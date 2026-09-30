"""Joint glyph/color optimization over the complete rendered frame."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import blue_noise, linear_to_srgb, srgb_to_linear
from .defaults import DEFAULT_USE_SHORTLIST, DEFAULT_GLYPH_ENTROPY_STRENGTH
from .settings import finite_float
from .tone import quantize_playback_lab, srgb_to_oklab


@dataclass
class Grid:
    glyph: np.ndarray  # rows x columns, uint8
    color: np.ndarray  # rows x columns x OKLab, float32
    background: np.ndarray | None = None  # one OKLab color for the complete image


def _cells(image: torch.Tensor, rows: int, cols: int, height: int, width: int) -> torch.Tensor:
    return image.permute(1, 2, 0).reshape(rows, height, cols, width, 3).permute(0, 2, 1, 3, 4).reshape(-1, height * width, 3)


def _render(mask: torch.Tensor, color: torch.Tensor, rows: int, cols: int,
            height: int, width: int, background: torch.Tensor | None = None) -> torch.Tensor:
    coverage = mask.reshape(rows, cols, height, width).permute(0, 2, 1, 3).reshape(rows * height, cols * width)
    foreground = color.reshape(rows, cols, 3).repeat_interleave(height, 0).repeat_interleave(width, 1)
    pixels = coverage[..., None] * foreground
    if background is not None:
        pixels = pixels + (1 - coverage[..., None]) * background.reshape(3)
    return pixels.permute(2, 0, 1).unsqueeze(0)


def _choose_background(image: torch.Tensor, choice: str) -> torch.Tensor:
    if choice != "auto":
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", choice):
            raise ValueError("Background must be 'auto' or a #RRGGBB color")
        srgb = np.array([int(choice[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)
        return image.new_tensor(srgb_to_oklab(srgb))
    # Choose one shadow color for the entire image. Keep it dark enough that
    # bright glyphs can still describe images with no dark regions.
    sampled = F.avg_pool2d(image.unsqueeze(0), 4, ceil_mode=True).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    lightness = sampled[:, 0]
    background = sampled[lightness <= torch.quantile(lightness, 0.1)].median(0).values
    return background * (0.18 / background[0].clamp_min(1e-8)).clamp(max=1)


DETAIL_LOSSES = ("default", "local_color", "chroma_contrast", "color_tail")
DITHER_TO_LOGIT_SCALE = 100.0


def _glyph_distribution(weights: torch.Tensor, indices: torch.Tensor,
                        glyph_count: int) -> torch.Tensor:
    """Pool candidate probabilities by glyph identity across the entire frame.

    Scatter by identity, rather than shortlist position, also handles repeated
    candidates (including appended transition endpoints).
    """
    return weights.new_zeros(glyph_count).scatter_add(
        0, indices.reshape(-1), weights.reshape(-1)) / len(weights)


def _negative_glyph_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Negative marginal Shannon entropy in nats; unused glyphs cost zero."""
    return (probabilities * probabilities.clamp_min(1e-12).log()).sum()


def _glyph_entropy_move_cost(glyph: torch.Tensor, indices: torch.Tensor,
                             counts: torch.Tensor) -> torch.Tensor:
    """Exact entropy change for each individual proposed glyph replacement.

    Proposals are ranked individually; simultaneous changes must still pass
    the complete-frame objective check, since their histogram effects interact.
    """
    def contribution(count):
        p = count / len(glyph)
        return p * p.clamp_min(1e-12).log()

    current = contribution(counts)
    removal = contribution((counts - 1).clamp_min(0)) - current
    addition = contribution(counts + 1) - current
    cost = removal[glyph, None] + addition[indices]
    return torch.where(indices == glyph[:, None], 0, cost)


def _ink_lightness_reward(color: torch.Tensor, ink_presence: torch.Tensor) -> torch.Tensor:
    """Reward brighter ink with a marginal gain of (1 - L)^2.

    Integrating the remaining headroom keeps the reward monotone while fading
    its pressure toward white. Shadows retain the original unit slope. Empty
    glyphs receive no reward; soft glyph mixtures use their ink probability.
    """
    lightness = color[..., 0].clamp(0, 1)
    return ink_presence * lightness * (1 - lightness + lightness.square() / 3)


@lru_cache(maxsize=16)
def _color_matrices(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(matrix, device=device, dtype=dtype).T for matrix in (
        ((1.0, 0.3963377774, 0.2158037573),
         (1.0, -0.1055613458, -0.0638541728),
         (1.0, -0.0894841775, -1.2914855480)),
        ((4.0767416621, -3.3077115913, 0.2309699292),
         (-1.2684380046, 2.6097574011, -0.3413193965),
         (-0.0041960863, -0.7034186147, 1.7076147010)),
        ((0.4122214708, 0.5363325363, 0.0514459929),
         (0.2119034982, 0.6806995451, 0.1073969566),
         (0.0883024619, 0.2817188376, 0.6299787005)),
        ((0.2104542553, 0.7936177850, -0.0040720468),
         (1.9779984951, -2.4285922050, 0.4505937099),
         (0.0259040371, 0.7827717662, -0.8086757660)),
    ))


def _oklab_to_rgb(color: torch.Tensor) -> torch.Tensor:
    lab_lms, lms_rgb, _, _ = _color_matrices(color.device, color.dtype)
    return (color @ lab_lms).pow(3) @ lms_rgb


def _rgb_to_oklab(color: torch.Tensor) -> torch.Tensor:
    _, _, rgb_lms, lms_lab = _color_matrices(color.device, color.dtype)
    lms = (color @ rgb_lms).clamp_min(0)
    # The cube-root derivative is infinite at black. Keep exact forward values
    # with a finite derivative for colors fitted at the RGB gamut boundary.
    roots = lms.clamp_min(1e-12).pow(1 / 3)
    roots = roots + (lms.pow(1 / 3) - roots).detach()
    return roots @ lms_lab


def _color_parameter(lab: torch.Tensor) -> torch.nn.Parameter:
    """Fit continuous sRGB channels, so every update has a displayable ink."""
    with torch.no_grad():
        _, rgb = _gamut_projection(lab)
        return torch.nn.Parameter(linear_to_srgb(rgb))


def _playback_color(srgb: torch.Tensor) -> torch.Tensor:
    """Decode RGB8 ink with gradients through RGB, except for byte rounding."""
    srgb = srgb.clamp(0, 1)
    quantized = (srgb * 255).round() / 255
    srgb = srgb + (quantized - srgb).detach()
    return _rgb_to_oklab(srgb_to_linear(srgb))


def _gamut_projection(color: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the shader's twelve-step, lightness-preserving gamut mapping."""
    lab = color.clone()
    lab[..., 0].clamp_(0, 1)
    rgb = _oklab_to_rgb(lab)
    bad = ((rgb < -1e-5) | (rgb > 1 + 1e-5)).any(-1)
    if bool(bad.any()):
        trial = lab[bad].clone()
        chroma = trial[..., 1:].clone()
        low = torch.zeros_like(trial[..., 0])
        high = torch.ones_like(low)
        for _ in range(12):
            mid = (low + high) * 0.5
            trial[..., 1:] = chroma * mid[..., None]
            candidate = _oklab_to_rgb(trial)
            valid = ((candidate >= -1e-5) & (candidate <= 1 + 1e-5)).all(-1)
            low = torch.where(valid, mid, low)
            high = torch.where(valid, high, mid)
        trial[..., 1:] = chroma * low[..., None]
        lab[bad] = trial
        rgb[bad] = _oklab_to_rgb(trial)
    return lab, rgb


def _project_oklab_colors(color: torch.Tensor, *, quantize: bool = True) -> torch.Tensor:
    """Compose using RGB8 ink while retaining a useful straight-through gradient.

    Parameters stay continuous between updates. Quantization is applied to the
    rendered colors, matching the files and shader without zeroing gradients at
    every RGB byte boundary.
    """
    with torch.no_grad():
        projected, rgb = _gamut_projection(color.detach())
        if quantize:
            rgb = rgb.clamp(0, 1)
            srgb = torch.where(rgb <= 0.0031308, rgb * 12.92,
                               1.055 * rgb.pow(1 / 2.4) - 0.055)
            srgb = (srgb * 255).round() / 255
            rgb = torch.where(srgb <= 0.04045, srgb / 12.92,
                              ((srgb + 0.055) / 1.055).pow(2.4))
            projected = _rgb_to_oklab(rgb)
        else:
            # Most displayed pixels are already in gamut. Their OKLab values
            # need no RGB roundtrip; only the shader's tiny boundary clamps do.
            clipped = ((rgb < 0) | (rgb > 1)).any(-1)
            if bool(clipped.any()):
                projected[clipped] = _rgb_to_oklab(rgb[clipped].clamp(0, 1))
    return color + (projected - color).detach()


def _saturation_loss(rendered: torch.Tensor, target_lab: torch.Tensor,
                    chroma_scale: torch.Tensor | None = None, *,
                    invert_ink: bool = False, displayed_lab: torch.Tensor | None = None,
                    source_chroma: torch.Tensor | None = None,
                    source_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Preserve source saturation and hue after the OKLab display tone transform.

    A magnitude-only deficit also rewards unrelated hues. Measure the deficit
    along the source hue and penalize perpendicular chroma. Allow stronger ink
    in that hue so sparse glyphs can compensate for their background gaps.
    """
    if displayed_lab is None:
        displayed_lab = _display_headroom_lab(rendered, invert_ink=invert_ink)
    if source_chroma is None:
        source_chroma = (target_lab[:, 1:].square().sum(1, keepdim=True) + 1e-8).sqrt()
    scale = (torch.quantile(source_chroma.detach().flatten(), 0.95).clamp_min(0.02)
             if chroma_scale is None else chroma_scale)
    weight = (1 + 4 * (source_chroma / scale).clamp(max=1).square()
              if source_weight is None else source_weight)
    direction = target_lab[:, 1:] / source_chroma
    aligned_chroma = (displayed_lab[:, 1:] * direction).sum(1, keepdim=True)
    perpendicular = displayed_lab[:, 1:] - aligned_chroma * direction
    # Holding absolute chroma constant while lifting lightness washes out ink.
    # Preserve C/L on brighter pixels; retain the source chroma floor on darker
    # pixels so glyph gaps cannot satisfy the correction by turning black.
    lightness_ratio = (displayed_lab[:, :1] / target_lab[:, :1].clamp_min(1e-6)).clamp_min(1)
    required_chroma = source_chroma * lightness_ratio
    chroma_error = (torch.relu(required_chroma - aligned_chroma).square()
                    + perpendicular.square().sum(1, keepdim=True))
    return (weight * chroma_error).mean((1, 2, 3))


def _display_headroom_lab(lab: torch.Tensor, *, invert_ink: bool = False) -> torch.Tensor:
    """Apply playback's tone curve and gamut map to an OKLab image."""
    lightness = lab[:, :1]
    displayed_lightness = lightness + 2.0 * lightness * (1 - lightness).square()
    chroma_scale = displayed_lightness / lightness.clamp_min(1e-6)
    if invert_ink:
        displayed_lightness = 1 - displayed_lightness
    displayed_lab = torch.cat((displayed_lightness, lab[:, 1:] * chroma_scale), dim=1)
    # Playback gamut-maps the displayed result after composing coverage. Its
    # chroma compression must also enter the gradient: an identity backward
    # pass cannot see the saturation lost by brightening at a gamut boundary.
    return _display_gamut_projection(displayed_lab.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _display_gamut_projection(color: torch.Tensor) -> torch.Tensor:
    """Match playback gamut mapping with the active RGB boundary's gradient."""
    if not color.requires_grad:
        return _project_oklab_colors(color, quantize=False)
    with torch.no_grad():
        projected, rgb = _gamut_projection(color.detach())
        raw_rgb = _oklab_to_rgb(color.detach())
        bad = ((raw_rgb < -1e-5) | (raw_rgb > 1 + 1e-5)).any(-1)
        clipped = ((rgb < 0) | (rgb > 1)).any(-1)
        if bool(clipped.any()):
            projected[clipped] = _rgb_to_oklab(rgb[clipped].clamp(0, 1))
    result = color + (projected - color).detach()
    if bool(bad.any()):
        trial = color[bad].clone()
        trial[..., 0].clamp_(0, 1)
        chroma = trial[..., 1:]
        scale = ((projected[bad][..., 1:] * chroma.detach()).sum(-1)
                 / chroma.detach().square().sum(-1).clamp_min(1e-12))
        mapped = torch.cat((trial[..., :1], chroma * scale[..., None]), dim=-1)
        mapped_rgb = _oklab_to_rgb(mapped)
        # Differentiate the boundary equation RGB_channel(L, a*s, b*s) = 0
        # or 1 implicitly. Bisection supplies s; a zero-valued correction gives
        # its derivative without differentiating discrete bisection decisions.
        lab_lms, lms_rgb, _, _ = _color_matrices(color.device, color.dtype)
        chroma_lab = torch.cat((torch.zeros_like(trial[..., :1]), chroma), dim=-1)
        slope = (3 * (mapped @ lab_lms).square() * (chroma_lab @ lab_lms)) @ lms_rgb
        channel = torch.minimum(rgb[bad].abs(), (rgb[bad] - 1).abs()).argmin(-1, keepdim=True)
        boundary = mapped_rgb.gather(-1, channel).squeeze(-1)
        derivative = slope.gather(-1, channel).squeeze(-1).detach()
        derivative = torch.where(derivative >= 0, derivative.clamp_min(1e-8),
                                 derivative.clamp_max(-1e-8))
        scale = scale - (boundary - boundary.detach()) / derivative
        corrected = torch.cat((trial[..., :1], chroma * scale[..., None]), dim=-1)
        result = result.clone()
        result[bad] = corrected + (projected[bad] - corrected).detach()
    return result


def _sobel_gradients(image_lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = image_lab.new_tensor([
        [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
    ]).reshape(1, 1, 3, 3) / 8
    vertical = horizontal.transpose(-1, -2)
    horizontal = horizontal.expand(image_lab.shape[1], 1, 3, 3)
    vertical = vertical.expand(image_lab.shape[1], 1, 3, 3)
    padded = F.pad(image_lab, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, horizontal, groups=image_lab.shape[1])
    gy = F.conv2d(padded, vertical, groups=image_lab.shape[1])
    return gx, gy


def _gradient_loss(rendered_lab: torch.Tensor, target_lab: torch.Tensor,
                   target_gradients: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Match oriented Sobel edges in displayed OKLab, including chroma edges."""
    rendered_x, rendered_y = _sobel_gradients(rendered_lab)
    target_x, target_y = (_sobel_gradients(target_lab) if target_gradients is None
                          else target_gradients)
    return ((rendered_x - target_x).square() + (rendered_y - target_y).square()).sum(1).mean()


def _post_logit_noise(rows: int, cols: int, candidates: int, seed: int,
                      distribution: str, *, device: str,
                      dtype: torch.dtype) -> torch.Tensor:
    """Build bounded blue-noise or IID Gumbel perturbations per cell/candidate."""
    if distribution == "uniform":
        uniform = torch.as_tensor(np.stack([
            blue_noise((rows, cols), seed + 1 + candidate * 104729)
            for candidate in range(candidates)
        ], axis=-1), dtype=dtype, device=device).reshape(-1, candidates)
        return uniform - 0.5
    if distribution == "gumbel":
        # Independent uniforms per cell and candidate make standard Gumbel-Max
        # sampling exact when the caller uses a noise scale of 1.
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        uniform = torch.rand((rows * cols, candidates), generator=generator,
                             device=device, dtype=dtype)
        epsilon = torch.finfo(dtype).eps
        uniform = uniform.clamp(min=epsilon, max=1 - epsilon)
        return -torch.log(-torch.log(uniform))
    raise ValueError("logit_noise_distribution must be 'uniform' or 'gumbel'")


def _chroma(image: torch.Tensor) -> torch.Tensor:
    return image[:, 1:]


def _local_color_weight(target: torch.Tensor) -> torch.Tensor:
    """Give small, locally unusual colors more influence without a hand-drawn mask."""
    chroma = _chroma(target)
    local_mean = F.avg_pool2d(F.pad(chroma, (10, 10, 10, 10), mode="replicate"), 21, stride=1)
    contrast = (chroma - local_mean).square().sum(1, keepdim=True).sqrt()
    scale = torch.quantile(contrast.detach().flatten(), 0.95).clamp_min(0.02)
    return 1 + 12 * (contrast / scale).clamp(max=1).square()


def _loss(rendered: torch.Tensor, target: torch.Tensor, *,
          detail_loss: str = "default", local_weight: torch.Tensor | None = None,
          cell: tuple[int, int] | None = None,
          target_lab: torch.Tensor | None = None,
          target_pyramid: tuple[tuple[int, float, torch.Tensor], ...] | None = None,
          gradient_target_lab: torch.Tensor | None = None,
          gradient_target_gradients: tuple[torch.Tensor, torch.Tensor] | None = None,
          gradient_loss_strength: float = 0.0,
          invert_ink: bool = False, displayed_lab: torch.Tensor | None = None) -> torch.Tensor:
    if target_lab is None:
        target_lab = target
    residual = rendered - target_lab
    pixel_error = residual.square().sum(1, keepdim=True)
    # Pixel color, low-frequency color, and edge/structure differences.
    loss = ((local_weight * pixel_error).mean() if detail_loss == "local_color"
            else pixel_error.mean())
    if target_pyramid is None:
        target_pyramid = tuple((factor, weight, F.avg_pool2d(target_lab, factor))
                               for factor, weight in ((2, 1.0), (4, 1.5), (8, 2.0))
                               if min(rendered.shape[-2:]) >= factor)
    for factor, weight, pooled_target in target_pyramid:
        pooled = (F.avg_pool2d(rendered, factor) - pooled_target).square()
        loss = loss + weight * pooled.sum(1, keepdim=True).mean()
    difference = residual[:, :1]
    if difference.shape[-1] > 1:
        loss = loss + 0.35 * (difference[..., 1:] - difference[..., :-1]).square().mean()
    if difference.shape[-2] > 1:
        loss = loss + 0.35 * (difference[..., 1:, :] - difference[..., :-1, :]).square().mean()
    if detail_loss == "chroma_contrast":
        chroma_difference = residual[:, 1:]
        local_mean = F.avg_pool2d(F.pad(chroma_difference, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        loss = loss + 2.0 * (chroma_difference - local_mean).square().mean()
    elif detail_loss == "color_tail":
        if cell is None:
            raise ValueError("color_tail requires a cell size")
        height, width = cell
        chroma = residual[:, 1:]
        chroma_error = chroma.square().mean(1).squeeze(0)
        rows, cols = chroma_error.shape[0] // height, chroma_error.shape[1] // width
        patches = chroma_error.reshape(rows, height, cols, width).permute(0, 2, 1, 3).reshape(-1, height * width)
        loss = loss + 0.75 * patches.topk(max(1, height * width // 10), dim=1).values.mean()
    if gradient_loss_strength:
        if gradient_target_lab is None:
            gradient_target_lab = target_lab
        if displayed_lab is None:
            displayed_lab = _display_headroom_lab(rendered, invert_ink=invert_ink)
        loss = loss + gradient_loss_strength * _gradient_loss(
            displayed_lab, gradient_target_lab, gradient_target_gradients)
    return loss


def _error_movement_samples(image: torch.Tensor, cell: tuple[int, int],
                            directions: torch.Tensor) -> torch.Tensor:
    """Sort color projections in overlapping neighborhoods of at most 2x2 glyphs.

    Matching equal ranks is one-dimensional optimal transport: pixel errors
    may move inside a neighborhood, but color mass cannot disappear. Multiple
    OKLab directions distinguish a mixture hue from its two source colors.
    """
    height, width = cell
    window_height = min(2 * height, image.shape[-2])
    window_width = min(2 * width, image.shape[-1])
    projected = torch.einsum("nchw,dc->ndhw", image, directions)
    patches = projected.unfold(2, window_height, height).unfold(3, window_width, width)
    return patches.flatten(-2).sort(dim=-1).values


@dataclass
class _FrameObjective:
    """Constant image-derived inputs shared by every optimization step."""
    target: torch.Tensor
    detail_loss: str
    cell: tuple[int, int]
    invert_ink: bool
    gradient_strength: float
    saturation_strength: float
    gradient_target: torch.Tensor | None = None
    saturation_target: torch.Tensor | None = None
    glyph_error_movement_strength: float = 0.0
    error_movement_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.error_movement_directions = self.error_movement_samples = None
        if self.glyph_error_movement_strength:
            # Axes, face diagonals and body diagonals give balanced sampling
            # of lightness/chroma without favoring a particular source hue.
            directions = self.target.new_tensor([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
                [0, 1, 1], [0, 1, -1],
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            ])
            self.error_movement_directions = directions / directions.norm(dim=1, keepdim=True)
            source = self.error_movement_target if self.error_movement_target is not None else self.target
            self.error_movement_samples = _error_movement_samples(
                source, self.cell, self.error_movement_directions).detach()
        self.local_weight = (_local_color_weight(self.target)
                             if self.detail_loss in ("local_color", "chroma_contrast", "color_tail")
                             else None)
        self.target_pyramid = tuple(
            (factor, weight, F.avg_pool2d(self.target, factor).detach())
            for factor, weight in ((2, 1.0), (4, 1.5), (8, 2.0))
            if min(self.target.shape[-2:]) >= factor)
        self.gradient_target_gradients = (_sobel_gradients(self.gradient_target)
                                          if self.gradient_strength else None)
        self.saturation_scale = None
        self.saturation_chroma = self.saturation_weight = None
        if self.saturation_strength:
            self.saturation_chroma = (self.saturation_target[:, 1:].square().sum(1, keepdim=True) + 1e-8).sqrt()
            self.saturation_scale = torch.quantile(self.saturation_chroma.flatten(), 0.95).clamp_min(0.02)
            self.saturation_weight = 1 + 4 * (
                self.saturation_chroma / self.saturation_scale).clamp(max=1).square()

    def __call__(self, rendered: torch.Tensor) -> torch.Tensor:
        displayed = (_display_headroom_lab(rendered, invert_ink=self.invert_ink)
                     if self.gradient_strength or self.saturation_strength or self.glyph_error_movement_strength else None)
        loss = _loss(rendered, self.target, detail_loss=self.detail_loss,
                     local_weight=self.local_weight, cell=self.cell,
                     target_pyramid=self.target_pyramid,
                     gradient_target_lab=self.gradient_target,
                     gradient_target_gradients=self.gradient_target_gradients,
                     gradient_loss_strength=self.gradient_strength,
                     invert_ink=self.invert_ink, displayed_lab=displayed)
        if self.saturation_strength:
            loss = loss + self.saturation_strength * _saturation_loss(
                rendered, self.saturation_target, self.saturation_scale,
                invert_ink=self.invert_ink, displayed_lab=displayed,
                source_chroma=self.saturation_chroma,
                source_weight=self.saturation_weight).mean()
        if self.glyph_error_movement_strength:
            samples = _error_movement_samples(displayed, self.cell, self.error_movement_directions)
            # Projection directions have covariance I/3; multiply by three
            # to keep the scale comparable to squared OKLab pixel distance.
            loss = loss + self.glyph_error_movement_strength * 3 * (
                samples - self.error_movement_samples).square().mean()
        return loss


def _candidate_fit(cells: torch.Tensor, masks: torch.Tensor, count: int,
                   background_cells: torch.Tensor,
                   pixel_weight: torch.Tensor | None = None,
                   glyph_luminance_bonus: float = 0.0,
                   gradient_target_cells: torch.Tensor | None = None,
                   gradient_target_gradients: tuple[torch.Tensor, torch.Tensor] | None = None,
                   gradient_shortlist_strength: float = 0.0,
                   invert_ink: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    mask_flat = masks.flatten(1)
    denominator = mask_flat.square().sum(1).clamp_min(1e-8)
    mask_square = mask_flat.square()
    ink_presence = mask_flat.gt(0).any(1)
    all_indices = []
    all_colors = []
    cell_height, cell_width = masks.shape[1:]
    # Bound the pixel workload as well as the cell count: physical cells grow
    # at high display resolutions, so 128 cells can otherwise allocate several
    # hundred MB for each shortlist/Sobel/gamut intermediate.
    cell_limit = 128 if gradient_target_cells is not None and gradient_shortlist_strength else 2048
    chunk_size = max(1, min(cell_limit, 2_000_000 //
                            (masks.shape[0] * cell_height * cell_width)))
    for start in range(0, cells.shape[0], chunk_size):
        cell = cells[start:start + chunk_size]
        underpaint = background_cells[start:start + len(cell)]
        weight = (pixel_weight[start:start + len(cell)] if pixel_weight is not None else None)
        if weight is None:
            numerator = (torch.einsum("npc,gp->ngc", cell - underpaint, mask_flat)
                         + torch.einsum("npc,gp->ngc", underpaint, mask_square))
            fitted_denominator = denominator[None, :]
        else:
            fitted_denominator = torch.einsum("np,gp->ng", weight, mask_square).clamp_min(1e-8)
            numerator = (torch.einsum("npc,gp->ngc", weight[..., None] * (cell - underpaint), mask_flat)
                         + torch.einsum("npc,gp->ngc", weight[..., None] * underpaint, mask_square))
        color = _project_oklab_colors(numerator / fitted_denominator[..., None])
        candidate_pixels = (mask_flat[None, :, :, None] * color[:, :, None, :]
                            + (1 - mask_flat[None, :, :, None]) * underpaint[:, None, :, :])
        perceptual_error = (candidate_pixels - cell[:, None]).square().sum(3)
        if weight is None:
            error = perceptual_error.mean(2)
        else:
            error = ((perceptual_error * weight[:, None]).sum(2)
                     / weight.sum(1, keepdim=True).clamp_min(1e-8))
        if glyph_luminance_bonus:
            error = error - glyph_luminance_bonus * _ink_lightness_reward(color, ink_presence)
        if gradient_target_cells is not None and gradient_shortlist_strength:
            # Rank glyphs partly by how their displayed OKLab edges match this
            # cell's source edges. Neighboring-cell edges are handled later by
            # the full-frame objective.
            candidate_lab = candidate_pixels.reshape(
                len(cell) * masks.shape[0], cell_height, cell_width, 3
            ).permute(0, 3, 1, 2)
            candidate_display = _display_headroom_lab(candidate_lab, invert_ink=invert_ink)
            candidate_x, candidate_y = _sobel_gradients(candidate_display)
            if gradient_target_gradients is None:
                source_cell = gradient_target_cells[start:start + len(cell)].reshape(
                    len(cell), cell_height, cell_width, 3
                ).permute(0, 3, 1, 2)
                source_x, source_y = _sobel_gradients(source_cell)
            else:
                source_x, source_y = (component[start:start + len(cell)]
                                      for component in gradient_target_gradients)
            candidate_count = masks.shape[0]
            source_x = source_x[:, None].expand(-1, candidate_count, -1, -1, -1).reshape_as(candidate_x)
            source_y = source_y[:, None].expand(-1, candidate_count, -1, -1, -1).reshape_as(candidate_y)
            gradient_error = (
                (candidate_x - source_x).square().sum(1).mean((1, 2))
                + (candidate_y - source_y).square().sum(1).mean((1, 2))
            ).reshape(len(cell), candidate_count)
            error = error + gradient_shortlist_strength * gradient_error
        indices = error.topk(min(count, masks.shape[0]), largest=False).indices
        all_indices.append(indices)
        all_colors.append(color.gather(1, indices[..., None].expand(-1, -1, 3)))
    return torch.cat(all_indices), torch.cat(all_colors)


def render_grid(grid: Grid, masks: np.ndarray, device: str = "cpu") -> np.ndarray:
    rows, cols = grid.glyph.shape
    height, width = masks.shape[1:]
    with torch.no_grad():
        bank = torch.as_tensor(masks, dtype=torch.float32, device=device).flatten(1)
        glyph = torch.as_tensor(grid.glyph.astype(np.int64), device=device).flatten()
        color = torch.as_tensor(grid.color, dtype=torch.float32, device=device).reshape(-1, 3)
        background = (torch.as_tensor(grid.background, dtype=torch.float32, device=device).reshape(3)
                      if grid.background is not None else None)
        image = _render(bank[glyph], color, rows, cols, height, width, background)
    return image.squeeze(0).permute(1, 2, 0).cpu().numpy()


def fit_fixed_glyphs(image_srgb: np.ndarray, masks_np: np.ndarray, glyphs: np.ndarray, *,
                     background_oklab: np.ndarray, steps: int = 24, device: str = "cpu",
                     previous: Grid | None = None, destination: Grid | None = None,
                     progress: float | None = None,
                     on_progress: Callable[[float, str], None] | None = None,
                     detail_loss: str = "default", glyph_luminance_bonus: float = 0.0,
                     saturation_loss_strength: float | None = None,
                     saturation_target_srgb: np.ndarray | None = None,
                     saturation_target_oklab: np.ndarray | None = None,
                     gradient_loss_strength: float = 1.5,
                     gradient_target_srgb: np.ndarray | None = None,
                     gradient_target_oklab: np.ndarray | None = None,
                     invert_ink: bool = False, input_oklab: bool = False) -> Grid:
    """Fit playback ink colors for already selected glyphs without a glyph solve."""
    if detail_loss not in DETAIL_LOSSES:
        raise ValueError(f"Unknown detail loss: {detail_loss}")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    glyph_luminance_bonus = finite_float(glyph_luminance_bonus, "glyph_luminance_bonus", minimum=0)
    if saturation_loss_strength is None:
        saturation_loss_strength = 20.0 if glyph_luminance_bonus > 0 else 0.0
    saturation_loss_strength = finite_float(saturation_loss_strength, "saturation_loss_strength", minimum=0)
    gradient_loss_strength = finite_float(gradient_loss_strength, "gradient_loss_strength", minimum=0)
    if len(masks_np) > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    height, width = masks_np.shape[1:]
    rows, cols = glyphs.shape
    if glyphs.size == 0 or glyphs.min() < 0 or glyphs.max() >= len(masks_np):
        raise ValueError("Fixed glyphs must refer to the provided mask bank")
    image_height, image_width = image_srgb.shape[:2]
    if (rows, cols) != ((image_height + height - 1) // height,
                       (image_width + width - 1) // width):
        raise ValueError("Fixed glyphs must use the target image grid")
    for endpoint in (previous, destination):
        if endpoint is not None and endpoint.glyph.shape != glyphs.shape:
            raise ValueError("Transition endpoints must use the same grid")
    pad = (0, cols * width - image_width, 0, rows * height - image_height)
    source_lab = (np.asarray(image_srgb, dtype=np.float32) if input_oklab
                  else srgb_to_oklab(image_srgb))

    def target_tensor(lab: np.ndarray) -> torch.Tensor:
        if lab.shape != source_lab.shape:
            raise ValueError("Color objective targets must match the source image shape")
        tensor = torch.as_tensor(lab, dtype=torch.float32, device=device).permute(2, 0, 1)
        return F.pad(tensor.unsqueeze(0), pad, mode="replicate").detach()

    target = target_tensor(source_lab)
    gradient_target = saturation_target = None
    if gradient_loss_strength:
        gradient_source = (gradient_target_oklab if gradient_target_oklab is not None
                           else srgb_to_oklab(gradient_target_srgb)
                           if gradient_target_srgb is not None else source_lab)
        gradient_target = target_tensor(gradient_source)
    if saturation_loss_strength:
        saturation_source = (saturation_target_oklab if saturation_target_oklab is not None
                             else srgb_to_oklab(saturation_target_srgb)
                             if saturation_target_srgb is not None else source_lab)
        saturation_target = target_tensor(saturation_source)
    objective = _FrameObjective(target, detail_loss, (height, width), invert_ink,
                                gradient_loss_strength, saturation_loss_strength,
                                gradient_target, saturation_target)
    background = _project_oklab_colors(torch.as_tensor(
        background_oklab, dtype=torch.float32, device=device).reshape(3))
    bank = torch.as_tensor(masks_np, dtype=torch.float32, device=device).flatten(1)
    mask = bank[torch.as_tensor(glyphs.astype(np.int64), device=device).flatten()]
    ink_presence = mask.gt(0).any(1)
    cells = _cells(target.squeeze(0), rows, cols, height, width)
    weights = torch.ones_like(mask)
    if detail_loss == "local_color":
        weights = objective.local_weight.reshape(rows, height, cols, width).permute(
            0, 2, 1, 3).reshape_as(mask)
    numerator = (weights[..., None] * mask[..., None]
                 * (cells - (1 - mask[..., None]) * background)).sum(1)
    denominator = (weights * mask.square()).sum(1).clamp_min(1e-8)
    color_parameter = _color_parameter(numerator / denominator[..., None])
    previous_colors = (torch.as_tensor(previous.color, dtype=torch.float32,
                                      device=device).reshape(-1, 3)
                       if previous is not None else None)
    scheduled_colors = None
    if previous is not None and destination is not None and progress is not None:
        scheduled_colors = ((1 - float(progress)) * previous_colors
                            + float(progress) * torch.as_tensor(destination.color,
                                dtype=torch.float32, device=device).reshape(-1, 3))

    def color_loss(color: torch.Tensor) -> torch.Tensor:
        loss = objective(_render(mask, color, rows, cols, height, width, background))
        if glyph_luminance_bonus:
            loss = loss - glyph_luminance_bonus * _ink_lightness_reward(color, ink_presence).mean()
        if previous_colors is not None:
            loss = loss + 0.010 * (color - previous_colors).square().mean()
        if scheduled_colors is not None:
            loss = loss + 0.004 * (color - scheduled_colors).square().mean()
        return loss

    # Preserve the best representable fit. Color objectives may disagree at a
    # gamut boundary, and the final Adam step need not be the best one.
    best_color = _playback_color(color_parameter.detach())
    best_loss = color_loss(best_color).detach()
    optimizer = torch.optim.Adam((color_parameter,), lr=0.01)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        color = _playback_color(color_parameter)
        loss = color_loss(color)
        best_color = torch.where(loss.detach() < best_loss, color.detach(), best_color)
        best_loss = torch.minimum(loss.detach(), best_loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            color_parameter.clamp_(0, 1)
        if on_progress is not None:
            on_progress((step + 1) / max(steps, 1), "Fitting path glyph colors")
    with torch.no_grad():
        color = _playback_color(color_parameter)
        best_color = torch.where(color_loss(color) < best_loss, color, best_color)
    if on_progress is not None and steps == 0:
        on_progress(1.0, "Fitting path glyph colors")
    return Grid(glyphs.astype(np.uint8).copy(),
                quantize_playback_lab(best_color.cpu().numpy().reshape(rows, cols, 3)),
                quantize_playback_lab(background.cpu().numpy()))


def solve(image_srgb: np.ndarray, masks_np: np.ndarray, *, seed: int = 0,
          steps: int = 36, candidates: int = 6, dither: float = 0.005,
          use_shortlist: bool = DEFAULT_USE_SHORTLIST,
          logit_noise_distribution: str = "uniform",
          background: str = "auto", background_oklab: np.ndarray | None = None,
          device: str = "cpu", previous: Grid | None = None,
          destination: Grid | None = None, progress: float | None = None,
          on_progress: Callable[[float, str], None] | None = None,
          detail_loss: str = "default",
          objective_space: str = "oklab",
          glyph_luminance_bonus: float = 0.0,
          saturation_loss_strength: float | None = None,
          saturation_target_srgb: np.ndarray | None = None,
          saturation_target_oklab: np.ndarray | None = None,
          gradient_loss_strength: float = 1.5,
          gradient_target_srgb: np.ndarray | None = None,
          gradient_target_oklab: np.ndarray | None = None,
          gradient_shortlist_strength: float | None = None,
          glyph_error_movement_strength: float = 0.0,
          glyph_entropy_strength: float = DEFAULT_GLYPH_ENTROPY_STRENGTH,
          error_movement_target_oklab: np.ndarray | None = None,
          invert_ink: bool = False,
          input_oklab: bool = False) -> Grid:
    """Optimize every font glyph per cell with OKLab scoring and bounded RGB ink.

    Shortlisting is retained only as an explicit comparison mode. The candidate
    cap and shortlist gradient strength have no effect on the normal solve.
    """
    if masks_np.shape[0] > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    if not isinstance(use_shortlist, bool):
        raise ValueError("use_shortlist must be a boolean")
    if detail_loss not in DETAIL_LOSSES:
        raise ValueError(f"Unknown detail loss: {detail_loss}")
    if objective_space != "oklab":
        raise ValueError("The optimizer operates only in OKLab.")
    glyph_luminance_bonus = finite_float(glyph_luminance_bonus, "glyph_luminance_bonus", minimum=0)
    if saturation_loss_strength is None:
        saturation_loss_strength = 20.0 if glyph_luminance_bonus > 0 else 0.0
    saturation_loss_strength = finite_float(saturation_loss_strength, "saturation_loss_strength", minimum=0)
    gradient_loss_strength = finite_float(gradient_loss_strength, "gradient_loss_strength", minimum=0)
    glyph_error_movement_strength = finite_float(
        glyph_error_movement_strength, "glyph_error_movement_strength", minimum=0)
    glyph_entropy_strength = finite_float(glyph_entropy_strength, "glyph_entropy_strength", minimum=0)
    if (glyph_error_movement_strength and error_movement_target_oklab is not None
            and error_movement_target_oklab.shape != image_srgb.shape):
        raise ValueError("Error movement target must match the source image shape")
    if gradient_shortlist_strength is None:
        gradient_shortlist_strength = gradient_loss_strength
    gradient_shortlist_strength = finite_float(gradient_shortlist_strength, "gradient_shortlist_strength", minimum=0)
    dither = finite_float(dither, "dither", minimum=0)
    if logit_noise_distribution not in ("uniform", "gumbel"):
        raise ValueError("logit_noise_distribution must be 'uniform' or 'gumbel'")
    # Preserve existing UI/CLI values while expressing perturbation in logit
    # units; in Gumbel mode dither=0.01 gives unit-scale Gumbel-Max.
    logit_noise_multiplier = dither * DITHER_TO_LOGIT_SCALE
    cell_height, cell_width = masks_np.shape[1:]
    original_height, original_width = image_srgb.shape[:2]
    rows = (original_height + cell_height - 1) // cell_height
    cols = (original_width + cell_width - 1) // cell_width
    pad_height = rows * cell_height - original_height
    pad_width = cols * cell_width - original_width
    image_lab_np = (np.asarray(image_srgb, dtype=np.float32) if input_oklab
                    else srgb_to_oklab(image_srgb))
    image = torch.as_tensor(image_lab_np, dtype=torch.float32, device=device).permute(2, 0, 1)
    image = F.pad(image.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0)
    target = image.unsqueeze(0)
    saturation_target_lab = None
    saturation_target_cells = None
    saturation_chroma_scale = None
    gradient_target_lab = None
    def target_tensor(lab: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(lab, dtype=torch.float32, device=device).permute(2, 0, 1)
        return F.pad(tensor.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0)

    if gradient_loss_strength or (use_shortlist and gradient_shortlist_strength):
        if gradient_target_oklab is not None:
            source_lab = gradient_target_oklab
        elif gradient_target_srgb is not None:
            source_lab = srgb_to_oklab(gradient_target_srgb)
        else:
            source_lab = image_lab_np
        gradient_target_lab = target_tensor(source_lab).unsqueeze(0).detach()
    if saturation_loss_strength:
        if saturation_target_oklab is not None:
            source_lab = saturation_target_oklab
        elif saturation_target_srgb is not None:
            source_lab = srgb_to_oklab(saturation_target_srgb)
        else:
            source_lab = image_lab_np
        saturation_target_lab = target_tensor(source_lab).unsqueeze(0).detach()
        saturation_chroma = (saturation_target_lab[:, 1:].square().sum(1, keepdim=True)
                             + 1e-8).sqrt()
        saturation_chroma_scale = torch.quantile(saturation_chroma.flatten(), 0.95).clamp_min(0.02)
        saturation_target_cells = _cells(saturation_target_lab.squeeze(0), rows, cols,
                                         cell_height, cell_width)
    gradient_target_cells = (
        _cells(gradient_target_lab.squeeze(0), rows, cols, cell_height, cell_width)
        if gradient_target_lab is not None else None
    )
    gradient_cell_gradients = (_sobel_gradients(gradient_target_cells.reshape(
        rows * cols, cell_height, cell_width, 3).permute(0, 3, 1, 2))
        if use_shortlist and gradient_shortlist_strength and gradient_target_cells is not None else None)
    cells = _cells(image, rows, cols, cell_height, cell_width)
    background_color = (_choose_background(image, background) if background_oklab is None
                        else torch.as_tensor(background_oklab, dtype=torch.float32, device=device).reshape(3))
    background_color = _project_oklab_colors(background_color)
    background_cells = background_color.reshape(1, 1, 3).expand(len(cells), cell_height * cell_width, 3)
    masks = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
    bank = masks.flatten(1)
    noise = torch.as_tensor(blue_noise((rows, cols), seed).flatten(), device=device)
    frame_objective = _FrameObjective(target.detach(), detail_loss,
                                     (cell_height, cell_width), invert_ink,
                                     gradient_loss_strength, saturation_loss_strength,
                                     gradient_target_lab, saturation_target_lab,
                                     glyph_error_movement_strength,
                                     target_tensor(error_movement_target_oklab).unsqueeze(0).detach()
                                     if glyph_error_movement_strength and error_movement_target_oklab is not None
                                     else None)
    local_weight = frame_objective.local_weight
    target_lab = target.detach()
    target_lab_cells = _cells(target_lab.squeeze(0), rows, cols, cell_height, cell_width)
    pixel_weight = None
    if local_weight is not None:
        pixel_weight = (local_weight.squeeze(0).squeeze(0).reshape(rows, cell_height, cols, cell_width)
                        .permute(0, 2, 1, 3).reshape(-1, cell_height * cell_width))
    if not use_shortlist:
        # Every glyph has a logit in every cell. Bypass candidate ranking and
        # retain the whole bank during both soft and discrete optimization.
        indices = torch.arange(len(bank), device=device).expand(len(cells), -1)
        fitted = None
    elif local_weight is not None:
        regular_count = max(1, candidates // 2)
        weighted_count = candidates - regular_count
        regular_indices, regular_colors = _candidate_fit(
            cells, masks, regular_count, background_cells,
            glyph_luminance_bonus=glyph_luminance_bonus,
            gradient_target_cells=gradient_target_cells,
            gradient_target_gradients=gradient_cell_gradients,
            gradient_shortlist_strength=gradient_shortlist_strength,
            invert_ink=invert_ink)
        weighted_indices, weighted_colors = _candidate_fit(
            cells, masks, weighted_count, background_cells, pixel_weight,
            glyph_luminance_bonus=glyph_luminance_bonus,
            gradient_target_cells=gradient_target_cells,
            gradient_target_gradients=gradient_cell_gradients,
            gradient_shortlist_strength=gradient_shortlist_strength,
            invert_ink=invert_ink)
        indices = torch.cat((regular_indices, weighted_indices), dim=1)
        fitted = torch.cat((regular_colors, weighted_colors), dim=1)
    else:
        indices, fitted = _candidate_fit(cells, masks, candidates,
                                         background_cells,
                                         glyph_luminance_bonus=glyph_luminance_bonus,
                                         gradient_target_cells=gradient_target_cells,
                                         gradient_target_gradients=gradient_cell_gradients,
                                         gradient_shortlist_strength=gradient_shortlist_strength,
                                         invert_ink=invert_ink)

    if previous is not None and destination is not None:
        if previous.glyph.shape != (rows, cols) or destination.glyph.shape != (rows, cols):
            raise ValueError("Transition endpoints must use the same grid")
        if use_shortlist:
            indices = torch.cat((indices,
                torch.as_tensor(previous.glyph.astype(np.int64), device=device).reshape(-1, 1),
                torch.as_tensor(destination.glyph.astype(np.int64), device=device).reshape(-1, 1)), dim=1)
            extra = []
            for grid in (previous, destination):
                extra.append(torch.as_tensor(grid.color, dtype=torch.float32, device=device).reshape(-1, 1, 3))
            fitted = torch.cat((fitted, *extra), dim=1)

    sample_masks = bank[indices]
    candidate_ink_presence = sample_masks.gt(0).any(2)
    fitting_weight = pixel_weight if detail_loss == "local_color" else None
    mask_den = (sample_masks.square().sum(2).clamp_min(1e-8) if fitting_weight is None
                else torch.einsum("np,nkp->nk", fitting_weight, sample_masks.square()).clamp_min(1e-8))
    fitting_cells = cells - background_cells
    if fitting_weight is not None:
        fitting_cells = fitting_cells * fitting_weight[..., None]
    numerator = torch.einsum("npc,nkp->nkc", fitting_cells, sample_masks)
    background_fit = background_cells if fitting_weight is None else background_cells * fitting_weight[..., None]
    numerator = numerator + torch.einsum("npc,nkp->nkc", background_fit, sample_masks.square())
    candidate_colors = _project_oklab_colors(numerator / mask_den[..., None])
    if fitted is None:
        fitted = candidate_colors
    candidate_render = (sample_masks[..., None] * candidate_colors[:, :, None, :]
                        + (1 - sample_masks[..., None]) * background_cells[:, None, :, :])
    perceptual_error = (candidate_render - target_lab_cells[:, None]).square().sum(3)
    if fitting_weight is None:
        initial_errors = perceptual_error.mean(2)
    else:
        initial_errors = ((perceptual_error * fitting_weight[:, None]).sum(2)
                          / fitting_weight.sum(1, keepdim=True).clamp_min(1e-8))
    if glyph_luminance_bonus:
        initial_errors = initial_errors - glyph_luminance_bonus * _ink_lightness_reward(
            candidate_colors, candidate_ink_presence)
    if saturation_loss_strength:
        batch, count = candidate_render.shape[:2]
        candidate_images = candidate_render.reshape(-1, cell_height, cell_width, 3).permute(0, 3, 1, 2)
        candidate_targets = (saturation_target_cells[:, None]
                             .expand(-1, count, -1, -1)
                             .reshape(-1, cell_height, cell_width, 3)
                             .permute(0, 3, 1, 2))
        saturation_errors = _saturation_loss(
            candidate_images, candidate_targets, saturation_chroma_scale,
            invert_ink=invert_ink).reshape(batch, count)
        initial_errors = initial_errors + saturation_loss_strength * saturation_errors
    if previous is not None and destination is not None:
        chosen = torch.where(noise < float(progress),
                             torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                             torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        initial_errors = initial_errors + 0.015 * (indices != chosen[:, None]).float()
    logits = torch.nn.Parameter((-initial_errors / 0.02).clamp(-20, 0).detach())
    first = initial_errors.argmin(1)
    initial_color = _project_oklab_colors(fitted[torch.arange(len(fitted), device=device), first])
    color_parameter = _color_parameter(initial_color)
    optimizer = torch.optim.Adam(({"params": (logits,), "lr": 0.12},
                                  {"params": (color_parameter,), "lr": 0.03}))
    prev_masks = prev_colors = scheduled_masks = None
    if previous is not None and destination is not None:
        prev_masks = bank[torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten()]
        prev_colors = torch.as_tensor(previous.color, dtype=torch.float32, device=device).reshape(-1, 3)
        scheduled = torch.where(noise < float(progress),
            torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
            torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        scheduled_masks = bank[scheduled]

    polish_steps = max(4, steps // 4)
    total_work = steps + 4 + polish_steps
    def frame_loss(rendered: torch.Tensor) -> torch.Tensor:
        return frame_objective(rendered)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = max(0.3, 1.0 - 0.7 * step / max(1, steps - 1))
        weights = torch.softmax(logits / temperature, dim=1)
        mask = (weights[..., None] * sample_masks).sum(1)
        color = _playback_color(color_parameter)
        rendered = _render(mask, color, rows, cols, cell_height, cell_width, background_color)
        loss = frame_loss(rendered)
        if glyph_luminance_bonus:
            ink_presence = (weights * candidate_ink_presence).sum(1)
            loss = loss - glyph_luminance_bonus * _ink_lightness_reward(color, ink_presence).mean()
        loss = loss + 0.006 * (-(weights * weights.clamp_min(1e-9).log()).sum(1)).mean()
        if glyph_entropy_strength:
            loss = loss + glyph_entropy_strength * _negative_glyph_entropy(
                _glyph_distribution(weights, indices, len(bank)))
        if prev_masks is not None:
            loss = loss + 0.025 * (mask - prev_masks).square().mean()
            loss = loss + 0.010 * (color - prev_colors).square().mean()
            loss = loss + 0.008 * (mask - scheduled_masks).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            color_parameter.clamp_(0, 1)
        if on_progress is not None:
            on_progress((step + 1) / total_work, "Optimizing glyphs and colors")

    with torch.no_grad():
        selection_logits = logits
        if logit_noise_multiplier:
            candidate_noise = _post_logit_noise(
                rows, cols, logits.shape[1], seed,
                logit_noise_distribution, device=device, dtype=logits.dtype,
            )
            selection_logits = logits + logit_noise_multiplier * candidate_noise
        selected = selection_logits.argmax(1)
        glyph = indices.gather(1, selected[:, None]).squeeze(1)
        hard_masks = bank[glyph]
        hard_color = _playback_color(color_parameter.detach())

    def hard_score(chosen_masks: torch.Tensor, chosen_color: torch.Tensor,
                   chosen_glyph: torch.Tensor) -> torch.Tensor:
        score = frame_loss(_render(chosen_masks, chosen_color, rows, cols, cell_height, cell_width, background_color))
        if glyph_entropy_strength:
            counts = torch.bincount(chosen_glyph, minlength=len(bank)).to(chosen_color.dtype)
            score = score + glyph_entropy_strength * _negative_glyph_entropy(counts / len(chosen_glyph))
        if glyph_luminance_bonus:
            score = score - glyph_luminance_bonus * _ink_lightness_reward(
                chosen_color, chosen_masks.gt(0).any(1)).mean()
        if prev_masks is not None:
            score = score + 0.025 * (chosen_masks - prev_masks).square().mean()
            score = score + 0.010 * (chosen_color - prev_colors).square().mean()
            score = score + 0.008 * (chosen_masks - scheduled_masks).square().mean()
        return score

    # Make actual discrete changes in four separated cell groups. Each proposal
    # is accepted only if the complete-frame rendered objective improves.
    row_ids = torch.arange(rows, device=device).repeat_interleave(cols)
    col_ids = torch.arange(cols, device=device).repeat(rows)
    for parity in range(4):
        rendered = _render(hard_masks, hard_color, rows, cols, cell_height, cell_width, background_color).detach().requires_grad_()
        gradient = torch.autograd.grad(frame_loss(rendered), rendered)[0]
        gradient_cells = _cells(gradient.squeeze(0), rows, cols, cell_height, cell_width)
        with torch.no_grad():
            current_pixels = (hard_masks[..., None] * hard_color[:, None, :]
                              + (1 - hard_masks[..., None]) * background_cells)
            delta = candidate_render - current_pixels[:, None]
            gain = (gradient_cells[:, None] * delta).sum((2, 3))
            gain = gain + delta.square().sum((2, 3)) / (len(glyph) * cell_height * cell_width * 3)
            if glyph_entropy_strength:
                counts = torch.bincount(glyph, minlength=len(bank)).to(gain.dtype)
                gain = gain + glyph_entropy_strength * _glyph_entropy_move_cost(glyph, indices, counts)
            if glyph_luminance_bonus:
                gain = gain - glyph_luminance_bonus * (
                    _ink_lightness_reward(candidate_colors, candidate_ink_presence)
                    - _ink_lightness_reward(hard_color, hard_masks.gt(0).any(1))[:, None]
                ) / len(glyph)
            if prev_masks is not None:
                gain = gain + 0.025 * ((sample_masks - prev_masks[:, None]).square().mean(2)
                                       - (hard_masks - prev_masks).square().mean(1)[:, None]) / len(glyph)
                gain = gain + 0.008 * ((sample_masks - scheduled_masks[:, None]).square().mean(2)
                                       - (hard_masks - scheduled_masks).square().mean(1)[:, None]) / len(glyph)
            best = gain.argmin(1)
            group = (row_ids % 2) * 2 + col_ids % 2 == parity
            changes = group & (gain.gather(1, best[:, None]).squeeze(1) < -1e-10)
            if bool(changes.any()):
                trial_glyph = glyph.clone()
                trial_color = hard_color.clone()
                trial_glyph[changes] = indices[changes, best[changes]]
                trial_color[changes] = candidate_colors[changes, best[changes]]
                trial_masks = bank[trial_glyph]
                if hard_score(trial_masks, trial_color, trial_glyph) < hard_score(hard_masks, hard_color, glyph):
                    glyph, hard_color, hard_masks = trial_glyph, trial_color, trial_masks
        if on_progress is not None:
            on_progress((steps + parity + 1) / total_work, "Refining character boundaries")

    # Final colors are optimized against the actual discrete glyphs.
    color_parameter = _color_parameter(hard_color)
    color_optimizer = torch.optim.Adam((color_parameter,), lr=0.01)
    best_color = hard_color.detach().clone()
    best_loss = hard_score(hard_masks, best_color, glyph).detach()
    for polish_step in range(polish_steps):
        color_optimizer.zero_grad(set_to_none=True)
        color = _playback_color(color_parameter)
        loss = hard_score(hard_masks, color, glyph)
        # Retain the best persisted fit, including the pre-polish colors.
        score = loss.detach()
        best_color = torch.where(score < best_loss, color.detach(), best_color)
        best_loss = torch.minimum(score, best_loss)
        loss.backward()
        color_optimizer.step()
        with torch.no_grad():
            color_parameter.clamp_(0, 1)
        if on_progress is not None:
            on_progress((steps + 4 + polish_step + 1) / total_work, "Polishing colors")

    with torch.no_grad():
        color = _playback_color(color_parameter)
        best_color = torch.where(hard_score(hard_masks, color, glyph) < best_loss, color, best_color)

    return Grid(glyph.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8),
                quantize_playback_lab(best_color.cpu().numpy().reshape(rows, cols, 3)),
                quantize_playback_lab(background_color.detach().cpu().numpy()))
