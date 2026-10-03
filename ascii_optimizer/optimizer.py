"""Joint glyph/color optimization over the complete rendered frame."""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import blue_noise, linear_to_srgb, srgb_to_linear
from .defaults import (
    DEFAULT_GLYPH_ENTROPY_STRENGTH,
    DEFAULT_PERCEPTUAL_LOSS,
    DEFAULT_POLISH_STEPS,
    DEFAULT_REFINEMENT_SWEEPS,
    DEFAULT_USE_SHORTLIST,
    PERCEPTUAL_WEIGHTS,
    PERCEPTUAL_GLYPH_STROKE_WEIGHT,
    PERCEPTUAL_PIXEL_WEIGHT,
    PERCEPTUAL_CHROMA_WEIGHT,
)
from .diagnostics import SolverTrace
from .perceptual import SpatialMetrics, ppd_for_width
from .settings import finite_float
from .tone import encode_headroom_lab, quantize_playback_lab, srgb_to_oklab


@dataclass
class Grid:
    glyph: np.ndarray  # rows x columns, uint8
    color: np.ndarray  # rows x columns x OKLab, float32
    background: np.ndarray | None = None  # one OKLab color for the complete image


def _cells(image: torch.Tensor, rows: int, cols: int, height: int, width: int) -> torch.Tensor:
    return image.permute(1, 2, 0).reshape(rows, height, cols, width, 3).permute(0, 2, 1, 3, 4).reshape(-1, height * width, 3)


def _render(mask: torch.Tensor, color: torch.Tensor, rows: int, cols: int,
            height: int, width: int, background: torch.Tensor | None = None) -> torch.Tensor:
    # Compose before rearranging cells: expanding foreground colors over the
    # whole image needlessly copied them twice, including in every backward pass.
    pixels = mask[..., None] * color[:, None, :]
    if background is not None:
        pixels = pixels + (1 - mask[..., None]) * background.reshape(3)
    pixels = pixels.reshape(rows, cols, height, width, 3).permute(0, 2, 1, 3, 4).reshape(rows * height, cols * width, 3)
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


def _glyph_counts(glyph: torch.Tensor, glyph_count: int, dtype: torch.dtype) -> torch.Tensor:
    """Fixed-size histogram; bincount synchronizes to infer its output size."""
    return torch.zeros(glyph_count, device=glyph.device, dtype=dtype).scatter_add_(
        0, glyph, torch.ones(glyph.shape, device=glyph.device, dtype=dtype))


def _hard_weights(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """One glyph in the forward pass, softmax derivatives in the backward pass."""
    soft = torch.softmax(logits / temperature, dim=1)
    hard = F.one_hot(soft.argmax(1), soft.shape[1]).to(soft.dtype)
    return hard + (soft - soft.detach())


_HARD_WEIGHTS_IMPLEMENTATION = _hard_weights


def _layout_error(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cell-aligned subpatch color errors; never pool away the entire glyph."""
    shape = (min(4, rendered.shape[-2]), min(2, rendered.shape[-1]))
    return (F.adaptive_avg_pool2d(rendered, shape)
            - F.adaptive_avg_pool2d(target, shape)).square().sum(1).mean((-2, -1))


@torch.no_grad()
def _accept_discrete_proposals(glyph: torch.Tensor, color: torch.Tensor,
                               bank: torch.Tensor, cell_ids: torch.Tensor,
                               proposed_glyph: torch.Tensor, proposed_color: torch.Tensor,
                               score: Callable, diagnostics: SolverTrace | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate batches against the whole objective, splitting rejected batches."""
    pending = [cell_ids] if len(cell_ids) else []
    current_score = score(bank[glyph], color, glyph).detach()
    while pending:
        ids = pending.pop()
        trial_glyph, trial_color = glyph.clone(), color.clone()
        trial_glyph[ids] = proposed_glyph[ids]
        trial_color[ids] = proposed_color[ids]
        trial_score = score(bank[trial_glyph], trial_color, trial_glyph).detach()
        if diagnostics is not None:
            diagnostics.counters["acceptance_trials"] = diagnostics.counters.get("acceptance_trials", 0) + 1
        if trial_score < current_score:
            glyph, color, current_score = trial_glyph, trial_color, trial_score
            if diagnostics is not None:
                diagnostics.counters["accepted_cells"] = diagnostics.counters.get("accepted_cells", 0) + len(ids)
        elif len(ids) > 1:
            midpoint = len(ids) // 2
            pending.extend((ids[midpoint:], ids[:midpoint]))
    return glyph, color


def _glyph_entropy_move_cost(glyph: torch.Tensor, indices: torch.Tensor,
                             counts: torch.Tensor,
                             cell_ids: torch.Tensor | None = None) -> torch.Tensor:
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
    current_glyph = glyph if cell_ids is None else glyph[cell_ids]
    cost = removal[current_glyph, None] + addition[indices]
    return torch.where(indices == current_glyph[:, None], 0, cost)


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
    linear_lms = color @ rgb_lms
    lms = linear_lms + (linear_lms.clamp_min(0) - linear_lms).detach()
    # The cube-root derivative is infinite at black. Keep exact forward values
    # with a finite derivative for colors fitted at the RGB gamut boundary.
    # RGB8's first nonzero linear channel is about 3e-4. A finite surrogate
    # slope below that scale lets rounded black ink recover during fitting,
    # instead of becoming an absorbing state through clamp's zero derivative.
    safe = lms.clamp_min(1e-4)
    safe = lms + (safe - lms).detach()
    roots = safe.pow(1 / 3)
    roots = roots + (lms.pow(1 / 3) - roots).detach()
    return roots @ lms_lab


def _color_parameter(lab: torch.Tensor) -> torch.nn.Parameter:
    """Fit continuous sRGB channels, so every update has a displayable ink."""
    with torch.no_grad():
        _, rgb = _gamut_projection(lab)
        return torch.nn.Parameter(linear_to_srgb(rgb))


def _playback_color(srgb: torch.Tensor, *, neutral_ink: bool = False) -> torch.Tensor:
    """Decode RGB8 ink with gradients through RGB, except for byte rounding."""
    # Parameters are projected after every update. Preserve a derivative at
    # RGB boundaries so a channel that reaches zero can move back into gamut.
    if neutral_ink:
        srgb = srgb.mean(-1, keepdim=True).expand_as(srgb)
    srgb = srgb + (srgb.clamp(0, 1) - srgb).detach()
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
    """Symmetrically preserve source C/L and hue in squared OKLab units.

    Compare displayed a,b with source a/L,b/L times displayed L. This avoids
    dividing by rendered lightness and gives black/neutral sources no invented
    chroma. Both excess and missing chroma are errors, including on dark pixels.
    Pixel/structure losses, rather than a chroma floor, constrain glyph gaps.
    """
    if displayed_lab is None:
        displayed_lab = _display_headroom_lab(rendered, invert_ink=invert_ink)
    if source_chroma is None:
        source_chroma = target_lab[:, 1:].norm(dim=1, keepdim=True)
    scale = (torch.quantile(source_chroma.detach().flatten(), 0.95).clamp_min(0.02)
             if chroma_scale is None else chroma_scale)
    weight = (1 + 4 * (source_chroma / scale).clamp(max=1).square()
              if source_weight is None else source_weight)
    source_chromaticity = target_lab[:, 1:] / target_lab[:, :1].clamp_min(1e-6)
    required_chroma = source_chromaticity * displayed_lab[:, :1]
    chroma_error = (displayed_lab[:, 1:] - required_chroma).square().sum(1, keepdim=True)
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


def _relative_chroma_loss(displayed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve pastel chroma as well as saturated colors, without a gray target.

    Normalize hue/chroma error by the source-defined C/L target at displayed
    lightness. Absolute squared error alone makes losing a pale tint nearly
    free compared with geometry. The soft denominator keeps black/near-gray
    derivatives finite; exactly achromatic sources have zero weight.
    """
    chroma = target[:, 1:]
    source_energy = chroma.square().sum(1, keepdim=True)
    expected = chroma / target[:, :1].clamp_min(1e-6) * displayed[:, :1]
    error = (displayed[:, 1:] - expected).square().sum(1, keepdim=True)
    weight = source_energy / (source_energy + 0.002 ** 2)
    return weight * error / (expected.square().sum(1, keepdim=True) + 0.002 ** 2)


def _sobel_filters(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3) / 8
    return (horizontal.expand(image.shape[1], 1, 3, 3),
            horizontal.transpose(-1, -2).expand(image.shape[1], 1, 3, 3))


def _sobel_gradients(image_lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal, vertical = _sobel_filters(image_lab)
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
          invert_ink: bool = False, displayed_lab: torch.Tensor | None = None,
          components: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    if target_lab is None:
        target_lab = target
    residual = rendered - target_lab
    pixel_error = residual.square().sum(1, keepdim=True)
    # Pixel color, low-frequency color, and edge/structure differences.
    loss = ((local_weight * pixel_error).mean() if detail_loss == "local_color"
            else pixel_error.mean())
    if components is not None:
        components["pixel"] = loss
    if target_pyramid is None:
        target_pyramid = tuple((factor, weight, F.avg_pool2d(target_lab, factor))
                               for factor, weight in ((2, 1.0), (4, 1.5), (8, 2.0))
                               if min(rendered.shape[-2:]) >= factor)
    for factor, weight, pooled_target in target_pyramid:
        pooled = (F.avg_pool2d(rendered, factor) - pooled_target).square()
        term = weight * pooled.sum(1, keepdim=True).mean()
        loss = loss + term
        if components is not None:
            components[f"pool_{factor}"] = term
    difference = residual[:, :1]
    if difference.shape[-1] > 1:
        term = 0.35 * (difference[..., 1:] - difference[..., :-1]).square().mean()
        loss = loss + term
        if components is not None:
            components["luminance_horizontal"] = term
    if difference.shape[-2] > 1:
        term = 0.35 * (difference[..., 1:, :] - difference[..., :-1, :]).square().mean()
        loss = loss + term
        if components is not None:
            components["luminance_vertical"] = term
    if detail_loss == "chroma_contrast":
        chroma_difference = residual[:, 1:]
        local_mean = F.avg_pool2d(F.pad(chroma_difference, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        term = 2.0 * (chroma_difference - local_mean).square().mean()
        loss = loss + term
        if components is not None:
            components["chroma_contrast"] = term
    elif detail_loss == "color_tail":
        if cell is None:
            raise ValueError("color_tail requires a cell size")
        height, width = cell
        chroma = residual[:, 1:]
        chroma_error = chroma.square().mean(1).squeeze(0)
        rows, cols = chroma_error.shape[0] // height, chroma_error.shape[1] // width
        patches = chroma_error.reshape(rows, height, cols, width).permute(0, 2, 1, 3).reshape(-1, height * width)
        term = 0.75 * patches.topk(max(1, height * width // 10), dim=1).values.mean()
        loss = loss + term
        if components is not None:
            components["color_tail"] = term
    if gradient_loss_strength:
        if gradient_target_lab is None:
            gradient_target_lab = target_lab
        if displayed_lab is None:
            displayed_lab = _display_headroom_lab(rendered, invert_ink=invert_ink)
        term = gradient_loss_strength * _gradient_loss(
            displayed_lab, gradient_target_lab, gradient_target_gradients)
        loss = loss + term
        if components is not None:
            components["displayed_sobel"] = term
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
    layout_loss_strength: float = 0.0
    spatial_metrics: SpatialMetrics | None = None
    contour_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.spatial_metrics is not None and hasattr(self.spatial_metrics, "encoded_target"):
            self.target = self.spatial_metrics.encoded_target
            low, high = self.spatial_metrics.reference_range
            def adapt(lab):
                if lab is None:
                    return None
                rgb = _oklab_to_rgb(lab.permute(0, 2, 3, 1))
                return _rgb_to_oklab(low + rgb * (high - low)).permute(0, 3, 1, 2).detach()
            self.gradient_target = adapt(self.gradient_target)
            self.saturation_target = adapt(self.saturation_target)
            self.error_movement_target = adapt(self.error_movement_target)
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
            self.saturation_chroma = self.saturation_target[:, 1:].norm(dim=1, keepdim=True)
            self.saturation_scale = torch.quantile(self.saturation_chroma.flatten(), 0.95).clamp_min(0.02)
            self.saturation_weight = 1 + 4 * (
                self.saturation_chroma / self.saturation_scale).clamp(max=1).square()

    def __call__(self, rendered: torch.Tensor, *,
                 components: dict[str, torch.Tensor] | None = None,
                 displayed_lab: torch.Tensor | None = None) -> torch.Tensor:
        displayed = displayed_lab
        if displayed is None and (self.gradient_strength or self.saturation_strength or self.glyph_error_movement_strength or self.spatial_metrics is not None):
            displayed = _display_headroom_lab(rendered, invert_ink=self.invert_ink)
        loss = _loss(rendered, self.target, detail_loss=self.detail_loss,
                     local_weight=self.local_weight, cell=self.cell,
                     target_pyramid=self.target_pyramid,
                     gradient_target_lab=self.gradient_target,
                     gradient_target_gradients=self.gradient_target_gradients,
                     gradient_loss_strength=self.gradient_strength,
                     invert_ink=self.invert_ink, displayed_lab=displayed,
                     components=(pixel_components := {}) if components is not None else None)
        if self.spatial_metrics is not None:
            loss = PERCEPTUAL_PIXEL_WEIGHT * loss
        if components is not None:
            scale = PERCEPTUAL_PIXEL_WEIGHT if self.spatial_metrics is not None else 1.0
            components.update({name: scale * value for name, value in pixel_components.items()})
        if self.saturation_strength:
            term = self.saturation_strength * _saturation_loss(
                rendered, self.saturation_target, self.saturation_scale,
                invert_ink=self.invert_ink, displayed_lab=displayed,
                source_chroma=self.saturation_chroma,
                source_weight=self.saturation_weight).mean()
            loss = loss + term
            if components is not None:
                components["saturation"] = term
            if self.spatial_metrics is not None:
                term = PERCEPTUAL_CHROMA_WEIGHT * _relative_chroma_loss(displayed, self.saturation_target).mean()
                loss = loss + term
                if components is not None:
                    components["relative_chroma"] = term
        if self.glyph_error_movement_strength:
            samples = _error_movement_samples(displayed, self.cell, self.error_movement_directions)
            # Projection directions have covariance I/3; multiply by three
            # to keep the scale comparable to squared OKLab pixel distance.
            term = self.glyph_error_movement_strength * 3 * (
                samples - self.error_movement_samples).square().mean()
            loss = loss + term
            if components is not None:
                components["error_movement"] = term
        if self.layout_loss_strength:
            height, width = self.cell
            rows, cols = self.target.shape[-2] // height, self.target.shape[-1] // width
            def cell_images(image):
                return _cells(image.squeeze(0), rows, cols, height, width).reshape(
                    -1, height, width, 3).permute(0, 3, 1, 2)
            term = self.layout_loss_strength * _layout_error(
                cell_images(rendered), cell_images(self.target)).mean()
            loss = loss + term
            if components is not None:
                components["layout"] = term
        if self.spatial_metrics is not None:
            rgb = _oklab_to_rgb(displayed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            terms, _ = self.spatial_metrics.terms(rgb, selected=PERCEPTUAL_WEIGHTS)
            for name, weight in PERCEPTUAL_WEIGHTS.items():
                if name != "spatial_lightness":
                    weight *= self.contour_scale
                term = weight * terms[name]
                loss = loss + term
                if components is not None:
                    components[name] = term
        return loss


def _replicate_padding_adjoint(padded: torch.Tensor, padding: int) -> torch.Tensor:
    """Fold a padded gradient back through replicate padding, including corners."""
    horizontal = padded[..., :, padding:-padding].clone()
    horizontal[..., 0] += padded[..., :, :padding].sum(-1)
    horizontal[..., -1] += padded[..., :, -padding:].sum(-1)
    result = horizontal[..., padding:-padding, :].clone()
    result[..., 0, :] += horizontal[..., :padding, :].sum(-2)
    result[..., -1, :] += horizontal[..., -padding:, :].sum(-2)
    return result


def _display_ink_table(color: torch.Tensor, background: torch.Tensor,
                       coverage: torch.Tensor, *, invert_ink: bool) -> torch.Tensor:
    """Map each actual atlas coverage value once for a cell's current ink.

    Glyphs share the same ink/background in a cell. Looking up their repeated
    coverage values is exact, and avoids tone/gamut mapping the same composed
    color separately for every occurrence in every candidate glyph.
    """
    alpha = coverage[None, None, None]
    pixels = alpha * color[:, :, None, None] + (1 - alpha) * background[None, :, None, None]
    return _display_headroom_lab(pixels, invert_ink=invert_ink).squeeze(2)


class _ReplacementState:
    """Frame-wide residual/adjoint fields shared by all replacement batches.

    For every quadratic term ||A*x-target||², the exact replacement delta is
    <gradient, delta> + ||A*delta||². Convolutions and pooling adjoints prepare
    the gradient once. Nonlinear display mapping supplies a separate pixel
    delta in displayed OKLab; its Sobel loss is then quadratic in that delta.
    """

    @torch.no_grad()
    def __init__(self, objective: _FrameObjective, rendered: torch.Tensor):
        if objective.spatial_metrics is not None:
            raise ValueError("Exact replacement scoring does not support the perceptual losses")
        self.objective = objective
        self.rendered = rendered.detach()
        self.height, self.width = objective.cell
        self.frame_height, self.frame_width = rendered.shape[-2:]
        self.rows = self.frame_height // self.height
        self.cols = self.frame_width // self.width
        self.count = self.rows * self.cols
        self.area = self.frame_height * self.frame_width
        self.base = self.cells(self.rendered)
        self.target = self.cells(objective.target)
        residual = self.rendered - objective.target
        weight = objective.local_weight if objective.detail_loss == "local_color" else None
        self.pixel_weight = self.cells(weight) if weight is not None else None
        gradient = 2 * residual / self.area
        if weight is not None:
            gradient = gradient * weight
        for factor, strength, target in objective.target_pyramid:
            pooled = F.avg_pool2d(self.rendered, factor) - target
            adjoint = pooled.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
            adjoint = F.pad(adjoint, (0, self.frame_width - adjoint.shape[-1],
                                     0, self.frame_height - adjoint.shape[-2]))
            gradient += (2 * strength / (target.shape[-2] * target.shape[-1] * factor ** 2)) * adjoint
        lightness = residual[:, :1]
        if self.frame_width > 1:
            difference = (0.7 / (self.frame_height * (self.frame_width - 1))) * (
                lightness[..., 1:] - lightness[..., :-1])
            gradient[:, :1, :, :-1] -= difference
            gradient[:, :1, :, 1:] += difference
        if self.frame_height > 1:
            difference = (0.7 / ((self.frame_height - 1) * self.frame_width)) * (
                lightness[..., 1:, :] - lightness[..., :-1, :])
            gradient[:, :1, :-1, :] -= difference
            gradient[:, :1, 1:, :] += difference
        if objective.detail_loss == "chroma_contrast":
            chroma = residual[:, 1:]
            contrast = chroma - F.avg_pool2d(F.pad(chroma, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
            kernel = chroma.new_ones((2, 1, 9, 9)) / 81
            adjoint = _replicate_padding_adjoint(F.conv_transpose2d(contrast, kernel, groups=2), 4)
            gradient[:, 1:] += 2 * (contrast - adjoint) / self.area
        if objective.layout_loss_strength:
            with torch.enable_grad():
                cells = self.base.detach().requires_grad_()
                layout = objective.layout_loss_strength * _layout_error(cells, self.target).mean()
                cell_gradient = torch.autograd.grad(layout, cells)[0]
            gradient += cell_gradient.reshape(self.rows, self.cols, 3, self.height, self.width).permute(
                2, 0, 3, 1, 4).reshape_as(gradient)
        self.gradient = self.cells(gradient)
        self.displayed = self.displayed_cells = self.display_gradient = None
        self.sobel_kernels = _sobel_filters(rendered)
        self.delta_sobel_kernels = torch.stack((self.sobel_kernels[0][0], self.sobel_kernels[1][0]))
        if objective.gradient_strength or objective.saturation_strength or objective.glyph_error_movement_strength:
            self.displayed = _display_headroom_lab(self.rendered, invert_ink=objective.invert_ink)
            self.displayed_cells = self.cells(self.displayed)
        if objective.gradient_strength:
            gx, gy = _sobel_gradients(self.displayed)
            tx, ty = objective.gradient_target_gradients
            horizontal, vertical = self.sobel_kernels
            adjoint = (F.conv_transpose2d(gx - tx, horizontal, groups=3)
                       + F.conv_transpose2d(gy - ty, vertical, groups=3))
            self.display_gradient = self.cells(_replicate_padding_adjoint(adjoint, 1)) * (
                2 * objective.gradient_strength / self.area)
        self.saturation_target = self.saturation_chroma = self.saturation_weight = self.saturation_before = None
        if objective.saturation_strength:
            self.saturation_target = self.cells(objective.saturation_target)
            self.saturation_chroma = self.cells(objective.saturation_chroma)
            self.saturation_weight = self.cells(objective.saturation_weight)
            self.saturation_before = _saturation_loss(
                self.base, self.saturation_target, objective.saturation_scale,
                invert_ink=objective.invert_ink, displayed_lab=self.displayed_cells,
                source_chroma=self.saturation_chroma, source_weight=self.saturation_weight)
        self.tail_before = None
        if objective.detail_loss == "color_tail":
            self.tail_before = self.tail_error(self.base, self.target)
        self.transport_before = None
        if objective.glyph_error_movement_strength:
            samples = _error_movement_samples(self.displayed, objective.cell, objective.error_movement_directions)
            self.transport_before = (samples - objective.error_movement_samples).square().sum((1, 4))

    def cells(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        return image[0].reshape(channels, self.rows, self.height, self.cols, self.width).permute(
            1, 3, 0, 2, 4).reshape(self.count, channels, self.height, self.width)

    def tail_error(self, pixels: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        error = (pixels[:, 1:] - targets[:, 1:]).square().mean(1).flatten(1)
        return error.topk(max(1, self.height * self.width // 10), dim=1).values.mean(1)


class _ReplacementBatchScorer:
    """Full-frame batch checks without remapping unchanged pixels each time.

    Acceptance only combines the unchanged cells and their fixed proposals.
    Display mapping is pixel-local, so both choices can be prepared once. Each
    check still evaluates the complete image objective, including interactions
    between simultaneous changes; no approximate acceptance delta is used.
    """

    @torch.no_grad()
    def __init__(self, state: _ReplacementState, bank: torch.Tensor,
                 glyph: torch.Tensor, color: torch.Tensor,
                 proposed_glyph: torch.Tensor, proposed_color: torch.Tensor,
                 cell_ids: torch.Tensor, background: torch.Tensor):
        self.state = state
        self.glyph, self.color = glyph, color
        self.pixels = state.base.clone()
        mask = bank[proposed_glyph[cell_ids]][:, None].reshape(-1, 1, state.height, state.width)
        pixels = mask * proposed_color[cell_ids, :, None, None] + (1 - mask) * background[None, :, None, None]
        self.pixels[cell_ids] = pixels
        self.displayed = None
        if state.displayed_cells is not None:
            self.displayed = state.displayed_cells.clone()
            self.displayed[cell_ids] = _display_headroom_lab(pixels, invert_ink=state.objective.invert_ink)

    def image(self, cells: torch.Tensor) -> torch.Tensor:
        state = self.state
        return cells.permute(0, 2, 3, 1).reshape(state.rows, state.cols, state.height, state.width, 3).permute(
            0, 2, 1, 3, 4).reshape(state.frame_height, state.frame_width, 3).permute(2, 0, 1)[None]

    @torch.no_grad()
    def __call__(self, glyph: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        changed = ((glyph != self.glyph) | (color != self.color).any(1))[:, None, None, None]
        rendered = self.image(torch.where(changed, self.pixels, self.state.base))
        displayed = (self.image(torch.where(changed, self.displayed, self.state.displayed_cells))
                     if self.displayed is not None else None)
        return self.state.objective(rendered, displayed_lab=displayed)


class _ReplacementContext:
    """Exact candidate deltas using cell pixels and cached frame adjoint fields.

    No candidate copies a surrounding rendered image. Only the change itself
    enters pooling and edge kernels. Small zero/replicate halos account for
    affected outputs at neighboring pixels and the actual frame boundaries.
    """

    def __init__(self, objective: _FrameObjective, rendered: torch.Tensor,
                 cell_ids: torch.Tensor, state: _ReplacementState | None = None):
        self.state = _ReplacementState(objective, rendered) if state is None else state
        self.objective = objective
        self.rendered = self.state.rendered
        self.cell_ids = cell_ids
        self.height, self.width = objective.cell
        self.frame_height, self.frame_width = rendered.shape[-2:]
        self.rows, self.cols = self.state.rows, self.state.cols
        self.cell_y = cell_ids // self.cols * self.height
        self.cell_x = cell_ids % self.cols * self.width
        self.base = self.state.base[cell_ids]
        self.target = self.state.target[cell_ids]
        self.gradient = self.state.gradient[cell_ids]
        self.pixel_weight = self.state.pixel_weight[cell_ids] if self.state.pixel_weight is not None else None
        self.halos = {}
        self.pool_indices = None
        self.pool_valid = {}
        if any(self.height % factor or self.width % factor for factor, _, _ in objective.target_pyramid):
            # Worst phase for a cell lattice is 8-gcd(cell_dimension, 8).
            self.pool_height = ((self.height + 8 - gcd(self.height, 8) + 7) // 8) * 8
            self.pool_width = ((self.width + 8 - gcd(self.width, 8) + 7) // 8) * 8
            origin_y, origin_x = self.cell_y // 8 * 8, self.cell_x // 8 * 8
            device = rendered.device
            self.pool_indices = (
                (self.cell_y - origin_y)[:, None, None] * self.pool_width
                + (self.cell_x - origin_x)[:, None, None]
                + torch.arange(self.height, device=device)[None, :, None] * self.pool_width
                + torch.arange(self.width, device=device)[None, None, :]).flatten(1)
            for factor, _, target in objective.target_pyramid:
                yy = origin_y[:, None, None] // factor + torch.arange(self.pool_height // factor, device=device)[None, :, None]
                xx = origin_x[:, None, None] // factor + torch.arange(self.pool_width // factor, device=device)[None, None, :]
                self.pool_valid[factor] = ((yy < target.shape[-2]) & (xx < target.shape[-1]))[:, None]

    @staticmethod
    def reduce(value: torch.Tensor) -> torch.Tensor:
        return value.sum((-3, -2, -1), dtype=torch.float64)

    def halo(self, delta: torch.Tensor, padding: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad only the pixel change, replicating it at physical frame edges."""
        batch, count, channels = delta.shape[:3]
        if padding not in self.halos:
            device = delta.device
            yy = torch.arange(-padding, self.height + padding, device=device)[None, :, None]
            xx = torch.arange(-padding, self.width + padding, device=device)[None, None, :]
            frame_y = self.cell_y[:, None, None] + yy
            frame_x = self.cell_x[:, None, None] + xx
            local_y = frame_y.clamp(0, self.frame_height - 1) - self.cell_y[:, None, None] + padding
            local_x = frame_x.clamp(0, self.frame_width - 1) - self.cell_x[:, None, None] + padding
            indices = (local_y * (self.width + 2 * padding) + local_x).flatten(1)
            valid = ((frame_y >= 0) & (frame_y < self.frame_height)
                     & (frame_x >= 0) & (frame_x < self.frame_width))[:, None]
            self.halos[padding] = indices, valid
        indices, valid = self.halos[padding]
        padded = F.pad(delta, (padding, padding, padding, padding)).flatten(-2)
        padded = padded.gather(-1, indices[:, None, None].expand(batch, count, channels, -1))
        return (padded.reshape(batch * count, channels, self.height + 2 * padding, self.width + 2 * padding),
                valid[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1))

    def quadratic_energy(self, delta: torch.Tensor) -> torch.Tensor:
        state = self.state
        objective = self.objective
        batch, count = delta.shape[:2]
        square = delta.square()
        energy = self.reduce(square if self.pixel_weight is None else square * self.pixel_weight[:, None]) / state.area
        aligned = None
        for factor, strength, target in objective.target_pyramid:
            if self.height % factor == 0 and self.width % factor == 0:
                pooled = F.avg_pool2d(delta.flatten(0, 1), factor).reshape(batch, count, 3, self.height // factor, self.width // factor)
                norm = self.reduce(pooled.square())
            else:
                if aligned is None:
                    aligned = delta.new_zeros((batch, count, 3, self.pool_height * self.pool_width)).scatter(
                        -1, self.pool_indices[:, None, None].expand(batch, count, 3, -1), delta.flatten(-2))
                    aligned = aligned.reshape(batch * count, 3, self.pool_height, self.pool_width)
                pooled = F.avg_pool2d(aligned, factor).reshape(batch, count, 3, self.pool_height // factor, self.pool_width // factor)
                norm = self.reduce(pooled.square() * self.pool_valid[factor][:, None])
            energy += strength * norm / (target.shape[-2] * target.shape[-1])
        lightness = delta[:, :, 0]
        if self.frame_width > 1:
            norm = (lightness[..., 1:] - lightness[..., :-1]).square().sum((-2, -1), dtype=torch.float64)
            norm += lightness[..., 0].square().sum(-1) * (self.cell_x > 0)[:, None]
            norm += lightness[..., -1].square().sum(-1) * (self.cell_x + self.width < self.frame_width)[:, None]
            energy += 0.35 * norm / (self.frame_height * (self.frame_width - 1))
        if self.frame_height > 1:
            norm = (lightness[..., 1:, :] - lightness[..., :-1, :]).square().sum((-2, -1), dtype=torch.float64)
            norm += lightness[..., 0, :].square().sum(-1) * (self.cell_y > 0)[:, None]
            norm += lightness[..., -1, :].square().sum(-1) * (self.cell_y + self.height < self.frame_height)[:, None]
            energy += 0.35 * norm / ((self.frame_height - 1) * self.frame_width)
        if objective.detail_loss == "chroma_contrast":
            padded, valid = self.halo(delta[:, :, 1:], 8)
            contrast = padded[..., 4:-4, 4:-4] - F.avg_pool2d(padded, 9, stride=1)
            norm = self.reduce(contrast.square() * valid[..., 4:-4, 4:-4]).reshape(batch, count)
            energy += norm / state.area
        if objective.layout_loss_strength:
            cells = delta.flatten(0, 1)
            energy += objective.layout_loss_strength * _layout_error(cells, torch.zeros_like(cells)).reshape(batch, count) / state.count
        return energy

    def scores(self, replacements: torch.Tensor,
               displayed_replacements: torch.Tensor | None = None) -> torch.Tensor:
        """Exact full-objective change for each independent cell replacement."""
        state, objective = self.state, self.objective
        batch, count = replacements.shape[:2]
        delta = replacements - self.base[:, None]
        result = self.reduce(delta * self.gradient[:, None]) + self.quadratic_energy(delta)
        pixels = replacements.flatten(0, 1)
        def expand(values):
            return values[self.cell_ids, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        displayed = None
        if objective.gradient_strength or objective.saturation_strength or objective.glyph_error_movement_strength:
            displayed = (_display_headroom_lab(pixels, invert_ink=objective.invert_ink)
                         if displayed_replacements is None else displayed_replacements.flatten(0, 1))
        if objective.gradient_strength:
            display_delta = displayed.reshape_as(replacements) - state.displayed_cells[self.cell_ids, None]
            result += self.reduce(display_delta * state.display_gradient[self.cell_ids, None])
            padded, valid = self.halo(display_delta, 2)
            # Compute both Sobel directions in one convolution. Treat color
            # channels as batch entries to avoid tiny grouped convolutions.
            response = F.conv2d(padded.reshape(-1, 1, self.height + 4, self.width + 4),
                                state.delta_sobel_kernels)
            response = response.reshape(batch * count, 6, self.height + 2, self.width + 2)
            norm = self.reduce(response.square() * valid[..., 1:-1, 1:-1]).reshape(batch, count)
            result += objective.gradient_strength * norm / state.area
        if objective.saturation_strength:
            error = _saturation_loss(pixels, expand(state.saturation_target), objective.saturation_scale,
                invert_ink=objective.invert_ink, displayed_lab=displayed,
                source_chroma=expand(state.saturation_chroma), source_weight=expand(state.saturation_weight))
            result += objective.saturation_strength * (
                error.reshape(batch, count).double() - state.saturation_before[self.cell_ids, None].double()) / state.count
        if objective.detail_loss == "color_tail":
            error = state.tail_error(pixels, self.target[:, None].expand_as(replacements).flatten(0, 1))
            result += 0.75 * (error.reshape(batch, count) - state.tail_before[self.cell_ids, None]) / state.count
        if objective.glyph_error_movement_strength:
            display_cells = displayed.reshape_as(replacements)
            for i, cell_id in enumerate(self.cell_ids.tolist()):
                row, col = divmod(cell_id, self.cols)
                for wy in range(max(0, row - 1), min(row, max(0, self.rows - 2)) + 1):
                    for wx in range(max(0, col - 1), min(col, max(0, self.cols - 2)) + 1):
                        y, x = wy * self.height, wx * self.width
                        window = state.displayed[:, :, y:y + 2 * self.height, x:x + 2 * self.width]
                        trial = window.expand(count, -1, -1, -1).clone()
                        dy, dx = row * self.height - y, col * self.width - x
                        trial[:, :, dy:dy + self.height, dx:dx + self.width] = display_cells[i]
                        samples = _error_movement_samples(trial, objective.cell, objective.error_movement_directions)
                        target = objective.error_movement_samples[:, :, wy:wy + 1, wx:wx + 1]
                        error = (samples - target).square().sum((1, 2, 3, 4))
                        difference = error - state.transport_before[:, wy, wx]
                        result[i] = result[i] + (objective.glyph_error_movement_strength * 3
                                                / objective.error_movement_samples.numel()) * difference
        return result


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


def _make_spatial_metrics(image_srgb, *, source_srgb, source_oklab, input_oklab,
                          pad, cell_height, device, viewing_ppd, masks, background,
                          invert_ink):
    """Map source contrast into the atlas's displayed range once per solve.

    A white source patch targets the brightest attainable mean glyph tile,
    rather than unattainable solid-screen white. This fixed, frame-wide affine
    linear-RGB mapping preserves shading through highlights, and depends only
    on the actual atlas, uniform background and display transform.
    """
    if source_srgb is not None:
        if source_srgb.shape != image_srgb.shape:
            raise ValueError("Perceptual target must match the source image shape")
        source = torch.as_tensor(source_srgb, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        rgb = srgb_to_linear(source)
    elif source_oklab is not None or input_oklab:
        lab = source_oklab if source_oklab is not None else image_srgb
        if lab.shape != image_srgb.shape:
            raise ValueError("Perceptual target must match the source image shape")
        source = torch.as_tensor(lab, dtype=torch.float32, device=device)
        rgb = _oklab_to_rgb(source).permute(2, 0, 1)[None]
    else:
        source = torch.as_tensor(image_srgb, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        rgb = srgb_to_linear(source)
    ppd = ppd_for_width(image_srgb.shape[1]) if viewing_ppd is None else viewing_ppd
    with torch.no_grad():
        atlas = torch.as_tensor(masks, dtype=torch.float32, device=device)[:, None]
        white = _rgb_to_oklab(atlas.new_ones(3))[None, :, None, None]
        bg = background[None, :, None, None]
        displayed = _display_headroom_lab(atlas * white + (1 - atlas) * bg, invert_ink=invert_ink)
        atlas_rgb = _oklab_to_rgb(displayed.permute(0, 2, 3, 1)).mean((1, 2))
        bg_rgb = _oklab_to_rgb(_display_headroom_lab(bg, invert_ink=invert_ink).reshape(3))
        y = atlas_rgb @ atlas_rgb.new_tensor([0.2126, 0.7152, 0.0722])
        low, high = ((atlas_rgb[y.argmin()], bg_rgb) if invert_ink
                     else (bg_rgb, atlas_rgb[y.argmax()]))
        reference = low[None, :, None, None] + rgb * (high - low)[None, :, None, None]
        reference_lab = _rgb_to_oklab(reference.permute(0, 2, 3, 1))
        encoded = encode_headroom_lab(reference_lab.squeeze(0).cpu().numpy(), invert_ink=invert_ink)
        encoded = torch.as_tensor(encoded, device=device).permute(2, 0, 1)[None]
    model = SpatialMetrics(F.pad(reference, pad, mode="replicate"), ppd=ppd, cell_height=cell_height)
    model.reference_range = (low, high)
    model.encoded_target = F.pad(encoded, pad, mode="replicate")
    return model


@torch.no_grad()
def _spatial_ink_fit(model, masks, background, *, invert_ink):
    """Fit each glyph to the same local linear-light mean and source hue.

    A shared ink makes a swap from a dense to a thin glyph look too dark even
    when brighter ink can match the target. This small neutral-ink response
    table gives every candidate a useful color before its geometry competes.
    Full-frame optimization still scores actual gamut-mapped color pixels.
    """
    height, width = masks.shape[-2:]
    rows, cols = model.source_rgb.shape[-2] // height, model.source_rgb.shape[-1] // width
    mean_rgb = _cells(model.source_rgb.squeeze(0), rows, cols, height, width).mean(1)
    target_y = mean_rgb @ mean_rgb.new_tensor([0.2126, 0.7152, 0.0722])
    levels = torch.linspace(0, 1, 65, device=masks.device, dtype=masks.dtype)
    ink = torch.stack((levels, levels * 0, levels * 0), 1)
    alpha = masks[:, None, None]
    pixels = alpha * ink[None, :, :, None, None] + (1 - alpha) * background[None, None, :, None, None]
    shown = _display_headroom_lab(pixels.flatten(0, 1), invert_ink=invert_ink)
    rgb = _oklab_to_rgb(shown.permute(0, 2, 3, 1)).mean((1, 2))
    response = (rgb @ rgb.new_tensor([0.2126, 0.7152, 0.0722])).reshape(len(masks), -1)
    if invert_ink:
        response, levels = response.flip(1), levels.flip(0)
    upper = torch.searchsorted(response.contiguous(), target_y[None].expand(len(masks), -1).contiguous()).clamp(1, len(levels) - 1)
    lower = upper - 1
    y0, y1 = response.gather(1, lower), response.gather(1, upper)
    fraction = ((target_y[None] - y0) / (y1 - y0).clamp_min(1e-8)).clamp(0, 1)
    light = (levels[lower] + fraction * (levels[upper] - levels[lower])).T
    lab = _rgb_to_oklab(mean_rgb)
    ratio = lab[:, 1:] / lab[:, :1].clamp_min(1e-6)
    chroma = light[..., None] * ratio[:, None]
    if invert_ink:
        boosted = light + 2 * light * (1 - light).square()
        chroma = chroma * ((1 - boosted) / boosted.clamp_min(1e-6))[..., None]
    colors = torch.cat((light[..., None], chroma), 2)
    return _project_oklab_colors(colors).detach()


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
                     perceptual_loss: bool = DEFAULT_PERCEPTUAL_LOSS,
                     perceptual_target_srgb: np.ndarray | None = None,
                     viewing_ppd: float | None = None,
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
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if viewing_ppd is not None:
        viewing_ppd = finite_float(viewing_ppd, "viewing_ppd", minimum=1, maximum=1024)
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
    background = _project_oklab_colors(torch.as_tensor(
        background_oklab, dtype=torch.float32, device=device).reshape(3))
    objective = _FrameObjective(target, detail_loss, (height, width), invert_ink,
                                gradient_loss_strength, saturation_loss_strength,
                                gradient_target, saturation_target,
                                spatial_metrics=_make_spatial_metrics(
                                    image_srgb, source_srgb=perceptual_target_srgb
                                    if perceptual_target_srgb is not None else gradient_target_srgb
                                    if gradient_target_srgb is not None else saturation_target_srgb,
                                    source_oklab=gradient_target_oklab
                                    if gradient_target_oklab is not None else saturation_target_oklab,
                                    input_oklab=input_oklab,
                                    pad=pad, cell_height=height, device=device, viewing_ppd=viewing_ppd,
                                    masks=masks_np, background=background, invert_ink=invert_ink)
                                if perceptual_loss else None)
    neutral_ink = (perceptual_loss and objective.spatial_metrics.neutral_source
                   and bool(background[1:].abs().amax() < 1e-6))
    bank = torch.as_tensor(masks_np, dtype=torch.float32, device=device).flatten(1)
    mask = bank[torch.as_tensor(glyphs.astype(np.int64), device=device).flatten()]
    ink_presence = mask.gt(0).any(1)
    cells = _cells(objective.target.squeeze(0), rows, cols, height, width)
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
    best_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
    best_loss = color_loss(best_color).detach()
    optimizer = torch.optim.Adam((color_parameter,), lr=0.003 if perceptual_loss else 0.01)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        color = _playback_color(color_parameter, neutral_ink=neutral_ink)
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
        color = _playback_color(color_parameter, neutral_ink=neutral_ink)
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
          layout_loss_strength: float = 0.0,
          refinement_sweeps: int = DEFAULT_REFINEMENT_SWEEPS,
          polish_steps: int = DEFAULT_POLISH_STEPS,
          postprocess: bool | None = None,
          perceptual_loss: bool = DEFAULT_PERCEPTUAL_LOSS,
          perceptual_target_srgb: np.ndarray | None = None,
          viewing_ppd: float | None = None,
          flow_costs: torch.Tensor | np.ndarray | None = None,
          flow_strength: float = 0.0,
          diagnostics: SolverTrace | None = None,
          invert_ink: bool = False,
          input_oklab: bool = False) -> Grid:
    """Optimize every font glyph per cell with OKLab scoring and bounded RGB ink.

    Shortlisting is retained only as an explicit comparison mode. The candidate
    cap and shortlist gradient strength have no effect on the normal solve.
    Spatial brightness and circular contour preservation participate in Adam
    and final color polish. The exact refinement cache supports
    only the previous objective, available with perceptual_loss=False.
    """
    if masks_np.shape[0] > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    if not isinstance(use_shortlist, bool):
        raise ValueError("use_shortlist must be a boolean")
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if postprocess is None:
        postprocess = not perceptual_loss
    if not isinstance(postprocess, bool):
        raise ValueError("postprocess must be a boolean")
    if viewing_ppd is not None:
        viewing_ppd = finite_float(viewing_ppd, "viewing_ppd", minimum=1, maximum=1024)
    if detail_loss not in DETAIL_LOSSES:
        raise ValueError(f"Unknown detail loss: {detail_loss}")
    if objective_space != "oklab":
        raise ValueError("The optimizer operates only in OKLab.")
    for name, value in (("steps", steps), ("refinement_sweeps", refinement_sweeps),
                        ("polish_steps", polish_steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if perceptual_loss and refinement_sweeps:
        raise ValueError("Refinement does not support the perceptual losses; use perceptual_loss=False for legacy comparisons")
    glyph_luminance_bonus = finite_float(glyph_luminance_bonus, "glyph_luminance_bonus", minimum=0)
    if saturation_loss_strength is None:
        saturation_loss_strength = 20.0 if glyph_luminance_bonus > 0 else 0.0
    saturation_loss_strength = finite_float(saturation_loss_strength, "saturation_loss_strength", minimum=0)
    gradient_loss_strength = finite_float(gradient_loss_strength, "gradient_loss_strength", minimum=0)
    glyph_error_movement_strength = finite_float(
        glyph_error_movement_strength, "glyph_error_movement_strength", minimum=0)
    glyph_entropy_strength = finite_float(glyph_entropy_strength, "glyph_entropy_strength", minimum=0)
    flow_strength = finite_float(flow_strength, "flow_strength", 0, 2)
    if flow_strength and flow_costs is None:
        raise ValueError("Flow strength requires flow/glyph costs")
    if flow_strength and (not perceptual_loss or refinement_sweeps):
        raise ValueError("Flow glyph selection requires the perceptual full-frame solver")
    layout_loss_strength = finite_float(layout_loss_strength, "layout_loss_strength", minimum=0)
    if (glyph_error_movement_strength and error_movement_target_oklab is not None
            and error_movement_target_oklab.shape != image_srgb.shape):
        raise ValueError("Error movement target must match the source image shape")
    if gradient_shortlist_strength is None:
        gradient_shortlist_strength = gradient_loss_strength
    gradient_shortlist_strength = finite_float(gradient_shortlist_strength, "gradient_shortlist_strength", minimum=0)
    dither = finite_float(dither, "dither", minimum=0)
    if not postprocess:
        dither = 0.0
        polish_steps = 0
    if logit_noise_distribution not in ("uniform", "gumbel"):
        raise ValueError("logit_noise_distribution must be 'uniform' or 'gumbel'")
    # Preserve existing UI/CLI values while expressing perturbation in logit
    # units; in Gumbel mode dither=0.01 gives unit-scale Gumbel-Max.
    logit_noise_multiplier = dither * DITHER_TO_LOGIT_SCALE
    cell_height, cell_width = masks_np.shape[1:]
    original_height, original_width = image_srgb.shape[:2]
    rows = (original_height + cell_height - 1) // cell_height
    cols = (original_width + cell_width - 1) // cell_width
    if flow_costs is not None:
        flow_costs = torch.as_tensor(flow_costs, dtype=torch.float32, device=device).detach()
        if flow_costs.shape != (rows * cols, len(masks_np)) or not bool(torch.isfinite(flow_costs).all()):
            raise ValueError("Flow costs must be finite and match every cell and glyph")
    if not flow_strength:
        flow_costs = None
    if diagnostics is not None:
        diagnostics.start(device, rows=rows, cols=cols, steps=steps, seed=seed,
                          postprocess=postprocess,
                          dither=dither, cell=[cell_width, cell_height],
                          glyph_count=len(masks_np), use_shortlist=use_shortlist,
                          refinement_sweeps=int(refinement_sweeps), polish_steps=int(polish_steps))
    def section(name, **metadata):
        return diagnostics.section(name, **metadata) if diagnostics is not None else nullcontext()
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
        saturation_chroma = saturation_target_lab[:, 1:].norm(dim=1, keepdim=True)
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
                                     else None, layout_loss_strength,
                                     spatial_metrics=_make_spatial_metrics(
                                         image_srgb, source_srgb=perceptual_target_srgb
                                         if perceptual_target_srgb is not None else gradient_target_srgb
                                         if gradient_target_srgb is not None else saturation_target_srgb,
                                         source_oklab=gradient_target_oklab
                                         if gradient_target_oklab is not None else saturation_target_oklab,
                                         input_oklab=input_oklab,
                                         pad=(0, pad_width, 0, pad_height), cell_height=cell_height,
                                         device=device, viewing_ppd=viewing_ppd, masks=masks_np,
                                         background=background_color, invert_ink=invert_ink)
                                     if perceptual_loss else None)
    if flow_costs is not None:
        frame_objective.contour_scale = 1 / (1 + 3 * flow_strength)
    neutral_ink = (perceptual_loss and frame_objective.spatial_metrics.neutral_source
                   and bool(background_color[1:].abs().amax() < 1e-6))
    stroke_costs = (frame_objective.spatial_metrics.glyph_stroke_costs(masks)
                    if perceptual_loss and flow_costs is None else None)
    cells = _cells(frame_objective.target.squeeze(0), rows, cols, cell_height, cell_width)
    saturation_target_lab = frame_objective.saturation_target
    if saturation_target_lab is not None:
        saturation_target_cells = _cells(saturation_target_lab.squeeze(0), rows, cols, cell_height, cell_width)
        saturation_chroma_scale = frame_objective.saturation_scale
    if frame_objective.gradient_target is not None:
        gradient_target_cells = _cells(frame_objective.gradient_target.squeeze(0), rows, cols, cell_height, cell_width)
        if use_shortlist and gradient_shortlist_strength:
            gradient_cell_gradients = _sobel_gradients(gradient_target_cells.reshape(
                rows * cols, cell_height, cell_width, 3).permute(0, 3, 1, 2))
    if diagnostics is not None:
        diagnostics.metadata.update(coefficients={
            "pixel": 1.0, "pool_2": 1.0, "pool_4": 1.5, "pool_8": 2.0,
            "luminance_horizontal": .35, "luminance_vertical": .35,
            "chroma_contrast": 2.0, "color_tail": .75,
            "displayed_sobel": gradient_loss_strength, "saturation": saturation_loss_strength,
            "error_movement": 3 * glyph_error_movement_strength, "layout": layout_loss_strength,
            "glyph_entropy": glyph_entropy_strength, "ink_reward": -glyph_luminance_bonus,
            "glyph_flow": flow_strength,
            "transition_mask": .025, "transition_color": .010, "transition_schedule": .008,
            **({**{name: weight * (frame_objective.contour_scale if name != "spatial_lightness" else 1)
                    for name, weight in PERCEPTUAL_WEIGHTS.items()},
                "glyph_stroke_alignment": PERCEPTUAL_GLYPH_STROKE_WEIGHT if stroke_costs is not None else 0.0}
               if perceptual_loss else {})},
            perceptual_loss=perceptual_loss,
            neutral_ink=neutral_ink,
            viewing_ppd=frame_objective.spatial_metrics.ppd if perceptual_loss else None,
            replacement_diagnostics="Disabled for nonlinear perceptual losses" if perceptual_loss else "Exact",
            saturation_definition="Symmetric displayed a,b - source (a/L,b/L) times displayed L; "
                                  "squared OKLab error, source chroma weights 1..5. Black source chroma is zero.",
            visual_definition="Probability-weighted variance of composed encoded-OKLab glyph pixels and "
                              "oriented Sobel changes including a zero halo, at current ink/background. "
                              "These linear diagnostics exclude display tone/gamut mapping.",
            gradient_definition="RMS derivatives of weighted training terms in logits and bounded RGB ink parameters; "
                                "includes the straight-through and scheduled-temperature derivatives.")
        if perceptual_loss:
            diagnostics.metadata["coefficients"]["relative_chroma"] = PERCEPTUAL_CHROMA_WEIGHT
            for name in ("pixel", "pool_2", "pool_4", "pool_8", "luminance_horizontal",
                         "luminance_vertical", "chroma_contrast", "color_tail", "displayed_sobel"):
                diagnostics.metadata["coefficients"][name] *= PERCEPTUAL_PIXEL_WEIGHT
            diagnostics.metadata["reference_range_linear_rgb"] = [v.cpu().tolist()
                for v in frame_objective.spatial_metrics.reference_range]
            diagnostics.metadata["stroke_definition"] = (
                "Squared unsigned OKLab contour tensor distance plus 0.25 times raster coverage; "
                "source bandpass scales H/4..H and H/2..2H, coherence and energy weights.")
            if stroke_costs is not None:
                diagnostics.register(glyph_stroke_costs=stroke_costs)
            diagnostics.metadata["contour_scale"] = frame_objective.contour_scale
            if flow_costs is not None:
                diagnostics.register(glyph_flow_costs=flow_costs)
    local_weight = frame_objective.local_weight
    target_lab = frame_objective.target.detach()
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

    sample_masks = (bank[indices] if use_shortlist else
                    bank[None].expand(len(cells), -1, -1))
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
    if perceptual_loss:
        spatial_colors = _spatial_ink_fit(frame_objective.spatial_metrics, masks, background_color, invert_ink=invert_ink)
        candidate_colors = spatial_colors.gather(1, indices[..., None].expand(-1, -1, 3))
        fitted = candidate_colors
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
    if perceptual_loss:
        initial_errors = PERCEPTUAL_PIXEL_WEIGHT * initial_errors
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
        if perceptual_loss:
            displayed_candidates = _display_headroom_lab(candidate_images, invert_ink=invert_ink)
            relative = _relative_chroma_loss(displayed_candidates, candidate_targets).mean((1, 2, 3)).reshape(batch, count)
            initial_errors = initial_errors + PERCEPTUAL_CHROMA_WEIGHT * relative
    if stroke_costs is not None:
        initial_errors = initial_errors + PERCEPTUAL_GLYPH_STROKE_WEIGHT * stroke_costs.gather(1, indices)
    if flow_costs is not None:
        initial_errors = initial_errors + flow_strength * flow_costs.gather(1, indices)
    if previous is not None and destination is not None:
        chosen = torch.where(noise < float(progress),
                             torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                             torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        initial_errors = initial_errors + 0.015 * (indices != chosen[:, None]).float()
    # The local fit is a starting preference, not an exclusion of candidates
    # that can win the spatial objective. Keep all candidates trainable.
    logit_scale, logit_floor = (0.1, -4) if perceptual_loss else (0.02, -20)
    logits = torch.nn.Parameter((-(initial_errors - initial_errors.amin(1, keepdim=True)) / logit_scale).clamp(logit_floor, 0).detach())
    first = initial_errors.argmin(1)
    initial_color = _project_oklab_colors(fitted[torch.arange(len(fitted), device=device), first])
    color_parameter = _color_parameter(initial_color)
    optimizer = torch.optim.Adam(({"params": (logits,), "lr": 0.12},
                                  {"params": (color_parameter,), "lr": 0.01 if perceptual_loss else 0.03}))
    prev_masks = prev_colors = scheduled_masks = None
    if previous is not None and destination is not None:
        prev_masks = bank[torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten()]
        prev_colors = torch.as_tensor(previous.color, dtype=torch.float32, device=device).reshape(-1, 3)
        scheduled = torch.where(noise < float(progress),
            torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
            torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        scheduled_masks = bank[scheduled]

    coverage_values = coverage_indices = candidate_displayed = None
    if refinement_sweeps and (gradient_loss_strength or saturation_loss_strength or glyph_error_movement_strength):
        # The atlas usually has at most 256 coverage values. Skip the lookup
        # for arbitrary masks whose unique values would provide little reuse.
        values, inverse = torch.unique(bank, sorted=True, return_inverse=True)
        if len(values) * 2 < bank.numel():
            coverage_values, coverage_indices = values, inverse
        # Analytic candidate inks never change. Retain their display mapping
        # only for smaller frames, keeping the additional cache below 64 MiB.
        if (coverage_values is not None
                and candidate_render.numel() * candidate_render.element_size() <= 64 * 1024 ** 2):
            with torch.no_grad():
                candidate_images = candidate_render.reshape(
                    -1, cell_height, cell_width, 3).permute(0, 3, 1, 2)
                candidate_displayed = _display_headroom_lab(candidate_images, invert_ink=invert_ink).reshape(
                    len(cells), indices.shape[1], 3, cell_height, cell_width)

    total_work = steps + 6 * refinement_sweeps + polish_steps
    def frame_loss(rendered: torch.Tensor, components=None) -> torch.Tensor:
        return (frame_objective(rendered) if components is None
                else frame_objective(rendered, components=components))

    def score_terms(frame_value, chosen_masks, chosen_color, chosen_glyph, components=None):
        score = frame_value.double()
        def add(name, value):
            nonlocal score
            score = score + value
            if components is not None:
                components[name] = value
        if glyph_entropy_strength:
            counts = _glyph_counts(chosen_glyph, len(bank), chosen_color.dtype)
            add("glyph_entropy", glyph_entropy_strength * _negative_glyph_entropy(counts / len(chosen_glyph)))
        if stroke_costs is not None:
            add("glyph_stroke_alignment", PERCEPTUAL_GLYPH_STROKE_WEIGHT *
                stroke_costs.gather(1, chosen_glyph[:, None]).mean())
        if flow_costs is not None:
            add("glyph_flow", flow_strength * flow_costs.gather(1, chosen_glyph[:, None]).mean())
        if glyph_luminance_bonus:
            add("ink_reward", -glyph_luminance_bonus * _ink_lightness_reward(
                chosen_color, chosen_masks.gt(0).any(1)).mean())
        if prev_masks is not None:
            add("transition_mask", 0.025 * (chosen_masks - prev_masks).square().mean())
            add("transition_color", 0.010 * (chosen_color - prev_colors).square().mean())
            add("transition_schedule", 0.008 * (chosen_masks - scheduled_masks).square().mean())
        return score

    def hard_score(chosen_masks: torch.Tensor, chosen_color: torch.Tensor,
                   chosen_glyph: torch.Tensor, components=None) -> torch.Tensor:
        value = frame_loss(_render(chosen_masks, chosen_color, rows, cols, cell_height, cell_width, background_color), components)
        return score_terms(value, chosen_masks, chosen_color, chosen_glyph, components)

    @torch.no_grad()
    def observe(stage, chosen_glyph, chosen_color, *, logged_logits=None, temperature=None,
                score=None, components=None, **metadata):
        if diagnostics is None:
            return
        if score is None:
            components = {}
            score = hard_score(bank[chosen_glyph], chosen_color, chosen_glyph, components)
        diagnostics.record(stage, loss=score, components=components,
                           glyph=chosen_glyph, color=chosen_color, logits=logged_logits,
                           temperature=temperature, **metadata)

    if diagnostics is not None:
        diagnostics.register(candidate_indices=indices, background=background_color,
                             selection_noise=torch.zeros_like(logits))
        diagnostics.register_geometry(masks, *_sobel_filters(masks[:, None]))
        initial_glyph = indices.gather(1, logits.argmax(1)[:, None]).squeeze(1)
        observe("initial", initial_glyph, _playback_color(color_parameter.detach(), neutral_ink=neutral_ink),
                logged_logits=logits, temperature=1., step=0)

    for step in range(steps):
        with section("adam_step", step=step):
            optimizer.zero_grad(set_to_none=True)
            temperature = max(0.3, 1.0 - 0.7 * step / max(1, steps - 1))
            weights = _hard_weights(logits, temperature)
            # Contract directly instead of materializing cell × glyph × pixel
            # products on every Adam step. The full bank is shared by cells.
            mask = (weights @ bank if not use_shortlist else
                    torch.bmm(weights[:, None], sample_masks).squeeze(1))
            actual_glyph = indices.gather(1, weights.argmax(1)[:, None]).squeeze(1)
            if _hard_weights is _HARD_WEIGHTS_IMPLEMENTATION:
                # Keep atlas pixels exact even when the caller enables TF32
                # or another reduced-precision matrix backend. Contraction
                # supplies only the surrogate gradient of the hard selection.
                mask = bank[actual_glyph] + (mask - mask.detach())
            color = _playback_color(color_parameter, neutral_ink=neutral_ink)
            if perceptual_loss and not use_shortlist:
                rendered = _render(bank[actual_glyph], color, rows, cols, cell_height, cell_width, background_color)
                fitted_pixels = torch.einsum("nk,nkc,kp->npc", weights,
                                            candidate_colors - background_color, bank) + background_color
                fitted_frame = fitted_pixels.reshape(rows, cols, cell_height, cell_width, 3).permute(
                    4, 0, 2, 1, 3).reshape_as(rendered)
                rendered = rendered + (fitted_frame - fitted_frame.detach())
            else:
                rendered = _render(mask, color, rows, cols, cell_height, cell_width, background_color)
            components = {} if diagnostics is not None else None
            frame_value = frame_loss(rendered, components)
            loss = frame_value
            training_terms = dict(components) if diagnostics is not None else None
            def training_term(name, value):
                nonlocal loss
                loss = loss + value
                if training_terms is not None:
                    training_terms[name] = value
            if glyph_luminance_bonus:
                ink_presence = (weights * candidate_ink_presence).sum(1)
                training_term("ink_reward", -glyph_luminance_bonus * _ink_lightness_reward(color, ink_presence).mean())
            if glyph_entropy_strength:
                training_term("glyph_entropy", glyph_entropy_strength * _negative_glyph_entropy(
                    _glyph_distribution(weights, indices, len(bank))))
            if stroke_costs is not None:
                training_term("glyph_stroke_alignment", PERCEPTUAL_GLYPH_STROKE_WEIGHT *
                              (weights * stroke_costs.gather(1, indices)).sum(1).mean())
            if flow_costs is not None:
                training_term("glyph_flow", flow_strength *
                              (weights * flow_costs.gather(1, indices)).sum(1).mean())
            if prev_masks is not None:
                training_term("transition_mask", 0.025 * (mask - prev_masks).square().mean())
                training_term("transition_color", 0.010 * (color - prev_colors).square().mean())
                training_term("transition_schedule", 0.008 * (mask - scheduled_masks).square().mean())
            gradients = None
            if diagnostics is not None and (step % diagnostics.gradient_sample_every == 0 or step == steps - 1):
                gradients = diagnostics.sample_gradients({**training_terms, "total": loss},
                                                         {"logits": logits, "ink_rgb": color_parameter})
            if diagnostics is not None:
                # Hard forward already evaluated these exact pixels. Reuse that
                # frame loss instead of rendering and tone-mapping the frame twice.
                with torch.no_grad():
                    if _hard_weights is _HARD_WEIGHTS_IMPLEMENTATION:
                        score = score_terms(frame_value.detach(), mask.detach(), color.detach(), actual_glyph, components)
                    else:
                        # Comparison scripts can replace hard forward with blending.
                        components = {}
                        score = hard_score(bank[actual_glyph], color.detach(), actual_glyph, components)
                    observe("adam_step", actual_glyph, color, logged_logits=logits, temperature=temperature,
                            score=score, components=components, step=step, gradients=gradients)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                color_parameter.clamp_(0, 1)
            if on_progress is not None:
                on_progress((step + 1) / total_work, "Optimizing glyphs and colors")

    with torch.no_grad():
        if diagnostics is not None:
            final_glyph = indices.gather(1, logits.argmax(1)[:, None]).squeeze(1)
            observe("solver_final", final_glyph, _playback_color(color_parameter.detach(), neutral_ink=neutral_ink),
                    logged_logits=logits, temperature=0.3 if steps > 1 else 1., step=steps)
        if not postprocess:
            glyph = indices.gather(1, logits.argmax(1)[:, None]).squeeze(1)
            hard_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
            observe("final", glyph, hard_color)
            if on_progress is not None:
                on_progress(1.0, "Finished optimizing glyphs and colors")
            return Grid(glyph.cpu().numpy().reshape(rows, cols).astype(np.uint8),
                        quantize_playback_lab(hard_color.cpu().numpy().reshape(rows, cols, 3)),
                        quantize_playback_lab(background_color.detach().cpu().numpy()))
        selection_logits = logits
        if logit_noise_multiplier:
            candidate_noise = _post_logit_noise(
                rows, cols, logits.shape[1], seed,
                logit_noise_distribution, device=device, dtype=logits.dtype,
            )
            selection_logits = logits + logit_noise_multiplier * candidate_noise
            if diagnostics is not None:
                diagnostics.register(selection_noise=logit_noise_multiplier * candidate_noise)
        selected = selection_logits.argmax(1)
        glyph = indices.gather(1, selected[:, None]).squeeze(1)
        hard_masks = bank[glyph]
        hard_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
        observe("after_dither", glyph, hard_color)

    observe("pre_refinement", glyph, hard_color)

    # Geometry is known on the CPU; avoid GPU nonzero and its shape sync for
    # the identical parity groups on every sweep.
    parity_map = (np.arange(rows)[:, None] % 2) * 2 + np.arange(cols)[None, :] % 2
    parity_groups = [torch.as_tensor(np.flatnonzero(parity_map == parity), device=device)
                     for parity in range(4)]
    completed_work = steps

    def move_penalty(cell_ids, proposed_glyph, proposed_masks, proposed_color):
        """Exact changes to terms not represented by the rendered pixel image."""
        penalty = proposed_color.new_zeros(proposed_glyph.shape, dtype=torch.float64)
        if glyph_entropy_strength:
            penalty = penalty + glyph_entropy_strength * _glyph_entropy_move_cost(
                glyph, proposed_glyph, current_counts, cell_ids)
        if glyph_luminance_bonus:
            penalty = penalty - glyph_luminance_bonus * (
                _ink_lightness_reward(proposed_color, proposed_masks.gt(0).any(2))
                - _ink_lightness_reward(hard_color[cell_ids], hard_masks[cell_ids].gt(0).any(1))[:, None]
            ) / len(glyph)
        if prev_masks is not None:
            penalty = penalty + 0.025 * (
                (proposed_masks - prev_masks[cell_ids, None]).square().mean(2)
                - (hard_masks[cell_ids] - prev_masks[cell_ids]).square().mean(1)[:, None]) / len(glyph)
            penalty = penalty + 0.008 * (
                (proposed_masks - scheduled_masks[cell_ids, None]).square().mean(2)
                - (hard_masks[cell_ids] - scheduled_masks[cell_ids]).square().mean(1)[:, None]) / len(glyph)
            penalty = penalty + 0.010 * (
                (proposed_color - prev_colors[cell_ids, None]).square().mean(2)
                - (hard_color[cell_ids] - prev_colors[cell_ids]).square().mean(1)[:, None]) / len(glyph)
        return penalty

    def replacement_pixels(masks, colors):
        return (masks[..., None] * colors[:, :, None]
                + (1 - masks[..., None]) * background_color).reshape(
                    len(masks), -1, cell_height, cell_width, 3).permute(0, 1, 4, 2, 3)

    @torch.no_grad()
    def log_replacements(stage):
        if diagnostics is None or perceptual_loss or not getattr(diagnostics, "include_replacements", True):
            return
        nonlocal current_counts
        current_counts = _glyph_counts(glyph, len(bank), hard_color.dtype) if glyph_entropy_strength else None
        rendered = _render(hard_masks, hard_color, rows, cols, cell_height, cell_width, background_color)
        state = _ReplacementState(frame_objective, rendered)
        deltas = hard_color.new_empty(indices.shape, dtype=torch.float64)
        cell_area = (cell_height + 16) * (cell_width + 16)
        chunk_size = max(1, min(256, 1_000_000 // (indices.shape[1] * cell_area)))
        for ids in torch.arange(len(glyph), device=device).split(chunk_size):
            context = _ReplacementContext(frame_objective, rendered, ids, state)
            colors = hard_color[ids, None].expand(-1, indices.shape[1], -1)
            pixels = replacement_pixels(sample_masks[ids], colors)
            deltas[ids] = context.scores(pixels) + move_penalty(ids, indices[ids], sample_masks[ids], colors)
        diagnostics.record_replacements(stage, deltas, glyph)

    current_counts = None
    log_replacements("pre_refinement")

    def polish_colors(count, report_progress=False, phase="sweep_polish"):
        nonlocal hard_color
        if count == 0:
            return
        with section("color_polish", phase=phase, count=count):
            # Refit ink to the final glyph geometry before polishing. A glyph
            # changed by Adam or dither can need a very different ink than the
            # shared color learned for its previous shape.
            with torch.no_grad():
                if perceptual_loss:
                    fitted_ink = spatial_colors[torch.arange(len(glyph), device=device), glyph]
                else:
                    den = hard_masks.square().sum(1).clamp_min(1e-8)
                    fitted_ink = ((cells - background_cells) * hard_masks[..., None]
                                  + background_cells * hard_masks.square()[..., None]).sum(1) / den[:, None]
                    fitted_ink = _project_oklab_colors(fitted_ink)
                fitted_ink = _playback_color(_color_parameter(fitted_ink), neutral_ink=neutral_ink)
                if hard_score(hard_masks, fitted_ink, glyph) < hard_score(hard_masks, hard_color, glyph):
                    hard_color = fitted_ink
            parameter = _color_parameter(hard_color)
            color_optimizer = torch.optim.Adam((parameter,), lr=0.003 if perceptual_loss else 0.01)
            best_color = hard_color.detach().clone()
            best_score = hard_score(hard_masks, best_color, glyph).detach()
            for polish_step in range(count):
                color_optimizer.zero_grad(set_to_none=True)
                color = _playback_color(parameter, neutral_ink=neutral_ink)
                components = {} if diagnostics is not None else None
                loss = hard_score(hard_masks, color, glyph, components)
                better = loss.detach() < best_score
                best_color = torch.where(better, color.detach(), best_color)
                best_score = torch.minimum(best_score, loss.detach())
                observe(phase, glyph, color, score=loss, components=components, color_step=polish_step)
                loss.backward()
                color_optimizer.step()
                with torch.no_grad():
                    parameter.clamp_(0, 1)
                if report_progress and on_progress is not None:
                    on_progress((completed_work + polish_step + 1) / total_work, "Polishing colors")
            with torch.no_grad():
                color = _playback_color(parameter, neutral_ink=neutral_ink)
                score = hard_score(hard_masks, color, glyph)
                best_color = torch.where(score < best_score, color, best_color)
            hard_color = best_color

    for sweep in range(refinement_sweeps):
        sweep_score = hard_score(hard_masks, hard_color, glyph).detach()
        for parity in range(4):
            with section("prepare_responses", sweep=sweep, parity=parity):
                group_ids = parity_groups[parity]
                current_counts = _glyph_counts(glyph, len(bank), hard_color.dtype) if glyph_entropy_strength else None
                rendered = _render(hard_masks, hard_color, rows, cols, cell_height, cell_width, background_color).detach()
                state = _ReplacementState(frame_objective, rendered)
                with torch.no_grad():
                    ink_table = (_display_ink_table(hard_color, background_color, coverage_values, invert_ink=invert_ink)
                                 if coverage_values is not None else None)
                proposed_glyph, proposed_color = glyph.clone(), hard_color.clone()
                improving_ids = []
                proposed_gain = hard_color.new_full((len(glyph),), float("inf"), dtype=torch.float64)
            # Include both the pixel-fitted ink and the current context-fitted
            # ink for every glyph, without restricting the glyph bank.
            halo = 16 if detail_loss == "chroma_contrast" else 4
            cell_area = (cell_height + halo) * (cell_width + halo)
            chunk_size = max(1, min(256, 1_000_000 // (2 * indices.shape[1] * cell_area)))
            for cell_ids in group_ids.split(chunk_size):
                with section("candidate_ranking", sweep=sweep, parity=parity):
                    if not len(cell_ids):
                        continue
                    context = _ReplacementContext(frame_objective, rendered, cell_ids, state)
                    with torch.no_grad():
                        proposal_indices = indices[cell_ids].repeat(1, 2)
                        proposal_masks = sample_masks[cell_ids].repeat(1, 2, 1)
                        colors = torch.cat((candidate_colors[cell_ids],
                                            hard_color[cell_ids, None].expand(-1, indices.shape[1], -1)), dim=1)
                        pixels = replacement_pixels(proposal_masks, colors)
                        displayed_pixels = None
                        if ink_table is not None:
                            count = indices.shape[1]
                            fitted_display = (candidate_displayed[cell_ids] if candidate_displayed is not None
                                else _display_headroom_lab(pixels[:, :count].flatten(0, 1), invert_ink=invert_ink).reshape(
                                    len(cell_ids), count, 3, cell_height, cell_width))
                            lookup_indices = coverage_indices[indices[cell_ids]].flatten(2)
                            current_display = ink_table[cell_ids, None].expand(-1, count, -1, -1).gather(
                                -1, lookup_indices[:, :, None].expand(-1, -1, 3, -1)).reshape(
                                    len(cell_ids), count, 3, cell_height, cell_width)
                            displayed_pixels = torch.cat((fitted_display, current_display), dim=1)
                        gains = context.scores(pixels, displayed_pixels)
                        gains = gains + move_penalty(cell_ids, proposal_indices, proposal_masks, colors)
                        selected = gains.argmin(1)
                        batch_ids = torch.arange(len(cell_ids), device=device)
                        proposed_glyph[cell_ids] = proposal_indices[batch_ids, selected]
                        proposed_color[cell_ids] = colors[batch_ids, selected]
                        proposed_gain[cell_ids] = gains[batch_ids, selected]
            # Rank every glyph in bounded batches, then fit the selected inks
            # together. Tiny per-ranking-batch Adam/gradient passes would repeat
            # display mapping and dispatch hundreds of small kernels.
            fit_chunk_size = max(1, min(4096, 2_000_000 // cell_area))
            for cell_ids in group_ids.split(fit_chunk_size):
                with section("candidate_color_fit", sweep=sweep, parity=parity):
                    if not len(cell_ids):
                        continue
                    context = _ReplacementContext(frame_objective, rendered, cell_ids, state)
                    selected_glyph = proposed_glyph[cell_ids, None]
                    selected_mask = bank[selected_glyph]
                    selected_color = proposed_color[cell_ids, None].clone()
                    selected_gain = proposed_gain[cell_ids].clone()
                    parameter = _color_parameter(selected_color)
                    fitter = torch.optim.Adam((parameter,), lr=0.01)
                    for _ in range(2):
                        fitter.zero_grad(set_to_none=True)
                        color = _playback_color(parameter, neutral_ink=neutral_ink)
                        gain = (context.scores(replacement_pixels(selected_mask, color))
                                + move_penalty(cell_ids, selected_glyph, selected_mask, color)).squeeze(1)
                        with torch.no_grad():
                            better = gain < selected_gain
                            selected_color = torch.where(better[:, None, None], color.detach(), selected_color)
                            selected_gain = torch.minimum(selected_gain, gain.detach())
                        (gain.sum() * len(glyph)).backward()
                        fitter.step()
                        with torch.no_grad():
                            parameter.clamp_(0, 1)
                    with torch.no_grad():
                        color = _playback_color(parameter, neutral_ink=neutral_ink)
                        gain = (context.scores(replacement_pixels(selected_mask, color))
                                + move_penalty(cell_ids, selected_glyph, selected_mask, color)).squeeze(1)
                        better = gain < selected_gain
                        selected_color = torch.where(better[:, None, None], color, selected_color)
                        selected_gain = torch.minimum(selected_gain, gain)
                        changes = selected_gain < -1e-10
                        ids = cell_ids[changes]
                        proposed_color[cell_ids] = selected_color[:, 0]
                        improving_ids.append(ids)
            with torch.no_grad():
                if improving_ids:
                    with section("acceptance", sweep=sweep, parity=parity):
                        changes = torch.cat(improving_ids)
                        if len(changes):
                            cached_score = _ReplacementBatchScorer(
                                state, bank, glyph, hard_color, proposed_glyph, proposed_color, changes, background_color)
                            def acceptance_score(chosen_masks, chosen_color, chosen_glyph):
                                return score_terms(cached_score(chosen_glyph, chosen_color),
                                                   chosen_masks, chosen_color, chosen_glyph)
                            glyph, hard_color = _accept_discrete_proposals(
                                glyph, hard_color, bank, changes, proposed_glyph, proposed_color, acceptance_score,
                                diagnostics=diagnostics)
                            hard_masks = bank[glyph]
            observe("refinement_group", glyph, hard_color, sweep=sweep, parity=parity)
            completed_work += 1
            if on_progress is not None:
                on_progress(completed_work / total_work, "Scoring glyphs with neighboring pixels")
        polish_colors(2)
        observe("refinement_sweep", glyph, hard_color, sweep=sweep)
        completed_work += 2
        if on_progress is not None:
            on_progress(completed_work / total_work, "Refitting ink with discrete glyphs")
        if hard_score(hard_masks, hard_color, glyph).detach() >= sweep_score - 1e-10:
            break
    # Account for skipped sweeps without making progress move backward.
    completed_work = steps + 6 * refinement_sweeps
    observe("post_refinement", glyph, hard_color)
    log_replacements("post_refinement")

    # Retain the best RGB8 colors, including the pre-polish fit.
    polish_colors(polish_steps, report_progress=True, phase="final_polish")
    observe("final", glyph, hard_color)
    if on_progress is not None:
        on_progress(1.0, "Finished optimizing glyphs and colors")

    return Grid(glyph.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8),
                quantize_playback_lab(hard_color.cpu().numpy().reshape(rows, cols, 3)),
                quantize_playback_lab(background_color.detach().cpu().numpy()))
