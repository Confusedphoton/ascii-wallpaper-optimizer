"""Joint glyph, color, and SDF stroke-weight optimization over the complete frame."""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import (
    blue_noise,
    decode_stroke_weight_u8,
    encode_stroke_weight_u8,
    linear_to_srgb,
    sdf_to_mask,
    srgb_to_linear,
)
from .defaults import (
    DEFAULT_GLYPH_ENTROPY_STRENGTH,
    DEFAULT_PERCEPTUAL_LOSS,
    PERCEPTUAL_CHROMA_WEIGHT,
    PERCEPTUAL_GLYPH_STROKE_WEIGHT,
    PERCEPTUAL_PIXEL_WEIGHT,
    PERCEPTUAL_WEIGHTS,
)
from .diagnostics import SolverTrace
from .perceptual import SpatialMetrics, ppd_for_width
from .settings import finite_float
from .tone import encode_headroom_lab, quantize_playback_lab, srgb_to_oklab

DITHER_TO_LOGIT_SCALE = 100.0


@dataclass
class Grid:
    glyph: np.ndarray  # rows x columns, uint8
    color: np.ndarray  # rows x columns x OKLab, float32
    background: np.ndarray | None = None  # one OKLab color for the complete image
    weight: np.ndarray | None = None  # optional rows x columns SDF stroke weight in [-1, 1], float32


def _cells(image: torch.Tensor, rows: int, cols: int, height: int, width: int) -> torch.Tensor:
    return image.permute(1, 2, 0).reshape(rows, height, cols, width, 3).permute(0, 2, 1, 3, 4).reshape(-1, height * width, 3)


def _render(mask: torch.Tensor, color: torch.Tensor, rows: int, cols: int,
            height: int, width: int, background: torch.Tensor | None = None) -> torch.Tensor:
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
    sampled = F.avg_pool2d(image.unsqueeze(0), 4, ceil_mode=True).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    lightness = sampled[:, 0]
    background = sampled[lightness <= torch.quantile(lightness, 0.1)].median(0).values
    return background * (0.18 / background[0].clamp_min(1e-8)).clamp(max=1)


def _glyph_distribution(weights: torch.Tensor, indices: torch.Tensor,
                        glyph_count: int) -> torch.Tensor:
    """Pool candidate probabilities by glyph identity across the entire frame."""
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


def _ink_lightness_reward(color: torch.Tensor, ink_presence: torch.Tensor) -> torch.Tensor:
    """Reward brighter ink with a marginal gain of (1 - L)^2."""
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
    if neutral_ink:
        srgb = srgb.mean(-1, keepdim=True).expand_as(srgb)
    srgb = srgb + (srgb.clamp(0, 1) - srgb).detach()
    quantized = (srgb * 255).round() / 255
    srgb = srgb + (quantized - srgb).detach()
    return _rgb_to_oklab(srgb_to_linear(srgb))


def _playback_weight(weight: torch.Tensor) -> torch.Tensor:
    """Quantize stroke weight in [-1, 1] to glyph.png G-channel bytes with STE."""
    clamped = weight + (weight.clamp(-1.0, 1.0) - weight).detach()
    quantized = ((128.0 + clamped * 127.0).round().clamp(0.0, 255.0) - 128.0) / 127.0
    return clamped + (quantized - clamped).detach()


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
    """Compose using RGB8 ink while retaining a useful straight-through gradient."""
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
            clipped = ((rgb < 0) | (rgb > 1)).any(-1)
            if bool(clipped.any()):
                projected[clipped] = _rgb_to_oklab(rgb[clipped].clamp(0, 1))
    return color + (projected - color).detach()


def _saturation_loss(rendered: torch.Tensor, target_lab: torch.Tensor,
                     chroma_scale: torch.Tensor | None = None, *,
                     invert_ink: bool = False, displayed_lab: torch.Tensor | None = None,
                     source_chroma: torch.Tensor | None = None,
                     source_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetrically preserve source C/L and hue in squared OKLab units."""
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
    """Preserve pastel chroma as well as saturated colors, without a gray target."""
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


def _post_logit_noise(rows: int, cols: int, candidates: int, seed: int, *,
                      device: str, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic blue-noise perturbations per cell and candidate glyph."""
    uniform = torch.as_tensor(np.stack([
        blue_noise((rows, cols), seed + 1 + candidate * 104729)
        for candidate in range(candidates)
    ], axis=-1), dtype=dtype, device=device).reshape(-1, candidates)
    return uniform - 0.5


def _loss(rendered: torch.Tensor, target: torch.Tensor, *,
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
    loss = residual.square().sum(1, keepdim=True).mean()
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


@dataclass
class _FrameObjective:
    """Constant image-derived inputs shared by every optimization step."""
    target: torch.Tensor
    cell: tuple[int, int]
    invert_ink: bool
    gradient_strength: float
    saturation_strength: float
    gradient_target: torch.Tensor | None = None
    saturation_target: torch.Tensor | None = None
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
        if displayed is None and (self.gradient_strength or self.saturation_strength or self.spatial_metrics is not None):
            displayed = _display_headroom_lab(rendered, invert_ink=self.invert_ink)
        pixel_components = {} if components is not None else None
        loss = _loss(rendered, self.target,
                     target_pyramid=self.target_pyramid,
                     gradient_target_lab=self.gradient_target,
                     gradient_target_gradients=self.gradient_target_gradients,
                     gradient_loss_strength=self.gradient_strength,
                     invert_ink=self.invert_ink, displayed_lab=displayed,
                     components=pixel_components)
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


def render_grid(grid: Grid, masks: np.ndarray, device: str = "cpu", *,
                sdf_spread: float = 0.0) -> np.ndarray:
    rows, cols = grid.glyph.shape
    height, width = masks.shape[1:]
    with torch.no_grad():
        bank = torch.as_tensor(masks, dtype=torch.float32, device=device).flatten(1)
        glyph = torch.as_tensor(grid.glyph.astype(np.int64), device=device).flatten()
        color = torch.as_tensor(grid.color, dtype=torch.float32, device=device).reshape(-1, 3)
        background = (torch.as_tensor(grid.background, dtype=torch.float32, device=device).reshape(3)
                      if grid.background is not None else None)
        cell_masks = bank[glyph]
        if sdf_spread > 0.0:
            weight = (torch.as_tensor(grid.weight, dtype=torch.float32, device=device).reshape(-1, 1)
                      if grid.weight is not None else None)
            cell_masks = sdf_to_mask(cell_masks, weight, spread=sdf_spread)
        image = _render(cell_masks, color, rows, cols, height, width, background)
    return image.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _make_spatial_metrics(image_srgb, *, source_srgb, source_oklab, input_oklab,
                          pad, cell_height, device, viewing_ppd, masks, background,
                          invert_ink, sdf_spread: float = 0.0):
    """Map source contrast into the atlas's displayed range once per solve."""
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
        if sdf_spread > 0.0:
            # Calibrate target range at neutral SDF weight so bright colored regions
            # stay saturated and use SDF bolding to gain headroom without bleaching white.
            atlas = sdf_to_mask(atlas, 0.0, spread=sdf_spread)
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
    """Fit each glyph to the same local linear-light mean and source hue."""
    height, width = masks.shape[-2:]
    rows, cols = model.source_rgb.shape[-2] // height, model.source_rgb.shape[-1] // width
    mean_rgb = _cells(model.source_rgb.squeeze(0), rows, cols, height, width).mean(1)
    target_y = mean_rgb @ mean_rgb.new_tensor([0.2126, 0.7152, 0.0722])
    levels = torch.linspace(0, 1, 65, device=masks.device, dtype=masks.dtype)
    ink = torch.stack((levels, levels * 0, levels * 0), 1)
    response = masks.new_empty(len(masks) * len(levels))
    tiles_per_batch = max(1, 1_000_000 // (height * width))
    for start in range(0, len(response), tiles_per_batch):
        ids = torch.arange(start, min(start + tiles_per_batch, len(response)), device=masks.device)
        alpha = masks[ids // len(levels), None]
        pixels = alpha * ink[ids % len(levels), :, None, None] + (1 - alpha) * background[None, :, None, None]
        shown = _display_headroom_lab(pixels, invert_ink=invert_ink)
        rgb = _oklab_to_rgb(shown.permute(0, 2, 3, 1)).mean((1, 2))
        response[start:start + len(ids)] = rgb @ rgb.new_tensor([0.2126, 0.7152, 0.0722])
    response = response.reshape(len(masks), -1)
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
                     glyph_luminance_bonus: float = 0.0,
                     saturation_loss_strength: float | None = None,
                     saturation_target_srgb: np.ndarray | None = None,
                     saturation_target_oklab: np.ndarray | None = None,
                     gradient_loss_strength: float = 1.5,
                     gradient_target_srgb: np.ndarray | None = None,
                     gradient_target_oklab: np.ndarray | None = None,
                     perceptual_loss: bool = DEFAULT_PERCEPTUAL_LOSS,
                     perceptual_target_srgb: np.ndarray | None = None,
                     viewing_ppd: float | None = None,
                     invert_ink: bool = False, input_oklab: bool = False,
                     sdf_spread: float = 0.0,
                     initial_weight: np.ndarray | None = None,
                     solve_context: dict | None = None,
                     temporal_neighbors: tuple[Grid, ...] = (), temporal_strength: float = 0.0) -> Grid:
    """Fit playback ink colors (and SDF stroke weights when active) for fixed glyphs."""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    glyph_luminance_bonus = finite_float(glyph_luminance_bonus, "glyph_luminance_bonus", minimum=0)
    temporal_strength = finite_float(temporal_strength, "temporal_strength", 0, 2)
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
    configuration = (device, invert_ink, perceptual_loss, input_oklab, viewing_ppd,
                     gradient_loss_strength, saturation_loss_strength, sdf_spread)
    sources = (image_srgb, masks_np, background_oklab, gradient_target_srgb, gradient_target_oklab,
               saturation_target_srgb, saturation_target_oklab, perceptual_target_srgb)
    prepared = solve_context.get("fixed_glyph_objective") if solve_context is not None else None
    if (prepared is None or configuration != prepared[0]
            or any(current is not original for current, original in zip(sources, prepared[1]))):
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
        spatial_metrics = solve_context.get("spatial_metrics") if solve_context is not None else None
        if perceptual_loss and spatial_metrics is None:
            spatial_metrics = _make_spatial_metrics(
                image_srgb,
                source_srgb=(perceptual_target_srgb if perceptual_target_srgb is not None
                             else gradient_target_srgb if gradient_target_srgb is not None
                             else saturation_target_srgb),
                source_oklab=(gradient_target_oklab if gradient_target_oklab is not None
                              else saturation_target_oklab),
                input_oklab=input_oklab, pad=pad, cell_height=height, device=device,
                viewing_ppd=viewing_ppd, masks=masks_np, background=background,
                invert_ink=invert_ink, sdf_spread=sdf_spread)
            if solve_context is not None:
                solve_context["spatial_metrics"] = spatial_metrics
        objective = _FrameObjective(target, (height, width), invert_ink,
                                    gradient_loss_strength, saturation_loss_strength,
                                    gradient_target, saturation_target,
                                    spatial_metrics=spatial_metrics if perceptual_loss else None)
        if solve_context is not None:
            solve_context["fixed_glyph_objective"] = (configuration, sources, objective, background)
    else:
        _, _, objective, background = prepared

    neutral_ink = (perceptual_loss and objective.spatial_metrics.neutral_source
                   and bool(background[1:].abs().amax() < 1e-6))
    atlas_bank = torch.as_tensor(masks_np, dtype=torch.float32, device=device).flatten(1)
    neutral_bank = sdf_to_mask(atlas_bank, spread=sdf_spread) if sdf_spread > 0.0 else atlas_bank
    glyph_ids = torch.as_tensor(glyphs.astype(np.int64), device=device).flatten()
    cell_sdf = atlas_bank[glyph_ids]
    neutral_mask = neutral_bank[glyph_ids]
    ink_presence = neutral_mask.gt(0).any(1)

    if perceptual_loss and solve_context is not None and "candidate_colors" in solve_context:
        source_colors = solve_context["candidate_colors"]
        cached_colors = solve_context.get("candidate_colors_device")
        if cached_colors is None or cached_colors[0] is not source_colors or cached_colors[1] != device:
            candidates = torch.as_tensor(source_colors, dtype=torch.float32, device=device)
            solve_context["candidate_colors_device"] = (source_colors, device, candidates)
        else:
            candidates = cached_colors[2]
        if candidates.shape != (rows * cols, len(masks_np), 3):
            raise ValueError("Cached candidate colors must match the source grid and atlas")
        initial_color = candidates[torch.arange(rows * cols, device=device), glyph_ids]
    else:
        cells = _cells(objective.target.squeeze(0), rows, cols, height, width)
        numerator = (neutral_mask[..., None] * (cells - (1 - neutral_mask[..., None]) * background)).sum(1)
        denominator = neutral_mask.square().sum(1).clamp_min(1e-8)
        initial_color = numerator / denominator[..., None]

    color_parameter = _color_parameter(initial_color)
    weight_parameter = None
    param_groups = [{"params": (color_parameter,), "lr": 0.003 if perceptual_loss else 0.01}]
    if sdf_spread > 0.0:
        w0 = (torch.as_tensor(initial_weight, dtype=torch.float32, device=device).reshape(-1)
              if initial_weight is not None else torch.zeros(rows * cols, dtype=torch.float32, device=device))
        weight_parameter = torch.nn.Parameter(w0.clamp(-1.0, 1.0))
        param_groups.append({"params": (weight_parameter,), "lr": 0.03})

    previous_colors = (torch.as_tensor(previous.color, dtype=torch.float32,
                                       device=device).reshape(-1, 3)
                       if previous is not None else None)
    scheduled_colors = None
    if previous is not None and destination is not None and progress is not None:
        scheduled_colors = ((1 - float(progress)) * previous_colors
                            + float(progress) * torch.as_tensor(destination.color,
                                dtype=torch.float32, device=device).reshape(-1, 3))

    temporal_means = []
    for reference in temporal_neighbors if temporal_strength else ():
        if reference.glyph.shape != (rows, cols) or reference.color.shape != (rows, cols, 3):
            raise ValueError("Temporal references must use the same glyph grid")
        ref_ids = torch.as_tensor(reference.glyph.astype(np.int64), device=device).flatten()
        ref_masks = atlas_bank[ref_ids]
        if sdf_spread > 0.0:
            ref_w = (torch.as_tensor(reference.weight, dtype=torch.float32, device=device).reshape(-1, 1)
                     if reference.weight is not None else None)
            ref_masks = sdf_to_mask(ref_masks, ref_w, spread=sdf_spread)
        ref_colors = torch.as_tensor(reference.color, device=device).reshape(-1, 3)
        ref_frame = _render(ref_masks, ref_colors, rows, cols, height, width, background)
        temporal_means.append(F.adaptive_avg_pool2d(_display_headroom_lab(ref_frame, invert_ink=invert_ink),
                                                    (rows, cols)).detach())

    def current_mask(w: torch.Tensor | None) -> torch.Tensor:
        if sdf_spread <= 0.0:
            return neutral_mask
        return sdf_to_mask(cell_sdf, w[:, None], spread=sdf_spread)

    def color_loss(color: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        rendered = _render(current_mask(w), color, rows, cols, height, width, background)
        displayed = _display_headroom_lab(rendered, invert_ink=invert_ink) if temporal_means else None
        loss = objective(rendered, displayed_lab=displayed)
        if temporal_means:
            means = F.adaptive_avg_pool2d(displayed, (rows, cols))
            loss = loss + temporal_strength * sum((means - ref).square().mean() for ref in temporal_means) / len(temporal_means)
        if glyph_luminance_bonus:
            loss = loss - glyph_luminance_bonus * _ink_lightness_reward(color, ink_presence).mean()
        if previous_colors is not None:
            loss = loss + 0.010 * (color - previous_colors).square().mean()
        if scheduled_colors is not None:
            loss = loss + 0.004 * (color - scheduled_colors).square().mean()
        return loss

    best_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
    best_weight = _playback_weight(weight_parameter.detach()) if weight_parameter is not None else None
    best_loss = color_loss(best_color, best_weight).detach()
    optimizer = torch.optim.Adam(param_groups)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        color = _playback_color(color_parameter, neutral_ink=neutral_ink)
        weight = _playback_weight(weight_parameter) if weight_parameter is not None else None
        loss = color_loss(color, weight)
        better = loss.detach() < best_loss
        best_color = torch.where(better, color.detach(), best_color)
        if best_weight is not None and weight is not None:
            best_weight = torch.where(better, weight.detach(), best_weight)
        best_loss = torch.minimum(best_loss, loss.detach())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            color_parameter.clamp_(0, 1)
            if weight_parameter is not None:
                weight_parameter.clamp_(-1.0, 1.0)
        if on_progress is not None:
            on_progress((step + 1) / max(steps, 1), "Fitting path glyph colors")
    with torch.no_grad():
        color = _playback_color(color_parameter, neutral_ink=neutral_ink)
        weight = _playback_weight(weight_parameter) if weight_parameter is not None else None
        better = color_loss(color, weight) < best_loss
        best_color = torch.where(better, color, best_color)
        if best_weight is not None and weight is not None:
            best_weight = torch.where(better, weight, best_weight)
    if on_progress is not None and steps == 0:
        on_progress(1.0, "Fitting path glyph colors")
    final_weight = (decode_stroke_weight_u8(encode_stroke_weight_u8(
        best_weight.cpu().numpy().reshape(rows, cols)))
        if best_weight is not None else None)
    return Grid(glyphs.astype(np.uint8).copy(),
                quantize_playback_lab(best_color.cpu().numpy().reshape(rows, cols, 3)),
                quantize_playback_lab(background.cpu().numpy()),
                weight=final_weight)


def solve(image_srgb: np.ndarray, masks_np: np.ndarray, *, seed: int = 0,
          steps: int = 36, dither: float = 0.005,
          background: str = "auto", background_oklab: np.ndarray | None = None,
          device: str = "cpu", previous: Grid | None = None,
          destination: Grid | None = None, progress: float | None = None,
          on_progress: Callable[[float, str], None] | None = None,
          glyph_luminance_bonus: float = 0.0,
          saturation_loss_strength: float | None = None,
          saturation_target_srgb: np.ndarray | None = None,
          saturation_target_oklab: np.ndarray | None = None,
          gradient_loss_strength: float = 1.5,
          gradient_target_srgb: np.ndarray | None = None,
          gradient_target_oklab: np.ndarray | None = None,
          glyph_entropy_strength: float = DEFAULT_GLYPH_ENTROPY_STRENGTH,
          postprocess: bool | None = None,
          perceptual_loss: bool = DEFAULT_PERCEPTUAL_LOSS,
          perceptual_target_srgb: np.ndarray | None = None,
          viewing_ppd: float | None = None,
          flow_costs: torch.Tensor | np.ndarray | None = None,
          flow_strength: float = 0.0,
          sdf_spread: float = 0.0,
          solve_context: dict | None = None,
          diagnostics: SolverTrace | None = None,
          invert_ink: bool = False,
          input_oklab: bool = False) -> Grid:
    """Optimize every font glyph per cell with OKLab scoring, bounded RGB ink, and optional SDF weight."""
    if masks_np.shape[0] > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if postprocess is None:
        postprocess = not perceptual_loss
    if not isinstance(postprocess, bool):
        raise ValueError("postprocess must be a boolean")
    if viewing_ppd is not None:
        viewing_ppd = finite_float(viewing_ppd, "viewing_ppd", minimum=1, maximum=1024)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    glyph_luminance_bonus = finite_float(glyph_luminance_bonus, "glyph_luminance_bonus", minimum=0)
    if saturation_loss_strength is None:
        saturation_loss_strength = 20.0 if glyph_luminance_bonus > 0 else 0.0
    saturation_loss_strength = finite_float(saturation_loss_strength, "saturation_loss_strength", minimum=0)
    gradient_loss_strength = finite_float(gradient_loss_strength, "gradient_loss_strength", minimum=0)
    glyph_entropy_strength = finite_float(glyph_entropy_strength, "glyph_entropy_strength", minimum=0)
    flow_strength = finite_float(flow_strength, "flow_strength", 0, 2)
    if flow_strength and flow_costs is None:
        raise ValueError("Flow strength requires flow/glyph costs")
    if flow_strength and not perceptual_loss:
        raise ValueError("Flow glyph selection requires the perceptual full-frame solver")
    dither = finite_float(dither, "dither", minimum=0)
    if not postprocess:
        dither = 0.0
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
                          postprocess=postprocess, dither=dither, cell=[cell_width, cell_height],
                          glyph_count=len(masks_np), sdf_spread=float(sdf_spread))

    def section(name, **metadata):
        return diagnostics.section(name, **metadata) if diagnostics is not None else nullcontext()

    pad_height = rows * cell_height - original_height
    pad_width = cols * cell_width - original_width
    image_lab_np = (np.asarray(image_srgb, dtype=np.float32) if input_oklab
                    else srgb_to_oklab(image_srgb))
    image = torch.as_tensor(image_lab_np, dtype=torch.float32, device=device).permute(2, 0, 1)
    image = F.pad(image.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0)
    target = image.unsqueeze(0)

    def target_tensor(lab: np.ndarray) -> torch.Tensor:
        if lab.shape != image_lab_np.shape:
            raise ValueError("Color objective targets must match the source image shape")
        tensor = torch.as_tensor(lab, dtype=torch.float32, device=device).permute(2, 0, 1)
        return F.pad(tensor.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0)

    saturation_target_lab = None
    if saturation_loss_strength:
        saturation_source = (saturation_target_oklab if saturation_target_oklab is not None
                             else srgb_to_oklab(saturation_target_srgb)
                             if saturation_target_srgb is not None else image_lab_np)
        saturation_target_lab = target_tensor(saturation_source).unsqueeze(0).detach()

    gradient_target_lab = None
    if gradient_loss_strength:
        gradient_source = (gradient_target_oklab if gradient_target_oklab is not None
                           else srgb_to_oklab(gradient_target_srgb)
                           if gradient_target_srgb is not None else image_lab_np)
        gradient_target_lab = target_tensor(gradient_source).unsqueeze(0).detach()

    atlas_tensor = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
    masks = sdf_to_mask(atlas_tensor, spread=sdf_spread) if sdf_spread > 0.0 else atlas_tensor
    sdf_bank = atlas_tensor.flatten(1)
    bank = masks.flatten(1)

    background_color = (torch.as_tensor(background_oklab, dtype=torch.float32, device=device).reshape(3)
                        if background_oklab is not None else _choose_background(image, background))
    background_color = _project_oklab_colors(background_color)
    background_cells = background_color.reshape(1, 1, 3).expand(rows * cols, cell_height * cell_width, 3)

    noise = torch.as_tensor(blue_noise((rows, cols), seed), dtype=torch.float32, device=device).flatten()
    spatial_metrics = None
    if perceptual_loss:
        spatial_metrics = solve_context.get("spatial_metrics") if solve_context is not None else None
        if spatial_metrics is None:
            spatial_metrics = _make_spatial_metrics(
                image_srgb,
                source_srgb=(perceptual_target_srgb if perceptual_target_srgb is not None
                             else gradient_target_srgb if gradient_target_srgb is not None
                             else saturation_target_srgb),
                source_oklab=(gradient_target_oklab if gradient_target_oklab is not None
                              else saturation_target_oklab),
                input_oklab=input_oklab, pad=(0, pad_width, 0, pad_height),
                cell_height=cell_height, device=device, viewing_ppd=viewing_ppd,
                masks=masks_np, background=background_color, invert_ink=invert_ink,
                sdf_spread=sdf_spread)
            if solve_context is not None:
                solve_context["spatial_metrics"] = spatial_metrics

    frame_objective = _FrameObjective(
        target.detach(), (cell_height, cell_width), invert_ink,
        gradient_loss_strength, saturation_loss_strength,
        gradient_target_lab, saturation_target_lab,
        spatial_metrics=spatial_metrics)
    if flow_costs is not None:
        frame_objective.contour_scale = 1 / (1 + 3 * flow_strength)
    neutral_ink = (perceptual_loss and frame_objective.spatial_metrics.neutral_source
                   and bool(background_color[1:].abs().amax() < 1e-6))
    stroke_costs = (frame_objective.spatial_metrics.glyph_stroke_costs(masks)
                    if perceptual_loss and flow_costs is None else None)

    cells = _cells(frame_objective.target.squeeze(0), rows, cols, cell_height, cell_width)
    saturation_target_lab = frame_objective.saturation_target
    saturation_target_cells = (_cells(saturation_target_lab.squeeze(0), rows, cols, cell_height, cell_width)
                               if saturation_target_lab is not None else None)
    saturation_chroma_scale = frame_objective.saturation_scale

    if diagnostics is not None:
        diagnostics.metadata.update(
            coefficients={
                "pixel": 1.0, "pool_2": 1.0, "pool_4": 1.5, "pool_8": 2.0,
                "luminance_horizontal": .35, "luminance_vertical": .35,
                "displayed_sobel": gradient_loss_strength, "saturation": saturation_loss_strength,
                "glyph_entropy": glyph_entropy_strength, "ink_reward": -glyph_luminance_bonus,
                "glyph_flow": flow_strength,
                "transition_mask": .025, "transition_color": .010, "transition_schedule": .008,
                **({**{name: weight * (frame_objective.contour_scale if name != "spatial_lightness" else 1)
                        for name, weight in PERCEPTUAL_WEIGHTS.items()},
                    "glyph_stroke_alignment": PERCEPTUAL_GLYPH_STROKE_WEIGHT if stroke_costs is not None else 0.0}
                   if perceptual_loss else {}),
            },
            perceptual_loss=perceptual_loss,
            neutral_ink=neutral_ink,
            viewing_ppd=frame_objective.spatial_metrics.ppd if perceptual_loss else None,
            sdf_spread=float(sdf_spread),
        )
        if perceptual_loss:
            diagnostics.metadata["coefficients"]["relative_chroma"] = PERCEPTUAL_CHROMA_WEIGHT
            for name in ("pixel", "pool_2", "pool_4", "pool_8", "luminance_horizontal",
                         "luminance_vertical", "displayed_sobel"):
                diagnostics.metadata["coefficients"][name] *= PERCEPTUAL_PIXEL_WEIGHT
            diagnostics.metadata["reference_range_linear_rgb"] = [
                v.cpu().tolist() for v in frame_objective.spatial_metrics.reference_range
            ]
            if stroke_costs is not None:
                diagnostics.register(glyph_stroke_costs=stroke_costs)
            diagnostics.metadata["contour_scale"] = frame_objective.contour_scale
            if flow_costs is not None:
                diagnostics.register(glyph_flow_costs=flow_costs)

    indices = torch.arange(len(bank), device=device).expand(len(cells), -1)
    if previous is not None and destination is not None:
        if previous.glyph.shape != (rows, cols) or destination.glyph.shape != (rows, cols):
            raise ValueError("Transition endpoints must use the same grid")

    sample_masks = bank[None].expand(len(cells), -1, -1)
    candidate_ink_presence = bank.gt(0).any(1)[indices]
    if perceptual_loss:
        spatial_colors = _spatial_ink_fit(frame_objective.spatial_metrics, masks, background_color, invert_ink=invert_ink)
        candidate_colors = spatial_colors
    else:
        candidate_colors = cells.new_empty((*indices.shape, 3))

    tiles_per_batch = max(1, 1_000_000 // bank.shape[1])
    candidates_per_batch = min(indices.shape[1], tiles_per_batch)
    cells_per_batch = max(1, tiles_per_batch // candidates_per_batch)
    initial_errors = cells.new_empty(indices.shape)
    for start in range(0, len(cells), cells_per_batch):
        batch = slice(start, start + cells_per_batch)
        for first_candidate in range(0, indices.shape[1], candidates_per_batch):
            candidates_slice = slice(first_candidate, first_candidate + candidates_per_batch)
            alpha = sample_masks[batch, candidates_slice]
            underpaint = background_cells[batch]
            if not perceptual_loss:
                square = alpha.square()
                difference = cells[batch] - underpaint
                denominator = square.sum(2)
                numerator = (torch.einsum("npc,nkp->nkc", difference, alpha)
                             + torch.einsum("npc,nkp->nkc", underpaint, square))
                candidate_colors[batch, candidates_slice] = _project_oklab_colors(numerator / denominator.clamp_min(1e-8)[..., None])
            colors = candidate_colors[batch, candidates_slice]
            raster = alpha[..., None] * colors[:, :, None, :] + (1 - alpha[..., None]) * underpaint[:, None]
            score = (raster - cells[batch, None]).square().sum(3).mean(2)
            if perceptual_loss:
                score = PERCEPTUAL_PIXEL_WEIGHT * score
            if glyph_luminance_bonus:
                score = score - glyph_luminance_bonus * _ink_lightness_reward(colors, candidate_ink_presence[batch, candidates_slice])
            if saturation_loss_strength:
                count = alpha.shape[1]
                candidate_images = raster.reshape(-1, cell_height, cell_width, 3).permute(0, 3, 1, 2)
                candidate_targets = (saturation_target_cells[batch, None]
                                     .expand(-1, count, -1, -1)
                                     .reshape(-1, cell_height, cell_width, 3)
                                     .permute(0, 3, 1, 2))
                displayed = _display_headroom_lab(candidate_images, invert_ink=invert_ink)
                saturation_errors = _saturation_loss(
                    candidate_images, candidate_targets, saturation_chroma_scale,
                    invert_ink=invert_ink, displayed_lab=displayed).reshape_as(score)
                score = score + saturation_loss_strength * saturation_errors
                if perceptual_loss:
                    relative = _relative_chroma_loss(displayed, candidate_targets).mean((1, 2, 3))
                    score = score + PERCEPTUAL_CHROMA_WEIGHT * relative.reshape_as(score)
            initial_errors[batch, candidates_slice] = score
    del raster
    if saturation_loss_strength:
        del candidate_images, candidate_targets, displayed
    if stroke_costs is not None:
        initial_errors = initial_errors + PERCEPTUAL_GLYPH_STROKE_WEIGHT * stroke_costs
    if solve_context is not None:
        solve_context["image_unary"] = initial_errors.detach().cpu().numpy().copy()
        solve_context["candidate_colors"] = candidate_colors.detach().cpu().numpy().copy()
    if flow_costs is not None:
        initial_errors = initial_errors + flow_strength * flow_costs
    if previous is not None and destination is not None:
        chosen = torch.where(noise < float(progress),
                             torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                             torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        initial_errors = initial_errors + 0.015 * (indices != chosen[:, None]).float()

    logit_scale, logit_floor = (0.1, -4) if perceptual_loss else (0.02, -20)
    logits = torch.nn.Parameter((-(initial_errors - initial_errors.amin(1, keepdim=True)) / logit_scale).clamp(logit_floor, 0).detach())
    first = initial_errors.argmin(1)
    initial_color = _project_oklab_colors(candidate_colors[torch.arange(len(candidate_colors), device=device), first])
    color_parameter = _color_parameter(initial_color)
    weight_parameter = (torch.nn.Parameter(torch.zeros(rows * cols, device=device, dtype=torch.float32))
                        if sdf_spread > 0.0 else None)

    param_groups = [
        {"params": (logits,), "lr": 0.12},
        {"params": (color_parameter,), "lr": 0.01 if perceptual_loss else 0.03},
    ]
    if weight_parameter is not None:
        param_groups.append({"params": (weight_parameter,), "lr": 0.05})
    optimizer = torch.optim.Adam(param_groups)

    prev_masks = prev_colors = scheduled_masks = None
    if previous is not None and destination is not None:
        prev_masks = bank[torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten()]
        prev_colors = torch.as_tensor(previous.color, dtype=torch.float32, device=device).reshape(-1, 3)
        scheduled = torch.where(noise < float(progress),
                                torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                                torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        scheduled_masks = bank[scheduled]

    total_work = max(1, steps)

    def frame_loss(rendered: torch.Tensor, components=None) -> torch.Tensor:
        return (frame_objective(rendered) if components is None
                else frame_objective(rendered, components=components))

    def eval_masks(glyph_ids: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        if sdf_spread <= 0.0:
            return bank[glyph_ids]
        return sdf_to_mask(sdf_bank[glyph_ids], None if w is None else w[:, None], spread=sdf_spread)

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
    def observe(stage, chosen_glyph, chosen_color, *, chosen_weight=None, logged_logits=None,
                temperature=None, score=None, components=None, **metadata):
        if diagnostics is None:
            return
        if score is None:
            components = {}
            score = hard_score(eval_masks(chosen_glyph, chosen_weight), chosen_color, chosen_glyph, components)
        diagnostics.record(stage, loss=score, components=components,
                           glyph=chosen_glyph, color=chosen_color, logits=logged_logits,
                           temperature=temperature, **metadata)

    if diagnostics is not None:
        diagnostics.register(candidate_indices=indices, background=background_color,
                             selection_noise=torch.zeros_like(logits))
        diagnostics.register_geometry(masks, *_sobel_filters(masks[:, None]))
        initial_glyph = logits.argmax(1)
        init_w = _playback_weight(weight_parameter.detach()) if weight_parameter is not None else None
        observe("initial", initial_glyph, _playback_color(color_parameter.detach(), neutral_ink=neutral_ink),
                chosen_weight=init_w, logged_logits=logits, temperature=1., step=0)

    for step in range(steps):
        with section("adam_step", step=step):
            optimizer.zero_grad(set_to_none=True)
            temperature = max(0.3, 1.0 - 0.7 * step / max(1, steps - 1))
            weights = _hard_weights(logits, temperature)
            actual_glyph = weights.argmax(1)
            stroke_weight = _playback_weight(weight_parameter) if weight_parameter is not None else None
            hard_mask = eval_masks(actual_glyph, stroke_weight)
            soft_mask = weights @ bank
            if _hard_weights is _HARD_WEIGHTS_IMPLEMENTATION:
                mask = hard_mask + (soft_mask - soft_mask.detach())
            else:
                mask = soft_mask
            color = _playback_color(color_parameter, neutral_ink=neutral_ink)
            if perceptual_loss:
                rendered = _render(hard_mask, color, rows, cols, cell_height, cell_width, background_color)
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
                              (weights * stroke_costs).sum(1).mean())
            if flow_costs is not None:
                training_term("glyph_flow", flow_strength * (weights * flow_costs).sum(1).mean())
            if prev_masks is not None:
                training_term("transition_mask", 0.025 * (mask - prev_masks).square().mean())
                training_term("transition_color", 0.010 * (color - prev_colors).square().mean())
                training_term("transition_schedule", 0.008 * (mask - scheduled_masks).square().mean())

            gradients = None
            if diagnostics is not None and (step % diagnostics.gradient_sample_every == 0 or step == steps - 1):
                tracked = {"logits": logits, "ink_rgb": color_parameter}
                if weight_parameter is not None:
                    tracked["stroke_weight"] = weight_parameter
                gradients = diagnostics.sample_gradients({**training_terms, "total": loss}, tracked)
            if diagnostics is not None:
                with torch.no_grad():
                    if _hard_weights is _HARD_WEIGHTS_IMPLEMENTATION:
                        score = score_terms(frame_value.detach(), hard_mask.detach(), color.detach(), actual_glyph, components)
                    else:
                        components = {}
                        score = hard_score(hard_mask.detach(), color.detach(), actual_glyph, components)
                    observe("adam_step", actual_glyph, color, chosen_weight=stroke_weight,
                            logged_logits=logits, temperature=temperature,
                            score=score, components=components, step=step, gradients=gradients)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                color_parameter.clamp_(0, 1)
                if weight_parameter is not None:
                    weight_parameter.clamp_(-1.0, 1.0)
            if on_progress is not None:
                on_progress((step + 1) / total_work, "Optimizing glyphs and colors")

    def pack_result(final_glyph: torch.Tensor, final_color: torch.Tensor,
                    final_weight: torch.Tensor | None) -> Grid:
        weight_np = (decode_stroke_weight_u8(encode_stroke_weight_u8(
            final_weight.detach().cpu().numpy().reshape(rows, cols)))
            if final_weight is not None else None)
        return Grid(final_glyph.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8),
                    quantize_playback_lab(final_color.detach().cpu().numpy().reshape(rows, cols, 3)),
                    quantize_playback_lab(background_color.detach().cpu().numpy()),
                    weight=weight_np)

    with torch.no_grad():
        terminal_weight = _playback_weight(weight_parameter.detach()) if weight_parameter is not None else None
        if diagnostics is not None:
            final_glyph = logits.argmax(1)
            observe("solver_final", final_glyph, _playback_color(color_parameter.detach(), neutral_ink=neutral_ink),
                    chosen_weight=terminal_weight, logged_logits=logits,
                    temperature=0.3 if steps > 1 else 1., step=steps)
        if not postprocess:
            glyph = logits.argmax(1)
            hard_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
            observe("final", glyph, hard_color, chosen_weight=terminal_weight)
            if on_progress is not None:
                on_progress(1.0, "Finished optimizing glyphs and colors")
            return pack_result(glyph, hard_color, terminal_weight)

        selection_logits = logits
        if logit_noise_multiplier:
            candidate_noise = _post_logit_noise(
                rows, cols, logits.shape[1], seed, device=device, dtype=logits.dtype)
            selection_logits = logits + logit_noise_multiplier * candidate_noise
            if diagnostics is not None:
                diagnostics.register(selection_noise=logit_noise_multiplier * candidate_noise)
        glyph = selection_logits.argmax(1)
        hard_weight = terminal_weight
        hard_color = _playback_color(color_parameter.detach(), neutral_ink=neutral_ink)
        observe("after_dither", glyph, hard_color, chosen_weight=hard_weight)

    observe("final", glyph, hard_color, chosen_weight=hard_weight)
    if on_progress is not None:
        on_progress(1.0, "Finished optimizing glyphs and colors")
    return pack_result(glyph, hard_color, hard_weight)
