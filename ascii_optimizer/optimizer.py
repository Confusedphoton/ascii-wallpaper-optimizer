"""Joint glyph/color optimization over the complete rendered frame."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import blue_noise
from .tone import srgb_to_oklab


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


def _ink_lightness(color: torch.Tensor) -> torch.Tensor:
    """Return the L component of OKLab ink colors."""
    return color[..., 0]


def _project_oklab_colors(color: torch.Tensor) -> torch.Tensor:
    """Keep optimizer parameters in a conservative, bounded OKLab range."""
    return torch.cat((color[..., :1].clamp(0, 1), color[..., 1:].clamp(-0.4, 0.4)), dim=-1)


def _saturation_loss(rendered: torch.Tensor, target_lab: torch.Tensor,
                    chroma_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Penalize loss of source chroma after the OKLab display tone transform."""
    displayed_lab = _display_headroom_lab(rendered)
    source_chroma = (target_lab[:, 1:].square().sum(1, keepdim=True) + 1e-8).sqrt()
    output_chroma = (displayed_lab[:, 1:].square().sum(1, keepdim=True) + 1e-8).sqrt()
    scale = (torch.quantile(source_chroma.detach().flatten(), 0.95).clamp_min(0.02)
             if chroma_scale is None else chroma_scale)
    weight = 1 + 4 * (source_chroma / scale).clamp(max=1).square()
    chroma_deficit = torch.relu(source_chroma - output_chroma)
    return (weight * chroma_deficit.square()).mean((1, 2, 3))


def _display_headroom_lab(lab: torch.Tensor, *, invert_ink: bool = False) -> torch.Tensor:
    """Apply the display transform to an OKLab image without leaving OKLab."""
    lightness = lab[:, :1]
    displayed_lightness = lightness + 2.0 * lightness * (1 - lightness).square()
    chroma_scale = displayed_lightness / lightness.clamp_min(1e-6)
    if invert_ink:
        displayed_lightness = 1 - displayed_lightness
    displayed_lab = torch.cat((displayed_lightness, lab[:, 1:] * chroma_scale), dim=1)
    return displayed_lab


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


def _gradient_loss(rendered_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
    """Match oriented Sobel edges in displayed OKLab, including chroma edges."""
    rendered_x, rendered_y = _sobel_gradients(rendered_lab)
    target_x, target_y = _sobel_gradients(target_lab)
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
          gradient_target_lab: torch.Tensor | None = None,
          gradient_loss_strength: float = 0.0,
          invert_ink: bool = False) -> torch.Tensor:
    if target_lab is None:
        target_lab = target
    residual = rendered - target_lab
    pixel_error = residual.square().sum(1, keepdim=True)
    # Pixel color, low-frequency color, and edge/structure differences.
    loss = ((local_weight * pixel_error).mean() if detail_loss == "local_color"
            else pixel_error.mean())
    for factor, weight in ((2, 1.0), (4, 1.5), (8, 2.0)):
        if min(rendered.shape[-2:]) >= factor:
            pooled = F.avg_pool2d(residual, factor).square()
            scale_error = pooled.sum(1, keepdim=True)
            loss = loss + weight * scale_error.mean()
    difference = residual[:, :1]
    loss = loss + 0.35 * (difference[..., 1:] - difference[..., :-1]).square().mean()
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
        displayed_lab = _display_headroom_lab(rendered, invert_ink=invert_ink)
        loss = loss + gradient_loss_strength * _gradient_loss(displayed_lab, gradient_target_lab)
    return loss


def _candidate_fit(cells: torch.Tensor, masks: torch.Tensor, count: int,
                   background_cells: torch.Tensor,
                   pixel_weight: torch.Tensor | None = None,
                   glyph_luminance_bonus: float = 0.0,
                   gradient_target_cells: torch.Tensor | None = None,
                   gradient_shortlist_strength: float = 0.0,
                   invert_ink: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    mask_flat = masks.flatten(1)
    denominator = mask_flat.square().sum(1).clamp_min(1e-8)
    mask_square = mask_flat.square()
    all_indices = []
    all_colors = []
    cell_height, cell_width = masks.shape[1:]
    chunk_size = 128 if gradient_target_cells is not None and gradient_shortlist_strength else 2048
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
            error = error - glyph_luminance_bonus * _ink_lightness(color)
        if gradient_target_cells is not None and gradient_shortlist_strength:
            # Rank glyphs partly by how their displayed OKLab edges match this
            # cell's source edges. Neighboring-cell edges are handled later by
            # the full-frame objective.
            candidate_lab = candidate_pixels.reshape(
                len(cell) * masks.shape[0], cell_height, cell_width, 3
            ).permute(0, 3, 1, 2)
            candidate_display = _display_headroom_lab(candidate_lab, invert_ink=invert_ink)
            candidate_x, candidate_y = _sobel_gradients(candidate_display)
            source_cell = gradient_target_cells[start:start + len(cell)].reshape(
                len(cell), cell_height, cell_width, 3
            ).permute(0, 3, 1, 2)
            source_x, source_y = _sobel_gradients(source_cell)
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


def solve(image_srgb: np.ndarray, masks_np: np.ndarray, *, seed: int = 0,
          steps: int = 36, candidates: int = 6, dither: float = 0.005,
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
          invert_ink: bool = False,
          input_oklab: bool = False) -> Grid:
    """Optimize one complete grid with OKLab colors, scores, and parameters."""
    if masks_np.shape[0] > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    if detail_loss not in DETAIL_LOSSES:
        raise ValueError(f"Unknown detail loss: {detail_loss}")
    if objective_space != "oklab":
        raise ValueError("The optimizer operates only in OKLab.")
    if glyph_luminance_bonus < 0:
        raise ValueError("glyph_luminance_bonus must be nonnegative")
    if saturation_loss_strength is None:
        saturation_loss_strength = 20.0 if glyph_luminance_bonus > 0 else 0.0
    if saturation_loss_strength < 0:
        raise ValueError("saturation_loss_strength must be nonnegative")
    if gradient_loss_strength < 0:
        raise ValueError("gradient_loss_strength must be nonnegative")
    if gradient_shortlist_strength is None:
        gradient_shortlist_strength = gradient_loss_strength
    if gradient_shortlist_strength < 0:
        raise ValueError("gradient_shortlist_strength must be nonnegative")
    if not np.isfinite(dither) or dither < 0:
        raise ValueError("dither must be finite and nonnegative")
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

    if gradient_loss_strength or gradient_shortlist_strength:
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
    cells = _cells(image, rows, cols, cell_height, cell_width)
    background_color = (_choose_background(image, background) if background_oklab is None
                        else torch.as_tensor(background_oklab, dtype=torch.float32, device=device).reshape(3))
    background_cells = background_color.reshape(1, 1, 3).expand(len(cells), cell_height * cell_width, 3)
    masks = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
    bank = masks.flatten(1)
    noise = torch.as_tensor(blue_noise((rows, cols), seed).flatten(), device=device)
    local_weight = _local_color_weight(target) if detail_loss in ("local_color", "chroma_contrast", "color_tail") else None
    target_lab = target.detach()
    target_lab_cells = _cells(target_lab.squeeze(0), rows, cols, cell_height, cell_width)
    pixel_weight = None
    if local_weight is not None:
        pixel_weight = (local_weight.squeeze(0).squeeze(0).reshape(rows, cell_height, cols, cell_width)
                        .permute(0, 2, 1, 3).reshape(-1, cell_height * cell_width))
        regular_count = max(1, candidates // 2)
        weighted_count = candidates - regular_count
        regular_indices, regular_colors = _candidate_fit(
            cells, masks, regular_count, background_cells,
            glyph_luminance_bonus=glyph_luminance_bonus,
            gradient_target_cells=gradient_target_cells,
            gradient_shortlist_strength=gradient_shortlist_strength,
            invert_ink=invert_ink)
        weighted_indices, weighted_colors = _candidate_fit(
            cells, masks, weighted_count, background_cells, pixel_weight,
            glyph_luminance_bonus=glyph_luminance_bonus,
            gradient_target_cells=gradient_target_cells,
            gradient_shortlist_strength=gradient_shortlist_strength,
            invert_ink=invert_ink)
        indices = torch.cat((regular_indices, weighted_indices), dim=1)
        fitted = torch.cat((regular_colors, weighted_colors), dim=1)
    else:
        indices, fitted = _candidate_fit(cells, masks, candidates,
                                         background_cells,
                                         glyph_luminance_bonus=glyph_luminance_bonus,
                                         gradient_target_cells=gradient_target_cells,
                                         gradient_shortlist_strength=gradient_shortlist_strength,
                                         invert_ink=invert_ink)

    if previous is not None and destination is not None:
        if previous.glyph.shape != (rows, cols) or destination.glyph.shape != (rows, cols):
            raise ValueError("Transition endpoints must use the same grid")
        indices = torch.cat((indices,
            torch.as_tensor(previous.glyph.astype(np.int64), device=device).reshape(-1, 1),
            torch.as_tensor(destination.glyph.astype(np.int64), device=device).reshape(-1, 1)), dim=1)
        extra = []
        for grid in (previous, destination):
            extra.append(torch.as_tensor(grid.color, dtype=torch.float32, device=device).reshape(-1, 1, 3))
        fitted = torch.cat((fitted, *extra), dim=1)

    sample_masks = bank[indices]
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
    candidate_render = (sample_masks[..., None] * candidate_colors[:, :, None, :]
                        + (1 - sample_masks[..., None]) * background_cells[:, None, :, :])
    perceptual_error = (candidate_render - target_lab_cells[:, None]).square().sum(3)
    if fitting_weight is None:
        initial_errors = perceptual_error.mean(2)
    else:
        initial_errors = ((perceptual_error * fitting_weight[:, None]).sum(2)
                          / fitting_weight.sum(1, keepdim=True).clamp_min(1e-8))
    if glyph_luminance_bonus:
        initial_errors = initial_errors - glyph_luminance_bonus * _ink_lightness(candidate_colors)
    if saturation_loss_strength:
        batch, count = candidate_render.shape[:2]
        candidate_images = candidate_render.reshape(-1, cell_height, cell_width, 3).permute(0, 3, 1, 2)
        candidate_targets = (saturation_target_cells[:, None]
                             .expand(-1, count, -1, -1)
                             .reshape(-1, cell_height, cell_width, 3)
                             .permute(0, 3, 1, 2))
        saturation_errors = _saturation_loss(
            candidate_images, candidate_targets, saturation_chroma_scale).reshape(batch, count)
        initial_errors = initial_errors + saturation_loss_strength * saturation_errors
    if previous is not None and destination is not None:
        chosen = torch.where(noise < float(progress),
                             torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                             torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        initial_errors = initial_errors + 0.015 * (indices != chosen[:, None]).float()
    logits = torch.nn.Parameter((-initial_errors / 0.02).clamp(-20, 0).detach())
    first = initial_errors.argmin(1)
    initial_color = _project_oklab_colors(fitted[torch.arange(len(fitted), device=device), first])
    color_parameter = torch.nn.Parameter(initial_color.detach())
    optimizer = torch.optim.Adam((logits, color_parameter), lr=0.12)
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
        loss = _loss(rendered, target, detail_loss=detail_loss, local_weight=local_weight,
                     cell=(cell_height, cell_width), target_lab=target_lab,
                     gradient_target_lab=gradient_target_lab,
                     gradient_loss_strength=gradient_loss_strength,
                     invert_ink=invert_ink)
        if saturation_loss_strength:
            loss = loss + saturation_loss_strength * _saturation_loss(
                rendered, saturation_target_lab, saturation_chroma_scale).mean()
        return loss

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = max(0.3, 1.0 - 0.7 * step / max(1, steps - 1))
        weights = torch.softmax(logits / temperature, dim=1)
        mask = (weights[..., None] * sample_masks).sum(1)
        color = color_parameter
        rendered = _render(mask, color, rows, cols, cell_height, cell_width, background_color)
        loss = frame_loss(rendered)
        if glyph_luminance_bonus:
            loss = loss - glyph_luminance_bonus * _ink_lightness(color).mean()
        loss = loss + 0.006 * (-(weights * weights.clamp_min(1e-9).log()).sum(1)).mean()
        if prev_masks is not None:
            loss = loss + 0.025 * (mask - prev_masks).square().mean()
            loss = loss + 0.010 * (color - prev_colors).square().mean()
            loss = loss + 0.008 * (mask - scheduled_masks).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            color_parameter.copy_(_project_oklab_colors(color_parameter))
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
        hard_color = color_parameter.detach()

    def hard_score(chosen_masks: torch.Tensor, chosen_color: torch.Tensor) -> torch.Tensor:
        score = frame_loss(_render(chosen_masks, chosen_color, rows, cols, cell_height, cell_width, background_color))
        if glyph_luminance_bonus:
            score = score - glyph_luminance_bonus * _ink_lightness(chosen_color).mean()
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
            if glyph_luminance_bonus:
                gain = gain - glyph_luminance_bonus * (
                    _ink_lightness(candidate_colors) - _ink_lightness(hard_color)[:, None]
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
                if hard_score(trial_masks, trial_color) < hard_score(hard_masks, hard_color):
                    glyph, hard_color, hard_masks = trial_glyph, trial_color, trial_masks
        if on_progress is not None:
            on_progress((steps + parity + 1) / total_work, "Refining character boundaries")

    # Final colors are optimized against the actual discrete glyphs.
    color_parameter = torch.nn.Parameter(hard_color.detach().clone())
    color_optimizer = torch.optim.Adam((color_parameter,), lr=0.04)
    for polish_step in range(polish_steps):
        color_optimizer.zero_grad(set_to_none=True)
        color = color_parameter
        loss = frame_loss(_render(hard_masks, color, rows, cols, cell_height, cell_width, background_color))
        if glyph_luminance_bonus:
            loss = loss - glyph_luminance_bonus * _ink_lightness(color).mean()
        if prev_colors is not None:
            loss = loss + 0.010 * (color - prev_colors).square().mean()
        loss.backward()
        color_optimizer.step()
        with torch.no_grad():
            color_parameter.copy_(_project_oklab_colors(color_parameter))
        if on_progress is not None:
            on_progress((steps + 4 + polish_step + 1) / total_work, "Polishing colors")

    return Grid(glyph.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8),
                color_parameter.detach().cpu().numpy().reshape(rows, cols, 3),
                background_color.detach().cpu().numpy())
