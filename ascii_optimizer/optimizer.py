"""Joint glyph/color optimization over the complete rendered frame."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import blue_noise, srgb_to_linear


@dataclass
class Grid:
    glyph: np.ndarray  # rows x columns, uint8
    color: np.ndarray  # rows x columns x RGB, linear float32
    background: np.ndarray | None = None  # one RGB color for the complete image, linear float32


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
        return srgb_to_linear(image.new_tensor([int(choice[i:i + 2], 16) / 255 for i in (1, 3, 5)]))
    # Choose one shadow color for the entire image. Keep it dark enough that
    # bright glyphs can still describe images with no dark regions.
    sampled = F.avg_pool2d(image.unsqueeze(0), 4, ceil_mode=True).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    luma = (sampled * sampled.new_tensor([0.2126, 0.7152, 0.0722])).sum(1)
    background = sampled[luma <= torch.quantile(luma, 0.1)].median(0).values
    background_luma = (background * background.new_tensor([0.2126, 0.7152, 0.0722])).sum()
    return background * (0.18 / background_luma.clamp_min(1e-8)).clamp(max=1)


def _loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = rendered - target
    # Linear-light pixel, low-frequency color, and edge/structure differences.
    loss = residual.square().mean()
    for factor, weight in ((2, 1.0), (4, 1.5), (8, 2.0)):
        if min(rendered.shape[-2:]) >= factor:
            loss = loss + weight * F.avg_pool2d(residual, factor).square().mean()
    luma = (rendered * rendered.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    target_luma = (target * target.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    difference = luma - target_luma
    loss = loss + 0.35 * (difference[..., 1:] - difference[..., :-1]).square().mean()
    loss = loss + 0.35 * (difference[..., 1:, :] - difference[..., :-1, :]).square().mean()
    return loss


def _candidate_fit(cells: torch.Tensor, masks: torch.Tensor, count: int,
                   noise: torch.Tensor, dither: float,
                   background_cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_flat = masks.flatten(1)
    denominator = mask_flat.square().sum(1).clamp_min(1e-8)
    mask_square = mask_flat.square()
    coverage = mask_flat.mean(1)
    all_indices = []
    all_colors = []
    for start in range(0, cells.shape[0], 2048):
        cell = cells[start:start + 2048]
        underpaint = background_cells[start:start + len(cell)]
        numerator = (torch.einsum("npc,gp->ngc", cell - underpaint, mask_flat)
                     + torch.einsum("npc,gp->ngc", underpaint, mask_square))
        color = (numerator / denominator[None, :, None]).clamp(0, 1)
        cell_background = cell * underpaint
        background_squared = underpaint.square()
        target_energy = ((cell - underpaint).square().sum((1, 2))[:, None]
                         + 2 * torch.einsum("np,gp->ng", (cell_background - background_squared).sum(2), mask_flat)
                         + torch.einsum("np,gp->ng", background_squared.sum(2), mask_square))
        error = target_energy - 2 * (numerator * color).sum(2) + (color.square().sum(2) * denominator[None, :])
        error = error / (cell.shape[1] * 3)
        error = error + dither * (noise[start:start + len(cell), None] - 0.5) * (coverage[None, :] - coverage.mean())
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
          background: str = "auto", background_linear: np.ndarray | None = None,
          device: str = "cpu", previous: Grid | None = None,
          destination: Grid | None = None, progress: float | None = None,
          on_progress: Callable[[float, str], None] | None = None) -> Grid:
    """Optimize one complete grid; temporal frames also depend on their neighbors."""
    if masks_np.shape[0] > 256:
        raise ValueError("The glyph map supports at most 256 characters")
    cell_height, cell_width = masks_np.shape[1:]
    original_height, original_width = image_srgb.shape[:2]
    rows = (original_height + cell_height - 1) // cell_height
    cols = (original_width + cell_width - 1) // cell_width
    pad_height = rows * cell_height - original_height
    pad_width = cols * cell_width - original_width
    image = torch.as_tensor(image_srgb, dtype=torch.float32, device=device).permute(2, 0, 1)
    image = srgb_to_linear(F.pad(image.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0))
    target = image.unsqueeze(0)
    cells = _cells(image, rows, cols, cell_height, cell_width)
    background_color = (_choose_background(image, background) if background_linear is None
                        else torch.as_tensor(background_linear, dtype=torch.float32, device=device).reshape(3))
    background_cells = background_color.reshape(1, 1, 3).expand(len(cells), cell_height * cell_width, 3)
    masks = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
    bank = masks.flatten(1)
    noise = torch.as_tensor(blue_noise((rows, cols), seed).flatten(), device=device)
    indices, fitted = _candidate_fit(cells, masks, candidates, noise, dither, background_cells)

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
    mask_den = sample_masks.square().sum(2).clamp_min(1e-8)
    numerator = torch.einsum("npc,nkp->nkc", cells - background_cells, sample_masks)
    numerator = numerator + torch.einsum("npc,nkp->nkc", background_cells, sample_masks.square())
    candidate_colors = (numerator / mask_den[..., None]).clamp(0, 1)
    candidate_render = (sample_masks[..., None] * candidate_colors[:, :, None, :]
                        + (1 - sample_masks[..., None]) * background_cells[:, None, :, :])
    initial_errors = (candidate_render - cells[:, None]).square().mean((2, 3))
    if previous is not None and destination is not None:
        chosen = torch.where(noise < float(progress),
                             torch.as_tensor(destination.glyph.astype(np.int64), device=device).flatten(),
                             torch.as_tensor(previous.glyph.astype(np.int64), device=device).flatten())
        initial_errors = initial_errors + 0.015 * (indices != chosen[:, None]).float()
    logits = torch.nn.Parameter((-initial_errors / 0.02).clamp(-20, 0).detach())
    first = initial_errors.argmin(1)
    initial_color = fitted[torch.arange(len(fitted), device=device), first].clamp(0.001, 0.999)
    color_parameter = torch.nn.Parameter(torch.logit(initial_color).detach())
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
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = max(0.3, 1.0 - 0.7 * step / max(1, steps - 1))
        weights = torch.softmax(logits / temperature, dim=1)
        mask = (weights[..., None] * sample_masks).sum(1)
        color = torch.sigmoid(color_parameter)
        rendered = _render(mask, color, rows, cols, cell_height, cell_width, background_color)
        loss = _loss(rendered, target)
        loss = loss + 0.006 * (-(weights * weights.clamp_min(1e-9).log()).sum(1)).mean()
        loss = loss + dither * ((noise - 0.5) * mask.mean(1)).mean()
        if prev_masks is not None:
            loss = loss + 0.025 * (mask - prev_masks).square().mean()
            loss = loss + 0.010 * (color - prev_colors).square().mean()
            loss = loss + 0.008 * (mask - scheduled_masks).square().mean()
        loss.backward()
        optimizer.step()
        if on_progress is not None:
            on_progress((step + 1) / total_work, "Optimizing glyphs and colors")

    with torch.no_grad():
        selected = logits.argmax(1)
        glyph = indices.gather(1, selected[:, None]).squeeze(1)
        hard_masks = bank[glyph]
        hard_color = torch.sigmoid(color_parameter).detach()

    def hard_score(chosen_masks: torch.Tensor, chosen_color: torch.Tensor) -> torch.Tensor:
        score = _loss(_render(chosen_masks, chosen_color, rows, cols, cell_height, cell_width, background_color), target)
        score = score + dither * ((noise - 0.5) * chosen_masks.mean(1)).mean()
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
        gradient = torch.autograd.grad(_loss(rendered, target), rendered)[0]
        gradient_cells = _cells(gradient.squeeze(0), rows, cols, cell_height, cell_width)
        with torch.no_grad():
            current_pixels = (hard_masks[..., None] * hard_color[:, None, :]
                              + (1 - hard_masks[..., None]) * background_cells)
            delta = candidate_render - current_pixels[:, None]
            gain = (gradient_cells[:, None] * delta).sum((2, 3))
            gain = gain + delta.square().sum((2, 3)) / (len(glyph) * cell_height * cell_width * 3)
            gain = gain + dither * (noise[:, None] - 0.5) * (sample_masks.mean(2) - hard_masks.mean(1)[:, None]) / len(glyph)
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
    color_parameter = torch.nn.Parameter(torch.logit(hard_color.clamp(0.001, 0.999)))
    color_optimizer = torch.optim.Adam((color_parameter,), lr=0.04)
    for polish_step in range(polish_steps):
        color_optimizer.zero_grad(set_to_none=True)
        color = torch.sigmoid(color_parameter)
        loss = _loss(_render(hard_masks, color, rows, cols, cell_height, cell_width, background_color), target)
        if prev_colors is not None:
            loss = loss + 0.010 * (color - prev_colors).square().mean()
        loss.backward()
        color_optimizer.step()
        if on_progress is not None:
            on_progress((steps + 4 + polish_step + 1) / total_work, "Polishing colors")

    return Grid(glyph.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8),
                torch.sigmoid(color_parameter).detach().cpu().numpy().reshape(rows, cols, 3),
                background_color.detach().cpu().numpy())
