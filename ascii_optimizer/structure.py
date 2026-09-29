"""Structure-first ASCII glyph selection with blue-noise tone dithering."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .assets import blue_noise
from .defaults import STRUCTURE_CHARS
from .optimizer import Grid
from .tone import quantize_playback_lab, srgb_to_oklab


STRUCTURE_DITHER = 0.06


def _source_cells(source: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    height, width = source.shape[:2]
    cell_height, cell_width = cell
    rows = (height + cell_height - 1) // cell_height
    cols = (width + cell_width - 1) // cell_width
    source_lab = np.pad(srgb_to_oklab(source),
                        ((0, rows * cell_height - height),
                         (0, cols * cell_width - width), (0, 0)), mode="edge")
    return source_lab.reshape(rows, cell_height, cols, cell_width, 3).transpose(0, 2, 1, 3, 4)


def _foreground(cells: np.ndarray) -> np.ndarray:
    foreground = cells.mean((2, 3))
    foreground[..., 0] = np.minimum(1, 0.48 + 0.52 * foreground[..., 0])
    return quantize_playback_lab(foreground)


def fit_structure_colors(source: np.ndarray, masks: np.ndarray, glyphs: np.ndarray,
                         background: np.ndarray) -> Grid:
    """Assign the structure style's bright per-cell ink to fixed path glyphs."""
    cells = _source_cells(source, masks.shape[1:])
    if glyphs.shape != cells.shape[:2]:
        raise ValueError("Fixed glyphs must use the target image grid")
    return Grid(glyphs.astype(np.uint8).copy(), _foreground(cells),
                quantize_playback_lab(background))


def solve_structure(source: np.ndarray, masks: np.ndarray, background: np.ndarray, *,
                    seed: int = 0, dither: float = STRUCTURE_DITHER,
                    previous: Grid | None = None, destination: Grid | None = None,
                    progress: float | None = None,
                    on_progress: Callable[[float, str], None] | None = None) -> Grid:
    """Match glyph coverage and within-cell luminance texture, keeping one dark canvas."""
    height, width = source.shape[:2]
    cell_height, cell_width = masks.shape[1:]
    rows = (height + cell_height - 1) // cell_height
    cols = (width + cell_width - 1) // cell_width
    if previous is not None and destination is not None:
        if previous.glyph.shape != (rows, cols) or destination.glyph.shape != (rows, cols):
            raise ValueError("Transition endpoints must use the same grid")

    cells = _source_cells(source, (cell_height, cell_width))
    source_lightness = cells[..., 0].mean((2, 3))
    foreground = _foreground(cells)
    background = quantize_playback_lab(background)

    desired_coverage = 0.01 + 0.35 * source_lightness ** 1.35
    if dither:
        desired_coverage = np.clip(
            desired_coverage + dither * (blue_noise((rows, cols), seed) - 0.5), 0, 1)
    coverage = masks.mean((1, 2))
    source_texture = cells[..., 0].copy()
    source_texture -= source_texture.mean((2, 3), keepdims=True)
    contrast = (foreground[..., 0] - float(background[0]))[..., None, None]
    schedule = None
    if previous is not None and destination is not None:
        schedule = np.where(blue_noise((rows, cols), seed + 1009) < float(progress),
                            destination.glyph, previous.glyph)

    best_score = np.full((rows, cols), np.inf, dtype=np.float32)
    best_glyph = np.zeros((rows, cols), dtype=np.uint8)
    for index, mask in enumerate(masks):
        glyph_texture = contrast * (mask - coverage[index])
        texture_error = ((source_texture - glyph_texture) ** 2).mean((2, 3))
        score = 1.5 * (desired_coverage - coverage[index]) ** 2 + texture_error
        if schedule is not None:
            score = score + 0.003 * (index != schedule) + 0.001 * (index != previous.glyph)
        selected = score < best_score
        best_score = np.where(selected, score, best_score)
        best_glyph = np.where(selected, index, best_glyph)
        if on_progress is not None:
            on_progress((index + 1) / len(masks), "Selecting structure-first glyphs")
    return Grid(best_glyph, foreground, background)
