"""Compare experimental local glyph error transport with the current renderer.

Run through the numerical launcher so ROCm does not probe unrelated NPUs:
    .venv/bin/python -m ascii_optimizer.launcher scripts/compare_error_movement.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import experiment_utils as base
from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.tone import srgb_to_oklab


SOURCES = (
    ("tail", Path("screenshots/test_image.png"), (720, 65, 960, 240)),
    ("blush", Path("screenshots/149642007_p0-无题.jpg"), (185, 375, 370, 490)),
)


def focus_metrics(reference: np.ndarray, rendered: np.ndarray,
                  crop: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = crop
    source, output = reference[y1:y2, x1:x2], rendered[y1:y2, x1:x2]
    return {**base.image_metrics(source, output),
            "gradient_rmse_x100": base.gradient_rmse(source, output)}


def tail_boundary_metrics(reference, grid, masks, crop) -> dict:
    """Count green ink only in cells containing both source blue and yellow.

    Source yellow is itself slightly chartreuse. Requiring green to exceed
    both red and blue by 12% separates pooled green from that source yellow.
    The thresholds are diagnostic heuristics, not a perceptual quality score.
    """
    height, width = masks.shape[1:]
    rows, cols = grid.glyph.shape
    pixels = np.pad(reference, ((0, rows * height - reference.shape[0]),
                               (0, cols * width - reference.shape[1]), (0, 0)), mode="edge")
    red, green, blue = np.moveaxis(pixels, -1, 0)
    yellow = (red > 0.35) & (green > 0.35) & (blue < 0.18)
    bluish = (blue > red * 1.08) & (blue > green * 1.01) & (blue > 0.12)

    def fraction(mask):
        return mask.reshape(rows, height, cols, width).mean((1, 3))

    x1, y1, x2, y2 = crop
    yy, xx = np.mgrid[:rows, :cols]
    selected = ((fraction(yellow) >= 0.08) & (fraction(bluish) >= 0.08)
                & (xx * width >= x1) & (xx * width < x2)
                & (yy * height >= y1) & (yy * height < y2))
    ink = base.oklab_to_srgb(base.display_tone_lab(grid.color))
    r, g, b = np.moveaxis(ink, -1, 0)
    green_ink = (g > 1.12 * r) & (g > 1.12 * b)
    return {"source_blue_yellow_boundary_cells": int(selected.sum()),
            "pooled_green_ink_cells": int((selected & green_ink).sum()),
            "pooled_green_ink_fraction": float(green_ink[selected].mean()) if selected.any() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rendered/error_movement"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cells", nargs="+", default=["12x24", "8x16"])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 1, 4, 12])
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        cells = [tuple(map(int, cell.split("x"))) for cell in args.cells]
        if any(len(cell) != 2 or min(cell) < 1 for cell in cells):
            raise ValueError
    except ValueError:
        parser.error("cells must be positive WIDTHxHEIGHT pairs")
    if args.width < 1 or args.steps < 0 or any(not np.isfinite(s) or s < 0 for s in args.strengths):
        parser.error("width must be positive, and steps/strengths finite and nonnegative")
    strengths = list(dict.fromkeys([0.0, *args.strengths]))
    base.reset_runs()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"method": "local sliced squared Wasserstein color transport in overlapping 2x2 glyph windows",
              "production_default": 0.0, "width": args.width, "cells": cells,
              "strengths": strengths, "steps": args.steps, "seed": 0,
              "dither": base.DITHER, "glyphs": base.DEFAULT_CHARS, "images": {}}
    for name, path, crop in SOURCES:
        size = base.source_size(path, args.width)
        reference = base.load_target(path, size, "fill", False)
        crop = tuple(round(c * args.width / 960) for c in crop)
        image_report = {"source": str(path), "size": size, "focus_crop": crop, "grids": {}}
        for cell in cells:
            directory = args.output / name / f"{cell[0]}x{cell[1]}"
            directory.mkdir(parents=True, exist_ok=True)
            masks = glyph_masks(resolve_font("Monospace"), list(base.DEFAULT_CHARS), cell)
            source_image = base.image(reference)
            source_image.save(directory / "source.png")
            panels = [("Source", source_image)]
            variants = {}
            baseline_glyph = None
            for strength in strengths:
                key = "before" if strength == 0 else f"movement_{strength:g}"
                print(f"{name} {cell}: {key}", flush=True)
                grid = base.solve_style(reference, masks, style="headroom", invert_ink=False,
                    background="auto", seed=0, steps=args.steps, candidates=base.CANDIDATES,
                    dither=base.DITHER, structure_dither=0.06, device=args.device,
                    glyph_error_movement_strength=strength)
                if strength == 0:
                    baseline_glyph = grid.glyph.copy()
                rendered = base.preview(grid, masks, size)
                shown = base.image(rendered)
                shown.save(directory / f"{key}.png")
                np.savez_compressed(directory / f"{key}.npz", glyph=grid.glyph,
                                    color=grid.color, background=grid.background)
                panels.append(("Before (current default)" if strength == 0 else f"Error movement = {strength:g}", shown))
                variants[key] = {"strength": strength, "experiment": grid.experiment,
                    **base.image_metrics(reference, rendered),
                    **base.glyph_metrics(grid, masks, base.DEFAULT_CHARS, baseline_glyph),
                    "focus": focus_metrics(reference, rendered, crop),
                    "mean_display_lightness": float(srgb_to_oklab(rendered)[..., 0].mean())}
                if name == "tail":
                    variants[key]["boundary"] = tail_boundary_metrics(reference, grid, masks, crop)
            base.contact_sheet(panels, directory / "comparison.png", columns=3, panel=(640, round(size[1] * 640 / size[0])))
            base.contact_sheet(panels, directory / "focus_comparison.png", crop=crop, columns=3, zoom=3)
            for strength in strengths[1:]:
                # Review one variant at a time at exact pixel scale.
                key = f"movement_{strength:g}"
                pair = [panels[0], panels[1], next(p for p in panels if p[0] == f"Error movement = {strength:g}")]
                base.contact_sheet(pair, directory / f"focus_before_after_{strength:g}.png", crop=crop, columns=3, zoom=3)
            image_report["grids"][f"{cell[0]}x{cell[1]}"] = variants
        report["images"][name] = image_report
    base.write_report(args.output / "metrics.json", report)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
