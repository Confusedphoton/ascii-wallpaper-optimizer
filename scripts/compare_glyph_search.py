"""Compare six-glyph shortlists against optimization over the complete glyph bank.

    .venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_search.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import experiment_utils as base
from compare_error_movement import SOURCES, focus_metrics, tail_boundary_metrics
from ascii_optimizer import optimizer
from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.defaults import DEFAULT_USE_SHORTLIST


def region_cells(grid, masks, crop):
    height, width = masks.shape[1:]
    yy, xx = np.mgrid[:grid.glyph.shape[0], :grid.glyph.shape[1]]
    x1, y1, x2, y2 = crop
    return ((xx * width + width / 2 >= x1) & (xx * width + width / 2 < x2)
            & (yy * height + height / 2 >= y1) & (yy * height + height / 2 < y2))


def selection_metrics(grid, masks, shortlist, crop, width, name):
    outside = ~(grid.glyph[..., None] == shortlist).any(-1)
    focus = region_cells(grid, masks, crop)
    result = {"selected_outside_original_shortlist_fraction": float(outside.mean()),
              "focus_outside_original_shortlist_fraction": float(outside[focus].mean()) if focus.any() else None}
    if name == "blush":
        scale = width / 960
        cheeks = np.zeros_like(focus)
        for box in ((208, 438, 255, 474), (308, 414, 344, 456)):
            cheeks |= region_cells(grid, masks, tuple(round(c * scale) for c in box))
        coverage = masks[grid.glyph].mean((-2, -1))
        glyphs, counts = np.unique(grid.glyph[cheeks], return_counts=True)
        result["cheeks"] = {"cells": int(cheeks.sum()),
            "glyph_histogram": {base.DEFAULT_CHARS[int(g)]: int(n) for g, n in zip(glyphs, counts)},
            "mean_coverage": float(coverage[cheeks].mean()) if cheeks.any() else None,
            "thin_glyph_fraction_coverage_le_0_20": float((coverage[cheeks] <= 0.2).mean()) if cheeks.any() else None,
            "outside_original_shortlist_fraction": float(outside[cheeks].mean()) if cheeks.any() else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rendered/glyph_search"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cells", nargs="+", default=["12x24", "8x16"])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 4, 12, 32])
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--dither", type=float, default=base.DITHER)
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
    if not np.isfinite(args.dither) or args.dither < 0:
        parser.error("dither must be finite and nonnegative")
    strengths = list(dict.fromkeys([0.0, *args.strengths]))
    base.reset_runs()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"method": "all glyphs in every cell, with no shortlist ranking or pruning",
              "production_use_shortlist": DEFAULT_USE_SHORTLIST, "production_error_movement_strength": 0,
              "width": args.width, "cells": cells, "strengths": strengths,
              "steps": args.steps, "seed": 0, "dither": args.dither,
              "glyphs": base.DEFAULT_CHARS, "images": {}}
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
            panels = {}
            variants = {}
            baseline_glyph = shortlist = None
            shortlist_fit = optimizer._candidate_fit
            captured = []
            adam_step = torch.optim.Adam.step
            noise_function = optimizer._post_logit_noise
            last_logits = candidate_noise = None

            def capture(*fit_args, **fit_kwargs):
                indices, colors = shortlist_fit(*fit_args, **fit_kwargs)
                captured.append(indices.detach().cpu().numpy())
                return indices, colors

            def capture_step(adam, *step_args, **step_kwargs):
                nonlocal last_logits
                result = adam_step(adam, *step_args, **step_kwargs)
                if len(adam.param_groups) == 2:
                    last_logits = adam.param_groups[0]["params"][0].detach().clone()
                return result

            def capture_noise(*noise_args, **noise_kwargs):
                nonlocal candidate_noise
                candidate_noise = noise_function(*noise_args, **noise_kwargs)
                return candidate_noise

            for use_shortlist in (True, False):
                for strength in strengths:
                    key = f"{'shortlist' if use_shortlist else 'all'}_movement_{strength:g}"
                    print(f"{name} {cell}: {key}", flush=True)
                    last_logits = candidate_noise = None
                    with patch.object(optimizer, "_candidate_fit", capture), \
                            patch.object(torch.optim.Adam, "step", capture_step), \
                            patch.object(optimizer, "_post_logit_noise", capture_noise):
                        grid = base.solve_style(reference, masks, style="headroom", invert_ink=False,
                            background="auto", seed=0, steps=args.steps, candidates=base.CANDIDATES,
                            dither=args.dither, structure_dither=0.06, device=args.device,
                            use_shortlist=use_shortlist, glyph_error_movement_strength=strength)
                    if baseline_glyph is None:
                        baseline_glyph = grid.glyph.copy()
                        shortlist = captured[0].reshape(*grid.glyph.shape, -1)
                        np.save(directory / "original_shortlist.npy", shortlist)
                    rendered = base.preview(grid, masks, size)
                    shown = base.image(rendered)
                    shown.save(directory / f"{key}.png")
                    np.savez_compressed(directory / f"{key}.npz", glyph=grid.glyph,
                                        color=grid.color, background=grid.background)
                    panels[key] = (f"{'Shortlist (6)' if use_shortlist else f'All glyphs ({len(masks)})'} | movement {strength:g}", shown)
                    variants[key] = {"use_shortlist": use_shortlist, "strength": strength,
                        "glyphs_per_cell": min(base.CANDIDATES, len(masks)) if use_shortlist else len(masks),
                        "experiment": grid.experiment, **base.image_metrics(reference, rendered),
                        **base.glyph_metrics(grid, masks, base.DEFAULT_CHARS, baseline_glyph),
                        "focus": focus_metrics(reference, rendered, crop),
                        **selection_metrics(grid, masks, shortlist, crop, args.width, name)}
                    if name == "tail":
                        variants[key]["boundary"] = tail_boundary_metrics(reference, grid, masks, crop)
                    if last_logits is not None:
                        with torch.no_grad():
                            perturbation = (args.dither * optimizer.DITHER_TO_LOGIT_SCALE * candidate_noise
                                            if candidate_noise is not None else 0)
                            changed = ((last_logits + perturbation).argmax(1) != last_logits.argmax(1))
                            changed = changed.cpu().numpy().reshape(grid.glyph.shape)
                            top_two = last_logits.topk(min(2, last_logits.shape[1]), dim=1).values
                            margin = (top_two[:, 0] - top_two[:, -1]).cpu().numpy()
                        focus = region_cells(grid, masks, crop)
                        variants[key]["logit_noise"] = {
                            "pre_refinement_winner_change_fraction": float(changed.mean()),
                            "focus_pre_refinement_winner_change_fraction": float(changed[focus].mean()) if focus.any() else None,
                            "top_two_logit_margin_mean": float(margin.mean()),
                            "top_two_logit_margin_median": float(np.median(margin)),
                            "top_two_logit_margin_le_noise_span_fraction": float((margin <= args.dither * optimizer.DITHER_TO_LOGIT_SCALE).mean()),
                        }
            for strength in strengths:
                pair = [("Source", source_image), panels[f"shortlist_movement_{strength:g}"],
                        panels[f"all_movement_{strength:g}"]]
                base.contact_sheet(pair, directory / f"focus_movement_{strength:g}.png", crop=crop, columns=3, zoom=3)
            for strength in strengths[1:]:
                factorial = [panels["shortlist_movement_0"], panels["all_movement_0"],
                             panels[f"shortlist_movement_{strength:g}"], panels[f"all_movement_{strength:g}"]]
                base.contact_sheet(factorial, directory / f"factorial_focus_{strength:g}.png", crop=crop, columns=2, zoom=3)
            overview = [("Source", source_image), *panels.values()]
            base.contact_sheet(overview, directory / "comparison.png", columns=3, panel=(480, round(size[1] / 2)))
            base.contact_sheet(overview, directory / "focus_comparison.png", crop=crop, columns=3, zoom=2)
            image_report["grids"][f"{cell[0]}x{cell[1]}"] = variants
        report["images"][name] = image_report
    base.write_report(args.output / "metrics.json", report)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
