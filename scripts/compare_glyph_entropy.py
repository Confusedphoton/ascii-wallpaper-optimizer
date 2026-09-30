"""Review opt-in frame-wide glyph entropy against the unchanged renderer.

    .venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_entropy.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import experiment_utils as base
from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.defaults import DEFAULT_GLYPH_ENTROPY_STRENGTH


SOURCES = (*base.SOURCES,
           ("viviana", Path("screenshots/143381492_p0-Viviana.jpg"), (330, 270, 530, 440)))
TRACKED_GLYPHS = "@MW`´&#"
REQUESTED_GLYPHS = "@MW´"


def tracked_glyph_metrics(grid, chars: str) -> dict:
    shares = {char: float(np.isin(grid.glyph, [i for i, glyph in enumerate(chars) if glyph == char]).mean())
              if char in chars else None for char in TRACKED_GLYPHS}
    return {"tracked_glyph_shares": shares,
            "requested_glyphs_present": [char for char in REQUESTED_GLYPHS if char in chars],
            "requested_glyph_share": sum(shares[char] or 0 for char in REQUESTED_GLYPHS)}


def write_review(output: Path, report: dict) -> None:
    lines = ["# Glyph entropy review", "",
             f"Production glyph entropy strength: **{report['production_default']:g}**. "
             "Strength zero is the control with entropy disabled.", "",
             "The loss adds `strength × sum(p × ln(p))` for the frame-wide glyph histogram. "
             "Higher entropy rewards more varied glyph use. It does not penalize particular glyphs. "
             "Soft selection and discrete acceptance use the same marginal objective; the existing "
             "per-cell sharpening term stays active.", "",
             f"Settings: width {report['width']}, {report['steps']} optimization steps, seed 0, "
             f"dither {report['dither']}; all other objectives use production defaults.", "",
             f"Glyph bank: `{report['glyphs'].replace('`', '[backtick]')}`", "",
             "Absent glyphs are listed as missing in metrics.json. The current default bank includes "
             "M, W, and ´, but lacks @. The user clarified that the typed backtick meant ´. "
             "A custom bank is always held constant "
             "between its own baseline and entropy variants.", "",
             "The table reports **actual discrete outputs**, not soft probabilities. "
             "Higher entropy alone does not establish visual quality. ΔE measures displayed color "
             "error against the source (lower is better). Compare the faces, shading, and backgrounds "
             "in the full previews and the enlarged glyph crops.", "",
             "| Image / cell | Strength | Entropy (bits) | M + W | @ + M + W + ´ | ΔE ×100 | Glyphs changed |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, image_report in report["images"].items():
        for cell, variants in image_report["grids"].items():
            for variant in variants.values():
                shares = variant["tracked_glyph_shares"]
                lines.append(f"| {name} / {cell} | {variant['strength']:g} | "
                             f"{variant['glyph_entropy_bits']:.3f} | "
                             f"{100 * ((shares['M'] or 0) + (shares['W'] or 0)):.1f}% | "
                             f"{100 * variant['requested_glyph_share']:.1f}% | "
                             f"{variant['mean_oklab_delta_e_x100']:.2f} | "
                             f"{100 * variant['glyph_change_fraction_vs_baseline']:.1f}% |")
    for name, image_report in report["images"].items():
        for cell in image_report["grids"]:
            directory = f"{name}/{cell}"
            lines.extend(["", f"## {name} — {cell}", "",
                          f"![Source, baseline, entropy sweep]({directory}/comparison.png)", "",
                          f"![Enlarged detail comparison]({directory}/focus_comparison.png)", ""])
            for strength in report["strengths"][1:]:
                lines.append(f"[Full-size before / after, strength {strength:g}]"
                             f"({directory}/before_after_{strength:g}.png)")
    (output / "review.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rendered/glyph_entropy"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cells", nargs="+", default=["12x24", "8x16"])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 0.005, DEFAULT_GLYPH_ENTROPY_STRENGTH, 0.02, 0.06])
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--dither", type=float, default=base.DITHER)
    parser.add_argument("--chars", default=base.DEFAULT_CHARS,
                        help="Keep this glyph bank identical across baseline and entropy variants")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        cells = [tuple(map(int, cell.lower().split("x"))) for cell in args.cells]
        if any(len(cell) != 2 or min(cell) < 1 for cell in cells):
            raise ValueError
    except ValueError:
        parser.error("cells must be positive WIDTHxHEIGHT pairs")
    if args.width < 1 or args.steps < 0 or any(not np.isfinite(s) or s < 0 for s in args.strengths):
        parser.error("width must be positive, and steps/strengths finite and nonnegative")
    if not np.isfinite(args.dither) or args.dither < 0:
        parser.error("dither must be finite and nonnegative")
    if not args.chars or len(args.chars) > 256 or len(set(args.chars)) != len(args.chars):
        parser.error("chars must contain 1–256 unique glyphs")
    strengths = list(dict.fromkeys([0.0, *args.strengths]))
    base.reset_runs()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"method": "negative frame-wide marginal Shannon glyph entropy in nats",
              "production_default": DEFAULT_GLYPH_ENTROPY_STRENGTH, "width": args.width, "cells": cells,
              "strengths": strengths, "steps": args.steps, "seed": 0,
              "dither": args.dither, "glyphs": args.chars,
              "bank_matches_production_default": args.chars == base.DEFAULT_CHARS, "images": {}}
    for name, path, original_crop in SOURCES:
        size = base.source_size(path, args.width)
        reference = base.load_target(path, size, "fill", False)
        crop = tuple(round(c * args.width / 960) for c in original_crop)
        image_report = {"source": str(path), "size": size, "focus_crop": crop, "grids": {}}
        report["images"][name] = image_report
        for cell in cells:
            directory = args.output / name / f"{cell[0]}x{cell[1]}"
            directory.mkdir(parents=True, exist_ok=True)
            masks = glyph_masks(resolve_font("Monospace"), list(args.chars), cell)
            source_image = base.image(reference)
            source_image.save(directory / "source.png")
            panels = [("Source", source_image)]
            variants = {}
            baseline_glyph = None
            for strength in strengths:
                key = "before" if strength == 0 else f"entropy_{strength:g}"
                print(f"{name} {cell}: {key}", flush=True)
                grid = base.solve_style(reference, masks, style="headroom", invert_ink=False,
                    background="auto", seed=0, steps=args.steps, candidates=base.CANDIDATES,
                    dither=args.dither, structure_dither=0.06, device=args.device,
                    glyph_entropy_strength=strength)
                if strength == 0:
                    baseline_glyph = grid.glyph.copy()
                rendered = base.preview(grid, masks, size)
                shown = base.image(rendered)
                shown.save(directory / f"{key}.png")
                np.savez_compressed(directory / f"{key}.npz", glyph=grid.glyph,
                                    color=grid.color, background=grid.background)
                panels.append(("Before (strength 0)" if strength == 0 else f"Entropy = {strength:g}", shown))
                variants[key] = {"strength": strength, "experiment": grid.experiment,
                    **base.image_metrics(reference, rendered),
                    **base.glyph_metrics(grid, masks, args.chars, baseline_glyph),
                    **tracked_glyph_metrics(grid, args.chars),
                    "gradient_rmse_x100": base.gradient_rmse(reference, rendered),
                    "focus": base.image_metrics(reference[crop[1]:crop[3], crop[0]:crop[2]],
                                                 rendered[crop[1]:crop[3], crop[0]:crop[2]])}
            base.contact_sheet(panels, directory / "comparison.png", columns=3,
                               panel=(640, round(size[1] * 640 / size[0])))
            base.contact_sheet(panels, directory / "focus_comparison.png", crop=crop, columns=3, zoom=3)
            for index, strength in enumerate(strengths[1:], 2):
                pair = [panels[0], panels[1], panels[index]]
                base.contact_sheet(pair, directory / f"before_after_{strength:g}.png", columns=3)
                base.contact_sheet(pair, directory / f"focus_before_after_{strength:g}.png",
                                   crop=crop, columns=3, zoom=3)
            image_report["grids"][f"{cell[0]}x{cell[1]}"] = variants
            # Save progress so completed comparisons remain reviewable if interrupted.
            base.write_report(args.output / "metrics.json", report)
            write_review(args.output, report)
    print(f"Wrote {args.output / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
