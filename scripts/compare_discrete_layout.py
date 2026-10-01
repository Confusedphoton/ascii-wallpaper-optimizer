"""Compare the previous renderer, exact contextual refinement, and hard glyphs.

Run before committing with --baseline-ref HEAD, or supply an older commit/file:
    .venv/bin/python -m ascii_optimizer.launcher scripts/compare_discrete_layout.py --device cuda
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

import experiment_utils as base
from ascii_optimizer import cli, optimizer
from ascii_optimizer.assets import glyph_masks, resolve_font
from compare_error_movement import SOURCES, focus_metrics, tail_boundary_metrics


def load_baseline(source: Path, suffix: str = "baseline"):
    name = f"ascii_optimizer._discrete_layout_{suffix}"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_review(output: Path, report: dict) -> None:
    links = []
    lines = ["# Discrete glyph layout comparison", "",
             "All runs hold the source, font, glyph bank, seed, steps, and production objective weights fixed.",
             "`soft_exact` isolates exact neighboring-pixel refinement; `hard_exact` also removes mask blending.",
             "`hard_layout` adds the experimental cell-aligned patch term. Lower errors are better.", "",
             "| Image / cell | Variant | Common discrete objective | Edge RMSE ×100 | Focus ΔE ×100 | Seconds | GPU peak MiB |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, image in report["images"].items():
        for cell, variants in image["grids"].items():
            for key, metrics in variants.items():
                run = metrics["experiment"]
                memory = run.get("cuda_peak_allocated_bytes", 0) / 1024 ** 2
                lines.append(f"| {name} / {cell} | {key} | {metrics['common_discrete_objective']:.6f} | "
                             f"{metrics['gradient_rmse_x100']:.3f} | {metrics['focus']['mean_oklab_delta_e_x100']:.3f} | "
                             f"{run['seconds']:.2f} | {memory:.1f} |")
            directory = f"{name}/{cell}"
            links.append(f"[{name}, {cell}: complete previews]({directory}/comparison.png) · "
                         f"[enlarged focus]({directory}/focus_comparison.png)")
    lines.extend(["", *links])
    lines.extend(["", "The common objective excludes the experimental layout term and includes production glyph variety and ink rewards.",
                  "Pixel and edge metrics measure serialized RGB8 playback. GPU memory is measured per solve; RSS is the process high-water mark.",
                  "See `metrics.json` for source hashes, baseline code hash, settings, and complete measurements.", ""])
    if report.get("reference_sha256"):
        lines.extend(["`reference` uses the additional supplied optimizer snapshot, saved as `optimizer_reference.py`.", ""])
    (output / "review.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rendered/discrete_layout"))
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cells", nargs="+", default=["12x24", "8x16"])
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--refinement-sweeps", type=int, default=4)
    parser.add_argument("--polish-steps", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layout-strength", type=float, default=0.5)
    parser.add_argument("--variants", nargs="+", choices=("before", "reference", "soft_exact", "hard_exact", "hard_layout"))
    args = parser.parse_args()
    if args.variants is None:
        args.variants = ["before", "soft_exact", "hard_exact", "hard_layout"]
        if args.reference_file is not None:
            args.variants.insert(1, "reference")
    if "reference" in args.variants and args.reference_file is None:
        parser.error("the reference variant requires --reference-file")
    try:
        cells = [tuple(map(int, value.split("x"))) for value in args.cells]
        if any(len(cell) != 2 or min(cell) < 1 for cell in cells):
            raise ValueError
    except ValueError:
        parser.error("cells must be positive WIDTHxHEIGHT pairs")
    if args.width < 1 or min(args.steps, args.refinement_sweeps, args.polish_steps) < 0 or not np.isfinite(args.layout_strength) or args.layout_strength < 0:
        parser.error("width must be positive; steps and layout strength must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    source = (args.baseline_file.read_bytes() if args.baseline_file else subprocess.run(
        ["git", "show", f"{args.baseline_ref}:ascii_optimizer/optimizer.py"], capture_output=True, check=True).stdout)
    baseline_path = args.output.resolve() / "optimizer_before.py"
    baseline_path.write_bytes(source)
    baseline = load_baseline(baseline_path)
    reference_module = reference_source = None
    if args.reference_file is not None:
        reference_source = args.reference_file.read_bytes()
        reference_path = args.output.resolve() / "optimizer_reference.py"
        reference_path.write_bytes(reference_source)
        reference_module = load_baseline(reference_path, "reference")
    renderer_source = Path(optimizer.__file__).read_bytes()
    (args.output / "optimizer_after.py").write_bytes(renderer_source)
    base.reset_runs()
    report = {"method": "exact single-cell replacement deltas with neighboring pixels and global pooling alignment",
              "baseline_ref": args.baseline_ref if args.baseline_file is None else None,
              "baseline_sha256": hashlib.sha256(source).hexdigest(),
              "reference_sha256": hashlib.sha256(reference_source).hexdigest() if reference_source is not None else None,
              "renderer_sha256": hashlib.sha256(renderer_source).hexdigest(), "variants": args.variants,
              "width": args.width, "cells": cells, "steps": args.steps,
              "refinement_sweeps": args.refinement_sweeps, "polish_steps": args.polish_steps,
              "layout_strength": args.layout_strength, "seed": base.SEED, "dither": base.DITHER,
              "glyphs": base.DEFAULT_CHARS, "images": {}}
    for name, path, crop in SOURCES:
        size = base.source_size(path, args.width)
        reference = base.load_target(path, size, "fill", False)
        crop = tuple(round(coordinate * args.width / 960) for coordinate in crop)
        image_report = {"source": str(path), "size": size, "focus_crop": crop, "grids": {}}
        report["images"][name] = image_report
        for cell in cells:
            directory = args.output / name / f"{cell[0]}x{cell[1]}"
            directory.mkdir(parents=True, exist_ok=True)
            masks = glyph_masks(resolve_font("Monospace"), list(base.DEFAULT_CHARS), cell)
            source_image = base.image(reference)
            source_image.save(directory / "source.png")
            panels = [("Source", source_image)]
            variants = {}
            baseline_glyph = None
            for key, module, soft, layout in (
                ("before", baseline, False, 0.),
                ("reference", reference_module, False, 0.),
                ("soft_exact", optimizer, True, 0.),
                ("hard_exact", optimizer, False, 0.),
                ("hard_layout", optimizer, False, args.layout_strength),
            ):
                if key not in args.variants:
                    continue
                print(f"{name} {cell}: {key}", flush=True)
                objectives = []
                original_objective = module._FrameObjective
                def capture_objective(*positional, **keyword):
                    objective = original_objective(*positional, **keyword)
                    objectives.append(objective)
                    return objective
                parameters = inspect.signature(module.solve).parameters
                def solve(*positional, **keyword):
                    if "perceptual_loss" in parameters:
                        keyword["perceptual_loss"] = False
                    if module is optimizer:
                        keyword["layout_loss_strength"] = layout
                    if "refinement_sweeps" in parameters:
                        keyword["refinement_sweeps"] = args.refinement_sweeps
                    if "polish_steps" in parameters:
                        keyword["polish_steps"] = args.polish_steps
                    # Older snapshots predate independent budgets. Keep their
                    # own historical behavior and record which settings apply.
                    return module.solve(*positional, **{key: value for key, value in keyword.items() if key in parameters})
                # Use the CLI's source encoding and production defaults for every
                # variant, while capturing the exact objective for final scoring.
                with patch.object(cli, "solve", solve), patch.object(module, "_FrameObjective", capture_objective):
                    relaxation = (patch.object(optimizer, "_hard_weights", lambda logits, temperature: torch.softmax(logits / temperature, 1))
                                  if soft else nullcontext())
                    with relaxation:
                        grid = base.solve_style(reference, masks, style="headroom", invert_ink=False,
                            background="auto", seed=base.SEED, steps=args.steps, dither=base.DITHER,
                            structure_dither=0.06, device=args.device)
                if baseline_glyph is None:
                    baseline_glyph = grid.glyph.copy()
                rendered = base.preview(grid, masks, size)
                shown = base.image(rendered)
                shown.save(directory / f"{key}.png")
                np.savez_compressed(directory / f"{key}.npz", glyph=grid.glyph, color=grid.color, background=grid.background)
                panels.append((key, shown))
                objective = objectives[0]
                # Score every saved grid using the same non-layout objective.
                common = optimizer._FrameObjective(objective.target, objective.detail_loss, objective.cell,
                    objective.invert_ink, objective.gradient_strength, objective.saturation_strength,
                    objective.gradient_target, objective.saturation_target)
                lab = torch.as_tensor(optimizer.render_grid(grid, masks, args.device), device=args.device).permute(2, 0, 1)[None]
                with torch.no_grad():
                    score = common(lab).double()
                    counts = torch.bincount(torch.as_tensor(grid.glyph.astype(np.int64), device=args.device).flatten(), minlength=len(masks)).float()
                    score += base.DEFAULT_GLYPH_ENTROPY_STRENGTH * optimizer._negative_glyph_entropy(counts / grid.glyph.size)
                    ink = torch.as_tensor(grid.color, device=args.device).reshape(-1, 3)
                    presence = torch.as_tensor(masks, device=args.device).flatten(1).gt(0).any(1)[torch.as_tensor(grid.glyph.astype(np.int64), device=args.device).flatten()]
                    score -= base.DEFAULT_GLYPH_LUMINANCE_BONUS * optimizer._ink_lightness_reward(ink, presence).mean()
                variants[key] = {"layout_loss_strength": layout, "hard_forward": hasattr(module, "_hard_weights") and not soft,
                    "refinement_sweeps": args.refinement_sweeps if "refinement_sweeps" in parameters else "historical",
                    "polish_steps": args.polish_steps if "polish_steps" in parameters else "historical",
                    "common_discrete_objective": float(score), "experiment": grid.experiment,
                    **base.image_metrics(reference, rendered),
                    **base.glyph_metrics(grid, masks, base.DEFAULT_CHARS, baseline_glyph),
                    "gradient_rmse_x100": base.gradient_rmse(reference, rendered),
                    "focus": focus_metrics(reference, rendered, crop)}
                if name == "tail":
                    variants[key]["boundary"] = tail_boundary_metrics(reference, grid, masks, crop)
            base.contact_sheet(panels, directory / "comparison.png", columns=3)
            base.contact_sheet(panels, directory / "focus_comparison.png", crop=crop, columns=3, zoom=3)
            image_report["grids"][f"{cell[0]}x{cell[1]}"] = variants
            base.write_report(args.output / "metrics.json", report)
            write_review(args.output, report)
    print(f"Wrote {args.output / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
