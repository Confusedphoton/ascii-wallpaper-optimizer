"""Render reproducible flow-domain comparisons for the FLIP web viewer.

  .venv/bin/python -m ascii_optimizer.launcher scripts/compare_flow.py --device cuda
  .venv/bin/python scripts/view_renders.py --report rendered/flow_domains/review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import experiment_utils as base
from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import (DEFAULT_CHARS, FLOW_CHARS, DEFAULT_FLOW_STRENGTH,
                                      DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH, ppd_for_width)
from ascii_optimizer.runtime import select_device

SOURCES = {
    "tree": Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg"),
    "viviana": Path("screenshots/143381492_p0-Viviana.jpg"),
    "snow": Path("screenshots/149642007_p0-无题.jpg"),
    "typhon": Path("screenshots/__typhon_and_typhoeus_arknights_and_1_more_drawn_by_mokacoat__4cc6db689e0953c4728dc3e2db472b19.jpg"),
}
VARIANTS = {
    "perceptual_standard": {"style": "headroom", "chars": DEFAULT_CHARS},
    "perceptual_wide": {"style": "headroom", "chars": FLOW_CHARS},
    "flow_calm": {"style": "flow", "flow_eddies": 0.0},
    "flow_default": {"style": "flow"},
    "flow_eddies": {"style": "flow", "flow_eddies": 1.4},
    "flow_fine": {"style": "flow", "flow_scale": 3.0},
}


def plot_flow(field, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    u, v = field.velocity[0].cpu().numpy()
    speed = np.hypot(u, v)
    h, w = speed.shape
    x = (np.arange(w) + 0.5) * field.shape[1] / w
    y = (np.arange(h) + 0.5) * field.shape[0] / h
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 6 * h / w)), layout="constrained")
    flow = axes[0].streamplot(x, y, u, v, color=speed, cmap="viridis", norm=Normalize(0, 5),
                              density=1.4, linewidth=0.65, arrowsize=0.6)
    fig.colorbar(flow.lines, ax=axes[0], label="Flow speed (simulation samples / step)")
    axes[0].set_title("Frozen forced flow")
    confidence = axes[1].imshow(field.source_confidence[0, 0].cpu().numpy(), origin="lower",
                               extent=(0, field.shape[1], 0, field.shape[0]), cmap="cividis", vmin=0, vmax=1)
    fig.colorbar(confidence, ax=axes[1], label="Image-axis confidence (0–1)")
    axes[1].set_title("Image guidance")
    for ax in axes:
        ax.set(xlabel="Image $x$ (pixels)", ylabel="Image $y$ (pixels)", aspect="equal")
        ax.invert_yaxis()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cell", type=int, nargs=2, default=(8, 16), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--font", default="Monospace")
    parser.add_argument("--images", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--output", type=Path, default=Path("rendered/flow_domains/review"))
    parser.add_argument("--flow-strength", type=float, default=DEFAULT_FLOW_STRENGTH)
    parser.add_argument("--glyph-variety", type=float, default=DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH)
    args = parser.parse_args()
    if min(args.width, *args.cell, args.steps) < 1:
        parser.error("Width, cell dimensions and steps must be positive")
    device = select_device(args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    font = resolve_font(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"settings": {"width": args.width, "cell": args.cell, "steps": args.steps,
                            "seed": args.seed, "glyph_variety": args.glyph_variety}, "images": {}}
    for name in args.images:
        source = SOURCES[name]
        size = base.source_size(source, args.width)
        target = load_target(source, size, "fill", False)
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        base.image(target).save(directory / "source.png")
        image_report = {"source": str(source), "size": size, "cell": args.cell,
                        "ppd": ppd_for_width(size[0]), "variants": {}}
        report["images"][name] = image_report
        panels = [("Source", base.image(target))]
        baseline = None
        for variant in args.variants:
            settings = dict(VARIANTS[variant])
            chars = settings.pop("chars", FLOW_CHARS)
            masks = glyph_masks(font, list(chars), tuple(args.cell))
            destination = directory / variant
            destination.mkdir(parents=True, exist_ok=True)
            flow = {}
            print(f"{name}: {variant} ({len(chars)} glyphs, {size})", flush=True)
            grid = base.solve_style(target, masks, invert_ink=False, background="auto",
                                    seed=args.seed, steps=args.steps, dither=0.005, structure_dither=0.06,
                                    device=device, glyph_entropy_strength=args.glyph_variety,
                                    flow_strength=args.flow_strength, flow_diagnostics=flow, **settings)
            shown = base.preview(grid, masks, size)
            base.image(shown).save(destination / "final.png")
            np.savez_compressed(destination / "grid.npz", glyph=grid.glyph, color=grid.color,
                                background=grid.background, chars=np.asarray(list(chars)))
            measurements = {**base.image_metrics(target, shown), **base.glyph_metrics(grid, masks, chars),
                            "seconds": grid.experiment["seconds"], "bank_size": len(chars),
                            "settings": grid.experiment["effective_objectives"]}
            import flip_evaluator
            heatmap, mean, _ = flip_evaluator.evaluate(target, shown, "LDR", inputsRGB=True,
                applyMagma=True, parameters={"ppd": image_report["ppd"]})
            base.image(heatmap).save(destination / "flip.png")
            measurements["flip_mean"] = float(mean)
            if flow:
                field, costs = flow.pop("field"), flow.pop("costs")
                chosen = torch.as_tensor(grid.glyph.astype(np.int64), device=device).flatten()
                measurements["flow"] = {**flow, "selected_flow_cost": float(costs.gather(1, chosen[:, None]).mean())}
                np.savez_compressed(destination / "field.npz", velocity=field.velocity.cpu().numpy(),
                                    confidence=field.source_confidence.cpu().numpy())
                plot_flow(field, destination / "flow.png")
            if variant == "perceptual_wide":
                baseline = grid.glyph.copy()
            if baseline is not None and variant.startswith("flow_"):
                measurements["glyph_change_vs_wide"] = float((grid.glyph != baseline).mean())
            image_report["variants"][variant] = measurements
            panels.append((variant.replace("_", " "), base.image(shown)))
            print(f"  FLIP {mean:.4f}; {measurements['distinct_glyphs']} glyphs; "
                  f"effective {measurements['effective_glyph_count']:.1f}; {measurements['seconds']:.1f}s", flush=True)
            base.write_report(args.output / "metrics.json", report, sources=[SOURCES[n] for n in args.images], font=args.font)
            del grid, masks, flow
        base.contact_sheet(panels, directory / "comparison.png", columns=3, panel=(480, round(480 * size[1] / size[0])))
        crop = (round(size[0] * .25), round(size[1] * .25), round(size[0] * .6), round(size[1] * .5))
        base.contact_sheet(panels, directory / "detail.png", columns=3, crop=crop, zoom=3)
    lines = ["# Flow domains review", "", "All previews use serialized RGB8 playback colors and actual font masks.",
             "The expanded bank has 96 glyphs. Perceptual wide is the control with the same bank, seed, color fitting and variety weight.",
             "", "| Image | Variant | FLIP | Used glyphs | Effective glyph count | Most common share | Solve seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, data in report["images"].items():
        for variant, item in data["variants"].items():
            lines.append(f"| {name} | {variant} | {item['flip_mean']:.4f} | {item['distinct_glyphs']} | "
                         f"{item['effective_glyph_count']:.1f} | {item['most_common_glyph_share']:.1%} | {item['seconds']:.1f} |")
        lines += ["", f"[{name} comparison]({name}/comparison.png) · [glyph detail]({name}/detail.png)", ""]
    (args.output / "review.md").write_text("\n".join(lines) + "\n")
    print(f"Review saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
