"""Trace one production render for source/current/revised objective comparisons.

Run each revision into a different --label directory, then use view_renders.py
with the source.png and the respective final.png files. No wallpaper state is
modified. Settings can be read from a completed render's manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import experiment_utils as base
from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS, ppd_for_width
from ascii_optimizer.diagnostics import SolverTrace
from ascii_optimizer.optimizer import Grid
from ascii_optimizer.perceptual import SpatialMetrics
from ascii_optimizer.revision import render_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rendered/objective_review"))
    parser.add_argument("--label", default="current")
    parser.add_argument("--width", type=int)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess", action="store_true",
                        help="Compare the optional dither/color-polish output instead of the normal Adam terminal")
    args = parser.parse_args()
    sig = json.loads(args.manifest.read_text())["signature"] if args.manifest else {}
    native_size = sig.get("size", [1707, 1067])
    width = args.width or native_size[0]
    size = (width, round(native_size[1] * width / native_size[0]))
    scale = width / native_size[0]
    cell = tuple(max(1, round(x * scale)) for x in sig.get("cell", [7, 14]))
    chars = sig.get("chars", DEFAULT_CHARS)
    font = resolve_font(sig.get("font", "Adwaita Mono"))
    masks = glyph_masks(font, list(chars), cell)
    target = load_target(args.image, size, sig.get("crop", "fill"), sig.get("grayscale", False))
    output = args.output / args.label
    output.mkdir(parents=True, exist_ok=True)
    base.image(target).save(args.output / "source.png")
    torch.set_num_threads(8)
    trace = SolverTrace(gradient_sample_every=max(1, args.steps), include_replacements=False)
    revision = render_revision()
    kwargs = dict(style="headroom", invert_ink=sig.get("invert_ink", False),
                  background=sig.get("background", "auto"), seed=sig.get("seed", 0),
                  steps=args.steps, dither=sig.get("dither", 0.005), structure_dither=0,
                  device=args.device, glyph_luminance_bonus=sig.get("glyph_luminance_bonus", 0.12),
                  gradient_loss_strength=sig.get("gradient_loss_strength", 1.5),
                  glyph_entropy_strength=sig.get("glyph_entropy_strength", 0.01),
                  diagnostics=trace)
    if args.postprocess:
        kwargs["postprocess"] = True
    started = time.monotonic()
    def progress(fraction, stage):
        if fraction == 1 or fraction * 45 % 6 < 1:
            print(f"{args.label}: {fraction:.0%} {stage} ({time.monotonic()-started:.1f}s)", flush=True)
    grid = base.solve_style(target, masks, **kwargs, on_progress=progress)
    shown = base.preview(grid, masks, size, invert_ink=kwargs["invert_ink"])
    base.image(shown).save(output / "final.png")
    saved = trace.save(output)
    if "reference_range_linear_rgb" in saved["metadata"]:
        from ascii_optimizer.assets import linear_to_srgb
        low, high = (torch.tensor(v) for v in saved["metadata"]["reference_range_linear_rgb"])
        linear = base.srgb_to_linear(torch.from_numpy(target))
        reference = linear_to_srgb(low + linear * (high - low)).numpy()
        base.image(reference).save(output / "brightness_reference.png")
    states = np.load(output / "grids.npz")
    stages = []
    for row in saved["records"]:
        step = row.get("step", 0)
        if row["stage"] == "adam_step" and step + 1 not in (6, 12, 24, args.steps):
            continue
        key = row["stage"] + (f"_{step+1}" if row["stage"] == "adam_step" else "")
        idx = row["grid_index"]
        state = Grid(states["glyph"][idx], states["color"][idx], states["background"])
        preview = base.preview(state, masks, size, invert_ink=kwargs["invert_ink"])
        base.image(preview).save(output / f"{key}.png")
        stages.append(dict(stage=key, loss=row["loss"], components=row["components"],
                           glyph_histogram=base.glyph_metrics(state, masks, chars)["glyph_histogram"]))
    model = SpatialMetrics(base.srgb_to_linear(torch.from_numpy(target).permute(2, 0, 1)[None]).to(args.device),
                           ppd=ppd_for_width(width), cell_height=cell[1])
    metrics = dict(revision=revision, source=str(args.image.resolve()),
                   source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                   size=size, cell=cell, chars=chars, steps=args.steps,
                   settings={k:v for k,v in kwargs.items() if k != "diagnostics"},
                   seconds=time.monotonic()-started,
                   image=base.image_metrics(target, shown), glyph=base.glyph_metrics(grid, masks, chars),
                   spatial=model.measure(base.srgb_to_linear(torch.from_numpy(shown).permute(2,0,1)[None]).to(args.device)),
                   stages=stages)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"seconds":metrics["seconds"], "image":metrics["image"],
                      "orientation_w1":metrics["spatial"]["orientation_w1"],
                      "stroke_error":metrics["spatial"]["orientation_stroke_tensor"]}), flush=True)


if __name__ == "__main__":
    main()
