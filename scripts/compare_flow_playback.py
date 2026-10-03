"""Generate actual serialized flow clips and seam diagnostics for the FLIP viewer.

  .venv/bin/python -m ascii_optimizer.launcher scripts/compare_flow_playback.py --device cuda
  .venv/bin/python scripts/view_renders.py --report rendered/flow_playback/review
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import experiment_utils as base
from compare_flow import SOURCES
from ascii_optimizer import cli
from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import FLOW_CHARS, ppd_for_width
from ascii_optimizer.runtime import select_device
from ascii_optimizer.tone import srgb_to_oklab

VARIANTS = {
    "flow_static": {},
    "flow_animated": {"flow_animated": True},
    "flow_animated_smooth": {"flow_animated": True, "flow_temporal_strength": 1.4},
    "flow_animated_stable": {"flow_animated": True, "flow_temporal_strength": 2.0},
    "flow_animated_strong": {"flow_animated": True, "flow_eddies": 1.4},
}


def review_variant(directory, target, masks, size, ppd):
    manifest = json.loads((directory / "assets/manifest.json").read_text())
    first = cli._load_grid(directory / "assets/stills/0000")
    initial = base.preview(first, masks, size)
    base.image(initial).save(directory / "final.png")
    result = {**base.image_metrics(target, initial), **base.glyph_metrics(first, masks, FLOW_CHARS)}
    info = manifest.get("loops", {}).get("0000")
    if not info:
        return result
    previews = directory / "frames"
    previews.mkdir(exist_ok=True)
    count = info["frame_count"]
    differences, lightness, sampled_flip = [], [], {}
    previous, first_pixels, previous_delta = None, None, None
    acceleration = []
    sample_indices = {0, count - 1, count // 4, count // 2, 3 * count // 4}
    first_delta = last_delta = None
    for index in range(count):
        grid = cli._load_grid(directory / "assets" / info["path"] / f"{index:04d}")
        pixels = base.preview(grid, masks, size)
        base.image(pixels).save(previews / f"{index:04d}.png")
        if index == 0:
            first_pixels = pixels
        if previous is not None:
            delta = pixels - previous
            differences.append(float(np.sqrt(np.mean(delta**2))))
            lightness.append(float(np.abs(srgb_to_oklab(pixels)[..., 0] - srgb_to_oklab(previous)[..., 0]).mean()))
            if first_delta is None:
                first_delta = delta
            if previous_delta is not None:
                acceleration.append(float(np.sqrt(np.mean((delta - previous_delta)**2))))
            previous_delta = delta
            last_delta = delta
        if index in sample_indices:
            import flip_evaluator
            _, mean, _ = flip_evaluator.evaluate(target, pixels, "LDR", inputsRGB=True,
                applyMagma=False, parameters={"ppd": ppd})
            sampled_flip[str(index)] = float(mean)
        previous = pixels
    seam = first_pixels - previous
    seam_rms = float(np.sqrt(np.mean(seam**2)))
    seam_acceleration = max(float(np.sqrt(np.mean((seam - last_delta)**2))),
                            float(np.sqrt(np.mean((first_delta - seam)**2))))
    result.update(clip={"duration_ms": info["duration_ms"],
                        "frames": [f"frames/{index:04d}.png" for index in range(count)]},
                  closure=info["closure"], pixel_step_mean=float(np.mean(differences)),
                  pixel_step_p95=float(np.quantile(differences, .95)), pixel_seam_rms=seam_rms,
                  pixel_seam_ratio=seam_rms / max(float(np.mean(differences)), 1e-8),
                  lightness_step_mean=float(np.mean(lightness)),
                  lightness_seam=float(np.abs(srgb_to_oklab(first_pixels)[..., 0] - srgb_to_oklab(previous)[..., 0]).mean()),
                  acceleration_p95=float(np.quantile(acceleration, .95)), seam_acceleration=seam_acceleration,
                  sampled_flip=sampled_flip)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cell", type=int, nargs=2, default=(8, 16))
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--frame-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--font", default="Monospace")
    parser.add_argument("--images", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["flow_static", "flow_animated", "flow_animated_strong"])
    parser.add_argument("--output", type=Path, default=Path("rendered/flow_playback/review"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    font = resolve_font(args.font)
    masks = glyph_masks(font, list(FLOW_CHARS), tuple(args.cell))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args) | {"output": str(args.output)}, "images": {}}
    for name in args.images:
        size = base.source_size(SOURCES[name], args.width)
        target = load_target(SOURCES[name], size, "fill", False)
        scene = args.output / name
        scene.mkdir(parents=True, exist_ok=True)
        base.image(target).save(scene / "source.png")
        data = {"source": str(SOURCES[name]), "size": size, "ppd": ppd_for_width(size[0]), "variants": {}}
        report["images"][name] = data
        for variant in args.variants:
            destination = scene / variant
            arguments = [str(SOURCES[name]), "--output", str(destination / "assets"),
                         "--size", f"{size[0]}x{size[1]}", "--cell", f"{args.cell[0]}x{args.cell[1]}",
                         "--style", "flow", "--font", str(font), "--device", device,
                         "--steps", str(args.steps), "--flow-frame-steps", str(args.frame_steps),
                         "--flow-loop-seconds", str(args.seconds), "--flow-fps", str(args.fps),
                         "--seed", str(args.seed)]
            for key, value in VARIANTS[variant].items():
                arguments += ["--" + key.replace("_", "-")] + ([] if value is True else [str(value)])
            if args.force:
                arguments.append("--force")
            if args.progress:
                arguments.append("--progress-json")
            print(f"{name}: {variant}, {size}, {args.seconds * args.fps} animation frames", flush=True)
            began = time.perf_counter()
            cli.build(cli.make_parser().parse_args(arguments))
            measurements = review_variant(destination, target, masks, size, data["ppd"])
            measurements["seconds"] = time.perf_counter() - began
            data["variants"][variant] = measurements
            base.write_report(args.output / "metrics.json", report, sources=[SOURCES[n] for n in args.images], font=args.font)
            print(f"  {measurements['seconds']:.1f}s; seam/ordinary step {measurements.get('pixel_seam_ratio', 0):.3f}", flush=True)
    lines = ["# Animated flow review", "", "The viewer plays serialized RGB8 renders; P plays/pauses and Inspect loop seam focuses on the wrap.", "",
             "| Image | Variant | Seconds | Glyph changes/step | Seam/ordinary pixel step | Field seam/ordinary step |",
             "|---|---|---:|---:|---:|---:|"]
    for name, data in report["images"].items():
        for variant, item in data["variants"].items():
            closure = item.get("closure", {})
            lines.append(f"| {name} | {variant} | {item['seconds']:.1f} | {closure.get('glyph_change_mean', 0):.1%} | "
                         f"{item.get('pixel_seam_ratio', 0):.3f} | {closure.get('velocity_seam_ratio', 0):.3f} |")
    (args.output / "review.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
