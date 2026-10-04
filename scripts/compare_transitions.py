"""Compare saved slideshow transitions with current code using identical stills.

  .venv/bin/python -m ascii_optimizer.launcher scripts/compare_transitions.py /path/to/assets --output rendered/transitions/review
  .venv/bin/python scripts/view_renders.py --report rendered/transitions/review

The input assets remain untouched. Review previews are composed at the saved
display resolution before resizing; measurements use the full-resolution RGB8
output. The viewer repeats a complete slideshow with a short hold at each still.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from ascii_optimizer import cli
from ascii_optimizer.optimizer import render_grid
from ascii_optimizer.revision import render_revision
from ascii_optimizer.tone import display_tone_lab, oklab_to_srgb, srgb_to_oklab
from ascii_optimizer.transition import TYPES, glyph_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path, help="Completed slideshow asset directory")
    parser.add_argument("--output", type=Path, default=Path("rendered/transitions/review"))
    parser.add_argument("--width", type=int, default=1280, help="Review preview width")
    parser.add_argument("--duration-ms", type=int, default=1500, help="Playback duration of each transition")
    parser.add_argument("--transition-type", choices=TYPES, help="Override the saved transition type")
    args = parser.parse_args()
    original = args.assets.resolve()
    output = args.output.resolve()
    if output == original or original in output.parents or output in original.parents:
        parser.error("Review output must be separate from the input assets")
    if args.width < 1 or args.duration_ms < 1:
        parser.error("Width and duration must be positive")
    manifest = json.loads((original / "manifest.json").read_text())
    count, frame_count = manifest["image_count"], manifest["frame_count"]
    if count < 2:
        parser.error("The input must contain at least two slideshow images")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    options = SimpleNamespace(**(manifest["signature"] | {
        "size": tuple(manifest["output_size"]), "frames": frame_count,
        "invert_ink": manifest["invert_ink"],
        "transition_type": args.transition_type or manifest["signature"].get("transition_type", "glyph-path"),
    }))
    cell = tuple(manifest["cell_size"])
    with Image.open(original / "atlas.png") as atlas:
        alpha = np.asarray(atlas.convert("RGBA"), dtype=np.float32)[..., 3] / 255
    masks = alpha.reshape(cell[1], manifest["atlas_count"], cell[0]).transpose(1, 0, 2)
    paths = glyph_paths(masks)
    revised = output / "interpolated_assets"
    revised.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original / "stills", revised / "stills", dirs_exist_ok=True)
    if manifest.get("loops"):
        shutil.copytree(original / "loops", revised / "loops", dirs_exist_ok=True)
    for filename in ("atlas.png", "metadata.png"):
        shutil.copyfile(original / filename, revised / filename)
    backgrounds = {key: value for key, value in manifest["background_labs"].items()
                   if not key.startswith("transitions/")}
    for index in range(count):
        backgrounds.update(cli._render_transition(index, [Path(p) for p in options.images],
            revised, options, masks, cell, paths, lambda *_: None))
    updated = manifest | {"background_labs": backgrounds, "generation": "transition-review",
                          "signature": manifest["signature"] | {"render_revision": render_revision(),
                              "transition_type": options.transition_type}}
    (revised / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
    scene = output / "slideshow"
    scene.mkdir(exist_ok=True)
    report = {"input_assets": str(original), "input_signature": manifest["signature"],
              "render_revision": render_revision(), "images": {"slideshow": {
                  "label": "Your slideshow", "ppd": manifest["signature"].get("viewing_ppd") or 67,
                  "variants": {}}}}
    preview_size = (min(args.width, options.size[0]),
                    max(1, round(options.size[1] * min(args.width, options.size[0]) / options.size[0])))
    revised_variant = "earth_mover" if options.transition_type == "earth-mover" else "oklab_interpolation"
    for variant, assets in (("original", original), (revised_variant, revised)):
        directory = scene / variant
        directory.mkdir(exist_ok=True)
        frames, metrics = [], {}
        print(f"Composing {variant} at {options.size}, previews {preview_size}", flush=True)
        for index in range(count):
            following = (index + 1) % count
            pair = f"{index:04d}-{following:04d}"
            lightness, ink_lightness, pixel_steps = [], [], []
            previous = None
            pair_frames = []
            for frame in range(frame_count):
                grid = cli._load_grid(assets / "transitions" / pair / f"{frame:04d}")
                lab = display_tone_lab(render_grid(grid, masks)[:options.size[1], :options.size[0]],
                                       invert_ink=options.invert_ink)
                pixels = np.rint(oklab_to_srgb(lab) * 255).astype(np.uint8)
                lightness.append(float(srgb_to_oklab(pixels.astype(np.float32) / 255)[..., 0].mean()))
                ink_lightness.append(float(grid.color[..., 0].mean()))
                if previous is not None:
                    pixel_steps.append(float(np.abs(pixels.astype(np.float32) - previous).mean() / 255))
                previous = pixels.astype(np.float32)
                filename = f"{pair}-{frame:04d}.png"
                image = Image.fromarray(pixels).resize(preview_size, Image.Resampling.LANCZOS)
                image.save(directory / filename)
                pair_frames.append(filename)
                if index == 0 and frame == 0:
                    image.save(scene / "source.png")
                if index == 0 and frame == frame_count // 2:
                    image.save(directory / "final.png")
            frames.extend([pair_frames[0]] * round(500 / (args.duration_ms / frame_count)))
            frames.extend(pair_frames)
            metrics[pair] = {"displayed_oklab_lightness": lightness, "ink_oklab_lightness": ink_lightness,
                             "pixel_steps": pixel_steps,
                             "first_frame_lightness_change": lightness[1] - lightness[0],
                             "last_frame_lightness_change": lightness[-1] - lightness[-2],
                             "pixel_step_max": max(pixel_steps)}
            print(f"  {pair}: entry lightness change {lightness[1] - lightness[0]:+.5f}, "
                  f"exit {lightness[-1] - lightness[-2]:+.5f}", flush=True)
        report["images"]["slideshow"]["variants"][variant] = {
            "clip": {"frames": frames, "duration_ms": round(len(frames) * args.duration_ms / frame_count)},
            "pairs": metrics,
        }
        # A contiguous sequence also supports video export at the saved timing.
        video_frames = directory / "video_frames"
        shutil.rmtree(video_frames, ignore_errors=True)
        video_frames.mkdir(exist_ok=True)
        for number, frame in enumerate(frames):
            shutil.copyfile(directory / frame, video_frames / f"{number:04d}.png")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Review report: {output}", flush=True)


if __name__ == "__main__":
    main()
