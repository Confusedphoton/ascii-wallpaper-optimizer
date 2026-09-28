"""Batch entry point for still grids and ASCII-aware slideshow transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import __version__
from .assets import glyph_masks, linear_to_srgb, load_target, resolve_font, save_atlas, save_metadata
from .optimizer import Grid, render_grid, solve


DEFAULT_CHARS = " .´:rszunvcxLOGDQ0KNMW&@"


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(item) for item in value.lower().split("x"))
        if width > 0 and height > 0:
            return width, height
    except (ValueError, TypeError):
        pass
    raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_grid(grid: Grid, directory: Path, preview: bool, masks: np.ndarray,
               size: tuple[int, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.zeros((*grid.glyph.shape, 3), dtype=np.uint8)
    indices[:, :, 0] = grid.glyph
    Image.fromarray(indices, "RGB").save(directory / "glyph.png")
    color = linear_to_srgb(torch.as_tensor(grid.color)).numpy()
    Image.fromarray(np.rint(color * 255).astype(np.uint8), "RGB").save(directory / "color.png")
    if preview:
        pixels = render_grid(grid, masks)[:size[1], :size[0]]
        color = linear_to_srgb(torch.as_tensor(pixels)).numpy()
        Image.fromarray(np.rint(color * 255).astype(np.uint8), "RGB").save(directory / "preview.png")


def _load_grid(directory: Path) -> Grid:
    glyph = np.asarray(Image.open(directory / "glyph.png").convert("RGB"))[:, :, 0].copy()
    encoded = np.asarray(Image.open(directory / "color.png").convert("RGB"), dtype=np.float32) / 255
    from .assets import srgb_to_linear
    color = srgb_to_linear(torch.as_tensor(encoded)).numpy()
    return Grid(glyph, color)


def build(args: argparse.Namespace) -> None:
    report_progress = getattr(args, "progress_json", False)
    total_units = len(args.images) + (len(args.images) * (args.frames - 2) if len(args.images) > 1 else 0)
    completed_units = 0

    def report(fraction: float, stage: str) -> None:
        if report_progress:
            print("ASCII_PROGRESS " + json.dumps({
                "progress": min(1.0, (completed_units + fraction) / max(1, total_units)),
                "stage": stage,
            }), flush=True)

    report(0.0, "Preparing input images")
    font = resolve_font(args.font)
    chars = list(args.chars)
    if not chars or len(chars) > 256 or len(set(chars)) != len(chars):
        raise ValueError("Characters must be 1 to 256 distinct glyphs")
    cell = args.cell or ((2, 4), (4, 8), (8, 16))[args.detail]
    masks = glyph_masks(font, chars, cell)
    images = [path.expanduser().resolve() for path in args.images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = {
        "version": __version__, "images": [str(path) for path in images],
        "image_hashes": [_file_hash(path) for path in images],
        "font": str(font), "font_hash": _file_hash(font),
        "size": list(args.size), "cell": list(cell), "chars": args.chars,
        "crop": args.crop, "grayscale": args.grayscale,
        "frames": args.frames, "steps": args.steps,
        "transition_steps": args.transition_steps,
        "candidates": args.candidates, "dither": args.dither, "seed": args.seed,
    }
    output = args.output.expanduser().resolve()
    manifest = output / "manifest.json"
    old = {}
    if manifest.exists() and not args.force:
        try:
            old = json.loads(manifest.read_text())
            expected_stills = [output / "stills" / f"{i:04d}" / f"{kind}.png"
                               for i in range(len(images)) for kind in ("glyph", "color", "preview")]
            expected_pairs = [output / "transitions" / f"{i:04d}-{(i+1)%len(images):04d}" / f"{frame:04d}" / f"{kind}.png"
                              for i in range(len(images)) for frame in range(args.frames)
                              for kind in ("glyph", "color")] if len(images) > 1 else []
            if (old.get("signature") == signature and (output / "atlas.png").exists()
                    and (output / "metadata.png").exists()
                    and all(p.exists() for p in expected_stills + expected_pairs)):
                print(f"Already rendered: {output}")
                report(1.0, "Ready")
                return
        except (OSError, ValueError):
            pass
    output.mkdir(parents=True, exist_ok=True)
    save_atlas(masks, output / "atlas.png")
    save_metadata(args.size, cell, len(chars), output / "metadata.png")
    old_signature = old.get("signature", {})
    settings_match = all(old_signature.get(key) == value for key, value in signature.items()
                         if key not in ("images", "image_hashes"))
    old_hashes = old_signature.get("image_hashes", [])
    targets = [load_target(path, args.size, args.crop, args.grayscale) for path in images]
    stills = []
    for index, target in enumerate(targets):
        still_dir = output / "stills" / f"{index:04d}"
        if (settings_match and index < len(old_hashes)
                and old_hashes[index] == signature["image_hashes"][index]
                and all((still_dir / f"{kind}.png").exists() for kind in ("glyph", "color", "preview"))):
            print(f"Reusing still {index + 1}/{len(images)}", flush=True)
            stills.append(_load_grid(still_dir))
            completed_units += 1
            report(0.0, f"Reused image {index + 1}/{len(images)}")
            continue
        print(f"Optimizing still {index + 1}/{len(images)}: {images[index]}", flush=True)
        grid = solve(target, masks, seed=args.seed + index, steps=args.steps,
                     candidates=args.candidates, dither=args.dither, device=args.device,
                     on_progress=lambda fraction, stage, index=index: report(
                         fraction, f"Image {index + 1}/{len(images)}: {stage}"))
        stills.append(grid)
        _save_grid(grid, still_dir, True, masks, args.size)
        completed_units += 1
        report(0.0, f"Finished image {index + 1}/{len(images)}")
    if len(images) > 1:
        for index in range(len(images)):
            following = (index + 1) % len(images)
            base = output / "transitions" / f"{index:04d}-{following:04d}"
            if (settings_match and index < len(old_hashes) and following < len(old_hashes)
                    and old_hashes[index] == signature["image_hashes"][index]
                    and old_hashes[following] == signature["image_hashes"][following]
                    and all((base / f"{frame:04d}" / f"{kind}.png").exists()
                            for frame in range(args.frames) for kind in ("glyph", "color"))):
                print(f"Reusing transition {index + 1}/{len(images)}", flush=True)
                completed_units += args.frames - 2
                report(0.0, f"Reused transition {index + 1}/{len(images)}")
                continue
            print(f"Optimizing transition {index + 1}/{len(images)}", flush=True)
            previous = stills[index]
            for frame in range(args.frames):
                fraction = frame / (args.frames - 1)
                progress = fraction * fraction * (3 - 2 * fraction)
                if frame == 0:
                    grid = stills[index]
                elif frame == args.frames - 1:
                    grid = stills[following]
                else:
                    target = (1 - progress) * targets[index] + progress * targets[following]
                    grid = solve(target, masks, seed=args.seed + index * 10007,
                                 steps=args.transition_steps, candidates=args.candidates,
                                 dither=args.dither, device=args.device, previous=previous,
                                 destination=stills[following], progress=progress,
                                 on_progress=lambda fraction, stage, index=index, frame=frame: report(
                                     fraction, f"Transition {index + 1}/{len(images)}, frame {frame + 1}/{args.frames}: {stage}"))
                _save_grid(grid, base / f"{frame:04d}", False, masks, args.size)
                previous = grid
                if frame not in (0, args.frames - 1):
                    completed_units += 1
                    report(0.0, f"Finished transition frame {frame + 1}/{args.frames}")
    manifest.write_text(json.dumps({"signature": signature, "image_count": len(images),
                                    "frame_count": args.frames, "grid": [stills[0].glyph.shape[1], stills[0].glyph.shape[0]]}, indent=2) + "\n")
    print(f"Rendered assets: {output}")
    report(1.0, "Ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Source images in slideshow order")
    parser.add_argument("--output", type=Path, required=True, help="Generated asset directory")
    parser.add_argument("--size", type=_size, required=True, help="Wallpaper size, WIDTHxHEIGHT")
    parser.add_argument("--font", default="Monospace", help="Font family or font file")
    parser.add_argument("--chars", default=DEFAULT_CHARS, help="Distinct characters, including a space")
    parser.add_argument("--detail", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--cell", type=_size, help="Override cell dimensions")
    parser.add_argument("--crop", choices=("fill", "fit"), default="fill")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frames", type=int, default=16, help="Frames per slideshow transition")
    parser.add_argument("--steps", type=int, default=36, help="Still optimization steps")
    parser.add_argument("--transition-steps", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--dither", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 2 or args.steps < 1 or args.transition_steps < 1 or args.candidates < 1:
        parser.error("frames must be >= 2 and optimization counts must be positive")
    build(args)


if __name__ == "__main__":
    main()
