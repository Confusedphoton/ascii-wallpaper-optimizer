"""Batch entry point for still grids and ASCII-aware slideshow transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from . import __version__
from .assets import cell_for_display, glyph_masks, load_target, resolve_font, save_atlas, save_metadata
from .defaults import DEFAULT_CHARS, DEFAULT_GLYPH_LUMINANCE_BONUS, DEFAULT_GRADIENT_LOSS_STRENGTH
from .optimizer import Grid, _choose_background, render_grid, solve
from .structure import STRUCTURE_CHARS, STRUCTURE_DITHER, solve_structure
from .tone import (display_tone_lab, encode_headroom_lab,
                   invert_source, oklab_to_srgb, srgb_to_oklab)
from .transition import FIELDS, PACES, ease, glyph_paths, local_progress, path_glyphs, phase_offsets, recolor


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
               size: tuple[int, int], invert_ink: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.zeros((*grid.glyph.shape, 3), dtype=np.uint8)
    indices[:, :, 0] = grid.glyph
    Image.fromarray(indices, "RGB").save(directory / "glyph.png")
    color = oklab_to_srgb(grid.color)
    Image.fromarray(np.rint(color * 255).astype(np.uint8), "RGB").save(directory / "color.png")
    background = np.zeros(3, dtype=np.float32) if grid.background is None else grid.background
    background_color = oklab_to_srgb(background).reshape(1, 1, 3)
    Image.fromarray(np.rint(background_color * 255).astype(np.uint8), "RGB").save(directory / "background.png")
    if preview:
        pixels = display_tone_lab(render_grid(grid, masks)[:size[1], :size[0]], invert_ink=invert_ink)
        color = oklab_to_srgb(pixels)
        Image.fromarray(np.rint(color * 255).astype(np.uint8), "RGB").save(directory / "preview.png")


def _load_grid(directory: Path) -> Grid:
    glyph = np.asarray(Image.open(directory / "glyph.png").convert("RGB"))[:, :, 0].copy()
    encoded = np.asarray(Image.open(directory / "color.png").convert("RGB"), dtype=np.float32) / 255
    color = srgb_to_oklab(encoded)
    encoded_background = np.asarray(Image.open(directory / "background.png").convert("RGB"), dtype=np.float32) / 255
    background = srgb_to_oklab(encoded_background).reshape(3)
    return Grid(glyph, color, background)


def solve_style(target: np.ndarray, masks: np.ndarray, *, style: str,
                invert_ink: bool, background: str, seed: int, steps: int,
                candidates: int, dither: float, structure_dither: float,
                logit_noise_distribution: str = "uniform",
                device: str, previous: Grid | None = None,
                destination: Grid | None = None, progress: float | None = None,
                on_progress: Callable[[float, str], None] | None = None,
                detail_loss: str = "default",
                objective_space: str = "oklab",
                glyph_luminance_bonus: float = 0.0,
                saturation_loss_strength: float | None = None,
                gradient_loss_strength: float | None = None) -> Grid:
    """Prepare a grid whose displayed pixels follow the chosen ink polarity."""
    if style not in ("headroom", "structure"):
        raise ValueError(f"Unknown render style: {style}")
    if objective_space != "oklab":
        raise ValueError("The renderer operates only in OKLab.")
    if gradient_loss_strength is None:
        gradient_loss_strength = DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0
    if style == "structure" and glyph_luminance_bonus:
        raise ValueError("glyph_luminance_bonus is only available for the headroom objective")
    if style == "structure" and saturation_loss_strength:
        raise ValueError("saturation_loss_strength is only available for the headroom objective")
    if style == "structure" and gradient_loss_strength:
        raise ValueError("gradient_loss_strength is only available for the headroom objective")
    if style == "structure" and detail_loss != "default":
        raise ValueError("Feature/detail losses are not implemented for structure-first glyph selection")
    source_lab = srgb_to_oklab(target)
    source_lab_tensor = torch.from_numpy(source_lab).permute(2, 0, 1)
    original_background = _choose_background(source_lab_tensor, background).numpy()
    if invert_ink:
        solve_background = (np.zeros(3, dtype=np.float32) if background == "auto"
                            else original_background)
    else:
        solve_background = encode_headroom_lab(original_background.reshape(1, 1, 3))[0, 0]

    if style == "headroom":
        encoded_target = encode_headroom_lab(source_lab, invert_ink=invert_ink)
        return solve(encoded_target, masks, seed=seed, steps=steps,
                     candidates=candidates, dither=dither,
                     logit_noise_distribution=logit_noise_distribution,
                     background_oklab=solve_background, device=device,
                     input_oklab=True,
                     previous=previous, destination=destination,
                     progress=progress, on_progress=on_progress,
                     detail_loss=detail_loss,
                     objective_space=objective_space,
                     glyph_luminance_bonus=glyph_luminance_bonus,
                     saturation_loss_strength=saturation_loss_strength,
                     saturation_target_oklab=source_lab,
                     gradient_loss_strength=gradient_loss_strength,
                     gradient_target_oklab=source_lab,
                     invert_ink=invert_ink)

    selection_target = invert_source(target) if invert_ink else target
    selection_background = solve_background if invert_ink else original_background
    grid = solve_structure(selection_target, masks, selection_background,
                           seed=seed, dither=structure_dither,
                           previous=previous, destination=destination,
                           progress=progress, on_progress=on_progress)
    grid.background = solve_background
    return grid


def build(args: argparse.Namespace) -> None:
    style = getattr(args, "style", "headroom")
    invert_ink = bool(getattr(args, "invert_ink", False))
    glyph_luminance_bonus = getattr(args, "glyph_luminance_bonus", None)
    if glyph_luminance_bonus is None:
        glyph_luminance_bonus = DEFAULT_GLYPH_LUMINANCE_BONUS if style == "headroom" else 0.0
    glyph_luminance_bonus = float(glyph_luminance_bonus)
    if (not math.isfinite(glyph_luminance_bonus)
            or not 0 <= glyph_luminance_bonus <= DEFAULT_GLYPH_LUMINANCE_BONUS
            or (style == "structure" and glyph_luminance_bonus != 0)):
        raise ValueError("Glyph luminance bonus must be between 0 and 0.12 and is only available for headroom rendering.")
    gradient_loss_strength = getattr(args, "gradient_loss_strength", None)
    if gradient_loss_strength is None:
        gradient_loss_strength = DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0
    structure_dither = float(getattr(args, "structure_dither", STRUCTURE_DITHER))
    transition_pace = getattr(args, "transition_pace", "ease-in-out")
    transition_field = getattr(args, "transition_field", "blue-noise")
    transition_spread = float(getattr(args, "transition_spread", 0.65))
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
    effective_chars = STRUCTURE_CHARS if style == "structure" and args.chars == DEFAULT_CHARS else args.chars
    chars = list(effective_chars)
    if not chars or len(chars) > 256 or len(set(chars)) != len(chars):
        raise ValueError("Characters must be 1 to 256 distinct glyphs")
    cell = args.cell or cell_for_display(args.size, args.detail)
    masks = glyph_masks(font, chars, cell)
    images = [path.expanduser().resolve() for path in args.images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = {
        "version": __version__, "images": [str(path) for path in images],
        "image_hashes": [_file_hash(path) for path in images],
        "font": str(font), "font_hash": _file_hash(font),
        "size": list(args.size), "cell": list(cell), "chars": effective_chars,
        "style": style, "invert_ink": invert_ink, "structure_dither": structure_dither,
        "gradient_loss_strength": float(gradient_loss_strength),
        "glyph_luminance_bonus": glyph_luminance_bonus,
        "crop": args.crop, "grayscale": args.grayscale,
        "frames": args.frames, "steps": args.steps,
        "transition_steps": args.transition_steps,
        "transition_pace": transition_pace, "transition_field": transition_field,
        "transition_spread": transition_spread,
        "candidates": args.candidates, "dither": args.dither, "background": args.background, "seed": args.seed,
    }
    output = args.output.expanduser().resolve()
    manifest = output / "manifest.json"
    old = {}
    if manifest.exists() and not args.force:
        try:
            old = json.loads(manifest.read_text())
            expected_stills = [output / "stills" / f"{i:04d}" / f"{kind}.png"
                               for i in range(len(images)) for kind in ("glyph", "color", "background", "preview")]
            expected_pairs = [output / "transitions" / f"{i:04d}-{(i+1)%len(images):04d}" / f"{frame:04d}" / f"{kind}.png"
                              for i in range(len(images)) for frame in range(args.frames)
                              for kind in ("glyph", "color", "background")] if len(images) > 1 else []
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
    save_metadata(args.size, cell, len(chars), output / "metadata.png", invert_ink=invert_ink)
    old_signature = old.get("signature", {})
    still_settings_match = all(old_signature.get(key) == value for key, value in signature.items()
                               if key not in ("images", "image_hashes", "frames", "transition_steps",
                                              "transition_pace", "transition_field", "transition_spread"))
    transition_settings_match = still_settings_match and all(
        old_signature.get(key) == signature[key] for key in
        ("frames", "transition_steps", "transition_pace", "transition_field", "transition_spread"))
    old_hashes = old_signature.get("image_hashes", [])
    targets = [load_target(path, args.size, args.crop, args.grayscale) for path in images]
    stills = []
    for index, target in enumerate(targets):
        still_dir = output / "stills" / f"{index:04d}"
        if (still_settings_match and index < len(old_hashes)
                and old_hashes[index] == signature["image_hashes"][index]
                and all((still_dir / f"{kind}.png").exists() for kind in ("glyph", "color", "background", "preview"))):
            print(f"Reusing still {index + 1}/{len(images)}", flush=True)
            stills.append(_load_grid(still_dir))
            completed_units += 1
            report(0.0, f"Reused image {index + 1}/{len(images)}")
            continue
        print(f"Optimizing still {index + 1}/{len(images)}: {images[index]}", flush=True)
        grid = solve_style(target, masks, style=style, invert_ink=invert_ink,
                     seed=args.seed + index, steps=args.steps,
                     candidates=args.candidates, dither=args.dither, structure_dither=structure_dither,
                     background=args.background, device=args.device,
                     glyph_luminance_bonus=glyph_luminance_bonus,
                     gradient_loss_strength=float(gradient_loss_strength),
                     on_progress=lambda fraction, stage, index=index: report(
                         fraction, f"Image {index + 1}/{len(images)}: {stage}"))
        stills.append(grid)
        _save_grid(grid, still_dir, True, masks, args.size, invert_ink)
        completed_units += 1
        report(0.0, f"Finished image {index + 1}/{len(images)}")
    if len(images) > 1:
        paths = glyph_paths(masks)
        for index in range(len(images)):
            following = (index + 1) % len(images)
            base = output / "transitions" / f"{index:04d}-{following:04d}"
            if (transition_settings_match and index < len(old_hashes) and following < len(old_hashes)
                    and old_hashes[index] == signature["image_hashes"][index]
                    and old_hashes[following] == signature["image_hashes"][following]
                    and all((base / f"{frame:04d}" / f"{kind}.png").exists()
                            for frame in range(args.frames) for kind in ("glyph", "color", "background"))):
                print(f"Reusing transition {index + 1}/{len(images)}", flush=True)
                completed_units += args.frames - 2
                report(0.0, f"Reused transition {index + 1}/{len(images)}")
                continue
            print(f"Optimizing transition {index + 1}/{len(images)}", flush=True)
            previous = stills[index]
            shape = previous.glyph.shape
            longest = max(len(paths[int(a)][int(b)]) - 1 for a, b in np.unique(
                np.stack((previous.glyph.ravel(), stills[following].glyph.ravel()), axis=1), axis=0))
            if longest > args.frames - 1:
                raise ValueError(f"This transition needs at least {longest + 1} frames for adjacent glyph steps")
            offsets = phase_offsets(shape, transition_field, args.seed + index * 10007,
                                    targets[index])
            cell_height, cell_width = cell[1], cell[0]
            for frame in range(args.frames):
                fraction = frame / (args.frames - 1)
                progress = ease(fraction, transition_pace)
                if frame == 0:
                    grid = stills[index]
                elif frame == args.frames - 1:
                    grid = stills[following]
                else:
                    local = local_progress(progress, offsets, transition_spread)
                    pixel_progress = np.repeat(np.repeat(local, cell_height, axis=0), cell_width, axis=1)
                    pixel_progress = pixel_progress[:args.size[1], :args.size[0], None]
                    target = (1 - pixel_progress) * targets[index] + pixel_progress * targets[following]
                    grid = solve_style(target, masks, style=style, invert_ink=invert_ink,
                                 seed=args.seed + index * 10007,
                                 steps=args.transition_steps, candidates=args.candidates,
                                 dither=args.dither, structure_dither=structure_dither,
                                 background=args.background, device=args.device, previous=previous,
                                 destination=stills[following], progress=float(local.mean()),
                                 glyph_luminance_bonus=glyph_luminance_bonus,
                                 gradient_loss_strength=float(gradient_loss_strength),
                                 on_progress=lambda fraction, stage, index=index, frame=frame: report(
                                     fraction, f"Transition {index + 1}/{len(images)}, frame {frame + 1}/{args.frames}: {stage}"))
                    grid.glyph = path_glyphs(stills[index].glyph, stills[following].glyph, local, paths,
                                             previous.glyph, args.frames - 1 - frame)
                    if style == "headroom":
                        grid.color = recolor(encode_headroom_lab(srgb_to_oklab(target), invert_ink=invert_ink),
                                             masks, grid.glyph, grid.background)
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
    parser.add_argument("--detail", type=int, choices=(0, 1, 2, 3, 4), default=1)
    parser.add_argument("--cell", type=_size, help="Override cell dimensions")
    parser.add_argument("--crop", choices=("fill", "fit"), default="fill")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frames", type=int, default=16, help="Frames per slideshow transition")
    parser.add_argument("--transition-pace", choices=PACES, default="ease-in-out")
    parser.add_argument("--transition-field", choices=FIELDS, default="blue-noise")
    parser.add_argument("--transition-spread", type=float, default=0.65)
    parser.add_argument("--steps", type=int, default=36, help="Still optimization steps")
    parser.add_argument("--transition-steps", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--dither", type=float, default=0.005,
                        help="Post-optimization glyph-logit noise (default: blue-noise uniform; Gumbel scale is dither × 100)")
    parser.add_argument("--gradient-loss-strength", type=float, default=None,
                        help="Weight for displayed OKLab Sobel gradients (default: 1.5 for headroom)")
    parser.add_argument("--glyph-luminance-bonus", type=float, default=None,
                        help="Bonus for lighter glyph ink with automatic chrominance correction (default: 0.12 for headroom)")
    parser.add_argument("--style", choices=("headroom", "structure"), default="headroom")
    parser.add_argument("--invert-ink", action="store_true", help="Dark ink on a light canvas")
    parser.add_argument("--structure-dither", type=float, default=STRUCTURE_DITHER,
                        help="Blue-noise coverage dither for the structure style")
    parser.add_argument("--background", default="auto", help="auto or a fixed #RRGGBB color")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 2 or args.steps < 1 or args.transition_steps < 1 or args.candidates < 1:
        parser.error("frames must be >= 2 and optimization counts must be positive")
    if args.background != "auto" and not re.fullmatch(r"#[0-9a-fA-F]{6}", args.background):
        parser.error("background must be auto or a #RRGGBB color")
    if not 0 <= args.dither <= 0.05 or not 0 <= args.structure_dither <= 0.12:
        parser.error("dither must be 0..0.05 and structure-dither must be 0..0.12")
    if not 0 <= args.transition_spread <= 0.9:
        parser.error("transition-spread must be 0..0.9")
    if args.gradient_loss_strength is not None and args.gradient_loss_strength < 0:
        parser.error("gradient-loss-strength must be nonnegative")
    if args.glyph_luminance_bonus is not None and not 0 <= args.glyph_luminance_bonus <= DEFAULT_GLYPH_LUMINANCE_BONUS:
        parser.error("glyph-luminance-bonus must be 0..0.12")
    if args.style == "structure" and args.glyph_luminance_bonus not in (None, 0):
        parser.error("glyph-luminance-bonus is only available for headroom rendering")
    build(args)


if __name__ == "__main__":
    main()
