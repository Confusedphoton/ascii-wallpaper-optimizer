"""Batch entry point for still grids and ASCII-aware slideshow transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from . import __version__
from .assets import (
    cell_for_display,
    glyph_masks,
    load_target,
    resolve_font,
    save_atlas,
    save_metadata,
)
from .defaults import (
    DEFAULT_CHARS,
    DEFAULT_GLYPH_ENTROPY_STRENGTH,
    DEFAULT_GLYPH_LUMINANCE_BONUS,
    DEFAULT_GRADIENT_LOSS_STRENGTH,
    DEFAULT_PERCEPTUAL_LOSS,
    DEFAULT_POLISH_STEPS,
    DEFAULT_REFINEMENT_SWEEPS,
    DEFAULT_USE_SHORTLIST,
    MAX_GLYPH_ENTROPY_STRENGTH,
    MAX_GLYPH_LUMINANCE_BONUS,
    PERCEPTUAL_WEIGHTS,
    ppd_for_width,
)
from .optimizer import (
    Grid,
    _choose_background,
    _color_matrices,
    fit_fixed_glyphs,
    render_grid,
    solve,
)
from .revision import render_revision
from .runtime import select_device
from .settings import normalize_options, validate_size
from .structure import STRUCTURE_CHARS, fit_structure_colors, solve_structure
from .tone import (
    display_tone_lab,
    encode_headroom_lab,
    invert_source,
    oklab_to_srgb,
    playback_rgb8,
    quantize_playback_lab,
    srgb_to_oklab,
)
from .transition import (
    FIELDS,
    PACES,
    ease,
    glyph_paths,
    local_progress,
    path_glyphs,
    phase_offsets,
)

MAP_KINDS = ("glyph", "color", "background")
STILL_KINDS = (*MAP_KINDS, "preview")
TRANSITION_SETTINGS = {"frames", "transition_steps", "transition_pace", "transition_field", "transition_spread"}


def _atomic_json(path: Path, value: dict) -> None:
    """Publish JSON only after the complete temporary file has been written."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
               size: tuple[int, int], invert_ink: bool = False) -> Grid:
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.zeros((*grid.glyph.shape, 3), dtype=np.uint8)
    indices[:, :, 0] = grid.glyph
    Image.fromarray(indices, "RGB").save(directory / "glyph.png")
    color = playback_rgb8(grid.color)
    Image.fromarray(color, "RGB").save(directory / "color.png")
    background = np.zeros(3, dtype=np.float32) if grid.background is None else grid.background
    background_color = playback_rgb8(background).reshape(1, 1, 3)
    Image.fromarray(background_color, "RGB").save(directory / "background.png")
    persisted = Grid(grid.glyph.copy(), srgb_to_oklab(color.astype(np.float32) / 255),
                     srgb_to_oklab(background_color.astype(np.float32) / 255).reshape(3))
    if preview:
        pixels = display_tone_lab(render_grid(persisted, masks)[:size[1], :size[0]], invert_ink=invert_ink)
        color = oklab_to_srgb(pixels)
        Image.fromarray(np.rint(color * 255).astype(np.uint8), "RGB").save(directory / "preview.png")
    return persisted


def _load_grid(directory: Path) -> Grid:
    with Image.open(directory / "glyph.png") as image:
        glyph = np.asarray(image.convert("RGB"))[:, :, 0].copy()
    with Image.open(directory / "color.png") as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    color = srgb_to_oklab(encoded)
    with Image.open(directory / "background.png") as image:
        encoded_background = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    background = srgb_to_oklab(encoded_background).reshape(3)
    return Grid(glyph, color, background)


def _copy_grid(source: Path, destination: Path, *, preview: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for kind in STILL_KINDS if preview else MAP_KINDS:
        shutil.copyfile(source / f"{kind}.png", destination / f"{kind}.png")


def _cached_still(entry: Path, signature: dict) -> Path | None:
    try:
        metadata = json.loads((entry / "cache.json").read_text())
        if not isinstance(metadata, dict):
            return None
        generation = metadata.get("generation", "")
        if metadata.get("signature") != signature or not re.fullmatch(r"[a-f0-9]{32}", generation):
            return None
        directory = entry / generation
        if all((directory / f"{kind}.png").is_file() for kind in STILL_KINDS):
            grid = _load_grid(directory)
            width, height = signature["size"]
            cell_width, cell_height = signature["cell"]
            expected = ((height + cell_height - 1) // cell_height,
                        (width + cell_width - 1) // cell_width)
            if grid.glyph.shape != expected or grid.color.shape != (*expected, 3):
                return None
            if grid.glyph.max() >= len(signature["chars"]):
                return None
            with Image.open(directory / "preview.png") as image:
                if image.size != (width, height):
                    return None
                image.verify()
            return directory
    except (OSError, ValueError, TypeError):
        pass
    return None


def _cache_still(source: Path, entry: Path, signature: dict) -> None:
    """Keep each cached generation immutable, and switch its index atomically."""
    entry.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    temporary = entry / f".{generation}.tmp"
    try:
        _copy_grid(source, temporary, preview=True)
        temporary.replace(entry / generation)
        _atomic_json(entry / "cache.json", {"signature": signature, "generation": generation})
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _backgrounds(source_lab: np.ndarray, background: str,
                 invert_ink: bool) -> tuple[np.ndarray, np.ndarray]:
    original = _choose_background(torch.from_numpy(source_lab).permute(2, 0, 1), background).numpy()
    encoded = (np.zeros(3, dtype=np.float32) if invert_ink and background == "auto"
               else encode_headroom_lab(original.reshape(1, 1, 3), invert_ink=invert_ink)[0, 0])
    return original, quantize_playback_lab(encoded)


def solve_style(target: np.ndarray, masks: np.ndarray, *, style: str,
                invert_ink: bool, background: str, seed: int, steps: int,
                candidates: int = 6, dither: float, structure_dither: float,
                use_shortlist: bool = DEFAULT_USE_SHORTLIST,
                logit_noise_distribution: str = "uniform",
                device: str, previous: Grid | None = None,
                destination: Grid | None = None, progress: float | None = None,
                on_progress: Callable[[float, str], None] | None = None,
                detail_loss: str = "default",
                objective_space: str = "oklab",
                glyph_luminance_bonus: float = 0.0,
                saturation_loss_strength: float | None = None,
                glyph_error_movement_strength: float = 0.0,
                glyph_entropy_strength: float | None = None,
                gradient_loss_strength: float | None = None,
                refinement_sweeps: int = DEFAULT_REFINEMENT_SWEEPS,
                polish_steps: int = DEFAULT_POLISH_STEPS,
                perceptual_loss: bool | None = None,
                viewing_ppd: float | None = None,
                diagnostics=None) -> Grid:
    """Prepare a grid whose displayed pixels follow the chosen ink polarity."""
    if style not in ("headroom", "structure"):
        raise ValueError(f"Unknown render style: {style}")
    if objective_space != "oklab":
        raise ValueError("The renderer operates only in OKLab.")
    if gradient_loss_strength is None:
        gradient_loss_strength = DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0
    if glyph_entropy_strength is None:
        glyph_entropy_strength = DEFAULT_GLYPH_ENTROPY_STRENGTH if style == "headroom" else 0.0
    if perceptual_loss is None:
        perceptual_loss = DEFAULT_PERCEPTUAL_LOSS and style == "headroom"
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if style == "structure" and perceptual_loss:
        raise ValueError("perceptual_loss is only available for the headroom objective")
    if style == "structure" and glyph_luminance_bonus:
        raise ValueError("glyph_luminance_bonus is only available for the headroom objective")
    if style == "structure" and saturation_loss_strength:
        raise ValueError("saturation_loss_strength is only available for the headroom objective")
    if style == "structure" and gradient_loss_strength:
        raise ValueError("gradient_loss_strength is only available for the headroom objective")
    if style == "structure" and glyph_error_movement_strength:
        raise ValueError("glyph_error_movement_strength is only available for the headroom objective")
    if style == "structure" and glyph_entropy_strength:
        raise ValueError("glyph_entropy_strength is only available for the headroom objective")
    if style == "structure" and detail_loss != "default":
        raise ValueError("Feature/detail losses are not implemented for structure-first glyph selection")
    source_lab = srgb_to_oklab(target)
    original_background, solve_background = _backgrounds(source_lab, background, invert_ink)

    if style == "headroom":
        encoded_target = encode_headroom_lab(source_lab, invert_ink=invert_ink)
        return solve(encoded_target, masks, seed=seed, steps=steps,
                     candidates=candidates, dither=dither,
                     use_shortlist=use_shortlist,
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
                     glyph_error_movement_strength=glyph_error_movement_strength,
                     glyph_entropy_strength=glyph_entropy_strength,
                     error_movement_target_oklab=source_lab,
                     gradient_loss_strength=gradient_loss_strength,
                     gradient_target_oklab=source_lab,
                     refinement_sweeps=refinement_sweeps, polish_steps=polish_steps,
                     perceptual_loss=perceptual_loss, perceptual_target_srgb=target,
                     viewing_ppd=viewing_ppd,
                     **({"diagnostics": diagnostics} if diagnostics is not None else {}),
                     invert_ink=invert_ink)

    selection_target = invert_source(target) if invert_ink else target
    selection_background = solve_background if invert_ink else original_background
    grid = solve_structure(selection_target, masks, selection_background,
                           seed=seed, dither=structure_dither,
                           previous=previous, destination=destination,
                           progress=progress, on_progress=on_progress)
    grid.background = solve_background
    return grid


def _render_transition(index: int, images: list[Path], output: Path,
                       args: argparse.Namespace, masks: np.ndarray, cell: tuple[int, int],
                       paths: list[list[list[int]]], report: Callable[[float, str], None]) -> dict:
    """Render one pair in order, retaining its previous-frame color objective."""
    following = (index + 1) % len(images)
    base = output / "transitions" / f"{index:04d}-{following:04d}"
    backgrounds = {}
    print(f"Optimizing transition {index + 1}/{len(images)}", flush=True)
    start = _load_grid(output / "stills" / f"{index:04d}")
    destination = _load_grid(output / "stills" / f"{following:04d}")
    previous = start
    shape = previous.glyph.shape
    longest = max(len(paths[int(a)][int(b)]) - 1 for a, b in np.unique(
        np.stack((previous.glyph.ravel(), destination.glyph.ravel()), axis=1), axis=0))
    if longest > args.frames - 1:
        raise ValueError(f"This transition needs at least {longest + 1} frames for adjacent glyph steps")
    source_target = load_target(images[index], args.size, args.crop, args.grayscale)
    destination_target = load_target(images[following], args.size, args.crop, args.grayscale)
    offsets = phase_offsets(shape, args.transition_field, args.seed + index * 10007,
                            source_target)
    cell_height, cell_width = cell[1], cell[0]
    for frame in range(args.frames):
        fraction = frame / (args.frames - 1)
        progress = ease(fraction, args.transition_pace)
        if frame == 0:
            grid = start
        elif frame == args.frames - 1:
            grid = destination
        else:
            local = local_progress(progress, offsets, args.transition_spread)
            pixel_progress = np.repeat(np.repeat(local, cell_height, axis=0), cell_width, axis=1)
            pixel_progress = pixel_progress[:args.size[1], :args.size[0], None]
            target = (1 - pixel_progress) * source_target + pixel_progress * destination_target
            glyphs = path_glyphs(start.glyph, destination.glyph, local, paths,
                                previous.glyph, args.frames - 1 - frame)
            source_lab = srgb_to_oklab(target)
            original_background, solve_background = _backgrounds(source_lab, args.background, args.invert_ink)
            if args.style == "headroom":
                grid = fit_fixed_glyphs(
                    encode_headroom_lab(source_lab, invert_ink=args.invert_ink), masks, glyphs,
                    background_oklab=solve_background, input_oklab=True,
                    steps=args.transition_steps, device=args.device,
                    previous=previous, destination=destination, progress=float(local.mean()),
                    glyph_luminance_bonus=args.glyph_luminance_bonus,
                    gradient_loss_strength=float(args.gradient_loss_strength),
                    saturation_target_oklab=source_lab, gradient_target_oklab=source_lab,
                    perceptual_target_srgb=target,
                    perceptual_loss=args.perceptual_loss,
                    invert_ink=args.invert_ink,
                    on_progress=lambda fraction, stage, index=index, frame=frame: report(
                        frame - 1 + fraction, f"Transition {index + 1}/{len(images)}, frame {frame + 1}/{args.frames}: {stage}"))
            else:
                selection = invert_source(target) if args.invert_ink else target
                grid = fit_structure_colors(selection, masks, glyphs,
                                            solve_background if args.invert_ink else original_background)
                grid.background = solve_background
            del target, source_lab
        frame_dir = base / f"{frame:04d}"
        if frame in (0, args.frames - 1):
            _copy_grid(output / "stills" / f"{index if frame == 0 else following:04d}", frame_dir)
        else:
            grid = _save_grid(grid, frame_dir, False, masks, args.size, args.invert_ink)
        backgrounds[f"transitions/{index:04d}-{following:04d}/{frame:04d}"] = grid.background.tolist()
        previous = grid
        if frame not in (0, args.frames - 1):
            report(float(frame), f"Transition {index + 1}/{len(images)}: finished frame {frame + 1}/{args.frames}")
    return backgrounds


def build(args: argparse.Namespace) -> None:
    vars(args).update(normalize_options(vars(args)))
    transition_workers = getattr(args, "transition_workers", 2)
    if isinstance(transition_workers, bool) or not isinstance(transition_workers, int) or transition_workers < 1:
        raise ValueError("transition_workers must be a positive integer")
    args.device = select_device(args.device)
    args.size = validate_size(args.size)
    style, invert_ink = args.style, args.invert_ink
    glyph_luminance_bonus = args.glyph_luminance_bonus
    gradient_loss_strength = args.gradient_loss_strength
    structure_dither = args.structure_dither
    if not args.images:
        raise ValueError("At least one source image is required")
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
    cell = validate_size(args.cell or cell_for_display(args.size, args.detail), max_dimension=255)
    masks = glyph_masks(font, chars, cell)
    images = [path.expanduser().resolve() for path in args.images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = {
        "version": __version__, "render_revision": render_revision(), "images": [str(path) for path in images],
        "image_hashes": [_file_hash(path) for path in images],
        "font": str(font), "font_hash": _file_hash(font),
        "size": list(args.size), "cell": list(cell), "detail": args.detail, "chars": effective_chars,
        "style": style, "invert_ink": invert_ink, "structure_dither": structure_dither,
        "gradient_loss_strength": float(gradient_loss_strength),
        "glyph_luminance_bonus": glyph_luminance_bonus,
        "glyph_entropy_strength": args.glyph_entropy_strength,
        "perceptual_loss": args.perceptual_loss,
        "perceptual_weights": PERCEPTUAL_WEIGHTS if args.perceptual_loss else {},
        "viewing_ppd": ppd_for_width(args.size[0]) if args.perceptual_loss else None,
        "crop": args.crop, "grayscale": args.grayscale,
        "frames": args.frames, "steps": args.steps,
        "transition_steps": args.transition_steps,
        "transition_pace": args.transition_pace, "transition_field": args.transition_field,
        "transition_spread": args.transition_spread,
        "glyph_search": "all" if style == "headroom" else "structure",
        "dither": args.dither, "background": args.background, "seed": args.seed,
    }
    output = args.output.expanduser().resolve()
    still_cache = getattr(args, "still_cache", None)
    if still_cache is not None:
        still_cache = Path(still_cache).expanduser().resolve()
    manifest = output / "manifest.json"
    old = {}
    if manifest.exists() and not args.force:
        try:
            old = json.loads(manifest.read_text())
            if not isinstance(old, dict):
                old = {}
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
    if not isinstance(old_signature, dict):
        old_signature = {}
    still_settings_match = all(old_signature.get(key) == value for key, value in signature.items()
                               if key not in {"images", "image_hashes", *TRANSITION_SETTINGS})
    transition_settings_match = still_settings_match and all(
        old_signature.get(key) == signature[key] for key in
        ("frames", "transition_steps", "transition_pace", "transition_field", "transition_spread"))
    old_hashes = old_signature.get("image_hashes", [])
    if not isinstance(old_hashes, list):
        old_hashes = []
    # Keep grids on disk. Each transition worker loads only its active pair,
    # so source-image memory is bounded by worker count, not slideshow length.
    background_labs: dict[str, list[float]] = {}
    grid_shape = None
    for index, image in enumerate(images):
        still_dir = output / "stills" / f"{index:04d}"
        still_signature = {key: value for key, value in signature.items()
                           if key not in {"images", "image_hashes", "font", *TRANSITION_SETTINGS}}
        still_signature.update(image_hash=signature["image_hashes"][index], seed=args.seed + index)
        cache_key = hashlib.sha256(json.dumps(still_signature, sort_keys=True).encode()).hexdigest()
        cache_entry = still_cache / cache_key if still_cache is not None else None
        cached = _cached_still(cache_entry, still_signature) if cache_entry is not None and not args.force else None
        if (still_settings_match and index < len(old_hashes)
                and old_hashes[index] == signature["image_hashes"][index]
                and all((still_dir / f"{kind}.png").exists() for kind in STILL_KINDS)):
            print(f"Reusing still {index + 1}/{len(images)}", flush=True)
            grid = _load_grid(still_dir)
            if cache_entry is not None and cached is None:
                _cache_still(still_dir, cache_entry, still_signature)
        elif cached is not None:
            print(f"Reusing cached still {index + 1}/{len(images)}", flush=True)
            _copy_grid(cached, still_dir, preview=True)
            grid = _load_grid(still_dir)
        else:
            print(f"Optimizing still {index + 1}/{len(images)}: {image}", flush=True)
            target = load_target(image, args.size, args.crop, args.grayscale)
            grid = solve_style(target, masks, style=style, invert_ink=invert_ink,
                         seed=args.seed + index, steps=args.steps,
                         dither=args.dither, structure_dither=structure_dither,
                         background=args.background, device=args.device,
                         glyph_luminance_bonus=glyph_luminance_bonus,
                         glyph_entropy_strength=args.glyph_entropy_strength,
                         gradient_loss_strength=float(gradient_loss_strength),
                         perceptual_loss=args.perceptual_loss,
                         on_progress=lambda fraction, stage, index=index: report(
                             fraction, f"Image {index + 1}/{len(images)}: {stage}"))
            grid = _save_grid(grid, still_dir, True, masks, args.size, invert_ink)
            del target
            if cache_entry is not None:
                _cache_still(still_dir, cache_entry, still_signature)
        grid_shape = grid.glyph.shape
        background_labs[f"stills/{index:04d}"] = grid.background.tolist()
        del grid
        completed_units += 1
        report(0.0, f"Finished image {index + 1}/{len(images)}")
    if len(images) > 1:
        paths = glyph_paths(masks)
        pending_pairs = []
        for index in range(len(images)):
            following = (index + 1) % len(images)
            base = output / "transitions" / f"{index:04d}-{following:04d}"
            if (transition_settings_match and index < len(old_hashes) and following < len(old_hashes)
                    and old_hashes[index] == signature["image_hashes"][index]
                    and old_hashes[following] == signature["image_hashes"][following]
                    and all((base / f"{frame:04d}" / f"{kind}.png").exists()
                            for frame in range(args.frames) for kind in ("glyph", "color", "background"))):
                print(f"Reusing transition {index + 1}/{len(images)}", flush=True)
                for frame in range(args.frames):
                    key = f"transitions/{index:04d}-{following:04d}/{frame:04d}"
                    background_labs[key] = _load_grid(base / f"{frame:04d}").background.tolist()
                completed_units += args.frames - 2
                report(0.0, f"Reused transition {index + 1}/{len(images)}")
                continue
            pending_pairs.append(index)
        if pending_pairs:
            workers = min(transition_workers, len(pending_pairs))
            print(f"Rendering transitions with {workers} worker(s) on {args.device}", flush=True)
            lock = threading.Lock()
            cancelled = threading.Event()
            pair_progress = {index: 0.0 for index in pending_pairs}

            def pair_report(index: int, units: float, stage: str) -> None:
                if cancelled.is_set():
                    raise CancelledError("Transition render cancelled")
                with lock:
                    pair_progress[index] = max(pair_progress[index], units)
                    report(sum(pair_progress.values()), stage)

            gpu = style == "headroom" and torch.device(args.device).type == "cuda"
            if gpu:
                # These cached tensors are shared by the streams. Populate them
                # before workers start and wait for their uploads to complete.
                device = torch.device(args.device)
                _color_matrices(torch.device("cuda", torch.cuda.current_device()
                                             if device.index is None else device.index), torch.float32)
                torch.cuda.synchronize(device)
            old_threads = torch.get_num_threads()
            if workers > 1:
                torch.set_num_threads(min(old_threads, max(1, (os.cpu_count() or 1) // workers)))

            def render_pair(index: int) -> dict:
                stream = torch.cuda.Stream(device=args.device) if gpu else None
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    try:
                        pair_report(index, 0.0, f"Starting transition {index + 1}/{len(images)}")
                        return _render_transition(index, images, output, args, masks, cell, paths,
                                                  lambda units, stage: pair_report(index, units, stage))
                    finally:
                        if stream is not None:
                            stream.synchronize()

            try:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ascii-transition") as executor:
                    futures = [executor.submit(render_pair, index) for index in pending_pairs]
                    try:
                        for future in as_completed(futures):
                            background_labs.update(future.result())
                    except BaseException:
                        cancelled.set()
                        for future in futures:
                            future.cancel()
                        raise
            finally:
                if workers > 1:
                    torch.set_num_threads(old_threads)
            completed_units += len(pending_pairs) * (args.frames - 2)
    _atomic_json(manifest, {"signature": signature, "generation": getattr(args, "generation", None) or uuid.uuid4().hex,
                            "image_count": len(images), "frame_count": args.frames,
                            "grid": [grid_shape[1], grid_shape[0]], "output_size": list(args.size),
                            "cell_size": list(cell), "atlas_count": len(chars), "invert_ink": invert_ink,
                            "background_labs": background_labs})
    print(f"Rendered assets: {output}")
    report(1.0, "Ready")


def make_parser() -> argparse.ArgumentParser:
    defaults = normalize_options({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Source images in slideshow order")
    parser.add_argument("--output", type=Path, required=True, help="Generated asset directory")
    parser.add_argument("--size", type=_size, required=True, help="Wallpaper size, WIDTHxHEIGHT")
    parser.add_argument("--font", default=defaults["font"], help="Font family or font file")
    parser.add_argument("--chars", default=DEFAULT_CHARS, help="Distinct characters, including a space")
    parser.add_argument("--detail", type=int, choices=(0, 1, 2, 3, 4), default=defaults["detail"])
    parser.add_argument("--cell", type=_size, help="Override cell dimensions")
    parser.add_argument("--crop", choices=("fill", "fit"), default=defaults["crop"])
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frames", type=int, default=defaults["frames"], help="Frames per slideshow transition")
    parser.add_argument("--transition-pace", choices=PACES, default=defaults["transition_pace"])
    parser.add_argument("--transition-field", choices=FIELDS, default=defaults["transition_field"])
    parser.add_argument("--transition-spread", type=float, default=defaults["transition_spread"])
    parser.add_argument("--steps", type=int, default=defaults["steps"], help="Still optimization steps")
    parser.add_argument("--transition-steps", type=int, default=defaults["transition_steps"])
    # Accept old commands without restricting the full glyph-bank solve.
    parser.add_argument("--candidates", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dither", type=float, default=defaults["dither"],
                        help="Post-optimization glyph-logit noise (default: blue-noise uniform; Gumbel scale is dither × 100)")
    parser.add_argument("--gradient-loss-strength", type=float, default=None,
                        help="Weight for displayed OKLab Sobel gradients (default: 1.5 for headroom)")
    parser.add_argument("--glyph-entropy-strength", type=float, default=None,
                        help=("Weight for frame-wide glyph variety; 0 disables "
                              f"(default: {DEFAULT_GLYPH_ENTROPY_STRENGTH:g} for headroom, "
                              f"0 for structure; max: {MAX_GLYPH_ENTROPY_STRENGTH:g})"))
    parser.add_argument("--glyph-luminance-bonus", type=float, default=None,
                        help=("Bonus for lighter glyph ink with automatic chrominance correction "
                              f"(default: {DEFAULT_GLYPH_LUMINANCE_BONUS:g}, "
                              f"max: {MAX_GLYPH_LUMINANCE_BONUS:g})"))
    parser.add_argument("--style", choices=("headroom", "structure"), default=defaults["style"])
    parser.add_argument("--no-perceptual-loss", dest="perceptual_loss", action="store_false",
                        help="Disable the default perceptual objective for headroom rendering")
    parser.set_defaults(perceptual_loss=None)
    parser.add_argument("--invert-ink", action="store_true", help="Dark ink on a light canvas")
    parser.add_argument("--structure-dither", type=float, default=defaults["structure_dither"],
                        help="Blue-noise coverage dither for the structure style")
    parser.add_argument("--background", default=defaults["background"], help="auto or a fixed #RRGGBB color")
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda (also used for ROCm)")
    parser.add_argument("--transition-workers", type=int,
                        default=os.environ.get("ASCII_WALLPAPER_TRANSITION_WORKERS", "2"),
                        help="Concurrent transition pairs (default: 2; use 1 for lower memory use)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--still-cache", type=Path, help="Reusable still-asset cache shared across render variants")
    parser.add_argument("--generation", help=argparse.SUPPRESS)
    parser.add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        build(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
