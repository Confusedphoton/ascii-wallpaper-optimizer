"""Render image slideshows or videos, and import/export compressed ASCII archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
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
    DEFAULT_FLOW_STRENGTH,
    DEFAULT_FLOW_EDDIES,
    DEFAULT_FLOW_SCALE,
    DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH,
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
    fit_fixed_glyphs,
    render_grid,
    solve,
)
from .revision import render_revision
from .runtime import select_device
from .settings import VIDEO_PLAYBACK_OPTIONS, normalize_options, validate_size
from .structure import STRUCTURE_CHARS, solve_structure
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
    TYPES,
    ease,
    earth_mover_destinations,
    glyph_paths,
    local_progress,
    path_glyphs,
    phase_offsets,
    transport_cells,
)
from .video import decode_frames, frame_durations, probe_video

MAP_KINDS = ("glyph", "color", "background")
STILL_KINDS = (*MAP_KINDS, "preview")
TRANSITION_SETTINGS = {"frames", "transition_steps", "transition_type", "transition_pace", "transition_field", "transition_spread"}


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


def _cached_still(entry: Path, signature: dict, *, preview: bool = True) -> Path | None:
    try:
        metadata = json.loads((entry / "cache.json").read_text())
        if not isinstance(metadata, dict):
            return None
        generation = metadata.get("generation", "")
        if metadata.get("signature") != signature or not re.fullmatch(r"[a-f0-9]{32}", generation):
            return None
        directory = entry / generation
        if all((directory / f"{kind}.png").is_file() for kind in STILL_KINDS if preview or kind != "preview"):
            grid = _load_grid(directory)
            width, height = signature["size"]
            cell_width, cell_height = signature["cell"]
            expected = ((height + cell_height - 1) // cell_height,
                        (width + cell_width - 1) // cell_width)
            if grid.glyph.shape != expected or grid.color.shape != (*expected, 3):
                return None
            if grid.glyph.max() >= len(signature["chars"]):
                return None
            if preview:
                with Image.open(directory / "preview.png") as image:
                    if image.size != (width, height):
                        return None
                    image.verify()
            return directory
    except (OSError, ValueError, TypeError):
        pass
    return None


def _cache_still(source: Path, entry: Path, signature: dict, *, preview: bool = True) -> None:
    """Keep each cached generation immutable, and switch its index atomically."""
    entry.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    temporary = entry / f".{generation}.tmp"
    try:
        _copy_grid(source, temporary, preview=preview)
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
                postprocess: bool | None = None,
                perceptual_loss: bool | None = None,
                viewing_ppd: float | None = None,
                flow_strength: float = DEFAULT_FLOW_STRENGTH,
                flow_eddies: float = DEFAULT_FLOW_EDDIES,
                flow_scale: float = DEFAULT_FLOW_SCALE,
                flow_diagnostics: dict | None = None,
                flow_costs=None,
                solve_context: dict | None = None,
                diagnostics=None) -> Grid:
    """Prepare a grid whose displayed pixels follow the chosen ink polarity."""
    if style not in ("headroom", "structure", "flow"):
        raise ValueError(f"Unknown render style: {style}")
    if objective_space != "oklab":
        raise ValueError("The renderer operates only in OKLab.")
    if gradient_loss_strength is None:
        gradient_loss_strength = DEFAULT_GRADIENT_LOSS_STRENGTH if style != "structure" else 0.0
    if glyph_entropy_strength is None:
        glyph_entropy_strength = (DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH if style == "flow" else
                                  DEFAULT_GLYPH_ENTROPY_STRENGTH if style == "headroom" else 0.0)
    if perceptual_loss is None:
        perceptual_loss = DEFAULT_PERCEPTUAL_LOSS and style != "structure"
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if style == "flow" and not perceptual_loss:
        raise ValueError("Flow rendering requires perceptual color fitting")
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

    if style in ("headroom", "flow"):
        solve_progress = on_progress
        if style == "flow":
            from .flow import simulate_flow, glyph_flow_costs
            from .settings import finite_float
            flow_strength = finite_float(flow_strength, "flow_strength", 0, 2)
            flow_eddies = finite_float(flow_eddies, "flow_eddies", 0, 2)
            flow_scale = finite_float(flow_scale, "flow_scale", 1, 24)
            if flow_strength and flow_costs is None:
                field = simulate_flow(target, masks.shape[1:], seed=seed, eddies=flow_eddies,
                                      scale=flow_scale, device=device,
                                      on_progress=(lambda f, stage: on_progress(0.15 * f, stage))
                                      if on_progress is not None else None)
                flow_costs, summary = glyph_flow_costs(field, torch.as_tensor(masks, device=device))
                if flow_diagnostics is not None:
                    flow_diagnostics.update(field=field, costs=flow_costs, **field.diagnostics, **summary)
                if diagnostics is not None:
                    diagnostics.metadata["flow"] = {**field.diagnostics, **summary}
                if on_progress is not None:
                    solve_progress = lambda f, stage: on_progress(0.15 + 0.85 * f, stage)
        encoded_target = encode_headroom_lab(source_lab, invert_ink=invert_ink)
        return solve(encoded_target, masks, seed=seed, steps=steps,
                     candidates=candidates, dither=dither,
                     use_shortlist=use_shortlist,
                     logit_noise_distribution=logit_noise_distribution,
                     background_oklab=solve_background, device=device,
                     input_oklab=True,
                     previous=previous, destination=destination,
                     progress=progress, on_progress=solve_progress,
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
                     postprocess=postprocess,
                     perceptual_loss=perceptual_loss, perceptual_target_srgb=target,
                     viewing_ppd=viewing_ppd,
                     flow_costs=flow_costs, flow_strength=flow_strength if flow_costs is not None else 0.0,
                     solve_context=solve_context,
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


def _render_flow_image(target, masks, args, loop_dir, seed, on_progress):
    from .flow_playback import render_flow_clip
    count = args.flow_loop_seconds * args.flow_fps
    common = dict(style="flow", invert_ink=args.invert_ink, background=args.background,
                  seed=seed, dither=args.dither, structure_dither=args.structure_dither,
                  device=args.device, glyph_luminance_bonus=args.glyph_luminance_bonus,
                  glyph_entropy_strength=args.glyph_entropy_strength,
                  gradient_loss_strength=args.gradient_loss_strength,
                  flow_strength=args.flow_strength, flow_eddies=args.flow_eddies,
                  flow_scale=args.flow_scale, perceptual_loss=True)
    if not args.flow_strength:
        grid = solve_style(target, masks, steps=args.steps, **common)
        grids, diagnostics = [grid] * count, {"method": "static-zero-influence"}
    else:
        def solve_frame(costs, steps, context):
            return solve_style(target, masks, steps=steps, flow_costs=costs,
                               solve_context=context, **common)
        source_lab = srgb_to_oklab(target)
        _, fixed_background = _backgrounds(source_lab, args.background, args.invert_ink)
        encoded = encode_headroom_lab(source_lab, invert_ink=args.invert_ink)
        def fit_colors(glyphs, neighbors, context):
            return fit_fixed_glyphs(encoded, masks, glyphs, background_oklab=fixed_background,
                input_oklab=True, steps=args.flow_frame_steps, device=args.device,
                glyph_luminance_bonus=args.glyph_luminance_bonus,
                gradient_loss_strength=args.gradient_loss_strength, gradient_target_oklab=source_lab,
                saturation_target_oklab=source_lab, perceptual_target_srgb=target,
                perceptual_loss=True, invert_ink=args.invert_ink,
                solve_context=context, temporal_neighbors=neighbors,
                temporal_strength=args.flow_temporal_strength)
        grids, diagnostics = render_flow_clip(target, masks, seed=seed, eddies=args.flow_eddies,
            scale=args.flow_scale, device=args.device, frames=count, steps=args.steps,
            frame_steps=args.flow_frame_steps, temporal_strength=args.flow_temporal_strength,
            solve_frame=solve_frame, fit_colors=fit_colors, flow_strength=args.flow_strength,
            glyph_entropy_strength=args.glyph_entropy_strength, on_progress=on_progress)
    for frame, grid in enumerate(grids):
        _save_grid(grid, loop_dir / f"{frame:04d}", False, masks, args.size, args.invert_ink)
    info = {"frame_count": count, "duration_ms": args.flow_loop_seconds * 1000,
            "loop": True, "anchor_frame": 0, "closure": diagnostics}
    _atomic_json(loop_dir / "clip.json", info)
    return info


def _cached_loop(entry, signature, count):
    try:
        metadata = json.loads((entry / "cache.json").read_text())
        if (metadata.get("signature") != signature
                or not re.fullmatch(r"[a-f0-9]{32}", metadata.get("generation", ""))):
            return None
        directory = entry / metadata["generation"]
        info = json.loads((directory / "clip.json").read_text())
        if (info["frame_count"] != count or info["duration_ms"] != signature["flow_loop_seconds"] * 1000
                or info.get("loop") is not True):
            return None
        for frame in range(count):
            grid = _load_grid(directory / f"{frame:04d}")
            width, height = signature["size"]
            cw, ch = signature["cell"]
            if (grid.glyph.shape != (math.ceil(height/ch), math.ceil(width/cw))
                    or grid.color.shape != (*grid.glyph.shape, 3)
                    or grid.glyph.max() >= len(signature["chars"])):
                return None
        return directory
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _cache_loop(source, entry, signature):
    entry.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    temporary = entry / f".{generation}.tmp"
    try:
        shutil.copytree(source, temporary)
        temporary.replace(entry / generation)
        _atomic_json(entry / "cache.json", {"signature": signature, "generation": generation})
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _render_transition(index: int, images: list[Path], output: Path,
                       args: argparse.Namespace, masks: np.ndarray, cell: tuple[int, int],
                       paths: list[list[list[int]]], report: Callable[[float, str], None]) -> dict:
    """Prepare glyph morphs or spatial transport from the saved endpoints."""
    following = (index + 1) % len(images)
    base = output / "transitions" / f"{index:04d}-{following:04d}"
    backgrounds = {}
    print(f"Rendering transition {index + 1}/{len(images)}", flush=True)
    start = _load_grid(output / "stills" / f"{index:04d}")
    destination = _load_grid(output / "stills" / f"{following:04d}")
    shape = start.glyph.shape
    destinations = None
    target_glyphs, target_colors = destination.glyph, destination.color
    if getattr(args, "transition_type", "glyph-path") == "earth-mover":
        report(0.0, f"Transition {index + 1}/{len(images)}: matching earth mover regions")
        destinations = earth_mover_destinations(
            start.glyph, destination.glyph, start.color, destination.color,
            start.background, destination.background, masks, invert_ink=args.invert_ink)
        target_glyphs = destination.glyph.ravel()[destinations]
        target_colors = destination.color.reshape(-1, 3)[destinations]
    previous_glyphs = start.glyph
    longest = max(len(paths[int(a)][int(b)]) - 1 for a, b in np.unique(
        np.stack((start.glyph.ravel(), target_glyphs.ravel()), axis=1), axis=0))
    if longest > args.frames - 1:
        raise ValueError(f"This transition needs at least {longest + 1} frames for adjacent glyph steps")
    source_target = (load_target(images[index], args.size, args.crop, args.grayscale)
                     if args.transition_field == "salience" else None)
    offsets = phase_offsets(shape, args.transition_field, args.seed + index * 10007,
                            source_target)
    del source_target
    for frame in range(args.frames):
        fraction = frame / (args.frames - 1)
        progress = ease(fraction, args.transition_pace)
        if frame == 0:
            grid = start
        elif frame == args.frames - 1:
            grid = destination
        else:
            local = local_progress(progress, offsets, args.transition_spread)
            glyphs = path_glyphs(start.glyph, target_glyphs, local, paths,
                                previous_glyphs, args.frames - 1 - frame)
            previous_glyphs = glyphs
            # These are already decoded playback OKLab colors. Use the same
            # per-cell clock as the glyphs, anchored to the stills rather than
            # accumulating a fit from the preceding transition frame.
            color = (1 - local[..., None]) * start.color + local[..., None] * target_colors
            # Backgrounds are uniform, so follow the shared eased clock.
            background = (1 - progress) * start.background + progress * destination.background
            if destinations is not None:
                glyphs, color = transport_cells(glyphs, color, background, destinations, local, masks)
            grid = Grid(glyphs, color, background)
        frame_dir = base / f"{frame:04d}"
        if frame in (0, args.frames - 1):
            _copy_grid(output / "stills" / f"{index if frame == 0 else following:04d}", frame_dir)
        else:
            grid = _save_grid(grid, frame_dir, False, masks, args.size, args.invert_ink)
        backgrounds[f"transitions/{index:04d}-{following:04d}/{frame:04d}"] = grid.background.tolist()
        if frame not in (0, args.frames - 1):
            report(float(frame), f"Transition {index + 1}/{len(images)}: finished frame {frame + 1}/{args.frames}")
    return backgrounds


def _video_intermediates(start: Grid, destination: Grid, args: argparse.Namespace,
                         masks: np.ndarray, paths: list[list[list[int]]],
                         source: np.ndarray, index: int):
    """Sample the transition at any requested budget, including a single frame."""
    destinations = None
    target_glyphs, target_colors = destination.glyph, destination.color
    if args.transition_type == "earth-mover":
        destinations = earth_mover_destinations(
            start.glyph, destination.glyph, start.color, destination.color,
            start.background, destination.background, masks, invert_ink=args.invert_ink)
        target_glyphs = destination.glyph.ravel()[destinations]
        target_colors = destination.color.reshape(-1, 3)[destinations]
    offsets = phase_offsets(start.glyph.shape, args.transition_field,
                            args.seed + index * 10007,
                            source if args.transition_field == "salience" else None)
    for frame in range(1, args.video_transition_frames + 1):
        progress = ease(frame / (args.video_transition_frames + 1), args.transition_pace)
        local = local_progress(progress, offsets, args.transition_spread)
        # Videos specify an exact interpolation count. Sampling a glyph path
        # directly lets a short interval skip glyphs rather than rejecting it.
        glyphs = path_glyphs(start.glyph, target_glyphs, local, paths)
        color = (1 - local[..., None]) * start.color + local[..., None] * target_colors
        background = (1 - progress) * start.background + progress * destination.background
        if destinations is not None:
            glyphs, color = transport_cells(glyphs, color, background, destinations, local, masks)
        yield Grid(glyphs, color, background)


def _video_endpoint_settings(signature: dict) -> dict:
    return {**{key: value for key, value in signature.items()
               if key not in {"images", "image_hashes", "font", *VIDEO_PLAYBACK_OPTIONS}},
            "image_hash": signature["image_hashes"][0]}


def _video_endpoint_sources(directories, signature):
    """Map source frame numbers to the original maps in compatible generations."""
    settings = _video_endpoint_settings(signature)
    endpoints = {}
    for directory in directories:
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["schema_version"] != 3:
                continue
            previous = manifest["signature"]
            if _video_endpoint_settings(previous) != settings:
                continue
            stride = int(previous["frame_stride"])
            spacing = int(previous["video_transition_frames"]) + 1
            count = int(manifest["video"]["frame_count"])
            if stride < 1 or spacing < 1 or count < 1 or (count - 1) % spacing:
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            continue
        for index, frame in enumerate(range(0, count, spacing)):
            endpoints.setdefault(index * stride, []).append(directory / "video" / f"{frame:04d}")
    return endpoints


def _read_video_endpoint(paths, size, cell, glyph_count):
    expected = tuple(math.ceil(dimension / step) for dimension, step in zip(reversed(size), reversed(cell)))
    for path in paths:
        try:
            grid = _load_grid(path)
            if grid.glyph.shape == expected and grid.color.shape == (*expected, 3) and grid.glyph.max() < glyph_count:
                return grid
        except (OSError, ValueError):
            continue
    return None


def _build_video(args: argparse.Namespace, source: Path, output: Path,
                 signature: dict, masks: np.ndarray, cell: tuple[int, int]) -> None:
    """Optimize only sampled frames, then emit a timestamp-preserving clip."""
    info = probe_video(source).prefix(args.video_preview_ms)
    times = info.sampled_times(args.frame_stride)
    durations = frame_durations(info, args.frame_stride, args.video_transition_frames)
    count = len(durations)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.force:
        try:
            old = json.loads(manifest_path.read_text())
            if (isinstance(old, dict) and old.get("signature") == signature and old.get("schema_version") == 3
                    and old.get("video", {}).get("frame_count") == count
                    and old["video"].get("duration_ms") == info.duration_ms
                    and all((output / name).is_file() for name in
                            ("atlas.png", "metadata.png", "stills/0000/preview.png"))
                    and all((output / "video" / f"{frame:04d}" / f"{kind}.png").is_file()
                            for frame in range(count) for kind in MAP_KINDS)
                    and all((output / "stills/0000" / f"{kind}.png").is_file() for kind in MAP_KINDS)):
                print(f"Already rendered: {output}")
                if getattr(args, "progress_json", False):
                    print('ASCII_PROGRESS {"progress": 1.0, "stage": "Ready"}', flush=True)
                return
        except (OSError, ValueError, TypeError, KeyError):
            pass
    # Stage replacements so a compatible previous directory remains readable
    # throughout an in-place CLI rebuild and until all new frames are ready.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ascii-video-", dir=output.parent) as temporary:
        stage = Path(temporary) / "render"
        stage.mkdir()
        reusable = [output, *getattr(args, "reuse_video_from", ())]
        _write_video(args, source, stage, signature, masks, cell, info, times, durations, reusable)
        (output / "stills").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="previous-", dir=temporary) as backup:
            previous = Path(backup)
            replaced, installed = [], []
            try:
                (previous / "stills").mkdir()
                names = ("atlas.png", "metadata.png", "video", "stills/0000")
                for name in ("manifest.json", *names):
                    if (output / name).exists():
                        (output / name).rename(previous / name)
                        replaced.append(name)
                for name in (*names, "manifest.json"):
                    (stage / name).rename(output / name)
                    installed.append(name)
            except OSError:
                for name in reversed(installed):
                    (output / name).rename(stage / name)
                for name in reversed(replaced):
                    (previous / name).rename(output / name)
                raise
    print(f"Rendered video assets: {output}")
    if getattr(args, "progress_json", False):
        print('ASCII_PROGRESS {"progress": 1.0, "stage": "Ready"}', flush=True)


def _write_video(args, source, output, signature, masks, cell, info, times, durations, reuse_from):
    count = len(durations)
    save_atlas(masks, output / "atlas.png")
    save_metadata(args.size, cell, len(args.chars), output / "metadata.png", invert_ink=args.invert_ink)
    backgrounds = {}
    emitted = 0
    completed = 0
    previous = None
    previous_target = None
    paths = glyph_paths(masks) if args.video_transition_frames else None
    still_cache = getattr(args, "still_cache", None)
    if still_cache is not None:
        still_cache = Path(still_cache).expanduser().resolve()
    endpoint_settings = _video_endpoint_settings(signature)
    reusable = (_video_endpoint_sources(reuse_from, signature)
                if not args.force else {})

    def report(fraction: float, stage: str) -> None:
        if getattr(args, "progress_json", False):
            print("ASCII_PROGRESS " + json.dumps({
                "progress": min(1.0, (completed + fraction) / count), "stage": stage,
            }), flush=True)

    def save(grid: Grid) -> Grid:
        nonlocal emitted
        directory = output / "video" / f"{emitted:04d}"
        persisted = _save_grid(grid, directory, False, masks, args.size, args.invert_ink)
        backgrounds[f"video/{emitted:04d}"] = persisted.background.tolist()
        emitted += 1
        return persisted

    limit = {"limit": len(times)} if args.video_preview_ms else {}
    decoded = decode_frames(source, args.size, args.crop, args.grayscale, args.frame_stride, **limit)
    sampled_count = 0
    try:
        for index, target in enumerate(decoded):
            if index >= len(times):
                raise ValueError("Decoded video frame count does not match its timing metadata")
            original_index = index * args.frame_stride
            endpoint_signature = dict(endpoint_settings, source_frame=original_index,
                                      seed=args.seed + original_index)
            cache_key = hashlib.sha256(json.dumps(endpoint_signature, sort_keys=True).encode()).hexdigest()
            cache_entry = still_cache / cache_key if still_cache is not None else None
            cached = (_cached_still(cache_entry, endpoint_signature, preview=False)
                      if cache_entry is not None and not args.force else None)
            endpoint = (_load_grid(cached) if cached is not None else
                        _read_video_endpoint(reusable.get(original_index, ()), args.size, cell, len(masks)))
            reused = endpoint is not None
            if reused:
                print(f"Reusing cached video frame {index + 1}/{len(times)}", flush=True)
            else:
                print(f"Optimizing video frame {index + 1}/{len(times)}", flush=True)
                grid = solve_style(target, masks, style=args.style, invert_ink=args.invert_ink,
                                   seed=args.seed + original_index, steps=args.steps, dither=args.dither,
                                   structure_dither=args.structure_dither, background=args.background,
                                   device=args.device, glyph_luminance_bonus=args.glyph_luminance_bonus,
                                   glyph_entropy_strength=args.glyph_entropy_strength,
                                   gradient_loss_strength=float(args.gradient_loss_strength),
                                   perceptual_loss=args.perceptual_loss, flow_strength=args.flow_strength,
                                   flow_eddies=args.flow_eddies, flow_scale=args.flow_scale,
                                   on_progress=lambda fraction, stage, index=index: report(
                                       fraction, f"Video frame {index + 1}/{len(times)}: {stage}"))
                # Quantize endpoints before interpolating, exactly like slideshows.
                endpoint = Grid(grid.glyph, quantize_playback_lab(grid.color),
                                quantize_playback_lab(grid.background))
            if previous is not None and args.video_transition_frames:
                report(0.0, f"Video transition {index}/{len(times) - 1}: {args.transition_type}")
                for intermediate in _video_intermediates(
                        previous, endpoint, args, masks, paths, previous_target, index - 1):
                    save(intermediate)
                    completed += 1
                    report(0.0, f"Finished interpolated video frame {emitted}/{count}")
            previous = save(endpoint)
            if cache_entry is not None and not reused:
                _cache_still(output / "video" / f"{emitted - 1:04d}", cache_entry,
                             endpoint_signature, preview=False)
            if index == 0:
                anchor = output / "stills/0000"
                _save_grid(previous, anchor, True, masks, args.size, args.invert_ink)
                _copy_grid(output / "video/0000", anchor)
                backgrounds["stills/0000"] = previous.background.tolist()
            previous_target = target if args.video_transition_frames else None
            completed += 1
            sampled_count += 1
            report(0.0, f"Finished video frame {index + 1}/{len(times)}")
    finally:
        decoded.close()
    if sampled_count != len(times) or emitted != count:
        raise ValueError("Decoded video frame count does not match its timing metadata")
    video = {"path": "video", "frame_count": count, "duration_ms": info.duration_ms}
    if all(math.isclose(duration, durations[0], rel_tol=1e-9, abs_tol=1e-6) for duration in durations):
        video["frame_duration_ms"] = info.duration_ms / count
    else:
        video["frame_durations_ms"] = durations
    _atomic_json(output / "manifest.json", {
        "schema_version": 3, "video": video, "loops": {}, "signature": signature,
        "generation": getattr(args, "generation", None) or uuid.uuid4().hex,
        "image_count": 1, "frame_count": args.frames,
        "grid": [previous.glyph.shape[1], previous.glyph.shape[0]],
        "output_size": list(args.size), "cell_size": list(cell),
        "atlas_count": len(args.chars), "invert_ink": args.invert_ink,
        "background_labs": backgrounds,
    })


def _import_archive(args: argparse.Namespace, output: Path) -> None:
    from .archive import extract_archive

    if len(args.images) != 1:
        raise ValueError("Archive input requires exactly one source archive")
    source = args.images[0].expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_relative_to(output):
        raise ValueError("Archive output must not contain the input archive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.force:
        raise ValueError("Archive output must be absent or empty; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ascii-import-", dir=output.parent) as temporary:
        stage = Path(temporary) / "assets"
        manifest = extract_archive(source, stage)
        manifest["signature"] = {"input_format": "archive", "images": [str(source)],
                                 "image_hashes": [_file_hash(source)]}
        manifest["generation"] = getattr(args, "generation", None) or uuid.uuid4().hex
        _atomic_json(stage / "manifest.json", manifest)
        backup = Path(temporary) / "previous"
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
    print(f"Imported ASCII archive: {output}")
    if getattr(args, "progress_json", False):
        print('ASCII_PROGRESS {"progress": 1.0, "stage": "Ready"}', flush=True)


def build(args: argparse.Namespace) -> None:
    """Choose an input/output container before preparing numerical rendering."""
    input_format = getattr(args, "input_format", "images")
    if input_format not in ("images", "video", "archive"):
        raise ValueError("Invalid input_format")
    output_format = getattr(args, "output_format", "directory")
    if output_format not in ("directory", "zip"):
        raise ValueError("Invalid output_format")
    output = args.output.expanduser().resolve()
    if any(output == Path(path).expanduser().resolve() for path in args.images):
        raise ValueError("Output must differ from every input source")
    if output_format == "zip":
        from .archive import write_archive

        with tempfile.TemporaryDirectory(prefix="ascii-export-") as temporary:
            inner = argparse.Namespace(**vars(args))
            inner.output = Path(temporary) / "assets"
            inner.output_format = "directory"
            build(inner)
            write_archive(inner.output, output)
        print(f"Compressed ASCII archive: {output}")
        return
    if input_format == "archive":
        _import_archive(args, output)
        return
    if args.size is None:
        raise ValueError("--size is required for image and video rendering")
    _build_directory(args)


def _build_directory(args: argparse.Namespace) -> None:
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
    if args.input_format == "video" and len(args.images) != 1:
        raise ValueError("Video input requires exactly one source video")
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
    effective_chars = args.chars  # normalize_options already resolves style defaults
    chars = list(effective_chars)
    if not chars or len(chars) > 256 or len(set(chars)) != len(chars):
        raise ValueError("Characters must be 1 to 256 distinct glyphs")
    cell = validate_size(args.cell or cell_for_display(args.size, args.detail), max_dimension=255)
    masks = glyph_masks(font, chars, cell)
    loop_count = args.flow_loop_seconds * args.flow_fps if args.flow_animated else 0
    cells = math.ceil(args.size[0]/cell[0]) * math.ceil(args.size[1]/cell[1])
    if loop_count and cells * loop_count * len(args.images) > 40_000_000:
        raise ValueError("Animated render is too large; reduce detail, loop length, frame rate or image count")
    images = [path.expanduser().resolve() for path in args.images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = {
        "input_format": args.input_format,
        "frame_stride": args.frame_stride, "video_transition_frames": args.video_transition_frames,
        "version": __version__, "render_revision": render_revision(), "images": [str(path) for path in images],
        "image_hashes": [_file_hash(path) for path in images],
        "font": str(font), "font_hash": _file_hash(font),
        "size": list(args.size), "cell": list(cell), "detail": args.detail, "chars": effective_chars,
        "style": style, "invert_ink": invert_ink, "structure_dither": structure_dither,
        "flow_strength": args.flow_strength, "flow_eddies": args.flow_eddies, "flow_scale": args.flow_scale,
        **{name: getattr(args, name) for name in ("flow_animated", "flow_loop_seconds", "flow_fps",
           "flow_frame_steps", "flow_temporal_strength")},
        "gradient_loss_strength": float(gradient_loss_strength),
        "glyph_luminance_bonus": glyph_luminance_bonus,
        "glyph_entropy_strength": args.glyph_entropy_strength,
        "perceptual_loss": args.perceptual_loss,
        "output_stage": ("adam_terminal" if args.perceptual_loss else
                         "postprocessed" if style == "headroom" else "structure"),
        "perceptual_weights": PERCEPTUAL_WEIGHTS if args.perceptual_loss else {},
        "viewing_ppd": ppd_for_width(args.size[0]) if args.perceptual_loss else None,
        "crop": args.crop, "grayscale": args.grayscale,
        "frames": args.frames, "steps": args.steps,
        "transition_steps": args.transition_steps,
        "transition_type": args.transition_type,
        "transition_pace": args.transition_pace, "transition_field": args.transition_field,
        "transition_spread": args.transition_spread,
        "glyph_search": "all" if style != "structure" else "structure",
        "dither": args.dither, "background": args.background, "seed": args.seed,
    }
    output = args.output.expanduser().resolve()
    if args.input_format == "video":
        signature.update(frame_stride=args.frame_stride, video_transition_frames=args.video_transition_frames,
                         video_preview_ms=args.video_preview_ms)
        _build_video(args, images[0], output, signature, masks, cell)
        return
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
            expected_loops = [output / "loops" / f"{i:04d}" / f"{frame:04d}" / f"{kind}.png"
                              for i in range(len(images)) for frame in range(loop_count) for kind in MAP_KINDS]
            if (old.get("signature") == signature and (output / "atlas.png").exists()
                    and (output / "metadata.png").exists()
                    and all(p.exists() for p in expected_stills + expected_pairs + expected_loops)
                    and (not loop_count or (old.get("schema_version") == 2 and all(
                        old.get("loops", {}).get(f"{i:04d}", {}).get("frame_count") == loop_count
                        and old["loops"][f"{i:04d}"].get("path") == f"loops/{i:04d}"
                        and old["loops"][f"{i:04d}"].get("duration_ms") == args.flow_loop_seconds * 1000
                        and old["loops"][f"{i:04d}"].get("loop") is True for i in range(len(images)))))):
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
        TRANSITION_SETTINGS)
    old_hashes = old_signature.get("image_hashes", [])
    if not isinstance(old_hashes, list):
        old_hashes = []
    # Keep grids on disk. Each transition worker loads only its active pair,
    # so source-image memory is bounded by worker count, not slideshow length.
    background_labs: dict[str, list[float]] = {}
    grid_shape = None
    loops = {}
    for index, image in enumerate(images):
        still_dir = output / "stills" / f"{index:04d}"
        still_signature = {key: value for key, value in signature.items()
                           if key not in {"images", "image_hashes", "font", *TRANSITION_SETTINGS}}
        still_signature.update(image_hash=signature["image_hashes"][index], seed=args.seed + index)
        cache_key = hashlib.sha256(json.dumps(still_signature, sort_keys=True).encode()).hexdigest()
        cache_entry = still_cache / cache_key if still_cache is not None else None
        cached = _cached_still(cache_entry, still_signature) if cache_entry is not None and not args.force else None
        if args.flow_animated:
            loop_dir = output / "loops" / f"{index:04d}"
            loop_cache = still_cache.parent / "loops" / cache_key if still_cache is not None else None
            cached_clip = _cached_loop(loop_cache, still_signature, loop_count) if loop_cache and not args.force else None
            if cached_clip is not None:
                shutil.rmtree(loop_dir, ignore_errors=True)
                shutil.copytree(cached_clip, loop_dir)
                info = json.loads((loop_dir / "clip.json").read_text())
            else:
                target = load_target(image, args.size, args.crop, args.grayscale)
                info = _render_flow_image(target, masks, args, loop_dir, args.seed + index,
                    lambda fraction, stage: report(fraction, f"Image {index + 1}/{len(images)}: {stage}"))
                del target
                if loop_cache is not None:
                    _cache_loop(loop_dir, loop_cache, still_signature)
            loops[f"{index:04d}"] = dict(info, path=f"loops/{index:04d}")
            grid = _save_grid(_load_grid(loop_dir / "0000"), still_dir, True, masks, args.size, invert_ink)
            _copy_grid(loop_dir / "0000", still_dir)
            for frame in range(loop_count):
                background_labs[f"loops/{index:04d}/{frame:04d}"] = grid.background.tolist()
        elif (still_settings_match and index < len(old_hashes)
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
                         flow_strength=args.flow_strength, flow_eddies=args.flow_eddies,
                         flow_scale=args.flow_scale,
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

            old_threads = torch.get_num_threads()
            if workers > 1:
                torch.set_num_threads(min(old_threads, max(1, (os.cpu_count() or 1) // workers)))

            def render_pair(index: int) -> dict:
                pair_report(index, 0.0, f"Starting transition {index + 1}/{len(images)}")
                return _render_transition(index, images, output, args, masks, cell, paths,
                                          lambda units, stage: pair_report(index, units, stage))

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
    _atomic_json(manifest, {"schema_version": 2, "loops": loops, "signature": signature, "generation": getattr(args, "generation", None) or uuid.uuid4().hex,
                            "image_count": len(images), "frame_count": args.frames,
                            "grid": [grid_shape[1], grid_shape[0]], "output_size": list(args.size),
                            "cell_size": list(cell), "atlas_count": len(chars), "invert_ink": invert_ink,
                            "background_labs": background_labs})
    print(f"Rendered assets: {output}")
    report(1.0, "Ready")


def make_parser() -> argparse.ArgumentParser:
    defaults = normalize_options({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Source images, one video, or one ASCII archive")
    parser.add_argument("--input-format", choices=("images", "video", "archive"), default="images")
    parser.add_argument("--output", type=Path, required=True, help="Generated asset directory or compressed archive")
    parser.add_argument("--output-format", choices=("directory", "zip"), default="directory",
                        help="Directory of assets or an explicitly compressed ZIP archive")
    parser.add_argument("--size", type=_size, help="Wallpaper size, WIDTHxHEIGHT (required when rendering)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Render every Nth source video frame without changing playback duration")
    parser.add_argument("--video-transition-frames", type=int, default=0,
                        help="Interpolated frames between sampled video frames (default: 0)")
    parser.add_argument("--video-preview-ms", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--reuse-video-from", type=Path, action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--font", default=defaults["font"], help="Font family or font file")
    parser.add_argument("--chars", default=DEFAULT_CHARS, help="Distinct characters, including a space")
    parser.add_argument("--detail", type=int, choices=(0, 1, 2, 3, 4), default=defaults["detail"])
    parser.add_argument("--cell", type=_size, help="Override cell dimensions")
    parser.add_argument("--crop", choices=("fill", "fit"), default=defaults["crop"])
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frames", type=int, default=defaults["frames"], help="Frames per slideshow transition")
    parser.add_argument("--transition-type", choices=TYPES, default=defaults["transition_type"],
                        help="Glyph changes in place or earth mover transport across cells")
    parser.add_argument("--transition-pace", choices=PACES, default=defaults["transition_pace"])
    parser.add_argument("--transition-field", choices=FIELDS, default=defaults["transition_field"])
    parser.add_argument("--transition-spread", type=float, default=defaults["transition_spread"])
    parser.add_argument("--steps", type=int, default=defaults["steps"], help="Still optimization steps")
    # Accepted for saved settings and old commands; transitions interpolate.
    parser.add_argument("--transition-steps", type=int, default=defaults["transition_steps"],
                        help=argparse.SUPPRESS)
    # Accept old commands without restricting the full glyph-bank solve.
    parser.add_argument("--candidates", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dither", type=float, default=defaults["dither"],
                        help="Glyph-logit noise for classic headroom; perceptual headroom returns the raw Adam terminal without post-optimization noise")
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
    parser.add_argument("--style", choices=("headroom", "structure", "flow"), default=defaults["style"])
    parser.add_argument("--flow-strength", type=float, default=DEFAULT_FLOW_STRENGTH,
                        help="Flow influence in glyph selection, 0–2 (0 reproduces perceptual headroom with the same glyph bank)")
    parser.add_argument("--flow-eddies", type=float, default=DEFAULT_FLOW_EDDIES,
                        help="Eddy forcing strength, 0–2")
    parser.add_argument("--flow-scale", type=float, default=DEFAULT_FLOW_SCALE,
                        help="Domain size in glyph widths, 1–24")
    parser.add_argument("--flow-animated", action="store_true", help="Prepare a looping offline flow animation")
    parser.add_argument("--flow-loop-seconds", type=int, default=defaults["flow_loop_seconds"])
    parser.add_argument("--flow-fps", type=int, default=defaults["flow_fps"])
    parser.add_argument("--flow-frame-steps", type=int, default=defaults["flow_frame_steps"], help="Optimization updates per animated frame")
    parser.add_argument("--flow-temporal-strength", type=float, default=defaults["flow_temporal_strength"], help="Cyclic raster/color coherence weight")
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
