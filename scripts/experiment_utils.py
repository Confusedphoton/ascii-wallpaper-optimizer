"""Shared playback previews, metrics, and reproducibility for render experiments.

Grid colors and composed frames are OKLab. Only source/output image boundaries
use sRGB; previews round-trip the same RGB8 foreground/background as playback.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ascii_optimizer import __version__
from ascii_optimizer.assets import glyph_masks, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.cli import solve_style as _solve_style
from ascii_optimizer.defaults import (DEFAULT_CHARS, DEFAULT_GLYPH_LUMINANCE_BONUS,
                                      DEFAULT_GRADIENT_LOSS_STRENGTH, DEFAULT_GLYPH_ENTROPY_STRENGTH)
from ascii_optimizer.optimizer import Grid, _sobel_gradients, render_grid
from ascii_optimizer.tone import (display_tone_lab, headroom_curve, oklab_to_rgb,
                                  oklab_to_srgb, quantize_playback_lab, srgb_to_oklab)


WIDTH, CELL, STEPS, CANDIDATES, SEED, DITHER = 960, (8, 16), 36, 6, 0, 0.005
SOURCE = Path("screenshots/test_image.png")
SIZE = (960, 540)
EXPANDED_CHARS = " .´:-_/|+rszunvcxLOGDQ0KNMW&@"
SOURCES = (
    ("test_image", SOURCE, (205, 105, 380, 215)),
    ("tree", Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg"), (140, 95, 270, 215)),
    ("snow", Path("screenshots/149642007_p0-无题.jpg"), (175, 325, 365, 475)),
)
SNOW_BELT = (365, 215, 480, 310)
_RUNS: list[dict] = []


def image(array: np.ndarray) -> Image.Image:
    """Create an output RGB8 image from a displayed sRGB array."""
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def playback_grid(grid: Grid) -> Grid:
    background = np.zeros(3, np.float32) if grid.background is None else grid.background
    return Grid(grid.glyph, quantize_playback_lab(grid.color), quantize_playback_lab(background))


def _lighten_at_fixed_chroma(lab: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Raise L up to the gamut boundary while preserving the existing a,b."""
    lab = quantize_playback_lab(lab)
    low = lab[..., 0].copy()
    high = np.maximum(low, np.clip(requested, 0, 1))
    for _ in range(16):
        mid = (low + high) * 0.5
        trial = lab.copy()
        trial[..., 0] = mid
        rgb = oklab_to_rgb(trial)
        valid = ((rgb >= -1e-5) & (rgb <= 1 + 1e-5)).all(-1)
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    lab[..., 0] = low
    return quantize_playback_lab(lab)


def brighten_structure(grid: Grid, *, ink_gain: float, background_lift: float = 0) -> Grid:
    """Experimental color adjustment with the glyph map held fixed."""
    persisted = playback_grid(grid)
    color = _lighten_at_fixed_chroma(persisted.color, headroom_curve(persisted.color[..., 0], ink_gain))
    background = _lighten_at_fixed_chroma(persisted.background,
                                         persisted.background[0] + background_lift)
    return Grid(grid.glyph.copy(), color, background)


def preview(grid: Grid, masks: np.ndarray, size: tuple[int, int], *,
            invert_ink: bool = False, display_transform: bool = True) -> np.ndarray:
    """Compose serialized maps; return the exact displayed RGB8 colors as floats."""
    masks = np.rint(np.clip(masks, 0, 1) * 255).astype(np.float32) / 255
    lab = render_grid(playback_grid(grid), masks)[:size[1], :size[0]]
    if display_transform:
        lab = display_tone_lab(lab, invert_ink=invert_ink)
    # Metrics measure the saved preview, avoiding a second unquantized output.
    return np.asarray(image(oklab_to_srgb(lab)), dtype=np.float32) / 255


def coverage_metrics(grid: Grid, masks: np.ndarray) -> dict[str, float]:
    coverage = masks[grid.glyph].mean((2, 3))
    return {"mean_glyph_coverage": float(coverage.mean()),
            "dense_glyph_fraction": float((coverage >= 0.3).mean())}


def image_metrics(reference: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    delta = srgb_to_oklab(reference) - srgb_to_oklab(rendered)
    delta_e = np.linalg.norm(delta, axis=-1)
    reference_linear = srgb_to_linear(torch.from_numpy(reference)).numpy()
    rendered_linear = srgb_to_linear(torch.from_numpy(rendered)).numpy()
    return {"mean_oklab_delta_e_x100": float(delta_e.mean() * 100),
            "median_oklab_delta_e_x100": float(np.median(delta_e) * 100),
            "p90_oklab_delta_e_x100": float(np.quantile(delta_e, 0.9) * 100),
            "linear_rgb_rmse": float(np.sqrt(np.mean((reference_linear - rendered_linear) ** 2))),
            "oklab_lightness_rmse_x100": float(np.sqrt(np.mean(delta[..., 0] ** 2)) * 100),
            "oklab_chroma_rmse_x100": float(np.sqrt(np.mean(delta[..., 1:] ** 2) * 2) * 100)}


def glyph_metrics(grid: Grid, masks: np.ndarray, chars: str,
                  base_glyph: np.ndarray | None = None) -> dict:
    indices, counts = np.unique(grid.glyph, return_counts=True)
    probabilities = counts.astype(np.float64) / grid.glyph.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    coverage = masks.mean((1, 2))
    used_coverage = coverage[grid.glyph]
    ink_lightness = playback_grid(grid).color[..., 0]
    most_common = int(indices[counts.argmax()])
    densest = int(coverage.argmax())
    metrics = {**coverage_metrics(grid, masks),
               "distinct_glyphs": len(indices), "glyph_entropy_bits": entropy,
               "effective_glyph_count": float(2 ** entropy),
               "most_common_glyph": chars[most_common],
               "most_common_glyph_share": float(counts.max() / grid.glyph.size),
               "highest_coverage_glyph": chars[densest],
               "highest_coverage_glyph_coverage": float(coverage[densest]),
               "highest_coverage_glyph_share": float((grid.glyph == densest).mean()),
               "median_glyph_coverage": float(np.median(used_coverage)),
               "dense_glyph_share_coverage_ge_0_30": float((used_coverage >= 0.3).mean()),
               "mean_ink_oklab_lightness": float(ink_lightness.mean()),
               "ink_oklab_lightness_std": float(ink_lightness.std()),
               "glyph_histogram": [{"glyph": chars[int(index)], "count": int(count),
                                    "share": float(count / grid.glyph.size),
                                    "coverage": float(coverage[int(index)])}
                                   for index, count in sorted(zip(indices, counts), key=lambda item: -item[1])]}
    if base_glyph is not None:
        metrics["glyph_change_fraction_vs_baseline"] = float((grid.glyph != base_glyph).mean())
    return metrics


def gradients(srgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(srgb_to_oklab(srgb)).permute(2, 0, 1).unsqueeze(0)
    gx, gy = _sobel_gradients(tensor)
    return gx[0].permute(1, 2, 0).numpy(), gy[0].permute(1, 2, 0).numpy()


def gradient_rmse(reference: np.ndarray, rendered: np.ndarray) -> float:
    sx, sy = gradients(reference)
    ox, oy = gradients(rendered)
    return float(np.sqrt(((ox - sx) ** 2 + (oy - sy) ** 2).sum(-1).mean()) * 100)


def red_chroma_metrics(reference: np.ndarray, rendered: np.ndarray,
                       crop: tuple[int, int, int, int] = SNOW_BELT) -> dict:
    x1, y1, x2, y2 = crop
    source = reference[y1:y2, x1:x2]
    output = rendered[y1:y2, x1:x2]
    red, green, blue = np.moveaxis(source, -1, 0)
    selected = (red > 0.25) & (red > 1.25 * green) & (red > 1.3 * blue)
    result = {"red_belt_sample_pixels": int(selected.sum())}
    if not selected.any():
        return {**result, **dict.fromkeys(("red_belt_target_mean_chroma", "red_belt_output_mean_chroma",
                                          "red_belt_chroma_retention", "red_belt_chroma_rmse"))}
    source_lab, output_lab = srgb_to_oklab(source), srgb_to_oklab(output)
    source_chroma = np.linalg.norm(source_lab[selected, 1:], axis=-1).mean()
    output_chroma = np.linalg.norm(output_lab[selected, 1:], axis=-1).mean()
    return {**result, "red_belt_target_mean_chroma": float(source_chroma),
            "red_belt_output_mean_chroma": float(output_chroma),
            "red_belt_chroma_retention": float(output_chroma / max(source_chroma, 1e-8)),
            "red_belt_chroma_rmse": float(np.sqrt(np.mean((source_lab[selected, 1:] - output_lab[selected, 1:]) ** 2)))}


def source_size(path: Path, width: int) -> tuple[int, int]:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        return width, max(1, round(width * source.height / source.width))


def contact_sheet(previews: list[tuple[str, Image.Image]], path: Path, *,
                  crop: tuple[int, int, int, int] | None = None,
                  columns: int = 3, panel: tuple[int, int] | None = None,
                  zoom: int = 1) -> None:
    if not previews or columns < 1 or zoom < 1:
        raise ValueError("A contact sheet needs images, positive columns, and positive zoom")
    if panel is None:
        panel = (previews[0][1].size if crop is None else
                 ((crop[2] - crop[0]) * zoom, (crop[3] - crop[1]) * zoom))
    if min(panel) < 1:
        raise ValueError("Contact-sheet panels must have positive dimensions")
    gap, heading = 14, 38
    rows = (len(previews) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel[0] + (columns + 1) * gap,
                              rows * (panel[1] + heading) + (rows + 1) * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 19)
    for index, (title, original) in enumerate(previews):
        x = gap + index % columns * (panel[0] + gap)
        y = gap + index // columns * (panel[1] + heading + gap)
        draw.text((x, y), title, fill="white", font=font)
        shown = original.crop(crop) if crop else original
        scale = min(panel[0] / shown.width, panel[1] / shown.height)
        shown = shown.resize((max(1, round(shown.width * scale)), max(1, round(shown.height * scale))),
                             Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
        sheet.paste(shown, (x + (panel[0] - shown.width) // 2,
                            y + heading + (panel[1] - shown.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def reset_runs() -> None:
    _RUNS.clear()


def solve_style(target: np.ndarray, masks: np.ndarray, **kwargs) -> Grid:
    """Measure a solve using production style defaults unless explicitly overridden."""
    style = kwargs.get("style", "headroom")
    kwargs.setdefault("glyph_luminance_bonus", DEFAULT_GLYPH_LUMINANCE_BONUS if style == "headroom" else 0.0)
    kwargs.setdefault("gradient_loss_strength", DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0)
    arguments = inspect.signature(_solve_style).bind(target, masks, **kwargs)
    arguments.apply_defaults()
    device = kwargs.get("device", "cpu")
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    grid = _solve_style(target, masks, **kwargs)
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    effective = {name: value for name, value in arguments.arguments.items()
                 if isinstance(value, (str, bool, int, float)) or value is None}
    if effective.get("gradient_loss_strength") is None:
        effective["gradient_loss_strength"] = DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0
    if effective.get("glyph_entropy_strength") is None:
        effective["glyph_entropy_strength"] = DEFAULT_GLYPH_ENTROPY_STRENGTH if style == "headroom" else 0.0
    if effective.get("saturation_loss_strength") is None:
        effective["saturation_loss_strength"] = 20.0 if effective["glyph_luminance_bonus"] > 0 else 0.0
    effective["glyphs_per_cell"] = (min(effective["candidates"], len(masks))
                                   if style == "headroom" and effective["use_shortlist"] else len(masks))
    if not effective["use_shortlist"] or style != "headroom":
        effective.pop("candidates", None)
    effective.pop("on_progress", None)
    measurements = {"run_index": len(_RUNS), "seconds": elapsed, "effective_objectives": effective,
                    "source_array_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                    "masks_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
                    "source_shape": list(target.shape), "mask_shape": list(masks.shape),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if platform.system() == "Darwin" else 1024)}
    if cuda:
        measurements["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    _RUNS.append(measurements)
    grid.experiment = measurements
    return grid


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, results, *, sources=(), font="Monospace") -> None:
    """Include source/font hashes, revision, effective objectives, and measurements."""
    paths = set(Path(source).expanduser().resolve() for source in sources)

    def discover(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "source" and isinstance(item, str):
                    paths.add(Path(item).expanduser().resolve())
                discover(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                discover(item)

    discover(results)
    font_path = resolve_font(font)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=False).stdout.strip() or None
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True,
                           text=True, check=False).stdout.strip()
    renderer_directory = Path(__file__).resolve().parents[1] / "ascii_optimizer"
    experiment_path = Path(inspect.stack()[1].filename).resolve()
    provenance = {"renderer_version": __version__, "git_revision": revision, "working_tree_dirty": bool(dirty),
                  "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "font": str(font_path), "font_sha256": _hash_file(font_path),
                  "renderer_sources_sha256": {source.name: _hash_file(source) for source in sorted(renderer_directory.glob("*.py"))},
                  "experiment_utils_sha256": _hash_file(Path(__file__)),
                  "experiment_source": str(experiment_path),
                  "experiment_source_sha256": _hash_file(experiment_path) if experiment_path.is_file() else None,
                  "sources": [{"path": str(source), "sha256": _hash_file(source)} for source in sorted(paths)],
                  "preview_format": "RGB8 sRGB after RGB8 ink/background playback composition in OKLab",
                  "source_preparation": {"crop": "fill", "grayscale": False},
                  "memory_scope": "process lifetime high-water RSS; per-solve CUDA peak when used",
                  "runs": list(_RUNS)}
    report = {**results, "reproducibility": provenance} if isinstance(results, dict) else {"variants": results, "reproducibility": provenance}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
