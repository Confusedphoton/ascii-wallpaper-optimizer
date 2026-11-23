"""Standalone lab runner for rapid renderer experiments (decoupled from QML/service).

Write a small experimental renderer function:

    def render(source_srgb, atlas, sdf_spread, *, chars, cell, device, seed) -> Grid

and pass it to `run_lab(...)` to generate contact sheets, zoomed crops, and a
report compatible with `scripts/view_renders.py --report <output>`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_utils as utils
from ascii_optimizer.assets import glyph_sdf_atlas, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS, ppd_for_width
from ascii_optimizer.optimizer import Grid

FIXTURES: dict[str, Path] = {
    "viviana": ROOT / "screenshots/143381492_p0-Viviana.jpg",
    "tree": ROOT / "screenshots/149254100_p0-被树傀儡抓住惹.jpg",
    "snow": ROOT / "screenshots/149642007_p0-无题.jpg",
    "typhon": ROOT / "screenshots/__typhon_and_typhoeus_arknights_and_1_more_drawn_by_mokacoat__4cc6db689e0953c4728dc3e2db472b19.jpg",
}

LabRenderer = Callable[..., Grid]


def run_lab(
    variants: dict[str, tuple[LabRenderer, dict]],
    output: Path,
    *,
    images: tuple[str, ...] = ("viviana", "tree", "typhon"),
    width: int = 960,
    cell: tuple[int, int] = (8, 16),
    font: str = "Monospace",
    default_chars: str = DEFAULT_CHARS,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Run arbitrary experimental renderers side-by-side and emit a viewer report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font_path = resolve_font(font)
    atlas_cache: dict[str, tuple[np.ndarray, float]] = {}
    report = {
        "settings": {"width": width, "cell": list(cell), "font": font, "seed": seed},
        "images": {},
    }

    for name in images:
        source_path = FIXTURES[name]
        size = utils.source_size(source_path, width)
        target = load_target(source_path, size, "fill", False)
        scene_dir = output / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        utils.image(target).save(scene_dir / "source.png")
        scene_report = {
            "source": str(source_path),
            "size": list(size),
            "cell": list(cell),
            "ppd": ppd_for_width(size[0]),
            "variants": {},
        }
        report["images"][name] = scene_report
        panels = [("Source", utils.image(target))]

        for variant_name, (fn, kwargs) in variants.items():
            chars = kwargs.get("chars", default_chars)
            use_sdf = kwargs.get("use_sdf", True)
            cache_key = f"{use_sdf}:{chars}"
            if cache_key not in atlas_cache:
                if use_sdf:
                    atlas_cache[cache_key] = glyph_sdf_atlas(font_path, list(chars), cell)
                else:
                    atlas_cache[cache_key] = (utils.glyph_masks(font_path, list(chars), cell), 0.0)
            atlas, sdf_spread = atlas_cache[cache_key]
            variant_dir = scene_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            call_kwargs = {"chars": chars, "cell": cell, "device": device, "seed": seed, **kwargs}
            started = time.perf_counter()
            grid = fn(target, atlas, sdf_spread, **call_kwargs)
            elapsed = time.perf_counter() - started
            shown = utils.preview(grid, atlas, size, sdf_spread=sdf_spread)
            utils.image(shown).save(variant_dir / "final.png")
            scene_report["variants"][variant_name] = {
                **utils.image_metrics(target, shown),
                **utils.glyph_metrics(grid, atlas, chars, sdf_spread=sdf_spread),
                "sdf_spread": sdf_spread,
                "seconds": elapsed,
            }
            panels.append((variant_name.replace("_", " "), utils.image(shown)))

        utils.contact_sheet(panels, scene_dir / "comparison.png", columns=min(3, len(panels)),
                            panel=(480, round(480 * size[1] / size[0])))
        crop = (round(size[0] * 0.25), round(size[1] * 0.20), round(size[0] * 0.60), round(size[1] * 0.55))
        utils.contact_sheet(panels, scene_dir / "detail.png", columns=min(3, len(panels)), crop=crop, zoom=3)

    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
