"""Compare one-candidate and six-candidate OKLab solves.

Run from the repository root with:
    .venv/bin/python scripts/compare_color_spaces.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS
from experiment_utils import (coverage_metrics, image_metrics, preview as _render,
                              image as _image, reset_runs, source_size, solve_style, write_report, contact_sheet)


OUTPUT = Path("rendered/color_spaces")
SOURCES = [
    ("test_image", Path("screenshots/test_image.png")),
    ("tree", Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg")),
    ("snow", Path("screenshots/149642007_p0-无题.jpg")),
    ("viviana_gradient", Path("screenshots/143381492_p0-Viviana.jpg")),
]
WIDTH = 960
CELL = (8, 16)
STEPS = 36
CANDIDATES = 6
VARIANTS = [
    ("reference", "Source"),
    ("one_candidate", "1 candidate · OKLab"),
    ("six_candidates", "6 candidates · OKLab"),
]


def _metrics(reference, rendered, base_grid, grid, masks):
    return {**image_metrics(reference, rendered), **coverage_metrics(grid, masks),
            "glyph_change_fraction_vs_one_candidate": float((grid.glyph != base_grid.glyph).mean())}


def _sheet(images, path):
    contact_sheet([(title, images[key]) for key, title in VARIANTS], path)


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    masks = glyph_masks(resolve_font("Monospace"), list(DEFAULT_CHARS), CELL)
    results = {
        "settings": {
            "width": WIDTH,
            "cell": CELL,
            "steps": STEPS,
            "candidates": CANDIDATES,
            "seed": 0,
            "dither": 0.005,
            "objective": "squared Euclidean OKLab (Delta E OK)",
        },
        "images": {},
    }
    for label, path in SOURCES:
        size = source_size(path, WIDTH)
        reference = load_target(path, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(parents=True, exist_ok=True)
        previews = {"reference": _image(reference)}
        previews["reference"].save(directory / "reference.png")
        image_results = {"source": str(path), "size": size, "variants": {}}
        grids = {}
        for count, key, title in ((1, "one_candidate", "1 candidate · OKLab"),
                                  (CANDIDATES, "six_candidates", "6 candidates · OKLab")):
            print(f"{label}: solving {title}", flush=True)
            grid = solve_style(
                reference, masks, style="headroom", invert_ink=False,
                background="auto", seed=0, steps=STEPS, candidates=count,
                use_shortlist=True,
                dither=0.005, structure_dither=0.06, device="cpu",
            )
            grids[key] = grid
            rendered = _render(grid, masks, size)
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            image_results["variants"][key] = {
                "title": title,
                "experiment": grid.experiment,
                "metrics": _metrics(reference, rendered, grids.get("one_candidate", grid),
                                     grid, masks),
            }
        # The one-candidate solve is the glyph-change comparison anchor.
        for key in ("one_candidate", "six_candidates"):
            image_results["variants"][key]["metrics"]["glyph_change_fraction_vs_one_candidate"] = float(
                np.mean(grids[key].glyph != grids["one_candidate"].glyph))
        _sheet(previews, directory / "comparison.png")
        results["images"][label] = image_results
        print(f"{label}: wrote {directory}", flush=True)
    write_report(OUTPUT / "metrics.json", results)
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
