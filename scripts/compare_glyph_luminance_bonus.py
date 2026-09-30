"""Compare brighter-ink rewards against the current glyph/color objective.

Run from the repository root with:
    .venv/bin/python scripts/compare_glyph_luminance_bonus.py

Each bonus subtracts ``strength * mean(L - L² + L³/3)`` for nonempty glyphs
from the current full-frame objective, tapering the incentive as ink lightness
approaches white. Zero removes the lighter-ink reward. This sweep disables the
saturation objective to isolate the reward; production enables both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS
from experiment_utils import (
    contact_sheet, glyph_metrics as _glyph_metrics, image as _image,
    image_metrics as _image_metrics, preview as _preview, reset_runs, source_size,
    solve_style, write_report,
)

OUTPUT = Path("rendered/glyph_luminance_bonus")
WIDTH = 960
CELL = (8, 16)
STEPS = 36
CANDIDATES = 6
SEED = 0
DITHER = 0.005
BONUSES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12)
SOURCES = (
    ("test_image", Path("screenshots/test_image.png"), (205, 105, 380, 215)),
    ("tree", Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg"), (140, 95, 270, 215)),
    ("snow", Path("screenshots/149642007_p0-无题.jpg"), (175, 325, 365, 475)),
)


def _sheet(previews, path, crop=None):
    keys = ["reference", *(f"bonus_{bonus:g}" for bonus in BONUSES)]
    contact_sheet([("Source" if key == "reference" else
                    f"β = {key.removeprefix('bonus_')}", previews[key]) for key in keys],
                  path, crop=crop)


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chars = DEFAULT_CHARS
    masks = glyph_masks(resolve_font("Monospace"), list(chars), CELL)
    results = {
        "settings": {
            "width": WIDTH,
            "cell": CELL,
            "steps": STEPS,
            "candidates": CANDIDATES,
            "seed": SEED,
            "dither": DITHER,
            "glyphs": chars,
            "objective": "current full-frame OKLab objective minus bonus * mean(L - L² + L³/3) for nonempty glyphs",
            "bonus_strengths": BONUSES,
        },
        "images": {},
    }
    for label, source, focus_crop in SOURCES:
        size = source_size(source, WIDTH)
        reference = load_target(source, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(parents=True, exist_ok=True)
        previews = {"reference": _image(reference)}
        previews["reference"].save(directory / "reference.png")
        variants = {}
        current_grid = None
        for bonus in BONUSES:
            print(f"{label}: solving lightness bonus {bonus:g}", flush=True)
            grid = solve_style(
                reference, masks, style="headroom", invert_ink=False,
                background="auto", seed=SEED, steps=STEPS,
                candidates=CANDIDATES, dither=DITHER,
                structure_dither=0.06, device="cpu",
                glyph_luminance_bonus=bonus,
                saturation_loss_strength=0.0,
            )
            if bonus == 0:
                current_grid = grid
            rendered = _preview(grid, masks, size)
            key = f"bonus_{bonus:g}"
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            variants[key] = {
                "bonus_strength": bonus,
                "experiment": grid.experiment,
                **_image_metrics(reference, rendered),
                **_glyph_metrics(grid, masks, chars, None if bonus == 0 else current_grid.glyph),
            }
        assert current_grid is not None
        _sheet(previews, directory / "comparison.png")
        _sheet(previews, directory / "focus_comparison.png", focus_crop)
        results["images"][label] = {
            "source": str(source),
            "size": size,
            "focus_crop": focus_crop,
            "variants": variants,
        }
    write_report(OUTPUT / "metrics.json", results)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
