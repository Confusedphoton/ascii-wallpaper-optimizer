"""Compare one-candidate and six-candidate OKLab solves.

Run from the repository root with:
    .venv/bin/python scripts/compare_color_spaces.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.cli import DEFAULT_CHARS, solve_style
from ascii_optimizer.optimizer import render_grid
from ascii_optimizer.tone import display_tone_lab, oklab_to_srgb, srgb_to_oklab


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


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def _render(grid, masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    lab = render_grid(grid, masks)[:size[1], :size[0]]
    return oklab_to_srgb(display_tone_lab(lab))


def _metrics(reference: np.ndarray, rendered: np.ndarray, base_grid,
             grid, masks: np.ndarray) -> dict:
    reference_lab = srgb_to_oklab(reference)
    rendered_lab = srgb_to_oklab(rendered)
    delta = reference_lab - rendered_lab
    delta_e = np.sqrt(np.square(delta).sum(-1))
    coverage = masks[grid.glyph].mean((1, 2))
    return {
        "mean_oklab_delta_e_x100": float(delta_e.mean() * 100),
        "median_oklab_delta_e_x100": float(np.median(delta_e) * 100),
        "p90_oklab_delta_e_x100": float(np.quantile(delta_e, 0.9) * 100),
        "oklab_lightness_rmse_x100": float(np.sqrt(np.mean(delta[..., 0] ** 2)) * 100),
        "oklab_chroma_rmse_x100": float(np.sqrt(np.mean(np.square(delta[..., 1:])) * 2) * 100),
        "glyph_change_fraction_vs_one_candidate": float(np.mean(grid.glyph != base_grid.glyph)),
        "mean_glyph_coverage": float(coverage.mean()),
        "dense_glyph_fraction": float(np.mean(coverage >= 0.3)),
    }


def _sheet(images: dict[str, Image.Image], path: Path) -> None:
    width, height = images["reference"].size
    gap, heading = 16, 42
    sheet = Image.new("RGB", (3 * width + 4 * gap, height + heading + 2 * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 22)
    for index, (key, title) in enumerate(VARIANTS):
        x = gap + index * (width + gap)
        draw.text((x, gap), title, fill="white", font=font)
        sheet.paste(images[key], (x, gap + heading))
    sheet.save(path)


def main() -> None:
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
        with Image.open(path) as source_image:
            size = (WIDTH, round(WIDTH * source_image.height / source_image.width))
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
                dither=0.005, structure_dither=0.06, device="cpu",
            )
            grids[key] = grid
            rendered = _render(grid, masks, size)
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            image_results["variants"][key] = {
                "title": title,
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
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
