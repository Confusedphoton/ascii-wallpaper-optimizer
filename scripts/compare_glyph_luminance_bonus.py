"""Compare brighter-ink rewards against the current glyph/color objective.

Run from the repository root with:
    .venv/bin/python scripts/compare_glyph_luminance_bonus.py

Each bonus subtracts ``strength * mean(OKLab_L(ink color))`` from the current
full-frame objective. A strength of zero is the unmodified objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ascii_optimizer.assets import (
    glyph_masks,
    linear_to_srgb,
    load_target,
    resolve_font,
    srgb_to_linear,
)
from ascii_optimizer.cli import DEFAULT_CHARS, solve_style
from ascii_optimizer.optimizer import Grid, render_grid
from ascii_optimizer.tone import display_tone, rgb_to_oklab


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


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def _preview(grid: Grid, masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    linear = render_grid(grid, masks)[:size[1], :size[0]]
    return linear_to_srgb(torch.from_numpy(display_tone(linear))).numpy()


def _glyph_metrics(grid: Grid, masks: np.ndarray, chars: str, base_glyph: np.ndarray | None) -> dict:
    indices, counts = np.unique(grid.glyph, return_counts=True)
    total = grid.glyph.size
    probabilities = counts.astype(np.float64) / total
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    coverage = masks.mean((1, 2))
    max_coverage_index = int(coverage.argmax())
    histogram = [
        {"glyph": chars[int(index)], "count": int(count), "share": float(count / total),
         "coverage": float(coverage[int(index)])}
        for index, count in sorted(zip(indices, counts), key=lambda item: -item[1])
    ]
    used_coverage = coverage[grid.glyph]
    ink_lightness = rgb_to_oklab(grid.color)[..., 0]
    most_common_index = int(indices[counts.argmax()])
    metrics = {
        "distinct_glyphs": int(len(indices)),
        "glyph_entropy_bits": entropy,
        "effective_glyph_count": float(2 ** entropy),
        "most_common_glyph": chars[most_common_index],
        "most_common_glyph_share": float(counts.max() / total),
        "highest_coverage_glyph": chars[max_coverage_index],
        "highest_coverage_glyph_coverage": float(coverage[max_coverage_index]),
        "highest_coverage_glyph_share": float(np.mean(grid.glyph == max_coverage_index)),
        "mean_glyph_coverage": float(used_coverage.mean()),
        "median_glyph_coverage": float(np.median(used_coverage)),
        "dense_glyph_share_coverage_ge_0_30": float(np.mean(used_coverage >= 0.30)),
        "mean_ink_oklab_lightness": float(ink_lightness.mean()),
        "ink_oklab_lightness_std": float(ink_lightness.std()),
        "glyph_histogram": histogram,
    }
    if base_glyph is not None:
        metrics["glyph_change_fraction_vs_current"] = float(np.mean(grid.glyph != base_glyph))
    return metrics


def _image_metrics(reference: np.ndarray, rendered: np.ndarray) -> dict:
    reference_linear = srgb_to_linear(torch.from_numpy(reference)).numpy()
    rendered_linear = srgb_to_linear(torch.from_numpy(rendered)).numpy()
    delta = rgb_to_oklab(reference_linear) - rgb_to_oklab(rendered_linear)
    delta_e = np.sqrt(np.square(delta).sum(-1))
    return {
        "mean_oklab_delta_e_x100": float(delta_e.mean() * 100),
        "p90_oklab_delta_e_x100": float(np.quantile(delta_e, 0.9) * 100),
        "linear_rgb_rmse": float(np.sqrt(np.mean((reference_linear - rendered_linear) ** 2))),
        "oklab_lightness_rmse_x100": float(np.sqrt(np.mean(delta[..., 0] ** 2)) * 100),
        "oklab_chroma_rmse_x100": float(np.sqrt(np.mean(np.square(delta[..., 1:])) * 2) * 100),
    }


def _sheet(previews: dict[str, Image.Image], path: Path,
           crop: tuple[int, int, int, int] | None = None) -> None:
    keys = ["reference", *(f"bonus_{bonus:g}" for bonus in BONUSES)]
    width, height = previews["reference"].size
    gap, heading = 14, 38
    panel = (width, height) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    columns = 3
    rows = (len(keys) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel[0] + (columns + 1) * gap,
                               rows * (panel[1] + heading) + (rows + 1) * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 19)
    for index, key in enumerate(keys):
        x = gap + index % columns * (panel[0] + gap)
        y = gap + index // columns * (panel[1] + heading + gap)
        title = "Source" if key == "reference" else f"β = {key.removeprefix('bonus_')}"
        draw.text((x, y), title, fill="white", font=font)
        image = previews[key]
        if crop is not None:
            image = image.crop(crop)
        if crop is None:
            image = image.resize(panel, Image.Resampling.LANCZOS)
        else:
            image = image.resize(panel, Image.Resampling.NEAREST)
        sheet.paste(image, (x, y + heading))
    sheet.save(path)


def main() -> None:
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
            "objective": "current full-frame OKLab objective minus bonus * mean OKLab lightness of fitted ink colors",
            "bonus_strengths": BONUSES,
        },
        "images": {},
    }
    for label, source, focus_crop in SOURCES:
        with Image.open(source) as original:
            size = (WIDTH, round(WIDTH * original.height / original.width))
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
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
