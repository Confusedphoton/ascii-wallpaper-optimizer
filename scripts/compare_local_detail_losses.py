"""Render the three screenshot sources with small-color-detail loss experiments.

Run from the repository root: .venv/bin/python scripts/compare_local_detail_losses.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ascii_optimizer.assets import glyph_masks, linear_to_srgb, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.cli import DEFAULT_CHARS, solve_style
from ascii_optimizer.optimizer import render_grid
from ascii_optimizer.structure import STRUCTURE_CHARS
from ascii_optimizer.tone import display_tone, rgb_to_oklab


OUTPUT = Path("rendered/local_detail_losses")
SOURCES = [
    ("original", Path("screenshots/test_image.png"), (205, 105, 380, 215),
     [(245, 166, 270, 192), (300, 140, 327, 170)]),
    ("tree", Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg"), (140, 95, 270, 215),
     [(175, 135, 197, 165), (230, 160, 255, 195)]),
    ("snow", Path("screenshots/149642007_p0-无题.jpg"), (175, 325, 365, 475),
     [(215, 413, 249, 453), (295, 380, 335, 425)]),
]
VARIANTS = [
    ("reference", "Reference"),
    ("default", "Current default"),
    ("structure", "Structure first"),
    ("local_color", "Local color weight"),
    ("chroma_contrast", "Chroma contrast"),
    ("color_tail", "Worst color pixels"),
]
CELL = (8, 16)
STEPS = 36
CANDIDATES = 6


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def _preview(grid, masks, size):
    linear = render_grid(grid, masks)[:size[1], :size[0]]
    return linear_to_srgb(torch.from_numpy(display_tone(linear))).numpy()


def _metrics(reference: np.ndarray, rendered: np.ndarray, eye_box: tuple[int, int, int, int],
             iris_boxes: list[tuple[int, int, int, int]]) -> dict:
    reference_linear = srgb_to_linear(torch.from_numpy(reference)).numpy()
    rendered_linear = srgb_to_linear(torch.from_numpy(rendered)).numpy()
    eye = np.s_[eye_box[1]:eye_box[3], eye_box[0]:eye_box[2]]
    reference_lab = rgb_to_oklab(reference_linear[eye])
    rendered_lab = rgb_to_oklab(rendered_linear[eye])
    chroma_error = reference_lab[..., 1:] - rendered_lab[..., 1:]
    reference_full_lab = rgb_to_oklab(reference_linear)
    rendered_full_lab = rgb_to_oklab(rendered_linear)
    iris_errors = []
    for x1, y1, x2, y2 in iris_boxes:
        iris_errors.append(float(np.sqrt(np.mean(
            (reference_full_lab[y1:y2, x1:x2, 1:] - rendered_full_lab[y1:y2, x1:x2, 1:]) ** 2))))
    return {
        "whole_image_linear_rgb_rmse": float(np.sqrt(np.mean((reference_linear - rendered_linear) ** 2))),
        "eye_crop_oklab_chroma_rmse": float(np.sqrt(np.mean(chroma_error ** 2))),
        "iris_boxes_oklab_chroma_rmse": float(np.mean(iris_errors)),
        "eye_crop_oklab_lightness_rmse": float(np.sqrt(np.mean((reference_lab[..., 0] - rendered_lab[..., 0]) ** 2))),
    }


def _sheet(previews: dict[str, Image.Image], path: Path,
           crop: tuple[int, int, int, int] | None = None,
           variants: list[tuple[str, str]] = VARIANTS, columns: int = 3) -> None:
    panel = (480, 300) if crop is None else (420, 300)
    gap, heading = 14, 38
    rows = (len(variants) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel[0] + (columns + 1) * gap,
                              rows * (panel[1] + heading) + (rows + 1) * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 19)
    for index, (key, title) in enumerate(variants):
        x = gap + index % columns * (panel[0] + gap)
        y = gap + index // columns * (panel[1] + heading + gap)
        draw.text((x, y), title, fill="white", font=font)
        image = previews[key].copy()
        if crop is not None:
            image = image.crop(crop)
        scale = min(panel[0] / image.width, panel[1] / image.height)
        shown = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
        sheet.paste(shown, (x + (panel[0] - shown.width) // 2,
                            y + heading + (panel[1] - shown.height) // 2))
    sheet.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = resolve_font("Monospace")
    default_masks = glyph_masks(font, list(DEFAULT_CHARS), CELL)
    structure_masks = glyph_masks(font, list(STRUCTURE_CHARS), CELL)
    results = {"settings": {"width": 960, "cell": CELL, "steps": STEPS,
                            "candidates": CANDIDATES, "seed": 0, "dither": 0.005}, "images": {}}
    for label, path, eye_box, iris_boxes in SOURCES:
        with Image.open(path) as original:
            size = (960, round(960 * original.height / original.width))
        reference = load_target(path, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(exist_ok=True)
        previews = {"reference": _image(reference)}
        previews["reference"].save(directory / "reference.png")
        image_results = {"source": str(path), "size": size, "eye_crop": eye_box,
                         "iris_boxes": iris_boxes, "variants": {}}
        for key, title in VARIANTS[1:]:
            style = "structure" if key == "structure" else "headroom"
            masks = structure_masks if style == "structure" else default_masks
            grid = solve_style(reference, masks, style=style, invert_ink=False,
                               background="auto", seed=0, steps=STEPS,
                               candidates=CANDIDATES, dither=0.005,
                               structure_dither=0.06, device="cpu",
                               detail_loss=key if style == "headroom" else "default")
            rendered = _preview(grid, masks, size)
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            image_results["variants"][key] = _metrics(reference, rendered, eye_box, iris_boxes)
            print(f"{label}: {title}", flush=True)
        _sheet(previews, directory / "comparison.png")
        _sheet(previews, directory / "eye_detail.png", eye_box)
        results["images"][label] = image_results
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
