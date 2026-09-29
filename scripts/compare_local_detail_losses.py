"""Render the three screenshot sources with small-color-detail loss experiments.

Run from the repository root: .venv/bin/python scripts/compare_local_detail_losses.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.defaults import DEFAULT_CHARS
from experiment_utils import (image as _image, preview as _preview, reset_runs, source_size,
                              solve_style, write_report, contact_sheet)
from ascii_optimizer.structure import STRUCTURE_CHARS
from ascii_optimizer.tone import srgb_to_oklab


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
WIDTH = 960
CELL = (8, 16)
STEPS = 36
CANDIDATES = 6


def _metrics(reference: np.ndarray, rendered: np.ndarray, eye_box: tuple[int, int, int, int],
             iris_boxes: list[tuple[int, int, int, int]]) -> dict:
    reference_linear = srgb_to_linear(torch.from_numpy(reference)).numpy()
    rendered_linear = srgb_to_linear(torch.from_numpy(rendered)).numpy()
    eye = np.s_[eye_box[1]:eye_box[3], eye_box[0]:eye_box[2]]
    reference_lab = srgb_to_oklab(reference[eye])
    rendered_lab = srgb_to_oklab(rendered[eye])
    chroma_error = reference_lab[..., 1:] - rendered_lab[..., 1:]
    reference_full_lab = srgb_to_oklab(reference)
    rendered_full_lab = srgb_to_oklab(rendered)
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


def _sheet(previews, path, crop=None, variants=VARIANTS, columns=3):
    contact_sheet([(title, previews[key]) for key, title in variants], path,
                  crop=crop, columns=columns, panel=(480, 300) if crop is None else (420, 300))


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = resolve_font("Monospace")
    default_masks = glyph_masks(font, list(DEFAULT_CHARS), CELL)
    structure_masks = glyph_masks(font, list(STRUCTURE_CHARS), CELL)
    results = {"settings": {"width": WIDTH, "cell": CELL, "steps": STEPS,
                            "candidates": CANDIDATES, "seed": 0, "dither": 0.005}, "images": {}}
    for label, path, eye_box, iris_boxes in SOURCES:
        size = source_size(path, WIDTH)
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
            image_results["variants"][key] = {"experiment": grid.experiment,
                **_metrics(reference, rendered, eye_box, iris_boxes)}
            print(f"{label}: {title}", flush=True)
        _sheet(previews, directory / "comparison.png")
        _sheet(previews, directory / "eye_detail.png", eye_box)
        results["images"][label] = image_results
    write_report(OUTPUT / "metrics.json", results)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
