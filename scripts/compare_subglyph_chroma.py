"""Compare a within-cell perceptual chroma objective against existing renderers.

Run from the repository root: .venv/bin/python scripts/compare_subglyph_chroma.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.cli import DEFAULT_CHARS, solve_style
from ascii_optimizer.optimizer import _feature_chroma_mask, _oklab
from ascii_optimizer.structure import STRUCTURE_CHARS
from ascii_optimizer.tone import encode_headroom, rgb_to_oklab
from compare_local_detail_losses import SOURCES, CELL, STEPS, CANDIDATES, _image, _metrics, _preview, _sheet


OUTPUT = Path("rendered/subglyph_chroma")
VARIANTS = [
    ("reference", "Reference"),
    ("default", "Current default"),
    ("structure", "Structure first"),
    ("feature_0_5", "Feature chroma 0.5"),
    ("feature_2", "Feature chroma 2"),
    ("feature_6", "Feature chroma 6"),
    ("contrast_0_5", "Feature contrast 0.5"),
    ("contrast_2", "Feature contrast 2"),
    ("contrast_6", "Feature contrast 6"),
]
MODES = {
    "feature_0_5": ("feature_chroma", 0.5),
    "feature_2": ("feature_chroma", 2.0),
    "feature_6": ("feature_chroma", 6.0),
    "contrast_0_5": ("feature_contrast", 0.5),
    "contrast_2": ("feature_contrast", 2.0),
    "contrast_6": ("feature_contrast", 6.0),
}
FULL_IMAGE_VARIANTS = VARIANTS[:6]


def _full_image_sheet(previews: dict[str, Image.Image], path: Path) -> None:
    width, height = previews["reference"].size
    gap, heading = 16, 40
    sheet = Image.new("RGB", (3 * width + 4 * gap,
                              2 * (height + heading) + 3 * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 22)
    for index, (key, title) in enumerate(FULL_IMAGE_VARIANTS):
        x = gap + (index % 3) * (width + gap)
        y = gap + (index // 3) * (height + heading + gap)
        draw.text((x, y), title, fill="white", font=font)
        sheet.paste(previews[key], (x, y + heading))
    sheet.save(path)


def _feature_mask(reference: np.ndarray) -> np.ndarray:
    encoded = encode_headroom(reference)
    target = srgb_to_linear(torch.from_numpy(encoded)).permute(2, 0, 1).unsqueeze(0)
    pad_height = (-target.shape[-2]) % CELL[1]
    pad_width = (-target.shape[-1]) % CELL[0]
    target = F.pad(target, (0, pad_width, 0, pad_height), mode="replicate")
    mask = _feature_chroma_mask(_oklab(target), (CELL[1], CELL[0]))
    return mask[0, 0, :reference.shape[0], :reference.shape[1]].numpy().astype(bool)


def _feature_metrics(reference: np.ndarray, rendered: np.ndarray, feature: np.ndarray,
                     iris_boxes: list[tuple[int, int, int, int]]) -> dict:
    source_lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(reference)).numpy())
    output_lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(rendered)).numpy())
    error = (source_lab[..., 1:] - output_lab[..., 1:]) ** 2
    iris_feature = np.zeros_like(feature)
    magenta_iris = np.zeros_like(feature)
    for x1, y1, x2, y2 in iris_boxes:
        iris_feature[y1:y2, x1:x2] = feature[y1:y2, x1:x2]
        a = source_lab[y1:y2, x1:x2, 1]
        magenta_iris[y1:y2, x1:x2] = a >= np.quantile(a, 0.8)
    return {
        "selected_feature_chroma_rmse": float(np.sqrt(error[feature].mean())),
        "selected_iris_feature_chroma_rmse": float(np.sqrt(error[iris_feature].mean())),
        "iris_magenta_chroma_rmse": float(np.sqrt(error[magenta_iris].mean())),
        "iris_magenta_a_retention": float(output_lab[magenta_iris, 1].mean() / source_lab[magenta_iris, 1].mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = resolve_font("Monospace")
    masks = glyph_masks(font, list(DEFAULT_CHARS), CELL)
    structure_masks = glyph_masks(font, list(STRUCTURE_CHARS), CELL)
    results = {"settings": {"width": 960, "cell": CELL, "steps": STEPS,
                            "candidates": CANDIDATES, "seed": 0, "dither": 0.005,
                            "feature_threshold_oklab": 0.015}, "images": {}}
    for label, path, eye_box, iris_boxes in SOURCES:
        with Image.open(path) as original:
            size = (960, round(960 * original.height / original.width))
        reference = load_target(path, size, "fill", False)
        feature = _feature_mask(reference)
        directory = OUTPUT / label
        directory.mkdir(exist_ok=True)
        highlight = reference.copy() * 0.45
        highlight[feature] = [1.0, 0.1, 0.75]
        _image(highlight).save(directory / "selected_features.png")
        previews = {"reference": _image(reference)}
        previews["reference"].save(directory / "reference.png")
        image_results = {"source": str(path), "size": size, "eye_crop": eye_box,
                         "iris_boxes": iris_boxes, "selected_feature_fraction": float(feature.mean()),
                         "variants": {}}
        for key, title in VARIANTS[1:]:
            style = "structure" if key == "structure" else "headroom"
            bank = structure_masks if style == "structure" else masks
            grid = solve_style(reference, bank, style=style, invert_ink=False,
                               background="auto", seed=0, steps=STEPS,
                               candidates=CANDIDATES, dither=0.005,
                               structure_dither=0.06, device="cpu",
                               detail_loss=MODES[key][0] if key in MODES else "default",
                               feature_strength=MODES[key][1] if key in MODES else 1.0)
            rendered = _preview(grid, bank, size)
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            image_results["variants"][key] = {
                **_metrics(reference, rendered, eye_box, iris_boxes),
                **_feature_metrics(reference, rendered, feature, iris_boxes),
            }
            print(f"{label}: {title}", flush=True)
        _sheet(previews, directory / "comparison.png", variants=VARIANTS)
        _sheet(previews, directory / "eye_detail.png", eye_box, VARIANTS)
        _sheet(previews, directory / "eye_key_comparison.png", eye_box,
               [VARIANTS[index] for index in (0, 1, 3, 5)], columns=2)
        _full_image_sheet(previews, directory / "full_image_comparison.png")
        results["images"][label] = image_results
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
