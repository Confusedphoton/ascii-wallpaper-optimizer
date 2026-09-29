"""Compare the glyph lightness bonus with a display-space saturation loss.

Run from the repository root with:
    .venv/bin/python scripts/compare_glyph_saturation_loss.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import compare_glyph_luminance_bonus as base
from ascii_optimizer.assets import srgb_to_linear
from ascii_optimizer.tone import rgb_to_oklab


OUTPUT = Path("rendered/glyph_luminance_saturation")
SATURATION_LOSS = 20.0
VARIANTS = (
    ("current", "Current β=0", 0.0, 0.0),
    ("beta_0_02", "β=.02", 0.02, 0.0),
    ("beta_0_02_sat", "β=.02 + sat loss", 0.02, None),
    ("beta_0_08", "β=.08", 0.08, 0.0),
    ("beta_0_08_sat", "β=.08 + sat loss", 0.08, None),
    ("beta_0_12", "β=.12", 0.12, 0.0),
    ("beta_0_12_sat", "β=.12 + sat loss", 0.12, None),
)
SNOW_BELT = (365, 215, 480, 310)


def _red_belt_metrics(reference: np.ndarray, rendered: np.ndarray) -> dict:
    x1, y1, x2, y2 = SNOW_BELT
    source = reference[y1:y2, x1:x2]
    output = rendered[y1:y2, x1:x2]
    red, green, blue = np.moveaxis(source, -1, 0)
    red_pixels = (red > 0.25) & (red > 1.25 * green) & (red > 1.3 * blue)
    source_lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(source)).numpy())
    output_lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(output)).numpy())
    source_chroma = np.linalg.norm(source_lab[..., 1:], axis=-1)
    output_chroma = np.linalg.norm(output_lab[..., 1:], axis=-1)
    return {
        "red_belt_sample_pixels": int(red_pixels.sum()),
        "red_belt_target_mean_chroma": float(source_chroma[red_pixels].mean()),
        "red_belt_output_mean_chroma": float(output_chroma[red_pixels].mean()),
        "red_belt_chroma_retention": float(output_chroma[red_pixels].mean()
                                            / source_chroma[red_pixels].mean()),
        "red_belt_chroma_rmse": float(np.sqrt(np.mean(
            (source_lab[red_pixels, 1:] - output_lab[red_pixels, 1:]) ** 2))),
    }


def _sheet(previews: dict[str, Image.Image], path: Path,
           crop: tuple[int, int, int, int] | None = None, zoom: int = 1) -> None:
    keys = ["reference", *(key for key, _, _, _ in VARIANTS)]
    width, height = previews["reference"].size
    gap, heading, columns = 14, 38, 3
    panel = ((width, height) if crop is None else
             ((crop[2] - crop[0]) * zoom, (crop[3] - crop[1]) * zoom))
    rows = (len(keys) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel[0] + (columns + 1) * gap,
                               rows * (panel[1] + heading) + (rows + 1) * gap), "#191b21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(base.resolve_font("DejaVu Sans")), 19)
    labels = {key: label for key, label, _, _ in VARIANTS}
    for index, key in enumerate(keys):
        x = gap + index % columns * (panel[0] + gap)
        y = gap + index // columns * (panel[1] + heading + gap)
        draw.text((x, y), "Source" if key == "reference" else labels[key],
                  fill="white", font=font)
        image = previews[key]
        if crop is not None:
            image = image.crop(crop).resize(panel, Image.Resampling.NEAREST)
        sheet.paste(image, (x, y + heading))
    sheet.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    masks = base.glyph_masks(base.resolve_font("Monospace"), list(base.DEFAULT_CHARS), base.CELL)
    results = {
        "settings": {
            "width": base.WIDTH,
            "cell": base.CELL,
            "steps": base.STEPS,
            "candidates": base.CANDIDATES,
            "seed": base.SEED,
            "dither": base.DITHER,
            "default_saturation_loss_strength_for_nonzero_bonus": SATURATION_LOSS,
            "saturation_loss": "weighted squared deficit in displayed OKLab chroma plus out-of-gamut headroom penalty",
            "variants": [{"key": key, "label": label, "glyph_luminance_bonus": bonus,
                          "saturation_loss_strength": saturation}
                         for key, label, bonus, saturation in VARIANTS],
        },
        "images": {},
    }
    for label, source, focus_crop in base.SOURCES:
        with Image.open(source) as original:
            size = (base.WIDTH, round(base.WIDTH * original.height / original.width))
        reference = base.load_target(source, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(parents=True, exist_ok=True)
        previews = {"reference": base._image(reference)}
        previews["reference"].save(directory / "reference.png")
        metrics = {}
        current_grid = None
        for key, title, bonus, saturation in VARIANTS:
            print(f"{label}: {title}", flush=True)
            effective_saturation = (SATURATION_LOSS if bonus > 0 else 0.0) if saturation is None else saturation
            grid = base.solve_style(
                reference, masks, style="headroom", invert_ink=False,
                background="auto", seed=base.SEED, steps=base.STEPS,
                candidates=base.CANDIDATES, dither=base.DITHER,
                structure_dither=0.06, device="cpu",
                glyph_luminance_bonus=bonus,
                saturation_loss_strength=saturation,
            )
            if key == "current":
                current_grid = grid
            rendered = base._preview(grid, masks, size)
            previews[key] = base._image(rendered)
            previews[key].save(directory / f"{key}.png")
            variant_metrics = {
                "glyph_luminance_bonus": bonus,
                "saturation_loss_strength": effective_saturation,
                **base._image_metrics(reference, rendered),
                **base._glyph_metrics(grid, masks, base.DEFAULT_CHARS,
                                      None if key == "current" else current_grid.glyph),
            }
            if label == "snow":
                variant_metrics.update(_red_belt_metrics(reference, rendered))
            metrics[key] = variant_metrics
        _sheet(previews, directory / "comparison.png")
        actual_focus = SNOW_BELT if label == "snow" else focus_crop
        _sheet(previews, directory / "focus_comparison.png", actual_focus,
               zoom=3 if label == "snow" else 1)
        results["images"][label] = {
            "source": str(source),
            "size": size,
            "focus_crop": actual_focus,
            "variants": metrics,
        }
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
