"""Compare Sobel gradient-loss amplitudes on the screenshot test images.

Run from the repository root with:
    .venv/bin/python scripts/compare_gradient_loss.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import experiment_utils as base
from ascii_optimizer.assets import glyph_masks, resolve_font
from experiment_utils import SNOW_BELT, red_chroma_metrics as _red_belt_metrics, preview as _preview


OUTPUT = Path("rendered/gradient_loss_default_sweep")
BONUS = 0.12
GRADIENT_STRENGTHS = (0.0, 0.5, 1.0, 1.5, 2.0, 4.0)


def _edge_metrics(reference: np.ndarray, rendered: np.ndarray) -> dict:
    source_x, source_y = base.gradients(reference)
    output_x, output_y = base.gradients(rendered)
    source_magnitude = np.sqrt(np.square(source_x).sum(-1) + np.square(source_y).sum(-1))
    output_magnitude = np.sqrt(np.square(output_x).sum(-1) + np.square(output_y).sum(-1))
    edge_mask = source_magnitude >= np.quantile(source_magnitude, 0.85)
    delta_x, delta_y = output_x - source_x, output_y - source_y
    squared_error = np.square(delta_x).sum(-1) + np.square(delta_y).sum(-1)
    return {
        "edge_gradient_rmse_x100": float(np.sqrt(squared_error.mean() / 6) * 100),
        "source_edge_gradient_rmse_x100": float(np.sqrt(squared_error[edge_mask].mean() / 6) * 100),
        "source_edge_magnitude_mae_x100": float(np.abs(
            output_magnitude[edge_mask] - source_magnitude[edge_mask]).mean() * 100),
        "source_edge_magnitude_retention": float(
            output_magnitude[edge_mask].mean() / max(source_magnitude[edge_mask].mean(), 1e-8)),
        "source_edge_pixel_fraction": float(edge_mask.mean()),
    }


def _sheet(previews, labels, path, crop=None, zoom=1):
    items = [("Source", previews["reference"]), *((title, previews[key]) for key, title in labels.items())]
    base.contact_sheet(items, path, crop=crop, zoom=zoom)


def main() -> None:
    base.reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chars = base.DEFAULT_CHARS
    masks = glyph_masks(resolve_font("Monospace"), list(chars), base.CELL)
    labels = {f"strength_{strength:g}": f"β=.12, gradient={strength:g}"
              for strength in GRADIENT_STRENGTHS}
    results = {
        "settings": {
            "width": base.WIDTH,
            "cell": base.CELL,
            "steps": base.STEPS,
            "candidates": base.CANDIDATES,
            "seed": base.SEED,
            "glyphs": chars,
            "glyph_luminance_bonus": BONUS,
            "saturation_loss_strength": 20.0,
            "gradient_loss": "Sobel gradients of displayed OKLab L,a,b, squared oriented-response error",
            "gradient_strengths": GRADIENT_STRENGTHS,
        },
        "images": {},
    }
    for image_name, source, focus_crop in base.SOURCES:
        size = base.source_size(source, base.WIDTH)
        reference = base.load_target(source, size, "fill", False)
        directory = OUTPUT / image_name
        directory.mkdir(parents=True, exist_ok=True)
        previews = {"reference": base.image(reference)}
        previews["reference"].save(directory / "reference.png")
        variants = {}
        baseline_grid = None
        for strength in GRADIENT_STRENGTHS:
            key = f"strength_{strength:g}"
            print(f"{image_name}: gradient loss strength {strength:g}", flush=True)
            grid = base.solve_style(
                reference, masks, style="headroom", invert_ink=False,
                background="auto", seed=base.SEED, steps=base.STEPS,
                candidates=base.CANDIDATES, dither=base.DITHER,
                structure_dither=0.06, device="cpu",
                glyph_luminance_bonus=BONUS,
                gradient_loss_strength=strength,
            )
            if strength == 0:
                baseline_grid = grid
            rendered = _preview(grid, masks, size)
            previews[key] = base.image(rendered)
            previews[key].save(directory / f"{key}.png")
            variant_metrics = {
                "gradient_loss_strength": strength,
                "experiment": grid.experiment,
                **base.image_metrics(reference, rendered),
                **_edge_metrics(reference, rendered),
                **base.glyph_metrics(grid, masks, chars,
                                      None if strength == 0 else baseline_grid.glyph),
                "added_glyph_shares": {
                    char: float(np.mean(grid.glyph == chars.index(char))) for char in "\\=^"
                },
            }
            if image_name == "snow":
                variant_metrics.update(_red_belt_metrics(reference, rendered))
            variants[key] = variant_metrics
        assert baseline_grid is not None
        _sheet(previews, labels, directory / "comparison.png")
        actual_focus = SNOW_BELT if image_name == "snow" else focus_crop
        _sheet(previews, labels, directory / "focus_comparison.png", actual_focus,
               zoom=3 if image_name == "snow" else 2)
        results["images"][image_name] = {
            "source": str(source),
            "size": size,
            "focus_crop": actual_focus,
            "variants": variants,
        }
    base.write_report(OUTPUT / "metrics.json", results)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
