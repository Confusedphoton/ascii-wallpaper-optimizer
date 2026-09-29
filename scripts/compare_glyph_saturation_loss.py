"""Compare the glyph lightness bonus with a display-space saturation loss.

Run from the repository root with:
    .venv/bin/python scripts/compare_glyph_saturation_loss.py
"""

from __future__ import annotations

from pathlib import Path


import experiment_utils as base
from experiment_utils import SNOW_BELT, red_chroma_metrics as _red_belt_metrics


OUTPUT = Path("rendered/glyph_luminance_saturation")
SATURATION_LOSS = 20.0
VARIANTS = (
    ("current", "Baseline β=0", 0.0, 0.0),
    ("beta_0_02", "β=.02", 0.02, 0.0),
    ("beta_0_02_sat", "β=.02 + sat loss", 0.02, None),
    ("beta_0_08", "β=.08", 0.08, 0.0),
    ("beta_0_08_sat", "β=.08 + sat loss", 0.08, None),
    ("beta_0_12", "β=.12", 0.12, 0.0),
    ("beta_0_12_sat", "β=.12 + sat loss", 0.12, None),
)


def _sheet(previews, path, crop=None, zoom=1):
    labels = [("reference", "Source"), *((key, title) for key, title, _, _ in VARIANTS)]
    base.contact_sheet([(title, previews[key]) for key, title in labels], path, crop=crop, zoom=zoom)


def main() -> None:
    base.reset_runs()
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
            "saturation_loss": "weighted source-hue chroma deficit and perpendicular chroma error in displayed OKLab with RGB8-constrained ink",
            "variants": [{"key": key, "label": label, "glyph_luminance_bonus": bonus,
                          "saturation_loss_strength": saturation}
                         for key, label, bonus, saturation in VARIANTS],
        },
        "images": {},
    }
    for label, source, focus_crop in base.SOURCES:
        size = base.source_size(source, base.WIDTH)
        reference = base.load_target(source, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(parents=True, exist_ok=True)
        previews = {"reference": base.image(reference)}
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
            rendered = base.preview(grid, masks, size)
            previews[key] = base.image(rendered)
            previews[key].save(directory / f"{key}.png")
            variant_metrics = {
                "glyph_luminance_bonus": bonus,
                "experiment": grid.experiment,
                "saturation_loss_strength": effective_saturation,
                **base.image_metrics(reference, rendered),
                **base.glyph_metrics(grid, masks, base.DEFAULT_CHARS,
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
    base.write_report(OUTPUT / "metrics.json", results)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
