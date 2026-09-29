"""Compare glyph palettes and dither using the production structure renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS, STRUCTURE_CHARS
from ascii_optimizer.tone import srgb_to_oklab
from experiment_utils import (CELL, EXPANDED_CHARS, SIZE, SOURCE, contact_sheet,
                              coverage_metrics, image, image_metrics, preview,
                              reset_runs, solve_style, write_report)


OUTPUT = Path("rendered/structure_glyphs")


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    specs = [
        ("Default headroom palette", "original", DEFAULT_CHARS, 0.0),
        ("Expanded mid-density palette", "expanded", EXPANDED_CHARS, 0.0),
        ("Production structure palette", "production", STRUCTURE_CHARS, 0.0),
        ("Production · subtle dither", "dither_subtle", STRUCTURE_CHARS, 0.03),
        ("Production · stronger dither", "dither_strong", STRUCTURE_CHARS, 0.06),
    ]
    panels = [("Source", image(source))]
    report = []
    for title, name, chars, dither in specs:
        masks = glyph_masks(resolve_font("Monospace"), list(chars), CELL)
        grid = solve_style(source, masks, style="structure", invert_ink=False, background="auto",
                           seed=0, steps=1, candidates=1, dither=0, structure_dither=dither, device="cpu")
        rendered = preview(grid, masks, SIZE)
        shown = image(rendered)
        shown.save(OUTPUT / f"{name}.png")
        panels.append((title, shown))
        counts = np.bincount(grid.glyph.ravel(), minlength=len(chars))
        coverage = masks.mean((1, 2))
        report.append({"name": name, "title": title, "chars": chars, "structure_dither": dither,
                       "mean_display_lightness": float(srgb_to_oklab(rendered)[..., 0].mean()),
                       "low_coverage_fraction": float(counts[coverage < 0.08].sum() / counts.sum()),
                       "intermediate_fraction": float(counts[(coverage >= 0.08) & (coverage < 0.18)].sum() / counts.sum()),
                       "blank_fraction": float(counts[chars.index(" ")] / counts.sum()),
                       "top_glyphs": [[chars[i], int(counts[i])] for i in np.argsort(-counts)[:8]],
                       "experiment": grid.experiment,
                       **image_metrics(source, rendered), **coverage_metrics(grid, masks)})
    contact_sheet(panels, OUTPUT / "comparison.png", columns=2,
                  panel=(max(1, SIZE[0] // 2), max(1, SIZE[1] // 2)))
    contact_sheet(panels, OUTPUT / "comparison_detail.png", columns=2,
                  crop=(SIZE[0] * 17 // 100, SIZE[1] * 12 // 100, SIZE[0] * 67 // 100, SIZE[1] * 79 // 100))
    write_report(OUTPUT / "comparison.json", {"source": str(SOURCE), "size": SIZE, "cell": CELL, "variants": report})
    print(f"Wrote {OUTPUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
