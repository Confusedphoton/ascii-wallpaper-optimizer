"""Compare display transforms on a fixed production structure glyph map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import STRUCTURE_CHARS
from ascii_optimizer.optimizer import Grid
from ascii_optimizer.tone import encode_headroom_lab, srgb_to_oklab
from experiment_utils import (CELL, SIZE, SOURCE, brighten_structure, contact_sheet,
                              image, image_metrics, preview, reset_runs, solve_style, write_report)


OUTPUT = Path("rendered/structure_headroom")
CHARS = STRUCTURE_CHARS
GAIN = 2.0


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    masks = glyph_masks(resolve_font("Monospace"), list(CHARS), CELL)
    base = solve_style(source, masks, style="structure", invert_ink=False, background="auto",
                       seed=0, steps=1, candidates=1, dither=0, structure_dither=0.06, device="cpu")
    chosen = brighten_structure(base, ink_gain=GAIN)
    compressed_background = encode_headroom_lab(base.background.reshape(1, 1, 3))[0, 0]
    pinned = Grid(base.glyph.copy(), base.color.copy(), compressed_background)
    pinned_bright = Grid(chosen.glyph.copy(), chosen.color.copy(), compressed_background)
    variants = [
        ("Production structure display", "production", base, True),
        ("Brighter ink · direct display", "selected", chosen, False),
        ("Brighter ink + full display boost", "full_headroom", chosen, True),
        ("Pinned canvas · one display boost", "pinned_once", pinned, True),
        ("Pinned canvas · brighter ink + boost", "pinned_twice", pinned_bright, True),
    ]
    panels = [("Source", image(source))]
    report = []
    for title, name, grid, transform in variants:
        np.testing.assert_array_equal(grid.glyph, base.glyph)
        rendered = preview(grid, masks, SIZE, display_transform=transform)
        shown = image(rendered)
        shown.save(OUTPUT / f"{name}.png")
        panels.append((title, shown))
        lab = srgb_to_oklab(rendered)
        report.append({"name": name, "title": title, "glyphs_match_production": True,
                       "display_transform": transform, "ink_gain": GAIN if grid is chosen or grid is pinned_bright else 0.0,
                       "pinned_background": grid is pinned or grid is pinned_bright,
                       "mean_display_lightness": float(lab[..., 0].mean()),
                       "corner_lightness": float(lab[:32, :32, 0].mean()), **image_metrics(source, rendered)})
    contact_sheet(panels, OUTPUT / "comparison.png", columns=2,
                  panel=(max(1, SIZE[0] // 2), max(1, SIZE[1] // 2)))
    contact_sheet(panels, OUTPUT / "comparison_detail.png", columns=2,
                  crop=(SIZE[0] * 17 // 100, SIZE[1] * 12 // 100, SIZE[0] * 67 // 100, SIZE[1] * 79 // 100))
    write_report(OUTPUT / "comparison.json", {"source": str(SOURCE), "size": SIZE, "cell": CELL,
                 "base_experiment": base.experiment, "variants": report,
                 "note": "Direct display and repeated/pinned boosts are experimental previews; production uses one full display transform."})
    print(f"Wrote {OUTPUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
