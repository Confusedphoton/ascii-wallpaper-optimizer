"""Compare bracket additions using the production structure renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.tone import srgb_to_oklab
from experiment_utils import (CELL, EXPANDED_CHARS, SIZE, SOURCE, contact_sheet,
                              coverage_metrics, image, image_metrics, preview,
                              reset_runs, solve_style, write_report)


OUTPUT = Path("rendered/structure_brackets")


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    variants = [
        ("Selected · strong dither", "selected", EXPANDED_CHARS),
        ("Add [ ] { }", "brackets", EXPANDED_CHARS + "[]{}"),
        ("Add brackets + ( )", "brackets_parens", EXPANDED_CHARS + "[]{}()"),
        ("Brackets, no |", "no_pipe", EXPANDED_CHARS.replace("|", "") + "[]{}()"),
    ]
    panels = [("Source", image(source))]
    report = []
    for title, name, chars in variants:
        masks = glyph_masks(resolve_font("Monospace"), list(chars), CELL)
        grid = solve_style(source, masks, style="structure", invert_ink=False, background="auto",
                           seed=0, steps=1, candidates=1, dither=0, structure_dither=0.06, device="cpu")
        rendered = preview(grid, masks, SIZE)
        shown = image(rendered)
        shown.save(OUTPUT / f"{name}.png")
        panels.append((title, shown))
        counts = np.bincount(grid.glyph.ravel(), minlength=len(chars))
        count = counts.sum()
        def fraction(symbols):
            return float(sum(counts[chars.index(char)] for char in symbols if char in chars) / count)
        report.append({"name": name, "title": title, "chars": chars,
                       "pipe_fraction": fraction("|"), "bracket_fraction": fraction("[]{}"),
                       "parenthesis_fraction": fraction("()"),
                       "mean_display_lightness": float(srgb_to_oklab(rendered)[..., 0].mean()),
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
