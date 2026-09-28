"""Compare bracket additions to the selected structure-first strong-dither style."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.optimizer import _choose_background, render_grid
from compare_color_strategies import brighten_structure, rgb_to_oklab, structure_first, to_image
from compare_structure_glyphs import CELL, EXPANDED_CHARS, SIZE, SOURCE, sheet


OUTPUT = Path("rendered/structure_brackets")
OPTIONS = {"density_floor": 0.01, "density_span": 0.35,
           "density_exponent": 1.35, "dither_strength": 0.06, "seed": 0}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    linear = srgb_to_linear(torch.from_numpy(source)).permute(2, 0, 1)
    background = _choose_background(linear, "auto").numpy()
    variants = [
        ("Selected · strong dither", "selected", EXPANDED_CHARS),
        ("Add [ ] { }", "brackets", EXPANDED_CHARS + "[]{}"),
        ("Add brackets + ( )", "brackets_parens", EXPANDED_CHARS + "[]{}()"),
        ("Brackets, no |", "no_pipe", EXPANDED_CHARS.replace("|", "") + "[]{}()"),
    ]
    previews: list[tuple[str, Image.Image]] = []
    report = []
    for title, name, chars in variants:
        masks = glyph_masks(resolve_font("Monospace"), list(chars), CELL)
        grid = structure_first(source, masks, background, **OPTIONS)
        grid = brighten_structure(grid, ink_gain=2.0, background_lift=0)
        rendered = render_grid(grid, masks)[:SIZE[1], :SIZE[0]]
        preview = to_image(rendered)
        preview.save(OUTPUT / f"{name}.png")
        previews.append((title, preview))
        counts = np.bincount(grid.glyph.ravel(), minlength=len(chars))
        count = counts.sum()
        fraction = lambda symbols: float(sum(counts[chars.index(char)] for char in symbols if char in chars) / count)
        report.append({
            "name": name, "title": title, "chars": chars,
            "pipe_fraction": fraction("|"), "bracket_fraction": fraction("[]{}"),
            "parenthesis_fraction": fraction("()"),
            "mean_display_lightness": float(rgb_to_oklab(rendered)[..., 0].mean()),
        })
    sheet(previews, OUTPUT / "comparison.png")
    sheet(previews, OUTPUT / "comparison_detail.png", (163, 65, 643, 427))
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUTPUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
