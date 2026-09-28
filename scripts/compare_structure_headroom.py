"""Compare full-frame headroom boosting on the chosen structure-first glyph map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ascii_optimizer.assets import glyph_masks, linear_to_srgb, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.optimizer import Grid, _choose_background, render_grid
from compare_color_strategies import (
    brighten_structure, headroom_input, headroom_output, rgb_to_oklab,
    structure_first, to_image,
)
from compare_structure_glyphs import CELL, EXPANDED_CHARS, SIZE, SOURCE, sheet


OUTPUT = Path("rendered/structure_headroom")
CHARS = EXPANDED_CHARS + "[]{}"
OPTIONS = {"density_floor": 0.01, "density_span": 0.35,
           "density_exponent": 1.35, "dither_strength": 0.06, "seed": 0}
GAIN = 2.0


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    masks = glyph_masks(resolve_font("Monospace"), list(CHARS), CELL)
    linear_source = srgb_to_linear(torch.from_numpy(source)).permute(2, 0, 1)
    background = _choose_background(linear_source, "auto").numpy()
    base = structure_first(source, masks, background, **OPTIONS)
    chosen = brighten_structure(base, ink_gain=GAIN, background_lift=0)
    chosen_linear = render_grid(chosen, masks)[:SIZE[1], :SIZE[0]]

    # This is the exact output-side transform used in the headroom experiment.
    boosted_linear = headroom_output(chosen_linear, GAIN)

    # Precompress the one uniform background so applying the same output
    # transform restores its original dark shade. The glyph map stays fixed.
    background_srgb = linear_to_srgb(torch.from_numpy(chosen.background.reshape(1, 1, 3))).numpy()
    encoded_background = headroom_input(background_srgb, GAIN)
    solve_background = srgb_to_linear(torch.from_numpy(encoded_background)).numpy().reshape(3)
    once_grid = Grid(base.glyph.copy(), base.color.copy(), solve_background)
    once_linear = headroom_output(render_grid(once_grid, masks)[:SIZE[1], :SIZE[0]], GAIN)
    twice_grid = Grid(chosen.glyph.copy(), chosen.color.copy(), solve_background)
    twice_linear = headroom_output(render_grid(twice_grid, masks)[:SIZE[1], :SIZE[0]], GAIN)
    np.testing.assert_array_equal(chosen.glyph, once_grid.glyph)
    np.testing.assert_array_equal(chosen.glyph, twice_grid.glyph)

    previews: list[tuple[str, Image.Image]] = []
    variants = [
        ("Selected · brighter ink", "selected", chosen_linear),
        ("Full-image boost", "full_headroom", boosted_linear),
        ("Boost once · dark canvas", "pinned_once", once_linear),
        ("Boost twice · dark canvas", "pinned_twice", twice_linear),
    ]
    report = []
    for title, name, linear in variants:
        preview = to_image(linear)
        preview.save(OUTPUT / f"{name}.png")
        previews.append((title, preview))
        report.append({"name": name, "title": title,
                       "mean_display_lightness": float(rgb_to_oklab(linear)[..., 0].mean()),
                       "corner_lightness": float(rgb_to_oklab(linear[:32, :32])[..., 0].mean())})
    sheet(previews, OUTPUT / "comparison.png")
    sheet(previews, OUTPUT / "comparison_detail.png", (163, 65, 643, 427))
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUTPUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
