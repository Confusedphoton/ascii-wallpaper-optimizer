"""Compare production OKLab render styles and controlled color adjustments.

The structure color adjustments preserve glyph maps and use RGB8-representable
inks. All previews use the same composition and display transform as playback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS, STRUCTURE_CHARS
from ascii_optimizer.tone import srgb_to_oklab
from experiment_utils import (brighten_structure, contact_sheet, coverage_metrics,
                              image, image_metrics, preview, reset_runs,
                              solve_style, write_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=Path("screenshots/test_image.png"))
    parser.add_argument("--output", type=Path, default=Path("rendered/color_strategies"))
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--cell", default="8x16")
    parser.add_argument("--steps", type=int, default=36)
    args = parser.parse_args()
    size = tuple(map(int, args.size.split("x")))
    cell = tuple(map(int, args.cell.split("x")))
    if len(size) != 2 or len(cell) != 2 or min(*size, *cell, args.steps) < 1:
        parser.error("size, cell, and steps must be positive")
    reset_runs()
    args.output.mkdir(parents=True, exist_ok=True)
    source = load_target(args.image, size, "fill", False)
    font = resolve_font("Monospace")
    headroom_masks = glyph_masks(font, list(DEFAULT_CHARS), cell)
    structure_masks = glyph_masks(font, list(STRUCTURE_CHARS), cell)
    variants = [
        ("Headroom · current defaults", "headroom", "headroom", {}),
        ("Headroom · no lighter-ink reward", "no_bonus", "headroom", {"glyph_luminance_bonus": 0.0}),
        ("Headroom · no gradient objective", "no_gradient", "headroom", {"gradient_loss_strength": 0.0}),
        ("Structure · current defaults", "structure", "structure", {}),
        ("Structure · brighter ink", "structure_bright_ink", "structure", {}),
        ("Structure · ink + canvas", "structure_bright_canvas", "structure", {}),
        ("Headroom · inverted ink", "inverse_ink", "headroom", {"invert_ink": True}),
    ]
    panels = [("Source image", image(source))]
    report = []
    structure_grid = None
    for title, name, style, options in variants:
        print(f"Rendering {title}", flush=True)
        masks = structure_masks if style == "structure" else headroom_masks
        if name.startswith("structure_bright"):
            grid = brighten_structure(structure_grid, ink_gain=2.0,
                                      background_lift=0.12 if name.endswith("canvas") else 0.0)
            np.testing.assert_array_equal(grid.glyph, structure_grid.glyph)
        else:
            grid = solve_style(source, masks, style=style, invert_ink=options.get("invert_ink", False),
                               background="auto", seed=0, steps=args.steps, candidates=6,
                               dither=0.005, structure_dither=0.06, device="cpu",
                               **{key: value for key, value in options.items() if key != "invert_ink"})
            if name == "structure":
                structure_grid = grid
        rendered = preview(grid, masks, size, invert_ink=options.get("invert_ink", False))
        shown = image(rendered)
        shown.save(args.output / f"{name}.png")
        panels.append((title, shown))
        report.append({"name": name, "title": title, "style": style,
                       "options": options, "experiment": getattr(grid, "experiment", None),
                       "glyphs_match_original_structure": bool(name.startswith("structure_bright")),
                       "color_adjustment": {"ink_gain": 2.0, "background_lift": 0.12 if name.endswith("canvas") else 0.0}
                       if name.startswith("structure_bright") else None,
                       "mean_display_lightness": float(srgb_to_oklab(rendered)[..., 0].mean()),
                       **image_metrics(source, rendered), **coverage_metrics(grid, masks)})
    contact_sheet(panels, args.output / "comparison.png", panel=(max(1, size[0] // 2), max(1, size[1] // 2)))
    contact_sheet(panels, args.output / "comparison_detail.png",
                  crop=(size[0] * 17 // 100, size[1] * 12 // 100,
                        size[0] * 67 // 100, size[1] * 79 // 100))
    write_report(args.output / "comparison.json", {"source": str(args.image), "size": size,
                 "cell": cell, "steps": args.steps, "variants": report})
    print(f"Wrote {args.output / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
