"""Compare glyph palettes and density curves for the structure-first preview."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.cli import DEFAULT_CHARS
from ascii_optimizer.optimizer import _choose_background, render_grid
from compare_color_strategies import brighten_structure, rgb_to_oklab, structure_first, to_image


SOURCE = Path("screenshots/test_image.png")
OUTPUT = Path("rendered/structure_glyphs")
SIZE = (960, 540)
CELL = (8, 16)
EXPANDED_CHARS = " .´:-_/|+rszunvcxLOGDQ0KNMW&@"


def sheet(previews: list[tuple[str, Image.Image]], path: Path,
          crop: tuple[int, int, int, int] | None = None) -> None:
    panel = (480, 270) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    margin, heading = 18, 44
    rows = (len(previews) + 1) // 2
    result = Image.new("RGB", (2 * (panel[0] + margin) + margin,
                               rows * (panel[1] + heading + margin) + margin), "#15171d")
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 20)
    for index, (title, image) in enumerate(previews):
        x = margin + index % 2 * (panel[0] + margin)
        y = margin + index // 2 * (panel[1] + heading + margin)
        draw.text((x, y), title, fill="white", font=font)
        shown = image.resize(panel, Image.Resampling.LANCZOS) if crop is None else image.crop(crop)
        result.paste(shown, (x, y + heading))
    result.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_target(SOURCE, SIZE, "fill", False)
    linear = srgb_to_linear(torch.from_numpy(source)).permute(2, 0, 1)
    background = _choose_background(linear, "auto").numpy()
    specs = [
        ("Original structure", "original", DEFAULT_CHARS, {}),
        ("Add mid-density glyphs", "expanded", EXPANDED_CHARS, {}),
        ("Mid-density tone curve", "expanded_curve", EXPANDED_CHARS,
         {"density_floor": 0.01, "density_span": 0.35, "density_exponent": 1.35}),
        ("Tone curve + subtle dither", "expanded_dither_subtle", EXPANDED_CHARS,
         {"density_floor": 0.01, "density_span": 0.35, "density_exponent": 1.35,
          "dither_strength": 0.03, "seed": 0}),
        ("Tone curve + stronger dither", "expanded_dither_strong", EXPANDED_CHARS,
         {"density_floor": 0.01, "density_span": 0.35, "density_exponent": 1.35,
          "dither_strength": 0.06, "seed": 0}),
    ]
    previews = [("Source", Image.fromarray(np.rint(source * 255).astype(np.uint8), "RGB"))]
    report = []
    for title, name, chars, options in specs:
        masks = glyph_masks(resolve_font("Monospace"), list(chars), CELL)
        grid = structure_first(source, masks, background, **options)
        grid = brighten_structure(grid, ink_gain=2.0, background_lift=0)
        rendered = render_grid(grid, masks)[:SIZE[1], :SIZE[0]]
        image = to_image(rendered)
        image.save(OUTPUT / f"{name}.png")
        previews.append((title, image))
        counts = np.bincount(grid.glyph.ravel(), minlength=len(chars))
        coverage = masks.mean((1, 2))
        report.append({
            "name": name, "title": title, "chars": chars, "curve": options,
            "mean_display_lightness": float(rgb_to_oklab(rendered)[..., 0].mean()),
            "low_coverage_fraction": float(counts[coverage < 0.08].sum() / counts.sum()),
            "intermediate_fraction": float(counts[(coverage >= 0.08) & (coverage < 0.18)].sum() / counts.sum()),
            "blank_fraction": float(counts[chars.index(" ")] / counts.sum()),
            "top_glyphs": [[chars[i], int(counts[i])] for i in np.argsort(-counts)[:8]],
        })
    sheet(previews, OUTPUT / "comparison.png")
    sheet(previews, OUTPUT / "comparison_detail.png", (163, 65, 643, 427))
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUTPUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
