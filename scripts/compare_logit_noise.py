"""Compare bounded blue-noise and IID Gumbel-Max on optimized glyph logits.

Run from the repository root with:
    .venv/bin/python scripts/compare_logit_noise.py

Gumbel values are sampled from independent PyTorch uniforms for each cell and
candidate, using the inverse CDF rather than a softmax helper. A Gumbel scale of
1 is the exact Gumbel-Max sample from softmax(logits).
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_optimizer.assets import glyph_masks, load_target, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS
from experiment_utils import (coverage_metrics, image_metrics, preview as _render,
                              image as _image, reset_runs, source_size, solve_style, write_report, gradient_rmse as _gradient_rmse)
from ascii_optimizer.optimizer import DITHER_TO_LOGIT_SCALE


OUTPUT = Path("rendered/logit_noise")
SOURCES = [
    ("test_image", Path("screenshots/test_image.png")),
    ("tree", Path("screenshots/149254100_p0-被树傀儡抓住惹.jpg")),
    ("snow", Path("screenshots/149642007_p0-无题.jpg")),
    ("viviana_gradient", Path("screenshots/143381492_p0-Viviana.jpg")),
]
WIDTH = 720
CELL = (8, 16)
STEPS = 24
CANDIDATES = 6
SEED = 0
DITHER_LEVELS = (0.0025, 0.005, 0.01, 0.02)
PANEL_WIDTH = 480
GUMBEL_STD = np.pi / np.sqrt(6)


def _metrics(reference, rendered, grid, baseline_grid, masks):
    return {**image_metrics(reference, rendered), **coverage_metrics(grid, masks),
            "oklab_gradient_rmse_x100": _gradient_rmse(reference, rendered),
            "glyph_change_fraction_vs_no_noise": float((grid.glyph != baseline_grid.glyph).mean())}


def _fit_panel(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _comparison_sheet(previews: dict[str, Image.Image], panel_titles: dict[str, str],
                      path: Path) -> None:
    methods = (
        ("uniform", "Bounded blue-noise uniform (plus matched Max control)",
         [f"uniform_{level:g}" for level in DITHER_LEVELS] + ["uniform_max_match"]),
        ("gumbel", "IID Gumbel (matched sweep plus exact Gumbel-Max)",
         [f"gumbel_matched_{level:g}" for level in DITHER_LEVELS] + ["gumbel_max"]),
    )
    panel_images = {key: _fit_panel(image, PANEL_WIDTH) for key, image in previews.items()}
    panel_height = panel_images["reference"].height
    columns = 7
    margin, title_height, row_title_height = 14, 28, 24
    row_height = row_title_height + title_height + panel_height
    sheet = Image.new(
        "RGB",
        (columns * PANEL_WIDTH + (columns + 1) * margin,
         2 * row_height + 3 * margin),
        "#191b21",
    )
    draw = ImageDraw.Draw(sheet)
    font_path = str(resolve_font("DejaVu Sans"))
    title_font = ImageFont.truetype(font_path, 16)
    row_font = ImageFont.truetype(font_path, 18)
    for row, (method, heading, keys) in enumerate(methods):
        y = margin + row * (row_height + margin)
        draw.text((margin, y), heading, fill="white", font=row_font)
        y += row_title_height
        items = [("reference", "Source"), ("baseline", "No noise")]
        items.extend((key, panel_titles[key]) for key in keys)
        for column, (key, title) in enumerate(items):
            x = margin + column * (PANEL_WIDTH + margin)
            draw.text((x, y), title, fill="#d8dbe3", font=title_font)
            sheet.paste(panel_images[key], (x, y + title_height))
    sheet.save(path)


def _html_report(results: dict) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Post-logit noise comparison</title>",
        "<style>body{font:15px system-ui;background:#17191f;color:#eee;margin:2rem}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse;margin:1rem 0 2rem}"
        "td,th{padding:.4rem .7rem;border:1px solid #555;text-align:right}"
        "th:first-child,td:first-child{text-align:left}section{margin:3rem 0}</style>",
        "<h1>Post-optimization logit noise comparison</h1>",
        f"<p>{WIDTH}px wide; {CELL[1]}×{CELL[0]} cells; {STEPS} optimizer steps; "
        f"all {len(DEFAULT_CHARS)} glyphs per cell; seed {SEED}. The dither setting is "
        f"multiplied by {DITHER_TO_LOGIT_SCALE:g}. Uniform panels use bounded "
        "blue-noise ranks. Gumbel panels use IID torch.rand samples transformed by "
        "-log(-log(u)). The amplitude sweep pairs each uniform panel with an IID "
        "Gumbel panel at the same noise standard deviation. The final pair compares "
        "exact scale-1 Gumbel-Max sampling with a variance-matched blue-noise uniform "
        "control. Each noisy argmax is followed by deterministic full-frame glyph "
        "refinement.</p>",
    ]
    for label, image in results["images"].items():
        parts.append(f"<section><h2>{html.escape(label)}</h2>")
        parts.append(f"<p>Source: {html.escape(image['source'])}</p>")
        parts.append(f"<a href='{html.escape(label)}/comparison.png'><img src='{html.escape(label)}/comparison.png'></a>")
        parts.append("<table><thead><tr><th>Variant</th><th>Mean ΔE OK ×100</th>"
                     "<th>P90 ΔE OK ×100</th><th>Gradient RMSE ×100</th>"
                     "<th>Glyph changes</th><th>Mean coverage</th><th>Dense cells</th>"
                     "</tr></thead><tbody>")
        for variant in image["variants"].values():
            metrics = variant["metrics"]
            parts.append(
                f"<tr><td>{html.escape(variant['title'])}</td>"
                f"<td>{metrics['mean_oklab_delta_e_x100']:.3f}</td>"
                f"<td>{metrics['p90_oklab_delta_e_x100']:.3f}</td>"
                f"<td>{metrics['oklab_gradient_rmse_x100']:.3f}</td>"
                f"<td>{metrics['glyph_change_fraction_vs_no_noise']:.1%}</td>"
                f"<td>{metrics['mean_glyph_coverage']:.3f}</td>"
                f"<td>{metrics['dense_glyph_fraction']:.1%}</td></tr>"
            )
        parts.append("</tbody></table></section>")
    return "\n".join(parts) + "\n"


def main() -> None:
    reset_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    masks = glyph_masks(resolve_font("Monospace"), list(DEFAULT_CHARS), CELL)
    settings = {
        "width": WIDTH,
        "cell": list(CELL),
        "steps": STEPS,
        "glyphs_per_cell": len(DEFAULT_CHARS),
        "seed": SEED,
        "glyphs": DEFAULT_CHARS,
        "dither_levels": list(DITHER_LEVELS),
        "dither_to_logit_scale": DITHER_TO_LOGIT_SCALE,
        "gumbel_transform": "IID torch.rand per cell/candidate; -torch.log(-torch.log(u))",
        "gumbel_scale_1": "exact Gumbel-Max argmax sample from softmax(logits)",
        "matched_gumbel_scale_from_uniform_scale": "uniform scale / (pi * sqrt(2))",
        "sources": [str(path) for _, path in SOURCES],
    }
    results = {"settings": settings, "images": {}}
    variants = [("baseline", "No noise", "uniform", 0.0)]
    panel_titles = {"baseline": "No noise"}
    for level in DITHER_LEVELS:
        uniform_scale = level * DITHER_TO_LOGIT_SCALE
        uniform_key = f"uniform_{level:g}"
        uniform_title = (f"Uniform scale {uniform_scale:g} · "
                         f"SD {uniform_scale / np.sqrt(12):.3f}")
        variants.append((uniform_key, uniform_title, "uniform", level))
        panel_titles[uniform_key] = uniform_title

        gumbel_scale = uniform_scale / (np.pi * np.sqrt(2))
        gumbel_dither = gumbel_scale / DITHER_TO_LOGIT_SCALE
        gumbel_key = f"gumbel_matched_{level:g}"
        gumbel_title = (f"Gumbel scale {gumbel_scale:.3f} · "
                        f"SD {gumbel_scale * GUMBEL_STD:.3f}")
        variants.append((gumbel_key, gumbel_title, "gumbel", gumbel_dither))
        panel_titles[gumbel_key] = gumbel_title

    max_scale = GUMBEL_STD * np.sqrt(12)
    uniform_max_dither = max_scale / DITHER_TO_LOGIT_SCALE
    variants.append(("uniform_max_match", "Blue-noise uniform matched to Gumbel-Max",
                     "uniform", uniform_max_dither))
    panel_titles["uniform_max_match"] = f"Uniform · SD {GUMBEL_STD:.3f} · matched Max"
    variants.append(("gumbel_max", "IID Gumbel-Max exact, scale 1", "gumbel",
                     1.0 / DITHER_TO_LOGIT_SCALE))
    panel_titles["gumbel_max"] = f"Exact Gumbel-Max · scale 1 · SD {GUMBEL_STD:.3f}"

    for label, source_path in SOURCES:
        size = source_size(source_path, WIDTH)
        reference = load_target(source_path, size, "fill", False)
        directory = OUTPUT / label
        directory.mkdir(parents=True, exist_ok=True)
        for level in DITHER_LEVELS:
            (directory / f"gumbel_{level:g}.png").unlink(missing_ok=True)
        previews = {"reference": _image(reference)}
        previews["reference"].save(directory / "reference.png")
        image_results = {"source": str(source_path), "size": list(size), "variants": {}}
        grids = {}
        rendered_arrays = {}
        for key, title, distribution, level in variants:
            print(f"{label}: solving {title}", flush=True)
            grid = solve_style(
                reference, masks, style="headroom", invert_ink=False,
                background="auto", seed=SEED, steps=STEPS, candidates=CANDIDATES,
                dither=level, logit_noise_distribution=distribution,
                structure_dither=0.06, device="cpu",
            )
            grids[key] = grid
            rendered = _render(grid, masks, size)
            rendered_arrays[key] = rendered
            previews[key] = _image(rendered)
            previews[key].save(directory / f"{key}.png")
            noise_scale = level * DITHER_TO_LOGIT_SCALE
            noise_std = (noise_scale / np.sqrt(12) if distribution == "uniform"
                         else noise_scale * GUMBEL_STD)
            image_results["variants"][key] = {
                "title": title,
                "experiment": grid.experiment,
                "distribution": distribution,
                "dither_level": level,
                "logit_multiplier": noise_scale,
                "logit_noise_std": noise_std,
                "exact_gumbel_max": bool(distribution == "gumbel" and np.isclose(noise_scale, 1.0)),
                "metrics": {},
            }
        baseline = grids["baseline"]
        for key, _, _, _ in variants:
            image_results["variants"][key]["metrics"] = _metrics(
                reference, rendered_arrays[key], grids[key], baseline, masks,
            )
        _comparison_sheet(previews, panel_titles, directory / "comparison.png")
        results["images"][label] = image_results
        print(f"{label}: wrote {directory}", flush=True)

    write_report(OUTPUT / "metrics.json", results)
    (OUTPUT / "index.html").write_text(_html_report(results))
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
