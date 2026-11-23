"""Compare v6_curves & v7_balanced vs v8 SDF-interval coverage (unlocks directional aurora/glacier strokes while preventing belly voids!).

Run:
  PYTHONPATH=. .venv/bin/python -m ascii_optimizer.launcher experiments/compare_stroke_graph.py --device cuda
  .venv/bin/python scripts/view_renders.py --report rendered/stroke_graph_v8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_utils as utils
from ascii_optimizer.renderers import solve_style
from ascii_optimizer.runtime import select_device
from experiments.runner import run_lab
from experiments.stroke_graph import GRAPH_CURVE_CHARS, render_stroke_graph


def _sdf_headroom(target, atlas, sdf_spread, *, device, seed, steps=36, **_kwargs):
    return solve_style(target, atlas, style="headroom", device=device, seed=seed,
                       steps=steps, sdf_spread=sdf_spread)


def _save_named_crop(scene_dir: Path, variant_names: list[str], crop_rel: tuple[float, float, float, float], out_name: str) -> None:
    source = Image.open(scene_dir / "source.png")
    w, h = source.size
    crop = (round(w * crop_rel[0]), round(h * crop_rel[1]), round(w * crop_rel[2]), round(h * crop_rel[3]))
    panels = [("Source", source)]
    for name in variant_names:
        panels.append((name.replace("_", " "), Image.open(scene_dir / name / "final.png")))
    utils.contact_sheet(panels, scene_dir / out_name, columns=min(3, len(panels)), crop=crop, zoom=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--images", nargs="+", default=["viviana", "typhon", "tree"])
    parser.add_argument("--output", type=Path, default=Path("rendered/stroke_graph_v8"))
    args = parser.parse_args()
    device = select_device(args.device)

    variants = {
        "sdf_headroom": (_sdf_headroom, {"use_sdf": True, "steps": 36}),
        "v7_balanced": (render_stroke_graph, {
            "use_sdf": True,
            "chars": GRAPH_CURVE_CHARS,
            "band_weight": 3.4,
            "weber_norm": True,
            "sdf_interval_cov": False,
            "chroma_orient_gain": 2.5,
            "smooth_flow_gain": 0.50,
            "color_cov_gain": 3.0,
        }),
        "v8_sdf_interval": (render_stroke_graph, {
            "use_sdf": True,
            "chars": GRAPH_CURVE_CHARS,
            "band_weight": 3.4,
            "weber_norm": True,
            "sdf_interval_cov": True,
            "chroma_orient_gain": 3.0,
            "smooth_flow_gain": 0.65,
            "color_cov_gain": 3.0,
        }),
        "v8_sdf_flow_strong": (render_stroke_graph, {
            "use_sdf": True,
            "chars": GRAPH_CURVE_CHARS,
            "band_weight": 3.4,
            "weber_norm": True,
            "sdf_interval_cov": True,
            "chroma_orient_gain": 3.5,
            "smooth_flow_gain": 0.90,
            "color_cov_gain": 3.0,
            "coupling": 0.34,
        }),
    }

    report = run_lab(
        variants=variants,
        output=args.output,
        images=tuple(args.images),
        width=args.width,
        device=device,
    )

    vnames = list(variants.keys())
    if "viviana" in report["images"]:
        _save_named_crop(args.output / "viviana", vnames, (0.20, 0.38, 0.50, 0.80), "face_detail.png")
        _save_named_crop(args.output / "viviana", vnames, (0.42, 0.45, 0.82, 0.92), "belly_detail.png")
    if "typhon" in report["images"]:
        _save_named_crop(args.output / "typhon", vnames, (0.02, 0.02, 0.58, 0.65), "aurora_detail.png")

    for name, scene in report["images"].items():
        print(f"=== {name} ===")
        for variant, metrics in scene["variants"].items():
            print(f"  {variant:20s}: dE={metrics['mean_oklab_delta_e_x100']:.2f} "
                  f"L_rmse={metrics['oklab_lightness_rmse_x100']:.2f} "
                  f"C_rmse={metrics['oklab_chroma_rmse_x100']:.2f} "
                  f"cov={metrics['mean_glyph_coverage']:.3f} "
                  f"glyphs={metrics['distinct_glyphs']:2d} "
                  f"({metrics['seconds']:.2f}s)")


if __name__ == "__main__":
    main()
