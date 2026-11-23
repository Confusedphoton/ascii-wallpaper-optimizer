"""Quick lab comparison: legacy alpha atlas vs. new SDF variable-weight atlas.

Run:
  .venv/bin/python -m ascii_optimizer.launcher experiments/compare_sdf.py --device cuda
  .venv/bin/python scripts/view_renders.py --report rendered/sdf_lab
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ascii_optimizer.defaults import STROKE_CHARS
from ascii_optimizer.renderers import solve_style
from ascii_optimizer.runtime import select_device
from experiments.runner import run_lab


def _headroom(target, atlas, sdf_spread, *, device, seed, steps=36, **_kwargs):
    return solve_style(target, atlas, style="headroom", device=device, seed=seed,
                       steps=steps, sdf_spread=sdf_spread)


def _stroke(target, atlas, sdf_spread, *, device, seed, **_kwargs):
    return solve_style(target, atlas, style="stroke", device=device, seed=seed,
                       sdf_spread=sdf_spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--images", nargs="+", default=["viviana", "typhon"])
    parser.add_argument("--output", type=Path, default=Path("rendered/sdf_lab"))
    args = parser.parse_args()
    device = select_device(args.device)
    report = run_lab(
        variants={
            "alpha_headroom": (_headroom, {"use_sdf": False, "steps": 36}),
            "sdf_headroom": (_headroom, {"use_sdf": True, "steps": 36}),
            "sdf_stroke": (_stroke, {"use_sdf": True, "chars": STROKE_CHARS}),
        },
        output=args.output,
        images=tuple(args.images),
        width=args.width,
        device=device,
    )
    for name, scene in report["images"].items():
        print(f"=== {name} ===")
        for variant, metrics in scene["variants"].items():
            print(f"  {variant:16s}: dE={metrics['mean_oklab_delta_e_x100']:.2f} "
                  f"L_rmse={metrics['oklab_lightness_rmse_x100']:.2f} "
                  f"cov={metrics['mean_glyph_coverage']:.3f} "
                  f"glyphs={metrics['distinct_glyphs']} "
                  f"({metrics['seconds']:.2f}s)")


if __name__ == "__main__":
    main()
