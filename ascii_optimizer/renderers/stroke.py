"""Connected-stroke MRF / graph renderer registration."""

from __future__ import annotations

from ..optimizer import Grid
from ..stroke_graph import solve_stroke_graph
from . import register


def render_stroke(target, masks, *, source_lab, original_background, solve_background,
                  invert_ink, background, seed, device, on_progress, sdf_spread,
                  **_unused) -> Grid:
    return solve_stroke_graph(
        target, masks,
        sdf_spread=sdf_spread,
        background=background,
        original_background=original_background,
        solve_background=solve_background,
        source_lab=source_lab,
        invert_ink=invert_ink,
        seed=seed,
        device=device,
        on_progress=on_progress,
    )


register("stroke", render_stroke)
