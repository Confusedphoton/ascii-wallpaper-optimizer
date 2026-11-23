"""Lab adapter re-exporting the production connected-stroke MRF renderer."""

from __future__ import annotations

from ascii_optimizer.stroke_graph import (
    GRAPH_ASCII_CHARS,
    GRAPH_CURVE_CHARS,
    GRAPH_FLOWING_CHARS,
    GRAPH_ORGANIC_CHARS,
    GRAPH_SYMBOL_CHARS,
    _glyph_axial,
    _oklab_morph_close,
    _seam_compatibilities,
    solve_stroke_graph,
)


def render_stroke_graph(source_srgb, atlas, sdf_spread, **kwargs):
    return solve_stroke_graph(source_srgb, atlas, sdf_spread=sdf_spread, **kwargs)


__all__ = [
    "GRAPH_ASCII_CHARS",
    "GRAPH_CURVE_CHARS",
    "GRAPH_FLOWING_CHARS",
    "GRAPH_ORGANIC_CHARS",
    "GRAPH_SYMBOL_CHARS",
    "_glyph_axial",
    "_oklab_morph_close",
    "_seam_compatibilities",
    "render_stroke_graph",
    "solve_stroke_graph",
]
