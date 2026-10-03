"""Defaults shared by the batch renderer and optimizer styles."""

import math

DEFAULT_CHARS = " .´:rszunvcxLOGDQ0KNMW&#\\/=^|[]{}"
DEFAULT_USE_SHORTLIST = False
DEFAULT_GLYPH_ENTROPY_STRENGTH = 0.01
MAX_GLYPH_ENTROPY_STRENGTH = 0.1
STRUCTURE_CHARS = " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&#[]{}"
# Printable ASCII offers thin strokes, bends, junctions, loops and fine texture.
# Keep the accent used by the other styles; every glyph remains a candidate.
FLOW_CHARS = "".join(chr(code) for code in range(32, 127)) + "´"
DEFAULT_FLOW_STRENGTH = 0.4
DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH = 0.02
DEFAULT_FLOW_EDDIES = 0.65
DEFAULT_FLOW_SCALE = 6.0
DEFAULT_GRADIENT_LOSS_STRENGTH = 1.5
DEFAULT_GLYPH_LUMINANCE_BONUS = 0.12
MAX_GLYPH_LUMINANCE_BONUS = 0.5
DEFAULT_REFINEMENT_SWEEPS = 0
DEFAULT_POLISH_STEPS = 9
DEFAULT_PERCEPTUAL_LOSS = True
PERCEPTUAL_GLYPH_STROKE_WEIGHT = 0.4
PERCEPTUAL_PIXEL_WEIGHT = 0.1
PERCEPTUAL_CHROMA_WEIGHT = 0.2
PERCEPTUAL_WEIGHTS = {"spatial_lightness": 8.0, "orientation_w1": 0.2,
                      "contour_strength": 0.005}
# Nominal viewing geometry used for the approved experiment.
DEFAULT_VIEWING_DISTANCE_METERS = 0.6096
DEFAULT_DISPLAY_DIAGONAL_INCHES = 13.0


def ppd_for_width(width, distance=DEFAULT_VIEWING_DISTANCE_METERS,
                  diagonal_inches=DEFAULT_DISPLAY_DIAGONAL_INCHES):
    """Image pixels per degree across the nominal 13-inch 16:10 display."""
    physical_width = diagonal_inches * 0.0254 * 16 / math.sqrt(16**2 + 10**2)
    return distance * width / physical_width * math.pi / 180
