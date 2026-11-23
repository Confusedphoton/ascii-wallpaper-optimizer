"""Single source of truth for renderer defaults, bounds, and option schemas."""

from __future__ import annotations

import math

DEFAULT_CHARS = " .´:rszunvcxLOGDQ0KNMW&#\\/=^|[]{}"
# Connected-stroke organic graphic + lowercase curve fills (no boxy [] or double-bar =).
STROKE_CHARS = " .'`:;-_~+|/\\(){}<>^v*cszoxX#%@&$"
# Printable ASCII offers thin strokes, bends, junctions, loops and fine texture.
# Keep the accent used by the other styles; every glyph remains a candidate.
FLOW_CHARS = "".join(chr(code) for code in range(32, 127)) + "´"

STYLE_CHARS: dict[str, str] = {
    "headroom": DEFAULT_CHARS,
    "stroke": STROKE_CHARS,
    "flow": FLOW_CHARS,
}

DEFAULT_FLOW_STRENGTH = 0.4
DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH = 0.02
DEFAULT_FLOW_EDDIES = 0.65
DEFAULT_FLOW_SCALE = 6.0
DEFAULT_FLOW_LOOP_SECONDS = 10
DEFAULT_FLOW_FPS = 12
DEFAULT_FLOW_FRAME_STEPS = 12
DEFAULT_FLOW_TEMPORAL_STRENGTH = 0.8

DEFAULT_GRADIENT_LOSS_STRENGTH = 1.5
DEFAULT_GLYPH_LUMINANCE_BONUS = 0.12
MAX_GLYPH_LUMINANCE_BONUS = 0.5
DEFAULT_GLYPH_ENTROPY_STRENGTH = 0.01
MAX_GLYPH_ENTROPY_STRENGTH = 0.1
DEFAULT_PERCEPTUAL_LOSS = True

PERCEPTUAL_GLYPH_STROKE_WEIGHT = 0.4
PERCEPTUAL_PIXEL_WEIGHT = 0.1
PERCEPTUAL_CHROMA_WEIGHT = 0.2
PERCEPTUAL_WEIGHTS = {
    "spatial_lightness": 8.0,
    "orientation_w1": 0.2,
    "contour_strength": 0.005,
}

# Signed-distance-field glyph atlas parameters (shared by Python solver and GLSL).
DEFAULT_SDF_WEIGHT_RANGE = 0.18
DEFAULT_SDF_THIN_RANGE = 0.06
DEFAULT_SDF_SPREAD_RATIO = 0.5
DEFAULT_SDF_MIN_SPREAD = 2.0
DEFAULT_SDF_SUPERSAMPLE = 8

# Nominal viewing geometry used for the perceptual model.
DEFAULT_VIEWING_DISTANCE_METERS = 0.6096
DEFAULT_DISPLAY_DIAGONAL_INCHES = 13.0

# Canonical option schema: name -> default value for standard ("headroom") renders.
OPTION_DEFAULTS: dict[str, object] = {
    "input_format": "images",
    "frame_stride": 1,
    "video_transition_frames": 0,
    "video_preview_ms": 0,
    "style": "headroom",
    "perceptual_loss": DEFAULT_PERCEPTUAL_LOSS,
    "font": "Monospace",
    "chars": DEFAULT_CHARS,
    "invert_ink": False,
    "grayscale": False,
    "detail": 1,
    "frames": 16,
    "steps": 36,
    "transition_steps": 16,
    "seed": 0,
    "dither": 0.005,
    "transition_type": "glyph-path",
    "transition_pace": "ease-in-out",
    "transition_field": "blue-noise",
    "transition_spread": 0.65,
    "gradient_loss_strength": DEFAULT_GRADIENT_LOSS_STRENGTH,
    "glyph_luminance_bonus": DEFAULT_GLYPH_LUMINANCE_BONUS,
    "glyph_entropy_strength": DEFAULT_GLYPH_ENTROPY_STRENGTH,
    "flow_strength": DEFAULT_FLOW_STRENGTH,
    "flow_eddies": DEFAULT_FLOW_EDDIES,
    "flow_scale": DEFAULT_FLOW_SCALE,
    "flow_animated": False,
    "flow_loop_seconds": DEFAULT_FLOW_LOOP_SECONDS,
    "flow_fps": DEFAULT_FLOW_FPS,
    "flow_frame_steps": DEFAULT_FLOW_FRAME_STEPS,
    "flow_temporal_strength": DEFAULT_FLOW_TEMPORAL_STRENGTH,
    "crop": "fill",
    "background": "auto",
}

STYLE_OPTION_DEFAULTS: dict[str, dict[str, object]] = {
    "stroke": {
        "perceptual_loss": False,
        "dither": 0.0,
        "gradient_loss_strength": 0.0,
        "glyph_luminance_bonus": 0.0,
        "glyph_entropy_strength": 0.0,
    },
    "flow": {
        "glyph_entropy_strength": DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH,
    },
}

INT_OPTION_BOUNDS: dict[str, tuple[int, int | None]] = {
    "frame_stride": (1, 10000),
    "video_transition_frames": (0, 118),
    "video_preview_ms": (0, 3000),
    "detail": (0, 4),
    "frames": (2, 120),
    "steps": (1, None),
    "transition_steps": (1, None),
    "seed": (0, None),
    "flow_loop_seconds": (1, 30),
    "flow_fps": (4, 24),
    "flow_frame_steps": (2, 96),
}

FLOAT_OPTION_BOUNDS: dict[str, tuple[float, float]] = {
    "dither": (0.0, 0.05),
    "transition_spread": (0.0, 0.9),
    "gradient_loss_strength": (0.0, 4.0),
    "glyph_luminance_bonus": (0.0, MAX_GLYPH_LUMINANCE_BONUS),
    "glyph_entropy_strength": (0.0, MAX_GLYPH_ENTROPY_STRENGTH),
    "flow_strength": (0.0, 2.0),
    "flow_eddies": (0.0, 2.0),
    "flow_scale": (1.0, 24.0),
    "flow_temporal_strength": (0.0, 2.0),
}

CHOICE_OPTIONS: dict[str, tuple[str, ...]] = {
    "input_format": ("images", "video", "archive"),
    "style": ("headroom", "stroke", "flow"),
    "transition_type": ("glyph-path", "earth-mover"),
    "transition_pace": ("linear", "ease-in", "ease-out", "ease-in-out"),
    "transition_field": ("blue-noise", "ripple", "spiral", "directional", "edge", "salience"),
    "crop": ("fill", "fit"),
}

BOOL_OPTIONS: tuple[str, ...] = ("invert_ink", "grayscale", "flow_animated")
FLOW_OPTIONS: tuple[str, ...] = ("flow_strength", "flow_eddies", "flow_scale")
FLOW_ANIMATED_OPTIONS: tuple[str, ...] = (
    "flow_loop_seconds", "flow_fps", "flow_frame_steps", "flow_temporal_strength",
)
VIDEO_OPTIONS: tuple[str, ...] = ("frame_stride", "video_transition_frames", "video_preview_ms")
TRANSITION_OPTION_NAMES: frozenset[str] = frozenset({
    "frames", "transition_steps", "transition_type", "transition_pace",
    "transition_field", "transition_spread",
})

# Shared option list copied into render signatures and CLI invocations.
SIGNATURE_OPTION_NAMES: tuple[str, ...] = (
    "input_format", "frame_stride", "video_transition_frames",
    "chars", "detail", "grayscale", "frames", "dither",
    "transition_type", "transition_pace", "transition_field", "transition_spread",
    "gradient_loss_strength", "glyph_luminance_bonus", "glyph_entropy_strength",
    "style", "invert_ink", "background", "crop", "steps", "transition_steps", "seed",
    "perceptual_loss", "flow_strength", "flow_eddies", "flow_scale",
    "flow_animated", "flow_loop_seconds", "flow_fps", "flow_frame_steps",
    "flow_temporal_strength",
)


def style_default(style: str, name: str):
    """Return the effective default for an option under the chosen style."""
    if name in STYLE_OPTION_DEFAULTS.get(style, {}):
        return STYLE_OPTION_DEFAULTS[style][name]
    return OPTION_DEFAULTS[name]


def ppd_for_width(width, distance=DEFAULT_VIEWING_DISTANCE_METERS,
                  diagonal_inches=DEFAULT_DISPLAY_DIAGONAL_INCHES):
    """Image pixels per degree across the nominal 13-inch 16:10 display."""
    physical_width = diagonal_inches * 0.0254 * 16 / math.sqrt(16**2 + 10**2)
    return distance * width / physical_width * math.pi / 180
