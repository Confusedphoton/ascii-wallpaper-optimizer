"""Shared, numerical-runtime-free defaults and validation for render requests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .defaults import (DEFAULT_CHARS, STRUCTURE_CHARS, FLOW_CHARS, DEFAULT_FLOW_STRENGTH,
                       DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH,
                       DEFAULT_FLOW_EDDIES, DEFAULT_FLOW_SCALE, DEFAULT_FLOW_LOOP_SECONDS, DEFAULT_FLOW_FPS,
                       DEFAULT_FLOW_FRAME_STEPS, DEFAULT_FLOW_TEMPORAL_STRENGTH, DEFAULT_PERCEPTUAL_LOSS, DEFAULT_GLYPH_LUMINANCE_BONUS,
                       DEFAULT_GRADIENT_LOSS_STRENGTH, MAX_GLYPH_LUMINANCE_BONUS,
                       DEFAULT_GLYPH_ENTROPY_STRENGTH, MAX_GLYPH_ENTROPY_STRENGTH)


def finite_float(value, name: str, minimum: float | None = None,
                 maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if ((minimum is not None and result < minimum)
            or (maximum is not None and result > maximum)):
        raise ValueError(f"{name} is out of range")
    return result


def _integer(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} is out of range")
        return value
    number = finite_float(value, name, minimum, maximum)
    if isinstance(value, bool) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def validate_size(value, max_dimension: int = 65535) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("size must contain width and height")
    return tuple(_integer(item, "display dimension", 1, max_dimension) for item in value)


def normalize_options(raw: Mapping) -> dict:
    """Validate common CLI/service options, leaving unrelated arguments intact."""
    if not isinstance(raw, Mapping):
        raise ValueError("Render settings must be an object")
    options = dict(raw)
    # Legacy clients may still send a shortlist size. It no longer affects
    # rendering, and must not create separate cache variants.
    options.pop("candidates", None)
    style = options.setdefault("style", "headroom")
    if style not in ("headroom", "structure", "flow"):
        raise ValueError("Unknown render style")
    perceptual_loss = options.get("perceptual_loss")
    if perceptual_loss is None:
        perceptual_loss = DEFAULT_PERCEPTUAL_LOSS if style != "structure" else False
        options["perceptual_loss"] = perceptual_loss
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if style == "structure" and perceptual_loss:
        raise ValueError("perceptual_loss is only available for headroom rendering")
    if style == "flow" and not perceptual_loss:
        raise ValueError("Flow rendering requires perceptual color fitting")
    options.setdefault("font", "Monospace")
    if not isinstance(options["font"], str) or not options["font"].strip():
        raise ValueError("font must be a nonempty family or path")
    chars = options.get("chars", DEFAULT_CHARS)
    if not isinstance(chars, str):
        raise ValueError("chars must be a string")
    if chars.startswith(r"\s"):
        chars = " " + chars[2:]
    if not 1 <= len(chars) <= 256 or len(set(chars)) != len(chars):
        raise ValueError("Characters must be 1 to 256 distinct glyphs")
    options["chars"] = ({"structure": STRUCTURE_CHARS, "flow": FLOW_CHARS}.get(style, chars)
                        if chars == DEFAULT_CHARS else chars)
    for name, default in (("invert_ink", False), ("grayscale", False)):
        value = options.setdefault(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    for name, default, low, high in (
        ("detail", 1, 0, 4), ("frames", 16, 2, 120),
        ("steps", 36, 1, None), ("transition_steps", 16, 1, None),
        ("seed", 0, 0, None),
    ):
        options[name] = _integer(options.get(name, default), name, low, high)
    for name, default, high in (
        ("dither", 0.005, 0.05), ("structure_dither", 0.06, 0.12),
        ("transition_spread", 0.65, 0.9),
        ("gradient_loss_strength", DEFAULT_GRADIENT_LOSS_STRENGTH if style != "structure" else 0.0, 4.0),
        ("glyph_luminance_bonus", DEFAULT_GLYPH_LUMINANCE_BONUS if style != "structure" else 0.0,
         MAX_GLYPH_LUMINANCE_BONUS),
        ("glyph_entropy_strength", DEFAULT_FLOW_GLYPH_ENTROPY_STRENGTH if style == "flow" else
         DEFAULT_GLYPH_ENTROPY_STRENGTH if style == "headroom" else 0.0,
         MAX_GLYPH_ENTROPY_STRENGTH),
    ):
        value = options.get(name)
        options[name] = finite_float(default if value is None else value, name, 0, high)
    for name, default, low, high in (
        ("flow_strength", DEFAULT_FLOW_STRENGTH, 0, 2),
        ("flow_eddies", DEFAULT_FLOW_EDDIES, 0, 2),
        ("flow_scale", DEFAULT_FLOW_SCALE, 1, 24),
    ):
        value = options.get(name)
        options[name] = finite_float(default if value is None else value, name, low, high)
        if style != "flow":
            # Inactive settings must not create redundant cache variants.
            options[name] = default
    animated = options.setdefault("flow_animated", False)
    if not isinstance(animated, bool):
        raise ValueError("flow_animated must be a boolean")
    options["flow_animated"] = animated and style == "flow"
    for name, default, low, high in (
        ("flow_loop_seconds", DEFAULT_FLOW_LOOP_SECONDS, 1, 30),
        ("flow_fps", DEFAULT_FLOW_FPS, 4, 24),
        ("flow_frame_steps", DEFAULT_FLOW_FRAME_STEPS, 2, 96),
    ):
        options[name] = _integer(options.get(name, default), name, low, high)
        if not options["flow_animated"]:
            options[name] = default
    options["flow_temporal_strength"] = finite_float(options.get("flow_temporal_strength",
        DEFAULT_FLOW_TEMPORAL_STRENGTH), "flow_temporal_strength", 0, 2)
    if not options["flow_animated"]:
        options["flow_temporal_strength"] = DEFAULT_FLOW_TEMPORAL_STRENGTH
    if style == "structure" and (options["gradient_loss_strength"] or options["glyph_luminance_bonus"]):
        raise ValueError("Gradient loss and glyph luminance bonus are only available for headroom rendering")
    if style == "structure" and options["glyph_entropy_strength"]:
        raise ValueError("glyph_entropy_strength is only available for headroom rendering")
    for name, default, choices in (
        ("transition_type", "glyph-path", ("glyph-path", "earth-mover")),
        ("transition_pace", "ease-in-out", ("linear", "ease-in", "ease-out", "ease-in-out")),
        ("transition_field", "blue-noise", ("blue-noise", "ripple", "spiral", "directional", "edge", "salience")),
        ("crop", "fill", ("fill", "fit")),
    ):
        if options.setdefault(name, default) not in choices:
            raise ValueError(f"Invalid {name}")
    background = options.setdefault("background", "auto")
    if not isinstance(background, str) or (background != "auto" and not re.fullmatch(r"#[0-9a-fA-F]{6}", background)):
        raise ValueError("Background must be auto or a #RRGGBB color")
    options["background"] = background.lower()
    return options
