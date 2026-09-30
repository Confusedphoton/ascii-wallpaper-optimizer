"""Shared, numerical-runtime-free defaults and validation for render requests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .defaults import (DEFAULT_CHARS, STRUCTURE_CHARS, DEFAULT_GLYPH_LUMINANCE_BONUS,
                       DEFAULT_GRADIENT_LOSS_STRENGTH, MAX_GLYPH_LUMINANCE_BONUS)


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
    style = options.setdefault("style", "headroom")
    if style not in ("headroom", "structure"):
        raise ValueError("Unknown render style")
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
    options["chars"] = STRUCTURE_CHARS if style == "structure" and chars == DEFAULT_CHARS else chars
    for name, default in (("invert_ink", False), ("grayscale", False)):
        value = options.setdefault(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    for name, default, low, high in (
        ("detail", 1, 0, 4), ("frames", 16, 2, 120),
        ("steps", 36, 1, None), ("transition_steps", 16, 1, None),
        ("candidates", 6, 1, 256), ("seed", 0, 0, None),
    ):
        options[name] = _integer(options.get(name, default), name, low, high)
    for name, default, high in (
        ("dither", 0.005, 0.05), ("structure_dither", 0.06, 0.12),
        ("transition_spread", 0.65, 0.9),
        ("gradient_loss_strength", DEFAULT_GRADIENT_LOSS_STRENGTH if style == "headroom" else 0.0, 4.0),
        ("glyph_luminance_bonus", DEFAULT_GLYPH_LUMINANCE_BONUS if style == "headroom" else 0.0,
         MAX_GLYPH_LUMINANCE_BONUS),
    ):
        value = options.get(name)
        options[name] = finite_float(default if value is None else value, name, 0, high)
    if style == "structure" and (options["gradient_loss_strength"] or options["glyph_luminance_bonus"]):
        raise ValueError("Gradient loss and glyph luminance bonus are only available for headroom rendering")
    for name, default, choices in (
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
