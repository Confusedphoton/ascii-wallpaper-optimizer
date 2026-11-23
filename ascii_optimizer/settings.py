"""Shared, numerical-runtime-free defaults and validation for render requests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .defaults import (
    BOOL_OPTIONS,
    CHOICE_OPTIONS,
    DEFAULT_CHARS,
    FLOAT_OPTION_BOUNDS,
    FLOW_ANIMATED_OPTIONS,
    FLOW_OPTIONS,
    INT_OPTION_BOUNDS,
    OPTION_DEFAULTS,
    SIGNATURE_OPTION_NAMES,
    STYLE_CHARS,
    TRANSITION_OPTION_NAMES,
    VIDEO_OPTIONS,
    style_default,
)

VIDEO_PLAYBACK_OPTIONS = frozenset({*VIDEO_OPTIONS, *TRANSITION_OPTION_NAMES})

# Options forwarded by the render service as --kebab-case <value> CLI flags.
CLI_VALUE_OPTION_NAMES: tuple[str, ...] = tuple(
    name for name in SIGNATURE_OPTION_NAMES
    if name not in (*BOOL_OPTIONS, "perceptual_loss")
)


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
    # Legacy clients may still send removed options; drop them so they do not
    # split cache variants.
    options.pop("candidates", None)
    options.pop("structure_dither", None)

    for name, choices in CHOICE_OPTIONS.items():
        value = options.setdefault(name, OPTION_DEFAULTS[name])
        if value not in choices:
            if name == "input_format":
                raise ValueError("Unknown input format")
            if name == "style":
                raise ValueError("Unknown render style")
            raise ValueError(f"Invalid {name}")

    input_format = options["input_format"]
    style = options["style"]

    perceptual_loss = options.get("perceptual_loss")
    if perceptual_loss is None:
        perceptual_loss = style_default(style, "perceptual_loss")
        options["perceptual_loss"] = perceptual_loss
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")
    if style == "flow" and not perceptual_loss:
        raise ValueError("Flow rendering requires perceptual color fitting")

    options.setdefault("font", OPTION_DEFAULTS["font"])
    if not isinstance(options["font"], str) or not options["font"].strip():
        raise ValueError("font must be a nonempty family or path")

    chars = options.get("chars", DEFAULT_CHARS)
    if not isinstance(chars, str):
        raise ValueError("chars must be a string")
    if chars.startswith(r"\s"):
        chars = " " + chars[2:]
    if not 1 <= len(chars) <= 256 or len(set(chars)) != len(chars):
        raise ValueError("Characters must be 1 to 256 distinct glyphs")
    options["chars"] = STYLE_CHARS.get(style, chars) if chars == DEFAULT_CHARS else chars

    for name in BOOL_OPTIONS:
        value = options.setdefault(name, OPTION_DEFAULTS[name])
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")

    for name, (low, high) in INT_OPTION_BOUNDS.items():
        value = options.get(name)
        default = style_default(style, name)
        options[name] = _integer(default if value is None else value, name, low, high)

    for name, (low, high) in FLOAT_OPTION_BOUNDS.items():
        value = options.get(name)
        default = style_default(style, name)
        options[name] = finite_float(default if value is None else value, name, low, high)

    # Zero out inactive mode/style options so they never split cache keys.
    if input_format != "video":
        for name in VIDEO_OPTIONS:
            options[name] = OPTION_DEFAULTS[name]
    if style != "flow":
        for name in FLOW_OPTIONS:
            options[name] = OPTION_DEFAULTS[name]
    if style == "stroke":
        options["perceptual_loss"] = False
        for name in ("dither", "gradient_loss_strength", "glyph_luminance_bonus", "glyph_entropy_strength"):
            options[name] = 0.0
    options["flow_animated"] = bool(options["flow_animated"] and style == "flow" and input_format != "video")
    if not options["flow_animated"]:
        for name in FLOW_ANIMATED_OPTIONS:
            options[name] = OPTION_DEFAULTS[name]

    background = options.setdefault("background", OPTION_DEFAULTS["background"])
    if not isinstance(background, str) or (background != "auto" and not re.fullmatch(r"#[0-9a-fA-F]{6}", background)):
        raise ValueError("Background must be auto or a #RRGGBB color")
    options["background"] = background.lower()
    return options
