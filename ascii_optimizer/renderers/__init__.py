"""Pluggable ASCII renderer registry decoupled from CLI, service, and UI plumbing."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..defaults import (
    DEFAULT_FLOW_EDDIES,
    DEFAULT_FLOW_SCALE,
    DEFAULT_FLOW_STRENGTH,
    style_default,
)
from ..optimizer import Grid, _choose_background
from ..tone import encode_headroom_lab, quantize_playback_lab, srgb_to_oklab

RendererFn = Callable[..., Grid]
RENDERERS: dict[str, RendererFn] = {}


def register(name: str, renderer: RendererFn) -> None:
    """Register a render style by name."""
    RENDERERS[name] = renderer


def get_renderer(name: str) -> RendererFn:
    if name not in RENDERERS:
        raise ValueError(f"Unknown render style: {name}")
    return RENDERERS[name]


def available_styles() -> tuple[str, ...]:
    return tuple(RENDERERS)


def resolve_backgrounds(source_lab: np.ndarray, background: str,
                        invert_ink: bool) -> tuple[np.ndarray, np.ndarray]:
    original = _choose_background(torch.from_numpy(source_lab).permute(2, 0, 1), background).numpy()
    encoded = (np.zeros(3, dtype=np.float32) if invert_ink and background == "auto"
               else encode_headroom_lab(original.reshape(1, 1, 3), invert_ink=invert_ink)[0, 0])
    return original, quantize_playback_lab(encoded)


def solve_style(target: np.ndarray, masks: np.ndarray, *, style: str,
                invert_ink: bool = False, background: str = "auto", seed: int = 0,
                steps: int = 36, dither: float = 0.005,
                device: str = "cpu", previous: Grid | None = None,
                destination: Grid | None = None, progress: float | None = None,
                on_progress: Callable[[float, str], None] | None = None,
                glyph_luminance_bonus: float | None = None,
                saturation_loss_strength: float | None = None,
                glyph_entropy_strength: float | None = None,
                gradient_loss_strength: float | None = None,
                postprocess: bool | None = None,
                perceptual_loss: bool | None = None,
                viewing_ppd: float | None = None,
                flow_strength: float = DEFAULT_FLOW_STRENGTH,
                flow_eddies: float = DEFAULT_FLOW_EDDIES,
                flow_scale: float = DEFAULT_FLOW_SCALE,
                flow_diagnostics: dict | None = None,
                flow_costs=None,
                sdf_spread: float = 0.0,
                solve_context: dict | None = None,
                diagnostics=None,
                **extra) -> Grid:
    """Dispatch a render request through the pluggable renderer registry."""
    renderer = get_renderer(style)
    if gradient_loss_strength is None:
        gradient_loss_strength = style_default(style, "gradient_loss_strength") if style in ("headroom", "flow") else 0.0
    if glyph_luminance_bonus is None:
        glyph_luminance_bonus = style_default(style, "glyph_luminance_bonus") if style in ("headroom", "flow") else 0.0
    if glyph_entropy_strength is None:
        glyph_entropy_strength = style_default(style, "glyph_entropy_strength") if style in ("headroom", "flow") else 0.0
    if perceptual_loss is None:
        perceptual_loss = style_default(style, "perceptual_loss") if style in ("headroom", "flow") else False
    if not isinstance(perceptual_loss, bool):
        raise ValueError("perceptual_loss must be a boolean")

    source_lab = srgb_to_oklab(target)
    original_background, solve_background = resolve_backgrounds(source_lab, background, invert_ink)

    return renderer(
        target, masks,
        source_lab=source_lab,
        original_background=original_background,
        solve_background=solve_background,
        invert_ink=invert_ink,
        background=background,
        seed=seed,
        steps=steps,
        dither=dither,
        device=device,
        previous=previous,
        destination=destination,
        progress=progress,
        on_progress=on_progress,
        glyph_luminance_bonus=glyph_luminance_bonus,
        saturation_loss_strength=saturation_loss_strength,
        glyph_entropy_strength=glyph_entropy_strength,
        gradient_loss_strength=gradient_loss_strength,
        postprocess=postprocess,
        perceptual_loss=perceptual_loss,
        viewing_ppd=viewing_ppd,
        flow_strength=flow_strength,
        flow_eddies=flow_eddies,
        flow_scale=flow_scale,
        flow_diagnostics=flow_diagnostics,
        flow_costs=flow_costs,
        sdf_spread=sdf_spread,
        solve_context=solve_context,
        diagnostics=diagnostics,
        **extra,
    )


# Register built-in renderers.
from . import flow as _flow  # noqa: E402, F401
from . import headroom as _headroom  # noqa: E402, F401
from . import stroke as _stroke  # noqa: E402, F401
