"""Perceptual and classic headroom renderer."""

from __future__ import annotations

from ..optimizer import Grid, solve
from ..tone import encode_headroom_lab
from . import register


def render_headroom(target, masks, *, source_lab, solve_background, invert_ink,
                    seed, steps, dither, device, previous, destination, progress,
                    on_progress, glyph_luminance_bonus, saturation_loss_strength,
                    glyph_entropy_strength, gradient_loss_strength,
                    postprocess, perceptual_loss, viewing_ppd, sdf_spread,
                    solve_context, diagnostics, **_unused) -> Grid:
    encoded_target = encode_headroom_lab(source_lab, invert_ink=invert_ink)
    return solve(
        encoded_target, masks,
        seed=seed, steps=steps, dither=dither,
        background_oklab=solve_background, device=device, input_oklab=True,
        previous=previous, destination=destination, progress=progress,
        on_progress=on_progress,
        glyph_luminance_bonus=glyph_luminance_bonus,
        saturation_loss_strength=saturation_loss_strength,
        saturation_target_oklab=source_lab,
        glyph_entropy_strength=glyph_entropy_strength,
        gradient_loss_strength=gradient_loss_strength,
        gradient_target_oklab=source_lab,
        postprocess=postprocess,
        perceptual_loss=perceptual_loss, perceptual_target_srgb=target,
        viewing_ppd=viewing_ppd, sdf_spread=sdf_spread,
        solve_context=solve_context, diagnostics=diagnostics,
        invert_ink=invert_ink,
    )


register("headroom", render_headroom)
