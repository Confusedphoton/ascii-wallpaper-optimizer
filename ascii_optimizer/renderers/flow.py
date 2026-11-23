"""Forced-fluid flow domains renderer."""

from __future__ import annotations

import torch

from ..assets import sdf_to_mask
from ..flow import glyph_flow_costs, simulate_flow
from ..optimizer import Grid, solve
from ..settings import finite_float
from ..tone import encode_headroom_lab
from . import register


def render_flow(target, masks, *, source_lab, solve_background, invert_ink,
                seed, steps, dither, device, previous, destination, progress,
                on_progress, glyph_luminance_bonus, saturation_loss_strength,
                glyph_entropy_strength, gradient_loss_strength,
                postprocess, perceptual_loss, viewing_ppd, flow_strength,
                flow_eddies, flow_scale, flow_diagnostics, flow_costs,
                sdf_spread, solve_context, diagnostics, **_unused) -> Grid:
    if not perceptual_loss:
        raise ValueError("Flow rendering requires perceptual color fitting")
    flow_strength = finite_float(flow_strength, "flow_strength", 0, 2)
    flow_eddies = finite_float(flow_eddies, "flow_eddies", 0, 2)
    flow_scale = finite_float(flow_scale, "flow_scale", 1, 24)

    solve_progress = on_progress
    if flow_strength and flow_costs is None:
        eval_masks = sdf_to_mask(masks, spread=sdf_spread) if sdf_spread > 0.0 else masks
        field = simulate_flow(
            target, eval_masks.shape[1:], seed=seed, eddies=flow_eddies,
            scale=flow_scale, device=device,
            on_progress=(lambda f, stage: on_progress(0.15 * f, stage))
            if on_progress is not None else None,
        )
        flow_costs, summary = glyph_flow_costs(field, torch.as_tensor(eval_masks, device=device))
        if flow_diagnostics is not None:
            flow_diagnostics.update(field=field, costs=flow_costs, **field.diagnostics, **summary)
        if diagnostics is not None:
            diagnostics.metadata["flow"] = {**field.diagnostics, **summary}
        if on_progress is not None:
            solve_progress = lambda f, stage: on_progress(0.15 + 0.85 * f, stage)

    encoded_target = encode_headroom_lab(source_lab, invert_ink=invert_ink)
    return solve(
        encoded_target, masks,
        seed=seed, steps=steps, dither=dither,
        background_oklab=solve_background, device=device, input_oklab=True,
        previous=previous, destination=destination, progress=progress,
        on_progress=solve_progress,
        glyph_luminance_bonus=glyph_luminance_bonus,
        saturation_loss_strength=saturation_loss_strength,
        saturation_target_oklab=source_lab,
        glyph_entropy_strength=glyph_entropy_strength,
        gradient_loss_strength=gradient_loss_strength,
        gradient_target_oklab=source_lab,
        postprocess=postprocess,
        perceptual_loss=perceptual_loss, perceptual_target_srgb=target,
        viewing_ppd=viewing_ppd,
        flow_costs=flow_costs,
        flow_strength=flow_strength if flow_costs is not None else 0.0,
        sdf_spread=sdf_spread,
        solve_context=solve_context, diagnostics=diagnostics,
        invert_ink=invert_ink,
    )


register("flow", render_flow)
