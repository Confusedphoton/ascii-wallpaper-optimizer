"""State ownership, temporal boundaries, circular fitting, and clip persistence."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest
import torch

from ascii_optimizer import cli
from ascii_optimizer.flow import prepare_flow, simulate_flow, simulate_flow_loop, glyph_flow_costs, prepare_glyph_flow
from ascii_optimizer.optimizer import Grid
from ascii_optimizer.settings import normalize_options


@pytest.fixture(autouse=True)
def modest_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_stateful_steps_match_static_snapshot_and_snapshots_own_their_data():
    source = np.full((24, 32, 3), .4, np.float32)
    simulation = prepare_flow(source, (8, 4), seed=3)
    simulation.advance(5)
    frozen = simulation.snapshot()
    original = frozen.velocity.clone()
    simulation.advance(7)
    assert torch.equal(frozen.velocity, original)
    expected = simulate_flow(source, (8, 4), seed=3, steps=12)
    assert torch.equal(simulation.snapshot().velocity, expected.velocity)
    assert simulation.velocity.shape[-1] > expected.velocity.shape[-1]


def test_temporal_loop_has_motion_a_normal_wrap_and_seeded_repeatability():
    source = np.full((24, 32, 3), .4, np.float32)
    first = simulate_flow_loop(source, (8, 4), seed=2, frames=24)
    second = simulate_flow_loop(source, (8, 4), seed=2, frames=24)
    assert len(first.fields) == 24
    assert first.diagnostics["method"] == "temporal-overlap"
    assert first.diagnostics["velocity_step_mean"] > 1e-3
    assert first.diagnostics["velocity_seam_ratio"] < 2
    assert first.diagnostics["divergence_after_projection"] < 1e-5
    for a, b in zip(first.fields, second.fields):
        assert torch.equal(a.velocity, b.velocity)
        assert torch.isfinite(a.velocity).all()
        assert a.confidence.min() >= 0 and a.confidence.max() <= 1


def test_cached_atlas_descriptors_preserve_static_costs():
    source = np.full((16, 16, 3), .4, np.float32)
    field = simulate_flow(source, (8, 4), steps=3)
    masks = torch.rand(4, 8, 4)
    expected, _ = glyph_flow_costs(field, masks)
    actual, _ = glyph_flow_costs(field, masks, prepare_glyph_flow(masks))
    assert torch.equal(expected, actual)


def test_temporal_cost_uses_rasters_not_glyph_identity():
    from ascii_optimizer.flow_playback import select_cyclic_glyphs
    masks = np.full((3, 8, 4), .25, np.float32)
    unary = np.tile([1., .5, 0.], (4, 8, 1)).astype(np.float32)
    labels, _ = select_cyclic_glyphs(unary, masks, strength=2)
    assert (labels == 2).all()


def test_clip_fitting_includes_both_wrap_neighbors_and_shares_source_context():
    from ascii_optimizer.flow_playback import render_flow_clip
    source = np.full((16, 16, 3), .35, np.float32)
    masks = np.full((3, 8, 4), .25, np.float32)
    calls = []
    def prepare(costs, budget, context):
        context["image_unary"] = np.zeros((8, 3), np.float32)
        context["candidate_colors"] = np.zeros((8, 3, 3), np.float32)
        return Grid(np.zeros((2, 4), np.uint8), np.zeros((2, 4, 3), np.float32), np.zeros(3))
    def color_fit(glyphs, neighbors, context):
        calls.append((neighbors, context))
        return Grid(glyphs, np.full((2, 4, 3), len(calls) / 100, np.float32), np.zeros(3))
    grids, _ = render_flow_clip(source, masks, seed=0, eddies=.65, scale=6, device="cpu",
        frames=4, steps=3, frame_steps=2, temporal_strength=.12,
        solve_frame=prepare, fit_colors=color_fit)
    assert len(grids) == 4
    assert calls[0][0][0].glyph.shape == (2, 4)
    assert calls[3][0][1] is calls[1][0][0]  # Last phase receives phase 0 as its next neighbor.
    assert all(call[1] is calls[0][1] for call in calls)


def test_cyclic_layout_removes_isolated_chatter_without_erasing_sustained_motion():
    from ascii_optimizer.flow_playback import select_cyclic_glyphs
    # A brief improvement cannot pay for two raster jumps. A sustained change
    # through the wrap can pay for its entry and exit and must remain visible.
    masks = np.array([[[0., 0.]], [[1., 1.]]], np.float32)
    unary = np.zeros((12, 1, 2), np.float32)
    unary[:, :, 1] = 1
    unary[5, :, 1] = -.1
    unary[[10, 11, 0, 1], :, 1] = -1
    labels, diagnostics = select_cyclic_glyphs(unary, masks, .3)
    assert labels[5, 0] == 0
    assert (labels[[10, 11, 0, 1], 0] == 1).all()
    assert diagnostics["layout_objective_final"] <= diagnostics["layout_objective_initial"]


def test_clip_colors_reuse_source_referenced_perceptual_starts():
    from ascii_optimizer.optimizer import fit_fixed_glyphs, render_grid
    from ascii_optimizer.tone import encode_headroom_lab, srgb_to_oklab
    source = np.full((8, 8, 3), .8, np.float32)
    masks = np.zeros((2, 8, 4), np.float32)
    masks[1, :, 1:3] = 1
    context = {}
    cli.solve_style(source, masks, style="flow", steps=0, flow_strength=0,
                    solve_context=context, glyph_entropy_strength=0, device="cpu",
                    invert_ink=False, background="#000000", seed=0, dither=0)
    glyphs = np.ones((1, 2), np.uint8)
    options = dict(background_oklab=np.zeros(3), input_oklab=True, steps=0,
                   perceptual_target_srgb=source, glyph_luminance_bonus=0,
                   gradient_loss_strength=0, saturation_loss_strength=0)
    encoded = encode_headroom_lab(srgb_to_oklab(source))
    pixel_start = fit_fixed_glyphs(encoded, masks, glyphs, **options)
    perceptual_start = fit_fixed_glyphs(encoded, masks, glyphs, solve_context=context, **options)
    # Bright sparse ink needs the linear-light fit retained by the source
    # solve. A short clip color budget must not restart from dark pixel ink.
    assert render_grid(perceptual_start, masks).mean() > render_grid(pixel_start, masks).mean() + .05


@pytest.mark.parametrize("image_count", [1, 2])
def test_animated_cli_anchor_cache_and_service_reject_missing_loop_frame(tmp_path, monkeypatch, image_count):
    from ascii_optimizer.service import RenderQueue
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 16), "#6688aa").save(source)
    images = [source]
    if image_count == 2:
        second_source = tmp_path / "second.png"
        Image.new("RGB", (32, 16), "#cc7733").save(second_source)
        images.append(second_source)
    cache = tmp_path / "cache/stills"
    def arguments(output):
        return cli.make_parser().parse_args([*(str(image) for image in images), "--output", str(output), "--size", "32x16",
            "--cell", "4x8", "--style", "flow", "--flow-animated", "--flow-loop-seconds", "1",
            "--flow-fps", "4", "--steps", "2", "--flow-frame-steps", "2", "--device", "cpu",
            "--still-cache", str(cache)])
    output = tmp_path / "first"
    cli.build(arguments(output))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frame_count"] == 16  # Transition count has not become the clip count.
    assert manifest["loops"]["0000"]["frame_count"] == 4
    assert len(manifest["loops"]) == image_count
    for index in range(image_count):
        for kind in ("glyph", "color", "background"):
            anchor = (output / f"loops/{index:04d}/0000" / f"{kind}.png").read_bytes()
            assert (output / f"stills/{index:04d}" / f"{kind}.png").read_bytes() == anchor
            if image_count > 1:
                pair = output / f"transitions/{index:04d}-{(index + 1) % image_count:04d}"
                assert (pair / "0000" / f"{kind}.png").read_bytes() == anchor
                next_anchor = (output / f"loops/{(index + 1) % image_count:04d}/0000" / f"{kind}.png").read_bytes()
                assert (pair / "0015" / f"{kind}.png").read_bytes() == next_anchor
    def uncached(*args, **kwargs):
        pytest.fail("Reusable clip cache should avoid simulation and fitting")
    monkeypatch.setattr(cli, "_render_flow_image", uncached)
    second = tmp_path / "second"
    cli.build(arguments(second))
    queue = RenderQueue(tmp_path / "service", start_worker=False)
    assert queue._manifest(second, manifest["signature"]) is not None
    (second / "loops/0000/0003/color.png").unlink()
    assert queue._manifest(second, manifest["signature"]) is None


def test_inactive_animation_options_are_canonical_and_active_options_are_validated():
    assert normalize_options({"flow_fps": 24}) == normalize_options({})
    for key, value in (("flow_fps", 25), ("flow_loop_seconds", 0), ("flow_animated", 1)):
        with pytest.raises(ValueError):
            normalize_options({"style": "flow", "flow_animated": True, key: value})
    a = normalize_options({"style": "flow", "flow_animated": True})
    b = normalize_options({"style": "flow", "flow_animated": True, "flow_fps": 24})
    assert a != b
