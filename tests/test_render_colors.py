from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ascii_optimizer import optimizer
from ascii_optimizer.assets import load_target
from ascii_optimizer.optimizer import (Grid, _FrameObjective, _display_headroom_lab,
                                       _loss, _playback_color, _project_oklab_colors,
                                       _saturation_loss,
                                       fit_fixed_glyphs, render_grid, solve)
from ascii_optimizer.structure import fit_structure_colors, solve_structure
from ascii_optimizer.tone import (display_tone_lab, encode_headroom_lab,
                                  oklab_to_srgb, playback_rgb8,
                                  quantize_playback_lab, srgb_to_oklab)


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def test_solver_colors_and_background_are_exact_playback_rgb8_roundtrips():
    source = np.full((8, 8, 3), [0.9, 0.03, 0.15], dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.full((4, 4), 0.3),
                      np.ones((4, 4)))).astype(np.float32)
    grid = solve(encode_headroom_lab(srgb_to_oklab(source)), masks,
                 input_oklab=True, steps=3, background="#203040",
                 gradient_target_srgb=source)
    decoded = Grid(grid.glyph.copy(),
                   srgb_to_oklab(playback_rgb8(grid.color).astype(np.float32) / 255),
                   srgb_to_oklab(playback_rgb8(grid.background).astype(np.float32) / 255))
    np.testing.assert_array_equal(grid.color, decoded.color)
    np.testing.assert_array_equal(grid.background, decoded.background)
    np.testing.assert_array_equal(render_grid(grid, masks), render_grid(decoded, masks))


def test_playback_projection_keeps_gradients_across_rgb_byte_boundaries():
    color = torch.tensor([[0.8, 0.3, -0.3]], requires_grad=True)
    projected = _project_oklab_colors(color)
    projected.square().sum().backward()
    assert torch.isfinite(color.grad).all()
    assert color.grad.abs().sum() > 0
    np.testing.assert_allclose(projected.detach().numpy(),
                               quantize_playback_lab(color.detach().numpy()), atol=3e-6)


def test_rgb_ink_has_finite_gradients_at_black_white_and_saturated_boundaries():
    srgb = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1],
                         [0.21, 0.43, 0.67]], dtype=torch.float32, requires_grad=True)
    lab = _playback_color(srgb)
    lab.sum().backward()
    assert torch.isfinite(srgb.grad).all()
    assert srgb.grad[-1].abs().sum() > 0
    expected = srgb_to_oklab(np.rint(srgb.detach().numpy() * 255) / 255)
    np.testing.assert_allclose(lab.detach().numpy(), expected, atol=3e-7)


@pytest.mark.parametrize("wrong_hue", ["opposite", "perpendicular"])
def test_chrominance_correction_rejects_unrelated_hues_with_sufficient_saturation(wrong_hue):
    target = torch.tensor([0.5, 0.02, -0.03]).reshape(1, 3, 1, 1)
    shifted = target.clone()
    if wrong_hue == "opposite":
        shifted[:, 1:] *= -1
    else:
        shifted[:, 1] -= 2 * target[:, 2]
        shifted[:, 2] += 2 * target[:, 1]
    matching_loss = _saturation_loss(target, target, displayed_lab=target)
    shifted_loss = _saturation_loss(shifted, target, displayed_lab=shifted)
    assert matching_loss < 1e-10
    assert shifted_loss > 0.001


def test_light_ink_reward_boosts_shadows_and_tapers_at_white():
    colors = torch.zeros((5, 3))
    colors[:, 0] = torch.tensor([0.01, 0.1, 0.5, 0.9, 1])
    colors.requires_grad_()
    rewards = optimizer._ink_lightness_reward(colors, torch.ones(5))
    rewards.sum().backward()
    slopes = colors.grad[:, 0]
    assert (rewards[1:] > rewards[:-1]).all()
    assert (slopes[:-1] > slopes[1:]).all()
    assert slopes[1] > 0.5  # useful shadow lift
    assert slopes[-2] < 0.02  # little incentive to bleach highlights
    assert slopes[-1].abs() < 1e-6
    assert optimizer._ink_lightness_reward(colors, torch.zeros(5)).count_nonzero() == 0


@pytest.mark.parametrize("lift", [1.1, 1.8])
def test_chrominance_correction_preserves_saturation_when_lightness_increases(lift):
    target = torch.tensor([0.4, 0.04, -0.03]).reshape(1, 3, 1, 1)
    preserving_saturation = target * lift
    holding_chroma = target.clone()
    holding_chroma[:, :1] *= lift
    # Absolute chroma alone cannot distinguish these two brighter colors.
    # The correction must accept constant C/L and reject the washed-out color.
    assert _saturation_loss(target, target, displayed_lab=preserving_saturation) < 1e-10
    assert _saturation_loss(target, target, displayed_lab=holding_chroma) > 1e-5


@pytest.mark.parametrize("lightness_factor", [.25, 1., 1.8])
def test_saturation_preservation_penalizes_excess_and_deficit_symmetrically(lightness_factor):
    source = torch.tensor([.4, .04, -.03], dtype=torch.float64).reshape(1, 3, 1, 1)
    matching = source * lightness_factor
    low, high = matching.clone(), matching.clone()
    low[:, 1:] *= .5
    high[:, 1:] *= 1.5
    assert _saturation_loss(matching, source, displayed_lab=matching).item() < 1e-20
    deficit = _saturation_loss(low, source, displayed_lab=low)
    excess = _saturation_loss(high, source, displayed_lab=high)
    assert deficit.item() > 1e-5
    torch.testing.assert_close(deficit, excess, atol=1e-15, rtol=1e-12)


@pytest.mark.parametrize("source_lightness", [0., 1e-7, .001, .4])
def test_neutral_saturation_has_no_dark_source_artifact(source_lightness):
    source = torch.tensor([source_lightness, 0., 0.]).reshape(1, 3, 1, 1)
    gray = torch.tensor([.4, 0., 0.]).reshape(1, 3, 1, 1).requires_grad_()
    loss = _saturation_loss(gray, source, displayed_lab=gray)
    assert loss.item() == 0
    loss.sum().backward()
    assert torch.isfinite(gray.grad).all()
    tinted = gray.detach().clone()
    tinted[:, 1] = .02
    assert _saturation_loss(tinted, source, displayed_lab=tinted).item() > 0


def test_rgb8_black_ink_has_a_finite_gradient_toward_a_brighter_target():
    rgb = torch.zeros((1, 3), requires_grad=True)
    lab = optimizer._playback_color(rgb)
    assert lab.count_nonzero() == 0
    (lab[:, 0] - .2).square().sum().backward()
    assert torch.isfinite(rgb.grad).all()
    assert (rgb.grad < 0).all()


@pytest.mark.parametrize("color", [[0.65, 0.15, 0.03], [0.6, -0.1, -0.15]])
def test_display_saturation_gradient_tracks_gamut_compression(color):
    lab = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    tensor = _image_tensor(lab).requires_grad_()
    displayed = _display_headroom_lab(tensor)
    saturation = displayed[:, 1:].norm(dim=1) / displayed[:, 0]
    saturation.sum().backward()

    def playback_saturation(lightness_delta):
        shifted = lab.copy()
        shifted[..., 0] += lightness_delta
        visible = srgb_to_oklab(oklab_to_srgb(display_tone_lab(shifted)))
        return (np.linalg.norm(visible[..., 1:], axis=-1) / visible[..., 0]).item()

    epsilon = 0.005  # larger than playback's twelve-step gamut-map tolerance
    difference = (playback_saturation(epsilon) - playback_saturation(-epsilon)) / (2 * epsilon)
    assert tensor.grad[:, 0].item() == pytest.approx(difference, abs=0.015)


@pytest.mark.parametrize("invert_ink", [False, True])
def test_display_gradient_is_finite_at_black_white_and_primary_boundaries(invert_ink):
    srgb = torch.tensor([[[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]],
                        dtype=torch.float32, requires_grad=True)
    displayed = _display_headroom_lab(_playback_color(srgb).permute(0, 3, 1, 2),
                                     invert_ink=invert_ink)
    displayed.square().sum().backward()
    assert torch.isfinite(srgb.grad).all()


@pytest.mark.parametrize("fixed_glyphs", [False, True])
@pytest.mark.parametrize("bonus", [0.12, 0.5])
@pytest.mark.parametrize("refinement_sweeps", [0, 2])
def test_light_ink_bonus_preserves_highlight_saturation_and_lifts_shadows(fixed_glyphs, bonus, refinement_sweeps):
    # Include pastel highlights and the hue-specific brightness limits of all
    # three RGB primaries, plus a shadow whose brightness should still improve.
    swatches = np.array([[0.12, 0.05, 0.08], [0.3, 0.15, 0.22],
                         [1, 0.5, 0.7], [1, 0.85, 0.9],
                         [0.6, 0.8, 1], [0.85, 0.93, 1],
                         [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    source_lab = srgb_to_oklab(np.repeat(np.repeat(swatches[None], 4, 0), 4, 1))
    target = encode_headroom_lab(source_lab)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.full((4, 4), 0.5),
                      np.ones((4, 4)))).astype(np.float32)
    settings = dict(input_oklab=True, steps=36, saturation_loss_strength=20,
                    saturation_target_oklab=source_lab, gradient_loss_strength=0,
                    # Isolate the ink reward's legacy calibration from the
                    # default source-brightness preservation objective.
                    perceptual_loss=False)

    def displayed_cells(strength):
        if fixed_glyphs:
            grid = fit_fixed_glyphs(target, masks, np.full((1, len(swatches)), 3, dtype=np.uint8),
                                    background_oklab=np.zeros(3, dtype=np.float32),
                                    glyph_luminance_bonus=strength, **settings)
        else:
            grid = solve(target, masks, candidates=4, dither=0, background="#000000",
                         glyph_luminance_bonus=strength, refinement_sweeps=refinement_sweeps,
                         glyph_entropy_strength=0, **settings)
        displayed = srgb_to_oklab(oklab_to_srgb(display_tone_lab(render_grid(grid, masks))))
        return displayed.reshape(1, 4, len(swatches), 4, 3).mean((0, 1, 3))

    baseline, boosted = displayed_cells(0), displayed_cells(bonus)
    source = srgb_to_oklab(swatches)
    saturation = np.linalg.norm(boosted[2:, 1:], axis=-1) / boosted[2:, 0]
    expected = np.linalg.norm(source[2:, 1:], axis=-1) / source[2:, 0]
    # Near-neutral RGB8 pastels also need an absolute tolerance; their small
    # chroma makes a relative ratio sensitive to individual color bytes.
    np.testing.assert_allclose(saturation, expected, rtol=0.05, atol=0.001)
    # Empty glyphs must not collect the bonus and erase the deepest shadow.
    assert boosted[0, 0] > 0.05
    if fixed_glyphs:
        assert boosted[1, 0] > baseline[1, 0] + 0.015
    elif refinement_sweeps:
        # Discrete contextual search can already lift one shadow through its
        # glyph choice. The reward must still lift shadows without undoing it.
        assert (boosted[:2, 0] >= baseline[:2, 0] - 0.003).all()
        assert (boosted[:2, 0] - baseline[:2, 0]).max() > 0.015
    else:
        # Geometry can change with the reward; brighter ink need not increase
        # a sparse cell's mean displayed lightness. Keep both shadows visible.
        assert (boosted[:2, 0] > 0.05).all()


def _sparse_masks():
    masks = np.zeros((4, 4, 4), dtype=np.float32)
    masks[1] = np.eye(4)
    masks[2, :, 1:3] = 1
    masks[3] = np.indices((4, 4)).sum(0) % 2
    return masks


@pytest.mark.parametrize("fixed_glyphs", [False, True])
def test_neutral_source_stays_neutral_with_light_ink_bonus(fixed_glyphs):
    source_lab = srgb_to_oklab(np.full((16, 16, 3), 0.5, dtype=np.float32))
    target = encode_headroom_lab(source_lab)
    masks = _sparse_masks()
    settings = dict(input_oklab=True, steps=36, glyph_luminance_bonus=0.12,
                    saturation_target_oklab=source_lab, gradient_target_oklab=source_lab,
                    gradient_loss_strength=1.5)
    if fixed_glyphs:
        grid = fit_fixed_glyphs(target, masks, np.ones((4, 4), dtype=np.uint8),
                                background_oklab=np.zeros(3, dtype=np.float32), **settings)
    else:
        grid = solve(target, masks, candidates=4, dither=0.014, background="#000000", **settings)
    displayed = srgb_to_oklab(oklab_to_srgb(display_tone_lab(render_grid(grid, masks))))
    assert np.abs(displayed[..., 1:]).max() < 0.001


def test_faint_purple_source_does_not_turn_yellow_during_gamut_constrained_solve():
    source_lab = srgb_to_oklab(np.full((16, 16, 3), [0.25, 0.24, 0.27], dtype=np.float32))
    masks = _sparse_masks()
    grid = solve(encode_headroom_lab(source_lab), masks, input_oklab=True,
                 steps=36, candidates=4, dither=0.014, background="#000000",
                 glyph_luminance_bonus=0.12, saturation_target_oklab=source_lab,
                 gradient_target_oklab=source_lab, gradient_loss_strength=1.5)
    displayed = srgb_to_oklab(oklab_to_srgb(display_tone_lab(render_grid(grid, masks))))
    cells = displayed.reshape(4, 4, 4, 4, 3).mean((1, 3))
    # Empty dithered cells display the neutral canvas, whose hue is undefined.
    visible = masks[grid.glyph].sum((2, 3)) > 0
    assert visible.any()
    assert (cells[..., 1][visible] > 0).all()
    assert (cells[..., 2][visible] < 0).all()
    assert np.sqrt(np.mean((cells[..., 1:] - source_lab[0, 0, 1:]) ** 2)) < 0.01


@pytest.mark.parametrize("invert_ink", [False, True])
def test_display_objective_uses_the_shaders_gamut_mapped_colors(invert_ink):
    colors = np.array([[[1, 0, 0], [0, 0, 1], [0.2, 0.8, 0.1],
                        [0, 0, 0], [1, 1, 1]]], dtype=np.float32)
    lab = srgb_to_oklab(colors)
    expected = srgb_to_oklab(oklab_to_srgb(display_tone_lab(lab, invert_ink=invert_ink)))
    actual = _display_headroom_lab(_image_tensor(lab), invert_ink=invert_ink)
    np.testing.assert_allclose(actual.squeeze(0).permute(1, 2, 0).numpy(),
                               expected, atol=3e-6)


def test_fixed_glyph_fit_retains_paths_and_steps_improve_the_retained_color_objective():
    source = np.random.default_rng(8).random((8, 12, 3), dtype=np.float32)
    source_lab = srgb_to_oklab(source)
    encoded = encode_headroom_lab(source_lab)
    masks = np.zeros((3, 4, 3), dtype=np.float32)
    masks[1, :3] = 0.6
    masks[2] = 0.3
    glyphs = np.array([[1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    settings = dict(background_oklab=np.zeros(3, dtype=np.float32), input_oklab=True,
                    gradient_loss_strength=1.5, gradient_target_oklab=source_lab)
    initial = fit_fixed_glyphs(encoded, masks, glyphs, steps=0, **settings)
    fitted = fit_fixed_glyphs(encoded, masks, glyphs, steps=12, **settings)

    def score(grid):
        spatial = optimizer._make_spatial_metrics(
            encoded, source_srgb=None, source_oklab=source_lab, input_oklab=True,
            pad=(0, 0, 0, 0), cell_height=4, device="cpu", viewing_ppd=None,
            masks=masks, background=torch.zeros(3), invert_ink=False)
        objective = _FrameObjective(_image_tensor(encoded), "default", (4, 3), False,
                                    1.5, 0, _image_tensor(source_lab), spatial_metrics=spatial)
        return objective(_image_tensor(render_grid(grid, masks)))

    np.testing.assert_array_equal(fitted.glyph, glyphs)
    np.testing.assert_array_equal(fitted.color, quantize_playback_lab(fitted.color))
    assert score(fitted) < score(initial) * 0.99
    assert not np.array_equal(playback_rgb8(initial.color), playback_rgb8(fitted.color))


def test_cached_target_losses_preserve_values_and_compute_target_sobel_once(monkeypatch):
    target = torch.rand((1, 3, 8, 8))
    rendered = target * 0.8
    expected = _loss(rendered, target, gradient_loss_strength=0.7)
    calls = []
    original = optimizer._sobel_gradients

    def record(image):
        calls.append(image)
        return original(image)

    monkeypatch.setattr(optimizer, "_sobel_gradients", record)
    objective = _FrameObjective(target, "default", (4, 4), False, 0.7, 0,
                                gradient_target=target)
    torch.testing.assert_close(objective(rendered), expected)
    torch.testing.assert_close(objective(rendered), expected)
    assert len(calls) == 3  # one target gradient, two rendered gradients


def test_structure_fixed_color_fit_matches_still_ink_without_reselecting_glyphs():
    source = np.random.default_rng(19).random((8, 8, 3), dtype=np.float32)
    original = source.copy()
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    background = srgb_to_oklab(np.array([0.06, 0.02, 0.08], dtype=np.float32))
    still = solve_structure(source, masks, background, dither=0)
    path_glyphs = (still.glyph + 1) % len(masks)
    fitted = fit_structure_colors(source, masks, path_glyphs, background)
    np.testing.assert_array_equal(fitted.glyph, path_glyphs)
    np.testing.assert_array_equal(fitted.color, still.color)
    np.testing.assert_array_equal(fitted.background, still.background)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("setting", ["gradient_loss_strength", "gradient_shortlist_strength",
                                     "saturation_loss_strength", "glyph_luminance_bonus",
                                     "glyph_error_movement_strength"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_solver_rejects_nonfinite_objective_weights(setting, value):
    with pytest.raises(ValueError, match=setting):
        solve(np.zeros((4, 4, 3), dtype=np.float32),
              np.zeros((1, 4, 4), dtype=np.float32), **{setting: value})


def test_error_movement_preserves_local_palette_instead_of_averaging_it():
    # Two colors with the same neighborhood mean: averaging them destroys
    # palette information; exchanging their positions conserves it.
    source = torch.tensor([0.65, 0.12, -0.10]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
    source[:, 1:, :, 2:] *= -1
    directions = _FrameObjective(source, "default", (2, 2), False, 0, 0,
                                 glyph_error_movement_strength=1).error_movement_directions
    samples = optimizer._error_movement_samples(source, (2, 2), directions)
    swapped = source.flip(-1)
    torch.testing.assert_close(optimizer._error_movement_samples(swapped, (2, 2), directions), samples)
    averaged = source.mean((-2, -1), keepdim=True).expand_as(source).clone().requires_grad_()
    loss = (optimizer._error_movement_samples(averaged, (2, 2), directions) - samples).square().mean()
    assert loss > 0.001
    loss.backward()
    assert torch.isfinite(averaged.grad).all()
    assert averaged.grad.abs().sum() > 0


def test_error_movement_cannot_move_color_outside_neighboring_glyphs():
    source = torch.zeros((1, 3, 4, 12))
    source[:, 0] = 0.5
    source[:, 1, :, :2] = 0.2
    directions = torch.eye(3)
    samples = optimizer._error_movement_samples(source, (2, 2), directions)
    moved = source.roll(8, dims=-1)
    assert (optimizer._error_movement_samples(moved, (2, 2), directions) - samples).square().mean() > 0


@pytest.mark.parametrize("invert_ink", [False, True])
def test_error_movement_compares_displayed_colors_to_original_source(invert_ink):
    source = np.full((4, 4, 3), [0.7, 0.3, 0.4], np.float32)
    target = _image_tensor(srgb_to_oklab(source))
    encoded = _image_tensor(encode_headroom_lab(srgb_to_oklab(source), invert_ink=invert_ink))
    baseline = _FrameObjective(encoded, "default", (2, 2), invert_ink, 0, 0)
    movement = _FrameObjective(encoded, "default", (2, 2), invert_ink, 0, 0,
                               glyph_error_movement_strength=4, error_movement_target=target)
    torch.testing.assert_close(movement(encoded), baseline(encoded), atol=1e-8, rtol=0)


def test_disabled_error_movement_is_identical_and_skips_transport(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Disabled error movement must not compute transport")
    monkeypatch.setattr(optimizer, "_error_movement_samples", unexpected)
    source = np.random.default_rng(19).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    baseline = solve(source, masks, steps=2)
    disabled = solve(source, masks, steps=2, glyph_error_movement_strength=0)
    np.testing.assert_array_equal(baseline.glyph, disabled.glyph)
    np.testing.assert_array_equal(baseline.color, disabled.color)


@pytest.mark.parametrize("shape", [(3, 3), (3, 9), (9, 3)])
def test_error_movement_handles_partial_and_single_cell_grids(shape):
    source = np.random.default_rng(3).random((*shape, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    grid = solve(source, masks, steps=1, glyph_error_movement_strength=4)
    assert grid.glyph.shape == tuple((side + 3) // 4 for side in shape)
    assert np.isfinite(grid.color).all()


def test_error_movement_rejects_negative_strength_and_mismatched_target():
    source = np.zeros((4, 4, 3), np.float32)
    masks = np.ones((1, 4, 4), np.float32)
    with pytest.raises(ValueError, match="glyph_error_movement_strength"):
        solve(source, masks, glyph_error_movement_strength=-1)
    with pytest.raises(ValueError, match="target must match"):
        solve(source, masks, glyph_error_movement_strength=1,
              error_movement_target_oklab=np.zeros((3, 3, 3), np.float32))


@pytest.mark.parametrize("detail_loss", ["default", "local_color"])
@pytest.mark.parametrize("with_endpoints", [False, True])
def test_unshortlisted_solve_includes_every_glyph_and_bypasses_ranking(monkeypatch, detail_loss, with_endpoints):
    source = np.random.default_rng(5).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.fliplr(np.eye(4)), np.ones((4, 4)))).astype(np.float32)

    def unexpected(*args, **kwargs):
        raise AssertionError("All-glyph optimization must bypass shortlist fitting")

    monkeypatch.setattr(optimizer, "_candidate_fit", unexpected)
    original_softmax = torch.softmax
    counts = []

    def record(logits, *args, **kwargs):
        counts.append(logits.shape[1])
        return original_softmax(logits, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", record)
    endpoint_args = {}
    if with_endpoints:
        endpoint = Grid(np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.float32))
        endpoint_args = dict(previous=endpoint, destination=endpoint, progress=0.5)
    grid = solve(source, masks, use_shortlist=False, candidates=1, steps=2,
                 detail_loss=detail_loss, glyph_error_movement_strength=4, **endpoint_args)
    assert counts == [len(masks)] * 2  # endpoint glyphs are already present
    assert np.isfinite(grid.color).all()
    np.testing.assert_array_equal(grid.color, quantize_playback_lab(grid.color))


def test_unshortlisted_solve_ignores_candidate_limit_and_shortlist_gradient_strength():
    source = np.random.default_rng(5).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    first = solve(source, masks, steps=2, use_shortlist=False, candidates=1,
                  gradient_shortlist_strength=0)
    second = solve(source, masks, steps=2, use_shortlist=False, candidates=99,
                   gradient_shortlist_strength=100)
    np.testing.assert_array_equal(first.glyph, second.glyph)
    np.testing.assert_array_equal(first.color, second.color)


def test_default_full_bank_matches_approved_mode_and_requires_a_boolean(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Default rendering must not rank a glyph shortlist")
    monkeypatch.setattr(optimizer, "_candidate_fit", unexpected)
    source = np.random.default_rng(5).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    default = solve(source, masks, steps=2)
    explicit = solve(source, masks, steps=2, use_shortlist=False)
    np.testing.assert_array_equal(default.glyph, explicit.glyph)
    np.testing.assert_array_equal(default.color, explicit.color)
    with pytest.raises(ValueError, match="use_shortlist"):
        solve(source, masks, use_shortlist="false")


@pytest.mark.parametrize("distribution", ["uniform", "gumbel"])
def test_post_logit_noise_is_seeded_and_valid_for_each_distribution(distribution):
    settings = dict(rows=24, cols=24, candidates=3,
                    distribution=distribution, device="cpu", dtype=torch.float32)
    first = optimizer._post_logit_noise(seed=19, **settings)
    torch.testing.assert_close(first, optimizer._post_logit_noise(seed=19, **settings))
    assert not torch.equal(first, optimizer._post_logit_noise(seed=20, **settings))
    assert first.shape == (576, 3)
    assert torch.isfinite(first).all()
    assert not torch.equal(first[:, 0], first[:, 1])
    if distribution == "uniform":
        assert first.min() >= -0.5 and first.max() <= 0.5


def test_unit_scale_gumbel_argmax_samples_the_logits_softmax_distribution():
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    noise = optimizer._post_logit_noise(200, 200, 3, 61, "gumbel",
                                        device="cpu", dtype=torch.float32)
    selected = (probabilities.log() + noise).argmax(1)
    observed = torch.bincount(selected, minlength=3).float() / len(selected)
    # A seeded 40k sample keeps this check stable while rejecting correlated,
    # non-Gumbel noise and incorrect scale/sign transforms.
    torch.testing.assert_close(observed, probabilities, atol=0.01, rtol=0)


def test_post_logit_noise_rejects_unknown_distributions():
    with pytest.raises(ValueError, match="uniform.*gumbel"):
        optimizer._post_logit_noise(2, 2, 3, 0, "normal",
                                    device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("grayscale", [False, True])
def test_fit_crop_preserves_image_extent_and_black_borders_in_both_color_modes(tmp_path, grayscale):
    path = tmp_path / "wide.png"
    Image.new("RGB", (4, 2), (220, 70, 10)).save(path)
    fitted = load_target(path, (4, 4), "fit", grayscale)
    assert fitted.shape == (4, 4, 3)
    np.testing.assert_array_equal(fitted[[0, 3]], np.zeros((2, 4, 3)))
    expected = np.array([108] * 3 if grayscale else [220, 70, 10], dtype=np.float32) / 255
    np.testing.assert_allclose(fitted[1:3], np.broadcast_to(expected, (2, 4, 3)), atol=1e-7)
    # Fill occupies the entire requested image instead of retaining fit borders.
    filled = load_target(path, (4, 4), "fill", grayscale)
    np.testing.assert_allclose(filled, np.broadcast_to(expected, (4, 4, 3)), atol=1e-7)


def test_profiled_image_is_converted_to_srgb_before_color_solving(tmp_path):
    from io import BytesIO
    from PIL import ImageCms

    # LAB has a different channel encoding; treating its bytes as RGB produces
    # a strong hue shift. A tagged TIFF exercises the actual CMS conversion.
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    original = Image.new("LAB", (4, 4), (135, 180, 110))
    path = tmp_path / "profiled.tif"
    original.save(path, icc_profile=profile)
    expected = ImageCms.profileToProfile(original, ImageCms.ImageCmsProfile(BytesIO(profile)),
                                        ImageCms.createProfile("sRGB"), outputMode="RGB")
    actual = load_target(path, (4, 4), "fill", False)
    np.testing.assert_array_equal(np.rint(actual * 255).astype(np.uint8), np.asarray(expected))


def test_invalid_color_profile_keeps_image_readable(tmp_path):
    path = tmp_path / "invalid-profile.png"
    Image.new("RGB", (4, 4), (90, 120, 150)).save(path, icc_profile=b"invalid ICC")
    actual = load_target(path, (4, 4), "fill", False)
    np.testing.assert_allclose(actual[0, 0], np.array([90, 120, 150]) / 255, atol=1e-7)
