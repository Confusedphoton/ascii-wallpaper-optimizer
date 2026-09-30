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
    assert (cells[..., 1] > 0).all()
    assert (cells[..., 2] < 0).all()
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
        return _loss(_image_tensor(render_grid(grid, masks)), _image_tensor(encoded),
                     gradient_target_lab=_image_tensor(source_lab), gradient_loss_strength=1.5)

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
                                     "saturation_loss_strength", "glyph_luminance_bonus"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_solver_rejects_nonfinite_objective_weights(setting, value):
    with pytest.raises(ValueError, match=setting):
        solve(np.zeros((4, 4, 3), dtype=np.float32),
              np.zeros((1, 4, 4), dtype=np.float32), **{setting: value})


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
