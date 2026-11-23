"""Tests for exact SDF glyph atlases, stroke-weight optimization, and pluggable renderers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ascii_optimizer import optimizer, renderers
from ascii_optimizer.assets import (
    _bounded_edt,
    decode_stroke_weight_u8,
    encode_stroke_weight_u8,
    glyph_sdf_atlas,
    resolve_font,
    sdf_to_mask,
)
from ascii_optimizer.optimizer import Grid


def test_bounded_edt_matches_exact_euclidean_distances():
    feature = np.zeros((1, 11, 11), dtype=np.bool_)
    feature[0, 5, 5] = True
    dist = _bounded_edt(feature, radius=6)[0]
    y, x = np.mgrid[:11, :11]
    expected = np.minimum(np.hypot(y - 5, x - 5), 6.0).astype(np.float32)
    np.testing.assert_allclose(dist, expected, atol=1e-5)


def test_stroke_weight_byte_encoding_preserves_neutral_and_endpoints():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    encoded = encode_stroke_weight_u8(values)
    assert encoded.tolist() == [1, 64, 128, 192, 255]
    decoded = decode_stroke_weight_u8(encoded)
    assert decoded[2] == 0.0
    assert decoded[0] == pytest.approx(-1.0)
    assert decoded[4] == pytest.approx(1.0)


def test_sdf_atlas_preserves_empty_space_and_monotone_stroke_dilation():
    atlas, spread = glyph_sdf_atlas(resolve_font("DejaVu Sans Mono"), [" ", "/", "#"], (8, 16))
    assert spread >= 2.0
    assert atlas.shape == (3, 16, 8)
    assert np.all(atlas[0] == 0.0)
    # Empty space stays empty even at maximum stroke weight.
    assert sdf_to_mask(atlas[0], 1.0, spread=spread).max() == 0.0
    # Non-empty glyphs strictly thicken as weight increases.
    thin = sdf_to_mask(atlas[1], -0.8, spread=spread).mean()
    neutral = sdf_to_mask(atlas[1], 0.0, spread=spread).mean()
    bold = sdf_to_mask(atlas[1], 0.8, spread=spread).mean()
    assert 0.0 < thin < neutral < bold < 1.0


def test_sdf_to_mask_numpy_torch_parity_and_weight_gradient():
    atlas, spread = glyph_sdf_atlas(resolve_font("DejaVu Sans Mono"), [" ", "/"], (8, 16))
    weight_np = np.array([[-0.6], [0.7]], dtype=np.float32)
    np_mask = sdf_to_mask(atlas, weight_np[:, :, None], spread=spread)
    weight_t = torch.tensor(weight_np, requires_grad=True)
    torch_mask = sdf_to_mask(torch.from_numpy(atlas), weight_t[:, :, None], spread=spread)
    np.testing.assert_allclose(torch_mask.detach().numpy(), np_mask, atol=1e-6)
    torch_mask.sum().backward()
    assert torch.isfinite(weight_t.grad).all()
    assert weight_t.grad[0].item() == 0.0  # empty space has zero gradient
    assert weight_t.grad[1].item() > 0.0   # thicker weight increases ink coverage


def test_sdf_solver_adapts_stroke_weight_to_brightness():
    atlas, spread = glyph_sdf_atlas(resolve_font("DejaVu Sans Mono"), [" ", "/"], (8, 16))
    source = np.zeros((16, 16, 3), dtype=np.float32)
    source[:, :8] = 0.08
    source[:, 8:] = 0.92
    grid = optimizer.solve(source, atlas, steps=16, background="#000000",
                           gradient_loss_strength=0.0, glyph_entropy_strength=0.0,
                           sdf_spread=spread)
    assert grid.weight is not None
    assert grid.weight.shape == (1, 2)
    # Bright cell should dilate stroke weight higher than the dark cell.
    assert grid.weight[0, 1] > grid.weight[0, 0]


def test_renderer_registry_allows_pluggable_styles_without_touching_cli():
    called = {}

    def custom_renderer(target, masks, *, solve_background, **_kwargs):
        called["shape"] = target.shape
        rows = target.shape[0] // masks.shape[1]
        cols = target.shape[1] // masks.shape[2]
        return Grid(np.zeros((rows, cols), dtype=np.uint8),
                    np.zeros((rows, cols, 3), dtype=np.float32),
                    solve_background)

    renderers.register("test_custom_lab", custom_renderer)
    try:
        assert "test_custom_lab" in renderers.available_styles()
        grid = renderers.solve_style(np.zeros((8, 8, 3), dtype=np.float32),
                                     np.ones((1, 4, 4), dtype=np.float32),
                                     style="test_custom_lab")
        assert called["shape"] == (8, 8, 3)
        assert grid.glyph.shape == (2, 2)
    finally:
        renderers.RENDERERS.pop("test_custom_lab", None)
