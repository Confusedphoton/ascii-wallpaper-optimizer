from __future__ import annotations

import numpy as np
import pytest
import torch

from ascii_optimizer import optimizer
from ascii_optimizer.diagnostics import SolverTrace
from ascii_optimizer.perceptual import SpatialMetrics, luminance


@pytest.mark.parametrize("invert", [False, True])
def test_reference_uses_actual_atlas_range_and_keeps_highlight_differences(invert):
    source = np.broadcast_to(np.linspace(0.5, 1, 32, dtype=np.float32)[None, :, None], (32, 32, 3)).copy()
    masks = np.stack((np.zeros((8, 8)), np.eye(8), np.tri(8))).astype(np.float32)
    background = torch.tensor([0.0, 0.0, 0.0])
    model = optimizer._make_spatial_metrics(
        source, source_srgb=source, source_oklab=None, input_oklab=False,
        pad=(0, 0, 0, 0), cell_height=8, device="cpu", viewing_ppd=60,
        masks=masks, background=background, invert_ink=invert,
    )
    low, high = model.reference_range
    alpha = torch.from_numpy(masks)[:, None]
    atlas = alpha * torch.tensor([1.0, 0.0, 0.0])[None, :, None, None]
    shown = optimizer._display_headroom_lab(atlas, invert_ink=invert)
    actual = optimizer._oklab_to_rgb(shown.permute(0, 2, 3, 1)).mean((1, 2))
    y = actual @ torch.tensor([0.2126, 0.7152, 0.0722])
    torch.testing.assert_close(low if invert else high, actual[y.argmin() if invert else y.argmax()])
    assert torch.all(high > low)
    # No source-highlight plateau is introduced by the global range mapping.
    assert (luminance(model.source_rgb)[..., 1:] > luminance(model.source_rgb)[..., :-1]).all()


def test_chromatic_diagonal_contours_choose_aligned_rasters_and_are_permutation_invariant():
    y, x = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
    wave = torch.sin((x - y) * 0.22) * 0.035
    source = torch.stack((0.2 + wave, 0.2 - wave * (0.2126 / 0.7152), torch.full_like(wave, 0.2)))[None]
    torch.testing.assert_close(luminance(source), torch.full((1, 1, 96, 96), 0.2))
    model = SpatialMetrics(source, ppd=60, cell_height=16)
    masks = torch.stack((torch.eye(16), torch.eye(16).flip(1), torch.ones(16, 16)))
    costs = model.glyph_stroke_costs(masks).reshape(6, 6, 3)[1:-1, 1:-1]
    assert (costs[..., 0] < costs[..., 1]).all()
    assert (costs[..., 0] < costs[..., 2]).all()
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model.glyph_stroke_costs(masks[permutation]),
                               model.glyph_stroke_costs(masks)[:, permutation])
    flat = SpatialMetrics(torch.full_like(source, 0.2), ppd=60, cell_height=16)
    assert flat.glyph_stroke_costs(masks).abs().max() < 1e-8


def test_thin_and_dense_candidates_fit_the_same_attainable_mean_with_different_ink():
    masks = torch.stack((torch.eye(16), torch.tril(torch.ones(16, 16))))
    source = torch.full((1, 3, 32, 32), 0.03)
    model = SpatialMetrics(source, ppd=60, cell_height=16)
    colors = optimizer._spatial_ink_fit(model, masks, torch.zeros(3), invert_ink=False)
    assert (colors[:, 0, 0] > colors[:, 1, 0]).all()
    pixels = masks[None, :, None] * colors[:, :, :, None, None]
    shown = optimizer._display_headroom_lab(pixels.flatten(0, 1))
    rgb = optimizer._oklab_to_rgb(shown.permute(0, 2, 3, 1)).mean((1, 2))
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.03), atol=0.001, rtol=0)


def test_relative_chroma_preserves_pastels_without_coloring_neutral_or_black_sources():
    target = torch.tensor([0.8, 0.015, -0.01]).reshape(1, 3, 1, 1)
    faithful = target * 0.7
    white = faithful.clone()
    white[:, 1:] = 0
    wrong_hue = faithful.clone()
    wrong_hue[:, 1:] *= -1
    assert optimizer._relative_chroma_loss(faithful, target).item() < 1e-8
    assert optimizer._relative_chroma_loss(white, target).item() > 0.9
    assert optimizer._relative_chroma_loss(wrong_hue, target).item() > 3.6
    for lightness in (0.0, 0.5):
        neutral = torch.tensor([lightness, 0, 0]).reshape(1, 3, 1, 1)
        candidate = white.clone().requires_grad_()
        loss = optimizer._relative_chroma_loss(candidate, neutral).sum()
        loss.backward()
        assert loss.item() == 0
        assert torch.isfinite(candidate.grad).all()
    candidate = white.clone().requires_grad_()
    optimizer._relative_chroma_loss(candidate, target).sum().backward()
    assert torch.isfinite(candidate.grad).all()
    assert (candidate.grad[:, 1:] * target[:, 1:]).sum() < 0


@pytest.mark.parametrize("perceptual", [False, True])
def test_dithered_terminal_layout_is_returned_even_when_initialization_scores_better(monkeypatch, perceptual):
    masks = np.stack((np.ones((4, 4)), np.zeros((4, 4)))).astype(np.float32)
    source = np.full((8, 8, 3), 0.7, np.float32)
    def force_empty(rows, cols, candidates, *args, **kwargs):
        noise = torch.zeros(rows * cols, candidates, device=kwargs["device"], dtype=kwargs["dtype"])
        noise[:, 1] = 1000
        return noise
    monkeypatch.setattr(optimizer, "_post_logit_noise", force_empty)
    trace = SolverTrace(include_replacements=False)
    grid = optimizer.solve(source, masks, steps=2, dither=0.01, polish_steps=0,
                           glyph_entropy_strength=0, perceptual_loss=perceptual,
                           postprocess=True, diagnostics=trace)
    states = {r["stage"]: r["tensors"] for r in trace.records}
    assert states["initial"]["loss"] < states["after_dither"]["loss"]
    assert (grid.glyph == 1).all()
    torch.testing.assert_close(states["after_dither"]["glyph"], states["pre_refinement"]["glyph"])
    assert "solver_best" not in states


@pytest.mark.parametrize("invert", [False, True])
def test_default_style_returns_exact_adam_terminal_despite_saved_dither(monkeypatch, invert):
    from ascii_optimizer.cli import solve_style
    from ascii_optimizer.tone import playback_rgb8

    source = np.random.default_rng(12).uniform(0.1, 0.8, (8, 8, 3)).astype(np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError("Normal renders must not apply post-Adam noise")
    monkeypatch.setattr(optimizer, "_post_logit_noise", forbidden)
    trace = SolverTrace(include_replacements=False)
    grid = solve_style(source, masks, style="headroom", invert_ink=invert,
                       background="auto", seed=0, steps=3, dither=0.016,
                       structure_dither=0.06, device="cpu", diagnostics=trace)
    states = {r["stage"]: r["tensors"] for r in trace.records}
    terminal = states["solver_final"]
    np.testing.assert_array_equal(grid.glyph.ravel(), terminal["glyph"].numpy())
    np.testing.assert_array_equal(playback_rgb8(grid.color.reshape(-1, 3)),
                                  playback_rgb8(terminal["color"].numpy()))
    torch.testing.assert_close(states["final"]["glyph"], terminal["glyph"])
    torch.testing.assert_close(states["final"]["color"], terminal["color"])
    assert "after_dither" not in states and "final_polish" not in states
