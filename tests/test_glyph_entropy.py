from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ascii_optimizer import optimizer
from ascii_optimizer.cli import solve_style
from ascii_optimizer.defaults import DEFAULT_GLYPH_ENTROPY_STRENGTH


def test_marginal_entropy_rewards_different_hard_glyphs_without_softening_cells():
    weights = torch.ones((4, 1))
    diverse = optimizer._glyph_distribution(weights, torch.tensor([[0], [1], [0], [1]]), 3)
    collapsed = optimizer._glyph_distribution(weights, torch.zeros((4, 1), dtype=torch.long), 3)
    torch.testing.assert_close(diverse, torch.tensor([0.5, 0.5, 0.0]))
    assert optimizer._negative_glyph_entropy(diverse).item() == pytest.approx(-math.log(2))
    assert optimizer._negative_glyph_entropy(collapsed).item() == 0


def test_distribution_uses_glyph_ids_and_merges_duplicate_shortlist_candidates():
    logits = torch.tensor([[2.0, 0.0, -1.0], [1.0, 0.0, 2.0]], requires_grad=True)
    weights = logits.softmax(1)
    indices = torch.tensor([[2, 0, 2], [0, 1, 0]])
    probabilities = optimizer._glyph_distribution(weights, indices, 4)
    expected = torch.stack((weights[0, 1] + weights[1, 0] + weights[1, 2],
                            weights[1, 1], weights[0, 0] + weights[0, 2],
                            weights.new_zeros(()))) / 2
    torch.testing.assert_close(probabilities, expected)
    loss = optimizer._negative_glyph_entropy(probabilities)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(optimizer._glyph_distribution(weights[:, permutation], indices[:, permutation], 4),
                               probabilities)


def test_entropy_gradient_moves_probability_away_from_the_dominant_glyph():
    logits = torch.tensor([[2.0, 0.0]] * 4, requires_grad=True)
    indices = torch.tensor([[0, 1]] * 4)
    optimizer._negative_glyph_entropy(optimizer._glyph_distribution(logits.softmax(1), indices, 2)).backward()
    assert (logits.grad[:, 0] > 0).all()
    assert (logits.grad[:, 1] < 0).all()


def test_discrete_move_cost_matches_exact_entropy_including_unused_and_same_glyphs():
    glyph = torch.tensor([0, 0, 0, 1])
    indices = torch.arange(3).expand(4, -1)
    counts = torch.bincount(glyph, minlength=3).float()
    costs = optimizer._glyph_entropy_move_cost(glyph, indices, counts)
    before = optimizer._negative_glyph_entropy(counts / len(glyph))
    for cell in range(4):
        for candidate in range(3):
            trial = glyph.clone()
            trial[cell] = candidate
            after = optimizer._negative_glyph_entropy(torch.bincount(trial, minlength=3).float() / len(glyph))
            torch.testing.assert_close(costs[cell, candidate], after - before)


def test_zero_entropy_strength_skips_histogram_work_and_is_deterministic(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled entropy must skip histogram work")
    for name in ("_glyph_distribution", "_negative_glyph_entropy", "_glyph_entropy_move_cost"):
        monkeypatch.setattr(optimizer, name, unexpected)
    source = np.random.default_rng(5).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    default = optimizer.solve(source, masks, steps=2, glyph_entropy_strength=0)
    disabled = optimizer.solve(source, masks, steps=2, glyph_entropy_strength=0)
    for field in ("glyph", "color", "background"):
        np.testing.assert_array_equal(getattr(default, field), getattr(disabled, field))


def test_discrete_refinement_uses_entropy_even_without_soft_optimization():
    source = np.full((16, 16, 3), 0.5, np.float32)
    masks = np.ones((2, 4, 4), np.float32)
    baseline = optimizer.solve(source, masks, steps=0, dither=0, glyph_entropy_strength=0, refinement_sweeps=2,
                               perceptual_loss=False)
    regularized = optimizer.solve(source, masks, steps=0, dither=0, glyph_entropy_strength=0.02, refinement_sweeps=2,
                                  perceptual_loss=False)
    assert np.unique(baseline.glyph).size == 1
    assert np.unique(regularized.glyph).size == 2
    assert (regularized.glyph == 0).mean() == pytest.approx(0.5)
    np.testing.assert_array_equal(optimizer.render_grid(baseline, masks), optimizer.render_grid(regularized, masks))


def test_single_glyph_entropy_is_finite_and_cannot_change_the_output():
    source = np.full((3, 5, 3), 0.5, np.float32)
    masks = np.ones((1, 4, 4), np.float32)
    baseline = optimizer.solve(source, masks, steps=2, dither=0)
    regularized = optimizer.solve(source, masks, steps=2, dither=0, glyph_entropy_strength=1)
    np.testing.assert_array_equal(baseline.glyph, regularized.glyph)
    np.testing.assert_array_equal(baseline.color, regularized.color)


def test_adopted_default_matches_explicit_strength():
    source = np.random.default_rng(8).random((8, 8, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    default = optimizer.solve(source, masks, steps=2)
    explicit = optimizer.solve(source, masks, steps=2, glyph_entropy_strength=DEFAULT_GLYPH_ENTROPY_STRENGTH)
    assert DEFAULT_GLYPH_ENTROPY_STRENGTH == 0.01
    np.testing.assert_array_equal(default.glyph, explicit.glyph)
    np.testing.assert_array_equal(default.color, explicit.color)


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf")])
def test_invalid_entropy_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="glyph_entropy_strength"):
        optimizer.solve(np.zeros((4, 4, 3), np.float32), np.ones((1, 4, 4), np.float32),
                        glyph_entropy_strength=strength)


@pytest.mark.parametrize("use_shortlist", [False, True])
def test_entropy_solve_handles_padding_and_duplicate_transition_endpoints(use_shortlist):
    source = np.random.default_rng(2).random((5, 7, 3), dtype=np.float32)
    masks = np.stack((np.zeros((4, 4)), np.eye(4), np.ones((4, 4)))).astype(np.float32)
    endpoint = optimizer.Grid(np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.float32))
    settings = dict(steps=2, dither=0, candidates=2, use_shortlist=use_shortlist,
                    previous=endpoint, destination=endpoint, progress=0.5, glyph_entropy_strength=0.02)
    first = optimizer.solve(source, masks, **settings)
    second = optimizer.solve(source, masks, **settings)
    assert first.glyph.shape == (2, 2)
    assert np.isfinite(first.color).all()
    np.testing.assert_array_equal(first.glyph, second.glyph)
    np.testing.assert_array_equal(first.color, second.color)


def test_style_api_forwards_entropy_and_rejects_structure_selection(monkeypatch):
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        return optimizer.Grid(np.zeros((1, 1), np.uint8), np.zeros((1, 1, 3), np.float32))
    monkeypatch.setattr("ascii_optimizer.cli.solve", capture)
    settings = dict(invert_ink=False, background="auto", seed=0, steps=1,
                    dither=0, structure_dither=0, device="cpu", glyph_entropy_strength=0.02)
    source, masks = np.zeros((4, 4, 3), np.float32), np.ones((1, 4, 4), np.float32)
    solve_style(source, masks, style="headroom", **settings)
    assert captured["glyph_entropy_strength"] == 0.02
    with pytest.raises(ValueError, match="glyph_entropy_strength"):
        solve_style(source, masks, style="structure", **settings)
