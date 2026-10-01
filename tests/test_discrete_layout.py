from __future__ import annotations

import numpy as np
import pytest
import torch

from ascii_optimizer import optimizer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cell_composition_matches_expanded_frame_pixels_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device")
    generator = torch.Generator().manual_seed(8)
    mask = torch.rand((6, 15), generator=generator).to(device).requires_grad_()
    color = torch.rand((6, 3), generator=generator).to(device).requires_grad_()
    background = torch.tensor([.12, .01, -.02], device=device, requires_grad=True)
    actual = optimizer._render(mask, color, 2, 3, 3, 5, background)
    coverage = mask.reshape(2, 3, 3, 5).permute(0, 2, 1, 3).reshape(6, 15)
    foreground = color.reshape(2, 3, 3).repeat_interleave(3, 0).repeat_interleave(5, 1)
    expected = (coverage[..., None] * foreground + (1 - coverage[..., None]) * background).permute(2, 0, 1)[None]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weight = torch.rand(actual.shape, generator=generator).to(device)
    actual_grad = torch.autograd.grad((actual * weight).sum(), (mask, color, background), retain_graph=True)
    expected_grad = torch.autograd.grad((expected * weight).sum(), (mask, color, background))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


def test_hard_selection_cannot_render_a_complementary_glyph_blend():
    logits = torch.tensor([[0.1, 0.0]], requires_grad=True)
    masks = torch.tensor([[[1., 0.], [0., 1.]]])
    weights = optimizer._hard_weights(logits, 0.7)
    rendered = (weights[..., None] * masks).sum(1)
    torch.testing.assert_close(rendered, masks[:, 0])
    rendered[:, 0].sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_mask_matrix_contraction_preserves_hard_pixels_and_logit_gradient():
    generator = torch.Generator().manual_seed(4)
    bank = torch.rand((7, 15), generator=generator, dtype=torch.float64)
    logits = torch.rand((8, 7), generator=generator, dtype=torch.float64, requires_grad=True)
    weights = optimizer._hard_weights(logits, .5)
    direct = weights @ bank
    expanded = (weights[..., None] * bank[None]).sum(1)
    torch.testing.assert_close(direct, expanded, atol=0, rtol=0)
    weighting = torch.rand(direct.shape, generator=generator, dtype=torch.float64)
    left = torch.autograd.grad((direct * weighting).sum(), logits, retain_graph=True)[0]
    right = torch.autograd.grad((expanded * weighting).sum(), logits)[0]
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)


def test_optimizer_only_scores_actual_glyph_masks(monkeypatch):
    masks = np.stack((np.eye(4), 1 - np.eye(4))).astype(np.float32)
    original = optimizer._render
    matmul = torch.Tensor.__matmul__
    def rounded_contraction(left, right):
        result = matmul(left, right)
        if left.shape == (4, 2) and right.shape == (2, 16):
            # Simulate a matrix backend rounding atlas values. The forward
            # glyph still has to be copied exactly from the actual bank.
            result = result + .0001
        return result
    monkeypatch.setattr(torch.Tensor, "__matmul__", rounded_contraction)
    calls = []
    def record(mask, *args, **kwargs):
        bank = torch.as_tensor(masks, device=mask.device).flatten(1)
        assert (mask[:, None] == bank[None]).all(2).any(1).all()
        calls.append(True)
        return original(mask, *args, **kwargs)
    monkeypatch.setattr(optimizer, "_render", record)
    optimizer.solve(np.full((8, 8, 3), 0.5, np.float32), masks, steps=3, dither=0)
    assert calls


def test_rejected_batch_is_split_to_retain_an_individually_improving_move():
    glyph = torch.zeros(2, dtype=torch.long)
    color = torch.zeros(2, 3)
    bank = torch.tensor([[0.], [1.]])
    def score(mask, color, glyph):
        return (glyph.sum().float() - 1).square()
    result, _ = optimizer._accept_discrete_proposals(glyph, color, bank, torch.arange(2),
                                                     torch.ones_like(glyph), color, score)
    assert result.sum() == 1
    assert score(bank[result], color, result) < score(bank[glyph], color, glyph)


@pytest.mark.parametrize("detail", optimizer.DETAIL_LOSSES)
@pytest.mark.parametrize("invert", [False, True])
def test_batch_acceptance_reuses_pixel_local_display_mapping_for_arbitrary_subsets(detail, invert):
    generator = torch.Generator().manual_seed(77)
    dtype = torch.float64
    cell, rows, cols = (3, 5), 2, 3
    target = torch.rand((1, 3, 6, 15), generator=generator, dtype=dtype) * .08
    target[:, 0] += .4
    objective = optimizer._FrameObjective(target, detail, cell, invert, 1.5, 2., target, target, .5, target, .8)
    bank = torch.rand((4, 15), generator=generator, dtype=dtype)
    glyph = torch.tensor([0, 1, 2, 3, 0, 1])
    color = torch.rand((6, 3), generator=generator, dtype=dtype) * .1
    color[:, 0] += .4
    background = torch.tensor([.15, .01, -.01], dtype=dtype)
    proposed_glyph = (glyph + 1) % len(bank)
    # Include a color-only change, which must not be inferred from glyph IDs.
    proposed_glyph[0] = glyph[0]
    proposed_color = color * 1.02
    rendered = optimizer._render(bank[glyph], color, rows, cols, *cell, background)
    state = optimizer._ReplacementState(objective, rendered)
    cached = optimizer._ReplacementBatchScorer(state, bank, glyph, color, proposed_glyph, proposed_color,
                                                torch.arange(6), background)
    for ids in ([], [0], [1, 3, 5], list(range(6))):
        trial_glyph, trial_color = glyph.clone(), color.clone()
        trial_glyph[ids], trial_color[ids] = proposed_glyph[ids], proposed_color[ids]
        expected = objective(optimizer._render(bank[trial_glyph], trial_color, rows, cols, *cell, background))
        torch.testing.assert_close(cached(trial_glyph, trial_color), expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("cell,grid", [((3, 5), (3, 4)), ((8, 12), (3, 3)), ((1, 1), (1, 1)), ((4, 3), (1, 3))])
@pytest.mark.parametrize("detail", optimizer.DETAIL_LOSSES)
@pytest.mark.parametrize("invert", [False, True])
def test_replacement_scores_match_full_frame_with_pool_alignment_and_boundaries(cell, grid, detail, invert):
    generator = torch.Generator().manual_seed(43)
    shape = (1, 3, cell[0] * grid[0], cell[1] * grid[1])
    target = torch.rand(shape, generator=generator, dtype=torch.float64) * 0.1
    target[:, 0] += 0.4
    rendered = target + torch.rand(shape, generator=generator, dtype=torch.float64) * 0.03
    objective = optimizer._FrameObjective(target, detail, cell, invert, 1.5, 2.0, target, target,
                                           0.5, target, 0.8)
    ids = torch.tensor(sorted({0, grid[0] * grid[1] // 2, grid[0] * grid[1] - 1}))
    replacements = torch.rand((len(ids), 2, 3, *cell), generator=generator, dtype=torch.float64) * 0.1
    replacements[:, :, 0] += 0.4
    context = optimizer._ReplacementContext(objective, rendered, ids)
    actual = context.scores(replacements)
    before = objective(rendered)
    for i, cell_id in enumerate(ids.tolist()):
        row, col = divmod(cell_id, grid[1])
        for candidate in range(2):
            trial = rendered.clone()
            trial[:, :, row * cell[0]:(row + 1) * cell[0], col * cell[1]:(col + 1) * cell[1]] = replacements[i, candidate]
            torch.testing.assert_close(actual[i, candidate], objective(trial) - before, atol=1e-10, rtol=1e-7)


def test_replacement_color_gradient_matches_full_frame_gradient():
    target = torch.full((1, 3, 9, 15), 0.1, dtype=torch.float64)
    target[:, 0] = 0.4
    objective = optimizer._FrameObjective(target, "chroma_contrast", (3, 5), False, 1.5, 2., target, target,
                                           0.3, target, 1.)
    rendered = target * 0.9
    pixels = (target[:, :, :3, :5] * 1.1).unsqueeze(0).detach().requires_grad_()
    context = optimizer._ReplacementContext(objective, rendered, torch.tensor([0]))
    actual = torch.autograd.grad(context.scores(pixels).sum(), pixels)[0]
    trial = rendered.clone()
    trial[:, :, :3, :5] = pixels[0]
    expected = torch.autograd.grad(objective(trial), pixels)[0]
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-6)


def test_candidate_batches_share_frame_responses_without_filtering_neighbors_again(monkeypatch):
    generator = torch.Generator().manual_seed(91)
    target = torch.rand((1, 3, 12, 20), generator=generator, dtype=torch.float64) * 0.1
    target[:, 0] += 0.4
    rendered = target + 0.02
    objective = optimizer._FrameObjective(target, "default", (3, 5), False, 1.5, 2., target, target)
    state = optimizer._ReplacementState(objective, rendered)
    pixels = target[:, :, :3, :5].clone()
    expected = []
    for cell_id in (0, 15):
        row, col = divmod(cell_id, 4)
        trial = rendered.clone()
        trial[:, :, row * 3:(row + 1) * 3, col * 5:(col + 1) * 5] = pixels
        expected.append(objective(trial) - objective(rendered))
    def unexpected(*args, **kwargs):
        raise AssertionError("Candidate batches must reuse frame-wide convolution responses")
    monkeypatch.setattr(optimizer._ReplacementState, "__init__", unexpected)
    monkeypatch.setattr(optimizer, "_sobel_gradients", unexpected)
    for i, cell_id in enumerate((0, 15)):
        context = optimizer._ReplacementContext(objective, rendered, torch.tensor([cell_id]), state)
        actual = context.scores(pixels[None])[0, 0]
        torch.testing.assert_close(actual, expected[i], atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize("invert", [False, True])
def test_coverage_lookup_preserves_displayed_scores_and_color_gradients(invert):
    dtype = torch.float64
    bank = torch.tensor([[[0., .1, .7], [.7, .1, 0.]], [[.7, 1., .1], [0., .1, 1.]]], dtype=dtype)
    coverage, lookup = torch.unique(bank, sorted=True, return_inverse=True)
    color = torch.tensor([[.6, .12, -.1]], dtype=dtype, requires_grad=True)
    background = torch.tensor([.05, .01, -.005], dtype=dtype)
    pixels = (bank[None, :, None] * color[:, None, :, None, None]
              + (1 - bank[None, :, None]) * background[None, None, :, None, None])
    direct_display = optimizer._display_headroom_lab(pixels.flatten(0, 1), invert_ink=invert).reshape_as(pixels)
    table = optimizer._display_ink_table(color, background, coverage, invert_ink=invert)
    cached = table[:, None].expand(-1, 2, -1, -1).gather(
        -1, lookup.flatten(1)[None, :, None].expand(1, 2, 3, 6)).reshape_as(pixels)
    torch.testing.assert_close(cached, direct_display, atol=1e-12, rtol=1e-10)
    target = torch.zeros((1, 3, 2, 6), dtype=dtype)
    target[:, 0] = .4
    objective = optimizer._FrameObjective(target, "default", (2, 3), invert, 1.5, 2., target, target)
    context = optimizer._ReplacementContext(objective, target, torch.tensor([0]))
    direct = context.scores(pixels)
    actual = context.scores(pixels, cached)
    torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-10)
    expected_gradient = torch.autograd.grad(direct.sum(), color, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), color)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_context_matches_full_frame_at_display_gamut_boundaries(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device available")
    target = torch.rand((1, 3, 6, 15), generator=torch.Generator().manual_seed(7)).to(device)
    target[:, 0] = 0.7
    target[:, 1:] = (target[:, 1:] - 0.5) * 0.4
    rendered = target * 0.9
    objective = optimizer._FrameObjective(target, "default", (3, 5), True, 1.5, 2., target, target)
    ids = torch.tensor([0, 5], device=device)
    pixels = torch.zeros((2, 2, 3, 3, 5), device=device)
    pixels[:, :, 0] = 0.95
    pixels[:, 0, 1] = 0.25
    pixels[:, 1, 2] = -0.25
    context = optimizer._ReplacementContext(objective, rendered, ids)
    scores = context.scores(pixels)
    before = objective(rendered)
    for i, cell_id in enumerate(ids.tolist()):
        row, col = divmod(cell_id, 3)
        for candidate in range(2):
            trial = rendered.clone()
            trial[:, :, row * 3:(row + 1) * 3, col * 5:(col + 1) * 5] = pixels[i, candidate]
            torch.testing.assert_close(scores[i, candidate].float(), objective(trial) - before,
                                       atol=2e-6, rtol=1e-4)


def test_neighboring_edges_can_reverse_a_cells_isolated_candidate_ranking():
    target = torch.zeros((1, 3, 4, 8), dtype=torch.float64)
    target[:, 0] = 0.5
    rendered = target.clone()
    rendered[:, 0, :, 4:] = 0.9
    objective = optimizer._FrameObjective(target, "default", (4, 4), False, 0., 0.)
    exact = target[:, :, :, :4]
    slightly_brighter = exact.clone()
    slightly_brighter[:, 0, :, -1] += 0.08
    candidates = torch.stack((exact[0], slightly_brighter[0]))[None]
    assert (candidates[0, 0] - exact[0]).square().sum() < (candidates[0, 1] - exact[0]).square().sum()
    scores = optimizer._ReplacementContext(objective, rendered, torch.tensor([0])).scores(candidates)
    assert scores[0, 1] < scores[0, 0]


def test_equal_coverage_glyphs_follow_subglyph_pixel_arrangement():
    left = np.zeros((4, 4), np.float32)
    left[:, :2] = 1
    masks = np.stack((left[:, ::-1], left))
    source = np.repeat((left * 0.6)[..., None], 3, axis=-1)
    grid = optimizer.solve(source, masks, steps=0, dither=0, background="#000000",
                           gradient_loss_strength=0, glyph_entropy_strength=0)
    assert grid.glyph[0, 0] == 1


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf")])
def test_layout_strength_rejects_invalid_values(strength):
    with pytest.raises(ValueError, match="layout_loss_strength"):
        optimizer.solve(np.zeros((4, 4, 3), np.float32), np.ones((1, 4, 4), np.float32),
                        layout_loss_strength=strength)
