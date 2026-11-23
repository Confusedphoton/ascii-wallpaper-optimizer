from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ascii_optimizer import optimizer
from ascii_optimizer.diagnostics import SolverTrace


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("steps", [0, 3])
def test_trace_records_actual_states_and_logits_without_changing_output(tmp_path, steps, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device")
    masks = np.stack((np.zeros((3, 4)), np.ones((3, 4)), np.eye(3, 4), 1 - np.eye(3, 4))).astype(np.float32)
    target = np.random.default_rng(7).uniform(.15, .7, (5, 7, 3)).astype(np.float32)
    kwargs = dict(steps=steps, dither=.005, device=device, glyph_luminance_bonus=.12,
                  perceptual_loss=False)
    plain = optimizer.solve(target, masks, **kwargs)
    trace = SolverTrace()
    traced = optimizer.solve(target, masks, **kwargs, diagnostics=trace)
    np.testing.assert_array_equal(plain.glyph, traced.glyph)
    np.testing.assert_array_equal(plain.color, traced.color)
    report = trace.save(tmp_path)
    grids = np.load(tmp_path / "grids.npz")
    logits = np.load(tmp_path / "logits.npz")
    assert logits["logits"].shape == (steps + 2, 4, len(masks))
    assert grids["glyph"].shape[1:] == (2, 2)
    terminal = next(row for row in report["records"] if row["stage"] == "solver_final")
    terminal_logits = logits["logits"][terminal["logit_index"]]
    predicted = np.take_along_axis(logits["candidate_indices"], terminal_logits.argmax(1)[:, None], axis=1)
    np.testing.assert_array_equal(predicted.ravel(), grids["glyph"][terminal["grid_index"]].ravel())
    dithered = next(row for row in report["records"] if row["stage"] == "after_dither")
    predicted = np.take_along_axis(logits["candidate_indices"],
                                   (terminal_logits + logits["selection_noise"]).argmax(1)[:, None], axis=1)
    np.testing.assert_array_equal(predicted.ravel(), grids["glyph"][dithered["grid_index"]].ravel())
    for row in report["records"]:
        assert sum(row["components"].values()) == pytest.approx(row["loss"], abs=1e-6)
        assert row["cpu_seconds"] >= 0
        if device == "cuda":
            assert row["gpu_timeline_seconds"] >= 0
        for term, values in row["loss_terms"].items():
            assert values["raw_error"] * values["coefficient"] == pytest.approx(row["components"][term])
        if "weighted_gradient_rms" in row:
            assert all(np.isfinite(v) and v >= 0 for values in row["weighted_gradient_rms"].values() for v in values.values())
            assert row["weighted_gradient_rms"]["total"]["logits"] > 0
    final = report["records"][-1]
    assert final["stage"] == "final"
    np.testing.assert_array_equal(grids["glyph"][final["grid_index"]], traced.glyph)
    np.testing.assert_allclose(grids["color"][final["grid_index"]], traced.color, atol=1e-6)
    assert json.loads((tmp_path / "trace.json").read_text())["records"] == report["records"]


def test_visual_variance_matches_rasterized_candidates_and_glyph_permutation(tmp_path):
    masks = np.stack((np.eye(4), 1 - np.eye(4), np.fliplr(np.eye(4)))).astype(np.float32)
    target = np.full((4, 4, 3), .5, np.float32)
    outputs = []
    for order in (np.arange(3), np.array([2, 0, 1])):
        trace = SolverTrace()
        grid = optimizer.solve(target, masks[order], steps=0,
                               dither=0, diagnostics=trace, glyph_entropy_strength=0)
        directory = tmp_path / str(len(outputs))
        trace.save(directory)
        values = np.load(directory / "logits.npz")["logits"][0]
        p = np.exp(values - values.max(1, keepdims=True)).astype(np.float64)
        p /= p.sum(1, keepdims=True)
        candidates = np.stack([optimizer.render_grid(optimizer.Grid(np.zeros((1, 1), np.uint8),
            grid.color, grid.background), mask[None]) for mask in masks[order]])
        mean = np.sum(p.reshape(-1, 1, 1, 1) * candidates, axis=0)
        expected = np.sum(p.reshape(-1, 1, 1) * ((candidates - mean) ** 2).sum(-1), axis=0).mean()
        visual = np.load(directory / "visual.npz")
        assert visual["pixel_variance"][0, 0] == pytest.approx(expected, rel=2e-6)
        differences = torch.from_numpy((candidates - mean).astype(np.float64)).permute(0, 3, 1, 2)
        padded = torch.nn.functional.pad(differences, (2, 2, 2, 2))
        horizontal, vertical = optimizer._sobel_filters(differences)
        gx = torch.nn.functional.conv2d(padded, horizontal, groups=3)
        gy = torch.nn.functional.conv2d(padded, vertical, groups=3)
        expected_edges = ((gx.square() + gy.square()).sum(1).mean((1, 2)).numpy() * p.ravel()).sum()
        assert visual["edge_variance"][0, 0] == pytest.approx(expected_edges, rel=2e-6)
        outputs.append(visual["pixel_variance"][0, 0])
    assert outputs[0] == pytest.approx(outputs[1], rel=1e-6)


def test_gradient_sampling_reports_coefficient_effect_and_does_not_accumulate_gradients():
    parameter = torch.tensor([1., 2.], requires_grad=True)
    trace = SolverTrace()
    raw = parameter.square().mean()
    norms = trace.sample_gradients({"unweighted": raw, "weighted": 20 * raw}, {"parameter": parameter})
    torch.testing.assert_close(norms["weighted/parameter"], 20 * norms["unweighted/parameter"])
    assert parameter.grad is None
    raw.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([1., 2.]))


def test_normal_render_returns_raw_adam_terminal(tmp_path):
    trace = SolverTrace()
    optimizer.solve(np.full((4, 4, 3), .4, np.float32), np.ones((1, 4, 4), np.float32),
                    steps=1, diagnostics=trace)
    report = trace.save(tmp_path)
    assert report["metadata"]["postprocess"] is False
    assert not any(row["stage"] in ("after_dither", "final_polish") for row in report["records"])


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_iteration_budgets_are_rejected(value):
    with pytest.raises(ValueError, match="steps"):
        optimizer.solve(np.zeros((4, 4, 3), np.float32), np.ones((1, 4, 4), np.float32), steps=value)


def test_loss_components_sum_to_full_objective():
    target = torch.rand((1, 3, 8, 8), generator=torch.Generator().manual_seed(3), dtype=torch.float64) * .08
    target[:, 0] += .4
    rendered = target * .9
    objective = optimizer._FrameObjective(target, (4, 4), False, 1.5, 2., target, target)
    components = {}
    loss = objective(rendered, components=components)
    torch.testing.assert_close(loss, objective(rendered), atol=0, rtol=0)
    torch.testing.assert_close(sum(components.values()), loss, atol=1e-14, rtol=1e-12)
