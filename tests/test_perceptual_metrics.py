from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("perceptual_metrics")


def test_circular_transport_wraparound_and_reference_geodesic(metrics):
    n = 12
    p = torch.zeros(1, n, 1, 1)
    p[:, 0] = 1
    for index in range(n):
        q = torch.zeros_like(p)
        q[:, index] = 1
        assert float(metrics.circular_w1(p, q)) == pytest.approx(
            min(index, n - index) / n
        )
        for rotation in (1, 5):
            assert metrics.circular_w1(
                p.roll(rotation, 1), q.roll(rotation, 1)
            ) == pytest.approx(metrics.circular_w1(p, q))


def test_lightness_black_and_white_and_finite_black_derivative(metrics):
    y = torch.tensor([0.0, 0.001, 1.0], requires_grad=True)
    l = metrics.lightness(y)
    assert float(l[0].detach()) == pytest.approx(0)
    assert float(l[-1].detach()) == pytest.approx(1)
    l.sum().backward()
    assert torch.isfinite(y.grad).all() and (y.grad > 0).all()


def test_spatial_brightness_integrates_linear_light_before_nonlinearity(metrics):
    rgb = torch.zeros(1, 3, 64, 64)
    rgb[:, :, ::2] = 0.2
    ref = torch.full_like(rgb, 0.1)
    model = metrics.SpatialMetrics(ref, ppd=60)
    terms, _ = model.terms(rgb)
    assert terms["spatial_lightness"] == pytest.approx(0, abs=1e-9)
    dark = model.measure(torch.full_like(rgb, 0.05))
    bright = model.measure(torch.full_like(rgb, 0.2))
    assert dark["scales"][1]["lightness_bias"] < 0
    assert bright["scales"][1]["lightness_bias"] > 0


def test_orientation_detects_rotation_and_missing_contours_without_polarity_penalty(
    metrics,
):
    x = torch.linspace(-1, 1, 96)
    step = (x > 0).float() * 0.4 + 0.1
    source = step[None, None, None, :].expand(1, 3, 96, 96).clone()
    model = metrics.SpatialMetrics(source, ppd=60, cell_height=16)
    same = model.measure(source)
    rotated = model.measure(source.transpose(-1, -2))
    missing = model.measure(torch.full_like(source, 0.1))
    reversed_ = model.measure(source.flip(-1))
    assert same["orientation_w1"] == pytest.approx(0, abs=1e-7)
    assert rotated["orientation_w1"] > same["orientation_w1"] + 0.01
    assert rotated["orientation_tensor"] > 0.01
    assert missing["contour_strength"] > 0.01
    assert reversed_["orientation_w1"] == pytest.approx(0, abs=2e-6)


def test_metric_gradients_are_finite_and_change_the_candidate(metrics):
    torch.manual_seed(8)
    source = torch.rand(1, 3, 64, 64) * 0.2
    model = metrics.SpatialMetrics(source, ppd=60)
    candidate = (source * 0.8).detach().requires_grad_()
    terms, _ = model.terms(candidate)
    sum(terms.values()).backward()
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0


def test_installed_profile_parser_preserves_glyph_identity_and_active_cache(
    metrics, tmp_path
):
    import json

    script = importlib.import_module("compare_perceptual")
    first, second = tmp_path / "a", tmp_path / "b"
    for directory, chars in ((first, " .@\\"), (second, " .W\\")):
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"signature": {"chars": chars}})
        )
    config = tmp_path / "plasma.conf"
    config.write_text(
        "[Containments][48][Wallpaper][org.asciiwallpaper.optimizer][Wallpaper]\n"
        f"RenderDirectory={first}\nRenderDirectoryB={second}\n"
    )
    profiles = script.installed_profiles(config)
    assert profiles["tree"]["signature"]["chars"] == " .@\\"
    assert profiles["viviana"]["signature"]["chars"] == " .W\\"
    assert profiles["viviana"]["installed"] == second / "stills/0000"


def test_experimental_objective_matches_independent_weighted_metrics_and_gradient(
    metrics,
):
    from ascii_optimizer import optimizer

    source = torch.rand(1, 3, 32, 32) * 0.2
    model = metrics.SpatialMetrics(source, ppd=60)
    target = optimizer._rgb_to_oklab(source.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    base = optimizer._FrameObjective(
        target, "default", (4, 4), False, 1.5, 0, target, None
    )
    weights = {
        "spatial_lightness": 2.0,
        "orientation_w1": 0.2,
        "contour_strength": 0.005,
    }
    wrapped = metrics.ExperimentalObjective(base, model, weights, replace_sobel=True)
    rendered = (target * 0.9).detach().requires_grad_()
    components = {}
    loss = wrapped(rendered, components=components)
    displayed = optimizer._display_headroom_lab(rendered)
    rgb = optimizer._oklab_to_rgb(displayed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    terms, _ = model.terms(rgb)
    expected = base(rendered) + sum(weights[k] * terms[k] for k in weights)
    torch.testing.assert_close(loss, expected)
    assert "displayed_sobel" not in components
    for k, weight in weights.items():
        torch.testing.assert_close(components[k], weight * terms[k])
    loss.backward()
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad.abs().sum() > 0


def test_default_objective_and_selected_metrics_match_reviewed_experiment(metrics):
    from ascii_optimizer import optimizer
    from ascii_optimizer.defaults import PERCEPTUAL_WEIGHTS

    source = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(17)) * 0.3
    model = metrics.SpatialMetrics(source, ppd=60, cell_height=8)
    target = optimizer._rgb_to_oklab(source.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    baseline = optimizer._FrameObjective(
        target, "default", (8, 8), False, 1.5, 20, target, target
    )
    reviewed = metrics.ExperimentalObjective(baseline, model, PERCEPTUAL_WEIGHTS)
    production = optimizer._FrameObjective(
        target, "default", (8, 8), False, 1.5, 20, target, target, spatial_metrics=model
    )
    candidate = (target * 0.9).detach().requires_grad_()
    expected = reviewed(candidate)
    components = {}
    actual = production(candidate, components=components)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(actual, candidate)[0]
    expected_gradient = torch.autograd.grad(expected, candidate)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    assert set(PERCEPTUAL_WEIGHTS) <= components.keys()
    assert "displayed_sobel" in components and "saturation" in components
    terms, _ = model.terms(source, selected=PERCEPTUAL_WEIGHTS)
    assert "orientation_tensor" not in terms


@pytest.mark.parametrize("invert", [False, True])
def test_default_solver_uses_original_source_and_logs_perceptual_losses(
    metrics, invert
):
    import numpy as np

    from ascii_optimizer import cli, optimizer
    from ascii_optimizer.diagnostics import SolverTrace
    from ascii_optimizer.perceptual import ppd_for_width

    source = np.random.default_rng(27).uniform(0.05, 0.6, (9, 13, 3)).astype(np.float32)
    masks = np.stack((np.eye(4), np.ones((4, 4)))).astype(np.float32)
    trace = SolverTrace()
    cli.solve_style(
        source,
        masks,
        style="headroom",
        invert_ink=invert,
        background="auto",
        seed=0,
        dither=0,
        structure_dither=0.06,
        device="cpu",
        steps=2,
        polish_steps=2,
        diagnostics=trace,
    )
    assert trace.metadata["perceptual_loss"] is True
    assert trace.metadata["viewing_ppd"] == pytest.approx(ppd_for_width(13))
    assert trace.metadata["coefficients"]["spatial_lightness"] == 2
    assert trace.metadata["replacement_diagnostics"].startswith("Disabled")
    for row in trace.records:
        assert {
            "loss/spatial_lightness",
            "loss/orientation_w1",
            "loss/contour_strength",
        } <= row["tensors"].keys()
        assert torch.isfinite(row["tensors"]["loss"])
    assert not trace.replacements
    with pytest.raises(ValueError, match="perceptual_loss=False"):
        optimizer.solve(source, masks, refinement_sweeps=1)


def test_source_features_are_cached_and_original_rgb_is_used(metrics, monkeypatch):
    import numpy as np

    from ascii_optimizer import cli, optimizer
    from ascii_optimizer.assets import srgb_to_linear

    captured = []
    original = optimizer.SpatialMetrics

    def capture(source, **kwargs):
        captured.append(source)
        return original(source, **kwargs)

    monkeypatch.setattr(optimizer, "SpatialMetrics", capture)
    source = np.full((8, 8, 3), (0.15, 0.3, 0.5), dtype=np.float32)
    masks = np.ones((1, 4, 4), dtype=np.float32)
    cli.solve_style(
        source,
        masks,
        style="headroom",
        steps=1,
        invert_ink=True,
        background="auto",
        seed=0,
        dither=0,
        structure_dither=0.06,
        device="cpu",
    )
    expected = srgb_to_linear(torch.from_numpy(source).permute(2, 0, 1)[None])
    torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)
    model = original(expected, ppd=60)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gaussian kernels must be prepared before the loss loop")

    monkeypatch.setattr(
        importlib.import_module("ascii_optimizer.perceptual"),
        "gaussian_kernel",
        forbidden,
    )
    model.terms(expected)
