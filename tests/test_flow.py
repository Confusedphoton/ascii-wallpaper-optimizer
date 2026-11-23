"""Physical invariants, font geometry and production flow-style integration."""

import numpy as np
import pytest
import torch

from ascii_optimizer import cli, optimizer
from ascii_optimizer.defaults import DEFAULT_CHARS, FLOW_CHARS
from ascii_optimizer.flow import FlowField, _SpectralDomain, glyph_flow_costs, simulate_flow
from ascii_optimizer.settings import normalize_options


@pytest.fixture(autouse=True)
def modest_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_projection_removes_divergence_without_erasing_a_uniform_current():
    torch.manual_seed(3)
    domain = _SpectralDomain(24, 40, "cpu")
    velocity = torch.randn(1, 2, 24, 40)
    projected = domain.project(velocity)
    assert domain.divergence(projected).square().mean() < domain.divergence(velocity).square().mean() * 1e-10
    assert torch.allclose(projected.mean((2, 3)), velocity.mean((2, 3)), atol=1e-6)
    uniform = torch.ones_like(velocity)
    assert torch.allclose(domain.project(uniform), uniform, atol=1e-6)


def test_forced_snapshot_is_finite_seeded_and_handles_flat_and_chromatic_images():
    source = np.full((32, 48, 3), .4, np.float32)
    first = simulate_flow(source, (8, 4), seed=5, steps=8)
    second = simulate_flow(source, (8, 4), seed=5, steps=8)
    other = simulate_flow(source, (8, 4), seed=7, steps=8)
    assert torch.equal(first.velocity, second.velocity)
    assert not torch.allclose(first.velocity, other.velocity)
    assert torch.isfinite(first.velocity).all()
    assert first.source_confidence.max() < 1e-5
    assert first.diagnostics["mean_speed"] > 0
    assert first.diagnostics["divergence_after_projection"] < 1e-5
    assert first.velocity.shape[-1] / first.velocity.shape[-2] == pytest.approx(48 / 32, rel=.06)
    source[:, 24:] = [.1, .55, .35]
    source[:, :24] = [.6, .2, .5]
    chromatic = simulate_flow(source, (8, 4), steps=8)
    assert chromatic.source_confidence.max() > .25


def straight_field(horizontal=True):
    velocity = torch.zeros(1, 2, 16, 16)
    velocity[:, 0 if horizontal else 1] = 1
    return FlowField(velocity, torch.ones(1, 1, 16, 16), torch.ones(1, 1, 16, 16),
                     (16, 16), (16, 16), 1., {})


def test_glyph_geometry_follows_flow_axis_and_ignores_velocity_sign():
    masks = torch.zeros(3, 16, 16)
    masks[1, 7:9, 2:14] = 1
    masks[2, 2:14, 7:9] = 1
    horizontal, _ = glyph_flow_costs(straight_field(), masks)
    vertical, _ = glyph_flow_costs(straight_field(False), masks)
    assert horizontal[0, 1] < horizontal[0, 2]
    assert vertical[0, 2] < vertical[0, 1]
    reverse = straight_field()
    reverse.velocity.neg_()
    costs, _ = glyph_flow_costs(reverse, masks)
    assert torch.equal(costs, horizontal)
    assert torch.isfinite(horizontal).all()


def test_curved_stream_matches_a_loop_over_a_straight_cross():
    y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    radius = (x.square() + y.square()).sqrt()
    masks = torch.stack(((radius > .55) & (radius < .72),
                         (x.abs() < .07) | (y.abs() < .07))).float()
    field = FlowField(torch.stack((-y, x))[None], torch.ones(1, 1, 32, 32),
                      torch.ones(1, 1, 32, 32), (32, 32), (32, 32), 1., {})
    costs, _ = glyph_flow_costs(field, masks)
    assert costs[0, 0] + .2 < costs[0, 1]


def test_diffusion_damps_nyquist_stripes():
    domain = _SpectralDomain(24, 40, "cpu")
    stripes = (torch.arange(40) % 2).float() * 2 - 1
    velocity = stripes[None, None, None].expand(1, 2, 24, 40).clone()
    assert domain.diffuse(velocity, 1, 1).square().mean() < velocity.square().mean() * .02


def test_flow_style_zero_weight_reproduces_same_bank_perceptual_output():
    rng = np.random.default_rng(3)
    source = rng.random((16, 16, 3), dtype=np.float32)
    masks = rng.random((5, 8, 4), dtype=np.float32)
    common = dict(invert_ink=False, background="auto", seed=2, steps=3,
                  dither=0.005, glyph_entropy_strength=0.01, device="cpu")
    headroom = cli.solve_style(source, masks, style="headroom", **common)
    zero_flow = cli.solve_style(source, masks, style="flow", flow_strength=0, **common)
    assert np.array_equal(headroom.glyph, zero_flow.glyph)
    assert np.array_equal(headroom.color, zero_flow.color)


def test_flow_cost_remains_in_optimization_and_trace():
    from ascii_optimizer.diagnostics import SolverTrace
    # Identical masks make image/color costs indifferent to glyph identity.
    # The simulated geometry cost must decide the output through all updates.
    masks = np.full((3, 8, 4), .25, np.float32)
    source = np.full((16, 16, 3), .35, np.float32)
    costs = np.tile([1., .5, 0.], (8, 1)).astype(np.float32)
    trace = SolverTrace()
    grid = optimizer.solve(source, masks, steps=4, flow_costs=costs, flow_strength=.8,
                           glyph_entropy_strength=0, diagnostics=trace)
    assert (grid.glyph == 2).all()
    assert trace.metadata["coefficients"]["glyph_flow"] == .8
    assert trace.metadata["contour_scale"] < 1


def test_flow_bank_is_broad_unique_and_custom_banks_survive():
    flow = normalize_options({"style": "flow"})
    assert flow["chars"] == FLOW_CHARS
    assert len(FLOW_CHARS) == len(set(FLOW_CHARS)) == 96
    assert set("=@G|/\\-+0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") <= set(FLOW_CHARS)
    assert flow["glyph_entropy_strength"] > 0
    assert flow["perceptual_loss"]
    assert normalize_options({"style": "flow", "chars": " .G@"})["chars"] == " .G@"
    assert normalize_options({"chars": DEFAULT_CHARS})["chars"] == DEFAULT_CHARS


@pytest.mark.parametrize("name,value", [("flow_strength", float("nan")), ("flow_strength", 2.1),
                                        ("flow_eddies", -1), ("flow_scale", .5), ("flow_scale", 25)])
def test_flow_settings_validate_bounds(name, value):
    with pytest.raises(ValueError):
        normalize_options({"style": "flow", name: value})


def test_flow_settings_and_algorithm_enter_cache_identity(tmp_path, monkeypatch):
    from ascii_optimizer import service
    from ascii_optimizer.revision import RENDER_MODULES
    from PIL import Image
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "#6688aa").save(source)
    monkeypatch.setattr(service, "_screen_size", lambda _screen, size: size)
    raw = {"images": [str(source)], "size": [32, 32], "style": "flow"}
    a = service._validated_request(raw)
    b = service._validated_request(dict(raw, flow_eddies=1.2))
    assert a != b and a["glyph_search"] == "all"
    assert a["chars"] == FLOW_CHARS
    assert "flow.py" in RENDER_MODULES


def test_flow_cli_writes_full_bank_and_exact_slideshow_endpoints(tmp_path):
    import json
    from PIL import Image
    sources = []
    for index, color in enumerate(("#446688", "#885544")):
        path = tmp_path / f"source{index}.png"
        Image.new("RGB", (32, 16), color).save(path)
        sources.append(str(path))
    output = tmp_path / "render"
    args = cli.make_parser().parse_args([*sources, "--output", str(output), "--size", "32x16",
        "--cell", "4x8", "--style", "flow", "--steps", "2", "--transition-steps", "2",
        "--frames", "16", "--transition-workers", "1", "--device", "cpu"])
    cli.build(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["atlas_count"] == 96
    assert manifest["signature"]["style"] == "flow"
    assert manifest["signature"]["glyph_entropy_strength"] == .02
    for kind in ("glyph", "color", "background"):
        start = output / "transitions/0000-0001/0000" / f"{kind}.png"
        finish = output / "transitions/0000-0001/0015" / f"{kind}.png"
        assert start.read_bytes() == (output / "stills/0000" / f"{kind}.png").read_bytes()
        assert finish.read_bytes() == (output / "stills/0001" / f"{kind}.png").read_bytes()
    assert (output / "transitions/0000-0001/0001/color.png").is_file()
