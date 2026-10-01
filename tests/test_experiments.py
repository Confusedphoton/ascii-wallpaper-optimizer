from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ascii_optimizer import cli
from ascii_optimizer.defaults import DEFAULT_GLYPH_LUMINANCE_BONUS
from ascii_optimizer.optimizer import Grid
from ascii_optimizer import optimizer


@pytest.fixture
def experiments(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("experiment_utils")
    module.reset_runs()
    return module


@pytest.mark.parametrize("invert_ink", [False, True])
def test_experiment_preview_matches_serialized_playback_preview(experiments, tmp_path, invert_ink):
    masks = np.array([[[0, 64 / 255], [128 / 255, 1]]], np.float32)
    grid = Grid(np.array([[0]], np.uint8), np.array([[[0.72, 0.32, -0.27]]], np.float32),
                np.array([0.17, -0.04, 0.04], np.float32))
    cli._save_grid(grid, tmp_path, True, masks, (2, 2), invert_ink)
    preview = experiments.image(experiments.preview(grid, masks, (2, 2), invert_ink=invert_ink))
    with Image.open(tmp_path / "preview.png") as expected:
        np.testing.assert_array_equal(np.asarray(preview), np.asarray(expected))


def test_dense_glyph_fraction_averages_pixels_within_each_cell(experiments):
    masks = np.stack([np.full((4, 5), value) for value in (0.1, 0.6, 0.9)]).astype(np.float32)
    glyph = np.array([[0, 0, 1], [1, 2, 2]], np.uint8)
    grid = Grid(glyph, np.zeros((2, 3, 3), np.float32))
    metrics = experiments.coverage_metrics(grid, masks)
    assert metrics["mean_glyph_coverage"] == pytest.approx((0.1 + 0.6 + 0.9) / 3)
    assert metrics["dense_glyph_fraction"] == pytest.approx(4 / 6)
    for name in ("compare_color_spaces", "compare_logit_noise"):
        script = importlib.import_module(name)
        reference = np.zeros((8, 15, 3), np.float32)
        result = script._metrics(reference, reference, grid, grid, masks)
        assert result["dense_glyph_fraction"] == pytest.approx(4 / 6)


def test_glyph_ink_lightness_is_the_stored_oklab_l_component(experiments):
    masks = np.ones((1, 2, 2), np.float32)
    grid = Grid(np.zeros((1, 2), np.uint8),
                np.array([[[0.6, 0.05, -0.04], [0.8, -0.03, 0.02]]], np.float32))
    metrics = experiments.glyph_metrics(grid, masks, "#")
    expected = experiments.playback_grid(grid).color[..., 0]
    assert metrics["mean_ink_oklab_lightness"] == pytest.approx(expected.mean())
    assert metrics["ink_oklab_lightness_std"] == pytest.approx(expected.std())


@pytest.mark.parametrize("entropy_strength", [0, .01])
def test_solver_trace_script_exports_stage_comparisons_and_raw_logits(experiments, monkeypatch, tmp_path, entropy_strength):
    script = importlib.import_module("trace_solver")
    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (90, 130, 150)).save(source)
    masks = np.stack((np.zeros((2, 3)), np.ones((2, 3)), np.eye(2, 3))).astype(np.float32)
    monkeypatch.setattr(script, "SOURCES", [("fixture", source, (0, 0, 960, 640))])
    monkeypatch.setattr(script, "glyph_masks", lambda *args: masks)
    monkeypatch.setattr(experiments, "DEFAULT_CHARS", " #/")
    output = tmp_path / "trace"
    monkeypatch.setattr(sys, "argv", ["trace_solver.py", "--output", str(output), "--width", "12",
                                     "--cell", "3x2", "--steps", "2", "--step-budgets", "2", "--no-plots",
                                     "--timing-repeats", "1", "--refinement-budgets", "0", "1", "--no-matched-time",
                                     "--glyph-entropy-strength", str(entropy_strength)])
    script.main()
    report = json.loads((output / "metrics.json").read_text())
    run = report["images"]["fixture"]
    assert run["budgets"][0]["matches_traced_output"] is True
    stages = run["analysis"]["stages"]
    adam = run["analysis"]["adam_stages"]
    assert adam["adam_step_0"] == stages["initial"]
    assert adam["solver_final"] == stages["solver_final"]
    assert run["budgets"][0]["loss"] == pytest.approx(stages["final"]["loss"], rel=1e-5, abs=1e-7)
    assert report["glyph_entropy_strength"] == entropy_strength
    assert stages["final"]["loss_terms"].get("glyph_entropy", {}).get("coefficient", 0) == entropy_strength
    assert stages["final"]["loss"] <= stages["pre_refinement"]["loss"]
    assert stages["final"]["regions"]["uniform"]["cells"] > 0
    data = np.load(output / "fixture" / "logits.npz")
    assert data["logits"].shape == (4, 16, 3)
    assert (output / "fixture" / "stage_comparison.png").is_file()
    assert (output / "fixture" / "adam_comparison.png").is_file()
    assert (output / "fixture" / "adam_focus.png").is_file()
    assert (output / "fixture" / "focus_comparison.png").is_file()
    assert (output / "fixture" / "visual.npz").is_file()
    assert (output / "fixture" / "replacements_pre_refinement.npz").is_file()
    assert [run["refinement_sweeps"] for run in run["refinement_budgets"]] == [0, 1]
    assert all(run["polish_steps"] == 9 for run in run["refinement_budgets"])


@pytest.mark.parametrize("fixed_time", [None, 2.])
def test_matched_compute_calibration_reports_actual_tolerance(experiments, monkeypatch, fixed_time):
    script = importlib.import_module("trace_solver")
    calls = []
    def run(target, masks, kwargs, *, steps, sweeps, polish_steps, repeats):
        calls.append((steps, sweeps, polish_steps, repeats))
        seconds = .2 + steps * .02 if fixed_time is None else fixed_time
        return {"seconds": seconds, "steps": steps}, None, None
    monkeypatch.setattr(script, "ordinary_run", run)
    baseline = [{"seconds": .2, "steps": 0}, {"seconds": .56, "steps": 36}]
    budget = {"seconds": 1.2, "refinement_sweeps": 2}
    result, _, _ = script.matched_adam(None, None, {}, baseline, budget, 9, 3)
    assert all(sweeps == 0 and polish == 9 and repeats == 3 for _, sweeps, polish, repeats in calls)
    assert result["within_10_percent"] is (fixed_time is None)
    assert result["time_ratio"] == pytest.approx(result["seconds"] / 1.2)
    if fixed_time is None:
        assert result["steps"] == 50
    else:
        assert len(calls) == 3


def test_report_records_hashes_effective_objectives_and_measurements(experiments, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), (90, 130, 150)).save(source)
    with Image.open(source) as image:
        target = np.asarray(image, np.float32) / 255
    masks = np.stack((np.zeros((2, 2)), np.ones((2, 2)))).astype(np.float32)
    grid = experiments.solve_style(target, masks, style="headroom", invert_ink=False, background="auto",
                                   seed=3, steps=1, candidates=1, dither=0, structure_dither=0, device="cpu",
                                   gradient_loss_strength=0)
    path = tmp_path / "report.json"
    experiments.write_report(path, {"source": str(source), "variant": grid.experiment})
    report = json.loads(path.read_text())
    provenance = report["reproducibility"]
    assert provenance["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(provenance["font_sha256"]) == 64
    assert len(provenance["renderer_sources_sha256"]["optimizer.py"]) == 64
    assert len(provenance["git_revision"]) == 40
    run = provenance["runs"][0]
    assert run["seconds"] > 0
    assert run["process_peak_rss_bytes"] > 0
    assert run["effective_objectives"]["glyph_luminance_bonus"] == DEFAULT_GLYPH_LUMINANCE_BONUS
    assert run["effective_objectives"]["glyph_entropy_strength"] == 0.01
    assert run["effective_objectives"]["saturation_loss_strength"] == 20
    assert run["effective_objectives"]["use_shortlist"] is False
    assert run["effective_objectives"]["glyphs_per_cell"] == len(masks)
    assert "candidates" not in run["effective_objectives"]


def test_red_detail_metric_handles_a_crop_without_red_pixels(experiments):
    reference = np.zeros((4, 4, 3), np.float32)
    metrics = experiments.red_chroma_metrics(reference, reference, (0, 0, 4, 4))
    assert metrics["red_belt_sample_pixels"] == 0
    assert metrics["red_belt_chroma_retention"] is None
    json.dumps(metrics, allow_nan=False)


def test_entropy_review_tracks_clarified_glyphs_and_marks_absent_glyphs(experiments):
    script = importlib.import_module("compare_glyph_entropy")
    grid = Grid(np.array([[0, 0, 1, 2]], np.uint8), np.zeros((1, 4, 3), np.float32))
    metrics = script.tracked_glyph_metrics(grid, "MW´")
    assert metrics["requested_glyph_share"] == 1
    assert metrics["tracked_glyph_shares"]["M"] == 0.5
    assert metrics["tracked_glyph_shares"]["´"] == 0.25
    assert metrics["tracked_glyph_shares"]["@"] is None
    assert metrics["tracked_glyph_shares"]["`"] is None
    assert metrics["requested_glyphs_present"] == ["M", "W", "´"]
    assert len(script.SOURCES) == 4


@pytest.mark.parametrize("name", [
    "compare_glyph_luminance_bonus", "compare_glyph_saturation_loss", "compare_gradient_loss",
    "compare_color_spaces", "compare_logit_noise", "compare_local_detail_losses",
    "compare_structure_glyphs", "compare_bracket_glyphs", "compare_structure_headroom",
])
def test_comparison_entry_points_run_on_a_small_fixture(experiments, monkeypatch, tmp_path, name):
    """Exercise real solves/report output while keeping sweeps small."""
    script = importlib.import_module(name)
    path = tmp_path / "source.png"
    pixels = np.zeros((8, 12, 3), np.uint8)
    pixels[:, 6:] = (170, 70, 120)
    Image.fromarray(pixels).save(path)
    crop = (1, 1, 8, 7)
    for target in (experiments, script):
        for setting, value in (("WIDTH", 12), ("SIZE", (12, 8)), ("CELL", (3, 4)),
                               ("STEPS", 1), ("CANDIDATES", 2), ("SOURCE", path),
                               ("SOURCES", (("fixture", path, crop),))):
            if hasattr(target, setting):
                monkeypatch.setattr(target, setting, value)
    monkeypatch.setattr(script, "OUTPUT", tmp_path / name)
    if hasattr(script, "BONUSES"):
        monkeypatch.setattr(script, "BONUSES", (0.0, 0.12))
    if name == "compare_gradient_loss":
        monkeypatch.setattr(script, "GRADIENT_STRENGTHS", (0.0, 1.5))
    if name == "compare_glyph_saturation_loss":
        monkeypatch.setattr(script, "VARIANTS", (("current", "β=0", 0, 0), ("sat", "β=.12", 0.12, None)))
    if name in ("compare_color_spaces", "compare_logit_noise"):
        monkeypatch.setattr(script, "SOURCES", [("fixture", path)])
    if name == "compare_logit_noise":
        monkeypatch.setattr(script, "DITHER_LEVELS", (0.005,))
        monkeypatch.setattr(script, "PANEL_WIDTH", 48)
    if name == "compare_local_detail_losses":
        monkeypatch.setattr(script, "SOURCES", [("fixture", path, crop, [(2, 2, 5, 5)])])
    script.main()
    reports = list(script.OUTPUT.glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["reproducibility"]["runs"]
    assert list(script.OUTPUT.rglob("comparison.png"))


def test_color_strategy_cli_runs_on_a_small_fixture(experiments, monkeypatch, tmp_path):
    script = importlib.import_module("compare_color_strategies")
    path = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (120, 60, 170)).save(path)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [str(Path(script.__file__)), str(path), "--output", str(output),
                                      "--size", "12x8", "--cell", "3x4", "--steps", "1"])
    script.main()
    report = json.loads((output / "comparison.json").read_text())
    assert len(report["variants"]) == 7
    assert len(report["reproducibility"]["runs"]) == 5


def test_error_movement_comparison_runs_with_production_defaults(experiments, monkeypatch, tmp_path):
    script = importlib.import_module("compare_error_movement")
    path = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (120, 60, 170)).save(path)
    output = tmp_path / "comparison"
    monkeypatch.setattr(script, "SOURCES", (("tail", path, (0, 0, 960, 640)),))
    monkeypatch.setattr(sys, "argv", [script.__file__, "--width", "12", "--cells", "3x4",
                                      "--steps", "1", "--strengths", "4", "--output", str(output)])
    script.main()
    report = json.loads((output / "metrics.json").read_text())
    variants = report["images"]["tail"]["grids"]["3x4"]
    assert set(variants) == {"before", "movement_4"}
    assert report["production_default"] == 0
    assert variants["before"]["experiment"]["effective_objectives"]["glyph_error_movement_strength"] == 0
    assert variants["movement_4"]["experiment"]["effective_objectives"]["glyph_error_movement_strength"] == 4
    assert (output / "tail/3x4/focus_before_after_4.png").exists()


@pytest.mark.parametrize("with_reference", [False, True])
def test_discrete_layout_comparison_records_actual_objectives_and_source_snapshots(experiments, monkeypatch, tmp_path, with_reference):
    script = importlib.import_module("compare_discrete_layout")
    path = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (120, 60, 170)).save(path)
    baseline = tmp_path / "baseline.py"
    baseline.write_bytes(Path(optimizer.__file__).read_bytes())
    output = tmp_path / "comparison"
    monkeypatch.setattr(script, "SOURCES", (("fixture", path, (0, 0, 960, 640)),))
    monkeypatch.setattr(experiments, "DEFAULT_CHARS", " .#")
    arguments = [script.__file__, "--width", "12", "--cells", "3x4",
                 "--steps", "1", "--baseline-file", str(baseline), "--output", str(output)]
    if with_reference:
        arguments.extend(["--reference-file", str(baseline)])
    monkeypatch.setattr(sys, "argv", arguments)
    script.main()
    report = json.loads((output / "metrics.json").read_text())
    variants = report["images"]["fixture"]["grids"]["3x4"]
    expected = {"before", "soft_exact", "hard_exact", "hard_layout"}
    if with_reference:
        expected.add("reference")
        assert report["reference_sha256"] == hashlib.sha256((output / "optimizer_reference.py").read_bytes()).hexdigest()
    assert set(variants) == expected
    assert variants["hard_exact"]["hard_forward"]
    assert not variants["soft_exact"]["hard_forward"]
    assert variants["hard_exact"]["layout_loss_strength"] == 0
    assert variants["hard_layout"]["layout_loss_strength"] == 0.5
    assert all(np.isfinite(value["common_discrete_objective"]) for value in variants.values())
    assert report["renderer_sha256"] == hashlib.sha256((output / "optimizer_after.py").read_bytes()).hexdigest()
    assert report["baseline_sha256"] == hashlib.sha256((output / "optimizer_before.py").read_bytes()).hexdigest()
    assert (output / "fixture/3x4/focus_comparison.png").exists()
    assert (output / "review.md").exists()


@pytest.mark.parametrize("dither", [0.0, 0.005])
def test_glyph_search_comparison_records_full_bank_and_shortlist(experiments, monkeypatch, tmp_path, dither):
    script = importlib.import_module("compare_glyph_search")
    path = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (120, 60, 170)).save(path)
    output = tmp_path / "comparison"
    monkeypatch.setattr(script, "SOURCES", (("blush", path, (0, 0, 960, 640)),))
    monkeypatch.setattr(sys, "argv", [script.__file__, "--width", "12", "--cells", "3x4",
                                      "--steps", "1", "--strengths", "4", "--dither", str(dither),
                                      "--output", str(output)])
    script.main()
    report = json.loads((output / "metrics.json").read_text())
    variants = report["images"]["blush"]["grids"]["3x4"]
    assert len(variants) == 4
    full = variants["all_movement_4"]
    assert full["glyphs_per_cell"] == len(experiments.DEFAULT_CHARS)
    assert full["experiment"]["effective_objectives"]["use_shortlist"] is False
    assert variants["shortlist_movement_4"]["glyphs_per_cell"] == 6
    assert report["production_use_shortlist"] is False
    assert report["dither"] == dither
    assert full["experiment"]["effective_objectives"]["dither"] == dither
    assert 0 <= full["logit_noise"]["pre_refinement_winner_change_fraction"] <= 1
    if dither == 0:
        assert full["logit_noise"]["pre_refinement_winner_change_fraction"] == 0
    assert (output / "blush/3x4/factorial_focus_4.png").exists()
