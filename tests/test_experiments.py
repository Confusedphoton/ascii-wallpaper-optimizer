from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ascii_optimizer import cli
from ascii_optimizer.defaults import DEFAULT_GLYPH_LUMINANCE_BONUS
from ascii_optimizer.optimizer import Grid


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


def test_glyph_ink_lightness_is_the_stored_oklab_l_component(experiments):
    masks = np.ones((1, 2, 2), np.float32)
    grid = Grid(np.zeros((1, 2), np.uint8),
                np.array([[[0.6, 0.05, -0.04], [0.8, -0.03, 0.02]]], np.float32))
    metrics = experiments.glyph_metrics(grid, masks, "#")
    expected = experiments.playback_grid(grid).color[..., 0]
    assert metrics["mean_ink_oklab_lightness"] == pytest.approx(expected.mean())
    assert metrics["ink_oklab_lightness_std"] == pytest.approx(expected.std())


def test_report_records_hashes_effective_objectives_and_measurements(experiments, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), (90, 130, 150)).save(source)
    with Image.open(source) as image:
        target = np.asarray(image, np.float32) / 255
    masks = np.stack((np.zeros((2, 2)), np.ones((2, 2)))).astype(np.float32)
    grid = experiments.solve_style(target, masks, style="headroom", invert_ink=False, background="auto",
                                   seed=3, steps=1, dither=0, device="cpu",
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
    assert run["effective_objectives"]["glyphs_per_cell"] == len(masks)


def test_red_detail_metric_handles_a_crop_without_red_pixels(experiments):
    reference = np.zeros((4, 4, 3), np.float32)
    metrics = experiments.red_chroma_metrics(reference, reference, (0, 0, 4, 4))
    assert metrics["red_belt_sample_pixels"] == 0
    assert metrics["red_belt_chroma_retention"] is None
    json.dumps(metrics, allow_nan=False)
