from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ascii_optimizer.assets import blue_noise, cell_for_display
from ascii_optimizer.cli import build
from ascii_optimizer.optimizer import Grid, _loss, render_grid, solve
from ascii_optimizer.service import _validated_request


def test_render_uses_mask_once_and_preserves_linear_color():
    masks = np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
    grid = Grid(np.array([[0]], dtype=np.uint8),
                np.array([[[0.8, 0.4, 0.2]]], dtype=np.float32))
    image = render_grid(grid, masks)
    np.testing.assert_allclose(image[0, 1], [0.4, 0.2, 0.1], atol=1e-6)
    np.testing.assert_allclose(image[1, 0], [0.8, 0.4, 0.2], atol=1e-6)


def test_underpaint_fills_glyph_gaps_without_changing_full_coverage():
    masks = np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
    grid = Grid(np.array([[0]], dtype=np.uint8),
                np.array([[[0.8, 0.4, 0.2]]], dtype=np.float32),
                np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32))
    image = render_grid(grid, masks)
    np.testing.assert_allclose(image[0, 0], [0.2, 0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(image[0, 1], [0.5, 0.35, 0.3], atol=1e-6)
    np.testing.assert_allclose(image[1, 0], [0.8, 0.4, 0.2], atol=1e-6)


def test_detail_uses_physical_display_size():
    cells = [cell_for_display((2560, 1600), detail) for detail in range(3)]
    assert cells == [(14, 28), (21, 42), (32, 64)]
    assert [(2560 + w - 1) // w for w, _ in cells] == [183, 122, 80]
    assert cell_for_display((1280, 800), 2) == (16, 32)


def test_full_frame_loss_couples_cells_across_boundary():
    target = torch.zeros((1, 3, 8, 8))
    rendered = target.clone()
    rendered[:, :, :, 4:] = 0.4
    rendered.requires_grad_()
    _loss(rendered, target).backward()
    # The left cell is locally exact; its gradient comes from multiscale context.
    assert rendered.grad[:, :, :, 3].abs().sum() > 0


def test_blue_noise_is_seeded_and_low_frequency_suppressed():
    first = blue_noise((64, 64), 7)
    np.testing.assert_array_equal(first, blue_noise((64, 64), 7))
    assert not np.array_equal(first, blue_noise((64, 64), 8))
    spectrum = np.abs(np.fft.fft2(first - 0.5)) ** 2
    assert spectrum[0:3, 0:3].mean() < spectrum[16:24, 16:24].mean()


def test_solve_deterministic_and_keeps_grid_shape():
    source = np.zeros((9, 9, 3), dtype=np.float32)
    source[:, :5] = 0.8
    masks = np.stack((np.zeros((4, 3)), np.ones((4, 3))), axis=0).astype(np.float32)
    first = solve(source, masks, steps=2, seed=3)
    second = solve(source, masks, steps=2, seed=3)
    assert first.glyph.shape == (3, 3)
    np.testing.assert_array_equal(first.glyph, second.glyph)
    np.testing.assert_allclose(first.color, second.color)


def test_progress_reports_completed_solve_and_space_glyph(tmp_path: Path):
    image = tmp_path / "input.png"
    Image.new("RGB", (9, 9), (160, 80, 40)).save(image)
    request = _validated_request({"images": [str(image)], "font": "Monospace",
        "chars": "\\s.#", "detail": 1, "grayscale": False, "frames": 2,
        "dither": 0.005, "size": [9, 9], "screen": "nonexistent"})
    assert request["chars"] == " .#"
    updates = []
    masks = np.stack((np.zeros((4, 3)), np.ones((4, 3))), axis=0).astype(np.float32)
    solve(np.zeros((9, 9, 3), dtype=np.float32), masks, steps=2,
          on_progress=lambda fraction, stage: updates.append((fraction, stage)))
    assert updates[-1][0] == 1.0
    assert all(left[0] <= right[0] for left, right in zip(updates, updates[1:]))


def test_batch_transition_endpoints_equal_stills(tmp_path: Path, capsys):
    images = []
    for index in range(2):
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[:, index * 5:index * 5 + 5] = [220, 130, 60]
        path = tmp_path / f"{index}.png"
        Image.fromarray(image).save(path)
        images.append(path)
    output = tmp_path / "out"
    args = argparse.Namespace(images=images, output=output, size=(13, 9),
        font="DejaVu Sans Mono", chars=" .#", cell=(3, 4), detail=1,
        crop="fill", grayscale=False, frames=3, steps=2, transition_steps=2,
        candidates=3, dither=0.003, seed=1, device="cpu", force=False)
    build(args)
    pair = output / "transitions" / "0000-0001"
    for kind in ("glyph", "color", "background"):
        assert (pair / "0000" / f"{kind}.png").read_bytes() == (output / "stills" / "0000" / f"{kind}.png").read_bytes()
        assert (pair / "0002" / f"{kind}.png").read_bytes() == (output / "stills" / "0001" / f"{kind}.png").read_bytes()
    assert json.loads((output / "manifest.json").read_text())["frame_count"] == 3
    assert Image.open(output / "stills" / "0000" / "preview.png").size == (13, 9)
    metadata = np.asarray(Image.open(output / "metadata.png"))
    assert tuple(metadata[0, 0]) == (0, 13, 3)
    assert tuple(metadata[0, 1]) == (0, 9, 4)
    build(args)
    assert "Already rendered" in capsys.readouterr().out
    args.dither = 0.02
    build(args)
    assert json.loads((output / "manifest.json").read_text())["signature"]["dither"] == 0.02
    assert "Already rendered" not in capsys.readouterr().out
