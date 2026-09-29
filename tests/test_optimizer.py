from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ascii_optimizer.assets import blue_noise, cell_for_display, glyph_masks, linear_to_srgb, resolve_font, srgb_to_linear
from ascii_optimizer.cli import build, solve_style
from ascii_optimizer.optimizer import Grid, _feature_pixel_loss, _loss, render_grid, solve
from ascii_optimizer.service import _validated_request
from ascii_optimizer.structure import STRUCTURE_CHARS
from ascii_optimizer.tone import display_tone, encode_headroom
from ascii_optimizer.transition import ease, glyph_paths, local_progress, path_glyphs, phase_offsets
from ascii_optimizer import service


def test_render_uses_mask_once_and_preserves_linear_color():
    masks = np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
    grid = Grid(np.array([[0]], dtype=np.uint8),
                np.array([[[0.8, 0.4, 0.2]]], dtype=np.float32))
    image = render_grid(grid, masks)
    np.testing.assert_allclose(image[0, 1], [0.4, 0.2, 0.1], atol=1e-6)
    np.testing.assert_allclose(image[1, 0], [0.8, 0.4, 0.2], atol=1e-6)


def test_uniform_background_fills_glyph_gaps_without_changing_full_coverage():
    masks = np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
    grid = Grid(np.array([[0]], dtype=np.uint8),
                np.array([[[0.8, 0.4, 0.2]]], dtype=np.float32),
                np.array([0.2, 0.3, 0.4], dtype=np.float32))
    image = render_grid(grid, masks)
    np.testing.assert_allclose(image[0, 0], [0.2, 0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(image[0, 1], [0.5, 0.35, 0.3], atol=1e-6)
    np.testing.assert_allclose(image[1, 0], [0.8, 0.4, 0.2], atol=1e-6)


def test_fixed_and_dynamic_background_use_one_color_for_the_whole_grid():
    source = np.zeros((9, 9, 3), dtype=np.float32)
    source[:, :5] = [0.8, 0.4, 0.2]
    masks = np.stack((np.zeros((4, 3)), np.ones((4, 3))), axis=0).astype(np.float32)
    fixed = solve(source, masks, steps=2, background="#203040")
    np.testing.assert_allclose(fixed.background,
        srgb_to_linear(torch.tensor([32, 48, 64], dtype=torch.float32) / 255).numpy(), atol=1e-6)
    assert fixed.background.shape == (3,)
    dynamic = solve(source, masks, steps=2, background="auto")
    assert dynamic.background.shape == (3,)
    assert (dynamic.background >= 0).all()


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


def test_detail_losses_can_preserve_a_single_colored_pixel_over_a_larger_gray_patch():
    source = np.zeros((8, 8, 3), dtype=np.float32)
    source[:2, :2] = 0.65
    source[5, 5] = [1.0, 0.05, 0.8]
    masks = np.zeros((3, 8, 8), dtype=np.float32)
    masks[1, :2, :2] = 1
    masks[2, 5, 5] = 1
    settings = dict(steps=20, candidates=3, dither=0, background="#000000")
    assert solve(source, masks, detail_loss="default", **settings).glyph[0, 0] == 1
    assert solve(source, masks, detail_loss="local_color", **settings).glyph[0, 0] == 2
    assert solve(source, masks, detail_loss="color_tail", **settings).glyph[0, 0] == 2
    assert solve(source, masks, detail_loss="feature_chroma", feature_strength=0.5, **settings).glyph[0, 0] == 2


def test_feature_loss_gives_sparse_and_dense_feature_cells_equal_weight():
    error = torch.zeros((1, 1, 2, 4))
    selected = torch.zeros_like(error)
    error[0, 0, 0, 0] = 1
    selected[0, 0, 0, 0] = 1
    selected[0, 0, :, 2:] = 1
    assert _feature_pixel_loss(error, selected, (2, 2)).item() == 0.5


def test_blue_noise_is_seeded_and_low_frequency_suppressed():
    first = blue_noise((64, 64), 7)
    np.testing.assert_array_equal(first, blue_noise((64, 64), 7))
    assert not np.array_equal(first, blue_noise((64, 64), 8))
    spectrum = np.abs(np.fft.fft2(first - 0.5)) ** 2
    assert spectrum[0:3, 0:3].mean() < spectrum[16:24, 16:24].mean()


def test_transition_paths_walk_neighbor_glyphs_without_skipping():
    masks = np.zeros((4, 2, 2), dtype=np.float32)
    masks[1, 0, 0] = 1
    masks[2, 0] = 1
    masks[3, :2, :2] = 1
    paths = glyph_paths(masks, neighbors=1)
    start = np.array([[0]], dtype=np.uint8)
    end = np.array([[3]], dtype=np.uint8)
    path = paths[0][3]
    assert len(path) > 2
    previous = start
    seen = [0]
    for frame in range(1, len(path)):
        progress = np.array([[frame / (len(path) - 1)]], dtype=np.float32)
        current = path_glyphs(start, end, progress, paths, previous, len(path) - 1 - frame)
        seen.append(int(current[0, 0]))
        previous = current
    assert seen == path


def test_transition_offsets_and_pacing_are_deterministic():
    offsets = phase_offsets((16, 16), "blue-noise", 4)
    np.testing.assert_array_equal(offsets, phase_offsets((16, 16), "blue-noise", 4))
    source = np.zeros((32, 32, 3), dtype=np.float32)
    source[:, 16:] = 1
    for field in ("ripple", "spiral", "directional", "edge", "salience"):
        phases = phase_offsets((16, 16), field, 4, source)
        assert phases.shape == offsets.shape and 0 <= phases.min() <= phases.max() <= 1
    local = local_progress(0.5, offsets, 0.65)
    assert local.min() < 0.1 and local.max() > 0.9
    assert ease(0.25, "ease-in") < 0.25 < ease(0.25, "ease-out")
    assert np.array_equal(local_progress(0, offsets, 0.65), np.zeros_like(offsets))
    assert np.array_equal(local_progress(1, offsets, 0.65), np.ones_like(offsets))


def test_profile_override_persists(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(service, "OVERRIDE_PATH", tmp_path / "override.json")
    assert service.get_override() == "auto"
    assert service.set_override("B") == "B"
    assert service.get_override() == "B"


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
    custom = _validated_request({"images": [str(image)], "font": "Monospace",
        "chars": "\\s.#", "detail": 1, "grayscale": False, "frames": 2,
        "dither": 0.005, "background": "#203040", "size": [9, 9], "screen": "nonexistent"})
    assert request["background"] == "auto"
    assert custom["background"] == "#203040"
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
        candidates=3, dither=0.003, background="auto", seed=1, device="cpu", force=False)
    build(args)
    pair = output / "transitions" / "0000-0001"
    for kind in ("glyph", "color", "background"):
        assert (pair / "0000" / f"{kind}.png").read_bytes() == (output / "stills" / "0000" / f"{kind}.png").read_bytes()
        assert (pair / "0002" / f"{kind}.png").read_bytes() == (output / "stills" / "0001" / f"{kind}.png").read_bytes()
    font_paths = glyph_paths(glyph_masks(resolve_font(args.font), list(args.chars), args.cell))
    frames = [np.asarray(Image.open(pair / f"{frame:04d}" / "glyph.png"))[:, :, 0]
              for frame in range(args.frames)]
    for row, col in np.ndindex(frames[0].shape):
        path = font_paths[int(frames[0][row, col])][int(frames[-1][row, col])]
        steps = [path.index(int(frame[row, col])) for frame in frames]
        assert all(next_step - step in (0, 1) for step, next_step in zip(steps, steps[1:]))
    assert json.loads((output / "manifest.json").read_text())["frame_count"] == 3
    assert Image.open(output / "stills" / "0000" / "preview.png").size == (13, 9)
    assert Image.open(output / "stills" / "0000" / "background.png").size == (1, 1)
    metadata = np.asarray(Image.open(output / "metadata.png"))
    assert tuple(metadata[0, 0]) == (0, 13, 3)
    assert tuple(metadata[0, 1]) == (0, 9, 4)
    build(args)
    assert "Already rendered" in capsys.readouterr().out
    args.dither = 0.02
    build(args)
    assert json.loads((output / "manifest.json").read_text())["signature"]["dither"] == 0.02
    assert "Already rendered" not in capsys.readouterr().out
    args.background = "#203040"
    build(args)
    assert json.loads((output / "manifest.json").read_text())["signature"]["background"] == "#203040"


def test_headroom_encoding_restores_source_colors_after_display():
    colors = np.array([[[0.85, 0.65, 0.1], [0.3, 0.45, 0.72],
                        [0.05, 0.04, 0.03]]], dtype=np.float32)
    encoded = encode_headroom(colors)
    displayed = display_tone(srgb_to_linear(torch.from_numpy(encoded)).numpy())
    restored = linear_to_srgb(torch.from_numpy(displayed)).numpy()
    np.testing.assert_allclose(restored, colors, atol=0.002)
    # Mid-gamut colors also survive a solver-side polarity change.
    inverted = encode_headroom(colors[:, 1:2], invert_ink=True)
    output = display_tone(srgb_to_linear(torch.from_numpy(inverted)).numpy(), invert_ink=True)
    np.testing.assert_allclose(linear_to_srgb(torch.from_numpy(output)).numpy(),
                               colors[:, 1:2], atol=0.002)


def test_inverting_structure_changes_selected_glyphs():
    source = np.zeros((8, 8, 3), dtype=np.float32)
    source[:, :4] = 0.04
    source[:, 4:] = 0.96
    masks = np.zeros((3, 4, 4), dtype=np.float32)
    masks[1, :2, :2] = 1
    masks[2, :3, :3] = 1
    settings = dict(style="structure", background="auto", seed=1, steps=2,
                    candidates=3, dither=0, structure_dither=0, device="cpu")
    normal = solve_style(source, masks, invert_ink=False, **settings)
    inverted = solve_style(source, masks, invert_ink=True, **settings)
    assert not np.array_equal(normal.glyph, inverted.glyph)
    assert "|" in STRUCTURE_CHARS and all(char in STRUCTURE_CHARS for char in "[]{}")
    with pytest.raises(ValueError, match="not implemented for structure-first"):
        solve_style(source, masks, invert_ink=False, detail_loss="feature_chroma", **settings)


def test_inverting_headroom_changes_solved_ink():
    source = np.full((8, 8, 3), 0.7, dtype=np.float32)
    masks = np.zeros((3, 4, 4), dtype=np.float32)
    masks[1, :2, :2] = 1
    masks[2, :3, :3] = 1
    settings = dict(style="headroom", background="auto", seed=1, steps=2,
                    candidates=3, dither=0, structure_dither=0, device="cpu")
    normal = solve_style(source, masks, invert_ink=False, **settings)
    inverted = solve_style(source, masks, invert_ink=True, **settings)
    assert inverted.color.mean() < normal.color.mean() * 0.5


def test_render_style_and_inversion_are_saved_with_assets(tmp_path: Path):
    image = tmp_path / "input.png"
    Image.new("RGB", (12, 8), (100, 130, 170)).save(image)
    second = tmp_path / "second.png"
    Image.new("RGB", (12, 8), (160, 90, 70)).save(second)
    args = argparse.Namespace(images=[image, second], output=tmp_path / "out", size=(12, 8),
        font="DejaVu Sans Mono", chars=" .#", cell=(3, 4), detail=1,
        crop="fill", grayscale=False, frames=3, steps=2, transition_steps=2,
        candidates=3, dither=0.003, structure_dither=0.06, style="structure",
        invert_ink=True, background="auto", seed=1, device="cpu", force=False)
    build(args)
    metadata = np.asarray(Image.open(args.output / "metadata.png"))
    assert tuple(metadata[0, 2]) == (2, 1, 0)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["signature"]["style"] == "structure"
    assert manifest["signature"]["invert_ink"] is True
    for kind in ("glyph", "color", "background"):
        assert (args.output / "transitions/0000-0001/0000" / f"{kind}.png").read_bytes() == (args.output / "stills/0000" / f"{kind}.png").read_bytes()
        assert (args.output / "transitions/0000-0001/0002" / f"{kind}.png").read_bytes() == (args.output / "stills/0001" / f"{kind}.png").read_bytes()
    assert (args.output / "transitions/0000-0001/0001/glyph.png").exists()
    request = _validated_request({"images": [str(image)], "font": "Monospace",
        "chars": " .#", "detail": 1, "grayscale": False, "frames": 2,
        "dither": 0.005, "structure_dither": 0.06, "style": "structure",
        "invert_ink": True, "size": [12, 8], "screen": "nonexistent"})
    assert request["style"] == "structure" and request["invert_ink"] is True
