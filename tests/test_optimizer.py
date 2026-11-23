from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ascii_optimizer import optimizer, service
from ascii_optimizer.assets import (
    blue_noise,
    cell_for_display,
    glyph_sdf_atlas,
    linear_to_srgb,
    resolve_font,
    sdf_spread_for_cell,
    sdf_to_mask,
    srgb_to_linear,
)
from ascii_optimizer.cli import DEFAULT_CHARS, build, make_parser, solve_style
from ascii_optimizer.defaults import DEFAULT_GRADIENT_LOSS_STRENGTH
from ascii_optimizer.optimizer import Grid, _loss, render_grid, solve
from ascii_optimizer.service import _validated_request
from ascii_optimizer.settings import normalize_options
from ascii_optimizer.tone import display_tone, encode_headroom, srgb_to_oklab
from ascii_optimizer.transition import ease, glyph_paths, local_progress, path_glyphs, phase_offsets


@pytest.fixture(autouse=True)
def use_requested_screen_size(monkeypatch):
    monkeypatch.setattr(service, "_screen_size", lambda name, fallback: tuple(fallback))


@pytest.fixture
def batch_args():
    def create(*, images, output, size, **options):
        args = make_parser().parse_args([*map(str, images), "--output", str(output),
                                         "--size", f"{size[0]}x{size[1]}"])
        vars(args).update(options)
        return args
    return create


def test_render_uses_mask_once_and_preserves_oklab_color():
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


def test_equal_coverage_glyphs_follow_subglyph_pixel_arrangement():
    left = np.zeros((4, 4), np.float32)
    left[:, :2] = 1
    masks = np.stack((left[:, ::-1], left))
    source = np.repeat((left * 0.6)[..., None], 3, axis=-1)
    grid = optimizer.solve(source, masks, steps=0, dither=0, background="#000000",
                           gradient_loss_strength=0, glyph_entropy_strength=0)
    assert grid.glyph[0, 0] == 1


def test_fixed_and_dynamic_background_use_one_color_for_the_whole_grid():
    source = np.zeros((9, 9, 3), dtype=np.float32)
    source[:, :5] = [0.8, 0.4, 0.2]
    masks = np.stack((np.zeros((4, 3)), np.ones((4, 3))), axis=0).astype(np.float32)
    fixed = solve(source, masks, steps=2, background="#203040")
    np.testing.assert_allclose(fixed.background,
        srgb_to_oklab(np.array([32, 48, 64], dtype=np.float32) / 255), atol=1e-6)
    assert fixed.background.shape == (3,)
    dynamic = solve(source, masks, steps=2, background="auto")
    assert dynamic.background.shape == (3,)
    assert 0 <= dynamic.background[0] <= 1
    assert np.isfinite(dynamic.background).all()


def test_detail_uses_physical_display_size():
    cells = [cell_for_display((2560, 1600), detail) for detail in range(5)]
    assert cells == [(14, 28), (21, 42), (32, 64), (11, 22), (8, 16)]
    assert [(2560 + w - 1) // w for w, _ in cells] == [183, 122, 80, 233, 320]
    assert cell_for_display((1280, 800), 2) == (16, 32)


@pytest.mark.parametrize("suffix", ["", "B"])
def test_plasma_defaults_match_shared_options(suffix):
    namespace = {"k": "http://www.kde.org/standards/kcfg/1.0"}
    config = ET.parse(Path(__file__).resolve().parents[1] / "contents/config/main.xml").getroot()
    entries = {entry.get("name"): entry for entry in config.findall(".//k:entry", namespace)}
    encoded_chars = ",".join(
        r"\s" if char == " " else r"\\" if char == "\\" else char
        for char in DEFAULT_CHARS
    )
    assert entries["AsciiChars" + suffix].find("k:default", namespace).text == encoded_chars
    defaults = normalize_options({})
    for name, option in (("Detail", "detail"), ("Dither", "dither"),
                         ("GradientLossStrength", "gradient_loss_strength"),
                         ("GlyphLuminanceBonus", "glyph_luminance_bonus"),
                         ("GlyphEntropyStrength", "glyph_entropy_strength"),
                         ("TransitionFrames", "frames"), ("TransitionSpread", "transition_spread")):
        assert float(entries[name + suffix].find("k:default", namespace).text) == defaults[option]


def test_full_frame_loss_couples_cells_across_boundary():
    target = torch.zeros((1, 3, 8, 8))
    rendered = target.clone()
    rendered[:, :, :, 4:] = 0.4
    rendered.requires_grad_()
    _loss(rendered, target).backward()
    assert rendered.grad[:, :, :, 3].abs().sum() > 0


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


def test_batch_transition_endpoints_equal_stills(tmp_path: Path, capsys, batch_args):
    images = []
    for index in range(2):
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[:, index * 5:index * 5 + 5] = [220, 130, 60]
        path = tmp_path / f"{index}.png"
        Image.fromarray(image).save(path)
        images.append(path)
    output = tmp_path / "out"
    args = batch_args(images=images, output=output, size=(13, 9),
        font="DejaVu Sans Mono", chars=" .#", cell=(3, 4), detail=1,
        crop="fill", grayscale=False, frames=3, steps=2, transition_steps=2,
        candidates=3, dither=0.003, background="auto", seed=1, device="cpu", force=False)
    build(args)
    pair = output / "transitions" / "0000-0001"
    for kind in ("glyph", "color", "background"):
        assert (pair / "0000" / f"{kind}.png").read_bytes() == (output / "stills" / "0000" / f"{kind}.png").read_bytes()
        assert (pair / "0002" / f"{kind}.png").read_bytes() == (output / "stills" / "0001" / f"{kind}.png").read_bytes()
    sdf_atlas, spread = glyph_sdf_atlas(resolve_font(args.font), list(args.chars), args.cell)
    font_paths = glyph_paths(sdf_to_mask(sdf_atlas, spread=spread))
    frames = [np.asarray(Image.open(pair / f"{frame:04d}" / "glyph.png"))[:, :, 0]
              for frame in range(args.frames)]
    for row, col in np.ndindex(frames[0].shape):
        path = font_paths[int(frames[0][row, col])][int(frames[-1][row, col])]
        steps = [path.index(int(frame[row, col])) for frame in frames]
        assert all(next_step - step in (0, 1) for step, next_step in zip(steps, steps[1:]))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frame_count"] == 3
    assert manifest["atlas_type"] == "sdf"
    assert manifest["sdf_spread"] == spread
    assert Image.open(output / "stills" / "0000" / "preview.png").size == (13, 9)
    assert Image.open(output / "stills" / "0000" / "background.png").size == (1, 1)
    metadata = np.asarray(Image.open(output / "metadata.png"))
    assert tuple(metadata[0, 0]) == (0, 13, 3)
    assert tuple(metadata[0, 1]) == (0, 9, 4)
    assert metadata[0, 2, 2] == round(spread * 16)
    assert manifest["signature"]["gradient_loss_strength"] == DEFAULT_GRADIENT_LOSS_STRENGTH
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
    inverted = encode_headroom(colors[:, 1:2], invert_ink=True)
    output = display_tone(srgb_to_linear(torch.from_numpy(inverted)).numpy(), invert_ink=True)
    np.testing.assert_allclose(linear_to_srgb(torch.from_numpy(output)).numpy(),
                               colors[:, 1:2], atol=0.002)


def test_inverting_headroom_changes_solved_ink():
    source = np.full((8, 8, 3), 0.7, dtype=np.float32)
    masks = np.zeros((3, 4, 4), dtype=np.float32)
    masks[1, :2, :2] = 1
    masks[2, :3, :3] = 1
    settings = dict(style="headroom", background="auto", seed=1, steps=2,
                    dither=0, device="cpu")
    normal = solve_style(source, masks, invert_ink=False, **settings)
    inverted = solve_style(source, masks, invert_ink=True, **settings)
    from ascii_optimizer.tone import display_tone_lab
    normal_ink = display_tone_lab(normal.color)
    inverted_ink = display_tone_lab(inverted.color, invert_ink=True)
    assert inverted_ink[..., 0].mean() < normal_ink[..., 0].mean() * 0.5


def test_render_style_and_inversion_are_saved_with_assets(tmp_path: Path, batch_args):
    image = tmp_path / "input.png"
    Image.new("RGB", (12, 8), (100, 130, 170)).save(image)
    second = tmp_path / "second.png"
    Image.new("RGB", (12, 8), (160, 90, 70)).save(second)
    args = batch_args(images=[image, second], output=tmp_path / "out", size=(12, 8),
        font="DejaVu Sans Mono", chars=" .#", cell=(3, 4), detail=1,
        crop="fill", grayscale=False, frames=3, steps=2, transition_steps=2,
        candidates=3, dither=0.003, style="headroom",
        invert_ink=True, background="auto", seed=1, device="cpu", force=False)
    build(args)
    metadata = np.asarray(Image.open(args.output / "metadata.png"))
    expected_spread_byte = round(sdf_spread_for_cell((3, 4)) * 16)
    assert tuple(metadata[0, 2]) == (2, 1, expected_spread_byte)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["signature"]["style"] == "headroom"
    assert manifest["signature"]["invert_ink"] is True
    for kind in ("glyph", "color", "background"):
        assert (args.output / "transitions/0000-0001/0000" / f"{kind}.png").read_bytes() == (args.output / "stills/0000" / f"{kind}.png").read_bytes()
        assert (args.output / "transitions/0000-0001/0002" / f"{kind}.png").read_bytes() == (args.output / "stills/0001" / f"{kind}.png").read_bytes()
    assert (args.output / "transitions/0000-0001/0001/glyph.png").exists()
    request = _validated_request({"images": [str(image)], "font": "Monospace",
        "chars": " .#", "detail": 1, "grayscale": False, "frames": 2,
        "dither": 0.005, "style": "headroom",
        "invert_ink": True, "size": [12, 8], "screen": "nonexistent"})
    assert request["style"] == "headroom" and request["invert_ink"] is True
