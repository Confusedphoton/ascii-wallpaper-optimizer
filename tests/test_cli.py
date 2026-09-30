from __future__ import annotations

import json
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ascii_optimizer import cli
from ascii_optimizer.optimizer import Grid, render_grid
from ascii_optimizer.settings import normalize_options
from ascii_optimizer.tone import display_tone_lab, oklab_to_srgb, srgb_to_oklab


@pytest.fixture
def fast_batch(tmp_path: Path, monkeypatch):
    """Use cheap deterministic solvers to exercise real asset/cache behavior."""
    font = tmp_path / "font.ttf"
    font.write_bytes(b"test font content")
    masks = np.stack((np.zeros((2, 2)), np.ones((2, 2)))).astype(np.float32)
    monkeypatch.setattr(cli, "resolve_font", lambda _: font)
    monkeypatch.setattr(cli, "glyph_masks", lambda *_: masks)
    calls = []

    def solve_still(target, masks, **kwargs):
        calls.append(kwargs["seed"])
        rows, cols = target.shape[0] // 2, target.shape[1] // 2
        glyph = np.full((rows, cols), int(target.mean() > 0.4), np.uint8)
        color = np.broadcast_to(srgb_to_oklab(target.mean((0, 1))), (rows, cols, 3)).copy()
        return Grid(glyph, color, np.zeros(3, np.float32))

    def fit_colors(target, masks, glyphs, **kwargs):
        color = np.broadcast_to(target.mean((0, 1)), (*glyphs.shape, 3)).copy()
        return Grid(glyphs.copy(), color, kwargs["background_oklab"])

    monkeypatch.setattr(cli, "solve_style", solve_still)
    monkeypatch.setattr(cli, "fit_fixed_glyphs", fit_colors)

    def arguments(name="first", count=2, **overrides):
        images = []
        for index in range(count):
            path = tmp_path / f"source-{index}.png"
            if not path.exists():
                Image.new("RGB", (4, 4), (40 + index * 35,) * 3).save(path)
            images.append(path)
        parsed = cli.make_parser().parse_args([
            *map(str, images), "--output", str(tmp_path / name), "--size", "4x4",
            "--cell", "2x2", "--chars", " .", "--frames", "3", "--steps", "2",
            "--transition-steps", "2", "--still-cache", str(tmp_path / "cache"),
            "--device", "cpu",
        ])
        vars(parsed).update(overrides)
        return parsed

    return arguments, calls, masks


def test_cli_defaults_follow_shared_validation():
    args = cli.make_parser().parse_args(["example.png", "--output", "render", "--size", "8x6"])
    options = normalize_options(vars(args))
    for name, expected in normalize_options({}).items():
        assert options[name] == expected


def test_still_cache_reuses_transition_variants_and_keeps_copies_independent(fast_batch):
    arguments, calls, _ = fast_batch
    first = arguments()
    cli.build(first)
    assert len(calls) == 2
    old_map = (first.output / "stills/0000/color.png").read_bytes()
    second = arguments("second", frames=5, transition_steps=7, transition_field="ripple")
    cli.build(second)
    assert len(calls) == 2
    assert (second.output / "stills/0000/color.png").read_bytes() == old_map
    (first.output / "stills/0000/color.png").write_bytes(b"unrelated overwrite")
    third = arguments("third", frames=3)
    cli.build(third)
    assert len(calls) == 2
    assert (third.output / "stills/0000/color.png").read_bytes() == old_map
    assert json.loads((first.output / "manifest.json").read_text())["frame_count"] == 3
    assert json.loads((second.output / "manifest.json").read_text())["frame_count"] == 5


def test_force_rebuilds_stills_and_atomically_replaces_cache_index(fast_batch):
    arguments, calls, _ = fast_batch
    first = arguments()
    cli.build(first)
    indexes = list(first.still_cache.glob("*/cache.json"))
    generations = {path: json.loads(path.read_text())["generation"] for path in indexes}
    second = arguments("second", force=True)
    cli.build(second)
    assert len(calls) == 4
    for path, generation in generations.items():
        replacement = json.loads(path.read_text())["generation"]
        assert replacement != generation
        assert (path.parent / generation / "color.png").is_file()
        assert (path.parent / replacement / "color.png").is_file()
    assert not list(first.still_cache.rglob("*.tmp"))


def test_cache_identity_includes_content_and_effective_seed(fast_batch):
    arguments, calls, _ = fast_batch
    first = arguments(count=1)
    cli.build(first)
    second = arguments("second", count=1, seed=9)
    cli.build(second)
    assert calls == [0, 9]
    Image.new("RGB", (4, 4), (200, 100, 70)).save(first.images[0])
    cli.build(arguments("third", count=1, seed=9))
    assert calls == [0, 9, 9]


def test_corrupted_still_cache_is_regenerated(fast_batch):
    arguments, calls, _ = fast_batch
    first = arguments(count=1)
    cli.build(first)
    index = next(first.still_cache.glob("*/cache.json"))
    generation = json.loads(index.read_text())["generation"]
    (index.parent / generation / "color.png").write_bytes(b"invalid PNG")
    second = arguments("second", count=1)
    cli.build(second)
    assert calls == [0, 0]
    cli._load_grid(second.output / "stills/0000")


def test_preview_and_manifest_backgrounds_use_the_serialized_maps(tmp_path: Path):
    masks = np.array([[[0, 0.25], [0.5, 1]]], dtype=np.float32)
    grid = Grid(np.array([[0]], dtype=np.uint8),
                np.array([[[0.7, 0.35, 0.25]]], dtype=np.float32),
                np.array([0.22, 0.01, -0.03], dtype=np.float32))
    cli._save_grid(grid, tmp_path, True, masks, (2, 2), invert_ink=True)
    persisted = cli._load_grid(tmp_path)
    expected = np.rint(oklab_to_srgb(display_tone_lab(
        render_grid(persisted, masks), invert_ink=True)) * 255).astype(np.uint8)
    with Image.open(tmp_path / "preview.png") as image:
        np.testing.assert_array_equal(np.asarray(image), expected)


def test_transition_endpoints_copy_exact_still_bytes_and_manifest_describes_every_frame(fast_batch):
    arguments, _, _ = fast_batch
    args = arguments(generation="requested-generation")
    cli.build(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["generation"] == "requested-generation"
    assert manifest["output_size"] == [4, 4]
    assert manifest["cell_size"] == [2, 2]
    assert manifest["atlas_count"] == 2
    assert len(manifest["background_labs"]) == 2 + 2 * args.frames
    for index in range(2):
        following = (index + 1) % 2
        pair = args.output / f"transitions/{index:04d}-{following:04d}"
        for frame, still_index in ((0, index), (args.frames - 1, following)):
            for kind in cli.MAP_KINDS:
                assert (pair / f"{frame:04d}/{kind}.png").read_bytes() == (
                    args.output / f"stills/{still_index:04d}/{kind}.png").read_bytes()
        for frame in range(args.frames):
            relative = f"transitions/{index:04d}-{following:04d}/{frame:04d}"
            np.testing.assert_array_equal(manifest["background_labs"][relative],
                                          cli._load_grid(args.output / relative).background)


@pytest.mark.parametrize("workers", [1, 2])
def test_slideshow_source_memory_is_bounded_by_workers(fast_batch, monkeypatch, workers):
    arguments, _, _ = fast_batch
    original = cli.load_target
    references = []
    maximum = 0

    def track_target(*args, **kwargs):
        nonlocal maximum
        result = original(*args, **kwargs)
        references.append(weakref.ref(result))
        maximum = max(maximum, sum(reference() is not None for reference in references))
        return result

    monkeypatch.setattr(cli, "load_target", track_target)
    cli.build(arguments(count=8, transition_workers=workers))
    assert 2 <= maximum <= 2 * workers
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("invert_ink", [False, True])
@pytest.mark.parametrize("background", ["#000000", "#ffffff", "#203040"])
def test_fixed_background_is_the_selected_display_color(invert_ink, background):
    rgb = np.array([int(background[index:index + 2], 16) / 255
                    for index in (1, 3, 5)], dtype=np.float32)
    _, encoded = cli._backgrounds(srgb_to_oklab(np.zeros((2, 2, 3), np.float32)),
                                 background, invert_ink)
    displayed = oklab_to_srgb(display_tone_lab(encoded, invert_ink=invert_ink))
    np.testing.assert_allclose(displayed, rgb, atol=2 / 255)


def test_failed_render_does_not_publish_a_partial_manifest(fast_batch, monkeypatch):
    arguments, _, _ = fast_batch

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(cli, "solve_style", fail)
    args = arguments()
    with pytest.raises(RuntimeError, match="solver failed"):
        cli.build(args)
    assert not (args.output / "manifest.json").exists()
    assert not list(args.output.glob("*.tmp"))


def test_transition_pairs_overlap_and_progress_never_goes_backwards(fast_batch, monkeypatch, capsys):
    arguments, _, _ = fast_batch
    original = cli.fit_fixed_glyphs
    barrier = threading.Barrier(2, timeout=10)
    active = set()

    def concurrent_fit(*args, **kwargs):
        active.add(threading.current_thread().name)
        barrier.wait()
        kwargs["on_progress"](0.5, "Fitting colors")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "fit_fixed_glyphs", concurrent_fit)
    args = arguments(transition_workers=2, progress_json=True)
    cli.build(args)
    assert len(active) == 2
    updates = [json.loads(line.removeprefix("ASCII_PROGRESS ")) for line in
               capsys.readouterr().out.splitlines() if line.startswith("ASCII_PROGRESS ")]
    progress = [update["progress"] for update in updates]
    assert progress == sorted(progress)
    assert progress[-1] == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parallel_real_color_solver_matches_serial_assets(fast_batch, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU is available")
    from ascii_optimizer.optimizer import fit_fixed_glyphs

    arguments, _, _ = fast_batch
    streams = set()

    def real_fit(*args, **kwargs):
        if device == "cuda":
            streams.add(torch.cuda.current_stream().cuda_stream)
        return fit_fixed_glyphs(*args, **kwargs)

    monkeypatch.setattr(cli, "fit_fixed_glyphs", real_fit)
    serial = arguments("serial", frames=5, device=device, transition_workers=1)
    parallel = arguments("parallel", frames=5, device=device, transition_workers=2)
    cli.build(serial)
    streams.clear()
    cli.build(parallel)
    if device == "cuda":
        assert len(streams) == 2
    for path in serial.output.rglob("*.png"):
        assert path.read_bytes() == (parallel.output / path.relative_to(serial.output)).read_bytes()


def test_parallel_failure_cancels_other_pairs_without_publishing(fast_batch, monkeypatch):
    arguments, _, _ = fast_batch
    barrier = threading.Barrier(2, timeout=10)

    def fail(*args, **kwargs):
        barrier.wait()
        raise RuntimeError("transition failed")

    monkeypatch.setattr(cli, "fit_fixed_glyphs", fail)
    args = arguments(count=8, transition_workers=2)
    with pytest.raises(RuntimeError, match="transition failed"):
        cli.build(args)
    assert not (args.output / "manifest.json").exists()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_counts_are_rejected(fast_batch, workers):
    arguments, _, _ = fast_batch
    with pytest.raises(ValueError, match="transition_workers"):
        cli.build(arguments(transition_workers=workers))
