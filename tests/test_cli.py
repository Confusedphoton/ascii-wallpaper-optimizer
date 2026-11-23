from __future__ import annotations

import json
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ascii_optimizer import cli
from ascii_optimizer.optimizer import Grid, render_grid
from ascii_optimizer.settings import normalize_options
from ascii_optimizer.tone import display_tone_lab, oklab_to_srgb, playback_rgb8, srgb_to_oklab


@pytest.fixture
def fast_batch(tmp_path: Path, monkeypatch):
    """Use cheap deterministic solvers to exercise real asset/cache behavior."""
    font = tmp_path / "font.ttf"
    font.write_bytes(b"test font content")
    masks = np.stack((np.zeros((2, 2)), np.ones((2, 2)))).astype(np.float32)
    monkeypatch.setattr(cli, "resolve_font", lambda _: font)
    monkeypatch.setattr(cli, "glyph_sdf_atlas", lambda *_: (masks, 0.0))
    calls = []

    def solve_still(target, masks, **kwargs):
        calls.append(kwargs["seed"])
        rows, cols = target.shape[0] // 2, target.shape[1] // 2
        glyph = np.full((rows, cols), int(target.mean() > 0.4), np.uint8)
        color = np.broadcast_to(srgb_to_oklab(target.mean((0, 1))), (rows, cols, 3)).copy()
        return Grid(glyph, color, np.zeros(3, np.float32))

    monkeypatch.setattr(cli, "solve_style", solve_still)

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


@pytest.mark.parametrize("strength", [0, 0.004, 0.025, 0.1])
def test_cli_entropy_option_is_adjustable(strength):
    args = cli.make_parser().parse_args(["image.png", "--output", "render", "--size", "8x8",
                                        "--glyph-entropy-strength", str(strength)])
    assert normalize_options(vars(args))["glyph_entropy_strength"] == strength


def test_entropy_reaches_solver_and_invalidates_still_cache(fast_batch, monkeypatch):
    arguments, calls, _ = fast_batch
    original = cli.solve_style
    strengths = []
    def capture(*args, **kwargs):
        strengths.append(kwargs["glyph_entropy_strength"])
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, "solve_style", capture)
    args = arguments(count=1)
    cli.build(args)
    assert strengths == [0.01]
    assert json.loads((args.output / "manifest.json").read_text())["signature"]["glyph_entropy_strength"] == 0.01
    cli.build(args)
    assert len(calls) == 1
    args.glyph_entropy_strength = 0
    cli.build(args)
    assert strengths == [0.01, 0]
    assert json.loads((args.output / "manifest.json").read_text())["signature"]["glyph_entropy_strength"] == 0


def test_legacy_structure_style_is_rejected():
    with pytest.raises(ValueError, match="Unknown render style"):
        normalize_options({"style": "structure"})


def test_legacy_candidate_flag_does_not_restrict_rendering_or_split_cache(fast_batch):
    arguments, calls, _ = fast_batch
    args = arguments(count=1, candidates=1)
    cli.build(args)
    signature = json.loads((args.output / "manifest.json").read_text())["signature"]
    assert signature["glyph_search"] == "all"
    assert "candidates" not in signature
    assert len(calls) == 1
    args.candidates = 256
    cli.build(args)
    assert len(calls) == 1
    parser = cli.make_parser()
    assert "--candidates" not in parser.format_help()
    assert parser.parse_args(["image.png", "--output", "render", "--size", "8x8", "--candidates", "1"]).candidates == 1


def test_batch_uses_full_glyph_bank_with_legacy_candidate_limit(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (190, 120, 140)).save(source)
    output = tmp_path / "output"
    args = cli.make_parser().parse_args([str(source), "--output", str(output), "--size", "8x8",
        "--cell", "4x4", "--chars", " .#|", "--candidates", "1", "--steps", "1", "--device", "cpu"])
    cli.build(args)
    assert (output / "stills/0000/preview.png").exists()


def test_still_cache_reuses_transition_variants_and_keeps_copies_independent(fast_batch):
    arguments, calls, _ = fast_batch
    first = arguments()
    cli.build(first)
    assert len(calls) == 2
    old_map = (first.output / "stills/0000/color.png").read_bytes()
    second = arguments("second", frames=5, transition_steps=7, transition_field="ripple",
                       transition_type="earth-mover")
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


@pytest.mark.parametrize("transition_type", ["glyph-path", "earth-mover"])
def test_transition_endpoints_copy_exact_still_bytes_and_manifest_describes_every_frame(fast_batch, transition_type):
    arguments, _, _ = fast_batch
    args = arguments(generation="requested-generation", transition_type=transition_type)
    cli.build(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["generation"] == "requested-generation"
    assert manifest["output_size"] == [4, 4]
    assert manifest["cell_size"] == [2, 2]
    assert manifest["atlas_count"] == 2
    assert manifest["signature"]["transition_type"] == transition_type
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
@pytest.mark.parametrize("field", ["blue-noise", "salience"])
def test_slideshow_source_memory_is_bounded_by_workers(fast_batch, monkeypatch, workers, field):
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
    cli.build(arguments(count=8, transition_workers=workers, transition_field=field))
    assert 1 <= maximum <= workers
    # Only salience needs a source image during transitions, once per pair.
    assert len(references) == 8 * (2 if field == "salience" else 1)
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("style,perceptual", [("headroom", True), ("headroom", False),
                                             ("flow", True)])
@pytest.mark.parametrize("invert_ink", [False, True])
def test_transition_colors_follow_saved_keyframes_without_refitting(
        fast_batch, monkeypatch, style, perceptual, invert_ink):
    arguments, _, masks = fast_batch
    args = arguments(style=style, perceptual_loss=perceptual, invert_ink=invert_ink,
                     frames=5, transition_pace="linear", transition_spread=0.5)
    start = Grid(np.ones((2, 2), np.uint8),
                 srgb_to_oklab(np.full((2, 2, 3), 0.25, np.float32)),
                 srgb_to_oklab(np.full(3, 0.05, np.float32)))
    end = Grid(start.glyph.copy(), srgb_to_oklab(np.full((2, 2, 3), 0.75, np.float32)),
               srgb_to_oklab(np.full(3, 0.15, np.float32)))
    start = cli._save_grid(start, args.output / "stills/0000", False, masks, args.size, invert_ink)
    end = cli._save_grid(end, args.output / "stills/0001", False, masks, args.size, invert_ink)
    # Half the cells wait until halfway through; the rest start immediately.
    monkeypatch.setattr(cli, "phase_offsets", lambda *args: np.array([[0, 1], [0, 1]], np.float32))
    def unexpected(*args, **kwargs):
        raise AssertionError("Transition colors must come from saved keyframes")
    monkeypatch.setattr(cli, "fit_fixed_glyphs", unexpected)
    monkeypatch.setattr(cli, "load_target", unexpected)
    cli._render_transition(0, args.images, args.output, args, masks, (2, 2),
                           cli.glyph_paths(masks), lambda *_: None)
    pair = args.output / "transitions/0000-0001"
    first = cli._load_grid(pair / "0001")
    # A delayed cell keeps its exact keyframe ink instead of darkening on entry.
    np.testing.assert_array_equal(playback_rgb8(first.color[:, 1]), playback_rgb8(start.color[:, 1]))
    np.testing.assert_array_equal(playback_rgb8(first.color[:, 0]),
                                  playback_rgb8((start.color[:, 0] + end.color[:, 0]) / 2))
    np.testing.assert_array_equal(playback_rgb8(first.background),
                                  playback_rgb8(0.75 * start.background + 0.25 * end.background))
    # RGB blending would produce 128 here; the perceptual midpoint is darker.
    assert playback_rgb8(first.color)[0, 0, 0] < 128
    for frame, endpoint in ((0, start), (4, end)):
        np.testing.assert_array_equal(playback_rgb8(cli._load_grid(pair / f"{frame:04d}").color),
                                      playback_rgb8(endpoint.color))


def test_earth_mover_prepared_frame_moves_ink_to_a_middle_cell(fast_batch):
    arguments, _, masks = fast_batch
    args = arguments(frames=5, transition_type="earth-mover", transition_pace="linear",
                     transition_spread=0, size=(10, 2))
    glyphs = np.array([[1, 0, 0, 0, 0]], np.uint8)
    colors = np.broadcast_to(np.array([0.8, 0, 0], np.float32), (1, 5, 3)).copy()
    background = np.zeros(3, np.float32)
    cli._save_grid(Grid(glyphs, colors, background), args.output / "stills/0000", False, masks, args.size)
    cli._save_grid(Grid(glyphs[:, ::-1], colors, background), args.output / "stills/0001", False, masks, args.size)
    cli._render_transition(0, args.images, args.output, args, masks, (2, 2),
                           cli.glyph_paths(masks), lambda *_: None)
    midpoint = cli._load_grid(args.output / "transitions/0000-0001/0002")
    np.testing.assert_array_equal(midpoint.glyph, [[0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(playback_rgb8(midpoint.color[0, 2]), playback_rgb8(colors[0, 0]))


def test_transition_type_changes_reuse_stills_but_regenerate_transitions(fast_batch, monkeypatch):
    arguments, calls, _ = fast_batch
    args = arguments()
    cli.build(args)
    rendered = []
    original = cli._render_transition

    def track(index, *args, **kwargs):
        rendered.append(index)
        return original(index, *args, **kwargs)

    monkeypatch.setattr(cli, "_render_transition", track)
    cli.build(arguments(transition_type="earth-mover"))
    assert calls == [0, 1]
    assert sorted(rendered) == [0, 1]
    rendered.clear()
    cli.build(arguments(transition_type="earth-mover"))
    assert rendered == []


@pytest.mark.parametrize("transition_type", [None, "earth-mover"])
def test_transition_review_supports_legacy_manifests_and_earth_mover(fast_batch, monkeypatch, transition_type):
    import sys
    from scripts import compare_transitions

    arguments, _, _ = fast_batch
    args = arguments(style="headroom", perceptual_loss=False)
    cli.build(args)
    original_path = args.output / "manifest.json"
    manifest = json.loads(original_path.read_text())
    manifest["signature"].pop("transition_type")
    original_path.write_text(json.dumps(manifest))
    original_bytes = original_path.read_bytes()
    output = args.output.parent / "review"
    command = ["compare_transitions", str(args.output), "--output", str(output)]
    if transition_type:
        command.extend(["--transition-type", transition_type])
    monkeypatch.setattr(sys, "argv", command)
    monkeypatch.setattr(compare_transitions.torch, "set_num_threads", lambda _: None)
    compare_transitions.main()
    revised = json.loads((output / "interpolated_assets/manifest.json").read_text())
    assert revised["signature"]["transition_type"] == (transition_type or "glyph-path")
    report = json.loads((output / "metrics.json").read_text())
    scene = report["images"]["slideshow"]
    assert scene["ppd"] == 67
    assert ("earth_mover" if transition_type else "oklab_interpolation") in scene["variants"]
    assert original_path.read_bytes() == original_bytes


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
    original = cli._render_transition
    barrier = threading.Barrier(2, timeout=10)
    active = set()

    def concurrent_pair(*args, **kwargs):
        active.add(threading.current_thread().name)
        barrier.wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_render_transition", concurrent_pair)
    args = arguments(transition_workers=2, progress_json=True)
    cli.build(args)
    assert len(active) == 2
    updates = [json.loads(line.removeprefix("ASCII_PROGRESS ")) for line in
               capsys.readouterr().out.splitlines() if line.startswith("ASCII_PROGRESS ")]
    progress = [update["progress"] for update in updates]
    assert progress == sorted(progress)
    assert progress[-1] == 1


@pytest.mark.parametrize("transition_type", ["glyph-path", "earth-mover"])
def test_parallel_transitions_match_serial_assets(fast_batch, transition_type):
    arguments, _, _ = fast_batch
    serial = arguments("serial", frames=5, transition_workers=1, transition_type=transition_type)
    parallel = arguments("parallel", frames=5, transition_workers=2, transition_type=transition_type)
    cli.build(serial)
    cli.build(parallel)
    for path in serial.output.rglob("*.png"):
        assert path.read_bytes() == (parallel.output / path.relative_to(serial.output)).read_bytes()


def test_parallel_failure_cancels_other_pairs_without_publishing(fast_batch, monkeypatch):
    arguments, _, _ = fast_batch
    barrier = threading.Barrier(2, timeout=10)

    def fail(*args, **kwargs):
        barrier.wait()
        raise RuntimeError("transition failed")

    monkeypatch.setattr(cli, "_render_transition", fail)
    args = arguments(count=8, transition_workers=2)
    with pytest.raises(RuntimeError, match="transition failed"):
        cli.build(args)
    assert not (args.output / "manifest.json").exists()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_counts_are_rejected(fast_batch, workers):
    arguments, _, _ = fast_batch
    with pytest.raises(ValueError, match="transition_workers"):
        cli.build(arguments(transition_workers=workers))
