from __future__ import annotations

import json
import shutil
import subprocess
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ascii_optimizer import cli, video
from ascii_optimizer.optimizer import Grid
from ascii_optimizer.tone import srgb_to_oklab


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is not installed")
    frames = tmp_path / "source"
    frames.mkdir()
    for index in range(5):
        Image.new("RGB", (8, 4), (20 + index * 50, 35, 40)).save(frames / f"{index}.png")
    destination = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "4", "-i", str(frames / "%d.png"),
                    "-c:v", "ffv1", str(destination)], check=True, capture_output=True)
    return destination


@pytest.fixture
def video_batch(tmp_path, monkeypatch, clip):
    font = tmp_path / "font.ttf"
    font.write_bytes(b"test font")
    masks = np.stack((np.zeros((2, 2)), np.ones((2, 2)))).astype(np.float32)
    monkeypatch.setattr(cli, "resolve_font", lambda _: font)
    monkeypatch.setattr(cli, "glyph_masks", lambda *_: masks)
    targets = []

    def solve(target, masks, **kwargs):
        # Store weak references: decoding must not retain all source frames.
        targets.append(weakref.ref(target))
        rows, cols = target.shape[0] // 2, target.shape[1] // 2
        return Grid(np.full((rows, cols), int(target[..., 0].mean() > 0.4), np.uint8),
                    np.broadcast_to(srgb_to_oklab(target.mean((0, 1))), (rows, cols, 3)).copy(),
                    np.zeros(3, np.float32))

    monkeypatch.setattr(cli, "solve_style", solve)

    def arguments(*extra):
        return cli.make_parser().parse_args([
            str(clip), "--input-format", "video", "--output", str(tmp_path / "render"),
            "--size", "4x4", "--cell", "2x2", "--chars", " .", "--device", "cpu", *extra,
        ])

    return arguments, targets, masks


def test_real_probe_and_sampled_decode_preserve_frame_timestamps(clip):
    info = video.probe_video(clip)
    assert info.frame_times_ms == pytest.approx((0, 250, 500, 750, 1000))
    assert info.duration_ms == pytest.approx(1250)
    targets = list(video.decode_frames(clip, (4, 4), "fill", False, 2))
    assert len(targets) == 3
    for frame, index in zip(targets, (0, 2, 4)):
        assert frame.shape == (4, 4, 3)
        assert frame.dtype == np.float32
        np.testing.assert_allclose(frame[0, 0], np.array([20 + index * 50, 35, 40]) / 255, atol=1 / 255)


def test_video_fit_and_grayscale(clip):
    target = next(video.decode_frames(clip, (4, 4), "fit", True, 5))
    np.testing.assert_array_equal(target[0], 0)
    np.testing.assert_array_equal(target[-1], 0)
    assert target[1].mean() > 0
    np.testing.assert_array_equal(target[..., 0], target[..., 1])
    np.testing.assert_array_equal(target[..., 1], target[..., 2])


def test_variable_timestamps_and_subsampling_keep_full_duration():
    info = video.VideoInfo((0, 40, 150, 190, 310), 390)
    assert video.frame_durations(info, stride=2, transition_frames=1) == [75, 75, 80, 80, 80]
    assert sum(video.frame_durations(info, stride=2, transition_frames=1)) == 390
    assert video.frame_durations(info, stride=100, transition_frames=3) == [390]


def test_probe_retains_variable_frame_metadata(monkeypatch):
    metadata = {"streams": [{"avg_frame_rate": "25/1", "duration": "0.39"}],
                "frames": [{"best_effort_timestamp_time": str(time / 1000)}
                           for time in (5000, 5040, 5150, 5190, 5310)]}
    monkeypatch.setattr(video, "_executable", lambda name: name)
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, json.dumps(metadata), ""))
    info = video.probe_video(Path("vfr.mp4"))
    assert info.frame_times_ms == pytest.approx((0, 40, 150, 190, 310))
    assert info.duration_ms == pytest.approx(390)


def test_probe_video_duration_ignores_longer_audio_track(monkeypatch):
    metadata = {"streams": [{"avg_frame_rate": "4/1"}], "format": {"duration": "10"},
                "frames": [{"best_effort_timestamp_time": str(index / 4), "pkt_duration_time": "0.25"}
                           for index in range(5)]}
    monkeypatch.setattr(video, "_executable", lambda name: name)
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, json.dumps(metadata), ""))
    assert video.probe_video(Path("audio-longer.mp4")).duration_ms == 1250


def test_missing_ffmpeg_has_actionable_error(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="requires ffmpeg"):
        video.probe_video(Path("clip.mp4"))


def test_cli_video_default_has_no_interpolated_frames_and_reuses_complete_render(video_batch):
    arguments, targets, _ = video_batch
    args = arguments()
    cli.build(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["image_count"] == 1
    assert manifest["loops"] == {}
    assert manifest["signature"]["video_transition_frames"] == 0
    assert manifest["video"] == {"path": "video", "frame_count": 5,
                                 "duration_ms": 1250.0, "frame_duration_ms": 250.0}
    assert len(targets) == 5
    assert all(target() is None for target in targets)
    assert (args.output / "stills/0000/preview.png").is_file()
    for kind in cli.MAP_KINDS:
        assert (args.output / "stills/0000" / f"{kind}.png").read_bytes() == (
            args.output / "video/0000" / f"{kind}.png").read_bytes()
    assert set(manifest["background_labs"]) == {"stills/0000", *(f"video/{index:04d}" for index in range(5))}
    cli.build(args)
    assert len(targets) == 5


def test_video_rebuilds_when_existing_manifest_is_not_an_object(video_batch):
    arguments, targets, _ = video_batch
    args = arguments()
    args.output.mkdir()
    (args.output / "manifest.json").write_text("[]")
    cli.build(args)
    assert len(targets) == 5
    assert json.loads((args.output / "manifest.json").read_text())["schema_version"] == 3


@pytest.mark.parametrize("transition_type", ["glyph-path", "earth-mover"])
def test_cli_video_subsamples_and_interpolates_exactly_without_loop_seam(video_batch, transition_type):
    arguments, targets, _ = video_batch
    args = arguments("--frame-stride", "3", "--video-transition-frames", "2",
                     "--transition-type", transition_type)
    cli.build(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert len(targets) == 2
    assert manifest["video"] == {"path": "video", "frame_count": 4, "duration_ms": 1250.0,
                                 "frame_durations_ms": [250.0, 250.0, 250.0, 500.0]}
    assert len(list((args.output / "video").iterdir())) == 4
    for index in range(4):
        persisted = cli._load_grid(args.output / "video" / f"{index:04d}")
        np.testing.assert_allclose(persisted.background,
                                   manifest["background_labs"][f"video/{index:04d}"])


def test_single_interpolated_frame_can_skip_long_glyph_paths(video_batch):
    arguments, _, _ = video_batch
    args = arguments("--video-transition-frames", "1")
    masks = np.array([[[0]], [[0.3]], [[0.6]], [[1]]], np.float32)
    start = Grid(np.array([[0]], np.uint8), np.ones((1, 1, 3), np.float32), np.zeros(3, np.float32))
    end = Grid(np.array([[3]], np.uint8), np.ones((1, 1, 3), np.float32), np.zeros(3, np.float32))
    paths = [[[a] if a == b else list(range(a, b + 1)) if a < b else list(range(a, b - 1, -1))
              for b in range(4)] for a in range(4)]
    frames = list(cli._video_intermediates(start, end, args, masks, paths, np.zeros((1, 1, 3)), 0))
    assert len(frames) == 1


def test_video_endpoint_cache_reuses_transition_variants_and_original_frame_seeds(video_batch, tmp_path):
    arguments, targets, _ = video_batch
    cache = tmp_path / "cache"
    cli.build(arguments("--frame-stride", "2", "--still-cache", str(cache)))
    assert len(targets) == 3
    assert not list(cache.rglob("preview.png"))
    source_frames = {}
    for metadata in cache.glob("*/cache.json"):
        signature = json.loads(metadata.read_text())["signature"]
        source_frames[signature["source_frame"]] = signature["seed"]
        assert "frame_stride" not in signature
        assert "video_transition_frames" not in signature
        assert "transition_type" not in signature
    assert source_frames == {0: 0, 2: 2, 4: 4}
    cli.build(arguments("--frame-stride", "2", "--still-cache", str(cache),
                        "--video-transition-frames", "2", "--transition-type", "earth-mover"))
    assert len(targets) == 3
    cli.build(arguments("--frame-stride", "1", "--still-cache", str(cache)))
    assert len(targets) == 5  # Only source frames 1 and 3 were absent.


def test_video_archive_export_and_import_need_no_render_settings(video_batch, monkeypatch, tmp_path):
    from ascii_optimizer.archive import read_archive_manifest

    arguments, _, _ = video_batch
    archive = tmp_path / "clip.ascii.zip"
    cli.build(arguments("--output-format", "zip", "--output", str(archive), "--frame-stride", "2",
                        "--video-transition-frames", "1", "--transition-type", "earth-mover"))
    exported = read_archive_manifest(archive)
    assert exported["video"]["frame_count"] == 5

    def forbidden(*args, **kwargs):
        raise AssertionError("Importing an ASCII archive must not prepare a renderer")

    monkeypatch.setattr(cli, "resolve_font", forbidden)
    monkeypatch.setattr(cli, "select_device", forbidden)
    imported = tmp_path / "imported"
    args = cli.make_parser().parse_args([str(archive), "--input-format", "archive", "--output", str(imported)])
    cli.build(args)
    manifest = json.loads((imported / "manifest.json").read_text())
    assert manifest["video"] == exported["video"]
    assert manifest["signature"] == {"input_format": "archive", "images": [str(archive)],
                                     "image_hashes": [cli._file_hash(archive)]}
    assert (imported / "video/0004/glyph.png").is_file()


def test_images_and_video_still_require_size():
    args = cli.make_parser().parse_args(["clip.mp4", "--input-format", "video", "--output", "render"])
    with pytest.raises(ValueError, match="--size is required"):
        cli.build(args)
