from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pytest

from ascii_optimizer import archive, service
from ascii_optimizer.settings import normalize_options


def _manifest(kind: str) -> dict:
    manifest = {
        "schema_version": 3 if kind == "video" else 2,
        "signature": {"images": ["original.png"], "transition_type": "earth-mover"},
        "generation": "source-generation",
        "image_count": 2 if kind == "slideshow" else 1,
        "frame_count": 3,
        "grid": [2, 2],
        "output_size": [4, 4],
        "cell_size": [2, 2],
        "atlas_count": 2,
        "invert_ink": False,
        "loops": {},
    }
    if kind == "video":
        manifest["video"] = {"path": "video", "frame_count": 3, "duration_ms": 100,
                             "frame_durations_ms": [25, 25, 50]}
    if kind == "flow":
        manifest["loops"] = {"0000": {"path": "loops/0000", "frame_count": 2,
                                       "duration_ms": 1000, "loop": True}}
    return manifest


def _assets(directory: Path, manifest: dict) -> None:
    for name in archive.manifest_asset_paths(manifest):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(manifest).encode() if name == "manifest.json" else b"playback map")


def _source_archive(tmp_path: Path, kind: str) -> tuple[Path, dict]:
    manifest = _manifest(kind)
    directory = tmp_path / "prepared"
    _assets(directory, manifest)
    destination = tmp_path / "prepared.ascii.zip"
    archive.write_archive(directory, destination)
    return destination, manifest


def _publish(queue: service.RenderQueue, reply: dict, manifest: dict) -> dict:
    job = queue.jobs[reply["id"]]
    manifest = dict(manifest, signature=job["settings"], generation=job["generation"])
    _assets(Path(job["directory"]), manifest)
    queue._publish(job)
    return manifest


@pytest.fixture
def video_request(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video source identity")
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font source identity")
    monkeypatch.setattr(service, "_screen_size", lambda name, fallback: tuple(fallback))
    return {"input_format": "video", "images": [str(source)], "font": str(font),
            "size": [4, 4], "chars": " .", "frames": 3, "steps": 1,
            "transition_steps": 1, "transition_type": "earth-mover"}


def test_video_request_defaults_and_options_reach_renderer_and_split_cache(tmp_path, video_request):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    default = queue.submit(video_request)
    assert default["settings"]["video_transition_frames"] == 0
    assert default["settings"]["frame_stride"] == 1
    interpolated = queue.submit(dict(video_request, video_transition_frames=2))
    sampled = queue.submit(dict(video_request, frame_stride=3))
    assert len({default["id"], interpolated["id"], sampled["id"]}) == 3
    command = queue._command(queue.jobs[interpolated["id"]])
    assert command[command.index("--input-format") + 1] == "video"
    assert command[command.index("--video-transition-frames") + 1] == "2"
    assert command[command.index("--transition-type") + 1] == "earth-mover"
    sampled_command = queue._command(queue.jobs[sampled["id"]])
    assert sampled_command[sampled_command.index("--frame-stride") + 1] == "3"


def test_video_publication_retains_timing_and_missing_frame_queues_new_generation(tmp_path, video_request):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(video_request)
    manifest = _publish(queue, first, _manifest("video"))
    ready = queue.submit(video_request)
    assert ready["state"] == "ready"
    assert ready["manifest"]["video"] == manifest["video"]
    assert ready["directory"] == first["directory"]
    (Path(first["directory"]) / "video/0002/color.png").unlink()
    replacement = queue.submit(video_request)
    assert replacement["state"] == "queued"
    assert replacement["id"] == first["id"]
    assert replacement["generation"] != first["generation"]
    _publish(queue, replacement, _manifest("video"))
    assert queue.status(first["id"])["state"] == "ready"


@pytest.mark.parametrize("corruption", ["missing-frame", "bad-timing", "wrong-schema"])
def test_incomplete_or_invalid_video_cannot_be_published(tmp_path, video_request, corruption):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    reply = queue.submit(video_request)
    job = queue.jobs[reply["id"]]
    manifest = dict(_manifest("video"), signature=job["settings"], generation=job["generation"])
    directory = Path(job["directory"])
    _assets(directory, manifest)
    if corruption == "missing-frame":
        (directory / "video/0001/glyph.png").unlink()
    elif corruption == "bad-timing":
        manifest["video"]["frame_durations_ms"] = [25, 25]
        (directory / "manifest.json").write_text(json.dumps(manifest))
    else:
        manifest["schema_version"] = 2
        (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="complete matching asset generation"):
        queue._publish(job)
    assert not (directory.parent / "current.json").exists()
    assert job["state"] == "queued"


@pytest.mark.parametrize("kind", ["slideshow", "video", "flow"])
def test_archive_import_command_publishes_manifest_playback_layout(tmp_path, monkeypatch, kind):
    # Exercise the CLI arguments produced by the service, including the lack
    # of render size/font options when a prepared archive is imported.
    from ascii_optimizer import cli

    source, source_manifest = _source_archive(tmp_path, kind)
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    reply = queue.submit({"input_format": "archive", "images": [str(source)],
                          "font": "missing font", "size": [0, 0], "steps": 0})
    job = queue.jobs[reply["id"]]
    assert job["settings"] == {"input_format": "archive", "images": [str(source)],
                               "image_hashes": [service._file_hash(source)]}
    command = queue._command(job)
    assert "--size" not in command
    assert "--font" not in command
    assert "--still-cache" not in command
    assert command[command.index("--generation") + 1] == job["generation"]

    def unexpected(*args, **kwargs):
        raise AssertionError("Archive import must not prepare a numerical render")

    monkeypatch.setattr(cli, "resolve_font", unexpected)
    monkeypatch.setattr(cli, "select_device", unexpected)
    args = cli.make_parser().parse_args(command[command.index("ascii_optimizer.cli") + 1:])
    cli.build(args)
    queue._publish(job)
    ready = queue.submit({"input_format": "archive", "images": [str(source)]})
    assert ready["state"] == "ready"
    assert ready["manifest"]["generation"] == job["generation"]
    assert ready["manifest"]["image_count"] == source_manifest["image_count"]
    assert ready["manifest"]["loops"] == source_manifest["loops"]
    if kind == "video":
        assert ready["manifest"]["video"] == source_manifest["video"]
    assert set(path for path in archive.manifest_asset_paths(ready["manifest"])) == set(
        str(path.relative_to(Path(ready["directory"])))
        for path in Path(ready["directory"]).rglob("*") if path.is_file())


@pytest.mark.parametrize("kind,missing", [
    ("slideshow", "stills/0001/preview.png"),
    ("slideshow", "transitions/0001-0000/0002/background.png"),
    ("video", "video/0002/color.png"),
    ("flow", "loops/0000/0001/glyph.png"),
])
def test_archive_cache_recovery_checks_manifest_frames_beyond_single_source_file(tmp_path, kind, missing):
    source, manifest = _source_archive(tmp_path, kind)
    request = {"input_format": "archive", "images": [str(source)]}
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(request)
    _publish(queue, first, manifest)
    assert len(first["settings"]["images"]) == 1
    (Path(first["directory"]) / missing).unlink()
    restarted = service.RenderQueue(queue.cache, start_worker=False)
    assert restarted.status(first["id"]) is None
    replacement = restarted.submit(request)
    assert replacement["state"] == "queued"
    assert replacement["generation"] != first["generation"]


def test_archive_reconnect_survives_missing_source_and_renderer_changes(tmp_path, monkeypatch):
    source, manifest = _source_archive(tmp_path, "video")
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit({"input_format": "archive", "images": [str(source)]})
    _publish(queue, first, manifest)
    source.unlink()

    def unexpected(*args, **kwargs):
        raise AssertionError("Read-only reconnection must use published archive assets")

    monkeypatch.setattr(service, "render_revision", unexpected)
    restarted = service.RenderQueue(queue.cache, start_worker=False)
    ready = restarted.status(first["id"], "wallpaper")
    assert ready["state"] == "ready"
    assert ready["manifest"]["video"] == manifest["video"]
    assert "manifest" not in restarted.status(first["id"], "wallpaper", first["generation"])
    assert restarted.pending.empty()


def test_archive_identity_uses_content_and_ignores_inactive_render_options(tmp_path):
    source, manifest = _source_archive(tmp_path, "video")
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    request = {"input_format": "archive", "images": [str(source)]}
    first = queue.submit(request)
    same = queue.submit(dict(request, steps=200, font="another font", frame_stride=100,
                             video_transition_frames=10, size=[1000, 1000]))
    assert same["id"] == first["id"]
    manifest["generation"] = "updated-source-generation"
    _assets(tmp_path / "prepared", manifest)
    archive.write_archive(tmp_path / "prepared", source)
    changed = queue.submit(request)
    assert changed["id"] != first["id"]


def test_archive_request_validation_rejects_missing_playback_frames_before_queueing(tmp_path):
    source, manifest = _source_archive(tmp_path, "video")
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    with pytest.raises(ValueError, match="missing required playback assets"):
        queue.submit({"input_format": "archive", "images": [str(source)]})
    assert queue.jobs == {}
    assert queue.pending.empty()


@pytest.mark.parametrize("input_format", ["video", "archive"])
def test_media_input_rejects_multiple_source_files(tmp_path, video_request, input_format):
    second = tmp_path / "other"
    second.write_bytes(b"another source")
    raw = dict(video_request, input_format=input_format,
               images=[*video_request["images"], str(second)])
    with pytest.raises(ValueError, match="exactly one"):
        service._validated_request(raw)


def test_archive_request_validation_imports_no_numerical_runtime(tmp_path):
    source, _ = _source_archive(tmp_path, "video")
    program = (
        "import sys; from ascii_optimizer import service; "
        f"settings = service._validated_request({{'input_format': 'archive', 'images': [{str(source)!r}]}}); "
        "assert settings['input_format'] == 'archive'; "
        "assert not {'torch', 'numpy', 'PIL'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.parametrize("name,value", [
    ("frame_stride", 0), ("frame_stride", -1), ("frame_stride", 1.5),
    ("frame_stride", True), ("frame_stride", float("nan")), ("frame_stride", 10001),
    ("video_transition_frames", -1), ("video_transition_frames", 119),
    ("video_transition_frames", 1.5), ("video_transition_frames", True),
])
def test_video_sampling_and_interpolation_options_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        normalize_options({"input_format": "video", name: value})


def test_video_sampling_and_interpolation_option_defaults_and_limits():
    defaults = normalize_options({"input_format": "video"})
    assert defaults["frame_stride"] == 1
    assert defaults["video_transition_frames"] == 0
    maximum = normalize_options({"input_format": "video", "frame_stride": 10000,
                                 "video_transition_frames": 118})
    assert maximum["frame_stride"] == 10000
    assert maximum["video_transition_frames"] == 118


@pytest.mark.parametrize("input_format", ["images", "archive"])
def test_inactive_video_options_have_one_canonical_cache_identity(input_format):
    options = normalize_options({"input_format": input_format, "frame_stride": 8,
                                 "video_transition_frames": 12})
    assert options["frame_stride"] == 1
    assert options["video_transition_frames"] == 0
    assert options == normalize_options({"input_format": input_format})


def test_real_video_worker_reuses_complete_endpoints_after_still_eviction(tmp_path, monkeypatch, capsys):
    if any(not shutil.which(name) for name in ("ffmpeg", "ffprobe", "fc-match")):
        pytest.skip("Video worker requires FFmpeg and fontconfig")
    source = tmp_path / "short.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=8x8:r=8", "-frames:v", "2", "-c:v", "ffv1", str(source)],
                   check=True, capture_output=True)
    monkeypatch.setattr(service, "_screen_size", lambda name, fallback: tuple(fallback))
    queue = service.RenderQueue(tmp_path / "cache", start_worker=True)
    queue.configure_cache(1)
    settings = {"input_format": "video", "images": [str(source)], "font": "DejaVu Sans Mono",
                "size": [4, 4], "chars": " .", "style": "structure", "grayscale": True,
                "steps": 1, "transition_steps": 1, "client_id": "video-review"}

    def render(**overrides):
        reply = queue.submit(dict(settings, **overrides))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ready = queue.status(reply["id"], settings["client_id"])
            if ready["state"] in ("ready", "error"):
                assert ready["state"] == "ready", ready.get("error")
                return ready
            time.sleep(0.05)
        with queue.lock:
            process = queue.jobs[reply["id"]].get("process")
            if process is not None:
                process.terminate()
        pytest.fail("Video render worker did not finish within 30 seconds")

    original = render()
    capsys.readouterr()
    ready = render(video_transition_frames=1, transition_type="earth-mover")
    assert capsys.readouterr().out.count("Reusing cached video frame") == 2
    for before, after in [(0, 0), (1, 2)]:
        for name in ("glyph.png", "color.png", "background.png"):
            old = Path(original["directory"]) / "video" / f"{before:04d}" / name
            new = Path(ready["directory"]) / "video" / f"{after:04d}" / name
            assert old.read_bytes() == new.read_bytes()
    assert ready["manifest"]["schema_version"] == 3
    assert ready["manifest"]["video"]["frame_count"] == 3
    assert ready["manifest"]["video"]["duration_ms"] == pytest.approx(250)
    assert ready["manifest"]["video"]["frame_durations_ms"] == pytest.approx([62.5, 62.5, 125])
    assert ready["manifest"]["signature"]["transition_type"] == "earth-mover"
    assert (Path(ready["directory"]) / "video/0002/glyph.png").is_file()
