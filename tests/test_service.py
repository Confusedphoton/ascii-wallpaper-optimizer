from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from ascii_optimizer import service
from ascii_optimizer.settings import normalize_options


def test_screen_size_without_display_environment_uses_hint_without_starting_qt(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("A headless service must not start kscreen-doctor")
    monkeypatch.setattr(service.subprocess, "run", unexpected)
    assert service._screen_size("DP-1", [3414, 2134]) == (3414, 2134)


def test_screen_size_with_display_environment_selects_physical_mode(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    output = {"outputs": [{"name": "DP-1", "enabled": True, "currentModeId": "native",
                            "modes": [{"id": "native", "size": {"width": 3840, "height": 2160}}]}]}
    monkeypatch.setattr(service.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, json.dumps(output)))
    assert service._screen_size("DP-1", [1920, 1080]) == (3840, 2160)


@pytest.fixture
def request_settings(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (170, 90, 50)).save(image)
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font identity fixture")
    monkeypatch.setattr(service, "_screen_size", lambda name, size: tuple(size))
    return {"images": [str(image)], "font": str(font), "size": [8, 8], "chars": " .#",
            "frames": 3, "steps": 2, "transition_steps": 1, "candidates": 3,
            "dither": 0, "gradient_loss_strength": 0, "glyph_luminance_bonus": 0}


def publish(render_queue, reply):
    job = render_queue.jobs[reply["id"]]
    directory = Path(job["directory"])
    settings = job["settings"]
    paths = [directory / "atlas.png", directory / "metadata.png"]
    paths.extend(directory / "stills" / f"{index:04d}" / f"{kind}.png"
                 for index in range(len(settings["images"]))
                 for kind in ("glyph", "color", "background", "preview"))
    if len(settings["images"]) > 1:
        paths.extend(directory / "transitions" / f"{index:04d}-{(index + 1) % len(settings['images']):04d}" /
                     f"{frame:04d}" / f"{kind}.png" for index in range(len(settings["images"]))
                     for frame in range(settings["frames"]) for kind in ("glyph", "color", "background"))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1)).save(path)
    (directory / "manifest.json").write_text(json.dumps({"signature": settings,
        "generation": job["generation"], "image_count": len(settings["images"]),
        "frame_count": settings["frames"]}))
    render_queue._publish(job)
    return job


def test_transition_variants_have_immutable_directories_and_return_correct_assets(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = render_queue.submit(request_settings)
    publish(render_queue, first)
    second = render_queue.submit(dict(request_settings, frames=5))
    publish(render_queue, second)
    returned = render_queue.submit(request_settings)
    assert first["directory"] != second["directory"]
    assert returned["directory"] == first["directory"]
    assert returned["state"] == "ready"
    assert returned["manifest"]["frame_count"] == 3
    restarted = service.RenderQueue(render_queue.cache, start_worker=False)
    assert restarted.submit(request_settings)["generation"] == first["generation"]


def test_obsolete_candidate_limits_do_not_split_service_jobs(tmp_path, request_settings):
    assert "candidates" not in normalize_options(request_settings)
    first_settings = service._validated_request(dict(request_settings, candidates=1))
    assert first_settings == service._validated_request(dict(request_settings, candidates=256))
    assert first_settings["glyph_search"] == "all"
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(dict(request_settings, candidates=1))
    second = queue.submit(dict(request_settings, candidates=256))
    assert first["id"] == second["id"]
    assert "--candidates" not in queue._command(queue.jobs[first["id"]])


def test_forced_generation_is_visible_to_existing_consumers_and_reuses_still_cache(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = render_queue.submit(dict(request_settings, client_id="wallpaper"))
    publish(render_queue, first)
    assert "manifest" not in render_queue.status(first["id"], "wallpaper", first["generation"])
    forced = render_queue.submit(dict(request_settings, client_id="settings", force=True))
    assert forced["id"] == first["id"]
    assert forced["directory"] != first["directory"]
    assert render_queue.jobs[first["id"]]["clients"] == {"wallpaper", "settings"}
    # A repeated forced POST while rendering must not create another generation.
    assert render_queue.submit(dict(request_settings, client_id="settings", force=True))["generation"] == forced["generation"]
    command = render_queue._command(render_queue.jobs[forced["id"]])
    assert command[command.index("--still-cache") + 1] == str(render_queue.cache / "stills")
    publish(render_queue, forced)
    changed = render_queue.status(first["id"], "wallpaper", first["generation"])
    assert changed["generation"] == forced["generation"]
    assert changed["manifest"]["generation"] == forced["generation"]
    assert Path(first["directory"]).is_dir()


def test_superseded_work_is_removed_but_shared_work_keeps_running(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = render_queue.submit(dict(request_settings, client_id="preview"))
    render_queue.submit(dict(request_settings, client_id="wallpaper"))
    render_queue.submit(dict(request_settings, client_id="preview", dither=0.01))
    assert render_queue.status(first["id"])["state"] == "queued"
    render_queue.submit(dict(request_settings, client_id="wallpaper", dither=0.02))
    assert render_queue.status(first["id"])["state"] == "superseded"
    assert render_queue.pending.qsize() == 2
    # A fast slider sweep retains only the last pending request from this client.
    for value in range(25):
        render_queue.submit(dict(request_settings, client_id="preview", dither=value / 1000))
    assert render_queue.pending.qsize() == 2


def test_deleted_cache_assets_are_regenerated(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = render_queue.submit(request_settings)
    publish(render_queue, first)
    (Path(first["directory"]) / "stills/0000/color.png").unlink()
    assert render_queue.submit(request_settings)["state"] == "queued"


def test_read_only_status_restores_saved_assets_after_source_changes(tmp_path, request_settings, monkeypatch):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(request_settings)
    publish(queue, first)
    restarted = service.RenderQueue(queue.cache, start_worker=False)
    # A status query must not validate against current renderer or input files.
    monkeypatch.setattr(service, "render_revision", lambda: "new-source-revision")
    Path(request_settings["images"][0]).unlink()
    Path(request_settings["font"]).unlink()
    result = restarted.status(first["id"], "wallpaper", first["generation"])
    assert result["state"] == "ready"
    assert result["directory"] == first["directory"]
    assert result["settings"] == first["settings"]
    assert restarted.pending.empty()
    assert restarted.jobs[first["id"]]["clients"] == {"wallpaper"}
    assert restarted.status("../missing") is None
    assert restarted.pending.empty()


def test_explicit_render_still_uses_current_source_revision(tmp_path, request_settings, monkeypatch):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(request_settings)
    publish(queue, first)
    monkeypatch.setattr(service, "render_revision", lambda: "new-source-revision")
    assert queue.status(first["id"])["generation"] == first["generation"]
    updated = queue.submit(dict(request_settings, force=True))
    assert updated["id"] != first["id"]
    assert updated["settings"]["render_revision"] == "new-source-revision"


def test_content_and_font_changes_invalidate_service_keys(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = render_queue.submit(request_settings)
    Path(request_settings["font"]).write_bytes(b"changed font")
    changed_font = render_queue.submit(request_settings)
    Image.new("RGB", (8, 8), (20, 120, 180)).save(request_settings["images"][0])
    changed_image = render_queue.submit(request_settings)
    assert len({first["id"], changed_font["id"], changed_image["id"]}) == 3


def test_service_imports_no_numerical_runtime():
    subprocess.run([sys.executable, "-c", "import sys; import ascii_optimizer.service; "
                    "assert not {'torch', 'numpy', 'PIL'} & sys.modules.keys()"], check=True)


@pytest.mark.parametrize("name", ["gradient_loss_strength", "glyph_luminance_bonus", "glyph_entropy_strength", "dither", "structure_dither", "transition_spread"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_shared_options_reject_nonfinite_values(name, value):
    with pytest.raises(ValueError, match=name):
        normalize_options({name: value})


@pytest.mark.parametrize("strength", [-0.001, 0.101])
def test_shared_options_bound_glyph_entropy_strength(strength):
    with pytest.raises(ValueError, match="glyph_entropy_strength"):
        normalize_options({"glyph_entropy_strength": strength})


def test_service_entropy_defaults_and_changes_job_identity_and_command(tmp_path, request_settings):
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(request_settings)
    settings = queue.jobs[first["id"]]["settings"]
    assert settings["glyph_entropy_strength"] == 0.01
    command = queue._command(queue.jobs[first["id"]])
    assert command[command.index("--glyph-entropy-strength") + 1] == "0.01"
    changed = queue.submit(dict(request_settings, glyph_entropy_strength=0.025))
    assert changed["id"] != first["id"]


@pytest.mark.parametrize("payload", ["[]", "null", "3", "invalid JSON"])
def test_corrupt_override_falls_back_to_auto(tmp_path, monkeypatch, payload):
    path = tmp_path / "override.json"
    path.write_text(payload)
    monkeypatch.setattr(service, "OVERRIDE_PATH", path)
    assert service.get_override() == "auto"


def test_http_validation_status_and_manifest_generation(tmp_path, request_settings):
    render_queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    class Handler(service.Handler):
        queue = render_queue
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/v1/render", data=json.dumps(dict(request_settings, gradient_loss_strength=float("nan"))).encode(),
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as failure:
            urlopen(request, timeout=3)
        assert failure.value.code == 400
        assert "finite" in json.load(failure.value)["error"]
        request = Request(base + "/v1/render", data=json.dumps(request_settings).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            job = json.load(response)
        publish(render_queue, job)
        with urlopen(base + "/v1/status/" + job["id"], timeout=3) as response:
            assert json.load(response)["manifest"]["generation"] == job["generation"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_worker_publishes_real_default_structure_assets_and_recovers_disk_cache(tmp_path, request_settings):
    font = subprocess.check_output(["fc-match", "-f", "%{file}", "DejaVu Sans Mono"], text=True).strip()
    raw = dict(request_settings, font=font, style="structure", chars=service.normalize_options({})["chars"])
    raw.pop("gradient_loss_strength")
    raw.pop("glyph_luminance_bonus")
    render_queue = service.RenderQueue(tmp_path / "cache")
    submitted = render_queue.submit(raw)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = render_queue.status(submitted["id"])
        if result["state"] in ("ready", "error"):
            break
        time.sleep(0.02)
    assert result["state"] == "ready", result
    assert result["manifest"]["signature"]["chars"] == normalize_options({"style": "structure"})["chars"]
    assert Image.open(Path(result["directory"]) / "stills/0000/preview.png").size == (8, 8)
    restarted = service.RenderQueue(render_queue.cache, start_worker=False)
    assert restarted.submit(raw)["directory"] == result["directory"]
