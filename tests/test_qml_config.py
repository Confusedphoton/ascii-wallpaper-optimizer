"""Exercise the real settings page with Plasma's initial-property/save contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from test_qml import _assets, _qml_runner


@pytest.fixture
def config_service(tmp_path: Path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            status = 200
            if self.path == "/v1/source-info":
                if payload["images"] == ["/wrong-kind.jpg"] and payload["input_format"] == "video":
                    status, reply = 400, {"error": "Choose a video."}
                elif payload["input_format"] == "video":
                    reply = {"source_frames": 120, "duration_ms": 4000}
                else:
                    reply = {}
            elif self.path == "/v1/render":
                video = payload["input_format"] == "video"
                generation = payload.get("background", "auto")
                if video:
                    generation = "excerpt" if payload.get("video_preview_ms") else "full"
                reply = {"id": "settings", "state": "ready", "stage": "Ready", "progress": 1,
                         "generation": generation, "settings": payload,
                         "directory": str(tmp_path / "assets"), "image_count": 1, "frame_count": 4,
                         "manifest": {"output_size": [4, 4], "cell_size": [2, 2], "atlas_count": 1}}
            else:
                reply = {"maximum": payload.get("maximum", 20), "count": 0}
            body = json.dumps(reply).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("qml_case", ["config", "video_preview"])
def test_wallpaper_settings_persistence_and_preview(tmp_path: Path, config_service, qml_case) -> None:
    from scripts.build_plasma_package import build, find_qsb

    runner = _qml_runner()
    try:
        compiler = find_qsb()
    except RuntimeError as error:
        pytest.skip(str(error))
    wallpaper, _ = build(destination=tmp_path / "dist", qsb=compiler)
    ui = wallpaper / "contents/ui"
    (ui / "qmldir").write_text("module Wallpaper\nConfig 1.0 config.qml\n")
    (tmp_path / "Wallpaper").symlink_to(ui, target_is_directory=True)
    assets = tmp_path / "assets"
    _assets(assets)
    support = tmp_path / "TestSupport"
    support.mkdir()
    (support / "qmldir").write_text("module TestSupport\nsingleton TestConfig 1.0 TestConfig.qml\n")
    (support / "TestConfig.qml").write_text(
        "pragma Singleton\nimport QtQuick\nQtObject {\n"
        f"    property string fixtureRoot: {json.dumps(str(assets))}\n"
        f"    property string serviceUrl: {json.dumps(config_service[0])}\n"
        "}\n"
    )
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="rhi",
                       QSG_RHI_BACKEND="opengl", EGL_PLATFORM="surfaceless",
                       XDG_CONFIG_HOME=str(tmp_path / "config"))
    result = subprocess.run([runner, "-input", str(Path(__file__).parent / "qml_config" / f"tst_{qml_case}.qml"),
                             "-import", str(tmp_path)], env=environment,
                            text=True, errors="replace", capture_output=True, timeout=45)
    output = result.stdout + result.stderr
    if 'module "org.kde.kirigami" is not installed' in output:
        pytest.skip("Kirigami is not installed")
    assert result.returncode == 0, output
    assert all(message not in output for message in
               ("Binding loop detected", "TypeError:", "ReferenceError:", "Cannot assign")), output
    if qml_case == "config":
        renders = [payload for path, payload in config_service[1]
                   if path == "/v1/render" and payload["images"] == ["/apply.png"]]
        assert any(payload["background"] == "#112233" and not payload["force"] for payload in renders)
        assert any(path == "/v1/cache/clear" for path, _ in config_service[1])
        assert any(path == "/v1/cache" and payload["maximum"] == 7 for path, payload in config_service[1])
        return
    video_renders = [payload for path, payload in config_service[1]
                     if path == "/v1/render" and payload["images"] == ["/preview-video.mkv"]]
    assert [(render.get("video_preview_ms", 0), render["force"]) for render in video_renders] == [
        (3000, False), (0, False),
    ]
