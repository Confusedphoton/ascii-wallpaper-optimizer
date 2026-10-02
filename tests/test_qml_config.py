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
            if self.path == "/v1/render":
                reply = {"id": "settings", "state": "ready", "stage": "Ready", "progress": 1,
                         "generation": payload.get("background", "auto"), "settings": payload,
                         "directory": str(tmp_path / "assets"), "image_count": 1, "frame_count": 4,
                         "manifest": {"output_size": [4, 4], "cell_size": [2, 2], "atlas_count": 1}}
            else:
                reply = {"maximum": payload.get("maximum", 20), "count": 0}
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_wallpaper_settings_persistence_and_preview(tmp_path: Path, config_service) -> None:
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
    result = subprocess.run([runner, "-input", str(Path(__file__).parent / "qml_config"),
                             "-import", str(tmp_path)], env=environment,
                            text=True, errors="replace", capture_output=True, timeout=45)
    output = result.stdout + result.stderr
    if 'module "org.kde.kirigami" is not installed' in output:
        pytest.skip("Kirigami is not installed")
    assert result.returncode == 0, output
    assert all(message not in output for message in
               ("Binding loop detected", "TypeError:", "ReferenceError:", "Cannot assign")), output
    renders = [payload for path, payload in config_service[1]
               if path == "/v1/render" and payload["images"] == ["/apply.png"]]
    assert any(payload["background"] == "#112233" and not payload["force"] for payload in renders)
    assert any(path == "/v1/cache/clear" for path, _ in config_service[1])
    assert any(path == "/v1/cache" and payload["maximum"] == 7 for path, payload in config_service[1])
