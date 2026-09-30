"""Run portable Qt Quick lifecycle contracts against a local renderer stub."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from PIL import Image
import pytest


def _qml_runner() -> str:
    runner = shutil.which("qmltestrunner")
    for candidate in ("/usr/lib/qt6/bin/qmltestrunner", "/usr/lib/x86_64-linux-gnu/qt6/bin/qmltestrunner", runner):
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Qt 6 qmltestrunner is not installed")


def _assets(directory: Path, *, complete: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(directory / "atlas.png")
    metadata = Image.new("RGB", (3, 1))
    metadata.putdata([(0, 4, 2), (0, 4, 2), (0, 0, 0)])
    metadata.save(directory / "metadata.png")
    frames = ["stills/0000", "stills/0001"]
    frames.extend(f"transitions/0000-0001/{frame:04d}" for frame in (1, 2))
    for relative in frames:
        frame = directory / relative
        frame.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2), 0).save(frame / "glyph.png")
        Image.new("RGB", (1, 1), (0, 0, 0)).save(frame / "background.png")
        if complete:
            Image.new("RGB", (2, 2), (180, 120, 70)).save(frame / "color.png")


def test_qml_playback_and_client_lifecycle(tmp_path: Path) -> None:
    runner = _qml_runner()
    from ascii_optimizer.settings import normalize_options
    fixture_root = tmp_path / "assets"
    _assets(fixture_root / "old")
    _assets(fixture_root / "new")
    _assets(fixture_root / "partial", complete=False)
    counts: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value: dict, status: int = 200) -> None:
            body = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def job(self, name: str, generation: str, directory: str = "old") -> dict:
            return {"id": name, "state": "ready", "stage": "Ready", "progress": 1,
                    "directory": str(fixture_root / directory), "generation": generation,
                    "image_count": 1, "frame_count": 4, "manifest": {"image_count": 1}}

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["client_id"]
            name = Path(request["images"][0]).stem
            counts[name] = counts.get(name, 0) + 1
            if name == "restart" and counts[name] == 2:
                assert not request["force"], "Reconnecting must reuse the completed render"
            if name == "retry" and counts[name] == 1:
                self.reply({"error": "temporarily unavailable"}, 503)
            elif name == "timeout" and counts[name] == 1:
                time.sleep(0.3)
                self.reply(self.job(name, "late-reply"))
            elif name == "queued":
                self.reply(dict(self.job(name, "queued-ready"), state="queued", progress=0))
            else:
                self.reply(self.job(name, "refresh-old" if name == "refresh" else name + "-ready"))

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            assert query.get("client_id")
            assert "generation" in query
            name = parsed.path.rsplit("/", 1)[-1]
            if name == "restart":
                counts["restart-status"] = counts.get("restart-status", 0) + 1
                if counts["restart-status"] == 1:
                    self.reply({"error": "Unknown render job"}, 404)
                    return
            if name == "slow-status":
                time.sleep(0.2)
            self.reply(self.job(name, "refresh-new" if name == "refresh" else name + "-ready",
                                "new" if name == "refresh" else "old"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    support = tmp_path / "TestSupport"
    support.mkdir()
    (support / "qmldir").write_text("module TestSupport\nsingleton TestConfig 1.0 TestConfig.qml\n")
    (support / "TestConfig.qml").write_text(
        "pragma Singleton\nimport QtQuick\nQtObject {\n"
        f"    property var renderDefaults: {json.dumps(normalize_options({}))}\n"
        f"    property string fixtureRoot: {json.dumps(str(fixture_root))}\n"
        f"    property string serviceUrl: \"http://127.0.0.1:{server.server_port}\"\n"
        "}\n"
    )
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software")
    try:
        result = subprocess.run([runner, "-input", str(Path(__file__).parent / "qml"),
                                 "-import", str(tmp_path)], env=environment,
                                text=True, capture_output=True, timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
    output = result.stdout + result.stderr
    if any(f'module "{module}" is not installed' in output for module in ("QtTest", "QtQuick")):
        pytest.skip("Qt Quick Test QML module is not installed")
    assert result.returncode == 0, output
    assert all(message not in output for message in ("Binding loop detected", "TypeError:", "ReferenceError:")), output
    assert counts["retry"] == 2
    assert counts["timeout"] == 2
    assert counts["restart"] == 2
