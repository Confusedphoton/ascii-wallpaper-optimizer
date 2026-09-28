"""Small local render queue for the Plasma QML wallpaper.

The HTTP thread imports no numerical libraries. Each render runs in a separate
process, so loading PyTorch and optimizing pixels never blocks Plasma's UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __version__

PORT = 48741
ADDRESS = "127.0.0.1"
MAX_BODY = 65536


def _screen_size(name: str, fallback: list[int]) -> tuple[int, int]:
    """Get the active physical mode; Plasma's QML sizes are logical pixels."""
    try:
        result = subprocess.run(["kscreen-doctor", "-j"], capture_output=True,
                                text=True, check=True, timeout=3)
        outputs = json.loads(result.stdout)["outputs"]
        enabled = [output for output in outputs if output.get("enabled")]
        for output in enabled:
            if output.get("name") == name or (not name and len(enabled) == 1):
                mode = next(mode for mode in output["modes"]
                            if mode["id"] == output["currentModeId"])
                size = mode["size"]
                return int(size["width"]), int(size["height"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError):
        pass
    return int(fallback[0]), int(fallback[1])


def _validated_request(raw: dict) -> dict:
    images = [str(Path(path).expanduser().resolve()) for path in raw["images"]]
    if not images or len(images) > 50 or any(not Path(path).is_file() for path in images):
        raise ValueError("Select at least one readable image (at most 50).")
    chars = str(raw["chars"])
    if chars.startswith("\\s"):
        chars = " " + chars[2:]
    if not chars or len(chars) > 256 or len(chars) != len(set(chars)):
        raise ValueError("Characters must be distinct and contain 1 to 256 glyphs.")
    detail = int(raw["detail"])
    frames = int(raw["frames"])
    dither = float(raw["dither"])
    if detail not in (0, 1, 2) or not 2 <= frames <= 120 or not 0 <= dither <= 0.05:
        raise ValueError("Detail, frame count, or dithering strength is out of range.")
    hint = raw.get("size", [1920, 1080])
    if not isinstance(hint, list) or len(hint) != 2 or any(not 1 <= int(v) <= 16384 for v in hint):
        raise ValueError("Invalid display size.")
    size = _screen_size(str(raw.get("screen", "")), hint)
    if any(not 1 <= value <= 16384 for value in size):
        raise ValueError("Invalid physical display size.")
    image_hashes = []
    for path in images:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        image_hashes.append(digest.hexdigest())
    return {
        "version": __version__, "images": images, "image_hashes": image_hashes, "font": str(raw["font"]),
        "chars": chars, "detail": detail, "grayscale": bool(raw["grayscale"]),
        "frames": frames, "dither": dither, "size": list(size),
    }


class RenderQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pending: queue.Queue[str] = queue.Queue()
        self.cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ascii-wallpaper-optimizer"
        self.cache.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, name="ascii-render-queue", daemon=True).start()

    def submit(self, raw: dict) -> dict:
        settings = _validated_request(raw)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]
        with self.lock:
            old = self.jobs.get(key)
            if old and old["state"] in ("queued", "running"):
                return self._public(old)
            if old and old["state"] == "ready" and not raw.get("force"):
                return self._public(old)
            job = {"id": key, "state": "queued", "progress": 0.0,
                   "stage": "Waiting for renderer", "error": "", "settings": settings,
                   "directory": str(self.cache / key), "force": bool(raw.get("force"))}
            self.jobs[key] = job
            self.pending.put(key)
            return self._public(job)

    def status(self, key: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(key)
            return self._public(job) if job else None

    @staticmethod
    def _public(job: dict) -> dict:
        return {name: job[name] for name in ("id", "state", "progress", "stage", "error", "directory")}

    def _worker(self) -> None:
        while True:
            key = self.pending.get()
            with self.lock:
                job = self.jobs[key]
                job.update(state="running", stage="Starting optimizer")
                settings = job["settings"]
                directory = job["directory"]
            command = [sys.executable, "-m", "ascii_optimizer.cli",
                       "--output", directory, "--size", f"{settings['size'][0]}x{settings['size'][1]}",
                       "--font", settings["font"], "--chars", settings["chars"],
                       "--detail", str(settings["detail"]), "--frames", str(settings["frames"]),
                       "--dither", str(settings["dither"]), "--progress-json"]
            if settings["grayscale"]:
                command.append("--grayscale")
            if job["force"]:
                command.append("--force")
            command.extend(settings["images"])
            recent_lines: list[str] = []
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        line = line.strip()
                        if line.startswith("ASCII_PROGRESS "):
                            update = json.loads(line[len("ASCII_PROGRESS "):])
                            with self.lock:
                                job["progress"] = float(update["progress"])
                                job["stage"] = str(update["stage"])
                        elif line:
                            recent_lines.append(line)
                            recent_lines = recent_lines[-12:]
                    code = process.wait()
                if code:
                    raise RuntimeError("\n".join(recent_lines[-6:]) or f"Renderer exited with code {code}.")
                with self.lock:
                    job.update(state="ready", progress=1.0, stage="Ready")
            except Exception as error:
                with self.lock:
                    job.update(state="error", stage="Render failed", error=str(error))
                traceback.print_exc()
            finally:
                self.pending.task_done()


class Handler(BaseHTTPRequestHandler):
    queue: RenderQueue

    def log_message(self, format: str, *args: object) -> None:
        if not self.path.startswith("/v1/status/"):
            super().log_message(format, *args)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/health":
            self._json(200, {"service": "ascii-wallpaper-optimizer", "version": 1})
        elif path.startswith("/v1/status/"):
            job = self.queue.status(path.removeprefix("/v1/status/"))
            self._json(200 if job else 404, job or {"error": "Unknown render job"})
        else:
            self._json(404, {"error": "Unknown endpoint"})

    def do_POST(self) -> None:
        if self.path != "/v1/render":
            self._json(404, {"error": "Unknown endpoint"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._json(415, {"error": "JSON content type required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Render request is too large or empty.")
            request = json.loads(self.rfile.read(length))
            self._json(200, self.queue.submit(request))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})


def main() -> None:
    Handler.queue = RenderQueue()
    server = ThreadingHTTPServer((ADDRESS, PORT), Handler)
    print(f"ASCII wallpaper render service listening on {ADDRESS}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
