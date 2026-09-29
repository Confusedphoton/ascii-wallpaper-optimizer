"""Small local render queue for the Plasma QML wallpaper.

The HTTP thread imports no numerical libraries. Each render runs in a separate
process, so loading PyTorch and optimizing pixels never blocks Plasma's UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from .settings import normalize_options, validate_size

PORT = 48741
ADDRESS = "127.0.0.1"
MAX_BODY = 65536
CONSUMER_TTL = 90.0
MAX_RETAINED_JOBS = 256
MAX_PENDING_JOBS = 64
OVERRIDE_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ascii-wallpaper-optimizer" / "override.json"
override_lock = threading.Lock()
hash_lock = threading.Lock()
file_hashes: dict[tuple, str] = {}


class PendingJobs(queue.Queue):
    """A queue that can remove a consumer's obsolete work before dispatch."""

    def discard(self, token: tuple[str, str]) -> None:
        with self.mutex:
            try:
                self.queue.remove(token)
            except ValueError:
                return  # The worker already took it and will finish its accounting.
            self.unfinished_tasks -= 1
            if self.unfinished_tasks == 0:
                self.all_tasks_done.notify_all()
            self.not_full.notify()


def get_override() -> str:
    with override_lock:
        try:
            value = json.loads(OVERRIDE_PATH.read_text())
            profile = value.get("profile", "auto") if isinstance(value, dict) else "auto"
            return profile if profile in ("auto", "A", "B") else "auto"
        except (OSError, ValueError, TypeError):
            return "auto"


def set_override(profile: str) -> str:
    if profile not in ("auto", "A", "B"):
        raise ValueError("Override must be auto, A, or B.")
    with override_lock:
        OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = OVERRIDE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps({"profile": profile}) + "\n")
        temporary.replace(OVERRIDE_PATH)
    return profile


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
    options = normalize_options(raw)
    selected = raw.get("images")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 50 or any(not isinstance(path, str) for path in selected):
        raise ValueError("Select at least one readable image (at most 50).")
    images = [str(Path(path).expanduser().resolve()) for path in selected]
    if any(not Path(path).is_file() for path in images):
        raise ValueError("Select readable image files.")
    hint = validate_size(raw.get("size", [1920, 1080]), 16384)
    size = validate_size(_screen_size(str(raw.get("screen", "")), list(hint)), 16384)
    font = Path(options["font"]).expanduser()
    if not font.is_file():
        result = subprocess.run(["fc-match", "-f", "%{file}", options["font"]],
                                check=True, capture_output=True, text=True, timeout=3)
        font = Path(result.stdout.strip())
    if not font.is_file():
        raise ValueError("Could not find the selected font")
    names = ("chars", "detail", "grayscale", "frames", "dither", "structure_dither",
             "transition_pace", "transition_field", "transition_spread", "gradient_loss_strength",
             "glyph_luminance_bonus", "style", "invert_ink", "background", "crop", "steps",
             "transition_steps", "candidates", "seed")
    return dict({name: options[name] for name in names}, version=__version__, images=images,
                image_hashes=[_file_hash(Path(path)) for path in images],
                font=str(font.resolve()), font_hash=_file_hash(font), size=list(size))


def _file_hash(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with hash_lock:
        cached = file_hashes.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    with hash_lock:
        if len(file_hashes) >= 512:
            file_hashes.clear()
        file_hashes[key] = value
    return value


class RenderQueue:
    def __init__(self, cache: Path | None = None, *, start_worker: bool = True) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pending: PendingJobs = PendingJobs(maxsize=MAX_PENDING_JOBS)
        self.consumers: dict[str, tuple[str, float]] = {}
        self.cache = cache or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ascii-wallpaper-optimizer"
        self.cache.mkdir(parents=True, exist_ok=True)
        if start_worker:
            threading.Thread(target=self._worker, name="ascii-render-queue", daemon=True).start()

    def _detach(self, client: str) -> None:
        old = self.consumers.pop(client, None)
        job = self.jobs.get(old[0]) if old else None
        if not job:
            return
        job["clients"].discard(client)
        if not job["clients"] and not job.get("unmanaged") and job["state"] in ("queued", "running"):
            self.pending.discard((job["id"], job["generation"]))
            job.update(state="superseded", stage="Replaced by newer settings")
            process = job.get("process")
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    def _expire(self) -> None:
        now = time.monotonic()
        for client, (_, seen) in list(self.consumers.items()):
            if now - seen > CONSUMER_TTL:
                self._detach(client)
        if len(self.jobs) > MAX_RETAINED_JOBS:
            for key, job in list(self.jobs.items()):
                if not job["clients"] and job["state"] not in ("queued", "running"):
                    del self.jobs[key]
                    if len(self.jobs) <= MAX_RETAINED_JOBS:
                        break

    def _manifest(self, directory: Path, settings: dict) -> dict | None:
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            signature = manifest["signature"]
            if (not isinstance(signature, dict) or any(signature.get(name) != value for name, value in settings.items())):
                return None
            count, frames = len(settings["images"]), settings["frames"]
            expected = [directory / "atlas.png", directory / "metadata.png"]
            expected.extend(directory / "stills" / f"{index:04d}" / f"{kind}.png"
                            for index in range(count) for kind in ("glyph", "color", "background", "preview"))
            if count > 1:
                expected.extend(directory / "transitions" / f"{index:04d}-{(index + 1) % count:04d}" /
                                f"{frame:04d}" / f"{kind}.png" for index in range(count)
                                for frame in range(frames) for kind in ("glyph", "color", "background"))
            return manifest if all(path.is_file() for path in expected) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _cached_job(self, key: str, settings: dict) -> dict | None:
        try:
            saved = json.loads((self.cache / "variants" / key / "current.json").read_text())
            directory = Path(saved["directory"])
            manifest = self._manifest(directory, settings)
            if manifest is None:
                return None
            return dict(id=key, state="ready", progress=1.0, stage="Ready", error="",
                        settings=settings, directory=str(directory), generation=str(manifest["generation"]),
                        manifest=manifest, force=False, clients=set())
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def submit(self, raw: dict) -> dict:
        settings = _validated_request(raw)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]
        client = raw.get("client_id", "")
        if not isinstance(client, str) or (client and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", client)):
            raise ValueError("Invalid client ID")
        if "force" in raw and not isinstance(raw["force"], bool):
            raise ValueError("force must be a boolean")
        with self.lock:
            self._expire()
            if client and self.consumers.get(client, (None,))[0] != key:
                self._detach(client)
            old = self.jobs.get(key)
            if not old and not raw.get("force"):
                old = self._cached_job(key, settings)
            if old and (old["state"] in ("queued", "running") or (old["state"] == "ready" and not raw.get("force"))):
                if old["state"] != "ready" or self._manifest(Path(old["directory"]), settings):
                    self.jobs[key] = old
                    self._attach(old, client)
                    return self._public(old)
            generation = uuid.uuid4().hex
            if self.pending.full():
                raise queue.Full("Render queue is full; retry after an existing render finishes")
            job = {"id": key, "state": "queued", "progress": 0.0,
                   "stage": "Waiting for renderer", "error": "", "settings": settings,
                   "directory": str(self.cache / "variants" / key / generation),
                   "generation": generation, "force": bool(raw.get("force")),
                   "clients": set(old["clients"]) if old else set()}
            self.jobs[key] = job
            self._attach(job, client)
            self.pending.put((key, generation))
            return self._public(job)

    def _attach(self, job: dict, client: str) -> None:
        if client:
            job["clients"].add(client)
            self.consumers[client] = (job["id"], time.monotonic())
        else:
            job["unmanaged"] = True

    def status(self, key: str, client: str = "", generation: str = "") -> dict | None:
        with self.lock:
            if client and self.consumers.get(client, (None,))[0] == key:
                self.consumers[client] = (key, time.monotonic())
            self._expire()
            job = self.jobs.get(key)
            return self._public(job, generation) if job else None

    @staticmethod
    def _public(job: dict, generation: str = "") -> dict:
        result = {name: job[name] for name in ("id", "state", "progress", "stage", "error", "directory", "generation", "settings")}
        result.update(image_count=len(job["settings"]["images"]), frame_count=job["settings"]["frames"])
        if job["state"] == "ready" and generation != job["generation"]:
            result["manifest"] = job["manifest"]
        return result

    def _publish(self, job: dict) -> None:
        directory = Path(job["directory"])
        manifest = self._manifest(directory, job["settings"])
        if manifest is None or manifest.get("generation") != job["generation"]:
            raise RuntimeError("Renderer did not publish a complete matching asset generation")
        reference = directory.parent / "current.json"
        temporary = reference.with_name(f"current-{job['generation']}.tmp")
        temporary.write_text(json.dumps({"directory": str(directory), "generation": job["generation"]}) + "\n")
        temporary.replace(reference)
        job.update(state="ready", progress=1.0, stage="Ready", manifest=manifest)

    def _worker(self) -> None:
        while True:
            key, generation = self.pending.get()
            try:
                with self.lock:
                    self._expire()
                    job = self.jobs.get(key)
                    if not job or job["generation"] != generation or job["state"] == "superseded":
                        continue
                    job.update(state="running", stage="Starting optimizer")
                command = self._command(job)
                recent_lines: list[str] = []
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    with self.lock:
                        job["process"] = process
                        if job["state"] == "superseded":
                            process.terminate()
                    assert process.stdout is not None
                    for line in process.stdout:
                        line = line.strip()
                        if line.startswith("ASCII_PROGRESS "):
                            update = json.loads(line[len("ASCII_PROGRESS "):])
                            with self.lock:
                                if job["state"] != "superseded":
                                    job["progress"] = float(update["progress"])
                                    job["stage"] = str(update["stage"])
                        elif line:
                            recent_lines.append(line)
                            recent_lines = recent_lines[-12:]
                    code = process.wait()
                if code:
                    raise RuntimeError("\n".join(recent_lines[-6:]) or f"Renderer exited with code {code}.")
                with self.lock:
                    if job["state"] != "superseded":
                        self._publish(job)
            except Exception as error:
                with self.lock:
                    if job["state"] != "superseded":
                        job.update(state="error", stage="Render failed", error=str(error))
                        traceback.print_exc()
            finally:
                if "job" in locals() and job:
                    job.pop("process", None)
                self.pending.task_done()

    def _command(self, job: dict) -> list[str]:
        settings = job["settings"]
        command = [sys.executable, "-m", "ascii_optimizer.cli", "--output", job["directory"],
                   "--generation", job["generation"], "--still-cache", str(self.cache / "stills"),
                   "--size", f"{settings['size'][0]}x{settings['size'][1]}", "--progress-json"]
        for name in ("font", "chars", "detail", "frames", "transition_pace", "transition_field",
                     "transition_spread", "dither", "gradient_loss_strength", "glyph_luminance_bonus",
                     "style", "structure_dither", "background", "crop", "steps", "transition_steps", "candidates", "seed"):
            command.extend(("--" + name.replace("_", "-"), str(settings[name])))
        for name in ("invert_ink", "grayscale"):
            if settings[name]:
                command.append("--" + name.replace("_", "-"))
        if job["force"]:
            command.append("--force")
        command.extend(settings["images"])
        return command


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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/v1/health":
            self._json(200, {"service": "ascii-wallpaper-optimizer", "version": 1})
        elif path == "/v1/override":
            self._json(200, {"profile": get_override()})
        elif path.startswith("/v1/status/"):
            query = parse_qs(parsed.query)
            job = self.queue.status(path.removeprefix("/v1/status/"),
                                    query.get("client_id", [""])[0], query.get("generation", [""])[0])
            self._json(200 if job else 404, job or {"error": "Unknown render job"})
        else:
            self._json(404, {"error": "Unknown endpoint"})

    def do_POST(self) -> None:
        if self.path not in ("/v1/render", "/v1/override"):
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
            if self.path == "/v1/override":
                self._json(200, {"profile": set_override(request["profile"])})
            else:
                self._json(200, self.queue.submit(request))
        except queue.Full as error:
            self._json(503, {"error": str(error)})
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            self._json(400, {"error": str(error)})


def main() -> None:
    Handler.queue = RenderQueue()
    server = ThreadingHTTPServer((ADDRESS, PORT), Handler)
    print(f"ASCII wallpaper render service listening on {ADDRESS}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
