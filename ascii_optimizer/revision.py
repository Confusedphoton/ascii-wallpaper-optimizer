"""Identify the renderer implementation used to generate cached assets."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

RENDER_MODULES = (
    "assets.py", "cli.py", "defaults.py", "optimizer.py", "settings.py",
    "stroke_graph.py", "tone.py", "transition.py", "perceptual.py", "flow.py",
    "flow_playback.py", "video.py", "archive.py", "revision.py",
    "renderers/__init__.py", "renderers/headroom.py",
    "renderers/stroke.py", "renderers/flow.py",
)


@lru_cache(maxsize=8)
def _digest(files: tuple[tuple[Path, str, int, int], ...]) -> str:
    digest = hashlib.sha256()
    for path, name, _, _ in files:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_revision() -> str:
    """Invalidate both variant and still caches after renderer code changes."""
    directory = Path(__file__).parent
    files = []
    for name in RENDER_MODULES:
        path = directory / name
        stat = path.stat()
        files.append((path, name, stat.st_mtime_ns, stat.st_ctime_ns))
    return _digest(tuple(files))
