"""Compressed, self-contained playback archives using only the standard library.

An archive has the same relative paths as a rendered asset directory.  Only
files used by its manifest are exported, so cached frames from older renders
cannot accidentally become part of an archive.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Iterator
import zipfile


_MAP_KINDS = ("glyph", "color", "background")
_STILL_KINDS = (*_MAP_KINDS, "preview")
_ARCHIVE_COMMENT = b"ASCII Wallpaper rendered assets"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid archive manifest {name}: expected a positive integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid archive manifest {name}: expected a positive number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"Invalid archive manifest {name}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid archive manifest {name}: expected a finite positive number")
    return number


def _safe_relative(name: str, *, directory: bool = False) -> str:
    """Reject paths whose interpretation can change during extraction."""
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid empty archive path")
    original = name
    if directory and name.endswith("/"):
        name = name[:-1]
    if (not name or name in (".", "..") or "\\" in name or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError(f"Unsafe archive path: {original!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or any(part in (".", "..") for part in path.parts)
            or path.as_posix() != name):
        raise ValueError(f"Unsafe archive path: {original!r}")
    return name


def _manifest_layout(manifest: dict) -> tuple[int, int, dict, dict | None]:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid archive manifest: expected an object")
    schema = manifest.get("schema_version", 1)
    if isinstance(schema, bool) or not isinstance(schema, int) or schema not in (1, 2, 3):
        raise ValueError("Unsupported archive manifest schema_version")
    if not isinstance(manifest.get("signature"), dict):
        raise ValueError("Invalid archive manifest signature")
    count = _positive_int(manifest.get("image_count"), "image_count")
    frames = _positive_int(manifest.get("frame_count"), "frame_count")
    if count > 1 and frames < 2:
        raise ValueError("Invalid archive manifest frame_count: transitions need two endpoints")

    loops = manifest.get("loops", {})
    if not isinstance(loops, dict):
        raise ValueError("Invalid archive manifest loops")
    for key, loop in loops.items():
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                or key != f"{int(key):04d}" or int(key) >= count):
            raise ValueError("Invalid archive manifest loop index")
        if not isinstance(loop, dict) or loop.get("path") != f"loops/{key}":
            raise ValueError("Invalid archive manifest loop path")
        _positive_int(loop.get("frame_count"), "loop frame_count")
        _positive_number(loop.get("duration_ms"), "loop duration_ms")
        if loop.get("loop") is not True:
            raise ValueError("Invalid archive manifest loop flag")
    if loops and schema < 2:
        raise ValueError("Loop archives require manifest schema_version 2 or later")

    video = manifest.get("video")
    if video is not None:
        if (schema != 3 or count != 1 or loops
                or not isinstance(video, dict) or video.get("path") != "video"):
            raise ValueError("Invalid archive manifest video layout")
        video_count = _positive_int(video.get("frame_count"), "video frame_count")
        duration = _positive_number(video.get("duration_ms"), "video duration_ms")
        if "frame_durations_ms" in video:
            durations = video["frame_durations_ms"]
            if not isinstance(durations, list) or len(durations) != video_count:
                raise ValueError("Invalid archive manifest video frame_durations_ms")
            try:
                total = math.fsum(_positive_number(value, "video frame duration") for value in durations)
            except OverflowError as error:
                raise ValueError("Invalid archive manifest video duration_ms") from error
            if "frame_duration_ms" in video:
                raise ValueError("Specify only one video frame duration representation")
        else:
            try:
                total = video_count * _positive_number(video.get("frame_duration_ms"),
                                                     "video frame_duration_ms")
            except OverflowError as error:
                raise ValueError("Invalid archive manifest video duration_ms") from error
        if not math.isclose(duration, total, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Invalid archive manifest video duration_ms: frame durations do not sum to duration")
    elif schema == 3:
        raise ValueError("Archive manifest schema_version 3 requires video metadata")
    return count, frames, loops, video


def _asset_paths(layout: tuple[int, int, dict, dict | None]) -> Iterator[str]:
    count, frames, loops, video = layout
    yield "manifest.json"
    yield "atlas.png"
    yield "metadata.png"
    for index in range(count):
        for kind in _STILL_KINDS:
            yield f"stills/{index:04d}/{kind}.png"
    if count > 1:
        for index in range(count):
            for frame in range(frames):
                for kind in _MAP_KINDS:
                    yield f"transitions/{index:04d}-{(index + 1) % count:04d}/{frame:04d}/{kind}.png"
    for key, loop in loops.items():
        for frame in range(loop["frame_count"]):
            for kind in _MAP_KINDS:
                yield f"loops/{key}/{frame:04d}/{kind}.png"
    if video is not None:
        for frame in range(video["frame_count"]):
            for kind in _MAP_KINDS:
                yield f"video/{frame:04d}/{kind}.png"


def manifest_asset_paths(manifest: dict) -> list[str]:
    """Validate a playback manifest and list its required relative file paths."""
    return list(_asset_paths(_manifest_layout(manifest)))


def _member_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    for info in archive.infolist():
        # ZipInfo.filename truncates NUL bytes; orig_filename retains them.
        name = _safe_relative(info.orig_filename, directory=info.is_dir())
        if name in members:
            raise ValueError(f"Duplicate archive path: {name}")
        file_type = stat.S_IFMT(info.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Archive member is not a regular file or directory: {name}")
        if ((file_type == stat.S_IFDIR and not info.is_dir())
                or (file_type == stat.S_IFREG and info.is_dir())):
            raise ValueError(f"Archive member has inconsistent file type: {name}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted archive members are unsupported: {name}")
        members[name] = info
    for name in members:
        for parent in PurePosixPath(name).parents:
            info = members.get(parent.as_posix())
            if info is not None and not info.is_dir():
                raise ValueError(f"Archive file conflicts with a directory: {parent}")
    return members


def _archive_contents(archive: zipfile.ZipFile) -> tuple[dict, dict[str, zipfile.ZipInfo], list[str]]:
    members = _member_index(archive)
    manifest_info = members.get("manifest.json")
    if manifest_info is None or manifest_info.is_dir():
        raise ValueError("Archive is missing manifest.json")
    try:
        manifest = json.loads(archive.read(manifest_info))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError("Invalid archive manifest.json") from error
    layout = _manifest_layout(manifest)
    count, frames, loops, video = layout
    required_count = (3 + count * len(_STILL_KINDS)
                      + (count * frames * len(_MAP_KINDS) if count > 1 else 0)
                      + sum(loop["frame_count"] * len(_MAP_KINDS) for loop in loops.values())
                      + (video["frame_count"] * len(_MAP_KINDS) if video else 0))
    # Check before materializing a potentially enormous, untrusted frame count.
    if required_count > len(members):
        raise ValueError("Archive is missing required playback assets")
    paths = list(_asset_paths(layout))
    for name in paths:
        info = members.get(name)
        if info is None or info.is_dir() or info.file_size == 0:
            raise ValueError(f"Archive is missing required playback asset: {name}")
    return manifest, members, paths


def read_archive_manifest(source: Path) -> dict:
    """Read and validate an archive without extracting its playback assets."""
    try:
        with zipfile.ZipFile(source) as archive:
            manifest, _, _ = _archive_contents(archive)
            return manifest
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ValueError(f"Invalid rendered asset archive: {source}") from error


def _source_file(directory: Path, name: str) -> Path:
    path = directory
    if directory.is_symlink():
        raise ValueError(f"Archive source directory is a symbolic link: {directory}")
    for part in PurePosixPath(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Archive source contains a symbolic link: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing rendered playback asset: {path}")
    return path


def write_archive(directory: Path, destination: Path) -> None:
    """Write a ZIP_DEFLATED archive, atomically replacing destination on success."""
    directory = Path(directory).expanduser().absolute()
    destination = Path(destination).expanduser().absolute()
    manifest_path = _source_file(directory, "manifest.json")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError("Invalid rendered manifest.json") from error
    layout = _manifest_layout(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                             dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            archive.comment = _ARCHIVE_COMMENT
            for name in _asset_paths(layout):
                archive.write(_source_file(directory, name), name)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _empty_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("Archive extraction destination must not be a symbolic link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Archive extraction destination must be absent or an empty directory")


def extract_archive(source: Path, destination: Path) -> dict:
    """Validate, then atomically extract into an absent or empty destination.

    Files unrelated to the manifest are ignored after validating their paths.
    A failed validation or extraction leaves the destination untouched.
    """
    destination = Path(destination).expanduser().absolute()
    _empty_destination(destination)
    temporary_path = None
    try:
        with zipfile.ZipFile(source) as archive:
            manifest, members, paths = _archive_contents(archive)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.",
                                                    suffix=".tmp", dir=destination.parent))
            for name in paths:
                path = temporary_path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(members[name]) as incoming, path.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            _empty_destination(destination)
            os.replace(temporary_path, destination)
            return manifest
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ValueError(f"Invalid rendered asset archive: {source}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            shutil.rmtree(temporary_path)
