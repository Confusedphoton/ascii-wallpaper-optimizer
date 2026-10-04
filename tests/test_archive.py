from __future__ import annotations

import json
from pathlib import Path
import stat
import zipfile

import pytest

from ascii_optimizer import archive


def _manifest(*, count=2, loops=False, video=False):
    result = {
        "schema_version": 3 if video else 2,
        "signature": {"images": [f"source-{index}.png" for index in range(count)],
                      "transition_type": "earth-mover"},
        "generation": "original-generation",
        "image_count": count,
        "frame_count": 3,
        "output_size": [8, 8],
        "cell_size": [2, 4],
        "grid": [4, 2],
        "atlas_count": 2,
        "loops": {},
    }
    if loops:
        result["loops"] = {f"{index:04d}": {"path": f"loops/{index:04d}", "frame_count": 2,
                                         "duration_ms": 1000, "loop": True}
                           for index in range(count)}
    if video:
        result["video"] = {"path": "video", "frame_count": 3, "duration_ms": 100,
                           "frame_durations_ms": [25, 25, 50]}
    return result


def _rendered(directory: Path, manifest: dict) -> list[str]:
    paths = archive.manifest_asset_paths(manifest)
    for name in paths:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(manifest).encode() if name == "manifest.json"
                         else b"playback map " + name.encode() + b"\0" * 128)
    return paths


def _zip(path: Path, manifest: dict, *, extras=(), omit=()):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in archive.manifest_asset_paths(manifest):
            if name not in omit:
                package.writestr(name, json.dumps(manifest) if name == "manifest.json" else b"map")
        for member in extras:
            package.writestr(member, b"extra")


@pytest.mark.parametrize("kind", ["slideshow", "flow", "video"])
def test_compressed_archive_roundtrip_preserves_playback_and_excludes_stale_assets(tmp_path, kind):
    manifest = _manifest(count=1 if kind == "video" else 2,
                         loops=kind == "flow", video=kind == "video")
    rendered = tmp_path / "rendered"
    paths = _rendered(rendered, manifest)
    stale = rendered / "video/9999/preview.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale previous render")
    (rendered / "diagnostics.json").write_text("{}")
    compressed = tmp_path / "background.ascii.zip"
    archive.write_archive(rendered, compressed)

    with zipfile.ZipFile(compressed) as package:
        assert set(package.namelist()) == set(paths)
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in package.infolist())
    assert archive.read_archive_manifest(compressed) == manifest
    destination = tmp_path / "playback"
    assert not destination.exists()
    assert archive.extract_archive(compressed, destination) == manifest
    assert set(str(path.relative_to(destination)) for path in destination.rglob("*")
               if path.is_file()) == set(paths)
    for name in paths:
        assert (destination / name).read_bytes() == (rendered / name).read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_archive_write_keeps_previous_archive_and_removes_temporary_file(tmp_path):
    manifest = _manifest(count=1)
    directory = tmp_path / "render"
    _rendered(directory, manifest)
    destination = tmp_path / "prepared.zip"
    archive.write_archive(directory, destination)
    previous = destination.read_bytes()
    (directory / "stills/0000/color.png").unlink()
    with pytest.raises(ValueError, match="Missing rendered playback asset"):
        archive.write_archive(directory, destination)
    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("member", ["../outside", "/outside", "video/../outside", "video//outside",
                                  "./outside", ".", "./", "video\\outside", "C:/outside", "video/file:stream"])
def test_unsafe_paths_are_rejected_before_extraction(tmp_path, member):
    package = tmp_path / "unsafe.zip"
    _zip(package, _manifest(count=1), extras=[member])
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="archive path"):
        archive.read_archive_manifest(package)
    with pytest.raises(ValueError, match="archive path"):
        archive.extract_archive(package, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside").exists()


def test_duplicate_paths_are_rejected(tmp_path):
    package = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _zip(package, _manifest(count=1), extras=["atlas.png"])
    with pytest.raises(ValueError, match="Duplicate archive path"):
        archive.extract_archive(package, tmp_path / "extract")


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_non_regular_archive_members_are_rejected(tmp_path, file_type):
    member = zipfile.ZipInfo("extra")
    member.create_system = 3
    member.external_attr = (file_type | 0o777) << 16
    package = tmp_path / "special.zip"
    _zip(package, _manifest(count=1), extras=[member])
    with pytest.raises(ValueError, match="not a regular file"):
        archive.extract_archive(package, tmp_path / "extract")


def test_file_and_directory_conflicts_are_rejected(tmp_path):
    package = tmp_path / "conflict.zip"
    _zip(package, _manifest(count=1), extras=["stills"])
    with pytest.raises(ValueError, match="conflicts with a directory"):
        archive.read_archive_manifest(package)


def test_missing_referenced_frame_is_rejected_without_publishing(tmp_path):
    package = tmp_path / "incomplete.zip"
    _zip(package, _manifest(count=1, video=True), omit=["video/0002/color.png"])
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(ValueError, match="missing required playback asset"):
        archive.extract_archive(package, destination)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_empty_asset_is_rejected(tmp_path):
    package = tmp_path / "empty.zip"
    manifest = _manifest(count=1)
    with zipfile.ZipFile(package, "w") as output:
        for name in archive.manifest_asset_paths(manifest):
            output.writestr(name, json.dumps(manifest) if name == "manifest.json" else b"")
    with pytest.raises(ValueError, match="missing required playback asset"):
        archive.read_archive_manifest(package)


@pytest.mark.parametrize("update", [
    {"path": "../video"}, {"frame_count": True}, {"frame_count": 0},
    {"duration_ms": float("nan")}, {"duration_ms": 101},
    {"frame_durations_ms": [25, 25]}, {"frame_durations_ms": [0, 50, 50]},
    {"frame_durations_ms": [1e308, 1e308, 1e308]},
    {"frame_duration_ms": 25},
])
def test_invalid_video_timing_and_layout_are_rejected(update):
    manifest = _manifest(count=1, video=True)
    manifest["video"].update(update)
    with pytest.raises(ValueError, match="archive manifest|video frame duration"):
        archive.manifest_asset_paths(manifest)


def test_uniform_video_timing_and_large_frame_names_are_supported():
    manifest = _manifest(count=1, video=True)
    manifest["video"] = {"path": "video", "frame_count": 10001,
                         "frame_duration_ms": 10, "duration_ms": 100010}
    paths = archive.manifest_asset_paths(manifest)
    assert "video/10000/glyph.png" in paths
    assert len(paths) == 7 + 3 * 10001


def test_untrusted_huge_frame_count_is_rejected_before_materializing_paths(tmp_path):
    package = tmp_path / "huge.zip"
    manifest = _manifest(count=1, video=True)
    manifest["video"] = {"path": "video", "frame_count": 10**12,
                         "frame_duration_ms": 1, "duration_ms": 10**12}
    with zipfile.ZipFile(package, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="missing required playback assets"):
        archive.read_archive_manifest(package)


def test_extraction_failure_leaves_destination_empty_and_cleans_staging(tmp_path, monkeypatch):
    package = tmp_path / "valid.zip"
    _zip(package, _manifest(count=1))
    destination = tmp_path / "extract"
    destination.mkdir()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(archive.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="disk full"):
        archive.extract_archive(package, destination)
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".*.tmp"))


def test_nonempty_extraction_destination_is_preserved(tmp_path):
    package = tmp_path / "valid.zip"
    _zip(package, _manifest(count=1))
    destination = tmp_path / "extract"
    destination.mkdir()
    existing = destination / "existing"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError, match="absent or an empty directory"):
        archive.extract_archive(package, destination)
    assert existing.read_bytes() == b"keep"


def test_symlink_source_assets_and_extraction_destinations_are_rejected(tmp_path):
    directory = tmp_path / "render"
    _rendered(directory, _manifest(count=1))
    atlas = directory / "atlas.png"
    external = tmp_path / "external.png"
    atlas.rename(external)
    atlas.symlink_to(external)
    with pytest.raises(ValueError, match="symbolic link"):
        archive.write_archive(directory, tmp_path / "output.zip")
    package = tmp_path / "valid.zip"
    _zip(package, _manifest(count=1))
    actual = tmp_path / "actual"
    actual.mkdir()
    destination = tmp_path / "extract"
    destination.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        archive.extract_archive(package, destination)


def test_harmless_unreferenced_archive_members_are_not_extracted(tmp_path):
    package = tmp_path / "extra.zip"
    _zip(package, _manifest(count=1), extras=["stills/9999/color.png", "notes.txt"])
    destination = tmp_path / "extract"
    archive.extract_archive(package, destination)
    assert not (destination / "stills/9999").exists()
    assert not (destination / "notes.txt").exists()
