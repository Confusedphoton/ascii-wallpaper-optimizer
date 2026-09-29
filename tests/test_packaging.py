from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plasma_builder", ROOT / "scripts/build_plasma_package.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def compiler():
    try:
        return builder.find_qsb()
    except RuntimeError:
        pytest.skip("Qt 6 shader compiler is not installed")


def test_package_compiles_current_sources_and_ignores_stale_binaries(tmp_path):
    qsb = compiler()
    wallpaper, applet = builder.build(destination=tmp_path / "dist", qsb=qsb)
    shaders = wallpaper / "contents/ui/shaders"
    for source in shaders.glob("*.frag"):
        compiled = source.with_suffix(source.suffix + ".qsb")
        assert compiled.stat().st_size > 0
        dump = subprocess.check_output([qsb, "--dump", str(compiled)], text=True)
        assert "backgroundAvailable" in dump
        assert "metadataAvailable" in dump
    assert (applet / "contents/ui/main.qml").is_file()
    assert (wallpaper / "contents/ui/AsciiFrame.qml").is_file()


def test_compile_failure_leaves_existing_package_available(tmp_path):
    compiler_path = tmp_path / "broken-qsb"
    compiler_path.write_text("#!/bin/sh\nexit 1\n")
    compiler_path.chmod(0o755)
    destination = tmp_path / "dist"
    existing = destination / "plasma-wallpaper"
    existing.mkdir(parents=True)
    marker = existing / "working-package"
    marker.write_text("previous successful build")
    with pytest.raises(subprocess.CalledProcessError):
        builder.build(destination=destination, qsb=str(compiler_path))
    assert marker.read_text() == "previous successful build"
    assert not list(destination.glob("plasma-stage-*"))
