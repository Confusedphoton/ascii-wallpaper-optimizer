from __future__ import annotations

import sys

import pytest

from ascii_optimizer import launcher


@pytest.fixture
def devices(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ACCEL_PATH", tmp_path)
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    return tmp_path


def test_no_npu_needs_no_wrapper(devices, monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    assert launcher.numerical_command(["-m", "ascii_optimizer.runtime"]) == [
        sys.executable, "-m", "ascii_optimizer.runtime"]


def test_npu_is_hidden_without_hiding_gpu_devices(devices, monkeypatch):
    (devices / "accel0").touch()
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/usr/bin/bwrap")
    command = launcher.numerical_command(["-m", "ascii_optimizer.cli", "image with spaces.png"])
    assert command == ["/usr/bin/bwrap", "--die-with-parent", "--bind", "/", "/",
                       "--dev-bind", "/dev", "/dev", "--tmpfs", "/dev/accel",
                       sys.executable, "-m", "ascii_optimizer.cli", "image with spaces.png"]


def test_missing_wrapper_fails_before_loading_gpu_runtime(devices, monkeypatch):
    (devices / "accel0").touch()
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install bubblewrap"):
        launcher.numerical_command(["-m", "ascii_optimizer.cli"])
