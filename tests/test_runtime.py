from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ascii_optimizer import runtime


def test_auto_device_uses_gpu_and_reports_rocm(monkeypatch, capsys):
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_name", lambda _: "Test AMD GPU")
    monkeypatch.setattr(runtime.torch.version, "hip", "7.2")
    assert runtime.select_device() == "cuda"
    assert "Test AMD GPU (ROCm" in capsys.readouterr().out


def test_explicit_gpu_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    assert runtime.select_device() == "cpu"
    with pytest.raises(ValueError, match="setup_environment.py"):
        runtime.select_device("cuda")


@pytest.mark.parametrize("vendors,backend", [({"AMD"}, "rocm7.2"), (set(), "cpu"), ({"NVIDIA"}, "auto")])
def test_setup_selects_backend_without_recreating_existing_venv(tmp_path, monkeypatch, vendors, backend):
    path = Path(__file__).resolve().parents[1] / "scripts/setup_environment.py"
    spec = importlib.util.spec_from_file_location("setup_environment", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(setup, "VENV", tmp_path)
    monkeypatch.setattr(setup, "detect_gpu_vendors", lambda: vendors)
    monkeypatch.setattr(sys, "argv", [str(path)])
    commands = []
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **_: commands.append(command))
    setup.main()
    assert len(commands) == 2
    assert commands[0][commands[0].index("--torch-backend") + 1] == backend
    assert commands[1][:5] == [str(python), "-m", "ascii_optimizer.launcher", "-m", "ascii_optimizer.runtime"]
