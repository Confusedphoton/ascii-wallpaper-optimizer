"""Exercise the real settings page with Plasma's initial-property/save contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from test_qml import _assets, _qml_runner


def test_wallpaper_settings_persistence_and_preview(tmp_path: Path) -> None:
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
