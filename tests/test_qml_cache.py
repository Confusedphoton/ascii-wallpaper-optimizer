"""Verify composed-frame caching through Qt Quick's real shader pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from test_qml import _qml_runner


@pytest.mark.parametrize("invert_ink", [False, True])
@pytest.mark.parametrize("device_pixel_ratio", [1, 2])
def test_qml_composed_frame_cache(tmp_path: Path, invert_ink: bool, device_pixel_ratio: int) -> None:
    runner = _qml_runner()
    if device_pixel_ratio > 1 and os.environ.get("ASCII_WALLPAPER_TEST_NATIVE_QML") != "1":
        pytest.skip("Set ASCII_WALLPAPER_TEST_NATIVE_QML=1 to run native display tests")
    if device_pixel_ratio > 1 and not os.environ.get("DISPLAY"):
        pytest.skip("Scaled Qt rendering requires an X display")
    from ascii_optimizer.assets import save_atlas, save_metadata
    from ascii_optimizer.cli import _save_grid
    from ascii_optimizer.optimizer import Grid
    from ascii_optimizer.tone import srgb_to_oklab
    from scripts.build_plasma_package import build, find_qsb

    try:
        compiler = find_qsb()
    except RuntimeError as error:
        pytest.skip(str(error))
    wallpaper, _ = build(destination=tmp_path / "dist", qsb=compiler)
    ui = wallpaper / "contents/ui"
    (ui / "qmldir").write_text("module Wallpaper\nAsciiShader 1.0 AsciiShader.qml\nRenderClient 1.0 RenderClient.qml\n")
    (tmp_path / "Wallpaper").symlink_to(ui, target_is_directory=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    mask = np.array([[0, 64, 128, 255], [255, 128, 64, 0],
                     [128, 255, 0, 64], [64, 0, 255, 128]], dtype=np.float32) / 255
    masks = np.stack((mask, mask[:, ::-1]))
    foreground = np.array([[[255, 0, 0], [0, 0, 255], [128, 128, 128], [0, 255, 0]],
                           [[255, 110, 0], [180, 20, 220], [255, 255, 255], [0, 0, 0]]],
                          dtype=np.float32) / 255
    glyphs = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    background_labs, pixels = {}, {}
    for index, name in enumerate(("stills/0000", "stills/0001",
                                 "transitions/0000-0001/0001", "transitions/0000-0001/0002")):
        colors = np.roll(foreground, index, axis=1)
        background = np.array([20 + index * 10, 30, 48], dtype=np.float32) / 255
        grid = Grid(glyphs, srgb_to_oklab(colors), srgb_to_oklab(background))
        persisted = _save_grid(grid, assets / name, True, masks, (16, 8), invert_ink=invert_ink)
        background_labs[name] = persisted.background.tolist()
        pixels[name] = np.asarray(Image.open(assets / name / "preview.png")).ravel().tolist()
    save_atlas(masks, assets / "atlas.png")
    save_metadata((16, 8), (4, 4), 2, assets / "metadata.png", invert_ink=invert_ink)
    manifest = {"image_count": 2, "frame_count": 4, "output_size": [16, 8],
                "cell_size": [4, 4], "atlas_count": 2, "invert_ink": invert_ink,
                "background_labs": background_labs}
    support = tmp_path / "TestSupport"
    support.mkdir()
    (support / "qmldir").write_text("module TestSupport\nsingleton TestConfig 1.0 TestConfig.qml\n")
    (support / "TestConfig.qml").write_text(
        "pragma Singleton\nimport QtQuick\nQtObject {\n"
        f"    property string fixtureRoot: {json.dumps(str(assets))}\n"
        f"    property var manifest: {json.dumps(manifest)}\n"
        f"    property var pixels: {json.dumps(pixels)}\n"
        "}\n"
    )
    # Force RHI, since the offscreen platform otherwise selects the software
    # adaptation. Its GLX backing surface does not support scaled windows, so
    # use the native X platform for the high-DPI cases.
    platform = "xcb" if device_pixel_ratio > 1 else "offscreen"
    environment = dict(os.environ, QT_QPA_PLATFORM=platform, QT_QUICK_BACKEND="rhi",
                       QSG_RHI_BACKEND="opengl", EGL_PLATFORM="surfaceless",
                       QT_SCALE_FACTOR=str(device_pixel_ratio))
    result = subprocess.run([runner, "-input", str(Path(__file__).parent / "qml_render"),
                             "-import", str(tmp_path)], env=environment,
                            text=True, capture_output=True, timeout=30)
    output = result.stdout + result.stderr
    if any(message in output for message in ("Failed to create OpenGL context",
                                             'module "QtTest" is not installed',
                                             'module "QtQuick" is not installed',
                                             "The Qt Quick RHI backend is unavailable")):
        pytest.skip("Qt Quick shader rendering is unavailable")
    assert result.returncode == 0, output
    assert all(message not in output for message in ("Binding loop detected", "TypeError:", "ReferenceError:")), output
