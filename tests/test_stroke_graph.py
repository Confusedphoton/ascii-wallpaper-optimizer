"""Tests for the promoted connected-stroke MRF (`stroke`) renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ascii_optimizer.assets import glyph_sdf_atlas, resolve_font, sdf_spread_for_cell
from ascii_optimizer.cli import build, make_parser
from ascii_optimizer.defaults import DEFAULT_CHARS, STROKE_CHARS
from ascii_optimizer.renderers import available_styles, solve_style
from ascii_optimizer.settings import normalize_options
from ascii_optimizer.tone import display_tone_lab


def test_stroke_style_is_registered_and_normalizes_cleanly():
    assert "stroke" in available_styles()
    opts = normalize_options({"style": "stroke"})
    assert opts["chars"] == STROKE_CHARS
    assert opts["perceptual_loss"] is False
    assert opts["gradient_loss_strength"] == 0.0
    assert opts["glyph_luminance_bonus"] == 0.0
    assert opts["glyph_entropy_strength"] == 0.0
    # Passing headroom-only weights alongside style="stroke" zeroes them cleanly instead of failing.
    opts2 = normalize_options({"style": "stroke", "gradient_loss_strength": 1.5, "glyph_luminance_bonus": 0.12})
    assert opts2["gradient_loss_strength"] == 0.0
    assert opts2["glyph_luminance_bonus"] == 0.0


def test_stroke_renderer_produces_valid_sdf_grid_and_supports_invert_ink():
    rng = np.random.default_rng(7)
    # Build a synthetic image with a diagonal bright stripe on a dark canvas.
    H, W = 64, 64
    yy, xx = np.mgrid[:H, :W]
    stripe = (np.abs(yy - xx) < 6).astype(np.float32)
    source = np.stack((0.12 + 0.72 * stripe, 0.10 + 0.55 * stripe, 0.16 + 0.45 * stripe), axis=-1)
    source = np.clip(source + rng.normal(0, 0.01, source.shape).astype(np.float32), 0.0, 1.0)

    font = resolve_font("DejaVu Sans Mono")
    atlas, spread = glyph_sdf_atlas(font, list(STROKE_CHARS), (8, 16))
    progress_calls = []

    normal = solve_style(
        source, atlas, style="stroke", sdf_spread=spread, seed=0, device="cpu",
        on_progress=lambda f, msg: progress_calls.append((f, msg)),
    )
    assert normal.glyph.shape == (4, 8)
    assert normal.weight is not None and normal.weight.shape == (4, 8)
    assert np.isfinite(normal.color).all() and np.isfinite(normal.weight).all()
    assert progress_calls and progress_calls[-1][0] == 1.0

    # Determinism check
    repeat = solve_style(source, atlas, style="stroke", sdf_spread=spread, seed=0, device="cpu")
    np.testing.assert_array_equal(normal.glyph, repeat.glyph)
    np.testing.assert_array_equal(normal.color, repeat.color)
    np.testing.assert_array_equal(normal.weight, repeat.weight)

    # Inverted ink: dark ink on bright canvas should produce darker displayed ink than normal mode.
    inverted = solve_style(source, atlas, style="stroke", sdf_spread=spread, invert_ink=True, seed=0, device="cpu")
    normal_L = display_tone_lab(normal.color)[..., 0].mean()
    inverted_L = display_tone_lab(inverted.color, invert_ink=True)[..., 0].mean()
    assert inverted_L < normal_L
    assert not np.array_equal(normal.glyph, inverted.glyph)


def test_stroke_cli_build_writes_sdf_assets_and_signature(tmp_path: Path):
    image = tmp_path / "scene.png"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :16] = (40, 20, 35)
    arr[:, 16:] = (220, 170, 110)
    Image.fromarray(arr).save(image)

    out = tmp_path / "out"
    args = make_parser().parse_args([
        str(image), "--output", str(out), "--size", "32x32", "--cell", "8x16",
        "--style", "stroke", "--device", "cpu", "--chars", DEFAULT_CHARS,
    ])
    build(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["signature"]["style"] == "stroke"
    assert manifest["signature"]["chars"] == STROKE_CHARS
    assert manifest["signature"]["output_stage"] == "stroke_graph"
    assert manifest["signature"]["glyph_search"] == "stroke_graph"
    meta = np.asarray(Image.open(out / "metadata.png"))
    assert meta[0, 2, 2] == round(sdf_spread_for_cell((8, 16)) * 16)
    assert (out / "stills/0000/preview.png").exists()
