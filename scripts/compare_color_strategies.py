"""Render a controlled visual comparison of experimental ASCII tone strategies.

The experimental variants are previews only. The iso-chroma variants reuse the
production solver with perceptual tone transforms around it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from ascii_optimizer.assets import blue_noise, glyph_masks, linear_to_srgb, load_target, resolve_font, srgb_to_linear
from ascii_optimizer.cli import DEFAULT_CHARS
from ascii_optimizer.optimizer import Grid, render_grid, solve


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                          [0.2119034982, 0.6806995451, 0.1073969566],
                          [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32).T
    return np.cbrt(np.maximum(lms, 0)) @ np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32).T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lms = lab @ np.array([[1.0, 0.3963377774, 0.2158037573],
                          [1.0, -0.1055613458, -0.0638541728],
                          [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32).T
    lms = lms ** 3
    return lms @ np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                            [-1.2684380046, 2.6097574011, -0.3413193965],
                            [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32).T


def in_gamut(lab: np.ndarray) -> np.ndarray:
    rgb = oklab_to_rgb(lab)
    return (rgb >= -1e-5).all(-1) & (rgb <= 1 + 1e-5).all(-1)


def gamut_map(lab: np.ndarray) -> np.ndarray:
    """Keep requested lightness and hue; reduce chroma only at gamut edges."""
    lab = lab.copy()
    lab[..., 0] = np.clip(lab[..., 0], 0, 1)
    bad = ~in_gamut(lab)
    if bad.any():
        chroma = lab[bad, 1:].copy()
        low = np.zeros(len(chroma), dtype=np.float32)
        high = np.ones(len(chroma), dtype=np.float32)
        for _ in range(12):
            mid = (low + high) * 0.5
            trial = lab[bad].copy()
            trial[:, 1:] = chroma * mid[:, None]
            valid = in_gamut(trial)
            low = np.where(valid, mid, low)
            high = np.where(valid, high, mid)
        lab[bad, 1:] = chroma * low[:, None]
    return np.clip(oklab_to_rgb(lab), 0, 1)


def lighten_at_fixed_chroma(linear: np.ndarray, new_lightness: np.ndarray) -> np.ndarray:
    """Raise OKLab lightness only as far as the existing chroma stays in gamut."""
    lab = rgb_to_oklab(linear)
    original = lab[..., 0].copy()
    requested = np.maximum(original, np.clip(new_lightness, 0, 1))
    lab[..., 0] = requested
    invalid = ~in_gamut(lab)
    if invalid.any():
        low = original[invalid].copy()
        high = requested[invalid].copy()
        trial = lab[invalid].copy()
        for _ in range(16):
            mid = (low + high) * 0.5
            trial[:, 0] = mid
            valid = in_gamut(trial)
            low = np.where(valid, mid, low)
            high = np.where(valid, high, mid)
        lab[invalid, 0] = low
    return np.clip(oklab_to_rgb(lab), 0, 1)


def headroom_curve(lightness: np.ndarray, gain: float) -> np.ndarray:
    """Boost shadows and midtones while tending to zero adjustment at white."""
    return lightness + gain * lightness * (1 - lightness) ** 2


def inverse_headroom_curve(lightness: np.ndarray, gain: float) -> np.ndarray:
    low = np.zeros_like(lightness)
    high = np.ones_like(lightness)
    for _ in range(18):
        midpoint = (low + high) * 0.5
        lower = headroom_curve(midpoint, gain) < lightness
        low = np.where(lower, midpoint, low)
        high = np.where(lower, high, midpoint)
    return (low + high) * 0.5


def headroom_input(source: np.ndarray, gain: float) -> np.ndarray:
    lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(source)).numpy())
    original_lightness = lab[..., 0].copy()
    solver_lightness = inverse_headroom_curve(original_lightness, gain)
    # Carry chroma through the compressed solve space reversibly. Holding a,b
    # fixed while darkening saturated colors would push them out of gamut.
    lab[..., 1:] *= (solver_lightness / np.maximum(original_lightness, 1e-6))[..., None]
    lab[..., 0] = solver_lightness
    return linear_to_srgb(torch.from_numpy(gamut_map(lab))).numpy()


def headroom_output(linear: np.ndarray, gain: float) -> np.ndarray:
    lab = rgb_to_oklab(linear)
    solver_lightness = lab[..., 0].copy()
    display_lightness = headroom_curve(solver_lightness, gain)
    lab[..., 1:] *= (display_lightness / np.maximum(solver_lightness, 1e-6))[..., None]
    lab[..., 0] = display_lightness
    return gamut_map(lab)


def tone_input(source: np.ndarray, strength: float, edge: float = 0,
               inverse: bool = False) -> np.ndarray:
    lab = rgb_to_oklab(srgb_to_linear(torch.from_numpy(source)).numpy())
    lightness = lab[..., 0]
    if edge:
        values = torch.from_numpy(lightness).unsqueeze(0).unsqueeze(0)
        smooth = F.avg_pool2d(F.pad(values, (8, 8, 8, 8), mode="reflect"), 17, stride=1)
        lightness = np.clip(lightness + edge * (lightness - smooth[0, 0].numpy()), 0, 1)
    lab[..., 0] = strength * (1 - lightness if inverse else lightness)
    linear = gamut_map(lab)
    return linear_to_srgb(torch.from_numpy(linear)).numpy()


def tone_output(linear: np.ndarray, strength: float, inverse: bool = False) -> np.ndarray:
    lab = rgb_to_oklab(linear)
    lab[..., 0] = 1 - lab[..., 0] / strength if inverse else lab[..., 0] / strength
    return gamut_map(lab)


def to_image(linear: np.ndarray) -> Image.Image:
    srgb = linear_to_srgb(torch.from_numpy(linear)).numpy()
    return Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8), "RGB")


def structure_first(source: np.ndarray, masks: np.ndarray, background: np.ndarray, *,
                    density_floor: float = 0.02, density_span: float = 0.34,
                    density_exponent: float = 1.65, dither_strength: float = 0,
                    seed: int = 0) -> Grid:
    """Choose glyphs by a tone density curve and local luminance texture."""
    height, width = source.shape[:2]
    cell_height, cell_width = masks.shape[1:]
    rows = (height + cell_height - 1) // cell_height
    cols = (width + cell_width - 1) // cell_width
    linear = srgb_to_linear(torch.from_numpy(source)).numpy()
    linear = np.pad(linear, ((0, rows * cell_height - height),
                             (0, cols * cell_width - width), (0, 0)), mode="edge")
    cells = linear.reshape(rows, cell_height, cols, cell_width, 3).transpose(0, 2, 1, 3, 4)
    mean = cells.mean((2, 3))
    lab = rgb_to_oklab(mean)
    source_lightness = lab[..., 0].copy()
    lab[..., 0] = np.minimum(1, 0.48 + 0.52 * source_lightness)
    foreground = gamut_map(lab)

    # The tone curve assigns midtones to sparse letters. The texture term then
    # selects the letter shape that best follows variation within each cell.
    desired_coverage = density_floor + density_span * source_lightness ** density_exponent
    if dither_strength:
        desired_coverage = np.clip(
            desired_coverage + dither_strength * (blue_noise((rows, cols), seed) - 0.5),
            0, 1,
        )
    mask_coverage = masks.mean((1, 2))
    source_texture = rgb_to_oklab(cells)[..., 0]
    source_texture -= source_texture.mean((2, 3), keepdims=True)
    foreground_lightness = rgb_to_oklab(foreground)[..., 0]
    background_lightness = float(rgb_to_oklab(background.reshape(1, 1, 3))[0, 0, 0])
    contrast = (foreground_lightness - background_lightness)[..., None, None]
    best_score = np.full((rows, cols), np.inf, dtype=np.float32)
    best_glyph = np.zeros((rows, cols), dtype=np.uint8)
    for index, mask in enumerate(masks):
        glyph_texture = contrast * (mask - mask_coverage[index])
        texture_error = ((source_texture - glyph_texture) ** 2).mean((2, 3))
        score = 1.5 * (desired_coverage - mask_coverage[index]) ** 2 + texture_error
        selected = score < best_score
        best_score = np.where(selected, score, best_score)
        best_glyph = np.where(selected, index, best_glyph)
    return Grid(best_glyph, foreground, background)


def brighten_structure(grid: Grid, *, ink_gain: float, background_lift: float) -> Grid:
    """Change only colors; every selected glyph remains identical."""
    ink_lightness = rgb_to_oklab(grid.color)[..., 0]
    color = lighten_at_fixed_chroma(grid.color, headroom_curve(ink_lightness, ink_gain))
    background = grid.background.reshape(1, 1, 3)
    background_lightness = rgb_to_oklab(background)[..., 0]
    brighter_background = lighten_at_fixed_chroma(background, background_lightness + background_lift)
    return Grid(grid.glyph.copy(), color, brighter_background.reshape(3))


def render_local_canvas(grid: Grid, masks: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Lift each cell's canvas by source tone without changing the glyph map."""
    rows, cols = grid.glyph.shape
    cell_height, cell_width = masks.shape[1:]
    height, width = source.shape[:2]
    linear = srgb_to_linear(torch.from_numpy(source)).numpy()
    linear = np.pad(linear, ((0, rows * cell_height - height),
                             (0, cols * cell_width - width), (0, 0)), mode="edge")
    cell_mean = linear.reshape(rows, cell_height, cols, cell_width, 3).mean((1, 3))
    source_lab = rgb_to_oklab(cell_mean)
    foreground_lightness = rgb_to_oklab(grid.color)[..., 0]
    background_lab = rgb_to_oklab(grid.background.reshape(1, 1, 3))
    local_lab = np.broadcast_to(background_lab, (rows, cols, 3)).copy()
    local_lab[..., 0] = np.minimum(
        background_lab[..., 0] + 0.36 * source_lab[..., 0] ** 1.5,
        foreground_lightness - 0.10,
    )
    local_lab[..., 1:] += 0.30 * source_lab[..., 1:]
    local_background = gamut_map(local_lab)
    background_map = torch.from_numpy(local_background).permute(2, 0, 1).unsqueeze(0)
    background_map = F.avg_pool2d(F.pad(background_map, (1, 1, 1, 1), mode="reflect"), 3)
    background_pixels = F.interpolate(background_map, size=(rows * cell_height, cols * cell_width),
                                      mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    mask = masks[grid.glyph].transpose(0, 2, 1, 3).reshape(rows * cell_height, cols * cell_width)
    foreground = np.repeat(np.repeat(grid.color, cell_height, axis=0), cell_width, axis=1)
    pixels = mask[..., None] * foreground + (1 - mask[..., None]) * background_pixels
    return pixels[:height, :width]


def contact_sheet(previews: list[tuple[str, Image.Image]], path: Path,
                  crop: tuple[int, int, int, int] | None = None) -> None:
    width, height = previews[0][1].size
    thumb_size = (width // 2, height // 2) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    margin, heading = 18, 52
    rows = (len(previews) + 2) // 3
    sheet = Image.new("RGB", (3 * (thumb_size[0] + margin) + margin,
                              rows * (thumb_size[1] + heading + margin) + margin), "#15171d")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(resolve_font("DejaVu Sans")), 20)
    for index, (title, preview) in enumerate(previews):
        x = margin + index % 3 * (thumb_size[0] + margin)
        y = margin + index // 3 * (thumb_size[1] + heading + margin)
        draw.text((x, y + 4), title, font=font, fill="white")
        panel = preview.resize(thumb_size, Image.Resampling.LANCZOS) if crop is None else preview.crop(crop)
        sheet.paste(panel, (x, y + heading))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=Path("screenshots/test_image.png"))
    parser.add_argument("--output", type=Path, default=Path("rendered/color_strategies"))
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--cell", default="8x16")
    parser.add_argument("--steps", type=int, default=36)
    args = parser.parse_args()
    size = tuple(map(int, args.size.split("x")))
    cell = tuple(map(int, args.cell.split("x")))
    args.output.mkdir(parents=True, exist_ok=True)
    source = load_target(args.image, size, "fill", False)
    masks = glyph_masks(resolve_font("Monospace"), list(DEFAULT_CHARS), cell)
    preview_specs = [
        ("Current solver", "current"),
        ("Global boost · previous", "iso_global_strong"),
        ("Headroom-aware boost", "iso_headroom"),
        ("Structure-first · original", "structure_first"),
        ("Structure · brighter ink", "structure_bright_ink"),
        ("Structure · ink + canvas", "structure_bright_canvas"),
        ("Structure · local canvas", "structure_local_canvas"),
        ("Inverse ink · option", "inverse_ink"),
    ]
    previews = [("Source image", Image.fromarray(np.rint(source * 255).astype(np.uint8), "RGB"))]
    baseline_background = None
    baseline_grid_background = None
    structure_grid = None
    report = []
    for title, name in preview_specs:
        print(f"Rendering {title}", flush=True)
        if name.startswith("structure_"):
            assert baseline_grid_background is not None
            if name == "structure_first":
                structure_grid = structure_first(source, masks, baseline_grid_background)
                grid = structure_grid
            else:
                assert structure_grid is not None
                grid = brighten_structure(structure_grid, ink_gain=2.0,
                                          background_lift=0.12 if name == "structure_bright_canvas" else 0.0)
                np.testing.assert_array_equal(grid.glyph, structure_grid.glyph)
        else:
            if name == "current":
                target, background = source, "auto"
            elif name == "inverse_ink":
                target, background = tone_input(source, 0.60, inverse=True), "#000000"
            else:
                assert baseline_background is not None
                base = np.array([int(baseline_background[i:i + 2], 16)
                                 for i in (1, 3, 5)], dtype=np.float32).reshape(1, 1, 3) / 255
                if name == "iso_global_strong":
                    target = tone_input(source, 0.28)
                    encoded = tone_input(base, 0.28)[0, 0]
                else:
                    target = headroom_input(source, 2.0)
                    encoded = headroom_input(base, 2.0)[0, 0]
                background = "#" + "".join(f"{round(float(c) * 255):02x}" for c in encoded)
            grid = solve(target, masks, steps=args.steps, candidates=6, dither=0.005,
                         background=background, device="cpu")
        if name == "current":
            baseline_grid_background = grid.background
            encoded = linear_to_srgb(torch.from_numpy(grid.background)).numpy()
            baseline_background = "#" + "".join(f"{round(float(c) * 255):02x}" for c in encoded)
        linear = (render_local_canvas(grid, masks, source) if name == "structure_local_canvas"
                  else render_grid(grid, masks)[:size[1], :size[0]])
        if name == "iso_global_strong":
            linear = tone_output(linear, 0.28)
        elif name == "iso_headroom":
            linear = headroom_output(linear, 2.0)
        elif name == "inverse_ink":
            linear = tone_output(linear, 0.60, inverse=True)
        preview = to_image(linear)
        preview.save(args.output / f"{name}.png")
        previews.append((title, preview))
        coverage = masks[grid.glyph].mean((2, 3))
        report.append({"name": name, "title": title,
                       "glyphs_match_original_structure": bool(
                           name.startswith("structure_") and structure_grid is not None
                           and np.array_equal(grid.glyph, structure_grid.glyph)),
                       "mean_display_lightness": float(rgb_to_oklab(linear)[..., 0].mean()),
                       "mean_mask_coverage": float(coverage.mean()),
                       "dense_glyph_fraction": float((coverage >= 0.3).mean())})
    contact_sheet(previews, args.output / "comparison.png")
    contact_sheet(previews, args.output / "comparison_detail.png",
                  (size[0] * 17 // 100, size[1] * 12 // 100,
                   size[0] * 67 // 100, size[1] * 79 // 100))
    contact_sheet(previews, args.output / "comparison_yellow.png",
                  (size[0] * 60 // 100, size[1] * 8 // 100,
                   size[0] * 96 // 100, size[1] * 53 // 100))
    (args.output / "comparison.json").write_text(json.dumps({
        "source": str(args.image.resolve()), "size": size, "cell": cell,
        "steps": args.steps, "background": baseline_background,
        "source_mean_lightness": float(rgb_to_oklab(
            srgb_to_linear(torch.from_numpy(source)).numpy())[..., 0].mean()),
        "variants": report,
    }, indent=2) + "\n")
    print(f"Wrote {args.output / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
