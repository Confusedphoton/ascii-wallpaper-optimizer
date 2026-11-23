"""Image and font preparation, using the same SDF/alpha atlases for solving and playback."""

from __future__ import annotations

from io import BytesIO
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

from .defaults import (
    DEFAULT_SDF_MIN_SPREAD,
    DEFAULT_SDF_SPREAD_RATIO,
    DEFAULT_SDF_SUPERSAMPLE,
    DEFAULT_SDF_THIN_RANGE,
    DEFAULT_SDF_WEIGHT_RANGE,
)

# Keep levels 0–2 stable for saved configs; levels 3–4 add denser grids.
DETAIL_COLUMNS = (180, 120, 80, 240, 320)


def cell_for_display(size: tuple[int, int], detail: int) -> tuple[int, int]:
    """Keep character density stable across monitor resolutions."""
    target_columns = DETAIL_COLUMNS[detail]
    width = min(127, max(2, round(size[0] / target_columns)))
    return width, width * 2


def resolve_font(name: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", name],
        check=True, capture_output=True, text=True,
    )
    path = Path(result.stdout.strip())
    if not path.is_file():
        raise ValueError(f"Could not find font: {name}")
    return path.resolve()


def sdf_spread_for_cell(cell: tuple[int, int]) -> float:
    """Choose an SDF spread radius (in cell pixels) representable in metadata.png."""
    raw = max(DEFAULT_SDF_MIN_SPREAD, min(cell) * DEFAULT_SDF_SPREAD_RATIO)
    # Quantize to 1/16 px so metadata.png (uint8) and manifest uniforms match exactly.
    return min(255, max(16, round(raw * 16))) / 16.0


def _bounded_edt(feature: np.ndarray, radius: int) -> np.ndarray:
    """Exact 2D Euclidean distance to True pixels in `feature` up to `radius`.

    Vectorized across leading batch dimensions `(..., H, W)` using a 2-pass
    separable squared-distance transform. Exact for every pixel whose true
    nearest feature is within `radius` (and saturates at `>= radius` beyond).
    """
    radius = max(1, int(radius))
    inf = np.float32((radius + 2) ** 2 * 2)
    f = np.where(feature, np.float32(0.0), inf)
    height, width = f.shape[-2:]

    # Pass 1: vertical min over dy in [-radius, radius].
    vert = f.copy()
    for dy in range(1, radius + 1):
        d2 = np.float32(dy * dy)
        if dy < height:
            np.minimum(vert[..., dy:, :], f[..., :-dy, :] + d2, out=vert[..., dy:, :])
            np.minimum(vert[..., :-dy, :], f[..., dy:, :] + d2, out=vert[..., :-dy, :])

    # Pass 2: horizontal min over dx in [-radius, radius].
    dist2 = vert.copy()
    for dx in range(1, radius + 1):
        d2 = np.float32(dx * dx)
        if dx < width:
            np.minimum(dist2[..., :, dx:], vert[..., :, :-dx] + d2, out=dist2[..., :, dx:])
            np.minimum(dist2[..., :, :-dx], vert[..., :, dx:] + d2, out=dist2[..., :, :-dx])

    return np.sqrt(np.minimum(dist2, np.float32(radius * radius)), dtype=np.float32)


def glyph_sdf_atlas(
    font_path: Path,
    chars: list[str],
    cell: tuple[int, int],
    scale: int | None = None,
    spread: float | None = None,
) -> tuple[np.ndarray, float]:
    """Build a byte-quantized Signed Distance Field atlas for the font and cell size.

    Values are normalized into [0, 1] with 0.5 on the exact glyph outline,
    >0.5 inside strokes, and <0.5 outside strokes. Empty glyphs are identically 0.
    """
    width, height = cell
    effective_spread = sdf_spread_for_cell(cell) if spread is None else float(spread)
    if effective_spread <= 0:
        raise ValueError("SDF spread must be positive")
    # Keep high-resolution rasterization crisp while bounding work on huge cells.
    if scale is None:
        scale = max(2, min(DEFAULT_SDF_SUPERSAMPLE, 128 // max(1, height)))
    pad = int(math.ceil(effective_spread * scale)) + 2
    hi_w, hi_h = width * scale, height * scale
    canvas_w, canvas_h = hi_w + 2 * pad, hi_h + 2 * pad
    font = ImageFont.truetype(str(font_path), hi_h)

    rasters = np.empty((len(chars), canvas_h, canvas_w), dtype=np.bool_)
    nonempty = np.zeros(len(chars), dtype=np.bool_)
    for index, char in enumerate(chars):
        image = Image.new("L", (canvas_w, canvas_h))
        draw = ImageDraw.Draw(image)
        draw.text((pad + hi_w / 2, pad + hi_h / 2), char, font=font,
                  anchor="mm", fill=255, stroke_width=0)
        inside = np.asarray(image, dtype=np.uint8) >= 128
        rasters[index] = inside
        nonempty[index] = bool(inside.any())

    atlas = np.zeros((len(chars), height, width), dtype=np.float32)
    if nonempty.any():
        active = rasters[nonempty]
        radius = int(math.ceil(effective_spread * scale)) + 1
        d_in = _bounded_edt(~active, radius)
        d_out = _bounded_edt(active, radius)
        # Half-subpixel shift places the zero-level set precisely between inside/outside pixels.
        signed_px = np.where(active, d_in - 0.5, -(d_out - 0.5)) / np.float32(scale)
        cropped = signed_px[:, pad:pad + hi_h, pad:pad + hi_w]
        downsampled = cropped.reshape(len(active), height, scale, width, scale).mean((2, 4))
        encoded = np.clip(0.5 + downsampled / (2.0 * effective_spread), 0.0, 1.0)
        # Keep non-empty glyph pixels above the shader's empty-glyph guard (1/255) only when nonzero.
        quantized = np.rint(encoded * 255.0).astype(np.float32) / 255.0
        atlas[nonempty] = quantized
    return atlas, effective_spread


def encode_stroke_weight_u8(weight: np.ndarray) -> np.ndarray:
    """Encode stroke weight in [-1, 1] into uint8 with 128 as exact neutral (0.0)."""
    return np.clip(np.rint(128.0 + np.asarray(weight, dtype=np.float32) * 127.0), 0, 255).astype(np.uint8)


def decode_stroke_weight_u8(encoded: np.ndarray) -> np.ndarray:
    """Decode uint8 stroke weight back to [-1, 1] matching ascii.frag."""
    return ((np.asarray(encoded, dtype=np.float32) - 128.0) / 127.0).astype(np.float32)


def sdf_to_mask(
    atlas: np.ndarray,
    weight=None,
    *,
    spread: float = 0.0,
    weight_range: float = DEFAULT_SDF_WEIGHT_RANGE,
    thin_range: float = DEFAULT_SDF_THIN_RANGE,
):
    """Evaluate antialiased ink coverage from an SDF atlas (or pass through alpha masks).

    Works identically on NumPy arrays and PyTorch tensors (differentiable w.r.t. `weight`).
    Positive weights dilate strokes up to `weight_range`; negative weights erode up to
    `thin_range` so thin strokes stay visible as delicate hairlines instead of vanishing.
    """
    if spread <= 0.0:
        return atlas
    aa = 0.5 / (2.0 * float(spread))
    is_numpy = isinstance(atlas, np.ndarray)
    if is_numpy:
        w = np.float32(0.0) if weight is None else np.asarray(weight, dtype=np.float32)
        bias = np.where(w >= 0.0, w * np.float32(weight_range), w * np.float32(thin_range))
        edge = np.float32(0.5) - bias
        t = np.clip((atlas - (edge - aa)) / (2.0 * aa), 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
        return np.where(atlas <= 0.004, np.float32(0.0), mask).astype(np.float32)

    import torch

    if weight is None:
        bias = 0.0
    elif isinstance(weight, torch.Tensor):
        bias = torch.where(weight >= 0.0, weight * weight_range, weight * thin_range)
    else:
        bias = weight * (weight_range if weight >= 0.0 else thin_range)
    edge = 0.5 - bias
    t = ((atlas - (edge - aa)) / (2.0 * aa)).clamp(0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    return torch.where(atlas <= 0.004, torch.zeros_like(mask), mask)


def glyph_masks(font_path: Path, chars: list[str], cell: tuple[int, int], scale: int = 4) -> np.ndarray:
    width, height = cell
    font = ImageFont.truetype(str(font_path), height * scale)
    masks = []
    for char in chars:
        image = Image.new("L", (width * scale, height * scale))
        draw = ImageDraw.Draw(image)
        draw.text((width * scale / 2, height * scale / 2), char, font=font,
                  anchor="mm", fill=255, stroke_width=0)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        masks.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(masks)


def save_atlas(masks: np.ndarray, path: Path) -> None:
    count, height, width = masks.shape
    atlas = np.zeros((height, count * width, 4), dtype=np.uint8)
    atlas[:, :, :3] = 255
    for index, mask in enumerate(masks):
        atlas[:, index * width:(index + 1) * width, 3] = np.rint(mask * 255).astype(np.uint8)
    Image.fromarray(atlas, "RGBA").save(path)


def save_metadata(size: tuple[int, int], cell: tuple[int, int], count: int, path: Path,
                  *, invert_ink: bool = False, sdf_spread: float = 0.0) -> None:
    width, height = size
    cell_width, cell_height = cell
    if max(width, height) > 65535 or max(cell_width, cell_height) > 255:
        raise ValueError("Output dimensions exceed the playback metadata format")
    encoded_spread = min(255, max(0, int(round(sdf_spread * 16))))
    pixels = np.array([[(width >> 8, width & 255, cell_width),
                        (height >> 8, height & 255, cell_height),
                        (count - 1, int(invert_ink), encoded_spread)]], dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)


def load_target(path: Path, size: tuple[int, int], crop: str, grayscale: bool) -> np.ndarray:
    width, height = size
    with Image.open(path) as original:
        source = ImageOps.exif_transpose(original)
        # OKLab conversion expects sRGB, not unconverted Display P3/Adobe RGB
        # channel values. Transform before resizing or calculating grayscale.
        profile = original.info.get("icc_profile")
        if profile:
            try:
                source = ImageCms.profileToProfile(
                    source, ImageCms.ImageCmsProfile(BytesIO(profile)),
                    ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError):
                # Preserve support for files with invalid or unsupported tags.
                source = source.convert("RGB")
        else:
            source = source.convert("RGB")
        if crop == "fill":
            source = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS)
        elif crop == "fit":
            source.thumbnail(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size)
            canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
            source = canvas
        else:
            raise ValueError(f"Unknown crop mode: {crop}")
        if grayscale:
            source = ImageOps.grayscale(source).convert("RGB")
        return np.asarray(source, dtype=np.float32) / 255.0


def srgb_to_linear(value):
    import torch
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    import torch
    value = value.clamp(0, 1)
    return torch.where(value <= 0.0031308, value * 12.92,
                       1.055 * torch.pow(value, 1 / 2.4) - 0.055)


def blue_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    """A deterministic rank map with low spatial frequencies suppressed."""
    rng = np.random.default_rng(seed)
    height, width = shape
    white = rng.standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    high_pass = np.minimum(radius / 0.18, 1.0) ** 1.5
    filtered = np.fft.ifft2(np.fft.fft2(white) * high_pass).real
    ranks = np.empty(filtered.size, dtype=np.float32)
    ranks[np.argsort(filtered, axis=None)] = (np.arange(filtered.size) + 0.5) / filtered.size
    return ranks.reshape(shape)
