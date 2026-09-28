"""Image and font preparation, using the same masks for solving and playback."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DETAIL_COLUMNS = (180, 120, 80)


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
                  *, invert_ink: bool = False) -> None:
    width, height = size
    cell_width, cell_height = cell
    if max(width, height) > 65535 or max(cell_width, cell_height) > 255:
        raise ValueError("Output dimensions exceed the playback metadata format")
    pixels = np.array([[(width >> 8, width & 255, cell_width),
                        (height >> 8, height & 255, cell_height),
                        (count - 1, int(invert_ink), 0)]], dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)


def load_target(path: Path, size: tuple[int, int], crop: str, grayscale: bool) -> np.ndarray:
    width, height = size
    with Image.open(path) as original:
        source = ImageOps.exif_transpose(original).convert("RGB")
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
