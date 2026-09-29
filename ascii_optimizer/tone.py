"""Perceptual tone preparation and display transforms for ASCII grids."""

from __future__ import annotations

import numpy as np
import torch

from .assets import linear_to_srgb, srgb_to_linear


HEADROOM_GAIN = 2.0


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB to OKLab (used only at color-space boundaries)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                          [0.2119034982, 0.6806995451, 0.1073969566],
                          [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32).T
    return np.cbrt(np.maximum(lms, 0)) @ np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32).T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab to linear RGB (used only at color-space boundaries)."""
    lms = lab @ np.array([[1.0, 0.3963377774, 0.2158037573],
                          [1.0, -0.1055613458, -0.0638541728],
                          [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32).T
    lms = lms ** 3
    return lms @ np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                            [-1.2684380046, 2.6097574011, -0.3413193965],
                            [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32).T


def gamut_map(lab: np.ndarray) -> np.ndarray:
    """Encode OKLab as in-gamut linear RGB, reducing chroma at gamut edges."""
    lab = lab.copy()
    lab[..., 0] = np.clip(lab[..., 0], 0, 1)
    rgb = oklab_to_rgb(lab)
    bad = ((rgb < -1e-5) | (rgb > 1 + 1e-5)).any(-1)
    if bad.any():
        chroma = lab[bad, 1:].copy()
        low = np.zeros(len(chroma), dtype=np.float32)
        high = np.ones(len(chroma), dtype=np.float32)
        trial = lab[bad].copy()
        for _ in range(12):
            mid = (low + high) * 0.5
            trial[:, 1:] = chroma * mid[:, None]
            candidate = oklab_to_rgb(trial)
            valid = ((candidate >= -1e-5) & (candidate <= 1 + 1e-5)).all(-1)
            low = np.where(valid, mid, low)
            high = np.where(valid, high, mid)
        lab[bad, 1:] = chroma * low[:, None]
        rgb = oklab_to_rgb(lab)
    return np.clip(rgb, 0, 1)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """Decode an sRGB image and convert it to the solver's OKLab space."""
    return rgb_to_oklab(srgb_to_linear(torch.as_tensor(srgb, dtype=torch.float32)).numpy())


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Gamut-map OKLab and encode sRGB for files and the playback shader."""
    return linear_to_srgb(torch.from_numpy(gamut_map(np.asarray(lab, dtype=np.float32)))).numpy()


def playback_rgb8(lab: np.ndarray) -> np.ndarray:
    """Encode precisely the RGB bytes stored in a playback color texture."""
    return np.rint(oklab_to_srgb(lab) * 255).astype(np.uint8)


def quantize_playback_lab(lab: np.ndarray) -> np.ndarray:
    """Project colors onto the RGB8 palette that the playback shader decodes.

    Use this at solver boundaries so previews, cached grids, and live playback
    all compose the same foreground and background colors.
    """
    return srgb_to_oklab(playback_rgb8(lab).astype(np.float32) / 255)


def headroom_curve(lightness: np.ndarray, gain: float = HEADROOM_GAIN) -> np.ndarray:
    return lightness + gain * lightness * (1 - lightness) ** 2


def inverse_headroom_curve(lightness: np.ndarray, gain: float = HEADROOM_GAIN) -> np.ndarray:
    low = np.zeros_like(lightness)
    high = np.ones_like(lightness)
    for _ in range(18):
        midpoint = (low + high) * 0.5
        lower = headroom_curve(midpoint, gain) < lightness
        low = np.where(lower, midpoint, low)
        high = np.where(lower, high, midpoint)
    return (low + high) * 0.5


def encode_headroom_lab(source_lab: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Precompensate an OKLab target for the output headroom curve."""
    lab = np.asarray(source_lab, dtype=np.float32).copy()
    target = 1 - lab[..., 0] if invert_ink else lab[..., 0].copy()
    encoded = inverse_headroom_curve(target)
    lab[..., 1:] *= (encoded / np.maximum(target, 1e-6))[..., None]
    lab[..., 0] = encoded
    return lab


def encode_headroom(source_srgb: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """sRGB boundary wrapper for headroom encoding."""
    return oklab_to_srgb(encode_headroom_lab(srgb_to_oklab(source_srgb), invert_ink=invert_ink))


def invert_source(source_srgb: np.ndarray) -> np.ndarray:
    """Prepare a dark-ink target for the structure-first glyph selector."""
    lab = srgb_to_oklab(source_srgb)
    lab[..., 0] = 1 - lab[..., 0]
    return oklab_to_srgb(lab)


def display_tone_lab(lab: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Apply the display lightness curve directly in OKLab."""
    lab = np.asarray(lab, dtype=np.float32).copy()
    original = lab[..., 0].copy()
    boosted = headroom_curve(original)
    lab[..., 1:] *= (boosted / np.maximum(original, 1e-6))[..., None]
    lab[..., 0] = boosted
    if invert_ink:
        lab[..., 0] = 1 - lab[..., 0]
    return lab


def display_tone(linear_rgb: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Color-boundary adapter returning linear RGB for older callers."""
    return gamut_map(display_tone_lab(rgb_to_oklab(linear_rgb), invert_ink=invert_ink))


def encoded_background(background_lab: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Encode an OKLab background for the headroom display curve."""
    return quantize_playback_lab(encode_headroom_lab(
        np.asarray(background_lab, dtype=np.float32).reshape(1, 1, 3),
        invert_ink=invert_ink))[0, 0]
