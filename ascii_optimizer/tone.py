"""Perceptual tone preparation and display transforms for ASCII grids."""

from __future__ import annotations

import numpy as np
import torch

from .assets import linear_to_srgb, srgb_to_linear


HEADROOM_GAIN = 2.0


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


def gamut_map(lab: np.ndarray) -> np.ndarray:
    """Keep requested lightness and hue, reducing chroma only at gamut edges."""
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


def encode_headroom(source_srgb: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Encode a visible target for the selected output lightness transform."""
    lab = rgb_to_oklab(srgb_to_linear(torch.as_tensor(source_srgb)).numpy())
    target = 1 - lab[..., 0] if invert_ink else lab[..., 0].copy()
    encoded = inverse_headroom_curve(target)
    lab[..., 1:] *= (encoded / np.maximum(target, 1e-6))[..., None]
    lab[..., 0] = encoded
    return linear_to_srgb(torch.from_numpy(gamut_map(lab))).numpy()


def invert_source(source_srgb: np.ndarray) -> np.ndarray:
    """Prepare a dark-ink target for the structure-first glyph selector."""
    lab = rgb_to_oklab(srgb_to_linear(torch.as_tensor(source_srgb)).numpy())
    lab[..., 0] = 1 - lab[..., 0]
    return linear_to_srgb(torch.from_numpy(gamut_map(lab))).numpy()


def display_tone(linear: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Apply the headroom curve, then optionally swap light and dark ink."""
    lab = rgb_to_oklab(linear)
    original = lab[..., 0].copy()
    boosted = headroom_curve(original)
    lab[..., 1:] *= (boosted / np.maximum(original, 1e-6))[..., None]
    lab[..., 0] = boosted
    displayed = gamut_map(lab)
    if invert_ink:
        inverted = rgb_to_oklab(displayed)
        inverted[..., 0] = 1 - inverted[..., 0]
        displayed = gamut_map(inverted)
    return displayed


def encoded_background(background_linear: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(torch.from_numpy(background_linear.reshape(1, 1, 3))).numpy()
    encoded = encode_headroom(srgb)
    return srgb_to_linear(torch.from_numpy(encoded)).numpy().reshape(3)
