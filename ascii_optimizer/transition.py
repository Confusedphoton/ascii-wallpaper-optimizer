"""Font-aware glyph paths and staggered slideshow timing."""

from __future__ import annotations

import numpy as np

from .assets import blue_noise


PACES = ("linear", "ease-in", "ease-out", "ease-in-out")
FIELDS = ("blue-noise", "ripple", "spiral", "directional", "edge", "salience")


def ease(t: float, pace: str) -> float:
    t = float(np.clip(t, 0, 1))
    if pace == "ease-in":
        return t * t
    if pace == "ease-out":
        return 1 - (1 - t) ** 2
    if pace == "ease-in-out":
        return t * t * (3 - 2 * t)
    if pace != "linear":
        raise ValueError(f"Unknown transition pace: {pace}")
    return t


def phase_offsets(shape: tuple[int, int], field: str, seed: int,
                  source: np.ndarray | None = None) -> np.ndarray:
    """Return deterministic phases in [0, 1], with blue noise in structured fields."""
    if field not in FIELDS:
        raise ValueError(f"Unknown transition field: {field}")
    rows, cols = shape
    y, x = np.mgrid[:rows, :cols].astype(np.float32)
    x = (x + 0.5) / max(cols, 1)
    y = (y + 0.5) / max(rows, 1)
    noise = blue_noise(shape, seed)
    if field == "blue-noise":
        return noise
    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    if field == "ripple":
        base = radius / np.sqrt(0.5)
    elif field == "spiral":
        angle = (np.arctan2(y - 0.5, x - 0.5) + np.pi) / (2 * np.pi)
        base = (angle + radius * 0.7) % 1
    elif field == "directional":
        base = x
    elif field == "edge":
        base = np.minimum.reduce((x, 1 - x, y, 1 - y)) * 2
    else:
        if source is None:
            raise ValueError("Salience needs a source image")
        from .tone import srgb_to_oklab

        lightness = srgb_to_oklab(source[..., :3])[..., 0]
        gy = np.zeros_like(lightness)
        gx = np.zeros_like(lightness)
        gy[1:] = lightness[1:] - lightness[:-1]
        gx[:, 1:] = lightness[:, 1:] - lightness[:, :-1]
        salience = np.sqrt(gx * gx + gy * gy)
        sy = np.minimum((y * source.shape[0]).astype(int), source.shape[0] - 1)
        sx = np.minimum((x * source.shape[1]).astype(int), source.shape[1] - 1)
        sampled = salience[sy, sx]
        scale = np.quantile(sampled, 0.95)
        base = 1 - np.clip(sampled / max(scale, 1e-8), 0, 1)
    return np.clip(0.82 * base + 0.18 * noise, 0, 1).astype(np.float32)


def local_progress(t: float, offsets: np.ndarray, spread: float) -> np.ndarray:
    """clamp(s * (t - delta_i)), with delta_i = spread * offsets_i."""
    if not 0 <= spread < 1:
        raise ValueError("Transition spread must be in [0, 1)")
    return np.clip((t - spread * offsets) / (1 - spread), 0, 1).astype(np.float32)


def glyph_paths(masks: np.ndarray, neighbors: int = 4) -> list[list[list[int]]]:
    """Shortest paths on a connected sparse graph of rendered font glyphs."""
    count = len(masks)
    flat = masks.reshape(count, -1).astype(np.float64)
    norms = (flat * flat).sum(1)
    distance = np.sqrt(np.maximum(0, norms[:, None] + norms[None, :] - 2 * flat @ flat.T) / flat.shape[1])
    # Neighboring coverage levels connect the graph; shape-nearest links allow
    # shortcuts only between glyphs of comparable visual density.
    links = np.zeros((count, count), dtype=bool)
    coverage = flat.mean(1)
    order = np.argsort(coverage)
    for a, b in zip(order[:-1], order[1:]):
        links[a, b] = links[b, a] = True
    for a in range(count):
        nearby = distance[a].copy()
        nearby[a] = np.inf
        nearby[np.abs(coverage - coverage[a]) > 0.08] = np.inf
        for b in np.argsort(nearby)[:neighbors]:
            if a != b and np.isfinite(nearby[b]):
                links[a, b] = links[b, a] = True
    cost = np.where(links, distance + 0.5 * np.abs(coverage[:, None] - coverage[None, :]) + 1e-6, np.inf)
    np.fill_diagonal(cost, 0)
    following = np.broadcast_to(np.arange(count), (count, count)).copy()
    for k in range(count):
        via = cost[:, k, None] + cost[k, None, :]
        better = via < cost
        cost = np.minimum(cost, via)
        following = np.where(better, following[:, k, None], following)
    paths = []
    for a in range(count):
        row = []
        for b in range(count):
            path = [a]
            while path[-1] != b:
                path.append(int(following[path[-1], b]))
            row.append(path)
        paths.append(row)
    return paths


def path_glyphs(start: np.ndarray, end: np.ndarray, progress: np.ndarray,
                paths: list[list[list[int]]], previous: np.ndarray | None = None,
                remaining_frames: int | None = None) -> np.ndarray:
    result = np.empty_like(start)
    for a, b in np.unique(np.stack((start.ravel(), end.ravel()), axis=1), axis=0):
        selected = (start == a) & (end == b)
        path = paths[int(a)][int(b)]
        step = np.minimum(len(path) - 1, np.floor(progress[selected] * len(path)).astype(int))
        if previous is not None and remaining_frames is not None:
            prior = previous[selected]
            current_step = np.array([path.index(int(glyph)) for glyph in prior])
            step = np.maximum(step, len(path) - 1 - remaining_frames)
            step = np.maximum(step, current_step)
            step = np.minimum(step, current_step + 1)
        result[selected] = np.asarray(path, dtype=np.uint8)[step]
    return result


def recolor(target: np.ndarray, masks: np.ndarray, glyphs: np.ndarray,
            background: np.ndarray) -> np.ndarray:
    """Fit fixed transition glyphs and ink colors directly in OKLab."""
    rows, cols = glyphs.shape
    height, width = masks.shape[1:]
    padded = np.pad(target, ((0, rows * height - target.shape[0]),
                             (0, cols * width - target.shape[1]), (0, 0)), mode="edge")
    cells = padded.reshape(
        rows, height, cols, width, 3).transpose(0, 2, 1, 3, 4)
    ink = masks[glyphs][..., None]
    under = background.reshape(1, 1, 1, 1, 3)
    numerator = (ink * (cells - (1 - ink) * under)).sum((2, 3))
    denominator = (ink * ink).sum((2, 3))
    colors = numerator / np.maximum(denominator, 1e-8)
    # Match the optimizer's bounded OKLab parameter range without converting
    # the fit back through RGB.
    colors[..., 0] = np.clip(colors[..., 0], 0, 1)
    colors[..., 1:] = np.clip(colors[..., 1:], -0.4, 0.4)
    return colors.astype(np.float32)
