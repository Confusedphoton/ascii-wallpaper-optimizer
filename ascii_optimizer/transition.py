"""Font-aware glyph paths, spatial transport, and staggered slideshow timing."""

from __future__ import annotations

import numpy as np

from .assets import blue_noise


PACES = ("linear", "ease-in", "ease-out", "ease-in-out")
FIELDS = ("blue-noise", "ripple", "spiral", "directional", "edge", "salience")
TYPES = ("glyph-path", "earth-mover")


def _minimum_assignment(cost: np.ndarray) -> np.ndarray:
    """Hungarian assignment for a small square cost matrix, without extra deps."""
    count = len(cost)
    row_price = np.zeros(count + 1)
    col_price = np.zeros(count + 1)
    matched = np.zeros(count + 1, dtype=int)
    for row in range(1, count + 1):
        matched[0] = row
        column = 0
        minimum = np.full(count + 1, np.inf)
        used = np.zeros(count + 1, dtype=bool)
        previous = np.zeros(count + 1, dtype=int)
        while True:
            used[column] = True
            current = matched[column]
            available = np.flatnonzero(~used[1:]) + 1
            reduced = cost[current - 1, available - 1] - row_price[current] - col_price[available]
            better = reduced < minimum[available]
            minimum[available[better]] = reduced[better]
            previous[available[better]] = column
            following = available[np.argmin(minimum[available])]
            delta = minimum[following]
            if not np.isfinite(delta):
                raise ValueError("Transport assignment has no feasible matching")
            row_price[matched[used]] += delta
            col_price[used] -= delta
            minimum[~used] -= delta
            column = following
            if matched[column] == 0:
                break
        while column:
            preceding = previous[column]
            matched[column] = matched[preceding]
            column = preceding
    assignment = np.empty(count, dtype=int)
    assignment[matched[1:] - 1] = np.arange(count)
    return assignment


def _transport_cost(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1)


def _spatial_groups(indices: np.ndarray, features: np.ndarray, depth: int = 3) -> list[np.ndarray]:
    """Split regions into equal-capacity groups along their longest spatial axis."""
    if depth == 0:
        return [indices]
    positions = features[indices, :2]
    axis = np.argmax(np.ptp(positions, axis=0))
    ordered = indices[np.argsort(positions[:, axis], kind="stable")]
    middle = len(ordered) // 2
    return (_spatial_groups(ordered[:middle], features, depth - 1)
            + _spatial_groups(ordered[middle:], features, depth - 1))


def earth_mover_destinations(start: np.ndarray, end: np.ndarray,
                             start_color: np.ndarray, end_color: np.ndarray,
                             start_background: np.ndarray, end_background: np.ndarray,
                             masks: np.ndarray, *, invert_ink: bool = False) -> np.ndarray:
    """Balanced cell transport using hierarchical Euclidean earth mover costs."""
    if start.shape != end.shape or start.ndim != 2:
        raise ValueError("Transport endpoints must use the same two-dimensional grid")
    from .tone import display_tone_lab

    rows, cols = start.shape
    y, x = np.mgrid[:rows, :cols]
    height, width = masks.shape[1:]
    scale = max(rows * height, cols * width)
    positions = np.stack(((x + 0.5) * width, (y + 0.5) * height), axis=-1) / scale
    coverage = masks.mean((1, 2))

    def features(glyphs, colors, background):
        ink = coverage[glyphs][..., None]
        average = ink * colors + (1 - ink) * background
        displayed = display_tone_lab(average, invert_ink=invert_ink)
        return np.concatenate((0.25 * positions, displayed,
                               0.5 * ink), axis=-1).reshape(-1, 6)

    source = features(start, start_color, start_background)
    target = features(end, end_color, end_background)
    destinations = np.empty(start.size, dtype=int)

    def assign(source_indices, target_indices):
        if len(source_indices) <= 64:
            matches = _minimum_assignment(_transport_cost(source[source_indices], target[target_indices]))
            destinations[source_indices] = target_indices[matches]
            return
        source_groups = _spatial_groups(source_indices, source)
        target_groups = _spatial_groups(target_indices, target)
        cost = _transport_cost(np.array([source[group].mean(0) for group in source_groups]),
                               np.array([target[group].mean(0) for group in target_groups]))
        sizes = np.array([len(group) for group in source_groups])
        target_sizes = np.array([len(group) for group in target_groups])
        cost[sizes[:, None] != target_sizes[None, :]] = np.inf
        for group, match in zip(source_groups, _minimum_assignment(cost)):
            assign(group, target_groups[match])

    indices = np.arange(start.size)
    assign(indices, indices)
    return destinations.reshape(start.shape)


def transport_cells(glyphs: np.ndarray, colors: np.ndarray, background: np.ndarray,
                    destinations: np.ndarray, progress: np.ndarray,
                    masks: np.ndarray, *,
                    weights: np.ndarray | None = None):
    """Splat traveling cells onto the grid, keeping one glyph at each location.

    When `weights` is provided, also returns the transported per-cell SDF weight map.
    """
    rows, cols = glyphs.shape
    y, x = np.mgrid[:rows, :cols]
    target_y, target_x = np.divmod(destinations, cols)
    moving_y = y + progress * (target_y - y)
    moving_x = x + progress * (target_x - x)
    top, left = np.floor(moving_y).astype(int), np.floor(moving_x).astype(int)
    dy, dx = moving_y - top, moving_x - left
    coverage = masks.mean((1, 2))[glyphs].ravel()
    visibility = coverage * np.linalg.norm(colors.reshape(-1, 3) - background, axis=1)
    locations, particles, splat_weights = [], [], []
    for oy, ox, weight in ((0, 0, (1 - dy) * (1 - dx)), (0, 1, (1 - dy) * dx),
                           (1, 0, dy * (1 - dx)), (1, 1, dy * dx)):
        ty, tx = top + oy, left + ox
        valid = (ty >= 0) & (ty < rows) & (tx >= 0) & (tx < cols) & (weight > 0)
        locations.append((ty[valid] * cols + tx[valid]))
        particles.append(np.flatnonzero(valid))
        splat_weights.append(weight[valid])
    locations = np.concatenate(locations)
    particles = np.concatenate(particles)
    splat_weights = np.concatenate(splat_weights)
    priority = splat_weights * (visibility[particles] + 1e-6)
    order = np.lexsort((particles, priority, locations))
    last = np.r_[locations[order][1:] != locations[order][:-1], True]
    winners = order[last]
    occupied = locations[winners]
    selected = particles[winners]
    result_glyphs = np.full(glyphs.size, np.argmin(masks.mean((1, 2))), dtype=glyphs.dtype)
    result_colors = np.broadcast_to(background, (glyphs.size, 3)).copy()
    result_glyphs[occupied] = glyphs.ravel()[selected]
    result_colors[occupied] = (background + splat_weights[winners, None]
                               * (colors.reshape(-1, 3)[selected] - background))
    if weights is None:
        return result_glyphs.reshape(glyphs.shape), result_colors.reshape(*glyphs.shape, 3)
    result_weights = np.zeros(glyphs.size, dtype=np.float32)
    result_weights[occupied] = weights.ravel()[selected]
    return (result_glyphs.reshape(glyphs.shape),
            result_colors.reshape(*glyphs.shape, 3),
            result_weights.reshape(glyphs.shape))


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
        y_edges = np.linspace(0, source.shape[0], rows + 1).astype(int)
        x_edges = np.linspace(0, source.shape[1], cols + 1).astype(int)
        integral = np.pad(salience.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        top, bottom = y_edges[:-1, None], y_edges[1:, None]
        left, right = x_edges[None, :-1], x_edges[None, 1:]
        area = (bottom - top) * (right - left)
        sampled = ((integral[bottom, right] - integral[top, right]
                    - integral[bottom, left] + integral[top, left])
                   / np.maximum(area, 1))
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
    from .tone import quantize_playback_lab

    return quantize_playback_lab(colors)
