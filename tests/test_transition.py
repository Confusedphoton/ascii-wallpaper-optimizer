import numpy as np
import pytest
import subprocess

from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS
from ascii_optimizer.transition import (earth_mover_destinations, glyph_paths, path_glyphs,
                                        phase_offsets, transport_cells)


def test_salience_pools_edges_between_cell_centers():
    flat = np.zeros((32, 32, 3), dtype=np.float32)
    edge = flat.copy()
    edge[:, 16:] = 1
    flat_phases = phase_offsets((4, 4), "salience", 3, flat)
    edge_phases = phase_offsets((4, 4), "salience", 3, edge)
    assert (edge_phases[:, 2] < flat_phases[:, 2]).all()
    np.testing.assert_array_equal(edge_phases[:, :2], flat_phases[:, :2])
    np.testing.assert_array_equal(edge_phases[:, 3], flat_phases[:, 3])


def test_real_font_paths_are_reachable_and_walk_long_paths_without_skipping():
    try:
        font = resolve_font("Monospace")
    except (OSError, ValueError, subprocess.SubprocessError):
        pytest.skip("No installed monospace font is available")
    masks = glyph_masks(font, list(DEFAULT_CHARS), (8, 16))
    paths = glyph_paths(masks)
    for start_index, row in enumerate(paths):
        for end_index, path in enumerate(row):
            assert path[0] == start_index and path[-1] == end_index
            assert len(path) == len(set(path))
    longest = max((path for row in paths for path in row), key=len)
    assert len(longest) > 2
    start = np.array([[longest[0]]], dtype=np.uint8)
    end = np.array([[longest[-1]]], dtype=np.uint8)
    previous = start
    frame_count = len(longest) + 3
    visited = [longest[0]]
    for frame in range(1, frame_count):
        # Delay the local phase, then jump it ahead. Remaining-frame scheduling
        # must still make each adjacent move and reach the endpoint on time.
        local = np.array([[0 if frame < frame_count // 2 else 1]], dtype=np.float32)
        current = path_glyphs(start, end, local, paths, previous,
                             remaining_frames=frame_count - 1 - frame)
        previous_step = longest.index(int(previous[0, 0]))
        current_step = longest.index(int(current[0, 0]))
        assert current_step - previous_step in (0, 1)
        visited.append(int(current[0, 0]))
        previous = current
    assert visited[-1] == longest[-1]
    assert list(dict.fromkeys(visited)) == longest


def test_transport_assignment_minimizes_cost_instead_of_greedily_matching():
    from itertools import permutations
    from ascii_optimizer.transition import _minimum_assignment

    rng = np.random.default_rng(42)
    for count in range(1, 7):
        cost = rng.uniform(0, 10, (count, count))
        assignment = _minimum_assignment(cost)
        expected = min(cost[np.arange(count), order].sum() for order in permutations(range(count)))
        assert cost[np.arange(count), assignment].sum() == pytest.approx(expected)
    # Equal-size region constraints can forbid individual links.
    np.testing.assert_array_equal(_minimum_assignment(np.array([[np.inf, 2], [1, np.inf]])), [1, 0])


@pytest.mark.parametrize("shape", [(1, 9), (9, 1)])
def test_earth_mover_transports_visible_ink_and_keeps_exact_endpoints(shape):
    masks = np.array([[[0]], [[1]]], np.float32)
    start = np.zeros(shape, np.uint8)
    end = start.copy()
    start.flat[1], end.flat[7] = 1, 1
    color = np.broadcast_to(np.array([0.8, 0, 0], np.float32), (*shape, 3)).copy()
    background = np.zeros(3, np.float32)
    destinations = earth_mover_destinations(start, end, color, color, background, background, masks)
    assert destinations.flat[1] == 7
    np.testing.assert_array_equal(np.sort(destinations.ravel()), np.arange(start.size))
    glyphs, colors = transport_cells(start, color, background, destinations,
                                    np.full(shape, 0.5), masks)
    assert glyphs.flat[4] == 1
    assert glyphs.flat[1] == glyphs.flat[7] == 0
    np.testing.assert_allclose(colors.reshape(-1, 3)[4], color.reshape(-1, 3)[1])
    for progress, expected in ((0, start), (1, end)):
        glyphs, colors = transport_cells(start, color, background, destinations,
                                        np.full(shape, progress), masks)
        np.testing.assert_array_equal(glyphs, expected)
        np.testing.assert_array_equal(colors, color)


def test_earth_mover_matches_colors_even_when_glyphs_are_the_same():
    masks = np.ones((1, 1, 1), np.float32)
    glyphs = np.zeros((1, 2), np.uint8)
    source = np.array([[[0.5, 0.3, 0], [0.5, -0.3, 0]]], np.float32)
    target = source[:, ::-1].copy()
    background = np.zeros(3, np.float32)
    mapping = earth_mover_destinations(glyphs, glyphs, source, target, background, background, masks)
    np.testing.assert_array_equal(mapping, [[1, 0]])


@pytest.mark.parametrize("shape", [(11, 13), (1, 137), (137, 1)])
def test_hierarchical_transport_is_bounded_bijective_and_stationary_for_identical_images(monkeypatch, shape):
    from ascii_optimizer import transition

    original = transition._transport_cost
    calls = []

    def bounded(source, target):
        calls.append((len(source), len(target)))
        assert max(len(source), len(target)) <= 64
        return original(source, target)

    monkeypatch.setattr(transition, "_transport_cost", bounded)
    rng = np.random.default_rng(10)
    masks = np.array([[[0]], [[1]]], np.float32)
    start = rng.integers(0, 2, shape, dtype=np.uint8)
    end = rng.integers(0, 2, shape, dtype=np.uint8)
    colors = rng.uniform(0.2, 0.8, (*shape, 3)).astype(np.float32)
    background = np.zeros(3, np.float32)
    identity = earth_mover_destinations(start, start, colors, colors, background, background, masks)
    np.testing.assert_array_equal(identity.ravel(), np.arange(start.size))
    mapping = earth_mover_destinations(start, end, colors, colors, background, background, masks)
    np.testing.assert_array_equal(np.sort(mapping.ravel()), np.arange(start.size))
    np.testing.assert_array_equal(mapping, earth_mover_destinations(
        start, end, colors, colors, background, background, masks))
    assert len(calls) > 1


def test_transport_collisions_prefer_visible_ink_and_handle_fonts_without_a_space():
    masks = np.array([[[0.1]], [[1]]], np.float32)
    glyphs = np.array([[0, 1, 0]], np.uint8)
    colors = np.array([[[0.1, 0, 0], [0.8, 0, 0], [0.1, 0, 0]]], np.float32)
    background = np.zeros(3, np.float32)
    # All contributions collide at the center at halfway progress.
    destinations = np.array([[2, 1, 0]])
    output, ink = transport_cells(glyphs, colors, background, destinations,
                                 np.full(glyphs.shape, 0.5), masks)
    np.testing.assert_array_equal(output, [[0, 1, 0]])
    np.testing.assert_array_equal(ink[0, 1], colors[0, 1])
    np.testing.assert_array_equal(ink[0, [0, 2]], np.zeros((2, 3)))


def test_transport_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="same two-dimensional grid"):
        earth_mover_destinations(np.zeros((1, 2)), np.zeros((2, 1)),
                                 None, None, None, None, None)
