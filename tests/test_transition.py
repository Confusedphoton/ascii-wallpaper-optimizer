import numpy as np
import pytest
import subprocess

from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.defaults import DEFAULT_CHARS
from ascii_optimizer.transition import glyph_paths, path_glyphs, phase_offsets


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
