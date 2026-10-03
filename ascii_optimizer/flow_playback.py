"""Offline cyclic glyph fitting for a temporally closed fluid trajectory."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .flow import simulate_flow_loop, prepare_glyph_flow, glyph_flow_costs
from .optimizer import Grid


def select_cyclic_glyphs(unary, masks, strength, entropy_strength=0, sweeps=8):
    """Refine the full bank around a circle using actual raster distances.

    Source candidate fits supply the local unary objective. Full-frame image
    losses remain in subsequent perceptual fixed-glyph color fitting.
    """
    frames, cells, bank_size = unary.shape
    bank = masks.reshape(bank_size, -1).astype(np.float32)
    distance = np.maximum(0, (bank * bank).mean(1)[:, None] + (bank * bank).mean(1)[None]
                          - 2 * (bank @ bank.T) / bank.shape[1])
    labels = np.argmin(unary, axis=2).astype(np.uint8)
    cell_ids = np.arange(cells)

    def entropy(choices):
        counts = np.bincount(choices, minlength=bank_size) / cells
        return float(entropy_strength * np.sum(counts * np.log(np.maximum(counts, 1e-12))))

    def frame_cost(index, choices):
        neighbors = (labels[(index - 1) % frames], labels[(index + 1) % frames])
        value = np.mean(unary[index, cell_ids, choices] + strength * sum(
                        distance[choices, reference] for reference in neighbors))
        return float(value) + entropy(choices)

    def objective():
        return float(sum(np.mean(unary[i, cell_ids, labels[i]] + strength *
                         distance[labels[i], labels[(i + 1) % frames]]) + entropy(labels[i])
                         for i in range(frames)))

    initial_objective = objective()
    for sweep in range(sweeps):
        changed = 0
        # Moving the starting phase avoids privileging phase zero's history.
        origin = (sweep * frames // 3) % frames
        direction = 1 if sweep % 2 == 0 else -1
        order = ((origin + direction * i) % frames for i in range(frames))
        for index in order:
            scores = np.array(unary[index], copy=True)
            for neighbor in ((index - 1) % frames, (index + 1) % frames):
                scores += strength * distance[:, labels[neighbor]].T
            if entropy_strength:
                counts = np.bincount(labels[index], minlength=bank_size)
                scores += entropy_strength * np.log((counts + 1) / cells)[None]
            scores[cell_ids, labels[index]] -= 1e-7
            proposed = np.argmin(scores, axis=1).astype(np.uint8)
            if frame_cost(index, proposed) <= frame_cost(index, labels[index]) + 1e-8:
                changed += int(np.count_nonzero(proposed != labels[index]))
                labels[index] = proposed
        if not changed:
            break
    return labels, {"layout_objective_initial": initial_objective,
                    "layout_objective_final": objective(), "layout_sweeps": sweep + 1}


def render_flow_clip(source, masks, *, seed, eddies, scale, device, frames,
                     steps, frame_steps, temporal_strength, solve_frame, fit_colors,
                     flow_strength=.4, glyph_entropy_strength=.02, on_progress=None):
    """Cyclic layout refinement followed by source-referenced color fitting.

    Repeated short joint Adam solves introduced glyph chatter in the pilot.
    Use cached full-bank candidate costs, a circular raster objective, and
    equal color-fitting budgets at every phase instead.
    """
    started = time.perf_counter()
    loop = simulate_flow_loop(source, masks.shape[1:], seed=seed, eddies=eddies,
                              scale=scale, device=device, frames=frames,
                              on_progress=(lambda f, stage: on_progress(.15*f, stage))
                              if on_progress else None)
    atlas = torch.as_tensor(masks, device=device)
    descriptors = prepare_glyph_flow(atlas)
    context = {}

    def costs(index):
        field = loop.fields[index]
        field = replace(field, velocity=field.velocity.to(device), confidence=field.confidence.to(device))
        return glyph_flow_costs(field, atlas, descriptors)[0]

    prepared = solve_frame(costs(0), steps, context)
    rows, cols = prepared.glyph.shape
    cells = rows * cols
    with tempfile.TemporaryDirectory(prefix="ascii-glyph-loop-") as temporary:
        unary = np.memmap(Path(temporary) / "costs.dat", mode="w+", dtype=np.float32,
                          shape=(frames, cells, len(masks)))
        for index in range(frames):
            unary[index] = context["image_unary"] + flow_strength * costs(index).cpu().numpy()
            if on_progress:
                on_progress(.15 + .15*(index + 1)/frames, f"Matching flow frame {index + 1}/{frames}")
        labels, layout_diagnostics = select_cyclic_glyphs(unary, masks, temporal_strength, glyph_entropy_strength)
        del unary
    colors = context["candidate_colors"]
    grids = [Grid(labels[index].reshape(rows, cols),
                  colors[np.arange(cells), labels[index]].reshape(rows, cols, 3), prepared.background.copy())
             for index in range(frames)]
    # Equal analytic starts/budgets avoid a color walk around the cycle. Both
    # neighbors, including last/first, constrain displayed per-cell tone.
    for pass_index, order in enumerate((range(frames), range(frames - 1, -1, -1))):
        for ordinal, index in enumerate(order):
            neighbors = (grids[(index - 1) % frames], grids[(index + 1) % frames])
            grids[index] = fit_colors(grids[index].glyph, neighbors, context)
            if on_progress:
                on_progress(.3 + .7*(pass_index*frames + ordinal + 1)/(2*frames),
                            f"Fitting flow colors {ordinal + 1}/{frames} (pass {pass_index + 1}/2)")
    glyph_changes = [float((grid.glyph != grids[(i + 1) % frames].glyph).mean())
                     for i, grid in enumerate(grids)]
    color_changes = [float(np.sqrt(np.mean((grid.color - grids[(i + 1) % frames].color)**2)))
                     for i, grid in enumerate(grids)]
    metadata = dict(loop.diagnostics, **layout_diagnostics, seconds=time.perf_counter() - started,
                    fitting="cyclic-full-bank-layout-and-perceptual-colors", temporal_strength=temporal_strength,
                    glyph_change_mean=float(np.mean(glyph_changes)), glyph_change_seam=glyph_changes[-1],
                    color_step_mean=float(np.mean(color_changes)), color_step_seam=color_changes[-1],
                    glyph_changes=glyph_changes, color_changes=color_changes)
    return grids, metadata
