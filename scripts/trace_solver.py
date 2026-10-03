"""Log solver losses, raw logits, stage grids, and optional sampled GPU profiles.

    .venv/bin/python -m ascii_optimizer.launcher scripts/trace_solver.py --device cuda

Plots require the optional diagnostics dependencies. Profiling and logging are
separate from ordinary timed runs; do not compare profiler/trace timings with
production solve times. All artifacts are written under --output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import experiment_utils as base
from ascii_optimizer import optimizer
from ascii_optimizer.assets import glyph_masks, resolve_font
from ascii_optimizer.diagnostics import SolverTrace
from ascii_optimizer.tone import srgb_to_oklab
from compare_error_movement import SOURCES as BASE_SOURCES, focus_metrics


SOURCES = (*BASE_SOURCES, ("viviana", Path("screenshots/143381492_p0-Viviana.jpg"),
                          (330, 270, 530, 440)))


STAGES = ("initial", "solver_final", "after_dither", "pre_refinement",
          "refinement_sweep", "post_refinement", "final")


def entropy_metrics(glyph, masks, color, selected):
    counts = np.bincount(glyph[selected].astype(int), minlength=len(masks))
    probability = counts[counts > 0] / max(1, counts.sum())
    return {"cells": int(selected.sum()),
            "glyph_entropy_bits": float(-(probability * np.log2(probability)).sum()),
            "mean_glyph_coverage": float(masks[glyph[selected]].mean()) if selected.any() else None,
            "mean_ink_oklab_lightness": float(color[selected, 0].mean()) if selected.any() else None,
            "glyph_counts": counts.tolist()}


def regions(reference, shape, cell):
    rows, cols = shape
    width, height = cell
    lab = srgb_to_oklab(reference)
    lab = np.pad(lab, ((0, rows * height - lab.shape[0]), (0, cols * width - lab.shape[1]), (0, 0)), mode="edge")
    cells = lab.reshape(rows, height, cols, width, 3).transpose(0, 2, 1, 3, 4).reshape(rows, cols, -1, 3)
    variance = cells.var(2).sum(2)
    return variance <= np.quantile(variance, .25), variance


def plot_trace(directory, report, stage_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = report["records"]
    time_key = "gpu_timeline_seconds" if records[0]["gpu_timeline_seconds"] is not None else "cpu_seconds"
    time = [row[time_key] for row in records]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    loss = [row["loss"] for row in records]
    axes[0, 0].plot(time, loss, color="tab:blue", label="Evaluated discrete state", linewidth=1)
    axes[0, 0].plot(time, np.minimum.accumulate(loss), color="tab:orange", label="Best evaluated loss", linewidth=2)
    pre = next(row for row in records if row["stage"] == "pre_refinement")
    post = next(row for row in records if row["stage"] == "post_refinement")
    for row, label in ((pre, "Refinement begins"), (post, "Final polish begins")):
        axes[0, 0].axvline(row[time_key], color=".55", linestyle="--", linewidth=1, label=label)
    axes[0, 0].set(xlabel="Elapsed diagnostic timeline (s)", ylabel="Weighted objective (dimensionless)")
    axes[0, 0].legend(fontsize=8)
    components = sorted({key for row in records for key in row["components"]})
    for name in components:
        axes[0, 1].plot(time, [row["components"].get(name, 0.) for row in records], label=name, linewidth=1)
    axes[0, 1].set(xlabel="Elapsed diagnostic timeline (s)", ylabel="Weighted loss contribution (dimensionless)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    logged = [row for row in records if row["stage"] in ("adam_step", "solver_final")]
    steps = [row["step"] for row in logged]
    axes[1, 0].plot(steps, [row["logit_summary"]["mean_softmax_entropy_bits"] for row in logged], label="Scheduled temperature")
    axes[1, 0].plot(steps, [row["logit_summary"]["mean_softmax_entropy_bits_at_temperature_1"] for row in logged], label="Fixed temperature = 1")
    axes[1, 0].set(xlabel="Adam updates completed", ylabel="Mean per-cell softmax entropy (bits)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(steps, [row["logit_summary"]["winner_margin_percentiles"][2] for row in logged], label="Median")
    axes[1, 1].fill_between(steps, [row["logit_summary"]["winner_margin_percentiles"][1] for row in logged],
                            [row["logit_summary"]["winner_margin_percentiles"][3] for row in logged], alpha=.2, label="10th–90th percentile across cells")
    axes[1, 1].set(xlabel="Adam updates completed", ylabel="Top-two raw logit gap (logit units)")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(directory / "loss_and_logits.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for kind, ax in zip(("pixel", "edge"), axes[0]):
        for suffix, label in (("", "Scheduled temperature"), ("_temperature_1", "Fixed temperature = 1")):
            ax.plot(steps, [row["visual_summary"][f"{kind}_variance{suffix}"] for row in logged], label=label)
        ax.set(xlabel="Adam updates completed", ylabel=f"Encoded OKLab {kind} variance (squared OKLab)")
        ax.legend(fontsize=8)
    sampled = [row for row in logged if "weighted_gradient_rms" in row]
    for parameter, ax in zip(("logits", "ink_rgb"), axes[1]):
        for name in sampled[0]["weighted_gradient_rms"] if sampled else []:
            ax.plot([row["step"] for row in sampled],
                    [row["weighted_gradient_rms"][name][parameter] for row in sampled], label=name)
        ax.set(xlabel="Adam updates completed", ylabel=f"Weighted derivative RMS ({parameter} parameter units)")
        values = [v[parameter] for row in sampled for v in row["weighted_gradient_rms"].values()]
        if values and max(values) > 100 * min((v for v in values if v > 0), default=1):
            ax.set_yscale("symlog", linthresh=min(v for v in values if v > 0))
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(directory / "visual_and_gradients.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for name in components:
        ax.plot(time, [row["loss_terms"].get(name, {}).get("raw_error", 0.) for row in records], label=name)
    ax.set(xlabel="Elapsed diagnostic timeline (s)", ylabel="Raw term error (before coefficient; units vary)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(directory / "raw_losses.png", dpi=160)
    plt.close(fig)

    selected = [(name, value) for name, value in stage_metrics.items() if name != "after_dither"]
    labels = [name.replace("refinement_sweep_", "sweep ") for name, _ in selected]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, key, label in zip(axes, ("glyph_entropy_bits", "mean_glyph_coverage", "mean_ink_oklab_lightness"),
                              ("Glyph histogram entropy (bits)", "Mean coverage (fraction)", "Mean ink lightness (OKLab)")):
        for region, legend in (("all", "Whole frame"), ("uniform", "Lowest-variance source cells")):
            ax.plot(labels, [value["regions"][region][key] for _, value in selected], marker="o", label=legend)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=65, labelsize=7)
    axes[0].legend(fontsize=8)
    fig.savefig(directory / "stage_diversity.png", dpi=160)
    plt.close(fig)

    data = np.load(directory / "logits.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for index, label in ((0, "Initial"), (len(data["logits"]) - 1, "After Adam")):
        logits = data["logits"][index]
        values = np.sort(logits, 1)
        gap = values[:, -1] - values[:, -min(2, values.shape[1])]
        axes[0].hist(gap, bins=40, histtype="step", label=label, density=True)
        axes[1].hist((logits - logits.max(1, keepdims=True)).ravel(), bins=50, histtype="step", label=label, density=True)
    axes[0].set(xlabel="Top-two raw logit gap (logit units)", ylabel="Probability density")
    axes[1].set(xlabel="Logit relative to cell winner (logit units)", ylabel="Probability density")
    for ax in axes:
        ax.legend()
    fig.savefig(directory / "logit_distributions.png", dpi=160)
    plt.close(fig)


def summarize(directory, report, reference, masks, chars, crop, cell, plots):
    data = np.load(directory / "grids.npz")
    uniform, variance = regions(reference, data["glyph"].shape[1:], cell)
    np.savez_compressed(directory / "regions.npz", uniform_cells=uniform, source_variance=variance)
    before = next(row for row in report["records"] if row["stage"] == "pre_refinement")
    before_glyph = data["glyph"][before["grid_index"]]
    stages, adam_stages = {}, {}
    panels, intermediate = [("Source", base.image(reference))], [("Source", base.image(reference))]
    for row in report["records"]:
        selected = row["stage"] in STAGES
        adam = row["stage"] == "adam_step" and row["step"] % 6 == 0
        if not selected and not adam:
            continue
        name = row["stage"] + (f"_{row['sweep'] + 1}" if "sweep" in row else "")
        if adam:
            name += f"_{row['step']}"
        index = row["grid_index"]
        grid = optimizer.Grid(data["glyph"][index], data["color"][index], data["background"])
        rendered = base.preview(grid, masks, (reference.shape[1], reference.shape[0]))
        shown = base.image(rendered)
        shown.save(directory / f"{name}.png")
        metrics = {"loss": row["loss"], "components": row["components"], "loss_terms": row["loss_terms"],
                            **base.image_metrics(reference, rendered),
                            **base.glyph_metrics(grid, masks, chars, before_glyph),
                            "gradient_rmse_x100": base.gradient_rmse(reference, rendered),
                            "focus": focus_metrics(reference, rendered, crop),
                            "regions": {"all": entropy_metrics(grid.glyph, masks, grid.color, np.ones_like(uniform)),
                                        "uniform": entropy_metrics(grid.glyph, masks, grid.color, uniform)}}
        if selected:
            stages[name] = metrics
            panels.append((name, shown))
        if adam:
            adam_stages[name] = metrics
            if row["step"]:
                intermediate.append((f"Raw Adam: {row['step']} updates", shown))
        if row["stage"] == "initial":
            intermediate.append(("Initialization: 0 updates", shown))
        elif row["stage"] == "solver_final":
            adam_stages[name] = metrics
            intermediate.append((f"Raw Adam: {row['step']} updates", shown))
        elif row["stage"] == "final":
            intermediate.append(("Returned frame after dither/refinement/polish", shown))
    base.contact_sheet(panels, directory / "stage_comparison.png", columns=3)
    base.contact_sheet(panels, directory / "focus_comparison.png", crop=crop, columns=3, zoom=3)
    base.contact_sheet(intermediate, directory / "adam_comparison.png", columns=3)
    base.contact_sheet(intermediate, directory / "adam_focus.png", crop=crop, columns=3, zoom=3)
    if plots:
        plot_trace(directory, report, stages)
    times = defaultdict(lambda: {"calls": 0, "cpu_seconds": 0., "gpu_timeline_seconds": 0.})
    for span in report["spans"]:
        totals = times[span["name"]]
        totals["calls"] += 1
        totals["cpu_seconds"] += span["cpu_seconds"]
        totals["gpu_timeline_seconds"] += span["gpu_timeline_seconds"] or 0.
    raw = np.load(directory / "logits.npz")
    visual = np.load(directory / "visual.npz")
    logit_regions = {}
    for row in report["records"]:
        if row["stage"] not in ("initial", "solver_final"):
            continue
        values = raw["logits"][row["logit_index"]]
        centered = (values - values.max(1, keepdims=True)).astype(np.float64)
        def entropy(temperature):
            probability = np.exp(centered / temperature)
            probability /= probability.sum(1, keepdims=True)
            return -(probability * np.log2(np.maximum(probability, 1e-300))).sum(1)
        scheduled, fixed = entropy(row["temperature"]), entropy(1.)
        winners = np.sort(values, axis=1)[:, -min(2, values.shape[1]):]
        margin = winners[:, -1] - winners[:, 0]
        noise_changes = (values + raw["selection_noise"]).argmax(1) != values.argmax(1)
        logit_regions[row["stage"]] = {}
        for region, selected in (("all", np.ones(values.shape[0], bool)), ("uniform", uniform.ravel())):
            logit_regions[row["stage"]][region] = {
                "mean_softmax_entropy_bits": float(scheduled[selected].mean()),
                "mean_softmax_entropy_bits_at_temperature_1": float(fixed[selected].mean()),
                "median_winner_margin": float(np.median(margin[selected])),
                "noise_winner_change_fraction": float(noise_changes[selected].mean()) if row["stage"] == "solver_final" else None,
                "visual": {key: float(visual[key][row["logit_index"]][selected].mean()) for key in visual.files
                           if "variance" in key or "distance" in key},
            }
    return {"stages": stages, "adam_stages": adam_stages, "phase_intervals": dict(times), "logit_regions": logit_regions,
            "acceptance_counters": report.get("counters", {})}


def ordinary_run(target, masks, kwargs, *, steps, sweeps, polish_steps, repeats):
    """Measure complete uninstrumented solves; score the last deterministic grid."""
    measurements = []
    for _ in range(repeats):
        objectives = []
        original = optimizer._FrameObjective
        def capture(*args, **kw):
            value = original(*args, **kw)
            objectives.append(value)
            return value
        with patch.object(optimizer, "_FrameObjective", capture):
            grid = base.solve_style(target, masks, **{**kwargs, "steps": steps,
                                   "refinement_sweeps": sweeps, "polish_steps": polish_steps})
        measurements.append(grid.experiment["seconds"])
    device = kwargs["device"]
    rendered = base.preview(grid, masks, (target.shape[1], target.shape[0]))
    frame = torch.as_tensor(optimizer.render_grid(grid, masks, device), device=device).permute(2, 0, 1)[None]
    components = {}
    with torch.no_grad():
        loss = objectives[0](frame, components=components).double()
        counts = torch.as_tensor(np.bincount(grid.glyph.ravel(), minlength=len(masks)), dtype=torch.float32, device=device)
        components["glyph_entropy"] = kwargs.get("glyph_entropy_strength", base.DEFAULT_GLYPH_ENTROPY_STRENGTH) * optimizer._negative_glyph_entropy(counts / grid.glyph.size)
        ink = torch.as_tensor(grid.color, device=device).reshape(-1, 3)
        presence = torch.as_tensor(masks, device=device).flatten(1).gt(0).any(1)[torch.as_tensor(grid.glyph.astype(int), device=device).flatten()]
        components["ink_reward"] = -kwargs.get("glyph_luminance_bonus", base.DEFAULT_GLYPH_LUMINANCE_BONUS) * optimizer._ink_lightness_reward(ink, presence).mean()
        loss += components["glyph_entropy"].double() + components["ink_reward"].double()
    source_lab, output_lab = srgb_to_oklab(target), srgb_to_oklab(rendered)
    source_saturation = np.linalg.norm(source_lab[..., 1:], axis=-1) / np.maximum(source_lab[..., 0], 1e-6)
    output_saturation = np.linalg.norm(output_lab[..., 1:], axis=-1) / np.maximum(output_lab[..., 0], 1e-6)
    valid = (source_lab[..., 0] > .05) & (output_lab[..., 0] > .05)
    saturation_difference = output_saturation[valid] - source_saturation[valid]
    result = {"steps": steps, "refinement_sweeps": sweeps, "polish_steps": polish_steps,
              "seconds": float(np.median(measurements)), "seconds_samples": measurements,
              "seconds_min": min(measurements), "seconds_max": max(measurements),
              "loss": float(loss), "components": {name: float(value) for name, value in components.items()},
              "gradient_rmse_x100": base.gradient_rmse(target, rendered),
              **base.image_metrics(target, rendered),
              "saturation_bias": float(saturation_difference.mean()) if valid.any() else None,
              "saturation_rmse": float(np.sqrt(np.mean(saturation_difference ** 2))) if valid.any() else None,
              "saturation_sample_fraction": float(valid.mean())}
    return result, grid, rendered


def matched_adam(target, masks, kwargs, baseline, budget, polish_steps, repeats):
    """Calibrate an Adam-only run to the measured refinement budget, within 10%."""
    intercept = baseline[0]["seconds"]
    slope = max(1e-5, (baseline[1]["seconds"] - intercept) / baseline[1]["steps"])
    steps = max(1, round((budget["seconds"] - intercept) / slope))
    attempts = []
    for _ in range(3):
        result, grid, rendered = ordinary_run(target, masks, kwargs, steps=steps, sweeps=0,
                                              polish_steps=polish_steps, repeats=repeats)
        attempts.append(result)
        ratio = result["seconds"] / budget["seconds"]
        if .9 <= ratio <= 1.1:
            break
        slope = max(1e-5, (result["seconds"] - intercept) / steps)
        steps = max(1, round((budget["seconds"] - intercept) / slope))
    result["target_seconds"] = budget["seconds"]
    result["target_refinement_sweeps"] = budget["refinement_sweeps"]
    result["time_ratio"] = result["seconds"] / budget["seconds"]
    result["within_10_percent"] = .9 <= result["time_ratio"] <= 1.1
    result["calibration_attempts"] = attempts[:-1]
    return result, grid, rendered


def sampled_profile(target, masks, kwargs, directory, stage):
    wait = 1 if stage == "adam" else kwargs["steps"] + 1
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(kwargs["device"]).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,
                               schedule=torch.profiler.schedule(wait=wait, warmup=1, active=2, repeat=1)) as profiler:
        base.solve_style(target, masks, **kwargs, on_progress=lambda *_: profiler.step())
    events = profiler.key_averages()
    (directory / f"profile_{stage}.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=35)
                                                     + "\n" + events.table(sort_by="self_cpu_time_total", row_limit=35))
    compact = [{"name": event.key, "calls": event.count, "self_cpu_us": event.self_cpu_time_total,
                "self_device_us": event.self_device_time_total} for event in events]
    (directory / f"profile_{stage}.json").write_text(json.dumps(compact, indent=2) + "\n")


def plot_ablation(directory, refinement, matched):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for runs, label, color in ((refinement, "Fixed optimizer steps + refinement", "tab:blue"),
                               (matched, "Optimizer only, matched time", "tab:orange")):
        if not runs:
            continue
        times = np.array([run["seconds"] for run in runs])
        error = np.array([[run["seconds"] - run["seconds_min"] for run in runs],
                          [run["seconds_max"] - run["seconds"] for run in runs]])
        for ax, key, ylabel in zip(axes, ("loss", "gradient_rmse_x100"),
                                  ("Weighted objective", r"Displayed edge RMSE ($\times100$)")):
            ax.errorbar(times, [run[key] for run in runs], xerr=error, marker="o", label=label,
                        color=color, capsize=3, linewidth=1)
            ax.set(xlabel="Ordinary solve time, median and observed range (s)", ylabel=ylabel)
        for run in runs:
            axes[0].annotate(f"{run['steps']} steps, {run['refinement_sweeps']} sweeps",
                             (run["seconds"], run["loss"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].legend(fontsize=8)
    fig.savefig(directory / "compute_comparison.png", dpi=160)
    plt.close(fig)


def write_review(output, report):
    lines = ["# Solver diagnostics", "",
             "All loss curves score actual discrete glyphs and RGB8 ink. Raw errors, coefficients, and weighted contributions are recorded separately; rewards are negative.",
             "Raw logits retain every cell and candidate, with candidate-to-glyph identities and temperatures in `logits.npz`.",
             "`grids.npz` contains every evaluated snapshot; `trace.json` maps array indices to stages, losses, and timings.",
             "Logit entropy at fixed temperature separates learned concentration from temperature annealing.",
             "Visual variance measures candidate raster/edge disagreement in encoded OKLab at current ink; it includes glyph geometry, but excludes display gamut mapping.",
             "Replacement archives measure actual single-cell whole-objective changes at fixed current RGB8 ink, including neighboring pixels and affected pooled regions.",
             "Uniform-region metrics use the lowest-variance quarter of source cells, held fixed across stages.", "",
             "Timings can be affected by other GPU jobs. Instrumented timelines include diagnostic overhead; ordinary solves are recorded separately.", "",
             "| Image | Stage | Objective | Glyph entropy, bits | Coverage | Ink lightness | Uniform entropy, bits |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for image, item in report["images"].items():
        for stage, metrics in item["analysis"]["stages"].items():
            lines.append(f"| {image} | {stage} | {metrics['loss']:.6f} | {metrics['glyph_entropy_bits']:.3f} | "
                         f"{metrics['mean_glyph_coverage']:.3f} | {metrics['mean_ink_oklab_lightness']:.3f} | "
                         f"{metrics['regions']['uniform']['glyph_entropy_bits']:.3f} |")
    for name, item in report["images"].items():
        lines += ["", f"## {name}", "", f"![Loss and logit curves]({name}/loss_and_logits.png)", "",
                  f"[Stage previews]({name}/stage_comparison.png) · [Focus previews]({name}/focus_comparison.png) · "
                  f"[Adam intermediates]({name}/adam_comparison.png)", "",
                  f"![Stage diversity]({name}/stage_diversity.png)", "",
                  f"![Logit distributions]({name}/logit_distributions.png)", "",
                  f"![Visual variance and weighted gradient RMS]({name}/visual_and_gradients.png)", "",
                  f"![Raw losses before coefficients]({name}/raw_losses.png)", "",
                  f"[Trace JSON]({name}/trace.json) · [Raw logits]({name}/logits.npz) · [All grid snapshots]({name}/grids.npz)", "",
                  "Raw Adam snapshots below are from one trajectory, before dithering, refinement, and final polish. Step zero is initialization. These are distinct from returned frames at independently configured step budgets; the temperature schedules differ.", "",
                  "| Completed Adam updates | Raw objective | Glyph entropy, bits | Uniform entropy, bits |",
                  "|---:|---:|---:|---:|"]
        for stage, metrics in item["analysis"].get("adam_stages", {}).items():
            step = report["steps"] if stage == "solver_final" else int(stage.removeprefix("adam_step_"))
            lines.append(f"| {step} | {metrics['loss']:.6f} | {metrics['glyph_entropy_bits']:.3f} | "
                         f"{metrics['regions']['uniform']['glyph_entropy_bits']:.3f} |")
        lines += ["", f"[Raw Adam focus comparison]({name}/adam_focus.png)", "",
                  "| Step budget | Ordinary solve, s | Final objective | Edge RMSE ×100 |", "|---:|---:|---:|---:|"]
        for run in item["budgets"]:
            lines.append(f"| {run['steps']} | {run['seconds']:.3f} | {run['loss']:.6f} | {run['gradient_rmse_x100']:.3f} |")
        lines += ["", "Each step budget has its own temperature schedule; refinement and final polish budgets are fixed independently. Times are medians of repeated uninstrumented solves."]
        lines += ["", f"[Step-budget preview comparison]({name}/budget_comparison.png)"]
        lines += ["", "| Refinement sweeps | Median time, s | Observed range, s | Objective | Edge RMSE ×100 | Saturation bias |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for run in item["refinement_budgets"]:
            bias = f"{run['saturation_bias']:.5f}" if run["saturation_bias"] is not None else "n/a"
            lines.append(f"| {run['refinement_sweeps']} | {run['seconds']:.3f} | {run['seconds_min']:.3f}–{run['seconds_max']:.3f} | "
                         f"{run['loss']:.6f} | {run['gradient_rmse_x100']:.3f} | {bias} |")
        lines += ["", "Saturation bias is output minus source C/L where both have L > 0.05; black/near-black pixels are excluded from this ratio diagnostic, not from the loss.",
                  "", f"[Refinement previews]({name}/refinement_comparison.png) · [Refinement detail]({name}/refinement_focus.png)",
                  "", "| Matched to sweeps | Adam-only steps | Median time, s | Time / target | Within 10% | Objective | Edge RMSE ×100 |",
                  "|---:|---:|---:|---:|---|---:|---:|"]
        for run in item["matched_time"]:
            lines.append(f"| {run['target_refinement_sweeps']} | {run['steps']} | {run['seconds']:.3f} | {run['time_ratio']:.3f} | "
                         f"{run['within_10_percent']} | {run['loss']:.6f} | {run['gradient_rmse_x100']:.3f} |")
        if item["matched_time"]:
            lines += ["", f"[Matched-time previews]({name}/matched_comparison.png) · [Matched-time detail]({name}/matched_focus.png)",
                      "", f"![Compute comparison; timing bars show observed min/max]({name}/compute_comparison.png)"]
        final = item["analysis"]["stages"]["final"]
        lines += ["", "| Final loss term | Raw error | Coefficient | Weighted contribution |", "|---|---:|---:|---:|"]
        for term, values in final["loss_terms"].items():
            lines.append(f"| {term} | {values['raw_error']:.6g} | {values['coefficient']:.6g} | {values['weighted_contribution']:.6g} |")
    (output / "review.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rendered/solver_diagnostics"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--cell", default="12x24")
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--step-budgets", nargs="+", type=int, default=[0, 12, 24, 36, 72])
    parser.add_argument("--refinement-sweeps", type=int, default=2)
    parser.add_argument("--polish-steps", type=int, default=9)
    parser.add_argument("--refinement-budgets", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--no-matched-time", action="store_true")
    parser.add_argument("--images", nargs="+", choices=[source[0] for source in SOURCES])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dither", type=float, default=base.DITHER)
    parser.add_argument("--glyph-entropy-strength", type=float, default=base.DEFAULT_GLYPH_ENTROPY_STRENGTH)
    parser.add_argument("--profile", nargs="*", choices=["adam", "refinement"], default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        cell = tuple(map(int, args.cell.split("x")))
        assert len(cell) == 2 and min(cell) > 0
    except (ValueError, AssertionError):
        parser.error("cell must be positive WIDTHxHEIGHT")
    if args.width < 1 or args.steps < 0 or min(args.step_budgets) < 0 or not np.isfinite(args.dither) or args.dither < 0:
        parser.error("width must be positive and steps/budgets/dither nonnegative and finite")
    if min(args.refinement_budgets + [args.refinement_sweeps, args.polish_steps]) < 0 or args.timing_repeats < 1:
        parser.error("refinement/polish budgets must be nonnegative and timing repeats positive")
    if not np.isfinite(args.glyph_entropy_strength) or args.glyph_entropy_strength < 0:
        parser.error("glyph entropy strength must be nonnegative and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(optimizer.__file__).read_bytes()
    (args.output / "optimizer_after.py").write_bytes(source)
    base.reset_runs()
    report = {"source_hash": hashlib.sha256(source).hexdigest(), "steps": args.steps, "width": args.width,
              "cell": cell, "dither": args.dither, "glyphs": base.DEFAULT_CHARS,
              "glyph_entropy_strength": args.glyph_entropy_strength,
              "refinement_sweeps": args.refinement_sweeps, "polish_steps": args.polish_steps,
              "timing_repeats": args.timing_repeats, "images": {}}
    masks = glyph_masks(resolve_font("Monospace"), list(base.DEFAULT_CHARS), cell)
    for name, path, crop in SOURCES:
        if (args.images and name not in args.images) or (not args.images and name == "viviana"):
            continue
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        size = base.source_size(path, args.width)
        target = base.load_target(path, size, "fill", False)
        crop = tuple(round(coordinate * args.width / 960) for coordinate in crop)
        kwargs = dict(style="headroom", invert_ink=False, background="auto", seed=0, steps=args.steps,
                      perceptual_loss=False,
                      dither=args.dither, structure_dither=.06, device=args.device,
                      glyph_entropy_strength=args.glyph_entropy_strength,
                      refinement_sweeps=args.refinement_sweeps, polish_steps=args.polish_steps)
        print(f"{name}: warmup", flush=True)
        base.solve_style(target, masks, **kwargs)
        trace = SolverTrace()
        print(f"{name}: trace", flush=True)
        traced_grid = base.solve_style(target, masks, **kwargs, diagnostics=trace)
        trace.metadata.update(effective_objectives=traced_grid.experiment["effective_objectives"],
                              glyphs=base.DEFAULT_CHARS, source=str(path), optimizer_sha256=report["source_hash"])
        traced = trace.save(directory)
        print(f"{name}: rendering stage comparisons", flush=True)
        analysis = summarize(directory, traced, target, masks, base.DEFAULT_CHARS, crop, cell, not args.no_plots)
        budgets = []
        for steps in dict.fromkeys(args.step_budgets):
            print(f"{name}: ordinary {steps}-step solve", flush=True)
            result, grid, rendered = ordinary_run(target, masks, kwargs, steps=steps,
                sweeps=args.refinement_sweeps, polish_steps=args.polish_steps, repeats=args.timing_repeats)
            base.image(rendered).save(directory / f"budget_{steps}.png")
            result["matches_traced_output"] = bool(np.array_equal(grid.glyph, traced_grid.glyph) and
                np.array_equal(grid.color, traced_grid.color)) if steps == args.steps else None
            budgets.append(result)
        from PIL import Image
        budget_panels = [("Source", base.image(target))]
        for run in budgets:
            with Image.open(directory / f"budget_{run['steps']}.png") as image:
                budget_panels.append((f"{run['steps']} Adam + {args.refinement_sweeps} sweeps", image.copy()))
        base.contact_sheet(budget_panels, directory / "budget_comparison.png", columns=3)
        refinement, matched = [], []
        refinement_panels = [("Source", base.image(target))]
        for sweeps in dict.fromkeys(args.refinement_budgets):
            print(f"{name}: ordinary {args.steps} steps + {sweeps} sweeps", flush=True)
            result, grid, rendered = ordinary_run(target, masks, kwargs, steps=args.steps, sweeps=sweeps,
                                                 polish_steps=args.polish_steps, repeats=args.timing_repeats)
            refinement.append(result)
            base.image(rendered).save(directory / f"refinement_{sweeps}.png")
            refinement_panels.append((f"{args.steps} Adam + {sweeps} sweeps", base.image(rendered)))
        base.contact_sheet(refinement_panels, directory / "refinement_comparison.png", columns=3)
        base.contact_sheet(refinement_panels, directory / "refinement_focus.png", columns=3, crop=crop, zoom=3)
        if not args.no_matched_time:
            baseline = []
            for steps in (0, max(36, args.steps)):
                print(f"{name}: calibrating Adam-only timing at {steps} steps", flush=True)
                result, _, _ = ordinary_run(target, masks, kwargs, steps=steps, sweeps=0,
                                            polish_steps=args.polish_steps, repeats=args.timing_repeats)
                baseline.append(result)
            panels = [("Source", base.image(target))]
            for budget in refinement:
                if budget["refinement_sweeps"] == 0:
                    continue
                print(f"{name}: matching Adam-only time to {budget['refinement_sweeps']} sweeps", flush=True)
                result, _, rendered = matched_adam(target, masks, kwargs, baseline, budget,
                                                   args.polish_steps, args.timing_repeats)
                matched.append(result)
                base.image(rendered).save(directory / f"matched_{budget['refinement_sweeps']}.png")
                with Image.open(directory / f"refinement_{budget['refinement_sweeps']}.png") as image:
                    panels.append((f"{args.steps} Adam + {budget['refinement_sweeps']} sweeps ({budget['seconds']:.2f}s)", image.copy()))
                panels.append((f"{result['steps']} Adam + 0 sweeps ({result['seconds']:.2f}s)", base.image(rendered)))
            base.contact_sheet(panels, directory / "matched_comparison.png", columns=3)
            base.contact_sheet(panels, directory / "matched_focus.png", columns=3, crop=crop, zoom=3)
        for stage in args.profile:
            print(f"{name}: sampled {stage} profile", flush=True)
            sampled_profile(target, masks, kwargs, directory, stage)
        if not args.no_plots:
            plot_ablation(directory, refinement, matched)
        report["images"][name] = {"source": str(path), "analysis": analysis, "budgets": budgets,
                                  "refinement_budgets": refinement, "matched_time": matched,
                                  "instrumented_seconds": traced_grid.experiment["seconds"]}
        base.write_report(args.output / "metrics.json", report)
        write_review(args.output, report)
    print(f"Wrote {args.output / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
