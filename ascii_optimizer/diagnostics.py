"""Opt-in solver traces. Keep tensors on their device until the solve finishes."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch


class SolverTrace:
    """Record evaluated states, raw logits, and CPU/GPU timeline intervals.

    Recording is deliberately opt-in: full logits and grid maps are cloned on
    their existing device, then transferred once after solving. Diagnostic
    evaluation/recording costs are included in traced timings; use an ordinary
    matched solve for production timing. GPU event intervals include host gaps
    between events, and are not a sum of kernel execution times.
    """

    def __init__(self, *, gradient_sample_every=12, include_replacements=True):
        if gradient_sample_every < 1:
            raise ValueError("gradient_sample_every must be positive")
        self.gradient_sample_every = gradient_sample_every
        self.include_replacements = include_replacements
        self.records = []
        self.spans = []
        self.static = {}
        self.metadata = {}
        self.counters = {}
        self.replacements = {}

    def sample_gradients(self, terms, parameters):
        """RMS of each weighted term's derivative in the actual Adam variables."""
        result = {}
        for name, value in terms.items():
            gradients = torch.autograd.grad(value, tuple(parameters.values()),
                                            retain_graph=True, allow_unused=True)
            for (parameter_name, parameter), gradient in zip(parameters.items(), gradients):
                result[f"{name}/{parameter_name}"] = (gradient.square().mean().sqrt()
                    if gradient is not None else parameter.new_zeros(()))
        return result

    def register_geometry(self, masks, horizontal, vertical):
        # Zero halos measure changes to edges across cell boundaries too. These
        # are linear encoded-OKLab diagnostics, not displayed/gamut-aware losses.
        pixels = masks.flatten(1).double()
        padded = torch.nn.functional.pad(masks[:, None], (2, 2, 2, 2))
        gx = torch.nn.functional.conv2d(padded, horizontal[:1]).flatten(1).double()
        gy = torch.nn.functional.conv2d(padded, vertical[:1]).flatten(1).double()
        self.register(pixel_gram=pixels @ pixels.T / pixels.shape[1],
                      edge_gram=(gx @ gx.T + gy @ gy.T) / gx.shape[1])

    def record_replacements(self, stage, deltas, current_glyph):
        self.replacements[stage] = {"objective_delta": deltas.detach().clone(),
                                    "current_glyph": current_glyph.detach().clone()}

    def start(self, device, **metadata):
        self.device = torch.device(device)
        self.metadata = metadata
        self.started = time.perf_counter()
        self.origin = self._event()

    def _event(self):
        if self.device.type != "cuda":
            return None
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return event

    def register(self, **tensors):
        self.static.update({key: value.detach().clone() for key, value in tensors.items()})

    def record(self, stage, *, loss, components, glyph, color, logits=None,
               temperature=None, gradients=None, **metadata):
        tensors = {"loss": loss, **{f"loss/{key}": value for key, value in components.items()},
                   "glyph": glyph, "color": color}
        if logits is not None:
            tensors["logits"] = logits
        if gradients is not None:
            tensors.update({f"gradient/{name}": value for name, value in gradients.items()})
        self.records.append({"stage": stage, "temperature": temperature, **metadata,
                             "counters": dict(self.counters),
                             "cpu_seconds": time.perf_counter() - self.started,
                             "event": self._event(),
                             "tensors": {key: value.detach().clone() for key, value in tensors.items()}})

    @contextmanager
    def section(self, name, **metadata):
        begin_cpu, begin_gpu = time.perf_counter(), self._event()
        with torch.profiler.record_function(f"solver.{name}"):
            yield
        self.spans.append({"name": name, **metadata,
                           "cpu_seconds": time.perf_counter() - begin_cpu,
                           "begin_gpu": begin_gpu, "end_gpu": self._event()})

    def save(self, directory: Path) -> dict:
        """Export a complete trace, with candidate identities and stage grids."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        static = {key: value.cpu().numpy() for key, value in self.static.items()}
        records, glyphs, colors, logits, temperatures, logit_records = [], [], [], [], [], []
        visual = {name: [] for name in ("pixel_variance", "edge_variance", "pixel_variance_temperature_1",
                                      "edge_variance_temperature_1", "top_two_pixel_distance_squared",
                                      "top_two_edge_distance_squared")}
        for index, original in enumerate(self.records):
            row = {key: value for key, value in original.items() if key not in ("event", "tensors")}
            row["gpu_timeline_seconds"] = (self.origin.elapsed_time(original["event"]) / 1000
                                            if self.origin is not None else None)
            tensors = {key: value.cpu().numpy() for key, value in original["tensors"].items()}
            row["loss"] = float(tensors["loss"])
            row["components"] = {key[5:]: float(value) for key, value in tensors.items() if key.startswith("loss/")}
            row["loss_terms"] = {name: {"raw_error": value / self.metadata["coefficients"][name],
                                         "coefficient": self.metadata["coefficients"][name],
                                         "weighted_contribution": value}
                                 for name, value in row["components"].items()}
            if any(key.startswith("gradient/") for key in tensors):
                row["weighted_gradient_rms"] = {}
                for key, value in tensors.items():
                    if key.startswith("gradient/"):
                        term, parameter = key[len("gradient/"):].split("/")
                        row["weighted_gradient_rms"].setdefault(term, {})[parameter] = float(value)
            row["grid_index"] = len(glyphs)
            glyphs.append(tensors["glyph"].reshape(self.metadata["rows"], self.metadata["cols"]))
            colors.append(tensors["color"].reshape(self.metadata["rows"], self.metadata["cols"], 3))
            if "logits" in tensors:
                values = tensors["logits"]
                temperature = row["temperature"]
                scaled = (values - values.max(1, keepdims=True)) / temperature
                probability = np.exp(scaled.astype(np.float64))
                probability /= probability.sum(1, keepdims=True)
                entropy = -(probability * np.log2(np.maximum(probability, 1e-300))).sum(1)
                winners = np.sort(values, axis=1)[:, -min(2, values.shape[1]):]
                margin = winners[:, -1] - winners[:, 0]
                row["logit_index"] = len(logits)
                row["logit_summary"] = {
                    "mean_softmax_entropy_bits": float(entropy.mean()),
                    "mean_winner_probability": float(probability.max(1).mean()),
                    "winner_margin_percentiles": np.percentile(margin, [0, 10, 50, 90, 100]).tolist(),
                }
                fixed = np.exp((values - values.max(1, keepdims=True)).astype(np.float64))
                fixed /= fixed.sum(1, keepdims=True)
                row["logit_summary"]["mean_softmax_entropy_bits_at_temperature_1"] = float(
                    -(fixed * np.log2(np.maximum(fixed, 1e-300))).sum(1).mean())
                contrast = ((tensors["color"] - static["background"]) ** 2).sum(-1)
                candidate_ids = static["candidate_indices"]
                top_two = np.argsort(values, axis=1)[:, -min(2, values.shape[1]):]
                pair = np.take_along_axis(candidate_ids, top_two, axis=1)
                for kind in ("pixel", "edge"):
                    gram = static[f"{kind}_gram"]
                    # Usually all cells share the full bank; avoid a cell x K x K allocation.
                    shared = np.all(candidate_ids == candidate_ids[:1])
                    selected = (gram[np.ix_(candidate_ids[0], candidate_ids[0])] if shared
                                else gram[candidate_ids[:, :, None], candidate_ids[:, None, :]])
                    for probabilities, suffix in ((probability, ""), (fixed, "_temperature_1")):
                        mean_squared = ((probabilities @ selected * probabilities).sum(1) if shared
                                        else np.einsum("nk,nkj,nj->n", probabilities, selected, probabilities))
                        second = (probabilities * gram.diagonal()[candidate_ids]).sum(1)
                        variance = np.maximum(0, second - mean_squared) * contrast
                        visual[f"{kind}_variance{suffix}"].append(variance)
                    a, b = pair[:, 0], pair[:, -1]
                    visual[f"top_two_{kind}_distance_squared"].append(
                        np.maximum(0, gram[a, a] + gram[b, b] - 2 * gram[a, b]) * contrast)
                row["visual_summary"] = {name: float(values[-1].mean()) for name, values in visual.items()}
                logits.append(values)
                temperatures.append(temperature)
                logit_records.append(index)
            records.append(row)
        np.savez_compressed(directory / "grids.npz", glyph=np.stack(glyphs).astype(np.uint8),
                            color=np.stack(colors), background=static["background"])
        np.savez_compressed(directory / "logits.npz", logits=np.stack(logits),
                            temperature=temperatures, record_index=logit_records,
                            candidate_indices=static["candidate_indices"],
                            selection_noise=static["selection_noise"])
        np.savez_compressed(directory / "visual.npz", **{name: np.stack(values) for name, values in visual.items()},
                            record_index=logit_records, pixel_gram=static["pixel_gram"], edge_gram=static["edge_gram"])
        if "glyph_stroke_costs" in static:
            np.savez_compressed(directory / "stroke_costs.npz", costs=static["glyph_stroke_costs"])
        replacements = {}
        for stage, tensors in self.replacements.items():
            arrays = {name: value.cpu().numpy() for name, value in tensors.items()}
            np.savez_compressed(directory / f"replacements_{stage}.npz", **arrays,
                                candidate_indices=static["candidate_indices"])
            best = arrays["objective_delta"].min(1)
            replacements[stage] = {"improvable_cell_fraction": float((best < -1e-10).mean()),
                                   "best_delta_percentiles": np.percentile(best, [0, 10, 50, 90, 100]).tolist(),
                                   "note": "Exact single-cell whole-objective changes with current RGB8 ink held fixed; "
                                           "neighbors and affected pooled regions included. Deltas are not additive across cells."}
        spans = []
        for original in self.spans:
            row = {key: value for key, value in original.items() if key not in ("begin_gpu", "end_gpu")}
            row["gpu_timeline_seconds"] = (original["begin_gpu"].elapsed_time(original["end_gpu"]) / 1000
                                             if self.origin is not None else None)
            spans.append(row)
        report = {"metadata": self.metadata, "seconds_before_export": finished - self.started,
                  "timing_note": "Diagnostic costs included; GPU event intervals include host gaps. "
                                 "Use uninstrumented matched runs for production timing.",
                  "records": records, "spans": spans, "replacements": replacements}
        report["counters"] = self.counters
        (directory / "trace.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
