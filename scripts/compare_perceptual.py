"""Opt-in brightness/contour experiments using installed wallpaper profiles.

  .venv/bin/python -m ascii_optimizer.launcher scripts/compare_perceptual.py --device cuda

Only this process patches the objective. No wallpaper/cache/settings writes.
Refinement is disabled: nonlinear metrics cannot use its quadratic delta cache.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import shutil
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import experiment_utils as base
import numpy as np
import torch
from perceptual_metrics import ExperimentalObjective, SpatialMetrics
from PIL import Image
from torch.nn import functional as F

from ascii_optimizer import cli, optimizer
from ascii_optimizer.assets import glyph_masks
from ascii_optimizer.diagnostics import SolverTrace

VARIANTS = {
    "baseline": ({}, False),
    "brightness_2": ({"spatial_lightness": 2.0}, False),
    "brightness_8": ({"spatial_lightness": 8.0}, False),
    "tensor": ({"orientation_tensor": 0.05, "contour_strength": 0.005}, False),
    "wasserstein": ({"orientation_w1": 0.2, "contour_strength": 0.005}, False),
    "wasserstein_replace": ({"orientation_w1": 0.2, "contour_strength": 0.005}, True),
    "combined": (
        {"spatial_lightness": 8.0, "orientation_w1": 0.2, "contour_strength": 0.005},
        True,
    ),
    "combined_tensor": (
        {
            "spatial_lightness": 8.0,
            "orientation_tensor": 0.05,
            "contour_strength": 0.005,
        },
        True,
    ),
    "combined_add": (
        {"spatial_lightness": 2.0, "orientation_w1": 0.2, "contour_strength": 0.005},
        False,
    ),
    "combined_tensor_add": (
        {
            "spatial_lightness": 2.0,
            "orientation_tensor": 0.05,
            "contour_strength": 0.005,
        },
        False,
    ),
    "combined_strokes": (
        {
            "spatial_lightness": 2.0,
            "orientation_stroke_tensor": 0.05,
            "contour_strength": 0.005,
        },
        False,
    ),
}


def installed_profiles(config):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(config)
    section = next(
        s
        for s in parser.sections()
        if s.endswith("[org.asciiwallpaper.optimizer][Wallpaper")
    )
    result = {}
    for name, field in (("tree", "RenderDirectory"), ("viviana", "RenderDirectoryB")):
        directory = Path(parser[section][field])
        manifest = json.loads((directory / "manifest.json").read_text())
        result[name] = {
            "signature": manifest["signature"],
            "installed": directory / "stills/0000",
        }
    return result


def image_rgb(target, device):
    from ascii_optimizer.assets import srgb_to_linear

    return srgb_to_linear(torch.as_tensor(target, device=device).permute(2, 0, 1)[None])


def ppd_for_width(width, distance=0.6096, diagonal_inches=13):
    physical_width = diagonal_inches * 0.0254 * 16 / math.sqrt(16**2 + 10**2)
    return distance * width / physical_width * math.pi / 180


def evaluate(target, shown, grid, masks, chars, device, cell, ppd):
    rgb = image_rgb(shown, device)
    model = SpatialMetrics(image_rgb(target, device), ppd=ppd, cell_height=cell[1])
    result = {
        **base.image_metrics(target, shown),
        **base.glyph_metrics(grid, masks, chars),
        "spatial": model.measure(rgb),
    }
    result["viewing_sensitivity"] = {
        str(factor): SpatialMetrics(
            image_rgb(target, device), ppd=ppd * factor, cell_height=cell[1]
        ).measure(rgb)
        for factor in (0.8, 1.2)
    }
    import flip_evaluator

    result["flip"] = {}
    for factor in (0.8, 1.0, 1.2):
        _, mean, _ = flip_evaluator.evaluate(
            target,
            shown,
            "LDR",
            inputsRGB=True,
            applyMagma=False,
            parameters={"ppd": ppd * factor},
        )
        result["flip"][str(factor)] = mean
    return result


def write_review(output, report):
    lines = [
        "# Spatial brightness and contour experiments",
        "",
        "These are script-only experiments; installed settings, cached renders, and production defaults are untouched.",
        "All new solves use current code and their installed profile’s font, glyph bank, dithering and weights. The saved installed reference may come from an earlier renderer revision; provenance is recorded separately.",
        "Source and displayed output are decoded to linear sRGB. Light is spatially integrated before CIE L*/100, using Gaussian standard deviations of 2.5, 5, and 10 arcminutes. These explicit scales are hypotheses, not a calibrated implementation of S-CIELAB.",
        "The spatial-lightness loss is symmetric, with fixed source shadow importance 1–4. Shadow diagnostics use source-filtered L* < 35; positive bias means too bright. Peak 500 nits is a capability, not a measurement of the actual SDR white, black level or ambient adaptation; no absolute-brightness equivalence is claimed.",
        "Contour features use Gaussian prefiltering plus overlapping windows two/four glyph heights wide in both axes. Tensor comparison uses unsigned axial moments; Wasserstein comparison uses 12 soft orientation bins on a circle with circumference pi. Equal-bin circular W1 uses cumulative differences minus their median. Reference edge salience/coherence weights are fixed. Both methods include a source-edge strength term to penalize erased contours.",
        "FLIP is the official NVIDIA evaluator, used independently at nominal and ±20% viewing scales. It is not the training objective. Lower FLIP is better overall; it cannot determine artistic preference or perceptual brightness equivalence by itself.",
        "All returned outputs include the same nine ink-polish updates. Refinement is disabled because its exact quadratic cache does not implement these nonlinear objectives. Raw trajectories are saved separately from objective-retained outputs. Each variant changes its selection objective, so total training losses are not cross-variant quality rankings.",
        "",
        "Research: [S-CIELAB halftone evaluation](https://web.stanford.edu/~jefarrel/Publications/1990s/1997_Zhang_Silverstein_Farrell_Wandell.pdf), [FLIP](https://research.nvidia.com/publication/flip), [circular transport](https://arxiv.org/abs/0906.5499), [structure tensor orientation](https://www.ipol.im/pub/art/2022/397/article.pdf).",
        "",
    ]
    for name, item in report["images"].items():
        lines += [
            f"## {name}",
            "",
            f"Image {item['size']}, cells {item['cell']}, {item['ppd']:.1f} image pixels/degree at two feet.",
            "",
            f"[Full comparison]({name}/comparison.png) · [Outline detail]({name}/focus.png) · [Shadow detail]({name}/shadows.png)",
            "",
            "| Variant | Shadow bias, L* | Shadow RMSE, L* | Whole-frame L* RMSE | Circular W1 | Tensor error | Edge-strength error | FLIP | Glyph entropy, bits | Solve s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant, r in item["variants"].items():
            spatial = r["spatial"]
            scale = spatial["scales"][1]
            lines.append(
                f"| {variant} | {scale['shadow_bias']:.3f} | {scale['shadow_rmse']:.3f} | {scale['lightness_rmse']:.3f} | "
                f"{spatial['orientation_w1']:.5f} | {spatial['orientation_tensor']:.5f} | {spatial['contour_strength']:.4f} | "
                f"{r['flip']['1.0']:.4f} | {r['glyph_entropy_bits']:.3f} | {r.get('ordinary_seconds', float('nan')):.2f} |"
            )
        lines += [
            "",
            "Timing is one warmed ordinary solve per variant, without tracing; concurrent desktop/GPU work can affect it. No timing uncertainty is inferred from a single run.",
            "",
        ]
        lines += [
            "| Variant | Source L* 0–10 bias | Source L* 10–20 bias | Source L* 20–35 bias |",
            "|---|---:|---:|---:|",
        ]
        for variant, r in item["variants"].items():
            bins = {
                b["low"]: b["bias"]
                for b in r["spatial"]["scales"][1].get("source_lightness_bins", [])
            }
            lines.append(
                "| "
                + variant
                + " | "
                + " | ".join(
                    f"{bins[k]:.3f}" if k in bins else "n/a" for k in (0, 10, 20)
                )
                + " |"
            )
        lines += [
            "",
            "Source-lightness bins are held fixed across variants. Positive bias in the darkest bin can coexist with negative overall shadow bias.",
            "",
        ]
        for variant in item["variants"]:
            if variant != "installed":
                lines.append(
                    f"[{variant}: raw Adam trajectory]({name}/{variant}/trajectory.png) · [trace]({name}/{variant}/trace.json)"
                )
        lines += [
            "",
            "Training weights: `"
            + json.dumps({k: VARIANTS[k] for k in item["variants"] if k in VARIANTS})
            + "`",
            "",
        ]
    (output / "review.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("rendered/perceptual_experiments")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config/plasma-org.kde.plasma.desktop-appletsrc",
    )
    parser.add_argument("--width", type=int, default=1707)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--images", nargs="+", choices=("viviana", "tree"), default=["viviana", "tree"]
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    args = parser.parse_args()
    if args.width < 64 or args.steps < 1:
        parser.error("width >= 64 and steps >= 1 required")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = installed_profiles(args.config)
    report = {
        "steps": args.steps,
        "device": args.device,
        "viewing": {
            "diagonal_inches": 13,
            "aspect": "16:10",
            "distance_m": 0.6096,
            "peak_capability_nits": 500,
        },
        "images": {},
        "source_hashes": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                optimizer.__file__,
                __file__,
                str(Path(__file__).with_name("perceptual_metrics.py")),
            )
        },
    }
    for path in (
        optimizer.__file__,
        __file__,
        str(Path(__file__).with_name("perceptual_metrics.py")),
    ):
        shutil.copyfile(path, args.output / Path(path).name)
    for name in args.images:
        profile = profiles[name]
        sig = profile["signature"]
        cell = tuple(sig["cell"])
        chars = sig["chars"]
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        size = (args.width, round(args.width * sig["size"][1] / sig["size"][0]))
        target = base.load_target(
            Path(sig["images"][0]), size, sig["crop"], sig["grayscale"]
        )
        masks = glyph_masks(Path(sig["font"]), list(chars), cell)
        ppd = ppd_for_width(args.width)
        base.image(target).save(directory / "source.png")
        kwargs = {
            "style": "headroom",
            "invert_ink": sig["invert_ink"],
            "background": sig["background"],
            "seed": sig["seed"],
            "steps": args.steps,
            "dither": sig["dither"],
            "structure_dither": sig["structure_dither"],
            "device": args.device,
            "refinement_sweeps": 0,
            "polish_steps": 9,
            "perceptual_loss": False,
            "gradient_loss_strength": sig["gradient_loss_strength"],
            "glyph_luminance_bonus": sig["glyph_luminance_bonus"],
            "glyph_entropy_strength": sig["glyph_entropy_strength"],
        }
        rgb = image_rgb(target, args.device)
        rgb = F.pad(
            rgb,
            (
                0,
                math.ceil(size[0] / cell[0]) * cell[0] - size[0],
                0,
                math.ceil(size[1] / cell[1]) * cell[1] - size[1],
            ),
            mode="replicate",
        )
        model = SpatialMetrics(rgb, ppd=ppd, cell_height=cell[1])
        item = {
            "size": size,
            "cell": cell,
            "ppd": ppd,
            "installed_signature": sig,
            "source_sha256": hashlib.sha256(
                Path(sig["images"][0]).read_bytes()
            ).hexdigest(),
            "variants": {},
        }
        report["images"][name] = item
        panels = [("Source", base.image(target))]
        # Keep the cached original at native resolution as a reference; do not
        # pretend a resized image has the same atlas/grid as these pilot runs.
        installed = profile["installed"] / "preview.png"
        shutil.copyfile(installed, directory / "installed_native.png")
        if list(size) == sig["size"]:
            with Image.open(installed) as im:
                shown = np.asarray(im, dtype=np.float32) / 255
            cached = cli._load_grid(profile["installed"])
            item["variants"]["installed"] = evaluate(
                target, shown, cached, masks, chars, args.device, cell, ppd
            )
            panels.append(("Installed cached output", base.image(shown)))
        print(name, "warmup", flush=True)
        base.solve_style(target, masks, **kwargs)
        for variant in args.variants:
            weights, replace = VARIANTS[variant]
            path = directory / variant
            path.mkdir(exist_ok=True)
            original = optimizer._FrameObjective

            def factory(
                *a,
                original=original,
                model=model,
                weights=weights,
                replace=replace,
                **k,
            ):
                return ExperimentalObjective(original(*a, **k), model, weights, replace)

            print(name, variant, "ordinary solve", flush=True)
            with ExitStack() as stack:
                if weights:
                    stack.enter_context(
                        patch.object(optimizer, "_FrameObjective", factory)
                    )
                grid = base.solve_style(target, masks, **kwargs)
            ordinary_seconds = grid.experiment["seconds"]
            shown = base.preview(grid, masks, size)
            base.image(shown).save(path / "final.png")
            np.savez_compressed(
                path / "grid.npz",
                glyph=grid.glyph,
                color=grid.color,
                background=grid.background,
            )
            result = evaluate(target, shown, grid, masks, chars, args.device, cell, ppd)
            result.update(
                ordinary_seconds=ordinary_seconds,
                weights=weights,
                replace_sobel=replace,
            )
            print(name, variant, "trace", flush=True)
            trace = SolverTrace(
                gradient_sample_every=max(args.steps, 1), include_replacements=False
            )
            with ExitStack() as stack:
                if weights:
                    stack.enter_context(
                        patch.object(optimizer, "_FrameObjective", factory)
                    )
                traced = base.solve_style(target, masks, **kwargs, diagnostics=trace)
            assert np.array_equal(grid.glyph, traced.glyph) and np.array_equal(
                grid.color, traced.color
            )
            trace.metadata["coefficients"].update(weights)
            if replace:
                trace.metadata["coefficients"]["displayed_sobel"] = 0
            trace.metadata.update(
                experimental_weights=weights,
                replace_sobel=replace,
                ppd=ppd,
                replacement_diagnostics="Disabled: nonlinear experimental objectives are unsupported by exact replacement cache",
            )
            saved = trace.save(path)
            data = np.load(path / "grids.npz")
            snapshots = [("Source", base.image(target))]
            raw_metrics = {}
            for row in saved["records"]:
                if (
                    row["stage"] == "initial"
                    or (row["stage"] == "adam_step" and row["step"] in (6, 12, 24))
                    or row["stage"] in ("solver_final", "final")
                ):
                    key = row["stage"] + (
                        f"_{row['step']}" if row["stage"] == "adam_step" else ""
                    )
                    idx = row["grid_index"]
                    raw = optimizer.Grid(
                        data["glyph"][idx], data["color"][idx], data["background"]
                    )
                    image = base.preview(raw, masks, size)
                    base.image(image).save(path / f"{key}.png")
                    snapshots.append((key, base.image(image)))
                    raw_metrics[key] = {
                        "loss": row["loss"],
                        "components": row["components"],
                        "spatial": SpatialMetrics(
                            image_rgb(target, args.device), ppd=ppd, cell_height=cell[1]
                        ).measure(image_rgb(image, args.device)),
                    }
            result["raw_stages"] = raw_metrics
            base.contact_sheet(
                snapshots, path / "trajectory.png", columns=3, panel=(960, 600)
            )
            item["variants"][variant] = result
            panels.append((variant, base.image(shown)))
            (args.output / "metrics.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        scale = args.width / 960
        focus = (330, 270, 530, 440) if name == "viviana" else (140, 95, 270, 215)
        shadow = (600, 40, 800, 200) if name == "viviana" else (330, 180, 530, 340)
        focus = tuple(round(x * scale) for x in focus)
        shadow = tuple(round(x * scale) for x in shadow)
        base.contact_sheet(
            panels, directory / "comparison.png", columns=3, panel=(960, 600)
        )
        base.contact_sheet(
            panels, directory / "focus.png", columns=3, crop=focus, panel=(600, 510)
        )
        base.contact_sheet(
            panels, directory / "shadows.png", columns=3, crop=shadow, panel=(600, 480)
        )
        write_review(args.output, report)
    print("Wrote", args.output / "review.md", flush=True)


if __name__ == "__main__":
    main()
