"""Plot saved perceptual experiments without re-running the optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_perceptual import image_rgb
from perceptual_metrics import SpatialMetrics
from PIL import Image


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = json.loads((args.directory / "metrics.json").read_text())
    names = list(report["images"])
    variants = list(
        dict.fromkeys(v for item in report["images"].values() for v in item["variants"])
    )
    palette = {v: plt.get_cmap("tab10")(i % 10) for i, v in enumerate(variants)}
    fig, axes = plt.subplots(
        len(names), 3, figsize=(17, 5 * len(names)), squeeze=False, layout="constrained"
    )
    for row, name in enumerate(names):
        item = report["images"][name]
        for v, r in item["variants"].items():
            bins = r["spatial"]["scales"][1].get("source_lightness_bins", [])
            if bins:
                axes[row, 0].plot(
                    [(b["low"] + b["high"]) / 2 for b in bins],
                    [b["bias"] for b in bins],
                    marker="o",
                    label=v,
                    color=palette[v],
                )
            axes[row, 2].plot(
                [float(f) * item["ppd"] for f in r["flip"]],
                list(r["flip"].values()),
                marker="o",
                label=v,
                color=palette[v],
            )
        base = item["variants"]["baseline"]["spatial"]
        keys = ("orientation_w1", "orientation_tensor", "orientation_stroke_tensor")
        for i, v in enumerate(item["variants"]):
            r = item["variants"][v]["spatial"]
            axes[row, 1].plot(
                range(3),
                [
                    100 * (r[k] / base[k] - 1) if k in r and k in base else np.nan
                    for k in keys
                ],
                marker="o",
                label=v,
                color=palette[v],
            )
        axes[row, 0].axhline(0, color=".5", linestyle="--", linewidth=1)
        axes[row, 0].set(
            title=f"{name}: spatial lightness",
            xlabel=r"Source lightness bin center ($L^*$)",
            ylabel=r"Mean output minus source ($\Delta L^*$)",
        )
        axes[row, 1].axhline(0, color=".5", linestyle="--", linewidth=1)
        axes[row, 1].set(
            xticks=range(3),
            xticklabels=("Circular transport", "Coarse tensor", "Stroke tensor"),
            title=f"{name}: orientation error",
            ylabel="Change from baseline (%)",
        )
        axes[row, 2].set(
            title=f"{name}: independent FLIP evaluation",
            xlabel="Image pixels per degree",
            ylabel="Mean FLIP error (dimensionless)",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=9, ncol=4)
    fig.savefig(args.directory / "measurements.png", dpi=160)
    plt.close(fig)
    for name, item in report["images"].items():
        directory = args.directory / name
        target = rgb(directory / "source.png")
        model = SpatialMetrics(
            image_rgb(target, "cpu"), ppd=item["ppd"], cell_height=item["cell"][1]
        )
        chosen = [
            v
            for v in (
                "baseline",
                "brightness_2",
                "tensor",
                "combined_add",
                "combined_tensor_add",
                "combined_strokes",
            )
            if v in item["variants"]
        ]
        maps = []
        for variant in chosen:
            shown = rgb(directory / variant / "final.png")
            _, ls = model.terms(image_rgb(shown, "cpu"), orientation=False)
            error = ((ls[1] - model.source_l[1]) * 100)[0, 0].numpy()
            error = np.where(model.source_l[1][0, 0].numpy() < 0.35, error, np.nan)
            maps.append(error)
        bound = max(abs(np.nanmin(a)) for a in maps)
        bound = max(bound, max(abs(np.nanmax(a)) for a in maps), 1)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        last = None
        for ax, v, error in zip(axes.ravel(), chosen, maps):
            last = ax.imshow(
                error,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                extent=(0, target.shape[1], target.shape[0], 0),
            )
            ax.set(title=v, xlabel="Image x (pixels)", ylabel="Image y (pixels)")
        for ax in axes.ravel()[len(chosen) :]:
            ax.set_visible(False)
        fig.colorbar(
            last,
            ax=list(axes.ravel()[: len(chosen)]),
            label=r"Shadow lightness difference ($\Delta L^*$); source $L^*<35$",
        )
        fig.savefig(directory / "shadow_error_maps.png", dpi=160)
        plt.close(fig)
    print("Wrote", args.directory / "measurements.png")


if __name__ == "__main__":
    main()
