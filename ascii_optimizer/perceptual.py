"""Spatial lightness and contour preservation for the headroom solver.

Light is integrated in linear RGB before a CIE L* transform. Gaussian scales
are explicit viewing hypotheses, not a calibrated S-CIELAB/FLIP implementation.
Unsigned contour orientations live on a circle of circumference pi.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .defaults import ppd_for_width  # noqa: F401


def gaussian_kernel(image, sigma):
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel /= kernel.sum()
    return kernel


def blur(image, sigma, kernel=None):
    kernel = gaussian_kernel(image, sigma) if kernel is None else kernel
    radius = kernel.numel() // 2
    channels = image.shape[1]
    out = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode="replicate"),
        kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1),
        groups=channels,
    )
    return F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"),
        kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1),
        groups=channels,
    )


def luminance(linear_rgb):
    return (
        linear_rgb
        * linear_rgb.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
    ).sum(1, keepdim=True)


def lightness(y):
    """CIE L*/100, reference white Y=1; finite derivative at black."""
    threshold = (6 / 29) ** 3
    root = y.clamp_min(threshold).pow(1 / 3)
    f = torch.where(y > threshold, root, y / (3 * (6 / 29) ** 2) + 4 / 29)
    return (116 * f - 16) / 100


def circular_w1(p, q):
    """Exact equal-bin circular W1, measured in fractions of circumference.

    Inputs are normalized nonnegative histograms; channel axis is angle.
    Subtracting the median chooses the optimal circular cut/flow offset.
    """
    cumulative = (p - q).cumsum(1)
    offset = cumulative.median(1, keepdim=True).values
    return (cumulative - offset).abs().mean(1, keepdim=True)


def gradients(image):
    from .optimizer import _sobel_gradients

    x, y = _sobel_gradients(image)
    return x, y


class SpatialMetrics:
    """Cache source features once; all differentiable work stays on its device."""

    def __init__(self, source_rgb, *, ppd, cell_height=16, downsample=4, bins=12):
        self.downsample = min(downsample, min(source_rgb.shape[-2:]))
        self.bins = bins
        self.ppd = ppd
        self.source_rgb = source_rgb
        # Achromatic sources use a tied RGB ink parameter, avoiding brightness
        # gradients introducing a hue into grayscale images.
        self.neutral_source = bool(
            (source_rgb.amax(1) - source_rgb.amin(1)).abs().amax() < 1e-6
        )
        # 2.5, 5 and 10 arcmin standard deviation: from glyph texture through
        # local shading. We report every scale, rather than assert one match.
        self.sigmas = tuple(
            max(0.5, ppd * arcmin / 60 / self.downsample) for arcmin in (2.5, 5, 10)
        )
        self.windows = tuple(
            max(3, round(cell_height * n / self.downsample)) for n in (2, 4)
        )
        self.kernels = {s: gaussian_kernel(source_rgb, s) for s in (*self.sigmas, 0.25)}
        angles = torch.arange(
            bins, device=source_rgb.device, dtype=source_rgb.dtype
        ) * (2 * math.pi / bins)
        self.directions = torch.stack((angles.cos(), angles.sin()), 1)
        self.source_y = self.reduce(luminance(source_rgb)).detach()
        self.source_l = tuple(
            lightness(self.blur(self.source_y, s)).detach() for s in self.sigmas
        )
        self.brightness_weights = tuple(
            1 + 3 * (1 - ref).square() for ref in self.source_l
        )
        # Fixed source-defined weights and masks prevent changing the region
        # membership by brightening/darkening the candidate.
        self.shadow = self.source_l[1] < 0.35
        self.source_features = tuple(self.features(self.source_y, i) for i in range(2))
        self.edge_weights = tuple(
            ref["mass"] / (ref["mass"] + 0.0001) for ref in self.source_features
        )

    @torch.no_grad()
    def glyph_stroke_costs(self, masks):
        """Compare actual atlas stroke tensors with source directions per cell.

        Angular costs are evaluated for every glyph, so the hard-choice
        surrogate receives each candidate's actual geometric cost instead of
        a linearization of a normalized tensor at the currently selected glyph.
        """
        height, width = masks.shape[-2:]
        padded = F.pad(masks[:, None], (2, 2, 2, 2))
        gx, gy = gradients(blur(padded, 0.5))
        xx, yy, xy = (v.mean((2, 3)) for v in (gx.square(), gy.square(), gx * gy))
        axial = torch.cat((xx - yy, 2 * xy), 1) / (xx + yy + 1e-8)
        coverage = masks.mean((1, 2))
        from .optimizer import _rgb_to_oklab
        lab = _rgb_to_oklab(self.source_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Source contours are coarser than raster strokes. Reduce only the
        # source field, aligned to cell boundaries, before its large blurs.
        stride = 2 if height % 2 == width % 2 == 0 and min(height, width) >= 8 else 1
        if stride > 1:
            lab = F.avg_pool2d(lab, stride)
        source_cell = (height // stride, width // stride)
        costs = []
        for sigma in (height / 4, height / 2):
            # Chromatic contours matter even at equal luminance. Remove the
            # broad shading ramp: cell colors can represent that ramp without
            # turning every glyph to its direction.
            field = blur(lab, max(0.5, sigma / stride)) - blur(lab, max(1.0, 4 * sigma / stride))
            sx, sy = gradients(field)
            sx, sy = sx / stride, sy / stride
            xx, yy, xy = (F.avg_pool2d(v, source_cell).flatten().unsqueeze(1)
                          for v in (sx.square().sum(1, keepdim=True),
                                    sy.square().sum(1, keepdim=True),
                                    (sx * sy).sum(1, keepdim=True)))
            mass = xx + yy
            ref = torch.cat((xx - yy, 2 * xy), 1) / (mass + 1e-8)
            coherence = ref.square().sum(1, keepdim=True).sqrt()
            weight = mass / (mass + 1e-6) * coherence
            # Prefer less ink when rasters explain a coherent contour equally
            # well. Tone matching still pays the cost if a thin glyph cannot
            # supply enough light. This applies to every raster, without names.
            costs.append(((ref[:, None] - axial[None]).square().sum(2)
                          + 0.25 * coverage[None]) * weight)
        return torch.stack(costs).mean(0)

    def blur(self, image, sigma):
        return blur(image, sigma, self.kernels[sigma])

    def reduce(self, y):
        return F.avg_pool2d(y, self.downsample)

    def pool(self, x, window):
        window = min(window, min(x.shape[-2:]))
        return F.avg_pool2d(x, window, stride=max(1, window // 2))

    def features(self, y, index, *, fine=False, histogram=True):
        # Blur luminance before differentiation; pool moments/histograms over
        # overlapping windows spanning two/four glyph heights in both axes.
        gx, gy = gradients(
            lightness(self.blur(y, 0.25 if fine else self.sigmas[index]))
        )
        energy = gx.square() + gy.square()
        xx = self.pool(gx.square(), self.windows[index])
        yy = self.pool(gy.square(), self.windows[index])
        xy = self.pool(gx * gy, self.windows[index])
        mass = xx + yy
        axial = torch.cat((xx - yy, 2 * xy), 1) / (mass + 1e-6)
        coherence = axial.square().sum(1, keepdim=True).sqrt()
        result = {"axial": axial, "mass": mass, "coherence": coherence}
        if not histogram:
            return result
        # Soft angular bins avoid atan2's undefined derivative at zero. The
        # cos(2 theta), sin(2 theta) representation identifies theta+pi.
        unit = torch.cat((gx.square() - gy.square(), 2 * gx * gy), 1) / (energy + 1e-8)
        logits = torch.einsum("bchw,kc->bkhw", unit, self.directions) * 8
        magnitude = (energy + 1e-8).sqrt() - 1e-4
        hist = self.pool(logits.softmax(1) * magnitude, self.windows[index])
        total = hist.sum(1, keepdim=True)
        # Pseudocount gives a valid uniform distribution at zero energy.
        hist = (hist + 1e-8 / self.bins) / (total + 1e-8)
        return {**result, "hist": hist}

    def terms(self, rgb, *, orientation=True, stroke=False, selected=None):
        selected = (
            set(selected)
            if selected is not None
            else {
                "spatial_lightness",
                "orientation_tensor",
                "orientation_w1",
                "contour_strength",
                *(("orientation_stroke_tensor",) if stroke else ()),
            }
        )
        stroke = stroke or "orientation_stroke_tensor" in selected
        y = self.reduce(luminance(rgb))
        displayed_l = tuple(lightness(self.blur(y, s)) for s in self.sigmas)
        brightness = torch.stack(
            [
                ((out - ref).square() * weight).mean()
                for out, ref, weight in zip(
                    displayed_l, self.source_l, self.brightness_weights
                )
            ]
        ).mean()
        result = {"spatial_lightness": brightness}
        if orientation:
            tensor, transport, strength, stroke_tensor = [], [], [], []
            for i, ref in enumerate(self.source_features):
                out = self.features(y, i, histogram="orientation_w1" in selected)
                # Fixed source edge salience, bounded and detached. A constant
                # image cannot evade the energy-matching term below.
                weight = self.edge_weights[i]
                direction_weight = weight * ref["coherence"]
                if "orientation_tensor" in selected:
                    tensor.append(
                        (
                            (out["axial"] - ref["axial"]).square().sum(1, keepdim=True)
                            * direction_weight
                        ).mean()
                    )
                if "orientation_w1" in selected:
                    transport.append(
                        (circular_w1(out["hist"], ref["hist"]) * weight).mean()
                    )
                if "contour_strength" in selected:
                    strength.append(
                        (
                            (
                                torch.log1p(out["mass"] / 0.0001)
                                - torch.log1p(ref["mass"] / 0.0001)
                            ).square()
                            * weight
                        ).mean()
                    )
                if stroke:
                    fine = self.features(y, i, fine=True, histogram=False)
                    stroke_tensor.append(
                        ((fine["axial"] - ref["axial"]).square().sum(1, keepdim=True)
                         * direction_weight).mean()
                    )
            for name, values in (
                ("orientation_tensor", tensor),
                ("orientation_w1", transport),
                ("contour_strength", strength),
            ):
                if values:
                    result[name] = torch.stack(values).mean()
            if stroke:
                result["orientation_stroke_tensor"] = torch.stack(stroke_tensor).mean()
        return result, displayed_l

    @torch.no_grad()
    def measure(self, rgb):
        terms, ls = self.terms(rgb, stroke=True)
        result = {name: float(value) for name, value in terms.items()}
        result["scales"] = []
        for sigma, ref, out in zip(self.sigmas, self.source_l, ls):
            difference = (out - ref) * 100
            shadow = ref < 0.35
            result["scales"].append(
                {
                    "sigma_pixels": sigma * self.downsample,
                    "lightness_bias": float(difference.mean()),
                    "lightness_rmse": float(difference.square().mean().sqrt()),
                    "shadow_bias": float(difference[shadow].mean())
                    if shadow.any()
                    else None,
                    "shadow_rmse": float(difference[shadow].square().mean().sqrt())
                    if shadow.any()
                    else None,
                    "shadow_fraction": float(shadow.float().mean()),
                    "source_lightness_bins": [],
                }
            )
            for low, high in ((0, 10), (10, 20), (20, 35), (35, 60), (60, 100)):
                selected = (ref * 100 >= low) & (ref * 100 < high)
                if selected.any():
                    result["scales"][-1]["source_lightness_bins"].append(
                        {
                            "low": low,
                            "high": high,
                            "fraction": float(selected.float().mean()),
                            "bias": float(difference[selected].mean()),
                            "rmse": float(difference[selected].square().mean().sqrt()),
                        }
                    )
        return result
