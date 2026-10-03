"""Image-guided forced flow and font-raster matching for static flow domains.

Velocity evolves on an isotropic, padded periodic domain. Spectral projection
uses consistent real derivative symbols and diffusion damps all modes. No fluid simulation
or glyph decisions run in the wallpaper shader.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .defaults import DEFAULT_FLOW_EDDIES, DEFAULT_FLOW_SCALE
from .perceptual import bandpass_moments, blur, gradients
from .settings import finite_float
from .tone import srgb_to_oklab


@dataclass
class FlowField:
    velocity: torch.Tensor  # [1, 2, H, W], x/y velocity in simulation samples/step
    confidence: torch.Tensor
    source_confidence: torch.Tensor
    shape: tuple[int, int]  # padded output image height/width
    cell: tuple[int, int]  # height/width
    spacing: float  # image pixels per simulation sample
    diagnostics: dict


class _SpectralDomain:
    def __init__(self, height, width, device):
        ky = torch.fft.fftfreq(height, device=device) * (2 * math.pi)
        kx = torch.fft.rfftfreq(width, device=device) * (2 * math.pi)
        self.laplacian = kx[None, :].square() + ky[:, None].square()
        # A real centered derivative has zero response at the Nyquist mode.
        if height % 2 == 0:
            ky[height // 2] = 0
        if width % 2 == 0:
            kx[-1] = 0
        self.kx, self.ky = kx[None, :], ky[:, None]
        self.k2 = self.kx.square() + self.ky.square()
        self.shape = (height, width)

    def project(self, velocity):
        spectrum = torch.fft.rfft2(velocity)
        along = (self.kx * spectrum[:, 0] + self.ky * spectrum[:, 1]) / self.k2.clamp_min(1e-12)
        spectrum = spectrum - torch.stack((self.kx * along, self.ky * along), 1)
        return torch.fft.irfft2(spectrum, s=self.shape)

    def diffuse(self, velocity, viscosity, dt):
        return torch.fft.irfft2(torch.fft.rfft2(velocity) / (1 + viscosity * dt * self.laplacian), s=self.shape)

    def divergence(self, velocity):
        spectrum = torch.fft.rfft2(velocity)
        return torch.fft.irfft2(1j * (self.kx * spectrum[:, 0] + self.ky * spectrum[:, 1]), s=self.shape)

    def curl(self, potential):
        spectrum = torch.fft.rfft2(potential)
        return torch.stack((torch.fft.irfft2(1j * self.ky * spectrum, s=self.shape),
                            torch.fft.irfft2(-1j * self.kx * spectrum, s=self.shape)), 1)


def _rms(value):
    return value.square().mean().sqrt()


def _advect(velocity, dt, coordinates):
    height, width = velocity.shape[-2:]
    x, y = coordinates
    px = (x - dt * velocity[:, 0]).remainder(width) + 1
    py = (y - dt * velocity[:, 1]).remainder(height) + 1
    grid = torch.stack((2 * px / (width + 1) - 1, 2 * py / (height + 1) - 1), -1)
    return F.grid_sample(F.pad(velocity, (1, 1, 1, 1), mode="circular"), grid,
                         mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def simulate_flow(source: np.ndarray, cell: tuple[int, int], *, seed: int = 0,
                  eddies: float = DEFAULT_FLOW_EDDIES, scale: float = DEFAULT_FLOW_SCALE,
                  device: str = "cpu", steps: int = 64,
                  on_progress: Callable[[float, str], None] | None = None) -> FlowField:
    """Freeze a developed forced field; parameters and seed define its snapshot."""
    eddies = finite_float(eddies, "flow_eddies", 0, 2)
    scale = finite_float(scale, "flow_scale", 1, 24)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Flow steps must be a nonnegative integer")
    ch, cw = cell
    if min(ch, cw) < 1 or source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
        raise ValueError("Flow needs an RGB image and positive cell dimensions")
    height = math.ceil(source.shape[0] / ch) * ch
    width = math.ceil(source.shape[1] / cw) * cw
    spacing = max(1.0, cw / 3)
    fh, fw = max(4, round(height / spacing)), max(4, round(width / spacing))
    # Use exact per-axis spacings when interpolating back to rectangular cells.
    # The interior is isotropic to rounding; the padding prevents seam guidance.
    margin = max(8, round(scale * cw / spacing))
    lab = torch.as_tensor(srgb_to_oklab(source), dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    lab = F.pad(lab, (0, width - source.shape[1], 0, height - source.shape[0]), mode="replicate")
    lab = F.interpolate(lab, size=(fh, fw), mode="bilinear", align_corners=False)
    lab = F.pad(lab, (margin, margin, margin, margin), mode="replicate")
    bands = [bandpass_moments(lab, max(0.75, factor * ch / spacing)) for factor in (0.25, 0.5)]
    xx, yy, xy = (sum(band[i] for band in bands) * 0.5 for i in range(3))
    xx, yy, xy = (blur(v, 1.25) for v in (xx, yy, xy))
    energy = xx + yy
    axial = -torch.cat((xx - yy, 2 * xy), 1) / energy.clamp_min(1e-12)
    coherence = axial.square().sum(1, keepdim=True).sqrt().clamp(0, 1)
    mass_scale = torch.quantile(energy, 0.7).clamp_min(1e-6)
    salience = energy / (energy + mass_scale * 0.15)
    confidence = (coherence * salience).clamp(0, 1)
    unit = axial / coherence.clamp_min(1e-6)
    domain = _SpectralDomain(*lab.shape[-2:], device)
    rng = np.random.default_rng(seed)
    white = torch.as_tensor(rng.standard_normal(domain.shape), dtype=torch.float32, device=device)[None]
    spectrum = torch.fft.rfft2(white)
    def noise(length):
        out = torch.fft.irfft2(spectrum * torch.exp(-0.5 * domain.laplacian * length ** 2), s=domain.shape)
        return out / _rms(out).clamp_min(1e-6)
    length = scale * cw / spacing / 3
    drive = domain.curl(noise(max(1.5, length * 1.8)))
    drive = drive / _rms(drive).clamp_min(1e-6)
    # Modulate a potential, then curl it. Direct velocity modulation would
    # create spurious sources/sinks at changes in the complexity envelope.
    envelope = 0.15 + 0.85 * salience[:, 0] * (1 - coherence[:, 0])
    eddy_force = domain.curl(noise(max(0.8, length * 0.5)) * blur(envelope[:, None], 1.5)[:, 0])
    eddy_force = eddy_force / _rms(eddy_force).clamp_min(1e-6)
    forcing = 0.12 * drive + 0.12 * eddies * eddy_force
    velocity = domain.project(drive * 0.6 + eddies * eddy_force * 0.2)
    h, w = domain.shape
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coordinates = (x[None].float(), y[None].float())
    before = after = velocity.new_tensor(0.)
    dt = 0.4
    for index in range(steps):
        velocity = velocity + dt * forcing
        ux, uy = velocity[:, :1], velocity[:, 1:]
        qx, qy = unit[:, :1], unit[:, 1:]
        tangent = torch.cat(((1 + qx) * ux + qy * uy,
                             qy * ux + (1 - qx) * uy), 1) * 0.5
        velocity = velocity - (velocity - tangent) * (1 - torch.exp(-dt * 1.6 * confidence))
        velocity = domain.diffuse(velocity, 0.12, dt) / (1 + 0.04 * dt)
        velocity = domain.project(velocity)
        velocity = _advect(velocity, dt, coordinates)
        before = _rms(domain.divergence(velocity))
        velocity = domain.project(velocity)
        after = _rms(domain.divergence(velocity))
        # Global rescaling bounds backtraces without distorting local axes.
        velocity /= (velocity.square().sum(1).sqrt().amax() / 5).clamp_min(1)
        if on_progress is not None and (index % 8 == 0 or index == steps - 1):
            on_progress((index + 1) / max(1, steps), "Developing flow domains")
    crop = (..., slice(margin, margin + fh), slice(margin, margin + fw))
    velocity, confidence = velocity[crop], confidence[crop]
    speed = velocity.square().sum(1, keepdim=True).sqrt()
    speed_confidence = speed / (speed + speed.mean().clamp_min(1e-5) * 0.1)
    return FlowField(velocity, speed_confidence, confidence, (height, width), cell, spacing,
                     {"steps": steps, "seed": seed, "eddies": eddies, "scale": scale,
                      "simulation_shape": [h, w], "spacing_pixels": spacing,
                      "divergence_before_projection": float(before),
                      "divergence_after_projection": float(after),
                      "mean_speed": float(speed.mean()), "boundary": "padded periodic"})


def _axial_bins(axial, bins=12):
    angles = torch.arange(bins, device=axial.device, dtype=axial.dtype) * (2 * math.pi / bins)
    directions = torch.stack((angles.cos(), angles.sin()), 1)
    return (torch.einsum("bchw,kc->bkhw", axial, directions) * 8).softmax(1)


def _cell_patches(field, rows, cols, patch=(4, 2)):
    """Cell-aligned descriptors; tall cells have more vertical subpatches."""
    ph, pw = patch
    value = F.adaptive_avg_pool2d(field, (rows * ph, cols * pw))
    return value[0].reshape(field.shape[1], rows, ph, cols, pw).permute(1, 3, 0, 2, 4).reshape(rows * cols, field.shape[1], ph * pw)


@torch.no_grad()
def glyph_flow_costs(field: FlowField, masks: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Font-aware unsigned direction distributions, local turning and spread.

    Costs retain cell-internal positions, so loops and crosses differ even when
    their global direction histograms agree. Empty glyphs have no flow demand.
    """
    height, width = field.shape
    ch, cw = field.cell
    rows, cols = height // ch, width // cw
    velocity = F.interpolate(field.velocity, size=field.shape, mode="bilinear", align_corners=False)
    energy = velocity.square().sum(1, keepdim=True)
    ux, uy = velocity[:, :1], velocity[:, 1:]
    axial = torch.cat((ux.square() - uy.square(), 2 * ux * uy), 1) / energy.clamp_min(1e-10)
    confidence = F.interpolate(field.confidence, size=field.shape, mode="bilinear", align_corners=False)
    target = _cell_patches(axial * confidence, rows, cols)
    confidence_patches = _cell_patches(confidence, rows, cols)
    target /= confidence_patches.clamp_min(1e-6)
    target_hist = _cell_patches(_axial_bins(axial) * confidence, rows, cols).sum(2)
    target_hist /= target_hist.sum(1, keepdim=True).clamp_min(1e-6)
    target_mean = target.mean(2)
    target_spread = (1 - target_mean.square().sum(1).sqrt()).clamp(0, 1)
    # Gradient normals rotated 90 degrees become actual glyph stroke tangents.
    gx, gy = gradients(blur(masks[:, None], 0.5))
    stroke_energy = gx.square() + gy.square()
    glyph_axial = -torch.cat((gx.square() - gy.square(), 2 * gx * gy), 1) / stroke_energy.clamp_min(1e-10)
    patch_energy = F.adaptive_avg_pool2d(stroke_energy, (4, 2)).flatten(2)
    glyph_patches = F.adaptive_avg_pool2d(glyph_axial * stroke_energy, (4, 2)).flatten(2) / patch_energy.clamp_min(1e-10)
    # Normalize patch importance by each raster, avoiding an automatic reward
    # for sparse or dense glyphs. Tone fitting handles achievable brightness.
    patch_weight = patch_energy / patch_energy.sum(2, keepdim=True).clamp_min(1e-10)
    glyph_hist = (_axial_bins(glyph_axial) * stroke_energy).sum((2, 3))
    glyph_hist /= glyph_hist.sum(1, keepdim=True).clamp_min(1e-10)
    glyph_mean = (glyph_patches * patch_weight).sum(2)
    glyph_spread = (1 - glyph_mean.square().sum(1).sqrt()).clamp(0, 1)
    # Bounded dissimilarity of cell-local axes, plus complete direction spread.
    local = ((target[:, None] - glyph_patches[None]).square() * patch_weight[None]).sum((2, 3))
    histogram = (target_hist[:, None].sqrt() - glyph_hist[None].sqrt()).square().sum(2)
    spread = (target_spread[:, None] - glyph_spread[None]).square()
    costs = (0.55 * local + 0.30 * histogram + 0.15 * spread) * confidence_patches.mean((1, 2))[:, None]
    presence = masks.flatten(1).gt(0).any(1)
    if bool(presence.any()):
        costs -= costs[:, presence].amin(1, keepdim=True)
    costs[:, ~presence] = 0
    return costs, {"mean_complexity": float(target_spread.mean()),
                   "glyph_spread": glyph_spread.cpu().tolist(),
                   "mean_flow_cost": float(costs.mean())}
