"""Image-guided forced flow and font-raster matching for static flow domains.

Velocity evolves on an isotropic, padded periodic domain. Spectral projection
uses consistent real derivative symbols and diffusion damps all modes. No fluid simulation
or glyph decisions run in the wallpaper shader.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
from pathlib import Path
import tempfile

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


@dataclass
class FlowSimulation:
    """Own the padded state; cropped snapshots cannot resume the solver."""
    velocity: torch.Tensor
    domain: _SpectralDomain
    coordinates: tuple
    forcing: torch.Tensor
    unit: torch.Tensor
    confidence: torch.Tensor
    crop: tuple
    shape: tuple[int, int]
    cell: tuple[int, int]
    spacing: float
    diagnostics: dict
    animated_forces: tuple[torch.Tensor, torch.Tensor]
    time: float = 0.0
    steps: int = 0

    @torch.no_grad()
    def advance(self, steps=1, *, dt=0.4, animated=False):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Flow steps must be a nonnegative integer")
        dt = finite_float(dt, "flow timestep", 1e-6, 0.4)
        if steps == 0:
            return
        before = after = self.velocity.new_tensor(0.)
        for _ in range(steps):
            forcing = self.forcing
            if animated:
                # Independent time scales, unrelated to the requested clip period.
                a, b = self.animated_forces
                forcing = forcing + a * math.sin(self.time * 0.071) + b * math.cos(self.time * 0.113)
            velocity = self.velocity + dt * forcing
            ux, uy = velocity[:, :1], velocity[:, 1:]
            qx, qy = self.unit[:, :1], self.unit[:, 1:]
            tangent = torch.cat(((1 + qx) * ux + qy * uy,
                                 qy * ux + (1 - qx) * uy), 1) * 0.5
            velocity = velocity - (velocity - tangent) * (1 - torch.exp(-dt * 1.6 * self.confidence))
            velocity = self.domain.diffuse(velocity, 0.12, dt) / (1 + 0.04 * dt)
            velocity = self.domain.project(velocity)
            velocity = _advect(velocity, dt, self.coordinates)
            before = _rms(self.domain.divergence(velocity))
            velocity = self.domain.project(velocity)
            after = _rms(self.domain.divergence(velocity))
            velocity /= (velocity.square().sum(1).sqrt().amax() / 5).clamp_min(1)
            self.velocity = velocity
            self.time += dt
            self.steps += 1
        self.diagnostics.update(steps=self.steps, divergence_before_projection=float(before),
                                divergence_after_projection=float(after))

    @torch.no_grad()
    def snapshot(self, velocity=None, *, cpu=False):
        velocity = (self.velocity if velocity is None else velocity)[self.crop].clone()
        confidence = self.confidence[self.crop].clone()
        if cpu:
            velocity, confidence = velocity.cpu(), confidence.cpu()
        speed = velocity.square().sum(1, keepdim=True).sqrt()
        speed_confidence = speed / (speed + speed.mean().clamp_min(1e-5) * 0.1)
        return FlowField(velocity, speed_confidence, confidence, self.shape, self.cell,
                         self.spacing, dict(self.diagnostics, mean_speed=float(speed.mean())))


@torch.no_grad()
def prepare_flow(source: np.ndarray, cell: tuple[int, int], *, seed=0,
                 eddies=DEFAULT_FLOW_EDDIES, scale=DEFAULT_FLOW_SCALE, device="cpu") -> FlowSimulation:
    eddies = finite_float(eddies, "flow_eddies", 0, 2)
    scale = finite_float(scale, "flow_scale", 1, 24)
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
    forces = []
    for _ in range(2):
        white = torch.as_tensor(rng.standard_normal(domain.shape), dtype=torch.float32, device=device)[None]
        potential = torch.fft.irfft2(torch.fft.rfft2(white) *
                    torch.exp(-0.5 * domain.laplacian * max(1.2, length) ** 2), s=domain.shape)
        force = domain.curl(potential * (0.4 + blur(envelope[:, None], 1.5)[:, 0]))
        forces.append(force / _rms(force).clamp_min(1e-6) * (0.09 + 0.06 * eddies))
    crop = (..., slice(margin, margin + fh), slice(margin, margin + fw))
    return FlowSimulation(velocity, domain, coordinates, forcing, unit, confidence, crop,
                          (height, width), cell, spacing,
                          {"seed": seed, "eddies": eddies, "scale": scale, "steps": 0,
                           "divergence_before_projection": 0.0, "divergence_after_projection": 0.0,
                           "simulation_shape": [h, w], "spacing_pixels": spacing,
                           "boundary": "padded periodic"}, tuple(forces))


@torch.no_grad()
def simulate_flow(source: np.ndarray, cell: tuple[int, int], *, seed=0,
                  eddies=DEFAULT_FLOW_EDDIES, scale=DEFAULT_FLOW_SCALE,
                  device="cpu", steps=64, on_progress=None) -> FlowField:
    simulation = prepare_flow(source, cell, seed=seed, eddies=eddies, scale=scale, device=device)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Flow steps must be a nonnegative integer")
    for index in range(steps):
        simulation.advance()
        if on_progress is not None and (index % 8 == 0 or index == steps - 1):
            on_progress((index + 1) / max(1, steps), "Developing flow domains")
    return simulation.snapshot()


@dataclass
class FlowLoop:
    fields: list[FlowField]
    diagnostics: dict


@torch.no_grad()
def simulate_flow_loop(source, cell, *, frames=120, seed=0, eddies=DEFAULT_FLOW_EDDIES,
                       scale=DEFAULT_FLOW_SCALE, device="cpu", on_progress=None) -> FlowLoop:
    """Impose a temporal boundary on a developed trajectory, not its forcing.

    Find the least mismatched window, then blend its head with a continued tail.
    All blending takes place on the full projected spatial domain. Trajectories
    are disk-backed; only the chosen cropped clip remains resident on the CPU.
    """
    if isinstance(frames, bool) or not isinstance(frames, int) or not 4 <= frames <= 720:
        raise ValueError("Flow clips need 4 to 720 frames")
    simulation = prepare_flow(source, cell, seed=seed, eddies=eddies, scale=scale, device=device)
    simulation.advance(96, animated=True)
    period = 48.0
    substeps = max(1, math.ceil(period / frames / 0.4))
    dt = period / frames / substeps
    overlap = max(2, round(frames * 0.25))
    count = 2 * frames + overlap + 2
    residuals = []
    divergences = []
    with tempfile.TemporaryDirectory(prefix="ascii-flow-") as temporary:
        history = np.memmap(Path(temporary) / "velocity.dat", mode="w+", dtype=np.float32,
                            shape=(count, *simulation.velocity.shape))
        for index in range(count):
            history[index] = simulation.velocity.cpu().numpy()
            if index >= frames + 1 and index - frames - 1 <= frames:
                start = index - frames - 1
                a, b = history[start], history[start + frames]
                amplitude = max(float(np.sqrt(np.mean(a * a))), 1e-6)
                mismatch = float(np.sqrt(np.mean((a - b) ** 2))) / amplitude
                da = history[start + 1] - a
                db = history[start + frames + 1] - b
                derivative = float(np.sqrt(np.mean((da - db) ** 2))) / amplitude
                residuals.append((mismatch + 2 * derivative, start, mismatch))
            simulation.advance(substeps, dt=dt, animated=True)
            if on_progress and index % 8 == 0:
                on_progress(0.8 * (index + 1) / count, "Finding a developed flow window")
        _, start, residual = min(residuals)
        fields, correction, speeds = [], [], []
        for index in range(frames):
            raw = torch.as_tensor(np.array(history[start + index]), device=device)
            velocity = raw
            if index < overlap:
                q = index / overlap
                weight = 6*q**5 - 15*q**4 + 10*q**3
                tail = torch.as_tensor(np.array(history[start + frames + index]), device=device)
                velocity = (1 - weight) * tail + weight * raw
                correction.append(float(_rms(velocity - raw) / _rms(raw).clamp_min(1e-6)))
            velocity = simulation.domain.project(velocity)
            divergences.append(float(_rms(simulation.domain.divergence(velocity))))
            fields.append(simulation.snapshot(velocity, cpu=True))
            speeds.append(fields[-1].diagnostics["mean_speed"])
            if on_progress and index % 8 == 0:
                on_progress(0.8 + 0.2 * (index + 1) / frames, "Closing the temporal flow boundary")
        del history
    increments = torch.stack([_rms(fields[(i + 1) % frames].velocity - field.velocity)
                              for i, field in enumerate(fields)])
    diagnostics = dict(simulation.diagnostics, method="temporal-overlap", frame_count=frames,
                       simulation_period=period, timestep=dt, substeps=substeps,
                       overlap_frames=overlap, selected_window=start, state_residual=residual,
                       correction_rms=max(correction, default=0),
                       velocity_step_mean=float(increments.mean()),
                       velocity_seam_ratio=float(increments[-1] / increments[:-1].mean().clamp_min(1e-8)),
                       minimum_speed=min(speeds), maximum_speed=max(speeds),
                       loop_divergence_max=max(divergences), forcing="independent time scales")
    if any(not bool(torch.isfinite(field.velocity).all()) for field in fields):
        raise ValueError("Flow loop contains non-finite velocities")
    if diagnostics["velocity_seam_ratio"] > 2.5:
        raise ValueError("Unable to close the flow boundary smoothly; try a different seed or lower eddy strength")
    if on_progress:
        on_progress(1, "Flow boundary prepared")
    return FlowLoop(fields, diagnostics)


def _axial_bins(axial, bins=12):
    angles = torch.arange(bins, device=axial.device, dtype=axial.dtype) * (2 * math.pi / bins)
    directions = torch.stack((angles.cos(), angles.sin()), 1)
    return (torch.einsum("bchw,kc->bkhw", axial, directions) * 8).softmax(1)


def _cell_patches(field, rows, cols, patch=(4, 2)):
    """Cell-aligned descriptors; tall cells have more vertical subpatches."""
    ph, pw = patch
    value = F.adaptive_avg_pool2d(field, (rows * ph, cols * pw))
    return value[0].reshape(field.shape[1], rows, ph, cols, pw).permute(1, 3, 0, 2, 4).reshape(rows * cols, field.shape[1], ph * pw)


@dataclass
class GlyphFlowDescriptors:
    patches: torch.Tensor
    weight: torch.Tensor
    histogram: torch.Tensor
    spread: torch.Tensor
    presence: torch.Tensor


@torch.no_grad()
def prepare_glyph_flow(masks: torch.Tensor) -> GlyphFlowDescriptors:
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
    return GlyphFlowDescriptors(glyph_patches, patch_weight, glyph_hist, glyph_spread,
                                masks.flatten(1).gt(0).any(1))


@torch.no_grad()
def glyph_flow_costs(field: FlowField, masks: torch.Tensor,
                     descriptors: GlyphFlowDescriptors | None = None) -> tuple[torch.Tensor, dict]:
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
    descriptors = descriptors or prepare_glyph_flow(masks)
    glyph_patches, patch_weight = descriptors.patches, descriptors.weight
    glyph_hist, glyph_spread = descriptors.histogram, descriptors.spread
    # Bounded dissimilarity of cell-local axes, plus complete direction spread.
    local = ((target[:, None] - glyph_patches[None]).square() * patch_weight[None]).sum((2, 3))
    histogram = (target_hist[:, None].sqrt() - glyph_hist[None].sqrt()).square().sum(2)
    spread = (target_spread[:, None] - glyph_spread[None]).square()
    costs = (0.55 * local + 0.30 * histogram + 0.15 * spread) * confidence_patches.mean((1, 2))[:, None]
    presence = descriptors.presence
    if bool(presence.any()):
        costs -= costs[:, presence].amin(1, keepdim=True)
    costs[:, ~presence] = 0
    return costs, {"mean_complexity": float(target_spread.mean()),
                   "glyph_spread": glyph_spread.cpu().tolist(),
                   "mean_flow_cost": float(costs.mean())}
