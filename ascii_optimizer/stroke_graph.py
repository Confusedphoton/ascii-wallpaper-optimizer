"""Connected-Stroke MRF / Graph ASCII renderer for fixed SDF glyph grids.

Precomputes 4-direction glyph-seam compatibility matrices directly from the
font's SDF rasters, then solves a tangent-gated 2D Markov Random Field via
parallel 4-color checkerboard sweeps so strokes chain continuously along real
contours, smooth atmospheric gradients (auroras/glaciers), and tapering strands.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .assets import (
    blue_noise,
    decode_stroke_weight_u8,
    encode_stroke_weight_u8,
    sdf_to_mask,
)
from .defaults import STROKE_CHARS
from .optimizer import Grid, _choose_background
from .perceptual import blur, gradients
from .tone import encode_headroom_lab, quantize_playback_lab, srgb_to_oklab

# Curated graphic banks for lab experiments and production defaults.
GRAPH_SYMBOL_CHARS = " .'`:;-_=~+|/\\()[]{}<>^v*#%@&$"
GRAPH_ORGANIC_CHARS = " .'`:;-_=~+|/\\(){}<>^v*xX#%@&$"
GRAPH_FLOWING_CHARS = " .'`:;-_~+|/\\(){}<>^v*xX#%@&$"
GRAPH_CURVE_CHARS = STROKE_CHARS
GRAPH_ASCII_CHARS = " .'`:;-_=~+|/\\()[]{}<>^v*xX#%@&$"


def _oklab_morph_close(lab: torch.Tensor, radius: int) -> torch.Tensor:
    """Lightness-ordered morphological closing in OKLab [1, 3, H, W].

    Bridges thin dark ink outlines (<= 2*radius px) inside solid bright regions
    without shifting hues or altering macro dark silhouettes (straps, buckles).
    """
    if radius <= 0:
        return lab
    k = 2 * radius + 1
    B, C, H, W = lab.shape
    flat_n = H * W

    def gather_by_L(img: torch.Tensor, negate: bool) -> torch.Tensor:
        L = -img[:, :1] if negate else img[:, :1]
        _, idx = F.max_pool2d(L, kernel_size=k, stride=1, padding=radius, return_indices=True)
        idx_flat = idx.reshape(B, 1, flat_n).expand(B, C, flat_n)
        return img.reshape(B, C, flat_n).gather(2, idx_flat).reshape(B, C, H, W)

    return gather_by_L(gather_by_L(lab, negate=False), negate=True)


def _seam_compatibilities(masks: np.ndarray, *, soft_curves: bool = True) -> dict[str, torch.Tensor]:
    """Build directional pairwise stroke-continuity bonus matrices [K, K]."""
    m = torch.from_numpy(masks).float()  # [K, H, W]
    K, H, W = m.shape
    bw = max(1, W // 4)
    bh = max(1, H // 4)
    cov = m.mean((1, 2))
    crisp = (1.0 - 0.48 * cov[:, None]) * (1.0 - 0.48 * cov[None, :])

    m_soft = blur(m[:, None], 1.35)[:, 0] if (soft_curves and min(H, W) >= 4) else m

    def pair_affinity(edge_a: torch.Tensor, edge_b: torch.Tensor,
                      reach_a: torch.Tensor, reach_b: torch.Tensor) -> torch.Tensor:
        a = edge_a.reshape(K, -1)
        b = edge_b.reshape(K, -1)
        a_energy = a.square().sum(1).sqrt()
        b_energy = b.square().sum(1).sqrt()
        overlap = (a @ b.T) / (a_energy[:, None] * b_energy[None, :] + 1e-6)
        reach = (reach_a.reshape(K, -1).amax(1)[:, None]
                 * reach_b.reshape(K, -1).amax(1)[None, :]).clamp(0.0, 1.0)
        return overlap * reach * crisp

    # Dampen horizontal self-chaining for extreme top/bottom-pinned glyphs (like '_' and '`').
    ys = torch.arange(H, dtype=torch.float32)[None, :, None]
    y_norm = (m * ys).sum((1, 2)) / m.sum((1, 2)).clamp_min(1e-6) / max(H - 1, 1)
    centered_y = (1.0 - 1.6 * (y_norm - 0.5).abs().clamp_min(0.28).sub(0.28)).clamp(0.35, 1.0)
    horiz_damp = centered_y[:, None] * centered_y[None, :] if soft_curves else 1.0

    horiz = pair_affinity(m_soft[:, :, -bw:], m_soft[:, :, :bw], m[:, :, -bw:], m[:, :, :bw]) * horiz_damp
    vert = pair_affinity(m_soft[:, -bh:, :], m_soft[:, :bh, :], m[:, -bh:, :], m[:, :bh, :])
    dbh = max(bh, H // 2) if soft_curves else bh
    dbw = max(bw, W // 2) if soft_curves else bw
    diag_dr = pair_affinity(m_soft[:, -dbh:, -dbw:], m_soft[:, :dbh, :dbw], m[:, -dbh:, -dbw:], m[:, :dbh, :dbw])
    diag_dl = pair_affinity(m_soft[:, -dbh:, :dbw], m_soft[:, :dbh, -dbw:], m[:, -dbh:, :dbw], m[:, :dbh, -dbw:])

    return {"horiz": horiz, "vert": vert, "diag_dr": diag_dr, "diag_dl": diag_dl}


def _glyph_axial(masks: np.ndarray) -> torch.Tensor:
    m = torch.from_numpy(masks)[:, None].float()
    if min(masks.shape[1:]) < 3:
        return torch.zeros((len(masks), 2), dtype=torch.float32)
    gx, gy = gradients(blur(m, 0.5))
    xx = (gx * gx).mean((2, 3))
    yy = (gy * gy).mean((2, 3))
    xy = (gx * gy).mean((2, 3))
    energy = (xx + yy).clamp_min(1e-8)
    return -torch.cat((xx - yy, 2.0 * xy), dim=1) / energy


@torch.no_grad()
def _fit_sdf_weight_and_ink(
    glyph_idx: torch.Tensor,
    atlas_t: torch.Tensor,
    sdf_spread: float,
    work_lab_cells: torch.Tensor,
    disp_lab_cells: torch.Tensor,
    enc_cells: torch.Tensor,
    bg_enc: torch.Tensor,
    *,
    invert_ink: bool = False,
    dominant_commit: float = 0.0,
    strand_gate: torch.Tensor | None = None,
    highlight_gate: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Closed-form foreground-dominant mask-weighted ink fit + bounded 8-way SDF stroke weight."""
    rows, cols = glyph_idx.shape
    cell_sdf = atlas_t[glyph_idx]  # [R, C, h, w]
    mean_work = work_lab_cells.mean((2, 3))  # [R, C, 3]
    bg_L = bg_enc[0]

    candidates = (
        (-0.55, -0.35, -0.15, 0.0, 0.18, 0.35, 0.50, 0.68)
        if sdf_spread > 0.0 else (0.0,)
    )
    best_err = torch.full((rows, cols), float("inf"), device=atlas_t.device)
    best_w = torch.zeros((rows, cols), device=atlas_t.device)
    best_ink = torch.zeros((rows, cols, 3), device=atlas_t.device)

    enc_L = enc_cells[..., 0]
    work_L = work_lab_cells[..., 0]
    disp_L = disp_lab_cells[..., 0]
    disp_ab = disp_lab_cells[..., 1:]

    if dominant_commit > 0.0:
        flat_L = work_L.flatten(2)
        med_L = flat_L.median(dim=2).values
        p86_L = flat_L.quantile(0.86, dim=2)
        boost = torch.zeros_like(med_L)
        if strand_gate is not None:
            boost = torch.maximum(boost, strand_gate)
        if highlight_gate is not None:
            boost = torch.maximum(boost, highlight_gate)
        thresh_L = (med_L + 0.90 * boost * (p86_L - med_L))[:, :, None, None]
        fg_bias = torch.sigmoid((work_L - thresh_L) * 18.0)
        fg_weight = (1.0 - dominant_commit) + dominant_commit * (0.10 + 1.90 * fg_bias)
    else:
        fg_weight = torch.ones_like(work_L)

    for w_val in candidates:
        m = sdf_to_mask(cell_sdf, w_val, spread=sdf_spread) if sdf_spread > 0.0 else cell_sdf
        mw = m * fg_weight
        mw_sum = mw.sum((2, 3))
        m2_sum = (m * mw).sum((2, 3)).clamp_min(1e-4)

        num_L = (mw * (enc_L - bg_L)).sum((2, 3))
        ink_enc_L = (bg_L + num_L / m2_sum).clamp(0.02, 0.98)

        has_ink = (mw_sum > 1e-3)[..., None]
        w_lab = (mw[..., None] * work_lab_cells).sum((2, 3)) / mw_sum.clamp_min(1e-4)[..., None]
        ink_work = torch.where(has_ink, w_lab, mean_work)

        ratio = ink_work[..., 1:] / ink_work[..., :1].clamp_min(0.06)
        ink_enc_ab = ratio * ink_enc_L[..., None]
        ink_enc = torch.cat((ink_enc_L[..., None], ink_enc_ab), dim=-1)

        pix_enc = bg_enc[None, None, None, None, :] + m[..., None] * (
            ink_enc[:, :, None, None, :] - bg_enc[None, None, None, None, :]
        )
        Lp = pix_enc[..., 0].clamp(0.0, 1.0)
        boosted = Lp + 2.0 * Lp * (1.0 - Lp).square()
        Ld = (1.0 - boosted) if invert_ink else boosted
        abd = pix_enc[..., 1:] * (boosted / Lp.clamp_min(1e-5))[..., None]
        err = (
            (Ld - disp_L).square().mean((2, 3))
            + 1.6 * (abd - disp_ab).square().sum(-1).mean((2, 3))
            + 0.0004 * (w_val * w_val)
        )

        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_w = torch.where(better, torch.full_like(best_w, w_val), best_w)
        best_ink = torch.where(better[..., None], ink_enc, best_ink)

    encoded_ink = quantize_playback_lab(best_ink.cpu().numpy())
    weight_np = (
        decode_stroke_weight_u8(encode_stroke_weight_u8(best_w.cpu().numpy()))
        if sdf_spread > 0.0 else None
    )
    return encoded_ink, weight_np


def solve_stroke_graph(
    source_srgb: np.ndarray,
    atlas: np.ndarray,
    *,
    sdf_spread: float = 0.0,
    background: str = "auto",
    original_background: np.ndarray | None = None,
    solve_background: np.ndarray | None = None,
    source_lab: np.ndarray | None = None,
    invert_ink: bool = False,
    coupling: float = 0.34,
    bridge_strength: float = 0.26,
    close_radius: int = 3,
    close_blend: float = 0.90,
    dominant_commit: float = 0.75,
    strand_boost: float = 0.75,
    strand_hysteresis: float = 0.65,
    feature_preserve: float = 0.95,
    pixel_feature_map: bool = True,
    band_weight: float = 3.4,
    weber_norm: bool = True,
    chroma_orient_gain: float = 3.5,
    smooth_flow_gain: float = 0.90,
    color_cov_gain: float = 3.0,
    sdf_interval_cov: bool = True,
    soft_curves: bool = True,
    sweeps: int = 8,
    dither: float = 0.025,
    seed: int = 0,
    device: str = "cpu",
    on_progress: Callable[[float, str], None] | None = None,
    **_unused,
) -> Grid:
    """Solve an SDF-aware connected-stroke 2D MRF on the fixed ASCII grid."""
    if len(atlas) == 0 or len(atlas) > 256:
        raise ValueError("The glyph map supports 1 to 256 characters")
    H, W = source_srgb.shape[:2]
    cell_h, cell_w = atlas.shape[1:]
    rows = (H + cell_h - 1) // cell_h
    cols = (W + cell_w - 1) // cell_w
    pad_h, pad_w = rows * cell_h - H, cols * cell_w - W

    src_lab = srgb_to_oklab(source_srgb) if source_lab is None else np.asarray(source_lab, dtype=np.float32)
    bg_orig = (
        _choose_background(torch.from_numpy(src_lab).permute(2, 0, 1), background).numpy()
        if original_background is None else np.asarray(original_background, dtype=np.float32)
    )
    if solve_background is None:
        bg_encoded = (
            np.zeros(3, dtype=np.float32) if (invert_ink and background == "auto")
            else quantize_playback_lab(
                encode_headroom_lab(bg_orig.reshape(1, 1, 3), invert_ink=invert_ink)[0, 0]
            )
        )
    else:
        bg_encoded = np.asarray(solve_background, dtype=np.float32)

    # In invert_ink mode, flip lightness so ink strokes still target high working L on a dark working bg.
    work_src_lab = src_lab.copy()
    work_bg = bg_orig.copy()
    if invert_ink:
        work_src_lab[..., 0] = 1.0 - work_src_lab[..., 0]
        work_bg[0] = 0.0 if background == "auto" else (1.0 - work_bg[0])
    bg_L = float(work_bg[0])

    padded_work_lab = np.pad(work_src_lab, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_disp_lab = np.pad(src_lab, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    dev = torch.device(device)
    atlas_t = torch.from_numpy(atlas).float().to(dev)
    neutral_masks = sdf_to_mask(atlas, spread=sdf_spread) if sdf_spread > 0.0 else atlas
    masks_t = torch.from_numpy(neutral_masks).float().to(dev)
    thin_masks_t = sdf_to_mask(atlas_t, -0.55, spread=sdf_spread) if sdf_spread > 0.0 else masks_t
    bold_masks_t = sdf_to_mask(atlas_t, 0.68, spread=sdf_spread) if sdf_spread > 0.0 else masks_t
    cov_t = masks_t.mean((1, 2))
    cov_min_t = thin_masks_t.mean((1, 2))
    cov_max_t = bold_masks_t.mean((1, 2))
    eff_cov_t = 0.70 * cov_t + 0.30 * (
        sdf_to_mask(atlas_t, 0.42, spread=sdf_spread).mean((1, 2)) if sdf_spread > 0.0 else cov_t
    )
    g_axial = _glyph_axial(neutral_masks).to(dev)

    if on_progress is not None:
        on_progress(0.15, "Analyzing contour orientation")

    # 1. Chroma-boosted structure tensor on raw working image for crisp & atmospheric tangents.
    raw_lab_t = torch.from_numpy(padded_work_lab).permute(2, 0, 1)[None].to(dev)
    raw_L_t = raw_lab_t[:, :1]
    raw_C_t = raw_lab_t[:, 1:].square().sum(1, keepdim=True).sqrt()
    orient_lab_t = torch.cat((raw_L_t, chroma_orient_gain * raw_lab_t[:, 1:]), dim=1)
    if min(rows * cell_h, cols * cell_w) >= 4:
        gx, gy = gradients(blur(orient_lab_t, max(0.7, cell_w * 0.14)))
        xx = blur((gx * gx).sum(1, keepdim=True), max(1.0, cell_w * 0.35))
        yy = blur((gy * gy).sum(1, keepdim=True), max(1.0, cell_w * 0.35))
        xy = blur((gx * gy).sum(1, keepdim=True), max(1.0, cell_w * 0.35))
        energy = (xx + yy).clamp_min(1e-8)
        tangent_axial = -torch.cat((xx - yy, 2.0 * xy), dim=1) / energy
        coherence = tangent_axial.square().sum(1, keepdim=True).sqrt().clamp(0.0, 1.0)
        edge_mag = energy.sqrt()
        edge_scale = torch.quantile(edge_mag.flatten(), 0.94).clamp_min(1e-4)
        edge_salience = (edge_mag / edge_scale).clamp(0.0, 1.0)
    else:
        tangent_axial = torch.zeros((1, 2, rows * cell_h, cols * cell_w), device=dev)
        coherence = torch.zeros((1, 1, rows * cell_h, cols * cell_w), device=dev)
        edge_salience = torch.zeros((1, 1, rows * cell_h, cols * cell_w), device=dev)

    # 2. Pixel-level focal feature detector (mouths/smiles/eyes/speculars) before closing.
    eff_close_radius = close_radius if min(H, W) >= 32 else 0
    if pixel_feature_map and feature_preserve > 0.0 and min(H, W) >= 16:
        local_L = blur(raw_L_t, 5.5)
        local_C = blur(raw_C_t, 5.5)
        bright_pop = ((raw_L_t - local_L - 0.04) / 0.14).clamp(0.0, 1.0)
        pale_pop = ((local_C - raw_C_t - 0.015) / 0.045).clamp(0.0, 1.0)
        highlight_pix = (bright_pop * (0.35 + 0.65 * pale_pop)).clamp(0.0, 1.0)
        near_highlight = F.max_pool2d(highlight_pix, kernel_size=13, stride=1, padding=6)
        dark_valley = ((local_L - raw_L_t - 0.18) / 0.25).clamp(0.0, 1.0)
        in_bright = ((local_L - bg_L - 0.16) / 0.22).clamp(0.0, 1.0)
        crease_pix = (dark_valley * near_highlight * in_bright).clamp(0.0, 1.0)
        feat_protect_pix = (feature_preserve * F.max_pool2d(
            torch.maximum(crease_pix, 0.7 * highlight_pix * near_highlight),
            kernel_size=5, stride=1, padding=2,
        )).clamp(0.0, 1.0)
        crease_cells = crease_pix[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
        highlight_cells = highlight_pix[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
        crease_gate = (feature_preserve * crease_cells.flatten(2).quantile(0.90, dim=2)).clamp(0.0, 1.0)
        highlight_gate = (feature_preserve * highlight_cells.flatten(2).quantile(0.90, dim=2)).clamp(0.0, 1.0)
        feature_gate = torch.maximum(crease_gate, 0.6 * highlight_gate)
    else:
        feat_protect_pix = torch.zeros_like(raw_L_t)
        crease_gate = torch.zeros((rows, cols), device=dev)
        highlight_gate = torch.zeros((rows, cols), device=dev)
        feature_gate = torch.zeros((rows, cols), device=dev)

    # 3. Opinionated region-completed fill target.
    if eff_close_radius > 0 and close_blend > 0.0:
        closed_lab_t = _oklab_morph_close(raw_lab_t, eff_close_radius)
        eff_close = (close_blend * (1.0 - 0.90 * feat_protect_pix)).clamp(0.0, 1.0)
        fill_lab_t = (1.0 - eff_close) * raw_lab_t + eff_close * closed_lab_t
    else:
        fill_lab_t = raw_lab_t

    L_t = fill_lab_t[:, :1]
    L_smooth = blur(L_t, 1.30) if min(rows * cell_h, cols * cell_w) >= 4 else L_t
    L_crisp = blur(raw_L_t, 0.50) if min(rows * cell_h, cols * cell_w) >= 4 else raw_L_t

    L_cells = L_t[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
    Ls_cells = L_smooth[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
    Lc_cells = L_crisp[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
    cell_L = L_cells.mean((2, 3))
    cell_L_p75 = L_cells.flatten(2).quantile(0.75, dim=2)
    cell_L_p92 = L_cells.flatten(2).quantile(0.92, dim=2)
    tex_smooth = Ls_cells - Ls_cells.mean((2, 3), keepdim=True)
    tex_crisp = Lc_cells - Lc_cells.mean((2, 3), keepdim=True)
    fg_tex_mix = feature_gate[:, :, None, None]
    cell_tex = (1.0 - fg_tex_mix) * tex_smooth + fg_tex_mix * tex_crisp

    cell_edge = edge_salience[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3)
    cell_edge_strength = cell_edge.amax((2, 3))
    cell_tangent = tangent_axial[0].reshape(2, rows, cell_h, cols, cell_w).permute(1, 3, 0, 2, 4).mean((3, 4))
    cell_coh = coherence[0, 0].reshape(rows, cell_h, cols, cell_w).permute(0, 2, 1, 3).mean((2, 3))

    # 4. Coherent bright-strand detection + 1-hop hysteresis.
    denom_L = max(1.0 - bg_L, 0.25)
    strand_above_bg = ((cell_L_p92 - bg_L - 0.04) / denom_L).clamp(0.0, 1.0)
    strand_contrast = ((cell_L_p92 - cell_L - 0.04) / 0.24).clamp(0.0, 1.0)
    strand_core = (strand_above_bg * strand_contrast * (0.35 + 0.65 * cell_coh)).clamp(0.0, 1.0)
    if strand_hysteresis > 0.0 and min(rows, cols) >= 3:
        bright_neighbor = F.max_pool2d(
            ((cell_L_p75 - bg_L - 0.12) / denom_L).clamp(0.0, 1.0)[None, None],
            kernel_size=3, stride=1, padding=1,
        )[0, 0]
        tip_signal = ((cell_L_p92 - bg_L - 0.08) / 0.20).clamp(0.0, 1.0) * strand_contrast
        strand_hyst = (strand_hysteresis * bright_neighbor * tip_signal).clamp(0.0, 1.0)
        strand_gate = (strand_boost * torch.maximum(strand_core, strand_hyst)).clamp(0.0, 1.0)
    else:
        strand_gate = (strand_boost * strand_core).clamp(0.0, 1.0)

    # --- Unary cost U[R, C, K] ---
    bg_ab = torch.tensor(work_bg[1:], dtype=torch.float32, device=dev)
    fill_ab_cells = fill_lab_t[0, 1:].reshape(2, rows, cell_h, cols, cell_w).permute(1, 3, 2, 4, 0)
    cell_ab = fill_ab_cells.mean((2, 3))
    dC2 = (cell_ab - bg_ab[None, None, :]).square().sum(-1)

    eff_L = 0.72 * cell_L + 0.28 * cell_L_p75
    eff_L = eff_L + 0.56 * strand_gate * (cell_L_p92 - eff_L)
    dL = (eff_L - bg_L).clamp_min(0.0)
    color_contrast = (dL.square() + color_cov_gain * dC2).sqrt()
    rel_L = (color_contrast / denom_L).clamp(0.0, 1.0)
    max_cov = float(eff_cov_t.max())
    desired_cov = 0.01 + (max_cov * 0.88) * rel_L.pow(0.80)
    desired_cov = desired_cov * (1.0 - 0.42 * crease_gate)
    if dither:
        bn = torch.from_numpy(blue_noise((rows, cols), seed)).to(dev)
        desired_cov = (desired_cov + dither * (bn - 0.5)).clamp(0.0, max_cov)

    if weber_norm:
        cov_scale = (desired_cov[:, :, None] + 0.07)
        if sdf_interval_cov and sdf_spread > 0.0:
            des = desired_cov[:, :, None]
            under = (cov_min_t[None, None, :] - des).clamp_min(0.0)
            over = (des - cov_max_t[None, None, :]).clamp_min(0.0)
            cov_cost = (1.35 * (under + over).square() + 0.22 * (des - eff_cov_t[None, None, :]).square()) / cov_scale
        else:
            cov_cost = 1.15 * (desired_cov[:, :, None] - eff_cov_t[None, None, :]).square() / cov_scale
    else:
        cov_cost = 2.6 * (desired_cov[:, :, None] - eff_cov_t[None, None, :]).square()

    band_cost = 0.0
    if band_weight > 0.0 and cell_h >= 6:
        b1 = cell_h // 3
        b2 = (2 * cell_h) // 3
        eff_masks_t = 0.70 * masks_t + 0.30 * bold_masks_t
        g_bands = torch.stack((
            eff_masks_t[:, :b1, :].mean((1, 2)),
            eff_masks_t[:, b1:b2, :].mean((1, 2)),
            eff_masks_t[:, b2:, :].mean((1, 2)),
        ), dim=-1)
        g_bands_min = torch.stack((
            thin_masks_t[:, :b1, :].mean((1, 2)),
            thin_masks_t[:, b1:b2, :].mean((1, 2)),
            thin_masks_t[:, b2:, :].mean((1, 2)),
        ), dim=-1)
        g_bands_max = torch.stack((
            bold_masks_t[:, :b1, :].mean((1, 2)),
            bold_masks_t[:, b1:b2, :].mean((1, 2)),
            bold_masks_t[:, b2:, :].mean((1, 2)),
        ), dim=-1)
        img_L_bands = torch.stack((
            L_cells[:, :, :b1, :].mean((2, 3)),
            L_cells[:, :, b1:b2, :].mean((2, 3)),
            L_cells[:, :, b2:, :].mean((2, 3)),
        ), dim=-1)
        img_ab_bands = torch.stack((
            fill_ab_cells[:, :, :b1, :, :].mean((2, 3)),
            fill_ab_cells[:, :, b1:b2, :, :].mean((2, 3)),
            fill_ab_cells[:, :, b2:, :, :].mean((2, 3)),
        ), dim=-2)
        dL_b = (img_L_bands - bg_L).clamp_min(0.0)
        dC2_b = (img_ab_bands - bg_ab[None, None, None, :]).square().sum(-1)
        rel_bands = ((dL_b.square() + color_cov_gain * dC2_b).sqrt() / denom_L).clamp(0.0, 1.0)
        des_bands = (0.01 + (max_cov * 0.88) * rel_bands.pow(0.80)) * (1.0 - 0.42 * crease_gate[:, :, None])
        if weber_norm:
            if sdf_interval_cov and sdf_spread > 0.0:
                db = des_bands[:, :, None, :]
                b_out = ((g_bands_min[None, None, :, :] - db).clamp_min(0.0)
                         + (db - g_bands_max[None, None, :, :]).clamp_min(0.0)).square()
                b_neu = (db - g_bands[None, None, :, :]).square()
                band_cost = (band_weight * 0.45) * ((1.20 * b_out + 0.22 * b_neu) / (db + 0.07)).mean(-1)
            else:
                band_err = ((des_bands[:, :, None, :] - g_bands[None, None, :, :]).square()
                            / (des_bands[:, :, None, :] + 0.07)).mean(-1)
                band_cost = (band_weight * 0.42) * band_err
        else:
            band_cost = band_weight * (des_bands[:, :, None, :] - g_bands[None, None, :, :]).square().mean(-1)

    smooth_masks = blur(masks_t[:, None], 1.25)[:, 0] if min(cell_h, cell_w) >= 4 else masks_t
    g_tex = smooth_masks - smooth_masks.mean((1, 2), keepdim=True)
    contrast = (0.45 + 0.55 * eff_L - bg_L).clamp_min(0.15)
    ct2 = cell_tex.square().mean((2, 3))[:, :, None]
    gt2 = g_tex.square().mean((1, 2))[None, None, :]
    cross = torch.einsum("rchw,khw->rck", cell_tex, g_tex) / (cell_h * cell_w)
    tex_cost = ct2 + (contrast[:, :, None].square()) * gt2 - 2.0 * contrast[:, :, None] * cross

    t_norm = cell_tangent.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    unit_tangent = cell_tangent / t_norm
    orient_dot = torch.einsum("rcd,kd->rck", unit_tangent, g_axial)
    grad_present = ((cell_edge_strength - 0.025) / 0.08).clamp(0.0, 1.0)
    smooth_flow = smooth_flow_gain * grad_present * cell_coh.pow(1.4)
    flow_salience = torch.maximum(cell_edge_strength * cell_coh, smooth_flow)
    orient_weight = torch.maximum(flow_salience, strand_gate)[:, :, None]
    orient_cost = 0.42 * orient_weight * (1.0 - orient_dot)

    unary = cov_cost + band_cost + (1.65 + 1.40 * feature_gate[:, :, None]) * tex_cost + orient_cost

    if on_progress is not None:
        on_progress(0.45, "Solving connected-stroke graph")

    # --- Tangent-gated 4-color checkerboard MRF ---
    compat = {name: mat.to(dev) for name, mat in _seam_compatibilities(neutral_masks, soft_curves=soft_curves).items()}
    contour_gate = torch.maximum((0.03 + 0.97 * flow_salience), 0.90 * strand_gate).clamp(0.0, 1.0)
    c2 = unit_tangent[..., 0]
    s2 = unit_tangent[..., 1]
    dir_h = contour_gate * (0.5 * (1.0 + c2)).square()
    dir_v = contour_gate * (0.5 * (1.0 - c2)).square()
    diag_pow = 1.30 if soft_curves else 2.0
    dir_dr = contour_gate * (0.5 * (1.0 + s2)).clamp_min(0.0).pow(diag_pow)
    dir_dl = contour_gate * (0.5 * (1.0 - s2)).clamp_min(0.0).pow(diag_pow)

    labels = unary.argmin(dim=-1)
    if (coupling > 0.0 or bridge_strength > 0.0) and sweeps > 0 and len(atlas) > 1:
        H_mat = compat["horiz"]
        V_mat = compat["vert"]
        DR_mat = compat["diag_dr"]
        DL_mat = compat["diag_dl"]
        bridge_scale = bridge_strength / max(coupling, 1e-4)
        for _ in range(sweeps):
            for pr in (0, 1):
                for pc in (0, 1):
                    bonus = torch.zeros_like(unary)
                    if cols > 1:
                        w_lr = 0.5 * (dir_h[:, :-1] + dir_h[:, 1:])
                        b_left = H_mat[labels[:, :-1]]
                        b_right = H_mat.T[labels[:, 1:]]
                        bonus[:, 1:] += w_lr[:, :, None] * b_left
                        bonus[:, :-1] += w_lr[:, :, None] * b_right
                        if bridge_strength > 0.0 and cols > 2:
                            both_h = (b_left[:, :-1] * b_right[:, 1:]).sqrt()
                            bonus[:, 1:-1] += bridge_scale * dir_h[:, 1:-1, None] * both_h
                    if rows > 1:
                        w_ud = 0.5 * (dir_v[:-1, :] + dir_v[1:, :])
                        b_up = V_mat[labels[:-1, :]]
                        b_down = V_mat.T[labels[1:, :]]
                        bonus[1:, :] += w_ud[:, :, None] * b_up
                        bonus[:-1, :] += w_ud[:, :, None] * b_down
                        if bridge_strength > 0.0 and rows > 2:
                            both_v = (b_up[:-1, :] * b_down[1:, :]).sqrt()
                            bonus[1:-1, :] += bridge_scale * dir_v[1:-1, :, None] * both_v
                    if rows > 1 and cols > 1:
                        w_dr = 0.5 * (dir_dr[:-1, :-1] + dir_dr[1:, 1:])
                        b_ul = DR_mat[labels[:-1, :-1]]
                        b_lr = DR_mat.T[labels[1:, 1:]]
                        bonus[1:, 1:] += 0.9 * w_dr[:, :, None] * b_ul
                        bonus[:-1, :-1] += 0.9 * w_dr[:, :, None] * b_lr
                        if bridge_strength > 0.0 and rows > 2 and cols > 2:
                            both_dr = (b_ul[:-1, :-1] * b_lr[1:, 1:]).sqrt()
                            bonus[1:-1, 1:-1] += bridge_scale * dir_dr[1:-1, 1:-1, None] * both_dr

                        w_dl = 0.5 * (dir_dl[:-1, 1:] + dir_dl[1:, :-1])
                        b_ur = DL_mat[labels[:-1, 1:]]
                        b_ll = DL_mat.T[labels[1:, :-1]]
                        bonus[1:, :-1] += 0.9 * w_dl[:, :, None] * b_ur
                        bonus[:-1, 1:] += 0.9 * w_dl[:, :, None] * b_ll
                        if bridge_strength > 0.0 and rows > 2 and cols > 2:
                            both_dl = (b_ur[:-1, 1:] * b_ll[1:, :-1]).sqrt()
                            bonus[1:-1, 1:-1] += bridge_scale * dir_dl[1:-1, 1:-1, None] * both_dl

                    total = unary - coupling * bonus
                    proposed = total.argmin(dim=-1)
                    labels[pr::2, pc::2] = proposed[pr::2, pc::2]

    glyph_np = labels.cpu().numpy().astype(np.uint8)

    if on_progress is not None:
        on_progress(0.80, "Fitting SDF stroke weights and ink")

    # --- Foreground-dominant mask-weighted ink fit + bounded 8-way SDF stroke weight ---
    fill_work_np = fill_lab_t[0].permute(1, 2, 0).cpu().numpy()
    fill_disp_np = fill_work_np.copy()
    if invert_ink:
        fill_disp_np[..., 0] = 1.0 - fill_disp_np[..., 0]
    fill_enc_np = encode_headroom_lab(fill_disp_np, invert_ink=invert_ink)

    work_cells_t = torch.from_numpy(fill_work_np).float().to(dev).reshape(
        rows, cell_h, cols, cell_w, 3).permute(0, 2, 1, 3, 4)
    disp_cells_t = torch.from_numpy(fill_disp_np).float().to(dev).reshape(
        rows, cell_h, cols, cell_w, 3).permute(0, 2, 1, 3, 4)
    enc_cells_t = torch.from_numpy(fill_enc_np).float().to(dev).reshape(
        rows, cell_h, cols, cell_w, 3).permute(0, 2, 1, 3, 4)
    bg_enc_t = torch.from_numpy(bg_encoded).float().to(dev)
    encoded_ink, weight_np = _fit_sdf_weight_and_ink(
        labels, atlas_t, sdf_spread, work_cells_t, disp_cells_t, enc_cells_t, bg_enc_t,
        invert_ink=invert_ink,
        dominant_commit=dominant_commit,
        strand_gate=strand_gate,
        highlight_gate=highlight_gate,
    )

    if on_progress is not None:
        on_progress(1.0, "Finished connected-stroke graph")

    return Grid(
        glyph=glyph_np,
        color=encoded_ink,
        background=bg_encoded,
        weight=weight_np,
    )
