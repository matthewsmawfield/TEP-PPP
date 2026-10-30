#!/usr/bin/env python3
"""
STEP 45 -- Temporal T4: time-resolved co-visibility regression.

The cross-sectional T4 (Step 13) is limited by the collinearity between
great-circle distance and mean co-visibility (r ~ -0.93) on ground-based
receiver networks.  This step exploits a feature the cross-sectional test
cannot: distance is FIXED for each station pair, while co-visibility
VARIES from epoch to epoch as satellites rise and set.

Method (two-stage regression):

  1. For each day and pair (i,j), compute per-epoch normalised residual
     products  p_{ij}(e) = r_i(e) * r_j(e) / (sigma_i * sigma_j)  at 48
     epochs (~30 min cadence), and per-epoch co-visibility J_{ij}(e)
     (Jaccard overlap of visible-satellite sets).  Co-visibility is
     computed from the day's REAL SP3 ephemerides at the same UTC
     seconds-of-day as the residual grid -- the nominal model
     constellation used previously is arbitrarily phased relative to
     Earth, which scrambles the per-epoch J and attenuates the
     common-view slope toward zero (measured: per-epoch J correlation
     with the real constellation ~0.17, kappa bias ~4x).

  2. Pooled regression with pair fixed effects:
        p_{ij}(e) = alpha_{ij} + kappa * J_{ij}(e) + epsilon
     kappa is the common-view effect, a SINGLE pooled slope estimated from
     within-pair variation in J across ALL pair-days (uncorrelated with
     distance by construction).  alpha_{ij} = mean_e(p - kappa*J) is the
     pair intercept -- the correlation extrapolated to zero co-visibility.
     An earlier implementation fitted beta PER PAIR from 48 epochs; that
     per-pair slope is almost pure noise (se ~ 1), and alpha_ij then
     carries -err_beta * Jbar_ij -- an errors-in-variables contaminant
     aligned with distance through the d-Jbar collinearity that attenuates
     the stage-2 partial correlation toward zero.  Pairs whose per-epoch
     co-visibility is identically ~0 (the cleanest, common-view-free
     regime) are INCLUDED with alpha = mean(p) rather than discarded.

  3. Stage 2: test whether alpha_{ij} shows distance structure after
     controlling for mean co-visibility Jbar_{ij}.  Fit
        alpha_{ij} = a + b * log(d_{ij}) + c * Jbar_{ij} + eta
     and report the partial correlation of alpha with log(d) | Jbar.
     Also reported: the same partial correlation with cos(Delta-lon)
     added to the control set, and partial corr(alpha, cos Delta-lon |
     Jbar) -- a longitude-phased (local-solar-time) systematic is the
     dominant non-distance structure in the broadcast product and must
     be separated, not mistaken for the field.

  4. Also fit exp(-d/lambda) to alpha_{ij} vs d and report lambda, R^2,
     and the frozen out-of-sample R^2 gain of adding the distance term to
     the co-visibility base model (day-folded, Step 13's functional form).
     The scored distance fit is the SEPARATED one: alpha is residualised on
     [Jbar, cos(Delta-lon)] before binning (the same control set the
     partial correlation uses), because the raw curve attributes neither
     the Jbar-aligned component (corr(d, Jbar) ~ -0.95) nor the
     longitude-phased solar-time systematic; the epoch-shuffle null is
     residualised identically, which also removes the kappa_null*Jbar(d)
     leakage that heavy-tailed the uncontrolled null R^2 distribution.

  5. Permutation null: shuffle epoch labels within each station (destroying
     cross-station correlation while preserving per-station distribution),
     recompute kappa, alpha_{ij} and the distance fit.  Report the sigma
     separation.

Key prediction: if TEP is real, alpha_{ij} (the non-common-view component)
should show distance structure with lambda in the 1,000-5,000 km band.  If
common-view is the only mechanism, alpha_{ij} should be ~0 for all pairs.
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402
from scripts.steps.step_13_commonview import (  # noqa: E402
    gps_constellation, visible_mask)

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260901)
C_M_PER_NS = 0.299792458
N_EPOCHS = 48
ELEV_MASK_RAD = np.radians(10.0)


def _clean_clk(v, tt):
    n = min(len(v), len(tt))
    v, tt = np.asarray(v, float)[:n], np.asarray(tt, float)[:n]
    if np.mean(np.isfinite(v)) < 0.9:
        return None
    g = np.isfinite(v)
    v = np.interp(np.arange(n), np.flatnonzero(g), v[g])
    r = v - np.polyval(np.polyfit(tt, v, 1), tt)
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    dt = float(np.nanmedian(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 30.0
    b = ts.bandpass(np.clip(r, -8 * s, 8 * s), dt) * C_M_PER_NS
    return b


def _resample_to_epochs(series, tt, grid):
    """Resample a cleaned residual series onto the fixed epoch grid (sod)."""
    n = min(len(series), len(tt))
    series, tt = np.asarray(series, float)[:n], np.asarray(tt, float)[:n]
    if n < len(grid):
        return None
    g = np.isfinite(series)
    if g.sum() < len(grid) // 2:
        return None
    if tt[g][0] > grid[0] + 3600 or tt[g][-1] < grid[-1] - 3600:
        return None
    return np.interp(grid, tt[g], series[g])


def load_real_visibility(date_tag, sts, epochs_s, systems=("G",)):
    """(visible, sin_elev) per (epoch, station, satellite) from the day's real
    SP3 ephemeris, evaluated at the UTC seconds-of-day in `epochs_s`.

    Returns None if the SP3 product is unavailable.  The satellite set is the
    requested constellation prefix(es); 'G' alone matches the broadcast-SPP
    product (GPS L1), and is used for the precise arm too so both arms share
    the same co-visibility definition.
    """
    year, doy = int(date_tag[:4]), int(date_tag[4:])
    prns = [f"{c}{i:02d}" for c in systems for i in range(1, 40)]
    meo = td.load_meo_series(year, doy, prns)
    if not meo:
        return None
    sat_list = sorted(meo)
    P = np.full((len(epochs_s), len(sat_list), 3), np.nan)
    for k, prn in enumerate(sat_list):
        t = meo[prn]["t"]
        pos = np.asarray(meo[prn]["station"].trajectory_km, float) * 1000.0
        for c in range(3):
            P[:, k, c] = np.interp(epochs_s, t, pos[:, c],
                                   left=np.nan, right=np.nan)
    sin_e = np.full((len(epochs_s), len(sts), len(sat_list)), np.nan)
    vis = np.zeros((len(epochs_s), len(sts), len(sat_list)), bool)
    for i, st in enumerate(sts):
        up = st.ecef_m / np.linalg.norm(st.ecef_m)
        los = P - st.ecef_m[None, None, :]
        ln = np.linalg.norm(los, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.einsum("esk,k->es", los, up) / np.clip(ln, 1e-9, None)
        sin_e[:, i] = np.clip(se, -1, 1)
        vis[:, i] = np.isfinite(ln) & (np.arcsin(np.clip(se, -1, 1)) > ELEV_MASK_RAD)
    return vis, sin_e


def _per_epoch_covisibility_from_vis(vis):
    """Jaccard J (n_pairs, n_epochs) + pair index from a visibility cube."""
    n_ep, n, _ = vis.shape
    pairs = list(itertools.combinations(range(n), 2))
    J = np.zeros((len(pairs), n_ep))
    for pi, (i, j) in enumerate(pairs):
        both = vis[:, i] & vis[:, j]
        either = vis[:, i] | vis[:, j]
        J[pi] = both.sum(1).astype(float) / np.clip(either.sum(1), 1, None)
    return J, pairs


def _per_epoch_covisibility(stations, epochs_s):
    n = len(stations)
    vis = np.zeros((len(epochs_s), n, 24), bool)
    for e, t in enumerate(epochs_s):
        sat = gps_constellation(float(t))
        for i, st in enumerate(stations):
            vis[e, i] = visible_mask(sat, st.ecef_m)
    pairs = list(itertools.combinations(range(n), 2))
    J = np.zeros((len(pairs), len(epochs_s)))
    for pi, (i, j) in enumerate(pairs):
        for e in range(len(epochs_s)):
            inter = float(np.sum(vis[e, i] & vis[e, j]))
            union = float(np.sum(vis[e, i] | vis[e, j]))
            J[pi, e] = inter / max(union, 1.0)
    return J, pairs


def _partial_corr(x, y, z):
    with np.errstate(all="ignore"):
        def resid(a, b):
            bs = (b - b.mean()) / (b.std() + 1e-9)
            A = np.column_stack([np.ones_like(bs), bs])
            beta = np.linalg.solve(A.T @ A + 1e-6 * np.eye(2), A.T @ a)
            return a - A @ beta
        rx, ry = resid(np.asarray(x, float), np.asarray(z, float)), \
                 resid(np.asarray(y, float), np.asarray(z, float))
        return float(np.corrcoef(rx, ry)[0, 1])


def _precompute_day(s, epochs_s, date_tag=None, systems=("G",)):
    """Clean, resample, and compute per-epoch products and co-visibility for one day.

    Returns (M_norm, J_pe, pairs_idx, nms_c, sts_c) or None.
    M_norm is (n_epochs, n_sta) normalised clock residuals.
    J_pe is (n_pairs, n_epochs) per-epoch co-visibility -- from the day's real
    SP3 ephemerides when `date_tag` is given and the product exists, else the
    nominal 24-satellite model.
    """
    nms = list(s)
    if len(nms) < 8:
        return None
    cleaned = {}
    for n in nms:
        c = _clean_clk(s[n]["resid"], s[n]["t"])
        if c is None:
            continue
        r = _resample_to_epochs(c, s[n]["t"], epochs_s)
        if r is not None:
            cleaned[n] = r
    if len(cleaned) < 8:
        return None
    nms_c = list(cleaned)
    sts_c = [s[n]["station"] for n in nms_c]
    M = np.column_stack([cleaned[n] for n in nms_c])
    sd = M.std(0); sd[sd < 1e-9] = np.nan
    M_norm = (M - M.mean(0)) / sd
    if date_tag is not None:
        rv = load_real_visibility(date_tag, sts_c, epochs_s, systems)
        if rv is not None:
            J_pe, pairs_idx = _per_epoch_covisibility_from_vis(rv[0])
            return M_norm, J_pe, pairs_idx, nms_c, sts_c
    J_pe, pairs_idx = _per_epoch_covisibility(sts_c, epochs_s)
    return M_norm, J_pe, pairs_idx, nms_c, sts_c


def _pair_geom(sts, pairs_idx):
    """dict pi -> (dist_km, |delta-lon| rad) for the pair index list."""
    lon = np.array([np.arctan2(st.ecef_m[1], st.ecef_m[0]) for st in sts])
    out = {}
    for pi, (i, j) in enumerate(pairs_idx):
        dl = abs(lon[i] - lon[j])
        out[pi] = (td.great_circle_km(sts[i], sts[j]), min(dl, 2 * np.pi - dl))
    return out


def _pair_table(M_norm, J_pe, pairs_idx, geom):
    """Per pair-day records (p_v, j_v, dist, dlon) used by stage 1/2."""
    out = []
    for pi, (i, j) in enumerate(pairs_idx):
        dist, dlon = geom[pi]
        if dist < 100 or dist > 20000:
            continue
        p = M_norm[:, i] * M_norm[:, j]
        jv = J_pe[pi]
        g = np.isfinite(p) & np.isfinite(jv)
        if g.sum() < 10:
            continue
        out.append((p[g], jv[g], dist, dlon))
    return out


def _pooled_kappa(day_tables):
    """Single common-view slope with pair fixed effects:

        kappa = sum_pd sum_e (J - Jbar_pd)(p - pbar_pd) / sum_pd sum_e (J - Jbar_pd)^2

    Pooled over all pair-days.  This is the stage-1 the method describes --
    one global common-view coefficient identified from within-pair J
    variation, not a noisy per-pair slope.
    """
    num = den = 0.0
    for p_v, j_v, *_ in day_tables:
        jc = j_v - j_v.mean()
        num += float(np.sum(jc * (p_v - p_v.mean())))
        den += float(np.sum(jc * jc))
    return num / den if den > 1e-12 else 0.0


def _alphas(day_tables, kappa):
    """alpha_pd = mean_e(p - kappa*J), the non-common-view intercept.

    Zero-variance-J pair-days (the common-view-free regime) are kept:
    their alpha is just mean(p) since kappa*J ~ 0 there.

    Note: with a POOLED kappa the correction term is kappa * Jbar_pd, a
    linear function of the stage-2 control Jbar, so the Jbar-controlled
    partial correlation of alpha is identical to that of the raw pair mean
    product -- kappa exercises only the un-binned distance fit (lambda, R^2,
    where no Jbar control is applied) and the epoch-shuffle nulls (through
    the recomputed null kappa).  Heterogeneous-kappa variants (per-pair,
    distance-binned) are NOT absorbed this way, which is what the step_45b
    spec battery measures.
    """
    a, d, jb, dl = [], [], [], []
    for p_v, j_v, dist, dlon in day_tables:
        a.append(float(p_v.mean() - kappa * j_v.mean()))
        d.append(dist); jb.append(float(j_v.mean())); dl.append(dlon)
    return np.array(a), np.array(d), np.array(jb), np.array(dl)


def _resid_multi(y, Z):
    """Residual of y on design Z (intercept added, columns standardised)."""
    Z = np.asarray(Z, float)
    Z = Z if Z.ndim == 2 else Z.reshape(-1, 1)
    Zs = (Z - Z.mean(0)) / np.clip(Z.std(0), 1e-9, None)
    A = np.column_stack([np.ones(len(y)), Zs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _binned_distance_fit(alpha, dist):
    """exp(-d/lam) fit to distance-binned alpha -- the stage-2 shape test."""
    ctr, mn, sem, cnt = ts.bin_by_distance(
        dist, alpha, n_bins=15, dmin=100.0,
        dmax=np.percentile(dist, 99), min_count=10)
    return ts.fit_correlation_length(
        ctr, mn, weight=1.0 / np.clip(sem, 1e-4, None) ** 2)


def _fit_distance(alpha, dist, jbar, dlon=None):
    """Stage 2: partial correlations and the binned distance fit to alpha."""
    pc = _partial_corr(alpha, np.log(dist + 1), jbar)
    pc_cov = _partial_corr(alpha, jbar, np.log(dist + 1))
    pc_dlon = np.nan
    pc_ctrl = np.nan
    if dlon is not None:
        pc_dlon = _partial_corr(alpha, np.cos(dlon), jbar)
        ra = _resid_multi(alpha, np.column_stack([jbar, np.cos(dlon)]))
        rl = _resid_multi(np.log(dist + 1), np.column_stack([jbar, np.cos(dlon)]))
        pc_ctrl = float(np.corrcoef(ra, rl)[0, 1]) if ra.std() > 0 and rl.std() > 0 else np.nan
    fit = _binned_distance_fit(alpha, dist)
    return pc, pc_cov, pc_dlon, pc_ctrl, fit


def _oos_r2_gain(alpha, dist, jbar, dlon, day_idx, lam=4000.0):
    """Frozen OOS R^2 gain: base = [Jbar, Jbar^2] vs base + exp(-d/lam),
    day-folded.  Also reported with cos(Delta-lon) added to the base.
    Returns (gain, gain_with_dlon_base)."""
    days = np.unique(day_idx)
    if len(days) < 2:
        return np.nan, np.nan
    feat = np.exp(-dist / lam)
    gains, gains_dl = [], []
    for dval in days:
        te = day_idx == dval
        tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        for extra, acc in ((False, gains), (True, gains_dl)):
            Xb = [jbar, jbar ** 2] + ([np.cos(dlon)] if extra else [])
            Xa = Xb + [feat]
            r2b = _oos_r2_one(Xb, alpha, tr, te)
            r2a = _oos_r2_one(Xa, alpha, tr, te)
            if np.isfinite(r2b) and np.isfinite(r2a):
                acc.append(r2a - r2b)
    return (float(np.mean(gains)) if gains else np.nan,
            float(np.mean(gains_dl)) if gains_dl else np.nan)


def _oos_r2_one(cols, y, tr, te):
    X = np.column_stack(cols)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
    pred = A[te] @ beta
    ss_res = float(np.sum((y[te] - pred) ** 2))
    ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def run_temporal_t4(product, date_tags, label, n_null=200, max_stations=None,
                    systems=("G",)):
    """Two-stage temporal T4 regression on a multi-day product."""
    days = list(td.load_multi_day(product, date_tags))
    if not days:
        return {"label": label, "n_days": 0, "error": "no data"}

    # UTC seconds-of-day, aligned with the residual grid and the SP3 epoch axis
    epochs_s = np.linspace(0.0, 86400.0, N_EPOCHS, endpoint=False)

    # Precompute all days once.  Co-visibility uses the day's real SP3
    # ephemerides (load_real_visibility) when available; the subsample path
    # below recomputes J from the same source for the kept stations.
    precomputed = []
    covis_source = "sp3"
    for dt_tag, s in days:
        pc = _precompute_day(s, epochs_s, date_tag=dt_tag, systems=systems)
        if pc is not None:
            # Optionally subsample stations for very large networks
            if max_stations and pc[0].shape[1] > max_stations:
                M_norm, J_pe, pairs_idx, nms_c, sts_c = pc
                idx = RNG.choice(len(nms_c), max_stations, replace=False)
                idx.sort()
                M_sub = M_norm[:, idx]
                nms_sub = [nms_c[k] for k in idx]
                sts_sub = [sts_c[k] for k in idx]
                # Recompute pairs and co-visibility for the subsample, from
                # the same ephemeris source as the full set.
                M_norm2 = (M_sub - M_sub.mean(0)) / (M_sub.std(0) + 1e-9)
                rv = load_real_visibility(dt_tag, sts_sub, epochs_s, systems)
                if rv is not None:
                    J_sub, pairs_sub = _per_epoch_covisibility_from_vis(rv[0])
                else:
                    J_sub, pairs_sub = _per_epoch_covisibility(sts_sub, epochs_s)
                pc = (M_norm2, J_sub, pairs_sub, nms_sub, sts_sub)
            precomputed.append(pc)
        else:
            covis_source = "mixed_or_nominal"

    if not precomputed:
        return {"label": label, "n_days": 0, "n_pairs": 0, "error": "no valid days"}

    # Stage 1: pooled common-view slope (pair fixed effects), then alpha.
    day_tables = []
    day_idx = []
    geoms = []
    for di, (M_norm, J_pe, pairs_idx, nms_c, sts_c) in enumerate(precomputed):
        geo = _pair_geom(sts_c, pairs_idx)
        geoms.append(geo)
        tbl = _pair_table(M_norm, J_pe, pairs_idx, geo)
        day_tables.extend(tbl)
        day_idx.extend([di] * len(tbl))

    if len(day_tables) < 100:
        return {"label": label, "n_days": len(precomputed), "n_pairs": len(day_tables),
                "error": "insufficient pairs"}

    day_idx = np.array(day_idx)
    n_zero_j = int(sum(1 for _, jv, *_ in day_tables
                     if (jv - jv.mean()).std() < 1e-6))
    kappa = _pooled_kappa(day_tables)
    alpha, dist, jbar, dlon = _alphas(day_tables, kappa)

    pc, pc_cov, pc_dlon, pc_ctrl, fit = _fit_distance(alpha, dist, jbar, dlon)
    oos_gain, oos_gain_dl = _oos_r2_gain(alpha, dist, jbar, dlon, day_idx)

    # The SEPARATED binned distance fit is the quantity distance_fit_
    # significant must judge.  The raw alpha-vs-distance curve cannot
    # attribute its structure: corr(d, Jbar) ~ -0.95 makes the smooth
    # component ambiguous, and the cos-dlon solar-time systematic (the
    # dominant measured non-distance structure, PC(alpha, cos dlon | Jbar)
    # ~ +0.69 on broadcast) enters the binned means uncontrolled.  Alpha is
    # therefore residualised on [Jbar, cos(dlon)] -- exactly the control set
    # the partial correlation already uses (pc_ctrl) -- before binning, and
    # the SAME residualisation is applied inside the epoch-shuffle null.
    # An unresidualised null fit is also degenerate in a second way: the
    # null kappa is nonzero at each draw, so alpha_null = pbar_shuf -
    # kappa_null*Jbar retains a random kappa*Jbar(d) distance-aligned term
    # that the uncontrolled binned fit scores as spurious R^2 (the measured
    # null-R^2 distribution is heavy-tailed, p95 ~ 0.8 on broadcast), which
    # silently deflated the sigma of the real fit.
    a_sep = _resid_multi(alpha, np.column_stack([jbar, np.cos(dlon)]))
    fit_sep = _binned_distance_fit(a_sep, dist)

    # Permutation null: shuffle epoch labels within each station; the pooled
    # kappa is recomputed per shuffle so the null exercises the full estimator.
    null_pcs = []
    null_lambdas = []
    null_r2s = []
    null_pc_ctrl = []
    null_sep_lambdas = []
    null_sep_r2s = []
    for _ in range(n_null):
        nt = []
        for di, (M_norm, J_pe, pairs_idx, nms_c, sts_c) in enumerate(precomputed):
            M_shuf = M_norm.copy()
            for col in range(M_shuf.shape[1]):
                RNG.shuffle(M_shuf[:, col])
            nt.extend(_pair_table(M_shuf, J_pe, pairs_idx, geoms[di]))
        if len(nt) < 100:
            continue
        kn = _pooled_kappa(nt)
        na, nd, nj, nl = _alphas(nt, kn)
        npc, _, _, npc_c, nfit = _fit_distance(na, nd, nj, nl)
        null_pcs.append(npc)
        if np.isfinite(npc_c):
            null_pc_ctrl.append(npc_c)
        if np.isfinite(nfit["lam"]):
            null_lambdas.append(nfit["lam"])
            null_r2s.append(nfit["R2"])
        nfit_sep = _binned_distance_fit(
            _resid_multi(na, np.column_stack([nj, np.cos(nl)])), nd)
        if np.isfinite(nfit_sep["lam"]):
            null_sep_lambdas.append(nfit_sep["lam"])
            null_sep_r2s.append(nfit_sep["R2"])

    null_pcs = np.array(null_pcs)
    null_lambdas = np.array(null_lambdas)
    null_r2s = np.array(null_r2s)
    null_pc_ctrl = np.array(null_pc_ctrl)
    null_sep_lambdas = np.array(null_sep_lambdas)
    null_sep_r2s = np.array(null_sep_r2s)

    if len(null_pcs) > 5:
        pc_sigma = float(abs(pc - np.mean(null_pcs)) / (np.std(null_pcs) + 1e-9))
        pc_p_value = float(np.mean(np.abs(null_pcs) >= abs(pc)))
    else:
        pc_sigma = float("nan")
        pc_p_value = float("nan")

    if len(null_r2s) > 5:
        r2_sigma = float(abs(fit["R2"] - np.mean(null_r2s)) / (np.std(null_r2s) + 1e-9))
        r2_p_value = float(np.mean(np.abs(null_r2s) >= abs(fit["R2"])))
    else:
        r2_sigma = float("nan")
        r2_p_value = float("nan")

    if len(null_sep_r2s) > 5:
        r2_sep_sigma = float(
            abs(fit_sep["R2"] - np.mean(null_sep_r2s))
            / (np.std(null_sep_r2s) + 1e-9))
        r2_sep_p_value = float(
            np.mean(np.abs(null_sep_r2s) >= abs(fit_sep["R2"])))
    else:
        r2_sep_sigma = float("nan")
        r2_sep_p_value = float("nan")

    # T4 temporal verdict -- two SEPARATE conditions, reported independently
    # so the step's own JSON cannot be read as a clean common-view separation
    # when it is not one:
    #
    #   distance_fit_significant : the binned-distance fit to the SEPARATED
    #       alpha (residualised on [Jbar, cos(dlon)] before binning, the same
    #       control set as the partial correlation) is significant vs the
    #       identically residualised epoch-shuffle null (R^2 sigma > 3),
    #       recovers a TEP-band lambda, and R^2 > 0.2.  Because both the real
    #       and null fits are evaluated on the residualised quantity, the
    #       kappa_null*Jbar(d) leakage that inflated the uncontrolled null
    #       R^2 distribution is removed by construction, and the surviving
    #       curve is distance structure that NEITHER co-visibility NOR the
    #       longitude-phased solar-time systematic can produce.  The raw
    #       (uncontrolled) fit is still reported descriptively below.
    #
    #   common_view_separated : the Jbar-controlled partial correlation of
    #       alpha with log(d) is itself significant (|PC| > 0.10 and PC sigma
    #       > 3 vs the shuffle null).  THIS is the condition that would show
    #       the distance structure survives co-visibility control.
    #
    #   frozen_S2 : the pre-registered criterion, |PC| > 0.15 AND OOS R^2 gain
    #       > 0.02, now evaluated literally (the OOS gain was previously only
    #       quoted in the criterion text and never computed here).
    #
    # T4_temporal_pass requires distance_fit_significant AND
    # common_view_separated.  The audit consumes the flags separately.
    r2_significant = bool(np.isfinite(r2_sep_sigma) and r2_sep_sigma > 3.0)
    lambda_in_band = bool(
        np.isfinite(fit_sep["lam"]) and 500 < fit_sep["lam"] < 20000)
    distance_fit_significant = bool(
        r2_significant and lambda_in_band and fit_sep["R2"] > 0.2)
    common_view_separated = bool(
        np.isfinite(pc_sigma) and pc_sigma > 3.0 and abs(pc) > 0.10)
    passes = bool(distance_fit_significant and common_view_separated)
    frozen_s2 = bool(abs(pc) > 0.15 and np.isfinite(oos_gain) and oos_gain > 0.02)

    print(f"\n[{label}] {len(precomputed)} days, {len(alpha):,} pair-days "
          f"({n_zero_j} zero-co-visibility)")
    print(f"  corr(dist, Jbar) = {np.corrcoef(dist, jbar)[0,1]:+.3f}")
    print(f"  pooled common-view slope kappa = {kappa:+.4f}")
    print(f"  Stage 2: partial corr(alpha, log-d | Jbar) = {pc:+.4f}")
    print(f"           partial corr(alpha, Jbar | log-d) = {pc_cov:+.4f}")
    print(f"           partial corr(alpha, cos-dlon | Jbar) = {pc_dlon:+.4f}")
    print(f"           partial corr(alpha, log-d | Jbar+cos-dlon) = {pc_ctrl:+.4f}")
    print(f"  OOS R^2 gain of distance term: {oos_gain:+.4f} "
          f"(with cos-dlon base: {oos_gain_dl:+.4f})")
    print(f"  Distance fit to alpha (raw): lambda = {fit['lam']:.0f} km, "
          f"R^2 = {fit['R2']:.3f}")
    print(f"  Distance fit to alpha (Jbar+cos-dlon separated): "
          f"lambda = {fit_sep['lam']:.0f} km, R^2 = {fit_sep['R2']:.3f}")
    print(f"  Permutation null ({len(null_pcs)} trials): "
          f"PC sigma = {pc_sigma:.1f}, p = {pc_p_value:.4f}")
    print(f"  R^2 sigma = {r2_sigma:.1f}, p = {r2_p_value:.4f}  "
          f"(raw; separated: sigma = {r2_sep_sigma:.1f}, "
          f"p = {r2_sep_p_value:.4f})")
    print(f"  distance_fit_significant = {distance_fit_significant}  "
          f"common_view_separated = {common_view_separated}  "
          f"frozen_S2 = {frozen_s2}")
    print(f"  T4 temporal pass (BOTH conditions): {passes}")

    return {
        "label": label,
        "n_days": len(precomputed),
        "n_pairs": len(alpha),
        "n_zero_covisibility_pairs": n_zero_j,
        "covis_source": covis_source,
        "covis_epochs_sod": True,
        "corr_dist_jbar": float(np.corrcoef(dist, jbar)[0, 1]),
        "pooled_common_view_kappa": float(kappa),
        "partial_corr_alpha_logd_given_jbar": pc,
        "partial_corr_alpha_jbar_given_logd": pc_cov,
        "partial_corr_alpha_cosdlon_given_jbar": pc_dlon,
        "partial_corr_alpha_logd_given_jbar_cosdlon": pc_ctrl,
        "oos_r2_gain_distance": oos_gain,
        "oos_r2_gain_distance_with_dlon_base": oos_gain_dl,
        "distance_fit_lambda_km": float(fit["lam"]),
        "distance_fit_R2": float(fit["R2"]),
        "distance_fit_separated_lambda_km": float(fit_sep["lam"]),
        "distance_fit_separated_R2": float(fit_sep["R2"]),
        "distance_fit_separated_controls": (
            "alpha residualised on [Jbar, cos(Delta-lon)] before binning; "
            "the null applies the identical residualisation, so the "
            "kappa_null*Jbar(d) leakage that heavy-tailed the uncontrolled "
            "null R^2 distribution is removed by construction"),
        "null_pc_sigma": pc_sigma,
        "null_pc_p_value": pc_p_value,
        "null_pc_ctrl_median": float(np.nanmedian(null_pc_ctrl)) if len(null_pc_ctrl) else float("nan"),
        "null_r2_sigma": r2_sigma,
        "null_r2_p_value": r2_p_value,
        "null_lambda_median": float(np.nanmedian(null_lambdas)) if len(null_lambdas) else float("nan"),
        "null_r2_median": float(np.nanmedian(null_r2s)) if len(null_r2s) else float("nan"),
        "null_sep_r2_sigma": r2_sep_sigma,
        "null_sep_r2_p_value": r2_sep_p_value,
        "null_sep_lambda_median": float(np.nanmedian(null_sep_lambdas)) if len(null_sep_lambdas) else float("nan"),
        "null_sep_r2_median": float(np.nanmedian(null_sep_r2s)) if len(null_sep_r2s) else float("nan"),
        "distance_fit_significant": distance_fit_significant,
        "common_view_separated": common_view_separated,
        "frozen_S2": frozen_s2,
        "T4_temporal_pass": passes,
        "stage1": ("pooled common-view slope kappa with pair fixed effects "
                   "(alpha = mean_e[p - kappa*J]); per-pair beta was replaced: "
                   "its noise entered alpha multiplied by Jbar and attenuated "
                   "the stage-2 partial correlation toward zero through the "
                   "d-Jbar collinearity (errors-in-variables)."),
        "covisibility": ("per-epoch Jaccard of visible-satellite sets computed "
                         "from the day's real SP3 ephemerides at UTC "
                         "seconds-of-day (GPS-only; the nominal 24-satellite "
                         "model is arbitrarily Earth-phased and scrambles the "
                         "per-epoch J, attenuating kappa ~4x)."),
        "T4_temporal_rule": (
            "distance_fit_significant = (separated R^2 > 0.2 AND separated "
            "R^2 sigma > 3.0 vs the identically residualised epoch-shuffle "
            "null AND 500 < separated lambda < 20000); "
            "common_view_separated = (|partial corr(alpha, log d | Jbar)| > 0.10 "
            "AND PC sigma > 3.0); T4_temporal_pass requires BOTH. The scored "
            "distance fit is on alpha residualised on [Jbar, cos(Delta-lon)] "
            "-- the raw binned fit is descriptive only, since it attributes "
            "neither the Jbar-aligned nor the longitude-phased components. "
            "frozen_S2 = (|PC| > 0.15 AND OOS R^2 gain > 0.02)."),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "test": "temporal_t4",
        "description": ("Time-resolved co-visibility regression: exploits "
                        "within-pair J variation (distance fixed, J varies) "
                        "to separate common-view from distance structure"),
        "timestamp": datetime.now().isoformat(),
        "arms": {},
    }

    # Broadcast global (7 days, 52 stations)
    bdays = [f"2024{d:03d}" for d in range(100, 107)]
    res["arms"]["broadcast_global"] = run_temporal_t4(
        "broadcast", bdays, "broadcast_global", n_null=200)

    # Precise COD (30 evenly-spaced days, subsample to 60 stations for speed)
    cod_days = td.available_dates("clk_cod", 2024)
    if len(cod_days) >= 30:
        idx = np.linspace(0, len(cod_days) - 1, 30).round().astype(int)
        days = [cod_days[i] for i in idx]
        res["arms"]["precise_cod"] = run_temporal_t4(
            "clk_cod", days, "precise_cod", n_null=100, max_stations=60)

    # Headline
    b = res["arms"].get("broadcast_global", {})
    p = res["arms"].get("precise_cod", {})
    res["summary"] = {
        "broadcast_passes": b.get("T4_temporal_pass", False),
        "precise_passes": p.get("T4_temporal_pass", False),
        "broadcast_distance_fit_significant": b.get("distance_fit_significant", False),
        "precise_distance_fit_significant": p.get("distance_fit_significant", False),
        "broadcast_common_view_separated": b.get("common_view_separated", False),
        "precise_common_view_separated": p.get("common_view_separated", False),
        "broadcast_pc": b.get("partial_corr_alpha_logd_given_jbar", float("nan")),
        "precise_pc": p.get("partial_corr_alpha_logd_given_jbar", float("nan")),
        "broadcast_lambda": b.get("distance_fit_lambda_km", float("nan")),
        "precise_lambda": p.get("distance_fit_lambda_km", float("nan")),
        "broadcast_sep_lambda": b.get("distance_fit_separated_lambda_km",
                                     float("nan")),
        "precise_sep_lambda": p.get("distance_fit_separated_lambda_km",
                                    float("nan")),
        "broadcast_sep_r2": b.get("distance_fit_separated_R2", float("nan")),
        "precise_sep_r2": p.get("distance_fit_separated_R2", float("nan")),
        "broadcast_sep_r2_sigma": b.get("null_sep_r2_sigma", float("nan")),
        "precise_sep_r2_sigma": p.get("null_sep_r2_sigma", float("nan")),
        "broadcast_r2": b.get("distance_fit_R2", float("nan")),
        "precise_r2": p.get("distance_fit_R2", float("nan")),
        "broadcast_r2_sigma": b.get("null_r2_sigma", float("nan")),
        "precise_r2_sigma": p.get("null_r2_sigma", float("nan")),
        "broadcast_pc_sigma": b.get("null_pc_sigma", float("nan")),
        "precise_pc_sigma": p.get("null_pc_sigma", float("nan")),
        "broadcast_pc_ctrl": b.get("partial_corr_alpha_logd_given_jbar_cosdlon", float("nan")),
        "precise_pc_ctrl": p.get("partial_corr_alpha_logd_given_jbar_cosdlon", float("nan")),
        "broadcast_pc_dlon": b.get("partial_corr_alpha_cosdlon_given_jbar", float("nan")),
        "precise_pc_dlon": p.get("partial_corr_alpha_cosdlon_given_jbar", float("nan")),
        "broadcast_oos_gain": b.get("oos_r2_gain_distance", float("nan")),
        "precise_oos_gain": p.get("oos_r2_gain_distance", float("nan")),
        "broadcast_kappa": b.get("pooled_common_view_kappa", float("nan")),
        "precise_kappa": p.get("pooled_common_view_kappa", float("nan")),
        "broadcast_frozen_S2": b.get("frozen_S2", False),
        "precise_frozen_S2": p.get("frozen_S2", False),
    }

    outpath = OUT / "step_45_temporal_t4.json"
    outpath.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved {outpath}")
    s = res["summary"]
    print("\n" + "=" * 72)
    print("TEMPORAL T4 RESULT")
    print("=" * 72)
    print(f"  Broadcast: PC={s['broadcast_pc']:+.4f}, lambda={s['broadcast_lambda']:.0f} km, "
          f"R^2={s['broadcast_r2']:.3f}, R^2sig={s['broadcast_r2_sigma']:.1f}, "
          f"pass={s['broadcast_passes']}")
    print(f"  Precise:   PC={s['precise_pc']:+.4f}, lambda={s['precise_lambda']:.0f} km, "
          f"R^2={s['precise_r2']:.3f}, R^2sig={s['precise_r2_sigma']:.1f}, "
          f"pass={s['precise_passes']}")


if __name__ == "__main__":
    main()
