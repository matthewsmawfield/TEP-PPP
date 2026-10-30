#!/usr/bin/env python3
"""
STEP 96 -- Leftover-clock supremacy (the ambitious flagship, Level 1).

The claim this test is built to support, and nothing weaker:

    A frozen TEP phase field, built only from other clocks and prior-day
    data, predicts a held-out station's proper time better than that
    station's own GNSS solution can -- and better than a data-driven
    spatial map fitted to a different period -- on an independent year
    and on a non-European network.

Design
------
The precise analysis-centre clock product (COD0MGXFIN) IS the full-sky
production PPP clock estimate: CODE already used every visible satellite
to estimate each station's clock.  We hold out one station at a time and
predict its current-day clock from the rest of the network only, using a
phase/coherence field FROZEN from a strictly prior window.

One estimand for every predictor
--------------------------------
    y_i(t) = clk_i(t) - clk_i(t_a)

the accumulated clock offset of station i since the last pre-window epoch
t_a (the last time that receiver had a fix).  This is exactly what a
receiver in outage must predict.  Every predictor below estimates the
same y_i(t) and is scored against the same truth.

Predictors
----------
  zero            predict no drift (y_hat = 0)
  linrate         y_hat = rate * (t - t_a), rate from the target pre-window
                  (the deployable "persistence trend")
  ar1             first-order Gauss-Markov forecast of y_i from its own
                  pre-window
  free_gp         GP whose (form, length scale, nugget) are LOO-selected on
                  the prior window's REFERENCE network (never the target),
                  applied to the current-day cross-section {y_j(t)}
  empirical_cov   BLUP kriging with a correlation function fitted to the
                  prior-window reference-network sample covariance -- the
                  data-driven spatial map a frozen physics kernel must beat
                  to earn a physics claim
  tep_bandpass    prior-day phase-cos Gram (10-500 uHz) (x) fixed 4,200 km
                  Gaussian covariance -- the current registered predictor
  tep_lfmf        prior MULTI-day phase-cos Gram with the low-frequency edge
                  dropped to a few uHz (keeps the slow drift the band-pass
                  discards) (x) fixed 4,200 km Gaussian

Frozen protocol: the TEP predictor specification (bands, covariance form,
scale, nugget) is fixed here from the 2024 development corpus and is NOT
re-selected on 2025.

Pass (Level 1): a single frozen TEP variant beats zero AND linrate AND
ar1 AND free_gp AND empirical_cov, with a positive day-clustered 95% CI,
on 2025 AND on at least one non-European sub-network.

Output: results/outputs/step_96_leftover_clock_supremacy.json
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td                        # noqa: E402
from scripts.lib.tep_phase import spectral_phase_gram         # noqa: E402

OUT = ROOT / "results" / "outputs" / os.environ.get(
    "TEP_S96_OUT", "step_96_leftover_clock_supremacy.json")
RNG = np.random.default_rng(20260903)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.  A global
# suppression masks every overflow/divide-by-zero in the whole process,
# including bugs.  The only place ill-conditioned numerics are expected is
# the free-GP LOO grid (large condition numbers on near-singular rho
# matrices); that loop is wrapped in a local np.errstate context below and
# its NaN/inf outputs are guarded with isfinite checks.

C_M_PER_NS = 0.299792458
DT_S = 300.0
PRODUCT = os.environ.get("TEP_S96_PRODUCT", "clk_cod")
N_DAYS = int(os.environ.get("TEP_S96_DAYS", "30"))
EVAL_STRIDE = int(os.environ.get("TEP_S96_STRIDE", "12"))
GRAM_DAYS = int(os.environ.get("TEP_S96_GRAM_DAYS", "5"))
PREWINDOW_S = 7200.0
MIN_REFS = 8

# ---- frozen TEP predictor specification (fixed on 2024, not re-tuned) ----
LAMBDA_KM = 4200.0
NUGGET = 0.70
BP_LO_HZ, BP_HI_HZ = 10e-6, 500e-6
LF_HI_HZ = 500e-6                       # LF+MF keeps the same high edge
# one full cycle per prior window: period == GRAM_DAYS days.  This keeps the
# slow drift the 10 uHz (~1 day) band-pass edge discards.
LF_LO_HZ = 1.0 / (GRAM_DAYS * 86400.0)

DNR_SWITCH = float(os.environ.get("TEP_S96_DNR", "10.0"))  # frozen per-station drift gate
EMP_HIST_DAYS = int(os.environ.get("TEP_S96_EMP_HIST", "5"))  # empirical-map history (matches GRAM_DAYS for fairness)
MAX_NET = int(os.environ.get("TEP_S96_MAX_NET", "100"))       # cap per-cell network size
CV_FORMS = ("gaussian", "exponential", "matern15")
CV_LAM = (150.0, 400.0, 900.0, 2000.0, 4200.0, 9000.0, 20000.0)
CV_NUG = (0.15, 0.35, 0.70, 1.27)
EMP_LAM = (200.0, 400.0, 800.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0)

# Geographically COHERENT sub-networks: each cluster's maximum inter-station
# separation is within ~1 correlation length (~4,200 km), so a single fixed
# lambda can actually relate its members.  GLOBAL and the ocean-spanning
# ASIA_OCEANIA grab-bag are reported for context but do not carry the verdict.
REGIONS = {
    "GLOBAL":        lambda la, lo: True,
    "EUROPE":        lambda la, lo: 34.0 <= la <= 72.0 and -12.0 <= lo <= 40.0,
    "NORTH_AMERICA": lambda la, lo: 22.0 <= la <= 62.0 and -128.0 <= lo <= -60.0,
    "EAST_ASIA":     lambda la, lo: 18.0 <= la <= 50.0 and 100.0 <= lo <= 150.0,
    "ASIA_OCEANIA":  lambda la, lo: -50.0 <= la <= 55.0 and 95.0 <= lo <= 180.0,
}
COHERENT_REGIONS = ("EUROPE", "NORTH_AMERICA", "EAST_ASIA")
NON_EURO_REGIONS = ("NORTH_AMERICA", "EAST_ASIA")

BASELINES = ("zero", "linrate", "ar1", "free_gp", "empirical_cov",
             "empirical_cov_full", "empirical_cov_full_hybrid")
# The Level-1 pass requires beating every DEPLOYABLE, TARGET-BLIND predictor:
# persistence, its trend, AR1, a free GP, and a causal parametric empirical-
# covariance map.  empirical_cov_full / _hybrid additionally use the held-out
# station's OWN historical covariance with its neighbours -- a target-aware,
# non-cold-start baseline that TEP is deliberately blind to -- so they are
# reported for context but not required for the pass.
REQUIRED_BASELINES = ("zero", "linrate", "ar1", "free_gp", "empirical_cov")
# TEP-specificity nulls: the primary arm must additionally beat all of these
# to show the DISTANCE + PHASE structure of the kernel (not just network
# aiding or drift handling) is doing the work.
SPECIFICITY_NULLS = ("netmean_gated", "tep_lfmf_gated_distscr",
                     "tep_lfmf_gated_phasescr")
# *_hybrid: the target keeps its OWN pre-window drift rate (a per-station
# nuisance, not field structure) and the spatial predictor estimates only the
# residual y_j - rate_j*h.  empirical_cov_full_hybrid gets the SAME treatment
# so the physics-vs-data-map comparison stays apples-to-apples.
# tep_lfmf_gated: the rate removal is applied PER STATION, gated by that
# station's own causal drift-to-noise ratio -- a reference/target whose
# pre-window looks like white noise keeps its raw value; one with a clear
# deterministic slope has that slope carried by its own trend and the TEP
# field predicts the rest.  No regional switch, no tuning beyond one frozen
# DNR threshold.
TEP_ARMS = ("tep_bandpass", "tep_lfmf", "tep_lfmf_hybrid",
            "tep_bandpass_hybrid", "tep_lfmf_gated", "tep_lfmf_gated_screened",
            # TEP-specificity nulls: the primary arm must beat all of these
            "netmean_gated", "tep_lfmf_gated_distscr", "tep_lfmf_gated_phasescr")


def _prev(tag, k=1):
    return (datetime.strptime(tag, "%Y%j") - timedelta(days=k)).strftime("%Y%j")


_DIST_CACHE: dict = {}


def load_day(tag, region_fn, need_dist=True):
    try:
        s = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
    except (EOFError, OSError, ValueError):
        return None                      # truncated / unreadable product file
    if not s or len(s) < 12:
        return None
    names = [n for n in s
             if region_fn(s[n]["station"].lat_deg, s[n]["station"].lon_deg)]
    if len(names) < MIN_REFS + 3:
        return None
    raw = np.column_stack([s[n]["raw"] for n in names]).astype(float) * C_M_PER_NS
    t = s[names[0]]["t"].astype(float)
    if not need_dist:
        return names, t, raw, None
    stn = [s[n]["station"] for n in names]
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        ci = _DIST_CACHE.setdefault(names[i], {})
        for j in range(i + 1, len(names)):
            key = names[j]
            dij = ci.get(key)
            if dij is None:
                dij = td.great_circle_km(stn[i], stn[j])
                ci[key] = dij
                _DIST_CACHE.setdefault(key, {})[names[i]] = dij
            D[i, j] = D[j, i] = dij
    return names, t, raw, D


def _interp_gaps(y, min_good=20):
    good = np.isfinite(y)
    if good.all():
        return y, 1.0
    if good.sum() < min_good:
        return None, 0.0
    ix = np.arange(len(y))
    return np.interp(ix, ix[good], y[good]), good.mean()


def multi_day_matrix(tag, region_fn, ndays):
    """Stack `ndays` strictly-prior UTC days into one continuous matrix on the
    stations present on every prior day."""
    mats, common = [], None
    for k in range(ndays, 0, -1):
        d = load_day(_prev(tag, k), region_fn, need_dist=False)
        if d is None:
            return None
        nm, _, raw, _ = d
        col = {n: raw[:, i] for i, n in enumerate(nm)}
        common = set(col) if common is None else (common & set(col))
        mats.append(col)
    if not common or len(common) < MIN_REFS + 1:
        return None
    order = sorted(common)
    return order, np.vstack([np.column_stack([c[n] for n in order]) for c in mats])


def prior_gram(tag, cur_names, region_fn, ndays, f_lo, f_hi):
    md = multi_day_matrix(tag, region_fn, ndays)
    if md is None:
        return None
    pnames, Y = md
    cols, kept = [], []
    for j, nm in enumerate(pnames):
        if nm not in cur_names:
            continue
        y, frac = _interp_gaps(Y[:, j])
        if y is None or frac < 0.9 or not (1e-4 < np.std(y) < 1e6):
            continue
        cols.append(y)
        kept.append(nm)
    if len(kept) < MIN_REFS + 1:
        return None
    try:
        g = spectral_phase_gram(np.column_stack(cols), DT_S,
                                f_lo_hz=f_lo, f_hi_hz=f_hi)
    except ValueError:
        return None
    return kept, g["predictive_gram"]


def prior_detrended(tag, cur_names, region_fn, ndays):
    """Prior-window per-station linearly-detrended matrix for the free-GP and
    empirical-covariance baselines.  Each column is standardised to unit
    variance after detrending so the sample covariance is numerically stable
    over multi-day windows (raw clock offsets in metres can overflow the
    Ptm.T @ Ptm matmul on 8-day stacks)."""
    md = multi_day_matrix(tag, region_fn, ndays)
    if md is None:
        return None
    pnames, Y = md
    q = np.arange(Y.shape[0]) - Y.shape[0] / 2.0
    cols, kept = [], []
    for j, nm in enumerate(pnames):
        if nm not in cur_names:
            continue
        y, frac = _interp_gaps(Y[:, j])
        if y is None or frac < 0.9:
            continue
        y = y - np.polyval(np.polyfit(q, y, 1), q)
        sd = float(np.std(y))
        if not (1e-4 < sd < 1e6):
            continue
        cols.append(y / sd)
        kept.append(nm)
    if len(kept) < MIN_REFS + 1:
        return None
    return kept, np.column_stack(cols)


def _reg_solve(K, c):
    # Every K here is (PSD Hadamard product) + nugget*I, so it is already
    # strictly PD.  Try Cholesky (fast); only on failure fall back to a
    # ridge + lstsq.  Avoids an O(n^3) eigendecomposition per held-out
    # station, which dominated the runtime on the 277-station global net.
    K = 0.5 * (K + K.T)
    try:
        L = np.linalg.cholesky(K)
        return np.linalg.solve(L.T, np.linalg.solve(L, c))
    except np.linalg.LinAlgError:
        n = len(c)
        tr = np.trace(K) / n if n else 1.0
        return np.linalg.lstsq(K + 1e-4 * tr * np.eye(n), c, rcond=None)[0]


def _rho_form(d, form, lam):
    if form == "gaussian":
        return np.exp(-0.5 * (d / lam) ** 2)
    if form == "matern15":
        r = np.sqrt(3.0) * d / lam
        return (1.0 + r) * np.exp(-r)
    return np.exp(-d / lam)


def _causal_fill(v):
    """Forward-fill NaNs (last-observation-carried-forward), then back-fill any
    leading NaNs.  No future value ever influences a past sample -- this is what
    a real receiver/network does during an outage."""
    v = np.asarray(v, float).copy()
    n = len(v)
    last = np.nan
    for k in range(n):
        if np.isfinite(v[k]):
            last = v[k]
        else:
            v[k] = last
    # leading NaNs: back-fill with the first finite value (only affects the
    # warm-up region, never an evaluation epoch)
    if not np.isfinite(v[0]):
        first = next((x for x in v if np.isfinite(x)), 0.0)
        v[~np.isfinite(v)] = first
    return v


def _accumulated_offset(clk_m, npre):
    """y_i(t) = clk_i(t) - clk_i(t_a), t_a = last pre-window epoch.
    Returns (y, y_observed_mask, (rate, span), ar1_params).  Gaps are
    CAUSALLY filled (LOCF); `y_observed_mask` is True only where the clock
    was genuinely observed, so the held-out truth is never an interpolated
    value that borrowed from after the outage."""
    finite = np.isfinite(clk_m)
    if finite.sum() < 20:
        return None, None, None, None
    filled = _causal_fill(clk_m)
    anchor = filled[npre - 1]
    y = filled - anchor
    lo = max(0, npre - int(PREWINDOW_S / DT_S))
    seg = filled[lo:npre]
    span = float(np.ptp(seg)) if len(seg) else 0.0
    if len(seg) >= 6:
        xx = np.arange(len(seg), dtype=float)
        coef = np.polyfit(xx, seg, 1)
        rate = float(coef[0])
        resid = seg - np.polyval(coef, xx)
        detr_noise = float(np.std(resid))
        seg_var = float(np.var(seg))
        # linear-fit R^2: how well a straight line explains the pre-window
        r2 = 1.0 - (detr_noise ** 2) / seg_var if seg_var > 1e-12 else 0.0
        # drift-to-noise ratio over the pre-window
        dnr_raw = abs(rate) * len(seg) / max(detr_noise, 1e-6)
        # gate signal for carrying the station's OWN linear rate instead of
        # letting the field predict it.  All three must hold, so the gate
        # fires only for unambiguous free-running / rubidium drift and never
        # for an H-maser with a tiny real trend:
        #   (a) a straight line explains >97% of the pre-window,
        #   (b) that line is steep vs the residual scatter (DNR high), and
        #   (c) the pre-window drift is physically large (> ~7 ns == 2 m).
        pre_drift_m = abs(rate) * len(seg)
        dnr = dnr_raw if (r2 > 0.97 and pre_drift_m > 2.0) else 0.0
    else:
        rate, dnr = 0.0, 0.0
    d = np.diff(seg)
    if len(d) >= 8:
        mu = float(np.mean(d))
        dc = d - mu
        den = float(dc[:-1] @ dc[:-1])
        phi = float(dc[:-1] @ dc[1:]) / den if den > 1e-12 else 0.0
        phi = max(min(phi, 0.999), -0.999)
        last = float(dc[-1])
    else:
        mu, phi, last = rate, 0.0, 0.0
    return y, finite, (rate, span, dnr), (mu, phi, last)


def _ar1_forecast(ar1p, horizon_steps):
    """Accumulated offset predicted by the AR(1) increment model over
    `horizon_steps` samples."""
    mu, phi, last = ar1p
    h = max(int(horizon_steps), 1)
    if abs(phi) < 1e-9:
        return float(mu * h)
    # sum_{k=1..h} [ mu + phi^k * last ]
    geom = phi * (1.0 - phi ** h) / (1.0 - phi)
    return float(mu * h + geom * last)


def run(days, region_fn):
    per_day = {}
    npre = int(PREWINDOW_S / DT_S)

    for tag in days:
        cur = load_day(tag, region_fn)
        if cur is None:
            continue
        names, t, _, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur:
            continue

        g_bp = prior_gram(tag, names, region_fn, 1, BP_LO_HZ, BP_HI_HZ)
        g_lf = prior_gram(tag, names, region_fn, GRAM_DAYS, LF_LO_HZ, LF_HI_HZ)
        pdt = prior_detrended(tag, names, region_fn, GRAM_DAYS)
        # longer prior window for the empirical-covariance baselines only, so
        # the "reuse the recent map" baseline is not artificially starved of
        # history relative to what a real system would keep
        pdt_emp = (pdt if EMP_HIST_DAYS <= GRAM_DAYS
                   else prior_detrended(tag, names, region_fn, EMP_HIST_DAYS))
        if g_bp is None or g_lf is None or pdt is None or pdt_emp is None:
            continue

        common = [n for n in names
                  if n in (set(g_bp[0]) & set(g_lf[0]) & set(pdt[0])
                           & set(pdt_emp[0]))]
        if len(common) < MIN_REFS + 2:
            continue
        G_bp = g_bp[1][np.ix_([g_bp[0].index(n) for n in common],
                              [g_bp[0].index(n) for n in common])]
        G_lf = g_lf[1][np.ix_([g_lf[0].index(n) for n in common],
                              [g_lf[0].index(n) for n in common])]
        P = pdt[1][:, [pdt[0].index(n) for n in common]]
        P_emp = pdt_emp[1][:, [pdt_emp[0].index(n) for n in common]]
        Dg = D[np.ix_([names.index(n) for n in common],
                      [names.index(n) for n in common])]

        # current-day accumulated-offset matrix + pre-window rate/AR1 params
        Y_cols, Obs_cols, rates, spans, dnrs, ar1ps, n_t = (
            [], [], [], [], [], [], None)
        ok = True
        for n in common:
            clk = s_cur[n]["raw"].astype(float) * C_M_PER_NS
            if n_t is None:
                n_t = len(clk)
            if len(clk) != n_t:
                ok = False
                break
            y, obs, rs, ar1p = _accumulated_offset(clk, npre)
            if y is None:
                ok = False
                break
            Y_cols.append(y)
            Obs_cols.append(obs)
            rates.append(rs[0])
            spans.append(rs[1])
            dnrs.append(rs[2])
            ar1ps.append(ar1p)
        if not ok or n_t is None or npre >= n_t - 1:
            continue
        Y = np.column_stack(Y_cols)
        Obs = np.column_stack(Obs_cols)          # genuinely-observed mask
        rates = np.asarray(rates)
        spans = np.asarray(spans)
        dnrs = np.asarray(dnrs)

        # Causal station QC only.  A station is excluded when its PRIOR-day
        # detrended residual is degenerate/unstable, or its PRE-window drift
        # already exceeds a holdover cap (~50 m == 167 ns over <=2 h is a
        # steered / free-running clock, not a leftover-clock target; cf.
        # step_55 MAX_PLAUSIBLE_CLOCK_M).  No post-anchor (future) value is
        # consulted for selection.
        MAX_PRE_DRIFT_M = 50.0
        sd = np.nanstd(P, axis=0)
        med = np.median(sd)
        keep = [i for i in range(len(common))
                if 0.01 < sd[i] < 12.0 and sd[i] < 3.5 * med
                and np.max(np.abs(Y[:npre, i])) < MAX_PRE_DRIFT_M
                and Obs[npre:, i].mean() > 0.3]
        if len(keep) < MIN_REFS + 2:
            continue
        # cap the network size: the very large GLOBAL grab-bag is a sanity
        # check, not a coherent-network claim, and a 140-station representative
        # sample keeps the per-station LOO tractable.
        if len(keep) > MAX_NET:
            keep = sorted(RNG.choice(keep, MAX_NET, replace=False).tolist())
        common = [common[i] for i in keep]
        G_bp = G_bp[np.ix_(keep, keep)]
        G_lf = G_lf[np.ix_(keep, keep)]
        P = P[:, keep]
        P_emp = P_emp[:, keep]
        Y = Y[:, keep]
        Obs = Obs[:, keep]
        rates = rates[keep]
        spans = spans[keep]
        dnrs = dnrs[keep]
        ar1ps = [ar1ps[i] for i in keep]
        Dg = Dg[np.ix_(keep, keep)]
        ng = len(common)

        # per-horizon-bin squared errors: short (<1 h), mid (1-4 h), long (>4 h)
        day_sq = {hb: {a: [] for a in BASELINES + TEP_ARMS}
                  for hb in ("all", "short", "mid", "long")}
        eval_i = list(range(npre, Y.shape[0] - 1, EVAL_STRIDE))

        def _hbins(h_steps):
            hs = h_steps * DT_S
            out = ["all"]
            if hs < 3600.0:
                out.append("short")
            elif hs <= 14400.0:
                out.append("mid")
            else:
                out.append("long")
            return out

        # ---- precompute the prior-window sample correlation ONCE per day ----
        # (a) common-mode-removed correlation for the parametric empirical fit
        # Guard NaN/inf in the matmuls (some stations have gaps that produce
        # non-finite values); the resulting CORR matrices are clipped below.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            Pcm = P_emp - P_emp.mean(1, keepdims=True)
            C_cm = (Pcm.T @ Pcm) / Pcm.shape[0]
            dcm = np.clip(np.diag(C_cm), 1e-9, None)
            CORR_cm = C_cm / np.sqrt(np.outer(dcm, dcm))
            # (b) raw shrunk correlation for the target-aware "reuse the map" baseline
            Ptm = P_emp - P_emp.mean(0, keepdims=True)
            S_full = (Ptm.T @ Ptm) / Ptm.shape[0]
            ds = np.clip(np.diag(S_full), 1e-9, None)
            CORR_full = S_full / np.sqrt(np.outer(ds, ds))
        CORR_cm = np.nan_to_num(CORR_cm, nan=0.0, posinf=0.0, neginf=0.0)
        CORR_full = np.nan_to_num(CORR_full, nan=0.0, posinf=0.0, neginf=0.0)
        _SHRINK = 0.15
        CORR_full = (1.0 - _SHRINK) * CORR_full + _SHRINK * np.eye(ng)

        # per-station detrended pre-window fluctuation amplitude (for the
        # causal field-responsiveness / screening damp) -- computed once
        w48 = min(48, npre)
        seg_pre = Y[npre - w48:npre, :] - (
            rates[None, :] * np.arange(-w48, 0)[:, None])
        amp_all = np.std(seg_pre, axis=0)
        amp_all[~np.isfinite(amp_all) | (amp_all <= 0)] = np.nanmedian(amp_all)

        # ---- TEP-specificity nulls (per day) ----------------------------------
        # If the DISTANCE structure of the kernel is load-bearing, permuting the
        # station labels of the distance matrix (keeping the same distances,
        # assigning them to the wrong pairs) must degrade the prediction.  If
        # the PHASE Gram is load-bearing, permuting its labels must too.  And if
        # any of the win over persistence is just network common mode, an
        # UNWEIGHTED mean of the same cross-section must capture most of it.
        _dperm = RNG.permutation(ng)
        _gperm = RNG.permutation(ng)
        Dg_scr = Dg[np.ix_(_dperm, _dperm)]
        G_lf_scr = G_lf[np.ix_(_gperm, _gperm)]

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < MIN_REFS:
                continue
            drr = Dg[np.ix_(pool, pool)]
            drt = Dg[b, pool]
            rho_rr = np.exp(-0.5 * (drr / LAMBDA_KM) ** 2)
            rho_rt = np.exp(-0.5 * (drt / LAMBDA_KM) ** 2)

            try:
                w_bp = _reg_solve(rho_rr * G_bp[np.ix_(pool, pool)]
                                  + NUGGET * np.eye(len(pool)),
                                  rho_rt * G_bp[b, pool])
                w_lf = _reg_solve(rho_rr * G_lf[np.ix_(pool, pool)]
                                  + NUGGET * np.eye(len(pool)),
                                  rho_rt * G_lf[b, pool])
                # TEP-specificity nulls: same machinery, scrambled geometry / Gram
                d_rr_s = Dg_scr[np.ix_(pool, pool)]
                d_rt_s = Dg_scr[b, pool]
                rr_rr_s = np.exp(-0.5 * (d_rr_s / LAMBDA_KM) ** 2)
                rr_rt_s = np.exp(-0.5 * (d_rt_s / LAMBDA_KM) ** 2)
                w_lf_dscr = _reg_solve(
                    rr_rr_s * G_lf[np.ix_(pool, pool)] + NUGGET * np.eye(len(pool)),
                    rr_rt_s * G_lf[b, pool])
                w_lf_pscr = _reg_solve(
                    rho_rr * G_lf_scr[np.ix_(pool, pool)] + NUGGET * np.eye(len(pool)),
                    rho_rt * G_lf_scr[b, pool])
            except np.linalg.LinAlgError:
                continue

            # --- empirical_cov: parametric data-driven map.  Fit rho(d) to the
            #     prior-window reference sample correlation, then BLUP. ---
            corr = CORR_cm[np.ix_(pool, pool)]
            iu, ju = np.triu_indices(len(pool), 1)
            dv, cvv = drr[iu, ju], corr[iu, ju]
            bestL = min(EMP_LAM, key=lambda L: float(
                np.sum((cvv - np.exp(-0.5 * (dv / L) ** 2)) ** 2)))
            try:
                w_emp = _reg_solve(
                    np.exp(-0.5 * (drr / bestL) ** 2) + NUGGET * np.eye(len(pool)),
                    np.exp(-0.5 * (drt / bestL) ** 2))
            except np.linalg.LinAlgError:
                continue

            # --- empirical_cov_full: the TOUGH baseline -- shrunk FULL prior
            #     covariance, target row included: BLUP kriging with the
            #     empirically-measured recent covariance, using the held-out
            #     station's OWN historical correlations with each neighbour.
            #     "Literally reuse the recent spatial map."  Uses the SAME
            #     nugget as the TEP predictor so the comparison is
            #     apples-to-apples (only the covariance FORM differs). ---
            try:
                w_empf = _reg_solve(
                    CORR_full[np.ix_(pool, pool)]
                    + NUGGET * np.eye(len(pool)),
                    CORR_full[b, pool])
            except np.linalg.LinAlgError:
                w_empf = w_emp

            # free GP: (form, lam, nug) chosen by closed-form ridge LOO on the
            # prior-window reference network only.  LOO residual for ridge
            # kriging is e_i / (1 - H_ii) with H = R (R + nug I)^-1 -- no
            # per-station refit, so the whole grid is cheap even for a large
            # global pool.  Selection on a capped random subset of references
            # keeps it O(1) in network size and still leakage-free.
            sel = (list(range(len(pool))) if len(pool) <= 30
                   else list(RNG.choice(len(pool), 30, replace=False)))
            Psel = P[:, [pool[k] for k in sel]]
            dsel = drr[np.ix_(sel, sel)]
            best = (np.inf, None)
            # Local errstate: the free-GP LOO grid hits ill-conditioned
            # (near-singular) rho matrices on some grid points, producing
            # overflow/invalid in the matmuls below.  Suppress ONLY here so a
            # real bug elsewhere is not silently hidden; every grid point is
            # guarded by isfinite checks and skipped on failure.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for form in CV_FORMS:
                    for lam in CV_LAM:
                        Rs = _rho_form(dsel, form, lam)
                        for nug in CV_NUG:
                            try:
                                A_inv = np.linalg.solve(
                                    Rs + nug * np.eye(len(sel)), np.eye(len(sel)))
                            except np.linalg.LinAlgError:
                                continue
                            if not np.isfinite(A_inv).all():
                                continue
                            H = Rs @ A_inv
                            if not np.isfinite(H).all() or np.max(np.abs(H)) > 1e6:
                                continue
                            hd = np.clip(np.diag(H), -10.0, 1.0 - 1e-3)
                            fitted = Psel @ H.T
                            loo = (Psel - fitted) / (1.0 - hd)[None, :]
                            if not np.isfinite(loo).all():
                                continue
                            e = float(np.sqrt(np.mean(loo ** 2)))
                            if e < best[0]:
                                best = (e, (form, lam, nug))
            if best[1] is None:
                continue
            form, lam, nug = best[1]
            try:
                w_gp = _reg_solve(
                    _rho_form(drr, form, lam) + nug * np.eye(len(pool)),
                    _rho_form(drt, form, lam))
            except np.linalg.LinAlgError:
                continue

            yb = Y[:, b]
            yr = Y[:, pool]
            ob = Obs[:, b]
            ob_r = Obs[:, pool]
            rate_b = rates[b]
            rate_pool = rates[pool]
            span_b = max(spans[b], 1e-6)
            ar1_b = ar1ps[b]
            # per-station causal drift gate (1 where a deterministic slope
            # clearly dominates the pre-window, else 0)
            g_b = 1.0 if dnrs[b] > DNR_SWITCH else 0.0
            g_pool = (dnrs[pool] > DNR_SWITCH).astype(float)
            # causal field-responsiveness: this clock's own detrended
            # pre-window fluctuation relative to the network median.  A
            # screened (atomic) clock barely responds to the LF field, so its
            # spatial prediction is damped -- the quartz/atomic screening
            # result applied per station.  `amp_all` is precomputed per day.
            resp_b = float(np.clip(amp_all[b] / max(np.median(amp_all[pool]), 1e-6),
                                   0.25, 1.4))

            for ei in eval_i:
                # score only where the held-out clock was GENUINELY observed
                # -- never an LOCF-filled epoch (would understate every error)
                if not ob[ei]:
                    continue
                truth = yb[ei]
                if not np.isfinite(truth):
                    continue
                # references are live (not in outage); a real deployment uses
                # each reference's last reported value, which is exactly what
                # the causal LOCF fill in `yr` already provides.  Require a
                # minimum fraction genuinely reporting so a mostly-dead pool
                # cannot carry a prediction.
                if ob_r[ei].mean() < 0.5:
                    continue
                cs = yr[ei]
                if not np.isfinite(cs).all():
                    continue
                # elapsed epochs since the anchor: truth y[ei] = clk[ei] -
                # clk[npre-1] accumulates ei-(npre-1) steps, so every temporal
                # predictor must forecast over the same h (the earlier
                # h = ei - npre lagged every baseline by one 300 s epoch).
                h = ei - npre + 1
                # deployable persistence-trend, clipped so a noisy pre-window
                # slope cannot diverge past a few pre-window spans
                lr = float(np.clip(rate_b * h, -8.0 * span_b, 8.0 * span_b))
                ar = float(np.clip(_ar1_forecast(ar1_b, h),
                                   -8.0 * span_b, 8.0 * span_b))
                # residual cross-section: each reference minus its own
                # deterministic pre-window drift over the elapsed horizon
                cs_resid = cs - rate_pool * h
                # per-station gated residual: only stations with a clear
                # deterministic slope get their rate removed
                cs_gated = cs - g_pool * rate_pool * h
                p_lf_pure = float(w_lf @ cs)
                p_lf_hyb = lr + float(w_lf @ cs_resid)
                p_lf_gated = g_b * lr + float(w_lf @ cs_gated)
                # field-responsiveness-damped gated arm: scale the spatial
                # term by the target's causal responsiveness (screening)
                p_lf_gs = g_b * lr + resp_b * float(w_lf @ cs_gated)
                # --- TEP-specificity nulls (same gated construction) ---
                nrm = np.full(len(pool), 1.0 / len(pool))
                preds = {
                    "zero": 0.0,
                    "linrate": lr,
                    "ar1": ar,
                    "free_gp": float(w_gp @ cs),
                    "empirical_cov": float(w_emp @ cs),
                    "empirical_cov_full": float(w_empf @ cs),
                    "empirical_cov_full_hybrid": lr + float(w_empf @ cs_resid),
                    "tep_bandpass": float(w_bp @ cs),
                    "tep_lfmf": p_lf_pure,
                    "tep_bandpass_hybrid": lr + float(w_bp @ cs_resid),
                    "tep_lfmf_hybrid": p_lf_hyb,
                    "tep_lfmf_gated": p_lf_gated,
                    "tep_lfmf_gated_screened": p_lf_gs,
                    # null 1: unweighted mean of the SAME gated cross-section
                    # (isolates network common-mode from distance weighting)
                    "netmean_gated": g_b * lr + float(nrm @ cs_gated),
                    # null 2: gated arm with a DISTANCE-scrambled kernel
                    "tep_lfmf_gated_distscr": g_b * lr + float(w_lf_dscr @ cs_gated),
                    # null 3: gated arm with a PHASE-Gram-scrambled kernel
                    "tep_lfmf_gated_phasescr": g_b * lr + float(w_lf_pscr @ cs_gated),
                }
                for hb in _hbins(h):
                    for k, v in preds.items():
                        day_sq[hb][k].append((truth - v) ** 2)

        if any(len(day_sq["all"][k]) < 30 for k in day_sq["all"]):
            continue
        for hb in day_sq:
            if any(len(day_sq[hb][k]) < 10 for k in day_sq[hb]):
                continue
            rms = {k: float(np.sqrt(np.mean(day_sq[hb][k]))) for k in day_sq[hb]}
            per_day.setdefault(f"{hb}:__days", []).append(tag)
            for k in BASELINES + TEP_ARMS:
                per_day.setdefault(f"{hb}:{k}_rms", []).append(rms[k])
            for a in TEP_ARMS:
                for bl in BASELINES + SPECIFICITY_NULLS:
                    if bl == a:
                        continue
                    per_day.setdefault(f"{hb}:{a}_vs_{bl}", []).append(
                        100.0 * (1.0 - rms[a] / rms[bl]) if rms[bl] > 0 else 0.0)

    def ci(key):
        v = np.asarray(per_day.get(key, []), float)
        v = v[np.isfinite(v)]
        if len(v) < 4:
            return {"median": None, "ci95": None, "n": int(len(v))}
        bs = [float(np.median(RNG.choice(v, len(v), replace=True)))
              for _ in range(6000)]
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n": int(len(v)), "frac_positive": float(np.mean(v > 0))}

    out = {"n_days_used": len(per_day.get("all:tep_bandpass_rms", [])),
           "per_day": {k: v for k, v in per_day.items()},
           "by_horizon": {}}
    for hb in ("all", "short", "mid", "long"):
        h = {"median_rms_m": {k: (float(np.median(per_day[f"{hb}:{k}_rms"]))
                                  if per_day.get(f"{hb}:{k}_rms") else None)
                              for k in BASELINES + TEP_ARMS},
             "n_days": len(per_day.get(f"{hb}:tep_lfmf_rms", []))}
        for a in TEP_ARMS:
            for bl in BASELINES + SPECIFICITY_NULLS:
                if bl == a:
                    continue
                h[f"{a}_vs_{bl}_pct"] = ci(f"{hb}:{a}_vs_{bl}")
        for a in TEP_ARMS:
            h[f"{a}_beats_all_baselines"] = all(
                (h[f"{a}_vs_{bl}_pct"]["ci95"] or [-1])[0] > 0
                for bl in REQUIRED_BASELINES)
            h[f"{a}_beats_target_aware_map"] = bool(
                (h[f"{a}_vs_empirical_cov_full_pct"]["ci95"] or [-1])[0] > 0)
            h[f"{a}_beats_specificity_nulls"] = all(
                (h.get(f"{a}_vs_{n}_pct", {}).get("ci95") or [-1])[0] > 0
                for n in SPECIFICITY_NULLS if n != a)
        out["by_horizon"][hb] = h
    # top-level convenience: the 'all' horizon
    a0 = out["by_horizon"]["all"]
    out["median_rms_m"] = a0["median_rms_m"]
    for a in TEP_ARMS:
        for bl in BASELINES + SPECIFICITY_NULLS:
            if bl == a:
                continue
            out[f"{a}_vs_{bl}_pct"] = a0.get(f"{a}_vs_{bl}_pct")
        out[f"{a}_beats_all_baselines"] = a0[f"{a}_beats_all_baselines"]
        out[f"{a}_beats_specificity_nulls"] = a0.get(f"{a}_beats_specificity_nulls")
        # a softer, operationally-relevant flag: beats every required baseline
        # at the mid+long outage horizons (where a leftover clock drifts)
        out[f"{a}_beats_all_baselines_midlong"] = all(
            (out["by_horizon"][hb][f"{a}_vs_{bl}_pct"]["ci95"] or [-1])[0] > 0
            for hb in ("mid", "long") for bl in REQUIRED_BASELINES)
    return out


PRODUCTS = [p.strip() for p in
            os.environ.get("TEP_S96_PRODUCTS", "clk_cod,clk_cod_ops").split(",")
            if p.strip()]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    global PRODUCT
    results = {}
    for product in PRODUCTS:
        PRODUCT = product
        for year, label in ((2024, "dev_frozen"), (2025, "independent")):
            dates = td.available_dates(product, year)
            if not dates:
                print(f"[{product} {year}] no data, skipping", flush=True)
                continue
            usable = dates[GRAM_DAYS + 1:]
            days = (usable if len(usable) < N_DAYS
                    else [usable[i] for i in
                          np.linspace(0, len(usable) - 1, N_DAYS).astype(int)])
            _run_regions = (os.environ.get("TEP_S96_REGIONS", "").split(",")
                            if os.environ.get("TEP_S96_REGIONS") else list(REGIONS))
            for region in [r for r in REGIONS if r in _run_regions]:
                print(f"\n=== {product} / {year} ({label}) / {region} : "
                      f"{len(days)} days ===", flush=True)
                r = run(days, REGIONS[region])
                results[f"{product}_{year}_{region}"] = r
                mid = r.get("by_horizon", {}).get("mid", {})
                print("  mid-horizon rms:",
                      {k: (round(v, 2) if v else None)
                       for k, v in mid.get("median_rms_m", {}).items()}, flush=True)
                for a in TEP_ARMS:
                    row = {bl: (mid.get(f"{a}_vs_{bl}_pct", {}) or {}).get("median")
                           for bl in BASELINES}
                    print(f"  {a} mid vs baselines:",
                          {k: (round(v, 1) if v is not None else None)
                           for k, v in row.items()}, flush=True)

    def _passing_cells(variant, horizon):
        cells = []
        for product in PRODUCTS:
            for reg in COHERENT_REGIONS:
                hh = (results.get(f"{product}_2025_{reg}", {})
                      .get("by_horizon", {}).get(horizon, {}))
                # a cell counts only if the variant beats every REQUIRED
                # deployable baseline AND every TEP-specificity null (network
                # mean, distance-scramble, phase-scramble)
                if (hh.get(f"{variant}_beats_all_baselines")
                        and hh.get(f"{variant}_beats_specificity_nulls")):
                    cells.append((product, reg))
        return cells

    def _variant_passes(variant, horizon="mid"):
        """A frozen TEP variant passes Level 1 if, on the INDEPENDENT year, it
        beats EVERY baseline (positive day-clustered CI) at the given outage
        horizon on >= 2 distinct COHERENT 2025 sub-networks, including >= 1
        non-European one."""
        cells = _passing_cells(variant, horizon)
        regs = {r for _, r in cells}
        return bool(len(regs) >= 2 and (regs & set(NON_EURO_REGIONS)))

    verdict = {}
    for a in TEP_ARMS:
        for hb in ("mid", "all", "long"):
            verdict[f"{a}_level1_pass_{hb}"] = _variant_passes(a, hb)
            verdict[f"{a}_passing_cells_{hb}"] = [
                f"{p}/{r}" for p, r in _passing_cells(a, hb)]
    # Level 1 is a claim about the PHYSICS arm only: the specificity nulls
    # (netmean, distance-scramble, phase-scramble) are controls that must be
    # BEATEN, never arms that can satisfy the verdict.  Iterate over the real
    # TEP arms so a passing null cannot trigger the flagship pass.
    _REAL_TEP_ARMS = tuple(a for a in TEP_ARMS if a not in SPECIFICITY_NULLS)
    verdict["level1_pass"] = bool(any(
        verdict[f"{a}_level1_pass_mid"] for a in _REAL_TEP_ARMS))
    verdict["note"] = (
        "Level 1 asks whether a frozen TEP phase field beats zero/linrate/ar1/"
        "free_gp/empirical_cov on the held-out year AND a non-European network. "
        "'mid' = 1-4 h outage, the regime where a leftover clock has drifted "
        "enough to predict.")

    out = {
        "step": 96,
        "title": "Leftover-clock supremacy (Level 1 flagship)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "products": PRODUCTS,
        "frozen_spec": {
            "lambda_km": LAMBDA_KM, "nugget": NUGGET,
            "bandpass_hz": [BP_LO_HZ, BP_HI_HZ],
            "lfmf_hz": [LF_LO_HZ, LF_HI_HZ], "gram_days_lfmf": GRAM_DAYS,
            "estimand": "y_i(t) = clk_i(t) - clk_i(t_a); accumulated offset since last fix",
            "note": ("spec fixed on the 2024 corpus; 2025 scored once. "
                     "empirical_cov is the data-driven map a physics-fixed "
                     "kernel must beat out of year."),
        },
        "results": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nLevel 1 {'PASS' if verdict['level1_pass'] else 'NOT PASSED'} -- saved {OUT}")


if __name__ == "__main__":
    main()
