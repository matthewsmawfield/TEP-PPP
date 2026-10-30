#!/usr/bin/env python3
"""
STEP 55  --  Causal outage clock prediction on a DENSE reference network.

Step 48b swept the receiver-outage length and found a network TEP prior never
beats temporal extrapolation of the receiver's own clock -- but that test ran
on the sparse global IGS network (~20 stations, median nearest reference
~750 km) and it kept leaning on the receiver's stale clock anchor ("trend +
held anchor residual + network-kriged *change*").  Step 54 showed the
distance-structured clock skill is a function of reference density.

This step repeats the outage sweep on the dense EUREF/EPN network (~25
stations, median nearest reference ~130 km) and lets the network estimate the
receiver clock *directly* rather than as a correction to a stale anchor:

  * the receiver has a clean clock history up to an anchor epoch t0, then
    loses all satellites for a gap G;
  * a causal offset+drift model is fitted for the receiver AND for every
    reference station on the 2 h window ending at t0;
  * at t0+G the references (which never lost their satellites) are observed;
    the receiver clock residual (w.r.t. its own extrapolated trend) is
    predicted from the reference residuals at t0+G.

Predictors of the receiver clock at t0+G:
  persistence   trend extrapolated + last pre-outage residual held
  ar1           trend + first-order Gauss-Markov residual decay
  best_temporal deployable pick of persistence / ar1 from the pre-window phi
  net_mean      trend + flat (uniform) mean of the reference residuals
  tep           trend + distance-kriged reference residuals
                (fixed Gaussian covariance, lambda = 4,200 km, nugget = 0.35)
  tep_scramble  tep with the station labels permuted in the covariance
                -- the distance-structure null

Strictly causal: no target data at or after t0 enters any network predictor,
and the TEP covariance is fixed by the physics (Step 53), zero tuning.

Output: results/outputs/step_55_dense_outage_prediction.json
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td   # noqa: E402
from scripts.lib import tep_kernel as tk   # noqa: E402
from scripts.lib.tep_clock import (  # noqa: E402
    MS_WRAP_M, PRODUCT_WRAP_M, align_network_epochs, percentile_abs,
    unwrap_series,
)

OUT = PROJECT_ROOT / "results" / "outputs"
_OUT_SUFFIX = "" if os.environ.get("TEP_KERNEL_FORM", "gaussian") == "gaussian" else "_legendre"
RNG = np.random.default_rng(20260902)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.  A global
# suppression masks every overflow/divide-by-zero in the whole process,
# including bugs.  Ill-conditioned kriging solves are wrapped in local
# np.errstate contexts where they occur.

C_M_PER_NS = 0.299792458
# The ppp_dense estimator stores its clock state modulo 1,000,000 metres
# (visible directly as a ±500 km sawtooth).  Steered receivers also wrap at
# 1 ms = 299,792.458 m.  Both periods are unwrapped causally.
CLOCK_WRAP_M = PRODUCT_WRAP_M
MS_CLOCK_WRAP_M = MS_WRAP_M
# A steered GNSS receiver clock stays within ~1 ms (300 km) of GPS time.  The
# `precise` product carries a handful of German stations (LEIJ, BRMG, FFMJ,
# TIT2, POTS) whose unwrapped clock reaches 130-215 ms (4e7-6e7 m) -- these are
# free-running or unwrap-failed series, not receiver clocks.  Left in the scored
# set they hit every arm (persistence, TEP, mean, scramble) with the same
# ~1e7 m error, which is RMS-dominant and collapses all relative comparisons to
# 0.0%.  Rejecting a non-physical *observable value* is a data-validity filter,
# not performance-based censoring (it is the same class as the existing
# `not np.isfinite(truth)` reject), so it is applied symmetrically and needs no
# causal justification.  Genuine sub-10-ms resets remain in the score and are
# tracked separately as target_reset_like_events.
MAX_PLAUSIBLE_CLOCK_M = 3_000_000.0
LAMBDA_KM = 4200.0
# Correlation model for the fixed TEP arm.  'gaussian' is the historical
# choice and the default.  'legendre' is the measured angular structure
# (Step 81/86): the only form here that carries the negative correlation
# lobe beyond ~5,250 km, which every parametric decay form is structurally
# unable to represent.  Set TEP_KERNEL_FORM=legendre to compare.
TEP_FORM = os.environ.get('TEP_KERNEL_FORM', 'gaussian')
# Nuisance local/spatial variance ratio.  Unlike lambda, this is not predicted
# by TEP: 0.35 is frozen from the earlier Step 25 precise-clock nested-CV
# experiment (median selected nugget), not selected on these 45 outage days.
NUGGET = 0.35

DT_S = 300.0
PREWINDOW_S = 7200.0
GAPS_S = [600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0]
EVAL_STRIDE = int(os.environ.get("TEP_EVAL_STRIDE", "3"))
N_DAYS = int(os.environ.get("TEP_N_DAYS", "90"))
MIN_REFS = 8


def _rho(d):
    return np.exp(-0.5 * (d / LAMBDA_KM) ** 2)


def _unwrap_clock_m(x):
    """Causally unwrap ppp_dense 1,000 km and 1 ms receiver-clock branches."""
    return unwrap_series(x, periods=(CLOCK_WRAP_M, MS_CLOCK_WRAP_M))


def load_day(dt):
    # Do not reject a station because of an event later in the day.  Availability
    # and finite-value checks are applied at each causal anchor below.
    s = td.load_series("precise", dt, min_valid_frac=0.0, max_jumps=10_000)
    if not s or len(s) < 12:
        return None
    names = list(s)
    raw = np.column_stack([s[n]["raw"] for n in names]).astype(float) * C_M_PER_NS
    raw = np.column_stack([_unwrap_clock_m(raw[:, i]) for i in range(raw.shape[1])])
    # Stations that start on different 1 ms branches stay offset after a
    # per-series unwrap.  Align each epoch onto the network median so a
    # live-reference kriging solve and the target truth share a branch.
    raw = align_network_epochs(raw, periods=(CLOCK_WRAP_M, MS_CLOCK_WRAP_M))
    resid = np.column_stack([s[n]["resid"] for n in names]).astype(float) * C_M_PER_NS
    t = s[names[0]]["t"].astype(float)
    # Only static/non-target-dependent screening here.  Full-day variance and
    # jump cuts would let future target data decide whether an outage is scored.
    keep = [i for i in range(len(names)) if np.isfinite(raw[:, i]).any()]
    if len(keep) < 10:
        return None
    names = [names[i] for i in keep]
    raw = raw[:, keep]
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            D[i, j] = D[j, i] = td.great_circle_km(s[names[i]]["station"],
                                                   s[names[j]]["station"])
    return names, t, raw, D


PRED_KEYS = (
    "pers", "ar1", "best_temporal", "linear_30m", "linear_60m",
    "net_mean", "shrink_mean", "nearest", "idw1", "idw2",
    "causal_cv_spatial", "tep", "tep_scramble",
)

CV_KERNELS = tuple(
    (form, lam, nug)
    for form in ("gaussian", "exponential")
    for lam in (100.0, 250.0, 500.0, 1000.0, 2000.0, 4200.0, 8000.0, 20000.0)
    for nug in (0.15, 0.35, 0.70)
)


def _rho_param(d, form, lam):
    """Correlation model -- delegated to the shared kernel module so the
    measured 'legendre' form is available here without a second copy of the
    parametric forms.  gaussian/exponential are bit-identical to the previous
    local implementation."""
    return tk._cov(np.asarray(d, float), form, lam)


def _solve_weights(d_rr, d_t, form=None, lam=LAMBDA_KM, nug=NUGGET):
    if form is None:
        form = TEP_FORM
    K = _rho_param(d_rr, form, lam) + (nug + 1e-9) * np.eye(len(d_t))
    # Local errstate: ill-conditioned K matrices on some CV grid points
    # produce overflow/invalid in the solve.  Guard with isfinite and fall
    # back to equal weights on failure.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            w = np.linalg.solve(K, _rho_param(d_t, form, lam))
        except np.linalg.LinAlgError:
            w = np.full(len(d_t), 1.0 / len(d_t))
    if not np.all(np.isfinite(w)):
        w = np.full(len(d_t), 1.0 / len(d_t))
    return w


def _causal_spatial_weights(ref_resid, target_resid, d_rr, d_t):
    """Choose a conventional covariance model using target history only.

    The target is available before the simulated outage, so selecting a model
    from its pre-outage residual is deployable.  The fixed TEP model must be
    compared with this flexible, causally trained spatial competitor.
    """
    best = (np.inf, None, None)
    for form, lam, nug in CV_KERNELS:
        try:
            w = _solve_weights(d_rr, d_t, form, lam, nug)
        except np.linalg.LinAlgError:
            continue
        score = float(np.sqrt(np.mean((target_resid - ref_resid @ w) ** 2)))
        if np.isfinite(score) and score < best[0]:
            best = (score, w, (form, lam, nug))
    return best[1], best[2]


def run(days):
    npre = int(PREWINDOW_S / DT_S)
    acc = {g: {k: [] for k in PRED_KEYS} for g in GAPS_S}
    acc_abs = {g: {k: [] for k in PRED_KEYS} for g in GAPS_S}
    per_day = {g: {k: [] for k in PRED_KEYS} for g in GAPS_S}
    per_day_n = {g: [] for g in GAPS_S}
    cv_selections = {}
    reference_integrity_rejections = {g: 0 for g in GAPS_S}
    reference_integrity_events = {g: 0 for g in GAPS_S}
    target_reset_like_events = {g: 0 for g in GAPS_S}
    target_reset_like_examples = {g: [] for g in GAPS_S}

    for dt in days:
        dd = load_day(dt)
        if dd is None:
            continue
        names, t, raw, D = dd
        n = len(names)
        nep = len(t)
        if n - 1 < MIN_REFS:
            continue
        perm = RNG.permutation(n)
        Dscr = D[np.ix_(perm, perm)]
        day_sq = {g: {k: [] for k in PRED_KEYS} for g in GAPS_S}
        day_abs = {g: {k: [] for k in PRED_KEYS} for g in GAPS_S}

        for ti in range(npre, nep - 1, EVAL_STRIDE):
            sl = slice(ti - npre, ti + 1)
            tt = t[sl] - t[ti]
            Adm = np.vstack([tt, np.ones_like(tt)]).T
            # Causal integrity screen from the pre-outage window only.  It
            # rejects unresolved resets/degenerate clocks without consulting
            # the target or references after the anchor.
            pre = raw[sl]
            pre_sd = np.full(n, np.inf)
            for qi in range(n):
                if np.isfinite(pre[:, qi]).all():
                    qc = np.polyfit(tt, pre[:, qi], 1)
                    pre_sd[qi] = np.std(pre[:, qi] - np.polyval(qc, tt))
            finite_sd = pre_sd[np.isfinite(pre_sd) & (pre_sd > 0.01)]
            if len(finite_sd) < MIN_REFS + 1:
                continue
            med_sd = float(np.median(finite_sd))
            causal_good = ((pre_sd > 0.01) & (pre_sd < 60.0) &
                           (pre_sd < 4.0 * med_sd))
            for b in range(n):
                # Reference availability is determined only from observations
                # known at the anchor and (later) at the prediction epoch.
                if not causal_good[b]:
                    continue
                base_pool = [j for j in range(n) if j != b and causal_good[j] and
                             np.isfinite(raw[sl, j]).all()]
                if len(base_pool) < MIN_REFS:
                    continue
                y = raw[sl, b]
                if not np.isfinite(y).all():
                    continue
                a1, a0 = np.polyfit(tt, y, 1)
                # A receiver with a clock drift this large would be steered,
                # not left to free-run through an outage -- extrapolating it is
                # meaningless and pollutes every RMS.  Causal, target-only cut.
                if abs(a1) > 0.20:            # m/s  (~0.7 ns/s)
                    continue
                pool = base_pool
                Y = raw[sl][:, pool]
                if not np.isfinite(Y).all():
                    continue
                coef = np.linalg.lstsq(Adm, Y, rcond=None)[0]     # (2, npool): slope, intercept
                # same causal drift cut for the references: a station with this
                # drift will be steered, and extrapolating its trend across the
                # outage produces a huge false residual for the network arms.
                ref_ok = np.abs(coef[0]) <= 0.20
                if ref_ok.sum() < MIN_REFS:
                    continue
                pool = [p for p, ok in zip(pool, ref_ok) if ok]
                Y = raw[sl][:, pool]
                coef = np.linalg.lstsq(Adm, Y, rcond=None)[0]
                rb = y - (a0 + a1 * tt)
                last = rb[-1]
                r0, r1 = rb[:-1], rb[1:]
                phi = np.clip(np.sum(r0 * r1) / max(np.sum(r0 * r0), 1e-12), -0.999, 0.999)
                # Horizon-matched temporal model selection.
                #
                # The previous code selected persistence vs AR(1) from causal
                # *one-step* prediction error.  That is the wrong objective for
                # a predictor that is scored at horizons of 10 min to 8 h: AR(1)
                # shrinks the prediction toward zero by phi**k, so it has lower
                # one-step error whenever phi < 1, but at the actual prediction
                # horizon the phi**k decay accumulates and persistence (holding
                # the last residual) is often better.  The one-step criterion
                # systematically picked AR(1) even on days where persistence won
                # at every scored horizon, making the deployable temporal
                # baseline artificially weak.
                #
                # Fix: for each gap k, evaluate both models at that horizon on
                # the pre-window (predict rb[i+k] from rb[i]).  For gaps longer
                # than the pre-window (k >= npre-1) no in-window horizon match
                # exists; fall back to the longest available horizon, which is
                # the most informative proxy for the true gap.
                npre_rb = len(rb)

                def _horizon_use_ar1(k_step):
                    """True if AR(1) beats persistence at horizon k_step on
                    the pre-window residuals."""
                    # Need at least 3 origins: npre_rb - best_h >= 3, so
                    # best_h <= npre_rb - 3.  The previous npre_rb - 1 bound
                    # left only 1 origin for the longest horizon, always
                    # returning False and forcing persistence for long gaps.
                    best_h = min(k_step, npre_rb - 3)
                    if best_h < 1:
                        return False
                    origins = np.arange(0, npre_rb - best_h)
                    if len(origins) < 3:
                        return False
                    pers_err = rb[origins + best_h] - rb[origins]
                    ar1_err = rb[origins + best_h] - (phi ** best_h) * rb[origins]
                    return float(np.mean(ar1_err ** 2)) < float(np.mean(pers_err ** 2))

                d_t = D[b, pool]
                d_rr = D[np.ix_(pool, pool)]
                try:
                    w = _solve_weights(d_rr, d_t)
                    ws = _solve_weights(Dscr[np.ix_(pool, pool)], Dscr[b, pool])
                except np.linalg.LinAlgError:
                    continue

                ref_hist = Y - (Adm @ coef)
                wcv, cv_param = _causal_spatial_weights(ref_hist, rb, d_rr, d_t)
                if wcv is None:
                    continue
                cv_selections[str(cv_param)] = cv_selections.get(str(cv_param), 0) + 1

                order = np.argsort(d_t)
                nearest_j = int(order[0])
                inv1 = 1.0 / np.maximum(d_t, 1.0)
                inv2 = inv1 ** 2
                widw1 = inv1 / inv1.sum()
                widw2 = inv2 / inv2.sum()

                for g in GAPS_S:
                    k = int(round(g / DT_S))
                    if ti + k >= nep:
                        continue
                    truth = raw[ti + k, b]
                    if not np.isfinite(truth):
                        continue
                    if abs(truth) > MAX_PLAUSIBLE_CLOCK_M:
                        continue
                    base = a0 + a1 * g
                    future_refs = raw[ti + k, pool]
                    if not np.isfinite(future_refs).all():
                        continue
                    if np.any(np.abs(future_refs) > MAX_PLAUSIBLE_CLOCK_M):
                        continue
                    refres = future_refs - (coef[0] * g + coef[1])
                    if not np.isfinite(refres).all():
                        continue

                    # Causal network integrity: a product branch/reset at one
                    # live reference is visible at the prediction epoch and
                    # must not poison every multi-reference predictor.  Screen
                    # innovations robustly, then recompute every spatial arm on
                    # identical surviving support.
                    rmed = float(np.median(refres))
                    rmad = 1.4826 * float(np.median(np.abs(refres - rmed)))
                    limit = max(10.0, 8.0 * rmad)
                    ok_ref = np.abs(refres - rmed) <= limit
                    if ok_ref.sum() < MIN_REFS:
                        continue
                    if not np.all(ok_ref):
                        reference_integrity_events[g] += 1
                        reference_integrity_rejections[g] += int((~ok_ref).sum())
                    if np.all(ok_ref):
                        pool_g, refres_g = pool, refres
                        w_g, ws_g, wcv_g = w, ws, wcv
                        nearest_g = nearest_j
                        widw1_g, widw2_g = widw1, widw2
                    else:
                        pool_g = [pool[ii] for ii in np.flatnonzero(ok_ref)]
                        refres_g = refres[ok_ref]
                        ref_hist_g = ref_hist[:, ok_ref]
                        d_t_g = D[b, pool_g]
                        d_rr_g = D[np.ix_(pool_g, pool_g)]
                        try:
                            w_g = _solve_weights(d_rr_g, d_t_g)
                            ws_g = _solve_weights(
                                Dscr[np.ix_(pool_g, pool_g)], Dscr[b, pool_g])
                        except np.linalg.LinAlgError:
                            continue
                        wcv_g, _ = _causal_spatial_weights(
                            ref_hist_g, rb, d_rr_g, d_t_g)
                        if wcv_g is None:
                            continue
                        order_g = np.argsort(d_t_g)
                        nearest_g = int(order_g[0])
                        inv1_g = 1.0 / np.maximum(d_t_g, 1.0)
                        inv2_g = inv1_g ** 2
                        widw1_g = inv1_g / inv1_g.sum()
                        widw2_g = inv2_g / inv2_g.sum()
                    pers = base + last
                    ar1 = base + last * (phi ** k)
                    use_ar1_g = _horizon_use_ar1(k)
                    best_t = ar1 if use_ar1_g else pers
                    n30 = max(3, int(1800 / DT_S) + 1)
                    n60 = max(3, int(3600 / DT_S) + 1)
                    p30 = np.polyfit(tt[-n30:], y[-n30:], 1)
                    p60 = np.polyfit(tt[-n60:], y[-n60:], 1)
                    preds = {
                        "pers": pers,
                        "ar1": ar1,
                        "best_temporal": best_t,
                        "linear_30m": float(np.polyval(p30, g)),
                        "linear_60m": float(np.polyval(p60, g)),
                        "net_mean": base + float(refres_g.mean()),
                        "shrink_mean": base + float(
                            refres_g.sum() / (len(pool_g) + NUGGET)),
                        "nearest": base + float(refres_g[nearest_g]),
                        "idw1": base + float(widw1_g @ refres_g),
                        "idw2": base + float(widw2_g @ refres_g),
                        "causal_cv_spatial": base + float(wcv_g @ refres_g),
                        "tep": base + float(w_g @ refres_g),
                        "tep_scramble": base + float(ws_g @ refres_g),
                    }
                    # Causal data-quality screen: a jump in the target's
                    # pre-gap residual (deviation from linear trend) larger
                    # than 200 m in the last 10 minutes signals a clock reset,
                    # cycle slip, or unwrapping artefact that makes any
                    # prediction meaningless.  This filter uses only pre-gap
                    # target data -- no future truth, no predictor output --
                    # so it is fully causal and symmetric across all arms.
                    # Normal epochs have median pre-gap residual jumps of
                    # ~3 m (99th pct ~140 m); corrupted epochs have median
                    # ~680 m.  The 200 m threshold separates the two
                    # distributions cleanly.
                    pre_jump = float(abs(rb[-1] - rb[-max(2, int(600.0 / DT_S))]))
                    if pre_jump > 200.0:
                        continue
                    # Detectable clock discontinuity during the outage.
                    #
                    # A 2024005/2024017 audit shows the surviving large-deviation
                    # endpoints are genuine ~1-1.5 ms receiver-clock resets
                    # (GRAZ +1.49 ms, OP71 +1.14 ms) -- 2-6 epochs across 90 days.
                    # These are NOT smooth holdover errors: every predictor
                    # (persistence, TEP, mean, scramble) fails on them by the full
                    # ~300-450 km jump, so pooled sum-of-squares is dominated by a
                    # handful of shared failures and every relative comparison
                    # collapses to 0.0%.  A real-time service monitoring the live
                    # reference network sees the target clock jump a full
                    # millisecond and re-initialises; it does not carry the stale
                    # clock into a position.  This benchmark measures clock
                    # PREDICTION skill across a data gap, which is undefined when
                    # the clock is discontinuous.
                    #
                    # 5 km ~ 17 us is far beyond the p99.9 holdover error (~520 m) and
                    # far below the reset magnitudes (~3-5e5 m); trivially flagged.
                    # The exclusion is a data-validity decision on the observable,
                    # reported transparently as target_reset_like_events, not a
                    # performance-based filter (both arms are affected identically).
                    if abs(truth - base) > 5000.0:
                        target_reset_like_events[g] += 1
                        if len(target_reset_like_examples[g]) < 20:
                            target_reset_like_examples[g].append({
                                "day": str(dt), "station": names[b],
                                "anchor_s": float(t[ti]),
                                "prediction_s": float(t[ti + k]),
                                "gap_s": float(g),
                                "truth_m": float(truth),
                                "pretrend_prediction_m": float(base),
                                "endpoint_deviation_m": float(truth - base),
                                "deviation_us": float((truth - base) / C_M_PER_NS / 1000.0),
                            })
                        continue
                    for kk, v in preds.items():
                        err = float(truth - v)
                        day_sq[g][kk].append(err ** 2)
                        day_abs[g][kk].append(abs(err))

        for g in GAPS_S:
            if len(day_sq[g]["pers"]) >= 25:
                per_day_n[g].append(len(day_sq[g]["pers"]))
                for kk in PRED_KEYS:
                    arr = np.asarray(day_sq[g][kk], float)
                    acc[g][kk].extend(arr.tolist())
                    acc_abs[g][kk].extend(day_abs[g][kk])
                    per_day[g][kk].append(float(np.mean(arr)))

    arms = {}
    for g in GAPS_S:
        if not acc[g]["pers"]:
            continue
        rms = {kk: float(np.sqrt(np.mean(acc[g][kk]))) for kk in PRED_KEYS}
        p50 = {kk: percentile_abs(acc_abs[g][kk], 50) for kk in PRED_KEYS}
        p95 = {kk: percentile_abs(acc_abs[g][kk], 95) for kk in PRED_KEYS}
        p99 = {kk: percentile_abs(acc_abs[g][kk], 99) for kk in PRED_KEYS}

        def gain_ci(num_key, den_key):
            a = np.asarray(per_day[g][num_key])
            d = np.asarray(per_day[g][den_key])
            counts = np.asarray(per_day_n[g], float)
            if len(a) < 4:
                return None
            bs = []
            for idx in (RNG.integers(0, len(a), len(a)) for _ in range(4000)):
                ra = np.sqrt(np.average(a[idx], weights=counts[idx]))
                rd = np.sqrt(np.average(d[idx], weights=counts[idx]))
                bs.append(100.0 * (1.0 - ra / rd))
            return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

        temporal_rms = {k: rms[k] for k in
                        ("pers", "ar1", "best_temporal", "linear_30m", "linear_60m")}
        conventional_rms = {k: rms[k] for k in
                            ("net_mean", "shrink_mean", "nearest", "idw1", "idw2",
                             "causal_cv_spatial")}
        best_temporal_key = min(temporal_rms, key=temporal_rms.get)
        best_conventional_key = min(conventional_rms, key=conventional_rms.get)

        arms[str(int(g))] = {
            "n_eval": len(acc[g]["pers"]),
            "n_days": len(per_day[g]["pers"]),
            "rms_m": rms,
            "p50_m": p50,
            "p95_m": p95,
            "p99_m": p99,
            "best_temporal_posthoc": {"method": best_temporal_key,
                                       "rms_m": temporal_rms[best_temporal_key]},
            "best_conventional_spatial_posthoc": {"method": best_conventional_key,
                                                    "rms_m": conventional_rms[best_conventional_key]},
            "tep_vs_pers_rms_pct": 100.0 * (1.0 - rms["tep"] / rms["pers"]),
            "tep_vs_best_temporal_rms_pct": 100.0 * (1.0 - rms["tep"] / rms["best_temporal"]),
            "tep_vs_net_mean_rms_pct": 100.0 * (1.0 - rms["tep"] / rms["net_mean"]),
            # day-clustered point estimates: median over days of the per-day RMS
            # ratio, robust to a single heavy-tailed day
            "tep_vs_pers_rms_pct_dayclustered": (
                100.0 * (1.0 - float(np.median(
                    np.sqrt(np.asarray(per_day[g]["tep"])) /
                    np.sqrt(np.asarray(per_day[g]["pers"])))))
                if len(per_day[g]["pers"]) >= 4 else None),
            "tep_vs_net_mean_rms_pct_dayclustered": (
                100.0 * (1.0 - float(np.median(
                    np.sqrt(np.asarray(per_day[g]["tep"])) /
                    np.sqrt(np.asarray(per_day[g]["net_mean"])))))
                if len(per_day[g]["net_mean"]) >= 4 else None),
            "tep_vs_best_temporal_posthoc_rms_pct": 100.0 * (
                1.0 - rms["tep"] / temporal_rms[best_temporal_key]),
            "tep_vs_best_conventional_spatial_posthoc_rms_pct": 100.0 * (
                1.0 - rms["tep"] / conventional_rms[best_conventional_key]),
            "tep_vs_causal_cv_spatial_rms_pct": 100.0 * (
                1.0 - rms["tep"] / rms["causal_cv_spatial"]),
            "scramble_vs_net_mean_rms_pct": 100.0 * (1.0 - rms["tep_scramble"] / rms["net_mean"]),
            "tep_vs_pers_p50_pct": 100.0 * (1.0 - p50["tep"] / p50["pers"]),
            "tep_vs_best_temporal_p50_pct": 100.0 * (1.0 - p50["tep"] / p50["best_temporal"]),
            "tep_vs_causal_cv_spatial_p50_pct": 100.0 * (
                1.0 - p50["tep"] / p50["causal_cv_spatial"]),
            "tep_vs_pers_p99_pct": 100.0 * (1.0 - p99["tep"] / p99["pers"]),
            "tep_vs_pers_ci95": gain_ci("tep", "pers"),
            "tep_vs_best_temporal_ci95": gain_ci("tep", "best_temporal"),
            "tep_vs_net_mean_ci95": gain_ci("tep", "net_mean"),
            "tep_vs_best_temporal_posthoc_ci95": gain_ci("tep", best_temporal_key),
            "tep_vs_best_conventional_spatial_posthoc_ci95": gain_ci(
                "tep", best_conventional_key),
            "tep_vs_causal_cv_spatial_ci95": gain_ci("tep", "causal_cv_spatial"),
            "reference_integrity": {
                "events_with_rejection": int(reference_integrity_events[g]),
                "references_rejected": int(reference_integrity_rejections[g]),
                "rule": "abs(innovation - median) <= max(10 m, 8*MAD_sigma)",
            },
            "target_reset_like_events": int(target_reset_like_events[g]),
            "target_reset_like_examples": target_reset_like_examples[g][:10],
        }
    return arms, cv_selections


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates("precise", 2024)
    days = [dates[i] for i in np.linspace(0, len(dates) - 1, N_DAYS).astype(int)]

    arms, cv_selections = run(days)
    out = {
        "step": 55,
        "title": "Causal outage clock prediction on a dense reference network",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "network": ("dense EUREF/EPN precise clocks (~25 stations, "
                    "median nearest reference ~130 km)"),
        "design": ("causal outage sweep; the network estimates the receiver clock "
                   "residual directly from reference residuals at t0+G (causal per-station "
                   "trends on the 2 h pre-anchor window); TEP covariance fixed (Gaussian, "
                   "lambda = 4,200 km; nuisance nugget = 0.35 frozen from the "
                   "earlier Step 25 precise-clock nested CV), compared with temporal, "
                   "nearest-neighbour, IDW, flat/shrunk mean, and causally CV-selected "
                   "spatial covariance controls. Day membership is sampled in advance; "
                   "per-anchor station availability uses only the causal window and "
                   "reference values available at the prediction epoch."),
        "n_days": N_DAYS,
        "tep_kernel_form": TEP_FORM,
        "causal_cv_spatial_selections": cv_selections,
        "by_gap": arms,
    }
    if arms:
        gaps = sorted(arms, key=int)
        out["headline"] = {
            "tep_vs_persistence_rms_pct": {g: arms[g]["tep_vs_pers_rms_pct"] for g in gaps},
            "tep_vs_best_temporal_rms_pct": {g: arms[g]["tep_vs_best_temporal_rms_pct"] for g in gaps},
            "tep_vs_net_mean_rms_pct": {g: arms[g]["tep_vs_net_mean_rms_pct"] for g in gaps},
            "tep_vs_best_temporal_posthoc_rms_pct": {
                g: arms[g]["tep_vs_best_temporal_posthoc_rms_pct"] for g in gaps},
            "tep_vs_best_conventional_spatial_posthoc_rms_pct": {
                g: arms[g]["tep_vs_best_conventional_spatial_posthoc_rms_pct"] for g in gaps},
            "tep_vs_causal_cv_spatial_rms_pct": {
                g: arms[g]["tep_vs_causal_cv_spatial_rms_pct"] for g in gaps},
            "tep_vs_persistence_p50_pct": {g: arms[g]["tep_vs_pers_p50_pct"] for g in gaps},
            "tep_vs_best_temporal_p50_pct": {
                g: arms[g]["tep_vs_best_temporal_p50_pct"] for g in gaps},
            "distance_scramble_vs_net_mean_rms_pct": {g: arms[g]["scramble_vs_net_mean_rms_pct"] for g in gaps},
            "min_tep_vs_best_temporal_pct": min(arms[g]["tep_vs_best_temporal_rms_pct"] for g in gaps),
            "max_tep_vs_best_temporal_pct": max(arms[g]["tep_vs_best_temporal_rms_pct"] for g in gaps),
            "interpretation": (
                "Result-dependent interpretation intentionally omitted from the executable. "
                "Promotion requires positive paired day-cluster confidence intervals against "
                "the strongest temporal and conventional spatial controls, a near-zero "
                "distance-scramble result, and independent end-to-end position validation."),
        }

    path = OUT / f"step_55_dense_outage_prediction{_OUT_SUFFIX}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")
    print(f"{'gap':>7} {'pers RMS':>9} {'tep/pers':>10} {'tep/bestT':>11} "
          f"{'tep/bestS':>11} {'tep/netmean':>12} {'scr/netmean':>12}")
    for g in sorted(arms, key=int):
        a = arms[g]
        print(f"{g:>7} {a['rms_m']['pers']:>8.2f}m {a['tep_vs_pers_rms_pct']:>+9.1f}% "
              f"{a['tep_vs_best_temporal_posthoc_rms_pct']:>+10.1f}% "
              f"{a['tep_vs_best_conventional_spatial_posthoc_rms_pct']:>+10.1f}% "
              f"{a['tep_vs_net_mean_rms_pct']:>+11.1f}% "
              f"{a['scramble_vs_net_mean_rms_pct']:>+11.1f}%")


if __name__ == "__main__":
    main()
