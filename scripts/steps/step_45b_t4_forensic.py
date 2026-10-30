#!/usr/bin/env python3
"""
STEP 45b -- TEMPORAL T4 FORENSIC AUDIT.

Step 45 reports that the non-common-view component alpha_ij of the within-pair
co-visibility regression retains distance structure at lambda ~ 3.9-4.2 Mm
(R^2 sigma > 3 vs a within-station epoch-shuffle null).  But Step 45's own
numbers show the *partial correlation* of alpha with log(d) given mean
co-visibility Jbar is ~0 (PC sigma 0.6-1.3), and corr(d, Jbar) = -0.97 to -0.98.
The pass currently rests on the binned-distance R^2 criterion, which still
inherits the residual d-Jbar collinearity.

2026-10-29 update: the stage-1 estimator is corrected to the pooled
common-view slope (pair fixed effects) on real-SP3 co-visibility -- see
step_45_temporal_t4.py.  The deprecated per-pair OLS beta is retained as the
"perpair_beta" spec so the battery documents what it cost; "pooled_distbin"
and "dlon_control" are new robustness specs.

This step stress-tests the result with the full control battery:

  A. SPEC BATTERY -- refit stage 2 under alternative specifications, reporting
     BOTH the binned-distance lambda/R^2 AND the partial correlation of alpha
     with log d given the controls:
       baseline            pooled-kappa, real-SP3 J, linear Jbar control
       perpair_beta        deprecated per-pair OLS beta (attenuated stage-2)
       pooled_distbin      pooled kappa per distance decile (heterogeneity)
       dlon_control        controls = [Jbar, cos(delta-lon)] (solar-time syst.)
       jbar_nonlinear      controls = [Jbar, Jbar^2, Jbar^3]  (degenerate spec:
                           corr(d,Jbar) ~ -0.95, so a cubic in Jbar can
                           reparametrise the distance axis itself; its
                           partial correlation collapse is mechanical, not
                           evidence of confounding)
       covis_count         co-visibility = raw common-satellite count
       covis_overlap       co-visibility = |inter| / min(|Vi|, |Vj|)
       covis_elevweight    co-visibility = sum_common min(sin e_i, sin e_j)
       nsat_control        add mean visible-sat count as a stage-2 control
       day_fixed_effects   remove per-day mean of alpha before pooling
       matched_lowcovis    stage-1 uses only epochs with J < pair 33rd pctile
       beta_trimmed        drop pairs with |beta| outside the 10-90 pctile
       long_baseline       pairs with d > 2000 km only (weakest collinearity)

  B. LEAVE-ONE-STATION-OUT -- drop each station, refit baseline, report the
     lambda distribution and the fraction landing in the TEP band.

  C. LEAVE-ONE-DAY-OUT / LEAVE-ONE-WEEK-OUT -- same, over day blocks.

  D. STATION-CLUSTERED BOOTSTRAP -- resample stations with replacement,
     weight pairs by endpoint multiplicity, refit stage 2; report the
     lambda and R^2 95% CIs.  Replaces Step 45's pair-naive binned SEM.

Output: results/outputs/step_45b_t4_forensic.json
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
from scripts.lib import tep_data as td            # noqa: E402
from scripts.lib import tep_stats as ts           # noqa: E402
from scripts.steps.step_13_commonview import gps_constellation  # noqa: E402
from scripts.steps.step_45_temporal_t4 import (   # noqa: E402
    _precompute_day, _per_epoch_covisibility_from_vis,
    load_real_visibility, N_EPOCHS)

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260901)
ELEV_MASK_RAD = np.radians(10.0)
TEP_BAND = (2000.0, 6000.0)     # forensic band -- tighter than step 45's 500-20000


# --------------------------------------------------------------------------- vis
def _visible_elev(sat_ecef, sta_ecef_m):
    """(sin_elev array, visible bool array) for one station."""
    up = sta_ecef_m / np.linalg.norm(sta_ecef_m)
    los = sat_ecef - sta_ecef_m
    los = los / np.linalg.norm(los, axis=1, keepdims=True)
    sin_e = np.clip(los @ up, -1, 1)
    return sin_e, np.arcsin(sin_e) > ELEV_MASK_RAD


def _covis_variants(sts, epochs_s, date_tag=None):
    """Per-pair per-epoch co-visibility under four definitions.

    Returns dict of (n_pairs, n_epochs) arrays: jaccard, count, overlap, elev.
    When `date_tag` is given and the day's SP3 product exists, visibility
    comes from the REAL ephemerides at UTC seconds-of-day; otherwise the
    nominal 24-satellite model is used (arbitrarily Earth-phased, which
    scrambles per-epoch J -- the reason the pooled-kappa stage-1 is used).
    """
    n = len(sts)
    ne = len(epochs_s)
    rv = load_real_visibility(date_tag, sts, epochs_s) if date_tag else None
    if rv is not None:
        vis, sinE = rv
    else:
        sat_count = None
        for e, t in enumerate(epochs_s):
            sat = gps_constellation(float(t))
            if sat_count is None:
                sat_count = sat.shape[0]
                sinE = np.zeros((ne, n, sat_count))
                vis = np.zeros((ne, n, sat_count), bool)
            for i, st in enumerate(sts):
                se, vm = _visible_elev(sat, st.ecef_m)
                sinE[e, i] = se
                vis[e, i] = vm
    pairs = list(itertools.combinations(range(n), 2))
    out = {k: np.zeros((len(pairs), ne)) for k in ("jaccard", "count", "overlap", "elev")}
    for pi, (i, j) in enumerate(pairs):
        both = vis[:, i] & vis[:, j]
        either = vis[:, i] | vis[:, j]
        inter = both.sum(1).astype(float)
        union = np.clip(either.sum(1).astype(float), 1.0, None)
        mn = np.clip(np.minimum(vis[:, i].sum(1), vis[:, j].sum(1)).astype(float), 1.0, None)
        out["jaccard"][pi] = inter / union
        out["count"][pi] = inter
        out["overlap"][pi] = inter / mn
        ew = np.where(both, np.minimum(sinE[:, i], sinE[:, j]), 0.0).sum(1)
        out["elev"][pi] = ew
    nsat = vis.sum(2)  # (ne, n)
    return out, pairs, nsat


# ------------------------------------------------------------------ pair table
def _build_pair_table(precomputed, epochs_s):
    """One row per (pair, day). Cache each pair's co-visibility variants, and
    the per-day per-station normalised residual columns, so specs are cheap
    recomputes AND the null can shuffle the *underlying station series* (not the
    already-formed product -- alpha = mean(p) - beta*mean(J), and mean(p) is
    permutation-invariant, so shuffling p is a toothless null).

    `precomputed` entries are (date_tag, pc_tuple) so co-visibility uses the
    day's real SP3 ephemerides."""
    rows = []
    gstations = {}   # global station name -> index
    day_cols = []    # per day: (n_epochs, n_local_sta) float array
    for day_idx, (date_tag, (M_norm, _J_pe, _pairs_idx, nms_c, sts_c)) in enumerate(precomputed):
        cv, pairs, nsat = _covis_variants(sts_c, epochs_s, date_tag=date_tag)
        day_cols.append(np.asarray(M_norm, float))
        lons = np.array([np.arctan2(st.ecef_m[1], st.ecef_m[0]) for st in sts_c])
        for pi, (i, j) in enumerate(pairs):
            p = M_norm[:, i] * M_norm[:, j]
            if np.isfinite(p).sum() < 10:
                continue
            d = td.great_circle_km(sts_c[i], sts_c[j])
            if d < 100 or d > 20000:
                continue
            dl = abs(lons[i] - lons[j])
            for nm in (nms_c[i], nms_c[j]):
                gstations.setdefault(nm, len(gstations))
            rows.append({
                "si": gstations[nms_c[i]], "sj": gstations[nms_c[j]],
                "day": day_idx, "li": i, "lj": j, "dist": float(d),
                "dlon": float(min(dl, 2 * np.pi - dl)),
                "jaccard": cv["jaccard"][pi].astype(np.float32),
                "count": cv["count"][pi].astype(np.float32),
                "overlap": cv["overlap"][pi].astype(np.float32),
                "elev": cv["elev"][pi].astype(np.float32),
                "nsat": (0.5 * (nsat[:, i] + nsat[:, j])).astype(np.float32),
            })
    return rows, gstations, day_cols


# ------------------------------------------------------------------ regression
def _alpha_for_rows(rows, day_cols, covis_key="jaccard", matched_lowcovis=False,
                    beta_mode="pooled"):
    """Stage 1 -> (alpha, dist, jbar, beta, nsat_bar, day, si, sj).

    beta_mode:
      "pooled"   -- one common-view slope per spec call, pair fixed effects
                    (kappa = sum (J-Jbar)(p-pbar) / sum (J-Jbar)^2).  The
                    corrected estimator: a per-pair slope from 48 epochs is
                    almost pure noise, and alpha then carries -err*Jbar which
                    is distance-aligned and attenuates the stage-2 partial
                    correlation.
      "distbin"  -- pooled slope per distance decile (heterogeneous kappa).
      "perpair"  -- the deprecated per-pair OLS slope, kept as a spec so the
                    battery can show what it cost.
    `day_cols` supplies the per-day per-station residual columns; the product
    is formed fresh here so the epoch-shuffle null exercises the whole
    estimator (kappa included).
    """
    items = []
    for r in rows:
        M = day_cols[r["day"]]
        p_full = M[:, r["li"]] * M[:, r["lj"]]
        g = np.isfinite(p_full)
        if g.sum() < 10:
            continue
        p_v = p_full[g]
        j_v = np.asarray(r[covis_key], float)[g]
        ns = np.asarray(r["nsat"], float)[g]
        if matched_lowcovis:
            thr = np.percentile(j_v, 33.0)
            m = j_v <= thr
            if m.sum() < 6:
                continue
            p_v, j_v, ns = p_v[m], j_v[m], ns[m]
        items.append((p_v, j_v, ns, r))
    if not items:
        return None

    def _k(pp, jj):
        jc = jj - jj.mean()
        return float(np.sum(jc * (pp - pp.mean()))), float(np.sum(jc * jc))

    kappa = 0.0
    kappas_bin = None
    if beta_mode == "pooled":
        num = den = 0.0
        for p_v, j_v, _, _ in items:
            n_, d_ = _k(p_v, j_v)
            num += n_; den += d_
        kappa = num / den if den > 1e-12 else 0.0
    elif beta_mode == "distbin":
        edges = np.percentile([r["dist"] for *_, r in items], np.linspace(0, 100, 11))
        nb = np.zeros(10); db = np.zeros(10)
        for p_v, j_v, _, r in items:
            b = min(int(np.searchsorted(edges, r["dist"]) - 1), 9)
            if b < 0:
                b = 0
            n_, d_ = _k(p_v, j_v)
            nb[b] += n_; db[b] += d_
        kappas_bin = np.divide(nb, db, out=np.zeros_like(nb), where=db > 1e-12)

    out = []
    for p_v, j_v, ns, r in items:
        if beta_mode == "perpair":
            jc = j_v - j_v.mean()
            j_scale = abs(j_v).mean() + 1e-9
            if jc.std() < 1e-4 * j_scale:
                continue
            denom = float(np.sum(jc ** 2))
            if denom < 1e-9 * j_scale ** 2:
                continue
            beta = float(np.sum(jc * (p_v - p_v.mean())) / denom)
            # beta is a slope of a unit-scale residual product on co-visibility;
            # |beta| >> a few means jc is near-degenerate -> drop the pair rather
            # than let a 1e12 alpha poison the stage-2 regression.
            if not np.isfinite(beta) or abs(beta) > 50.0:
                continue
        elif beta_mode == "distbin":
            b = min(int(np.searchsorted(edges, r["dist"]) - 1), 9)
            beta = float(kappas_bin[max(b, 0)])
        else:
            beta = float(kappa)
        alpha = float(p_v.mean() - beta * j_v.mean())
        if not np.isfinite(alpha) or abs(alpha) > 50.0:
            continue
        out.append((alpha, r["dist"], float(j_v.mean()), beta,
                    float(ns.mean()), r["day"], r["si"], r["sj"], r["dlon"]))
    if not out:
        return None
    keys = ("alpha", "dist", "jbar", "beta", "nsat", "day", "si", "sj", "dlon")
    arr = {k: np.array([o[i] for o in out], float) for i, k in enumerate(keys)}
    return arr


def _resid(y, X):
    """Residual of y on design X (intercept added, columns standardised for
    conditioning, rank-deficiency tolerated)."""
    X = X if X.ndim == 2 else X.reshape(-1, 1)
    Xs = X - X.mean(0)
    sd = Xs.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = np.clip((X - X.mean(0)) / sd, -1e6, 1e6)
    A = np.column_stack([np.ones(len(y)), Xs])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        beta, *_ = np.linalg.lstsq(A, y, rcond=1e-8)
        r = y - A @ beta
    return r if np.all(np.isfinite(r)) else y - y.mean()


def _stage2(arr, controls=("jbar",), day_fe=False, dmin=100.0):
    """Partial correlation of alpha vs log d given controls, plus the binned
    exponential distance fit (lambda, R^2)."""
    alpha = arr["alpha"].copy()
    dist = arr["dist"]
    logd = np.log(dist + 1.0)

    ctrl_cols = []
    for c in controls:
        if c == "jbar":
            ctrl_cols.append(arr["jbar"])
        elif c == "jbar2":
            ctrl_cols.append(arr["jbar"] ** 2)
        elif c == "jbar3":
            ctrl_cols.append(arr["jbar"] ** 3)
        elif c == "nsat":
            ctrl_cols.append(arr["nsat"])
        elif c == "cosdlon":
            ctrl_cols.append(np.cos(arr["dlon"]))
    if day_fe:
        days = arr["day"].astype(int)
        for d in np.unique(days)[1:]:
            ctrl_cols.append((days == d).astype(float))
        # also remove per-day mean of alpha
        for d in np.unique(days):
            m = days == d
            alpha[m] = alpha[m] - alpha[m].mean()

    Z = np.column_stack(ctrl_cols) if ctrl_cols else np.zeros((len(alpha), 1))
    ra = _resid(alpha, Z)
    rl = _resid(logd, Z)
    pc = float(np.corrcoef(ra, rl)[0, 1]) if ra.std() > 0 and rl.std() > 0 else np.nan
    # same partial correlation with cos(delta-lon) added to the control set
    # (skipped if cosdlon is already a control), and the longitude-phased
    # (solar-time) component itself
    if "cosdlon" in controls:
        Zd = Z
    else:
        Zd = np.column_stack([Z, np.cos(arr["dlon"])])
    ra2 = _resid(alpha, Zd)
    rl2 = _resid(logd, Zd)
    pc_dlctrl = float(np.corrcoef(ra2, rl2)[0, 1]) if ra2.std() > 0 and rl2.std() > 0 else np.nan
    rd = _resid(np.cos(arr["dlon"]), Z)
    pc_dlon = float(np.corrcoef(ra, rd)[0, 1]) if ra.std() > 0 and rd.std() > 0 else np.nan

    ctr, mn, sem, cnt = ts.bin_by_distance(
        dist, alpha, n_bins=15, dmin=dmin,
        dmax=np.percentile(dist, 99), min_count=10)
    if len(ctr) < 4:
        return {"pc": pc, "pc_dlon": pc_dlon, "pc_dlctrl": pc_dlctrl,
                "lam": np.nan, "R2": np.nan, "n_bins": len(ctr), "n_pairs": len(alpha)}
    fit = ts.fit_correlation_length(ctr, mn, weight=1.0 / np.clip(sem, 1e-4, None) ** 2)
    return {"pc": pc, "pc_dlon": pc_dlon, "pc_dlctrl": pc_dlctrl,
            "lam": float(fit["lam"]), "R2": float(fit["R2"]),
            "n_bins": int(len(ctr)), "n_pairs": int(len(alpha))}


# ----------------------------------------------------------------- null (spec)
def _shuffle_day_cols(day_cols):
    """Permute the epoch order of every station column independently, per day.
    Destroys all cross-station co-fluctuation while preserving each station's
    marginal residual distribution -- so mean(M_i * M_j) collapses toward zero
    (unlike shuffling the already-formed product, whose mean is invariant)."""
    out = []
    for M in day_cols:
        Ms = M.copy()
        for c in range(Ms.shape[1]):
            Ms[:, c] = Ms[RNG.permutation(Ms.shape[0]), c]
        out.append(Ms)
    return out


def _null_r2_sigma(rows, day_cols, spec_fn, n_null=80):
    """Per-station epoch-shuffle null for one spec (shuffles the underlying
    station series, then re-forms all products). Returns (r2_sigma, null_lam,
    real)."""
    real = spec_fn(rows, day_cols)
    if not np.isfinite(real["R2"]):
        return np.nan, np.nan, real
    r2s, lams = [], []
    for _ in range(n_null):
        res = spec_fn(rows, _shuffle_day_cols(day_cols))
        if np.isfinite(res["R2"]):
            r2s.append(res["R2"])
        if np.isfinite(res["lam"]):
            lams.append(res["lam"])
    r2s = np.array(r2s)
    r2_sig = float((real["R2"] - r2s.mean()) / (r2s.std() + 1e-9)) if len(r2s) > 5 else np.nan
    return r2_sig, (float(np.median(lams)) if lams else np.nan), real


# ------------------------------------------------------------------- one arm
def _spec_battery(rows, day_cols):
    specs = {
        "baseline":        lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="pooled"), ("jbar",)),
        "perpair_beta":    lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="perpair"), ("jbar",)),
        "pooled_distbin":  lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="distbin"), ("jbar",)),
        "dlon_control":    lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="pooled"), ("jbar", "cosdlon")),
        "jbar_nonlinear":  lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="pooled"), ("jbar", "jbar2", "jbar3")),
        "covis_count":     lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "count", beta_mode="pooled"), ("jbar",)),
        "covis_overlap":   lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "overlap", beta_mode="pooled"), ("jbar",)),
        "covis_elevweight": lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "elev", beta_mode="pooled"), ("jbar",)),
        "nsat_control":    lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="pooled"), ("jbar", "nsat")),
        "day_fixed_effects": lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", beta_mode="pooled"), ("jbar",), day_fe=True),
        "matched_lowcovis": lambda rr, dc: _stage2(_alpha_for_rows(rr, dc, "jaccard", matched_lowcovis=True, beta_mode="pooled"), ("jbar",)),
        "beta_trimmed":    _spec_beta_trimmed,
        "long_baseline":   _spec_long_baseline,
    }
    out = {}
    for name, fn in specs.items():
        try:
            r2_sig, null_lam, real = _null_r2_sigma(rows, day_cols, fn,
                                                    n_null=60 if name != "baseline" else 120)
            out[name] = {
                "lambda_km": real["lam"], "R2": real["R2"],
                "partial_corr_alpha_logd_given_controls": real["pc"],
                "partial_corr_alpha_logd_given_controls_cosdlon": real.get("pc_dlctrl"),
                "partial_corr_alpha_cosdlon_given_controls": real.get("pc_dlon"),
                "n_pairs": real["n_pairs"], "n_bins": real["n_bins"],
                "null_r2_sigma": r2_sig, "null_lambda_median": null_lam,
                "lambda_in_TEP_band": bool(np.isfinite(real["lam"])
                                           and TEP_BAND[0] <= real["lam"] <= TEP_BAND[1]),
                "survives": bool(np.isfinite(real["lam"]) and np.isfinite(r2_sig)
                                 and r2_sig > 3.0 and real["R2"] > 0.3
                                 and 800 <= real["lam"] <= 12000),
            }
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)}
    return out


def _spec_beta_trimmed(rows, day_cols):
    a0 = _alpha_for_rows(rows, day_cols, "jaccard")
    if a0 is None:
        return {"pc": np.nan, "lam": np.nan, "R2": np.nan, "n_bins": 0, "n_pairs": 0}
    lo, hi = np.percentile(a0["beta"], [10, 90])
    keep = (a0["beta"] >= lo) & (a0["beta"] <= hi)
    return _stage2({k: v[keep] for k, v in a0.items()}, ("jbar",))


def _spec_long_baseline(rows, day_cols):
    a0 = _alpha_for_rows(rows, day_cols, "jaccard")
    if a0 is None:
        return {"pc": np.nan, "lam": np.nan, "R2": np.nan, "n_bins": 0, "n_pairs": 0}
    keep = a0["dist"] > 2000.0
    if keep.sum() < 200:
        return {"pc": np.nan, "lam": np.nan, "R2": np.nan, "n_bins": 0, "n_pairs": int(keep.sum())}
    return _stage2({k: v[keep] for k, v in a0.items()}, ("jbar",), dmin=2000.0)


def _loso(rows, gstations, day_cols):
    base = _stage2(_alpha_for_rows(rows, day_cols, "jaccard"), ("jbar",))
    lams = []
    names = {v: k for k, v in gstations.items()}
    for s_idx in sorted(names):
        sub = [r for r in rows if r["si"] != s_idx and r["sj"] != s_idx]
        if len(sub) < 300:
            continue
        res = _stage2(_alpha_for_rows(sub, day_cols, "jaccard"), ("jbar",))
        if np.isfinite(res["lam"]):
            lams.append(res["lam"])
    lams = np.array(lams)
    return {
        "baseline_lambda_km": base["lam"], "n_drops": int(len(lams)),
        "loso_lambda_median": float(np.median(lams)) if len(lams) else np.nan,
        "loso_lambda_iqr": [float(np.percentile(lams, 25)), float(np.percentile(lams, 75))] if len(lams) else [np.nan, np.nan],
        "loso_frac_in_TEP_band": float(np.mean((lams >= TEP_BAND[0]) & (lams <= TEP_BAND[1]))) if len(lams) else np.nan,
        "loso_max_shift_km": float(np.max(np.abs(lams - base["lam"]))) if len(lams) and np.isfinite(base["lam"]) else np.nan,
    }


def _lodo(rows, day_cols):
    days = sorted({r["day"] for r in rows})
    base = _stage2(_alpha_for_rows(rows, day_cols, "jaccard"), ("jbar",))
    lams = []
    for d in days:
        sub = [r for r in rows if r["day"] != d]
        res = _stage2(_alpha_for_rows(sub, day_cols, "jaccard"), ("jbar",))
        if np.isfinite(res["lam"]):
            lams.append(res["lam"])
    lams = np.array(lams)
    return {
        "baseline_lambda_km": base["lam"], "n_days": len(days),
        "lodo_lambda_median": float(np.median(lams)) if len(lams) else np.nan,
        "lodo_lambda_range": [float(lams.min()), float(lams.max())] if len(lams) else [np.nan, np.nan],
        "lodo_frac_in_TEP_band": float(np.mean((lams >= TEP_BAND[0]) & (lams <= TEP_BAND[1]))) if len(lams) else np.nan,
    }


def _station_bootstrap(rows, gstations, day_cols, n_boot=300):
    names = sorted(gstations.values())
    lam_s, r2_s, pc_s = [], [], []
    for _ in range(n_boot):
        draw = RNG.choice(names, size=len(names), replace=True)
        mult = {}
        for s in draw:
            mult[s] = mult.get(s, 0) + 1
        sub = []
        for r in rows:
            m = mult.get(r["si"], 0) * mult.get(r["sj"], 0)
            if m > 0:
                for _k in range(min(m, 4)):
                    sub.append(r)
        if len(sub) < 300:
            continue
        res = _stage2(_alpha_for_rows(sub, day_cols, "jaccard"), ("jbar",))
        if np.isfinite(res["lam"]):
            lam_s.append(res["lam"]); r2_s.append(res["R2"]); pc_s.append(res["pc"])
    lam_s, r2_s, pc_s = map(np.array, (lam_s, r2_s, pc_s))
    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]
    return {
        "n_boot_ok": int(len(lam_s)),
        "lambda_km_median": float(np.median(lam_s)) if len(lam_s) else np.nan,
        "lambda_km_ci95": ci(lam_s),
        "R2_median": float(np.median(r2_s)) if len(r2_s) else np.nan,
        "R2_ci95": ci(r2_s),
        "partial_corr_median": float(np.median(pc_s)) if len(pc_s) else np.nan,
        "partial_corr_ci95": ci(pc_s),
        "lambda_ci_in_TEP_band": bool(len(lam_s) and ci(lam_s)[0] >= 1000 and ci(lam_s)[1] <= 12000),
    }


def run_arm(product, date_tags, label, max_stations=None, systems=("G",)):
    days = list(td.load_multi_day(product, date_tags))
    # UTC seconds-of-day, aligned with the residual grid and SP3 epoch axis
    epochs_s = np.linspace(0.0, 86400.0, N_EPOCHS, endpoint=False)
    precomputed = []
    for _dt, s in days:
        pc = _precompute_day(s, epochs_s, date_tag=_dt, systems=systems)
        if pc is None:
            continue
        if max_stations and pc[0].shape[1] > max_stations:
            M_norm, _J, _P, nms_c, sts_c = pc
            idx = np.sort(RNG.choice(len(nms_c), max_stations, replace=False))
            M_sub = M_norm[:, idx]
            M_norm2 = (M_sub - M_sub.mean(0)) / (M_sub.std(0) + 1e-9)
            pc = (M_norm2, None, None, [nms_c[k] for k in idx], [sts_c[k] for k in idx])
        precomputed.append((_dt, pc))
    if not precomputed:
        return {"label": label, "error": "no data"}

    rows, gstations, day_cols = _build_pair_table(precomputed, epochs_s)
    print(f"[{label}] {len(precomputed)} days, {len(rows)} pair-days, {len(gstations)} stations",
          flush=True)

    arr0 = _alpha_for_rows(rows, day_cols, "jaccard")
    corr_d_j = float(np.corrcoef(arr0["dist"], arr0["jbar"])[0, 1])

    res = {
        "label": label, "n_days": len(precomputed), "n_pair_days": len(rows),
        "n_stations": len(gstations), "corr_dist_jbar": corr_d_j,
        "spec_battery": _spec_battery(rows, day_cols),
        "leave_one_station_out": _loso(rows, gstations, day_cols),
        "leave_one_day_out": _lodo(rows, day_cols),
        "station_clustered_bootstrap": _station_bootstrap(rows, gstations, day_cols),
    }

    sb = res["spec_battery"]
    n_survive = sum(1 for v in sb.values() if v.get("survives"))
    n_band = sum(1 for v in sb.values() if v.get("lambda_in_TEP_band"))
    res["verdict"] = {
        "n_specs": len(sb),
        "n_specs_survive_null": n_survive,
        "n_specs_lambda_in_TEP_band": n_band,
        "loso_stable": bool(res["leave_one_station_out"].get("loso_frac_in_TEP_band", 0) >= 0.8),
        "lodo_stable": bool(res["leave_one_day_out"].get("lodo_frac_in_TEP_band", 0) >= 0.8),
        "bootstrap_ci_excludes_short_and_long": res["station_clustered_bootstrap"].get("lambda_ci_in_TEP_band", False),
        "partial_corr_ever_significant": bool(any(
            abs(v.get("partial_corr_alpha_logd_given_controls", 0) or 0) > 0.1
            for v in sb.values())),
    }
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "test": "temporal_t4_forensic",
        "description": ("Full control battery for the Step 45 temporal T4: "
                        "spec battery, LOSO, LODO, station-clustered bootstrap. "
                        "Reports the partial correlation (honest d|Jbar "
                        "separator) alongside the binned-distance R^2 for every "
                        "spec."),
        "TEP_band_km": list(TEP_BAND),
        "timestamp": datetime.now().isoformat(),
        "arms": {},
    }

    bdays = [f"2024{d:03d}" for d in range(100, 107)]
    res["arms"]["broadcast_global"] = run_arm("broadcast", bdays, "broadcast_global")

    cod_days = td.available_dates("clk_cod", 2024)
    if len(cod_days) >= 20:
        idx = np.linspace(0, len(cod_days) - 1, 20).round().astype(int)
        res["arms"]["precise_cod"] = run_arm(
            "clk_cod", [cod_days[i] for i in idx], "precise_cod", max_stations=55)

    (OUT / "step_45b_t4_forensic.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved {OUT / 'step_45b_t4_forensic.json'}")
    for name, arm in res["arms"].items():
        if "verdict" not in arm:
            continue
        v = arm["verdict"]
        print(f"\n[{name}] corr(d,Jbar)={arm['corr_dist_jbar']:+.3f}")
        print(f"  specs surviving null: {v['n_specs_survive_null']}/{v['n_specs']}"
              f"   lambda in TEP band: {v['n_specs_lambda_in_TEP_band']}/{v['n_specs']}")
        print(f"  LOSO stable: {v['loso_stable']}   LODO stable: {v['lodo_stable']}"
              f"   bootstrap CI in band: {v['bootstrap_ci_excludes_short_and_long']}")
        print(f"  partial corr ever significant (|pc|>0.1): {v['partial_corr_ever_significant']}")
        b = arm["station_clustered_bootstrap"]
        print(f"  bootstrap lambda {b['lambda_km_median']:.0f} km "
              f"CI {b['lambda_km_ci95']}, R^2 {b['R2_median']:.2f} CI {b['R2_ci95']}")


if __name__ == "__main__":
    main()
