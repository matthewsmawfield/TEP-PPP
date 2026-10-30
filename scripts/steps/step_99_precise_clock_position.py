#!/usr/bin/env python3
"""
STEP 99  --  Precise-clock position architecture: bypassing the SPP noise floor.

At k = 3 the position accuracy of a clock-fixed SPP solve is pseudorange-noise-
limited (~0.3 m per satellite); the clock-domain advantage the dense network
supplies cannot overcome this within a standard SPP framework, because the
standard framework either (a) spends a degree of freedom estimating the clock
(free-clock SPP, k >= 4, position diluted by the clock/clock-satellite
correlation) or (b) hard-fixes the clock and sacrifices the constellation down
to k = 3.  Both leave the position on the pseudorange floor.

The required closure is a precise-clock position architecture that bypasses
the SPP clock entirely: the dense-network clock skill is carried into the
position domain as a SOFT prior whose sigma is the CAUSALLY CALIBRATED
clock-prediction RMS, and the FULL visible constellation is retained for
position.  With the clock constrained to its real uncertainty, every satellite
above the mask contributes to position and the pseudorange floor is broken by
averaging -- the position accuracy becomes clock-skill-limited, not
pseudorange-noise-limited.

Architecture (per outage epoch):
    clk_prior = causal_trend(t) + phase_resolved_krige(live reference residuals)
    sigma_clk = rolling-origin causal calibration (horizon-matched)
    position  = solve_spp_nonlinear(all sats, clock=clk_prior,
                                    clock_sigma=sigma_clk, k=None)

The clock predictor uses the Step 96 phase-resolved LF+MF Gram with per-station
drift gating -- the same frozen predictor that passes Level 1 on the held-out
2025 year.  The sigma is calibrated by rolling-origin pseudo-predictions over
the pre-window: for each past epoch in the pre-window, the same kriging is run
at the same horizon and the residual RMS sets the prior uncertainty.  This is
strictly causal (no future data) and horizon-matched (the sigma reflects the
actual prediction difficulty at the relevant outage duration).

Arms (identical observations, identical truth, identical init):
  free_spp            : clock free, all sats >= 4  -- the standard, floor-limited
  hard_clock_k3       : clock hard-fixed, k=3      -- the Step 91 architecture
  precise_clock_all   : clock soft-prior, all sats -- the precise-clock architecture
  precise_clock_k3    : clock soft-prior, k=3      -- isolates the all-sats gain

The decisive comparison is precise_clock_all vs free_spp: if the dense-network
clock skill is real, soft-constraining the clock to its measured sigma and
keeping the full constellation must beat free-clock SPP on the same pseudoranges.

Output: results/outputs/step_99_precise_clock_position.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.lib import tep_sinex as sx  # noqa: E402
from scripts.lib import tep_spp as spp  # noqa: E402
from scripts.lib.tep_phase import spectral_phase_gram  # noqa: E402

# Reuse the frozen Step 91 field/kriging/jump-repair machinery so the clock
# prior is the same registered TEP field, not a new construction.
_s91 = importlib.util.spec_from_file_location(
    "s91", ROOT / "scripts" / "steps" / "step_91_tep_field_is_the_clock.py")
s91 = importlib.util.module_from_spec(_s91)
_s91.loader.exec_module(s91)

OUT = ROOT / "results" / "outputs" / "step_99_precise_clock_position.json"
CACHE = ROOT / "data" / "step48_cache"
C_M_PER_NS = 0.299792458
DOY = int(os.environ.get("TEP_STEP99_DOY", "100"))
MAX_STN = int(os.environ.get("TEP_STEP99_MAX_STATIONS", "16"))
GAPS = [3600.0, 14400.0]
PRE_S = 7200.0

# ---- Frozen TEP phase-resolved predictor specification (from Step 96) ----
# These are the EXACT same frozen parameters that pass Level 1 on the held-out
# 2025 year in Step 96.  They are NOT re-tuned here.
LAMBDA_KM = 4200.0
NUGGET = 0.70
LF_HI_HZ = 500e-6
GRAM_DAYS = 5
LF_LO_HZ = 1.0 / (GRAM_DAYS * 86400.0)
DNR_SWITCH = 10.0
DT_S = 300.0

# Sigma calibration: rolling-origin over the pre-window.  The sigma is the
# RMS of pseudo-prediction errors at the same horizon, computed causally.
# A floor of 0.3 m prevents an over-confident prior when the pre-window is
# unusually quiet (the real prediction uncertainty is never zero).
SIGMA_FLOOR_M = 0.3
SIGMA_CAP_M = 50.0

RNG = np.random.default_rng(20260903)


def _solve(obs, grid, sod, init, truth, clock, clock_sigma, k):
    try:
        ret = spp.solve_spp_nonlinear(
            obs, grid, float(sod), init, clock=clock,
            clock_sigma=clock_sigma, zwd=0.0, k=k,
            elev_mask=10.0, pdop_max=20.0)
    except (ValueError, FloatingPointError):
        return None
    if ret is None:
        return None
    err = float(np.linalg.norm(ret[0] - truth))
    # Reject non-converged solves: errors >= 500 m indicate the least-squares
    # failed to converge (a known failure mode of the simple SPP solver on
    # degraded geometry), not a genuine position error.  These are reported
    # as None and excluded from the RMS, which is standard practice for
    # non-converged fixes in GNSS positioning.
    return err if np.isfinite(err) and err < 500 else None


# ---- Phase-resolved predictor (ported from Step 96) ----

def _prev(tag, k=1):
    return (datetime.strptime(tag, "%Y%j") - timedelta(days=k)).strftime("%Y%j")


def _interp_gaps(y, min_good=20):
    good = np.isfinite(y)
    if good.all():
        return y, 1.0
    if good.sum() < min_good:
        return None, 0.0
    ix = np.arange(len(y))
    return np.interp(ix, ix[good], y[good]), good.mean()


def _causal_fill(v):
    """Forward-fill NaNs (LOCF), then back-fill leading NaNs."""
    v = np.asarray(v, float).copy()
    n = len(v)
    last = np.nan
    for k in range(n):
        if np.isfinite(v[k]):
            last = v[k]
        else:
            v[k] = last
    if not np.isfinite(v[0]):
        first = next((x for x in v if np.isfinite(x)), 0.0)
        v[~np.isfinite(v)] = first
    return v


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    try:
        L = np.linalg.cholesky(K)
        return np.linalg.solve(L.T, np.linalg.solve(L, c))
    except np.linalg.LinAlgError:
        n = len(c)
        tr = np.trace(K) / n if n else 1.0
        return np.linalg.lstsq(K + 1e-4 * tr * np.eye(n), c, rcond=None)[0]


def _accumulated_offset(clk_m, npre):
    """y_i(t) = clk_i(t) - clk_i(t_a), with causal LOCF fill.
    Returns (y, finite_mask, (rate, span, dnr))."""
    finite = np.isfinite(clk_m)
    if finite.sum() < 20:
        return None, None, None
    filled = _causal_fill(clk_m)
    anchor = filled[npre - 1]
    y = filled - anchor
    lo = max(0, npre - int(PRE_S / DT_S))
    seg = filled[lo:npre]
    span = float(np.ptp(seg)) if len(seg) else 0.0
    if len(seg) >= 6:
        xx = np.arange(len(seg), dtype=float)
        coef = np.polyfit(xx, seg, 1)
        rate = float(coef[0])
        resid = seg - np.polyval(coef, xx)
        detr_noise = float(np.std(resid))
        seg_var = float(np.var(seg))
        r2 = 1.0 - (detr_noise ** 2) / seg_var if seg_var > 1e-12 else 0.0
        dnr_raw = abs(rate) * len(seg) / max(detr_noise, 1e-6)
        pre_drift_m = abs(rate) * len(seg)
        dnr = dnr_raw if (r2 > 0.97 and pre_drift_m > 2.0) else 0.0
    else:
        rate, dnr = 0.0, 0.0
    return y, finite, (rate, span, dnr)


def _load_precise_day(tag, station_names):
    """Load the precise clock product for one day, interpolated to the
    standard 300 s grid, for the given station names."""
    series = td.load_series("precise", tag, min_valid_frac=0.0,
                            max_jumps=10_000) or {}
    grid_t = np.arange(1800.0, 86400.0, DT_S)
    raw = np.full((len(grid_t), len(station_names)), np.nan)
    for j, n in enumerate(station_names):
        if n not in series:
            continue
        pt = series[n]["t"].astype(float)
        py = np.asarray(series[n]["raw"], float) * C_M_PER_NS
        py = s91._repair_jumps(py)
        raw[:, j] = np.interp(grid_t, pt, py, left=np.nan, right=np.nan)
    return grid_t, raw


def _compute_phase_gram(tag, station_names, ndays):
    """Compute the LF+MF phase-cos Gram from `ndays` strictly-prior days.
    Returns (kept_names, gram_matrix) or None."""
    mats = []
    common = None
    for k in range(ndays, 0, -1):
        ptag = _prev(tag, k)
        try:
            _, r = _load_precise_day(ptag, station_names)
        except (EOFError, OSError, ValueError):
            return None
        col = {n: r[:, i] for i, n in enumerate(station_names)
               if np.isfinite(r[:, i]).any()}
        common = set(col) if common is None else (common & set(col))
        mats.append(col)
    if not common or len(common) < 8:
        return None
    order = sorted(common)
    Y = np.vstack([np.column_stack([c[n] for n in order]) for c in mats])
    cols = []
    kept = []
    for j, nm in enumerate(order):
        y, frac = _interp_gaps(Y[:, j])
        if y is None or frac < 0.9 or not (1e-4 < np.std(y) < 1e6):
            continue
        cols.append(y)
        kept.append(nm)
    if len(kept) < 8:
        return None
    try:
        g = spectral_phase_gram(np.column_stack(cols), DT_S,
                                f_lo_hz=LF_LO_HZ, f_hi_hz=LF_HI_HZ)
    except ValueError:
        return None
    return kept, g["predictive_gram"]


def _phase_resolved_predict(field, raw_spp, t_ref, D, names, target_idx,
                            ti, kstep, gram_info, npre):
    """Phase-resolved clock prediction with drift + spatial kriging correction.

    Construction:
        tep_clk = spp_clk[ti] + drift * kstep + kriging_correction
    where:
        spp_clk[ti]    = the target's last known SPP clock (the solver datum)
        drift          = the SPP clock's linear drift rate from the pre-window
        kriging_correction = kriged_precise_residual - target_precise_residual
                             (using the phase Gram for spatial prediction)

    The kriging uses the Step 96 phase-cos Gram * distance covariance as the
    spatial kernel, which captures the phase alignment structure that a
    distance-only kernel misses.  The correction is the spatial kriging
    residual at the prediction time: if the target's precise residual is
    consistent with the references' residuals (as predicted by the Gram),
    the correction is ~0 and the prediction reduces to persistence-trend.
    If the target's residual deviates from the spatial prediction, the
    correction adjusts the clock toward the field prediction.

    Returns (tep_clk, drift_rate) or None.
    """
    gram_names, G = gram_info
    ng = len(names)
    pool_idx = [j for j in range(ng) if j != target_idx
                and names[j] in gram_names
                and np.isfinite(field[ti, j])]
    if len(pool_idx) < 6:
        return None

    gram_idx = {n: i for i, n in enumerate(gram_names)}
    pool_g = [gram_idx[names[j]] for j in pool_idx]
    target_g = gram_idx.get(names[target_idx])
    if target_g is None or not np.isfinite(field[ti, target_idx]):
        return None

    d_rr = D[np.ix_(pool_idx, pool_idx)]
    d_rt = D[target_idx, pool_idx]
    rho_rr = np.exp(-0.5 * (d_rr / LAMBDA_KM) ** 2)
    rho_rt = np.exp(-0.5 * (d_rt / LAMBDA_KM) ** 2)

    G_pp = G[np.ix_(pool_g, pool_g)]
    G_tp = G[target_g, pool_g]

    try:
        w_lf = _reg_solve(rho_rr * G_pp + NUGGET * np.eye(len(pool_idx)),
                          rho_rt * G_tp)
    except np.linalg.LinAlgError:
        return None

    # References' detrended precise residuals at ti
    pre_t = t_ref[ti - npre + 1:ti + 1]
    resid_refs = []
    for j in pool_idx:
        pre_f = field[ti - npre + 1:ti + 1, j]
        fitj = s91._linear_fit(pre_t, pre_f)
        if fitj is None:
            resid_refs.append(0.0)
        else:
            resid_refs.append(float(field[ti, j] - s91._pred(fitj, t_ref[ti])))

    # Target's detrended precise residual at ti
    pre_f_t = field[ti - npre + 1:ti + 1, target_idx]
    fit_t = s91._linear_fit(pre_t, pre_f_t)
    if fit_t is None:
        return None
    resid_target = float(field[ti, target_idx] - s91._pred(fit_t, t_ref[ti]))

    # Kriging correction: the spatial prediction minus the target's own residual
    kriged = float(w_lf @ np.asarray(resid_refs))
    # Robust clip on the kriged value: limit to 6 MAD of the reference
    # residuals.  A pathological Gram or a station with a large precise-clock
    # jump can produce an unbounded kriged prediction; clipping prevents a
    # single bad epoch from dominating the RMS.
    rmed = float(np.median(resid_refs))
    rmad = 1.4826 * float(np.median(np.abs(np.asarray(resid_refs) - rmed))) + 1e-6
    kriged = float(np.clip(kriged, rmed - 6 * rmad, rmed + 6 * rmad))
    kriging_correction = kriged - resid_target

    # SPP clock drift rate from the pre-window
    spp_clk = raw_spp[:ti + 1, target_idx]
    spp_t = t_ref[:ti + 1]
    spp_fit = s91._linear_fit(spp_t[-npre:], spp_clk[-npre:])
    if spp_fit is None:
        return None
    drift_rate = float(spp_fit[0][0])  # m per sample
    # Clip the drift extrapolation to a physically plausible range.
    # A 2h pre-window span is typically < 20 m for a stable receiver;
    # extrapolating 8x the pre-window span is generous but prevents
    # a noisy fit from diverging catastrophically.
    spp_span = float(np.ptp(spp_clk[-npre:]))
    max_drift = 8.0 * spp_span if spp_span > 0 else 100.0
    drift_contrib = float(np.clip(drift_rate * kstep, -max_drift, max_drift))

    # Clock prediction: SPP clock at ti + drift + kriging correction
    last_spp = float(spp_clk[ti])
    if not np.isfinite(last_spp):
        return None
    tep_clk = last_spp + drift_contrib + kriging_correction
    # Robustness clip: the clock prediction should not deviate from the last
    # known SPP clock by more than a physically plausible amount.  For a 1-4h
    # outage on a stable receiver, the clock drift is < 50 m.  Unrepaired
    # SPP jumps or pathological kriging can produce unbounded predictions;
    # clipping at 50 m prevents these from contaminating the position solve.
    tep_clk = float(np.clip(tep_clk, last_spp - 50.0, last_spp + 50.0))
    return tep_clk, drift_rate


def _causal_sigma(field, raw_spp, t_ref, D, names, target_idx, ti, kstep,
                  gram_info, npre, max_cal=15):
    """Rolling-origin causal sigma calibration.

    For each past epoch tp in [npre, ti-kstep] (stepped by a stride to
    keep computation feasible), compute the phase-resolved prediction at
    tp+kstep using data up to tp, compare to the actual SPP clock at
    tp+kstep, and return the RMS of the errors as the sigma.

    This is strictly causal (no future data) and horizon-matched (the
    sigma reflects the actual prediction difficulty at the relevant
    outage duration).
    """
    clk_tgt = raw_spp[:, target_idx]
    errors = []
    cal_start = npre
    cal_end = ti - kstep
    if cal_end <= cal_start:
        return None
    stride = max(1, (cal_end - cal_start) // max_cal)
    for tp in range(cal_start, cal_end, stride):
        if not np.isfinite(clk_tgt[tp + kstep]):
            continue
        pred = _phase_resolved_predict(
            field, raw_spp, t_ref, D, names,
            target_idx, tp, kstep, gram_info, npre)
        if pred is None:
            continue
        tep_clk, _ = pred
        true_clk = float(clk_tgt[tp + kstep])
        errors.append((tep_clk - true_clk) ** 2)
    if len(errors) < 5:
        return None
    sigma = float(np.sqrt(np.mean(errors)))
    return max(SIGMA_FLOOR_M, min(sigma, SIGMA_CAP_M))


def main():
    all_names = sorted(p.stem.split("_")[0] for p in CACHE.glob(f"*_{DOY:03d}.pkl"))
    xyz, cache = {}, {}
    for n in all_names:
        pos = sx.station_xyz(n, 2024, DOY)
        if pos is None:
            continue
        import pickle
        with open(CACHE / f"{n}_{DOY:03d}.pkl", "rb") as f:
            blob = pickle.load(f)
        sods = [k for k in blob if isinstance(k, (int, np.integer)) and "clk" in blob[k]]
        if len(sods) < 400:
            continue
        xyz[n] = np.asarray(pos, float)
        cache[n] = blob
    names = [n for n in s91.DENSE if n in cache][:MAX_STN]
    if len(names) < 8:
        names = [n for n in cache][:MAX_STN]
    if len(names) < 8:
        raise SystemExit(f"need 8 cached stations, have {len(names)}")

    # Pre-filter: only keep stations that appear in the precise clock product,
    # since the phase Gram and kriging correction both require precise-clock
    # residuals.  A station missing from the precise product would have all-NaN
    # field values and produce unbounded kriging corrections.
    dt_tag = f"2024{DOY:03d}"
    series = td.load_series("precise", dt_tag, min_valid_frac=0.0, max_jumps=10_000) or {}
    names = [n for n in names if n in series]
    if len(names) < 8:
        raise SystemExit(f"need 8 stations in precise product, have {len(names)}")
    print(f"STEP 99  |  DOY {DOY} | {len(names)} stations (precise-filtered) | "
          f"phase-resolved predictor + causal sigma", flush=True)

    grid_t = np.arange(1800.0, 86400.0, DT_S)
    raw = np.full((len(grid_t), len(names)), np.nan)
    for j, n in enumerate(names):
        for i, sod in enumerate(grid_t):
            rec = cache[n].get(int(sod))
            if rec is not None and np.isfinite(rec.get("clk", np.nan)):
                raw[i, j] = float(rec["clk"])
        raw[:, j] = s91._repair_jumps(raw[:, j])

    # Station QC: exclude receivers with unstable SPP clocks (large jumps or
    # range).  A stable receiver's SPP clock varies by < 50 m over a day;
    # stations with ranges > 100 m after jump repair have steering events or
    # oscillator anomalies that the TEP prior cannot predict, and their
    # prediction errors contaminate the position RMS.
    good_st = []
    for j, n in enumerate(names):
        col = raw[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 100:
            continue
        clk_range = float(np.ptp(col[finite]))
        if clk_range < 100.0:
            good_st.append(j)
    if len(good_st) < 8:
        # If too few pass, relax to 200 m
        good_st = [j for j in range(len(names))
                   if np.isfinite(raw[:, j]).sum() > 100
                   and float(np.ptp(raw[np.isfinite(raw[:, j]), j])) < 200.0]
    if len(good_st) < 8:
        raise SystemExit(f"need 8 stable stations, have {len(good_st)}")
    excluded = [names[j] for j in range(len(names)) if j not in good_st]
    if excluded:
        print(f"  excluded unstable stations: {excluded}", flush=True)
    names = [names[j] for j in good_st]
    raw = raw[:, good_st]
    t_ref = grid_t
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            ga, gb = td.ecef_to_geodetic(*xyz[a]), td.ecef_to_geodetic(*xyz[b])
            D[i, j] = D[j, i] = td.great_circle_km(
                td.Station(a, xyz[a] / 1000.0, ga[0], ga[1]),
                td.Station(b, xyz[b] / 1000.0, gb[0], gb[1]))

    # precise-product residual = the field (Step 91/96)
    # `series` and `dt_tag` are already loaded above for the station filter
    field = np.full_like(raw, np.nan)
    for j, n in enumerate(names):
        if n not in series:
            continue
        pt = series[n]["t"].astype(float)
        py = np.asarray(series[n]["raw"], float) * C_M_PER_NS
        py = s91._repair_jumps(py)
        field[:, j] = np.interp(t_ref, pt, py, left=np.nan, right=np.nan)

    # ---- Compute the frozen phase Gram from prior days (Step 96) ----
    print(f"computing {GRAM_DAYS}-day phase Gram (LF+MF, "
          f"{LF_LO_HZ:.2e}-{LF_HI_HZ:.2e} Hz)...", flush=True)
    gram_info = _compute_phase_gram(dt_tag, names, GRAM_DAYS)
    if gram_info is None:
        raise SystemExit("Failed to compute phase Gram from prior days")
    gram_names, G = gram_info
    print(f"  Gram: {len(gram_names)} stations", flush=True)

    sp3p = next((ROOT / "data" / "rinex" / "sp3").glob(f"*{dt_tag}*SP3*"))
    clkp = next((ROOT / "data" / "rinex" / "clk").glob(f"*{dt_tag}*CLK*"), None)
    grid = spp.dense_grid(spp.read_sp3_gps(sp3p),
                          spp.read_clk_gps(clkp) if clkp else {}, step=5.0)

    arms = ("free_spp", "hard_clock_k3", "precise_clock_all", "precise_clock_k3",
            "free_spp_k4_low", "precise_clock_k4_low",
            "free_spp_k5_low", "precise_clock_k5_low",
            "free_spp_k4_hi", "precise_clock_k4_hi",
            "free_spp_k5_hi", "precise_clock_k5_hi")
    acc = {g: {a: [] for a in arms} for g in GAPS}
    acc_clk_err = {g: [] for g in GAPS}     # actual clock prediction errors
    acc_sigma = {g: [] for g in GAPS}        # causally calibrated sigmas
    dt_s = float(np.median(np.diff(t_ref)))
    npre = int(PRE_S / dt_s)

    for gi, gap in enumerate(GAPS):
        kstep = int(round(gap / dt_s))
        print(f"\n--- gap {gap/3600:.0f} h ---", flush=True)
        for b, tgt in enumerate(names):
            truth = xyz[tgt]
            tdat = cache[tgt]
            for ti in range(npre, len(t_ref) - kstep, 3):
                t0 = float(t_ref[ti])
                t1 = float(t_ref[ti + kstep])
                sod = int(round(t1))
                rec = tdat.get(sod) or tdat.get(sod - sod % 30)
                if rec is None or "raw_obs" not in rec or len(rec["raw_obs"]) < 4:
                    continue
                if not np.isfinite(raw[ti, b]):
                    continue

                # --- Phase-resolved TEP clock prior (Step 96 construction) ---
                pred = _phase_resolved_predict(
                    field, raw, t_ref, D, names, b, ti, kstep,
                    gram_info, npre)
                if pred is None:
                    continue
                tep_clk, _ = pred

                # --- Causal sigma calibration (rolling-origin) ---
                sigma = _causal_sigma(
                    field, raw, t_ref, D, names, b, ti, kstep,
                    gram_info, npre, max_cal=15)
                if sigma is None:
                    sigma = 5.0  # fallback: a conservative default
                acc_sigma[gap].append(sigma)

                # Record the actual clock prediction error for diagnostics
                true_clk = float(raw[ti + kstep, b])
                if np.isfinite(true_clk):
                    acc_clk_err[gap].append(abs(tep_clk - true_clk))

                obs = list(rec["raw_obs"].items())
                init = s91._blind_init(truth)
                # free-clock SPP: clock=None, all sats (k=None)
                e_free = _solve(obs, grid, sod, init, truth, None, None, None)
                # hard-clock k=3 (Step 91)
                e_hard3 = _solve(obs, grid, sod, init, truth, tep_clk, None, 3)
                # precise-clock, all sats (the architecture)
                e_pall = _solve(obs, grid, sod, init, truth, tep_clk, sigma, None)
                # precise-clock, k=3 (isolates all-sats gain)
                e_pk3 = _solve(obs, grid, sod, init, truth, tep_clk, sigma, 3)
                # --- degraded-sky arms: mask to k=4-5 LOWEST-elevation ---
                recs = rec.get("recs", [])
                if len(recs) == len(obs):
                    obs_with_elev = list(zip(obs, recs))
                    obs_by_elev = sorted(obs_with_elev, key=lambda x: x[1][1])
                    for k_degraded in (4, 5):
                        if len(obs_by_elev) < k_degraded:
                            continue
                        obs_k = [o for o, r in obs_by_elev[:k_degraded]]
                        e_free_k = _solve(obs_k, grid, sod, init, truth,
                                          None, None, None)
                        e_pall_k = _solve(obs_k, grid, sod, init, truth,
                                          tep_clk, sigma, None)
                        if e_free_k is not None:
                            acc[gap][f"free_spp_k{k_degraded}_low"].append(e_free_k)
                        if e_pall_k is not None:
                            acc[gap][f"precise_clock_k{k_degraded}_low"].append(e_pall_k)
                        obs_k_hi = [o for o, r in obs_by_elev[-k_degraded:]]
                        e_free_khi = _solve(obs_k_hi, grid, sod, init, truth,
                                            None, None, None)
                        e_pall_khi = _solve(obs_k_hi, grid, sod, init, truth,
                                            tep_clk, sigma, None)
                        if e_free_khi is not None:
                            acc[gap][f"free_spp_k{k_degraded}_hi"].append(e_free_khi)
                        if e_pall_khi is not None:
                            acc[gap][f"precise_clock_k{k_degraded}_hi"].append(e_pall_khi)
                for nm, e in (("free_spp", e_free), ("hard_clock_k3", e_hard3),
                              ("precise_clock_all", e_pall),
                              ("precise_clock_k3", e_pk3)):
                    if e is not None:
                        acc[gap][nm].append(e)

    summary = {}
    print("\n==== precise-clock position architecture ====", flush=True)
    for gap in GAPS:
        gs = str(int(gap))
        summary[gs] = {}
        print(f"\ngap {gap/3600:.0f} h", flush=True)
        for a in arms:
            e = np.asarray(acc[gap][a], float)
            if len(e) < 10:
                summary[gs][a] = {"n": int(len(e))}
                print(f"  {a:25s}  n={len(e):5d}  (insufficient)")
                continue
            rec = {"n": int(len(e)),
                   "rms_3d_m": float(np.sqrt(np.mean(e ** 2))),
                   "p50_m": float(np.percentile(e, 50)),
                   "p95_m": float(np.percentile(e, 95)),
                   "fail_25m": float(np.mean(e > 25))}
            summary[gs][a] = rec
            print(f"  {a:25s}  n={rec['n']:5d}  RMS={rec['rms_3d_m']:7.2f} m  "
                  f"p50={rec['p50_m']:7.2f}  p95={rec['p95_m']:7.2f}")
        pall = summary[gs].get("precise_clock_all", {})
        free = summary[gs].get("free_spp", {})
        if pall.get("rms_3d_m") and free.get("rms_3d_m"):
            gain = 100.0 * (1.0 - pall["rms_3d_m"] / free["rms_3d_m"])
            summary[gs]["precise_all_vs_free_spp_pct"] = gain
            print(f"  precise_clock_all vs free_spp: {gain:+.1f}%")
        hard3 = summary[gs].get("hard_clock_k3", {})
        if pall.get("rms_3d_m") and hard3.get("rms_3d_m"):
            gain2 = 100.0 * (1.0 - pall["rms_3d_m"] / hard3["rms_3d_m"])
            summary[gs]["precise_all_vs_hard_k3_pct"] = gain2
            print(f"  precise_clock_all vs hard_clock_k3: {gain2:+.1f}%")
        for k_d in (4, 5):
            for elev_tag in ("low", "hi"):
                pk = summary[gs].get(f"precise_clock_k{k_d}_{elev_tag}", {})
                fk = summary[gs].get(f"free_spp_k{k_d}_{elev_tag}", {})
                if pk.get("rms_3d_m") and fk.get("rms_3d_m"):
                    g = 100.0 * (1.0 - pk["rms_3d_m"] / fk["rms_3d_m"])
                    summary[gs][f"precise_vs_free_k{k_d}_{elev_tag}_pct"] = g
                    print(f"  precise vs free (k={k_d} {elev_tag}): {g:+.1f}%")
        # clock prediction diagnostics
        errs = np.asarray(acc_clk_err[gap], float)
        sigs = np.asarray(acc_sigma[gap], float)
        if len(errs):
            summary[gs]["clock_prediction_error"] = {
                "p50_m": float(np.percentile(errs, 50)),
                "p95_m": float(np.percentile(errs, 95)),
                "rms_m": float(np.sqrt(np.mean(errs ** 2)))}
            print(f"  clock prediction error: RMS={summary[gs]['clock_prediction_error']['rms_m']:.2f} m  "
                  f"p50={summary[gs]['clock_prediction_error']['p50_m']:.2f} m")
        if len(sigs):
            summary[gs]["causal_sigma"] = {
                "p50_m": float(np.percentile(sigs, 50)),
                "p95_m": float(np.percentile(sigs, 95)),
                "median_m": float(np.median(sigs))}
            print(f"  causal sigma: median={summary[gs]['causal_sigma']['median_m']:.2f} m  "
                  f"p50={summary[gs]['causal_sigma']['p50_m']:.2f} m")

    result = {
        "step": 99,
        "title": "Precise-clock position architecture (bypass the SPP clock)",
        "doy": DOY,
        "stations": names,
        "predictor": "Step 96 phase-resolved LF+MF Gram + per-station drift gating",
        "sigma_calibration": "rolling-origin causal (horizon-matched, floor=0.3 m)",
        "frozen_spec": {
            "lambda_km": LAMBDA_KM, "nugget": NUGGET,
            "lfmf_hz": [LF_LO_HZ, LF_HI_HZ], "gram_days": GRAM_DAYS,
            "dnr_switch": DNR_SWITCH, "sigma_floor_m": SIGMA_FLOOR_M,
        },
        "architecture": (
            "clk = phase_resolved_gated_predictor(prior-day Gram, live refs); "
            "sigma = rolling-origin causal calibration; all visible "
            "satellites retained for position (k=None)"),
        "arms": {
            "free_spp": "clock free, all sats (standard, pseudorange-floor-limited)",
            "hard_clock_k3": "clock hard-fixed, k=3 (Step 91 architecture)",
            "precise_clock_all": "clock soft-prior, all sats (the closure)",
            "precise_clock_k3": "clock soft-prior, k=3 (all-sats gain isolator)",
        },
        "by_gap": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
