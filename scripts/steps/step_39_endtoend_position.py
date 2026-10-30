#!/usr/bin/env python3
"""
STEP 39 -- Linearised degraded-geometry positioning replay: P2 diagnostic.

Answers: "on genuinely unseen station-days, under real satellite obstruction,
does the TEP network clock prior produce better POSITIONS than the best
conventional clock model, against a SURVEYED-TRUTH coordinate?"

Differences from Step 33 (which this supersedes for the headline P2 result):
  * TRUTH = the official IGS daily-SINEX coordinate for that station and day
    (scripts/lib/tep_sinex.py), not the day's own SPP mean.  The SPP mean is
    empirically 10-17 m from the SINEX value for broadcast SPP -- an error
    comparable to the signal, and circular.
  * A full CONVENTIONAL CLOCK BATTERY, not just persistence + linear:
    ordinary (no prior), persistence, linear drift, quadratic drift,
    first-order Gauss-Markov (AR1), and the TEP distance kernel.  Every
    conventional model's prior sigma is its OWN measured pre-window
    prediction error (fair weighting).
  * 4/5/6/8 visible satellites are all first-class (not 3-satellite
    rank-deficiency -- "you merely made an impossible system solvable" has
    no purchase at k >= 4).
  * RELIABILITY metrics, not just the median: 3D / horizontal / vertical RMS,
    the 50/68/95/99th percentiles and the maximum, and the >10 m / >25 m /
    >50 m failure rates and solution availability.

Method per epoch inside an obstruction window that started at the last
full-sky epoch: keep k satellites, build A_i = [-e_i, 1], b_i = resp_i (the
full-sky post-fit pseudorange residual), solve the weighted normal equations
for [dx dy dz d(cdt)] with the method's clock pseudo-observation
[0,0,0,1] = (cdt_prior - cdt_ref), weight 1/sigma_method^2.  Position error =
|| (pos_rtk - truth_sinex) + dpos ||.

Important evidence boundary: this is a linearised replay around RTKLIB's
full-sky solution, not a raw-observation obstruction run.  The full-sky state
and post-fit residuals have already used the removed satellites.  This step is
therefore a mechanism/diagnostic check and MUST NOT by itself pass strict P2.
Strict P2 requires removing satellites from RINEX before an independent solver
run (or an equivalent raw-measurement estimator) and comparing with external
truth.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_sinex as sx      # noqa: E402
from scripts.steps.step_33_position_demo import (  # noqa: E402
    run_rtklib, parse_stat, los_unit, kernel_clock_prediction,
    _find_train_day,
)

OUT = PROJECT_ROOT / "results" / "outputs"
CACHE = PROJECT_ROOT / "data" / "step39_cache"
RNG = np.random.default_rng(20260939)

DAYS = [f"2024{d:03d}" for d in range(100, 107)]
GAP_LENGTHS_S = [600.0, 1800.0, 3600.0, 7200.0, 14400.0]
K_VALUES = [4, 5, 6, 8]
GAP_STRIDE_S = 3 * 3600.0
ELEV_MIN_DEG = 10.0
MAX_STATIONS = 60
TEMPORAL_METHODS = ("persistence", "linear", "quadratic", "ar1")
# When disk is tight, reuse the Step 33 RTKLIB cache and skip station-days
# without one (set via env TEP_STEP39_CACHE_ONLY=1).
import os as _os
CACHE_ONLY = _os.environ.get("TEP_STEP39_CACHE_ONLY", "0") == "1"


def _enu_rot(truth_xyz):
    x, y, z = truth_xyz
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y))
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def _solve(A, b, w, cdt_prior=None, sig=None):
    A = np.asarray(A, float); b = np.asarray(b, float); w = np.asarray(w, float)
    if cdt_prior is not None and sig and sig > 0:
        A = np.vstack([A, [0.0, 0.0, 0.0, 1.0]])
        b = np.append(b, cdt_prior)
        w = np.append(w, 1.0 / sig ** 2)
    # Solve the whitened design directly. Forming normal equations squares the
    # condition number and needlessly amplifies weak-geometry numerical error.
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w
    sv = np.linalg.svd(Aw, compute_uv=False)
    if sv[-1] < 1e-4 * sv[0]:
        return None
    try:
        x, _, rank, _ = np.linalg.lstsq(Aw, bw, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    if rank < 4:
        return None
    if np.linalg.norm(x[:3]) > 200_000.0:
        return None
    return x[:3]


def _ar1_predict(t_pre, c_pre, t_query, anchor):
    """First-order Gauss-Markov extrapolation of the clock from the pre-window
    samples.  Fit c[k] - mean ~ phi (c[k-1] - mean); predict forward."""
    if len(c_pre) < 5:
        return anchor
    dt = np.median(np.diff(t_pre))
    x = c_pre - np.mean(c_pre)
    num = np.sum(x[1:] * x[:-1]); den = np.sum(x[:-1] ** 2) + 1e-12
    phi = np.clip(num / den, -0.999, 0.999)
    steps = max(1, int(round((t_query - t_pre[-1]) / max(dt, 1.0))))
    return float(np.mean(c_pre) + (c_pre[-1] - np.mean(c_pre)) * phi ** steps)


def _poly_predict(t_pre, c_pre, t_query, deg):
    if len(c_pre) < deg + 2:
        return c_pre[-1]
    t0 = t_pre[-1]
    coef = np.polyfit(t_pre - t0, c_pre, deg)
    return float(np.polyval(coef, t_query - t0))


def _clock_predictions(t_hist, c_hist, query_times, kt, kres):
    """Forecast clock values at query_times from history ending beforehand."""
    anchor = float(c_hist[-1])
    rate = float((c_hist[-1] - c_hist[0]) /
                 max(t_hist[-1] - t_hist[0], 1.0))
    kres_anchor = float(np.interp(t_hist[-1], kt, kres))
    out = []
    for s in query_times:
        dt = float(s) - t_hist[-1]
        out.append({
            "persistence": anchor,
            "linear": anchor + rate * dt,
            "quadratic": _poly_predict(t_hist, c_hist, float(s), 2),
            "ar1": _ar1_predict(t_hist, c_hist, float(s), anchor),
            "kernel": anchor + (float(np.interp(s, kt, kres)) - kres_anchor),
        })
    return out


def _causal_prior_sigmas(sods, cdt_full, kt, kres, gap_start, gap_length):
    """Calibrate prior sigmas on a same-horizon pseudo-gap before evaluation.

    The calibration target ends before the 30-minute history used by the real
    evaluation gap, so no sample at or after ``gap_start`` can affect a weight.
    Returns (sigmas, calibration_end) or (None, None).
    """
    history_s = 1800.0
    cal_end = gap_start - history_s
    cal_start = cal_end - gap_length
    cal_hist = sods[(sods >= cal_start - history_s) & (sods < cal_start)]
    cal_query = sods[(sods >= cal_start) & (sods < cal_end)]
    if len(cal_hist) < 6 or len(cal_query) < 10:
        return None, None
    c_hist = np.interp(cal_hist, sods, cdt_full)
    truth = np.interp(cal_query, sods, cdt_full)
    preds = _clock_predictions(cal_hist, c_hist, cal_query, kt, kres)
    methods = ("persistence", "linear", "quadratic", "ar1", "kernel")
    sigmas = {}
    for method in methods:
        errors = np.abs(np.array([p[method] for p in preds], float) - truth)
        sigmas[method] = min(max(float(np.median(errors)), 0.05), 30.0)
    return sigmas, float(cal_end)


_S33_CACHE = PROJECT_ROOT / "data" / "step33_cache"
_S33_CACHE_V2 = PROJECT_ROOT / "data" / "step33_cache_v2"


def _station_run(day, code4, code9, tmp):
    import pickle
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{code4}_{day}.pkl"
    truth = sx.station_xyz(code4, 2024, int(day[4:]))
    if truth is None:
        return None
    for src in (cf, _S33_CACHE_V2 / f"{code4}_{day}.pkl",
                _S33_CACHE / f"{code4}_{day}.pkl"):
        if src.exists():
            try:
                c = pickle.load(open(src, "rb"))
                if "kp" in c and "ep" in c and c["kp"] and c["ep"]:
                    return {"kp": c["kp"], "ep": c["ep"], "truth": truth}
            except Exception:
                pass
    if CACHE_ONLY:
        return None
    kp = kernel_clock_prediction(day, code4, _find_train_day(day, DAYS))
    if kp is None:
        return None
    stat = run_rtklib(code9, int(day[4:]), tmp)
    if stat is None:
        return None
    ep = parse_stat(stat)
    if len(ep) < 200:
        return None
    out = {"kp": kp, "ep": ep}
    try:
        pickle.dump(out, open(cf, "wb"))
    except Exception:
        pass
    return {**out, "truth": truth}


def _eval_station(day, code4, code9, tmp, acc):
    r = _station_run(day, code4, code9, tmp)
    if r is None:
        return 0
    ep, kp, truth = r["ep"], r["kp"], r["truth"]
    Rmat = _enu_rot(truth)
    kt, kres = kp["t"], kp["kres_m"]
    sods = np.array(sorted(ep.keys()))
    cdt_full = np.array([ep[s]["cdt_rtk_m"] for s in sods])

    methods = ("ordinary",) + TEMPORAL_METHODS + ("adaptive_conventional", "kernel")
    n_used = 0
    for L in GAP_LENGTHS_S:
        for k in K_VALUES:
            key = (L, k)
            for g0 in np.arange(sods[0] + 3600.0, sods[-1] - L, GAP_STRIDE_S):
                pre = sods[(sods >= g0 - 1800.0) & (sods < g0)]
                if len(pre) < 6:
                    continue
                cpre = np.interp(pre, sods, cdt_full)

                win = sods[(sods >= g0) & (sods < g0 + L)]
                # Calibrate every weight on a same-horizon pseudo-gap that is
                # entirely before this evaluation gap.  The previous version
                # used abs(prediction - truth) from `win` itself, leaking the
                # test window into the normal-equation weights.
                msig, calibration_end = _causal_prior_sigmas(
                    sods, cdt_full, kt, kres, float(g0), float(L))
                if msig is None or not calibration_end < g0:
                    continue
                selected_conventional = min(TEMPORAL_METHODS, key=lambda m: msig[m])
                ep_rows = []
                all_preds = _clock_predictions(pre, cpre, win, kt, kres)
                for s, preds in zip(win, all_preds):
                    rec = ep[int(s)]
                    cdt_ref = rec["cdt_rtk_m"]
                    preds["adaptive_conventional"] = preds[selected_conventional]
                    ep_rows.append((rec, cdt_ref, preds))
                if len(ep_rows) < 10:
                    continue
                msig["adaptive_conventional"] = msig[selected_conventional]

                for rec, cdt_ref, preds in ep_rows:
                    alls = [q for q in rec["sats"] if q[1] >= ELEV_MIN_DEG]
                    if len(alls) < k:
                        continue
                    keep = sorted(alls, key=lambda q: -q[1])[:k]
                    e = np.array([los_unit(a, el) for a, el, _ in keep])
                    b = np.array([rr for _, _, rr in keep])
                    el = np.array([q[1] for q in keep])
                    A = np.column_stack([-e, np.ones(k)])
                    w = np.sin(np.radians(el)) ** 2
                    offs = rec["pos_rtk"] - truth
                    for m in methods:
                        if m == "ordinary":
                            dpos = _solve(A, b, w)
                        else:
                            dpos = _solve(A, b, w, preds[m] - cdt_ref, msig[m])
                        if dpos is None:
                            acc.setdefault(key, {}).setdefault(m, {"err": [], "enu": [], "fail": 0, "n": 0})
                            acc[key][m]["n"] += 1
                            acc[key][m]["fail"] += 1
                            continue
                        errv = offs + dpos
                        enu = Rmat @ errv
                        d = acc.setdefault(key, {}).setdefault(m, {"err": [], "enu": [], "fail": 0, "n": 0})
                        d["err"].append(float(np.linalg.norm(errv)))
                        d["enu"].append([float(enu[0]), float(enu[1]), float(enu[2])])
                        d["n"] += 1
                    n_used += 1
    return n_used


def _summ(d):
    err = np.array(d["err"], float)
    enu = np.array(d["enu"], float) if d["enu"] else np.zeros((0, 3))
    if len(err) < 20:
        return None
    h = np.hypot(enu[:, 0], enu[:, 1]) if len(enu) else np.array([np.nan])
    v = np.abs(enu[:, 2]) if len(enu) else np.array([np.nan])
    return {
        "n_epochs": int(d["n"]),
        "availability": float(len(err) / max(d["n"], 1)),
        "rms_3d_m": float(np.sqrt(np.mean(err ** 2))),
        "rms_h_m": float(np.sqrt(np.mean(h ** 2))),
        "rms_v_m": float(np.sqrt(np.mean(v ** 2))),
        "p50_m": float(np.percentile(err, 50)),
        "p68_m": float(np.percentile(err, 68)),
        "p95_m": float(np.percentile(err, 95)),
        "p99_m": float(np.percentile(err, 99)),
        "max_m": float(np.max(err)),
        "fail_10m": float(np.mean(err > 10.0)),
        "fail_25m": float(np.mean(err > 25.0)),
        "fail_50m": float(np.mean(err > 50.0)),
    }


def main():
    import tempfile
    OUT.mkdir(parents=True, exist_ok=True)
    acc = {}
    n_st = 0
    with tempfile.TemporaryDirectory() as td_:
        tmp = Path(td_)
        for day in DAYS:
            s = td.load_series("broadcast", day)
            if not s:
                continue
            code9_map = {}
            # resolve 9-char names from the obs dir
            import glob
            for f in glob.glob(str(PROJECT_ROOT / "data" / "rinex" / "obs" / "2024"
                                   / f"{int(day[4:]):03d}" / "*_R_*_MO.crx.gz")):
                c9 = Path(f).name.split("_")[0]
                code9_map[c9[:4]] = c9
            cands = [c for c in s if c in code9_map][:MAX_STATIONS]
            for code4 in cands:
                nu = _eval_station(day, code4, code9_map[code4], tmp, acc)
                if nu:
                    n_st += 1
            print(f"  {day}: {n_st} station-days processed")

    result = {"test": "linearized_postfit_position_replay_p2_diagnostic",
              "evidence_level": "diagnostic_only",
              "strict_p2_pass": False,
              "strict_p2_blocker": ("satellite obstruction is applied after a full-sky RTKLIB "
                                    "solution; raw observations must be thinned before solving"),
              "uses_full_sky_linearization": True,
              "prior_sigma_calibration": ("same-horizon rolling-origin pseudo-gap ending before "
                                          "the evaluation history; no evaluation-window truth"),
              "truth_source": "IGS daily SINEX",
              "n_station_days": n_st, "days": DAYS,
              "k_values": K_VALUES, "gap_lengths_s": GAP_LENGTHS_S,
              "by_gap_k": {}, "timestamp": datetime.now().isoformat()}

    for (L, k), md in sorted(acc.items()):
        summ = {m: _summ(d) for m, d in md.items()}
        summ = {m: v for m, v in summ.items() if v}
        if "kernel" not in summ:
            continue
        conv = [m for m in TEMPORAL_METHODS if m in summ]
        oracle = min(conv, key=lambda m: summ[m]["rms_3d_m"]) if conv else None
        deployable = "adaptive_conventional" if "adaptive_conventional" in summ else None
        row = {"summary": summ, "best_conventional_oracle": oracle,
               "deployable_conventional": deployable}
        if deployable:
            kr, br = summ["kernel"]["rms_3d_m"], summ[deployable]["rms_3d_m"]
            row["kernel_vs_deployable_conv_3d_pct"] = float(100 * (1 - kr / br)) if br > 0 else None
            row["kernel_vs_deployable_conv_fail25_ratio"] = (
                float(summ["kernel"]["fail_25m"] / summ[deployable]["fail_25m"])
                if summ[deployable]["fail_25m"] > 0 else None)
        if oracle:
            kr, br = summ["kernel"]["rms_3d_m"], summ[oracle]["rms_3d_m"]
            row["kernel_vs_oracle_conv_3d_pct"] = float(100 * (1 - kr / br)) if br > 0 else None
        result["by_gap_k"][f"L{int(L)}_k{k}"] = row
        if deployable:
            print(f"  L={int(L/60)}min k={k}: kernel 3D {summ['kernel']['rms_3d_m']:.2f} m "
                  f"vs causal selector {summ[deployable]['rms_3d_m']:.2f} m "
                  f"({row['kernel_vs_deployable_conv_3d_pct']:+.1f}%)  "
                  f">25m fail {summ['kernel']['fail_25m']*100:.1f}% vs "
                  f"{summ[deployable]['fail_25m']*100:.1f}%")

    # Reliability is operationally distinct from RMS. Preserve the promising
    # tail result as a diagnostic while the raw-observation P2 gate stays false.
    fail25_reductions = []
    four_hour_reductions = []
    for key, row in result["by_gap_k"].items():
        ratio = row.get("kernel_vs_deployable_conv_fail25_ratio")
        if ratio is None:
            continue
        reduction = float(100.0 * (1.0 - ratio))
        fail25_reductions.append(reduction)
        if key.startswith("L14400_"):
            four_hour_reductions.append(reduction)
    if fail25_reductions:
        result["integrity_tail_diagnostic"] = {
            "metric": "relative reduction in epochs with position error >25 m",
            "all_gap_satellite_cells_improve": bool(all(x > 0 for x in fail25_reductions)),
            "n_cells": len(fail25_reductions),
            "reduction_range_pct": [min(fail25_reductions), max(fail25_reductions)],
            "four_hour_reduction_range_pct": (
                [min(four_hour_reductions), max(four_hour_reductions)]
                if four_hour_reductions else None),
            "evidence_level": "postfit diagnostic only",
        }

    (OUT / "step_39_endtoend_position.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved {OUT / 'step_39_endtoend_position.json'}")


if __name__ == "__main__":
    main()
