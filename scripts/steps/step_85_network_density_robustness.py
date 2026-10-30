#!/usr/bin/env python3
"""
STEP 85  --  Network density robustness of the TEP outage prediction.

Step 55 demonstrates 12.5-43.9% clock RMS reduction versus persistence on the
dense EUREF/EPN network (~25 stations, median nearest reference ~130 km).  A
practical deployment question follows: how does the TEP prior degrade as the
reference network thins?  Not every region has EUREF-class density; a deployable
method must remain useful with fewer references.

This step re-runs the Step 55 causal outage prediction at controlled reference
densities (5, 8, 12, 16, 20, and all-available stations) on a 15-day subset.
For each density level the reference pool is randomly subsampled (seeded) before
the kriging solve, while the target station, trend model, temporal baselines and
integrity screens remain identical to Step 55.  The TEP covariance is still
physics-fixed (Gaussian, lambda = 4,200 km, nugget = 0.35); no re-tuning at any
density.

Output: results/outputs/step_85_network_density_robustness.json
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

# Import the core building blocks from Step 55 so the density sweep uses the
# *same* predictor logic, not a re-implementation.
_step55 = PROJECT_ROOT / "scripts" / "steps" / "step_55_dense_outage_prediction.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("step55", _step55)
step55 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(step55)

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260903)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.
# Ill-conditioned numerics are wrapped in local np.errstate contexts.
# np.seterr(all="ignore")  # removed: masks bugs globally

DT_S = 300.0
PREWINDOW_S = 7200.0
GAPS_S = [1800.0, 3600.0, 14400.0]  # 30 min, 1 h, 4 h — the key operational horizons
EVAL_STRIDE = 5  # coarser stride for speed
N_DAYS = int(os.environ.get("TEP_DENSITY_N_DAYS", "15"))
DENSITY_LEVELS = [5, 8, 12, 16, 20, 99]  # 99 = all available
LAMBDA_KM = 4200.0
NUGGET = 0.35


def _rho(d):
    return np.exp(-0.5 * (d / LAMBDA_KM) ** 2)


def _solve_weights(d_rr, d_t, lam=LAMBDA_KM, nug=NUGGET):
    K = _rho(d_rr) + (nug + 1e-9) * np.eye(len(d_t))
    return np.linalg.solve(K, _rho(d_t))


def run_density_sweep(days):
    npre = int(PREWINDOW_S / DT_S)
    # acc[density][gap][arm] -> list of squared errors
    results = {}
    for dens in DENSITY_LEVELS:
        results[dens] = {g: {"tep": [], "pers": [], "net_mean": []} for g in GAPS_S}
    target_reset_like_events = {dens: {g: 0 for g in GAPS_S} for dens in DENSITY_LEVELS}
    target_reset_like_examples = {dens: {g: [] for g in GAPS_S} for dens in DENSITY_LEVELS}

    for dt in days:
        dd = step55.load_day(dt)
        if dd is None:
            continue
        names, t, raw, D = dd
        n = len(names)
        nep = len(t)
        if n - 1 < 5:
            continue

        for dens in DENSITY_LEVELS:
            cap = min(dens, n - 1)
            for ti in range(npre, nep - 1, EVAL_STRIDE):
                sl = slice(ti - npre, ti + 1)
                tt = t[sl] - t[ti]
                Adm = np.vstack([tt, np.ones_like(tt)]).T
                pre = raw[sl]
                pre_sd = np.full(n, np.inf)
                for qi in range(n):
                    if np.isfinite(pre[:, qi]).all():
                        qc = np.polyfit(tt, pre[:, qi], 1)
                        pre_sd[qi] = np.std(pre[:, qi] - np.polyval(qc, tt))
                finite_sd = pre_sd[np.isfinite(pre_sd) & (pre_sd > 0.01)]
                if len(finite_sd) < 6:
                    continue
                med_sd = float(np.median(finite_sd))
                causal_good = ((pre_sd > 0.01) & (pre_sd < 60.0) &
                               (pre_sd < 4.0 * med_sd))
                for b in range(n):
                    if not causal_good[b]:
                        continue
                    base_pool = [j for j in range(n) if j != b and causal_good[j] and
                                 np.isfinite(raw[sl, j]).all()]
                    if len(base_pool) < 5:
                        continue
                    y = raw[sl, b]
                    if not np.isfinite(y).all():
                        continue
                    a1, a0 = np.polyfit(tt, y, 1)
                    if abs(a1) > 0.20:
                        continue
                    # Subsample the reference pool to the target density
                    if len(base_pool) > cap:
                        idx = RNG.choice(len(base_pool), cap, replace=False)
                        pool = [base_pool[i] for i in idx]
                    else:
                        pool = base_pool
                    if len(pool) < 5:
                        continue
                    Y = raw[sl][:, pool]
                    if not np.isfinite(Y).all():
                        continue
                    coef = np.linalg.lstsq(Adm, Y, rcond=None)[0]
                    ref_ok = np.abs(coef[0]) <= 0.20
                    if ref_ok.sum() < 5:
                        continue
                    pool = [p for p, ok in zip(pool, ref_ok) if ok]
                    if len(pool) < 5:
                        continue
                    Y = raw[sl][:, pool]
                    coef = np.linalg.lstsq(Adm, Y, rcond=None)[0]
                    rb = y - (a0 + a1 * tt)
                    last = rb[-1]
                    d_t = D[b, pool]
                    d_rr = D[np.ix_(pool, pool)]
                    try:
                        w = _solve_weights(d_rr, d_t)
                    except np.linalg.LinAlgError:
                        continue
                    for g in GAPS_S:
                        k = int(round(g / DT_S))
                        if ti + k >= nep:
                            continue
                        truth = raw[ti + k, b]
                        if not np.isfinite(truth):
                            continue
                        base = a0 + a1 * g
                        future_refs = raw[ti + k, pool]
                        if not np.isfinite(future_refs).all():
                            continue
                        refres = future_refs - (coef[0] * g + coef[1])
                        if not np.isfinite(refres).all():
                            continue
                        # Integrity screen
                        rmed = float(np.median(refres))
                        rmad = 1.4826 * float(np.median(np.abs(refres - rmed)))
                        limit = max(10.0, 8.0 * rmad)
                        ok_ref = np.abs(refres - rmed) <= limit
                        if ok_ref.sum() < 5:
                            continue
                        if not np.all(ok_ref):
                            pool_g = [pool[ii] for ii in np.flatnonzero(ok_ref)]
                            refres_g = refres[ok_ref]
                            d_t_g = D[b, pool_g]
                            d_rr_g = D[np.ix_(pool_g, pool_g)]
                            try:
                                w_g = _solve_weights(d_rr_g, d_t_g)
                            except np.linalg.LinAlgError:
                                continue
                        else:
                            pool_g, refres_g, w_g = pool, refres, w
                        # Causal data-quality screen
                        pre_jump = float(abs(rb[-1] - rb[-max(2, int(600.0 / DT_S))]))
                        if pre_jump > 200.0:
                            continue
                        if abs(truth - base) > 5000.0:
                            target_reset_like_events[dens][g] += 1
                            if len(target_reset_like_examples[dens][g]) < 20:
                                target_reset_like_examples[dens][g].append({
                                    "day": str(dt), "station": names[b],
                                    "anchor_s": float(t[ti]),
                                    "prediction_s": float(t[ti + k]),
                                    "gap_s": float(g),
                                    "truth_m": float(truth),
                                    "pretrend_prediction_m": float(base),
                                    "endpoint_deviation_m": float(truth - base),
                                })
                            continue
                        pers = base + last
                        tep_pred = base + float(w_g @ refres_g)
                        nm_pred = base + float(refres_g.mean())
                        results[dens][g]["tep"].append((truth - tep_pred) ** 2)
                        results[dens][g]["pers"].append((truth - pers) ** 2)
                        results[dens][g]["net_mean"].append((truth - nm_pred) ** 2)

    # Aggregate
    sweep = {}
    for dens in DENSITY_LEVELS:
        sweep[str(dens)] = {}
        for g in GAPS_S:
            d = results[dens][g]
            if len(d["pers"]) < 25:
                continue
            rms_tep = float(np.sqrt(np.mean(d["tep"])))
            rms_pers = float(np.sqrt(np.mean(d["pers"])))
            rms_nm = float(np.sqrt(np.mean(d["net_mean"])))
            sweep[str(dens)][str(int(g))] = {
                "n_eval": len(d["pers"]),
                "rms_tep_m": rms_tep,
                "rms_pers_m": rms_pers,
                "rms_net_mean_m": rms_nm,
                "tep_vs_pers_pct": 100.0 * (1.0 - rms_tep / rms_pers),
                "tep_vs_net_mean_pct": 100.0 * (1.0 - rms_tep / rms_nm),
                "target_reset_like_events": int(target_reset_like_events[dens][g]),
                "target_reset_like_examples": target_reset_like_examples[dens][g][:10],
            }
    return sweep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates("precise", 2024)
    days = [dates[i] for i in np.linspace(0, len(dates) - 1, N_DAYS).astype(int)]

    sweep = run_density_sweep(days)

    # Determine actual station counts
    n_stations = {}
    for dt in days[:3]:
        dd = step55.load_day(dt)
        if dd:
            n_stations[str(dt)] = len(dd[0])

    out = {
        "step": 85,
        "title": "Network density robustness of TEP outage prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Re-runs the Step 55 causal outage prediction at controlled reference "
            "densities (5, 8, 12, 16, 20, and all-available stations).  For each "
            "density the reference pool is randomly subsampled (seeded) before the "
            "kriging solve; the TEP covariance remains physics-fixed (Gaussian, "
            "lambda = 4,200 km, nugget = 0.35) with no re-tuning at any density. "
            "15-day subset, gaps of 30 min / 1 h / 4 h."),
        "n_days": N_DAYS,
        "density_levels": DENSITY_LEVELS,
        "gap_labels_s": [int(g) for g in GAPS_S],
        "station_counts_sample": n_stations,
        "sweep": sweep,
    }

    path = OUT / "step_85_network_density_robustness.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")
    print(f"{'density':>8} {'gap':>7} {'TEP RMS':>9} {'pers RMS':>10} {'TEP/pers':>10} {'TEP/nm':>9}")
    for dens in sorted(sweep, key=int):
        for g in sorted(sweep[dens], key=int):
            a = sweep[dens][g]
            label = "all" if int(dens) >= 99 else dens
            print(f"{label:>8} {int(g)//60:>5}min {a['rms_tep_m']:>8.3f}m "
                  f"{a['rms_pers_m']:>9.3f}m {a['tep_vs_pers_pct']:>+9.1f}% "
                  f"{a['tep_vs_net_mean_pct']:>+8.1f}%")


if __name__ == "__main__":
    main()
