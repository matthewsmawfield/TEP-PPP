#!/usr/bin/env python3
"""
STEP 87  --  Pre-window length sensitivity of the TEP outage prediction.

Step 55 uses a fixed 2-hour pre-anchor window for the causal trend fit and
the temporal baseline selection.  A reviewer may ask whether the result is
sensitive to this choice.  This step re-runs the Step 55 outage prediction
at 1 h, 2 h, and 4 h pre-window lengths on a 15-day subset, at the 1-hour
and 4-hour outage horizons.  The TEP covariance remains physics-fixed
(lambda = 4,200 km, nugget = 0.35) with no re-tuning.

Output: results/outputs/step_87_prewindow_sensitivity.json
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

_step55 = PROJECT_ROOT / "scripts" / "steps" / "step_55_dense_outage_prediction.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("step55", _step55)
step55 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(step55)

from scripts.lib import tep_data as td  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260905)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.
# Ill-conditioned numerics are wrapped in local np.errstate contexts.
# np.seterr(all="ignore")  # removed: masks bugs globally

DT_S = 300.0
GAPS_S = [3600.0, 14400.0]  # 1 h and 4 h
EVAL_STRIDE = 3
N_DAYS = int(os.environ.get("TEP_PREWIN_N_DAYS", "15"))
PREWINDOW_OPTIONS = [3600.0, 7200.0, 14400.0]  # 1 h, 2 h, 4 h
LAMBDA_KM = 4200.0
NUGGET = 0.35


def _rho(d):
    return np.exp(-0.5 * (d / LAMBDA_KM) ** 2)


def _solve_weights(d_rr, d_t):
    K = _rho(d_rr) + (NUGGET + 1e-9) * np.eye(len(d_t))
    return np.linalg.solve(K, _rho(d_t))


def run_prewindow_sweep(days):
    results = {}
    for prewin in PREWINDOW_OPTIONS:
        results[str(int(prewin))] = {g: {"tep": [], "pers": []} for g in GAPS_S}
        npre = int(prewin / DT_S)
    target_reset_like_events = {str(int(prewin)): {g: 0 for g in GAPS_S} for prewin in PREWINDOW_OPTIONS}
    target_reset_like_examples = {str(int(prewin)): {g: [] for g in GAPS_S} for prewin in PREWINDOW_OPTIONS}

    for dt in days:
        dd = step55.load_day(dt)
        if dd is None:
            continue
        names, t, raw, D = dd
        n = len(names)
        nep = len(t)
        if n - 1 < 8:
            continue

        for prewin in PREWINDOW_OPTIONS:
            npre = int(prewin / DT_S)
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
                if len(finite_sd) < 9:
                    continue
                med_sd = float(np.median(finite_sd))
                causal_good = ((pre_sd > 0.01) & (pre_sd < 60.0) &
                               (pre_sd < 4.0 * med_sd))
                for b in range(n):
                    if not causal_good[b]:
                        continue
                    base_pool = [j for j in range(n) if j != b and causal_good[j] and
                                 np.isfinite(raw[sl, j]).all()]
                    if len(base_pool) < 8:
                        continue
                    y = raw[sl, b]
                    if not np.isfinite(y).all():
                        continue
                    a1, a0 = np.polyfit(tt, y, 1)
                    if abs(a1) > 0.20:
                        continue
                    pool = base_pool
                    Y = raw[sl][:, pool]
                    if not np.isfinite(Y).all():
                        continue
                    coef = np.linalg.lstsq(Adm, Y, rcond=None)[0]
                    ref_ok = np.abs(coef[0]) <= 0.20
                    if ref_ok.sum() < 8:
                        continue
                    pool = [p for p, ok in zip(pool, ref_ok) if ok]
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
                    pre_jump = float(abs(rb[-1] - rb[-max(2, int(600.0 / DT_S))]))
                    if pre_jump > 200.0:
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
                        rmed = float(np.median(refres))
                        rmad = 1.4826 * float(np.median(np.abs(refres - rmed)))
                        limit = max(10.0, 8.0 * rmad)
                        ok_ref = np.abs(refres - rmed) <= limit
                        if ok_ref.sum() < 8:
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
                        if abs(truth - base) > 5000.0:
                            target_reset_like_events[str(int(prewin))][g] += 1
                            if len(target_reset_like_examples[str(int(prewin))][g]) < 20:
                                target_reset_like_examples[str(int(prewin))][g].append({
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
                        results[str(int(prewin))][g]["tep"].append((truth - tep_pred) ** 2)
                        results[str(int(prewin))][g]["pers"].append((truth - pers) ** 2)

    sweep = {}
    for prewin in PREWINDOW_OPTIONS:
        sweep[str(int(prewin))] = {}
        for g in GAPS_S:
            d = results[str(int(prewin))][g]
            if len(d["pers"]) < 25:
                continue
            rms_tep = float(np.sqrt(np.mean(d["tep"])))
            rms_pers = float(np.sqrt(np.mean(d["pers"])))
            sweep[str(int(prewin))][str(int(g))] = {
                "n_eval": len(d["pers"]),
                "rms_tep_m": rms_tep,
                "rms_pers_m": rms_pers,
                "tep_vs_pers_pct": 100.0 * (1.0 - rms_tep / rms_pers),
                "target_reset_like_events": int(target_reset_like_events[str(int(prewin))][g]),
                "target_reset_like_examples": target_reset_like_examples[str(int(prewin))][g][:10],
            }
    return sweep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates("precise", 2024)
    days = [dates[i] for i in np.linspace(0, len(dates) - 1, N_DAYS).astype(int)]

    sweep = run_prewindow_sweep(days)

    out = {
        "step": 87,
        "title": "Pre-window length sensitivity of TEP outage prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Re-runs the Step 55 causal outage prediction at 1 h, 2 h, and 4 h "
            "pre-window lengths on a 15-day subset. The TEP covariance remains "
            "physics-fixed (lambda = 4,200 km, nugget = 0.35) with no re-tuning. "
            "Tests whether the result is sensitive to the pre-window choice."),
        "n_days": N_DAYS,
        "prewindow_options_s": [int(p) for p in PREWINDOW_OPTIONS],
        "gap_labels_s": [int(g) for g in GAPS_S],
        "sweep": sweep,
    }

    path = OUT / "step_87_prewindow_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")
    print(f"{'prewin':>8} {'gap':>7} {'TEP RMS':>9} {'pers RMS':>10} {'TEP/pers':>10}")
    for prewin in sorted(sweep, key=int):
        for g in sorted(sweep[prewin], key=int):
            a = sweep[prewin][g]
            print(f"{int(prewin)//3600:>6}h {int(g)//60:>5}min "
                  f"{a['rms_tep_m']:>8.3f}m {a['rms_pers_m']:>9.3f}m "
                  f"{a['tep_vs_pers_pct']:>+9.1f}%")


if __name__ == "__main__":
    main()
