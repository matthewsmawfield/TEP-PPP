#!/usr/bin/env python3
"""
STEP 88  --  Per-station breakdown of the TEP outage prediction.

Step 55 reports aggregate RMS improvement across all station-days.  A reviewer
or deployer will ask: is the benefit broad, or driven by a few outlier
stations?  This step computes per-station TEP-vs-persistence RMS at the 1-hour
and 4-hour outage horizons, reports the fraction of stations where TEP wins,
the median per-station improvement, and the interquartile range.  It also
checks seasonal stability by splitting the 90 days into four quarters.

Output: results/outputs/step_88_per_station_breakdown.json
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse Step 55 machinery
_step55 = PROJECT_ROOT / "scripts" / "steps" / "step_55_dense_outage_prediction.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("step55", _step55)
step55 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(step55)

from scripts.lib import tep_data as td  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260904)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.
# Ill-conditioned numerics are wrapped in local np.errstate contexts.
# np.seterr(all="ignore")  # removed: masks bugs globally

DT_S = 300.0
PREWINDOW_S = 7200.0
GAPS_S = [3600.0, 14400.0]  # 1 h and 4 h
EVAL_STRIDE = 3
N_DAYS = int(os.environ.get("TEP_N_DAYS", "90"))
LAMBDA_KM = 4200.0
NUGGET = 0.35


def _rho(d):
    return np.exp(-0.5 * (d / LAMBDA_KM) ** 2)


def _solve_weights(d_rr, d_t):
    K = _rho(d_rr) + (NUGGET + 1e-9) * np.eye(len(d_t))
    return np.linalg.solve(K, _rho(d_t))


def run_station_breakdown(days):
    npre = int(PREWINDOW_S / DT_S)
    # per_station[name][gap] -> {"tep": [sq_errs], "pers": [sq_errs]}
    per_station = defaultdict(lambda: {g: {"tep": [], "pers": []} for g in GAPS_S})
    # per_quarter[quarter][gap] -> {"tep": [...], "pers": [...]}
    per_quarter = defaultdict(lambda: {g: {"tep": [], "pers": []} for g in GAPS_S})
    target_reset_like_events = {g: 0 for g in GAPS_S}
    target_reset_like_examples = {g: [] for g in GAPS_S}

    for day_idx, dt in enumerate(days):
        quarter = day_idx // max(1, len(days) // 4)
        dd = step55.load_day(dt)
        if dd is None:
            continue
        names, t, raw, D = dd
        n = len(names)
        nep = len(t)
        if n - 1 < 8:
            continue

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
                            })
                        continue
                    pers = base + last
                    tep_pred = base + float(w_g @ refres_g)
                    sq_tep = (truth - tep_pred) ** 2
                    sq_pers = (truth - pers) ** 2
                    per_station[names[b]][g]["tep"].append(sq_tep)
                    per_station[names[b]][g]["pers"].append(sq_pers)
                    per_quarter[quarter][g]["tep"].append(sq_tep)
                    per_quarter[quarter][g]["pers"].append(sq_pers)

    # Aggregate per-station
    station_results = {}
    for name in sorted(per_station):
        station_results[name] = {}
        for g in GAPS_S:
            d = per_station[name][g]
            if len(d["pers"]) < 10:
                continue
            rms_tep = float(np.sqrt(np.mean(d["tep"])))
            rms_pers = float(np.sqrt(np.mean(d["pers"])))
            station_results[name][str(int(g))] = {
                "n_eval": len(d["pers"]),
                "rms_tep_m": rms_tep,
                "rms_pers_m": rms_pers,
                "tep_vs_pers_pct": 100.0 * (1.0 - rms_tep / rms_pers),
                "tep_wins": rms_tep < rms_pers,
            }

    # Summary statistics
    summary = {}
    for g in GAPS_S:
        improvements = [station_results[s][str(int(g))]["tep_vs_pers_pct"]
                        for s in station_results
                        if str(int(g)) in station_results[s]]
        wins = sum(1 for s in station_results
                   if str(int(g)) in station_results[s] and
                   station_results[s][str(int(g))]["tep_wins"])
        total = len(improvements)
        # Formal sign test: under H0 (TEP = persistence), each station has
        # P(TEP wins) = 0.5.  The probability of k wins out of n is
        # Binomial(n, 0.5).  The one-sided p-value is P(X >= k).
        try:
            from scipy.stats import binomtest
            sign_p = float(binomtest(wins, total, 0.5, alternative="greater").pvalue)
        except (ImportError, Exception):
            # Manual binomial tail if scipy is unavailable
            from math import comb
            sign_p = sum(comb(total, k) for k in range(wins, total + 1)) / 2.0 ** total
        summary[str(int(g))] = {
            "n_stations": total,
            "tep_wins": wins,
            "tep_win_fraction": wins / total if total else 0,
            "sign_test_p_value": sign_p,
            "median_improvement_pct": float(np.median(improvements)) if improvements else 0,
            "p25_improvement_pct": float(np.percentile(improvements, 25)) if improvements else 0,
            "p75_improvement_pct": float(np.percentile(improvements, 75)) if improvements else 0,
            "min_improvement_pct": float(min(improvements)) if improvements else 0,
            "max_improvement_pct": float(max(improvements)) if improvements else 0,
            "target_reset_like_events": int(target_reset_like_events[g]),
            "target_reset_like_examples": target_reset_like_examples[g][:10],
        }

    # Seasonal stability
    seasonal = {}
    for q in sorted(per_quarter):
        seasonal[f"Q{q+1}"] = {}
        for g in GAPS_S:
            d = per_quarter[q][g]
            if len(d["pers"]) < 25:
                continue
            rms_tep = float(np.sqrt(np.mean(d["tep"])))
            rms_pers = float(np.sqrt(np.mean(d["pers"])))
            seasonal[f"Q{q+1}"][str(int(g))] = {
                "n_eval": len(d["pers"]),
                "rms_tep_m": rms_tep,
                "rms_pers_m": rms_pers,
                "tep_vs_pers_pct": 100.0 * (1.0 - rms_tep / rms_pers),
            }

    return station_results, summary, seasonal


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates("precise", 2024)
    days = [dates[i] for i in np.linspace(0, len(dates) - 1, N_DAYS).astype(int)]

    station_results, summary, seasonal = run_station_breakdown(days)

    out = {
        "step": 88,
        "title": "Per-station breakdown and seasonal stability of TEP outage prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Per-station TEP-vs-persistence RMS at 1 h and 4 h outage horizons "
            "on the dense EUREF network over 90 days. Reports the fraction of "
            "stations where TEP wins, the median per-station improvement, and "
            "the interquartile range. Seasonal stability is assessed by "
            "splitting the 90 days into four quarters."),
        "n_days": N_DAYS,
        "gaps_s": [int(g) for g in GAPS_S],
        "summary": summary,
        "seasonal_stability": seasonal,
        "per_station": station_results,
    }

    path = OUT / "step_88_per_station_breakdown.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")
    for g in sorted(summary, key=int):
        s = summary[g]
        print(f"Gap {int(g)//60:>4}min: {s['n_stations']} stations, "
              f"TEP wins {s['tep_wins']}/{s['n_stations']} "
              f"({s['tep_win_fraction']*100:.1f}%), "
              f"sign test p={s['sign_test_p_value']:.2e}, "
              f"median improvement {s['median_improvement_pct']:+.1f}%, "
              f"IQR [{s['p25_improvement_pct']:+.1f}, {s['p75_improvement_pct']:+.1f}]")
    print("\nSeasonal stability:")
    for q in sorted(seasonal):
        for g in sorted(seasonal[q], key=int):
            s = seasonal[q][g]
            print(f"  {q} gap {int(g)//60:>4}min: TEP/pers {s['tep_vs_pers_pct']:+.1f}% "
                  f"(n={s['n_eval']})")


if __name__ == "__main__":
    main()
