#!/usr/bin/env python3
"""
STEP 10  (Phase 1 / T1) -- Global correlation length + null controls.

Question: does the operational PPP clock dataset reproduce the distance-structured
correlation established by TEP-GNSS Papers 1-3/14 (lambda_T ~ 1000-5000 km,
R^2 > 0.5 on binned means, all null controls collapsing to R^2 ~ 0)?

Method: pool zero-lag Pearson correlations of detrended receiver-clock residuals
over every station pair and every available day, bin by great-circle distance,
fit C(r) = A exp(-r/lambda) + C0.  Repeat under three label-permutation nulls.

Two arms:
  global  -- broadcast-ephemeris SPP, ~50 stations to 18,000 km  (matches Paper 3)
  eu_prec -- precise clock products, ~27 European stations        (common-mode reference)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
import itertools

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td              # noqa: E402
from scripts.lib import tep_stats as ts             # noqa: E402

RNG = np.random.default_rng(20260901)
OUT = PROJECT_ROOT / "results" / "outputs"
N_NULL = 200


def pair_correlations(series: dict, band=True):
    """Return (dist_km, corr) arrays over all station pairs in one day.

    Residuals are band-passed to 10-500 uHz first so full-band white
    measurement noise does not dilute the correlation (cf. Papers 1-3).
    """
    names = list(series)
    cols = []
    for n in names:
        x = series[n]["resid"]
        dt = float(np.median(np.diff(series[n]["t"])))
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0
        cols.append(ts.bandpass(x, dt) if band else (x - np.mean(x)))
    R = np.column_stack(cols)
    R = R - R.mean(0)  # Remove time mean (per-station detrending)
    sd = R.std(0)
    sd[sd < 1e-9] = np.nan
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        C = (R.T @ R) / len(R) / np.outer(sd, sd)
    d, c = [], []
    for i, j in itertools.combinations(range(len(names)), 2):
        cij = C[i, j]
        if not np.isfinite(cij):
            continue
        d.append(td.great_circle_km(series[names[i]]["station"], series[names[j]]["station"]))
        c.append(float(cij))
    return np.array(d), np.array(c)


def pooled_fit(days_series, transform=None, dmax=None, seed=None):
    """Bin pooled pair correlations by distance and fit the exponential model."""
    dd, cc = [], []
    rng = np.random.default_rng(seed) if seed is not None else RNG
    for s in days_series:
        s2 = transform(s, rng) if transform else s
        d, c = pair_correlations(s2)
        dd.append(d); cc.append(c)
    dd = np.concatenate(dd); cc = np.concatenate(cc)
    centers, means, sems, counts = ts.bin_by_distance(dd, cc, n_bins=28, dmin=100.0,
                                                      dmax=dmax, min_count=15)
    fit = ts.fit_correlation_length(centers, means, weight=1.0 / np.clip(sems, 1e-9, None) ** 2)
    fit["bins"] = int(len(centers))
    fit["n_pairs"] = int(len(dd))
    return fit, (centers, means, sems, counts)


def run_arm(product: str, date_tags, dmax: float | None, label: str):
    from joblib import Parallel, delayed
    
    days = [s for _, s in td.load_multi_day(product, date_tags)]
    days = [s for s in days if len(s) >= 4]
    n_st = sorted({n for s in days for n in s})
    print(f"\n[{label}]  {len(days)} days, {len(n_st)} distinct stations")

    fit, binned = pooled_fit(days, dmax=dmax)
    print(f"  exponential fit: lambda = {fit['lam']:.0f} +/- {fit['sigma_lam']:.0f} km   "
          f"R2 = {fit['R2']:.3f}   A = {fit['A']:.2f}  C0 = {fit['C0']:+.2f}   "
          f"({fit['n_pairs']:,} pair-days, {fit['bins']} bins)")

    # ---- covariance-FORM selection (AIC) --------------------------------
    # Papers 1-2 select the Gaussian / squared-exponential covariance by AIC
    # (dAIC = -12.8 vs exponential).  Forcing the exponential here made lambda
    # and the reported amplitude misspecified: on the operational data the
    # exponential is driven to a NONPHYSICAL A > 1 with a large negative C0 to
    # mimic the Gaussian's flat core and sharp shoulder.  We therefore report
    # the AIC-SELECTED covariance as primary, and keep the exponential fit
    # alongside it for continuity with earlier versions.
    w_bin = 1.0 / np.clip(binned[2], 1e-9, None) ** 2
    mc = ts.fit_all_models(binned[0], binned[1], weight=w_bin)
    best_name = mc.get("best_model", "exponential")
    best = mc.get("models", {}).get(best_name, {})
    bp = best.get("params", [])
    bsig = best.get("sigma_params", [])
    lam_sel = float(bp[1]) if len(bp) > 1 else fit["lam"]
    sig_sel = float(bsig[1]) if len(bsig) > 1 else fit["sigma_lam"]
    r2_sel = float(best.get("R2", fit["R2"]))
    a_sel = float(bp[0]) if len(bp) > 2 else fit["A"]
    c0_sel = float(bp[2]) if len(bp) > 2 else fit["C0"]
    print(f"  AIC-selected form: {best_name}  lambda = {lam_sel:.0f} +/- {sig_sel:.0f} km  "
          f"R2 = {r2_sel:.3f}  A = {a_sel:.2f}  C0 = {c0_sel:+.2f}"
          if len(bp) > 2 else f"  AIC-selected form: {best_name}")
    print(f"    dAIC(exponential vs best) = "
          f"{mc.get('delta_aic_exponential_vs_best', float('nan')):.1f}")

    nulls = {}
    all_null_r2 = []
    for nm, fn in (("temporal_shuffle", ts.null_temporal_shuffle),
                   ("spatial_shuffle", ts.null_spatial_shuffle),
                   ("phase_randomise", ts.null_phase_randomise)):
        seeds = RNG.integers(0, 2**31, size=N_NULL)
        # threading backend: pair_correlations is dominated by GIL-releasing
        # numpy/scipy (bandpass, R.T@R) so threads overlap the compute, and --
        # unlike loky -- the ~300 MB-2 GB pooled day structure is shared, not
        # pickled per worker (which was OOM-killing workers on the full-year
        # global CLK arm).
        results = Parallel(n_jobs=8, backend="threading")(
            delayed(pooled_fit)(days, transform=fn, dmax=dmax, seed=s) for s in seeds
        )
        r2s = np.array([res[0]["R2"] for res in results], float)
        lams = np.array([res[0]["lam"] for res in results], float)
        all_null_r2.append(r2s)
        # Permutation p-value: proportion of FINITE nulls that exceed real R2.
        # NaN nulls (failed fits) are excluded, not counted as non-rejections.
        # Use the conservative Monte Carlo p-value (Barnard 1963, Davison &
        # Hinkley 1997): (#{null >= obs} + 1) / (N_finite + 1), which
        # guarantees p > 0 and avoids the anti-conservative p = 0 when no
        # null exceeds the observed value.
        finite = np.isfinite(r2s)
        n_finite = int(np.sum(finite))
        p = float((np.sum(r2s[finite] >= fit["R2"]) + 1) / (n_finite + 1))
        nulls[nm] = {"R2_mean": float(np.nanmean(r2s)), "R2_p95": float(np.nanpercentile(r2s, 95)),
                     "R2_max": float(np.nanmax(r2s)), "lam_median": float(np.nanmedian(lams)),
                     "perm_p": p}
        print(f"  null {nm:18s}: R2 mean {nulls[nm]['R2_mean']:+.3f}  p95 {nulls[nm]['R2_p95']:+.3f}  "
              f"perm-p(real>=null) {p:.3f}")
    null_pool = np.concatenate(all_null_r2)
    null_sep = (fit["R2"] - np.nanmean(null_pool)) / (np.nanstd(null_pool) + 1e-9)

    # Precompute per-day pair data (distances, correlations, station-name pairs)
    # so the bootstrap can weight pairs by station draw multiplicity without
    # recomputing correlations 200 times.
    per_day_pairs = []
    for s in days:
        d_arr, c_arr = pair_correlations(s)
        names_list = list(s)
        pair_names = [(names_list[i], names_list[j])
                      for i, j in itertools.combinations(range(len(names_list)), 2)]
        ok = np.isfinite(d_arr) & np.isfinite(c_arr)
        per_day_pairs.append((d_arr[ok], c_arr[ok], [pair_names[k] for k in np.where(ok)[0]]))

    # station-clustered bootstrap on lambda, honouring with-replacement
    # multiplicity: a station drawn k times weights its pairs by k_i * k_j.
    def lam_stat(sel):
        from collections import Counter
        counts = Counter(sel)
        dd, cc = [], []
        for d_arr, c_arr, pn in per_day_pairs:
            for k, (dij, cij) in enumerate(zip(d_arr, c_arr)):
                ni, nj = pn[k]
                wi = counts.get(ni, 0)
                wj = counts.get(nj, 0)
                w = wi * wj
                if w > 0:
                    dd.extend([dij] * w)
                    cc.extend([cij] * w)
        if len(dd) < 100:
            return np.nan
        dd, cc = np.array(dd), np.array(cc)
        centers, means, sems, counts_b = ts.bin_by_distance(dd, cc, n_bins=28, dmin=100.0,
                                                             dmax=dmax, min_count=15)
        if len(centers) < 4:
            return np.nan
        # Fit the SAME covariance form the point estimate uses, otherwise the
        # CI is for a different parameter than the reported lambda (the
        # exponential-fit CI [4665, 25754] did not even bracket the
        # Gaussian point estimate of 3480 km).
        wb = 1.0 / np.clip(sems, 1e-9, None) ** 2
        if best_name == "exponential":
            f = ts.fit_correlation_length(centers, means, weight=wb)
            return f["lam"] if f["R2"] > 0.2 else np.nan
        mcb = ts.fit_all_models(centers, means, weight=wb)
        mb = mcb.get("models", {}).get(best_name, {})
        pb = mb.get("params", [])
        if len(pb) < 2 or not np.isfinite(mb.get("R2", np.nan)) or mb["R2"] <= 0.2:
            return np.nan
        return float(pb[1])
    pt, lo, hi, _ = ts.station_clustered_bootstrap(lam_stat, n_st, n_boot=200, rng=RNG)
    print(f"  lambda 95% CI (station-clustered): [{lo:.0f}, {hi:.0f}] km")

    # T1 is judged on the AIC-SELECTED covariance (the exponential is
    # misspecified here -- see the model-comparison block above).
    passes = (1000 <= lam_sel <= 5000 and r2_sel > 0.5
              and all(v["perm_p"] < 0.05 for v in nulls.values())
              and null_sep > 3.0)
    print(f"  real R2 {r2_sel:.3f} vs null pool mean {np.nanmean(null_pool):.3f} "
          f"(separation {null_sep:.1f} sigma)")
    return {
        "label": label, "product": product, "n_days": len(days), "n_stations": len(n_st),
        # primary = AIC-selected covariance
        "lambda_km": lam_sel, "R2": r2_sel, "covariance_form": best_name,
        "A": a_sel, "C0": c0_sel,
        "sigma_lambda_km": sig_sel,
        "model_comparison": {
            "best_model": best_name,
            "delta_aic_exponential_vs_best": mc.get("delta_aic_exponential_vs_best"),
            "ranked_by_aic": mc.get("ranked_by_aic", []),
            "per_model": {k: {kk: vv for kk, vv in v.items() if kk != "params"}
                          for k, v in mc.get("models", {}).items()},
        },
        # exponential fit retained for continuity with earlier versions
        "exponential_fit": {
            "lambda_km": fit["lam"], "sigma_lambda_km": fit["sigma_lam"],
            "R2": fit["R2"], "A": fit["A"], "C0": fit["C0"],
        },
        "n_pair_days": fit["n_pairs"], "n_bins": fit["bins"],
        "lambda_ci95": [lo, hi], "nulls": nulls, "null_separation_sigma": float(null_sep),
        "binned": {"dist_km": binned[0].tolist(), "corr": binned[1].tolist(),
                   "sem": binned[2].tolist(), "count": binned[3].tolist()},
        "T1_pass": bool(passes),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global_days = [f"2024{d:03d}" for d in range(100, 107)]
    eu_days = td.available_dates("precise", 2024)[:60]
    # NOTE: the full-year global analysis-centre CLK arm (T1 on ~128 global
    # stations x ~360 days) is run by step_14_precise_global.py, which is the
    # dedicated global-precise step.  step_09 downloads that product.
    result = {
        "test": "T1_global_correlation_length",
        "global_broadcast": run_arm("broadcast", global_days, dmax=17000.0, label="global_broadcast"),
        "eu_precise": run_arm("precise", eu_days, dmax=1300.0, label="eu_precise"),
        "timestamp": datetime.now().isoformat(),
    }
    result["T1_pass"] = result["global_broadcast"]["T1_pass"]
    # The European precise arm is a complementary check on a different
    # product and aperture; it does not gate T1 (which is defined on the
    # global broadcast arm matching Paper 3), but its pass/fail is recorded
    # separately for transparency.
    result["T1_eu_precise_pass"] = result["eu_precise"]["T1_pass"]
    (OUT / "step_10_global_correlation.json").write_text(json.dumps(result, indent=2))
    print(f"\nT1 pass (global arm): {result['T1_pass']}")
    print(f"T1 pass (eu_precise arm): {result['T1_eu_precise_pass']}")
    print(f"saved {OUT / 'step_10_global_correlation.json'}")


if __name__ == "__main__":
    main()
