#!/usr/bin/env python3
"""
STEP 40 -- Band-pass filter sensitivity analysis.

Reviewer concern (C3): the 10-500 uHz band-pass is a strong preprocessing
choice that shapes the signal.  Is the correlation length robust to a
different band?

This step re-runs the T1 correlation-length fit on the global broadcast
product (DOY 100-106, 52 stations) and the COD precise product (90-day
subset) under four band-pass configurations:

  default    10-500 uHz   (the pre-registered band)
  wide       5-1000 uHz   (broader; includes more drift and more noise)
  narrow_lo  20-500 uHz   (higher high-pass; removes longer periods)
  narrow_hi  10-200 uHz   (lower low-pass; removes higher frequencies)
  raw        no band-pass (detrend only; the most hostile control)

For each band the pooled correlation-length fit is reported with R^2 and
the permutation-null separation sigma.  The correlation length is considered
robust if lambda remains in the same order of magnitude (1,000-10,000 km)
and the null separation remains > 3 sigma across bands.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td                       # noqa: E402
from scripts.lib import tep_stats as ts                      # noqa: E402
from scripts.lib import tep_kernel as tk                     # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"

BANDS = {
    "default":   (10.0, 500.0),
    "wide":      (5.0, 1000.0),
    "narrow_lo": (20.0, 500.0),
    "narrow_hi": (10.0, 200.0),
    "raw":       (None, None),       # no band-pass; detrend only
}


def _load_and_fit(product, days, band_lo, band_hi, dmax=19000.0):
    """Load days with a custom band-pass and fit the pooled correlation length."""
    all_series = []
    for d in days:
        s = td.load_series(product, d)
        if len(s) < 8:
            continue
        # Re-apply band-pass if requested; otherwise use detrended residuals
        names, X, D = tk.build_matrix(s, band_lo_uhz=band_lo,
                                      band_hi_uhz=band_hi,
                                      filter_by_std=True)
        if X is None or len(names) < 8:
            continue
        all_series.append((names, X, D))
    if not all_series:
        return None
    # Pooled pair correlations across all days
    from collections import defaultdict
    dist_bins = defaultdict(list)
    for names, X, D in all_series:
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isfinite(D[i, j]) or D[i, j] > dmax:
                    continue
                xi, xj = X[:, i], X[:, j]
                mask = np.isfinite(xi) & np.isfinite(xj)
                if mask.sum() < 30:
                    continue
                c = np.corrcoef(xi[mask], xj[mask])[0, 1]
                if np.isfinite(c):
                    dist_bins[int(D[i, j] / 500) * 500].append(c)
    if len(dist_bins) < 4:
        return None
    dists = np.array(sorted(dist_bins.keys()), float)
    vals = np.array([np.mean(dist_bins[d]) for d in dists])
    weights = np.array([len(dist_bins[d]) for d in dists], float)
    fit = ts.fit_correlation_length(dists, vals, weight=weights)
    # Null separation: temporal shuffle
    rng = np.random.default_rng(42)
    null_r2 = []
    for _ in range(25):
        null_vals = []
        for names, X, D in all_series:
            n = len(names)
            for i in range(n):
                for j in range(i + 1, n):
                    if not np.isfinite(D[i, j]) or D[i, j] > dmax:
                        continue
                    xi = X[:, i].copy()
                    rng.shuffle(xi)
                    xj = X[:, j]
                    mask = np.isfinite(xi) & np.isfinite(xj)
                    if mask.sum() < 30:
                        continue
                    c = np.corrcoef(xi[mask], xj[mask])[0, 1]
                    if np.isfinite(c):
                        null_vals.append((int(D[i, j] / 500) * 500, c))
        if not null_vals:
            continue
        nd = defaultdict(list)
        for d, c in null_vals:
            nd[d].append(c)
        if len(nd) < 4:
            continue
        ndists = np.array(sorted(nd.keys()), float)
        nvals = np.array([np.mean(nd[d]) for d in ndists])
        nweights = np.array([len(nd[d]) for d in ndists], float)
        nfit = ts.fit_correlation_length(ndists, nvals, weight=nweights)
        if np.isfinite(nfit["R2"]):
            null_r2.append(nfit["R2"])
    # A null fit's length scale is not a meaningful signal statistic: random
    # near-zero correlations can return any grid-edge lambda.  Test the fitted
    # explanatory power instead, which is the quantity the permutation destroys.
    if null_r2:
        null_mean = np.mean(null_r2)
        null_std = np.std(null_r2)
        sigma = (fit["R2"] - null_mean) / (null_std + 1e-9)
        p_empirical = (1.0 + np.sum(np.asarray(null_r2) >= fit["R2"])) / (1.0 + len(null_r2))
    else:
        sigma = float("nan")
        p_empirical = float("nan")
    return {
        "lambda_km": int(round(fit["lam"])) if np.isfinite(fit["lam"]) else None,
        "R2": float(round(fit["R2"], 3)) if np.isfinite(fit["R2"]) else None,
        "null_separation_sigma": float(round(sigma, 1)) if np.isfinite(sigma) else None,
        "null_empirical_p": float(p_empirical) if np.isfinite(p_empirical) else None,
        "null_statistic": "fitted_R2",
        "n_bins": len(dist_bins),
        "n_days": len(all_series),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("STEP 40 -- Band-pass filter sensitivity analysis")
    print("=" * 72)

    # Broadcast global (DOY 100-106)
    bdays = td.available_dates("broadcast", 2024)
    bdays = [d for d in bdays if 100 <= int(d[4:7]) <= 106]
    print(f"\nBroadcast global: {len(bdays)} days (DOY 100-106)")

    # COD precise (90-day subset)
    cdays = td.available_dates("clk_cod", 2024)
    idx = np.linspace(0, len(cdays) - 1, 30).round().astype(int)
    cdays = [cdays[i] for i in idx]
    print(f"COD precise: {len(cdays)} days (evenly-spaced subset)")

    results = {}
    for label, (lo, hi) in BANDS.items():
        print(f"\n--- Band: {label} ({lo}-{hi} uHz) ---")
        b_res = _load_and_fit("broadcast", bdays, lo, hi)
        if b_res:
            print(f"  broadcast: lambda={b_res['lambda_km']} km  R2={b_res['R2']}  "
                  f"sigma={b_res['null_separation_sigma']}  n_days={b_res['n_days']}")
        else:
            print("  broadcast: (failed)")
        c_res = _load_and_fit("clk_cod", cdays, lo, hi)
        if c_res:
            print(f"  precise:   lambda={c_res['lambda_km']} km  R2={c_res['R2']}  "
                  f"sigma={c_res['null_separation_sigma']}  n_days={c_res['n_days']}")
        else:
            print("  precise:   (failed)")
        results[label] = {"band_lo_uhz": lo, "band_hi_uhz": hi,
                          "broadcast": b_res, "precise": c_res}

    # Robustness summary
    lams_b = [v["broadcast"]["lambda_km"] for v in results.values()
              if v["broadcast"] and v["broadcast"]["lambda_km"]]
    lams_c = [v["precise"]["lambda_km"] for v in results.values()
              if v["precise"] and v["precise"]["lambda_km"]]
    fits_b = [v["broadcast"] for v in results.values() if v.get("broadcast")]
    fits_c = [v["precise"] for v in results.values() if v.get("precise")]
    broadcast_robust = (len(fits_b) == len(BANDS) and
                        all(1000 <= l <= 10000 for l in lams_b if l) and
                        all((arm.get("null_separation_sigma") or -np.inf) > 3.0
                            for arm in fits_b))
    precise_robust = (len(fits_c) == len(BANDS) and
                      all(1000 <= l <= 10000 for l in lams_c if l) and
                      all((arm.get("null_separation_sigma") or -np.inf) > 3.0
                          for arm in fits_c))
    robust = broadcast_robust and precise_robust
    print(f"\n{'=' * 72}")
    print(f"  Broadcast lambda range: {min(lams_b) if lams_b else '?'}-"
          f"{max(lams_b) if lams_b else '?'} km")
    print(f"  Precise lambda range:   {min(lams_c) if lams_c else '?'}-"
          f"{max(lams_c) if lams_c else '?'} km")
    print(f"  Broadcast robust: {broadcast_robust}")
    print(f"  Precise robust:   {precise_robust}")
    print(f"  Joint robust:     {robust}")
    print(f"{'=' * 72}")

    result = {
        "test": "bandpass_sensitivity",
        "description": "T1 correlation-length fit under alternative band-pass configurations",
        "bands": results,
        "robustness_by_product": {
            "broadcast": broadcast_robust,
            "precise": precise_robust,
        },
        "robust": robust,
        "timestamp": datetime.now().isoformat(),
    }
    (OUT / "step_40_bandpass_sensitivity.json").write_text(json.dumps(result, indent=2))
    print(f"saved {OUT / 'step_40_bandpass_sensitivity.json'}")


if __name__ == "__main__":
    main()
