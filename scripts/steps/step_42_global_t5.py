#!/usr/bin/env python3
"""
STEP 42 -- Global T5: constellation/product independence at global aperture.

The step_27 T5 test compares broadcast vs precise correlations on <400 km
baselines (30-station European dense network), where distance structure is
degenerate. This step runs the same comparison at GLOBAL aperture (baselines
to 19,000 km) where distance structure is resolvable, using the 52-station
global broadcast set (DOY 100-106) and the COD precise receiver-clock product
on the same days.

The test asks: does the distance-dependent correlation survive the removal
of ~two orders of magnitude of satellite-clock error, at baselines where
distance structure is actually resolvable?
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402
from scripts.steps.step_27_discriminators import _clean, corr_matrix  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
DAYS = ["2024100", "2024101", "2024102", "2024103",
        "2024104", "2024105", "2024106"]


def collect_pairs(product, days):
    recs = []
    for day in days:
        s = td.load_series(product, day)
        if len(s) < 20:
            continue
        nms = list(s)
        cleaned, sts = [], []
        for nm in nms:
            b = _clean(s[nm]["resid"], s[nm]["t"])
            if b is not None:
                cleaned.append(b)
                sts.append(s[nm]["station"])
        if len(cleaned) < 20:
            continue
        Cm = corr_matrix(cleaned)
        for i, j in itertools.combinations(range(len(cleaned)), 2):
            if not np.isfinite(Cm[i, j]):
                continue
            dist = td.great_circle_km(sts[i], sts[j])
            if dist < 200 or dist > 19000:
                continue
            recs.append((dist, float(Cm[i, j])))
    return recs


def fit_exponential(recs, bin_km=1000.0):
    d = np.array([r[0] for r in recs])
    c = np.array([r[1] for r in recs])
    bins = np.arange(500, 19000, bin_km)
    centers = (bins[:-1] + bins[1:]) / 2
    means, ns = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 5:
            means.append(c[m].mean())
            ns.append(m.sum())
        else:
            means.append(np.nan)
            ns.append(0)
    means = np.array(means)
    centers = np.array(centers)
    valid = np.isfinite(means)

    def model(r, A, lam, C0):
        return A * np.exp(-r / lam) + C0

    try:
        popt, _ = curve_fit(model, centers[valid], means[valid],
                            p0=[0.5, 3000, 0.1], maxfev=10000)
        r2 = 1 - np.sum((means[valid] - model(centers[valid], *popt))**2) / \
             np.sum((means[valid] - means[valid].mean())**2)
    except Exception:
        popt, r2 = [np.nan, np.nan, np.nan], np.nan
    return {
        "n_pairs": len(recs),
        "lambda_km": float(popt[1]) if np.isfinite(popt[1]) else None,
        "A": float(popt[0]) if np.isfinite(popt[0]) else None,
        "C0": float(popt[2]) if np.isfinite(popt[2]) else None,
        "R2": float(r2) if np.isfinite(r2) else None,
        "bin_centers_km": centers.tolist(),
        "bin_means": means.tolist(),
        "bin_counts": ns,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Global T5: constellation/product independence at global aperture\n")
    print(f"Days: {DAYS}\n")

    bcast_recs = collect_pairs("broadcast", DAYS)
    precise_recs = collect_pairs("clk_cod", DAYS)

    print(f"Broadcast: {len(bcast_recs)} pairs")
    print(f"Precise (COD): {len(precise_recs)} pairs\n")

    bcast_fit = fit_exponential(bcast_recs)
    precise_fit = fit_exponential(precise_recs)

    print(f"Broadcast fit: lambda={bcast_fit['lambda_km']:.0f} km, "
          f"R2={bcast_fit['R2']:.3f}")
    print(f"Precise fit:   lambda={precise_fit['lambda_km']:.0f} km, "
          f"R2={precise_fit['R2']:.3f}")

    # Compare correlation at matched distances
    print("\nDistance-matched comparison:")
    bins = np.arange(1000, 16000, 2000.0)
    bd = np.array([r[0] for r in bcast_recs])
    bc = np.array([r[1] for r in bcast_recs])
    pd_ = np.array([r[0] for r in precise_recs])
    pc = np.array([r[1] for r in precise_recs])
    comparison = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bm = (bd >= lo) & (bd < hi)
        pm = (pd_ >= lo) & (pd_ < hi)
        if bm.sum() > 5 and pm.sum() > 5:
            b_mean = float(bc[bm].mean())
            p_mean = float(pc[pm].mean())
            ratio = p_mean / b_mean if abs(b_mean) > 1e-6 else np.nan
            print(f"  {lo:5.0f}-{hi:5.0f} km: broadcast={b_mean:+.4f}  "
                  f"precise={p_mean:+.4f}  ratio={ratio:.3f}")
            comparison.append({
                "dist_lo_km": lo, "dist_hi_km": hi,
                "broadcast_corr": b_mean, "precise_corr": p_mean,
                "ratio": ratio,
                "n_broadcast": int(bm.sum()), "n_precise": int(pm.sum()),
            })

    # Key T5 metric: does the precise product show distance-dependent
    # correlation at global aperture?
    precise_has_structure = (precise_fit["R2"] is not None and
                             precise_fit["R2"] > 0.8 and
                             precise_fit["lambda_km"] is not None and
                             1000 < precise_fit["lambda_km"] < 10000)

    res = {
        "test": "global_t5_constellation_independence",
        "days": DAYS,
        "broadcast": bcast_fit,
        "precise_cod": precise_fit,
        "distance_matched_comparison": comparison,
        "precise_has_distance_structure": precise_has_structure,
        "timestamp": datetime.now().isoformat(),
    }

    outpath = OUT / "step_42_global_t5.json"
    outpath.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved {outpath}")

    print("\n" + "=" * 72)
    print("GLOBAL T5 RESULT")
    print("=" * 72)
    print(f"  Precise product: lambda={precise_fit['lambda_km']:.0f} km, "
          f"R2={precise_fit['R2']:.3f}")
    print(f"  Distance structure present at global aperture: {precise_has_structure}")
    print(f"  Broadcast lambda={bcast_fit['lambda_km']:.0f} km (contaminated by "
          f"satellite common-mode, R2={bcast_fit['R2']:.3f})")


if __name__ == "__main__":
    main()
