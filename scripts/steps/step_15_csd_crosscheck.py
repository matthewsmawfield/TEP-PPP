#!/usr/bin/env python3
"""
STEP 15  -- Phase-coherent CSD cross-check of the global correlation.

Question: Papers 1-2 established the TEP signal using the phase-alignment
index cos(arg(Σ_f S_ij(f))) on cross-spectral densities.  Step 10 uses
zero-lag Pearson correlation on Butterworth band-passed residuals.  Do both
methods agree on the same dataset?

Method: for the same broadcast-SPP global days used by step_10, compute the
Paper 1-2 phase-alignment index cos(arg(Σ_f S_ij(f))) in the 10-500 uHz
band, bin the pair values by distance, fit the exponential model, and
compare lambda and R2 to the Pearson result.  Also run the model comparison
(AIC/BIC) on the CSD-binned data.

This is the methodological consistency check between the series' signature
technique and the operational pipeline's simplified approach.
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
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402

RNG = np.random.default_rng(20260901)
OUT = PROJECT_ROOT / "results" / "outputs"
N_NULL = 200


def pooled_csd_fit(days_series, transform=None, dmax=None, seed=None):
    """Bin pooled CSD pair coherences by distance and fit the exponential model."""
    dd, cc = [], []
    rng = np.random.default_rng(seed) if seed is not None else RNG
    for s in days_series:
        s2 = transform(s, rng) if transform else s
        d, c = ts.pair_correlations_csd(s2)
        dd.append(d)
        cc.append(c)
    dd = np.concatenate(dd)
    cc = np.concatenate(cc)
    centers, means, sems, counts = ts.bin_by_distance(dd, cc, n_bins=28, dmin=100.0,
                                                      dmax=dmax, min_count=15)
    w = 1.0 / np.clip(sems, 1e-4, None) ** 2
    # model comparison on CSD-binned data
    mc = ts.fit_all_models(centers, means, weight=w)
    best = mc.get("best_model", "none")
    if best in ("exponential", "gaussian"):
        mm = mc["models"][best]
        p = mm["params"]
        sig = mm.get("sigma_params") or [np.nan] * len(p)
        lam = p[1] if len(p) > 1 and p[0] > 0 else np.nan
        lam_sigma = sig[1] if len(sig) > 1 else np.nan
    else:
        lam = np.nan
        lam_sigma = np.nan
    fit = {
        "lam": float(lam),
        "sigma_lam": float(lam_sigma),
        "A": (mc["models"][best]["params"][0] if best in mc["models"] else np.nan),
        "C0": (mc["models"][best]["params"][-1] if best in mc["models"] else np.nan),
        "R2": (mc["models"][best]["R2"] if best in mc["models"] else np.nan),
        "best_model": best,
        "bins": int(len(centers)),
        "n_pairs": int(len(dd)),
    }
    return fit, (centers, means, sems, counts)


def run_arm(product, date_tags, dmax, label):
    from joblib import Parallel, delayed
    
    days = [s for _, s in td.load_multi_day(product, date_tags)]
    days = [s for s in days if len(s) >= 4]
    n_st = sorted({n for s in days for n in s})
    print(f"\n[{label}]  {len(days)} days, {len(n_st)} distinct stations (CSD method)")

    fit, binned = pooled_csd_fit(days, dmax=dmax)
    print(f"  CSD real:  lambda = {fit['lam']:.0f} +/- {fit['sigma_lam']:.0f} km   "
          f"R2 = {fit['R2']:.3f}   ({fit['n_pairs']:,} pair-days, {fit['bins']} bins)")

    # model comparison on CSD-binned data
    mc = ts.fit_all_models(binned[0], binned[1], weight=1.0 / np.clip(binned[2], 1e-4, None) ** 2)
    print(f"  CSD model comparison: best={mc['best_model']}  "
          f"delta_AIC(exp vs best)={mc['delta_aic_exponential_vs_best']:.1f}")
    for r in mc["ranked_by_aic"]:
        m = mc["models"][r]
        print(f"    {r:14s}  R2={m['R2']:.3f}  AIC={m['AIC']:.1f}  BIC={m['BIC']:.1f}")

    # null controls
    nulls = {}
    all_null_r2 = []
    for nm, fn in (("temporal_shuffle", ts.null_temporal_shuffle),
                   ("spatial_shuffle", ts.null_spatial_shuffle),
                   ("phase_randomise", ts.null_phase_randomise)):
        seeds = RNG.integers(0, 2**31, size=N_NULL)
        results = Parallel(n_jobs=-1)(
            delayed(pooled_csd_fit)(days, transform=fn, dmax=dmax, seed=s) for s in seeds
        )
        r2s = np.array([res[0]["R2"] for res in results], float)
        lams = np.array([res[0]["lam"] for res in results], float)
        all_null_r2.append(r2s)
        p = float(np.mean(r2s >= fit["R2"]))
        nulls[nm] = {"R2_mean": float(np.nanmean(r2s)), "R2_p95": float(np.nanpercentile(r2s, 95)),
                     "R2_max": float(np.nanmax(r2s)), "lam_median": float(np.nanmedian(lams)),
                     "perm_p": p}
        print(f"  CSD null {nm:18s}: R2 mean {nulls[nm]['R2_mean']:+.3f}  "
              f"perm-p {p:.3f}")
    null_pool = np.concatenate(all_null_r2)
    null_sep = (fit["R2"] - np.nanmean(null_pool)) / (np.nanstd(null_pool) + 1e-9)

    return {
        "label": label, "product": product, "n_days": len(days), "n_stations": len(n_st),
        "lambda_km": fit["lam"], "sigma_lambda_km": fit["sigma_lam"], "R2": fit["R2"],
        "A": fit["A"], "C0": fit["C0"], "n_pair_days": fit["n_pairs"], "n_bins": fit["bins"],
        "nulls": nulls, "null_separation_sigma": float(null_sep),
        "model_comparison": mc,
        "binned": {"dist_km": binned[0].tolist(), "coh": binned[1].tolist(),
                   "sem": binned[2].tolist(), "count": binned[3].tolist()},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global_days = [f"2024{d:03d}" for d in range(100, 107)]
    result = {
        "test": "csd_crosscheck",
        "description": "Phase-coherent CSD method (Papers 1-2 signature) vs Pearson (step_10)",
        "global_broadcast": run_arm("broadcast", global_days, dmax=17000.0, label="global_broadcast_csd"),
        "timestamp": datetime.now().isoformat(),
    }
    (OUT / "step_15_csd_crosscheck.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved {OUT / 'step_15_csd_crosscheck.json'}")


if __name__ == "__main__":
    main()
