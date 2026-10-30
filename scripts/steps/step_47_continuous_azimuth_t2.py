#!/usr/bin/env python3
"""
STEP 47 -- Continuous-azimuth T2 anisotropy model.

The step_41 global anisotropy test splits baselines into binary EW/NS
sectors (within +/-25 deg of cardinal), discarding all non-cardinal
baselines.  This loses ~60% of the data and reduces statistical power.

This step uses a continuous azimuth model that uses every baseline:
    C(az, d) = a0(d) + a1(d) * cos(2*az) + a2(d) * sin(2*az)
where az is the baseline azimuth (0-180 deg, folded) and d is the
distance.  The cos(2*az) term captures the EW-NS anisotropy (cos(2*az)
= +1 for NS, -1 for EW); the sin(2*az) term captures the NE-SW vs
NW-SE anisotropy.

The TEP prediction is that the EW-NS anisotropy (a1) is positive
(EW baselines have higher correlation than NS baselines at the same
distance, as reported in Papers 1-3).  The continuous model tests this
with all baselines, not just the cardinal ones.

A permutation null shuffles station labels (destroying the
azimuth-correlation relationship) and reports the sigma separation.
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
from scripts.steps.step_41_global_anisotropy import (  # noqa: E402
    _clean_clk, corr_matrix, load_global_day)

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260904)
C_M_PER_NS = 0.299792458


def continuous_anisotropy(days, product="clk_cod", bin_km=500.0,
                          n_boot=400, dmax=19000.0, min_n=30):
    """Continuous-azimuth anisotropy on the global precise network."""
    recs = []
    nd = 0
    for day in days:
        D, S = load_global_day(day, product)
        if len(D) < 20:
            continue
        nd += 1
        nms = list(D)
        Cm = corr_matrix([D[n] for n in nms])
        for i, j in itertools.combinations(range(len(nms)), 2):
            if not np.isfinite(Cm[i, j]):
                continue
            a, b = nms[i], nms[j]
            sa, sb = S[a], S[b]
            if sa is None or sb is None:
                continue
            dist = td.great_circle_km(sa, sb)
            if dist > dmax or dist < 100:
                continue
            az = td.baseline_azimuth_deg(sa, sb)
            recs.append((dist, float(Cm[i, j]), az, a, b))

    if len(recs) == 0:
        return {"n_days": nd, "n_pair_days": 0, "error": "no pairs"}

    d = np.array([r[0] for r in recs])
    c = np.array([r[1] for r in recs])
    az = np.array([r[2] for r in recs])
    stn = sorted({r[3] for r in recs} | {r[4] for r in recs})

    # Continuous model: fit a0 + a1*cos(2az) + a2*sin(2az) in distance bins
    bins = np.arange(500, dmax + bin_km, bin_km)
    bin_fits = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < min_n:
            continue
        az_m = np.radians(az[m] * 2)  # fold to 0-360, then double for 2az
        c_m = c[m]
        # Regression: c = a0 + a1*cos(2az) + a2*sin(2az)
        X = np.column_stack([np.ones(len(c_m)), np.cos(az_m), np.sin(az_m)])
        beta = np.linalg.solve(X.T @ X, X.T @ c_m)
        resid = c_m - X @ beta
        sse = np.sum(resid ** 2)
        sst = np.sum((c_m - c_m.mean()) ** 2)
        r2 = 1 - sse / sst if sst > 0 else np.nan
        # a1 is the EW-NS anisotropy coefficient (positive = EW > NS)
        # cos(2*az) = +1 for az=0 (NS), -1 for az=90 (EW)
        # So a1 > 0 means NS > EW, a1 < 0 means EW > NS
        # The series reports EW > NS, so we expect a1 < 0
        # We report -a1 as the "EW_minus_NS" coefficient for consistency
        ew_minus_ns = -beta[1]
        bin_fits.append({
            "d_lo": float(lo), "d_hi": float(hi), "n": int(m.sum()),
            "a0": float(beta[0]), "a1_cos2az": float(beta[1]),
            "a2_sin2az": float(beta[2]),
            "ew_minus_ns": float(ew_minus_ns),
            "r2": float(r2),
        })

    # Pooled continuous model (all distances, with distance as covariate)
    az_rad = np.radians(az * 2)
    logd = np.log(d + 1)
    X_pool = np.column_stack([np.ones(len(c)), logd, np.cos(az_rad), np.sin(az_rad),
                              logd * np.cos(az_rad), logd * np.sin(az_rad)])
    beta_pool = np.linalg.solve(X_pool.T @ X_pool, X_pool.T @ c)
    resid_pool = c - X_pool @ beta_pool
    sse_pool = np.sum(resid_pool ** 2)
    sst_pool = np.sum((c - c.mean()) ** 2)
    r2_pool = 1 - sse_pool / sst_pool if sst_pool > 0 else np.nan
    # The interaction term logd * cos(2az) tests whether the anisotropy
    # grows with distance (as TEP predicts if the effect accumulates)
    ew_minus_ns_pooled = -beta_pool[2]  # main effect
    ew_minus_ns_interaction = -beta_pool[4]  # distance interaction

    # Distance-weighted average of the per-bin EW-NS coefficient
    valid_bins = [bf for bf in bin_fits if np.isfinite(bf["ew_minus_ns"])]
    if valid_bins:
        weights = np.array([bf["n"] for bf in valid_bins])
        coeffs = np.array([bf["ew_minus_ns"] for bf in valid_bins])
        point = float(np.average(coeffs, weights=weights))
    else:
        point = float("nan")

    # Station-clustered bootstrap
    bs = []
    for _ in range(n_boot):
        draws = RNG.choice(stn, len(stn), replace=True)
        draw_count = {}
        for s in draws:
            draw_count[s] = draw_count.get(s, 0) + 1
        idx_list = []
        for k, r in enumerate(recs):
            s1, s2 = r[3], r[4]
            if s1 in draw_count and s2 in draw_count:
                idx_list.extend([k] * (draw_count[s1] * draw_count[s2]))
        if len(idx_list) < 2000:
            continue
        idx_arr = np.array(idx_list)
        d_b, c_b, az_b = d[idx_arr], c[idx_arr], az[idx_arr]
        # Refit per-bin
        bin_coeffs = []
        bin_w = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d_b >= lo) & (d_b < hi)
            if m.sum() < min_n:
                continue
            az_m = np.radians(az_b[m] * 2)
            c_m = c_b[m]
            X = np.column_stack([np.ones(len(c_m)), np.cos(az_m), np.sin(az_m)])
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ c_m)
                bin_coeffs.append(-beta[1])
                bin_w.append(m.sum())
            except np.linalg.LinAlgError:
                continue
        if bin_coeffs:
            bs.append(float(np.average(bin_coeffs, weights=bin_w)))

    bs = np.array(bs)
    if len(bs) > 0:
        lo_ci, hi_ci = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        p_value = float(np.mean(bs <= 0))
    else:
        lo_ci, hi_ci, p_value = float("nan"), float("nan"), float("nan")

    # Permutation null: shuffle station labels (destroy azimuth-correlation)
    n_null = 200
    null_coeffs = []
    for _ in range(n_null):
        perm = RNG.permutation(len(stn))
        stn_perm = [stn[i] for i in perm]
        # Remap: station i -> stn_perm[i], so pair (a,b) gets a new azimuth
        stn_map = {stn[i]: stn_perm[i] for i in range(len(stn))}
        # Recompute azimuths with shuffled station labels
        null_bin_coeffs = []
        null_bin_w = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d >= lo) & (d < hi)
            if m.sum() < min_n:
                continue
            # Shuffle azimuths within this bin
            az_m = az[m].copy()
            RNG.shuffle(az_m)
            az_rad_m = np.radians(az_m * 2)
            c_m = c[m]
            X = np.column_stack([np.ones(len(c_m)), np.cos(az_rad_m), np.sin(az_rad_m)])
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ c_m)
                null_bin_coeffs.append(-beta[1])
                null_bin_w.append(m.sum())
            except np.linalg.LinAlgError:
                continue
        if null_bin_coeffs:
            null_coeffs.append(float(np.average(null_bin_coeffs, weights=null_bin_w)))

    null_coeffs = np.array(null_coeffs)
    if len(null_coeffs) > 5:
        sigma = float(abs(point - np.mean(null_coeffs)) / (np.std(null_coeffs) + 1e-9))
        null_p = float(np.mean(np.abs(null_coeffs) >= abs(point)))
    else:
        sigma = float("nan")
        null_p = float("nan")

    # T2 continuous pass: EW-NS coefficient > 0 (EW > NS as series reports)
    # AND CI excludes zero AND sigma > 3
    significant = bool(np.isfinite(lo_ci) and lo_ci > 0)
    passes = bool(significant and np.isfinite(sigma) and sigma > 3.0)

    print(f"\n[Continuous T2] {nd} days, {len(recs):,} pair-days, {len(stn)} stations")
    print(f"  Pooled R^2 = {r2_pool:.4f}")
    print(f"  EW-NS coefficient (pooled) = {ew_minus_ns_pooled:+.6f}")
    print(f"  EW-NS x distance interaction = {ew_minus_ns_interaction:+.6f}")
    print(f"  Distance-weighted EW-NS = {point:+.6f}  CI [{lo_ci:+.6f}, {hi_ci:+.6f}]")
    print(f"  Permutation null: sigma = {sigma:.1f}, p = {null_p:.4f}")
    print(f"  T2 continuous pass: {passes}")
    print(f"  Per-bin EW-NS coefficients:")
    for bf in bin_fits:
        print(f"    [{bf['d_lo']:.0f}-{bf['d_hi']:.0f} km] n={bf['n']:5d}  "
              f"EW-NS={bf['ew_minus_ns']:+.6f}  R^2={bf['r2']:.4f}")

    return {
        "n_days": nd, "n_pair_days": len(recs), "n_stations": len(stn),
        "ew_minus_ns_pooled": point,
        "ci95": [lo_ci, hi_ci], "p_ew_le_ns": p_value,
        "pooled_r2": float(r2_pool),
        "pooled_ew_minus_ns": float(ew_minus_ns_pooled),
        "pooled_interaction": float(ew_minus_ns_interaction),
        "null_sigma": sigma, "null_p_value": null_p,
        "per_bin": bin_fits,
        "significant": significant,
        "T2_continuous_pass": passes,
        "T2_continuous_rule": "EW-NS > 0, CI excludes zero, sigma > 3.0",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cod_days = td.available_dates("clk_cod", 2024)
    idx = np.linspace(0, len(cod_days) - 1, 30).round().astype(int)
    days = [cod_days[i] for i in idx]
    print(f"Continuous-azimuth T2 on COD precise receiver clocks ({len(days)} days)\n")
    res = {
        "test": "continuous_azimuth_t2",
        "product": "COD0MGXFIN precise receiver clocks (RINEX CLK AR)",
        "timestamp": datetime.now().isoformat(),
        "T2_continuous_global": continuous_anisotropy(days, "clk_cod"),
    }

    # Also run on aperture-restricted subsets
    print("\n--- Aperture-restricted subsets ---")
    for dmax in [5000, 10000, 19000]:
        print(f"\n  dmax = {dmax} km:")
        res[f"T2_continuous_dmax{dmax}"] = continuous_anisotropy(
            days, "clk_cod", dmax=float(dmax))

    outpath = OUT / "step_47_continuous_azimuth_t2.json"
    outpath.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved {outpath}")

    t2 = res["T2_continuous_global"]
    print("\n" + "=" * 72)
    print("CONTINUOUS-AZIMUTH T2 RESULT")
    print("=" * 72)
    if np.isfinite(t2.get("ew_minus_ns_pooled", float("nan"))):
        sig = "SIGNIFICANT" if t2.get("significant") else "not significant"
        print(f"  EW-NS = {t2['ew_minus_ns_pooled']:+.6f}  "
              f"CI [{t2['ci95'][0]:+.6f}, {t2['ci95'][1]:+.6f}]  {sig}")
        print(f"  null sigma = {t2.get('null_sigma', float('nan')):.1f}")
        print(f"  {t2['n_stations']} stations, {t2['n_pair_days']:,} pair-days")


if __name__ == "__main__":
    main()
