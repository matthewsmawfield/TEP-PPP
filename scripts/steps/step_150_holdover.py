#!/usr/bin/env python3
"""
STEP 150  --  Operational clock-prediction holdover test.

Tests the multi-band TEP kernel in an operational holdover scenario:
predict a station's clock over simulated outage gaps of 1h, 4h, and 8h,
and compare the holdover error against:
  1. Persistence (hold the last known clock value)
  2. Linear drift extrapolation (hold last known clock + drift)
  3. Single-lambda TEP kernel
  4. Multi-band TEP kernel (the claim)

This is the operational test that Step 145 was supposed to be, but uses
the clock product data directly (no step48_cache needed).  The holdover
scenario is:
  - At time t0, the target station "loses GNSS" (simulated outage)
  - The network stations continue to provide clock data
  - Predict the target's clock at t0 + gap using:
    a. Persistence: clock(t0)
    b. Drift: clock(t0) + drift * gap
    c. Single-lambda TEP: kriging from the network's clock changes
    d. Multi-band TEP: per-band kriging from the network's clock changes
  - Measure the holdover error: |predicted - actual|

The key operational metric is the holdover error growth rate: how much
does the error grow per hour of outage, for each predictor?

Protocol: 2025 held-out data, gaps = 1h, 4h, 8h.

Output: results/outputs/step_150_holdover.json
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib.tep_phase import spectral_phase_gram
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_150_holdover.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2025

EVALUATION_DOYS = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 40
REGISTERED_RNG_SEED = 20260919

GAPS_S = [3600, 14400, 28800]  # 1h, 4h, 8h
GAP_LABELS = ["1h", "4h", "8h"]

BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

def _multi_scales(lam): return (lam / 1.6, lam, lam * 1.6)
def _rho_multi(d, scales): return np.mean([np.exp(-d / L) for L in scales], axis=0)

def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6: K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)

def _input_path(tag):
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists(): return p
    return None

def _band_phase_grams(x, dt_s):
    grams = []
    for band in BANDS:
        try:
            r = spectral_phase_gram(x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                    min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append((r["predictive_gram"], band))
        except ValueError: continue
    return grams if grams else None

def _prior_gram_bands(prev_day, cur_names):
    pnames, pt, praw, _ = prev_day
    spectral_cols, kept = [], []
    for nm in cur_names:
        if nm not in pnames: continue
        y = praw[:, pnames.index(nm)].astype(float)
        good = np.isfinite(y)
        if good.mean() < 0.9 or good.sum() < 20: continue
        if not good.all():
            ix = np.arange(len(y))
            y = np.interp(ix, ix[good], y[good])
        xq = np.arange(len(y)) - len(y) / 2.0
        sy = y - np.polyval(np.polyfit(xq, y, 2), xq)
        if not np.isfinite(sy).all() or not (1e-4 < np.std(sy) < 1e4): continue
        spectral_cols.append(sy); kept.append(nm)
    if len(kept) < s64.MIN_REFS + 1: return None
    bg = _band_phase_grams(np.column_stack(spectral_cols), s64.DT_S)
    return (kept, bg) if bg is not None else None

def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4: return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)
    print("Step 150: Operational clock-prediction holdover test", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Gaps: {GAP_LABELS}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t) and _input_path(s64._prev(t))]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        OUT.write_text(json.dumps({"step": 150, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    dt_s = s64.DT_S
    npre = int(s64.PREWINDOW_S / dt_s)

    # Per-gap, per-method squared errors
    per_day = {gap: {m: [] for m in ("persist", "drift", "single", "multi")} for gap in GAP_LABELS}
    # Per-gap, per-method errors (for RMS)
    per_day_err = {gap: {m: [] for m in ("persist", "drift", "single", "multi")} for gap in GAP_LABELS}

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None: continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur: continue

        pg_full = s64.prior_gram(prev, names)
        pg_bands = _prior_gram_bands(prev, names)
        if pg_full is None or pg_bands is None: continue

        gnames_f, G_full, P_full = pg_full
        gnames_b, band_grams = pg_bands
        common = sorted(set(gnames_f) & set(gnames_b))
        if len(common) < s64.MIN_REFS + 2: continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_b = [gnames_b.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]
        G_full = G_full[np.ix_(idx_f, idx_f)]
        band_grams = [(g[np.ix_(idx_b, idx_b)], band) for g, band in band_grams]
        P = P_full[:, idx_f]; Dg = D[np.ix_(idx_cur, idx_cur)]; gnames = common

        # Build clock residual matrix (metres)
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            ts = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for ki in range(len(clk_ns)):
                tk = ts[ki]
                mask = (ts <= tk) & (ts >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(ts[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS

        P = P.astype(float, copy=True)
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all(): P[:, j] = np.interp(np.arange(P.shape[0]), np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0); med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames)) if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2: continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        band_grams = [(g[np.ix_(keep, keep)], band) for g, band in band_grams]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        n_t = R.shape[0]

        # For each gap, evaluate at regular intervals
        for gap_s, gap_label in zip(GAPS_S, GAP_LABELS):
            kstep = int(round(gap_s / dt_s))
            # Evaluate at every 10th epoch in the valid range
            eval_is = list(range(npre, n_t - kstep - 1, max(1, kstep // 4)))

            for b in range(ng):
                pool = [j for j in range(ng) if j != b]
                if len(pool) < s64.MIN_REFS: continue

                # Pre-compute weights (distance × Gram structure)
                rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
                rho_rt_f = s64._rho_multi(Dg[b, pool])
                try:
                    w_full = _reg_solve(rho_rr_f * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                        rho_rt_f * G_full[b, pool])
                except np.linalg.LinAlgError: continue

                band_weights = []
                for G_band, band_info in band_grams:
                    scales = _multi_scales(band_info["lambda_km"])
                    rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                    rho_rt_b = _rho_multi(Dg[b, pool], scales)
                    try:
                        w = _reg_solve(rho_rr_b * G_band[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                       rho_rt_b * G_band[b, pool])
                        band_weights.append(w)
                    except np.linalg.LinAlgError: band_weights.append(None)

                for ti in eval_is:
                    truth = R[ti + kstep, b]
                    if not np.isfinite(truth): continue
                    clk_t0 = R[ti, b]
                    if not np.isfinite(clk_t0): continue

                    # Network clock changes over the gap — CAUSAL:
                    # predict reference changes from past data using AR(1),
                    # not from future R[ti+kstep, j] which would be leakage.
                    net_changes = []
                    for j in pool:
                        a0 = R[ti, j]
                        if not np.isfinite(a0):
                            net_changes.append(np.nan)
                            continue
                        seg_j = R[max(0, ti - npre):ti + 1, j]
                        good = np.isfinite(seg_j)
                        if good.sum() >= 6:
                            xx = np.arange(good.sum(), dtype=float)
                            coef = np.polyfit(xx, seg_j[good], 1)
                            a1_pred = a0 + coef[0] * kstep
                        else:
                            a1_pred = a0  # persistence
                        net_changes.append(a1_pred - a0)
                    net_changes = np.array(net_changes)
                    if not np.isfinite(net_changes).all(): continue

                    # Persistence: hold the last value
                    p_persist = clk_t0
                    e_persist = abs(truth - p_persist)

                    # Drift extrapolation: use the pre-window drift
                    seg = R[max(0, ti - npre):ti, b]
                    if len(seg) >= 6 and np.isfinite(seg).all():
                        xx = np.arange(len(seg), dtype=float)
                        coef = np.polyfit(xx, seg, 1)
                        p_drift = clk_t0 + coef[0] * kstep
                    else:
                        p_drift = clk_t0
                    e_drift = abs(truth - p_drift)

                    # Single-lambda TEP: predict the clock change from network changes
                    p_single_change = float(w_full @ net_changes)
                    p_single = clk_t0 + p_single_change
                    e_single = abs(truth - p_single)

                    # Multi-band TEP: average per-band predictions of the clock change
                    band_preds = [float(w @ net_changes) for w in band_weights if w is not None]
                    if band_preds:
                        p_multi_change = float(np.mean(band_preds))
                        p_multi = clk_t0 + p_multi_change
                    else:
                        p_multi = p_single
                    e_multi = abs(truth - p_multi)

                    per_day_err[gap_label]["persist"].append(e_persist)
                    per_day_err[gap_label]["drift"].append(e_drift)
                    per_day_err[gap_label]["single"].append(e_single)
                    per_day_err[gap_label]["multi"].append(e_multi)
                    per_day[gap_label]["persist"].append(e_persist ** 2)
                    per_day[gap_label]["drift"].append(e_drift ** 2)
                    per_day[gap_label]["single"].append(e_single ** 2)
                    per_day[gap_label]["multi"].append(e_multi ** 2)

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] 1h multi={np.sqrt(np.mean(per_day['1h']['multi'][-100:])):.2f}, "
                  f"single={np.sqrt(np.mean(per_day['1h']['single'][-100:])):.2f}, "
                  f"persist={np.sqrt(np.mean(per_day['1h']['persist'][-100:])):.2f} m", flush=True)

    n = len(per_day["1h"]["multi"])
    print(f"\n  {n} samples scored", flush=True)

    results = {"n_samples": n, "by_gap": {}}
    for gap_label in GAP_LABELS:
        sq = per_day[gap_label]
        rms = {m: float(np.sqrt(np.mean(v))) if v else None for m, v in sq.items()}
        # Multi vs each baseline
        comparisons = {}
        for base in ("persist", "drift", "single"):
            if rms["multi"] and rms[base]:
                vals = [100.0 * (1.0 - a / b) for a, b in
                        zip(per_day_err[gap_label]["multi"], per_day_err[gap_label][base]) if b > 0.001]
                comparisons[f"multi_vs_{base}_pct"] = _ci(vals, rng)
        results["by_gap"][gap_label] = {
            "rms_m": rms,
            "median_error_m": {m: float(np.median(per_day_err[gap_label][m])) if per_day_err[gap_label][m] else None
                               for m in ("persist", "drift", "single", "multi")},
            "p95_error_m": {m: float(np.percentile(per_day_err[gap_label][m], 95)) if per_day_err[gap_label][m] else None
                            for m in ("persist", "drift", "single", "multi")},
            "comparisons": comparisons,
        }

    interp = []
    for gap_label in GAP_LABELS:
        r = results["by_gap"][gap_label]
        rms = r["rms_m"]
        interp.append(f"{gap_label}: RMS persist={rms['persist']:.3f}, drift={rms['drift']:.3f}, "
                       f"single={rms['single']:.3f}, multi={rms['multi']:.3f} m")
        for base in ("persist", "drift", "single"):
            c = r["comparisons"].get(f"multi_vs_{base}_pct")
            if c and c.get("median") is not None:
                interp.append(f"  multi vs {base}: {c['median']:+.2f}%, CI95 [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]")

    verdict = {
        "step": 150, "title": "Operational clock-prediction holdover test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR, "n_samples": n,
        "gaps": GAP_LABELS, "gaps_s": GAPS_S,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "The multi-band TEP kernel should produce smaller holdover error than persistence, drift extrapolation, and the single-lambda kernel, with the advantage growing for longer gaps.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for gap_label in GAP_LABELS:
        r = results["by_gap"][gap_label]
        rms = r["rms_m"]
        print(f"  {gap_label}: persist={rms['persist']:.3f}, drift={rms['drift']:.3f}, "
              f"single={rms['single']:.3f}, multi={rms['multi']:.3f} m", flush=True)

if __name__ == "__main__":
    main()
