#!/usr/bin/env python3
"""
STEP 154  --  Holdover with AR(1)/AR(2) baselines.

Step 150 showed the multi-band TEP kernel beats single-lambda by +7%
in holdover, but persistence was slightly better than multi-band.
Persistence is a trivial baseline.  The real operational bar is AR(1)
and AR(2) — the standard deployable clock predictors used in GNSS
timing receivers.

This step adds AR(1) and AR(2) predictors fitted on the pre-window
data and extrapolated across the outage gap.  The multi-band TEP
kernel should beat these if it provides genuine operational value.

Also adds a "network AR(1)" baseline: predict the target's clock
change from the network's AR(1)-predicted clock change, kriged by
distance.  This tests whether the TEP network prior adds value beyond
what a simple network-aided AR(1) can provide.

Protocol: 2025 held-out data, gaps = 1h, 4h, 8h.

Output: results/outputs/step_154_holdover_ar.py
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

OUT = ROOT / "results" / "outputs" / "step_154_holdover_ar.json"
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
REGISTERED_RNG_SEED = 20260922

GAPS_S = [3600, 14400, 28800]
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

def _ar_predict(series, order, n_steps):
    """Fit AR(order) on the series and predict n_steps ahead.

    Uses the analytical k-step prediction formula to avoid recursive
    blow-up when AR coefficients are near or above 1.
    """
    n = len(series)
    if n < order + 5:
        return series[-1]  # fallback to persistence
    # Fit AR(order) by least squares: y[t] = c + phi_1*y[t-1] + ... + phi_order*y[t-order]
    Y = series[order:]
    X = np.column_stack([series[order - k - 1:n - k - 1] for k in range(order)])
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    try:
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return series[-1]
    c = coef[0]  # intercept
    phis = coef[1:]  # AR coefficients
    # Clip coefficients to prevent blow-up (|phi| <= 0.99)
    phis = np.clip(phis, -0.99, 0.99)
    mean = c / (1.0 - np.sum(phis)) if abs(np.sum(phis)) < 0.99 else float(np.mean(series))
    last = float(series[-1])
    if order == 1:
        # AR(1): p(k) = mean + phi^k * (last - mean)
        return float(mean + phis[0] ** n_steps * (last - mean))
    elif order == 2:
        # AR(2): use companion-matrix eigen-decomposition for the k-step
        # forecast.  Clip eigenvalues to the unit circle (|lambda| <= 0.99)
        # rather than scaling the matrix, which would break the companion
        # structure.  p(k) - mean = A^k @ (state - mean).
        prev = float(series[-2])
        A = np.array([[phis[0], phis[1]], [1.0, 0.0]])
        eigvals, eigvecs = np.linalg.eig(A)
        # Clip eigenvalues to ensure decay (stationarity)
        radii = np.abs(eigvals)
        eigvals = np.where(radii > 0.99, eigvals * (0.99 / radii), eigvals)
        state = np.array([last - mean, prev - mean])
        # A^k @ state via eigen-decomposition
        coeffs = np.linalg.solve(eigvecs, state)
        result = eigvecs @ (eigvals ** n_steps * coeffs)
        return float(mean + np.real(result[0]))
    else:
        return last  # fallback

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
    print("Step 154: Holdover with AR(1)/AR(2) baselines", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Gaps: {GAP_LABELS}", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t) and _input_path(s64._prev(t))]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        OUT.write_text(json.dumps({"step": 154, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    dt_s = s64.DT_S
    npre = int(s64.PREWINDOW_S / dt_s)

    methods = ("persist", "drift", "ar1", "ar2", "single", "multi", "net_ar1")
    per_day = {gap: {m: [] for m in methods} for gap in GAP_LABELS}
    per_day_err = {gap: {m: [] for m in methods} for gap in GAP_LABELS}

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

        for gap_s, gap_label in zip(GAPS_S, GAP_LABELS):
            kstep = int(round(gap_s / dt_s))
            eval_is = list(range(npre, n_t - kstep - 1, max(1, kstep // 4)))

            for b in range(ng):
                pool = [j for j in range(ng) if j != b]
                if len(pool) < s64.MIN_REFS: continue

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

                # Distance weights for network AR(1)
                rho_rr_net = s64._rho_multi(Dg[np.ix_(pool, pool)])
                rho_rt_net = s64._rho_multi(Dg[b, pool])

                for ti in eval_is:
                    truth = R[ti + kstep, b]
                    if not np.isfinite(truth): continue
                    clk_t0 = R[ti, b]
                    if not np.isfinite(clk_t0): continue

                    # AR(1)-predicted changes for each reference station (causal:
                    # uses only R[ti-npre:ti+1, j], not the future R[ti+kstep, j]).
                    # In operational holdover the target loses its fix but the
                    # CORS network keeps running; however the TEP spatial
                    # predictor must forecast from PAST reference data, not from
                    # oracle future values.  Using R[ti+kstep,j] would make this
                    # a gap-fill / nowcast, not a holdover test.
                    net_changes = []
                    for j in pool:
                        seg_j = R[max(0, ti - npre):ti + 1, j]
                        if len(seg_j) >= 10 and np.isfinite(seg_j).all():
                            p_j = _ar_predict(seg_j, 1, kstep)
                            net_changes.append(p_j - R[ti, j])
                        else:
                            net_changes.append(np.nan)
                    net_changes = np.array(net_changes)
                    if not np.isfinite(net_changes).all(): continue

                    # Persistence
                    p_persist = clk_t0
                    e_persist = abs(truth - p_persist)

                    # Drift extrapolation
                    seg = R[max(0, ti - npre):ti, b]
                    if len(seg) >= 6 and np.isfinite(seg).all():
                        xx = np.arange(len(seg), dtype=float)
                        coef = np.polyfit(xx, seg, 1)
                        p_drift = clk_t0 + coef[0] * kstep
                    else:
                        p_drift = clk_t0
                    e_drift = abs(truth - p_drift)

                    # AR(1) on the target's own history
                    seg_ar = R[max(0, ti - npre):ti + 1, b]
                    if len(seg_ar) >= 10 and np.isfinite(seg_ar).all():
                        p_ar1 = _ar_predict(seg_ar, 1, kstep)
                    else:
                        p_ar1 = clk_t0
                    e_ar1 = abs(truth - p_ar1)

                    # AR(2) on the target's own history
                    if len(seg_ar) >= 15 and np.isfinite(seg_ar).all():
                        p_ar2 = _ar_predict(seg_ar, 2, kstep)
                    else:
                        p_ar2 = p_ar1
                    e_ar2 = abs(truth - p_ar2)

                    # Single-lambda TEP
                    p_single_change = float(w_full @ net_changes)
                    p_single = clk_t0 + p_single_change
                    e_single = abs(truth - p_single)

                    # Multi-band TEP
                    band_preds = [float(w @ net_changes) for w in band_weights if w is not None]
                    if band_preds:
                        p_multi_change = float(np.mean(band_preds))
                        p_multi = clk_t0 + p_multi_change
                    else:
                        p_multi = p_single
                    e_multi = abs(truth - p_multi)

                    # Network AR(1): krig the AR(1)-predicted reference changes
                    # to target using distance weights only (no phase Gram).
                    # net_changes already holds the AR(1)-predicted changes.
                    if np.isfinite(net_changes).all():
                        try:
                            w_net = _reg_solve(rho_rr_net + s64.NUGGET * np.eye(len(pool)),
                                               rho_rt_net)
                            p_net_ar1_change = float(w_net @ net_changes)
                            p_net_ar1 = clk_t0 + p_net_ar1_change
                        except np.linalg.LinAlgError:
                            p_net_ar1 = p_single
                    else:
                        p_net_ar1 = p_single
                    e_net_ar1 = abs(truth - p_net_ar1)

                    for m, e in [("persist", e_persist), ("drift", e_drift),
                                 ("ar1", e_ar1), ("ar2", e_ar2),
                                 ("single", e_single), ("multi", e_multi),
                                 ("net_ar1", e_net_ar1)]:
                        # Clamp to prevent overflow in e**2
                        e = min(abs(e), 1e6) if np.isfinite(e) else 1e6
                        per_day_err[gap_label][m].append(e)
                        per_day[gap_label][m].append(e ** 2)

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] 1h: multi={np.sqrt(np.mean(per_day['1h']['multi'][-100:])):.2f}, "
                  f"ar1={np.sqrt(np.mean(per_day['1h']['ar1'][-100:])):.2f}, "
                  f"ar2={np.sqrt(np.mean(per_day['1h']['ar2'][-100:])):.2f} m", flush=True)

    n = len(per_day["1h"]["multi"])
    print(f"\n  {n} samples scored", flush=True)

    results = {"n_samples": n, "by_gap": {}}
    for gap_label in GAP_LABELS:
        sq = per_day[gap_label]
        rms = {m: float(np.sqrt(np.mean(v))) if v else None for m, v in sq.items()}
        comparisons = {}
        for base in methods:
            if rms["multi"] and rms[base] and base != "multi":
                vals = [100.0 * (1.0 - a / b) for a, b in
                        zip(per_day_err[gap_label]["multi"], per_day_err[gap_label][base]) if b > 0.001]
                comparisons[f"multi_vs_{base}_pct"] = _ci(vals, rng)
        results["by_gap"][gap_label] = {
            "rms_m": rms,
            "median_error_m": {m: float(np.median(per_day_err[gap_label][m])) if per_day_err[gap_label][m] else None
                               for m in methods},
            "comparisons": comparisons,
        }

    interp = []
    for gap_label in GAP_LABELS:
        r = results["by_gap"][gap_label]
        rms = r["rms_m"]
        interp.append(f"{gap_label}: RMS persist={rms['persist']:.3f}, drift={rms['drift']:.3f}, "
                       f"AR1={rms['ar1']:.3f}, AR2={rms['ar2']:.3f}, "
                       f"single={rms['single']:.3f}, multi={rms['multi']:.3f}, "
                       f"net_AR1={rms['net_ar1']:.3f} m")
        for base in ("persist", "drift", "ar1", "ar2", "single", "net_ar1"):
            c = r["comparisons"].get(f"multi_vs_{base}_pct")
            if c and c.get("median") is not None:
                interp.append(f"  multi vs {base}: {c['median']:+.2f}%, CI95 [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]")

    verdict = {
        "step": 154, "title": "Holdover with AR(1)/AR(2) baselines",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR, "n_samples": n,
        "gaps": GAP_LABELS, "gaps_s": GAPS_S,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "The multi-band TEP kernel should beat AR(1) and AR(2) in holdover, especially for longer gaps where the network's common-mode structure provides information that the target's own history cannot.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for gap_label in GAP_LABELS:
        r = results["by_gap"][gap_label]
        rms = r["rms_m"]
        print(f"  {gap_label}: AR1={rms['ar1']:.3f}, AR2={rms['ar2']:.3f}, "
              f"multi={rms['multi']:.3f}, net_AR1={rms['net_ar1']:.3f} m", flush=True)

if __name__ == "__main__":
    main()
