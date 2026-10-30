#!/usr/bin/env python3
"""
STEP 148  --  Finer frequency bands using all Paper 1 measured lambdas.

Paper 1 measured 9 frequency bands within the TEP range (10-500 uHz),
each with its own spatial correlation length.  The 3-band split (Step 134)
coarsely averaged these into 3 groups.  This step uses the FULL 9-band
structure:

  tidal_diurnal:      10-20 uHz,   lambda=3920 km  (IGS combined)
  tidal_semidiurnal:  20-30 uHz,   lambda=4701 km
  post_tidal_30_40:   30-40 uHz,   lambda=2453 km
  post_tidal_40_50:   40-50 uHz,   lambda=1410 km
  post_tidal_50_75:   50-75 uHz,   lambda=1192 km
  post_tidal_75_100:  75-100 uHz,  lambda=1067 km
  intermediate_100_200: 100-200 uHz, lambda=1260 km
  intermediate_200_350: 200-350 uHz, lambda=1383 km
  intermediate_350_500: 350-500 uHz, lambda=1432 km

The finer bands should capture the frequency-dependent spatial structure
more precisely than the 3-band split, potentially giving a stronger
predictive result.

NOTE: Some bands may have too few Fourier bins in a 24h/30s window
(resolution ~11.57 uHz).  Bands narrower than ~12 uHz (2 bins) are
merged with neighbours to maintain stability.

Protocol: 2025 held-out data, same as Step 134.

Output: results/outputs/step_148_finer_bands.json
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

OUT = ROOT / "results" / "outputs" / "step_148_finer_bands.json"
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
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260917

# Paper 1 IGS combined per-band lambdas (9 bands within TEP range)
# Bands narrower than ~12 uHz get min_bins=2
FINER_BANDS = [
    {"name": "tidal_diurnal",        "f_lo": 10e-6,   "f_hi": 20e-6,  "lambda_km": 3920.0, "min_bins": 2},
    {"name": "tidal_semidiurnal",    "f_lo": 20e-6,   "f_hi": 30e-6,  "lambda_km": 4701.0, "min_bins": 2},
    {"name": "post_tidal_30_40",     "f_lo": 30e-6,   "f_hi": 40e-6,  "lambda_km": 2453.0, "min_bins": 2},
    {"name": "post_tidal_40_50",     "f_lo": 40e-6,   "f_hi": 50e-6,  "lambda_km": 1410.0, "min_bins": 2},
    {"name": "post_tidal_50_75",     "f_lo": 50e-6,   "f_hi": 75e-6,  "lambda_km": 1192.0, "min_bins": 3},
    {"name": "post_tidal_75_100",    "f_lo": 75e-6,   "f_hi": 100e-6, "lambda_km": 1067.0, "min_bins": 3},
    {"name": "intermediate_100_200", "f_lo": 100e-6,  "f_hi": 200e-6, "lambda_km": 1260.0, "min_bins": 9},
    {"name": "intermediate_200_350", "f_lo": 200e-6,  "f_hi": 350e-6, "lambda_km": 1383.0, "min_bins": 13},
    {"name": "intermediate_350_500", "f_lo": 350e-6,  "f_hi": 500e-6, "lambda_km": 1432.0, "min_bins": 13},
]

# Also keep the 3-band spec for comparison
BANDS_3 = [
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

def _band_phase_grams(x, dt_s, bands):
    grams = []
    for band in bands:
        try:
            r = spectral_phase_gram(x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                    min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append((r["predictive_gram"], band))
        except ValueError: continue
    return grams if grams else None

def _prior_gram_bands(prev_day, cur_names, bands):
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
    bg = _band_phase_grams(np.column_stack(spectral_cols), s64.DT_S, bands)
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
    print("Step 148: Finer frequency bands (9-band)", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    for band in FINER_BANDS:
        print(f"  {band['name']:28s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t) and _input_path(s64._prev(t))]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        OUT.write_text(json.dumps({"step": 148, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in ("full_band_rms", "multi3_rms", "multi9_rms", "emp_rms",
                                "multi9_vs_full", "multi9_vs_emp", "multi9_vs_multi3")}

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None: continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur: continue

        pg_full = s64.prior_gram(prev, names)
        pg_3 = _prior_gram_bands(prev, names, BANDS_3)
        pg_9 = _prior_gram_bands(prev, names, FINER_BANDS)
        if pg_full is None or pg_3 is None or pg_9 is None: continue

        gnames_f, G_full, P_full = pg_full
        gnames_3, band_grams_3 = pg_3
        gnames_9, band_grams_9 = pg_9
        common = sorted(set(gnames_f) & set(gnames_3) & set(gnames_9))
        if len(common) < s64.MIN_REFS + 2: continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_3 = [gnames_3.index(n) for n in common]
        idx_9 = [gnames_9.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]
        G_full = G_full[np.ix_(idx_f, idx_f)]
        bg3 = [(g[np.ix_(idx_3, idx_3)], b) for g, b in band_grams_3]
        bg9 = [(g[np.ix_(idx_9, idx_9)], b) for g, b in band_grams_9]
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
        bg3 = [(g[np.ix_(keep, keep)], b) for g, b in bg3]
        bg9 = [(g[np.ix_(keep, keep)], b) for g, b in bg9]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "multi3", "multi9", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS: continue

            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(rho_rr_f * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError: continue

            def _band_weights(bg_list):
                ws = []
                for G_band, band_info in bg_list:
                    scales = _multi_scales(band_info["lambda_km"])
                    rho_rr = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                    rho_rt = _rho_multi(Dg[b, pool], scales)
                    try:
                        w = _reg_solve(rho_rr * G_band[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                       rho_rt * G_band[b, pool])
                        ws.append(w)
                    except np.linalg.LinAlgError: ws.append(None)
                return ws

            w3 = _band_weights(bg3)
            w9 = _band_weights(bg9)

            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu_p, ju_p = np.triu_indices(len(pool), 1)
            dv = Dg[np.ix_(pool, pool)][iu_p, ju_p]; cv = corr_emp[iu_p, ju_p]
            best_e = (np.inf, 4200.0)
            for Le in (200, 400, 800, 1500, 3000, 6000, 12000, 25000):
                pr = np.exp(-0.5 * (dv / Le) ** 2)
                sse = np.sum((cv - pr) ** 2)
                if sse < best_e[0]: best_e = (sse, Le)
            Le = best_e[1]
            try:
                w_emp = _reg_solve(np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / Le) ** 2) + s64.NUGGET * np.eye(len(pool)),
                                   np.exp(-0.5 * (Dg[b, pool] / Le) ** 2))
            except np.linalg.LinAlgError: w_emp = w_full

            B = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth): continue
                rr = A[ei]
                if not np.isfinite(rr).all(): continue
                p_full = float(w_full @ rr)
                bv3 = [float(w @ rr) for w in w3 if w is not None]
                p3 = float(np.mean(bv3)) if bv3 else p_full
                bv9 = [float(w @ rr) for w in w9 if w is not None]
                p9 = float(np.mean(bv9)) if bv9 else p_full
                p_emp = float(w_emp @ rr)
                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["multi3"].append((truth - p3) ** 2)
                day_sq["multi9"].append((truth - p9) ** 2)
                day_sq["emp"].append((truth - p_emp) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq): continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["multi3_rms"].append(rms["multi3"])
        per_day["multi9_rms"].append(rms["multi9"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["multi9_vs_full"].append(100.0 * (1.0 - rms["multi9"] / rms["full_band"]))
        per_day["multi9_vs_emp"].append(100.0 * (1.0 - rms["multi9"] / rms["emp"]))
        per_day["multi9_vs_multi3"].append(100.0 * (1.0 - rms["multi9"] / rms["multi3"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] full={np.median(per_day['full_band_rms']):.1f}, "
                  f"m3={np.median(per_day['multi3_rms']):.1f}, "
                  f"m9={np.median(per_day['multi9_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m", flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "multi9_vs_full_band_pct": _ci(per_day["multi9_vs_full"], rng),
        "multi9_vs_empirical_pct": _ci(per_day["multi9_vs_emp"], rng),
        "multi9_vs_multi3_pct": _ci(per_day["multi9_vs_multi3"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "multi3": float(np.median(per_day["multi3_rms"])) if n else None,
            "multi9": float(np.median(per_day["multi9_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interp = []
    mvf = results["multi9_vs_full_band_pct"]
    mve = results["multi9_vs_empirical_pct"]
    mv3 = results["multi9_vs_multi3_pct"]
    if mvf.get("median") is not None:
        interp.append(f"9-band vs full-band: median {mvf['median']:+.2f}%, CI95 [{mvf['ci95'][0]:+.2f}, {mvf['ci95'][1]:+.2f}].")
    if mve.get("median") is not None:
        interp.append(f"9-band vs empirical: median {mve['median']:+.2f}%, CI95 [{mve['ci95'][0]:+.2f}, {mve['ci95'][1]:+.2f}].")
    if mv3.get("median") is not None:
        interp.append(f"9-band vs 3-band: median {mv3['median']:+.2f}%, CI95 [{mv3['ci95'][0]:+.2f}, {mv3['ci95'][1]:+.2f}].")

    verdict = {
        "step": 148, "title": "Finer frequency bands (9-band)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR, "n_days_scored": n,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in FINER_BANDS],
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "The 9-band kernel should outperform the 3-band kernel by capturing the full frequency-dependent spatial structure measured by Paper 1.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for k, v in [("9vsFull", mvf), ("9vsEmp", mve), ("9vs3", mv3)]:
        if v.get("median") is not None: print(f"  {k}: {v['median']:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
