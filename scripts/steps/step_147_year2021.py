#!/usr/bin/env python3
"""
STEP 147  --  2021 truly independent year replication.

The multi-band kernel was developed on 2024 and tested on 2025.
2021 is a TRULY independent year — never used for development, tuning,
or parameter selection.  If the multi-band advantage replicates on 2021,
it is not an artefact of the 2024-2025 calendar period.

Uses the frozen 3-band specification from Step 134:
  tidal:       10-35 uHz,  lambda=4677 km
  post_tidal:  35-100 uHz, lambda=1502 km
  intermediate: 100-500 uHz, lambda=1459 km

Output: results/outputs/step_147_year2021.json
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

OUT = ROOT / "results" / "outputs" / "step_147_year2021.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2021

EVALUATION_DOYS = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 30
REGISTERED_RNG_SEED = 20260916

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
    print(f"Step 147: 2021 truly independent year replication", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t) and _input_path(s64._prev(t))]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 147, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in ("full_band_rms", "multi_band_rms", "emp_rms",
                                "multi_vs_full", "multi_vs_emp")}

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
        day_sq = {k: [] for k in ("full_band", "multi_band", "emp")}
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
                bv = [float(w @ rr) for w in band_weights if w is not None]
                p_multi = float(np.mean(bv)) if bv else p_full
                p_emp = float(w_emp @ rr)
                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["multi_band"].append((truth - p_multi) ** 2)
                day_sq["emp"].append((truth - p_emp) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq): continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["multi_band_rms"].append(rms["multi_band"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["multi_vs_full"].append(100.0 * (1.0 - rms["multi_band"] / rms["full_band"]))
        per_day["multi_vs_emp"].append(100.0 * (1.0 - rms["multi_band"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] full={np.median(per_day['full_band_rms']):.1f}, "
                  f"multi={np.median(per_day['multi_band_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m", flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "multi_band_vs_full_band_pct": _ci(per_day["multi_vs_full"], rng),
        "multi_band_vs_empirical_pct": _ci(per_day["multi_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "multi_band": float(np.median(per_day["multi_band_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interp = []
    mvf = results["multi_band_vs_full_band_pct"]
    mve = results["multi_band_vs_empirical_pct"]
    if mvf.get("median") is not None:
        interp.append(f"Multi-band vs full-band on 2021: median {mvf['median']:+.2f}%, CI95 [{mvf['ci95'][0]:+.2f}, {mvf['ci95'][1]:+.2f}].")
    if mve.get("median") is not None:
        interp.append(f"Multi-band vs empirical on 2021: median {mve['median']:+.2f}%, CI95 [{mve['ci95'][0]:+.2f}, {mve['ci95'][1]:+.2f}].")
        if mve["median"] > 0 and mve["ci95"][0] > 0:
            interp.append("The multi-band kernel beats empirical covariance on the truly independent 2021 year with a positive CI. This is the strongest replication test: 2021 was never used for development or tuning.")
        elif mve["median"] > 0:
            interp.append("The multi-band kernel is better than empirical covariance on 2021 but the CI includes zero.")
        else:
            interp.append("The multi-band kernel does NOT beat empirical covariance on 2021.")

    verdict = {
        "step": 147, "title": "2021 truly independent year replication",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR, "n_days_scored": n,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "The multi-band advantage should replicate on 2021, a year never used for development or tuning.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    if mvf.get("median") is not None: print(f"  vs full: {mvf['median']:+.2f}%", flush=True)
    if mve.get("median") is not None: print(f"  vs emp:  {mve['median']:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
