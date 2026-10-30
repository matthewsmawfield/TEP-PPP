#!/usr/bin/env python3
"""
STEP 142  --  Multi-year replication of the multi-band TEP prior.

The multi-band kernel (Step 134) beat empirical covariance by +0.56% on
2025 independent-year data.  This step replicates on 2024 development
data and tests whether the advantage is stable across years.

PROTOCOL:
  * Run the multi-band P1 gate on:
    1. 2024 development data (30 days)
    2. 2025 independent-year data (57 days)
  * Compare multi-band vs full-band and vs empirical covariance for each
    year.
  * The advantage should be present in both years.

Output: results/outputs/step_142_multi_year.json
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

OUT = ROOT / "results" / "outputs" / "step_142_multi_year.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
REGISTERED_RNG_SEED = 20260902
N_DAYS_PER_YEAR = 30

BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _band_phase_grams(x, dt_s):
    grams = []
    for band in BANDS:
        try:
            result = spectral_phase_gram(
                x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append(result["predictive_gram"])
        except ValueError:
            grams.append(None)
    return grams


def _prior_gram_bands(prev_day, cur_names):
    pnames, pt, praw, _ = prev_day
    spectral_cols, clean_raw_cols, kept = [], [], []
    for nm in cur_names:
        if nm not in pnames:
            continue
        y = praw[:, pnames.index(nm)].astype(float)
        good = np.isfinite(y)
        if good.mean() < 0.9 or good.sum() < 20:
            continue
        if not good.all():
            ix = np.arange(len(y))
            y = np.interp(ix, ix[good], y[good])
        clean_raw = y.copy()
        xq = np.arange(len(clean_raw)) - len(clean_raw) / 2.0
        spectral_y = clean_raw - np.polyval(np.polyfit(xq, clean_raw, 2), xq)
        if not np.isfinite(spectral_y).all() or not (1e-4 < np.std(spectral_y) < 1e4):
            continue
        spectral_cols.append(spectral_y)
        clean_raw_cols.append(clean_raw)
        kept.append(nm)
    if len(kept) < s64.MIN_REFS + 1:
        return None
    band_grams = _band_phase_grams(np.column_stack(spectral_cols), s64.DT_S)
    if all(g is None for g in band_grams):
        return None
    return kept, band_grams, np.column_stack(clean_raw_cols)


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def _run_year(year, n_days, rng):
    dates = td.available_dates(PRODUCT, year)
    if not dates:
        return {"n_days": 0, "error": "no data"}
    days = [dates[i] for i in
            np.linspace(1, len(dates) - 1, min(n_days, len(dates))).astype(int)]

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_rms", "multi_rms", "emp_rms",
        "multi_vs_full", "multi_vs_emp")}

    for k, tag in enumerate(days):
        prev_tag = s64._prev(tag)
        if _input_path(tag) is None or _input_path(prev_tag) is None:
            continue
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(prev_tag, product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                               max_jumps=10_000)
        if not s_cur:
            continue

        pg_full = s64.prior_gram(prev, names)
        pg_bands = _prior_gram_bands(prev, names)
        if pg_full is None or pg_bands is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_b, band_grams_raw, _ = pg_bands

        common = sorted(set(gnames_f) & set(gnames_b))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_b = [gnames_b.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        band_grams = [g[np.ix_(idx_b, idx_b)] if g is not None else None
                      for g in band_grams_raw]
        P = P_full[:, idx_f]
        Dg = D[np.ix_(idx_cur, idx_cur)]
        gnames = common

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
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]),
                                    np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        band_grams = [g[np.ix_(keep, keep)] if g is not None else None
                      for g in band_grams]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full", "multi", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            band_ws = []
            for bi, G_band in enumerate(band_grams):
                if G_band is None:
                    band_ws.append(None)
                    continue
                scales = _multi_scales(BANDS[bi]["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    band_ws.append(w_band)
                except np.linalg.LinAlgError:
                    band_ws.append(None)

            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu_p, ju_p = np.triu_indices(len(pool), 1)
            dv = Dg[np.ix_(pool, pool)][iu_p, ju_p]
            cv = corr_emp[iu_p, ju_p]
            best_e = (np.inf, 4200.0)
            for Le in (200, 400, 800, 1500, 3000, 6000, 12000, 25000):
                pr = np.exp(-0.5 * (dv / Le) ** 2)
                sse = np.sum((cv - pr) ** 2)
                if sse < best_e[0]:
                    best_e = (sse, Le)
            Le = best_e[1]
            ce_rr = np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / Le) ** 2)
            ce_rt = np.exp(-0.5 * (Dg[b, pool] / Le) ** 2)
            try:
                w_emp = _reg_solve(ce_rr + s64.NUGGET * np.eye(len(pool)), ce_rt)
            except np.linalg.LinAlgError:
                w_emp = w_full

            B = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                p_full = float(w_full @ rr)
                p_emp = float(w_emp @ rr)
                band_preds = [float(w @ rr) for w in band_ws if w is not None]
                p_multi = float(np.mean(band_preds)) if band_preds else p_full
                day_sq["full"].append((truth - p_full) ** 2)
                day_sq["multi"].append((truth - p_multi) ** 2)
                day_sq["emp"].append((truth - p_emp) ** 2)

        if any(len(day_sq[m]) < 30 for m in ("full", "multi", "emp")):
            continue
        rms = {m: float(np.sqrt(np.mean(day_sq[m])))
               for m in ("full", "multi", "emp")}
        per_day["full_rms"].append(rms["full"])
        per_day["multi_rms"].append(rms["multi"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["multi_vs_full"].append(100.0 * (1.0 - rms["multi"] / rms["full"]))
        per_day["multi_vs_emp"].append(100.0 * (1.0 - rms["multi"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{year} {k+1}/{len(days)}] "
                  f"full={np.median(per_day['full_rms']):.1f}, "
                  f"multi={np.median(per_day['multi_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_rms"])
    return {
        "n_days": n,
        "multi_vs_full_band_pct": _ci(per_day["multi_vs_full"], rng),
        "multi_vs_empirical_pct": _ci(per_day["multi_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_rms"])) if n else None,
            "multi_band": float(np.median(per_day["multi_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 142: Multi-year replication of multi-band TEP prior",
          flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print()

    all_results = {}
    for year in (2024, 2025):
        print(f"  Running {year}...", flush=True)
        all_results[str(year)] = _run_year(year, N_DAYS_PER_YEAR, rng)
        r = all_results[str(year)]
        mvf = r.get("multi_vs_full_band_pct", {})
        mve = r.get("multi_vs_empirical_pct", {})
        print(f"    {year}: n={r['n_days']}, "
              f"multi_vs_full={mvf.get('median')}, "
              f"multi_vs_emp={mve.get('median')}", flush=True)
        print()

    # Check consistency
    interpretations = []
    y24 = all_results.get("2024", {})
    y25 = all_results.get("2025", {})

    for yr, label in [("2024", "development"), ("2025", "independent-year")]:
        mvf = all_results[yr].get("multi_vs_full_band_pct", {})
        mve = all_results[yr].get("multi_vs_empirical_pct", {})
        if mvf.get("median") is not None:
            interpretations.append(
                f"{yr} ({label}): multi-band vs full-band "
                f"{mvf['median']:+.2f}% (CI95 [{mvf['ci95'][0]:+.2f}, "
                f"{mvf['ci95'][1]:+.2f}])")
        if mve.get("median") is not None:
            interpretations.append(
                f"{yr} ({label}): multi-band vs empirical "
                f"{mve['median']:+.2f}% (CI95 [{mve['ci95'][0]:+.2f}, "
                f"{mve['ci95'][1]:+.2f}])")

    # Consistency check
    mvf24 = y24.get("multi_vs_full_band_pct", {}).get("median")
    mvf25 = y25.get("multi_vs_full_band_pct", {}).get("median")
    if mvf24 is not None and mvf25 is not None:
        if mvf24 > 0 and mvf25 > 0:
            interpretations.append(
                f"The multi-band advantage is present in BOTH years "
                f"(2024: {mvf24:+.2f}%, 2025: {mvf25:+.2f}%). The frequency-"
                f"dependent lambdas from Paper 1 generalise across years.")
        else:
            interpretations.append(
                f"The multi-band advantage is NOT consistent across years "
                f"(2024: {mvf24:+.2f}%, 2025: {mvf25:+.2f}%).")

    mve24 = y24.get("multi_vs_empirical_pct", {}).get("median")
    mve25 = y25.get("multi_vs_empirical_pct", {}).get("median")
    if mve24 is not None and mve25 is not None:
        if mve24 > 0 and mve25 > 0:
            interpretations.append(
                f"The multi-band kernel beats empirical covariance in BOTH "
                f"years (2024: {mve24:+.2f}%, 2025: {mve25:+.2f}%). The "
                f"physics-fixed frequency-dependent lambdas generalise where "
                f"the data-driven covariance overfits, consistently across "
                f"years.")

    verdict = {
        "step": 142,
        "title": "Multi-year replication of multi-band TEP prior",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "bands": [{"name": b["name"], "lambda_km": b["lambda_km"]} for b in BANDS],
        "results": all_results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The multi-band kernel with Paper 1's frequency-dependent "
            "lambdas should beat both the full-band kernel and empirical "
            "covariance in both 2024 and 2025. The advantage should be "
            "stable across years, confirming that the physics-fixed "
            "frequency-dependent spatial scales generalise."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
