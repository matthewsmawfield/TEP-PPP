#!/usr/bin/env python3
"""
STEP 134  --  Multi-band frequency-dependent TEP prior.

USING THE FULL GRAVITATIONAL STRUCTURE CORRECTLY.

Paper 1 (TEP-GNSS, Jaipur) measured the frequency-dependent correlation
length of the TEP field:

  * Tidal band    (10-30 uHz):  lambda = 4,677 km  (gravitational forcing)
  * Post-tidal    (30-100 uHz): lambda = 1,502 km  (transition)
  * Intermediate  (100-500 uHz): lambda = 1,459 km (local)

The full-band kernel (Step 64/92) uses a single compromise lambda
(4,200 km) for all frequencies.  This under-fits the continuum (which
needs ~1,500 km) and slightly mis-fits the tides (which need 4,677 km).

The correct approach is a multi-band kernel: split the spectrum into
frequency bands, compute a separate phase Gram for each band, use the
physics-fixed lambda for that band, and combine the band-level
predictions.  Each band gets its correct spatial scale.

This should give MORE predictive power than the single-lambda full-band
kernel because:
  1. The tidal frequencies get their proper long scale (4,677 km)
  2. The continuum frequencies get their proper shorter scale (~1,500 km)
  3. Each band's spatial structure is properly captured

PROTOCOL:
  * Run the P1 gate on 2025 held-out data.
  * Compare:
    1. full_band:    single lambda=4,200 km, full-band phase Gram
    2. multi_band:   per-band lambda (4677/1502/1459 km), per-band Grams
    3. empirical:    data-driven covariance

  * TEP prediction: multi_band beats full_band because it uses the correct
    physics-fixed scale for each frequency band.  It beats empirical
    covariance because the physics-fixed scales generalise where the
    data-driven covariance overfits.

Output: results/outputs/step_134_multiband_prior.json
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

OUT = ROOT / "results" / "outputs" / "step_134_multiband_prior.json"
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
REGISTERED_RNG_SEED = 20260902

# Paper 1 frequency-dependent lambdas
# Band boundaries adjusted for 24h window resolvability:
#   At 30s/24h, frequency resolution = 1/86400 ≈ 11.57 uHz
#   Tidal band needs at least 2 bins; start at 9.84 uHz (min_low_cycles=0.85)
BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

# Multi-scale expansion for each band
def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _band_phase_grams(x, dt_s):
    """Compute a phase Gram for each frequency band.

    Returns a list of (gram, band_info) or None if any band fails.
    Bands that fail are skipped; if all fail, returns None.
    """
    grams = []
    for band in BANDS:
        try:
            result = spectral_phase_gram(
                x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append((result["predictive_gram"], band))
        except ValueError:
            continue
    if not grams:
        return None
    return grams


def _prior_gram_bands(prev_day, cur_names):
    """Build per-band phase Grams from the prior day."""
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
    if band_grams is None:
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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 134: Multi-band frequency-dependent TEP prior", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, "
              f"lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 134, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "multi_band_rms", "emp_rms",
        "multi_band_vs_full", "multi_band_vs_emp")}
    total_multi_fallbacks = 0

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                               max_jumps=10_000)
        if not s_cur:
            continue

        # Full-band Gram (registered)
        pg_full = s64.prior_gram(prev, names)
        # Multi-band Grams
        pg_bands = _prior_gram_bands(prev, names)

        if pg_full is None or pg_bands is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_b, band_grams, _ = pg_bands

        common = sorted(set(gnames_f) & set(gnames_b))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_b = [gnames_b.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        band_grams = [(g[np.ix_(idx_b, idx_b)], band)
                      for g, band in band_grams]
        P = P_full[:, idx_f]
        Dg = D[np.ix_(idx_cur, idx_cur)]
        gnames = common

        # Causal detrend
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
        band_grams = [(g[np.ix_(keep, keep)], band)
                      for g, band in band_grams]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "multi_band", "emp")}
        multi_fallback_count = 0
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band TEP kernel (single lambda)
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Multi-band TEP kernel: per-band prediction, then average
            band_preds = []
            band_valid = 0
            for G_band, band_info in band_grams:
                scales = _multi_scales(band_info["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    band_valid += 1
                except np.linalg.LinAlgError:
                    continue

            # Empirical covariance
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

            # Pre-compute multi-band weights for this target
            band_weights = []
            for G_band, band_info in band_grams:
                scales = _multi_scales(band_info["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    band_weights.append(w_band)
                except np.linalg.LinAlgError:
                    band_weights.append(None)

            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue

                p_full = float(w_full @ rr)
                p_emp = float(w_emp @ rr)

                # Multi-band: average predictions from valid bands
                band_preds_vals = []
                for w_band in band_weights:
                    if w_band is not None:
                        band_preds_vals.append(float(w_band @ rr))
                if band_preds_vals:
                    p_multi = float(np.mean(band_preds_vals))
                else:
                    p_multi = p_full
                    multi_fallback_count += 1

                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["multi_band"].append((truth - p_multi) ** 2)
                day_sq["emp"].append((truth - p_emp) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        total_multi_fallbacks += multi_fallback_count
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["multi_band_rms"].append(rms["multi_band"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["multi_band_vs_full"].append(
            100.0 * (1.0 - rms["multi_band"] / rms["full_band"]))
        per_day["multi_band_vs_emp"].append(
            100.0 * (1.0 - rms["multi_band"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"multi={np.median(per_day['multi_band_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "multi_band_vs_full_band_pct": _ci(per_day["multi_band_vs_full"], rng),
        "multi_band_vs_empirical_pct": _ci(per_day["multi_band_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "multi_band": float(np.median(per_day["multi_band_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    mvf = results["multi_band_vs_full_band_pct"]
    mve = results["multi_band_vs_empirical_pct"]

    if mvf.get("median") is not None:
        if mvf["median"] > 0 and mvf["ci95"] and mvf["ci95"][0] > 0:
            interpretations.append(
                f"The multi-band frequency-dependent TEP kernel BEATS the "
                f"single-lambda full-band kernel on independent-year data "
                f"(median {mvf['median']:+.2f}%, "
                f"CI95 [{mvf['ci95'][0]:+.2f}, {mvf['ci95'][1]:+.2f}]). "
                f"Using the correct physics-fixed lambda for each frequency "
                f"band (tidal=4677 km, post-tidal=1502 km, intermediate="
                f"1459 km, from Paper 1) captures more predictive power than "
                f"a single compromise lambda.  The gravitational tidal band "
                f"gets its proper long scale and the continuum gets its "
                f"proper shorter scale.")
        elif mvf["median"] > 0:
            interpretations.append(
                f"The multi-band kernel is better than the full-band kernel "
                f"(median {mvf['median']:+.2f}%, CI includes zero).")
        else:
            interpretations.append(
                f"The multi-band kernel does not beat the full-band kernel "
                f"({mvf['median']:+.2f}%).  The simple averaging of band-level "
                f"predictions may not be the optimal fusion strategy.")

    if mve.get("median") is not None:
        if mve["median"] > 0 and mve["ci95"] and mve["ci95"][0] > 0:
            interpretations.append(
                f"The multi-band TEP kernel BEATS empirical covariance on "
                f"independent-year data (median {mve['median']:+.2f}%, "
                f"CI95 [{mve['ci95'][0]:+.2f}, {mve['ci95'][1]:+.2f}]). "
                f"The physics-fixed frequency-dependent lambdas generalise "
                f"where the data-driven covariance overfits.  This is the "
                f"first positive P1 result on independent-year data using "
                f"the full gravitational structure of the TEP field.")
        elif mve["median"] > 0:
            interpretations.append(
                f"The multi-band kernel is better than empirical covariance "
                f"(median {mve['median']:+.2f}%, CI includes zero).")

    verdict = {
        "step": 134,
        "title": "Multi-band frequency-dependent TEP prior",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days_scored": n,
        "multi_band_fallback_count": total_multi_fallbacks,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6,
                    "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "fusion_method": "simple average of per-band predictions",
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The multi-band kernel uses the correct physics-fixed lambda "
            "for each frequency band (Paper 1: tidal=4677 km, post-tidal="
            "1502 km, intermediate=1459 km).  It should beat the single-"
            "lambda full-band kernel (4200 km) because each band gets its "
            "proper spatial scale.  It should beat empirical covariance "
            "because the physics-fixed scales generalise where the data-"
            "driven covariance overfits."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Multi-band vs full: {mvf.get('median')}", flush=True)
    print(f"  Multi-band vs emp:  {mve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
