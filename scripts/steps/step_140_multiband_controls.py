#!/usr/bin/env python3
"""
STEP 140  --  Multi-band TEP prior with full null controls and ablation.

BULLETPROOFING THE MULTI-BAND RESULT.

Step 134 showed +2.19% vs full-band and +0.56% vs empirical covariance.
Step 135 confirmed with dev-tuned weights.  But neither included the full
null control suite from Step 64/92:

  * geometry scramble (distance labels permuted)
  * phase scramble (phase labels permuted)
  * lambda scramble (wrong lambdas assigned to bands)
  * band ablation (remove one band at a time)

This step runs the full suite on 2025 independent-year data.

PROTOCOL:
  * Run the P1 gate on 2025 held-out data with:
    1. full_band:        single lambda=4200 km (Step 92 baseline)
    2. multi_band:       per-band lambdas, simple average (Step 134)
    3. empirical:        data-driven covariance
    4. geometry_scramble: multi-band with distance labels permuted
    5. phase_scramble:   multi-band with phase labels permuted
    6. lambda_scramble:  multi-band with lambdas swapped between bands
    7. no_tidal:         multi-band without the tidal band (ablation)
    8. no_post_tidal:    multi-band without the post-tidal band
    9. no_intermediate:  multi-band without the intermediate band

  * The multi-band kernel must beat ALL scrambles and ablations to be
    bulletproof.

Output: results/outputs/step_140_multiband_controls.json
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

OUT = ROOT / "results" / "outputs" / "step_140_multiband_controls.json"
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

BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

# Lambda scramble: swap lambdas between bands
LAMBDA_SCRAMBLE = [1502.0, 4677.0, 1459.0]  # tidal gets post-tidal's lambda, etc.


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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 140: Multi-band TEP prior with full null controls", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, "
              f"lambda={band['lambda_km']:.0f} km", flush=True)
    print(f"  Lambda scramble: {LAMBDA_SCRAMBLE}", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 140, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    methods = (
        "full_band", "multi_band", "empirical",
        "geometry_scramble", "phase_scramble", "lambda_scramble",
        "no_tidal", "no_post_tidal", "no_intermediate",
    )
    per_day = {f"{m}_rms": [] for m in methods}
    comparisons = (
        "multi_vs_full", "multi_vs_emp",
        "multi_vs_geo_scr", "multi_vs_phase_scr", "multi_vs_lambda_scr",
        "multi_vs_no_tidal", "multi_vs_no_post_tidal", "multi_vs_no_intermediate",
    )
    per_day.update({c: [] for c in comparisons})

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

        # Scramble permutations (frozen per day, generated after keep filter)
        # Placeholder — actual permutations generated after QC filtering
        Dscr = None
        phase_perm = None

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
        band_grams = [g[np.ix_(keep, keep)] if g is not None else None
                      for g in band_grams]
        Dg = Dg[np.ix_(keep, keep)]

        # Generate scramble permutations NOW (after keep filter, correct size)
        geo_perm = rng.permutation(len(gnames))
        phase_perm = rng.permutation(len(gnames))
        Dscr = Dg[np.ix_(geo_perm, geo_perm)]

        ng = len(gnames)
        day_sq = {m: [] for m in methods}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band TEP kernel
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Multi-band: per-band weights
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

            # Geometry scramble: multi-band with scrambled distances
            band_ws_geo = []
            for bi, G_band in enumerate(band_grams):
                if G_band is None:
                    band_ws_geo.append(None)
                    continue
                scales = _multi_scales(BANDS[bi]["lambda_km"])
                rho_rr_s = _rho_multi(Dscr[np.ix_(pool, pool)], scales)
                rho_rt_s = _rho_multi(Dscr[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_s * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_s * G_band[b, pool])
                    band_ws_geo.append(w_band)
                except np.linalg.LinAlgError:
                    band_ws_geo.append(None)

            # Phase scramble: multi-band with scrambled phase Grams
            band_ws_phase = []
            for bi, G_band in enumerate(band_grams):
                if G_band is None:
                    band_ws_phase.append(None)
                    continue
                Gscr_b = G_band[np.ix_(phase_perm, phase_perm)]
                scales = _multi_scales(BANDS[bi]["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * Gscr_b[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * Gscr_b[b, pool])
                    band_ws_phase.append(w_band)
                except np.linalg.LinAlgError:
                    band_ws_phase.append(None)

            # Lambda scramble: wrong lambdas assigned to bands
            band_ws_lambda = []
            for bi, G_band in enumerate(band_grams):
                if G_band is None:
                    band_ws_lambda.append(None)
                    continue
                scales = _multi_scales(LAMBDA_SCRAMBLE[bi])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    band_ws_lambda.append(w_band)
                except np.linalg.LinAlgError:
                    band_ws_lambda.append(None)

            # Ablation: remove one band at a time
            def _ablation_weights(skip_idx):
                ws = []
                for bi, G_band in enumerate(band_grams):
                    if bi == skip_idx or G_band is None:
                        ws.append(None)
                        continue
                    scales = _multi_scales(BANDS[bi]["lambda_km"])
                    rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                    rho_rt_b = _rho_multi(Dg[b, pool], scales)
                    try:
                        w_band = _reg_solve(
                            rho_rr_b * G_band[np.ix_(pool, pool)] +
                            s64.NUGGET * np.eye(len(pool)),
                            rho_rt_b * G_band[b, pool])
                        ws.append(w_band)
                    except np.linalg.LinAlgError:
                        ws.append(None)
                return ws

            band_ws_no_tidal = _ablation_weights(0)
            band_ws_no_post = _ablation_weights(1)
            band_ws_no_inter = _ablation_weights(2)

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

            def _fuse(bws):
                preds = []
                for w in bws:
                    if w is not None:
                        preds.append(float(w @ rr))
                return float(np.mean(preds)) if preds else p_full

            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue

                p_full = float(w_full @ rr)
                p_emp = float(w_emp @ rr)
                p_multi = _fuse(band_ws)
                p_geo = _fuse(band_ws_geo)
                p_phase = _fuse(band_ws_phase)
                p_lambda = _fuse(band_ws_lambda)
                p_no_tidal = _fuse(band_ws_no_tidal)
                p_no_post = _fuse(band_ws_no_post)
                p_no_inter = _fuse(band_ws_no_inter)

                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["multi_band"].append((truth - p_multi) ** 2)
                day_sq["empirical"].append((truth - p_emp) ** 2)
                day_sq["geometry_scramble"].append((truth - p_geo) ** 2)
                day_sq["phase_scramble"].append((truth - p_phase) ** 2)
                day_sq["lambda_scramble"].append((truth - p_lambda) ** 2)
                day_sq["no_tidal"].append((truth - p_no_tidal) ** 2)
                day_sq["no_post_tidal"].append((truth - p_no_post) ** 2)
                day_sq["no_intermediate"].append((truth - p_no_inter) ** 2)

        if any(len(day_sq[m]) < 30 for m in methods):
            continue
        rms = {m: float(np.sqrt(np.mean(day_sq[m]))) for m in methods}
        for m in methods:
            per_day[f"{m}_rms"].append(rms[m])

        per_day["multi_vs_full"].append(100.0 * (1.0 - rms["multi_band"] / rms["full_band"]))
        per_day["multi_vs_emp"].append(100.0 * (1.0 - rms["multi_band"] / rms["empirical"]))
        per_day["multi_vs_geo_scr"].append(100.0 * (1.0 - rms["multi_band"] / rms["geometry_scramble"]))
        per_day["multi_vs_phase_scr"].append(100.0 * (1.0 - rms["multi_band"] / rms["phase_scramble"]))
        per_day["multi_vs_lambda_scr"].append(100.0 * (1.0 - rms["multi_band"] / rms["lambda_scramble"]))
        per_day["multi_vs_no_tidal"].append(100.0 * (1.0 - rms["multi_band"] / rms["no_tidal"]))
        per_day["multi_vs_no_post_tidal"].append(100.0 * (1.0 - rms["multi_band"] / rms["no_post_tidal"]))
        per_day["multi_vs_no_intermediate"].append(100.0 * (1.0 - rms["multi_band"] / rms["no_intermediate"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"multi={np.median(per_day['multi_band_rms']):.1f}, "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"emp={np.median(per_day['empirical_rms']):.1f}, "
                  f"geo_scr={np.median(per_day['geometry_scramble_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "multi_vs_full_band_pct": _ci(per_day["multi_vs_full"], rng),
        "multi_vs_empirical_pct": _ci(per_day["multi_vs_emp"], rng),
        "multi_vs_geometry_scramble_pct": _ci(per_day["multi_vs_geo_scr"], rng),
        "multi_vs_phase_scramble_pct": _ci(per_day["multi_vs_phase_scr"], rng),
        "multi_vs_lambda_scramble_pct": _ci(per_day["multi_vs_lambda_scr"], rng),
        "multi_vs_no_tidal_pct": _ci(per_day["multi_vs_no_tidal"], rng),
        "multi_vs_no_post_tidal_pct": _ci(per_day["multi_vs_no_post_tidal"], rng),
        "multi_vs_no_intermediate_pct": _ci(per_day["multi_vs_no_intermediate"], rng),
        "median_rms_m": {m: float(np.median(per_day[f"{m}_rms"])) if n else None
                         for m in methods},
    }

    # Gate: multi-band must beat all controls with positive CI lower bound
    gate_checks = {}
    for label, key in [
        ("full_band", "multi_vs_full_band_pct"),
        ("empirical", "multi_vs_empirical_pct"),
        ("geometry_scramble", "multi_vs_geometry_scramble_pct"),
        ("phase_scramble", "multi_vs_phase_scramble_pct"),
        ("lambda_scramble", "multi_vs_lambda_scramble_pct"),
        ("no_tidal", "multi_vs_no_tidal_pct"),
        ("no_post_tidal", "multi_vs_no_post_tidal_pct"),
        ("no_intermediate", "multi_vs_no_intermediate_pct"),
    ]:
        ci = results[key]
        passed = ci.get("ci95") is not None and ci["ci95"][0] > 0
        gate_checks[label] = {
            "median": ci.get("median"),
            "ci95_lower": ci["ci95"][0] if ci.get("ci95") else None,
            "passed": passed,
        }
    results["gate_pass"] = all(g["passed"] for g in gate_checks.values())
    results["gate_checks"] = gate_checks

    interpretations = []
    for label, key in [
        ("full-band single-lambda", "multi_vs_full_band_pct"),
        ("empirical covariance", "multi_vs_empirical_pct"),
        ("geometry scramble", "multi_vs_geometry_scramble_pct"),
        ("phase scramble", "multi_vs_phase_scramble_pct"),
        ("lambda scramble", "multi_vs_lambda_scramble_pct"),
    ]:
        ci = results[key]
        if ci.get("median") is not None:
            status = "BEATS" if ci["ci95"][0] > 0 else "does not beat"
            interpretations.append(
                f"Multi-band vs {label}: {status} "
                f"(median {ci['median']:+.2f}%, "
                f"CI95 [{ci['ci95'][0]:+.2f}, {ci['ci95'][1]:+.2f}])")

    for label, key in [
        ("no_tidal (ablation)", "multi_vs_no_tidal_pct"),
        ("no_post_tidal (ablation)", "multi_vs_no_post_tidal_pct"),
        ("no_intermediate (ablation)", "multi_vs_no_intermediate_pct"),
    ]:
        ci = results[key]
        if ci.get("median") is not None:
            status = "band contributes" if ci["ci95"][0] > 0 else "band may not contribute"
            interpretations.append(
                f"Multi-band vs {label}: {status} "
                f"(median {ci['median']:+.2f}%, "
                f"CI95 [{ci['ci95'][0]:+.2f}, {ci['ci95'][1]:+.2f}])")

    if results["gate_pass"]:
        interpretations.append(
            f"GATE PASSED: the multi-band kernel beats all controls and "
            f"ablations with positive CI lower bounds. The frequency-"
            f"dependent lambdas from Paper 1 carry genuine physics-fixed "
            f"predictive structure that generalises to independent-year data.")
    else:
        failed = [k for k, v in gate_checks.items() if not v["passed"]]
        interpretations.append(
            f"GATE PARTIAL: the multi-band kernel does not beat: {failed}. "
            f"The result needs further investigation.")

    verdict = {
        "step": 140,
        "title": "Multi-band TEP prior with full null controls and ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days_scored": n,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6,
                    "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "lambda_scramble": LAMBDA_SCRAMBLE,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The multi-band kernel must beat all null controls (geometry "
            "scramble, phase scramble, lambda scramble) and all ablations "
            "(no_tidal, no_post_tidal, no_intermediate) with positive CI "
            "lower bounds. If it passes, the frequency-dependent lambdas "
            "from Paper 1 carry genuine physics-fixed predictive structure."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Gate: {'PASSED' if results['gate_pass'] else 'PARTIAL'}",
          flush=True)
    for label, check in gate_checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    {label:20s}: {status} "
              f"(median={check['median']:+.2f}%, "
              f"CI_lower={check['ci95_lower']:+.2f}%)")


if __name__ == "__main__":
    main()
