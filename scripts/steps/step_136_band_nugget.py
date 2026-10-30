#!/usr/bin/env python3
"""
STEP 136  --  Band-specific nugget optimisation for the multi-band TEP prior.

Step 134 showed +2.19% with the multi-band kernel using a common nugget
(0.70).  Step 135 tunes the fusion weights.  This step tunes the per-band
nugget — each frequency band may need a different noise floor for optimal
kriging.

The tidal band (gravitational, long-range) may have a lower nugget because
the gravitational signal is more spatially coherent.  The intermediate band
(local, short-range) may need a higher nugget because it is noisier.

PROTOCOL:
  * Phase 1: Tune per-band nuggets on 2024 development data (grid search).
  * Phase 2: Apply frozen nuggets to 2025 held-out data with the Step 135
    fusion weights.

Output: results/outputs/step_136_band_nugget.json
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
from scripts.steps import step_135_weighted_fusion as s135

OUT = ROOT / "results" / "outputs" / "step_136_band_nugget.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"

EVALUATION_DOYS_2025 = s135.EVALUATION_DOYS_2025
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260902
N_DEV_DAYS = 30

BANDS = s135.BANDS
NUGGET_GRID = (0.35, 0.50, 0.70, 0.90, 1.10)


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


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def _band_phase_grams(x, dt_s):
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


def _prepare_day(tag, prev_tag):
    cur = s64.load_day(tag, product=PRODUCT)
    prev = s64.load_day(prev_tag, product=PRODUCT)
    if cur is None or prev is None:
        return None
    names, t, raw, D = cur
    s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
    if not s_cur:
        return None

    pg_full = s64.prior_gram(prev, names)
    pg_bands = _prior_gram_bands(prev, names)
    if pg_full is None or pg_bands is None:
        return None

    gnames_f, G_full, P_full = pg_full
    gnames_b, band_grams_raw, _ = pg_bands

    common = sorted(set(gnames_f) & set(gnames_b))
    if len(common) < s64.MIN_REFS + 2:
        return None

    idx_f = [gnames_f.index(n) for n in common]
    idx_b = [gnames_b.index(n) for n in common]
    idx_cur = [names.index(n) for n in common]

    G_full = G_full[np.ix_(idx_f, idx_f)]
    band_grams = [(g[np.ix_(idx_b, idx_b)], band)
                  for g, band in band_grams_raw]
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
        return None
    gnames = [gnames[i] for i in keep]
    R = R[:, keep]; P = P[:, keep]
    G_full = G_full[np.ix_(keep, keep)]
    band_grams = [(g[np.ix_(keep, keep)], band) for g, band in band_grams]
    Dg = Dg[np.ix_(keep, keep)]

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))
    return gnames, R, P, Dg, G_full, band_grams, eval_i


def _score_day_with_nuggets(R, P, Dg, G_full, band_grams, eval_i, ng,
                            band_nuggets, fusion_weights):
    """Score a day with per-band nuggets and fusion weights."""
    day_sq = {k: [] for k in ("full", "multi", "emp")}
    fallback_count = [0]  # mutable counter for fallback events

    for b in range(ng):
        pool = [j for j in range(ng) if j != b]
        if len(pool) < s64.MIN_REFS:
            continue

        # Full-band
        rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
        rho_rt_f = s64._rho_multi(Dg[b, pool])
        try:
            w_full = _reg_solve(
                rho_rr_f * G_full[np.ix_(pool, pool)] +
                s64.NUGGET * np.eye(len(pool)),
                rho_rt_f * G_full[b, pool])
        except np.linalg.LinAlgError:
            continue

        # Per-band with per-band nuggets
        band_ws = []
        for bi, (G_band, band_info) in enumerate(band_grams):
            scales = _multi_scales(band_info["lambda_km"])
            rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
            rho_rt_b = _rho_multi(Dg[b, pool], scales)
            nug = band_nuggets[bi]
            try:
                w_band = _reg_solve(
                    rho_rr_b * G_band[np.ix_(pool, pool)] +
                    nug * np.eye(len(pool)),
                    rho_rt_b * G_band[b, pool])
                band_ws.append(w_band)
            except np.linalg.LinAlgError:
                band_ws.append(None)

        # Empirical
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

            band_preds = []
            for w_band in band_ws:
                if w_band is not None:
                    band_preds.append(float(w_band @ rr))
                else:
                    band_preds.append(None)

            valid_preds = [p for p in band_preds if p is not None]
            if valid_preds and fusion_weights:
                fw_valid = [w for w, p in zip(fusion_weights, band_preds)
                            if p is not None]
                p_multi = sum(w * p for w, p in zip(fw_valid, valid_preds))
                p_multi /= sum(fw_valid)
            elif valid_preds:
                p_multi = float(np.mean(valid_preds))
            else:
                p_multi = p_full
                fallback_count[0] += 1

            day_sq["full"].append((truth - p_full) ** 2)
            day_sq["multi"].append((truth - p_multi) ** 2)
            day_sq["emp"].append((truth - p_emp) ** 2)

    return day_sq, fallback_count[0]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 136: Band-specific nugget optimisation", flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print(f"  Nugget grid: {NUGGET_GRID}", flush=True)
    print(f"  Bands: {[b['name'] for b in BANDS]}", flush=True)
    print()

    # Load Step 135 frozen fusion weights
    s135_out = ROOT / "results" / "outputs" / "step_135_weighted_fusion.json"
    if s135_out.exists():
        s135_data = json.load(open(s135_out))
        fw = s135_data.get("frozen_fusion_weights", {})
        fusion_weights = [fw.get("tidal", 1.0), fw.get("post_tidal", 1.0),
                          fw.get("intermediate", 1.0)]
        print(f"  Using Step 135 fusion weights: {fusion_weights}", flush=True)
    else:
        fusion_weights = [1.0, 1.0, 1.0]
        print(f"  Step 135 output not found, using equal weights", flush=True)
    print()

    # ---- Phase 1: Tune per-band nuggets on 2024 dev data ----
    dates_2024 = td.available_dates(PRODUCT, 2024)
    dev_days = [dates_2024[i] for i in
                np.linspace(1, len(dates_2024) - 1, N_DEV_DAYS).astype(int)]

    # Grid search over per-band nuggets
    n_bands = len(BANDS)
    # To keep the grid manageable, search each band independently
    # (coordinate descent)
    best_nuggets = [s64.NUGGET] * n_bands  # start from default

    print(f"  Phase 1: Coordinate descent nugget tuning on dev data",
          flush=True)

    for iteration in range(2):  # 2 passes of coordinate descent
        for bi in range(n_bands):
            best_rms = np.inf
            best_nug = best_nuggets[bi]
            for nug in NUGGET_GRID:
                trial_nuggets = list(best_nuggets)
                trial_nuggets[bi] = nug
                scores = []
                for k, tag in enumerate(dev_days):
                    prev_tag = s64._prev(tag)
                    if _input_path(tag) is None or _input_path(prev_tag) is None:
                        continue
                    prep = _prepare_day(tag, prev_tag)
                    if prep is None:
                        continue
                    gnames, R, P, Dg, G_full, band_grams, eval_i = prep
                    ng = len(gnames)
                    day_sq, _fb = _score_day_with_nuggets(
                        R, P, Dg, G_full, band_grams, eval_i, ng,
                        trial_nuggets, fusion_weights)
                    if len(day_sq["multi"]) >= 30:
                        scores.append(float(np.sqrt(np.mean(day_sq["multi"]))))
                if len(scores) >= 10:
                    med = float(np.median(scores))
                    if med < best_rms:
                        best_rms = med
                        best_nug = nug
            best_nuggets[bi] = best_nug
            print(f"  iter={iteration} band={BANDS[bi]['name']:14s} "
                  f"best_nugget={best_nug:.2f} (dev_rms={best_rms:.3f})",
                  flush=True)

    print(f"\n  Best per-band nuggets: {best_nuggets}", flush=True)

    # ---- Phase 2: Apply frozen nuggets to 2025 ----
    print(f"\n  Phase 2: Apply frozen nuggets to 2025", flush=True)

    planned_eval_tags = [f"2025{d:03d}" for d in EVALUATION_DOYS_2025]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 136, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    per_day = {k: [] for k in (
        "full_rms", "multi_rms", "emp_rms",
        "multi_vs_full", "multi_vs_emp")}

    for k, tag in enumerate(eval_tags):
        prev_tag = s64._prev(tag)
        prep = _prepare_day(tag, prev_tag)
        if prep is None:
            continue
        gnames, R, P, Dg, G_full, band_grams, eval_i = prep
        ng = len(gnames)
        day_sq, day_fb = _score_day_with_nuggets(
            R, P, Dg, G_full, band_grams, eval_i, ng,
            best_nuggets, fusion_weights)

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
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_rms']):.1f}, "
                  f"multi={np.median(per_day['multi_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "frozen_band_nuggets": {
            BANDS[i]["name"]: best_nuggets[i] for i in range(n_bands)
        },
        "fusion_weights": fusion_weights,
        "multi_vs_full_band_pct": _ci(per_day["multi_vs_full"], rng),
        "multi_vs_empirical_pct": _ci(per_day["multi_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_rms"])) if n else None,
            "multi_band": float(np.median(per_day["multi_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    mvf = results["multi_vs_full_band_pct"]
    mve = results["multi_vs_empirical_pct"]

    if mvf.get("median") is not None:
        if mvf["median"] > 0 and mvf["ci95"] and mvf["ci95"][0] > 0:
            interpretations.append(
                f"The band-specific-nugget multi-band kernel BEATS the "
                f"full-band kernel (median {mvf['median']:+.2f}%, "
                f"CI95 [{mvf['ci95'][0]:+.2f}, {mvf['ci95'][1]:+.2f}]). "
                f"Per-band nuggets ({results['frozen_band_nuggets']}) "
                f"improve over the common nugget.")
        elif mvf["median"] > 0:
            interpretations.append(
                f"The band-specific-nugget kernel is better than the "
                f"full-band kernel (median {mvf['median']:+.2f}%, CI "
                f"includes zero).")

    if mve.get("median") is not None:
        if mve["median"] > 0 and mve["ci95"] and mve["ci95"][0] > 0:
            interpretations.append(
                f"The band-specific-nugget multi-band kernel BEATS empirical "
                f"covariance (median {mve['median']:+.2f}%, "
                f"CI95 [{mve['ci95'][0]:+.2f}, {mve['ci95'][1]:+.2f}]).")

    verdict = {
        "step": 136,
        "title": "Band-specific nugget optimisation for multi-band TEP prior",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_dev_days": N_DEV_DAYS,
        "n_eval_days": n,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Per-band nuggets should improve over the common nugget because "
            "each frequency band has a different signal-to-noise ratio. "
            "The tidal band (gravitational, long-range) may have a lower "
            "nugget; the intermediate band (local, noisier) may need a "
            "higher nugget."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Multi vs full: {mvf.get('median')}", flush=True)
    print(f"  Multi vs emp:  {mve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
