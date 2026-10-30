#!/usr/bin/env python3
"""
STEP 135  --  Weighted band fusion: optimising the multi-band combination.

Step 134 showed that the multi-band kernel (per-band lambda from Paper 1)
beats the single-lambda full-band kernel by +2.19% and empirical covariance
by +0.56% on 2025 independent-year data, using a SIMPLE AVERAGE of the
per-band predictions.

This step tests whether a WEIGHTED fusion does better.  Three approaches:

  1. dev_tuned:    Weights tuned on 2024 development data, applied frozen
                   to 2025.  This preserves the independent-year protocol.
  2. inv_var:      Inverse-variance weighting — each band weighted by the
                   inverse of its prior-day prediction variance.
  3. energy:       Energy weighting — each band weighted by its spectral
                   energy (more energetic bands get more weight).

The dev_tuned weights are found by grid search over the 2024 development
period, maximising the leave-one-station-out prediction accuracy.  The
frozen weights are then applied to 2025 without modification.

PROTOCOL:
  * Phase 1: Tune fusion weights on 2024 development data (30 days).
  * Phase 2: Apply frozen weights to 2025 held-out data.
  * Compare:
    1. full_band:    single lambda=4200 km
    2. multi_avg:    simple average (Step 134)
    3. multi_dev:    dev-tuned weighted fusion
    4. multi_iv:     inverse-variance fusion
    5. multi_energy: energy-weighted fusion
    6. empirical:    data-driven covariance

Output: results/outputs/step_135_weighted_fusion.json
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

OUT = ROOT / "results" / "outputs" / "step_135_weighted_fusion.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"

EVALUATION_DOYS_2025 = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260902
N_DEV_DAYS = 30

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
            grams.append((result["predictive_gram"], band,
                          result.get("n_bins", 0)))
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


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def _prepare_day(tag, prev_tag):
    """Load and prepare a day's data for multi-band prediction.

    Returns (gnames, R, P, Dg, G_full, band_data, eval_i) or None.
    band_data is a list of (G_band, band_info, n_bins, band_weights_per_target)
    """
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
    band_grams = [(g[np.ix_(idx_b, idx_b)], band, n_bins)
                  for g, band, n_bins in band_grams_raw]
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
        return None
    gnames = [gnames[i] for i in keep]
    R = R[:, keep]; P = P[:, keep]
    G_full = G_full[np.ix_(keep, keep)]
    band_grams = [(g[np.ix_(keep, keep)], band, n_bins)
                  for g, band, n_bins in band_grams]
    Dg = Dg[np.ix_(keep, keep)]

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

    return gnames, R, P, Dg, G_full, band_grams, eval_i


def _compute_weights(Dg, G_full, band_grams, ng):
    """Compute per-target prediction weights for full-band, each band, and empirical.

    Returns dict with:
      'full': list of w_full per target
      'bands': list of list of w_band per target (or None if failed)
      'emp': list of w_emp per target
      'band_energies': list of spectral energy per band (for energy weighting)
    """
    weights = {"full": [], "bands": [], "emp": [], "band_energies": []}

    for b in range(ng):
        pool = [j for j in range(ng) if j != b]
        if len(pool) < s64.MIN_REFS:
            weights["full"].append(None)
            weights["bands"].append([])
            weights["emp"].append(None)
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
            w_full = None
        weights["full"].append(w_full)

        # Per-band
        band_ws = []
        band_energies = []
        for G_band, band_info, n_bins in band_grams:
            scales = _multi_scales(band_info["lambda_km"])
            rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
            rho_rt_b = _rho_multi(Dg[b, pool], scales)
            try:
                w_band = _reg_solve(
                    rho_rr_b * G_band[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_b * G_band[b, pool])
                # Band energy: trace of the Gram
                energy = float(np.trace(G_band[np.ix_(pool, pool)]))
            except np.linalg.LinAlgError:
                w_band = None
                energy = 0.0
            band_ws.append(w_band)
            band_energies.append(energy)
        weights["bands"].append(band_ws)
        weights["band_energies"].append(band_energies)

        # Empirical (needs P — handled by caller)
        weights["emp"].append(None)  # filled by caller

    return weights


def _score_day(R, A_all, eval_i, weights, ng, fusion_weights=None,
               P=None, Dg=None):
    """Score a day with given fusion weights.

    Returns dict of {method: [squared_errors]}.
    """
    day_sq = {k: [] for k in ("full", "avg", "dev", "iv", "energy", "emp")}

    for b in range(ng):
        pool = [j for j in range(ng) if j != b]
        if len(pool) < s64.MIN_REFS:
            continue

        w_full = weights["full"][b]
        band_ws = weights["bands"][b]
        if w_full is None:
            continue

        # Empirical
        if P is not None and Dg is not None:
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
        else:
            w_emp = None

        B = R[:, b]; A = R[:, pool]
        band_energies = weights["band_energies"][b]

        for ei in eval_i:
            truth = B[ei]
            if not np.isfinite(truth):
                continue
            rr = A[ei]
            if not np.isfinite(rr).all():
                continue

            p_full = float(w_full @ rr)

            # Per-band predictions
            band_preds = []
            for w_band in band_ws:
                if w_band is not None:
                    band_preds.append(float(w_band @ rr))
                else:
                    band_preds.append(None)

            # Simple average
            valid_preds = [p for p in band_preds if p is not None]
            p_avg = float(np.mean(valid_preds)) if valid_preds else p_full

            # Dev-tuned weighted
            if fusion_weights and valid_preds:
                p_dev = sum(w * p for w, p in zip(fusion_weights, band_preds)
                            if p is not None)
                p_dev /= sum(w for w, p in zip(fusion_weights, band_preds)
                              if p is not None)
            else:
                p_dev = p_avg

            # Inverse-variance: weight by 1/var of band prediction
            # Use prior-day band prediction variance as proxy
            iv_weights = []
            for w_band in band_ws:
                if w_band is not None:
                    var = float(np.var(w_band))
                    iv_weights.append(1.0 / (var + 1e-12))
                else:
                    iv_weights.append(0.0)
            iv_sum = sum(iv_weights)
            if iv_sum > 0 and valid_preds:
                p_iv = sum(w * p for w, p in zip(iv_weights, band_preds)
                            if p is not None) / iv_sum
            else:
                p_iv = p_avg

            # Energy-weighted
            e_weights = [max(e, 0) for e in band_energies]
            e_sum = sum(e_weights)
            if e_sum > 0 and valid_preds:
                p_energy = sum(w * p for w, p in zip(e_weights, band_preds)
                                if p is not None) / e_sum
            else:
                p_energy = p_avg

            day_sq["full"].append((truth - p_full) ** 2)
            day_sq["avg"].append((truth - p_avg) ** 2)
            day_sq["dev"].append((truth - p_dev) ** 2)
            day_sq["iv"].append((truth - p_iv) ** 2)
            day_sq["energy"].append((truth - p_energy) ** 2)
            if w_emp is not None:
                day_sq["emp"].append((truth - float(w_emp @ rr)) ** 2)

    return day_sq


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 135: Weighted band fusion", flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print(f"  Phase 1: Tune fusion weights on 2024 dev data ({N_DEV_DAYS} days)",
          flush=True)
    print(f"  Phase 2: Apply frozen weights to 2025 held-out data", flush=True)
    print()

    # ---- Phase 1: Tune fusion weights on 2024 dev data ----
    dates_2024 = td.available_dates(PRODUCT, 2024)
    dev_days = [dates_2024[i] for i in
                np.linspace(1, len(dates_2024) - 1, N_DEV_DAYS).astype(int)]

    # Grid search over fusion weights
    # w_tidal, w_post_tidal, w_intermediate (normalised)
    weight_grid = []
    for wt in np.arange(0.0, 2.51, 0.25):
        for wp in np.arange(0.0, 2.51, 0.25):
            wi = 1.0  # fix intermediate at 1.0, vary the others
            weight_grid.append((wt, wp, wi))

    print(f"  Testing {len(weight_grid)} weight combinations on dev data",
          flush=True)

    dev_scores = {tuple(w): [] for w in weight_grid}
    dev_avg_scores = []

    for k, tag in enumerate(dev_days):
        prev_tag = s64._prev(tag)
        if _input_path(tag) is None or _input_path(prev_tag) is None:
            continue
        prep = _prepare_day(tag, prev_tag)
        if prep is None:
            continue
        gnames, R, P, Dg, G_full, band_grams, eval_i = prep
        ng = len(gnames)
        weights = _compute_weights(Dg, G_full, band_grams, ng)

        for fw in weight_grid:
            day_sq = _score_day(R, None, eval_i, weights, ng,
                                fusion_weights=fw, P=P, Dg=Dg)
            if len(day_sq["dev"]) >= 30:
                rms = float(np.sqrt(np.mean(day_sq["dev"])))
                dev_scores[tuple(fw)].append(rms)

        # Also score simple average
        day_sq_avg = _score_day(R, None, eval_i, weights, ng, P=P, Dg=Dg)
        if len(day_sq_avg["avg"]) >= 30:
            dev_avg_scores.append(float(np.sqrt(np.mean(day_sq_avg["avg"]))))

        if (k + 1) % 10 == 0:
            print(f"  [dev {k+1}/{len(dev_days)}]", flush=True)

    # Find best fusion weights
    best_w = None
    best_rms = np.inf
    for fw, scores in dev_scores.items():
        if len(scores) >= 10:
            med = float(np.median(scores))
            if med < best_rms:
                best_rms = med
                best_w = fw

    if best_w is None:
        print("  ERROR: no valid dev scores", flush=True)
        OUT.write_text(json.dumps({"step": 135, "error": "no dev scores"},
                                  indent=2))
        return

    print(f"\n  Best dev weights: tidal={best_w[0]:.2f}, "
          f"post_tidal={best_w[1]:.2f}, intermediate={best_w[2]:.2f}",
          flush=True)
    print(f"  Best dev RMS: {best_rms:.3f} m", flush=True)
    if dev_avg_scores:
        print(f"  Avg dev RMS: {np.median(dev_avg_scores):.3f} m", flush=True)

    # ---- Phase 2: Apply frozen weights to 2025 held-out data ----
    print(f"\n  Phase 2: Apply frozen weights to 2025", flush=True)

    planned_eval_tags = [f"2025{d:03d}" for d in EVALUATION_DOYS_2025]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 135, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    per_day = {k: [] for k in (
        "full_rms", "avg_rms", "dev_rms", "iv_rms", "energy_rms", "emp_rms",
        "dev_vs_full", "dev_vs_avg", "dev_vs_emp",
        "iv_vs_full", "iv_vs_emp",
        "energy_vs_full", "energy_vs_emp")}

    for k, tag in enumerate(eval_tags):
        prev_tag = s64._prev(tag)
        prep = _prepare_day(tag, prev_tag)
        if prep is None:
            continue
        gnames, R, P, Dg, G_full, band_grams, eval_i = prep
        ng = len(gnames)
        weights = _compute_weights(Dg, G_full, band_grams, ng)

        day_sq = _score_day(R, None, eval_i, weights, ng,
                            fusion_weights=best_w, P=P, Dg=Dg)

        if any(len(day_sq[m]) < 30 for m in
               ("full", "avg", "dev", "iv", "energy", "emp")):
            continue

        rms = {m: float(np.sqrt(np.mean(day_sq[m])))
               for m in ("full", "avg", "dev", "iv", "energy", "emp")}
        per_day["full_rms"].append(rms["full"])
        per_day["avg_rms"].append(rms["avg"])
        per_day["dev_rms"].append(rms["dev"])
        per_day["iv_rms"].append(rms["iv"])
        per_day["energy_rms"].append(rms["energy"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["dev_vs_full"].append(100.0 * (1.0 - rms["dev"] / rms["full"]))
        per_day["dev_vs_avg"].append(100.0 * (1.0 - rms["dev"] / rms["avg"]))
        per_day["dev_vs_emp"].append(100.0 * (1.0 - rms["dev"] / rms["emp"]))
        per_day["iv_vs_full"].append(100.0 * (1.0 - rms["iv"] / rms["full"]))
        per_day["iv_vs_emp"].append(100.0 * (1.0 - rms["iv"] / rms["emp"]))
        per_day["energy_vs_full"].append(100.0 * (1.0 - rms["energy"] / rms["full"]))
        per_day["energy_vs_emp"].append(100.0 * (1.0 - rms["energy"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_rms']):.1f}, "
                  f"dev={np.median(per_day['dev_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "frozen_fusion_weights": {
            "tidal": best_w[0],
            "post_tidal": best_w[1],
            "intermediate": best_w[2],
        },
        "dev_vs_full_band_pct": _ci(per_day["dev_vs_full"], rng),
        "dev_vs_avg_pct": _ci(per_day["dev_vs_avg"], rng),
        "dev_vs_empirical_pct": _ci(per_day["dev_vs_emp"], rng),
        "iv_vs_full_band_pct": _ci(per_day["iv_vs_full"], rng),
        "iv_vs_empirical_pct": _ci(per_day["iv_vs_emp"], rng),
        "energy_vs_full_band_pct": _ci(per_day["energy_vs_full"], rng),
        "energy_vs_empirical_pct": _ci(per_day["energy_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_rms"])) if n else None,
            "avg": float(np.median(per_day["avg_rms"])) if n else None,
            "dev_tuned": float(np.median(per_day["dev_rms"])) if n else None,
            "inv_var": float(np.median(per_day["iv_rms"])) if n else None,
            "energy": float(np.median(per_day["energy_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    dvf = results["dev_vs_full_band_pct"]
    dve = results["dev_vs_empirical_pct"]
    dva = results["dev_vs_avg_pct"]

    if dvf.get("median") is not None:
        if dvf["median"] > 0 and dvf["ci95"] and dvf["ci95"][0] > 0:
            interpretations.append(
                f"The dev-tuned weighted multi-band kernel BEATS the single-"
                f"lambda full-band kernel on independent-year data "
                f"(median {dvf['median']:+.2f}%, "
                f"CI95 [{dvf['ci95'][0]:+.2f}, {dvf['ci95'][1]:+.2f}]). "
                f"The frozen fusion weights (tidal={best_w[0]:.2f}, "
                f"post_tidal={best_w[1]:.2f}, intermediate={best_w[2]:.2f}) "
                f"were tuned on 2024 development data and applied without "
                f"modification to 2025.")
        elif dvf["median"] > 0:
            interpretations.append(
                f"The dev-tuned kernel is better than the full-band kernel "
                f"(median {dvf['median']:+.2f}%, CI includes zero).")

    if dva.get("median") is not None:
        if dva["median"] > 0:
            interpretations.append(
                f"The dev-tuned weights beat the simple average "
                f"(median {dva['median']:+.2f}%, "
                f"CI95 [{dva['ci95'][0]:+.2f}, {dva['ci95'][1]:+.2f}]). "
                f"The optimal fusion is not uniform — some bands carry more "
                f"predictive power than others.")
        else:
            interpretations.append(
                f"The dev-tuned weights do not beat the simple average "
                f"({dva['median']:+.2f}%).  Uniform fusion is sufficient.")

    if dve.get("median") is not None:
        if dve["median"] > 0 and dve["ci95"] and dve["ci95"][0] > 0:
            interpretations.append(
                f"The dev-tuned multi-band kernel BEATS empirical covariance "
                f"on independent-year data (median {dve['median']:+.2f}%, "
                f"CI95 [{dve['ci95'][0]:+.2f}, {dve['ci95'][1]:+.2f}]). "
                f"The physics-fixed frequency-dependent lambdas with tuned "
                f"fusion generalise where the data-driven covariance overfits.")

    # Compare fusion methods
    ivf = results["iv_vs_full_band_pct"]
    evf = results["energy_vs_full_band_pct"]
    if ivf.get("median") is not None:
        interpretations.append(
            f"Inverse-variance fusion vs full-band: {ivf['median']:+.2f}% "
            f"(CI95 [{ivf['ci95'][0]:+.2f}, {ivf['ci95'][1]:+.2f}]).")
    if evf.get("median") is not None:
        interpretations.append(
            f"Energy-weighted fusion vs full-band: {evf['median']:+.2f}% "
            f"(CI95 [{evf['ci95'][0]:+.2f}, {evf['ci95'][1]:+.2f}]).")

    verdict = {
        "step": 135,
        "title": "Weighted band fusion for multi-band TEP prior",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_dev_days": N_DEV_DAYS,
        "n_eval_days": n,
        "frozen_fusion_weights": results["frozen_fusion_weights"],
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The dev-tuned weighted fusion should beat the simple average "
            "(Step 134) because the bands carry different amounts of "
            "predictive power.  The frozen weights (tuned on 2024, applied "
            "to 2025) preserve the independent-year protocol."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Dev-tuned vs full: {dvf.get('median')}", flush=True)
    print(f"  Dev-tuned vs avg:  {dva.get('median')}", flush=True)
    print(f"  Dev-tuned vs emp:  {dve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
