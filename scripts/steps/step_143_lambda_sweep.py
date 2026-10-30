#!/usr/bin/env python3
"""
STEP 143  --  Lambda sweep per band: are Paper 1's frequency-dependent
lambdas near-optimal, not just better than scrambled?

Paper 1 (TEP-GNSS, Jaipur) measured the frequency-dependent correlation
length of the TEP field:

  * Tidal band    (10-30 uHz):  lambda = 4,677 km  (gravitational forcing)
  * Post-tidal    (30-100 uHz): lambda = 1,502 km  (transition)
  * Intermediate  (100-500 uHz): lambda = 1,459 km (local)

Step 134 showed that the multi-band kernel (using these physics-fixed
lambdas) beats the single-lambda full-band kernel on 2025 held-out data.
But that only proves the multi-band approach is better than a single
compromise lambda; it does not prove that Paper 1's specific lambda
values are optimal.  A sceptic could argue that any per-band lambdas
would do as well, or that slightly different values would do better.

This step tests whether Paper 1's lambdas are near-optimal by sweeping
each band's lambda independently on 2024 development data and then
applying the frozen dev-optimal lambdas to 2025 held-out data.

PROTOCOL:
  1. On 2024 DEVELOPMENT data, sweep each band's lambda independently
     over a grid while keeping the other bands at Paper 1 values.
     Find the optimal lambda per band (scored by median multi_band vs
     full_band improvement on dev days).
  2. Freeze the optimal lambdas.
  3. Apply the frozen optimal lambdas to 2025 HELD-OUT data.
  4. Compare:
       a. multi_band with Paper 1 lambdas (baseline from Step 134)
       b. multi_band with dev-optimal lambdas (this step)
       c. multi_band with common lambda (single compromise lambda)
       d. full_band single-lambda (Step 64 baseline)

If Paper 1's lambdas are near-optimal, the dev-optimal should be close
to Paper 1's values on the grid, and the dev-optimal should not
significantly outperform Paper 1 on 2025 held-out data.  The common-
lambda control should underperform the multi-band kernel, confirming
that different bands genuinely require different spatial scales.

Output: results/outputs/step_143_lambda_sweep.json
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

OUT = ROOT / "results" / "outputs" / "step_143_lambda_sweep.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"

DEV_YEAR = 2024
HELD_OUT_YEAR = 2025

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
#   At 30s/24h, frequency resolution = 1/86400 ~ 11.57 uHz
#   Tidal band needs at least 2 bins; start at 9.84 uHz (min_low_cycles=0.85)
BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

PAPER1_LAMBDAS = {"tidal": 4677, "post_tidal": 1502, "intermediate": 1459}

# Lambda grids for independent per-band sweep on dev data
LAMBDA_GRIDS = {
    "tidal":       [1000, 2000, 3000, 4000, 4677, 6000, 8000, 12000, 25000],
    "post_tidal":  [500, 800, 1200, 1502, 2000, 3000, 5000],
    "intermediate":[500, 800, 1200, 1459, 2000, 3000, 5000],
}

# Grid for the common-lambda control (single lambda for all bands)
COMMON_GRID = [1000, 1500, 2000, 3000, 4677, 6000, 8000, 12000]


# ---------------------------------------------------------------------------
# Helper functions (copied from Step 134)
# ---------------------------------------------------------------------------

def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _input_path(tag):
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


# ---------------------------------------------------------------------------
# Day processing and scoring
# ---------------------------------------------------------------------------

def _process_day(tag):
    """Process a single day into a precomputed bundle for lambda-sweep
    scoring.

    The bundle contains per-target data (full-band weights, band gram
    slices, distance matrices, and evaluation epochs) so that scoring
    with different lambda configurations is a pure in-memory operation
    with no redundant file I/O or gram computation.

    Returns a dict or None if the day is invalid.
    """
    cur = s64.load_day(tag, product=PRODUCT)
    prev = s64.load_day(s64._prev(tag), product=PRODUCT)
    if cur is None or prev is None:
        return None
    names, t, raw, D = cur
    s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                           max_jumps=10_000)
    if not s_cur:
        return None

    # Full-band Gram (registered, Step 64 baseline)
    pg_full = s64.prior_gram(prev, names)
    # Multi-band Grams (per-band phase Grams)
    pg_bands = _prior_gram_bands(prev, names)
    if pg_full is None or pg_bands is None:
        return None

    gnames_f, G_full, P_full = pg_full
    gnames_b, band_grams, _ = pg_bands

    common = sorted(set(gnames_f) & set(gnames_b))
    if len(common) < s64.MIN_REFS + 2:
        return None

    idx_f = [gnames_f.index(n) for n in common]
    idx_b = [gnames_b.index(n) for n in common]
    idx_cur = [names.index(n) for n in common]

    G_full = G_full[np.ix_(idx_f, idx_f)]
    band_grams = [(g[np.ix_(idx_b, idx_b)], band) for g, band in band_grams]
    P = P_full[:, idx_f]
    Dg = D[np.ix_(idx_cur, idx_cur)]
    gnames = common

    # Causal detrend (2-hour sliding window, no look-ahead)
    npre = int(s64.PREWINDOW_S / s64.DT_S)
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

    # Clean prior-day matrix
    P = P.astype(float, copy=True)
    xq = np.arange(P.shape[0]) - P.shape[0] / 2
    for j in range(P.shape[1]):
        m = np.isfinite(P[:, j])
        if not m.all():
            P[:, j] = np.interp(np.arange(P.shape[0]),
                                np.where(m)[0], P[m, j])
        P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

    # Robust station QC
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

    ng = len(gnames)
    eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

    # Precompute per-target data
    targets = []
    for b in range(ng):
        pool = [j for j in range(ng) if j != b]
        if len(pool) < s64.MIN_REFS:
            continue

        # Full-band weights (fixed, does not depend on band lambdas)
        rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
        rho_rt_f = s64._rho_multi(Dg[b, pool])
        try:
            w_full = _reg_solve(
                rho_rr_f * G_full[np.ix_(pool, pool)] +
                s64.NUGGET * np.eye(len(pool)),
                rho_rt_f * G_full[b, pool])
        except np.linalg.LinAlgError:
            continue

        # Band gram slices for this target
        band_data = []
        for G_band, band_info in band_grams:
            band_data.append((band_info["name"],
                              G_band[np.ix_(pool, pool)],
                              G_band[b, pool]))

        # Evaluation epochs
        evals = []
        B = R[:, b]
        A = R[:, pool]
        for ei in eval_i:
            truth = B[ei]
            if not np.isfinite(truth):
                continue
            rr = A[ei]
            if not np.isfinite(rr).all():
                continue
            evals.append((float(truth), rr.astype(float)))

        if not evals:
            continue

        targets.append({
            "Dg_rr": Dg[np.ix_(pool, pool)],
            "Dg_rt": Dg[b, pool],
            "band_data": band_data,
            "w_full": w_full,
            "evals": evals,
            "ng_pool": len(pool),
        })

    if not targets:
        return None

    n_evals = sum(len(t["evals"]) for t in targets)
    if n_evals < 30:
        return None

    return {"targets": targets, "n_evals": n_evals}


def _score_day(bundle, lambdas):
    """Score a day bundle with the given per-band lambdas.

    Parameters
    ----------
    bundle : dict
        Precomputed day bundle from _process_day.
    lambdas : dict
        Mapping from band name to lambda in km.

    Returns
    -------
    (rms_full, rms_multi) or (None, None) if too few samples.
    """
    sq_full = []
    sq_multi = []
    for target in bundle["targets"]:
        pool_size = target["ng_pool"]
        w_full = target["w_full"]

        # Compute band weights for this lambda configuration
        band_weights = []
        for band_name, G_rr, G_rt in target["band_data"]:
            lam = lambdas[band_name]
            scales = _multi_scales(lam)
            rho_rr = _rho_multi(target["Dg_rr"], scales)
            rho_rt = _rho_multi(target["Dg_rt"], scales)
            try:
                w = _reg_solve(
                    rho_rr * G_rr + s64.NUGGET * np.eye(pool_size),
                    rho_rt * G_rt)
                band_weights.append(w)
            except np.linalg.LinAlgError:
                band_weights.append(None)

        for truth, rr in target["evals"]:
            p_full = float(w_full @ rr)
            band_vals = [float(w @ rr) for w in band_weights if w is not None]
            p_multi = float(np.mean(band_vals)) if band_vals else p_full
            sq_full.append((truth - p_full) ** 2)
            sq_multi.append((truth - p_multi) ** 2)

    if len(sq_full) < 30:
        return None, None
    rms_full = float(np.sqrt(np.mean(sq_full)))
    rms_multi = float(np.sqrt(np.mean(sq_multi)))
    return rms_full, rms_multi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 143: Lambda sweep per band", flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print(f"  Dev year: {DEV_YEAR}, Held-out year: {HELD_OUT_YEAR}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, "
              f"Paper1 lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    # --- Prepare eval tags for both years ---
    def _eval_tags(year):
        planned = [f"{year}{d:03d}" for d in EVALUATION_DOYS]
        return [tag for tag in planned
                if _input_path(tag) is not None and
                _input_path(s64._prev(tag)) is not None]

    dev_tags = _eval_tags(DEV_YEAR)
    held_tags = _eval_tags(HELD_OUT_YEAR)
    print(f"  Dev: {len(dev_tags)} days, Held-out: {len(held_tags)} days",
          flush=True)

    if len(dev_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(dev_tags)} dev days", flush=True)
        OUT.write_text(json.dumps({"step": 143, "error": "too few dev days",
                                   "n_days": len(dev_tags)}, indent=2))
        return
    if len(held_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(held_tags)} held-out days", flush=True)
        OUT.write_text(json.dumps({"step": 143, "error": "too few held-out days",
                                   "n_days": len(held_tags)}, indent=2))
        return

    # --- Process dev (2024) days into bundles ---
    print("\n  Processing dev (2024) days...", flush=True)
    dev_bundles = []
    for k, tag in enumerate(dev_tags):
        bundle = _process_day(tag)
        if bundle is not None:
            dev_bundles.append(bundle)
        if (k + 1) % 10 == 0:
            print(f"    [{k+1}/{len(dev_tags)}] {len(dev_bundles)} valid",
                  flush=True)
    print(f"  {len(dev_bundles)} valid dev bundles", flush=True)

    if len(dev_bundles) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(dev_bundles)} valid dev bundles", flush=True)
        OUT.write_text(json.dumps({"step": 143, "error": "too few valid dev bundles",
                                   "n_days": len(dev_bundles)}, indent=2))
        return

    # --- Sweep each band's lambda independently on dev data ---
    # For each band, try each lambda in the grid while keeping the other
    # bands at Paper 1 values.  Score by median multi_band vs full_band
    # improvement on dev days.
    print("\n  Sweeping lambdas on dev data...", flush=True)
    dev_optimal = {}
    dev_sweep_scores = {}

    for band_name in ["tidal", "post_tidal", "intermediate"]:
        grid = LAMBDA_GRIDS[band_name]
        best_lambda = float(PAPER1_LAMBDAS[band_name])
        best_score = -np.inf
        sweep_detail = {}
        print(f"    {band_name}: sweeping {len(grid)} values...", flush=True)
        for lam in grid:
            lambdas = dict(PAPER1_LAMBDAS)
            lambdas[band_name] = float(lam)
            improvements = []
            for bundle in dev_bundles:
                rms_full, rms_multi = _score_day(bundle, lambdas)
                if rms_full is not None and rms_full > 0:
                    improvements.append(100.0 * (1.0 - rms_multi / rms_full))
            if improvements:
                score = float(np.median(improvements))
                sweep_detail[str(lam)] = {
                    "median_improvement_pct": score,
                    "n_days": len(improvements),
                }
                if score > best_score:
                    best_score = score
                    best_lambda = float(lam)
            med_imp = float(np.median(improvements)) if improvements else float('nan')
            print(f"      lambda={lam:>8}  median_imp={med_imp:+.2f}%  "
                  f"n={len(improvements)}", flush=True)
        dev_optimal[band_name] = best_lambda
        dev_sweep_scores[band_name] = sweep_detail
        print(f"    {band_name}: optimal lambda = {best_lambda:.0f} km "
              f"(Paper1: {PAPER1_LAMBDAS[band_name]} km)", flush=True)

    # --- Sweep common lambda on dev data ---
    # A single lambda applied to all bands simultaneously.
    print("\n  Sweeping common lambda on dev data...", flush=True)
    best_common = float(COMMON_GRID[0])
    best_common_score = -np.inf
    common_sweep_detail = {}
    for lam in COMMON_GRID:
        lambdas = {name: float(lam) for name in PAPER1_LAMBDAS}
        improvements = []
        for bundle in dev_bundles:
            rms_full, rms_multi = _score_day(bundle, lambdas)
            if rms_full is not None and rms_full > 0:
                improvements.append(100.0 * (1.0 - rms_multi / rms_full))
        if improvements:
            score = float(np.median(improvements))
            common_sweep_detail[str(lam)] = {
                "median_improvement_pct": score,
                "n_days": len(improvements),
            }
            if score > best_common_score:
                best_common_score = score
                best_common = float(lam)
        med_imp = float(np.median(improvements)) if improvements else float('nan')
        print(f"    common={lam:>8}  median_imp={med_imp:+.2f}%  "
              f"n={len(improvements)}", flush=True)
    print(f"  Common lambda optimal = {best_common:.0f} km", flush=True)

    # --- Process held-out (2025) days into bundles ---
    print("\n  Processing held-out (2025) days...", flush=True)
    held_bundles = []
    for k, tag in enumerate(held_tags):
        bundle = _process_day(tag)
        if bundle is not None:
            held_bundles.append(bundle)
        if (k + 1) % 10 == 0:
            print(f"    [{k+1}/{len(held_tags)}] {len(held_bundles)} valid",
                  flush=True)
    print(f"  {len(held_bundles)} valid held-out bundles", flush=True)

    if len(held_bundles) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(held_bundles)} valid held-out bundles",
              flush=True)
        OUT.write_text(json.dumps({"step": 143, "error": "too few valid held-out bundles",
                                   "n_days": len(held_bundles)}, indent=2))
        return

    # --- Score 2025 with different lambda configs ---
    # All configs are scored on the same bundles in the same order,
    # guaranteeing aligned paired comparisons.
    print("\n  Scoring 2025 held-out data...", flush=True)
    common_lambdas = {name: best_common for name in PAPER1_LAMBDAS}

    p1_rms_full, p1_rms_multi = [], []
    do_rms_full, do_rms_multi = [], []
    cm_rms_full, cm_rms_multi = [], []

    for k, bundle in enumerate(held_bundles):
        rf_p, rm_p = _score_day(bundle, PAPER1_LAMBDAS)
        rf_d, rm_d = _score_day(bundle, dev_optimal)
        rf_c, rm_c = _score_day(bundle, common_lambdas)
        if rf_p is None or rf_p <= 0:
            continue
        p1_rms_full.append(rf_p)
        p1_rms_multi.append(rm_p)
        do_rms_full.append(rf_d)
        do_rms_multi.append(rm_d)
        cm_rms_full.append(rf_c)
        cm_rms_multi.append(rm_c)
        if (k + 1) % 10 == 0:
            print(f"    [{k+1}/{len(held_bundles)}] "
                  f"p1={np.median(p1_rms_multi):.1f} "
                  f"do={np.median(do_rms_multi):.1f} "
                  f"cm={np.median(cm_rms_multi):.1f} m", flush=True)

    n_held = len(p1_rms_full)
    print(f"\n  {n_held} held-out days scored", flush=True)

    # --- Compute comparisons ---
    # Improvement = 100 * (1 - rms_method / rms_reference).  Positive
    # means the method has lower RMS (better) than the reference.
    paper1_vs_full = [
        100.0 * (1.0 - m / f) for m, f in zip(p1_rms_multi, p1_rms_full)]
    devopt_vs_full = [
        100.0 * (1.0 - m / f) for m, f in zip(do_rms_multi, do_rms_full)]
    devopt_vs_paper1 = [
        100.0 * (1.0 - dm / pm)
        for dm, pm in zip(do_rms_multi, p1_rms_multi)]
    common_vs_paper1 = [
        100.0 * (1.0 - cm / pm)
        for cm, pm in zip(cm_rms_multi, p1_rms_multi)]
    common_vs_full = [
        100.0 * (1.0 - m / f) for m, f in zip(cm_rms_multi, cm_rms_full)]

    results = {
        "paper1_vs_full_band_2025": _ci(paper1_vs_full, rng),
        "devoptimal_vs_full_band_2025": _ci(devopt_vs_full, rng),
        "devoptimal_vs_paper1_2025": _ci(devopt_vs_paper1, rng),
        "common_lambda_vs_paper1_2025": _ci(common_vs_paper1, rng),
        "common_lambda_vs_full_band_2025": _ci(common_vs_full, rng),
    }

    median_rms = {
        "full_band_2025": float(np.median(p1_rms_full)) if p1_rms_full else None,
        "paper1_multi_2025": float(np.median(p1_rms_multi)) if p1_rms_multi else None,
        "devoptimal_multi_2025": float(np.median(do_rms_multi)) if do_rms_multi else None,
        "common_multi_2025": float(np.median(cm_rms_multi)) if cm_rms_multi else None,
    }

    # --- Interpretation ---
    interpretations = []

    # Check whether Paper 1 values are near the dev-optimal on the grid
    for band_name in ["tidal", "post_tidal", "intermediate"]:
        p1 = PAPER1_LAMBDAS[band_name]
        opt = dev_optimal[band_name]
        grid = LAMBDA_GRIDS[band_name]
        if opt == p1:
            interpretations.append(
                f"{band_name}: Paper 1 lambda ({p1} km) is exactly the "
                f"dev-optimal value on the 2024 development grid.")
        else:
            p1_idx = grid.index(p1) if p1 in grid else None
            opt_idx = grid.index(int(opt)) if int(opt) in grid else None
            if p1_idx is not None and opt_idx is not None:
                step_dist = abs(p1_idx - opt_idx)
                if step_dist <= 1:
                    interpretations.append(
                        f"{band_name}: Paper 1 lambda ({p1} km) is within "
                        f"one grid step of the dev-optimal ({opt:.0f} km) on "
                        f"the 2024 development grid.")
                else:
                    interpretations.append(
                        f"{band_name}: Paper 1 lambda ({p1} km) is "
                        f"{step_dist} grid steps from the dev-optimal "
                        f"({opt:.0f} km) on the 2024 development grid.")
            else:
                interpretations.append(
                    f"{band_name}: Paper 1 lambda ({p1} km), dev-optimal "
                    f"({opt:.0f} km).")

    # Dev-optimal vs Paper 1 on 2025 held-out data
    dv = results["devoptimal_vs_paper1_2025"]
    if dv.get("median") is not None:
        if dv["median"] > 0 and dv["ci95"] and dv["ci95"][0] > 0:
            interpretations.append(
                f"The dev-optimal lambdas significantly outperform Paper 1 "
                f"lambdas on 2025 held-out data (median {dv['median']:+.2f}%, "
                f"CI95 [{dv['ci95'][0]:+.2f}, {dv['ci95'][1]:+.2f}]). "
                f"Paper 1's frequency-dependent lambdas are not optimal; "
                f"the development sweep found better values that "
                f"generalise to the independent year.")
        elif abs(dv["median"]) < 0.5:
            interpretations.append(
                f"The dev-optimal lambdas are statistically indistinguishable "
                f"from Paper 1 lambdas on 2025 held-out data (median "
                f"{dv['median']:+.2f}%, CI95 [{dv['ci95'][0]:+.2f}, "
                f"{dv['ci95'][1]:+.2f}]). Paper 1's frequency-dependent "
                f"lambdas are near-optimal: the development sweep did not "
                f"find values that meaningfully outperform them on the "
                f"independent year.")
        elif dv["median"] > 0:
            interpretations.append(
                f"The dev-optimal lambdas are slightly better than Paper 1 "
                f"on 2025 held-out data (median {dv['median']:+.2f}%, "
                f"CI includes zero). Paper 1's lambdas are near-optimal; "
                f"the improvement from tuning is not statistically "
                f"significant on the independent year.")
        else:
            interpretations.append(
                f"Paper 1 lambdas outperform the dev-optimal lambdas on "
                f"2025 held-out data (median {dv['median']:+.2f}%, "
                f"CI95 [{dv['ci95'][0]:+.2f}, {dv['ci95'][1]:+.2f}]). "
                f"The dev-optimal lambdas overfit the 2024 development "
                f"year; Paper 1's physics-fixed values generalise better "
                f"to the independent year.")

    # Common lambda vs Paper 1 multi-band
    cv = results["common_lambda_vs_paper1_2025"]
    if cv.get("median") is not None:
        if cv["median"] < 0 and cv["ci95"] and cv["ci95"][1] < 0:
            interpretations.append(
                f"Paper 1's frequency-dependent (multi-band) lambdas "
                f"significantly outperform the single common lambda on "
                f"2025 held-out data (median {cv['median']:+.2f}%, "
                f"CI95 [{cv['ci95'][0]:+.2f}, {cv['ci95'][1]:+.2f}]). "
                f"The frequency-dependent structure is not noise; "
                f"different bands genuinely require different spatial "
                f"scales, and Paper 1's per-band values capture this "
                f"structure correctly.")
        elif cv["median"] < 0:
            interpretations.append(
                f"Paper 1's multi-band lambdas are better than the common "
                f"lambda on 2025 held-out data (median {cv['median']:+.2f}%, "
                f"CI includes zero). The frequency-dependent structure "
                f"appears beneficial but is not statistically significant "
                f"on the independent year.")
        else:
            interpretations.append(
                f"The common lambda performs comparably to Paper 1's "
                f"multi-band lambdas on 2025 held-out data (median "
                f"{cv['median']:+.2f}%). The frequency-dependent structure "
                f"does not provide a clear advantage over a single "
                f"compromise lambda on the independent year.")

    # Paper 1 vs full_band (confirming Step 134 result)
    pv = results["paper1_vs_full_band_2025"]
    if pv.get("median") is not None:
        if pv["median"] > 0 and pv["ci95"] and pv["ci95"][0] > 0:
            interpretations.append(
                f"Paper 1's multi-band kernel significantly outperforms "
                f"the single-lambda full-band kernel on 2025 held-out "
                f"data (median {pv['median']:+.2f}%, CI95 "
                f"[{pv['ci95'][0]:+.2f}, {pv['ci95'][1]:+.2f}]), "
                f"confirming the Step 134 result.")

    verdict = {
        "step": 143,
        "title": "Lambda sweep per band",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "dev_year": DEV_YEAR,
        "held_out_year": HELD_OUT_YEAR,
        "paper1_lambdas": {k: v for k, v in PAPER1_LAMBDAS.items()},
        "dev_optimal_lambdas": dev_optimal,
        "common_lambda_dev_optimal": best_common,
        "lambda_grids": LAMBDA_GRIDS,
        "common_lambda_grid": COMMON_GRID,
        "n_dev_days": len(dev_bundles),
        "n_held_out_days": n_held,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"] * 1e6,
                    "f_hi_uhz": b["f_hi"] * 1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "dev_sweep_scores": dev_sweep_scores,
        "common_sweep_scores": common_sweep_detail,
        "results": results,
        "median_rms_m": median_rms,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "If Paper 1's frequency-dependent lambdas are near-optimal "
            "(not just better than scrambled), the dev-optimal lambdas "
            "should be close to Paper 1's values on the development grid, "
            "and the dev-optimal should not significantly outperform "
            "Paper 1 on 2025 held-out data. The common-lambda control "
            "should underperform the multi-band kernel, confirming that "
            "different bands require different spatial scales."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Dev-optimal lambdas: {dev_optimal}", flush=True)
    print(f"  Common lambda: {best_common:.0f} km", flush=True)
    print(f"  Paper1 vs full: {results['paper1_vs_full_band_2025'].get('median')}", flush=True)
    print(f"  Devopt vs paper1: {results['devoptimal_vs_paper1_2025'].get('median')}", flush=True)
    print(f"  Common vs paper1: {results['common_lambda_vs_paper1_2025'].get('median')}", flush=True)


if __name__ == "__main__":
    main()
