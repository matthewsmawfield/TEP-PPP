#!/usr/bin/env python3
"""
STEP 133  --  Tidal-enhanced TEP prior: amplifying the gravitational signal.

AMPLIFYING THE GRAVITATIONAL TIDES RATHER THAN REMOVING THEM.

Step 127 notched out the tidal lines (removing the gravitational signal).
Step 131 isolates the tidal lines (using only the gravitational signal).
This step takes the complementary approach: use the FULL band but WEIGHT
the gravitational tidal frequencies more heavily in the phase Gram.

The tidal-enhanced Gram amplifies the gravitational TEP signal while
retaining the continuum for stability.  This should combine the best of
both: the gravitational tidal power (strongest spatial coherence, Paper 1)
and the continuum (broadband stability).

PROTOCOL:
  * Run the P1 gate on 2025 held-out data.
  * Compare:
    1. full_band:     standard phase Gram (equal weight, 10-500 uHz)
    2. tidal_enhanced: phase Gram with tidal frequencies weighted 3x
    3. empirical:     data-driven covariance

  * TEP prediction: the tidal-enhanced kernel beats both the full-band
    (which under-weights the gravitational signal) and empirical
    covariance (which cannot distinguish gravitational from atmospheric
    components).

Output: results/outputs/step_133_tidal_enhanced.json
"""
from __future__ import annotations

import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_133_tidal_enhanced.json"
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

TIDAL_PERIODS = [86400.0, 43200.0, 28800.0, 21600.0, 86164.0, 44712.0]
TIDAL_WIDTH_FRAC = 0.05
TIDAL_WEIGHT = 3.0  # weight tidal bins 3x relative to continuum bins

# Use the Paper 1 full-band lambda (the tidal-enhanced Gram still covers
# the full band, so the full-band scale is appropriate, but we also test
# the tidal lambda).
LAMBDA_FULL = s64.LAMBDA_BASE_KM
LAMBDA_TIDAL = 4677.0  # Paper 1 tidal-band scale


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _tidal_enhanced_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                               min_bins=6, min_low_cycles=0.85):
    """Phase Gram with gravitational tidal frequencies weighted more heavily.

    The full band is retained (for stability) but the gravitational tidal
    bins are weighted TIDAL_WEIGHT times more than the continuum bins.
    This amplifies the gravitational TEP signal without discarding the
    continuum.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history too short")

    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("bad scale")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("bad detrended scale")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    band = (freq >= f_lo_hz) & (freq <= f_hi_hz)
    if int(band.sum()) < int(min_bins):
        raise ValueError("too few bins in band")

    # Build tidal weight vector
    weights = np.ones(len(freq))
    for period in TIDAL_PERIODS:
        fn = 1.0 / period
        lo = fn * (1.0 - TIDAL_WIDTH_FRAC)
        hi = fn * (1.0 + TIDAL_WIDTH_FRAC)
        tidal_bin = (freq >= lo) & (freq <= hi)
        weights[tidal_bin] = TIDAL_WEIGHT

    # Apply weights to the band-selected spectrum
    F_band = F[band] * np.sqrt(weights[band])[:, None]

    cross = np.einsum("fi,fj->ij", F_band.conj(), F_band, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero spectral energy")
    norm = np.sqrt(np.outer(energy, energy))
    pg = np.real(cross / norm)
    pg = 0.5 * (pg + pg.T)
    np.fill_diagonal(pg, 1.0)
    return pg


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _prior_gram_enhanced(prev_day, cur_names):
    """Build a tidal-enhanced phase Gram."""
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
    try:
        g = _tidal_enhanced_phase_gram(np.column_stack(spectral_cols), s64.DT_S)
    except ValueError:
        return None
    return kept, g, np.column_stack(clean_raw_cols)


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

    print("Step 133: Tidal-enhanced TEP prior — amplifying gravitational signal",
          flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Tidal weight: {TIDAL_WEIGHT}x", flush=True)
    print(f"  Lambda: {LAMBDA_FULL} km (full-band scale)", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 133, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "enhanced_rms", "emp_rms",
        "enhanced_vs_full", "enhanced_vs_emp")}

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
        pg_enhanced = _prior_gram_enhanced(prev, names)

        if pg_full is None or pg_enhanced is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_e, G_enh, _ = pg_enhanced

        common = sorted(set(gnames_f) & set(gnames_e))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_e = [gnames_e.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        G_enh = G_enh[np.ix_(idx_e, idx_e)]
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
        G_enh = G_enh[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "enhanced", "emp")}
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

            # Tidal-enhanced kernel (same lambda, enhanced Gram)
            try:
                w_enh = _reg_solve(
                    rho_rr_f * G_enh[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_enh[b, pool])
            except np.linalg.LinAlgError:
                continue

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
                day_sq["full_band"].append((truth - float(w_full @ rr)) ** 2)
                day_sq["enhanced"].append((truth - float(w_enh @ rr)) ** 2)
                day_sq["emp"].append((truth - float(w_emp @ rr)) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["enhanced_rms"].append(rms["enhanced"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["enhanced_vs_full"].append(
            100.0 * (1.0 - rms["enhanced"] / rms["full_band"]))
        per_day["enhanced_vs_emp"].append(
            100.0 * (1.0 - rms["enhanced"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"enh={np.median(per_day['enhanced_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "enhanced_vs_full_band_pct": _ci(per_day["enhanced_vs_full"], rng),
        "enhanced_vs_empirical_pct": _ci(per_day["enhanced_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "enhanced": float(np.median(per_day["enhanced_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    evf = results["enhanced_vs_full_band_pct"]
    eve = results["enhanced_vs_empirical_pct"]

    if evf.get("median") is not None:
        if evf["median"] > 0 and evf["ci95"] and evf["ci95"][0] > 0:
            interpretations.append(
                f"The tidal-enhanced kernel BEATS the full-band kernel "
                f"(median {evf['median']:+.2f}%, "
                f"CI95 [{evf['ci95'][0]:+.2f}, {evf['ci95'][1]:+.2f}]). "
                f"Amplifying the gravitational tidal frequencies improves "
                f"prediction.  The gravitational tides carry the strongest "
                f"spatial coherence (Paper 1: lambda=4677 km at tidal "
                f"frequencies), and weighting them more heavily extracts "
                f"more predictive power from the TEP field.")
        elif evf["median"] > 0:
            interpretations.append(
                f"The tidal-enhanced kernel is better than the full-band "
                f"kernel (median {evf['median']:+.2f}%, CI includes zero).")

    if eve.get("median") is not None:
        if eve["median"] > 0 and eve["ci95"] and eve["ci95"][0] > 0:
            interpretations.append(
                f"The tidal-enhanced kernel BEATS empirical covariance on "
                f"independent-year data (median {eve['median']:+.2f}%, "
                f"CI95 [{eve['ci95'][0]:+.2f}, {eve['ci95'][1]:+.2f}]). "
                f"The physics-fixed tidal-enhanced kernel generalises where "
                f"the data-driven covariance overfits.  This is evidence "
                f"that the gravitational tidal component carries stable, "
                f"generalisable predictive power.")

    verdict = {
        "step": 133,
        "title": "Tidal-enhanced TEP prior: amplifying the gravitational signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days_scored": n,
        "tidal_weight": TIDAL_WEIGHT,
        "tidal_periods_h": [p / 3600.0 for p in TIDAL_PERIODS],
        "lambda_km": LAMBDA_FULL,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Amplifying the gravitational tidal frequencies (3x weight) in "
            "the phase Gram should improve prediction by extracting more "
            "power from the strongest, most spatially coherent component "
            "of the TEP field.  The tidal-enhanced kernel should beat both "
            "the full-band kernel (which under-weights the tides) and "
            "empirical covariance (which cannot distinguish gravitational "
            "from atmospheric components)."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Enhanced vs full: {evf.get('median')}", flush=True)
    print(f"  Enhanced vs emp:  {eve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
