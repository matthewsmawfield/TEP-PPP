#!/usr/bin/env python3
"""
STEP 127  --  Notched TEP prior: exploiting the solar-illumination-independent
              TEP continuum for out-of-year generalisation.

EXPLOITING THE ATMOSPHERIC MODULATION AS A STRENGTH.

Step 92 showed that the full-band TEP kernel fails against empirical
covariance on 2025 held-out data (median -0.93%, CI includes zero).
Step 125 showed that the full-band solar elevation dependence (r=0.809)
is carried by the solar tidal lines; after notching, the TEP continuum
is solar-illumination-independent (r=0.334, day/night ratio inverts).

HYPOTHESIS: The full-band TEP kernel's failure on independent-year data
is caused by the solar tidal lines, which are atmospheric modulations
that may not generalise year-to-year.  The notched TEP continuum — the
solar-illumination-independent component — should generalise better.

PROTOCOL:
  * Run the Step 64/92 P1 gate on 2025 held-out data with THREE kernels:
    1. full_band: the registered full-band TEP kernel (baseline, Step 92)
    2. notched:   the TEP continuum with solar tidal lines removed
    3. empirical: the data-driven covariance (conventional comparator)
  * Compare notched vs full_band and notched vs empirical.

  * TEP exploitation prediction: notched beats full_band on independent-
    year data (the atmospheric modulation was the source of the failure),
    and notched beats empirical covariance (the TEP continuum is a stable
    physics-fixed structure that generalises where the data-driven
    covariance overfits).

Output: results/outputs/step_127_notched_prior.json
"""
from __future__ import annotations

import sys, json, hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib.tep_phase import spectral_phase_gram
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_127_notched_prior.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2025

# Same evaluation DOYs as Step 92 (frozen)
EVALUATION_DOYS = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260902

# Notch periods (seconds) — same as Step 121/125
NOTCH_PERIODS = [86400.0, 43200.0, 28800.0, 21600.0, 86164.0, 44712.0]
NOTCH_WIDTH_FRAC = 0.05

# Notched TEP kernel: use the Step 121 notched lambda (1877 km)
# as the physics-fixed scale for the notched continuum.
NOTCHED_LAMBDA_KM = 1877.0
NOTCHED_MULTI_SCALES_KM = (NOTCHED_LAMBDA_KM / 1.6,
                           NOTCHED_LAMBDA_KM,
                           NOTCHED_LAMBDA_KM * 1.6)


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _notched_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                        min_bins=6, min_low_cycles=0.85):
    """Phase Gram with solar tidal lines notched out.

    Identical to spectral_phase_gram but excludes frequency bins within
    ±NOTCH_WIDTH_FRAC of each notch frequency.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history is too short to resolve the registered low frequency")

    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero station scale in phase window")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero detrended scale in phase window")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    band = (freq >= f_lo_hz) & (freq <= f_hi_hz)

    # Notch out solar tidal lines
    notch_freqs = [1.0 / p for p in NOTCH_PERIODS if p > 0]
    notch_mask = np.zeros(len(freq), dtype=bool)
    for fn in notch_freqs:
        lo = fn * (1.0 - NOTCH_WIDTH_FRAC)
        hi = fn * (1.0 + NOTCH_WIDTH_FRAC)
        notch_mask |= (freq >= lo) & (freq <= hi)
    band &= ~notch_mask

    if int(band.sum()) < int(min_bins):
        raise ValueError("history provides too few Fourier bins after notching")
    F = F[band]

    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero spectral energy in at least one station after notching")
    norm = np.sqrt(np.outer(energy, energy))
    complex_coherence = cross / norm
    predictive_gram = np.real(complex_coherence)
    predictive_gram = 0.5 * (predictive_gram + predictive_gram.T)
    np.fill_diagonal(predictive_gram, 1.0)
    return predictive_gram


def _rho_multi_notched(d, lam_scale=1.0):
    """Multi-scale exponential covariance using the notched TEP continuum scale."""
    return np.mean([np.exp(-d / (L * lam_scale))
                    for L in NOTCHED_MULTI_SCALES_KM], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _prior_gram_notched(prev_day, cur_names):
    """Same as s64.prior_gram but uses the notched phase Gram."""
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
        g = _notched_phase_gram(np.column_stack(spectral_cols), s64.DT_S)
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

    print("Step 127: Notched TEP prior for out-of-year generalisation", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Notched lambda: {NOTCHED_LAMBDA_KM} km", flush=True)
    print(f"  Notch periods: {[f'{p/3600:.1f}h' for p in NOTCH_PERIODS]}", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    missing = [tag for tag in required_tags if _input_path(tag) is None]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs; "
              f"need {MIN_VALID_DAYS}", flush=True)
        OUT.write_text(json.dumps({"step": 127, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "notched_rms", "emp_rms",
        "notched_vs_full", "notched_vs_emp")}

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

        # Full-band phase Gram (registered TEP)
        pg_full = s64.prior_gram(prev, names)
        # Notched phase Gram (TEP continuum)
        pg_notched = _prior_gram_notched(prev, names)

        if pg_full is None or pg_notched is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_n, G_notch, P_notch = pg_notched

        # Use the intersection of stations admitted by both Grams
        common = sorted(set(gnames_f) & set(gnames_n))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_n = [gnames_n.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        G_notch = G_notch[np.ix_(idx_n, idx_n)]
        P = P_full[:, idx_f]  # use the full-band prior raw for empirical cov
        Dg = D[np.ix_(idx_cur, idx_cur)]
        gnames = common

        # Causal detrend of current-day residuals
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            t_s = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            win_s = 7200.0
            for ki in range(len(clk_ns)):
                tk = t_s[ki]
                mask = (t_s <= tk) & (t_s >= tk - win_s) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(t_s[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS

        # Detrend prior-day raw
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
            continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        G_notch = G_notch[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "notched", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band TEP kernel (registered, lambda=4200 km)
            rho_rr_full = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_full = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_full * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_full * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Notched TEP kernel (TEP continuum, lambda=1877 km)
            rho_rr_notch = _rho_multi_notched(Dg[np.ix_(pool, pool)])
            rho_rt_notch = _rho_multi_notched(Dg[b, pool])
            try:
                w_notch = _reg_solve(
                    rho_rr_notch * G_notch[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_notch * G_notch[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Empirical covariance (conventional comparator)
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

            B = R[:, b]
            A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                p_full = float(w_full @ rr)
                p_notch = float(w_notch @ rr)
                p_emp = float(w_emp @ rr)
                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["notched"].append((truth - p_notch) ** 2)
                day_sq["emp"].append((truth - p_emp) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["notched_rms"].append(rms["notched"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["notched_vs_full"].append(
            100.0 * (1.0 - rms["notched"] / rms["full_band"]))
        per_day["notched_vs_emp"].append(
            100.0 * (1.0 - rms["notched"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"notch={np.median(per_day['notched_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    print(f"\n  {len(per_day['full_band_rms'])} days scored", flush=True)

    results = {
        "n_days": len(per_day["full_band_rms"]),
        "notched_vs_full_band_pct": _ci(per_day["notched_vs_full"], rng),
        "notched_vs_empirical_pct": _ci(per_day["notched_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"]))
            if per_day["full_band_rms"] else None,
            "notched": float(np.median(per_day["notched_rms"]))
            if per_day["notched_rms"] else None,
            "empirical": float(np.median(per_day["emp_rms"]))
            if per_day["emp_rms"] else None,
        },
    }

    # Interpretation
    interpretations = []
    nvf = results["notched_vs_full_band_pct"]
    nve = results["notched_vs_empirical_pct"]

    if nvf.get("median") is not None:
        if nvf["median"] > 0 and nvf["ci95"] and nvf["ci95"][0] > 0:
            interpretations.append(
                f"The notched TEP continuum BEATS the full-band TEP kernel on "
                f"independent-year data (median {nvf['median']:+.2f}%, "
                f"CI95 [{nvf['ci95'][0]:+.2f}, {nvf['ci95'][1]:+.2f}]). "
                f"The full-band kernel's failure on 2025 data (Step 92: "
                f"-0.93%) was caused by the solar tidal lines — atmospheric "
                f"modulations that do not generalise year-to-year.  Removing "
                f"them yields a TEP continuum that generalises better.  This "
                f"directly exploits the Step 125 finding as a predictive "
                f"strength.")
        elif nvf["median"] > 0:
            interpretations.append(
                f"The notched TEP continuum is better than the full-band TEP "
                f"kernel on independent-year data (median {nvf['median']:+.2f}%, "
                f"but CI95 includes zero).  The improvement is in the expected "
                f"direction but not significant.")
        else:
            interpretations.append(
                f"The notched TEP continuum does NOT beat the full-band TEP "
                f"kernel on independent-year data (median {nvf['median']:+.2f}%). "
                f"The solar tidal lines were not the cause of the full-band "
                f"failure.")

    if nve.get("median") is not None:
        if nve["median"] > 0 and nve["ci95"] and nve["ci95"][0] > 0:
            interpretations.append(
                f"The notched TEP continuum BEATS empirical covariance on "
                f"independent-year data (median {nve['median']:+.2f}%, "
                f"CI95 [{nve['ci95'][0]:+.2f}, {nve['ci95'][1]:+.2f}]). "
                f"The physics-fixed TEP continuum (lambda=1877 km, frozen "
                f"from Step 121) generalises where the data-driven covariance "
                f"overfits.  This would be the first positive P1 result on "
                f"independent-year data.")
        elif nve["median"] > 0:
            interpretations.append(
                f"The notched TEP continuum is better than empirical "
                f"covariance (median {nve['median']:+.2f}%, but CI95 "
                f"includes zero).  The improvement is in the expected "
                f"direction but not significant.")
        else:
            interpretations.append(
                f"The notched TEP continuum does NOT beat empirical "
                f"covariance (median {nve['median']:+.2f}%).  The "
                f"physics-fixed kernel does not generalise better than "
                f"the data-driven covariance.")

    verdict = {
        "step": 127,
        "title": "Notched TEP prior: exploiting the solar-illumination-"
                 "independent TEP continuum for out-of-year generalisation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_evaluation_days": len(eval_tags),
        "n_days_scored": results["n_days"],
        "notch_periods_h": [p / 3600.0 for p in NOTCH_PERIODS],
        "notch_width_frac": NOTCH_WIDTH_FRAC,
        "notched_lambda_km": NOTCHED_LAMBDA_KM,
        "full_band_lambda_km": s64.LAMBDA_BASE_KM,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The full-band TEP kernel fails on 2025 data (Step 92: -0.93%) "
            "because the solar tidal lines are atmospheric modulations that "
            "do not generalise year-to-year.  The notched TEP continuum "
            "(lambda=1877 km, solar-illumination-independent per Step 125) "
            "should beat both the full-band TEP kernel and the empirical "
            "covariance on independent-year data.  This exploits the "
            "atmospheric modulation finding as a predictive strength: by "
            "removing the modulations, we isolate the stable TEP continuum "
            "that generalises across years."),
        "reference": {
            "step_92_full_band_vs_emp": -0.93,
            "step_121_notched_lambda_km": 1877.0,
            "step_125_notched_day_night_inversion": True,
        },
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Notched vs full: {nvf.get('median')}", flush=True)
    print(f"  Notched vs emp:  {nve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
