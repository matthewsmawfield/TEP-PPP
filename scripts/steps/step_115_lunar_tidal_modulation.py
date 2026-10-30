#!/usr/bin/env python3
"""
STEP 115  --  Lunar tidal modulation of the TEP clock correlation field.

THE KEY DISCRIMINATING EXPERIMENT.

If the GNSS clock correlation field is a proper-time field (TEP), it should
couple to the lunar tidal gravitational potential.  If it is an
atmospheric/ionospheric common-mode artefact, it should couple to the solar
cycle instead (24 h, 365.25 d) and show no lunar-month modulation.

The Moon's tidal acceleration varies by ~40% over the anomalistic month
(27.55 d) as the Moon moves between perigee and apogee.  The solar tidal
acceleration varies by ~10% over the year (perihelion to aphelion).  These
are independent periods — a lunar-month modulation of the clock field that
tracks the lunar tidal acceleration is a clean gravitational signature that
no atmospheric or ionospheric process produces.

Protocol:
  * Compute the phase-alignment index per station pair per 15-day window
    across the full 2024 year (366 days of CODE MGEX final clocks).
  * Fit the exponential lambda and the mean TEP-band phase alignment per
    window.
  * Compute the lunar tidal acceleration (from JPL ephemeris) and the solar
    tidal acceleration at each window centre.
  * Correlate lambda and phase-alignment amplitude with:
      - lunar tidal acceleration (TEP prediction: positive correlation)
      - solar tidal acceleration (atmospheric confound: should NOT correlate
        with lunar)
      - Earth-Moon distance (control: should anti-correlate with lunar tidal)
      - Earth-Sun distance (control)
  * Block permutation test for significance (lag-1 autocorrelation is high).

Pre-registered prediction: under TEP, the clock correlation length scale
lambda and/or the phase-alignment amplitude should correlate with the lunar
tidal acceleration at the anomalistic month period.  Under an atmospheric
interpretation, they should not.

Output: results/outputs/step_115_lunar_tidal_modulation.json
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_astronomical as ta

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_115_lunar_tidal_modulation.json"
PRODUCT = "clk_cod"
YEAR = 2024
WINDOW_DAYS = 15
STEP_DAYS = 10  # step the window by 10 days -> ~36 windows across the year
RNG = np.random.default_rng(20260906)


def _date_tag(year, doy):
    return f"{year}{doy:03d}"


def _doy_to_jd(year, doy):
    """Julian Date at 12h UT of the given DOY."""
    Y, M, D = year, 1, 0
    jd0 = (367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4)
           + np.floor(275 * M / 9) + D + 1721013.5)
    return jd0 + doy + 0.5


def analyze_window(date_tags, label):
    """Compute phase-alignment curve and exponential lambda for a window."""
    days = []
    for tag in date_tags:
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) >= 8:
            days.append(ser)

    if not days:
        return None

    all_pairs = []
    for ser in days:
        RF, STA = s53._spectra(ser)
        if len(RF) >= 8:
            all_pairs.extend(s53._pairs(RF, STA))

    if len(all_pairs) < 200:
        return None

    ctr, val, cnt = s53._curve(all_pairs)
    fit = s53._fit(ctr, val, cnt)
    exp_fit = fit.get("exponential") or {}

    # Mean phase alignment in the TEP band (1000-5000 km)
    tep_band = [(d, v) for d, v in all_pairs
                if 1000.0 <= d <= 5000.0]
    tep_mean = float(np.mean([v for _, v in tep_band])) if tep_band else 0.0

    # Mean phase alignment across all distances
    all_mean = float(np.mean([v for _, v in all_pairs]))

    return {
        "label": label,
        "n_days": len(days),
        "n_pairs": len(all_pairs),
        "n_bins": int(len(ctr)),
        "exponential_lambda_km": exp_fit.get("lambda_km"),
        "exponential_R2": exp_fit.get("R2"),
        "best_model": fit.get("best_model"),
        "R2": fit.get("R2"),
        "tep_band_mean_alignment": tep_mean,
        "all_distance_mean_alignment": all_mean,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 115: Lunar tidal modulation of the TEP clock correlation field",
          flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Window: {WINDOW_DAYS} days, step: {STEP_DAYS} days", flush=True)
    print()

    # Build windows across the full year
    all_doys = list(range(1, 367))
    windows = []
    start = 1
    while start + WINDOW_DAYS <= 367:
        win_doys = all_doys[start:start + WINDOW_DAYS]
        tags = [_date_tag(YEAR, d) for d in win_doys]
        jd_center = _doy_to_jd(YEAR, win_doys[WINDOW_DAYS // 2])

        # Lunar and solar tidal accelerations at window centre
        # Use the tep_astronomical functions (take datetime or epoch seconds)
        epoch = (jd_center - 2440587.5) * 86400.0  # JD to Unix epoch
        try:
            lunar_tidal = ta.lunar_tidal_accel(epoch)
            solar_tidal = ta.solar_tidal_accel(epoch)
            em_dist = ta.earth_moon_distance_km(epoch)
            es_dist = ta.earth_sun_distance_au(epoch)
        except Exception as e:
            print(f"  Skipping window at JD {jd_center}: {e}", flush=True)
            start += STEP_DAYS
            continue

        windows.append({
            "tags": tags,
            "jd_center": jd_center,
            "doy_center": win_doys[WINDOW_DAYS // 2],
            "lunar_tidal_accel": lunar_tidal,
            "solar_tidal_accel": solar_tidal,
            "earth_moon_dist_km": em_dist,
            "earth_sun_dist_au": es_dist,
        })
        start += STEP_DAYS

    print(f"  {len(windows)} windows across the year", flush=True)
    print()

    # Analyze each window
    results = []
    for i, w in enumerate(windows):
        res = analyze_window(w["tags"], f"window_{i}")
        if res is not None:
            res["jd_center"] = w["jd_center"]
            res["doy_center"] = w["doy_center"]
            res["lunar_tidal_accel"] = w["lunar_tidal_accel"]
            res["solar_tidal_accel"] = w["solar_tidal_accel"]
            res["earth_moon_dist_km"] = w["earth_moon_dist_km"]
            res["earth_sun_dist_au"] = w["earth_sun_dist_au"]
            results.append(res)
            print(f"  [{i+1}/{len(windows)}] DOY {res['doy_center']:3d}: "
                  f"lambda={res.get('exponential_lambda_km')}, "
                  f"R2={res.get('R2'):.3f}, "
                  f"tep_align={res['tep_band_mean_alignment']:.4f}, "
                  f"lunar_tidal={res['lunar_tidal_accel']:.4e}", flush=True)

    print(f"\n  {len(results)} windows with valid data", flush=True)

    if len(results) < 10:
        OUT.write_text(json.dumps({
            "step": 115, "error": "too few valid windows",
            "n_windows": len(results)}, indent=2))
        print("  ERROR: too few valid windows", flush=True)
        return

    # Extract time series
    lambdas = np.array([r["exponential_lambda_km"] for r in results
                        if r["exponential_lambda_km"] is not None
                        and np.isfinite(r["exponential_lambda_km"])], float)
    tep_aligns = np.array([r["tep_band_mean_alignment"] for r in results], float)
    lunar_tidals = np.array([r["lunar_tidal_accel"] for r in results], float)
    solar_tidals = np.array([r["solar_tidal_accel"] for r in results], float)
    em_dists = np.array([r["earth_moon_dist_km"] for r in results], float)
    es_dists = np.array([r["earth_sun_dist_au"] for r in results], float)
    doys = np.array([r["doy_center"] for r in results], float)

    # Align arrays — only use windows where lambda is finite
    good = np.array([r["exponential_lambda_km"] is not None
                     and np.isfinite(r["exponential_lambda_km"])
                     for r in results])
    lambdas = lambdas[good] if len(lambdas) == good.sum() else \
        np.array([r["exponential_lambda_km"] for r in results
                  if r["exponential_lambda_km"] is not None
                  and np.isfinite(r["exponential_lambda_km"])], float)

    # For correlation, need aligned arrays
    # Use tep_aligns (available for all) as the primary observable
    # since lambda fitting can fail for some windows

    def block_permutation_p(x, y, n_perm=500):
        """Block permutation p-value accounting for autocorrelation.

        Uses the Bartlett effective sample size to determine block size.
        """
        r_obs = float(np.corrcoef(x, y)[0, 1])
        # Lag-1 autocorrelation of x
        rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
        # Bartlett N_eff
        if abs(rho1) < 0.99:
            n_eff = len(x) * (1 - rho1) / (1 + rho1)
        else:
            n_eff = 2.0
        block_size = max(1, int(np.ceil(len(x) / max(n_eff, 2))))

        n_exc = 0
        for _ in range(n_perm):
            # Block permutation: permute blocks of size block_size
            n_blocks = len(x) // block_size
            if n_blocks < 2:
                # Fall back to full permutation
                yp = RNG.permutation(y)
            else:
                idx = np.arange(n_blocks) * block_size
                perm = RNG.permutation(n_blocks)
                yp = np.empty_like(y)
                for j, p in enumerate(perm):
                    yp[j * block_size:(j + 1) * block_size] = \
                        y[p * block_size:(p + 1) * block_size]
                # Handle remainder
                rem = len(x) % block_size
                if rem > 0:
                    yp[-rem:] = y[-rem:]
            r_perm = float(np.corrcoef(x, yp)[0, 1])
            if abs(r_perm) >= abs(r_obs):
                n_exc += 1
        return float((n_exc + 1) / (n_perm + 1)), r_obs, rho1, n_eff

    # Correlations
    correlations = {}

    # 1. TEP-band alignment vs lunar tidal acceleration (KEY TEST)
    p, r, rho1, n_eff = block_permutation_p(tep_aligns, lunar_tidals)
    correlations["tep_align_vs_lunar_tidal"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
        "n_windows": len(tep_aligns),
    }
    print(f"\n  TEP alignment vs lunar tidal: r={r:+.4f}, "
          f"p_block={p:.4f}, n_eff={n_eff:.1f}", flush=True)

    # 2. TEP-band alignment vs solar tidal acceleration (CONFOUND CHECK)
    p, r, rho1, n_eff = block_permutation_p(tep_aligns, solar_tidals)
    correlations["tep_align_vs_solar_tidal"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  TEP alignment vs solar tidal: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 3. TEP-band alignment vs Earth-Moon distance (CONTROL)
    p, r, rho1, n_eff = block_permutation_p(tep_aligns, em_dists)
    correlations["tep_align_vs_earth_moon_dist"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  TEP alignment vs EM distance: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 4. Lambda vs lunar tidal (if enough valid lambdas)
    if len(lambdas) >= 10:
        # Need to align lambdas with lunar_tidals
        lambdas_full = np.array([r["exponential_lambda_km"] if r["exponential_lambda_km"]
                                 is not None and np.isfinite(r["exponential_lambda_km"])
                                 else np.nan for r in results], float)
        ok = np.isfinite(lambdas_full)
        if ok.sum() >= 10:
            p, r, rho1, n_eff = block_permutation_p(
                lambdas_full[ok], lunar_tidals[ok])
            correlations["lambda_vs_lunar_tidal"] = {
                "r": r, "p_block_permutation": p,
                "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
                "n_windows": int(ok.sum()),
            }
            print(f"  Lambda vs lunar tidal: r={r:+.4f}, "
                  f"p_block={p:.4f}, n={int(ok.sum())}", flush=True)

    # 5. TEP alignment vs DOY (seasonal control)
    p, r, rho1, n_eff = block_permutation_p(tep_aligns, doys)
    correlations["tep_align_vs_doy"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  TEP alignment vs DOY (seasonal): r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 6. Partial correlation: TEP align vs lunar tidal, controlling for solar tidal
    # This removes any common seasonal trend
    from numpy.linalg import lstsq
    A = np.column_stack([solar_tidals, np.ones_like(solar_tidals)])
    resid_tep = tep_aligns - A @ lstsq(A, tep_aligns, rcond=None)[0]
    resid_lunar = lunar_tidals - A @ lstsq(A, lunar_tidals, rcond=None)[0]
    p, r, rho1, n_eff = block_permutation_p(resid_tep, resid_lunar)
    correlations["tep_align_vs_lunar_tidal_partial_solar"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
        "description": "Partial correlation: TEP alignment residual vs lunar tidal residual, after removing solar tidal from both",
    }
    print(f"  TEP align vs lunar (partial solar): r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 7. Partial correlation: TEP align vs lunar tidal, controlling for DOY
    A2 = np.column_stack([doys, np.ones_like(doys)])
    resid_tep2 = tep_aligns - A2 @ lstsq(A2, tep_aligns, rcond=None)[0]
    resid_lunar2 = lunar_tidals - A2 @ lstsq(A2, lunar_tidals, rcond=None)[0]
    p, r, rho1, n_eff = block_permutation_p(resid_tep2, resid_lunar2)
    correlations["tep_align_vs_lunar_tidal_partial_doy"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
        "description": "Partial correlation after removing linear DOY trend from both",
    }
    print(f"  TEP align vs lunar (partial DOY): r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # Interpretation
    key_r = correlations["tep_align_vs_lunar_tidal"]["r"]
    key_p = correlations["tep_align_vs_lunar_tidal"]["p_block_permutation"]
    partial_r = correlations["tep_align_vs_lunar_tidal_partial_solar"]["r"]
    partial_p = correlations["tep_align_vs_lunar_tidal_partial_solar"]["p_block_permutation"]

    # Seasonal trend analysis (secondary finding)
    doys = np.array([w["doy_center"] for w in results], float)
    lambdas = np.array([w["exponential_lambda_km"] for w in results], float)
    aligns = np.array([w["tep_band_mean_alignment"] for w in results], float)

    # Fit annual cycle: lambda = a + b * cos(2*pi*(doy - phi)/365.25)
    # Simple approach: correlate lambda with sin and cos of DOY
    sin_doy = np.sin(2 * np.pi * doys / 365.25)
    cos_doy = np.cos(2 * np.pi * doys / 365.25)
    r_lambda_sin = float(np.corrcoef(lambdas, sin_doy)[0, 1])
    r_lambda_cos = float(np.corrcoef(lambdas, cos_doy)[0, 1])
    r_align_sin = float(np.corrcoef(aligns, sin_doy)[0, 1])
    r_align_cos = float(np.corrcoef(aligns, cos_doy)[0, 1])

    # Summer/winter split
    summer_mask = (doys > 150) & (doys < 240)
    winter_mask = (doys < 80) | (doys > 330)
    summer_mean = float(lambdas[summer_mask].mean()) if summer_mask.sum() > 0 else None
    winter_mean = float(lambdas[winter_mask].mean()) if winter_mask.sum() > 0 else None
    lambda_cv = float(lambdas.std() / lambdas.mean()) if lambdas.mean() > 0 else None

    seasonal = {
        "lambda_cv_across_year": lambda_cv,
        "lambda_summer_mean_km": summer_mean,
        "lambda_winter_mean_km": winter_mean,
        "lambda_vs_sin_doy_r": r_lambda_sin,
        "lambda_vs_cos_doy_r": r_lambda_cos,
        "align_vs_sin_doy_r": r_align_sin,
        "align_vs_cos_doy_r": r_align_cos,
        "n_summer": int(summer_mask.sum()),
        "n_winter": int(winter_mask.sum()),
    }
    print(f"\n  Seasonal: lambda CV={lambda_cv:.3f}, summer={summer_mean:.0f}, "
          f"winter={winter_mean:.0f}", flush=True)
    print(f"  Lambda vs cos(DOY): r={r_lambda_cos:+.3f}, vs sin(DOY): "
          f"r={r_lambda_sin:+.3f}", flush=True)

    seasonal_note = ""
    if summer_mean and winter_mean and abs(summer_mean - winter_mean) > 500:
        seasonal_note = (
            f"  Secondary finding: lambda shows a strong seasonal cycle "
            f"(summer mean {summer_mean:.0f} km vs winter mean "
            f"{winter_mean:.0f} km, CV={lambda_cv:.3f}).  This seasonal "
            f"variation is much larger than any lunar modulation and is more "
            f"consistent with a solar-illumination-driven atmospheric/"
            f"ionospheric process than with a gravitational field.")

    if key_p < 0.05 and partial_p < 0.10:
        interpretation = (
            f"The TEP clock correlation field shows a significant lunar tidal "
            f"modulation (r={key_r:+.3f}, p={key_p:.3f}), remaining after "
            f"removing the solar tidal component (partial r={partial_r:+.3f}, "
            f"p={partial_p:.3f}).  This is consistent with a gravitational/"
            f"proper-time coupling and is not expected from atmospheric or "
            f"ionospheric common-mode effects, which track the solar cycle."
            f"{seasonal_note}")
    elif key_p < 0.05:
        interpretation = (
            f"The TEP clock correlation field shows a lunar tidal modulation "
            f"(r={key_r:+.3f}, p={key_p:.3f}), but the partial correlation "
            f"after removing solar tidal is not significant (partial "
            f"r={partial_r:+.3f}, p={partial_p:.3f}).  The lunar signal may "
            f"share a seasonal trend with the solar cycle."
            f"{seasonal_note}")
    else:
        interpretation = (
            f"No significant lunar tidal modulation of the TEP clock "
            f"correlation field (r={key_r:+.3f}, p={key_p:.3f}).  Under TEP "
            f"this is a null result; under an atmospheric interpretation it "
            f"is expected.  Note that with one year of data and high "
            f"autocorrelation, the effective sample size is small and a "
            f"decade of data may be needed to reach significance."
            f"{seasonal_note}")

    verdict = {
        "step": 115,
        "title": "Lunar tidal modulation of the TEP clock correlation field",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "window_days": WINDOW_DAYS,
        "step_days": STEP_DAYS,
        "n_windows": len(results),
        "windows": [{"jd_center": r["jd_center"],
                      "doy_center": r["doy_center"],
                      "n_pairs": r["n_pairs"],
                      "exponential_lambda_km": r["exponential_lambda_km"],
                      "R2": r["R2"],
                      "tep_band_mean_alignment": r["tep_band_mean_alignment"],
                      "lunar_tidal_accel": r["lunar_tidal_accel"],
                      "solar_tidal_accel": r["solar_tidal_accel"],
                      "earth_moon_dist_km": r["earth_moon_dist_km"],
                      "earth_sun_dist_au": r["earth_sun_dist_au"]}
                     for r in results],
        "correlations": correlations,
        "seasonal_analysis": seasonal,
        "interpretation": interpretation,
        "pre_registered_prediction": (
            "Under TEP, the clock correlation length scale and/or phase-"
            "alignment amplitude should correlate with the lunar tidal "
            "acceleration at the anomalistic month period (27.55 d).  Under "
            "an atmospheric/ionospheric interpretation, they should not — "
            "atmospheric effects track the solar cycle (24 h, 365.25 d), "
            "not the lunar month."),
        "design_note": (
            "The lunar tidal acceleration varies by ~40% over the anomalistic "
            "month, while the solar tidal acceleration varies by ~10% over "
            "the year.  These are independent periods.  A lunar-month "
            "modulation that tracks lunar_tidal_accel is a gravitational "
            "signature that no atmospheric process produces.  Block "
            "permutation accounts for the high lag-1 autocorrelation."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Interpretation: {interpretation}", flush=True)
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
