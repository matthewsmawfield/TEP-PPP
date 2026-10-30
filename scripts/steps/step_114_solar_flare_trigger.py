#!/usr/bin/env python3
"""
STEP 114  --  Solar flare / geomagnetic storm trigger: does the TEP clock
               correlation field change during a major storm?

F.1 of the experimental programme asks whether a strong geomagnetic storm
disturbs the GNSS clock correlation field in a way the TEP hypothesis would
predict (or, equally important, in a way it would not).  The strongest event
in the available 2024 corpus is the G4 (Severe) storm of 2024-03-24 (DOY 084),
when a CME from March 23 arrived at ~14:11 UTC producing Kp 8.7 conditions
that persisted for ~18 hours through March 25 (DOY 085).

Method:
  * Compute the Papers 1-2 phase-alignment index per station pair per day,
    using the same Step 53 infrastructure (linear detrend, 10-500 uHz band,
    40 log distance bins 50-13,000 km, AIC model selection).
  * Fit the distance decay (lambda, R^2, best model) for:
      - storm days:   DOY 084-085 (G4, Kp 8.7)
      - control days: DOY 080-082 (quiet, Kp <= 4)
      - recovery days: DOY 086-089 (post-storm)
  * Compare lambda, R^2, phase-alignment amplitude, and the null significance
    (per-station Fourier phase randomisation) across the three windows.
  * Also compute the EW/NS anisotropy ratio in the 500-800 km band for each
    window, since anisotropy is a key TEP signature.

The pre-registered prediction under a strict TEP-field hypothesis would be a
change in lambda or anisotropy during the storm.  Under a conventional
atmospheric/ionospheric interpretation, the storm would mainly affect
high-latitude stations and the amplitude of residuals rather than the
correlation length scale.  We report both.

This is a retrospective observational comparison, not a controlled experiment.
The storm is a natural event, and confounders (season, day-of-year geometry,
network composition) are addressed by using nearby control days rather than
days from a different season.

Output: results/outputs/step_114_solar_flare_trigger.json
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_stats as ts

# Reuse the Step 53 infrastructure exactly so the statistic cannot drift.
import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_114_solar_flare_trigger.json"
PRODUCT = "clk_cod"
YEAR = 2024

# Storm: 2024-03-24 (DOY 084), G4 Severe, Kp 8.7, CME arrival ~14:11 UTC.
# Storm persisted through DOY 085.
STORM_DOYS = ["2024084", "2024085"]
# Controls: quiet days DOY 080-082 (Kp <= 4, no storms).
CONTROL_DOYS = ["2024080", "2024081", "2024082"]
# Recovery: DOY 086-089 (post-storm, declining Kp).
RECOVERY_DOYS = ["2024086", "2024087", "2024088", "2024089"]

# Anisotropy band (from Step 89): 300-1500 km, EW vs NS.
ANISO_LO_KM, ANISO_HI_KM = 300.0, 1500.0


def _is_ew(az_deg, tol=45.0):
    """True if baseline azimuth is within tol of E-W (90 or 270 deg)."""
    a = az_deg % 180.0
    return abs(a - 90.0) <= tol


def _is_ns(az_deg, tol=45.0):
    """True if baseline azimuth is within tol of N-S (0 or 180 deg)."""
    a = az_deg % 180.0
    return a <= tol or (180.0 - a) <= tol


def _anisotropy_ratio(RF, STA):
    """EW minus NS mean phase-alignment in the 300-1500 km band."""
    nm = list(RF)
    ew_vals, ns_vals = [], []
    for i, j in itertools.combinations(range(len(nm)), 2):
        Fi, Fj = RF[nm[i]], RF[nm[j]]
        if Fi.shape != Fj.shape:
            continue
        d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
        if not (ANISO_LO_KM <= d <= ANISO_HI_KM):
            continue
        Ssum = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(Ssum) or Ssum == 0:
            continue
        cpsi = float(np.cos(np.angle(Ssum)))
        az = td.baseline_azimuth_midpoint_deg(STA[nm[i]], STA[nm[j]])
        if _is_ew(az):
            ew_vals.append(cpsi)
        elif _is_ns(az):
            ns_vals.append(cpsi)
    ew_mean = float(np.mean(ew_vals)) if ew_vals else 0.0
    ns_mean = float(np.mean(ns_vals)) if ns_vals else 0.0
    return {
        "ew_mean": ew_mean,
        "ns_mean": ns_mean,
        "ew_minus_ns": ew_mean - ns_mean,
        "n_ew": len(ew_vals),
        "n_ns": len(ns_vals),
    }


def _mean_residual_rms(series):
    """Mean band-limited residual RMS across stations (m)."""
    vals = []
    for n, v in series.items():
        r = np.asarray(v["resid"], float)
        ok = np.isfinite(r)
        if ok.sum() > 10:
            vals.append(float(np.std(r[ok])))
    return float(np.mean(vals)) if vals else 0.0


def _n_stations(series):
    return len(series)


def analyze_window(label, doys, n_null=100):
    """Compute phase-alignment curve, lambda fit, anisotropy, and nulls."""
    days = []
    for tag in doys:
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        if len(ser) >= 8:
            days.append((tag, ser))

    if not days:
        return {"label": label, "error": "no days with >= 8 stations"}

    all_pairs = []
    aniso_results = []
    rms_vals = []
    n_sta_vals = []
    for tag, ser in days:
        RF, STA = s53._spectra(ser)
        if len(RF) >= 8:
            all_pairs.extend(s53._pairs(RF, STA))
            aniso_results.append(_anisotropy_ratio(RF, STA))
        rms_vals.append(_mean_residual_rms(ser))
        n_sta_vals.append(_n_stations(ser))

    if len(all_pairs) < 200:
        return {"label": label, "error": f"only {len(all_pairs)} pairs",
                "n_days": len(days), "doys": doys}

    ctr, val, cnt = s53._curve(all_pairs)
    real = s53._fit(ctr, val, cnt)

    # Null: per-station Fourier phase randomisation
    null_r2 = []
    for i in range(n_null):
        rng = np.random.default_rng(107_000 + i)
        pn = []
        for tag, ser in days:
            RFn, STAn = s53._spectra(ser, phase_rng=rng)
            if len(RFn) >= 8:
                pn.extend(s53._pairs(RFn, STAn))
        if len(pn) < 200:
            continue
        f2 = s53._fit(*s53._curve(pn))
        r2 = f2.get("R2")
        if r2 is not None and np.isfinite(r2):
            null_r2.append(float(r2))

    r2_real = real.get("R2")
    if null_r2 and r2_real is not None:
        null_arr = np.asarray(null_r2)
        sd = float(np.std(null_arr))
        n_exc = int(np.sum(null_arr >= r2_real))
        sig = {
            "sigma": (float((r2_real - null_arr.mean()) / sd) if sd > 1e-6 else None),
            "n_null": len(null_arr),
            "n_exceeding": n_exc,
            "p_value": float((n_exc + 1) / (len(null_arr) + 1)),
            "null_R2_mean": float(null_arr.mean()),
            "null_R2_std": sd,
        }
    else:
        sig = {"sigma": None, "n_null": len(null_r2), "p_value": None}

    # Aggregate anisotropy
    ew_all = [a["ew_mean"] for a in aniso_results if a["n_ew"] > 0]
    ns_all = [a["ns_mean"] for a in aniso_results if a["n_ns"] > 0]
    ew_minus_ns_all = [a["ew_minus_ns"] for a in aniso_results
                       if a["n_ew"] > 0 and a["n_ns"] > 0]

    # The exponential lambda is the physically meaningful TEP length scale.
    # The power_law "lambda_km" is actually a dimensionless exponent, not a
    # length scale.  Report both but use the exponential for the summary.
    exp_fit = real.get("exponential") or {}
    exp_lambda = exp_fit.get("lambda_km")
    exp_r2 = exp_fit.get("R2")

    return {
        "label": label,
        "doys": doys,
        "n_days": len(days),
        "n_pairs": len(all_pairs),
        "n_bins": int(len(ctr)),
        "mean_n_stations": float(np.mean(n_sta_vals)) if n_sta_vals else 0,
        "mean_residual_rms_m": float(np.mean(rms_vals)) if rms_vals else 0.0,
        "fit": real,
        "lambda_km": exp_lambda,  # use exponential lambda as the canonical scale
        "R2": real.get("R2"),
        "best_model": real.get("best_model"),
        "exponential_lambda_km": exp_lambda,
        "exponential_R2": exp_r2,
        "delta_aic_exp_vs_best": real.get("delta_aic_exponential_vs_best"),
        "null_phase_randomise": sig,
        "anisotropy": {
            "ew_mean": float(np.mean(ew_all)) if ew_all else None,
            "ns_mean": float(np.mean(ns_all)) if ns_all else None,
            "ew_minus_ns_mean": (float(np.mean(ew_minus_ns_all))
                                 if ew_minus_ns_all else None),
            "per_day": aniso_results,
        },
        "binned_curve": {
            "distance_km": ctr.tolist(),
            "phase_alignment_index": val.tolist(),
            "n_pairs": cnt.tolist(),
        },
    }


def analyze_window_latstrat(label, doys, lat_cut=50.0):
    """Latitude-stratified phase-alignment analysis.

    Splits stations into high-latitude (|lat| > lat_cut) and low-latitude
    (|lat| < 30) subsets, computes the exponential lambda for each, and
    reports whether the storm response is latitude-dependent.

    A TEP field response should be latitude-independent; an auroral
    ionospheric response should be concentrated at high latitudes.
    """
    days = []
    for tag in doys:
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) >= 8:
            days.append((tag, ser))

    if not days:
        return {"label": label, "error": "no days"}

    def _lat_filtered_pairs(days, lat_lo, lat_hi):
        """Pairs where both stations have |lat| in [lat_lo, lat_hi)."""
        pairs = []
        for tag, ser in days:
            RF, STA = s53._spectra(ser)
            nm = list(RF)
            # Filter stations by latitude
            keep = set()
            for n in nm:
                lat = abs(STA[n].lat_deg)
                if lat_lo <= lat < lat_hi:
                    keep.add(n)
            if len(keep) < 6:
                continue
            for i, j in itertools.combinations(range(len(nm)), 2):
                ni, nj = nm[i], nm[j]
                if ni not in keep or nj not in keep:
                    continue
                Fi, Fj = RF[ni], RF[nj]
                if Fi.shape != Fj.shape:
                    continue
                d = td.great_circle_km(STA[ni], STA[nj])
                if not (50.0 <= d <= 13000.0):
                    continue
                Ssum = np.sum(np.conj(Fi) * Fj)
                if not np.isfinite(Ssum) or Ssum == 0:
                    continue
                pairs.append((d, float(np.cos(np.angle(Ssum)))))
        return pairs

    out = {"label": label}
    for sub_label, lat_lo, lat_hi in [
        ("high_lat", lat_cut, 90.0),
        ("low_lat", 0.0, 30.0),
    ]:
        pairs = _lat_filtered_pairs(days, lat_lo, lat_hi)
        if len(pairs) < 200:
            out[sub_label] = {"n_pairs": len(pairs), "error": "too few pairs"}
            continue
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        out[sub_label] = {
            "n_pairs": len(pairs),
            "n_bins": int(len(ctr)),
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "exponential_R2": exp_fit.get("R2"),
            "best_model": fit.get("best_model"),
            "R2": fit.get("R2"),
            "tep_band_mean_alignment": (float(np.mean(tep_band))
                                        if tep_band else None),
            "n_tep_band": len(tep_band),
        }
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 114: Solar flare / geomagnetic storm trigger analysis", flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print(f"  Storm: DOY 084-085 (2024-03-24/25, G4 Kp 8.7)", flush=True)
    print(f"  Control: DOY 080-082 (quiet)", flush=True)
    print(f"  Recovery: DOY 086-089 (post-storm)", flush=True)
    print()

    control = analyze_window("control_quiet", CONTROL_DOYS)
    print(f"  Control: {control.get('n_pairs', 0)} pairs, "
          f"exp_lambda={control.get('exponential_lambda_km')}, "
          f"R2={control.get('R2')}, best={control.get('best_model')}", flush=True)

    storm = analyze_window("storm_g4", STORM_DOYS)
    print(f"  Storm: {storm.get('n_pairs', 0)} pairs, "
          f"exp_lambda={storm.get('exponential_lambda_km')}, "
          f"R2={storm.get('R2')}, best={storm.get('best_model')}", flush=True)

    recovery = analyze_window("recovery", RECOVERY_DOYS)
    print(f"  Recovery: {recovery.get('n_pairs', 0)} pairs, "
          f"exp_lambda={recovery.get('exponential_lambda_km')}, "
          f"R2={recovery.get('R2')}, best={recovery.get('best_model')}", flush=True)

    # Latitude-stratified analysis (key TEP vs ionospheric discriminator)
    print(f"\n  Latitude stratification (|lat|>50 vs |lat|<30):", flush=True)
    control_lat = analyze_window_latstrat("control_lat", CONTROL_DOYS)
    storm_lat = analyze_window_latstrat("storm_lat", STORM_DOYS)
    for sub in ["high_lat", "low_lat"]:
        c = control_lat.get(sub, {})
        s = storm_lat.get(sub, {})
        cl = c.get("exponential_lambda_km")
        sl = s.get("exponential_lambda_km")
        print(f"    {sub}: control lambda={cl}, storm lambda={sl}, "
              f"n_control={c.get('n_pairs')}, n_storm={s.get('n_pairs')}",
              flush=True)

    # Comparison summary
    def _get(d, k):
        v = d.get(k)
        return float(v) if v is not None and np.isfinite(v) else None

    summary = {
        "control_lambda_km": _get(control, "lambda_km"),
        "storm_lambda_km": _get(storm, "lambda_km"),
        "recovery_lambda_km": _get(recovery, "lambda_km"),
        "control_R2": _get(control, "R2"),
        "storm_R2": _get(storm, "R2"),
        "recovery_R2": _get(recovery, "R2"),
        "control_residual_rms_m": _get(control, "mean_residual_rms_m"),
        "storm_residual_rms_m": _get(storm, "mean_residual_rms_m"),
        "recovery_residual_rms_m": _get(recovery, "mean_residual_rms_m"),
        "control_ew_minus_ns": (control.get("anisotropy", {}).get("ew_minus_ns_mean")),
        "storm_ew_minus_ns": (storm.get("anisotropy", {}).get("ew_minus_ns_mean")),
        "recovery_ew_minus_ns": (recovery.get("anisotropy", {}).get("ew_minus_ns_mean")),
    }

    # Delta lambda (storm vs control)
    cl, sl = summary["control_lambda_km"], summary["storm_lambda_km"]
    if cl is not None and sl is not None and cl > 0:
        summary["delta_lambda_pct"] = float((sl - cl) / cl * 100.0)
    else:
        summary["delta_lambda_pct"] = None

    # Delta R2
    cr, sr = summary["control_R2"], summary["storm_R2"]
    if cr is not None and sr is not None:
        summary["delta_R2_abs"] = float(sr - cr)
    else:
        summary["delta_R2_abs"] = None

    # Delta residual RMS
    crms, srms = summary["control_residual_rms_m"], summary["storm_residual_rms_m"]
    if crms is not None and srms is not None and crms > 0:
        summary["delta_rms_pct"] = float((srms - crms) / crms * 100.0)
    else:
        summary["delta_rms_pct"] = None

    # Interpretation
    interpretations = []
    dl = summary.get("delta_lambda_pct")
    if dl is not None:
        if abs(dl) > 20:
            direction = "shortening" if dl < 0 else "lengthening"
            interpretations.append(
                f"Exponential lambda changed by {dl:+.1f}% during the storm — a "
                f"{direction} of the correlation scale, which could indicate a "
                f"disturbance of the field or an atmospheric/ionospheric confound.")
        else:
            interpretations.append(
                f"Exponential lambda changed by only {dl:+.1f}% during the storm — "
                f"the correlation length scale is stable.")
    dr = summary.get("delta_R2_abs")
    if dr is not None:
        if abs(dr) > 0.05:
            interpretations.append(
                f"R^2 changed by {dr:+.3f} during the storm — the distance "
                f"structure {'strengthened' if dr > 0 else 'weakened'}.")
        else:
            interpretations.append(
                f"R^2 changed by only {dr:+.3f} during the storm — "
                f"the distance structure is stable.")
    drms = summary.get("delta_rms_pct")
    if drms is not None:
        if abs(drms) > 10:
            direction = "elevated" if drms > 0 else "reduced"
            interpretations.append(
                f"Residual RMS changed by {drms:+.1f}% during the storm — "
                f"{direction} clock noise, consistent with storm-driven "
                f"atmospheric/ionospheric perturbations.")
        else:
            interpretations.append(
                f"Residual RMS changed by only {drms:+.1f}% during the storm.")

    # Anisotropy change
    ca = summary.get("control_ew_minus_ns")
    sa = summary.get("storm_ew_minus_ns")
    if ca is not None and sa is not None:
        summary["delta_anisotropy_abs"] = float(sa - ca)
        if abs(sa - ca) > 0.02:
            interpretations.append(
                f"EW-NS anisotropy changed by {sa - ca:+.4f} during the storm.")
        else:
            interpretations.append(
                f"EW-NS anisotropy is stable (delta {sa - ca:+.4f}).")

    # Latitude-stratified interpretation (key TEP vs ionospheric discriminator)
    c_hi = control_lat.get("high_lat", {})
    s_hi = storm_lat.get("high_lat", {})
    c_lo = control_lat.get("low_lat", {})
    s_lo = storm_lat.get("low_lat", {})
    c_hi_l = c_hi.get("exponential_lambda_km")
    s_hi_l = s_hi.get("exponential_lambda_km")
    c_lo_l = c_lo.get("exponential_lambda_km")
    s_lo_l = s_lo.get("exponential_lambda_km")
    if c_hi_l and s_hi_l and c_lo_l and s_lo_l:
        hi_delta = (s_hi_l - c_hi_l) / c_hi_l * 100.0
        lo_delta = (s_lo_l - c_lo_l) / c_lo_l * 100.0
        summary["high_lat_delta_lambda_pct"] = float(hi_delta)
        summary["low_lat_delta_lambda_pct"] = float(lo_delta)
        if abs(hi_delta) > abs(lo_delta) * 1.5:
            interpretations.append(
                f"Latitude stratification: high-lat lambda changed by "
                f"{hi_delta:+.1f}% vs low-lat {lo_delta:+.1f}% — the storm "
                f"response is concentrated at high latitudes, consistent with "
                f"an auroral ionospheric effect rather than a global TEP field.")
        elif abs(lo_delta) > abs(hi_delta) * 1.5:
            interpretations.append(
                f"Latitude stratification: low-lat lambda changed by "
                f"{lo_delta:+.1f}% vs high-lat {hi_delta:+.1f}% — the storm "
                f"response is concentrated at low latitudes.")
        else:
            interpretations.append(
                f"Latitude stratification: high-lat lambda changed by "
                f"{hi_delta:+.1f}% vs low-lat {lo_delta:+.1f}% — the storm "
                f"response is approximately latitude-uniform, which is "
                f"consistent with a global field (TEP) but also with a "
                f"uniform ionospheric response.")

    verdict = {
        "step": 114,
        "title": "Solar flare / geomagnetic storm trigger: TEP field during G4 storm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "storm_event": {
            "date": "2024-03-24",
            "doy": "2024084",
            "level": "G4 (Severe)",
            "kp_max": 8.7,
            "cme_arrival_utc": "~14:11",
            "duration_hours": 18,
            "persisted_through": "2024-03-25 (DOY 085)",
        },
        "product": PRODUCT,
        "windows": {
            "control_quiet": control,
            "storm_g4": storm,
            "recovery": recovery,
        },
        "latitude_stratified": {
            "control": control_lat,
            "storm": storm_lat,
        },
        "summary": summary,
        "interpretation": interpretations,
        "design_note": (
            "Retrospective observational comparison.  The storm is a natural "
            "event; confounders (season, geometry, network composition) are "
            "addressed by using nearby control days.  A change in lambda or "
            "anisotropy during the storm is necessary but not sufficient for a "
            "TEP-field interpretation; atmospheric and ionospheric effects must "
            "be separated by latitude-stratified analysis and space-weather "
            "indices before any physical claim."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Delta lambda: {summary.get('delta_lambda_pct')}%", flush=True)
    print(f"  Delta R2: {summary.get('delta_R2_abs')}", flush=True)
    print(f"  Delta RMS: {summary.get('delta_rms_pct')}%", flush=True)
    print(f"  Delta aniso: {summary.get('delta_anisotropy_abs')}", flush=True)
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
