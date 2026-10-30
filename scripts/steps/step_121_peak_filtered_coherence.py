#!/usr/bin/env python3
"""
STEP 121  --  Peak-filtered distance coherence: continuum vs solar tidal lines.

THE CRITICAL TEST.

Step 119 showed that the network mean clock spectrum has strong peaks at
24h, 12h, 8h, and 6h (solar tides) on top of a 1/f^0.3 continuum.  Step 118
showed that the distance-structured coherence is dominated by the low-
frequency band (10-50 uHz, 5.6-28 h), which contains the 24h and 12h lines.

This experiment asks: does the distance-structured phase alignment survive
after the solar tidal lines are explicitly filtered out?

Protocol:
  * For each day of CODE MGEX data (30 days), compute per-station clock
    residual time series.
  * For each station, apply a notch filter that removes the 24h, 12h, 8h,
    and 6h periods (and their first harmonics).
  * Compute the phase-alignment distance decay on the filtered residuals.
  * Compare with the unfiltered distance decay.

Predictions:
  * TEP: the distance structure should survive, because a proper-time
    field is not concentrated at solar tidal periods.
  * Atmospheric: the distance structure should weaken or disappear,
    because the coherence is carried by the solar tidal lines.

Additionally, compute the distance decay using ONLY the continuum (masking
the peak regions in the frequency domain, as in Step 119's 1/f fit).

Output: results/outputs/step_121_peak_filtered_coherence.json
"""
from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import signal as scipy_signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_stats as ts

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_121_peak_filtered_coherence.json"
PRODUCT = "clk_cod"
N_DAYS = 30
F_LO, F_HI = 10e-6, 500e-6

# Periods to notch out (seconds)
NOTCH_PERIODS = [
    86400.0,   # 24h S1
    43200.0,   # 12h S2
    28800.0,   # 8h  S3
    21600.0,   # 6h  S4
    86164.0,   # 23h56m sidereal
    44712.0,   # 12h25m lunar M2
]
NOTCH_WIDTH_FRAC = 0.05  # ±5% around each period


def _spectra_notched(series, notch_freqs=None, f_lo=F_LO, f_hi=F_HI):
    """Per-station complex spectrum with optional notch filtering.

    If notch_freqs is provided, frequencies within ±NOTCH_WIDTH_FRAC of
    each notch frequency are zeroed in the spectrum before band selection.
    """
    dts = []
    for n in list(series)[:8]:
        t = np.asarray(series[n]["t"], float)
        if len(t) > 4:
            d = np.median(np.diff(t))
            if np.isfinite(d) and d > 0:
                dts.append(d)
    dt = float(np.median(dts)) if dts else 30.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 30.0
    n_grid = max(64, min(int(round(86400.0 / dt)), 2880))
    grid = np.arange(n_grid) * dt
    freq = np.fft.rfftfreq(n_grid, d=dt)
    band = (freq >= f_lo) & (freq <= f_hi)
    if band.sum() < 6:
        return {}, {}

    # Build notch mask
    if notch_freqs is not None:
        notch_mask = np.zeros(len(freq), dtype=bool)
        for fn in notch_freqs:
            if fn > 0:
                lo = fn * (1.0 - NOTCH_WIDTH_FRAC)
                hi = fn * (1.0 + NOTCH_WIDTH_FRAC)
                notch_mask |= (freq >= lo) & (freq <= hi)
        band &= ~notch_mask

    RF, STA = {}, {}
    for n in series:
        x_raw = np.asarray(series[n]["resid"], float)
        if np.nanstd(x_raw) > 20.0:
            continue
        t = np.asarray(series[n]["t"], float)
        x = x_raw
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 0.5 * n_grid:
            continue
        t, x = t[ok], x[ok]
        xr = np.interp(grid, t - t[0], x, left=x[0], right=x[-1])
        tt = np.arange(len(xr), dtype=float)
        xr = xr - np.polyval(np.polyfit(tt, xr, 1), tt)
        if not np.all(np.isfinite(xr)) or np.std(xr) < 1e-12:
            continue
        F = np.fft.rfft(xr)
        F[~band] = 0.0  # zero out non-band and notched frequencies
        # Only keep if we have enough in-band bins
        if band.sum() < 4:
            continue
        Fb = F[band]
        if Fb.size < 4 or not np.all(np.isfinite(Fb)):
            continue
        # Check that the notched spectrum still has signal
        if np.max(np.abs(Fb)) < 1e-10:
            continue
        RF[n] = Fb
        STA[n] = series[n]["station"]
    return RF, STA


def _pairs(RF, STA):
    nm = list(RF)
    out = []
    for i, j in itertools.combinations(range(len(nm)), 2):
        Fi, Fj = RF[nm[i]], RF[nm[j]]
        if Fi.shape != Fj.shape:
            continue
        d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
        if not (50.0 <= d <= 13000.0):
            continue
        Ssum = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(Ssum) or Ssum == 0:
            continue
        out.append((d, float(np.cos(np.angle(Ssum)))))
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 121: Peak-filtered distance coherence", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print(f"  Notch periods: {[f'{p/3600:.1f}h' for p in NOTCH_PERIODS]}", flush=True)
    print()

    # Convert notch periods to frequencies
    notch_freqs = [1.0 / p for p in NOTCH_PERIODS if p > 0]

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    full_pairs = []
    notched_pairs = []
    continuum_pairs = []  # notched = continuum (same thing here)

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        # Full band (no notch)
        RF, STA = s53._spectra(ser)
        if len(RF) >= 8:
            full_pairs.extend(_pairs(RF, STA))

        # Notched (solar tidal lines removed)
        RFn, STAn = _spectra_notched(ser, notch_freqs=notch_freqs)
        if len(RFn) >= 8:
            notched_pairs.extend(_pairs(RFn, STAn))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(selected)}] full={len(full_pairs)}, "
                  f"notched={len(notched_pairs)}", flush=True)

    print()
    results = {}

    for label, pairs in [("full_band", full_pairs), ("notched", notched_pairs)]:
        if len(pairs) < 200:
            results[label] = {"n_pairs": len(pairs), "error": "too few"}
            print(f"  {label}: too few pairs ({len(pairs)})", flush=True)
            continue
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        results[label] = {
            "n_pairs": len(pairs),
            "n_bins": int(len(ctr)),
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "exponential_R2": exp_fit.get("R2"),
            "best_model": fit.get("best_model"),
            "R2": fit.get("R2"),
            "tep_band_mean_alignment": (float(np.mean(tep_band))
                                        if tep_band else None),
            "n_tep_band": len(tep_band),
            "binned_curve": {
                "distance_km": ctr.tolist(),
                "phase_alignment_index": val.tolist(),
                "n_pairs": cnt.tolist(),
            },
        }
        print(f"  {label}: lambda={exp_fit.get('lambda_km')}, "
              f"R2={fit.get('R2'):.3f}, best={fit.get('best_model')}, "
              f"align={results[label]['tep_band_mean_alignment']:.4f}",
              flush=True)

    # Compare
    full_r2 = results.get("full_band", {}).get("R2")
    notched_r2 = results.get("notched", {}).get("R2")
    full_lambda = results.get("full_band", {}).get("exponential_lambda_km")
    notched_lambda = results.get("notched", {}).get("exponential_lambda_km")
    full_align = results.get("full_band", {}).get("tep_band_mean_alignment")
    notched_align = results.get("notched", {}).get("tep_band_mean_alignment")

    interpretations = []

    if full_r2 is not None and notched_r2 is not None:
        delta_r2 = notched_r2 - full_r2
        if abs(delta_r2) < 0.05:
            interpretations.append(
                f"R² changed by only {delta_r2:+.3f} after notching — the "
                f"distance structure is NOT carried by the solar tidal lines. "
                f"This is consistent with TEP (the field is not concentrated "
                f"at solar periods) and inconsistent with a purely solar-"
                f"tidal atmospheric interpretation.")
        elif delta_r2 < -0.05:
            interpretations.append(
                f"R² dropped by {delta_r2:+.3f} after notching — the distance "
                f"structure is partly carried by the solar tidal lines. "
                f"This is more consistent with an atmospheric interpretation.")
        else:
            interpretations.append(
                f"R² increased by {delta_r2:+.3f} after notching — removing "
                f"the solar tidal lines actually improved the distance fit, "
                f"suggesting the tidal lines add noise to the pairwise "
                f"coherence without contributing distance structure.")

    if full_align is not None and notched_align is not None:
        delta_align = notched_align - full_align
        interpretations.append(
            f"TEP-band alignment changed by {delta_align:+.4f} after notching "
            f"(from {full_align:.4f} to {notched_align:.4f}).")

    if full_lambda is not None and notched_lambda is not None:
        lambda_ratio = notched_lambda / full_lambda if full_lambda > 0 else None
        if lambda_ratio is not None:
            interpretations.append(
                f"Lambda ratio (notched/full) = {lambda_ratio:.3f}. "
                f"{'The correlation length is stable.' if 0.8 < lambda_ratio < 1.2 else 'The correlation length changed substantially.'}")

    verdict = {
        "step": 121,
        "title": "Peak-filtered distance coherence: continuum vs solar tidal lines",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "notch_periods_h": [p / 3600.0 for p in NOTCH_PERIODS],
        "notch_width_frac": NOTCH_WIDTH_FRAC,
        "results": results,
        "comparison": {
            "full_R2": full_r2,
            "notched_R2": notched_r2,
            "full_lambda_km": full_lambda,
            "notched_lambda_km": notched_lambda,
            "full_tep_alignment": full_align,
            "notched_tep_alignment": notched_align,
        },
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the distance-structured phase alignment should "
            "survive after removing the 24h, 12h, 8h, and 6h solar tidal "
            "lines, because a proper-time field is not concentrated at "
            "solar periods.  Under an atmospheric/ionospheric interpretation, "
            "the distance structure should weaken or disappear, because the "
            "coherence is carried by the solar tidal lines."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
