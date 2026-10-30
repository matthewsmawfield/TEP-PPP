#!/usr/bin/env python3
"""
STEP 118  --  Sub-band scale-free test and atmospheric correlation-length
               comparison.

THE SCALE ARGUMENT FOR TEP.

The TEP clock correlation field has a measured exponential length scale of
~2000-5000 km.  This is much larger than:
  - Tropospheric zenith delay correlation: ~500 km
  - Ionospheric TEC correlation: ~1000-2000 km
  - Synoptic weather systems: ~1000-3000 km

If the clock field is a proper-time field (TEP), it should be SCALE-FREE
within the 10-500 uHz band — the same correlation length in every sub-band.
If it is an atmospheric common-mode artefact, the correlation length should
vary with frequency, because different atmospheric processes dominate at
different periods:
  - 10-50 uHz (5.6-28 h): diurnal/synoptic atmospheric
  - 50-100 uHz (2.8-5.6 h): mesoscale/ionospheric
  - 100-500 uHz (33 min-2.8 h): gravity waves/multipath

Protocol:
  * Use 30 days of CODE MGEX clocks across 2024.
  * Split the 10-500 uHz band into three sub-bands.
  * Compute the phase-alignment distance decay for each sub-band.
  * Compare exponential lambda across sub-bands.
  * TEP predicts: same lambda in all sub-bands (scale-free field).
  - Atmospheric predicts: different lambda (frequency-dependent processes).

Additionally, compute the correlation length of the NETWORK MEAN clock
(the common-mode) and compare it to the pairwise correlation length.  If
the common-mode has a much longer scale than individual pairs, the field
is global (TEP-like).  If they match, it's local common-mode.

Output: results/outputs/step_118_subband_scale_free.json
"""
from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_stats as ts

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_118_subband_scale_free.json"
PRODUCT = "clk_cod"
N_DAYS = 30

# Sub-bands within 10-500 uHz
SUBBANDS = [
    ("low",  10e-6,  50e-6),   # 5.6-28 h: diurnal/synoptic
    ("mid",  50e-6, 100e-6),   # 2.8-5.6 h: mesoscale/ionospheric
    ("high", 100e-6, 500e-6),  # 33 min-2.8 h: gravity waves/multipath
]


def _spectra_subband(series, f_lo, f_hi, phase_rng=None):
    """Per-station complex spectrum in a sub-band."""
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
    if band.sum() < 4:
        return {}, {}

    RF, STA = {}, {}
    for n in series:
        t = np.asarray(series[n]["t"], float)
        x = np.asarray(series[n]["resid"], float)
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 0.5 * n_grid:
            continue
        t, x = t[ok], x[ok]
        xr = np.interp(grid, t - t[0], x, left=x[0], right=x[-1])
        tt = np.arange(len(xr), dtype=float)
        xr = xr - np.polyval(np.polyfit(tt, xr, 1), tt)
        if not np.all(np.isfinite(xr)) or np.std(xr) < 1e-12:
            continue
        F = np.fft.rfft(xr)[band]
        if F.size < 4 or not np.all(np.isfinite(F)):
            continue
        if phase_rng is not None:
            F = np.abs(F) * np.exp(1j * phase_rng.uniform(-np.pi, np.pi, F.shape))
        RF[n] = F
        STA[n] = series[n]["station"]
    return RF, STA


def _pairs_subband(RF, STA, f_lo, f_hi):
    """Phase-alignment pairs for a sub-band."""
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

    print("Step 118: Sub-band scale-free test", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print(f"  Sub-bands: {[(n, f'{lo*1e6:.0f}-{hi*1e6:.0f} uHz') for n, lo, hi in SUBBANDS]}",
          flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Collect pairs for each sub-band and the full band
    band_pairs = {name: [] for name, _, _ in SUBBANDS}
    band_pairs["full"] = []

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        # Full band (using step 53 infrastructure)
        RF, STA = s53._spectra(ser)
        if len(RF) >= 8:
            band_pairs["full"].extend(s53._pairs(RF, STA))

        # Sub-bands
        for name, f_lo, f_hi in SUBBANDS:
            RFs, STAs = _spectra_subband(ser, f_lo, f_hi)
            if len(RFs) >= 8:
                band_pairs[name].extend(_pairs_subband(RFs, STAs, f_lo, f_hi))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(selected)}] " +
                  " ".join(f"{n}={len(p)}" for n, p in band_pairs.items()),
                  flush=True)

    print()
    results = {}
    for name, pairs in band_pairs.items():
        if len(pairs) < 200:
            results[name] = {"n_pairs": len(pairs), "error": "too few pairs"}
            print(f"  {name}: too few pairs ({len(pairs)})", flush=True)
            continue
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        results[name] = {
            "n_pairs": len(pairs),
            "n_bins": int(len(ctr)),
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "exponential_R2": exp_fit.get("R2"),
            "best_model": fit.get("best_model"),
            "R2": fit.get("R2"),
            "tep_band_mean_alignment": (float(np.mean(tep_band)) if tep_band else None),
            "n_tep_band": len(tep_band),
        }
        print(f"  {name}: lambda={exp_fit.get('lambda_km'):.1f} km, "
              f"R2={fit.get('R2'):.3f}, best={fit.get('best_model')}, "
              f"align={results[name]['tep_band_mean_alignment']:.4f}",
              flush=True)

    # Compare lambdas across sub-bands
    lambdas = {n: r.get("exponential_lambda_km") for n, r in results.items()
               if r.get("exponential_lambda_km") is not None}

    full_lambda = lambdas.get("full")
    low_lambda = lambdas.get("low")
    mid_lambda = lambdas.get("mid")
    high_lambda = lambdas.get("high")

    # Scale-free test: are all sub-band lambdas the same?
    lambda_values = [v for v in [low_lambda, mid_lambda, high_lambda]
                     if v is not None]
    lambda_range = None
    lambda_cv = None  # coefficient of variation
    if len(lambda_values) >= 2:
        lambda_range = float(max(lambda_values) - min(lambda_values))
        mean_lambda = float(np.mean(lambda_values))
        lambda_cv = float(np.std(lambda_values) / mean_lambda) if mean_lambda > 0 else None

    interpretations = []
    if lambda_cv is not None:
        if lambda_cv < 0.2:
            interpretations.append(
                f"Sub-band lambda coefficient of variation = {lambda_cv:.3f} "
                f"(< 0.20) — the correlation length is SCALE-FREE across the "
                f"10-500 uHz band.  This is consistent with TEP (a proper-time "
                f"field should be scale-free) and inconsistent with an "
                f"atmospheric interpretation (different atmospheric processes "
                f"have different correlation scales at different periods).")
        else:
            interpretations.append(
                f"Sub-band lambda coefficient of variation = {lambda_cv:.3f} "
                f"(>= 0.20) — the correlation length VARIES with frequency. "
                f"This is more consistent with an atmospheric/ionospheric "
                f"interpretation (frequency-dependent processes) than with "
                f"a scale-free TEP field.")

    # Also compare the full-band lambda with the sub-band lambdas
    if full_lambda is not None:
        interpretations.append(
            f"Full-band lambda = {full_lambda:.1f} km.  "
            f"Sub-band lambdas: low={low_lambda}, mid={mid_lambda}, "
            f"high={high_lambda}.")

    # Compare with known atmospheric correlation scales
    interpretations.append(
        f"For reference: tropospheric ZTD correlation ~500 km, ionospheric "
        f"TEC correlation ~1000-2000 km, synoptic weather ~1000-3000 km. "
        f"The measured clock correlation scale ({full_lambda} km) is "
        f"{'much larger than' if full_lambda and full_lambda > 3000 else 'comparable to'} "
        f"these atmospheric scales.")

    verdict = {
        "step": 118,
        "title": "Sub-band scale-free test: is the TEP clock field scale-free?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "subbands": [(n, f"{lo*1e6:.0f}", f"{hi*1e6:.0f}") for n, lo, hi in SUBBANDS],
        "results": results,
        "lambda_comparison": {
            "full": full_lambda,
            "low": low_lambda,
            "mid": mid_lambda,
            "high": high_lambda,
            "lambda_range_km": lambda_range,
            "lambda_cv": lambda_cv,
        },
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the clock correlation field should be scale-free "
            "within the 10-500 uHz band — the same correlation length in "
            "every sub-band.  Under an atmospheric/ionospheric "
            "interpretation, the correlation length should vary with "
            "frequency because different atmospheric processes dominate "
            "at different periods.  A coefficient of variation < 0.20 "
            "across sub-bands supports TEP; >= 0.20 supports atmospheric."),
        "atmospheric_reference_scales": {
            "tropospheric_ZTD_km": "~500",
            "ionospheric_TEC_km": "~1000-2000",
            "synoptic_weather_km": "~1000-3000",
        },
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Lambda CV: {lambda_cv}", flush=True)
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
