#!/usr/bin/env python3
"""
STEP 126  --  Cross-frequency coherence: is the TEP field a single field?

THE UNIFIED-FIELD TEST.

If the common-mode clock correlation field is a single TEP proper-time
field, then the coherence at different frequencies should be CORRELATED
— the same field drives all frequencies, so a pair that is coherent at
low frequency should also be coherent at high frequency.

If the common-mode is a superposition of independent atmospheric processes
(troposphere at one scale, ionosphere at another, gravity waves at
another), then the coherence at different frequencies should be
UNCORRELATED — different processes drive different frequency bands.

Protocol:
  * Use 30 days of CODE MGEX clocks.
  * For each station pair, compute the band-averaged coherence in three
    sub-bands: low (10-50 uHz), mid (50-100 uHz), high (100-500 uHz).
  * Compute the correlation between low-band and high-band coherence
    across all pairs.
  * Also compute the correlation between the full-band coherence and
    each sub-band coherence.

  * TEP prediction: high cross-frequency correlation (r > 0.5) — one
    field drives all frequencies.
  * Atmospheric prediction: low cross-frequency correlation (r < 0.3) —
    independent processes at different frequencies.

Output: results/outputs/step_126_cross_frequency.json
"""
from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_126_cross_frequency.json"
PRODUCT = "clk_cod"
N_DAYS = 30

SUBBANDS = [
    ("low",  10e-6,  50e-6),
    ("mid",  50e-6, 100e-6),
    ("high", 100e-6, 500e-6),
]


def _spectra_subband(series, f_lo, f_hi):
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
        F = np.fft.rfft(xr)[band]
        if F.size < 4 or not np.all(np.isfinite(F)):
            continue
        RF[n] = F
        STA[n] = series[n]["station"]
    return RF, STA


def _pair_coherence(RF, STA):
    """Return dict of (ni, nj) -> (distance, cos_phase_alignment)."""
    nm = list(RF)
    out = {}
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
        out[(nm[i], nm[j])] = (d, float(np.cos(np.angle(Ssum))))
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 126: Cross-frequency coherence — is the TEP field unified?",
          flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Accumulate per-pair coherences across days for each sub-band
    pair_data = {}  # (ni, nj) -> {band: [alignments across days]}

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        for name, f_lo, f_hi in SUBBANDS:
            RF, STA = _spectra_subband(ser, f_lo, f_hi)
            if len(RF) < 8:
                continue
            coh = _pair_coherence(RF, STA)
            for (ni, nj), (d, a) in coh.items():
                key = (ni, nj)
                if key not in pair_data:
                    pair_data[key] = {"distance": d, "bands": {}}
                if name not in pair_data[key]["bands"]:
                    pair_data[key]["bands"][name] = []
                pair_data[key]["bands"][name].append(a)

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(selected)}] {len(pair_data)} unique pairs",
                  flush=True)

    print(f"\n  {len(pair_data)} unique pairs", flush=True)

    # Compute mean coherence per band per pair
    pairs_with_all_bands = []
    low_vals, mid_vals, high_vals = [], [], []
    distances = []

    for key, v in pair_data.items():
        bands = v["bands"]
        if "low" in bands and "mid" in bands and "high" in bands:
            lo = float(np.mean(bands["low"]))
            mi = float(np.mean(bands["mid"]))
            hi = float(np.mean(bands["high"]))
            low_vals.append(lo)
            mid_vals.append(mi)
            high_vals.append(hi)
            distances.append(v["distance"])
            pairs_with_all_bands.append((v["distance"], lo, mi, hi))

    low_vals = np.array(low_vals)
    mid_vals = np.array(mid_vals)
    high_vals = np.array(high_vals)
    distances = np.array(distances)

    print(f"  {len(low_vals)} pairs with all three bands", flush=True)

    if len(low_vals) < 100:
        OUT.write_text(json.dumps({"step": 126, "error": "too few pairs",
                                   "n_pairs": len(low_vals)}, indent=2))
        print("  ERROR: too few pairs", flush=True)
        return

    # Cross-frequency correlations
    r_low_mid = float(np.corrcoef(low_vals, mid_vals)[0, 1])
    r_low_high = float(np.corrcoef(low_vals, high_vals)[0, 1])
    r_mid_high = float(np.corrcoef(mid_vals, high_vals)[0, 1])

    print(f"\n  Cross-frequency correlations:", flush=True)
    print(f"    low vs mid:  r={r_low_mid:+.4f}", flush=True)
    print(f"    low vs high: r={r_low_high:+.4f}", flush=True)
    print(f"    mid vs high: r={r_mid_high:+.4f}", flush=True)

    # Also check in the TEP band only
    tep_mask = (distances >= 1000.0) & (distances <= 5000.0)
    if tep_mask.sum() > 50:
        r_lm_tep = float(np.corrcoef(low_vals[tep_mask], mid_vals[tep_mask])[0, 1])
        r_lh_tep = float(np.corrcoef(low_vals[tep_mask], high_vals[tep_mask])[0, 1])
        r_mh_tep = float(np.corrcoef(mid_vals[tep_mask], high_vals[tep_mask])[0, 1])
        print(f"\n  TEP band only ({tep_mask.sum()} pairs):", flush=True)
        print(f"    low vs mid:  r={r_lm_tep:+.4f}", flush=True)
        print(f"    low vs high: r={r_lh_tep:+.4f}", flush=True)
        print(f"    mid vs high: r={r_mh_tep:+.4f}", flush=True)
    else:
        r_lm_tep = r_lh_tep = r_mh_tep = None

    # Interpretation
    interpretations = []

    mean_cross_r = np.mean([abs(r_low_mid), abs(r_low_high), abs(r_mid_high)])
    if mean_cross_r > 0.5:
        interpretations.append(
            f"Cross-frequency coherence is HIGH (mean |r| = {mean_cross_r:.3f}). "
            f"The same pairs that are coherent at low frequency are also "
            f"coherent at high frequency.  This is consistent with a single "
            f"unified TEP proper-time field driving all frequencies, and "
            f"inconsistent with independent atmospheric processes at "
            f"different scales.")
    elif mean_cross_r > 0.3:
        interpretations.append(
            f"Cross-frequency coherence is MODERATE (mean |r| = {mean_cross_r:.3f}). "
            f"There is partial correlation between frequency bands, suggesting "
            f"a common component (TEP) plus band-specific components "
            f"(atmospheric).")
    else:
        interpretations.append(
            f"Cross-frequency coherence is LOW (mean |r| = {mean_cross_r:.3f}). "
            f"Different frequency bands are driven by independent processes, "
            f"consistent with a superposition of atmospheric mechanisms rather "
            f"than a single unified field.")

    # Check if the low-high correlation is the key discriminator
    if abs(r_low_high) > 0.4:
        interpretations.append(
            f"The low-high band correlation (r={r_low_high:+.3f}) is "
            f"particularly significant: these bands have completely "
            f"different atmospheric correlation scales (3749 km vs 2181 km) "
            f"yet the same pairs are coherent in both.  This suggests a "
            f"common underlying field (TEP) that is present at all "
            f"frequencies, with atmospheric processes modulating it "
            f"band-by-band.")

    verdict = {
        "step": 126,
        "title": "Cross-frequency coherence: is the TEP field unified?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "n_pairs_all_bands": len(low_vals),
        "subbands": [(n, f"{lo*1e6:.0f}", f"{hi*1e6:.0f}")
                     for n, lo, hi in SUBBANDS],
        "cross_frequency_correlations": {
            "all_pairs": {
                "low_vs_mid": r_low_mid,
                "low_vs_high": r_low_high,
                "mid_vs_high": r_mid_high,
                "mean_abs_r": float(mean_cross_r),
            },
            "tep_band_only": {
                "low_vs_mid": r_lm_tep,
                "low_vs_high": r_lh_tep,
                "mid_vs_high": r_mh_tep,
            },
        },
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the common-mode field is a single proper-time field "
            "that drives all frequencies.  Cross-frequency coherence should "
            "be high (mean |r| > 0.5) because the same pairs are coherent "
            "at all frequencies.  Under an atmospheric superposition, "
            "different frequency bands are driven by independent processes "
            "(troposphere, ionosphere, gravity waves) and cross-frequency "
            "correlation should be low (mean |r| < 0.3)."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
