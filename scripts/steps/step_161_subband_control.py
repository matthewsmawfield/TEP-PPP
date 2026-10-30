#!/usr/bin/env python3
"""
STEP 161  --  Critical control for Step 160: equal-width sub-bands.

Step 160 found Q/A ratio = 9x in the tidal band (10-35 uHz, 25 uHz wide)
but only 1.8x in the continuum (35-500 uHz, 465 uHz wide). The tidal band
has far fewer FFT bins than the continuum.

CONCERN: With fewer bins, the phase alignment index has higher variance.
If atomic clocks have lower SNR, fewer bins push their phase alignment
closer to zero (cos(random angle) -> 0), inflating the Q/A ratio. The
9x ratio could be a bin-count artefact, not a physics effect.

CONTROL: Split the continuum into narrow sub-bands matching the tidal
band's width (25 uHz). If the Q/A ratio is high only in the tidal band
and not in any continuum sub-band of equal width, the effect is real.
If some continuum sub-bands also show high Q/A, it's a bin-count artefact.

Also: split the tidal band itself into narrow sub-bands to localise
where the differentiation comes from.

Output: results/outputs/step_161_subband_control.json
"""
from __future__ import annotations
import sys, json, itertools
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps.step_103_atomic_quartz_gradients import _spectra, _clock_types
from scripts.steps.step_160_freq_clocktype import (
    _pair_phase_alignment_in_band, _pairs_by_type_in_band, _tep_band_pa, _decay_r2
)

OUT = ROOT / "results" / "outputs" / "step_161_subband_control.json"
PRODUCT = "clk_cod"
N_DAYS = 30
RNG = np.random.default_rng(20260920)

# Equal-width sub-bands (25 uHz each, matching the tidal band width)
SUBBAND_WIDTH = 25e-6  # 25 uHz
SUBBANDS = []
for start_uhz in range(10, 500, 25):
    f_lo = start_uhz * 1e-6
    f_hi = (start_uhz + 25) * 1e-6
    if f_hi > 500e-6:
        f_hi = 500e-6
    SUBBANDS.append((f"{start_uhz}-{start_uhz+25}uHz", f_lo, f_hi))

# Also test narrower 10 uHz sub-bands within the tidal region
NARROW_SUBBANDS = []
for start_uhz in range(8, 40, 5):
    f_lo = start_uhz * 1e-6
    f_hi = (start_uhz + 5) * 1e-6
    NARROW_SUBBANDS.append((f"{start_uhz}-{start_uhz+5}uHz", f_lo, f_hi))

TEP_BAND_KM = (1000.0, 5000.0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctype = _clock_types()
    print("Step 161: Sub-band control for clock-type screening", flush=True)
    print(f"  {len(SUBBANDS)} equal-width sub-bands (25 uHz each)", flush=True)
    print(f"  {len(NARROW_SUBBANDS)} narrow sub-bands (5 uHz each) in tidal region", flush=True)
    print(flush=True)

    dates_2025 = td.available_dates(PRODUCT, 2025)
    days = [dates_2025[i] for i in np.linspace(1, len(dates_2025) - 1, N_DAYS).astype(int)]
    print(f"  {len(days)} days (2025)", flush=True)

    # Collect pairs by sub-band and clock type
    wide_pairs = {sb[0]: {"QUARTZ": [], "ATOMIC": []} for sb in SUBBANDS}
    narrow_pairs = {sb[0]: {"QUARTZ": [], "ATOMIC": []} for sb in NARROW_SUBBANDS}

    for k, tag in enumerate(days):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        if not ser or len(ser) < 20:
            continue
        names = [n for n in ser if n in ctype and ctype[n] in ("QUARTZ", "ATOMIC")]
        if len(names) < 20:
            continue
        sub = {n: ser[n] for n in names}
        result = _spectra(sub)
        if result[0] is None:
            continue
        RF, STA = result
        nm = list(RF)
        if not nm:
            continue
        n_t = len(RF[nm[0]])
        t = ser[nm[0]]["t"]
        dt = float(np.median(np.diff(t)))
        n_grid = max(64, min(int(round(86400.0 / dt)), 2880))
        full_freq = np.fft.rfftfreq(n_grid, d=dt)
        band_mask = (full_freq >= 10e-6) & (full_freq <= 500e-6)
        band_freqs = full_freq[band_mask]
        if len(band_freqs) != n_t:
            band_freqs = np.linspace(10e-6, 500e-6, n_t)

        for sb_name, f_lo, f_hi in SUBBANDS:
            q, a = _pairs_by_type_in_band(RF, STA, band_freqs, ctype, f_lo, f_hi)
            wide_pairs[sb_name]["QUARTZ"].extend(q)
            wide_pairs[sb_name]["ATOMIC"].extend(a)

        for sb_name, f_lo, f_hi in NARROW_SUBBANDS:
            q, a = _pairs_by_type_in_band(RF, STA, band_freqs, ctype, f_lo, f_hi)
            narrow_pairs[sb_name]["QUARTZ"].extend(q)
            narrow_pairs[sb_name]["ATOMIC"].extend(a)

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(days)}]", flush=True)

    # Analyze wide sub-bands
    print(flush=True)
    print("=== Equal-width sub-bands (25 uHz each) ===", flush=True)
    print(f"  {'Band':20s} {'Q R2':>8s} {'A R2':>8s} {'Q pa':>8s} {'A pa':>8s} {'Q/A':>8s} {'n_Q':>8s} {'n_A':>8s}", flush=True)

    wide_results = {}
    for sb_name, f_lo, f_hi in SUBBANDS:
        q_pairs = wide_pairs[sb_name]["QUARTZ"]
        a_pairs = wide_pairs[sb_name]["ATOMIC"]
        q_r2 = _decay_r2(q_pairs)
        a_r2 = _decay_r2(a_pairs)
        q_pa = _tep_band_pa(q_pairs)
        a_pa = _tep_band_pa(a_pairs)
        ratio = q_pa / a_pa if q_pa and a_pa and a_pa > 0 else None
        wide_results[sb_name] = {
            "quartz": {"R2": q_r2, "tep_band_pa": q_pa, "n_pairs": len(q_pairs)},
            "atomic": {"R2": a_r2, "tep_band_pa": a_pa, "n_pairs": len(a_pairs)},
            "quartz_atomic_ratio": ratio,
        }
        q_r2_s = f"{q_r2:.3f}" if q_r2 is not None else "   N/A"
        a_r2_s = f"{a_r2:.3f}" if a_r2 is not None else "   N/A"
        q_pa_s = f"{q_pa:.4f}" if q_pa is not None else "   N/A"
        a_pa_s = f"{a_pa:.4f}" if a_pa is not None else "   N/A"
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "   N/A"
        marker = " <== TIDAL" if sb_name == "10-35uHz" else ""
        print(f"  {sb_name:20s} {q_r2_s:>8s} {a_r2_s:>8s} {q_pa_s:>8s} {a_pa_s:>8s} {ratio_s:>8s} "
              f"{len(q_pairs):>8d} {len(a_pairs):>8d}{marker}", flush=True)

    # Analyze narrow sub-bands
    print(flush=True)
    print("=== Narrow sub-bands (5 uHz each) in tidal region ===", flush=True)
    print(f"  {'Band':20s} {'Q R2':>8s} {'A R2':>8s} {'Q pa':>8s} {'A pa':>8s} {'Q/A':>8s} {'n_Q':>8s} {'n_A':>8s}", flush=True)

    narrow_results = {}
    for sb_name, f_lo, f_hi in NARROW_SUBBANDS:
        q_pairs = narrow_pairs[sb_name]["QUARTZ"]
        a_pairs = narrow_pairs[sb_name]["ATOMIC"]
        q_r2 = _decay_r2(q_pairs)
        a_r2 = _decay_r2(a_pairs)
        q_pa = _tep_band_pa(q_pairs)
        a_pa = _tep_band_pa(a_pairs)
        ratio = q_pa / a_pa if q_pa and a_pa and a_pa > 0 else None
        narrow_results[sb_name] = {
            "quartz": {"R2": q_r2, "tep_band_pa": q_pa, "n_pairs": len(q_pairs)},
            "atomic": {"R2": a_r2, "tep_band_pa": a_pa, "n_pairs": len(a_pairs)},
            "quartz_atomic_ratio": ratio,
        }
        q_r2_s = f"{q_r2:.3f}" if q_r2 is not None else "   N/A"
        a_r2_s = f"{a_r2:.3f}" if a_r2 is not None else "   N/A"
        q_pa_s = f"{q_pa:.4f}" if q_pa is not None else "   N/A"
        a_pa_s = f"{a_pa:.4f}" if a_pa is not None else "   N/A"
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "   N/A"
        # Mark which tidal lines are in this band
        lines = []
        if 10.676e-6 <= f_lo <= 10.676e-6 or 10.676e-6 >= f_lo and 10.676e-6 <= f_hi: lines.append("Q1")
        if 10.758e-6 >= f_lo and 10.758e-6 <= f_hi: lines.append("O1")
        if 11.574e-6 >= f_lo and 11.574e-6 <= f_hi: lines.append("S1")
        if 11.608e-6 >= f_lo and 11.608e-6 <= f_hi: lines.append("K1")
        if 21.422e-6 >= f_lo and 21.422e-6 <= f_hi: lines.append("N2")
        if 22.365e-6 >= f_lo and 22.365e-6 <= f_hi: lines.append("M2")
        if 23.148e-6 >= f_lo and 23.148e-6 <= f_hi: lines.append("S2")
        marker = f" <== {','.join(lines)}" if lines else ""
        print(f"  {sb_name:20s} {q_r2_s:>8s} {a_r2_s:>8s} {q_pa_s:>8s} {a_pa_s:>8s} {ratio_s:>8s} "
              f"{len(q_pairs):>8d} {len(a_pairs):>8d}{marker}", flush=True)

    # Key question: is the tidal band (10-35 uHz) Q/A ratio an outlier?
    tidal_ratio = wide_results.get("10-35uHz", {}).get("quartz_atomic_ratio")
    continuum_ratios = [wide_results[sb]["quartz_atomic_ratio"]
                        for sb_name, sb in wide_results.items()
                        if sb_name != "10-35uHz" and wide_results[sb_name]["quartz_atomic_ratio"] is not None]

    interp = []
    interp.append(f"Sub-band control: {len(SUBBANDS)} equal-width (25 uHz) sub-bands, {N_DAYS} days (2025)")
    interp.append("")

    if tidal_ratio is not None and continuum_ratios:
        cont_arr = np.array(continuum_ratios)
        percentile = float(np.mean(cont_arr < tidal_ratio)) * 100
        mean_cont = float(np.mean(cont_arr))
        std_cont = float(np.std(cont_arr))
        max_cont = float(np.max(cont_arr))
        z = (tidal_ratio - mean_cont) / std_cont if std_cont > 0 else 0

        interp.append(f"  Tidal band (10-35 uHz) Q/A ratio: {tidal_ratio:.2f}x")
        interp.append(f"  Continuum sub-bands: mean={mean_cont:.2f}x, std={std_cont:.2f}x, max={max_cont:.2f}x")
        interp.append(f"  Tidal band percentile: {percentile:.1f}%")
        interp.append(f"  Tidal band z-score vs continuum: {z:.1f}")
        interp.append("")

        if percentile > 95:
            interp.append("  The tidal band Q/A ratio is in the top 5% of all equal-width sub-bands.")
            interp.append("  The clock-type differentiation is NOT a bin-count artefact.")
            interp.append("  It is specific to the gravitational tidal frequencies.")
        elif percentile > 90:
            interp.append("  The tidal band Q/A ratio is in the top 10% of all equal-width sub-bands.")
            interp.append("  The clock-type differentiation is likely not a bin-count artefact.")
        else:
            interp.append(f"  The tidal band Q/A ratio is at the {percentile:.1f}th percentile of equal-width sub-bands.")
            interp.append("  The clock-type differentiation MAY be a bin-count artefact.")

    # Also check: which narrow sub-band has the highest Q/A?
    narrow_ratios = [(sb_name, narrow_results[sb_name].get("quartz_atomic_ratio"))
                     for sb_name in narrow_results
                     if narrow_results[sb_name].get("quartz_atomic_ratio") is not None]
    if narrow_ratios:
        narrow_ratios.sort(key=lambda x: x[1], reverse=True)
        interp.append("")
        interp.append("  Narrow sub-bands ranked by Q/A ratio:")
        for sb_name, ratio in narrow_ratios[:5]:
            interp.append(f"    {sb_name}: {ratio:.2f}x")

    verdict = {
        "step": 161,
        "title": "Sub-band control: is the tidal band Q/A ratio a bin-count artefact?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": 2025,
        "n_days": N_DAYS,
        "wide_subbands": wide_results,
        "narrow_subbands": narrow_results,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If the tidal band Q/A ratio is a real physics effect, it should be an outlier "
            "among equal-width sub-bands. If it is a bin-count artefact, other narrow "
            "sub-bands should show similarly high Q/A ratios."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for line in interp:
        print(line, flush=True)


if __name__ == "__main__":
    main()
