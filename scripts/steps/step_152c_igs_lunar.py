#!/usr/bin/env python3
"""
STEP 152c  --  Lunar tidal detection in IGS final clock products.

The critical test: do lunar tides (M2, O1) appear in IGS final clock
products (clk_cod), which use a global network adjustment that removes
common-mode signals?

If the lunar tides are a RTKLIB processing artefact, they should NOT
appear in IGS products.  If they are a real gravitational signal that
is common-mode, IGS processing may remove most of it, but any residual
differential component should still be detectable.

With 100+ IGS stations and 30+ days, the cross-station phase coherence
test has ~5000 pairs — far more powerful than the 21-pair RTKLIB test.

Output: results/outputs/step_152c_igs_lunar.json
"""
from __future__ import annotations
import sys, json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import tep_data

OUT = ROOT / "results" / "outputs" / "step_152c_igs_lunar.json"
META_PATH = ROOT / "data" / "meta" / "station_comprehensive.json"

DT = 300.0  # IGS products are 300s cadence
N_DAYS = 30  # Use 30-day window for maximum station count
C = 299792458.0

# Tidal frequencies (Hz)
M2_FREQ  = 1.0 / (12.42 * 3600.0)   # 22.365 uHz, lunar
O1_FREQ  = 1.0 / (25.82 * 3600.0)   # 10.758 uHz, lunar
S2_FREQ  = 2.0 / 86400.0            # 23.148 uHz, solar
S1_FREQ  = 1.0 / 86400.0            # 11.574 uHz, solar
K1_FREQ  = 1.0 / (23.93 * 3600.0)   # 11.608 uHz, mixed
P1_FREQ  = 1.0 / (24.07 * 3600.0)   # 11.542 uHz, solar

ALL_FREQS = [M2_FREQ, O1_FREQ, S2_FREQ, S1_FREQ, K1_FREQ, P1_FREQ]
ALL_NAMES = ["M2", "O1", "S2", "S1", "K1", "P1"]

RNG_SEED = 20260920


def harmonic_fit(resid, t_sec, freqs):
    """Simultaneous harmonic fit at given frequencies."""
    t0 = t_sec[0]
    t_c = t_sec - t0 - (t_sec[-1] - t_sec[0]) / 2.0
    t_s = (t_c[-1] - t_c[0]) / 2.0 if len(t_c) > 1 else 1.0
    t_n = t_c / t_s
    cols = [np.ones_like(t_n), t_n, t_n ** 2]
    for f in freqs:
        omega = 2.0 * np.pi * f * (t_sec - t0)
        cols.append(np.cos(omega))
        cols.append(np.sin(omega))
    design = np.column_stack(cols)
    coef, _, _, _ = lstsq(design, resid, cond=1e-12)
    resid_fit = resid - design @ coef
    noise_std = float(np.std(resid_fit))
    n = len(resid)
    results = []
    for i, f in enumerate(freqs):
        A = float(coef[3 + 2 * i])
        B = float(coef[3 + 2 * i + 1])
        amp = float(np.sqrt(A ** 2 + B ** 2))
        phase = float(np.arctan2(B, A))
        snr = amp / (noise_std / np.sqrt(n / 2.0)) if noise_std > 0 else 0.0
        results.append({"amp": amp, "phase": phase, "snr": snr, "A": A, "B": B})
    return results, noise_std


def find_best_window():
    """Find the 30-day window with the most stations available."""
    best_count = 0
    best_start = 1
    # We have 175 days of COD 2025
    for start in range(1, 176 - N_DAYS + 1):
        count = None
        common = None
        for doy in range(start, start + N_DAYS):
            tag = f'2025{doy:03d}'
            s = tep_data.load_series('clk_cod', tag, min_valid_frac=0.0)
            if not s:
                count = 0
                break
            stations = set(s.keys())
            if common is None:
                common = stations
            else:
                common = common & stations
        n = len(common) if common else 0
        if n > best_count:
            best_count = n
            best_start = start
    return best_start, best_count


def load_igs_station(st, doys):
    """Load clock residual for a station across multiple days."""
    all_resid = []
    all_t = []
    for doy in doys:
        tag = f'2025{doy:03d}'
        s = tep_data.load_series('clk_cod', tag, min_valid_frac=0.0)
        if not s or st not in s:
            return None
        d = s[st]
        r = np.asarray(d['resid'], float)
        t = np.asarray(d['t'], float) + (doy - 1) * 86400.0
        all_resid.append(r)
        all_t.append(t)
    resid = np.concatenate(all_resid)
    t = np.concatenate(all_t)
    # Remove NaN
    good = np.isfinite(resid)
    if good.mean() < 0.8:
        return None
    if not good.all():
        ix = np.arange(len(resid))
        resid = np.interp(ix, ix[good], resid[good])
    # Detrend
    tc = t - t[0]
    c = np.polyfit(tc, resid, 2)
    resid = resid - np.polyval(c, tc)
    return resid, t


def phase_prediction_test(stations_data, all_results, tide_idx, tide_freq, label):
    """Cross-station phase prediction test."""
    st_list = sorted(stations_data.keys())
    expected_corrs = []
    actual_corrs = []
    n_pairs = 0
    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            if s1 not in all_results or s2 not in all_results:
                continue
            d1 = stations_data[s1][0]  # resid
            d2 = stations_data[s2][0]
            n = min(len(d1), len(d2))
            if n < 100:
                continue
            dphase = all_results[s1]["results"][tide_idx]["phase"] - \
                     all_results[s2]["results"][tide_idx]["phase"]
            expected = float(np.cos(dphase))
            freqs = np.fft.fftfreq(n, DT)
            X1 = np.fft.fft(d1[:n])
            X2 = np.fft.fft(d2[:n])
            mask = (np.abs(freqs) >= tide_freq - 0.5e-6) & \
                   (np.abs(freqs) <= tide_freq + 0.5e-6)
            X1[~mask] = 0
            X2[~mask] = 0
            bp1 = np.real(np.fft.ifft(X1))
            bp2 = np.real(np.fft.ifft(X2))
            v1 = bp1 - bp1.mean()
            v2 = bp2 - bp2.mean()
            if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
                continue
            actual = float(np.corrcoef(v1, v2)[0, 1])
            if np.isfinite(actual):
                expected_corrs.append(expected)
                actual_corrs.append(actual)
                n_pairs += 1
    if len(expected_corrs) > 5:
        r_phase, p_phase = stats.pearsonr(expected_corrs, actual_corrs)
    else:
        r_phase, p_phase = None, None
    return {"r": float(r_phase) if r_phase is not None else None,
            "p": float(p_phase) if p_phase is not None else None,
            "n_pairs": n_pairs,
            "mean_actual_corr": float(np.mean(actual_corrs)) if actual_corrs else None}


def per_station_significance(resid, t, target_freq, freq_lo, freq_hi, exclude_freqs, seed):
    """Significance test against random nearby frequencies."""
    rng = np.random.default_rng(seed)
    target_res, _ = harmonic_fit(resid, t, [target_freq] + list(exclude_freqs))
    if target_res is None:
        return None, None, None
    target_amp = target_res[0]["amp"]
    random_amps = []
    for _ in range(200):
        f = rng.uniform(freq_lo, freq_hi)
        if any(abs(f - fr) < 0.3e-6 for fr in [target_freq] + list(exclude_freqs)):
            continue
        res, _ = harmonic_fit(resid, t, [f] + list(exclude_freqs))
        if res is not None:
            random_amps.append(res[0]["amp"])
    if not random_amps:
        return target_amp, None, None
    p = float(np.mean(np.array(random_amps) >= target_amp))
    return target_amp, float(np.median(random_amps)), p


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    meta = json.load(open(META_PATH))

    print("Step 152c: Lunar tidal detection in IGS final clock products", flush=True)
    print(f"  Product: clk_cod, Year: 2025", flush=True)
    print(f"  Window: {N_DAYS} days", flush=True)
    print(flush=True)

    # Find best window
    print("  Finding best 30-day window...", flush=True)
    best_start, best_count = find_best_window()
    doys = list(range(best_start, best_start + N_DAYS))
    T_SPAN = N_DAYS * 86400.0
    FREQ_RES = 1.0 / T_SPAN
    print(f"  Best window: DOY {best_start}-{best_start + N_DAYS - 1}, {best_count} stations", flush=True)
    print(f"  Frequency resolution: {FREQ_RES * 1e6:.3f} uHz", flush=True)
    print(f"  M2-S2 separation: {abs(S2_FREQ - M2_FREQ) / FREQ_RES:.1f} bins", flush=True)
    print(flush=True)

    # Get common station set
    common = None
    for doy in doys:
        tag = f'2025{doy:03d}'
        s = tep_data.load_series('clk_cod', tag, min_valid_frac=0.0)
        if not s: continue
        stations = set(s.keys())
        common = stations if common is None else common & stations
    common = sorted(common)
    print(f"  Common stations: {len(common)}", flush=True)

    # Load all stations
    print("  Loading station data...", flush=True)
    stations_data = {}
    skipped = 0
    for st in common:
        result = load_igs_station(st, doys)
        if result is None:
            skipped += 1
            continue
        resid, t = result
        if np.std(resid) > 100:  # Skip noisy stations
            skipped += 1
            continue
        stations_data[st] = (resid, t)
    print(f"  Loaded: {len(stations_data)} stations, skipped: {skipped}", flush=True)

    # Report noise levels
    stds = [np.std(stations_data[st][0]) for st in stations_data]
    print(f"  Residual std: mean={np.mean(stds):.3f} ns, median={np.median(stds):.3f} ns, "
          f"min={np.min(stds):.3f}, max={np.max(stds):.3f}", flush=True)
    print(flush=True)

    # Harmonic fit
    print("  Simultaneous harmonic fit: M2, O1, S2, S1, K1, P1", flush=True)
    all_results = {}
    for st in sorted(stations_data.keys()):
        resid, t = stations_data[st]
        results, noise = harmonic_fit(resid, t, ALL_FREQS)
        ct = meta.get(st, {}).get("clock", "UNKNOWN")
        all_results[st] = {"results": results, "clock_type": ct, "noise": noise}
    print(f"  Fitted {len(all_results)} stations", flush=True)

    # Summary amplitudes
    for name, idx in [("M2", 0), ("O1", 1), ("S2", 2), ("S1", 3)]:
        amps = [all_results[st]["results"][idx]["amp"] for st in all_results]
        snrs = [all_results[st]["results"][idx]["snr"] for st in all_results]
        print(f"    {name}: mean amp={np.mean(amps):.4f} ns, median SNR={np.median(snrs):.1f}, "
              f"max amp={np.max(amps):.4f} ns", flush=True)
    print(flush=True)

    # Phase prediction
    print("  Cross-station phase prediction", flush=True)
    phase_results = {}
    for idx, name in enumerate(ALL_NAMES):
        freq = ALL_FREQS[idx]
        pr = phase_prediction_test(stations_data, all_results, idx, freq, name)
        phase_results[name] = pr
        if pr["r"] is not None:
            sig = "***" if pr["p"] < 0.001 else "**" if pr["p"] < 0.01 else "*" if pr["p"] < 0.05 else ""
            print(f"    {name:4s}: r={pr['r']:+.4f}, p={pr['p']:.4f} {sig}, "
                  f"n_pairs={pr['n_pairs']}", flush=True)
    print(flush=True)

    # Per-station significance for M2 and O1
    print("  Per-station significance (200 random frequencies)", flush=True)
    sig_results = {}
    for name, idx, freq, flo, fhi in [("M2", 0, M2_FREQ, 18e-6, 27e-6),
                                       ("O1", 1, O1_FREQ, 8e-6, 14e-6)]:
        exclude = [f for f in ALL_FREQS if f != freq]
        ps = []
        for st in sorted(stations_data.keys())[:50]:  # Limit to 50 for speed
            resid, t = stations_data[st]
            amp, rand_med, p = per_station_significance(
                resid, t, freq, flo, fhi, exclude, RNG_SEED + hash(st + name) % 10000)
            if st not in sig_results:
                sig_results[st] = {}
            sig_results[st][name] = {"amp": amp, "random_median": rand_med, "p": p}
            if p is not None:
                ps.append(p)
        n_sig = sum(1 for p in ps if p < 0.05)
        print(f"    {name}: {n_sig}/{len(ps)} stations significant (p<0.05), "
              f"median p={np.median(ps):.4f}" if ps else f"    {name}: no data", flush=True)
    print(flush=True)

    # Clock-type screening
    print("  Clock-type screening", flush=True)
    quartz = [st for st in all_results if all_results[st]["clock_type"] == "QUARTZ"]
    atomic_types = ["ATOMIC", "HYDROGEN MASER", "CESIUM", "RUBIDIUM"]
    atomic = [st for st in all_results if all_results[st]["clock_type"] in atomic_types]
    print(f"    Quartz: {len(quartz)}, Atomic: {len(atomic)}", flush=True)
    clock_type = {}
    for name, idx in [("M2", 0), ("O1", 1), ("S2", 2), ("S1", 3)]:
        q = [all_results[st]["results"][idx]["amp"] for st in quartz]
        a = [all_results[st]["results"][idx]["amp"] for st in atomic]
        entry = {"quartz_mean": float(np.mean(q)) if q else None,
                 "quartz_n": len(q),
                 "atomic_mean": float(np.mean(a)) if a else None,
                 "atomic_n": len(a)}
        if len(q) > 1 and len(a) > 1:
            t_stat, p_val = stats.ttest_ind(q, a, equal_var=False)
            entry["p_value"] = float(p_val)
            print(f"    {name}: quartz={np.mean(q):.4f}±{np.std(q):.4f} (n={len(q)}), "
                  f"atomic={np.mean(a):.4f}±{np.std(a):.4f} (n={len(a)}), p={p_val:.4f}", flush=True)
        clock_type[name] = entry
    print(flush=True)

    # Build output
    per_station = {}
    for st in sorted(all_results.keys()):
        res = all_results[st]["results"]
        per_station[st] = {
            "clock_type": all_results[st]["clock_type"],
            "noise_std_ns": all_results[st]["noise"],
        }
        for i, name in enumerate(ALL_NAMES):
            per_station[st][name] = {"amp_ns": res[i]["amp"], "phase": res[i]["phase"], "snr": res[i]["snr"]}
        per_station[st]["significance"] = sig_results.get(st, {})

    lunar_amps = [all_results[st]["results"][0]["amp"] for st in all_results]  # M2
    solar_amps = [all_results[st]["results"][2]["amp"] for st in all_results]  # S2

    verdict = {
        "step": "152c",
        "title": "Lunar tidal detection in IGS final clock products",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": "clk_cod",
        "year": 2025,
        "window_doy": [best_start, best_start + N_DAYS - 1],
        "n_days": N_DAYS,
        "n_stations": len(all_results),
        "freq_resolution_uhz": FREQ_RES * 1e6,
        "mean_residual_std_ns": float(np.mean(stds)),
        "median_residual_std_ns": float(np.median(stds)),
        "per_station": per_station,
        "phase_prediction": phase_results,
        "clock_type_screening": clock_type,
        "summary": {
            "M2_lunar_mean_amp_ns": float(np.mean(lunar_amps)),
            "M2_lunar_median_amp_ns": float(np.median(lunar_amps)),
            "S2_solar_mean_amp_ns": float(np.mean(solar_amps)),
            "S2_solar_median_amp_ns": float(np.median(solar_amps)),
            "M2_phase_r": phase_results["M2"]["r"],
            "M2_phase_p": phase_results["M2"]["p"],
            "M2_n_pairs": phase_results["M2"]["n_pairs"],
            "S2_phase_r": phase_results["S2"]["r"],
            "S2_phase_p": phase_results["S2"]["p"],
        },
        "interpretation": [
            f"IGS final products (clk_cod): {len(all_results)} stations, {N_DAYS} days, "
            f"residual std median={np.median(stds):.3f} ns.",
            f"M2 (lunar): mean amp={np.mean(lunar_amps):.4f} ns, "
            f"median amp={np.median(lunar_amps):.4f} ns.",
            f"S2 (solar): mean amp={np.mean(solar_amps):.4f} ns, "
            f"median amp={np.median(solar_amps):.4f} ns.",
            f"M2 phase prediction: r={phase_results['M2']['r']}, "
            f"p={phase_results['M2']['p']}, n_pairs={phase_results['M2']['n_pairs']}."
            if phase_results['M2']['r'] is not None else "M2 phase prediction: insufficient data.",
            f"S2 phase prediction: r={phase_results['S2']['r']}, "
            f"p={phase_results['S2']['p']}, n_pairs={phase_results['S2']['n_pairs']}."
            if phase_results['S2']['r'] is not None else "S2 phase prediction: insufficient data.",
        ],
        "caveats": [
            f"IGS products at 300s cadence (vs 30s for RTKLIB PPP).",
            f"No station has all {N_DAYS} days; using best available window.",
            f"IGS global network adjustment removes common-mode signals.",
        ],
        "pre_registered_prediction": (
            "If the lunar tides in RTKLIB PPP are a real gravitational signal, "
            "they should also appear (at reduced amplitude) in IGS products with "
            "phase coherence across stations. If they are a processing artefact, "
            "they should NOT be phase-coherent in IGS products."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
