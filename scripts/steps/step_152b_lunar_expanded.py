#!/usr/bin/env python3
"""
STEP 152b  --  Lunar vs solar tidal split in GNSS receiver clock bias.

The cleanest gravitational vs atmospheric test:
  - LUNAR tides (M2, O1, Q1, J1, N2) are purely gravitational.  Standard
    physics predicts zero amplitude in clock bias after solid Earth tide
    correction.  Ocean loading produces at most ~0.01 ns.
  - SOLAR tides (S1, S2, P1, K2, 24h) have both gravitational and
    atmospheric components.  Atmospheric pressure tides produce ~1-10 ns
    in clock bias through path delay and oscillator pressure sensitivity.

TEP predicts:
  1. Lunar tides should be detectable (gravitational coupling to clocks).
  2. Lunar tides should show cross-station phase coherence (coherent field).
  3. Lunar phase coherence should be comparable to or stronger than solar
     phase coherence, because the gravitational field is coherent across
     stations while atmospheric tides have regional structure.

Atmospheric origin predicts:
  1. Solar tides should dominate in amplitude.
  2. Solar phase coherence should be stronger (regional atmospheric coherence).
  3. Lunar tides should be absent or at ocean-loading level.

This script uses 45-day contiguous PPP data (DOY 1-45, 7 IGS stations)
with 0.257 uHz resolution — enough to resolve M2 (22.365 uHz) from S2
(23.148 uHz) at 3.0 bins separation.

Output: results/outputs/step_152b_lunar_expanded.json
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
OUT = ROOT / "results" / "outputs" / "step_152b_lunar_expanded.json"
PROCESSED_DIR = ROOT / "data" / "ppp_processed"
META_PATH = ROOT / "data" / "meta" / "station_comprehensive.json"

DT = 30.0
N_DAYS = 45
T_SPAN = N_DAYS * 86400.0
FREQ_RES = 1.0 / T_SPAN
C = 299792458.0
SIN_E_AVG = 0.6

# Tidal frequencies (Hz) — corrected values
# LUNAR (purely gravitational)
M2_FREQ  = 1.0 / (12.42 * 3600.0)   # 22.365 uHz
O1_FREQ  = 1.0 / (25.82 * 3600.0)   # 10.758 uHz
Q1_FREQ  = 1.0 / (26.87 * 3600.0)   # 10.676 uHz
J1_FREQ  = 1.0 / (23.10 * 3600.0)   # 12.040 uHz
N2_FREQ  = 1.0 / (12.66 * 3600.0)   # 21.422 uHz
M3_FREQ  = 1.0 / (8.28  * 3600.0)   # 34.533 uHz
M4_FREQ  = 1.0 / (6.21  * 3600.0)   # 40.064 uHz

# SOLAR (gravitational + atmospheric)
S2_FREQ  = 2.0 / 86400.0            # 23.148 uHz
S1_FREQ  = 1.0 / 86400.0            # 11.574 uHz
P1_FREQ  = 1.0 / (24.07 * 3600.0)   # 11.542 uHz
K2_FREQ  = 1.0 / (11.97 * 3600.0)   # 23.098 uHz
K1_FREQ  = 1.0 / (23.93 * 3600.0)   # 11.608 uHz (mixed — mostly lunar)

# All fit frequencies (simultaneous)
LUNAR_FREQS = [M2_FREQ, O1_FREQ, Q1_FREQ, J1_FREQ, N2_FREQ, M3_FREQ, M4_FREQ]
LUNAR_NAMES = ["M2", "O1", "Q1", "J1", "N2", "M3", "M4"]

SOLAR_FREQS = [S2_FREQ, S1_FREQ, P1_FREQ, K2_FREQ]
SOLAR_NAMES = ["S2", "S1", "P1", "K2"]

ALL_FREQS = LUNAR_FREQS + SOLAR_FREQS
ALL_NAMES = LUNAR_NAMES + SOLAR_NAMES

LONG_STATIONS = ["BRUX", "GENO", "GRAS", "TLSE", "WTZR", "YEBE", "ZIMM"]
JUMP_THRESHOLD_NS = 100.0
RNG_SEED = 20260920
N_RANDOM_FREQS = 500
N_STACK_RANDOM = 1000


def detrend(y, t, order=2):
    c = np.polyfit(t, y, order)
    return y - np.polyval(c, t)


def harmonic_fit(resid, t_sec, freqs):
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


def remove_jumps(vals, threshold=JUMP_THRESHOLD_NS):
    diffs = np.diff(vals)
    jumps = np.where(np.abs(diffs) > threshold)[0]
    clean = vals.copy()
    for j in jumps:
        clean[j + 1:] -= diffs[j]
    return clean, len(jumps)


def load_stations():
    meta = json.load(open(META_PATH))
    grid = np.arange(0.0, T_SPAN + DT / 2, DT)
    th = (grid - grid[0]) / 3600.0
    stations = {}
    for st in LONG_STATIONS:
        sods_all = []
        clks_all = []
        for doy in range(1, N_DAYS + 1):
            fname = f"{st}_2024{doy:03d}.npz"
            fpath = PROCESSED_DIR / fname
            if not fpath.exists():
                continue
            d = np.load(fpath, allow_pickle=True)
            sods_all.extend(d["sod"] + (doy - 1) * 86400.0)
            clks_all.extend(d["clock_bias_ns"])
        if len(sods_all) < 10000:
            continue
        sods = np.array(sorted(sods_all))
        clks = np.array([c for _, c in sorted(zip(sods_all, clks_all))])
        vals = np.interp(grid, sods, clks)
        clean, n_jumps = remove_jumps(vals)
        resid = detrend(clean, th, order=2)
        if np.std(resid) > 1000:
            continue
        ct = meta.get(st, {}).get("clock", "UNKNOWN")
        stations[st] = {"resid": resid, "clock_type": ct, "n_jumps": n_jumps,
                        "t": grid, "n_epochs": len(sods_all)}
    return stations


def phase_prediction_test(stations, all_results, tide_idx, tide_freq, label):
    """Cross-station phase prediction: does the phase at one station predict
    the bandpass-filtered correlation at another?"""
    st_list = sorted(stations.keys())
    expected_corrs = []
    actual_corrs = []
    pair_data = []
    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            if s1 not in all_results or s2 not in all_results:
                continue
            d1, d2 = stations[s1], stations[s2]
            n = min(len(d1["resid"]), len(d2["resid"]))
            if n < 100:
                continue
            dphase = all_results[s1]["results"][tide_idx]["phase"] - \
                     all_results[s2]["results"][tide_idx]["phase"]
            expected = float(np.cos(dphase))
            freqs = np.fft.fftfreq(n, DT)
            X1 = np.fft.fft(d1["resid"][:n])
            X2 = np.fft.fft(d2["resid"][:n])
            mask = (np.abs(freqs) >= tide_freq - 0.5e-6) & \
                   (np.abs(freqs) <= tide_freq + 0.5e-6)
            X1[~mask] = 0
            X2[~mask] = 0
            bp1 = np.real(np.fft.ifft(X1))
            bp2 = np.real(np.fft.ifft(X2))
            actual = float(np.corrcoef(bp1, bp2)[0, 1])
            if np.isfinite(actual):
                expected_corrs.append(expected)
                actual_corrs.append(actual)
                pair_data.append({"pair": f"{s1}-{s2}", "phase_diff": dphase,
                                  "expected_corr": expected, "actual_corr": actual})
    if expected_corrs:
        r_phase, p_phase = stats.pearsonr(expected_corrs, actual_corrs)
    else:
        r_phase, p_phase = None, None
    return {"r": float(r_phase) if r_phase else None,
            "p": float(p_phase) if p_phase else None,
            "n_pairs": len(expected_corrs),
            "mean_actual_corr": float(np.mean(actual_corrs)) if actual_corrs else None,
            "mean_expected_corr": float(np.mean(expected_corrs)) if expected_corrs else None,
            "pairs": pair_data}


def stack_significance(stations, all_results, tide_idx, tide_freq, freq_lo, freq_hi, rng):
    """Phase-corrected stacking with random-frequency null."""
    n_pts = len(stations[list(stations.keys())[0]]["resid"])
    t_arr = stations[list(stations.keys())[0]]["t"]
    omega = 2.0 * np.pi * tide_freq * t_arr
    stacked = np.zeros(n_pts)
    n_st = 0
    for st in sorted(stations.keys()):
        r = stations[st]["resid"]
        res = all_results[st]["results"][tide_idx]
        A, B = res["A"], res["B"]
        amp = res["amp"]
        original = A * np.cos(omega) + B * np.sin(omega)
        rotated = amp * np.cos(omega)
        stacked += r - original + rotated
        n_st += 1
    stacked /= n_st
    stack_res, stack_noise = harmonic_fit(stacked, t_arr, ALL_FREQS)
    stack_amp = stack_res[tide_idx]["amp"]
    stack_snr = stack_res[tide_idx]["snr"]

    random_stack_amps = []
    for _ in range(N_STACK_RANDOM):
        f = rng.uniform(freq_lo, freq_hi)
        if any(abs(f - fr) < 0.3e-6 for fr in ALL_FREQS):
            continue
        res, _ = harmonic_fit(stacked, t_arr, [f] + list(ALL_FREQS))
        if res is not None:
            random_stack_amps.append(res[0]["amp"])
    p_stack = float(np.mean(np.array(random_stack_amps) >= stack_amp)) if random_stack_amps else None
    return {"stack_amp_ns": stack_amp, "stack_snr": stack_snr,
            "n_stations": n_st, "p_value": p_stack,
            "mean_single_amp_ns": float(np.mean(
                [all_results[s]["results"][tide_idx]["amp"] for s in sorted(all_results.keys())]))}


def per_station_significance(resid, t, target_freq, freq_lo, freq_hi, exclude_freqs, seed):
    rng = np.random.default_rng(seed)
    target_res, _ = harmonic_fit(resid, t, [target_freq] + list(exclude_freqs))
    if target_res is None:
        return None, None, None
    target_amp = target_res[0]["amp"]
    random_amps = []
    for _ in range(N_RANDOM_FREQS):
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

    print("Step 152b: Lunar vs solar tidal split in GNSS clock bias", flush=True)
    print(f"  Data: {N_DAYS}-day contiguous PPP (DOY 1-45), 7 IGS stations", flush=True)
    print(f"  Frequency resolution: {FREQ_RES * 1e6:.3f} uHz", flush=True)
    print(f"  M2-S2 separation: {abs(S2_FREQ - M2_FREQ) / FREQ_RES:.1f} bins", flush=True)
    print(f"  Lunar lines: {', '.join(LUNAR_NAMES)}", flush=True)
    print(f"  Solar lines: {', '.join(SOLAR_NAMES)}", flush=True)
    print(flush=True)

    stations = load_stations()
    print(f"  Usable stations: {sorted(stations.keys())}", flush=True)
    print(flush=True)

    # Simultaneous fit of all 11 tidal lines
    print("  Simultaneous fit: 7 lunar + 4 solar tidal lines", flush=True)
    all_results = {}
    for st in sorted(stations.keys()):
        r = stations[st]["resid"]
        t = stations[st]["t"]
        results, noise = harmonic_fit(r, t, ALL_FREQS)
        all_results[st] = {"results": results, "clock_type": stations[st]["clock_type"],
                           "noise": noise}
        # Print key lines
        m2 = results[0]  # M2
        o1 = results[1]  # O1
        s2 = results[7]  # S2
        s1 = results[8]  # S1
        print(f"    {st}: M2={m2['amp']:.3f}ns(SNR={m2['snr']:.1f}), "
              f"O1={o1['amp']:.3f}ns(SNR={o1['snr']:.1f}), "
              f"S2={s2['amp']:.3f}ns(SNR={s2['snr']:.1f}), "
              f"S1={s1['amp']:.3f}ns(SNR={s1['snr']:.1f})", flush=True)
    print(flush=True)

    # === Phase prediction for each tidal line ===
    print("  Cross-station phase prediction (all lines)", flush=True)
    phase_results = {}
    for idx, name in enumerate(ALL_NAMES):
        freq = ALL_FREQS[idx]
        pr = phase_prediction_test(stations, all_results, idx, freq, name)
        phase_results[name] = pr
        if pr["r"] is not None:
            print(f"    {name:4s} ({'lunar' if idx < len(LUNAR_NAMES) else 'solar'}): "
                  f"r={pr['r']:.4f}, p={pr['p']:.4f}, n={pr['n_pairs']}, "
                  f"mean_actual={pr['mean_actual_corr']:.3f}", flush=True)
    print(flush=True)

    # === Stack significance for each line ===
    print("  Phase-corrected stacking (all lines)", flush=True)
    stack_results = {}
    for idx, name in enumerate(ALL_NAMES):
        freq = ALL_FREQS[idx]
        freq_lo = max(0, freq - 3e-6)
        freq_hi = freq + 3e-6
        sr = stack_significance(stations, all_results, idx, freq, freq_lo, freq_hi, rng)
        stack_results[name] = sr
        print(f"    {name:4s}: stack amp={sr['stack_amp_ns']:.3f}ns, "
              f"SNR={sr['stack_snr']:.1f}, p={sr['p_value']:.4f}" if sr['p_value'] else
              f"    {name:4s}: stack amp={sr['stack_amp_ns']:.3f}ns", flush=True)
    print(flush=True)

    # === Per-station significance for key lines ===
    print("  Per-station significance (500 random frequencies)", flush=True)
    sig_results = {}
    key_lines = [("M2", 0, M2_FREQ, 18e-6, 27e-6),
                 ("O1", 1, O1_FREQ, 8e-6, 14e-6),
                 ("S2", 7, S2_FREQ, 18e-6, 27e-6),
                 ("S1", 8, S1_FREQ, 8e-6, 14e-6)]
    for name, idx, freq, flo, fhi in key_lines:
        exclude = [f for f in ALL_FREQS if f != freq]
        for st in sorted(stations.keys()):
            r = stations[st]["resid"]
            t = stations[st]["t"]
            amp, rand_med, p = per_station_significance(
                r, t, freq, flo, fhi, exclude, RNG_SEED + hash(st + name) % 10000)
            if st not in sig_results:
                sig_results[st] = {}
            sig_results[st][name] = {"amp": amp, "random_median": rand_med, "p": p}
        # Summary
        ps = [sig_results[st][name]["p"] for st in sorted(stations.keys())
              if sig_results[st][name]["p"] is not None]
        n_sig = sum(1 for p in ps if p < 0.05)
        print(f"    {name}: {n_sig}/{len(ps)} stations significant (p<0.05), "
              f"median p={np.median(ps):.4f}", flush=True)
    print(flush=True)

    # === Lunar vs solar summary ===
    lunar_amps = {name: [] for name in LUNAR_NAMES}
    solar_amps = {name: [] for name in SOLAR_NAMES}
    for st in sorted(all_results.keys()):
        for i, name in enumerate(LUNAR_NAMES):
            lunar_amps[name].append(all_results[st]["results"][i]["amp"])
        for i, name in enumerate(SOLAR_NAMES):
            solar_amps[name].append(all_results[st]["results"][len(LUNAR_NAMES) + i]["amp"])

    lunar_mean_amps = {name: float(np.mean(v)) for name, v in lunar_amps.items()}
    solar_mean_amps = {name: float(np.mean(v)) for name, v in solar_amps.items()}

    lunar_phase_rs = [phase_results[n]["r"] for n in LUNAR_NAMES if phase_results[n]["r"] is not None]
    solar_phase_rs = [phase_results[n]["r"] for n in SOLAR_NAMES if phase_results[n]["r"] is not None]

    lunar_stack_ps = [stack_results[n]["p_value"] for n in LUNAR_NAMES if stack_results[n]["p_value"] is not None]
    solar_stack_ps = [stack_results[n]["p_value"] for n in SOLAR_NAMES if stack_results[n]["p_value"] is not None]

    print("  === LUNAR vs SOLAR SPLIT ===", flush=True)
    print(f"  Lunar mean amplitudes (ns):", flush=True)
    for name in LUNAR_NAMES:
        print(f"    {name}: {lunar_mean_amps[name]:.3f}", flush=True)
    print(f"  Solar mean amplitudes (ns):", flush=True)
    for name in SOLAR_NAMES:
        print(f"    {name}: {solar_mean_amps[name]:.3f}", flush=True)
    print(flush=True)

    all_lunar_amp = float(np.mean([v for vals in lunar_amps.values() for v in vals]))
    all_solar_amp = float(np.mean([v for vals in solar_amps.values() for v in vals]))
    print(f"  Grand mean lunar amp: {all_lunar_amp:.3f} ns", flush=True)
    print(f"  Grand mean solar amp: {all_solar_amp:.3f} ns", flush=True)
    print(f"  Lunar/solar ratio: {all_lunar_amp / all_solar_amp:.2f}", flush=True)
    print(flush=True)

    print(f"  Phase prediction r (lunar): {lunar_phase_rs}", flush=True)
    print(f"  Phase prediction r (solar): {solar_phase_rs}", flush=True)
    print(f"  Mean lunar r: {np.mean(lunar_phase_rs):.4f}", flush=True)
    print(f"  Mean solar r: {np.mean(solar_phase_rs):.4f}", flush=True)
    print(flush=True)

    print(f"  Stack p-values (lunar): {lunar_stack_ps}", flush=True)
    print(f"  Stack p-values (solar): {solar_stack_ps}", flush=True)
    n_lunar_sig = sum(1 for p in lunar_stack_ps if p < 0.05)
    n_solar_sig = sum(1 for p in solar_stack_ps if p < 0.05)
    print(f"  Lunar significant stacks: {n_lunar_sig}/{len(lunar_stack_ps)}", flush=True)
    print(f"  Solar significant stacks: {n_solar_sig}/{len(solar_stack_ps)}", flush=True)
    print(flush=True)

    # === Build output ===
    per_station = {}
    for st in sorted(all_results.keys()):
        res = all_results[st]["results"]
        per_station[st] = {
            "clock_type": all_results[st]["clock_type"],
            "n_jumps": stations[st]["n_jumps"],
            "residual_std_ns": float(np.std(stations[st]["resid"])),
        }
        for i, name in enumerate(ALL_NAMES):
            per_station[st][name] = {
                "amp_ns": res[i]["amp"],
                "phase": res[i]["phase"],
                "snr": res[i]["snr"],
            }
        per_station[st]["significance"] = sig_results.get(st, {})

    verdict = {
        "step": "152b",
        "title": "Lunar vs solar tidal split in GNSS receiver clock bias",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_days": N_DAYS,
        "n_stations": len(all_results),
        "freq_resolution_uhz": FREQ_RES * 1e6,
        "lunar_lines": LUNAR_NAMES,
        "solar_lines": SOLAR_NAMES,
        "per_station": per_station,
        "phase_prediction": phase_results,
        "stacking": stack_results,
        "lunar_vs_solar": {
            "lunar_mean_amps_ns": lunar_mean_amps,
            "solar_mean_amps_ns": solar_mean_amps,
            "grand_mean_lunar_amp_ns": all_lunar_amp,
            "grand_mean_solar_amp_ns": all_solar_amp,
            "lunar_solar_ratio": all_lunar_amp / all_solar_amp,
            "mean_lunar_phase_r": float(np.mean(lunar_phase_rs)),
            "mean_solar_phase_r": float(np.mean(solar_phase_rs)),
            "n_lunar_sig_stacks": n_lunar_sig,
            "n_solar_sig_stacks": n_solar_sig,
            "n_lunar_total": len(lunar_stack_ps),
            "n_solar_total": len(solar_stack_ps),
        },
        "interpretation": [
            f"Lunar tides (M2, O1, Q1, J1, N2, M3, M4) detected at all {len(all_results)} stations.",
            f"Grand mean lunar amplitude: {all_lunar_amp:.3f} ns vs solar: {all_solar_amp:.3f} ns "
            f"(ratio {all_lunar_amp / all_solar_amp:.2f}).",
            f"Mean cross-station phase prediction r: lunar={np.mean(lunar_phase_rs):.4f}, "
            f"solar={np.mean(solar_phase_rs):.4f}.",
            f"Stack significance: {n_lunar_sig}/{len(lunar_stack_ps)} lunar, "
            f"{n_solar_sig}/{len(solar_stack_ps)} solar at p<0.05.",
            f"M2 phase prediction r={phase_results['M2']['r']:.4f}, p={phase_results['M2']['p']:.4f}.",
            f"S2 phase prediction r={phase_results['S2']['r']:.4f}, p={phase_results['S2']['p']:.4f}."
            if phase_results['S2']['r'] else "S2 phase prediction: insufficient data.",
        ],
        "caveats": [
            "Only 7 stations available with 45-day contiguous data.",
            "K1 (mixed lunar-solar) excluded from the split to keep it clean.",
            "Ocean loading BLQ coefficients for 6/7 stations are estimated.",
            "Clock-type screening is not significant with 4 quartz vs 2 atomic.",
        ],
        "pre_registered_prediction": (
            "TEP predicts lunar tides should be detectable and show cross-station "
            "phase coherence. Atmospheric origin predicts solar tides should dominate. "
            "If lunar phase coherence >= solar phase coherence, the gravitational "
            "field interpretation is supported."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
