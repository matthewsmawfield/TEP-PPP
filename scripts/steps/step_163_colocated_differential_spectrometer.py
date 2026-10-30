#!/usr/bin/env python3
"""
STEP 163  --  Co-located differential spectrometer: M2 vs S2 in the
              quartz-minus-atomic differential cross-spectrum.

The decisive test that existing CORS data can provide.

PRINCIPLE
---------
Two co-located GNSS receivers (same site, < 100 m separation) see identical
satellite clocks, orbits, ionosphere, troposphere, multipath, antenna phase
centre, cable delay, geometry, and co-visibility.  Form

    Delta(t) = resid_internal(t) - resid_external_atomic(t)

and every common-mode term cancels.  What survives is

    (g_q - g_a) * phi(t)  +  (alpha_q - alpha_a) * P_local(t)  +  noise

where g is the TEP coupling, phi(t) is the proper-time field, alpha is the
oscillator's barometric/thermal sensitivity, and P_local(t) is the local
pressure/temperature.

The cross-spectrum between Delta(t) and a REMOTE station's clock isolates
the distance-structured component:

    <conj(F_delta) * F_remote> ~ (g_q - g_a) * g_remote * <conj(phi) * phi>(d)

The local atmospheric term (alpha_q - alpha_a) * P_local does NOT contribute
coherently to the cross-spectrum because local pressure at the co-located
site is not coherent with a remote station 3000 km away at tidal frequencies.

PREDICTION
----------
TEP (gravitational):  M2 (lunar, 22.365 uHz) shows distance structure
                      in the differential cross-spectrum; S2 (solar, 23.148 uHz)
                      does not, because the S2 atmospheric component is local
                      and does not correlate at 3000 km.

Atmospheric:          S2 shows distance structure (if atmospheric pressure
                      correlates at 3000 km, which is unlikely but not
                      impossible); M2 does not.

Null:                 Neither M2 nor S2 shows distance structure in the
                      differential cross-spectrum.

This is the OPPOSITE of Step 162, which measured the clock-type RATIO of
spatial coherence (Q/A) and found S2 > M2.  The ratio includes the
atmospheric oscillator sensitivity; the co-located differential cancels it.

ESTIMATOR
---------
Multi-day FFT (13-day window, 0.891 uHz resolution) of the differential
and each remote station.  Cross-spectral phase alignment at each frequency
bin, fitted to A*exp(-d/lambda) + C0.

OUTPUT
------
results/outputs/step_163_colocated_differential_spectrometer.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

# Reuse Step 53 infrastructure
_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_163_colocated_differential_spectrometer.json"
AUDIT_PATH = ROOT / "data" / "meta" / "clock_anchor_audit_2024.json"
AUDIT = json.loads(AUDIT_PATH.read_text())

PRODUCT = "clk_cod_ops"
N_DAYS_WINDOW = 13       # multi-day FFT window
N_WINDOWS = 10           # number of non-overlapping 13-day windows
N_NULL = 500             # distance permutation null iterations
F_LO_UHZ = 8.0           # low frequency cutoff
F_HI_UHZ = 500.0         # high frequency cutoff
DT = 300.0               # 5-minute cadence (CODE operational product)
PRIMARY_MAX_M = 100.0    # co-located threshold

# Tidal frequencies (uHz)
TIDAL_LINES = {
    "O1":  10.758,  # lunar diurnal
    "Q1":  10.676,  # lunar diurnal
    "S1":  11.574,  # solar diurnal
    "P1":  11.542,  # solar diurnal
    "K1":  11.608,  # luni-solar diurnal
    "N2":  21.422,  # lunar semidiurnal
    "M2":  22.365,  # lunar semidiurnal (purely gravitational)
    "S2":  23.148,  # solar semidiurnal (atmospheric component)
    "K2":  23.098,  # luni-solar semidiurnal
    "M3":  34.533,  # lunar terdiurnal
}

# Off-tide control frequencies (uHz) — well-separated from real tides
CONTROL_FREQS = [15.0, 19.0, 27.0, 31.0, 33.0, 37.0, 41.0, 45.0]


def load_multiday_series(dates, stations_needed):
    """Load and concatenate multiple days of clock residuals for given stations."""
    series = {st: {"t": [], "resid": []} for st in stations_needed}
    station_coords = {}
    day_offset = 0

    for i, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10000)
        except Exception:
            continue
        for st in stations_needed:
            if st not in ser:
                continue
            r = ser[st]
            t = np.asarray(r["t"], float)
            x = np.asarray(r["resid"], float)
            valid = np.isfinite(t) & np.isfinite(x)
            if valid.sum() < 200:
                continue
            # Shift time to multi-day grid
            t_shifted = t[valid] + day_offset
            series[st]["t"].extend(t_shifted.tolist())
            series[st]["resid"].extend(x[valid].tolist())
            if st not in station_coords:
                station_coords[st] = r["station"]
        day_offset += 86400.0

    # Convert to arrays and detrend
    result = {}
    for st, data in series.items():
        if len(data["t"]) < 1000:
            continue
        t = np.asarray(data["t"], float)
        x = np.asarray(data["resid"], float)
        # Linear detrend
        if len(t) > 2:
            coeffs = np.polyfit(t, x, 1)
            x = x - np.polyval(coeffs, t)
        result[st] = {"t": t, "resid": x, "station": station_coords.get(st)}
    return result


def compute_multiday_fft(t, x, dt=DT):
    """Compute FFT on a regular grid."""
    t0 = t.min()
    t1 = t.max()
    n = int(np.round((t1 - t0) / dt)) + 1
    grid = np.arange(n) * dt + t0
    x_grid = np.interp(grid, t, x)
    # Remove mean
    x_grid = x_grid - np.mean(x_grid)
    # FFT
    F = np.fft.rfft(x_grid)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, F


def phase_alignment_at_freq(F_a, F_b, freqs, target_freq_uhz):
    """Compute phase alignment cos(arg(conj(F_a) * F_b)) at a target frequency."""
    target_hz = target_freq_uhz * 1e-6
    # Find nearest frequency bin
    idx = np.argmin(np.abs(freqs - target_hz))
    cross = np.conj(F_a[idx]) * F_b[idx]
    if cross == 0 or not np.isfinite(cross):
        return 0.0, 0.0
    return float(np.cos(np.angle(cross))), float(np.abs(cross))


def phase_alignment_in_band(F_a, F_b, freqs, f_lo_uhz, f_hi_uhz):
    """Sum cross-spectrum over a frequency band and return phase alignment."""
    f_lo_hz = f_lo_uhz * 1e-6
    f_hi_hz = f_hi_uhz * 1e-6
    mask = (freqs >= f_lo_hz) & (freqs <= f_hi_hz)
    if mask.sum() == 0:
        return 0.0, 0
    cross = np.conj(F_a[mask]) * F_b[mask]
    S = np.sum(cross)
    if S == 0 or not np.isfinite(S):
        return 0.0, int(mask.sum())
    return float(np.cos(np.angle(S))), int(mask.sum())


def fit_distance_structure(pairs):
    """Fit A*exp(-d/lambda) + C0 to (distance, phase_cos) pairs."""
    pairs = [(d, v) for d, v in pairs if 100 <= d <= 20000]
    if len(pairs) < 20:
        return None
    arr = np.asarray(pairs, float)
    arr = arr[np.argsort(arr[:, 0])]

    # Equal-count distance bins
    n_bins = min(15, max(6, len(arr) // 15))
    chunks = [c for c in np.array_split(arr, n_bins) if len(c) >= 3]
    if len(chunks) < 6:
        return None
    ctr = np.asarray([np.exp(np.mean(np.log(c[:, 0]))) for c in chunks])
    val = np.asarray([np.mean(c[:, 1]) for c in chunks])
    cnt = np.asarray([len(c) for c in chunks])

    f = s53._fit(ctr, val, cnt)
    if not f:
        return None
    e = f.get("exponential") or {}
    return {
        "lambda_km": e.get("lambda_km"),
        "R2_exponential": e.get("R2"),
        "best_model": f.get("best_model"),
        "n_bins": len(ctr),
        "n_pairs": len(arr),
        "mean_index": float(np.mean([p[1] for p in pairs])),
    }


def distance_permutation_null(pairs, observed_r2, seed):
    """Permute distances, preserve phase-cos values, refit."""
    if not pairs or observed_r2 is None:
        return None
    rng = np.random.default_rng(seed)
    distances = np.asarray([p[0] for p in pairs], float)
    values = np.asarray([p[1] for p in pairs], float)
    r2_list, lam_list = [], []
    for _ in range(N_NULL):
        f = fit_distance_structure(list(zip(rng.permutation(distances), values)))
        if f and f["R2_exponential"] is not None and f["lambda_km"] is not None:
            r2_list.append(f["R2_exponential"])
            lam_list.append(f["lambda_km"])
    r2_arr = np.asarray(r2_list)
    lam_arr = np.asarray(lam_list)
    if not len(r2_arr):
        return None
    return {
        "n_realisations": int(len(r2_arr)),
        "R2_median": float(np.median(r2_arr)),
        "R2_max": float(np.max(r2_arr)),
        "lambda_median_km": float(np.median(lam_arr)),
        "p_R2_ge_observed": float((1 + np.sum(r2_arr >= observed_r2)) / (1 + len(r2_arr))),
        "p_TEP_band_and_R2": float((1 + np.sum(
            (r2_arr >= observed_r2) & (lam_arr >= 1000) & (lam_arr <= 10000))) / (1 + len(r2_arr))),
    }


def analyze_window(dates, internal, atomic, all_stations):
    """Analyze one 13-day window for a co-located pair."""
    needed = [internal, atomic] + [s for s in all_stations if s not in (internal, atomic)]
    # Limit remote stations to a manageable number
    if len(needed) > 160:
        needed = needed[:160]

    series = load_multiday_series(dates, needed)
    if internal not in series or atomic not in series:
        return None

    # Compute differential
    si = series[internal]
    sa = series[atomic]
    # Interpolate both to a common grid
    t0 = max(si["t"].min(), sa["t"].min())
    t1 = min(si["t"].max(), sa["t"].max())
    if t1 - t0 < 800000:  # need at least ~9 days
        return None
    n = int(np.round((t1 - t0) / DT)) + 1
    grid = np.arange(n) * DT + t0
    qi = np.interp(grid, si["t"], si["resid"])
    qa = np.interp(grid, sa["t"], sa["resid"])
    delta = qi - qa
    delta = delta - np.mean(delta)

    # Multi-day FFT of the differential
    F_delta = np.fft.rfft(delta)
    freqs = np.fft.rfftfreq(n, d=DT)

    # For each remote station, compute FFT and cross-spectrum with differential
    site_coord = si["station"]  # co-located site coordinates
    results = {
        "n_samples": n,
        "duration_s": float(t1 - t0),
        "freq_resolution_uhz": float(1e6 / (n * DT)),
    }

    # Per-frequency line analysis
    line_results = {}
    for line_name, freq_uhz in TIDAL_LINES.items():
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs - target_hz))
        actual_freq = freqs[idx] * 1e6  # uHz

        # Accumulate phase alignment vs distance for each remote station
        pairs = []
        for st, sdata in series.items():
            if st in (internal, atomic) or "station" not in sdata:
                continue
            d = td.great_circle_km(site_coord, sdata["station"])
            if d < 100 or d > 20000:
                continue
            # Compute remote FFT on the same grid
            t_r = sdata["t"]
            x_r = sdata["resid"]
            # Interpolate to the same grid
            try:
                r_grid = np.interp(grid, t_r, x_r)
                r_grid = r_grid - np.mean(r_grid)
                F_r = np.fft.rfft(r_grid)
            except Exception:
                continue
            if F_r.shape != F_delta.shape:
                continue
            cross = np.conj(F_delta[idx]) * F_r[idx]
            if cross != 0 and np.isfinite(cross):
                pc = float(np.cos(np.angle(cross)))
                pairs.append((d, pc))

        if len(pairs) < 20:
            line_results[line_name] = {
                "freq_uhz": freq_uhz, "actual_freq_uhz": actual_freq,
                "n_pairs": len(pairs), "fit": None, "null": None,
            }
            continue

        fit = fit_distance_structure(pairs)
        null = distance_permutation_null(
            pairs, fit["R2_exponential"] if fit else None,
            seed=16300 + int(freq_uhz * 1000)) if fit else None

        line_results[line_name] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": actual_freq,
            "n_pairs": len(pairs),
            "fit": fit,
            "null": null,
            "mean_phase_cos": float(np.mean([p[1] for p in pairs])),
        }

    # Control frequencies
    control_results = {}
    for freq_uhz in CONTROL_FREQS:
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs - target_hz))
        actual_freq = freqs[idx] * 1e6

        pairs = []
        for st, sdata in series.items():
            if st in (internal, atomic) or "station" not in sdata:
                continue
            d = td.great_circle_km(site_coord, sdata["station"])
            if d < 100 or d > 20000:
                continue
            try:
                r_grid = np.interp(grid, sdata["t"], sdata["resid"])
                r_grid = r_grid - np.mean(r_grid)
                F_r = np.fft.rfft(r_grid)
            except Exception:
                continue
            if F_r.shape != F_delta.shape:
                continue
            cross = np.conj(F_delta[idx]) * F_r[idx]
            if cross != 0 and np.isfinite(cross):
                pc = float(np.cos(np.angle(cross)))
                pairs.append((d, pc))

        fit = fit_distance_structure(pairs) if len(pairs) >= 20 else None
        null = distance_permutation_null(
            pairs, fit["R2_exponential"] if fit else None,
            seed=16300 + int(freq_uhz * 1000)) if fit else None

        control_results[f"{freq_uhz:.0f}uHz"] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": actual_freq,
            "n_pairs": len(pairs),
            "fit": fit,
            "null": null,
            "mean_phase_cos": float(np.mean([p[1] for p in pairs])) if pairs else 0.0,
        }

    # Broadband (full TEP band) differential
    bb_pairs = []
    f_lo_hz = F_LO_UHZ * 1e-6
    f_hi_hz = F_HI_UHZ * 1e-6
    mask = (freqs >= f_lo_hz) & (freqs <= f_hi_hz)
    for st, sdata in series.items():
        if st in (internal, atomic) or "station" not in sdata:
            continue
        d = td.great_circle_km(site_coord, sdata["station"])
        if d < 100 or d > 20000:
            continue
        try:
            r_grid = np.interp(grid, sdata["t"], sdata["resid"])
            r_grid = r_grid - np.mean(r_grid)
            F_r = np.fft.rfft(r_grid)
        except Exception:
            continue
        if F_r.shape != F_delta.shape:
            continue
        cross = np.conj(F_delta[mask]) * F_r[mask]
        S = np.sum(cross)
        if S != 0 and np.isfinite(S):
            pc = float(np.cos(np.angle(S)))
            bb_pairs.append((d, pc))

    bb_fit = fit_distance_structure(bb_pairs) if len(bb_pairs) >= 20 else None
    bb_null = distance_permutation_null(
        bb_pairs, bb_fit["R2_exponential"] if bb_fit else None,
        seed=16399) if bb_fit else None

    results["broadband"] = {
        "f_lo_uhz": F_LO_UHZ, "f_hi_uhz": F_HI_UHZ,
        "n_pairs": len(bb_pairs),
        "fit": bb_fit, "null": bb_null,
        "mean_phase_cos": float(np.mean([p[1] for p in bb_pairs])) if bb_pairs else 0.0,
    }
    results["tidal_lines"] = line_results
    results["control_freqs"] = control_results

    # Also compute the differential power spectrum at tidal frequencies
    power_spectrum = np.abs(F_delta) ** 2
    power_at_tides = {}
    for line_name, freq_uhz in TIDAL_LINES.items():
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs - target_hz))
        # Average power in a small window around the target
        window = 3
        lo = max(0, idx - window)
        hi = min(len(power_spectrum), idx + window + 1)
        power_at_tides[line_name] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": float(freqs[idx] * 1e6),
            "power": float(np.mean(power_spectrum[lo:hi])),
            "peak_power": float(power_spectrum[idx]),
        }
    results["differential_power_at_tides"] = power_at_tides

    return results


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates_2024 = td.available_dates(PRODUCT, 2024)

    # Find co-located pairs from audit
    cands = []
    for _, c in AUDIT["pairs"].items():
        if c.get("exclusion_reason"):
            continue
        pair = (c["external_station"], c["internal_station"])
        sep_m = c["separation_m"]
        if sep_m <= PRIMARY_MAX_M and c.get("valid_internal_vs_free_atomic_contrast"):
            cands.append((pair[0], pair[1], sep_m))

    if not cands:
        OUT.write_text(json.dumps({"step": 163, "status": "no_colocated_pairs"}))
        print("no co-located pairs found")
        return

    print(f"STEP 163  |  {PRODUCT}, {N_DAYS_WINDOW}-day windows, {N_WINDOWS} windows")
    print(f"  Co-located pairs: {[(a, i) for a, i, _ in cands]}")

    # Get all station names from first available day
    ser0 = td.load_series(PRODUCT, dates_2024[0], min_valid_frac=0.0, max_jumps=10000)
    all_stations = list(ser0.keys())

    all_results = {}
    for atomic, internal, sep_m in cands:
        pair_name = f"{internal}-{atomic}"
        print(f"\n  === {pair_name} (sep={sep_m:.1f} m) ===")

        window_results = []
        for w in range(N_WINDOWS):
            start = w * N_DAYS_WINDOW
            end = start + N_DAYS_WINDOW
            if end > len(dates_2024):
                break
            window_dates = dates_2024[start:end]
            print(f"    window {w+1}: days {start}-{end-1} ({window_dates[0]}-{window_dates[-1]})", flush=True)

            result = analyze_window(window_dates, internal, atomic, all_stations)
            if result:
                result["window"] = w
                result["dates"] = window_dates
                window_results.append(result)

                # Print key results
                bb = result.get("broadband", {})
                bb_fit = bb.get("fit") or {}
                print(f"      broadband: lambda={bb_fit.get('lambda_km')}, "
                      f"R2={bb_fit.get('R2_exponential')}, n={bb.get('n_pairs')}")

                for line in ("M2", "S2"):
                    lr = result.get("tidal_lines", {}).get(line, {})
                    lf = lr.get("fit") or {}
                    print(f"      {line}: lambda={lf.get('lambda_km')}, "
                          f"R2={lf.get('R2_exponential')}, n={lr.get('n_pairs')}")
            else:
                print(f"      window {w+1}: insufficient data")

        if not window_results:
            print(f"    {pair_name}: no usable windows")
            continue

        # Aggregate across windows
        agg = aggregate_windows(window_results)
        agg["separation_m"] = sep_m
        agg["n_windows"] = len(window_results)
        all_results[pair_name] = agg

    # Compute summary statistics
    summary = compute_summary(all_results)

    output = {
        "step": 163,
        "title": "Co-located differential spectrometer: M2 vs S2 in the "
                 "quartz-minus-atomic differential cross-spectrum",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days_window": N_DAYS_WINDOW,
        "n_windows": N_WINDOWS,
        "freq_resolution_uhz": float(1e6 / (N_DAYS_WINDOW * 86400)),
        "design": "Multi-day FFT of co-located (internal - external_atomic) differential. "
                  "Cross-spectrum with remote stations at each tidal frequency. "
                  "The differential cancels satellite, atmospheric, orbital, and processing "
                  "common-mode. The cross-spectrum with remote stations isolates the "
                  "distance-structured gravitational component from local atmospheric effects.",
        "prediction": "TEP: M2 (lunar, gravitational) shows distance structure in the "
                      "differential cross-spectrum; S2 (solar, atmospheric) does not. "
                      "Atmospheric: S2 shows distance structure; M2 does not. "
                      "Null: neither shows distance structure.",
        "pairs": all_results,
        "summary": summary,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nSUMMARY:\n{json.dumps(summary, indent=2)}")


def aggregate_windows(window_results):
    """Aggregate per-frequency results across windows."""
    agg = {"windows": window_results}

    # Aggregate tidal line fits
    tidal_agg = {}
    for line in list(TIDAL_LINES.keys()):
        fits = []
        nulls = []
        for w in window_results:
            lr = w.get("tidal_lines", {}).get(line, {})
            f = lr.get("fit")
            if f and f.get("R2_exponential") is not None:
                fits.append(f)
            n = lr.get("null")
            if n:
                nulls.append(n)

        if fits:
            r2_vals = [f["R2_exponential"] for f in fits if f["R2_exponential"] is not None]
            lam_vals = [f["lambda_km"] for f in fits if f["lambda_km"] is not None]
            tidal_agg[line] = {
                "n_windows_with_fit": len(fits),
                "R2_mean": float(np.mean(r2_vals)) if r2_vals else None,
                "R2_median": float(np.median(r2_vals)) if r2_vals else None,
                "lambda_mean_km": float(np.mean(lam_vals)) if lam_vals else None,
                "lambda_median_km": float(np.median(lam_vals)) if lam_vals else None,
                "n_pairs_mean": float(np.mean([f["n_pairs"] for f in fits])),
                "p_null_median": float(np.median([n["p_R2_ge_observed"] for n in nulls])) if nulls else None,
            }
        else:
            tidal_agg[line] = {"n_windows_with_fit": 0}

    # Aggregate control frequencies
    control_agg = {}
    for w in window_results:
        for cf_name, cf_data in w.get("control_freqs", {}).items():
            if cf_name not in control_agg:
                control_agg[cf_name] = []
            f = cf_data.get("fit")
            if f and f.get("R2_exponential") is not None:
                control_agg[cf_name].append(f)

    control_summary = {}
    for cf_name, fits in control_agg.items():
        if fits:
            r2_vals = [f["R2_exponential"] for f in fits if f["R2_exponential"] is not None]
            lam_vals = [f["lambda_km"] for f in fits if f["lambda_km"] is not None]
            control_summary[cf_name] = {
                "n_windows": len(fits),
                "R2_mean": float(np.mean(r2_vals)) if r2_vals else None,
                "R2_median": float(np.median(r2_vals)) if r2_vals else None,
                "lambda_mean_km": float(np.mean(lam_vals)) if lam_vals else None,
            }

    # Aggregate broadband
    bb_fits = []
    for w in window_results:
        f = w.get("broadband", {}).get("fit")
        if f and f.get("R2_exponential") is not None:
            bb_fits.append(f)

    bb_summary = {}
    if bb_fits:
        r2_vals = [f["R2_exponential"] for f in bb_fits]
        lam_vals = [f["lambda_km"] for f in bb_fits]
        bb_summary = {
            "n_windows": len(bb_fits),
            "R2_mean": float(np.mean(r2_vals)),
            "R2_median": float(np.median(r2_vals)),
            "lambda_mean_km": float(np.mean(lam_vals)),
            "lambda_median_km": float(np.median(lam_vals)),
        }

    agg["tidal_lines_aggregated"] = tidal_agg
    agg["control_freqs_aggregated"] = control_summary
    agg["broadband_aggregated"] = bb_summary
    return agg


def compute_summary(all_results):
    """Compute the headline summary across all pairs."""
    summary = {"pairs": {}}

    for pair_name, pair_data in all_results.items():
        tidal = pair_data.get("tidal_lines_aggregated", {})
        control = pair_data.get("control_freqs_aggregated", {})
        bb = pair_data.get("broadband_aggregated", {})

        # M2 vs S2 comparison
        m2 = tidal.get("M2", {})
        s2 = tidal.get("S2", {})

        m2_r2 = m2.get("R2_mean")
        s2_r2 = s2.get("R2_mean")
        m2_lam = m2.get("lambda_mean_km")
        s2_lam = s2.get("lambda_mean_km")
        m2_p = m2.get("p_null_median")
        s2_p = s2.get("p_null_median")

        # Control frequency average
        ctrl_r2s = [c.get("R2_mean") for c in control.values() if c.get("R2_mean") is not None]
        ctrl_r2_mean = float(np.mean(ctrl_r2s)) if ctrl_r2s else None

        # Lunar vs solar means
        lunar_lines = ["O1", "Q1", "N2", "M2", "M3"]
        solar_lines = ["S1", "P1", "K1", "S2", "K2"]
        lunar_r2s = [tidal.get(l, {}).get("R2_mean") for l in lunar_lines
                     if tidal.get(l, {}).get("R2_mean") is not None]
        solar_r2s = [tidal.get(l, {}).get("R2_mean") for l in solar_lines
                     if tidal.get(l, {}).get("R2_mean") is not None]
        lunar_mean_r2 = float(np.mean(lunar_r2s)) if lunar_r2s else None
        solar_mean_r2 = float(np.mean(solar_r2s)) if solar_r2s else None

        # Verdict
        m2_in_tep_band = m2_lam is not None and 1000 <= m2_lam <= 10000
        s2_in_tep_band = s2_lam is not None and 1000 <= s2_lam <= 10000
        m2_significant = m2_p is not None and m2_p < 0.05
        s2_significant = s2_p is not None and s2_p < 0.05

        if m2_significant and m2_in_tep_band and not s2_significant:
            verdict = "M2_ONLY_TEP_CONSISTENT"
        elif s2_significant and s2_in_tep_band and not m2_significant:
            verdict = "S2_ONLY_ATMOSPHERIC"
        elif m2_significant and s2_in_tep_band and s2_significant and s2_in_tep_band:
            if m2_r2 > s2_r2:
                verdict = "BOTH_BUT_M2_STRONGER"
            else:
                verdict = "BOTH_BUT_S2_STRONGER"
        elif m2_significant and m2_in_tep_band:
            verdict = "M2_SIGNIFICANT_S2_NOT_TESTED"
        else:
            verdict = "NULL"

        summary["pairs"][pair_name] = {
            "M2_R2": m2_r2,
            "M2_lambda_km": m2_lam,
            "M2_p_null": m2_p,
            "M2_significant": m2_significant,
            "M2_in_TEP_band": m2_in_tep_band,
            "S2_R2": s2_r2,
            "S2_lambda_km": s2_lam,
            "S2_p_null": s2_p,
            "S2_significant": s2_significant,
            "S2_in_TEP_band": s2_in_tep_band,
            "control_R2_mean": ctrl_r2_mean,
            "lunar_mean_R2": lunar_mean_r2,
            "solar_mean_R2": solar_mean_r2,
            "broadband_R2_mean": bb.get("R2_mean"),
            "broadband_lambda_mean": bb.get("lambda_mean_km"),
            "verdict": verdict,
        }

    return summary


if __name__ == "__main__":
    main()
