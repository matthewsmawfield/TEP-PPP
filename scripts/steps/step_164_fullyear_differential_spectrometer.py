#!/usr/bin/env python3
"""
STEP 164  --  Co-located differential spectrometer: coherent multi-window
              accumulation with per-window detrending.

Step 163 used independent 13-day windows and averaged the per-window R2
values.  The result was positive but noisy: M2 R2=0.450 (2024), 0.635 (2025).

This step improves on Step 163 by accumulating the cross-spectrum
COHERENTLY across many 13-day windows.  Each window is independently
detrended (removing drift and jumps), but the cross-spectral products
conj(F_delta) * F_remote are accumulated across windows before computing
the phase alignment.  This gives:

  - The frequency resolution of a 13-day window (0.891 uHz)
  - The SNR of the full year (361/13 = 28 windows)
  - Per-window drift/jump removal

The phase alignment at frequency f is:

  cos(arg(sum_windows sum_remote conj(F_delta_w[f]) * F_remote_w[f]))

This is the coherent average of the cross-spectrum across all windows and
all remote stations, then the phase alignment is computed from the sum.

OUTPUT
------
results/outputs/step_164_fullyear_differential_spectrometer.json
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

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_164_fullyear_differential_spectrometer.json"
AUDIT_PATH = ROOT / "data" / "meta" / "clock_anchor_audit_2024.json"
AUDIT = json.loads(AUDIT_PATH.read_text()) if AUDIT_PATH.exists() else {"pairs": {}}

PRODUCT = "clk_cod_ops"
DT = 300.0
WINDOW_DAYS = 13
STEP_DAYS = 7           # overlap windows by 6 days
N_NULL = 1000

TIDAL_LINES = {
    "O1": 10.758, "Q1": 10.676, "S1": 11.574, "P1": 11.542, "K1": 11.608,
    "N2": 21.422, "M2": 22.365, "S2": 23.148, "K2": 23.098, "M3": 34.533,
}
CONTROL_FREQS = [13.0, 15.0, 17.0, 19.0, 25.0, 27.0, 29.0, 31.0, 33.0, 37.0, 39.0, 41.0, 43.0, 45.0]


def load_window_series(dates, stations_needed):
    """Load one 13-day window of clock residuals."""
    series = {st: {"t": [], "resid": []} for st in stations_needed}
    station_coords = {}
    day_offset = 0
    for tag in dates:
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
            series[st]["t"].extend((t[valid] + day_offset).tolist())
            series[st]["resid"].extend(x[valid].tolist())
            if st not in station_coords:
                station_coords[st] = r["station"]
        day_offset += 86400
    result = {}
    for st, data in series.items():
        if len(data["t"]) < 1000:
            continue
        t = np.asarray(data["t"], float)
        x = np.asarray(data["resid"], float)
        if len(t) > 2:
            coeffs = np.polyfit(t, x, 1)
            x = x - np.polyval(coeffs, t)
        result[st] = {"t": t, "resid": x, "station": station_coords.get(st)}
    return result


def compute_window_fft(t, x, dt=DT):
    """Compute FFT on a regular grid for one window."""
    t0 = t.min()
    t1 = t.max()
    n = int(np.round((t1 - t0) / dt)) + 1
    grid = np.arange(n) * dt + t0
    x_grid = np.interp(grid, t, x)
    x_grid = x_grid - np.mean(x_grid)
    F = np.fft.rfft(x_grid)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, F, n


def fit_distance_structure(pairs):
    pairs = [(d, v) for d, v in pairs if 100 <= d <= 20000]
    if len(pairs) < 20:
        return None
    arr = np.asarray(pairs, float)
    arr = arr[np.argsort(arr[:, 0])]
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


def analyze_fullyear(dates, internal, atomic, all_stations, year_label):
    """Coherent multi-window accumulation for a co-located pair."""
    needed = [internal, atomic] + [s for s in all_stations if s not in (internal, atomic)]
    if len(needed) > 160:
        needed = needed[:160]

    # Generate overlapping windows
    windows = []
    for start in range(0, len(dates) - WINDOW_DAYS + 1, STEP_DAYS):
        windows.append(dates[start:start + WINDOW_DAYS])
    print(f"  {len(windows)} overlapping {WINDOW_DAYS}-day windows "
          f"(step={STEP_DAYS} days)", flush=True)

    # For each window, compute FFTs and accumulate cross-spectral products
    # at each frequency bin, per remote station
    site_coord = None
    remote_coords = {}
    n_freq = None
    freqs_ref = None

    # Accumulators: for each remote station, accumulate conj(F_delta) * F_remote
    # across windows.  Keyed by station name.
    cross_accum = {}  # st -> complex array (per freq bin)
    cross_count = {}  # st -> int (number of windows accumulated)

    # Also accumulate differential power spectrum
    delta_power_accum = None
    delta_power_count = 0

    # Also accumulate individual clock cross-spectra
    internal_cross_accum = {}
    atomic_cross_accum = {}

    for w_idx, window_dates in enumerate(windows):
        if (w_idx + 1) % 10 == 0:
            print(f"    window {w_idx+1}/{len(windows)}", flush=True)
        series = load_window_series(window_dates, needed)
        if internal not in series or atomic not in series:
            continue

        si = series[internal]
        sa = series[atomic]
        if site_coord is None:
            site_coord = si["station"]

        # Build differential on common grid
        t0 = max(si["t"].min(), sa["t"].min())
        t1 = min(si["t"].max(), sa["t"].max())
        if t1 - t0 < 800000:
            continue
        n = int(np.round((t1 - t0) / DT)) + 1
        grid = np.arange(n) * DT + t0
        qi = np.interp(grid, si["t"], si["resid"])
        qa = np.interp(grid, sa["t"], sa["resid"])
        delta = qi - qa
        delta = delta - np.mean(delta)

        # Window FFT
        F_delta = np.fft.rfft(delta)
        freqs = np.fft.rfftfreq(n, d=DT)
        F_internal = np.fft.rfft(qi - np.mean(qi))
        F_atomic = np.fft.rfft(qa - np.mean(qa))

        if n_freq is None:
            n_freq = len(freqs)
            freqs_ref = freqs
            delta_power_accum = np.zeros(n_freq)
        elif len(freqs) != n_freq:
            # Different window length — skip (shouldn't happen with fixed WINDOW_DAYS)
            continue

        # Accumulate differential power
        delta_power_accum += np.abs(F_delta) ** 2
        delta_power_count += 1

        # For each remote station, compute and accumulate cross-spectrum
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

            if st not in cross_accum:
                cross_accum[st] = np.zeros(n_freq, dtype=complex)
                cross_count[st] = 0
                internal_cross_accum[st] = np.zeros(n_freq, dtype=complex)
                atomic_cross_accum[st] = np.zeros(n_freq, dtype=complex)
                remote_coords[st] = sdata["station"]

            cross_accum[st] += np.conj(F_delta) * F_r
            internal_cross_accum[st] += np.conj(F_internal) * F_r
            atomic_cross_accum[st] += np.conj(F_atomic) * F_r
            cross_count[st] += 1

    if delta_power_count == 0 or site_coord is None:
        return None

    print(f"  Accumulated {delta_power_count} windows, {len(cross_accum)} remote stations", flush=True)

    # Average differential power spectrum
    delta_power_avg = delta_power_accum / delta_power_count

    # Noise floor
    off_tide_mask = np.ones(n_freq, bool)
    for f_uhz in list(TIDAL_LINES.values()) + CONTROL_FREQS:
        for idx in range(n_freq):
            if abs(freqs_ref[idx] * 1e6 - f_uhz) < 1.0:
                off_tide_mask[idx] = False
    noise_floor = float(np.median(delta_power_avg[off_tide_mask & (freqs_ref > 5e-6) & (freqs_ref < 50e-6)]))

    results = {
        "year": year_label,
        "n_windows": delta_power_count,
        "n_remote_stations": len(cross_accum),
        "freq_resolution_uhz": float(1e6 / (WINDOW_DAYS * 86400)),
        "noise_floor": noise_floor,
    }

    # Power spectrum at tidal lines
    power_results = {}
    for line_name, freq_uhz in TIDAL_LINES.items():
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs_ref - target_hz))
        actual_freq = freqs_ref[idx] * 1e6
        power_results[line_name] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": actual_freq,
            "power": float(delta_power_avg[idx]),
            "snr": float(delta_power_avg[idx] / noise_floor) if noise_floor > 0 else 0,
        }
    results["power_at_tides"] = power_results

    # Cross-spectral distance fits at each tidal line
    # Using the COHERENTLY ACCUMULATED cross-spectrum
    print(f"  Computing coherently accumulated cross-spectral fits...", flush=True)
    line_results = {}
    for line_name, freq_uhz in TIDAL_LINES.items():
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs_ref - target_hz))
        actual_freq = freqs_ref[idx] * 1e6

        # Build (distance, phase_cos) pairs from accumulated cross-spectrum
        pairs = []
        pairs_internal = []
        pairs_atomic = []
        for st, cross in cross_accum.items():
            if cross_count[st] < 5:
                continue
            station_coord = remote_coords[st]
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue
            # Average cross-spectrum
            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                pc = float(np.cos(np.angle(avg_cross[idx])))
                pairs.append((d, pc))

            avg_int = internal_cross_accum[st] / cross_count[st]
            if avg_int[idx] != 0 and np.isfinite(avg_int[idx]):
                pc_i = float(np.cos(np.angle(avg_int[idx])))
                pairs_internal.append((d, pc_i))

            avg_atm = atomic_cross_accum[st] / cross_count[st]
            if avg_atm[idx] != 0 and np.isfinite(avg_atm[idx]):
                pc_a = float(np.cos(np.angle(avg_atm[idx])))
                pairs_atomic.append((d, pc_a))

        fit = fit_distance_structure(pairs) if len(pairs) >= 20 else None
        null = distance_permutation_null(
            pairs, fit["R2_exponential"] if fit else None,
            seed=16400 + int(freq_uhz * 1000)) if fit else None
        fit_int = fit_distance_structure(pairs_internal) if len(pairs_internal) >= 20 else None
        fit_atm = fit_distance_structure(pairs_atomic) if len(pairs_atomic) >= 20 else None

        line_results[line_name] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": actual_freq,
            "n_pairs": len(pairs),
            "differential_fit": fit,
            "differential_null": null,
            "internal_fit": fit_int,
            "atomic_fit": fit_atm,
            "mean_phase_cos": float(np.mean([p[1] for p in pairs])) if pairs else 0.0,
        }
        if fit:
            p_str = f"p={null['p_R2_ge_observed']:.4f}" if null else "p=N/A"
            print(f"    {line_name} ({freq_uhz:.3f} uHz): "
                  f"R2={fit['R2_exponential']:.3f}, lambda={fit['lambda_km']:.0f} km, {p_str}, "
                  f"n={len(pairs)}")
    results["tidal_lines"] = line_results

    # Control frequencies
    print(f"  Computing control frequencies...", flush=True)
    control_results = {}
    for freq_uhz in CONTROL_FREQS:
        target_hz = freq_uhz * 1e-6
        idx = np.argmin(np.abs(freqs_ref - target_hz))
        pairs = []
        for st, cross in cross_accum.items():
            if cross_count[st] < 5:
                continue
            station_coord = remote_coords[st]
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue
            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                pc = float(np.cos(np.angle(avg_cross[idx])))
                pairs.append((d, pc))
        fit = fit_distance_structure(pairs) if len(pairs) >= 20 else None
        null = distance_permutation_null(
            pairs, fit["R2_exponential"] if fit else None,
            seed=16600 + int(freq_uhz * 1000)) if fit else None
        control_results[f"{freq_uhz:.0f}uHz"] = {
            "freq_uhz": freq_uhz,
            "actual_freq_uhz": float(freqs_ref[idx] * 1e6),
            "n_pairs": len(pairs),
            "fit": fit,
            "null": null,
        }
    results["control_freqs"] = control_results

    # Broadband
    f_lo_hz = 8.0e-6
    f_hi_hz = 500.0e-6
    mask = (freqs_ref >= f_lo_hz) & (freqs_ref <= f_hi_hz)
    bb_pairs = []
    for st, cross in cross_accum.items():
        if cross_count[st] < 5:
            continue
        station_coord = remote_coords[st]
        d = td.great_circle_km(site_coord, station_coord)
        if d < 100 or d > 20000:
            continue
        avg_cross = cross / cross_count[st]
        S = np.sum(avg_cross[mask])
        if S != 0 and np.isfinite(S):
            pc = float(np.cos(np.angle(S)))
            bb_pairs.append((d, pc))
    bb_fit = fit_distance_structure(bb_pairs) if len(bb_pairs) >= 20 else None
    bb_null = distance_permutation_null(
        bb_pairs, bb_fit["R2_exponential"] if bb_fit else None,
        seed=16499) if bb_fit else None
    results["broadband"] = {"n_pairs": len(bb_pairs), "fit": bb_fit, "null": bb_null}

    return results


def compute_summary(all_results):
    summary = {"pairs": {}}
    for pair_name, pair_data in all_results.items():
        pair_summary = {}
        for year, result in pair_data.items():
            tidal = result.get("tidal_lines", {})
            control = result.get("control_freqs", {})
            power = result.get("power_at_tides", {})
            bb = result.get("broadband", {})

            m2 = tidal.get("M2", {})
            s2 = tidal.get("S2", {})
            m2_fit = m2.get("differential_fit") or {}
            s2_fit = s2.get("differential_fit") or {}
            m2_null = m2.get("differential_null") or {}
            s2_null = s2.get("differential_null") or {}

            ctrl_r2s = [c.get("fit", {}).get("R2_exponential") for c in control.values()
                        if c.get("fit") and c["fit"].get("R2_exponential") is not None]
            ctrl_r2_mean = float(np.mean(ctrl_r2s)) if ctrl_r2s else None

            lunar_lines = ["O1", "Q1", "N2", "M2", "M3"]
            solar_lines = ["S1", "P1", "K1", "S2", "K2"]
            lunar_r2s = [tidal.get(l, {}).get("differential_fit", {}).get("R2_exponential")
                         for l in lunar_lines
                         if tidal.get(l, {}).get("differential_fit", {}).get("R2_exponential") is not None]
            solar_r2s = [tidal.get(l, {}).get("differential_fit", {}).get("R2_exponential")
                         for l in solar_lines
                         if tidal.get(l, {}).get("differential_fit", {}).get("R2_exponential") is not None]
            lunar_mean_r2 = float(np.mean(lunar_r2s)) if lunar_r2s else None
            solar_mean_r2 = float(np.mean(solar_r2s)) if solar_r2s else None

            m2_snr = power.get("M2", {}).get("snr", 0)
            s2_snr = power.get("S2", {}).get("snr", 0)

            m2_int = (m2.get("internal_fit") or {}).get("R2_exponential")
            m2_atm = (m2.get("atomic_fit") or {}).get("R2_exponential")
            s2_int = (s2.get("internal_fit") or {}).get("R2_exponential")
            s2_atm = (s2.get("atomic_fit") or {}).get("R2_exponential")

            pair_summary[year] = {
                "M2_R2": m2_fit.get("R2_exponential"),
                "M2_lambda_km": m2_fit.get("lambda_km"),
                "M2_p_null": m2_null.get("p_R2_ge_observed"),
                "M2_n_pairs": m2.get("n_pairs"),
                "M2_snr": m2_snr,
                "M2_internal_R2": m2_int,
                "M2_atomic_R2": m2_atm,
                "S2_R2": s2_fit.get("R2_exponential"),
                "S2_lambda_km": s2_fit.get("lambda_km"),
                "S2_p_null": s2_null.get("p_R2_ge_observed"),
                "S2_n_pairs": s2.get("n_pairs"),
                "S2_snr": s2_snr,
                "S2_internal_R2": s2_int,
                "S2_atomic_R2": s2_atm,
                "control_R2_mean": ctrl_r2_mean,
                "lunar_mean_R2": lunar_mean_r2,
                "solar_mean_R2": solar_mean_r2,
                "broadband_R2": (bb.get("fit") or {}).get("R2_exponential"),
                "broadband_lambda": (bb.get("fit") or {}).get("lambda_km"),
                "n_windows": result.get("n_windows"),
                "n_remote_stations": result.get("n_remote_stations"),
            }
        summary["pairs"][pair_name] = pair_summary
    return summary


def _compute_tidal_band_colocated() -> dict:
    """Evaluate the 10-35 uHz gravitational tidal band on the 348-day
    station-collapsed co-located differential fingerprints (Step 77) so
    the co-located spatial decay scale is also resolved over the full
    tidal band, not only a single 0.89 uHz FFT bin."""
    s77_p = ROOT / "results" / "outputs" / "step_77_colocated_screening_differential.json"
    if not s77_p.exists():
        return {}
    s77_arms = json.loads(s77_p.read_text()).get("arms", {})
    out = {}

    def _fit_s77(pairs):
        pairs = [p for p in pairs if s53.DMIN_KM <= p[0] <= s53.DMAX_KM]
        if len(pairs) < 20:
            return None
        arr = np.asarray(pairs, float)
        arr = arr[np.argsort(arr[:, 0])]
        n_bins_target = min(15, max(6, len(arr) // 15))
        chunks = [c for c in np.array_split(arr, n_bins_target) if len(c) >= 3]
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
            "n_reference_stations": len(arr),
            "mean_index": float(np.mean(arr[:, 1])),
        }

    def _perm_s77(pairs, obs_r2, seed=16477):
        rng = np.random.default_rng(seed)
        dists = np.asarray([p[0] for p in pairs], float)
        vals = np.asarray([p[1] for p in pairs], float)
        r2s, lams = [], []
        for _ in range(N_NULL):
            f = _fit_s77(list(zip(rng.permutation(dists), vals)))
            if f and f["R2_exponential"] is not None and f["lambda_km"] is not None:
                r2s.append(float(f["R2_exponential"]))
                lams.append(float(f["lambda_km"]))
        r2s = np.asarray(r2s)
        lams = np.asarray(lams)
        return {
            "unit": "distant reference station (daily repeats collapsed)",
            "n_realisations": int(len(r2s)),
            "R2_median": float(np.median(r2s)),
            "lambda_median_km": float(np.median(lams)),
            "p_R2_ge_observed": float((1 + np.sum(r2s >= obs_r2)) / (1 + len(r2s))),
        }

    for arm_name in ("MAT1-MATE", "THTI-THTG"):
        arm = s77_arms.get(arm_name)
        if not arm:
            continue
        freqs = np.asarray(arm.get("mode_frequency_uhz", []), float)
        fps = arm.get("frequency_resolved_delta_fingerprints", [])
        if not len(freqs) or not fps:
            continue
        t_mask = (freqs >= 10.0) & (freqs <= 35.0)
        c_mask = (freqs > 35.0) & (freqs <= 500.0)
        arm_out = {}
        for label, mask in (("tidal_band_10_35uHz", t_mask),
                            ("continuum_35_500uHz", c_mask)):
            pairs = [
                (float(fp["distance_km"]),
                 float(np.mean(np.asarray(fp["mean_phase_cos_by_frequency"], float)[mask])))
                for fp in fps
            ]
            fit = _fit_s77(pairs)
            null = _perm_s77(pairs, fit["R2_exponential"], seed=16477) if fit else None
            arm_out[label] = {"fit": fit, "geometry_null": null}
        out[arm_name] = arm_out
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}

    cands = []
    for _, c in AUDIT["pairs"].items():
        if c.get("exclusion_reason"):
            continue
        if c["separation_m"] <= 100.0 and c.get("valid_internal_vs_free_atomic_contrast"):
            cands.append((c["external_station"], c["internal_station"], c["separation_m"]))

    dates_2024 = td.available_dates(PRODUCT, 2024)
    if cands and dates_2024:
        print(f"STEP 164  |  Co-located differential spectrometer (coherent multi-window)")
        print(f"  Window={WINDOW_DAYS}d, step={STEP_DAYS}d, pairs: {[(a,i) for a,i,_ in cands]}")

        ser0 = td.load_series(PRODUCT, dates_2024[0], min_valid_frac=0.0, max_jumps=10000)
        all_stations = list(ser0.keys())

        all_results = {}
        for atomic, internal, sep_m in cands:
            pair_name = f"{internal}-{atomic}"
            print(f"\n  === {pair_name} (sep={sep_m:.1f} m) ===")
            result = analyze_fullyear(dates_2024, internal, atomic, all_stations, 2024)
            if result:
                result["separation_m"] = sep_m
                all_results[pair_name] = {"2024": result}

        # 2025 replication for MAT1-MATE
        dates_2025 = td.available_dates(PRODUCT, 2025)
        if dates_2025 and "MAT1-MATE" in all_results:
            print(f"\n  2025 replication: {len(dates_2025)} days")
            ser0_25 = td.load_series(PRODUCT, dates_2025[0], min_valid_frac=0.0, max_jumps=10000)
            result_25 = analyze_fullyear(dates_2025, "MAT1", "MATE",
                                          list(ser0_25.keys()), 2025)
            if result_25:
                result_25["separation_m"] = 10.7
                all_results["MAT1-MATE"]["2025"] = result_25

        summary = compute_summary(all_results)
    else:
        all_results = existing.get("pairs", {})
        summary = existing.get("summary", {})

    tidal_band_colocated = _compute_tidal_band_colocated()
    summary["tidal_band_10_35uHz_colocated"] = tidal_band_colocated

    output = {
        "step": 164,
        "title": "Co-located differential spectrometer: coherent multi-window accumulation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "design": (
            f"Coherent accumulation of cross-spectra across {WINDOW_DAYS}-day overlapping windows "
            f"(step={STEP_DAYS}d).  Each window is independently detrended.  The cross-spectral "
            f"product conj(F_delta)*F_remote is accumulated across all windows before computing "
            f"phase alignment, giving the SNR of the full year with per-window drift removal."
        ),
        "pairs": all_results,
        "summary": summary,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nSUMMARY:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
