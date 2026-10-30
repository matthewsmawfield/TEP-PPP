#!/usr/bin/env python3
"""
STEP 166  --  Gravitational phase test on the co-located differential
              cross-spectrum.

THE COMBINED TEST: screening + gravitational identification in one.

Step 164 showed that the co-located differential (internal - external H-maser)
reveals M2 distance structure invisible in the atomic clock alone.  This is
the screening signature.

Step 165b showed that M2 clock phases across all stations match the
gravitational prediction cos(2*(lon1-lon2)) with Fisher p=3.53e-13.  This
identifies the signal as gravitational.

This step COMBINES them: for each remote station j, the cross-spectrum
between the co-located differential Delta(t) and station j's clock has a
phase at M2.  If this phase matches the gravitational prediction
2*(lon_site - lon_j), then the differential signal — which has already
cancelled all common-mode terms — is gravitational.

This is the strongest possible test with existing CORS data because it
requires ALL THREE properties simultaneously:
  1. The differential must have M2 power (screening: field couples
     differently to different oscillators)
  2. The M2 phase must match the gravitational prediction (identification:
     no atmospheric or processing effect can produce this)
  3. The phase must be coherent across remote stations at different
     longitudes (replication: not a local artefact)

PREDICTION:
  TEP: cos(phase_diff) correlates with cos(2*(lon_site - lon_j)) at M2.
       S2 shows weaker correlation (atmospheric phase scatter).
       Control frequencies show no correlation.

NULL:
  No correlation at any frequency.

OUTPUT:
  results/outputs/step_166_differential_gravitational_phase.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_166_differential_gravitational_phase.json"
AUDIT_PATH = ROOT / "data" / "meta" / "clock_anchor_audit_2024.json"
AUDIT = json.loads(AUDIT_PATH.read_text())

PRODUCT = "clk_cod_ops"
DT = 300.0
WINDOW_DAYS = 13
STEP_DAYS = 7
N_NULL = 1000

# Tidal frequencies (Hz)
TIDAL_FREQS_HZ = {
    "M2":  1.0 / (12.42 * 3600.0),
    "O1":  1.0 / (25.82 * 3600.0),
    "S2":  2.0 / 86400.0,
    "S1":  1.0 / 86400.0,
    "K1":  1.0 / (23.93 * 3600.0),
    "N2":  1.0 / (12.66 * 3600.0),
}

# Control frequencies (Hz)
CONTROL_FREQS_HZ = [13e-6, 15e-6, 17e-6, 19e-6, 25e-6, 27e-6, 29e-6, 31e-6,
                     33e-6, 35e-6, 37e-6, 39e-6, 41e-6, 43e-6, 45e-6]


def load_window_series(dates, stations_needed):
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


def analyze(dates, internal, atomic, all_stations, year_label):
    needed = [internal, atomic] + [s for s in all_stations if s not in (internal, atomic)]
    if len(needed) > 160:
        needed = needed[:160]

    windows = []
    for start in range(0, len(dates) - WINDOW_DAYS + 1, STEP_DAYS):
        windows.append(dates[start:start + WINDOW_DAYS])
    print(f"  {len(windows)} overlapping {WINDOW_DAYS}-day windows", flush=True)

    site_coord = None
    site_lon = None
    remote_coords = {}
    n_freq = None
    freqs_ref = None

    # Accumulate cross-spectra per remote station
    cross_accum = {}
    cross_count = {}

    # Also accumulate individual clock cross-spectra for comparison
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
            site_lon = si["station"].lon_deg

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

        F_delta = np.fft.rfft(delta)
        freqs = np.fft.rfftfreq(n, d=DT)
        F_internal = np.fft.rfft(qi - np.mean(qi))
        F_atomic = np.fft.rfft(qa - np.mean(qa))

        if n_freq is None:
            n_freq = len(freqs)
            freqs_ref = freqs
        elif len(freqs) != n_freq:
            continue

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

    if not cross_accum or site_coord is None:
        return None

    print(f"  {sum(1 for c in cross_count.values() if c >= 5)} remote stations with >=5 windows", flush=True)

    # For each frequency, compute the gravitational phase prediction test
    # For each remote station j:
    #   expected_phase_diff = 2*(lon_site - lon_j) for semidiurnal
    #   expected_phase_diff = 1*(lon_site - lon_j) for diurnal
    #   observed_phase_diff = angle(cross_spectrum at that freq)
    #   cos(expected) should correlate with cos(observed)

    results = {
        "year": year_label,
        "n_windows": max(cross_count.values()) if cross_count else 0,
        "n_remote_stations": sum(1 for c in cross_count.values() if c >= 5),
        "site_lon_deg": site_lon,
        "site_name": internal,
    }

    # Tidal lines
    print(f"  Gravitational phase test on differential cross-spectrum...", flush=True)
    line_results = {}
    for line_name, freq_hz in TIDAL_FREQS_HZ.items():
        idx = np.argmin(np.abs(freqs_ref - freq_hz))
        actual_freq = freqs_ref[idx] * 1e6  # uHz

        # Semidiurnal: phase diff = 2*dlon. Diurnal: phase diff = 1*dlon
        is_semidiurnal = "2" in line_name

        expected_cos = []
        observed_cos = []
        observed_cos_internal = []
        observed_cos_atomic = []
        distances = []
        station_lons = []

        for st, cross in cross_accum.items():
            if cross_count[st] < 5:
                continue
            station_coord = remote_coords[st]
            remote_lon = station_coord.lon_deg
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue

            dlon = site_lon - remote_lon
            if is_semidiurnal:
                expected_dphase = 2.0 * dlon
            else:
                expected_dphase = 1.0 * dlon
            expected_c = np.cos(np.radians(expected_dphase))

            # Observed phase from accumulated cross-spectrum
            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                obs_phase = np.angle(avg_cross[idx])
                obs_c = np.cos(obs_phase)
                expected_cos.append(expected_c)
                observed_cos.append(float(obs_c))
                distances.append(d)
                station_lons.append(remote_lon)

                # Individual clocks
                avg_int = internal_cross_accum[st] / cross_count[st]
                if avg_int[idx] != 0 and np.isfinite(avg_int[idx]):
                    observed_cos_internal.append(float(np.cos(np.angle(avg_int[idx]))))
                else:
                    observed_cos_internal.append(0.0)

                avg_atm = atomic_cross_accum[st] / cross_count[st]
                if avg_atm[idx] != 0 and np.isfinite(avg_atm[idx]):
                    observed_cos_atomic.append(float(np.cos(np.angle(avg_atm[idx]))))
                else:
                    observed_cos_atomic.append(0.0)

        n_pairs = len(expected_cos)
        if n_pairs < 20:
            line_results[line_name] = {
                "n_pairs": n_pairs, "r": None, "p": None,
                "r_internal": None, "r_atomic": None,
            }
            continue

        expected_cos = np.array(expected_cos)
        observed_cos = np.array(observed_cos)
        observed_cos_internal = np.array(observed_cos_internal)
        observed_cos_atomic = np.array(observed_cos_atomic)

        r, p = stats.pearsonr(expected_cos, observed_cos)
        r_int, p_int = stats.pearsonr(expected_cos, observed_cos_internal)
        r_atm, p_atm = stats.pearsonr(expected_cos, observed_cos_atomic)

        # Permutation null: shuffle the expected phases
        rng = np.random.default_rng(16600 + int(freq_hz * 1e6))
        null_rs = []
        for _ in range(N_NULL):
            perm = rng.permutation(expected_cos)
            r_null, _ = stats.pearsonr(perm, observed_cos)
            null_rs.append(r_null)
        null_rs = np.array(null_rs)
        p_perm = float((1 + np.sum(null_rs >= r)) / (1 + len(null_rs)))

        line_results[line_name] = {
            "n_pairs": n_pairs,
            "r": float(r),
            "p": float(p),
            "p_perm": p_perm,
            "r_internal": float(r_int) if r_int else None,
            "p_internal": float(p_int) if p_int else None,
            "r_atomic": float(r_atm) if r_atm else None,
            "p_atomic": float(p_atm) if p_atm else None,
            "null_r_median": float(np.median(null_rs)),
            "null_r_max": float(np.max(null_rs)),
            "mean_observed_cos": float(np.mean(observed_cos)),
            "mean_expected_cos": float(np.mean(expected_cos)),
        }

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {line_name} ({actual_freq:.3f} uHz): "
              f"r={r:.4f}, p={p:.6f} {sig}, p_perm={p_perm:.4f}, n={n_pairs} | "
              f"int r={r_int:.4f}, atm r={r_atm:.4f}")

    results["tidal_lines"] = line_results

    # Control frequencies
    print(f"  Control frequencies...", flush=True)
    control_results = {}
    for freq_hz in CONTROL_FREQS_HZ:
        idx = np.argmin(np.abs(freqs_ref - freq_hz))
        expected_cos = []
        observed_cos = []
        for st, cross in cross_accum.items():
            if cross_count[st] < 5:
                continue
            station_coord = remote_coords[st]
            remote_lon = station_coord.lon_deg
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue
            dlon = site_lon - remote_lon
            expected_c = np.cos(np.radians(2.0 * dlon))
            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                obs_c = float(np.cos(np.angle(avg_cross[idx])))
                expected_cos.append(expected_c)
                observed_cos.append(obs_c)
        if len(expected_cos) >= 20:
            r, p = stats.pearsonr(expected_cos, observed_cos)
            control_results[f"{freq_hz*1e6:.0f}uHz"] = {
                "n_pairs": len(expected_cos),
                "r": float(r), "p": float(p),
            }

    results["control_freqs"] = control_results
    ctrl_rs = [v["r"] for v in control_results.values() if v.get("r") is not None]
    results["control_r_mean"] = float(np.mean(ctrl_rs)) if ctrl_rs else None
    results["control_r_median"] = float(np.median(ctrl_rs)) if ctrl_rs else None

    return results


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    cands = []
    for _, c in AUDIT["pairs"].items():
        if c.get("exclusion_reason"):
            continue
        if c["separation_m"] <= 100.0 and c.get("valid_internal_vs_free_atomic_contrast"):
            cands.append((c["external_station"], c["internal_station"], c["separation_m"]))

    print(f"STEP 166  |  Gravitational phase test on co-located differential")
    print(f"  Combines screening (differential) + gravitational ID (phase)")
    print(f"  Pairs: {[(a, i) for a, i, _ in cands]}")

    dates_2024 = td.available_dates(PRODUCT, 2024)
    ser0 = td.load_series(PRODUCT, dates_2024[0], min_valid_frac=0.0, max_jumps=10000)
    all_stations = list(ser0.keys())

    all_results = {}
    for atomic, internal, sep_m in cands:
        pair_name = f"{internal}-{atomic}"
        print(f"\n  === {pair_name} (sep={sep_m:.1f} m) ===")
        result = analyze(dates_2024, internal, atomic, all_stations, 2024)
        if result:
            result["separation_m"] = sep_m
            all_results[pair_name] = {"2024": result}

    # 2025 replication for MAT1-MATE
    dates_2025 = td.available_dates(PRODUCT, 2025)
    if dates_2025 and "MAT1-MATE" in all_results:
        print(f"\n  2025 replication: MAT1-MATE")
        ser0_25 = td.load_series(PRODUCT, dates_2025[0], min_valid_frac=0.0, max_jumps=10000)
        result_25 = analyze(dates_2025, "MAT1", "MATE",
                            list(ser0_25.keys()), 2025)
        if result_25:
            result_25["separation_m"] = 10.7
            all_results["MAT1-MATE"]["2025"] = result_25

    # Summary
    summary = {}
    for pair_name, pair_data in all_results.items():
        pair_summary = {}
        for year, result in pair_data.items():
            tl = result.get("tidal_lines", {})
            ctrl_mean = result.get("control_r_mean", 0)
            m2 = tl.get("M2", {})
            s2 = tl.get("S2", {})
            o1 = tl.get("O1", {})
            pair_summary[year] = {
                "M2_r": m2.get("r"),
                "M2_p": m2.get("p"),
                "M2_p_perm": m2.get("p_perm"),
                "M2_r_internal": m2.get("r_internal"),
                "M2_r_atomic": m2.get("r_atomic"),
                "S2_r": s2.get("r"),
                "S2_p": s2.get("p"),
                "O1_r": o1.get("r"),
                "O1_p": o1.get("p"),
                "control_r_mean": ctrl_mean,
                "n_remote": result.get("n_remote_stations"),
                "n_windows": result.get("n_windows"),
            }
        summary[pair_name] = pair_summary

    output = {
        "step": 166,
        "title": "Gravitational phase test on the co-located differential cross-spectrum",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "design": (
            "Combines the screening signature (co-located differential cancels "
            "common-mode, reveals M2) with the gravitational identification "
            "(M2 phase matches cos(2*(lon1-lon2))). For each remote station j, "
            "the cross-spectrum between the differential and station j's clock "
            "has a phase at M2. If this phase matches the gravitational prediction, "
            "the differential signal is gravitational. This requires ALL THREE "
            "properties simultaneously: (1) differential has M2 power (screening), "
            "(2) M2 phase matches gravitational prediction (identification), "
            "(3) phase is coherent across remote stations at different longitudes "
            "(replication)."
        ),
        "prediction": (
            "TEP: M2 r > 0 and significantly exceeds control r. "
            "S2 r > 0 but weaker than M2 (atmospheric phase scatter). "
            "Control r ~ 0. "
            "M2 r_differential > M2 r_atomic (screening: differential reveals "
            "what atomic clock alone cannot see)."
        ),
        "pairs": all_results,
        "summary": summary,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nSUMMARY:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
