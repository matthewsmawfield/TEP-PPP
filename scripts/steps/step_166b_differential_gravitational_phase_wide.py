#!/usr/bin/env python3
"""
STEP 166b  --  Gravitational phase test on ALL co-located differentials
               with 30-day windows (resolves M2 from N2).

Step 166 used 13-day windows and only 3 audited pairs.  Problems:
  - M2 (22.365) and N2 (21.422) are 0.943 uHz apart, but 13-day resolution
    is 0.891 uHz — they hit the same FFT bin.
  - Only 3 pairs were tested; the IGS network has 30+ co-located sites.

This step fixes both:
  - 30-day windows give 0.386 uHz resolution — M2 and N2 are 2.4 bins apart.
  - ALL co-located pairs within 200m are tested, providing many independent
    differential tests.  Same-oscillator pairs serve as nulls; cross-oscillator
    pairs should show the gravitational phase signature.

The WTZ cluster alone has 7 receivers, giving 21 independent differential
tests at one site.

OUTPUT: results/outputs/step_166b_differential_gravitational_phase_wide.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_166b_differential_gravitational_phase_wide.json"

PRODUCT = "clk_cod_ops"
DT = 300.0
WINDOW_DAYS = 30
STEP_DAYS = 15
N_NULL = 500
MAX_SEP_M = 200.0

TIDAL_FREQS_HZ = {
    "M2":  1.0 / (12.42 * 3600.0),
    "O1":  1.0 / (25.82 * 3600.0),
    "S2":  2.0 / 86400.0,
    "N2":  1.0 / (12.66 * 3600.0),
    "K1":  1.0 / (23.93 * 3600.0),
    "S1":  1.0 / 86400.0,
}
CONTROL_FREQS_HZ = [13e-6, 15e-6, 17e-6, 19e-6, 25e-6, 27e-6, 29e-6, 31e-6,
                     35e-6, 39e-6, 43e-6]


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
        if len(data["t"]) < 2000:
            continue
        t = np.asarray(data["t"], float)
        x = np.asarray(data["resid"], float)
        if len(t) > 2:
            coeffs = np.polyfit(t, x, 1)
            x = x - np.polyval(coeffs, t)
        result[st] = {"t": t, "resid": x, "station": station_coords.get(st)}
    return result


def find_colocated_pairs(ser):
    """Find all co-located pairs within MAX_SEP_M."""
    stations = {}
    for st, data in ser.items():
        if 'station' in data:
            stations[st] = data['station']

    pairs = []
    for s1, s2 in combinations(stations.keys(), 2):
        c1 = stations[s1]
        c2 = stations[s2]
        if abs(c1.lat_deg - c2.lat_deg) > 0.05 or abs(c1.lon_deg - c2.lon_deg) > 0.05:
            continue
        d = td.great_circle_km(c1, c2) * 1000.0
        if d <= MAX_SEP_M:
            pairs.append((s1, s2, d))
    pairs.sort(key=lambda x: x[2])
    return pairs


def analyze_pair(dates, st1, st2, all_stations, year_label, sep_m):
    """Run gravitational phase test on one co-located differential."""
    needed = [st1, st2] + [s for s in all_stations if s not in (st1, st2)]
    if len(needed) > 160:
        needed = needed[:160]

    windows = []
    for start in range(0, len(dates) - WINDOW_DAYS + 1, STEP_DAYS):
        windows.append(dates[start:start + WINDOW_DAYS])
    if not windows:
        return None

    site_coord = None
    site_lon = None
    remote_coords = {}
    n_freq = None
    freqs_ref = None
    cross_accum = {}
    cross_count = {}
    internal_cross_accum = {}
    atomic_cross_accum = {}

    for w_idx, window_dates in enumerate(windows):
        series = load_window_series(window_dates, needed)
        if st1 not in series or st2 not in series:
            continue

        s1_data = series[st1]
        s2_data = series[st2]
        if site_coord is None:
            site_coord = s1_data["station"]
            site_lon = s1_data["station"].lon_deg

        t0 = max(s1_data["t"].min(), s2_data["t"].min())
        t1 = min(s1_data["t"].max(), s2_data["t"].max())
        if t1 - t0 < 2000000:
            continue
        n = int(np.round((t1 - t0) / DT)) + 1
        grid = np.arange(n) * DT + t0
        x1 = np.interp(grid, s1_data["t"], s1_data["resid"])
        x2 = np.interp(grid, s2_data["t"], s2_data["resid"])
        delta = x1 - x2
        delta = delta - np.mean(delta)

        F_delta = np.fft.rfft(delta)
        freqs = np.fft.rfftfreq(n, d=DT)
        F_s1 = np.fft.rfft(x1 - np.mean(x1))
        F_s2 = np.fft.rfft(x2 - np.mean(x2))

        if n_freq is None:
            n_freq = len(freqs)
            freqs_ref = freqs
        elif len(freqs) != n_freq:
            continue

        for st, sdata in series.items():
            if st in (st1, st2) or "station" not in sdata:
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
            internal_cross_accum[st] += np.conj(F_s1) * F_r
            atomic_cross_accum[st] += np.conj(F_s2) * F_r
            cross_count[st] += 1

    if not cross_accum or site_coord is None:
        return None

    n_remote = sum(1 for c in cross_count.values() if c >= 3)
    if n_remote < 20:
        return None

    # Gravitational phase test at each tidal line
    line_results = {}
    for line_name, freq_hz in TIDAL_FREQS_HZ.items():
        idx = np.argmin(np.abs(freqs_ref - freq_hz))
        actual_freq = freqs_ref[idx] * 1e6
        is_semidiurnal = "2" in line_name

        expected_cos = []
        observed_cos = []
        obs_cos_s1 = []
        obs_cos_s2 = []

        for st, cross in cross_accum.items():
            if cross_count[st] < 3:
                continue
            station_coord = remote_coords[st]
            remote_lon = station_coord.lon_deg
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue
            dlon = site_lon - remote_lon
            expected_dphase = 2.0 * dlon if is_semidiurnal else 1.0 * dlon
            expected_c = np.cos(np.radians(expected_dphase))

            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                obs_c = float(np.cos(np.angle(avg_cross[idx])))
                expected_cos.append(expected_c)
                observed_cos.append(obs_c)
                avg_s1 = internal_cross_accum[st] / cross_count[st]
                obs_cos_s1.append(float(np.cos(np.angle(avg_s1[idx]))) if avg_s1[idx] != 0 else 0.0)
                avg_s2 = atomic_cross_accum[st] / cross_count[st]
                obs_cos_s2.append(float(np.cos(np.angle(avg_s2[idx]))) if avg_s2[idx] != 0 else 0.0)

        n_pairs = len(expected_cos)
        if n_pairs < 20:
            line_results[line_name] = {"n_pairs": n_pairs, "r": None, "p": None}
            continue

        expected_cos = np.array(expected_cos)
        observed_cos = np.array(observed_cos)
        r, p = stats.pearsonr(expected_cos, observed_cos)

        # Permutation null
        rng = np.random.default_rng(16600 + int(freq_hz * 1e6))
        null_rs = []
        for _ in range(N_NULL):
            perm = rng.permutation(expected_cos)
            r_null, _ = stats.pearsonr(perm, observed_cos)
            null_rs.append(r_null)
        p_perm = float((1 + np.sum(np.array(null_rs) >= r)) / (1 + len(null_rs)))

        r_s1, _ = stats.pearsonr(expected_cos, np.array(obs_cos_s1))
        r_s2, _ = stats.pearsonr(expected_cos, np.array(obs_cos_s2))

        line_results[line_name] = {
            "n_pairs": n_pairs,
            "r": float(r), "p": float(p), "p_perm": p_perm,
            "r_s1": float(r_s1), "r_s2": float(r_s2),
            "actual_freq_uhz": actual_freq,
        }

    # Control frequencies
    ctrl_rs = []
    for freq_hz in CONTROL_FREQS_HZ:
        idx = np.argmin(np.abs(freqs_ref - freq_hz))
        exp_c = []
        obs_c = []
        for st, cross in cross_accum.items():
            if cross_count[st] < 3:
                continue
            station_coord = remote_coords[st]
            d = td.great_circle_km(site_coord, station_coord)
            if d < 100 or d > 20000:
                continue
            dlon = site_lon - station_coord.lon_deg
            exp_c.append(np.cos(np.radians(2.0 * dlon)))
            avg_cross = cross / cross_count[st]
            if avg_cross[idx] != 0 and np.isfinite(avg_cross[idx]):
                obs_c.append(float(np.cos(np.angle(avg_cross[idx]))))
        if len(exp_c) >= 20 and len(obs_c) >= 20:
            r, _ = stats.pearsonr(np.array(exp_c[:len(obs_c)]), np.array(obs_c))
            ctrl_rs.append(r)

    ctrl_r_mean = float(np.mean(ctrl_rs)) if ctrl_rs else None

    return {
        "year": year_label,
        "sep_m": sep_m,
        "n_windows": max(cross_count.values()) if cross_count else 0,
        "n_remote": n_remote,
        "tidal_lines": line_results,
        "control_r_mean": ctrl_r_mean,
        "freq_resolution_uhz": float(1e6 / (WINDOW_DAYS * 86400)),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"STEP 166b  |  Gravitational phase test on ALL co-located differentials")
    print(f"  Window={WINDOW_DAYS}d (resolution={1e6/(WINDOW_DAYS*86400):.3f} uHz), step={STEP_DAYS}d")
    print(f"  Max separation: {MAX_SEP_M} m")

    dates_2024 = td.available_dates(PRODUCT, 2024)
    ser0 = td.load_series(PRODUCT, dates_2024[0], min_valid_frac=0.0, max_jumps=10000)
    all_stations = list(ser0.keys())

    # Find all co-located pairs
    pairs = find_colocated_pairs(ser0)
    print(f"  Found {len(pairs)} co-located pairs (< {MAX_SEP_M} m)")
    for s1, s2, sep in pairs:
        print(f"    {s1:>4} - {s2:>4}: {sep:.1f} m")

    all_results = {}
    for st1, st2, sep_m in pairs:
        pair_name = f"{st1}-{st2}"
        print(f"\n  === {pair_name} (sep={sep_m:.1f} m) ===", flush=True)
        result = analyze_pair(dates_2024, st1, st2, all_stations, 2024, sep_m)
        if result:
            all_results[pair_name] = {"2024": result}
            tl = result.get("tidal_lines", {})
            m2 = tl.get("M2", {})
            s2 = tl.get("S2", {})
            n2 = tl.get("N2", {})
            ctrl = result.get("control_r_mean", 0)
            m2_r = m2.get("r", 0)
            m2_p = m2.get("p", 1)
            m2_pp = m2.get("p_perm", 1)
            n2_r = n2.get("r", 0)
            s2_r = s2.get("r", 0)
            sig = "***" if m2_p < 0.001 else "**" if m2_p < 0.01 else "*" if m2_p < 0.05 else ""
            print(f"    M2: r={m2_r:.4f}, p={m2_p:.6f} {sig}, p_perm={m2_pp:.4f} | "
                  f"N2: r={n2_r:.4f} | S2: r={s2_r:.4f} | ctrl={ctrl:.4f}")
        else:
            print(f"    Insufficient data")

    # 2025 replication for the strongest pairs
    dates_2025 = td.available_dates(PRODUCT, 2025)
    if dates_2025:
        ser0_25 = td.load_series(PRODUCT, dates_2025[0], min_valid_frac=0.0, max_jumps=10000)
        pairs_25 = find_colocated_pairs(ser0_25)
        print(f"\n  2025: {len(pairs_25)} pairs available")
        for st1, st2, sep_m in pairs_25:
            pair_name = f"{st1}-{st2}"
            if pair_name in all_results or f"{st2}-{st1}" in all_results:
                pn = pair_name if pair_name in all_results else f"{st2}-{st1}"
                print(f"\n  === {pn} 2025 replication ===", flush=True)
                result_25 = analyze_pair(dates_2025, st1, st2,
                                         list(ser0_25.keys()), 2025, sep_m)
                if result_25:
                    all_results[pn]["2025"] = result_25
                    tl = result_25.get("tidal_lines", {})
                    m2 = tl.get("M2", {})
                    m2_r = m2.get("r", 0)
                    m2_p = m2.get("p", 1)
                    m2_pp = m2.get("p_perm", 1)
                    sig = "***" if m2_p < 0.001 else "**" if m2_p < 0.01 else "*" if m2_p < 0.05 else ""
                    print(f"    M2: r={m2_r:.4f}, p={m2_p:.6f} {sig}, p_perm={m2_pp:.4f}")

    # Summary
    summary = {}
    for pair_name, pair_data in all_results.items():
        ps = {}
        for year, result in pair_data.items():
            tl = result.get("tidal_lines", {})
            m2 = tl.get("M2", {})
            n2 = tl.get("N2", {})
            s2 = tl.get("S2", {})
            o1 = tl.get("O1", {})
            ps[year] = {
                "M2_r": m2.get("r"),
                "M2_p": m2.get("p"),
                "M2_p_perm": m2.get("p_perm"),
                "N2_r": n2.get("r"),
                "S2_r": s2.get("r"),
                "O1_r": o1.get("r"),
                "control_r_mean": result.get("control_r_mean"),
                "n_remote": result.get("n_remote"),
                "sep_m": result.get("sep_m"),
            }
        summary[pair_name] = ps

    output = {
        "step": "166b",
        "title": "Gravitational phase test on ALL co-located differentials (30-day windows)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "design": (
            f"30-day windows (0.386 uHz resolution, resolves M2 from N2). "
            f"All co-located pairs within {MAX_SEP_M}m. Same-oscillator pairs "
            f"serve as nulls; cross-oscillator pairs should show gravitational "
            f"phase matching at M2."
        ),
        "pairs": all_results,
        "summary": summary,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nSUMMARY:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
