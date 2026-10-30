#!/usr/bin/env python3
"""Step 74: Comprehensive clock-type analysis using CODE precise clock products.

Processes all GPS + Galileo satellites and all ground stations from CODE CLK
files across 7 days (2024 DOY 097-103) for temporal stability.

Uses CODE AR (receiver) and AS (satellite) clock estimates directly — no RINEX
processing needed. This gives us analysis-center quality clocks for ~400+
stations and ~100+ satellites per day.

Clock types verified from:
- Satellites: NGA GPS catalog (GPS), ESA Galileo documentation
- Ground: IGS Network JSON (files.igs.org/pub/station/general/IGSNetwork.json)

Outputs:
- Per-station LF/MF/HF amplitudes (median across days)
- Same-type satellite vs ground comparisons
- Altitude, temperature, bedrock correlations
- Temporal stability (day-to-day variation)
"""
import gzip
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CLK_DIR = PROJECT_ROOT / "data" / "rinex" / "clk"
NAV_DIR = PROJECT_ROOT / "data" / "rinex" / "nav"
OUT_FILE = PROJECT_ROOT / "results" / "outputs" / "step_74c_comprehensive_clocks.json"

DOYS = ["097", "098", "099", "100", "101", "102", "103"]

C_M_PER_NS = 0.299792458
LF_MAX_HZ = 10e-6
MF_MIN_HZ, MF_MAX_HZ = 10e-6, 500e-6
HF_MIN_HZ = 500e-6

# Verified GPS satellite clock types from NGA catalog (April 2024)
GPS_CLOCK_TYPES = {
    "G02": ("IIR", "Rb"), "G03": ("IIF", "Rb"), "G04": ("III", "Rb"),
    "G05": ("IIR-M", "Rb"), "G06": ("IIF", "Rb"), "G07": ("IIR-M", "Rb"),
    "G08": ("IIF", "Cs"), "G09": ("IIF", "Rb"), "G10": ("IIF", "Rb"),
    "G11": ("III", "Rb"), "G12": ("IIR-M", "Rb"), "G13": ("IIR", "Rb"),
    "G14": ("III", "Rb"), "G15": ("IIR-M", "Rb"), "G16": ("IIR", "Rb"),
    "G17": ("IIR-M", "Rb"), "G18": ("III", "Rb"), "G19": ("IIR", "Rb"),
    "G20": ("IIR", "Rb"), "G21": ("IIR", "Rb"), "G22": ("IIR", "Rb"),
    "G23": ("III", "Rb"), "G24": ("IIF", "Cs"), "G25": ("IIF", "Rb"),
    "G26": ("IIF", "Rb"), "G27": ("IIF", "Rb"), "G28": ("III", "Rb"),
    "G29": ("IIR-M", "Rb"), "G30": ("IIF", "Rb"), "G31": ("IIR-M", "Rb"),
    "G32": ("IIF", "Rb"),
}

# Galileo satellites use Passive Hydrogen Masers (PHM)
# All Galileo FOC satellites use PHM clocks
GALILEO_CLOCK_TYPE = ("FOC", "PHM")


def lowpass(x, dt, f_high):
    fs = 1.0 / dt; nyq = fs / 2
    high = f_high / nyq if f_high < nyq else 0.999
    b, a = butter(4, high, btype="low")
    return filtfilt(b, a, x)

def highpass(x, dt, f_low):
    fs = 1.0 / dt; nyq = fs / 2
    low = f_low / nyq if f_low > 0 else 1e-6
    b, a = butter(4, low, btype="high")
    return filtfilt(b, a, x)

def bandpass(x, dt, f_low, f_high):
    fs = 1.0 / dt; nyq = fs / 2
    low = f_low / nyq if f_low > 0 else 1e-6
    high = f_high / nyq if f_high < nyq else 0.999
    if low >= high: return np.zeros_like(x)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x)

def compute_bands(x_m, dt):
    n = len(x_m)
    if n < 100: return None, None, None
    t = np.arange(n) * dt
    coeffs = np.polyfit(t, x_m, 2)
    detrended = x_m - np.polyval(coeffs, t)
    lf = lowpass(detrended, dt, LF_MAX_HZ)
    mf = bandpass(detrended, dt, MF_MIN_HZ, MF_MAX_HZ)
    hf = highpass(detrended, dt, HF_MIN_HZ)
    return (float(np.sqrt(np.mean(lf**2))),
            float(np.sqrt(np.mean(mf**2))),
            float(np.sqrt(np.mean(hf**2))))


def parse_clk_file(clk_path):
    """Parse a CODE CLK file for AR (receiver) and AS (satellite) records.

    Returns dicts of {id: [(t_rel, t_abs, clk_ns), ...]} where t_rel is
    seconds from first epoch and t_abs is absolute GPS seconds (day-of-month
    based, matching broadcast nav convention).
    """
    ar_clocks = {}
    as_clocks = {}
    ref_epoch = None

    with gzip.open(clk_path, "rt") as f:
        for line in f:
            if line.startswith("AR "):
                parts = line.split()
                station = parts[1][:4]
                t_abs = (int(parts[4]) * 86400 + int(parts[5]) * 3600 +
                         int(parts[6]) * 60 + float(parts[7]))
                if ref_epoch is None: ref_epoch = t_abs
                t_rel = t_abs - ref_epoch
                clk_s = float(parts[9])
                ar_clocks.setdefault(station, []).append((t_rel, t_abs, clk_s * 1e9))

            elif line.startswith("AS "):
                parts = line.split()
                sat = parts[1]
                t_abs = (int(parts[4]) * 86400 + int(parts[5]) * 3600 +
                         int(parts[6]) * 60 + float(parts[7]))
                if ref_epoch is None: ref_epoch = t_abs
                t_rel = t_abs - ref_epoch
                clk_s = float(parts[9])
                as_clocks.setdefault(sat, []).append((t_rel, t_abs, clk_s * 1e9))

    return ar_clocks, as_clocks


def parse_broadcast_clocks(nav_path):
    """Parse broadcast navigation file for GPS satellite clock steering.

    Handles RINEX format where negative values are concatenated with
    the preceding field (e.g. '00-4.56E-04' or '3.3E-04-1.7E-11').
    """
    import re
    # Pattern for RINEX scientific notation: [sign]d.ddddE[+-]dd
    float_pat = r'[-+]?\d+\.\d+E[-+]\d{2}'
    # Full line pattern: G## YYYY MM DD HH MM SS  a0 a1 a2
    line_pat = re.compile(
        r'^G(\d\d)\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'
        r'(\d{1,2}(?:\.\d+)?)\s*(' + float_pat + r')\s*(' + float_pat + r')\s*(' + float_pat + r')')

    with open(nav_path) as f:
        lines = f.readlines()
    header_end = next(i + 1 for i, l in enumerate(lines) if "END OF HEADER" in l)
    gps_clocks = {}
    i = header_end
    while i < len(lines):
        line = lines[i]
        if len(line) >= 3 and line[0] == "G" and line[1:3].strip().isdigit():
            sat = line[:3].strip()
            m = line_pat.match(line)
            if m:
                day = int(m.group(4)); hr = int(m.group(5))
                mn = int(m.group(6)); sec = float(m.group(7))
                toc = day * 86400 + hr * 3600 + mn * 60 + sec
                a0 = float(m.group(8)); a1 = float(m.group(9)); a2 = float(m.group(10))
                gps_clocks.setdefault(sat, []).append((toc, a0, a1, a2))
            i += 8
        else:
            i += 1
    return gps_clocks


def unwrap_satellite_clock(clk_ns, t_abs, bc_records):
    """Remove a0 steering from satellite clock, preserving a1/a2 drift."""
    bc_noa0 = np.zeros_like(t_abs)
    for j, t in enumerate(t_abs):
        best = None; best_dt = float("inf")
        for toc, a0, a1, a2 in bc_records:
            d = t - toc
            if 0 <= d < best_dt: best_dt = d; best = (toc, a0, a1, a2)
        if best is None: best = bc_records[0]; best_dt = t - best[0]
        toc, a0, a1, a2 = best
        bc_noa0[j] = (a1 * best_dt + a2 * best_dt**2) * 1e9
    return clk_ns - bc_noa0


def main() -> int:
    print("=== step_74: comprehensive clock-type analysis (7 days, CODE CLK) ===")

    # Load metadata
    igs_atomic = json.load(open(PROJECT_ROOT / "data" / "meta" / "igs_atomic_stations.json"))
    altitudes = json.load(open(PROJECT_ROOT / "data" / "meta" / "station_altitudes_all.json"))
    ct_corrected = json.load(open(PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"))
    igs_full = json.load(open(PROJECT_ROOT / "data" / "meta" / "igs_clock_type_full.json"))

    # Build clock type lookup from complete IGS Network JSON
    clock_type_lookup = {}
    for name, (cls, sub) in igs_full.items():
        clock_type_lookup[name] = (cls, sub, {})

    # Override with our corrected metadata (more accurate, date-matched)
    for name, info in ct_corrected["stations"].items():
        std = (info.get("standard_type") or "").upper()
        clk = info.get("corrected_type", "?")
        if clk == "ATOMIC":
            if "RUBIDIUM" in std: clock_type_lookup[name] = ("ATOMIC", "Rb", {})
            elif "CESIUM" in std: clock_type_lookup[name] = ("ATOMIC", "Cs", {})
            elif "MASER" in std: clock_type_lookup[name] = ("ATOMIC", "H-Maser", {})
            elif "UTC" in std: clock_type_lookup[name] = ("ATOMIC", "UTC", {})
            else: clock_type_lookup[name] = ("ATOMIC", "Other", {})
        elif clk == "QUARTZ":
            clock_type_lookup[name] = ("QUARTZ", "Internal", {})

    # Add altitude and country from IGS atomic list
    for s in igs_atomic:
        short_name = s["name"][:4]
        if short_name in clock_type_lookup:
            cls, sub, _ = clock_type_lookup[short_name]
            clock_type_lookup[short_name] = (cls, sub, s)

    # Process each day
    all_ground = defaultdict(list)  # station -> [(lf, mf, hf, doy), ...]
    all_sat = defaultdict(list)     # sat -> [(lf, mf, hf, doy), ...]
    sat_unwrapped = defaultdict(list)  # sat -> [(lf, mf, hf, doy), ...] (unwrapped)

    for doy in DOYS:
        clk_path = CLK_DIR / f"COD0MGXFIN_2024{doy}0000_01D_30S_CLK.CLK.gz"
        if not clk_path.exists():
            print(f"  DOY {doy}: no CLK file, skipping")
            continue

        print(f"  DOY {doy}: parsing CLK file...", end=" ", flush=True)
        ar_clocks, as_clocks = parse_clk_file(clk_path)
        print(f"{len(ar_clocks)} stations, {len(as_clocks)} satellites")

        # Process ground stations
        for station, vals in ar_clocks.items():
            if len(vals) < 100: continue
            t = np.array([v[0] for v in vals])
            clk_ns = np.array([v[2] for v in vals])
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 30.0
            if not np.isfinite(dt) or dt <= 0: dt = 30.0

            # Check for steering jumps (wrap)
            d = np.diff(clk_ns)
            if np.max(np.abs(d)) > 500000:  # >0.5ms jump = steering
                # Unwrap
                phase = clk_ns * (2 * np.pi / 1000000.0)
                clk_ns = np.unwrap(phase) * 1000000.0 / (2 * np.pi)

            clk_m = clk_ns * C_M_PER_NS
            lf, mf, hf = compute_bands(clk_m, dt)
            if lf is not None and lf < 100.0:  # exclude extreme outliers
                all_ground[station].append((lf, mf, hf, doy))

        # Process satellites
        nav_path = NAV_DIR / f"BRDC00IGS_R_2024{doy}0000_01D_MN.rnx"
        bc = parse_broadcast_clocks(nav_path) if nav_path.exists() else {}

        for sat, vals in as_clocks.items():
            if len(vals) < 100: continue
            t_rel = np.array([v[0] for v in vals])
            t_abs = np.array([v[1] for v in vals])
            clk_ns = np.array([v[2] for v in vals])
            dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 30.0
            if not np.isfinite(dt) or dt <= 0: dt = 30.0

            # Raw (steered) clock
            clk_m = clk_ns * C_M_PER_NS
            lf, mf, hf = compute_bands(clk_m, dt)
            if lf is not None:
                all_sat[sat].append((lf, mf, hf, doy))

            # Unwrapped (remove a0 steering) for GPS satellites with broadcast data
            if sat in GPS_CLOCK_TYPES and sat in bc:
                resid_ns = unwrap_satellite_clock(clk_ns, t_abs, bc[sat])
                resid_m = resid_ns * C_M_PER_NS
                lf2, mf2, hf2 = compute_bands(resid_m, dt)
                if lf2 is not None:
                    sat_unwrapped[sat].append((lf2, mf2, hf2, doy))

    # Aggregate: median across days
    print(f"\n=== Aggregating across {len(DOYS)} days ===")

    ground_results = {}
    for station, day_vals in all_ground.items():
        lfs = [v[0] for v in day_vals]
        mfs = [v[1] for v in day_vals]
        hfs = [v[2] for v in day_vals]
        doys_seen = [v[3] for v in day_vals]

        clk_info = clock_type_lookup.get(station, ("UNKNOWN", "Unknown", {}))
        clk_class, clk_subtype, extra = clk_info
        alt = altitudes.get(station)

        ground_results[station] = {
            "clock_class": clk_class,
            "clock_subtype": clk_subtype,
            "n_days": len(day_vals),
            "days": doys_seen,
            "lf_median_m": float(np.median(lfs)),
            "lf_std_m": float(np.std(lfs)),
            "lf_values": [float(v) for v in lfs],
            "mf_median_m": float(np.median(mfs)),
            "hf_median_m": float(np.median(hfs)),
            "altitude_m": alt,
            "country": extra.get("country", ""),
            "height": extra.get("height", ""),
        }

    sat_results = {}
    sat_unw_results = {}
    for sat, day_vals in all_sat.items():
        lfs = [v[0] for v in day_vals]
        mfs = [v[1] for v in day_vals]
        hfs = [v[2] for v in day_vals]

        if sat in GPS_CLOCK_TYPES:
            block, clk_type = GPS_CLOCK_TYPES[sat]
            constellation = "GPS"
        elif sat.startswith("E"):
            block, clk_type = GALILEO_CLOCK_TYPE
            constellation = "Galileo"
        else:
            block, clk_type = "?", "?"
            constellation = sat[0]

        sat_results[sat] = {
            "constellation": constellation,
            "block": block,
            "clock_type": clk_type,
            "n_days": len(day_vals),
            "lf_median_m": float(np.median(lfs)),
            "mf_median_m": float(np.median(mfs)),
            "hf_median_m": float(np.median(hfs)),
            "lf_values": [float(v) for v in lfs],
        }

        if sat in sat_unwrapped:
            uw_vals = sat_unwrapped[sat]
            uw_lfs = [v[0] for v in uw_vals]
            uw_mfs = [v[1] for v in uw_vals]
            uw_hfs = [v[2] for v in uw_vals]
            sat_unw_results[sat] = {
                "constellation": constellation,
                "block": block,
                "clock_type": clk_type,
                "n_days": len(uw_vals),
                "lf_median_m": float(np.median(uw_lfs)),
                "mf_median_m": float(np.median(uw_mfs)),
                "hf_median_m": float(np.median(uw_hfs)),
                "lf_values": [float(v) for v in uw_lfs],
            }

    # Summary statistics
    print(f"\nGround stations: {len(ground_results)}")
    print(f"Satellites (raw): {len(sat_results)}")
    print(f"Satellites (unwrapped): {len(sat_unw_results)}")

    # Clock type distribution
    type_counts = defaultdict(int)
    for r in ground_results.values():
        type_counts[(r["clock_class"], r["clock_subtype"])] += 1
    print(f"\nGround clock types:")
    for (cls, sub), n in sorted(type_counts.items()):
        print(f"  {cls}/{sub}: {n}")

    # Same-type comparisons
    print(f"\n=== Same-type satellite vs ground comparison ===")

    # Ground by subtype
    ground_by_type = defaultdict(list)
    for r in ground_results.values():
        if r["lf_median_m"] < 10.0:  # exclude outliers
            ground_by_type[r["clock_subtype"]].append(r["lf_median_m"])

    # Satellite unwrapped by type
    sat_by_type = defaultdict(list)
    for r in sat_unw_results.values():
        sat_by_type[r["clock_type"]].append(r["lf_median_m"])

    # Satellite raw by type
    sat_raw_by_type = defaultdict(list)
    for r in sat_results.values():
        if r["clock_type"] != "?":
            sat_raw_by_type[r["clock_type"]].append(r["lf_median_m"])

    for clk_type in ["Rb", "Cs", "PHM", "H-Maser"]:
        sat_unw = sat_by_type.get(clk_type, [])
        sat_raw = sat_raw_by_type.get(clk_type, [])
        ground = ground_by_type.get(clk_type, [])

        if sat_unw and ground:
            ratio = np.median(sat_unw) / np.median(ground)
            print(f"  {clk_type}: sat_unwrapped={np.median(sat_unw):.4f}m (n={len(sat_unw)}), "
                  f"ground={np.median(ground):.4f}m (n={len(ground)}), "
                  f"ratio={ratio:.1f}x")
        elif sat_raw and ground:
            print(f"  {clk_type}: sat_raw={np.median(sat_raw):.6f}m (n={len(sat_raw)}), "
                  f"ground={np.median(ground):.4f}m (n={len(ground)}), "
                  f"(no unwrap available)")
        elif ground:
            print(f"  {clk_type}: ground={np.median(ground):.4f}m (n={len(ground)}), no satellite data")
        elif sat_unw:
            print(f"  {clk_type}: sat_unwrapped={np.median(sat_unw):.4f}m (n={len(sat_unw)}), no ground data")

    # Cross-type: all satellite vs all ground atomic
    all_sat_unw = [r["lf_median_m"] for r in sat_unw_results.values()]
    all_ground_atomic = [r["lf_median_m"] for r in ground_results.values()
                         if r["clock_class"] == "ATOMIC" and r["lf_median_m"] < 10.0]
    all_ground_quartz = [r["lf_median_m"] for r in ground_results.values()
                         if r["clock_class"] == "QUARTZ" and r["lf_median_m"] < 10.0]

    if all_sat_unw and all_ground_atomic:
        print(f"\n  All satellite (unwrapped) / All ground atomic: "
              f"{np.median(all_sat_unw)/np.median(all_ground_atomic):.1f}x "
              f"(sat={np.median(all_sat_unw):.4f}m, ground={np.median(all_ground_atomic):.4f}m)")
    if all_ground_quartz and all_ground_atomic:
        print(f"  Ground quartz / ground atomic: "
              f"{np.median(all_ground_quartz)/np.median(all_ground_atomic):.2f}x")

    # Galileo vs GPS
    gal_lfs = [r["lf_median_m"] for r in sat_results.values() if r["constellation"] == "Galileo"]
    gps_raw_lfs = [r["lf_median_m"] for r in sat_results.values() if r["constellation"] == "GPS"]
    if gal_lfs and gps_raw_lfs:
        print(f"\n  Galileo (PHM, raw) / GPS (Rb+Cs, raw): "
              f"{np.median(gal_lfs)/np.median(gps_raw_lfs):.2f}x "
              f"(Gal={np.median(gal_lfs):.6f}m, GPS={np.median(gps_raw_lfs):.6f}m)")

    # Temporal stability
    print(f"\n=== Temporal stability (day-to-day variation) ===")
    stability = []
    for station, r in ground_results.items():
        if r["n_days"] >= 3 and r["lf_median_m"] < 10.0:
            cv = r["lf_std_m"] / r["lf_median_m"] if r["lf_median_m"] > 0 else 0
            stability.append((station, r["clock_subtype"], r["lf_median_m"], cv))

    stability.sort(key=lambda x: x[3])
    print(f"  Median CV: {np.median([s[3] for s in stability]):.2f}")
    print(f"  Most stable (lowest CV):")
    for s in stability[:5]:
        print(f"    {s[0]} ({s[1]}): LF={s[2]:.4f}m, CV={s[3]:.2f}")
    print(f"  Least stable (highest CV):")
    for s in stability[-5:]:
        print(f"    {s[0]} ({s[1]}): LF={s[2]:.4f}m, CV={s[3]:.2f}")

    # Altitude correlation
    print(f"\n=== Altitude vs LF correlation ===")
    alt_vals = []
    lf_vals = []
    for r in ground_results.values():
        if r["altitude_m"] is not None and r["lf_median_m"] < 10.0:
            alt_vals.append(r["altitude_m"])
            lf_vals.append(r["lf_median_m"])

    if len(alt_vals) > 10:
        r_p, p_p = stats.pearsonr(alt_vals, lf_vals)
        r_s, p_s = stats.spearmanr(alt_vals, lf_vals)
        print(f"  All stations (n={len(alt_vals)}): Pearson r={r_p:.3f} (p={p_p:.3f}), "
              f"Spearman rho={r_s:.3f} (p={p_s:.3f})")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "step": 74,
        "title": "Comprehensive clock-type analysis (7 days, CODE CLK)",
        "days": DOYS,
        "n_ground_stations": len(ground_results),
        "n_satellites_raw": len(sat_results),
        "n_satellites_unwrapped": len(sat_unw_results),
        "ground_stations": ground_results,
        "satellites_raw": sat_results,
        "satellites_unwrapped": sat_unw_results,
        "summary": {
            "same_type_ratios": {
                clk_type: {
                    "sat_unwrapped_median": float(np.median(sat_by_type.get(clk_type, [0]))) if sat_by_type.get(clk_type) else None,
                    "ground_median": float(np.median(ground_by_type.get(clk_type, [0]))) if ground_by_type.get(clk_type) else None,
                    "ratio": float(np.median(sat_by_type.get(clk_type, [0])) / np.median(ground_by_type.get(clk_type, [1]))) if sat_by_type.get(clk_type) and ground_by_type.get(clk_type) else None,
                    "n_sat": len(sat_by_type.get(clk_type, [])),
                    "n_ground": len(ground_by_type.get(clk_type, [])),
                }
                for clk_type in ["Rb", "Cs", "PHM", "H-Maser", "UTC", "Other"]
            },
            "all_sat_unwrapped_median": float(np.median(all_sat_unw)) if all_sat_unw else None,
            "all_ground_atomic_median": float(np.median(all_ground_atomic)) if all_ground_atomic else None,
            "all_ground_quartz_median": float(np.median(all_ground_quartz)) if all_ground_quartz else None,
            "temporal_stability_median_cv": float(np.median([s[3] for s in stability])) if stability else None,
        },
    }, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
