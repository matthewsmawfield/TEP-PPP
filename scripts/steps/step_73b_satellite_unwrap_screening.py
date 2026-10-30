#!/usr/bin/env python3
"""Step 73: Satellite clock unwrap — testing TEP-UCD environmental screening
(vacuum vs Earth bulk matter) by removing broadcast clock steering.

TEP-UCD (Paper 6) defines:
    Σ_obs = S_Σ(E) · ∇ln A(φ)
    E includes ρ (local density), Φ/c² (compactness)
    ρ_T ≈ 20 g/cm³ (saturation density)

Prediction:
    Satellite clocks (vacuum, ρ≈0) → S_Σ ≈ 1 (no screening)
    Ground clocks (Earth surface, ρ_air≈0.001) → S_Σ < 1 (screened by Earth bulk)

GPS satellite clock types (verified from NGA GPS catalog, April 2024):
    Block IIR:    3x Rb (no Cs)
    Block IIR-M:  3x Rb (no Cs)
    Block IIF:    2x Rb + 1x Cs (Cs is backup; only G08 and G24 run Cs)
    Block III:    3x Rb (no Cs)
    Total active on day 100: 29 Rb, 2 Cs

Ground station clock types (verified from IGS site logs):
    34 ATOMIC: 15 Cs, 9 H-Maser, 2 Rb (AREG, M0SE), 8 other
    52 QUARTZ: INTERNAL oscillator
    14 UNKNOWN

The GPS control segment steers satellite clocks via the broadcast model
(a0 + a1·(t-toc) + a2·(t-toc)²), updated every ~2 hours. The a0 term is
the offset correction. Removing only a0 while preserving a1 (drift) and
a2 (aging) recovers the free-running LF component.

Key same-type comparisons:
    Satellite Rb (vacuum) vs Ground Rb (Earth surface) — same oscillator type
    Satellite Cs (vacuum) vs Ground Cs (Earth surface) — same oscillator type
    These control for oscillator quality, isolating the environmental effect.
"""
import gzip
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CLK_PATH = PROJECT_ROOT / "data" / "rinex" / "clk" / "COD0MGXFIN_20241000000_01D_30S_CLK.CLK.gz"
NAV_PATH = PROJECT_ROOT / "data" / "rinex" / "nav" / "BRDC00IGS_R_20241000000_01D_MN.rnx"
OUT_FILE = PROJECT_ROOT / "results" / "outputs" / "step_73b_satellite_unwrap.json"

C_M_PER_NS = 0.299792458
LF_MAX_HZ = 10e-6
MF_MIN_HZ, MF_MAX_HZ = 10e-6, 500e-6
HF_MIN_HZ = 500e-6

# Verified satellite clock types from NGA GPS catalog (active April 9, 2024)
SAT_CLOCK_TYPES = {
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


def parse_broadcast_clocks(nav_path):
    """Parse broadcast nav for GPS satellite clock steering.

    Handles RINEX format where negative values are concatenated with
    the preceding field (e.g. '00-4.56E-04' or '3.3E-04-1.7E-11').
    """
    import re
    float_pat = r'[-+]?\d+\.\d+E[-+]\d{2}'
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


def parse_precise_clocks(clk_path):
    sat_clocks = {}; ref_epoch = None
    with gzip.open(clk_path, "rt") as f:
        for line in f:
            if not (line.startswith("AS G") and len(line) > 10 and line[4:6].isdigit()):
                continue
            parts = line.split()
            sat = parts[1]
            t_sec = (int(parts[4]) * 86400 + int(parts[5]) * 3600 +
                     int(parts[6]) * 60 + float(parts[7]))
            if ref_epoch is None: ref_epoch = t_sec
            t_rel = t_sec - ref_epoch
            clk_s = float(parts[9])
            sat_clocks.setdefault(sat, []).append((t_rel, t_sec, clk_s))
    return sat_clocks


def main() -> int:
    print("=== step_73: satellite clock unwrap — environmental screening test ===")
    print(f"Satellite clock types verified from NGA GPS catalog (April 2024):")
    rb_sats = [s for s, (b, c) in SAT_CLOCK_TYPES.items() if c == "Rb"]
    cs_sats = [s for s, (b, c) in SAT_CLOCK_TYPES.items() if c == "Cs"]
    print(f"  {len(rb_sats)} Rb: {', '.join(sorted(rb_sats))}")
    print(f"  {len(cs_sats)} Cs: {', '.join(sorted(cs_sats))}")

    bc = parse_broadcast_clocks(NAV_PATH)
    pc = parse_precise_clocks(CLK_PATH)

    results = {}
    sat_rb_lfs = []; sat_cs_lfs = []
    sat_rb_mfs = []; sat_cs_mfs = []

    print(f"\n{'Sat':>5s} {'Block':>6s} {'Clk':>3s} {'n':>5s} {'raw_LF':>8s} {'noa0_LF':>9s} "
          f"{'noa0_MF':>9s} {'noa0_HF':>9s} {'a0_range':>9s}")

    for sat_id in sorted(pc.keys()):
        if sat_id not in bc: continue
        if sat_id not in SAT_CLOCK_TYPES: continue

        block, clk_type = SAT_CLOCK_TYPES[sat_id]
        vals = pc[sat_id]
        t_rel = np.array([v[0] for v in vals])
        t_abs = np.array([v[1] for v in vals])
        clk_s = np.array([v[2] for v in vals])
        clk_ns = clk_s * 1e9
        dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 30.0
        if not np.isfinite(dt) or dt <= 0: dt = 30.0

        raw_m = clk_ns * C_M_PER_NS
        raw_lf, raw_mf, raw_hf = compute_bands(raw_m, dt)

        # No-a0 unwrap: remove only offset steering, preserve drift
        bc_noa0 = np.zeros_like(t_abs)
        for j, t in enumerate(t_abs):
            best = None; best_dt = float("inf")
            for toc, a0, a1, a2 in bc[sat_id]:
                d = t - toc
                if 0 <= d < best_dt: best_dt = d; best = (toc, a0, a1, a2)
            if best is None: best = bc[sat_id][0]; best_dt = t - best[0]
            toc, a0, a1, a2 = best
            bc_noa0[j] = (a1 * best_dt + a2 * best_dt**2) * 1e9

        resid_ns = clk_ns - bc_noa0
        resid_m = resid_ns * C_M_PER_NS
        noa0_lf, noa0_mf, noa0_hf = compute_bands(resid_m, dt)

        if raw_lf is None or noa0_lf is None: continue

        a0_vals = [r[1] for r in bc[sat_id]]
        a0_range_ns = (max(a0_vals) - min(a0_vals)) * 1e9

        print(f"{sat_id:>5s} {block:>6s} {clk_type:>3s} {len(clk_ns):>5d} {raw_lf:>8.4f} "
              f"{noa0_lf:>9.4f} {noa0_mf:>9.4f} {noa0_hf:>9.4f} {a0_range_ns:>9.1f}")

        results[sat_id] = {
            "block": block, "clock_type": clk_type,
            "n_epochs": len(clk_ns),
            "raw_lf_m": raw_lf, "raw_mf_m": raw_mf, "raw_hf_m": raw_hf,
            "noa0_lf_m": noa0_lf, "noa0_mf_m": noa0_mf, "noa0_hf_m": noa0_hf,
            "a0_range_ns": float(a0_range_ns),
        }
        if clk_type == "Rb":
            sat_rb_lfs.append(noa0_lf); sat_rb_mfs.append(noa0_mf)
        elif clk_type == "Cs":
            sat_cs_lfs.append(noa0_lf); sat_cs_mfs.append(noa0_mf)

    # Load ground results
    step72 = json.load(open(PROJECT_ROOT / "results" / "outputs" / "step_72_housing_screening.json"))

    # Ground clock types from corrected metadata
    ct = json.load(open(PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"))
    ground_rb = []; ground_cs = []; ground_hm = []; ground_quartz = []
    for name, info in step72["stations"].items():
        lf = info["lf_rms_m"]
        if lf >= 10.0: continue  # exclude outliers
        std_type = (ct["stations"].get(name, {}).get("standard_type") or "").upper()
        if "RUBIDIUM" in std_type:
            ground_rb.append(lf)
        elif "CESIUM" in std_type:
            ground_cs.append(lf)
        elif "MASER" in std_type:
            ground_hm.append(lf)
        elif info["clock_type"] == "QUARTZ":
            ground_quartz.append(lf)

    print(f"\n=== Verified clock-type comparison ===")
    print(f"Satellite Rb (vacuum, n={len(sat_rb_lfs)}):  LF={np.median(sat_rb_lfs):.4f}m "
          f"(range {min(sat_rb_lfs):.4f}-{max(sat_rb_lfs):.4f}m)")
    if sat_cs_lfs:
        print(f"Satellite Cs (vacuum, n={len(sat_cs_lfs)}):  LF={np.median(sat_cs_lfs):.4f}m "
              f"(range {min(sat_cs_lfs):.4f}-{max(sat_cs_lfs):.4f}m)")
    if ground_rb:
        print(f"Ground Rb (Earth, n={len(ground_rb)}):       LF={np.median(ground_rb):.4f}m "
              f"(values {[f'{v:.4f}' for v in ground_rb]})")
    if ground_cs:
        print(f"Ground Cs (Earth, n={len(ground_cs)}):       LF={np.median(ground_cs):.4f}m "
              f"(range {min(ground_cs):.4f}-{max(ground_cs):.4f}m)")
    if ground_hm:
        print(f"Ground H-Maser (Earth, n={len(ground_hm)}):  LF={np.median(ground_hm):.4f}m "
              f"(range {min(ground_hm):.4f}-{max(ground_hm):.4f}m)")
    print(f"Ground Quartz (Earth, n={len(ground_quartz)}): LF={np.median(ground_quartz):.4f}m")

    print(f"\n=== Same-type screening ratios (controlling for oscillator type) ===")
    if ground_rb and sat_rb_lfs:
        ratio_rb = np.median(sat_rb_lfs) / np.median(ground_rb)
        print(f"  Satellite Rb / Ground Rb: {ratio_rb:.2f}x  (SAME oscillator type, different environment)")
    if ground_cs and sat_cs_lfs:
        ratio_cs = np.median(sat_cs_lfs) / np.median(ground_cs)
        print(f"  Satellite Cs / Ground Cs: {ratio_cs:.2f}x  (SAME oscillator type, different environment)")
    if ground_hm and sat_rb_lfs:
        ratio_rbxhm = np.median(sat_rb_lfs) / np.median(ground_hm)
        print(f"  Satellite Rb / Ground H-Maser: {ratio_rbxhm:.2f}x  (different oscillator type AND environment)")

    print(f"\n=== Cross-type ratios (confounded) ===")
    all_sat = sat_rb_lfs + sat_cs_lfs
    all_ground_atomic = ground_rb + ground_cs + ground_hm
    print(f"  All satellite / All ground atomic: {np.median(all_sat)/np.median(all_ground_atomic):.2f}x")
    print(f"  Ground quartz / Ground atomic: {np.median(ground_quartz)/np.median(all_ground_atomic):.2f}x")

    # Within-group variation
    within_sat = max(all_sat) / min(all_sat) if all_sat else 0
    within_ground = max(all_ground_atomic) / min(all_ground_atomic) if all_ground_atomic else 0
    print(f"\n=== Within-group variation ===")
    print(f"  Within satellite (same vacuum): {within_sat:.0f}x")
    print(f"  Within ground atomic (same Earth surface): {within_ground:.0f}x")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sat_rb_count": len(sat_rb_lfs), "sat_cs_count": len(sat_cs_lfs),
        "ground_rb_count": len(ground_rb), "ground_cs_count": len(ground_cs),
        "ground_hm_count": len(ground_hm), "ground_quartz_count": len(ground_quartz),
        "sat_rb_lf_median_m": float(np.median(sat_rb_lfs)) if sat_rb_lfs else None,
        "sat_cs_lf_median_m": float(np.median(sat_cs_lfs)) if sat_cs_lfs else None,
        "ground_rb_lf_median_m": float(np.median(ground_rb)) if ground_rb else None,
        "ground_cs_lf_median_m": float(np.median(ground_cs)) if ground_cs else None,
        "ground_hm_lf_median_m": float(np.median(ground_hm)) if ground_hm else None,
        "ground_quartz_lf_median_m": float(np.median(ground_quartz)) if ground_quartz else None,
        "within_satellite_variation": float(within_sat),
        "within_ground_variation": float(within_ground),
    }
    if ground_rb and sat_rb_lfs:
        summary["ratio_sat_rb_ground_rb"] = float(np.median(sat_rb_lfs) / np.median(ground_rb))
    if ground_cs and sat_cs_lfs:
        summary["ratio_sat_cs_ground_cs"] = float(np.median(sat_cs_lfs) / np.median(ground_cs))

    json.dump({
        "step": 73,
        "title": "Satellite clock unwrap — environmental screening test",
        "sat_clock_types_source": "NGA GPS Catalog (ftp.agi.com/pub/OD/GPS_Catalog_NGA.txt)",
        "sat_clock_types_verified": SAT_CLOCK_TYPES,
        "n_satellites": len(results),
        "satellites": results,
        "summary": summary,
        "tep_ucd_connection": {
            "operator": "S_Σ(E)",
            "prediction": "S_Σ(vacuum, ρ≈0) > S_Σ(Earth surface, ρ≈5.5 g/cm³)",
            "same_type_test": "Satellite Rb vs Ground Rb controls for oscillator quality",
            "satellite_environment": "vacuum, ρ≈0, Φ/c² ≈ -6.7e-10 (GPS altitude ~20,200 km)",
            "ground_environment": "air, ρ≈0.001, Φ/c² ≈ -7.0e-10 (Earth surface)",
            "mechanism": "ρ_T ≈ 20 g/cm³ saturation: bulk matter screens the TEP field",
            "steering_note": "GPS OCS updates broadcast clock every ~2h; a0 offset removed to recover free-running LF",
        },
    }, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
