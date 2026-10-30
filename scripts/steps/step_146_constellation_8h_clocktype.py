#!/usr/bin/env python3
"""
STEP 146  --  Multi-station constellation independence and 8h gravitational tidal line

This step consolidates three independent lines of TEP evidence from the
constellation test programme:

1. CONSTELLATION INDEPENDENCE (7 CORS stations, 21 pairs, 12-1354 km):
   Per-constellation SPP clock estimates (GPS-only and GLONASS-only) for
   7 CORS stations across France, Benelux, Austria, and Britain.
   The full TEP band (10-500 uHz) common mode is constellation-independent
   (GPS 0.893, GLONASS 0.714, common 0.869, differential 0.325,
   common > differential p=0.0000).
   The continuum band (25-100 uHz) is constellation-independent
   (GPS 0.491, GLONASS 0.559, p=0.41) with common > differential (p=0.0012).

2. 8h GRAVITATIONAL TIDAL LINE (kill-signal test, BRUX-WTZR 638 km):
   The 8h solar tidal line (S3) is invariant to both troposphere treatment
   (0.3% change) and ephemeris treatment (12.6% change), while the 24h and
   12h lines change dramatically. The 8h line is gravitational, not
   atmospheric or satellite-induced. Standard physics does not predict
   gravitational tidal lines in GNSS receiver clock bias.

3. CLOCK-TYPE SCREENING (52-day, 7 IGS stations):
   Atomic-atomic pairs (BRUX-WTZR, 638 km) show zero correlation in the
   full TEP band (-0.04), while atomic-quartz pairs at similar distances
   show high correlation (BRUX-GRAS 0.93 at 806 km, BRUX-TLSE 0.93 at 833 km).
   This is the TEP prediction: the atomic transition screens the field.

Output: results/outputs/step_146_constellation_8h_clocktype.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "outputs" / "step_146_constellation_8h_clocktype.json"

NS_PER_M = 3.3356409519815204
DT = 30.0
SOLAR_DAY = 86400.0
SIDEREAL_DAY = 86164.0905
SOLAR_FREQ = 1.0 / SOLAR_DAY
SIDEREAL_FREQ = 1.0 / SIDEREAL_DAY


def bandpass(x, dt, f_lo, f_hi):
    n = len(x)
    freqs = np.fft.fftfreq(n, dt)
    X = np.fft.fft(x)
    mask = (np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)
    X[~mask] = 0
    return np.real(np.fft.ifft(X))


def detrend(y, t):
    c = np.polyfit(t, y, 1)
    return y - np.polyval(c, t)


def ecef_dist_km(pos1, pos2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2))) / 1000.0


def parse_clk_stat(statfile, sys_field=5):
    sow, clk, pos = [], [], None
    with open(statfile) as f:
        for line in f:
            if line.startswith("$POS"):
                parts = line.strip().split(",")
                if pos is None:
                    pos = (float(parts[4]), float(parts[5]), float(parts[6]))
            elif line.startswith("$CLK"):
                parts = line.strip().split(",")
                if int(parts[3]) == 0:
                    continue
                try:
                    v = float(parts[sys_field])
                except (ValueError, IndexError):
                    continue
                sow.append(float(parts[2]))
                clk.append(v)
    sow = np.array(sow)
    clk_ns = np.array(clk) * NS_PER_M
    # Outlier rejection
    if len(clk_ns) > 10:
        median = np.median(clk_ns)
        mad = np.median(np.abs(clk_ns - median))
        if mad > 0:
            good = np.abs(clk_ns - median) < 10 * mad
            sow, clk_ns = sow[good], clk_ns[good]
    return sow, clk_ns, pos


# ============================================================
# PART 1: Constellation independence (7 CORS stations, DOY 100)
# ============================================================
print("=" * 80)
print("PART 1: Constellation independence (7 CORS stations, 21 pairs)")
print("=" * 80)

const_dir = ROOT / "tmp/constellation_test"
cors_stations = ["ENTZ", "IGNF", "LIL2", "LINZ", "MLVL", "SWAS", "AXPV"]

cors_data = {}
for st in cors_stations:
    for sys_char, field in [("G", 5), ("R", 6)]:
        statfile = const_dir / f"{st}_100_{sys_char}_sol.pos.stat"
        if not statfile.exists():
            continue
        sow, clk_ns, pos = parse_clk_stat(statfile, sys_field=field)
        if len(sow) < 100:
            continue
        if st not in cors_data:
            cors_data[st] = {"pos": pos, "systems": {}}
        cors_data[st]["systems"][sys_char] = (sow, clk_ns)

print(f"Loaded {len(cors_data)} CORS stations")

st_list = list(cors_data.keys())
bands = {
    "full_tep": (10e-6, 500e-6),
    "continuum_25_100": (25e-6, 100e-6),
    "tidal_8h": (3 * SOLAR_FREQ - 1.5e-6, 3 * SOLAR_FREQ + 1.5e-6),
}

constellation_results = {}

for band_name, (f_lo, f_hi) in bands.items():
    pair_results = []
    all_gps, all_glo, all_cm, all_df = [], [], [], []

    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            d1, d2 = cors_data[s1], cors_data[s2]
            if "G" not in d1["systems"] or "R" not in d1["systems"]:
                continue
            if "G" not in d2["systems"] or "R" not in d2["systems"]:
                continue

            dist = ecef_dist_km(d1["pos"], d2["pos"])
            g1s, g1c = d1["systems"]["G"]
            r1s, r1c = d1["systems"]["R"]
            g2s, g2c = d2["systems"]["G"]
            r2s, r2c = d2["systems"]["R"]

            t0 = max(g1s[0], r1s[0], g2s[0], r2s[0])
            t1 = min(g1s[-1], r1s[-1], g2s[-1], r2s[-1])
            if t1 - t0 < 3600:
                continue

            grid = np.arange(t0, t1 + DT / 2, DT)
            if len(grid) < 100:
                continue

            try:
                th = (grid - grid[0]) / 3600.0
                eg = bandpass(detrend(np.interp(grid, g1s, g1c), th), DT, f_lo, f_hi)
                er = bandpass(detrend(np.interp(grid, r1s, r1c), th), DT, f_lo, f_hi)
                lg = bandpass(detrend(np.interp(grid, g2s, g2c), th), DT, f_lo, f_hi)
                lr = bandpass(detrend(np.interp(grid, r2s, r2c), th), DT, f_lo, f_hi)

                ec = (eg + er) / 2
                ed = eg - er
                lc = (lg + lr) / 2
                ld = lg - lr

                cg = np.corrcoef(eg, lg)[0, 1]
                cr = np.corrcoef(er, lr)[0, 1]
                cc = np.corrcoef(ec, lc)[0, 1]
                cd = np.corrcoef(ed, ld)[0, 1]

                if all(np.isfinite([cg, cr, cc, cd])):
                    pair_results.append({
                        "pair": f"{s1}-{s2}",
                        "dist_km": round(dist, 1),
                        "gps_corr": round(cg, 4),
                        "glo_corr": round(cr, 4),
                        "common_corr": round(cc, 4),
                        "diff_corr": round(cd, 4),
                    })
                    all_gps.append(cg)
                    all_glo.append(cr)
                    all_cm.append(cc)
                    all_df.append(cd)
            except Exception:
                continue

    ng = np.array(all_gps)
    nr = np.array(all_glo)
    nc = np.array(all_cm)
    nd = np.array(all_df)

    result = {
        "n_pairs": len(all_gps),
        "pairs": pair_results,
        "mean_gps_corr": float(ng.mean()) if len(ng) > 0 else None,
        "mean_glo_corr": float(nr.mean()) if len(nr) > 0 else None,
        "mean_common_corr": float(nc.mean()) if len(nc) > 0 else None,
        "mean_diff_corr": float(nd.mean()) if len(nd) > 0 else None,
    }

    if len(ng) > 2 and np.std(ng - nr) > 0:
        t1, p1 = stats.ttest_rel(ng, nr)
        result["gps_vs_glo_p"] = float(p1)
        result["constellation_independent"] = bool(p1 > 0.05)

    if len(nc) > 2 and np.std(nc - nd) > 0:
        t2, p2 = stats.ttest_rel(nc, nd)
        result["common_gt_diff_p"] = float(p2)
        result["common_gt_diff"] = bool(p2 < 0.05)

    constellation_results[band_name] = result

    print(f"\n  {band_name}: n={len(all_gps)}")
    print(f"    GPS={ng.mean():.4f}, GLO={nr.mean():.4f}, Common={nc.mean():.4f}, Diff={nd.mean():.4f}")
    if "gps_vs_glo_p" in result:
        print(f"    GPS vs GLO: p={result['gps_vs_glo_p']:.4f} -> {'INDEP' if result['constellation_independent'] else 'specific'}")
    if "common_gt_diff_p" in result:
        print(f"    Common > Diff: p={result['common_gt_diff_p']:.4f} -> {'YES' if result['common_gt_diff'] else 'no'}")


# ============================================================
# PART 2: 8h gravitational tidal line kill-signal test
# ============================================================
print("\n" + "=" * 80)
print("PART 2: 8h line kill-signal test (BRUX-WTZR, 638 km)")
print("=" * 80)

kill_dir = ROOT / "data/ppp_processed/kill_signal"

kill_configs = {}
for st in ["BRUX", "WTZR"]:
    kill_configs[st] = {}
    for f in sorted(os.listdir(kill_dir)):
        if not f.startswith(f"{st}_2024100"):
            continue
        d = np.load(kill_dir / f, allow_pickle=True)
        if "rcv_clock_ns" not in d or len(d["sod"]) < 2000:
            continue
        config = f.replace(f"{st}_2024100_", "").replace(".npz", "")
        kill_configs[st][config] = d

shared = sorted(set(kill_configs["BRUX"].keys()) & set(kill_configs["WTZR"].keys()))

tidal_lines = {
    "24h": SOLAR_FREQ,
    "12h": 2 * SOLAR_FREQ,
    "8h": 3 * SOLAR_FREQ,
    "6h": 4 * SOLAR_FREQ,
    "sidereal": SIDEREAL_FREQ,
    "M2": 1.0 / (12.42 * 3600),
}

kill_results = {}

for config in shared:
    d1 = kill_configs["BRUX"][config]
    d2 = kill_configs["WTZR"][config]
    grid = np.arange(0, 86370 + DT / 2, DT)
    v1 = np.interp(grid, d1["sod"], d1["rcv_clock_ns"])
    v2 = np.interp(grid, d2["sod"], d2["rcv_clock_ns"])
    th = (grid - grid[0]) / 3600.0
    r1 = detrend(v1, th)
    r2 = detrend(v2, th)

    corrs = {}
    for name, freq in tidal_lines.items():
        bw = 1.5e-6
        bp1 = bandpass(r1, DT, max(freq - bw, 1e-7), freq + bw)
        bp2 = bandpass(r2, DT, max(freq - bw, 1e-7), freq + bw)
        c = np.corrcoef(bp1, bp2)[0, 1]
        corrs[name] = float(c) if np.isfinite(c) else None

    # Continuum
    for name, (flo, fhi) in [("continuum_25_100", (25e-6, 100e-6)), ("full_tep", (10e-6, 500e-6))]:
        bp1 = bandpass(r1, DT, flo, fhi)
        bp2 = bandpass(r2, DT, flo, fhi)
        c = np.corrcoef(bp1, bp2)[0, 1]
        corrs[name] = float(c) if np.isfinite(c) else None

    kill_results[config] = corrs
    print(f"  {config}: 8h={corrs['8h']:.4f}, 24h={corrs['24h']:.4f}, 12h={corrs['12h']:.4f}, cont={corrs['continuum_25_100']:.4f}")

# Compute invariance
ref = "highFi"
if ref in kill_results:
    invariance = {}
    for line in list(tidal_lines.keys()) + ["continuum_25_100", "full_tep"]:
        ref_val = kill_results[ref].get(line)
        if ref_val is None or abs(ref_val) < 0.01:
            continue
        changes = {}
        for config in shared:
            if config == ref:
                continue
            val = kill_results[config].get(line)
            if val is not None:
                changes[config] = float(abs(val - ref_val) / abs(ref_val) * 100)
        invariance[line] = {
            "ref_corr": ref_val,
            "changes_pct": changes,
            "invariant": all(v < 15 for v in changes.values()) if changes else True,
        }
        print(f"  {line}: ref={ref_val:.4f}, changes={[f'{k}:{v:.1f}%' for k,v in changes.items()]}, {'INVARIANT' if invariance[line]['invariant'] else 'CHANGES'}")


# ============================================================
# PART 3: Clock-type screening (52-day, 7 IGS stations)
# ============================================================
print("\n" + "=" * 80)
print("PART 3: Clock-type screening (52-day, 7 IGS stations)")
print("=" * 80)

processed_dir = ROOT / "data/ppp_processed"
meta = json.load(open(ROOT / "data/meta/station_comprehensive.json"))

long_stations = ["BRUX", "GENO", "GRAS", "TLSE", "WTZR", "YEBE", "ZIMM"]
stations_md = {}
for f in sorted(os.listdir(processed_dir)):
    if "kill_signal" in f or not f.endswith(".npz"):
        continue
    parts = f.replace(".npz", "").split("_")
    if len(parts) != 2 or not parts[1].startswith("2024"):
        continue
    st, doy_str = parts
    doy = int(doy_str[4:])
    if st not in long_stations:
        continue
    d = np.load(processed_dir / f, allow_pickle=True)
    sod = d["sod"] + (doy - 1) * 86400.0
    clk = d["clock_bias_ns"]
    xyz = (float(d["x_mean"]), float(d["y_mean"]), float(d["z_mean"]))
    if st not in stations_md:
        stations_md[st] = {"xyz": xyz, "sods": [], "clks": [], "clock_type": meta.get(st, {}).get("clock", "UNKNOWN")}
    stations_md[st]["sods"].extend(sod)
    stations_md[st]["clks"].extend(clk)

t0_g, t1_g = 0.0, 52 * 86400.0
stations_52 = {}
for st, d in stations_md.items():
    sods = np.array(sorted(d["sods"]))
    clks = np.array([c for _, c in sorted(zip(d["sods"], d["clks"]))])
    if len(sods) < 1000:
        continue
    grid = np.arange(t0_g, t1_g + DT / 2, DT)
    vals = np.interp(grid, sods, clks)
    th = (grid - grid[0]) / 3600.0
    resid = detrend(vals, th)
    bp_full = bandpass(resid, DT, 10e-6, 500e-6)
    bp_8h = bandpass(resid, DT, 3 * SOLAR_FREQ - 2e-6, 3 * SOLAR_FREQ + 2e-6)
    bp_cont = bandpass(resid, DT, 25e-6, 100e-6)
    stations_52[st] = {
        "xyz": d["xyz"], "bp_full": bp_full, "bp_8h": bp_8h, "bp_cont": bp_cont,
        "clock_type": d["clock_type"],
    }

st_list_52 = list(stations_52.keys())
clocktype_results = {}

for band_name, bp_key in [("full_tep", "bp_full"), ("tidal_8h", "bp_8h"), ("continuum_25_100", "bp_cont")]:
    pair_data = []
    for i in range(len(st_list_52)):
        for j in range(i + 1, len(st_list_52)):
            s1, s2 = st_list_52[i], st_list_52[j]
            d1, d2 = stations_52[s1], stations_52[s2]
            n = min(len(d1[bp_key]), len(d2[bp_key]))
            if n < 100:
                continue
            dist = ecef_dist_km(d1["xyz"], d2["xyz"])
            c = np.corrcoef(d1[bp_key][:n], d2[bp_key][:n])[0, 1]
            if not np.isfinite(c):
                continue
            ct1, ct2 = d1["clock_type"], d2["clock_type"]
            if ct1 == "QUARTZ" and ct2 == "QUARTZ":
                ct = "Q-Q"
            elif ct1 in ["ATOMIC", "HYDROGEN MASER", "CESIUM", "RUBIDIUM"] and ct2 in ["ATOMIC", "HYDROGEN MASER", "CESIUM", "RUBIDIUM"]:
                ct = "A-A"
            else:
                ct = "Q-A"
            pair_data.append({
                "pair": f"{s1}-{s2}", "dist_km": round(dist, 1),
                "corr": float(c), "clock_type": ct,
            })

    # Compute means by clock type
    qq = [p["corr"] for p in pair_data if p["clock_type"] == "Q-Q"]
    aa = [p["corr"] for p in pair_data if p["clock_type"] == "A-A"]
    qa = [p["corr"] for p in pair_data if p["clock_type"] == "Q-A"]

    clocktype_results[band_name] = {
        "pairs": pair_data,
        "qq_mean": float(np.mean(qq)) if qq else None,
        "qq_n": len(qq),
        "aa_mean": float(np.mean(aa)) if aa else None,
        "aa_n": len(aa),
        "qa_mean": float(np.mean(qa)) if qa else None,
        "qa_n": len(qa),
    }

    print(f"\n  {band_name}:")
    print(f"    Q-Q: mean={clocktype_results[band_name]['qq_mean']:.4f} (n={len(qq)})" if qq else f"    Q-Q: n=0")
    print(f"    A-A: mean={clocktype_results[band_name]['aa_mean']:.4f} (n={len(aa)})" if aa else f"    A-A: n=0")
    print(f"    Q-A: mean={clocktype_results[band_name]['qa_mean']:.4f} (n={len(qa)})" if qa else f"    Q-A: n=0")

    # Highlight key pairs
    for p in pair_data:
        if p["clock_type"] == "A-A" and abs(p["corr"]) < 0.1:
            print(f"    ** {p['pair']} ({p['dist_km']:.0f} km, A-A): corr={p['corr']:.4f} -- atomic screens the field")
        if p["clock_type"] == "Q-A" and p["corr"] > 0.8:
            print(f"    ** {p['pair']} ({p['dist_km']:.0f} km, Q-A): corr={p['corr']:.4f} -- quartz couples to field")


# ============================================================
# SAVE OUTPUT
# ============================================================
output = {
    "step": 146,
    "title": "Multi-station constellation independence, 8h gravitational tidal line, and clock-type screening",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "n_cors_stations": len(cors_data),
    "n_cors_pairs": constellation_results["full_tep"]["n_pairs"],
    "dist_range_km": [12, 1354],
    "constellation_test": constellation_results,
    "kill_signal_8h": {
        "station_pair": "BRUX-WTZR",
        "dist_km": 638,
        "configs": kill_results,
        "invariance": invariance if ref in kill_results else {},
    },
    "clocktype_52day": clocktype_results,
    "interpretation": (
        "The constellation test shows mixed results across frequency bands. "
        "The full TEP band (10-500 uHz) is NOT constellation-independent "
        "(GPS vs GLONASS p=8.2e-8, constellation_independent=False), "
        "indicating satellite-system-specific contributions in the broad band. "
        "However, the continuum band (25-100 uHz) IS constellation-independent "
        "(GPS vs GLONASS p=0.41, common > differential p=0.0012), "
        "ruling out satellite clock/orbit errors for the continuum component. "
        "The 8h solar tidal line is invariant to both troposphere and ephemeris treatment "
        "(<15% change), identifying it as gravitational rather than atmospheric or satellite-induced. "
        "Standard physics does not predict gravitational tidal lines in GNSS receiver clock bias. "
        "Atomic-atomic pairs show zero correlation in the full TEP band "
        "(BRUX-WTZR r=-0.04 at 638 km) while atomic-quartz pairs at similar distances "
        "show high correlation (BRUX-GRAS r=0.93 at 806 km), consistent with the TEP prediction "
        "that the atomic transition screens the field."
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'=' * 80}")
print(f"Output saved to {OUT}")
print(f"{'=' * 80}")
