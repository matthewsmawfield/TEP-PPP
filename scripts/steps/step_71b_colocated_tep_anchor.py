#!/usr/bin/env python3
"""Step 71: Co-located clock-difference TEP field measurement.

Uses co-located atomic+quartz station pairs to directly measure the LF TEP
field.  At the same location, the quartz clock sees the full TEP field while
the atomic clock is screened from the LF component by 3.78x (Step 65).  Their
difference isolates the LF field:

    quartz_residual - atomic_residual = 0.74 * TEP_LF + noise_difference

This provides a direct, model-free measurement of the TEP field at each
anchor location — no kriging, no detrending, no spatial interpolation needed.

With multiple anchors across a network, the spatial structure of the LF field
can be mapped directly and compared to the TEP-predicted correlation length
(~2000 km from Step 65).  If the field is spatially coherent, a rover
observing the same LF field (via its own quartz-atomic difference, or via
a known quartz clock compared to the network's atomic prediction) could in
principle determine its position by matching the observed field to the
spatial map.

Co-located pairs found (2024 day 100):
  GRAC/GRAS  (France,      32 m)  OCXO / H-MASER
  RIO2/RGDG  (Argentina,   47 m)  INTERNAL / RUBIDIUM
  STJ3/STJO  (Canada,      47 m)  INTERNAL / CESIUM
  SUTM/SUTH  (S. Africa,  142 m)  INTERNAL / H-MASER
  TLSG/TLSE  (France,    1.3 km)  OCXO / H-MASER
  WAB2/ZIMM  (Switzerland, 5.2 km) OCXO / H-MASER

Output: ``results/outputs/step_71b_colocated_tep_anchor.json``
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib import tep_spp as spp
from scripts.lib import tep_dynamic as dyn
from scripts.lib import tep_kernel as tk
from scripts.lib import tep_data as td
from scripts.lib import tep_stats as ts

OUT = PROJECT_ROOT / "results" / "outputs"
OUT_FILE = OUT / "step_71b_colocated_tep_anchor.json"

# 2024 day 100 — European EUREF stations
OBS_DIR = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024" / "100"
SP3_PATH = PROJECT_ROOT / "data" / "rinex" / "sp3" / "COD0MGXFIN_20241000000_01D_05M_ORB.SP3.gz"
CLK_PATH = PROJECT_ROOT / "data" / "rinex" / "clk" / "COD0MGXFIN_20241000000_01D_30S_CLK.CLK.gz"

# Also check 2024 day 200 for temporal replication
OBS_DIR_200 = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024" / "200"
SP3_PATH_200 = PROJECT_ROOT / "data" / "rinex" / "sp3" / "COD0MGXFIN_20242000000_01D_05M_ORB.SP3.gz"
CLK_PATH_200 = PROJECT_ROOT / "data" / "rinex" / "clk" / "COD0MGXFIN_20242000000_01D_30S_CLK.CLK.gz"

# Co-located pairs (quartz, atomic, distance_m) — corrected from IGS site logs
# Clock types verified from http://garner.ucsd.edu/pub/docs/site_logs/
# for 2024-04-09 (day 100)
COLOCATED_PAIRS = [
    ("THTI", "THTG", 11),     # INTERNAL / EXTERNAL CESIUM — Tahiti, 11m
    ("TLSE", "TLSG", 1266),   # INTERNAL / EXTERNAL CESIUM — Toulouse, 1.3km
    ("FAA1", "THTG", 2533),   # INTERNAL / EXTERNAL CESIUM — Tahiti, 2.5km
    ("ZIMM", "WAB2", 5199),   # INTERNAL / steered H-MASER — Zimmerwald, 5.2km
]

# Step 65 screening factor
SCREENING_FACTOR_LF = 3.78


def _load_station_ecef(name, obs_dir):
    """Get station ECEF from RINEX header."""
    import gzip
    for f in obs_dir.iterdir():
        if f.name.startswith(name) and f.name.endswith(".crx.gz"):
            try:
                with gzip.open(f, "rt") as fh:
                    for line in fh:
                        if "APPROX POSITION XYZ" in line:
                            parts = line.split()
                            return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            except Exception:
                pass
    return None


def _build_clock_series(station_name, obs_dir, sp3_path, clk_path, tmp_dir):
    """Build clock series for a single station."""
    # Find the RINEX file
    rnx_path = None
    for f in obs_dir.iterdir():
        if f.name.startswith(station_name) and f.name.endswith(".crx.gz"):
            rnx_path = f
            break
    if rnx_path is None:
        return None

    station_out = tmp_dir / f"cors_{station_name}"
    station_out.mkdir(parents=True, exist_ok=True)

    try:
        rnx = dyn._decompress_crx(rnx_path, station_out)
        if rnx is None:
            return None
        hdr = dyn._rinex_header_xyz(rnx)
        if hdr is None:
            return None
        s = dyn.build_clock_series_precise_multi(rnx, sp3_path, clk_path, hdr,
                                                  station_name, systems=("G",))
        return s
    except Exception:
        return None


def _bandpass(y, dt, f_lo, f_hi):
    """Bandpass filter to [f_lo, f_hi] uHz."""
    if f_lo <= 0 and f_hi >= 5000:
        return y.copy()
    return ts.bandpass(y, dt, f_lo, f_hi)


def _great_circle_km(pos1, pos2):
    """Great-circle distance between two ECEF positions in km."""
    lat1 = np.arctan2(pos1[2], np.hypot(pos1[0], pos1[1]))
    lon1 = np.arctan2(pos1[1], pos1[0])
    lat2 = np.arctan2(pos2[2], np.hypot(pos2[0], pos2[1]))
    lon2 = np.arctan2(pos2[1], pos2[0])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main() -> int:
    print("=== step_71: co-located clock-difference TEP field measurement ===")

    # Band definitions (from Step 65)
    # LF: <10 uHz (periods >28h — the "drift")
    # MF: 10-500 uHz (the TEP band)
    # HF: >500 uHz (periods <33min — oscillator noise)
    BANDS = {
        "LF": (0, 10),
        "MF": (10, 500),
        "HF": (500, 5000),
        "FULL": (0, 5000),
    }

    results = []
    import tempfile
    tmp_base = PROJECT_ROOT / "tmp"
    tmp_base.mkdir(exist_ok=True)

    # Process day 100
    for day_label, obs_dir, sp3_path, clk_path in [
        ("2024-100", OBS_DIR, SP3_PATH, CLK_PATH),
    ]:
        print(f"\n--- Processing {day_label} ---")
        if not sp3_path.exists():
            print(f"  SP3 not found: {sp3_path}")
            continue

        with tempfile.TemporaryDirectory(prefix="step71_", dir=str(tmp_base)) as td_dir:
            tmp_dir = Path(td_dir)

            # Build clock series for all stations in the pairs
            all_stations = set()
            for q, a, _ in COLOCATED_PAIRS:
                all_stations.add(q)
                all_stations.add(a)

            series = {}
            for name in sorted(all_stations):
                print(f"  Building clock series for {name}...")
                s = _build_clock_series(name, obs_dir, sp3_path, clk_path, tmp_dir)
                if s is not None:
                    series[name] = s
                    print(f"    {len(s['t'])} epochs")
                else:
                    print(f"    FAILED")

            # Process each co-located pair
            for q_name, a_name, dist_m in COLOCATED_PAIRS:
                if q_name not in series or a_name not in series:
                    print(f"\n  Pair {q_name}/{a_name}: missing clock series")
                    continue

                q_s = series[q_name]
                a_s = series[a_name]

                # Align on common time grid
                t = np.asarray(q_s["t"], float)
                # raw is in nanoseconds; resid is robust-detrended in ns
                # But raw may have 1ms steering wraps that corrupt the detrend.
                # Unwrap the raw clock series before detrending.
                STEERING_PERIOD_NS = 1_000_000.0  # 1ms in ns
                q_raw_ns = np.asarray(q_s["raw"], float)
                a_raw_ns = np.asarray(a_s["raw"], float)

                # Unwrap 1ms steering (receiver resets clock to stay within ±0.5ms)
                q_unwrapped = (np.unwrap(q_raw_ns * (2.0 * np.pi / STEERING_PERIOD_NS))
                               * STEERING_PERIOD_NS / (2.0 * np.pi))
                a_unwrapped = (np.unwrap(a_raw_ns * (2.0 * np.pi / STEERING_PERIOD_NS))
                               * STEERING_PERIOD_NS / (2.0 * np.pi))

                # Detrend the unwrapped series
                q_resid = td.robust_detrend(q_unwrapped)
                a_resid = td.robust_detrend(a_unwrapped)

                # Convert to metres (c * dt)
                q_m = q_resid * dyn.C_M_PER_NS
                a_m = a_resid * dyn.C_M_PER_NS

                # The clock difference (quartz - atomic) — both already detrended
                diff_full = q_m - a_m

                # Bandpass the difference into LF, MF, HF
                dt = float(np.median(np.diff(t)))
                band_results = {}
                for band_name, (f_lo, f_hi) in BANDS.items():
                    diff_bp = _bandpass(diff_full, dt, f_lo, f_hi)
                    q_bp = _bandpass(q_m, dt, f_lo, f_hi)
                    a_bp = _bandpass(a_m, dt, f_lo, f_hi)

                    rms_diff = float(np.sqrt(np.mean(diff_bp ** 2)))
                    rms_q = float(np.sqrt(np.mean(q_bp ** 2)))
                    rms_a = float(np.sqrt(np.mean(a_bp ** 2)))

                    # The screening factor: quartz_rms / atomic_rms
                    if rms_a > 0:
                        screening = rms_q / rms_a
                    else:
                        screening = float('nan')

                    # The fraction of the quartz LF that is TEP field
                    # (vs oscillator noise): if the difference is purely TEP,
                    # then diff_rms = 0.74 * TEP_LF, and quartz_rms = TEP_LF + noise
                    # The predictable fraction = (diff_rms / 0.74)^2 / quartz_rms^2
                    if rms_diff > 0 and rms_q > 0:
                        tep_fraction = (rms_diff / 0.74) ** 2 / rms_q ** 2
                        tep_fraction = min(tep_fraction, 1.0)
                    else:
                        tep_fraction = 0.0

                    band_results[band_name] = {
                        "rms_quartz_m": rms_q,
                        "rms_atomic_m": rms_a,
                        "rms_difference_m": rms_diff,
                        "screening_factor": screening,
                        "tep_fraction_of_quartz": tep_fraction,
                    }

                # Cross-correlation between quartz and atomic in each band
                xcorr = {}
                for band_name, (f_lo, f_hi) in BANDS.items():
                    q_bp = _bandpass(q_m, dt, f_lo, f_hi)
                    a_bp = _bandpass(a_m, dt, f_lo, f_hi)
                    if np.std(q_bp) > 0 and np.std(a_bp) > 0:
                        cc = float(np.corrcoef(q_bp, a_bp)[0, 1])
                    else:
                        cc = 0.0
                    xcorr[band_name] = cc

                # Get positions
                q_xyz = _load_station_ecef(q_name, obs_dir)
                a_xyz = _load_station_ecef(a_name, obs_dir)

                pair_result = {
                    "quartz_station": q_name,
                    "atomic_station": a_name,
                    "separation_m": dist_m,
                    "quartz_xyz": q_xyz.tolist() if q_xyz is not None else None,
                    "atomic_xyz": a_xyz.tolist() if a_xyz is not None else None,
                    "quartz_clock_type": "INTERNAL (quartz)",
                    "atomic_clock_type": "EXTERNAL (atomic)",
                    "n_epochs": len(t),
                    "duration_s": float(t[-1] - t[0]),
                    "steering_unwrapped": True,
                    "bands": band_results,
                    "cross_correlation": xcorr,
                    "day": day_label,
                }
                results.append(pair_result)

                print(f"\n  Pair {q_name}/{a_name} ({dist_m}m):")
                for band in ["LF", "MF", "HF", "FULL"]:
                    b = band_results[band]
                    print(f"    {band}: quartz={b['rms_quartz_m']:.3f}m, "
                          f"atomic={b['rms_atomic_m']:.3f}m, "
                          f"diff={b['rms_difference_m']:.3f}m, "
                          f"screening={b['screening_factor']:.2f}x, "
                          f"tep_frac={b['tep_fraction_of_quartz']:.1%}, "
                          f"xcorr={xcorr[band]:.3f}")

    # Spatial analysis: compute pairwise field correlations between anchors
    print("\n=== Spatial field correlation between anchors ===")
    # Get LF band of the clock difference for each pair
    anchor_lf = {}
    for r in results:
        if r["quartz_xyz"] is not None:
            pos = np.array(r["quartz_xyz"])
            # The LF difference is a direct measurement of the TEP field
            # at this location.  We need the time series, not just the RMS.
            # For now, use the RMS as the field amplitude.
            anchor_lf[r["quartz_station"]] = {
                "pos": pos,
                "lf_rms": r["bands"]["LF"]["rms_difference_m"],
                "mf_rms": r["bands"]["MF"]["rms_difference_m"],
            }

    if len(anchor_lf) >= 2:
        print(f"  {len(anchor_lf)} anchors with positions:")
        distances = []
        lf_ratios = []
        for n1, d1 in anchor_lf.items():
            for n2, d2 in anchor_lf.items():
                if n1 >= n2:
                    continue
                dist = _great_circle_km(d1["pos"], d2["pos"])
                # Correlation of field amplitude vs distance
                distances.append(dist)
                lf_ratios.append(d1["lf_rms"] / d2["lf_rms"] if d2["lf_rms"] > 0 else 0)
                print(f"    {n1}-{n2}: {dist:.0f} km, "
                      f"LF ratio={d1['lf_rms']/d2['lf_rms'] if d2['lf_rms']>0 else 0:.2f}")

    # Summary
    print("\n=== Summary ===")
    print(f"Co-located pairs processed: {len(results)}")
    lf_screenings = [r["bands"]["LF"]["screening_factor"] for r in results
                     if np.isfinite(r["bands"]["LF"]["screening_factor"])]
    mf_screenings = [r["bands"]["MF"]["screening_factor"] for r in results
                     if np.isfinite(r["bands"]["MF"]["screening_factor"])]
    hf_screenings = [r["bands"]["HF"]["screening_factor"] for r in results
                     if np.isfinite(r["bands"]["HF"]["screening_factor"])]
    if lf_screenings:
        print(f"  LF screening: median={np.median(lf_screenings):.2f}x "
              f"(Step 65: 3.78x)")
        print(f"  MF screening: median={np.median(mf_screenings):.2f}x "
              f"(Step 65: 1.20x)")
        print(f"  HF screening: median={np.median(hf_screenings):.2f}x "
              f"(Step 65: 0.91x)")
    lf_xcorr = [r["cross_correlation"]["LF"] for r in results]
    mf_xcorr = [r["cross_correlation"]["MF"] for r in results]
    hf_xcorr = [r["cross_correlation"]["HF"] for r in results]
    print(f"  LF cross-corr (quartz-atomic at same site): "
          f"median={np.median(lf_xcorr):.3f}")
    print(f"  MF cross-corr: median={np.median(mf_xcorr):.3f}")
    print(f"  HF cross-corr: median={np.median(hf_xcorr):.3f}")

    lf_tep_frac = [r["bands"]["LF"]["tep_fraction_of_quartz"] for r in results]
    print(f"  LF TEP fraction of quartz variance: "
          f"median={np.median(lf_tep_frac):.1%}")

    output = {
        "step": 71,
        "title": "Co-located clock-difference TEP field measurement",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(results),
        "pairs": results,
        "summary": {
            "lf_screening_median": float(np.median(lf_screenings)) if lf_screenings else None,
            "mf_screening_median": float(np.median(mf_screenings)) if mf_screenings else None,
            "hf_screening_median": float(np.median(hf_screenings)) if hf_screenings else None,
            "lf_xcorr_median": float(np.median(lf_xcorr)) if lf_xcorr else None,
            "mf_xcorr_median": float(np.median(mf_xcorr)) if mf_xcorr else None,
            "hf_xcorr_median": float(np.median(hf_xcorr)) if hf_xcorr else None,
            "lf_tep_fraction_median": float(np.median(lf_tep_frac)) if lf_tep_frac else None,
            "step65_reference": {
                "lf_screening": 3.78,
                "mf_screening": 1.20,
                "hf_screening": 0.91,
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
