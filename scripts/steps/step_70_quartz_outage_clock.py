#!/usr/bin/env python3
"""Step 70: Clock-type-aware TEP outage test on quartz CORS receivers.

Uses the Step 65 finding that quartz clocks are 3.78x more sensitive to the
LF TEP field than atomic clocks.  Each quartz EUREF station is treated as a
simulated "rover" experiencing a clock outage, while the remaining stations
form the reference network.  The TEP kriging prior is tested in four modes:

1. ``none``           — standard kriging (all stations equal)
2. ``quartz_only``    — only quartz reference stations
3. ``screening_scaled`` — delta x 3.78 LF screening factor
4. ``type_weighted``  — quartz stations upweighted by screening factor

The test uses 24-hour archives (resolving the LF band) with outages from
10 minutes to 8 hours (covering MF and LF bands), directly testing whether
the clock-type-aware correction recovers the TEP field that atomic-dominated
networks underpredict for quartz receivers.

Output: ``results/outputs/step_70_quartz_outage_clock.json``
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib import tep_spp as spp
from scripts.lib import tep_dynamic as dyn
from scripts.lib import tep_kernel as tk
from datetime import datetime, timezone

OUT = PROJECT_ROOT / "results" / "outputs"
OUT_FILE = OUT / "step_70_quartz_outage_clock.json"

# 2024 day 100 — European EUREF stations (symlinked to avoid disk space)
DOY_DIR = PROJECT_ROOT / "tmp" / "eu_cors_2024_100"
SP3_PATH = PROJECT_ROOT / "data" / "rinex" / "sp3" / "COD0MGXFIN_20241000000_01D_05M_ORB.SP3.gz"
CLK_PATH = PROJECT_ROOT / "data" / "rinex" / "clk" / "COD0MGXFIN_20241000000_01D_30S_CLK.CLK.gz"

QUARTZ_TYPES = {"INTERNAL", "OCXO"}
ATOMIC_TYPES = {"H-MASER", "RUBIDIUM", "CESIUM"}
SCREENING_FACTOR = 3.78  # Step 65: quartz LF / atomic LF

# Outage durations to test (seconds)
OUTAGE_DURATIONS = [600, 1800, 3600, 7200, 14400, 28800]  # 10min to 8h


def _load_station_types():
    """Load clock types from corrected IGS site logs.

    Falls back to the original metadata if the corrected table is unavailable.
    The IGS site logs (parsed by scripts/utils/parse_igs_site_logs.py) provide
    the actual frequency standard active on 2024-04-09, which differs from
    the original metadata for 27 of 100 stations.
    """
    corrected_path = PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"
    if corrected_path.exists():
        ct = json.load(open(corrected_path))
        type_map = {}
        for name, info in ct.get("stations", {}).items():
            t = info.get("corrected_type", "UNKNOWN")
            type_map[name] = t
        return type_map
    # Fallback to original metadata
    meta = json.load(open(PROJECT_ROOT / "data" / "meta" / "optimal_100_metadata.json"))
    stations = meta.get("stations", {})
    type_map = {}
    for name, info in stations.items():
        ct = info.get("classification", {}).get("clock_type", "UNKNOWN")
        if ct in QUARTZ_TYPES:
            type_map[name] = "QUARTZ"
        elif ct in ATOMIC_TYPES:
            type_map[name] = "ATOMIC"
        else:
            type_map[name] = "UNKNOWN"
    return type_map


def _station_ecef(name):
    """Get station ECEF from RINEX header."""
    import gzip
    # Look in the original RINEX directory (symlinks point there)
    orig_dir = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024" / "100"
    rnx_path = None
    for f in orig_dir.iterdir():
        if f.name.startswith(name) and f.name.endswith(".crx.gz"):
            rnx_path = f
            break
    if rnx_path is None:
        # try symlink dir
        for f in DOY_DIR.iterdir():
            if f.name.startswith(name) and f.name.endswith(".crx.gz"):
                rnx_path = f
                break
    if rnx_path is None:
        return None
    try:
        with gzip.open(rnx_path, "rt") as f:
            for line in f:
                if "APPROX POSITION XYZ" in line:
                    parts = line.split()
                    return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
    except Exception:
        return None
    return None


def main() -> int:
    print("=== step_70: clock-type-aware TEP outage test on quartz CORS ===")
    type_map = _load_station_types()

    # List available stations — restrict to European for disk/memory
    eu_countries = {"BEL", "ITA", "CZE", "POL", "FRA", "DEU", "NLD", "CHE",
                    "AUT", "ESP", "PRT", "GBR", "IRL", "SVK", "SVN", "HUN",
                    "HRV", "ROU", "BUL", "GRC", "SWE", "NOR", "FIN", "LTU",
                    "LVA", "EST", "DNK", "LUX"}
    available = []
    for f in DOY_DIR.iterdir():
        if f.name.endswith(".crx.gz"):
            name = f.name[:4]
            country = f.name[6:9]
            if country not in eu_countries:
                continue
            ct = type_map.get(name, "UNKNOWN")
            if ct in ("QUARTZ", "ATOMIC"):
                available.append((name, ct, f))
    print(f"  Available: {len(available)} stations "
          f"({sum(1 for _,c,_ in available if c=='QUARTZ')} quartz, "
          f"{sum(1 for _,c,_ in available if c=='ATOMIC')} atomic)")

    # Load SP3/CLK (pass paths to build_cors_series_multi which re-reads per station)
    print("  Loading SP3/CLK...")
    sp3 = SP3_PATH
    clk = CLK_PATH
    sp3_dict = spp.read_sp3_multi(sp3)
    clk_dict = spp.read_clk_multi(clk)
    dense = spp.dense_grid_multi(sp3_dict, clk_dict, step=30.0)
    print(f"  SP3: {len(sp3_dict)} sats, CLK: {len(clk_dict)} sats")

    # Build clock series for all stations
    print("  Building clock series for all stations...")
    import tempfile
    tmp_base = PROJECT_ROOT / "tmp"
    tmp_base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step70_", dir=str(tmp_base)) as td_dir:
        all_series = dyn.build_cors_series_multi(
            DOY_DIR, sp3, clk, Path(td_dir), systems=("G",))
        print(f"  Clock series: {len(all_series)} stations")

        # For each quartz station, run outage test
        quartz_stations = [(n, f) for n, c, f in available if c == "QUARTZ"
                           and n in all_series]
        print(f"  European quartz targets: {len(quartz_stations)}")
        # Use up to 16 targets for better statistics
        targets = quartz_stations[:16]
        print(f"  Testing {len(targets)} targets: {[t[0] for t in targets]}")

        results = []
        for tgt_name, tgt_file in targets:
            print(f"\n  --- Target: {tgt_name} (QUARTZ) ---")
            tgt_series = all_series[tgt_name]
            tgt_xyz = _station_ecef(tgt_name)
            if tgt_xyz is None:
                print(f"    skip: no ECEF position")
                continue

            # Reference network = all other stations
            ref_series = {n: s for n, s in all_series.items()
                          if n != tgt_name and n in all_series}
            ref_types = {n: type_map.get(n, "UNKNOWN") for n in ref_series}
            n_quartz_ref = sum(1 for v in ref_types.values() if v == "QUARTZ")
            n_atomic_ref = sum(1 for v in ref_types.values() if v == "ATOMIC")
            print(f"    refs: {len(ref_series)} ({n_quartz_ref} quartz, {n_atomic_ref} atomic)")

            # Build the TEP context at the target position
            # Use a cutoff at noon (43200s) so we have pre-outage history
            kp_none = dyn.fixed_tep_context_from_cors(
                ref_series, tgt_xyz, cutoff_sod=43200.0,
                rover_clock_type="QUARTZ", screening_mode="none")
            kp_quartz = dyn.fixed_tep_context_from_cors(
                ref_series, tgt_xyz, cutoff_sod=43200.0,
                rover_clock_type="QUARTZ", screening_mode="quartz_only")
            kp_scaled = dyn.fixed_tep_context_from_cors(
                ref_series, tgt_xyz, cutoff_sod=43200.0,
                rover_clock_type="QUARTZ", screening_mode="screening_scaled")
            kp_weighted = dyn.fixed_tep_context_from_cors(
                ref_series, tgt_xyz, cutoff_sod=43200.0,
                rover_clock_type="QUARTZ", screening_mode="type_weighted")

            kps = {"none": kp_none, "quartz_only": kp_quartz,
                   "screening_scaled": kp_scaled, "type_weighted": kp_weighted}
            kps = {k: v for k, v in kps.items() if v is not None}
            if not kps:
                print(f"    skip: no valid kernel context")
                continue
            print(f"    kernel contexts: {list(kps.keys())}")

            # Target clock truth (from full-sky SPP solve)
            t_tgt = np.asarray(tgt_series["t"], float)
            y_tgt = np.asarray(tgt_series["raw"], float) * dyn.C_M_PER_NS

            # For each outage duration, predict the clock at the end
            # Anchor at the outage start, predict at outage start + duration
            anchor_sod = 43200.0  # noon
            target_result = {
                "station": tgt_name,
                "clock_type": "QUARTZ",
                "n_refs": len(ref_series),
                "n_quartz_refs": n_quartz_ref,
                "n_atomic_refs": n_atomic_ref,
                "kernel_modes": list(kps.keys()),
                "outages": {},
            }

            for dur in OUTAGE_DURATIONS:
                query_sod = anchor_sod + dur
                if query_sod > t_tgt[-1]:
                    continue

                # Truth: actual clock at query time
                truth_clk = float(np.interp(query_sod, t_tgt, y_tgt))
                # Anchor clock
                anchor_clk = float(np.interp(anchor_sod, t_tgt, y_tgt))
                # Temporal extrapolation (linear drift from pre-outage)
                pre = (t_tgt <= anchor_sod) & (t_tgt >= anchor_sod - 3600)
                if pre.sum() < 8:
                    continue
                q = t_tgt[pre] - anchor_sod
                fit = np.polyfit(q, y_tgt[pre], 1)
                temporal_pred = float(np.polyval(fit, dur))
                temporal_error = abs(temporal_pred - truth_clk)

                # TEP network prediction for each mode
                mode_errors = {}
                mode_deltas = {}
                for mode, kp in kps.items():
                    net = dyn.causal_tep_delta(kp, anchor_sod, query_sod,
                                               prewindow_s=3600.0)
                    if net is None:
                        mode_errors[mode] = None
                        mode_deltas[mode] = None
                        continue
                    nd, ns = net
                    mode_deltas[mode] = nd
                    tep_pred = temporal_pred + nd
                    mode_errors[mode] = abs(tep_pred - truth_clk)

                target_result["outages"][str(dur)] = {
                    "duration_s": dur,
                    "anchor_clk_m": anchor_clk,
                    "truth_clk_m": truth_clk,
                    "temporal_pred_m": temporal_pred,
                    "temporal_error_m": temporal_error,
                    "mode_errors_m": mode_errors,
                    "mode_deltas_m": mode_deltas,
                }
                # Print summary
                te = temporal_error
                best_mode = min(
                    [(m, e) for m, e in mode_errors.items() if e is not None],
                    key=lambda x: x[1], default=(None, None))
                print(f"    {dur}s: temporal={te:.2f}m, "
                      f"best={best_mode[0]}({best_mode[1]:.2f}m), "
                      f"deltas: " + " ".join(
                          f"{m}={d:.2f}" for m, d in mode_deltas.items()
                          if d is not None))

            results.append(target_result)

    # Aggregate: for each outage duration, compute median error by mode
    print("\n=== Aggregate results ===")
    aggregate = {}
    for dur in OUTAGE_DURATIONS:
        dur_s = str(dur)
        mode_errs = {"temporal": [], "none": [], "quartz_only": [],
                     "screening_scaled": [], "type_weighted": []}
        for r in results:
            o = r["outages"].get(dur_s)
            if o is None:
                continue
            mode_errs["temporal"].append(o["temporal_error_m"])
            for mode in ["none", "quartz_only", "screening_scaled", "type_weighted"]:
                e = o["mode_errors_m"].get(mode)
                if e is not None:
                    mode_errs[mode].append(e)
        agg = {}
        for mode, errs in mode_errs.items():
            if errs:
                arr = np.array(errs)
                agg[mode] = {
                    "n": len(arr),
                    "median_m": float(np.median(arr)),
                    "p95_m": float(np.percentile(arr, 95)),
                }
        # compute improvement vs temporal
        if "temporal" in agg and "none" in agg:
            agg["none"]["improvement_vs_temporal_pct"] = (
                (agg["temporal"]["median_m"] - agg["none"]["median_m"])
                / agg["temporal"]["median_m"] * 100)
        if "temporal" in agg and "screening_scaled" in agg:
            agg["screening_scaled"]["improvement_vs_temporal_pct"] = (
                (agg["temporal"]["median_m"] - agg["screening_scaled"]["median_m"])
                / agg["temporal"]["median_m"] * 100)
        aggregate[dur_s] = agg
        print(f"  {dur}s ({dur/60:.0f}min):")
        for mode in ["temporal", "none", "quartz_only", "screening_scaled", "type_weighted"]:
            if mode in agg:
                a = agg[mode]
                imp = a.get("improvement_vs_temporal_pct", 0)
                print(f"    {mode:20s}: median={a['median_m']:.3f}m, "
                      f"p95={a['p95_m']:.3f}m, n={a['n']}, "
                      f"improvement={imp:+.1f}%")

    output = {
        "step": 70,
        "title": "Clock-type-aware TEP outage test on quartz CORS receivers",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date_tag": "2024100",
        "n_targets": len(results),
        "outage_durations_s": OUTAGE_DURATIONS,
        "screening_factor": SCREENING_FACTOR,
        "step65_reference": "quartz LF / atomic LF = 3.78x",
        "targets": results,
        "aggregate": aggregate,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
