#!/usr/bin/env python3
"""Step 72: Housing as a screening layer — connecting clock-type screening
to the TEP-UCD environmental screening operator S_Sigma(E).

TEP-UCD (Paper 6, Eq. screening_operator) defines:

    Sigma_obs = S_Sigma(E) * grad ln A(phi)

where the environmental state E = {rho, Phi/c^2, grad_rho, grad_Phi,
compactness, R_T(M), proximity, T, z, boundary_geometry, coherence_volume}.

The clock *housing* — temperature stabilization, vibration isolation,
electromagnetic shielding, vault depth — is part of E.  Two clocks of the
same atomic type in different housings should have different screening
factors.  This script tests that prediction by:

1. Computing per-station LF/MF/HF clock amplitudes for all stations with
   IGS site-log temperature data.
2. Correlating temperature stabilization quality with LF amplitude,
   controlling for clock type.
3. Showing that the housing effect is comparable to the clock-type effect.

The key prediction: if housing matters (as TEP-UCD says it must), then
the partial correlation of temperature tolerance with LF amplitude,
controlling for clock type, should be significant.  The "clock-type"
screening factor (3.78x from Step 65) is then a confounded measurement
of clock-type + average housing quality.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib import tep_dynamic as dyn
from scripts.lib import tep_spp as spp

SP3 = PROJECT_ROOT / "data" / "rinex" / "sp3" / "COD0MGXFIN_20241000000_01D_05M_ORB.SP3.gz"
CLK = PROJECT_ROOT / "data" / "rinex" / "clk" / "COD0MGXFIN_20241000000_01D_30S_CLK.CLK.gz"
OBS_DIR = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024" / "100"
CT_FILE = PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"
HOUSING_FILE = PROJECT_ROOT / "data" / "meta" / "housing_screening.json"
ENV_FILE = PROJECT_ROOT / "data" / "meta" / "station_environment.json"
ALT_FILE = PROJECT_ROOT / "data" / "meta" / "station_altitudes_all.json"
OUT_FILE = PROJECT_ROOT / "results" / "outputs" / "step_72_housing_screening.json"

# Stations whose single-day LF amplitude exceeds this bound are treated as
# clock-jump artefacts and excluded from correlation estimators (they are
# still reported, per the exploratory-design disclosure).
LF_OUTLIER_CLIP_M = 10.0

# Frequency bands (same as Step 65)
# LF: < 10 uHz (periods > 28h) — the "drift" / TEP field
# MF: 10-500 uHz — the current TEP band
# HF: > 500 uHz (periods < 33 min) — oscillator noise
LF_MAX_HZ = 10e-6
MF_MIN_HZ = 10e-6
MF_MAX_HZ = 500e-6
HF_MIN_HZ = 500e-6


def bandpass(x, dt, f_low, f_high):
    """Zero-phase Butterworth bandpass."""
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    nyq = fs / 2
    low = f_low / nyq if f_low > 0 else 1e-6
    high = f_high / nyq if f_high < nyq else 0.999
    if low >= high:
        return np.zeros_like(x)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x)


def lowpass(x, dt, f_high):
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    nyq = fs / 2
    high = f_high / nyq if f_high < nyq else 0.999
    b, a = butter(4, high, btype="low")
    return filtfilt(b, a, x)


def highpass(x, dt, f_low):
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    nyq = fs / 2
    low = f_low / nyq if f_low > 0 else 1e-6
    b, a = butter(4, low, btype="high")
    return filtfilt(b, a, x)


def compute_band_amplitudes(clock_m, dt):
    """Compute LF, MF, HF RMS amplitudes in metres."""
    n = len(clock_m)
    if n < 100:
        return None, None, None

    # Detrend (remove linear + quadratic)
    t = np.arange(n) * dt
    coeffs = np.polyfit(t, clock_m, 2)
    detrended = clock_m - np.polyval(coeffs, t)

    # LF: lowpass at 10 uHz
    lf = lowpass(detrended, dt, LF_MAX_HZ)
    lf_rms = float(np.sqrt(np.mean(lf**2)))

    # MF: bandpass 10-500 uHz
    mf = bandpass(detrended, dt, MF_MIN_HZ, MF_MAX_HZ)
    mf_rms = float(np.sqrt(np.mean(mf**2)))

    # HF: highpass at 500 uHz
    hf = highpass(detrended, dt, HF_MIN_HZ)
    hf_rms = float(np.sqrt(np.mean(hf**2)))

    return lf_rms, mf_rms, hf_rms


def _analyse(results: dict) -> dict:
    """Aggregate housing/environment screening analysis.

    Persisted into the output JSON so every number the manuscript quotes from
    this step is verifiable from the committed artefact.
    """
    from scipy.stats import pearsonr, spearmanr, t as tdist

    housing = json.load(open(HOUSING_FILE)) if HOUSING_FILE.exists() else {"stations": {}}
    env = json.load(open(ENV_FILE)) if ENV_FILE.exists() else {}
    alt = json.load(open(ALT_FILE)) if ALT_FILE.exists() else {}
    hst = housing.get("stations", {})

    def _med(vals):
        return float(np.median(list(vals))) if len(list(vals)) else None

    analysis: dict = {}

    # --- site-log temperature census -------------------------------------
    n_ctrl = sum(1 for v in hst.values() if v.get("temp_controlled") is True)
    n_none = sum(
        1 for v in hst.values()
        if v.get("temp_controlled") is False
        or (v.get("temp_raw") and str(v["temp_raw"]).lower() == "none")
    )
    tol_vals = sorted(v["temp_tolerance"] for v in hst.values()
                      if v.get("temp_tolerance"))
    analysis["temperature_census"] = {
        "n_site_logs": len(hst),
        "n_controlled": n_ctrl,
        "n_none": n_none,
        "n_no_usable_entry": len(hst) - n_ctrl - n_none,
        "tolerance_min_c": tol_vals[0] if tol_vals else None,
        "tolerance_max_c": tol_vals[-1] if tol_vals else None,
    }

    # --- temperature-tolerance correlations, controlling for clock type ---
    quartz = [(n, r) for n, r in results.items()
              if r["clock_type"] == "QUARTZ" and r["temp_tolerance_numeric"] is not None]
    atomic = [(n, r) for n, r in results.items()
              if r["clock_type"] == "ATOMIC" and r["temp_tolerance_numeric"] is not None]
    all_st = quartz + atomic

    def _corr_block(group, label):
        if len(group) < 5:
            return None
        tols = np.array([r["temp_tolerance_numeric"] for _, r in group])
        lfs = np.array([r["lf_rms_m"] for _, r in group])
        mfs = np.array([r["mf_rms_m"] for _, r in group])
        hfs = np.array([r["hf_rms_m"] for _, r in group])
        return {
            "n": len(group),
            "r_temp_tol_lf": pearsonr(tols, lfs),
            "rho_temp_tol_lf": spearmanr(tols, lfs),
            "r_temp_tol_mf": pearsonr(tols, mfs),
            "r_temp_tol_hf": pearsonr(tols, hfs),
        }

    temp = {
        "quartz": _corr_block(quartz, "quartz"),
        "atomic": _corr_block(atomic, "atomic"),
    }
    if len(all_st) >= 10:
        tols = np.array([r["temp_tolerance_numeric"] for _, r in all_st])
        lfs = np.array([r["lf_rms_m"] for _, r in all_st])
        types = np.array([0 if r["clock_type"] == "QUARTZ" else 1 for _, r in all_st])
        r_all, p_all = pearsonr(tols, lfs)
        rho_all, p_rho_all = spearmanr(tols, lfs)
        r_tz = np.corrcoef(tols, types)[0, 1]
        r_lz = np.corrcoef(lfs, types)[0, 1]
        pc = (r_all - r_tz * r_lz) / np.sqrt((1 - r_tz**2) * (1 - r_lz**2))
        n = len(all_st)
        t_stat = pc * np.sqrt((n - 3) / max(1 - pc**2, 1e-12))
        p_pc = float(2.0 * tdist.sf(abs(t_stat), n - 3))
        mfs = np.array([r["mf_rms_m"] for _, r in all_st])
        hfs = np.array([r["hf_rms_m"] for _, r in all_st])
        temp["combined"] = {
            "n": n,
            "r_temp_tol_lf": [float(r_all), float(p_all)],
            "rho_temp_tol_lf": [float(rho_all), float(p_rho_all)],
            "partial_r_temp_tol_lf_given_clock_type": float(pc),
            "partial_r_p_value": p_pc,
            "r_temp_tol_clock_type": float(r_tz),
            "r_lf_clock_type": float(r_lz),
            "r_temp_tol_mf": float(pearsonr(tols, mfs)[0]),
            "r_temp_tol_hf": float(pearsonr(tols, hfs)[0]),
        }

    def _q_a_ratio(sub):
        q = [r["lf_rms_m"] for r in sub if r["clock_type"] == "QUARTZ"]
        a = [r["lf_rms_m"] for r in sub if r["clock_type"] == "ATOMIC"]
        if not q or not a:
            return None
        return {
            "quartz_atomic_lf_median_ratio": float(np.median(q) / np.median(a)),
            "quartz_lf_median_m": float(np.median(q)),
            "atomic_lf_median_m": float(np.median(a)),
            "n_quartz": len(q),
            "n_atomic": len(a),
        }

    temp["quartz_atomic_ratio_controlled"] = _q_a_ratio(
        [r for _, r in all_st if r["temp_controlled"]])
    temp["quartz_atomic_ratio_uncontrolled"] = _q_a_ratio(
        [r for _, r in all_st if r["temp_controlled"] is False])
    analysis["temperature"] = temp

    # --- monument foundation (bedrock vs building) ------------------------
    # Flags are non-exclusive: a building-mounted monument on bedrock carries
    # both attributes, so the census and the LF contrast count flag
    # membership independently and report the overlap.
    env_log = {n: v for n, v in env.items() if v.get("site_log_available")}
    n_bed = sum(1 for n in env_log
                if (env_log[n].get("environment") or {}).get("on_bedrock"))
    n_bld = sum(1 for n in env_log
                if (env_log[n].get("environment") or {}).get("on_building"))
    n_both = sum(
        1 for n in env_log
        if (env_log[n].get("environment") or {}).get("on_bedrock")
        and (env_log[n].get("environment") or {}).get("on_building"))
    found = {
        "n_site_logs": len(env_log),
        "n_bedrock": n_bed,
        "n_building": n_bld,
        "n_both_flags": n_both,
        "flags_nonexclusive": True,
    }
    for ct_name in ("ATOMIC", "QUARTZ"):
        for cls, flag in (("bedrock", "on_bedrock"),
                          ("building", "on_building")):
            vals = [
                r["lf_rms_m"] for n, r in results.items()
                if r["clock_type"] == ct_name
                and (env_log.get(n) or {}).get("environment", {}).get(flag)
            ]
            found[f"{ct_name.lower()}_lf_median_{cls}_m"] = _med(vals)
            found[f"n_{ct_name.lower()}_{cls}"] = len(vals)
    analysis["foundation"] = found

    # --- altitude ----------------------------------------------------------
    rows = []
    for n, r in results.items():
        a = alt.get(n)
        if a is None:
            try:
                a = float((env.get(n) or {}).get("parsed", {}).get("elevation_m"))
            except (TypeError, ValueError):
                continue
        rows.append((n, float(r["lf_rms_m"]), float(a), r["clock_type"]))
    if rows:
        alts_a = np.array([r[2] for r in rows])
        lfs_a = np.array([r[1] for r in rows])
        ct_a = np.array([r[3] for r in rows])
        ok = lfs_a < LF_OUTLIER_CLIP_M
        am = ct_a == "ATOMIC"
        analysis["altitude"] = {
            "n_stations": len(rows),
            "range_m": [float(alts_a.min()), float(alts_a.max())],
            "outlier_clip_lf_m": LF_OUTLIER_CLIP_M,
            "clipped_outliers": {
                n: float(l) for n, l, a, c in rows if l >= LF_OUTLIER_CLIP_M
            },
            "pearson_all_with_outliers": list(pearsonr(alts_a, lfs_a)),
            "spearman_all": list(spearmanr(alts_a, lfs_a)),
            "pearson_all_clipped": list(pearsonr(alts_a[ok], lfs_a[ok])),
            "n_clipped": int(ok.sum()),
            "pearson_atomic_clipped": list(pearsonr(alts_a[am & ok], lfs_a[am & ok])),
            "n_atomic_clipped": int((am & ok).sum()),
        }

    # --- national-metrology co-location -----------------------------------
    met = [n for n, v in env.items()
           if any("metrology" in (x or "").lower()
                  for x in (v.get("environment") or {}).get("environment_notes") or [])]
    met_atomic = [r["lf_rms_m"] for n, r in results.items()
                  if n in met and r["clock_type"] == "ATOMIC"]
    oth_atomic = [r["lf_rms_m"] for n, r in results.items()
                  if n not in met and r["clock_type"] == "ATOMIC"]
    analysis["metrology"] = {
        "flagged_stations": sorted(met),
        "n_flagged": len(met),
        "atomic_in_analysis": sorted(n for n, r in results.items()
                                     if n in met and r["clock_type"] == "ATOMIC"),
        "metrology_atomic_lf_median_m": _med(met_atomic),
        "other_atomic_lf_median_m": _med(oth_atomic),
        "ratio": (float(np.median(met_atomic) / np.median(oth_atomic))
                  if met_atomic and oth_atomic else None),
        "n_metrology_atomic": len(met_atomic),
        "n_other_atomic": len(oth_atomic),
    }

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="reuse the committed per-station table and only re-run the "
             "aggregate analysis block")
    args = parser.parse_args()

    print("=== step_72: housing as a screening layer (TEP-UCD S_Sigma(E)) ===")

    if args.analysis_only:
        results = json.load(open(OUT_FILE))["stations"]
        analysis = _analyse(results)
        doc = json.load(open(OUT_FILE))
        doc["stations"] = results
        doc["analysis"] = analysis
        json.dump(doc, open(OUT_FILE, "w"), indent=2)
        print(f"analysis persisted to {OUT_FILE}")
        return 0

    ct = json.load(open(CT_FILE))
    housing = json.load(open(HOUSING_FILE)) if HOUSING_FILE.exists() else {"stations": {}}

    # Select stations with meaningful temperature data
    stations_to_process = []
    for name, info in ct["stations"].items():
        h = housing.get("stations", {}).get(name, {})
        temp_raw = h.get("temp_raw")
        temp_ctrl = h.get("temp_controlled")
        # Only process stations with clear temperature info
        if temp_ctrl is not None or (temp_raw and temp_raw.lower() == "none"):
            stations_to_process.append(name)

    print(f"Stations with temperature data: {len(stations_to_process)}")

    # Build clock series and compute band amplitudes
    results = {}
    tmp_dir = PROJECT_ROOT / "tmp" / "step72"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(stations_to_process):
        h = housing["stations"][name]
        clock_type = ct["stations"][name].get("corrected_type", "UNKNOWN")

        # Find RINEX file
        rnx_path = None
        for f in OBS_DIR.iterdir():
            if f.name.startswith(name) and f.name.endswith(".crx.gz"):
                rnx_path = f
                break
        if rnx_path is None:
            continue

        # Build clock series
        out_dir = tmp_dir / name
        out_dir.mkdir(exist_ok=True)
        rnx_file = dyn._decompress_crx(rnx_path, out_dir)
        if rnx_file is None:
            continue
        hdr = dyn._rinex_header_xyz(rnx_file)
        if hdr is None:
            continue

        series = dyn.build_clock_series_precise_multi(
            rnx_file, SP3, CLK, hdr, name, systems=("G",))
        if series is None:
            continue

        # Convert to metres and unwrap 1ms steering
        raw_ns = np.asarray(series["raw"], float)
        unwrapped = np.unwrap(raw_ns * (2 * np.pi / 1e6)) * 1e6 / (2 * np.pi)
        clock_m = unwrapped * dyn.C_M_PER_NS

        # Time step
        t = np.asarray(series["t"], float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 30.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0

        lf, mf, hf = compute_band_amplitudes(clock_m, dt)
        if lf is None:
            continue

        # Temperature tolerance (lower = better housing)
        temp_ctrl = h.get("temp_controlled")
        temp_tol = h.get("temp_tolerance")
        if temp_ctrl is False:
            temp_tol_numeric = 99.0  # "none" = worst
        elif temp_tol is not None:
            temp_tol_numeric = float(temp_tol)
        else:
            temp_tol_numeric = None  # unknown

        results[name] = {
            "clock_type": clock_type,
            "standard_type": ct["stations"][name].get("standard_type"),
            "temp_raw": h.get("temp_raw"),
            "temp_controlled": temp_ctrl,
            "temp_tolerance": temp_tol,
            "temp_tolerance_numeric": temp_tol_numeric,
            "receiver": h.get("receiver"),
            "lf_rms_m": lf,
            "mf_rms_m": mf,
            "hf_rms_m": hf,
            "n_epochs": len(t),
        }

        temp_str = f"±{temp_tol}°C" if temp_tol else ("none" if temp_ctrl is False else "?")
        print(f"  {name:5s} ({clock_type:7s}): LF={lf:.3f}m, MF={mf:.3f}m, "
              f"HF={hf:.3f}m, temp={temp_str}")

    # Clean up
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Analysis
    print(f"\n=== Correlation analysis ===")
    quartz = [(n, r) for n, r in results.items()
              if r["clock_type"] == "QUARTZ" and r["temp_tolerance_numeric"] is not None]
    atomic = [(n, r) for n, r in results.items()
              if r["clock_type"] == "ATOMIC" and r["temp_tolerance_numeric"] is not None]
    all_st = quartz + atomic

    from scipy.stats import spearmanr, pearsonr

    def analyze(group, label):
        if len(group) < 5:
            print(f"\n{label}: n={len(group)} (too few for correlation)")
            return
        tols = np.array([r["temp_tolerance_numeric"] for _, r in group])
        lfs = np.array([r["lf_rms_m"] for _, r in group])
        mfs = np.array([r["mf_rms_m"] for _, r in group])
        hfs = np.array([r["hf_rms_m"] for _, r in group])

        r_lf, p_lf = pearsonr(tols, lfs)
        rho_lf, p_rho_lf = spearmanr(tols, lfs)
        r_mf, p_mf = pearsonr(tols, mfs)
        r_hf, p_hf = pearsonr(tols, hfs)

        print(f"\n{label} (n={len(group)}):")
        print(f"  r(temp_tol, LF) = {r_lf:.3f} (p={p_lf:.4f})")
        print(f"  rho(temp_tol, LF) = {rho_lf:.3f} (p={p_rho_lf:.4f})")
        print(f"  r(temp_tol, MF) = {r_mf:.3f} (p={p_mf:.4f})")
        print(f"  r(temp_tol, HF) = {r_hf:.3f} (p={p_hf:.4f})")

        # Print the data
        print(f"  {'Station':>6s} {'Tol':>6s} {'LF_m':>8s} {'MF_m':>8s} {'HF_m':>8s}")
        for name, r in sorted(group, key=lambda x: x[1]["temp_tolerance_numeric"]):
            print(f"  {name:>6s} {r['temp_tolerance_numeric']:>6.1f} "
                  f"{r['lf_rms_m']:>8.3f} {r['mf_rms_m']:>8.3f} {r['hf_rms_m']:>8.3f}")

    analyze(quartz, "QUARTZ")
    analyze(atomic, "ATOMIC")

    # Combined with partial correlation
    if len(all_st) >= 10:
        tols = np.array([r["temp_tolerance_numeric"] for _, r in all_st])
        lfs = np.array([r["lf_rms_m"] for _, r in all_st])
        types = np.array([0 if r["clock_type"] == "QUARTZ" else 1 for _, r in all_st])

        r_all, p_all = pearsonr(tols, lfs)
        rho_all, p_rho_all = spearmanr(tols, lfs)
        print(f"\nAll (n={len(all_st)}):")
        print(f"  r(temp_tol, LF) = {r_all:.3f} (p={p_all:.4f})")
        print(f"  rho(temp_tol, LF) = {rho_all:.3f} (p={p_rho_all:.4f})")

        # Partial correlation: temp_tol given clock_type
        r_tz = np.corrcoef(tols, types)[0, 1]
        r_lz = np.corrcoef(lfs, types)[0, 1]
        if abs(r_tz) < 0.999 and abs(r_lz) < 0.999:
            pc = (r_all - r_tz * r_lz) / np.sqrt((1 - r_tz**2) * (1 - r_lz**2))
            print(f"  partial r(temp_tol, LF | clock_type) = {pc:.3f}")
            print(f"  r(temp_tol, clock_type) = {r_tz:.3f}")
            print(f"  r(LF, clock_type) = {r_lz:.3f}")

        # Also check MF and HF
        mfs = np.array([r["mf_rms_m"] for _, r in all_st])
        hfs = np.array([r["hf_rms_m"] for _, r in all_st])
        r_mf, _ = pearsonr(tols, mfs)
        r_hf, _ = pearsonr(tols, hfs)
        print(f"  r(temp_tol, MF) = {r_mf:.3f}")
        print(f"  r(temp_tol, HF) = {r_hf:.3f}")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "step": 72,
        "title": "Housing as a screening layer",
        "n_stations": len(results),
        "stations": results,
        "analysis": _analyse(results),
        "tep_ucd_connection": {
            "operator": "S_Sigma(E)",
            "environmental_state": "E = {rho, Phi/c^2, grad_rho, grad_Phi, compactness, R_T(M), proximity, T, z, boundary_geometry, coherence_volume}",
            "housing_components": ["T (temperature)", "boundary_geometry (enclosure)", "coherence_volume"],
            "prediction": "Housing quality (temperature stabilization) should correlate with LF amplitude, controlling for clock type",
        },
    }, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
