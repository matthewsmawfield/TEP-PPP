#!/usr/bin/env python3
"""Parse temperature stabilization from IGS site logs and correlate with
clock LF amplitude to test the TEP-UCD environmental screening hypothesis.

TEP-UCD (Paper 6) defines the environmental screening operator:

    Sigma_obs = S_Sigma(E) * grad ln A(phi)

where the environmental state E includes temperature T, boundary geometry,
and coherence volume — i.e. the clock *housing*, not just the atomic
transition.  If the housing matters, then:

1. The "clock-type" screening factor (3.78x from Step 65) is a confounded
   measurement of clock-type + average housing quality.
2. Co-located pairs with mismatched housing should show anomalous screening
   (e.g. TLSE/TLSG where the cesium is in a worse thermal environment).
3. Across the network, LF amplitude should correlate with temperature
   stabilization quality, not just clock type.

This script tests prediction (3).
"""
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_LOG_URL = "http://garner.ucsd.edu/pub/docs/site_logs/{name}.log"
CT_FILE = PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"
OUT_FILE = PROJECT_ROOT / "data" / "meta" / "housing_screening.json"

# Step 65 output has per-station LF/MF/HF amplitudes
STEP65_FILE = PROJECT_ROOT / "results" / "outputs" / "step_65b_drift_decomposition.json"


def download_site_log(name: str) -> str | None:
    import urllib.request
    url = SITE_LOG_URL.format(name=name.lower())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def parse_temp_stabilization(log_text: str) -> dict | None:
    """Parse the Temperature Stabilization field from section 3 (Receiver).

    Returns {'value': float_or_None, 'tolerance': float_or_None, 'raw': str}
    where value is the setpoint in Celsius and tolerance is the ± band.
    'none' means explicitly no temperature control.
    """
    lines = log_text.split("\n")
    # Find the most recent (last) receiver entry's temperature stabilization
    last_temp = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s+Temperature Stabiliz\.\s*:\s*(.*)", line)
        if m:
            raw = m.group(1).strip()
            # Skip template placeholders and empty values
            if raw and not raw.startswith("(none") and raw != "(deg C) +/- (deg C)":
                last_temp = raw

    if last_temp is None:
        return None

    result = {"raw": last_temp, "value": None, "tolerance": None}

    if last_temp.lower() == "none":
        result["value"] = None
        result["tolerance"] = None
        result["controlled"] = False
        return result

    # Parse patterns like "21 +/- 3 C", "18.0 +/- 2.5", "1 K", "+/- 1 C"
    # Setpoint + tolerance
    m = re.match(r"(\d+\.?\d*)\s*\+/?-\s*(\d+\.?\d*)", last_temp)
    if m:
        result["value"] = float(m.group(1))
        result["tolerance"] = float(m.group(2))
        result["controlled"] = True
        return result

    # Tolerance only: "+/- 1 C"
    m = re.match(r"\+/?-\s*(\d+\.?\d*)", last_temp)
    if m:
        result["tolerance"] = float(m.group(1))
        result["controlled"] = True
        return result

    # Just a number
    m = re.match(r"(\d+\.?\d*)", last_temp)
    if m:
        result["value"] = float(m.group(1))
        result["controlled"] = True
        return result

    return result


def parse_receiver_model(log_text: str) -> str | None:
    """Get the most recent receiver model."""
    lines = log_text.split("\n")
    last_rx = None
    for line in lines:
        m = re.match(r"^\d+\.\d+\s+Receiver Type\s*:\s*(.+)", line)
        if m:
            val = m.group(1).strip()
            if val and not val.startswith("("):
                last_rx = val
    return last_rx


def main() -> int:
    ct = json.load(open(CT_FILE))
    stations = ct["stations"]

    # Load Step 65 per-station amplitudes if available
    step65 = {}
    if STEP65_FILE.exists():
        s65 = json.load(open(STEP65_FILE))
        for st in s65.get("stations", []):
            step65[st["name"]] = st

    print(f"Parsing temperature stabilization for {len(stations)} stations...")

    results = {}
    n_controlled = 0
    n_none = 0
    n_unknown = 0

    for name, info in stations.items():
        log_text = download_site_log(name)
        if log_text is None or len(log_text) < 100:
            results[name] = {
                "clock_type": info.get("corrected_type"),
                "temp_raw": None,
                "temp_controlled": None,
                "temp_tolerance": None,
                "receiver": None,
            }
            n_unknown += 1
            continue

        temp = parse_temp_stabilization(log_text)
        rx = parse_receiver_model(log_text)

        entry = {
            "clock_type": info.get("corrected_type"),
            "standard_type": info.get("standard_type"),
            "temp_raw": temp["raw"] if temp else None,
            "temp_controlled": temp.get("controlled") if temp else None,
            "temp_tolerance": temp.get("tolerance") if temp else None,
            "temp_setpoint": temp.get("value") if temp else None,
            "receiver": rx,
        }

        # Add Step 65 amplitudes if available
        if name in step65:
            s = step65[name]
            entry["lf_amplitude_m"] = s.get("lf_amplitude_m")
            entry["mf_amplitude_m"] = s.get("mf_amplitude_m")
            entry["hf_amplitude_m"] = s.get("hf_amplitude_m")
            entry["lf_rms_m"] = s.get("lf_rms_m")

        results[name] = entry

        if temp:
            if temp.get("controlled") == False:
                n_none += 1
            elif temp.get("controlled") == True:
                n_controlled += 1
        else:
            n_unknown += 1

        time.sleep(0.2)

    print(f"  Temperature controlled: {n_controlled}")
    print(f"  No control (none): {n_none}")
    print(f"  Unknown: {n_unknown}")

    # Analysis: correlate temperature tolerance with LF amplitude
    print("\n=== Housing vs LF amplitude ===")
    print(f"{'Station':>6s} {'Type':>7s} {'Temp':>10s} {'Tol':>6s} {'LF_m':>8s} {'MF_m':>8s}")
    quartz_with_temp = []
    atomic_with_temp = []
    for name, r in sorted(results.items()):
        if r.get("lf_amplitude_m") is None:
            continue
        tol = r.get("temp_tolerance")
        ctrl = r.get("temp_controlled")
        if ctrl is False or (ctrl is None and r.get("temp_raw") is None):
            temp_str = "none"
            tol_val = 99.0  # worst
        elif tol is not None:
            temp_str = f"±{tol}"
            tol_val = tol
        else:
            temp_str = "?"
            tol_val = None

        lf = r["lf_amplitude_m"]
        mf = r.get("mf_amplitude_m", float("nan"))
        ct = r["clock_type"]
        print(f"{name:>6s} {ct:>7s} {temp_str:>10s} {str(tol):>6s} {lf:>8.3f} {mf:>8.3f}")

        if tol_val is not None:
            if ct == "QUARTZ":
                quartz_with_temp.append((name, tol_val, lf))
            elif ct == "ATOMIC":
                atomic_with_temp.append((name, tol_val, lf))

    # Correlation within quartz group
    if len(quartz_with_temp) >= 5:
        tols = np.array([x[1] for x in quartz_with_temp])
        lfs = np.array([x[2] for x in quartz_with_temp])
        r_q = np.corrcoef(tols, lfs)[0, 1]
        print(f"\nQuartz: r(temp_tol, LF) = {r_q:.3f}, n={len(quartz_with_temp)}")
        # Spearman
        from scipy.stats import spearmanr
        rho_q, p_q = spearmanr(tols, lfs)
        print(f"Quartz: rho(temp_tol, LF) = {rho_q:.3f}, p={p_q:.4f}")

    if len(atomic_with_temp) >= 5:
        tols = np.array([x[1] for x in atomic_with_temp])
        lfs = np.array([x[2] for x in atomic_with_temp])
        r_a = np.corrcoef(tols, lfs)[0, 1]
        print(f"Atomic: r(temp_tol, LF) = {r_a:.3f}, n={len(atomic_with_temp)}")
        from scipy.stats import spearmanr
        rho_a, p_a = spearmanr(tols, lfs)
        print(f"Atomic: rho(temp_tol, LF) = {rho_a:.3f}, p={p_a:.4f}")

    # Combined
    all_with_temp = quartz_with_temp + atomic_with_temp
    if len(all_with_temp) >= 10:
        tols = np.array([x[1] for x in all_with_temp])
        lfs = np.array([x[2] for x in all_with_temp])
        types = np.array([0 if n in [x[0] for x in quartz_with_temp] else 1
                          for n, _, _ in all_with_temp])
        r_all = np.corrcoef(tols, lfs)[0, 1]
        print(f"All:    r(temp_tol, LF) = {r_all:.3f}, n={len(all_with_temp)}")
        # Partial correlation: temp_tol given clock_type
        # r(x,y|z) = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
        r_tz = np.corrcoef(tols, types)[0, 1]
        r_lz = np.corrcoef(lfs, types)[0, 1]
        if abs(r_tz) < 0.999 and abs(r_lz) < 0.999:
            pc = (r_all - r_tz * r_lz) / np.sqrt((1 - r_tz**2) * (1 - r_lz**2))
            print(f"All:    partial r(temp_tol, LF | clock_type) = {pc:.3f}")
        from scipy.stats import spearmanr
        rho_all, p_all = spearmanr(tols, lfs)
        print(f"All:    rho(temp_tol, LF) = {rho_all:.3f}, p={p_all:.4f}")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_stations": len(results),
        "n_controlled": n_controlled,
        "n_none": n_none,
        "n_unknown": n_unknown,
        "stations": results,
    }, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
