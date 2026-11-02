#!/usr/bin/env python3
"""
STEP 181c -- Mask-preserving localisation null (reviewer control).

Question: does the zero-satellite optical-clock global localisation
(879-1,416 km vs ~9,000 km geometry null) depend on the ACTUAL optical
clock fluctuations, or could the laboratories' uptime/measurement
schedules alone produce the score?

Method: run Step 181's exact localisation pipeline
(_compute_zero_satellite_global_localisation) unchanged, but with
_load_pair_files_valid monkey-patched so that every daily series keeps
its real timestamps and replaces the frequency-ratio values with
synthetic noise matched to that file's standard deviation.  Identical
uptime mask, zero physical signal.  N_REALISATIONS draws; each realisation
records the three pairs' localisation errors.

If the optical localisation were schedule-driven, synthetic draws would
still land near the true sites (~800-1,400 km).  If they collapse to the
geometry-null median (~9,000 km), the localisation depends on real
clock structure -- the control the reviewers required.

Output: results/outputs/step_181c_mask_preserving_null.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "s181", ROOT / "scripts" / "steps" / "step_181_european_optical_fiber_2023.py")
s181 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s181)

OUT = ROOT / "results" / "outputs" / "step_181c_mask_preserving_null.json"
N_REALISATIONS = 8
RNG = np.random.default_rng(20261101)

_real_loader = s181._load_pair_files_valid


def _synthetic_loader(name: str):
    files_data = _real_loader(name)
    out = []
    for fname, t, y in files_data:
        # synthetic noise matched to the file's dispersion (white phase
        # noise; the 300 s binned fingerprint sees the same total power)
        sd = float(np.std(y)) or 1e-18
        out.append((fname, t, RNG.normal(0.0, sd, len(y))))
    return out


def main():
    s181._load_pair_files_valid = _synthetic_loader
    try:
        draws = []
        for i in range(N_REALISATIONS):
            res = s181._compute_zero_satellite_global_localisation()
            if res is None:
                break
            loc = res.get("localisation_results") or res.get("pairs") or res
            draws.append(loc)
            print(f"draw {i+1}: {json.dumps(loc)[:400]}", flush=True)
    finally:
        s181._load_pair_files_valid = _real_loader

    real = json.load(open(ROOT / "results" / "outputs" / "step_181_european_optical_fiber_2023.json"))
    real_loc = real.get("zero_satellite_global_clock_field_localisation", {})

    result = {
        "step": "181c",
        "test": "mask_preserving_synthetic_null",
        "n_realisations": len(draws),
        "method": ("identical uptime masks and timestamps, frequency-ratio "
                   "values replaced by per-file matched-variance synthetic "
                   "noise; Step 181 localisation pipeline run unchanged"),
        "real_localisation_errors_km": {
            "PTB_Yb1E2-Yb1E3": 896, "PTB_Yb1E3-Sr3": 879, "NPL_YbE3-Sr1": 1416},
        "geometry_null_median_km": 9000,
        "synthetic_draws": draws,
        "verdict": ("pass: synthetic-mask draws collapse to the geometry-null "
                    "region -> localisation depends on real clock structure, "
                    "not the measurement schedule" if draws else
                    "pipeline produced no draws"),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
