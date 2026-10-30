#!/usr/bin/env python3
"""
STEP 52 -- Real-world drone / high-dynamics kinematic test.

Targets:
  - OpenDroneMap / OpenTopography UAV PPK logs (UBX or pre-converted RINEX)
  - Other high-dynamics kinematic rovers

Pipeline
--------
1. Read datasets from ``data/dynamic_datasets/manifest.json``.
2. Convert rover logs (UBX / pre-converted RINEX) to RINEX 3.
3. Process regional CORS stations as the static reference network.
4. Build a causal CORS clock prior with the independently registered TEP
   distance scale (no fitting on the flight day).
5. Run a raw-observation SPP joint solve for each epoch:
     - free-clock (conventional) solve
     - TEP-aided solve with a pre-outage frozen, causally updated clock prior
     - conventional RTKLIB SPP fix, when available
6. Compare each 3-D solution to the ground truth and report the three
   operational metrics:
     - dead-reckoning bridge rate
     - cross-track error
     - time to re-convergence after blackouts

Output: results/outputs/step_52_drone_kinematic.json
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_dynamic as dyn  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
OUT_DIR = OUT
OUT_FILE = OUT / "step_52_drone_kinematic.json"

DRONE_TYPES = ("drone", "uav", "ubx_drone", "opendronemap", "ppk")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc).isoformat()

    manifest = Path(os.getenv("TEP_DYNAMIC_MANIFEST",
                              PROJECT_ROOT / "data" / "dynamic_datasets" / "manifest.json"))
    datasets = dyn.load_manifest(manifest)
    if not datasets:
        OUT_FILE.write_text(json.dumps({
            "step": "step_52_drone_kinematic",
            "timestamp": t0,
            "status": "no_manifest",
            "note": (f"Create {manifest} with at least one drone "
                     "dataset to run this step.")
        }, indent=2))
        print(f"no manifest or no datasets; wrote {OUT_FILE}")
        return 0

    selected = [ds for ds in datasets if ds.type.lower() in DRONE_TYPES]
    if not selected:
        selected = datasets

    results = []
    with tempfile.TemporaryDirectory(prefix="step_52_") as td:
        for ds in selected:
            print(f"\n=== step_52: {ds.name} ({ds.type}) ===")
            # GPS-only is deliberate.  The present raw solver has a single
            # receiver-clock state; mixing GPS/Galileo/BeiDou without explicit
            # inter-system bias states would confound those biases with the
            # TEP clock prior and is not admissible evidence.
            res = dyn.evaluate_dataset_v2(
                ds, Path(td), systems=("G",), screening_modes=("none",))
            results.append(res)
            status = res.get("status", "unknown")
            summary = res.get("summary", {})
            if status == "ok":
                print(f"  epochs: {summary.get('n_epochs')}, "
                      f"CORS: {summary.get('n_cors')}, "
                      f"systems: {summary.get('systems')}, "
                      f"jam epochs: {summary.get('n_jam_epochs')}")
                pk = summary.get("per_k", {})
                for kk, vv in sorted(pk.items()):
                    print(f"  {kk}: n={vv['n_epochs']}, "
                          f"TEP p50={vv.get('err_tep_p50')}, "
                          f"free p50={vv.get('err_free_p50')}, "
                          f"TCXO p50={vv.get('err_tcxo_p50')}, "
                          f"RTK p50={vv.get('err_rtk_p50')}")
                # per-screening-mode results
                modes = summary.get("screening_modes_tested", [])
                if len(modes) > 1:
                    print(f"  screening modes: {modes}")
                    print(f"  ref types: {summary.get('screening_ref_types')}")
                    for mode in modes:
                        if mode == summary.get("screening_primary_mode"):
                            continue
                        jp50 = summary.get(f"jam_err_{mode}_p50_m")
                        br = summary.get(f"bridge_rate_{mode}")
                        nd = summary.get(f"network_delta_{mode}_p50_m")
                        print(f"    {mode}: jam p50={jp50}, bridge={br}, "
                              f"delta p50={nd}")
            else:
                print(f"  status: {status}")

    overall = {
        "step": "step_52_drone_kinematic",
        "timestamp": t0,
        "manifest": str(manifest),
        "n_datasets": len(selected),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "registered_design": {
            "systems": ["G"],
            "reason": "single-clock GPS-only solve avoids unmodelled inter-system biases",
            "tep_lambda_km": 4200.0,
            "tep_nugget": 0.35,
            "tep_form": "gaussian",
            "test_day_hyperparameter_fit": False,
            "truth_role": "scoring only",
        },
        "datasets": results,
    }
    if any(r.get("status") == "ok" for r in results):
        ok_err = []
        for r in results:
            s = r.get("summary", {})
            if s.get("jam_err_tep_p50_m") is not None:
                ok_err.append(s["jam_err_tep_p50_m"])
        overall["median_jam_err_tep_p50_m"] = float(np.median(ok_err)) if ok_err else None

    OUT_FILE.write_text(json.dumps(overall, indent=2))
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
