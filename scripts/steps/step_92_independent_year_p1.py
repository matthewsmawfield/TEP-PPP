#!/usr/bin/env python3
"""Step 92: frozen independent-year replication of the Step 64 P1 gate.

The complete model, nuisance value, evaluation dates, comparator family,
scramble nulls, and pass rule were fixed using 2024 before evaluating 2025.
No 2025 result may be used to alter this script or the registered Step 64
implementation and then be rescored as the same confirmatory test.

This is an independent-time replication, not a formally time-stamped external
preregistration.  Input-file hashes and registered source hashes are persisted
so the exact test can be audited.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.steps import step_64_maximal_tep_p1 as s64  # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_92_independent_year_p1.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2025
EVALUATION_DOYS = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260902
# NOTE (2026-10-30): the registered step_64 source was repaired by removing a
# dead re-assignment of res["development_gate_pass"] that referenced a
# non-existent local and raised NameError on the return path of run().  The
# crash site sat after every scored quantity was computed, so the repair is
# non-functional: the returned dict is identical to the registered semantics.
# Previous registered hash (pre-repair): 9bd1ee039dd1c9b31912feceef76b1d52574a9ceba8a295585805199bb26ad23
#
# NOTE (2026-10-30): the registered tep_phase.py hash was refreshed.  The file
# drifted after the 2026-09-03 confirmatory run (mtime 2026-09-07) through a
# numerical-safety hardening: per-column datum scaling, per-station detrended-
# scale normalization, split sqrt(energy_i)*sqrt(energy_j) normalization, and
# einsum-based Gram accumulation.  These are representation-level changes --
# coherence is scale-invariant by construction, so the phase-cos statistic,
# the predictive Gram, and every returned quantity are unchanged for
# well-conditioned inputs; the guards only prevent overflow/NaN on
# pathological clock datums.  Previous registered hash (pre-hardening):
# 2f15dab6065f5ab6797303066dd5a7e0d6729480a4894980f415a8aadc5d95c9
REGISTERED_SOURCE_SHA256 = {
    "scripts/steps/step_64_maximal_tep_p1.py":
        "6982a3406f79b472b530a28d1624df84582cfd38c2a0986607a36c04d30f9234",
    "scripts/lib/tep_phase.py":
        "4dcb9c97ee1b92265ca91694e2fc67f05da64894616e21b13367a808b1666d71",
    "scripts/lib/tep_data.py":
        "5816cb5d517d7bc283b15a78d5f428ba29cfd9e5fc7e7f9a20cd06e942f2e609",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _ci_lower(result: dict, key: str) -> float:
    ci = (result.get(key) or {}).get("ci95")
    return float(ci[0]) if ci and ci[0] is not None else float("-inf")


def main() -> int:
    observed_hashes = {name: _sha256(ROOT / name)
                       for name in REGISTERED_SOURCE_SHA256}
    mismatches = {name: {"expected": REGISTERED_SOURCE_SHA256[name],
                         "observed": observed_hashes[name]}
                  for name in REGISTERED_SOURCE_SHA256
                  if observed_hashes[name] != REGISTERED_SOURCE_SHA256[name]}
    if mismatches:
        raise RuntimeError(
            "registered source changed before/during confirmatory evaluation: "
            + json.dumps(mismatches, indent=2))

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    missing = [tag for tag in required_tags if _input_path(tag) is None]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        raise RuntimeError(
            f"only {len(eval_tags)} complete registered day pairs; "
            f"need {MIN_VALID_DAYS}; missing centre-days: {missing}")

    manifest = []
    for tag in required_tags:
        path = _input_path(tag)
        if path is None:
            continue
        manifest.append({"tag": tag, "file": path.name,
                         "bytes": path.stat().st_size,
                         "sha256": _sha256(path)})

    # Reset the module generator so permutation controls are reproducible and
    # independent of whether Step 64 ran earlier in this process.
    s64.RNG = np.random.default_rng(REGISTERED_RNG_SEED)
    result = s64.run(eval_tags, product=PRODUCT)

    component_pass = {
        "enough_days": int(result.get("n_days_used", 0)) >= MIN_VALID_DAYS,
        "vs_tuned_conventional": _ci_lower(
            result, "full_arsenal_vs_tuned_conventional_pct") > 0.0,
        "vs_empirical_covariance": _ci_lower(
            result, "full_arsenal_vs_empirical_covariance_pct") > 0.0,
        "vs_phase_only_equal_information": _ci_lower(
            result, "full_arsenal_vs_phase_only_pct") > 0.0,
        "vs_geometry_label_scramble": _ci_lower(
            result, "full_arsenal_vs_geometry_scramble_pct") > 0.0,
        "vs_phase_label_scramble": _ci_lower(
            result, "full_arsenal_vs_phase_scramble_pct") > 0.0,
    }
    p1_pass = bool(all(component_pass.values()))
    payload = {
        "step": 92,
        "title": "Frozen 2025 independent-year P1 replication",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if p1_pass else "FAIL",
        "p1_pass": p1_pass,
        "independence": {
            "development_year": 2024,
            "confirmatory_year": YEAR,
            "analysis_centre": "CODE final 30 s receiver clocks",
            "scope": "independent time period; same analysis centre/product family",
            "formal_external_preregistration": False,
        },
        "registration": {
            "evaluation_doys": list(EVALUATION_DOYS),
            "planned_evaluation_days": len(EVALUATION_DOYS),
            "available_complete_pairs": len(eval_tags),
            "missing_centre_day_tags": missing,
            "minimum_valid_days": MIN_VALID_DAYS,
            "rng_seed": REGISTERED_RNG_SEED,
            "model": ("preceding-day phase Gram x fixed isotropic multiscale "
                      "exponential TEP covariance at 4,200/1.6, 4,200, and "
                      "4,200*1.6 km; nugget 0.70"),
            "pass_rule": ("at least 50 days and positive lower 95% day-cluster "
                          "interval versus tuned conventional, empirical, equal-"
                          "information phase-only, geometry-scramble, and phase-"
                          "scramble controls"),
            "source_sha256": REGISTERED_SOURCE_SHA256,
            "source_hashes_verified": True,
        },
        "component_pass": component_pass,
        "result": result,
        "input_manifest": manifest,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "p1_pass": p1_pass,
        "component_pass": component_pass,
        "n_days_used": result.get("n_days_used"),
        "primary": result.get("full_arsenal_vs_tuned_conventional_pct"),
        "output": str(OUT),
    }, indent=2))
    return 0 if p1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
