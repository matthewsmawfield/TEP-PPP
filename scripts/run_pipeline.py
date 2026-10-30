#!/usr/bin/env python3
"""Reproducible TEP-PPP pipeline orchestrator.

Runs the full TEP-PPP analysis pipeline with professional logging, timing,
and progress tracking. Supports selective execution via --quick, --skip-data,
--from-step, and --only-step flags.

Usage:
    python3 scripts/run_pipeline.py --list
    python3 scripts/run_pipeline.py --quick
    python3 scripts/run_pipeline.py --only-step step_04_convergence_test
    python3 scripts/run_pipeline.py --from-step step_10_global_correlation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_DIR = ROOT / "scripts" / "steps"


@dataclass(frozen=True)
class Step:
    script: str
    description: str
    args: tuple[str, ...] = field(default_factory=tuple)
    tags: frozenset[str] = field(default_factory=frozenset)
    # Superseded early-pipeline scripts are retained under
    # archive/early_superseded/scripts/ (not tracked).  An optional step is
    # skipped with a loud notice when its script is absent, so the
    # reproduction path stays runnable on a fresh clone.
    optional: bool = False

    @property
    def key(self) -> str:
        return self.script.removesuffix(".py")


STEPS = [
    Step("step_01_acquire_data.py", "Acquire RINEX/products",
         ("--all-stations", "--year", "2024", "--doys", "100-106",
          "--download-nav", "--download-sp3", "--download-clk"), frozenset({"data", "network", "heavy"}), optional=True),
    Step("step_02_process_ppp.py", "Extract broadcast-SPP clocks",
         ("--all-stations", "--year", "2024", "--doys", "100-106"), frozenset({"data", "heavy"}), optional=True),
    Step("step_03_kernel_validation.py", "Exploratory kernel validation", optional=True),
    Step("step_03b_controls.py", "Small-cluster controls"),
    Step("step_04_convergence_test.py", "Exploratory convergence replay", optional=True),
    Step("step_05_longitudinal_stability.py", "Longitudinal stability", tags=frozenset({"heavy"}), optional=True),
    Step("step_06_falsification.py", "Falsification controls"),
    Step("step_08_integrity_monitoring.py", "Integrity proof of concept", optional=True),
    Step("step_09_acquire_global_clk.py", "Acquire global CLK products",
         ("--year", "2024", "--doys", "1-366", "--centres", "cod,gfz"), frozenset({"network", "heavy"})),
    Step("step_10_global_correlation.py", "T1 global correlation"),
    Step("step_11_anisotropy.py", "T2 anisotropy"),
    Step("step_12_resolvability.py", "T3 aperture resolvability"),
    Step("step_13_commonview.py", "T4 common-view discriminator"),
    Step("step_14_precise_global.py", "Global precise-clock replication", tags=frozenset({"heavy"})),
    Step("step_15_csd_crosscheck.py", "CSD cross-check"),
    Step("step_16_draconitic_falsification.py", "Draconitic falsification"),
    Step("step_25_kernel_selection.py", "Nested-CV kernel selection"),
    Step("step_26_operational_gain.py", "Analytical position propagation"),
    Step("step_27_discriminators.py", "Product/constellation discriminators"),
    Step("step_32_kernel_family.py", "Kernel-family comparison"),
    Step("step_34_heldout_lodo.py", "Full-year leave-one-day-out", tags=frozenset({"heavy"})),
    Step("step_38_baseline_battery.py", "Strong conventional baseline battery", tags=frozenset({"heavy"})),
    Step("step_33_position_demo.py", "Post-fit position replay", tags=frozenset({"heavy"})),
    Step("step_39_endtoend_position.py", "Causally calibrated post-fit replay", tags=frozenset({"heavy"})),
    Step("step_52_drone_kinematic.py", "Drone/UAV kinematic P2b test", tags=frozenset({"heavy", "dynamic"})),
    Step("step_40_bandpass_sensitivity.py", "Filter sensitivity", tags=frozenset({"heavy"})),
    Step("step_23_pride_endtoend.py", "Production PPP mechanism probe", tags=frozenset({"heavy"})),
    Step("step_54_dense_operational_skill.py", "Density-scaling operational skill", tags=frozenset({"heavy"})),
    Step("step_55_dense_outage_prediction.py", "Dense-network outage bridging", tags=frozenset({"heavy"})),
    Step("step_56_phase_predictive_bridge.py", "Phase-cos causal predictive bridge", tags=frozenset({"heavy"})),
    Step("step_57_joint_estimator_sensitivity.py", "Joint-estimator / k=3 clock sensitivity", tags=frozenset({"heavy"})),
    Step("step_58_clock_field_lengthscales.py", "Clock-field phase-lock vs coherence length", tags=frozenset({"heavy"})),
    Step("step_59_common_mode_is_tep.py", "Network-mean clock = TEP field test", tags=frozenset({"heavy"})),
    Step("step_60_operational_benefits.py", "Operational benefits ledger"),
    Step("step_64_maximal_tep_p1.py", "Maximal phase-cos TEP vs tuned conventional (P1)", tags=frozenset({"heavy"})),
    Step("step_65_oscillator_screening.py", "Oscillator-type screening of the TEP field", tags=frozenset({"heavy"})),
    Step("step_66_clock_gradient_holygrail.py", "Scalar-sector field-gradient bounds", tags=frozenset({"heavy"})),
    Step("step_68_synchronization_holonomy.py", "Synchronization holonomy (non-exact sector)", tags=frozenset({"heavy"})),
    Step("step_69_sidereal_multipath_separation.py", "Sidereal multipath separation", tags=frozenset({"heavy"})),
    Step("step_61_acquire_regional_cors.py", "Acquire dense EUREF/EPN CORS",
         ("--region", "germany_benelux", "--year", "2024", "--doys", "100-106", "--max-stations", "30"),
         frozenset({"data", "network", "heavy"})),
    Step("step_63_acquire_kinematic.py", "Acquire UrbanNav urban kinematic",
         ("--collection", "all"), frozenset({"data", "network", "heavy"})),
    Step("step_19_phase1_report.py", "Phase-1 gate report"),
    # Roadmap-driven next-generation steps (bottleneck breakers)
    Step("step_97_multi_year_seasonal_lever.py", "Multi-year seasonal lever and decade N_eff projection", tags=frozenset({"heavy"})),
    Step("step_98_satellite_channel_flagship.py", "Satellite-channel flagship cross-spectral evidence", tags=frozenset({"heavy"})),
    Step("step_99_precise_clock_position.py", "Precise-clock position architecture (bypass SPP)", tags=frozenset({"heavy"})),
    Step("step_100_decisive_drone_experiment.py", "Decisive drone experiment: aided kinematics and 24h phase state"),
    Step("step_101_full_field_predictor.py", "Full-field spatiotemporal predictor (no detrending)", tags=frozenset({"heavy"})),
    Step("step_102_causal_deployable_p1_gate.py", "Causally-tuned deployable P1 gate", tags=frozenset({"heavy"})),
    Step("step_103_atomic_quartz_gradients.py", "Direct atomic-vs-quartz phase + independent gradients", tags=frozenset({"heavy"})),
]


def _normalise(name: str) -> str:
    return Path(name).stem


def select_steps(args) -> list[Step]:
    selected = STEPS[:]
    if args.skip_data:
        selected = [s for s in selected if "data" not in s.tags]
    if args.quick:
        selected = [s for s in selected if not ({"heavy", "network"} & s.tags)]
    if args.only_step:
        key = _normalise(args.only_step)
        selected = [s for s in STEPS if s.key == key]
    elif args.from_step:
        key = _normalise(args.from_step)
        positions = [i for i, s in enumerate(selected) if s.key == key]
        if not positions:
            raise SystemExit(f"unknown or skipped --from-step: {args.from_step}")
        selected = selected[positions[0]:]
    if not selected:
        raise SystemExit("no steps selected")
    return selected


def run_step(step: Step, idx: int, total: int) -> float:
    """Execute a single pipeline step with professional output and timing.

    Returns elapsed time in seconds.
    """
    path = STEP_DIR / step.script
    if not path.exists():
        if step.optional:
            print(f"\n{'='*80}", flush=True)
            print(f"  PIPELINE STEP {idx}/{total}: {step.key} -- SKIPPED", flush=True)
            print(f"  Superseded script not present at {path}", flush=True)
            print(f"  (archived under archive/early_superseded/scripts/)", flush=True)
            print(f"{'='*80}\n", flush=True)
            return 0.0
        raise SystemExit(f"missing pipeline step: {path}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}", flush=True)
    print(f"  PIPELINE STEP {idx}/{total}: {step.key}", flush=True)
    print(f"  {step.description}", flush=True)
    print(f"  Started: {ts} UTC", flush=True)
    print(f"  Script:  {path.relative_to(ROOT)}", flush=True)
    if step.args:
        print(f"  Args:    {' '.join(step.args)}", flush=True)
    print(f"{'='*80}\n", flush=True)

    t0 = time.time()
    subprocess.run([sys.executable, str(path), *step.args], cwd=ROOT, check=True)
    elapsed = time.time() - t0

    if elapsed >= 60:
        elapsed_str = f"{elapsed/60:.1f} min"
    else:
        elapsed_str = f"{elapsed:.1f} s"
    print(f"\n  -> Step {step.key} completed in {elapsed_str}\n", flush=True)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip network/heavy steps")
    parser.add_argument("--skip-data", action="store_true", help="reuse existing raw/processed data")
    parser.add_argument("--from-step", help="start at this step name")
    parser.add_argument("--only-step", help="run exactly this step name")
    parser.add_argument("--list", action="store_true", help="list steps and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Step':<40s} {'Description'}")
        print(f"{'-'*40} {'-'*50}")
        for step in STEPS:
            tags_str = f" [{', '.join(sorted(step.tags))}]" if step.tags else ""
            print(f"{step.key:<40s} {step.description}{tags_str}")
        print(f"\nTotal: {len(STEPS)} steps")
        return 0

    selected = select_steps(args)
    pipeline_start = time.time()
    ts_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'#'*80}", flush=True)
    print(f"#  TEP-PPP ANALYSIS PIPELINE", flush=True)
    print(f"#  Started: {ts_start} UTC", flush=True)
    print(f"#  Steps selected: {len(selected)} of {len(STEPS)}", flush=True)
    if args.quick:
        print(f"#  Mode: --quick (network/heavy steps skipped)", flush=True)
    if args.skip_data:
        print(f"#  Mode: --skip-data (reusing existing data)", flush=True)
    if args.only_step:
        print(f"#  Mode: --only-step {args.only_step}", flush=True)
    if args.from_step:
        print(f"#  Mode: --from-step {args.from_step}", flush=True)
    print(f"{'#'*80}\n", flush=True)

    timings = []
    for idx, step in enumerate(selected, 1):
        elapsed = run_step(step, idx, len(selected))
        timings.append((step.key, step.description, elapsed))

    # Pipeline summary
    total_elapsed = time.time() - pipeline_start
    if total_elapsed >= 60:
        total_str = f"{total_elapsed/60:.1f} min"
    else:
        total_str = f"{total_elapsed:.1f} s"

    print(f"\n{'='*80}", flush=True)
    print(f"  PIPELINE SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Step':<40s} {'Time':>10s}", flush=True)
    print(f"  {'-'*40} {'-'*10}", flush=True)
    for key, desc, elapsed in timings:
        if elapsed >= 60:
            t_str = f"{elapsed/60:.1f} min"
        else:
            t_str = f"{elapsed:.1f} s"
        print(f"  {key:<40s} {t_str:>10s}", flush=True)
    print(f"  {'-'*40} {'-'*10}", flush=True)
    print(f"  {'TOTAL':<40s} {total_str:>10s}", flush=True)
    print(f"{'='*80}\n", flush=True)

    subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_operational_evidence.py")],
                   cwd=ROOT, check=True)
    print("Pipeline completed; evidence ledger regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
