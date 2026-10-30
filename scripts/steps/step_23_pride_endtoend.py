#!/usr/bin/env python3
"""
STEP 23  (Phase 2, end-to-end) -- 3-arm PPP with the patched PRIDE-PPP-AR engine.

Drives `bin/lsq` (with `lsq_add_clkcon`) on a real, fully-staged PRIDE run
directory, three ways:
    A  no prior                (unset TEP_CLK_PRIOR)
    B  network-mean clock prior (step_22 --arm B, two-pass)
    C  distance-kernel prior    (step_22 --arm C, two-pass)
plus a PROBE arm (tight, deliberately biased prior) that verifies the
constraint actually reaches the position state.

Result (ZIMM, DOY 100 2024, WUM MGEX products, kinematic multi-GNSS phase PPP):
    A / B / C converged 3D scatter are IDENTICAL to 0.001 cm.
    PROBE (sigma 0.02 m, +5 m bias) shifts the converged position by 4.7 cm.

Interpretation: in a clean 24 h multi-GNSS *phase* PPP solution the receiver
clock is already observation-limited to ~1 cm, so a network clock prior of
realistic quality (sigma ~0.5-1 m) is ~30x too loose to matter -- it is
correctly a no-op.  The prior only bites when it is tighter than the
observations (the PROBE), i.e. in real-time / single-frequency / few-satellite
/ degraded-geometry regimes.  That is the operational niche for a network
clock prior; standard post-processed multi-GNSS PPP does not need one.

This complements Step 21 (analytical, snapshot geometry: prior does help there
because it competes on equal footing with a single-epoch normal matrix) and
Step 20 (EKF, degraded regime, filter-limited).

Usage:
    python step_23_pride_endtoend.py --rundir <staged PRIDE 2024/100 dir> \\
        --rover ZIMM --rover-xyz "x,y,z" --date 2024100
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIDE = PROJECT_ROOT / "archive" / "PRIDE-PPPAR"
ATX = PRIDE / "table" / "igs20_2317.atx"
OUT = PROJECT_ROOT / "results" / "outputs"


def sdk_env():
    env = dict(os.environ)
    try:
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        env["LIBRARY_PATH"] = f"{sdk}/usr/lib"
    except Exception:
        pass
    return env


def converged_pos(kin_path):
    rows = []
    for line in Path(kin_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 5:
            try:
                rows.append([float(x) for x in p[2:5]])
            except ValueError:
                pass
    a = np.array(rows)
    c = a[len(a) // 3:]                       # drop convergence third
    return c.mean(0), float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))


def run_arm(rundir, tag, obs, prior_file, workroot, date_tag, rover):
    d = workroot / tag
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(rundir, d)
    for junk in ("rck_2024100", "res_2024100", "ztd_2024100", "abs_igs.atx"):
        (d / junk).unlink(missing_ok=True)
    (d / "abs_igs.atx").symlink_to(ATX)
    cfg = next(d.glob("config.*"))
    env = sdk_env()
    if prior_file:
        env["TEP_CLK_PRIOR"] = str(prior_file)
    else:
        env.pop("TEP_CLK_PRIOR", None)
    with open(d / "lsq.log", "w") as log:
        r = subprocess.run([str(PRIDE / "bin" / "lsq"), cfg.name, obs],
                           cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"WARNING: PRIDE lsq returned {r.returncode} for {tag}; "
              f"kinematic solution may be missing or invalid", file=sys.stderr)
    kin_file = d / f"kin_{date_tag}_{rover.lower()}"
    if not kin_file.exists():
        print(f"WARNING: kin file {kin_file} not found for {tag}", file=sys.stderr)
        return {"mean": None, "std_cm": None, "prior_applied": False}
    mean, std = converged_pos(kin_file)
    # Check that the prior was actually applied by looking for the
    # constraint-addition log line from lsq_add_clkcon.
    applied = "lsq_add_clkcon): receiver clock prior" in (d / "lsq.log").read_text()
    return {"mean": mean, "std_cm": std * 100, "prior_applied": applied}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--rover", default="ZIMM")
    ap.add_argument("--rover-xyz", required=True)
    ap.add_argument("--date", default="2024100")
    ap.add_argument("--product", default="clk_cod")
    a = ap.parse_args()

    rundir = Path(a.rundir).resolve()
    # Find the RINEX observation file in the parent directory.
    obs_files = sorted(rundir.parent.parent.glob("*.rnx"))
    if not obs_files:
        # fall back to the canonical naming convention
        canonical = rundir.parent.parent / f"{a.rover}00CHE.rnx"
        if canonical.exists():
            obs_files = [canonical]
    if not obs_files:
        print(f"ERROR: no RINEX (.rnx) files found in {rundir.parent.parent}", file=sys.stderr)
        sys.exit(1)
    obs = str(obs_files[0])
    work = Path("/tmp") / "step23_work"
    work.mkdir(exist_ok=True)

    # pass 1: arm A, to get the absolute clock trend
    a_res = run_arm(rundir, "armA", obs, None, work, a.date, a.rover)
    rck1 = work / "armA" / f"rck_{a.date}"

    gen = PROJECT_ROOT / "scripts" / "steps" / "step_22_make_clock_prior.py"
    priors = {}
    for arm in ("B", "C"):
        pf = work / f"prior_{arm}.cpr"
        subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                        "--product", a.product, "--arm", arm, "--rover-xyz", a.rover_xyz,
                        "--from-rck", str(rck1), "-o", str(pf)], check=True,
                       stdout=subprocess.DEVNULL)
        priors[arm] = pf

    b_res = run_arm(rundir, "armB", obs, priors["B"], work, a.date, a.rover)
    c_res = run_arm(rundir, "armC", obs, priors["C"], work, a.date, a.rover)

    # probe: tight, biased prior -> must shift the solution
    probe = work / "probe.cpr"
    d = np.loadtxt(priors["C"]); d[:, 2] += 5.0; d[:, 3] = 0.02
    np.savetxt(probe, d, fmt="%d %.2f %.6f %.6f")
    p_res = run_arm(rundir, "armProbe", obs, probe, work, a.date, a.rover)

    mA = a_res["mean"]
    def _offset(res):
        if res["mean"] is None or mA is None:
            return None
        return round(float(np.linalg.norm(res["mean"] - mA)) * 100, 3)
    def _scatter(res):
        return round(res["std_cm"], 2) if res["std_cm"] is not None else None
    result = {
        "test": "phase2_pride_endtoend_3arm",
        "rover": a.rover, "date": a.date, "engine": "PRIDE-PPP-AR 3.2.1 + lsq_add_clkcon",
        "mode": "kinematic multi-GNSS ionosphere-free phase PPP, WUM MGEX products",
        "arm_A_no_prior":       {"scatter_3d_cm": _scatter(a_res)},
        "arm_B_network_mean":   {"scatter_3d_cm": _scatter(b_res),
                                 "offset_vs_A_cm": _offset(b_res),
                                 "prior_applied": b_res["prior_applied"]},
        "arm_C_distance_kernel":{"scatter_3d_cm": _scatter(c_res),
                                 "offset_vs_A_cm": _offset(c_res),
                                 "prior_applied": c_res["prior_applied"]},
        "probe_tight_biased":   {"offset_vs_A_cm": _offset(p_res),
                                 "note": "sigma 0.02 m + 5 m bias -- verifies the constraint reaches position"},
        "finding": ("In clean 24 h multi-GNSS phase PPP the receiver clock is observation-limited "
                    "to ~1 cm, so a realistic-quality network clock prior (sigma ~0.5-1 m) is a "
                    "no-op. The prior's operational niche is real-time / single-frequency / "
                    "few-satellite / degraded-geometry PPP, not standard post-processed PPP."),
        "clean_solution_identical": bool(
            (b_res["prior_applied"] and c_res["prior_applied"]
             and _offset(b_res) is not None and _offset(c_res) is not None
             and _offset(b_res) < 0.01 and _offset(c_res) < 0.01)),
        "mechanism_verified": bool(p_res["prior_applied"] and p_res["mean"] is not None
                                   and mA is not None
                                   and np.linalg.norm(p_res["mean"] - mA) * 100 > 1.0),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step_23_pride_endtoend.json").write_text(json.dumps(result, indent=2))

    print("PRIDE-PPP-AR end-to-end, 3 arms (ZIMM DOY 100 2024):")
    print(f"  A no prior          : {a_res['std_cm']:.2f} cm 3D scatter")
    print(f"  B network-mean prior: {b_res['std_cm']:.2f} cm   offset vs A {result['arm_B_network_mean']['offset_vs_A_cm']} cm")
    print(f"  C distance kernel   : {c_res['std_cm']:.2f} cm   offset vs A {result['arm_C_distance_kernel']['offset_vs_A_cm']} cm")
    print(f"  PROBE (tight+biased): offset vs A {result['probe_tight_biased']['offset_vs_A_cm']} cm  "
          f"(mechanism verified: {result['mechanism_verified']})")
    print(f"\n  {result['finding']}")
    print(f"saved {OUT / 'step_23_pride_endtoend.json'}")


if __name__ == "__main__":
    main()
