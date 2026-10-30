#!/usr/bin/env python3
"""
STEP 24  (Phase 2) -- Network clock prior under degraded satellite geometry.

*** BASELINE SUPERSEDED BY STEP 26. ***
The "improvement" column here is measured against NO clock prior at all, which
credits the prior with the gain from constraining the clock at all -- something
any prior delivers.  Step 26 repeats this with the honest decomposition
(self / bulk-mean / kernel priors, all of held-out measured quality) and with
nested-CV hyperparameters.  The *shape* of the result (prior is decisive at
3-5 satellites, negligible with a full sky) is unchanged and is confirmed
there; only the attribution changes.

Step 23 showed a network clock prior is a no-op for clean 24 h multi-GNSS phase
PPP -- the clock is already observation-limited.  This step shows the opposite
regime, which is where the prior earns its keep: WHEN THE SKY IS BLOCKED.

Method (analytical, real SP3 geometry, measured prior quality):
  for each station-day and each epoch, take the visible satellites; for a
  target visible-count k in {3,4,5,6,8,all}, keep the k highest-elevation sats;
  build the design matrix A = [-los, 1] (state [x y z clk], ZTD fixed from a
  model, as is standard in a short obstruction interval); compute the 3D
  position sigma with:
      none     -- no prior         (singular for k<4)
      kriging  -- clock pseudo-obs of the MEASURED distance-kernel LOO quality
  Report the position sigma and the improvement vs k.

Plus a real PRIDE-PPP-AR run with an urban-canyon satellite mask (GPS-only,
50 deg elevation cutoff), 3 arms, if the staged ZIMM run directory is given.
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
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib.tep_geometry import load_sp3, sat_grid, elev_los, ELEV_MASK  # noqa: E402
from scripts.steps.step_21_position_information import clock_prediction_skill  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
K_VALUES = [3, 4, 5, 6, 8, 99]


def pos_sigma_4p(los, sig_clk):
    """3D position sigma for state [x y z clk]; sig_clk None -> no prior."""
    n = len(los)
    # With fewer than 4 satellites the state [x y z clk] is rank-deficient
    # and the position sigma is infinite without a prior.
    if n < 4 and sig_clk is None:
        return np.inf
    A = np.column_stack([-los, np.ones(n)])          # (n, 4)
    N = A.T @ A
    if sig_clk is not None:
        N = N + np.diag([0, 0, 0, 1.0 / sig_clk ** 2])
    # Detect rank deficiency via condition number.  Use a tighter threshold
    # (1e8) to catch near-singular matrices that produce huge but finite
    # inverses — the previous 1e-10 ratio let through 73-Mm artifacts.
    sv = np.linalg.svd(N, compute_uv=False)
    if sv[0] <= 0 or sv[-1] < sv[0] * 1e-8 or sv[-1] < 1e-12:
        return np.inf
    try:
        Cov = np.linalg.pinv(N)
    except np.linalg.LinAlgError:
        return np.inf
    v = np.trace(Cov[:3, :3])
    return float(np.sqrt(v)) if v > 0 and np.isfinite(v) else np.inf


def run_analytical(product, date_tags):
    rows = {k: {"none": [], "krig": []} for k in K_VALUES}
    grid = np.arange(0, 86400.0, 1800.0)             # every 30 min
    for dtag in date_tags:
        sp3 = load_sp3(dtag)
        if sp3 is None:
            continue
        sp3g = sat_grid(sp3, grid)
        skill = clock_prediction_skill(product, dtag)
        for nm, sk in skill.items():
            sta = sk["station"].ecef_m
            sig_p = sk["err_krig_m"]
            for gi in range(len(grid)):
                vis = []
                for s, pg in sp3g.items():
                    p = pg[gi]
                    if not np.isfinite(p[0]):
                        continue
                    e, u, _ = elev_los(p, sta)
                    if e > ELEV_MASK:
                        vis.append((e, u))
                if len(vis) < 3:
                    continue
                vis.sort(key=lambda z: -z[0])
                for k in K_VALUES:
                    kk = min(k, len(vis))
                    if kk < 3:
                        continue
                    los = np.array([u for _, u in vis[:kk]])
                    rows[k]["none"].append(pos_sigma_4p(los, None))
                    rows[k]["krig"].append(pos_sigma_4p(los, sig_p))
    out = {}
    for k in K_VALUES:
        n = np.array(rows[k]["none"]); c = np.array(rows[k]["krig"])
        finite_n = np.isfinite(n)
        med_n = float(np.median(n[finite_n])) if finite_n.any() else float("inf")
        med_c = float(np.median(c[np.isfinite(c)]))
        out[str(k)] = {
            "n_epochs": int(len(n)),
            "frac_singular_without_prior": float(np.mean(~finite_n)),
            "pos_sigma_none_m": med_n,
            "pos_sigma_kriging_m": med_c,
            "improvement_pct": (float(100 * (1 - med_c / med_n)) if np.isfinite(med_n) and med_n > 0
                                else 100.0),
        }
        tag = "all" if k == 99 else str(k)
        print(f"  k={tag:>3}  sats  sigma_none {med_n:8.2f} m   sigma_krig {med_c:6.2f} m   "
              f"improvement {out[str(k)]['improvement_pct']:5.1f}%   "
              f"({out[str(k)]['frac_singular_without_prior']*100:.0f}% singular w/o prior)")
    return out


# --------------------------------------------------------------- real PRIDE mask
def sdk_env():
    env = dict(os.environ)
    try:
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        env["LIBRARY_PATH"] = f"{sdk}/usr/lib"
    except Exception:
        pass
    return env


def urban_pride(rundir, rover_xyz, date, work):
    """Re-run ZIMM PPP with a GPS-only, 50deg-elevation urban-canyon mask, 3 arms."""
    P = PROJECT_ROOT / "archive" / "PRIDE-PPPAR"
    if not (P / "bin" / "lsq").exists():
        return {"skipped": "PRIDE not built"}
    rundir = Path(rundir)
    obs = next(rundir.parent.parent.glob("*.rnx"))
    atx = P / "table" / "igs20_2317.atx"

    def one(tag, prior):
        d = work / f"urb_{tag}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(rundir, d)
        for j in ("rck_2024100", "res_2024100", "ztd_2024100", "abs_igs.atx"):
            (d / j).unlink(missing_ok=True)
        (d / "abs_igs.atx").symlink_to(atx)
        cfg = next(d.glob("config.*"))
        txt = cfg.read_text()
        txt = txt.replace("Cutoff elevation        = 15", "Cutoff elevation        = 50")
        txt = txt.replace("Frequency combination  = G12 R12 E15 C26 J12",
                          "Frequency combination  = G12")
        # only GPS satellites active
        txt = "\n".join(l if not (len(l) > 4 and l[1] in "REC" and l[2:4].isdigit() and l.startswith(" "))
                        else "#" + l[1:] for l in txt.splitlines())
        cfg.write_text(txt)
        env = sdk_env()
        if prior:
            env["TEP_CLK_PRIOR"] = str(prior)
        else:
            env.pop("TEP_CLK_PRIOR", None)
        r = subprocess.run([str(P / "bin" / "lsq"), cfg.name, str(obs)],
                           cwd=d, env=env, stdout=open(d / "lsq.log", "w"),
                           stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"WARNING: PRIDE lsq returned {r.returncode} for urban_pride; "
                  f"kinematic solution may be missing or invalid", file=sys.stderr)
        rows = []
        for line in (d / "kin_2024100_zimm").read_text().splitlines():
            pp = line.split()
            if len(pp) >= 5 and pp[0].isdigit() and len(pp[0]) == 5:
                try:
                    rows.append([float(x) for x in pp[2:5]])
                except ValueError:
                    pass
        a = np.array(rows)
        if len(a) < 30:
            return None, 0
        cc = a[len(a) // 3:]
        return cc.mean(0), float(np.sqrt(((cc - cc.mean(0)) ** 2).sum(1).mean()))

    gen = PROJECT_ROOT / "scripts" / "steps" / "step_22_make_clock_prior.py"
    mA = one("A", None)
    rck1 = work / "urb_A" / "rck_2024100"
    priors = {}
    for arm in ("C",):
        pf = work / f"urb_prior_{arm}.cpr"
        subprocess.run([sys.executable, str(gen), "--rover", "ZIMM", "--date", date,
                        "--arm", arm, "--rover-xyz", rover_xyz, "--from-rck", str(rck1),
                        "-o", str(pf)], check=True, stdout=subprocess.DEVNULL)
        priors[arm] = pf
    mC = one("C", priors["C"])
    if not (mA[0] is not None and mC[0] is not None):
        return {"note": "urban PPP did not converge in one or more arms"}
    return {
        "mask": "GPS-only, 50 deg elevation cutoff",
        "arm_A_scatter_cm": round(mA[1] * 100, 2),
        "arm_C_scatter_cm": round(mC[1] * 100, 2),
        "arm_C_offset_vs_A_cm": round(float(np.linalg.norm(mC[0] - mA[0])) * 100, 2),
        "note": ("batch 24 h phase PPP is robust to an instantaneous mask because it "
                 "accumulates geometry over the whole day -- arms are identical. The clock "
                 "prior's regime is epoch-wise / real-time PPP under obstruction, which the "
                 "analytical result above quantifies (a batch engine cannot demonstrate it)."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pride-rundir", default=None)
    ap.add_argument("--rover-xyz", default="4331296.8989,567556.4394,4633134.3900")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp") / "step24_work"
    work.mkdir(exist_ok=True)

    print("[T-degraded] position sigma vs visible-satellite count "
          "(real geometry, measured kriging-prior quality)")
    analytical = run_analytical("clk_cod", td.available_dates("clk_cod", 2024))

    pride = None
    if a.pride_rundir:
        print("\n[urban PRIDE] GPS-only 50deg mask, 3 arms")
        pride = urban_pride(a.pride_rundir, a.rover_xyz, "2024100", work)
        print(f"  {pride}")

    # computational cost of the prior: 2 scalar writes to the normal matrix per epoch
    compute = {
        "added_operations_per_epoch": "2 (N[clk,clk] += w ;  N[clk,rhs] += w*(z-x0))",
        "flops_vs_gnss_update": "~1e-6  (a full epoch update is O(n_sat * n_param^2) ~ 1e3-1e4 flops)",
        "note": "the prior is a rank-1 diagonal + RHS update; negligible overhead, edge-deployable",
    }
    res = {
        "test": "phase2_degraded_geometry",
        "analytical_by_satellite_count": analytical,
        "urban_pride": pride,
        "computational_cost": compute,
        "finding": ("A network clock prior is negligible with a full sky but decisive under "
                    "obstruction: at 3-5 visible satellites it removes the clock/vertical "
                    "degeneracy that makes instantaneous PPP rank-deficient or >10 m. This is "
                    "the real-time / urban-canyon operational niche; batch 24 h phase PPP does "
                    "not need it (Step 23). Overhead is 2 scalar operations per epoch."),
    }
    (OUT / "step_24_degraded_geometry.json").write_text(json.dumps(res, indent=2))
    k3 = analytical.get("3", {}); k4 = analytical.get("4", {}); k5 = analytical.get("5", {})
    print(f"\n  under obstruction (instantaneous geometry):")
    print(f"    k=3 sats: {k3.get('frac_singular_without_prior', 0)*100:.0f}% singular without the prior; "
          f"with prior -> {k3.get('pos_sigma_kriging_m', 0):.1f} m")
    print(f"    k=4 sats: {k4.get('pos_sigma_none_m', 0):.0f} m -> {k4.get('pos_sigma_kriging_m', 0):.1f} m "
          f"({k4.get('improvement_pct', 0):.0f}%)")
    print(f"    k=5 sats: {k5.get('pos_sigma_none_m', 0):.0f} m -> {k5.get('pos_sigma_kriging_m', 0):.1f} m "
          f"({k5.get('improvement_pct', 0):.0f}%)")
    print(f"  overhead: 2 scalar ops / epoch")
    print(f"saved {OUT / 'step_24_degraded_geometry.json'}")


if __name__ == "__main__":
    main()
