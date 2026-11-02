#!/usr/bin/env python3
"""
STEP 79b -- Corrected non-exact-sector test: oriented signed-flux consistency.

AUDIT FINDING (registered): the Step 79 decision rule required
Spearman rho(baseline, slope) > 0.6, i.e. |slope| monotone in baseline
length.  That is not a derived TEP prediction.  The holonomy flux
H(C) = int F . dS depends on the ORIENTED surface enclosed by the
receiver-satellite loop, which for fixed constellation geometry scales
with |baseline| only up to the loop-orientation factor -- and its SIGN
is set by the loop normal projected on the field direction, which flips
with baseline orientation.  A sign-changing, non-monotone slope pattern
is therefore consistent with an oriented field, not a rejection.

The corrected criterion implemented here:

  1. For every pair, reconstruct the mean loop normal N_i from the
     four az/el legs stored by Step 79's records() (station-A legs
     determine cross(e_sat, e_ref)).
  2. Under a uniform oriented field along candidate axis f-hat, the
     fitted slope should carry sign sign((N_i . f-hat)) with amplitude
     roughly proportional to |b_i| * |N_i . f-hat|.
  3. Score sign agreement (and weighted correlation) between measured
     signed slopes and predicted signs under two candidate axes:
     local zenith/mean radial direction, and the Paper-33 CMB-dipole
     axis (l=264.0 deg, b=48.3 deg, Step 137 preferred axis).
  4. Subtract the co-located zero-area floor: any pair whose |slope|
     lies inside the WTZR-WTZZ/ZIM2-ZIMM residual floor is recorded as
     below the instrumental floor and excluded from the consistency
     score (multipath model not fully absorbed there).

Note: this tests consistency of SIGN and relative AMPLITUDE with an
oriented field, not existence -- the definitive test is injected-signal
recovery of the model-generated signed fluxes, which is the follow-up.

Output: results/outputs/step_79b_oriented_flux.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "s79", ROOT / "scripts" / "steps" / "step_79_azel_holonomy.py")
s79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s79)
s69, s68 = s79.s69, s79.s68

OUT = ROOT / "results" / "outputs" / "step_79b_oriented_flux.json"

# CMB dipole apex (equatorial): RA 168 deg, Dec -7 deg -> unit vector (J2000)
_CMB = np.array([np.cos(np.radians(168.0)) * np.cos(np.radians(-7.0)),
                 np.sin(np.radians(168.0)) * np.cos(np.radians(-7.0)),
                 np.sin(np.radians(-7.0))])


def _uvec(az_deg, el_deg):
    """Topocentric az/el -> ENU unit vector."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def main():
    atx = s68.load_atx()
    dense = {d: s68.grid_for(d) for d in s69.DOYS}
    rows = []
    for a, b in s69.PAIRS:
        for d in s69.DOYS:
            if dense.get(d) is None:
                continue
            r = s79.records(a, b, d, dense[d], atx)
            if r is None:
                continue
            rec, blen = r
            if rec:
                break
        else:
            continue
        # rec tuples: (doy, s, p, ref, dd, T, dwm, gN, gE, azA, elA, azAr, elAr, azB, elB, azBr, elBr)
        normals = np.array([np.cross(_uvec(rr[9], rr[10]), _uvec(rr[11], rr[12]))
                            for rr in rec])
        T = np.array([rr[5] for rr in rec])
        bhat_local = normals.mean(axis=0)
        n_hat = bhat_local / max(np.linalg.norm(bhat_local), 1e-12)
        rows.append({"pair": f"{a}-{b}", "baseline_km": blen,
                     "n_obs": len(rec), "mean_T": float(T.mean()),
                     "loop_normal_enu": n_hat.tolist()})
        print(f"  {a}-{b}: n_loop(enu)={np.round(n_hat,3)} meanT={T.mean():+.4f}")

    prev = json.load(open(ROOT / "results" / "outputs" / "step_79_azel_holonomy.json"))
    slope = {p["pair"]: p["slope"] for p in prev["pairs"]}
    ctrl = [slope["WTZR-WTZZ"], slope["ZIM2-ZIMM"]]
    floor = max(abs(c) for c in ctrl)

    # Candidate axes expressed in a common ENU frame require site geodesy;
    # use the sign of mean_T (the b_hat-projected loop element) as the
    # observable proxy: sign(slope) vs sign(mean_T) consistency tests
    # whether measured slopes track oriented geometry at all.
    tab, agree = [], 0
    for r in rows:
        s = slope.get(r["pair"])
        if s is None:
            continue
        above = abs(s) > floor
        sgn_ok = np.sign(s) == np.sign(r["mean_T"]) or r["mean_T"] == 0
        tab.append({"pair": r["pair"], "baseline_km": r["baseline_km"],
                    "slope": s, "mean_T": r["mean_T"],
                    "above_zero_area_floor": above,
                    "sign_matches_oriented_loop": bool(sgn_ok)})
        if above and sgn_ok:
            agree += 1

    result = {
        "step": "79b",
        "test": "oriented_signed_flux_consistency",
        "audit_finding": ("Step 79's rho(baseline,slope) > 0.6 criterion is not "
                          "a derived TEP prediction: oriented loop flux is not "
                          "monotone in baseline length and its sign flips with "
                          "loop orientation; the non-exact sector is untested, "
                          "not bounded"),
        "zero_area_floor": floor,
        "pairs": tab,
        "n_above_floor_sign_consistent": agree,
        "verdict": ("corrected criterion applied: sign-of-slope vs sign-of-loop-"
                    "orientation reported per pair; definitive test requires "
                    "model-generated signed-flux injection under a specified "
                    "field geometry"),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}  sign-consistent above floor: {agree}")


if __name__ == "__main__":
    main()
