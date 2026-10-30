#!/usr/bin/env python3
"""
STEP 79  --  Testing the non-exact (holonomy) sector with a multipath model that
             is NOT blind to it.

Step 72 proved the sidereal template cannot test this sector.  The loop
invariant T = (e_k x e_ref).b_hat is a function of satellite geometry, and
satellite geometry is what repeats every sidereal day, so a leave-one-day-out
sidereal template removes a STATIC holonomy along with the multipath: injected
recovery was 1.00 before cleaning and 0.00 after, on all eight pairs.  Step 69's
post-cleaning null was therefore null by construction, and Section 3.5h records
the non-exact sector as untested rather than bounded.

This step builds the estimator that Step 72's analysis pointed to but that was
never implemented.  The structural asymmetry it exploits:

    multipath  is a per-satellite, per-site function of local azimuth and
               elevation:  MP_site(az, el)
    T          is an irreducibly TWO-satellite cross product, which no
               single-satellite site map can represent

A double difference contains four (station, satellite) legs, so its multipath is

    MP_A(az_Ap, el_Ap) - MP_A(az_Ar, el_Ar) - MP_B(az_Bp, el_Bp) + MP_B(az_Br, el_Br)

Each MP_site is expanded on a coarse az/el grid and fitted JOINTLY with T, the
tropospheric gradients and per-arc offsets.  Because the az/el basis spans only
single-satellite site functions, it can absorb arbitrary multipath while leaving
T identifiable -- exactly the leverage that day-repetition destroys.

VALIDATION FIRST.  The method is proven to have power BEFORE the answer is
looked at: a synthetic holonomy of known slope is injected into the real double
difference and the fit must recover it.  Step 72 recovered 0.00 after sidereal
cleaning; anything comparable here would mean this estimator is blind too and
its answer must not be reported.  Only if recovery is near unity is the measured
slope interpretable.

Output: results/outputs/step_79_azel_holonomy.json
"""
from __future__ import annotations

import os as _os
import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "s69", ROOT / "scripts" / "steps" / "step_69_sidereal_multipath_separation.py")
s69 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s69)
s68 = s69.s68

# Elevation cut is a multipath lever, not a detail.  At the inherited 20 deg cut
# the zero-area control shows slope +0.384 (z=+9.8); at 35 deg it falls to +0.096
# (z=+2.8), a 75% drop, while the 282 km test pair loses only 31% and stays at
# z=+32.  Multipath is concentrated at low elevation, so a cut that leaves the
# control non-null is not measuring geometry.
ELEV_MIN_OVERRIDE = float(_os.environ.get("TEP_ELEV_MIN", "0")) or None

OUT = ROOT / "results" / "outputs" / ("step_79_azel_holonomy%s%s.json" % ("" if _os.environ.get("TEP_AZEL_GRID","12x6")=="12x6" else "_" + _os.environ["TEP_AZEL_GRID"],
   "" if not _os.environ.get("TEP_ELEV_MIN") else "_el" + _os.environ["TEP_ELEV_MIN"]))
RNG = np.random.default_rng(20260905)

# Grid resolution sets the multipath floor, so it is configurable and the run
# records which grid produced the numbers.  TEP_AZEL_GRID=36x18 runs the fine
# grid at which the zero-area control's slope has fallen 40% (see GRID_SWEEP).
_g = _os.environ.get("TEP_AZEL_GRID", "12x6").split("x")
N_AZ, N_EL = int(_g[0]), int(_g[1])
# Grid resolution is not a free choice: it sets the multipath floor.  A co-located
# pair encloses ~zero loop area, so its slope must vanish for a true holonomy --
# any slope there is residual multipath the grid failed to absorb.  GRID_SWEEP
# refines the grid and watches whether the control converges toward zero while a
# test pair's slope holds.  A control that shrinks with resolution while the test
# is stable means the earlier "site-systematics floor" was the grid, not physics.
GRID_SWEEP = [(12, 6), (24, 12), (36, 18)]
# ridge on the nuisance block only; T is left unpenalised
RIDGE = float(_os.environ.get('TEP_RIDGE', '1.0'))
EL_MAX = 90.0
INJECT = [0.005, 0.02, 0.05]
N_PERM = 300


def records(a, b, doy, dense, atx):
    """Like s69.pair_records but also returns the four (az, el) legs per DD."""
    da, db = s69.load_cached(a, doy), s69.load_cached(b, doy)
    if not da or not db:
        return None
    oa, xa, ta_, na_ = da
    ob, xb, tb_, nb_ = db
    bvec = xb - xa
    blen = float(np.linalg.norm(bvec)) / 1000.0
    bhat = bvec / max(np.linalg.norm(bvec), 1e-9)
    out = []
    for s in sorted(set(oa) & set(ob))[::s68.EPOCH_STRIDE]:
        common = sorted(set(oa[s]) & set(ob[s]))
        if len(common) < 4:
            continue
        geo = {}
        for p in common:
            ga = s68.sat_geom(dense, p, s, xa)
            gb = s68.sat_geom(dense, p, s, xb)
            if ga is None or gb is None or min(ga[2], gb[2]) < s68.ELEV_MIN:
                continue
            geo[p] = (ga, gb)
        if len(geo) < 4:
            continue
        ref = max(geo, key=lambda p: geo[p][0][2])
        ga_r, gb_r = geo[ref]
        car = s68.pcv_correction(atx, ta_, na_, ga_r[6], ga_r[7]) or 0.0
        cbr = s68.pcv_correction(atx, tb_, nb_, gb_r[6], gb_r[7]) or 0.0
        sd_ref = ((oa[s][ref] - car) - (ob[s][ref] - cbr)) - (ga_r[0] - gb_r[0])
        for p, (ga, gb) in geo.items():
            if p == ref:
                continue
            ca = s68.pcv_correction(atx, ta_, na_, ga[6], ga[7]) or 0.0
            cb = s68.pcv_correction(atx, tb_, nb_, gb[6], gb[7]) or 0.0
            dd = (((oa[s][p] - ca) - (ob[s][p] - cb)) - (ga[0] - gb[0])) - sd_ref
            T = float(np.dot(np.cross(ga[1], ga_r[1]), bhat))
            gN = ga[5] * np.cos(ga[4]) - ga_r[5] * np.cos(ga_r[4])
            gE = ga[5] * np.sin(ga[4]) - ga_r[5] * np.sin(ga_r[4])
            dwm = abs(ga[3] - ga_r[3])
            out.append((doy, s, p, ref, dd, T, dwm, gN, gE,
                        ga[4], ga[2], ga_r[4], ga_r[2],     # A: sat, ref
                        gb[4], gb[2], gb_r[4], gb_r[2]))    # B: sat, ref
    return (out, blen) if out else None


def arcs_local(recs):
    """s69.arcs_and_ambiguity, but preserving the four az/el legs.  The shared
    version rebuilds 9-tuples and drops them."""
    by = {}
    for r in recs:
        by.setdefault((r[0], r[2], r[3]), []).append(r)
    keep = []
    for _, v in by.items():
        v.sort(key=lambda r: r[1])
        sod = np.array([r[1] for r in v], float)
        dd = np.array([r[4] for r in v], float)
        brk = set(np.where(np.diff(sod) > 400)[0]) | set(np.where(np.abs(np.diff(dd)) > 0.05)[0])
        for sg in np.split(np.arange(len(v)), sorted(x + 1 for x in brk)):
            if len(sg) < s68.MIN_ARC:
                continue
            seg = dd[sg]
            if not np.isfinite(seg).all() or np.std(seg) > 0.5:
                continue
            med = float(np.median(seg))
            for idx in sg:
                r = v[idx]
                keep.append(r[:4] + (r[4] - med,) + r[5:])
    return keep


def cell(az_deg, el_deg):
    ia = int(np.clip(np.floor((az_deg % 360.0) / (360.0 / N_AZ)), 0, N_AZ - 1))
    ie = int(np.clip(np.floor(el_deg / (EL_MAX / N_EL)), 0, N_EL - 1))
    return ia * N_EL + ie


def design(rec, arc_id):
    """[T, dwm, gN, gE, azel cells A (+/-), azel cells B (-/+), arc offsets]."""
    n = len(rec)
    ncell = N_AZ * N_EL
    arcs = np.unique(arc_id)
    X = np.zeros((n, 4 + 2 * ncell + len(arcs)))
    for i, r in enumerate(rec):
        X[i, 0] = r[5]; X[i, 1] = r[6]; X[i, 2] = r[7]; X[i, 3] = r[8]
        X[i, 4 + cell(r[9], r[10])] += 1.0            # MP_A(sat)
        X[i, 4 + cell(r[11], r[12])] -= 1.0           # MP_A(ref)
        X[i, 4 + ncell + cell(r[13], r[14])] -= 1.0   # MP_B(sat)
        X[i, 4 + ncell + cell(r[15], r[16])] += 1.0   # MP_B(ref)
    for k, aid in enumerate(arcs):
        X[arc_id == aid, 4 + 2 * ncell + k] = 1.0
    return X


def slope(X, y, ridge=RIDGE):
    """Coefficient on T, with the NUISANCE block ridge-regularised.

    Unregularised least squares overfits here: a 24x12 grid gives 576 az/el
    columns per pair plus arc offsets, and with observations concentrated in a
    narrow elevation band many cells hold 0-2 points.  The multipath model then
    absorbs signal and the T coefficient becomes unstable -- measured directly,
    the zero-area control's slope flips sign between 12x6 (+0.096) and 24x12
    (-0.260) at the same elevation cut, which is not a physical quantity.

    T (column 0) is left UNPENALISED so the parameter of interest is not shrunk;
    the penalty applies only to the multipath, troposphere and arc columns.
    """
    try:
        n, k = X.shape
        P = np.eye(k) * ridge
        P[0, 0] = 0.0
        c = np.linalg.solve(X.T @ X + P, X.T @ y)
        return float(c[0])
    except np.linalg.LinAlgError:
        return None


def main():
    global N_AZ, N_EL
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if ELEV_MIN_OVERRIDE:
        s68.ELEV_MIN = ELEV_MIN_OVERRIDE
        print(f"elevation cut raised to {ELEV_MIN_OVERRIDE:.0f} deg (multipath control)")
    atx = s68.load_atx()
    dense = {d: s68.grid_for(d) for d in s69.DOYS}
    rows = []
    for a, b in s69.PAIRS:
        allrec, blen = [], None
        for d in s69.DOYS:
            if dense.get(d) is None:
                continue
            r = records(a, b, d, dense[d], atx)
            if r:
                allrec.extend(r[0]); blen = r[1]
        if len(allrec) < 800 or blen is None:
            continue
        rec = arcs_local(allrec)
        if len(rec) < 800:
            continue
        y = np.array([r[4] for r in rec], float)
        m = np.isfinite(y) & (np.abs(y) < 0.3)
        rec = [r for r, k in zip(rec, m) if k]; y = y[m]
        if len(rec) < 800:
            continue
        arc_id = np.array([f"{r[2]}_{r[3]}_{r[0]}" for r in rec])
        X = design(rec, arc_id)
        T = X[:, 0].copy()
        b0 = slope(X, y)
        if b0 is None:
            continue

        # --- validation: does this estimator recover an injected holonomy? ---
        rec_inj = []
        for amp in INJECT:
            bi = slope(X, y + amp * T)
            if bi is not None:
                rec_inj.append((bi - b0) / amp)
        recov = float(np.median(rec_inj)) if rec_inj else None

        # --- null: circular shift of T within each arc (preserves autocorrelation)
        nulls = []
        for _ in range(N_PERM):
            Ts = T.copy()
            for aid in np.unique(arc_id):
                sel = np.where(arc_id == aid)[0]
                if len(sel) > 3:
                    Ts[sel] = np.roll(T[sel], int(RNG.integers(1, len(sel))))
            Xs = X.copy(); Xs[:, 0] = Ts
            v = slope(Xs, y)
            if v is not None:
                nulls.append(v)
        nulls = np.array(nulls)
        z = float((b0 - nulls.mean()) / (nulls.std() + 1e-30)) if len(nulls) else None
        # STABILITY UNDER NUISANCE-MODEL REFINEMENT.
        # A physical coefficient must not depend on the resolution of the model
        # used to remove the nuisance; a leak from that nuisance does.  Measured
        # on the 3 mm zero-area control the slope swings 3x between 12x6 and
        # 24x12 (-0.187 -> -0.447) while the 282 km pair moves 7% (+0.096 ->
        # +0.103).  Only pairs whose slope is stable carry an interpretable term.
        _keep = (N_AZ, N_EL)
        slopes_by_grid = {}
        for naz, nel in ((12, 6), (24, 12)):
            N_AZ, N_EL = naz, nel
            Xg = design(rec, arc_id)
            sg = slope(Xg, y)
            if sg is not None:
                slopes_by_grid[f"{naz}x{nel}"] = sg
        N_AZ, N_EL = _keep
        sv = list(slopes_by_grid.values())
        instability = (abs(sv[0] - sv[1]) / max(abs(sv[0]), abs(sv[1]), 1e-12)
                       if len(sv) == 2 else None)
        rows.append({"pair": f"{a}-{b}", "baseline_km": blen, "n_obs": len(rec),
                     "slope": b0, "z": z, "injection_recovery": recov,
                     "slopes_by_grid": slopes_by_grid,
                     "grid_instability": instability,
                     "stable_under_refinement": bool(instability is not None
                                                     and instability < 0.25)})
        print(f"  {a}-{b:<6} {blen:8.3f} km  n={len(rec):6d}  "
              f"injection recovery {recov:.2f}  slope {b0:+.5f}  z {z:+.2f}", flush=True)

    # ---- grid-resolution sweep: is the floor physics or the model? ----
    sweep = []
    if _os.environ.get('TEP_AZEL_GRID','12x6') != '12x6':
        ctrl, tst = [], []
    else:
        ctrl = [pr for pr in s69.PAIRS if pr[0] == "WTZR"] or s69.PAIRS[:1]
        tst = [pr for pr in s69.PAIRS if pr[0] == "BRUX" and pr[1] == "WSRT"] or s69.PAIRS[-1:]
    for a, b in list(ctrl) + list(tst):
        allrec = []
        for dd_ in s69.DOYS:
            if dense.get(dd_) is None:
                continue
            r = records(a, b, dd_, dense[dd_], atx)
            if r:
                allrec.extend(r[0])
        if len(allrec) < 800:
            continue
        rc = arcs_local(allrec)
        yy = np.array([r[4] for r in rc], float)
        mm = np.isfinite(yy) & (np.abs(yy) < 0.3)
        rc = [r for r, k in zip(rc, mm) if k]; yy = yy[mm]
        aid = np.array([f"{r[2]}_{r[3]}_{r[0]}" for r in rc])
        for naz, nel in GRID_SWEEP:
            N_AZ, N_EL = naz, nel
            Xg = design(rc, aid); Tg = Xg[:, 0].copy()
            bg = slope(Xg, yy)
            if bg is None:
                continue
            rc_i = [(slope(Xg, yy + amp * Tg) - bg) / amp for amp in INJECT]
            nl = []
            for _ in range(60):
                Ts = Tg.copy()
                for q in np.unique(aid):
                    sel = np.where(aid == q)[0]
                    if len(sel) > 3:
                        Ts[sel] = np.roll(Tg[sel], int(RNG.integers(1, len(sel))))
                Xs = Xg.copy(); Xs[:, 0] = Ts
                vv = slope(Xs, yy)
                if vv is not None:
                    nl.append(vv)
            nl = np.array(nl)
            sweep.append({"pair": f"{a}-{b}", "n_az": naz, "n_el": nel,
                          "n_cells": naz * nel, "slope": bg,
                          "z": float((bg - nl.mean()) / (nl.std() + 1e-30)) if len(nl) else None,
                          "injection_recovery": float(np.median(rc_i)) if rc_i else None,
                          "is_zero_area_control": bool(a == "WTZR")})
            print(f"  [grid] {a}-{b:<6} {naz}x{nel} ({naz*nel} cells)  "
                  f"slope {bg:+.5f}  z {sweep[-1]['z']:+.2f}", flush=True)
    N_AZ, N_EL = GRID_SWEEP[0]

    if not rows:
        OUT.write_text(json.dumps({"step": 79, "status": "no_pairs"}, indent=2)); return
    rec_med = float(np.median([r["injection_recovery"] for r in rows
                               if r["injection_recovery"] is not None]))
    powered = rec_med > 0.5
    bl = np.array([r["baseline_km"] for r in rows])
    sl = np.abs([r["slope"] for r in rows])
    rho = float(np.corrcoef(np.argsort(np.argsort(bl)),
                            np.argsort(np.argsort(sl)))[0, 1]) if len(rows) > 3 else None
    detected = [r["pair"] for r in rows if r["z"] is not None and abs(r["z"]) > 5]
    stable = [r for r in rows if r.get("stable_under_refinement")]
    unstable = [r for r in rows if r.get("grid_instability") is not None
                and not r.get("stable_under_refinement")]
    rho_stable = None
    if len(stable) > 3:
        bl_s = np.array([r["baseline_km"] for r in stable])
        sl_s = np.abs([r["slope"] for r in stable])
        rho_stable = float(np.corrcoef(np.argsort(np.argsort(bl_s)),
                                       np.argsort(np.argsort(sl_s)))[0, 1])
    cs = [x for x in sweep if x["is_zero_area_control"]]
    ts = [x for x in sweep if not x["is_zero_area_control"]]
    ctrl_shrinks = bool(len(cs) >= 2 and abs(cs[-1]["slope"]) < 0.8 * abs(cs[0]["slope"]))
    test_stable = bool(len(ts) >= 2 and abs(abs(ts[-1]["slope"]) - abs(ts[0]["slope"]))
                       < 0.25 * abs(ts[0]["slope"]))
    verdict = {
        "grid_sweep": sweep,
        "control_slope_shrinks_with_resolution": ctrl_shrinks,
        "test_slope_stable_with_resolution": test_stable,
        "floor_is_model_not_physics": bool(ctrl_shrinks and test_stable),
        "grid_note": ("a zero-area control whose slope falls as the az/el grid is "
                      "refined was never a holonomy floor -- it was unmodelled "
                      "multipath; a test slope that holds while the control falls is "
                      "not multipath"),
        "median_injection_recovery": rec_med,
        "estimator_has_power": bool(powered),
        "comparison": "sidereal template recovered 0.00 (Step 72); this must be near 1.0 "
                      "for the measured slope to mean anything",
        "n_pairs_detected_z_gt_5": len(detected), "pairs_detected": detected,
        "slope_vs_baseline_spearman": rho,
        "n_stable_under_refinement": len(stable),
        "stable_pairs": [r["pair"] for r in stable],
        "unstable_pairs": [r["pair"] for r in unstable],
        "slope_vs_baseline_spearman_stable_only": rho_stable,
        "stability_note": ("a coefficient that moves with the resolution of the "
                           "nuisance model is a leak from that nuisance, not a "
                           "measurement; only stable pairs are interpretable"),
        "holonomy_signature": ("flux scales with enclosed loop area, so |slope| must grow "
                               "with baseline; rho > 0.6 with detections is the signature"),
        "conclusion": ("estimator BLIND - do not report the slope" if not powered else
                       ("non-exact sector DETECTED" if (len(detected) >= 3 and rho and rho > 0.6)
                        else "estimator has power; no holonomy above the multipath-modelled floor")),
    }
    OUT.write_text(json.dumps({
        "step": 79, "title": "Non-exact sector via a site-frame az/el multipath model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "grid": {"n_az": N_AZ, "n_el": N_EL},
        "pairs": rows, "verdict": verdict}, indent=2))
    print(f"\nmedian injection recovery {rec_med:.2f} "
          f"(sidereal template gave 0.00)  -> estimator has power: {powered}")
    print(f"VERDICT: {verdict['conclusion']}")


if __name__ == "__main__":
    main()
