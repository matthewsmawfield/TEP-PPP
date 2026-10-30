#!/usr/bin/env python3
"""
STEP 69  --  Sidereal separation of multipath from the closed-loop residual.

Step 68 tested the non-exact (disformal) TEP sector via the closed
synchronization loop A->sat1->B->sat2->A -- the carrier-phase double
difference.  Co-located controls were null as required, and a significant
direction-dependent term survived tropospheric-gradient and antenna-PCV
correction.  But it failed the decisive holonomy signature: the flux must scale
with enclosed loop AREA, and instead |slope| tracked each pair's own noise
(Spearman +0.89 vs noise, -0.14 vs baseline).  That is the signature of site
multipath, which ANTEX cannot remove because ANTEX is an antenna-MODEL average,
not a site calibration.

This step removes multipath physically rather than by modelling it.

!! SCOPE WARNING (see step_72b_sidereal_template_blindness.py) !!
This step measures MULTIPATH.  Its post-cleaning holonomy slopes are NOT a bound
on a static non-exact sector: the loop invariant T is a function of satellite
geometry, satellite geometry is what repeats every sidereal day, so the template
removes a static geometry-locked signal along with the multipath.  Step 72 shows
by injection that a static holonomy is recovered at 1.00 before cleaning and
0.00 after, on all eight pairs.  Report the 96.7% figure as multipath; do NOT
report the cleaned null as a physics bound.

GPS geometry repeats every sidereal day (86164.0905 s).  A satellite retraces
the same path over the same local reflectors, so MULTIPATH REPEATS with that
period.  A proper-time field effect does not: the field evolves on its own
timescale, uncorrelated with Earth's rotation relative to the stars.  So:

  * bin every double-difference residual by SIDEREAL phase and satellite;
  * build the multipath template from the OTHER days only (leave-one-day-out,
    so a day is never used to clean itself -- otherwise the subtraction eats
    genuine variance and biases the residual low);
  * subtract, suppressing multipath by ~sqrt(N-1);
  * re-run the holonomy regression on the cleaned residual and re-test whether
    the slope now scales with baseline.

Reported: the sidereal-repeating FRACTION of the double-difference variance
(how multipath-dominated the observable is), and the holonomy slope, z, and
slope-vs-baseline correlation before and after cleaning.

Output: results/outputs/step_69_sidereal_multipath_separation.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "s68", ROOT / "scripts" / "steps" / "step_68_synchronization_holonomy.py")
s68 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s68)

OUT = ROOT / "results" / "outputs" / "step_69_sidereal_multipath_separation.json"
RNG = np.random.default_rng(20260902)

SIDEREAL_S = 86164.0905
DOYS = [100, 101, 102, 103, 104, 105, 106]
SID_BIN_S = 240.0
MIN_DAYS_PER_BIN = 3
PAIRS = s68.PAIRS

_station_cache = {}


def load_cached(site, doy):
    k = (site, doy)
    if k not in _station_cache:
        _station_cache[k] = s68.load_station(site, doy)
    return _station_cache[k]


def pair_records(a, b, doy, dense, atx):
    """Per-observation double-difference records for one pair-day."""
    da, db = load_cached(a, doy), load_cached(b, doy)
    if not da or not db:
        return None
    oa, xa, ta_, na_ = da
    ob, xb, tb_, nb_ = db
    bvec = xb - xa
    blen = float(np.linalg.norm(bvec)) / 1000.0
    bhat = bvec / max(np.linalg.norm(bvec), 1e-9)
    sods = sorted(set(oa) & set(ob))[::s68.EPOCH_STRIDE]
    out = []
    for s in sods:
        common = sorted(set(oa[s]) & set(ob[s]))
        if len(common) < 4:
            continue
        geo = {}
        for p in common:
            ga = s68.sat_geom(dense, p, s, xa)
            gb = s68.sat_geom(dense, p, s, xb)
            if ga is None or gb is None:
                continue
            if min(ga[2], gb[2]) < s68.ELEV_MIN:
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
            abs_t = doy * 86400.0 + s
            sid = abs_t % SIDEREAL_S
            out.append((doy, s, p, ref, dd, T, dwm, gN, gE, sid))
    return (out, blen) if out else None


def arcs_and_ambiguity(recs):
    """Split into arcs on gaps/slips and remove the per-arc ambiguity."""
    by = {}
    for r in recs:
        by.setdefault((r[0], r[2], r[3]), []).append(r)
    keep = []
    for _, v in by.items():
        v.sort(key=lambda r: r[1])
        s = np.array([r[1] for r in v], float)
        d = np.array([r[4] for r in v], float)
        brk = set(np.where(np.diff(s) > 400)[0]) | set(np.where(np.abs(np.diff(d)) > 0.05)[0])
        for sg in np.split(np.arange(len(v)), sorted(x + 1 for x in brk)):
            if len(sg) < s68.MIN_ARC:
                continue
            dv = d[sg]
            if not np.isfinite(dv).all() or np.std(dv) > 0.5:
                continue
            med = np.median(dv)
            for k, idx in enumerate(sg):
                r = v[idx]
                keep.append((r[0], r[1], r[2], r[4] - med, r[5], r[6], r[7], r[8], r[9]))
    return keep


def fit(DD, TT, WW, GN, GE, arc_id):
    X = np.column_stack([TT, WW, GN, GE])
    try:
        coef, *_ = np.linalg.lstsq(X, DD, rcond=None)
    except np.linalg.LinAlgError:
        return None
    nulls = []
    arcs = np.unique(arc_id)
    for _ in range(400):
        Ts = TT.copy()
        for aid in arcs:
            sel = np.where(arc_id == aid)[0]
            if len(sel) > 3:
                Ts[sel] = np.roll(TT[sel], int(RNG.integers(1, len(sel))))
        try:
            cn, *_ = np.linalg.lstsq(np.column_stack([Ts, WW, GN, GE]), DD, rcond=None)
            nulls.append(cn[0])
        except np.linalg.LinAlgError:
            pass
    nulls = np.array(nulls)
    z = float((coef[0] - nulls.mean()) / (nulls.std() + 1e-30))
    return {"slope": float(coef[0]), "z": z, "rms_mm": float(np.std(DD) * 1000)}


def run():
    atx = s68.load_atx()
    dense = {d: s68.grid_for(d) for d in DOYS}
    results = []
    for a, b in PAIRS:
        allrec, blen = [], None
        for d in DOYS:
            if dense.get(d) is None:
                continue
            r = pair_records(a, b, d, dense[d], atx)
            if r:
                allrec.extend(r[0]); blen = r[1]
        if not allrec or blen is None:
            continue
        rec = arcs_and_ambiguity(allrec)
        if len(rec) < 800:
            continue
        doy = np.array([r[0] for r in rec]); prn = np.array([r[2] for r in rec])
        DD = np.array([r[3] for r in rec], float); TT = np.array([r[4] for r in rec], float)
        WW = np.array([r[5] for r in rec], float); GN = np.array([r[6] for r in rec], float)
        GE = np.array([r[7] for r in rec], float); SID = np.array([r[8] for r in rec], float)
        m = np.isfinite(DD) & (np.abs(DD) < 0.3)
        doy, prn, DD, TT, WW, GN, GE, SID = (x[m] for x in (doy, prn, DD, TT, WW, GN, GE, SID))
        if len(DD) < 800:
            continue
        arc_id = np.array([f"{p}_{d}" for p, d in zip(prn, doy)])
        _, arc_idx = np.unique(arc_id, return_inverse=True)

        # LEAVE-ONE-DAY-OUT sidereal multipath template
        sbin = (SID // SID_BIN_S).astype(int)
        key = np.array([f"{p}_{s}" for p, s in zip(prn, sbin)])
        clean = DD.copy()
        has = np.zeros(len(DD), bool)
        from collections import defaultdict
        grp = defaultdict(list)
        for i, k in enumerate(key):
            grp[k].append(i)
        for k, idxs in grp.items():
            if len(idxs) < MIN_DAYS_PER_BIN:
                continue
            ds = doy[idxs]
            for i in idxs:
                other = [j for j in idxs if doy[j] != doy[i]]
                if len(other) < MIN_DAYS_PER_BIN - 1:
                    continue
                clean[i] = DD[i] - np.median(DD[other])
                has[i] = True
        if has.sum() < 500:
            continue
        sid_frac = float(1.0 - np.var(clean[has]) / max(np.var(DD[has]), 1e-30))
        raw = fit(DD[has], TT[has], WW[has], GN[has], GE[has], arc_idx[has])
        cln = fit(clean[has], TT[has], WW[has], GN[has], GE[has], arc_idx[has])
        if not raw or not cln:
            continue
        results.append({
            "pair": f"{a}-{b}", "baseline_km": blen, "n_obs": int(has.sum()),
            "sidereal_repeating_variance_fraction": sid_frac,
            "raw": raw, "multipath_cleaned": cln,
        })
        print(f"  {a}-{b:<6} {blen:8.3f} km  sidereal-repeating {sid_frac*100:5.1f}%  "
              f"raw z {raw['z']:+7.1f} slope {raw['slope']:+.3f}  ->  "
              f"cleaned z {cln['z']:+7.1f} slope {cln['slope']:+.3f}  "
              f"(rms {raw['rms_mm']:.1f} -> {cln['rms_mm']:.1f} mm)", flush=True)

    tests = [r for r in results if r["baseline_km"] >= 1.0]
    ctrl = [r for r in results if r["baseline_km"] < 0.05]
    verdict = {}
    if len(tests) >= 4:
        bl = np.array([r["baseline_km"] for r in tests])
        for tag in ("raw", "multipath_cleaned"):
            sl = np.abs([r[tag]["slope"] for r in tests])
            nz = np.array([r[tag]["rms_mm"] for r in tests])
            def sp(x, y):
                return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
            verdict[f"{tag}_slope_vs_baseline_spearman"] = sp(bl, sl)
            verdict[f"{tag}_slope_vs_noise_spearman"] = sp(nz, sl)
            verdict[f"{tag}_n_detected"] = int(sum(1 for r in tests if abs(r[tag]["z"]) > 5))
        verdict["median_sidereal_repeating_fraction"] = float(
            np.median([r["sidereal_repeating_variance_fraction"] for r in tests]))
        verdict["controls_null_after_cleaning"] = bool(
            all(abs(r["multipath_cleaned"]["z"]) < 5 for r in ctrl)) if ctrl else None
        vb = verdict.get("multipath_cleaned_slope_vs_baseline_spearman", 0.0)
        vn = verdict.get("multipath_cleaned_slope_vs_noise_spearman", 0.0)
        verdict["holonomy_detected"] = bool(vb > 0.6 and vb > vn
                                            and verdict["multipath_cleaned_n_detected"] >= 3)
    return {"pairs": results, "verdict": verdict}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("SIDEREAL MULTIPATH SEPARATION (leave-one-day-out template)")
    res = run()
    OUT.write_text(json.dumps({
        "step": 69,
        "title": "Sidereal separation of multipath from the closed-loop (holonomy) residual",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "principle": ("GPS geometry repeats every 86164.0905 s, so multipath repeats with it "
                      "while a proper-time field effect does not; the template is built "
                      "leave-one-day-out so no day cleans itself"),
        "result": res}, indent=2))
    v = res["verdict"]
    print("\nVERDICT")
    for k, val in v.items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
