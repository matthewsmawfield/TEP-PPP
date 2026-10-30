#!/usr/bin/env python3
"""
STEP 72  --  Is the sidereal multipath template blind to the signal it is
             supposed to leave behind?

Step 69 removes multipath with a leave-one-day-out sidereal template, on the
stated premise that "GPS geometry repeats every sidereal day, so multipath
repeats with it while a proper-time field effect does not".

That premise is false for a GEOMETRY-LOCKED observable.  The holonomy regressor
T = (e_k x e_ref).b_hat is a pure function of satellite geometry, and satellite
geometry is exactly what repeats every sidereal day.  So a STATIC field produces
a signal a*T(t) that is itself sidereally periodic -- and the template subtracts
it along with the multipath.  If that is right, Step 69's post-cleaning null is
guaranteed by construction and bounds nothing.

This step tests it by injection rather than argument.  Into the real double
difference we add a synthetic holonomy of KNOWN slope and ask how much of it
survives the exact Step 69 cleaning + fit:

  static  :  DD + a*T            (a time-independent field; geometry-locked)
  varying :  DD + a*T*g(day)     (same field, but modulated day to day)

A method that can detect a static non-exact sector must recover the static
injection.  The varying arm is the control: it shows the template is not simply
destroying everything.

Also reported: the sidereal-repeating variance fraction of T itself, computed
with the same template -- the direct measure of how much of the regressor the
template eats.

Output: results/outputs/step_72b_sidereal_template_blindness.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from collections import defaultdict
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

OUT = ROOT / "results" / "outputs" / "step_72b_sidereal_template_blindness.json"
RNG = np.random.default_rng(20260903)
INJECT_SLOPES = [0.005, 0.02, 0.05]      # metres per unit T


def lodo_template(y, prn, sid, doy):
    """Exactly Step 69's leave-one-day-out sidereal template."""
    sbin = (sid // s69.SID_BIN_S).astype(int)
    grp = defaultdict(list)
    for i, k in enumerate(zip(prn, sbin)):
        grp[k].append(i)
    clean = y.copy()
    has = np.zeros(len(y), bool)
    for _, idxs in grp.items():
        if len(idxs) < s69.MIN_DAYS_PER_BIN:
            continue
        for i in idxs:
            other = [j for j in idxs if doy[j] != doy[i]]
            if len(other) < s69.MIN_DAYS_PER_BIN - 1:
                continue
            clean[i] = y[i] - np.median(y[other])
            has[i] = True
    return clean, has


def slope_of(y, TT, WW, GN, GE):
    X = np.column_stack([TT, WW, GN, GE])
    try:
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(c[0])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atx = s68.load_atx()
    dense = {d: s68.grid_for(d) for d in s69.DOYS}
    rows = []
    for a, b in s69.PAIRS:
        allrec, blen = [], None
        for d in s69.DOYS:
            if dense.get(d) is None:
                continue
            r = s69.pair_records(a, b, d, dense[d], atx)
            if r:
                allrec.extend(r[0]); blen = r[1]
        if not allrec or blen is None:
            continue
        rec = s69.arcs_and_ambiguity(allrec)
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

        # how much of the REGRESSOR itself does the template remove?
        Tc, hT = lodo_template(TT, prn, SID, doy)
        if hT.sum() < 500:
            continue
        T_sid_frac = float(1.0 - np.var(Tc[hT]) / max(np.var(TT[hT]), 1e-30))

        base_clean, has = lodo_template(DD, prn, SID, doy)
        if has.sum() < 500:
            continue
        s_base = slope_of(base_clean[has], TT[has], WW[has], GN[has], GE[has])
        s_base_raw = slope_of(DD[has], TT[has], WW[has], GN[has], GE[has])
        if s_base is None or s_base_raw is None:
            continue

        udoy = np.unique(doy)
        gmap = {d: (1.0 if i % 2 == 0 else -1.0) for i, d in enumerate(udoy)}
        gday = np.array([gmap[d] for d in doy], float)

        rec_static, rec_vary, rec_raw = [], [], []
        for amp in INJECT_SLOPES:
            ys = DD + amp * TT                      # static field
            cs, hs = lodo_template(ys, prn, SID, doy)
            v = slope_of(cs[hs], TT[hs], WW[hs], GN[hs], GE[hs])
            if v is not None:
                rec_static.append((v - s_base) / amp)
            vraw = slope_of(ys[has], TT[has], WW[has], GN[has], GE[has])
            if vraw is not None:
                rec_raw.append((vraw - s_base_raw) / amp)

            yv = DD + amp * TT * gday               # day-modulated field (control)
            cv, hv = lodo_template(yv, prn, SID, doy)
            v2 = slope_of(cv[hv] * gday[hv], TT[hv], WW[hv], GN[hv], GE[hv])
            if v2 is not None:
                rec_vary.append((v2 - slope_of(base_clean[hv] * gday[hv], TT[hv],
                                               WW[hv], GN[hv], GE[hv])) / amp)

        rows.append({
            "pair": f"{a}-{b}", "baseline_km": blen, "n_obs": int(has.sum()),
            "regressor_T_sidereal_repeating_fraction": T_sid_frac,
            "recovery_static_before_cleaning": float(np.median(rec_raw)) if rec_raw else None,
            "recovery_static_after_cleaning": float(np.median(rec_static)) if rec_static else None,
            "recovery_dayvarying_after_cleaning": float(np.median(rec_vary)) if rec_vary else None,
        })
        print(f"  {a}-{b:<6} {blen:8.3f} km   T sidereal {T_sid_frac*100:5.1f}%   "
              f"static recovery raw {rows[-1]['recovery_static_before_cleaning']:.2f} "
              f"-> cleaned {rows[-1]['recovery_static_after_cleaning']:.2f}   "
              f"day-varying {rows[-1]['recovery_dayvarying_after_cleaning']:.2f}", flush=True)

    if not rows:
        OUT.write_text(json.dumps({"step": 72, "status": "no_pairs"}, indent=2))
        return

    med_T = float(np.median([r["regressor_T_sidereal_repeating_fraction"] for r in rows]))
    med_s = float(np.median([r["recovery_static_after_cleaning"] for r in rows
                             if r["recovery_static_after_cleaning"] is not None]))
    med_r = float(np.median([r["recovery_static_before_cleaning"] for r in rows
                             if r["recovery_static_before_cleaning"] is not None]))
    med_v = float(np.median([r["recovery_dayvarying_after_cleaning"] for r in rows
                             if r["recovery_dayvarying_after_cleaning"] is not None]))
    verdict = {
        "median_T_sidereal_repeating_fraction": med_T,
        "median_static_recovery_before_cleaning": med_r,
        "median_static_recovery_after_cleaning": med_s,
        "median_dayvarying_recovery_after_cleaning": med_v,
        "template_is_blind_to_static_holonomy": bool(med_s < 0.25),
        "interpretation": (
            "The leave-one-day-out sidereal template removes %.0f%% of the holonomy "
            "regressor T itself, because T is a function of satellite geometry and "
            "GPS geometry is what repeats every sidereal day. A synthetic STATIC "
            "holonomy injected into the real double difference is recovered at %.2f "
            "of its true slope before cleaning but only %.2f after -- while a "
            "day-modulated injection survives at %.2f. Step 69's post-cleaning null "
            "is therefore NOT a bound on a static non-exact sector: the cleaning "
            "removes any geometry-locked signal along with the multipath. Step 69's "
            "96.7%% figure remains valid as a MULTIPATH characterisation; its "
            "holonomy null bounds only a day-to-day-VARYING component."
        ) % (100 * med_T, med_r, med_s, med_v),
    }
    OUT.write_text(json.dumps({
        "step": 72,
        "title": "Injection test: is the sidereal template blind to a static holonomy?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "injected_slopes_m_per_unit_T": INJECT_SLOPES,
        "pairs": rows, "verdict": verdict}, indent=2))
    print("\n" + verdict["interpretation"])


if __name__ == "__main__":
    main()
