#!/usr/bin/env python3
"""
STEP 108 -- The phase-cos Gram uplift on top of the distance kernel.

Step 107 established that the live rig must krige the network's ACCUMULATED
clock offset (not its instantaneous residual level), and calibrated a
distance-only Gaussian kernel for it.  This step measures what the Step 96
spectral Gram adds on top, because the answer decides whether the live rig
needs multi-day network history.

Held-out days 2024:022-050, 4 h horizon, causal Gram from the 5 strictly-prior
days, distance kernel lambda = 1500 km / nugget 0.70 (Step 107's calibration):

    target      distance only     distance x predictive_gram
    ---------   --------------    --------------------------
    ALL           +38.9 %                  +53.6 %
    ATOMIC        +57.2 %                  +68.1 %
    QUARTZ        +15.6 %                  +34.1 %     <-- the rig's class

The Gram MORE THAN DOUBLES the gain for a quartz target.  It also removes the
need for screening attenuation: alpha* rises from 0.62 (distance-only, Step
107) to ~1.08 with the Gram, in every clock class.  That is the tell that the
0.62 was never a property of quartz clocks -- it was compensating for a
distance-only kernel putting weight on stations that do not actually couple to
the target.  Weight them by measured spectral coherence instead and the
full-amplitude transfer is correct.

Two traps, both of which produce a CATASTROPHIC result rather than a mildly
worse one, and both of which cost a full debugging cycle here:

  * use `predictive_gram` (real part of the complex coherence), NOT
    `phase_cos`.  Kriging with phase_cos scores -69.5 %.
  * stack DETRENDED daily residuals.  Stacking raw days leaves a datum step at
    every day boundary, which corrupts the spectrum.

History requirement (the reason this is not yet live): 3 prior days recovers
+32.6 % for quartz, essentially the full 5-day benefit, but the live kriging
server keys station_clocks by SECOND-OF-DAY, so it silently overwrites across
midnight and can hold at most 24 h -- and only in memory.  Enabling the Gram on
the rig needs (a) re-keying that history to absolute time, (b) persisting it
across restarts, (c) a Gram endpoint, (d) alpha ~ 1.07 when the Gram is used.

Run:  python3 scripts/steps/step_108_phase_gram_uplift.py
      GD=3 python3 scripts/steps/step_108_phase_gram_uplift.py   # 3-day Gram
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "steps"))
from scripts.lib.tep_phase import spectral_phase_gram
from step_107_live_clock_prior_calibration import (  # noqa: E402
    load_day, gc_km, rolling_fit, kern)

LAM, NUG, FORM, NPRE, R2, H, DNR = 1500.0, 0.70, "gauss", 48, 0.97, 48, 5.0
GRAM_DAYS = int(__import__("os").environ.get("GD", "5"))
LF_HI = 500e-6
LF_LO = 1.0 / (GRAM_DAYS * 86400.0)
DT_S = 300.0
MIN_COMMON = 8


def _detrended_day(doy, names):
    """One day's clock residual per station, linearly detrended -- the datum
    and drift removed so days can be stacked without a step at the seam."""
    s, _ = load_day(doy)
    out = {}
    for n in names:
        v = s.get(n)
        if v is None:
            continue
        m = np.isfinite(v)
        if m.sum() < 200:
            continue
        xi = np.arange(len(v), dtype=float)
        y = v.copy()
        y[~m] = np.interp(xi[~m], xi[m], v[m])
        out[n] = y - np.polyval(np.polyfit(xi, y, 1), xi)
    return out


def build_gram(doy, names):
    mats, common = [], None
    for k in range(GRAM_DAYS + 3, 0, -1):
        if len(mats) >= GRAM_DAYS:
            break
        d = doy - k
        if d < 1:
            continue
        col = _detrended_day(d, names)
        if len(col) < MIN_COMMON:
            continue
        common = set(col) if common is None else (common & set(col))
        mats.append(col)
    if not common or len(common) < MIN_COMMON or len(mats) < 3:
        return None
    order = sorted(common)
    Y = np.vstack([np.column_stack([c[n] for n in order]) for c in mats])
    keep, cols = [], []
    for j, nm in enumerate(order):
        y = Y[:, j]
        if not np.isfinite(y).all():
            continue
        if not (1e-4 < np.std(y) < 1e6):
            continue
        cols.append(y); keep.append(nm)
    if len(keep) < MIN_COMMON:
        return None
    try:
        g = spectral_phase_gram(np.column_stack(cols), DT_S,
                                f_lo_hz=LF_LO, f_hi_hz=LF_HI)
    except ValueError as e:
        return None
    return keep, g["predictive_gram"], g["phase_cos"]


import json as _json
_CT = _json.loads((ROOT/"data"/"meta"/"igs_clock_type_full.json").read_text())
def _cls(nm):
    v = _CT.get(nm)
    return v[0] if v else "UNKNOWN"


def score(doys):
    acc = {}
    def _bump(cls_, key, err2):
        a = acc.setdefault((cls_, key), [0.0, 0]); a[0] += err2; a[1] += 1
    # alpha* accumulators per class for the gram variant
    ab = {}
    used = 0
    for doy in doys:
        s, geo = load_day(doy)
        names = sorted(s)
        if len(names) < 10:
            continue
        gg = build_gram(doy, names)
        if gg is None:
            continue
        gnames, PG, PC = gg
        gidx = {n: i for i, n in enumerate(gnames)}
        names = [n for n in names if n in gidx]
        if len(names) < 10:
            continue
        used += 1
        X = np.column_stack([s[n] for n in names])
        n_st = len(names)
        D = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(i+1, n_st):
                D[i, j] = D[j, i] = gc_km(geo[names[i]], geo[names[j]])
        gi = np.array([gidx[n] for n in names])
        PGm, PCm = PG[np.ix_(gi, gi)], PC[np.ix_(gi, gi)]
        rate, span, dnr, amp = rolling_fit(X, NPRE, R2)
        n_ep = X.shape[0]
        for ti in range(NPRE+1, n_ep-H-1, 4):
            a0, a1 = X[ti-1, :], X[ti+H, :]
            fin = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(rate[ti, :])
            if fin.sum() < 8:
                continue
            cs = a1 - a0
            gt = (dnr[ti, :] > DNR).astype(float)
            csg = cs - gt*rate[ti, :]*H
            lr = np.clip(rate[ti, :]*H, -8*span[ti, :], 8*span[ti, :])
            for b in np.flatnonzero(fin):
                pool = np.array([j for j in np.flatnonzero(fin) if j != b])
                if len(pool) < 6:
                    continue
                t = float(cs[b]); base = gt[b]*lr[b]
                if abs(t) > 1e4:
                    continue
                rr = kern(D[np.ix_(pool, pool)], LAM, FORM)
                rt = kern(D[b, pool], LAM, FORM)
                variants = {
                    "rho": (rr, rt),
                    "rho_pgram": (rr*PGm[np.ix_(pool, pool)], rt*PGm[b, pool]),
                    "rho_phasecos": (rr*PCm[np.ix_(pool, pool)], rt*PCm[b, pool]),
                    "pgram_only": (PGm[np.ix_(pool, pool)], PGm[b, pool]),
                }
                preds, ok = {}, True
                for k_, (A, bv) in variants.items():
                    try:
                        w = np.linalg.solve(A + NUG*np.eye(len(pool)), bv)
                    except np.linalg.LinAlgError:
                        ok = False; break
                    pv = base + float(w @ csg[pool])
                    if not np.isfinite(pv) or abs(pv) > 1e5:
                        ok = False; break
                    preds[k_] = pv
                if not ok:
                    continue
                c_ = _cls(names[b])
                for cc in (c_, "ALL"):
                    _bump(cc, "free", t*t)
                    _bump(cc, "persist", (t-base)**2)
                    for k_, pv in preds.items():
                        _bump(cc, k_, (t-pv)**2)
                    # alpha* on the gram spatial term
                    sp_ = preds["rho_pgram"] - base
                    r_ = t - base
                    e = ab.setdefault(cc, [0.0, 0.0])
                    e[0] += r_*sp_; e[1] += sp_*sp_
    print(f"days scored with a causal Gram: {used}")
    for cls_ in ("ALL", "ATOMIC", "QUARTZ"):
        if (cls_, "free") not in acc: continue
        fv, fn = acc[(cls_, "free")]
        if not fn: continue
        fr = math.sqrt(fv/fn)
        pv_, pn = acc[(cls_, "persist")]; pr = math.sqrt(pv_/pn)
        print(f"\n--- {cls_}  (n={fn}) ---")
        for k_ in ("persist", "rho", "rho_pgram", "pgram_only"):
            v, n = acc.get((cls_, k_), [0, 0])
            if not n: continue
            r = math.sqrt(v/n)
            print(f"  {k_:14s} rmse={r:8.4f} m  vs free {100*(1-r/fr):+7.2f}%  "
                  f"vs persist {100*(1-r/pr):+7.2f}%")
        num, den = ab.get(cls_, [0.0, 0.0])
        if den > 0:
            print(f"  alpha* (gram spatial term) = {num/den:.3f}")


score(list(range(22, 51)))
