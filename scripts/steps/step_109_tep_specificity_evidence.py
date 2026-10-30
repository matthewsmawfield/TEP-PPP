#!/usr/bin/env python3
"""
STEP 109 -- Separating a spatial FIELD from mundane common mode.

Steps 107/108 showed the live rig's clock prior works and how to weight it.
They do NOT show the gain is TEP.  The weakness is visible in Step 107's own
numbers: an UNWEIGHTED network mean already captures ~85 % of the skill
(6.248 m against 5.947 m for real geometry).  The first objection is that this
is shared satellite-clock, ephemeris and tropospheric error -- correlated
noise, not a field.  Two tests address it, on held-out days 2024:022-050.

T1  PRECISE-CLOCK REPLICATION.  Redo the transfer on precise-product clocks,
    target AND network consistently (mixing datums is the Step 99 units
    error).  Precise products remove satellite clock and orbit error outright.

        SPP clocks      +37.7 %
        PRECISE clocks  +27.0 %

    Roughly a quarter of the SPP skill WAS satellite clock/orbit common mode
    -- worth knowing -- but 27 % survives their complete removal, so they
    cannot be the mechanism.

T2  COMMON-MODE-REMOVED TRANSFER.  Subtract the network mean from the
    predictors AND the target.  A purely global common mode is then
    identically zero and can contribute no skill; whatever survives is
    spatial structure.  Scoring against a distance-PERMUTED control:

        clocks    TEP-specific skill      95 % CI        permuted   days +ve
        SPP           +4.57 %        [+4.01, +5.14]      -1.67 %     29/29
        PRECISE       +3.37 %        [+2.92, +3.85]      -1.47 %     29/29

    Day-clustered bootstrap, 2000 draws.  Every one of the 29 held-out days is
    positive (median +3.51 % on precise clocks, minimum +1.10 %), and
    scrambling which station sits at which distance does not merely erase the
    skill, it makes the prior actively harmful.

Read this honestly.  The TEP-ATTRIBUTABLE component is small -- a few percent
-- and most of the raw operational gain is common mode.  What T2 establishes
is that the small part is real, geometry-dependent, and survives on clocks
with satellite errors removed.  The large part is operationally valuable but
is not, on this evidence, distinguishable from correlated noise.

Run:  python3 scripts/steps/step_109_tep_specificity_evidence.py
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "steps"))
from step_107_live_clock_prior_calibration import (
    gc_km, rolling_fit, kern, repair_jumps, ecef_to_ll, GRID)

DENSE = ROOT / "data" / "ppp_dense"
C = 0.299792458
LAM, NUG, FORM, NPRE, R2, H, DNR = 1500.0, 0.70, "gauss", 48, 0.97, 48, 5.0
RNG = np.random.default_rng(4242)


def load_day_field(doy, field, year=2024):
    series, geo = {}, {}
    for f in sorted(DENSE.glob(f"*_{year}{doy:03d}.npz")):
        d = np.load(f, allow_pickle=True)
        nm = str(d["station"])
        if field not in d.files:
            continue
        y = np.asarray(d[field], float) * C
        t = np.asarray(d["timestamps"], float)
        # Some stations carry a precise array a couple of samples shorter than
        # the timestamp axis; align on the common prefix rather than drop them.
        k = min(len(y), len(t))
        if k < 200:
            continue
        y, t = y[:k], t[:k]
        m = np.isfinite(y) & np.isfinite(t)
        if m.sum() < 200:
            continue
        gy = np.interp(GRID, t[m], repair_jumps(y[m]), left=np.nan, right=np.nan)
        if np.isfinite(gy).sum() < 200:
            continue
        if np.nanmax(gy) - np.nanmin(gy) < 1e-6:
            continue
        series[nm] = gy
        geo[nm] = ecef_to_ll(float(d["x_mean"]), float(d["y_mean"]), float(d["z_mean"]))
    a = {nm: float(np.sqrt(np.nanmean((v[12:] - v[:-12]) ** 2)))
         for nm, v in series.items()}
    if len(a) >= 6:
        vals = np.array(list(a.values())); med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-9
        keep = {nm for nm, v in a.items() if abs(v - med) <= max(6 * mad, 0.5 * med)}
        series = {k: v for k, v in series.items() if k in keep}
        geo = {k: v for k, v in geo.items() if k in keep}
    return series, geo


def run(doys, field, cm_removed=False, perm=False):
    se = {k: 0.0 for k in ("free", "persist", "tep", "netmean")}
    n = 0
    for doy in doys:
        s, g = load_day_field(doy, field)
        names = sorted(s)
        if len(names) < 10:
            continue
        X = np.column_stack([s[n_] for n_ in names])
        n_st = len(names)
        D = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(i + 1, n_st):
                D[i, j] = D[j, i] = gc_km(g[names[i]], g[names[j]])
        if perm:
            idx = RNG.permutation(n_st); D = D[np.ix_(idx, idx)]
        rate, span, dnr, amp = rolling_fit(X, NPRE, R2)
        n_ep = X.shape[0]
        for ti in range(NPRE + 1, n_ep - H - 1, 4):
            a0, a1 = X[ti - 1, :], X[ti + H, :]
            fin = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(rate[ti, :])
            if fin.sum() < 8:
                continue
            cs_all = a1 - a0
            gt = (dnr[ti, :] > DNR).astype(float)
            csg_all = cs_all - gt * rate[ti, :] * H
            lr = np.clip(rate[ti, :] * H, -8 * span[ti, :], 8 * span[ti, :])
            for b in np.flatnonzero(fin):
                pool = np.array([j for j in np.flatnonzero(fin) if j != b])
                if len(pool) < 6:
                    continue
                base = gt[b] * lr[b]
                truth = float(cs_all[b]) - base      # what the network must predict
                csg = csg_all[pool].copy()
                if abs(truth) > 1e4:
                    continue
                if cm_removed:
                    # Strip the network mean from BOTH sides.  A purely global
                    # common mode is then identically zero in the predictors
                    # and cannot contribute any skill.
                    cm = float(np.mean(csg))
                    csg = csg - cm
                    truth = truth - cm
                krr = kern(D[np.ix_(pool, pool)], LAM, FORM)
                krt = kern(D[b, pool], LAM, FORM)
                try:
                    w = np.linalg.solve(krr + NUG * np.eye(len(pool)), krt)
                except np.linalg.LinAlgError:
                    continue
                pred = float(w @ csg)
                if not np.isfinite(pred):
                    continue
                se["free"] += truth * truth          # predicting zero
                se["persist"] += truth * truth
                se["tep"] += (truth - pred) ** 2
                se["netmean"] += (truth - float(np.mean(csg))) ** 2
                n += 1
    if n == 0:
        return None
    return {k: math.sqrt(v / n) for k, v in se.items()} | {"n": n}


DOYS = list(range(22, 51))
print("=" * 74)
print("T1  PRECISE-CLOCK REPLICATION  (satellite clock+orbit error removed)")
print("=" * 74)
for field, label in (("clock_bias_ns", "SPP clock (broadcast sat clocks)"),
                     ("precise_clock_bias_ns", "PRECISE-product clock")):
    r = run(DOYS, field)
    if r is None:
        print(f"  {label:34s} no data"); continue
    print(f"  {label:34s} predict-zero={r['free']:7.3f}m  "
          f"kriged={r['tep']:7.3f}m  skill={100*(1-r['tep']/r['free']):+6.2f}%  n={r['n']}")

print()
print("=" * 74)
print("T2  COMMON-MODE-REMOVED TRANSFER  (the discriminator)")
print("=" * 74)
for field, label in (("clock_bias_ns", "SPP clock"),
                     ("precise_clock_bias_ns", "PRECISE clock")):
    r = run(DOYS, field, cm_removed=True)
    if r is None:
        continue
    rp = run(DOYS, field, cm_removed=True, perm=True)
    print(f"  {label:16s} residual RMS={r['free']:7.3f}m  kriged={r['tep']:7.3f}m  "
          f"skill={100*(1-r['tep']/r['free']):+6.2f}%")
    if rp:
        print(f"  {'':16s} distance-PERMUTED kriged={rp['tep']:7.3f}m  "
              f"skill={100*(1-rp['tep']/rp['free']):+6.2f}%   "
              f"<-- real geometry must beat this")
