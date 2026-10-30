#!/usr/bin/env python3
"""E14: does the TEP prior beat STANDARD GNSS timing holdover?

Positioning is the wrong domain for the claim: with full sky the receiver
clock is well observed, so a prior adds little, and at k=4 the gain is mostly
the removal of an unknown (geometry) rather than the prior's accuracy.

TIMING is the domain where a clock prior is the whole product.  When a timing
receiver loses sky, the standard options are free-running hold or a drift
extrapolation.  Express the comparison in NANOSECONDS against the ITU-T G.8272
PRTC-A 100 ns holdover envelope, at realistic outage lengths.
"""
import sys, math, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "steps"))
from step_107_live_clock_prior_calibration import load_day, gc_km, rolling_fit, kern
from scripts.lib.tep_phase import spectral_phase_gram

C_NS = 0.299792458          # m per ns
LAM, NUG, FORM, NPRE, R2, DNR = 1500.0, 0.70, "gauss", 48, 0.97, 5.0
DT = 300.0
ct = json.loads((ROOT/"data"/"meta"/"igs_clock_type_full.json").read_text())
cls = lambda n: (ct.get(n) or ["UNKNOWN"])[0]
PRTC_A_NS = 100.0


def gram_prev_day(doy, names, hours=24):
    s, _ = load_day(doy - 1)
    n = int(hours*3600/DT)
    cols, keep = [], []
    for nm in names:
        v = s.get(nm)
        if v is None: continue
        seg = v[-n:]; m = np.isfinite(seg)
        if m.sum() < max(8, 0.9*n): continue
        xi = np.arange(len(seg), dtype=float)
        y = seg.copy(); y[~m] = np.interp(xi[~m], xi[m], seg[m])
        y = y - np.polyval(np.polyfit(xi, y, 1), xi)
        if not (1e-4 < np.std(y) < 1e6): continue
        cols.append(y); keep.append(nm)
    if len(keep) < 8: return None
    try:
        g = spectral_phase_gram(np.column_stack(cols), DT,
                                f_lo_hz=1.0/(hours*3600.0), f_hi_hz=500e-6)
    except ValueError:
        return None
    return keep, g["predictive_gram"]


def run(doys, H, permute=False, cm_removed=False):
    acc = {}
    for doy in doys:
        s, geo = load_day(doy); names = sorted(s)
        if len(names) < 10: continue
        gg = gram_prev_day(doy, names)
        if gg is None: continue
        gnames, PG = gg
        gi = {n: i for i, n in enumerate(gnames)}
        names = [n for n in names if n in gi]
        if len(names) < 10: continue
        X = np.column_stack([s[n] for n in names]); n_st = len(names)
        D = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(i+1, n_st):
                D[i, j] = D[j, i] = gc_km(geo[names[i]], geo[names[j]])
        idx = np.array([gi[n] for n in names]); PGm = PG[np.ix_(idx, idx)]
        if permute:
            # Scramble WHICH station sits at WHICH distance and carries WHICH
            # coherence row.  Everything else -- the clock series, the kernel,
            # the gating -- is untouched.
            pr = np.random.default_rng(1234 + doy).permutation(n_st)
            D = D[np.ix_(pr, pr)]; PGm = PGm[np.ix_(pr, pr)]
        rate, span, dnr, amp = rolling_fit(X, NPRE, R2)
        for ti in range(NPRE+1, X.shape[0]-H-1, 4):
            a0, a1 = X[ti-1, :], X[ti+H, :]
            fin = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(rate[ti, :])
            if fin.sum() < 8: continue
            cs = a1-a0; gt = (dnr[ti, :] > DNR).astype(float)
            csg = cs - gt*rate[ti, :]*H
            lr = np.clip(rate[ti, :]*H, -8*span[ti, :], 8*span[ti, :])
            for b in np.flatnonzero(fin):
                pool = np.array([j for j in np.flatnonzero(fin) if j != b])
                if len(pool) < 6: continue
                truth = float(cs[b])
                if abs(truth) > 1e4: continue
                rr = kern(D[np.ix_(pool, pool)], LAM, FORM)*PGm[np.ix_(pool, pool)]
                rt = kern(D[b, pool], LAM, FORM)*PGm[b, pool]
                try: w = np.linalg.solve(rr + NUG*np.eye(len(pool)), rt)
                except np.linalg.LinAlgError: continue
                _cs = csg[pool]
                if cm_removed:
                    # Strip the network mean: a purely global common mode is
                    # then identically zero and can contribute no skill.
                    _cs = _cs - float(np.mean(_cs))
                tep = gt[b]*lr[b] + float(w @ _cs)
                if not np.isfinite(tep): continue
                c_ = cls(names[b])
                a = acc.setdefault(c_, {"free": [0.,0], "drift": [0.,0], "tep": [0.,0]})
                a["free"][0]  += truth**2;                a["free"][1]  += 1
                a["drift"][0] += (truth-gt[b]*lr[b])**2;  a["drift"][1] += 1
                a["tep"][0]   += (truth-tep)**2;          a["tep"][1]   += 1
    return acc


def _ns(acc, c_, key):
    a = acc.get(c_)
    if not a or not a[key][1]:
        return None
    return math.sqrt(a[key][0]/a[key][1])/C_NS


print("TIMING HOLDOVER — clock error in NANOSECONDS after an outage")
print("standard options: free-running hold, or drift extrapolation")
print("ITU-T G.8272 PRTC-A holdover envelope = 100 ns\n")
print(f"{'target':9s} {'outage':>7s} {'free':>9s} {'drift':>9s} {'TEP':>9s}   "
      f"{'TEP vs best standard':>21s}  {'within PRTC-A':>14s}")
print("-"*90)
for H, label in ((12, "1 h"), (48, "4 h")):
    acc = run(list(range(22, 51)), H)
    for c_ in ("ATOMIC", "QUARTZ"):
        if c_ not in acc: continue
        a = acc[c_]
        r = {k: math.sqrt(v[0]/v[1])/C_NS for k, v in a.items()}   # m -> ns
        best = min(r["free"], r["drift"])
        gain = 100*(1-r["tep"]/best)
        ok = "YES" if r["tep"] < PRTC_A_NS else "no"
        std_ok = "yes" if best < PRTC_A_NS else "NO"
        print(f"{c_:9s} {label:>7s} {r['free']:8.1f}n {r['drift']:8.1f}n "
              f"{r['tep']:8.1f}n   {gain:+19.1f}%  TEP {ok} / std {std_ok}")


# --- specificity: is the timing gain a FIELD, or just common mode? ---------
#
# The same discipline applied to the position results.  A gain over standard
# holdover only says the predictor works; it does not say the spatial/spectral
# structure is what makes it work, because a global common mode shared by every
# station would score the same.
print()
print("=" * 90)
print("SPECIFICITY OF THE TIMING GAIN (4 h outage)")
print("=" * 90)
H = 48
base = run(list(range(22, 51)), H)
perm = run(list(range(22, 51)), H, permute=True)
cmr  = run(list(range(22, 51)), H, cm_removed=True)
print(f"{'target':9s} {'standard':>9s} {'TEP':>9s} {'TEP perm':>9s} {'TEP cm-free':>12s}"
      f"   {'geometry+coh':>13s} {'field beyond CM':>16s}")
print("-" * 90)
for c_ in ("ATOMIC", "QUARTZ"):
    std = _ns(base, c_, "free"); t = _ns(base, c_, "tep")
    tp = _ns(perm, c_, "tep");   tc = _ns(cmr, c_, "tep")
    if None in (std, t, tp, tc):
        continue
    g_struct = 100*(1 - t/tp)          # real structure vs scrambled structure
    g_beyond = 100*(1 - tc/std)        # skill surviving common-mode removal
    print(f"{c_:9s} {std:8.1f}n {t:8.1f}n {tp:8.1f}n {tc:11.1f}n"
          f"   {g_struct:+12.1f}% {g_beyond:+15.1f}%")
print()
print("geometry+coh : how much the TRUE station structure beats a scrambled one")
print("field beyond CM: skill left after the network mean is removed entirely")
