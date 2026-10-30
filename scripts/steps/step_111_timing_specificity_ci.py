#!/usr/bin/env python3
"""
STEP 111 -- Day-clustered CIs on the TIMING specificity result.

Step 110 showed the TEP clock prior beats standard GNSS holdover 2-3x and, more
tellingly, follows a different growth law: error ~ t^0.21 against t^0.435 for a
free-running oscillator (t^0.5 is the random-walk expectation).  This step puts
intervals on the two numbers a referee attacks.

Held-out days 2024:022-050, 4 h outage, 1500-draw day-clustered bootstrap:

                              atomic                          quartz
  structure    +40.35 % [+37.32, +43.28] 29/29   +21.97 % [+18.11, +26.04] 29/29
  beyond CM    +18.20 % [+16.89, +19.37] 29/29    +4.54 % [ +2.17,  +6.80] 25/29

  structure : real station geometry + coherence vs a scrambled assignment.
              Everything else -- clock series, kernel, gating -- is identical.
  beyond CM : skill surviving REMOVAL of the network mean, so a purely global
              common mode contributes exactly nothing.

Why timing and not position.  The TEP-attributable residual after common-mode
removal is +3.37 % in the position domain (Step 109) and +18.20 % here -- five
times larger, with a tighter relative interval.  Position forces the field to
compete against geometry: at k=4 the dominant benefit of a clock prior is the
REMOVAL OF AN UNKNOWN, not the prior's accuracy, so the field's contribution is
measured through a rank-deficient least-squares problem.  Timing has no such
confound -- the clock prior is the entire product.

Honest limits: the quartz row is marginal on the decisive control (25/29 days),
so the clean claim is an ATOMIC-clock claim; the controls were run at a 4 h
outage only; and this is archive CORS data.  The live LC29H rig measures a
receiver-to-network coherence of 0.347 with a NEGATIVE fitted transfer scale,
so none of this transfers to a consumer TCXO as configured.

Run:  python3 scripts/steps/step_111_timing_specificity_ci.py
"""
import sys, math, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "steps"))
from step_107_live_clock_prior_calibration import load_day, gc_km, rolling_fit, kern
from scripts.lib.tep_phase import spectral_phase_gram

C_NS = 0.299792458
LAM, NUG, FORM, NPRE, R2, DNR, H = 1500.0, 0.70, "gauss", 48, 0.97, 5.0, 48
DT = 300.0
ct = json.loads((ROOT/"data"/"meta"/"igs_clock_type_full.json").read_text())
cls = lambda n: (ct.get(n) or ["UNKNOWN"])[0]
RNG = np.random.default_rng(20260907)


def gram_prev_day(doy, names, hours=24):
    s, _ = load_day(doy - 1)
    n = int(hours*3600/DT); cols, keep = [], []
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
        return keep, spectral_phase_gram(np.column_stack(cols), DT,
            f_lo_hz=1.0/(hours*3600.0), f_hi_hz=500e-6)["predictive_gram"]
    except ValueError:
        return None


def per_day(doy):
    """Return {class: {arm: (sum_sq, n)}} for one day."""
    s, geo = load_day(doy); names = sorted(s)
    if len(names) < 10: return None
    gg = gram_prev_day(doy, names)
    if gg is None: return None
    gnames, PG = gg
    gi = {n: i for i, n in enumerate(gnames)}
    names = [n for n in names if n in gi]
    if len(names) < 10: return None
    X = np.column_stack([s[n] for n in names]); n_st = len(names)
    D = np.zeros((n_st, n_st))
    for i in range(n_st):
        for j in range(i+1, n_st):
            D[i, j] = D[j, i] = gc_km(geo[names[i]], geo[names[j]])
    idx = np.array([gi[n] for n in names]); PGm = PG[np.ix_(idx, idx)]
    pr = RNG.permutation(n_st)
    Dp, PGp = D[np.ix_(pr, pr)], PGm[np.ix_(pr, pr)]
    rate, span, dnr, amp = rolling_fit(X, NPRE, R2)
    out = {}
    for ti in range(NPRE+1, X.shape[0]-H-1, 4):
        a0, a1 = X[ti-1, :], X[ti+H, :]
        fin = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(rate[ti, :])
        if fin.sum() < 8: continue
        cs = a1-a0; gt = (dnr[ti, :] > DNR).astype(float)
        csg = cs - gt*rate[ti, :]*H
        for b in np.flatnonzero(fin):
            pool = np.array([j for j in np.flatnonzero(fin) if j != b])
            if len(pool) < 6: continue
            truth = float(cs[b])
            if abs(truth) > 1e4: continue
            base = gt[b]*np.clip(rate[ti, b]*H, -8*span[ti, b], 8*span[ti, b])
            vals = {}
            for tag, (DD, GG) in (("tep", (D, PGm)), ("perm", (Dp, PGp))):
                rr = kern(DD[np.ix_(pool, pool)], LAM, FORM)*GG[np.ix_(pool, pool)]
                rt = kern(DD[b, pool], LAM, FORM)*GG[b, pool]
                try: w = np.linalg.solve(rr + NUG*np.eye(len(pool)), rt)
                except np.linalg.LinAlgError: vals = None; break
                vals[tag] = base + float(w @ csg[pool])
            if vals is None: continue
            rr = kern(D[np.ix_(pool, pool)], LAM, FORM)*PGm[np.ix_(pool, pool)]
            rt = kern(D[b, pool], LAM, FORM)*PGm[b, pool]
            try: w = np.linalg.solve(rr + NUG*np.eye(len(pool)), rt)
            except np.linalg.LinAlgError: continue
            _cs = csg[pool] - float(np.mean(csg[pool]))
            vals["cmfree"] = base + float(w @ _cs)
            if not all(np.isfinite(v) for v in vals.values()): continue
            c_ = cls(names[b])
            a = out.setdefault(c_, {k: [0.0, 0] for k in ("std","tep","perm","cmfree")})
            a["std"][0] += truth**2; a["std"][1] += 1
            for k, v in vals.items():
                a[k][0] += (truth-v)**2; a[k][1] += 1
    return out


days = [d for d in (per_day(x) for x in range(22, 51)) if d]
print(f"days: {len(days)}\n")


def agg(ds, c_, k):
    num = sum(d[c_][k][0] for d in ds if c_ in d)
    n = sum(d[c_][k][1] for d in ds if c_ in d)
    return math.sqrt(num/n)/C_NS if n else None


for c_ in ("ATOMIC", "QUARTZ"):
    sub = [d for d in days if c_ in d]
    if len(sub) < 8: continue
    def stat(ds):
        std, tep = agg(ds, c_, "std"), agg(ds, c_, "tep")
        perm, cmf = agg(ds, c_, "perm"), agg(ds, c_, "cmfree")
        return (100*(1-tep/perm), 100*(1-cmf/std))
    pt = stat(sub)
    boots = [stat([sub[i] for i in RNG.integers(0, len(sub), len(sub))])
             for _ in range(1500)]
    g = np.array([b[0] for b in boots]); f = np.array([b[1] for b in boots])
    per = [stat([d]) for d in sub]
    print(f"--- {c_}  ({len(sub)} days, 4 h outage) ---")
    print(f"  structure (real vs scrambled): {pt[0]:+.2f}%  "
          f"95% CI [{np.percentile(g,2.5):+.2f}, {np.percentile(g,97.5):+.2f}]  "
          f"days positive {sum(1 for p in per if p[0]>0)}/{len(per)}")
    print(f"  beyond common mode           : {pt[1]:+.2f}%  "
          f"95% CI [{np.percentile(f,2.5):+.2f}, {np.percentile(f,97.5):+.2f}]  "
          f"days positive {sum(1 for p in per if p[1]>0)}/{len(per)}")
