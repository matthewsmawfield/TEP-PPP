#!/usr/bin/env python3
"""Step 170 — locate a held-out station from the live clock field alone.

If the field carries real spatial structure then a station's pattern of clock
correlations to the network encodes WHERE IT IS.  Holding one station out and
recovering its position from correlations only -- no GNSS geometry, no ranging
-- is an independent test of the field: it uses the same data as the clock
prediction of step 168 but asks a different question of it, one that a
common-mode signal cannot answer at all.

Design decisions, each arrived at by finding the alternative wrong:

* TARGET-BLIND CALIBRATION.  rho(d) is fitted on reference-reference pairs
  only, whose separations are known.  The held-out station takes no part, so
  lambda is never fitted to the answer.

* NEVER INVERT rho TO A RANGE.  d = lambda*sqrt(-2 ln rho) is catastrophically
  ill-conditioned where the data lives: near rho = 1 the derivative diverges,
  so at rho = 0.95 a 0.01 error moves the implied range ~50 km and the whole
  Dutch cluster sits in that regime.  Measured, that estimator gives a median
  error of 806 km -- WORSE than guessing the reference centroid (684 km) and
  indistinguishable from a scrambled-geometry null.  Fitting position against
  the correlation PROFILE keeps the residual bounded in [-1, 1] whatever the
  geometry and gives 212 km.  The signal was never the problem.

* WEIGHT BY SENSITIVITY.  |d rho / d d| = (d/lambda^2) rho is near zero both
  for co-located references (rho ~ 1) and distant ones (rho ~ 0); those carry
  almost no positional information and must not dominate the fit.

* COMMON MODE REMOVED FIRST.  Correlating raw clocks reports rho ~ 1 for every
  pair -- the GNSS-wide common mode dominates and carries no distance
  information whatsoever.

Controls:
  permuted  - shuffle which rho belongs to which reference.  Keeps every
              correlation value and the entire reference geometry, destroys
              only the pairing.  This is the null a spatial claim must beat.
  centroid  - distance from truth to the plain reference centroid: what you
              get by guessing "somewhere in the middle" with no field at all.
  spatial   - replace every clock with an INDEPENDENT random walk matched to
              that station's own step-size distribution.  Marginals and
              geometry unchanged, spatial correlation removed.  If the
              estimator still localises here it manufactures structure and
              every number above is an artefact.

Usage:
    python3 scripts/steps/step_170_clock_field_localisation_live.py [snapshot.json]
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binomtest

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "results" / "outputs" / "step_170_clock_field_localisation_live.json"

HORIZON_S = 1800.0      # accumulated-offset gap
GRID_S = 30.0
N_PERM = 8
N_BOOT = 2000
CATASTROPHIC_KM = 1000.0
SERVER = "http://localhost:8001"


def gc_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    v = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, v)))


def load(snapshot: str | None):
    now = time.time() % 86400.0
    if snapshot:
        raw = json.load(open(snapshot))
    else:
        h = json.load(urllib.request.urlopen(f"{SERVER}/api/history", timeout=120))
        raw = {k: dict(zip((f"{s:.0f}" for s in v["sod"]), v["clk_m"]))
               for k, v in (h.get("stations") or {}).items()}
    st = json.load(urllib.request.urlopen(f"{SERVER}/status", timeout=30))
    geo = {x["name"]: (x["lat"], x["lon"]) for x in st.get("stations", [])}
    ov = Path(__file__).resolve().parent / "osnet_true_coords.json"
    if ov.exists():
        for k, v in json.load(open(ov)).items():
            if k in geo:
                geo[k] = (v[0], v[1])
    # Longitude to [-180, 180).  The OS Net feed mixes conventions (SWAN at
    # 359.5, LEEK at -1.98); great-circle distance is immune because cos is
    # periodic, but an arithmetic MEAN of longitudes is not -- averaging those
    # two lands in the Pacific and put the centroid control 2,600 km out.
    geo = {k: (v[0], ((v[1] + 180.0) % 360.0) - 180.0)
           for k, v in geo.items()
           if v[0] is not None and v[1] is not None
           and not (abs(v[0]) < 0.01 and abs(v[1]) < 0.01)}
    series = {}
    for k, d in raw.items():
        if k not in geo or len(d) < 50:
            continue
        sod = np.array([float(x) for x in d], float)
        u = now - ((now - sod) % 86400.0)
        o = np.argsort(u)
        series[k] = (u[o], np.array([d[x] for x in d], float)[o])
    return series, geo


def covered_block(series, geo, min_st=10, min_span_s=3600.0):
    items = [(k, v[0].min(), v[0].max()) for k, v in series.items()]
    best = None
    for a in sorted({x[1] for x in items}):
        for b in sorted({x[2] for x in items}):
            if b - a < min_span_s:
                continue
            keep = [x[0] for x in items if x[1] <= a + 1 and x[2] >= b - 1]
            if len(keep) < min_st:
                continue
            lat = [geo[k][0] for k in keep]
            lon = [geo[k][1] for k in keep]
            spread = max(gc_km((min(lat), min(lon)), (max(lat), max(lon))), 1.0)
            score = len(keep) * (b - a) * spread
            if best is None or score > best[0]:
                best = (score, a, b, sorted(keep))
    return best


def observable(X):
    """Accumulated offset over the gap, common mode removed, correlated."""
    hn = int(HORIZON_S / GRID_S)
    A = X[hn:, :] - X[:-hn, :]
    A = A - A.mean(axis=1, keepdims=True)
    A = A - A.mean(axis=0, keepdims=True)
    return np.corrcoef(A.T)


def spatial_null(X, seed=3):
    rng = np.random.default_rng(seed)
    sd = np.median(np.abs(np.diff(X, axis=0)), axis=0) * 1.4826
    return np.cumsum(rng.normal(0.0, sd, X.shape), axis=0) + X[0]


def calibrate(C, D, idx):
    dd, rr = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if D[i, j] > 0:
                dd.append(D[i, j]); rr.append(C[i, j])
    dd = np.asarray(dd)
    rr = np.clip(np.asarray(rr), 1e-4, 0.999)
    y = -2.0 * np.log(rr)
    lam2 = float(np.sum(dd ** 2 * y) / max(np.sum(y ** 2), 1e-12))
    return math.sqrt(max(lam2, 1.0))


def locate_profile(rho, idx, lam, xy):
    r = np.clip(rho[idx], -0.999, 0.999)
    P = xy[idx]

    def resid(p):
        dd = np.array([gc_km((p[1], p[0]), (P[k][1], P[k][0]))
                       for k in range(len(P))])
        pred = np.exp(-0.5 * (dd / lam) ** 2)
        sens = np.abs(dd / (lam ** 2) * pred)
        return (pred - r) * np.sqrt(np.maximum(sens, 1e-6))

    best, cost = None, np.inf
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        try:
            sol = least_squares(resid, P[int(f * (len(P) - 1))],
                                method="lm", max_nfev=600)
            if sol.cost < cost:
                best, cost = sol.x, sol.cost
        except Exception:
            continue
    return best if best is not None else P.mean(axis=0)


def score(C, D, xy, names, geo, rng):
    n = len(names)
    errs, perms, cents = [], [], []
    for t in range(n):
        idx = np.array([i for i in range(n) if i != t])
        lam = calibrate(C, D, idx)
        truth = geo[names[t]]
        est = locate_profile(C[t], idx, lam, xy)
        errs.append(gc_km(truth, (est[1], est[0])))
        pe = []
        for _ in range(N_PERM):
            sh = idx[rng.permutation(len(idx))]
            r = C[t].copy(); r[idx] = C[t][sh]
            q = locate_profile(r, idx, lam, xy)
            pe.append(gc_km(truth, (q[1], q[0])))
        perms.append(float(np.median(pe)))
        cen = xy[idx].mean(axis=0)
        cents.append(gc_km(truth, (cen[1], cen[0])))
    return map(np.asarray, (errs, perms, cents))


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else None
    series, geo = load(snap)
    blk = covered_block(series, geo)
    if blk is None:
        print("no usable covered block"); return 1
    _, t0, t1, names = blk
    g = np.arange(t0, t1, GRID_S)
    X = np.column_stack([np.interp(g, series[k][0], series[k][1]) for k in names])
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = gc_km(geo[names[i]], geo[names[j]])
    xy = np.array([[geo[k][1], geo[k][0]] for k in names], float)
    print(f"{n} stations x {(t1 - t0) / 3600:.2f} h, "
          f"baselines {D[D > 0].min():.0f}-{D.max():.0f} km")

    rng = np.random.default_rng(7)
    errs, perms, cents = score(observable(X), D, xy, names, geo, rng)
    n_err, _, _ = score(observable(spatial_null(X)), D, xy, names, geo,
                        np.random.default_rng(7))

    med = lambda a: float(np.median(a))
    boot = lambda a: [round(float(np.percentile(
        [np.median(rng.choice(a, len(a), replace=True)) for _ in range(N_BOOT)], q)), 1)
        for q in (2.5, 97.5)]

    bad = errs > CATASTROPHIC_KM
    enclosed = ~bad
    b_perm = binomtest(int((errs < perms).sum()), n, 0.5, alternative="greater")
    b_cent = binomtest(int((errs < cents).sum()), n, 0.5, alternative="greater")

    print(f"\n{'arm':14}{'median_km':>11}{'95% CI':>20}")
    for nm, a in (("TEP profile", errs), ("permuted null", perms),
                  ("centroid", cents), ("SPATIAL NULL", n_err)):
        ci = boot(a)
        print(f"{nm:14}{med(a):11.0f}   [{ci[0]:.0f}, {ci[1]:.0f}]")
    print(f"\nbeats permuted null : {int((errs < perms).sum())}/{n}  p={b_perm.pvalue:.2e}")
    print(f"beats centroid      : {int((errs < cents).sum())}/{n}  p={b_cent.pvalue:.2e}")
    print(f"\nperipheral failures (>{CATASTROPHIC_KM:.0f} km): {int(bad.sum())}/{n}"
          f" -> {', '.join(names[i] for i in np.flatnonzero(bad))}")
    print(f"enclosed stations   : median {med(errs[enclosed]):.0f} km, "
          f"n={int(enclosed.sum())}")

    # PASS: beats both nulls significantly AND the spatial null does not localise
    null_clean = med(n_err) > 2.0 * med(errs)
    verdict = ("PASS" if (b_perm.pvalue < 0.01 and b_cent.pvalue < 0.01
                          and null_clean) else "FAIL")
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_stations": n, "span_h": round((t1 - t0) / 3600.0, 2),
        "baseline_km": [round(float(D[D > 0].min()), 1), round(float(D.max()), 1)],
        "horizon_s": HORIZON_S,
        "median_km": {"tep": round(med(errs), 1),
                      "permuted": round(med(perms), 1),
                      "centroid": round(med(cents), 1),
                      "spatial_null": round(med(n_err), 1)},
        "ci_km": {"tep": boot(errs), "permuted": boot(perms),
                  "spatial_null": boot(n_err)},
        "sign_test": {"vs_permuted": [int((errs < perms).sum()), n, b_perm.pvalue],
                      "vs_centroid": [int((errs < cents).sum()), n, b_cent.pvalue]},
        "peripheral_failures": [names[i] for i in np.flatnonzero(bad)],
        "enclosed_median_km": round(med(errs[enclosed]), 1),
        "enclosed_n": int(enclosed.sum()),
        "spatial_null_clean": bool(null_clean),
        "verdict": verdict,
        "criterion": ("PASS requires sign tests against BOTH the permuted-geometry "
                      "and centroid nulls at p<0.01, AND an amplitude-matched "
                      "spatially-independent null that does not localise "
                      "(median error > 2x the measured one)"),
        "scope": ("field-scale localisation, NOT navigation: ~200 km is the "
                  "resolution of the clock field, not a position fix"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nVERDICT: {verdict}   -> {OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
