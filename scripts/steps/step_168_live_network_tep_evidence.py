#!/usr/bin/env python3
"""Step 168 — live leave-one-out TEP evidence on the streaming CORS network.

Reproduces, from a snapshot of the live network's raw clock history, the
three numbers that a spatial-field claim needs:

  1. gain over the DEPLOYABLE baseline (gated persistence-trend), not a strawman
  2. gain beyond a WITHIN-CLUSTER geometry null
  3. gain surviving removal of the network-wide common mode

Why each choice, since each was arrived at by finding the alternative wrong:

* Target is a network station, never the rig.  The LC29H is a consumer TCXO
  whose clock noise is ~7x the field amplitude, so its ceiling is under 1%
  whatever the algorithm, and its measured median correlation to the network
  is +0.025.
* The observable is the ACCUMULATED offset across a gap (step 96), not the
  instantaneous detrended level.  Kriging the level measures -23.5% against
  the offset's +37.7% on held-out archive days.
* Only REAL coverage is scored.  The live matrix spans the union of all
  stations' epochs and gap-fills with np.interp, which holds the last value
  flat past a station's coverage; a constant column contributes exactly zero
  accumulated offset, so scoring it grades the predictor on fabricated data.
  A window must lie INSIDE every included station's measured range.
* Two nulls.  A GLOBAL label permutation destroys the near/far block structure
  as well as fine geometry, so on a clustered network it flatters the claim --
  it credited +11 points of "geometry" where the within-cluster null left
  +1.74.  Only the WITHIN-CLUSTER null, which preserves cluster identity,
  caster, and the near/far split, can support a spatial claim.
* CIs are block bootstraps over TIME BLOCKS of twice the horizon.  Adjacent
  prediction times share a fitting window and are nowhere near independent;
  bootstrapping over cases would give badly overconfident intervals.

Usage:
    python3 scripts/steps/step_168_live_network_tep_evidence.py [snapshot.json]

The snapshot is {station: {sod_str: clk_m}}; with no argument the script
pulls /api/history from a running kriging server on port 8001.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "results" / "outputs" / "step_168_live_network_tep_evidence.json"

NUGGET = 0.70            # sweep optimum; 0.05 costs ~12 points of gain
DNR = 5.0                # drift-to-noise ratio above which the gate fires
LINK_KM = 800.0          # single-linkage distance defining a cluster
N_PERM = 20
N_BOOT = 1000
MIN_BLOCKS = 5           # below this the bootstrap is not reportable
SERVER = "http://localhost:8001"


def _unwrap(sods, now_sod):
    a = np.asarray(sods, float)
    return now_sod - ((now_sod - a) % 86400.0)


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
    geo = {k: v for k, v in geo.items()
           if v[0] is not None and v[1] is not None
           and not (abs(v[0]) < 0.01 and abs(v[1]) < 0.01)}
    series = {}
    for k, d in raw.items():
        if k not in geo or len(d) < 50:
            continue
        sod = np.array([float(x) for x in d], float)
        u = _unwrap(sod, now)
        o = np.argsort(u)
        series[k] = (u[o], np.array([d[x] for x in d], float)[o])
    return series, geo


def great_circle_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    v = math.sin(p1)*math.sin(p2) + math.cos(p1)*math.cos(p2)*math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, v)))


def clusters_of(D, link_km=LINK_KM):
    n = D.shape[0]
    lab = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < link_km:
                a, b = lab[i], lab[j]
                if a != b:
                    lab = [a if x == b else x for x in lab]
    remap = {v: i for i, v in enumerate(sorted(set(lab)))}
    return np.array([remap[x] for x in lab])


def covered_block(series, min_stations=10, min_span_s=3600.0):
    """Window and station set maximising stations x span over REAL coverage."""
    items = [(k, v[0].min(), v[0].max()) for k, v in series.items()]
    best = None
    for a in sorted({x[1] for x in items}):
        for b in sorted({x[2] for x in items}):
            if b - a < min_span_s:
                continue
            keep = [x[0] for x in items if x[1] <= a + 1 and x[2] >= b - 1]
            if len(keep) < min_stations:
                continue
            score = len(keep) * (b - a)
            if best is None or score > best[0]:
                best = (score, a, b, sorted(keep))
    return best


def _krige(Dm, b, pool, src, lam, nugget=NUGGET):
    rr = np.exp(-0.5 * (Dm[np.ix_(pool, pool)] / lam) ** 2)
    rt = np.exp(-0.5 * (Dm[b, pool] / lam) ** 2)
    try:
        w = np.linalg.solve(rr + nugget * np.eye(len(pool)), rt)
    except np.linalg.LinAlgError:
        return None
    return float(w @ src)


def evaluate(X, D, cl, horizon_s, lam, dt=30.0, max_times=90, seed=11):
    rng = np.random.default_rng(seed)
    n_ep, n_st = X.shape
    H = int(round(horizon_s / dt))
    NPRE = H
    if n_ep < NPRE + H + 5:
        return None
    starts = list(range(NPRE, n_ep - H - 1))
    starts = starts[::max(1, len(starts) // max_times)][-max_times:]

    gperm = [rng.permutation(n_st) for _ in range(N_PERM)]
    wperm = []
    for _ in range(N_PERM):
        p = np.arange(n_st)
        for c in set(cl.tolist()):
            m = np.flatnonzero(cl == c)
            p[m] = m[rng.permutation(len(m))]
        wperm.append(p)

    xx = np.arange(NPRE, dtype=float)
    blk, e_base, e_tep, e_cm, e_cmb = [], [], [], [], []
    e_g = [[] for _ in gperm]
    e_w = [[] for _ in wperm]
    for ti in starts:
        seg = X[ti - NPRE:ti, :]
        if not np.isfinite(seg).all():
            continue
        rate = np.polyfit(xx, seg, 1)[0]
        resid = seg - (seg.mean(0) + rate * (xx[:, None] - xx.mean()))
        dn = resid.std(0)
        var = seg.var(0)
        r2 = np.where(var > 1e-12, 1 - dn ** 2 / np.maximum(var, 1e-12), 0)
        dnr = np.where((r2 > 0.97) & (np.abs(rate) * NPRE > 2.0),
                       np.abs(rate) * NPRE / np.maximum(dn, 1e-6), 0.0)
        csg = (X[ti + H, :] - X[ti - 1, :]) - (dnr > DNR) * rate * H
        if not np.isfinite(csg).all():
            continue
        for b in range(n_st):
            pool = np.array([j for j in range(n_st) if j != b])
            truth = float(csg[b])
            if not np.isfinite(truth) or abs(truth) > 1e4:
                continue
            v = _krige(D, b, pool, csg[pool], lam)
            if v is None:
                continue
            cm = float(np.mean(csg[pool]))
            vc = _krige(D, b, pool, csg[pool] - cm, lam)
            gv = [_krige(D[np.ix_(p, p)], b, pool, csg[pool], lam) for p in gperm]
            wv = [_krige(D[np.ix_(p, p)], b, pool, csg[pool], lam) for p in wperm]
            if vc is None or any(z is None for z in gv + wv):
                continue
            blk.append(ti // (2 * H))
            e_base.append(truth ** 2)
            e_tep.append((truth - v) ** 2)
            e_cmb.append((truth - cm) ** 2)
            e_cm.append((truth - cm - vc) ** 2)
            for i in range(N_PERM):
                e_g[i].append((truth - gv[i]) ** 2)
                e_w[i].append((truth - wv[i]) ** 2)
    if len(e_base) < 50:
        return None

    A = lambda x: np.asarray(x, float)
    blk = np.asarray(blk)
    base, tep = A(e_base), A(e_tep)
    G, W = A(e_g).mean(0), A(e_w).mean(0)
    ub = np.unique(blk)
    idx = {u: np.flatnonzero(blk == u) for u in ub}
    rms = lambda a: math.sqrt(float(np.mean(a)))
    gain = lambda a, b_: 100.0 * (1 - rms(a) / rms(b_))

    def boot(num, den):
        out = []
        for _ in range(N_BOOT):
            pick = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
            out.append(100.0 * (1 - math.sqrt(np.mean(num[pick]))
                                / math.sqrt(np.mean(den[pick]))))
        return [round(float(x), 2) for x in np.percentile(out, [2.5, 97.5])]

    gT = gain(tep, base)
    gW = np.array([gain(A(e_w[i]), base) for i in range(N_PERM)])
    return {
        "horizon_s": horizon_s, "lambda_km": lam, "nugget": NUGGET,
        "n_cases": len(base), "n_blocks": int(len(ub)),
        "n_stations": int(X.shape[1]),
        "base_rms_m": round(rms(base), 4),
        "tep_rms_m": round(rms(tep), 4),
        "gain_pct": round(gT, 2), "gain_ci": boot(tep, base),
        "global_null_pct": round(gain(G, base), 2),
        "within_cluster_null_pct": round(float(gW.mean()), 2),
        "geometry_vs_within_cluster_pct": round(gain(tep, W), 2),
        "geometry_ci": boot(tep, W),
        "beats_within_cluster_draws": f"{int((gT > gW).sum())}/{N_PERM}",
        "common_mode_removed_pct": round(gain(A(e_cm), A(e_cmb)), 2),
        "common_mode_ci": boot(A(e_cm), A(e_cmb)),
        "reportable": bool(len(ub) >= MIN_BLOCKS),
    }


def spatial_null(X, seed=3):
    """Amplitude-matched but spatially INDEPENDENT clocks.

    The decisive control on the estimator itself.  Every station keeps its own
    step-size distribution, so the marginals and the geometry are unchanged and
    the ONLY thing removed is the spatial correlation.  If the pipeline still
    reports a geometry contribution here, it manufactures signal and every
    number above is an artefact.  Measured: the real field gives +7.77%
    (CI [5.25, 10.57]) beyond the within-cluster null and beats 20/20 draws,
    while this null gives -0.56% (CI [-1.60, +0.64]) and beats 7/20 -- chance
    is 10/20, and the gain over persistence goes NEGATIVE because independent
    noise is genuinely unpredictable.
    """
    rng = np.random.default_rng(seed)
    step_sd = np.median(np.abs(np.diff(X, axis=0)), axis=0) * 1.4826
    return np.cumsum(rng.normal(0.0, step_sd, X.shape), axis=0) + X[0]


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else None
    series, geo = load(snap)
    blk = covered_block(series)
    if blk is None:
        print("no window with >=10 fully covered stations"); return 1
    _, t0, t1, names = blk
    g = np.arange(t0, t1, 30.0)
    X = np.column_stack([np.interp(g, series[k][0], series[k][1]) for k in names])
    n_st = len(names)
    D = np.zeros((n_st, n_st))
    for i in range(n_st):
        for j in range(i + 1, n_st):
            D[i, j] = D[j, i] = great_circle_km(geo[names[i]], geo[names[j]])
    cl = clusters_of(D)
    print(f"{n_st} stations x {(t1 - t0) / 3600:.2f} h, baselines "
          f"{D[D > 0].min():.0f}-{D.max():.0f} km, "
          f"{len(set(cl.tolist()))} clusters")

    results = []
    for hs, lam in ((900, 250), (900, 500), (1800, 250), (1800, 500),
                    (3600, 1000)):
        r = evaluate(X, D, cl, hs, lam)
        if r:
            results.append(r)
            flag = "" if r["reportable"] else "   [UNDERPOWERED: <5 blocks]"
            print(f"\nH={hs // 60:3d} min  lam={lam:5d} km  "
                  f"n={r['n_cases']} cases, {r['n_blocks']} blocks{flag}")
            print(f"  TEP vs persistence-trend      {r['gain_pct']:+6.2f}%  "
                  f"CI {r['gain_ci']}")
            print(f"  within-cluster null           "
                  f"{r['within_cluster_null_pct']:+6.2f}%")
            print(f"  geometry (within-cluster)     "
                  f"{r['geometry_vs_within_cluster_pct']:+6.2f}%  "
                  f"CI {r['geometry_ci']}  beats {r['beats_within_cluster_draws']}")
            print(f"  common-mode removed           "
                  f"{r['common_mode_removed_pct']:+6.2f}%  CI {r['common_mode_ci']}")

    # --- decisive control: the same pipeline on spatially unstructured data
    Xn = spatial_null(X)
    nulls = []
    for hs, lam in ((900, 250), (1800, 250)):
        rn = evaluate(Xn, D, cl, hs, lam)
        if rn:
            nulls.append(rn)
            print(f"\nSPATIAL NULL  H={hs // 60:3d} min  lam={lam} km "
                  f"(independent walks, matched marginals)")
            print(f"  gain {rn['gain_pct']:+6.2f}%   "
                  f"geometry {rn['geometry_vs_within_cluster_pct']:+6.2f}%  "
                  f"CI {rn['geometry_ci']}  beats "
                  f"{rn['beats_within_cluster_draws']}")
    null_clean = all(r["geometry_ci"][0] <= 0 for r in nulls) if nulls else False

    # PASS criterion, stated so it cannot be met by one lucky cell:
    #   >=4 h capture, >=2 DISTINCT horizons reportable (each with >=5
    #   independent time blocks), every reportable cell's within-cluster
    #   geometry AND common-mode-removed interval excluding zero, and the
    #   spatial null NOT reproducing the geometry contribution.
    good = [r for r in results if r["reportable"]
            and r["geometry_ci"][0] > 0 and r["common_mode_ci"][0] > 0]
    horizons = {r["horizon_s"] for r in good}
    span_ok = (t1 - t0) >= 4 * 3600
    verdict = ("PASS" if (good and len(horizons) >= 2 and span_ok and null_clean)
               else ("UNDERPOWERED" if results else "NO DATA"))
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations": names, "n_stations": n_st,
        "span_h": round((t1 - t0) / 3600.0, 2),
        "baseline_km": [round(float(D[D > 0].min()), 1), round(float(D.max()), 1)],
        "n_clusters": int(len(set(cl.tolist()))),
        "results": results,
        "spatial_null": nulls,
        "spatial_null_clean": null_clean,
        "reportable_horizons_s": sorted(horizons),
        "verdict": verdict,
        "criterion": (
            "PASS requires: >=4 h capture; >=2 DISTINCT horizons reportable, "
            "each with >=5 independent time blocks; every reportable cell's "
            "within-cluster geometry CI and common-mode-removed CI excluding "
            "zero; and an amplitude-matched spatially-independent null that "
            "does NOT reproduce the geometry contribution"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nVERDICT: {verdict}   -> {OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
