#!/usr/bin/env python3
"""
STEP 71  --  Separating per-station clock noise from the spatial field, on
             sidereal-cleaned carrier phase.

Every earlier short-baseline attempt hit the same wall: a station-pair
difference contains BOTH stations' local noise plus the field, so each pair is
only an upper bound -- and station noise varies by three orders of magnitude
across the network, which is why binning pair differences by distance (mixing
3 mm pairs with 5 m pairs) produced nothing usable.

Two things now make the decomposition tractable:

  1. Steps 69/70 showed 88-98% of the carrier-phase residual is sidereal-
     repeating multipath, removable with a leave-one-day-out sidereal template.
     That takes out the dominant nuisance.
  2. With multipath gone the remaining pair variance is well modelled as

         D_AB = n_A + n_B + 2 sigma^2 [1 - exp(-(d_AB/L)^2/2)]

     -- per-station noise plus a field structure function.  Fitting a
     PARAMETRIC F(d) (two free parameters) rather than per-bin values removes
     the degeneracy that collapsed an earlier per-bin attempt to a corner
     solution.

Efficiency: "phase minus geometry minus antenna PCV" is a PER-STATION quantity,
so it is computed once per station and every pair is then a cheap difference --
O(N) work instead of O(N^2).

Returns simultaneously:
  * n_i, each station's own clock+processing noise (a useful by-product: it
    ranks the network by clock quality);
  * sigma, the coherent field amplitude, and L, its correlation length.

If short-scale field structure exists between ~5 and 400 km it shows up as a
significant sigma with L in that range; if the field is smooth, sigma collapses
and the fit degenerates to pure station noise.

Also reported: the sidereal-repeating fraction per pair, which tests directly
whether an anomalous pair -- BRUX-REDU showed -0.2% repeatability where every
other pair showed 88-98% -- is explained by ONE station's oscillator rather
than by the baseline.

Output: results/outputs/step_71_station_field_decomposition.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
import itertools
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "s68", ROOT / "scripts" / "steps" / "step_68_synchronization_holonomy.py")
s68 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s68)

OUT = ROOT / "results" / "outputs" / "step_71_station_field_decomposition.json"
RNG = np.random.default_rng(20260902)
SIDEREAL_S = 86164.0905
DOYS = [100, 101, 102, 103, 104, 105, 106]
SID_BIN_S = 240.0
BAND = (10e-6, 500e-6)

STATIONS = ["WTZR", "WTZZ", "ZIM2", "ZIMM", "WAB2", "GOP6", "GOPE", "OBE4",
            "PADO", "VEN1", "MEDI", "BZR2", "KOS1", "WSRT", "DLF1", "BRUX",
            "REDU", "GENO", "IENG", "GRAS", "POTS", "PTBB", "GRAZ"]


def station_residuals(site, doy, dense, atx):
    """{(prn, sod): phase - geometry - PCV} for one station-day."""
    st = s68.load_station(site, doy)
    if not st:
        return None
    obs, xyz, typ, dneu = st
    out = {}
    for s in sorted(obs)[::s68.EPOCH_STRIDE]:
        for p, ph in obs[s].items():
            g = s68.sat_geom(dense, p, s, xyz)
            if g is None or g[2] < s68.ELEV_MIN:
                continue
            c = s68.pcv_correction(atx, typ, dneu, g[6], g[7])
            out[(p, s)] = ph - (c if c is not None else 0.0) - g[0]
    return (out, xyz) if out else None


def clean_pair(resA, resB, doy_of):
    """Per-arc ambiguity removal + LODO sidereal template, then the per-epoch
    satellite average = the station-pair clock difference."""
    keys = set(resA) & set(resB)
    if len(keys) < 300:
        return None
    by = defaultdict(list)
    for k in keys:
        _, p, s = k
        by[(doy_of[k], p)].append((s, resA[k] - resB[k]))
    rows = []
    for (d, p), v in by.items():
        v.sort()
        ss = np.array([x[0] for x in v], float)
        dv = np.array([x[1] for x in v], float)
        brk = set(np.where(np.diff(ss) > 400)[0]) | set(np.where(np.abs(np.diff(dv)) > 0.05)[0])
        for sg in np.split(np.arange(len(v)), sorted(x + 1 for x in brk)):
            if len(sg) < s68.MIN_ARC:
                continue
            seg = dv[sg]
            if not np.isfinite(seg).all() or np.std(seg) > 0.5:
                continue
            med = np.median(seg)
            for i in sg:
                rows.append((d, ss[i], p, dv[i] - med, (d * 86400.0 + ss[i]) % SIDEREAL_S))
    if len(rows) < 200:
        return None
    doy = np.array([r[0] for r in rows]); sod = np.array([r[1] for r in rows], float)
    prn = np.array([r[2] for r in rows]); val = np.array([r[3] for r in rows], float)
    sid = np.array([r[4] for r in rows], float)

    grp = defaultdict(list)
    for i, k in enumerate(zip(prn, (sid // SID_BIN_S).astype(int))):
        grp[k].append(i)
    clean = val.copy(); has = np.zeros(len(val), bool)
    for _, idxs in grp.items():
        if len(idxs) < 3:
            continue
        for i in idxs:
            other = [j for j in idxs if doy[j] != doy[i]]
            if len(other) >= 2:
                clean[i] = val[i] - np.median(val[other]); has[i] = True
    if has.sum() < 150:
        return None
    sid_frac = float(1.0 - np.var(clean[has]) / max(np.var(val[has]), 1e-30))

    agg = defaultdict(list)
    for i in np.where(has)[0]:
        agg[(doy[i], sod[i])].append(clean[i])
    ks = sorted(k for k, v in agg.items() if len(v) >= 3)
    if len(ks) < 100:
        return None
    t = np.array([k[0] * 86400.0 + k[1] for k in ks])
    y = np.array([np.mean(agg[k]) for k in ks])
    out = []
    for d in np.unique((t // 86400).astype(int)):
        m = ((t // 86400).astype(int) == d)
        if m.sum() < 40:
            continue
        g = np.arange(t[m].min(), t[m].max(), 120.0)
        z = np.interp(g, t[m], y[m])
        q = (g - g.mean()) / (g.std() or 1)
        z = z - np.polyval(np.polyfit(q, z, 2), q)
        F = np.fft.rfft(z); fr = np.fft.rfftfreq(len(z), 120.0)
        sel = (fr >= BAND[0]) & (fr <= BAND[1])
        if sel.sum() >= 4:
            out.append(2 * np.sum(np.abs(F[sel]) ** 2) / len(z) ** 2)
    if not out:
        return None
    return float(np.median(out)), sid_frac


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atx = s68.load_atx()
    res, xyzs, doy_of = defaultdict(dict), {}, {}
    for d in DOYS:
        dense = s68.grid_for(d)
        if dense is None:
            continue
        got = 0
        for site in STATIONS:
            r = station_residuals(site, d, dense, atx)
            if r is None:
                continue
            rr, xyz = r
            xyzs[site] = xyz
            got += 1
            for k, v in rr.items():
                # key MUST carry the day: (prn, sod) alone collides across days,
                # collapsing the seven-day stack into one and leaving the
                # leave-one-day-out sidereal template with no other days to use.
                res[site][(d,) + k] = v
                doy_of[(d,) + k] = d
        print(f"[{d}] residuals for {got} stations", flush=True)

    sites = sorted(s for s in STATIONS if len(res.get(s, {})) > 2000)
    print(f"\nusable stations: {len(sites)}", flush=True)
    pairs = []
    for a, b in itertools.combinations(sites, 2):
        d_km = float(np.linalg.norm(xyzs[a] - xyzs[b]) / 1000.0)
        if d_km > 700:
            continue
        r = clean_pair(res[a], res[b], doy_of)
        if r is None:
            continue
        D, sf = r
        pairs.append({"pair": f"{a}-{b}", "A": a, "B": b, "sep_km": d_km,
                      "D_m2": D, "rms_mm": float(np.sqrt(D) * 1000),
                      "sidereal_fraction": sf})
    print(f"usable pairs: {len(pairs)}", flush=True)
    if len(pairs) < 12:
        OUT.write_text(json.dumps({"step": 71, "status": "too_few_pairs",
                                   "pairs": pairs}, indent=2))
        return

    idx = {s: i for i, s in enumerate(sites)}
    dd = np.array([p["sep_km"] for p in pairs])
    y = np.array([p["D_m2"] for p in pairs])
    M = np.zeros((len(pairs), len(sites)))
    for k, p in enumerate(pairs):
        M[k, idx[p["A"]]] = 1.0
        M[k, idx[p["B"]]] = 1.0

    def solve_for(L, s2):
        rhs = y - 2 * s2 * (1 - np.exp(-0.5 * (dd / L) ** 2))
        n, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        n = np.clip(n, 0, None)
        pred = M @ n + 2 * s2 * (1 - np.exp(-0.5 * (dd / L) ** 2))
        return n, float(np.sum((pred - y) ** 2))

    best = (np.inf, None)
    for L in np.geomspace(5, 3000, 60):
        for s2 in np.linspace(0.0, float(max(y.max(), 1e-9)), 40):
            n, sse = solve_for(L, s2)
            if sse < best[0]:
                best = (sse, (float(L), float(s2), n))
    L, s2, nfit = best[1]
    _, sse0 = solve_for(1.0, 0.0)                     # station-noise-only null
    improve = float(1 - best[0] / sse0) if sse0 > 0 else 0.0

    noise = sorted((float(np.sqrt(max(v, 0)) * 1000), s) for s, v in zip(sites, nfit))
    verdict = {
        "n_pairs": len(pairs), "n_stations": len(sites),
        "field_amplitude_mm": float(np.sqrt(max(s2, 0)) * 1000),
        "correlation_length_km": L,
        "variance_explained_over_noise_only_model": improve,
        "field_significant": bool(improve > 0.05 and s2 > 0),
        "median_sidereal_fraction": float(np.median([p["sidereal_fraction"] for p in pairs])),
        "quietest_stations_mm": noise[:6],
        "noisiest_stations_mm": noise[-6:],
    }
    # ---- identifiability: how much of a field basis can this design even see? ----
    # Every row of M has exactly two 1s, so M @ 1 = 2*ones: a CONSTANT lives inside
    # the station-noise column space.  A field that is nearly flat across the
    # sampled baseline range is therefore algebraically indistinguishable from a
    # uniform shift of all n_i.  Quantify that before reading any amplitude.
    Q, _ = np.linalg.qr(M)
    perp = lambda v: v - Q @ (Q.T @ v)
    absorbed = {}
    for Ltest in (80, 200, 500, 1000, 2800, 4348):
        f = 2 * (1 - np.exp(-0.5 * (dd / Ltest) ** 2))
        absorbed[Ltest] = float(np.sum((M @ np.linalg.lstsq(M, f, rcond=None)[0]) ** 2)
                                / max(np.sum(f ** 2), 1e-30))

    def orthogonal_bound(sub):
        """95% upper bound on the field amplitude using ONLY the component of the
        field basis orthogonal to the station-noise span."""
        if len(sub) < 12:
            return None
        st = sorted({x for q in sub for x in (q["A"], q["B"])})
        ix = {x: i for i, x in enumerate(st)}
        Ms = np.zeros((len(sub), len(st)))
        for k, q in enumerate(sub):
            Ms[k, ix[q["A"]]] = 1.0
            Ms[k, ix[q["B"]]] = 1.0
        ds = np.array([q["sep_km"] for q in sub]); ys = np.array([q["D_m2"] for q in sub])
        Qs, _ = np.linalg.qr(Ms)
        pp = lambda v: v - Qs @ (Qs.T @ v)
        yp = pp(ys); best = None
        for Lt in np.geomspace(80, 2000, 40):
            f = pp(2 * (1 - np.exp(-0.5 * (ds / Lt) ** 2)))
            nf = float(f @ f)
            if nf < 1e-12:
                continue
            a = float(f @ yp) / nf
            r = yp - a * f
            se = float(np.sqrt(max((r @ r) / max(len(sub) - len(st) - 1, 1), 0) / nf))
            if best is None or abs(a / se) > abs(best[3]):
                best = (float(Lt), a, se, a / se)
        if best is None:
            return None
        Lt, a, se, z = best
        return {"L_km": Lt, "sigma2_m2": a, "sigma2_se_m2": se, "z": z,
                "amplitude_mm": float(np.sqrt(max(a, 0)) * 1000),
                "upper_95_mm": float(np.sqrt(max(a + 1.96 * se, 0)) * 1000),
                "detected": bool(z > 2)}

    # ---- per-station attribution of the non-repeating (non-multipath) residual ----
    attrib = {}
    for st_name in sites:
        sub = [q for q in pairs if st_name in (q["A"], q["B"])]
        if len(sub) < 3:
            continue
        r = np.array([q["rms_mm"] for q in sub]); sf = np.array([q["sidereal_fraction"] for q in sub])
        ds = np.array([q["sep_km"] for q in sub])
        attrib[st_name] = {
            "n_pairs": len(sub), "median_rms_mm": float(np.median(r)),
            "median_sidereal_fraction": float(np.median(sf)),
            # a FIELD effect must grow with separation; a station defect does not
            "rho_rms_vs_distance": float(np.corrcoef(ds, r)[0, 1]),
        }
    bad = sorted(attrib.items(), key=lambda kv: -kv[1]["median_rms_mm"])[:1]
    bad_site = bad[0][0] if bad else None

    bounds = {
        "all_pairs": orthogonal_bound(pairs),
        "excluding_%s" % bad_site: orthogonal_bound(
            [q for q in pairs if bad_site not in (q["A"], q["B"])]) if bad_site else None,
        "well_behaved_only": orthogonal_bound([q for q in pairs if q["sidereal_fraction"] > 0.5]),
    }

    OUT.write_text(json.dumps({
        "step": 71,
        "identifiability": {
            "note": "fraction of the field basis absorbed by the station-noise column "
                    "space; at >0.9 the design is effectively blind to that scale",
            "absorbed_fraction_by_lambda_km": absorbed,
            "n_pairs": len(pairs), "n_stations": len(sites)},
        "orthogonal_component_bounds": bounds,
        "station_attribution": attrib,
        "worst_station": bad_site,
        "title": "Per-station noise vs spatial field, sidereal-cleaned carrier phase",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "D_AB = n_A + n_B + 2 sigma^2 [1 - exp(-(d/L)^2/2)]",
        "pairs": sorted(pairs, key=lambda p: p["sep_km"]),
        "station_noise_mm": {s: float(np.sqrt(max(v, 0)) * 1000) for s, v in zip(sites, nfit)},
        "verdict": verdict}, indent=2))

    print("\nper-station clock+processing noise (mm):")
    for v, s in noise:
        print(f"   {s}  {v:8.1f}")
    print(f"\nfield amplitude {verdict['field_amplitude_mm']:.1f} mm   L = {L:.0f} km   "
          f"variance explained over noise-only: {improve*100:.1f}%")
    print(f"field significant: {verdict['field_significant']}")
    anom = [p for p in pairs if p["sidereal_fraction"] < 0.3]
    if anom:
        print("\nlow-sidereal (non-repeating) pairs -- station-attributable?")
        for p in sorted(anom, key=lambda q: q["sidereal_fraction"])[:12]:
            print(f"   {p['pair']:<12} {p['sep_km']:7.1f} km  "
                  f"sid {p['sidereal_fraction']*100:6.1f}%  rms {p['rms_mm']:7.1f} mm")


if __name__ == "__main__":
    main()
