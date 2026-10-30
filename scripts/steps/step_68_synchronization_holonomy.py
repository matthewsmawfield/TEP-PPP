#!/usr/bin/env python3
"""
STEP 68  --  THE SYNCHRONIZATION HOLONOMY: the distinctive TEP observable.

Every previous operational step in this pipeline measured the SCALAR sector of
the proper-time field -- clock offsets, their correlations, their spatial
gradient.  The TEP paper is explicit that this is not the distinctive
observable:

    "The operational observable is not a raw one-way speed and not the integral
     of a scalar proper-time differential.  It is the residual non-closure of
     synchronization transport around a closed loop."

    "a conformal-only scalar may affect local rates and open-path redshift
     comparisons, but it cannot by itself generate a closed-loop residual
     synchronization holonomy."

A scalar potential's gradient is an EXACT one-form, so its loop integral
vanishes identically.  Step 66's v.grad(phi) null was therefore guaranteed by
construction, not informative about TEP's non-exact sector.  The invariant is

    H_resid(C) = closed_loop_integral(sigma_tilde - sigma_GR)
               = surface_integral(F_tilde - F_GR),      C = boundary(Sigma)

nonzero only from disformal / non-metric (non-exact) transport structure.

In GNSS the closed synchronization loop is realised exactly by the
DOUBLE DIFFERENCE:

    station A --(satellite k1)--> station B --(satellite k2)--> station A

The double difference cancels both receiver clocks and both satellite clocks.
That cancellation IS the statement that the exact (conformal/scalar) part drops
out.  What survives, after geometry, atmosphere and the integer ambiguity, is
the non-exact part -- H_resid -- plus measurement noise.

TEST.  H_resid is a FLUX through the enclosed surface, so it must scale with
the loop's signed area and vanish as the loop degenerates.  The natural signed
invariant for the loop (A, k1, B, k2) is the scalar triple product

    T = (e_k1 x e_k2) . b        [b = baseline vector A->B, e = unit LOS]

which is signed, vanishes when the baseline lies in the plane of the two
lines of sight, and flips sign under k1<->k2 exactly as the double difference
does.  A nonzero regression slope of the double-difference residual on T is a
detection of synchronization curvature.

DISCRIMINATION FROM ATMOSPHERE.  Differential troposphere also grows with
baseline and elevation difference -- but it is UNSIGNED in T: it depends on
|mapping-function difference|, not on the orientation of the loop.  Both
regressors are fitted jointly, so the signed-area term is separated from the
atmospheric term.

CONTROLS.
  * co-located pair (WTZR/WTZZ, 3 m): b ~ 0 so T ~ 0 -- the slope MUST vanish;
    this calibrates the instrumental floor.
  * geometry shuffle: permute T against the residuals, destroying the loop
    association while preserving both distributions.
  * sign-flip null on the per-arc slopes.

Output: results/outputs/step_68_synchronization_holonomy.json
"""
from __future__ import annotations

import sys
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pathlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_dynamic as dyn   # noqa: E402
from scripts.lib import tep_spp as spp       # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_68_synchronization_holonomy.json"
CACHE = ROOT / "data" / "step67_cache"
WORK = Path("/tmp/step68_work")
RNG = np.random.default_rng(20260902)

F1, F2 = 1575.42e6, 1227.60e6
C = 299792458.0
LAM1, LAM2 = C / F1, C / F2
A_IF, B_IF = F1 ** 2 / (F1 ** 2 - F2 ** 2), -F2 ** 2 / (F1 ** 2 - F2 ** 2)
OMEGA = 7.2921151467e-5

DOYS = [100, 101, 102]
YEAR = 2024
EPOCH_STRIDE = 4            # 30 s data -> every 2 min
MIN_ARC = 25
ELEV_MIN = 20.0             # deg; high cut suppresses multipath and troposphere

PAIRS = [
    ("WTZR", "WTZZ"),   # 0.003 km  CONTROL: zero loop area
    ("ZIM2", "ZIMM"),   # 0.019 km  CONTROL
    ("WAB2", "ZIMM"),   # 5.2 km
    ("PADO", "VEN1"),   # 35.9 km
    ("BRUX", "REDU"),   # 104.8 km
    ("BRUX", "DLF1"),   # 132.2 km
    ("GRAS", "IENG"),   # 151.3 km
    ("BRUX", "WSRT"),   # 281.7 km
]



ATX_PATH = ROOT / "archive" / "PRIDE-PPPAR" / "table" / "igs20_2317.atx"
_ATX_CACHE = {}


def load_atx():
    """{antenna_type: {freq: (pco_neu_m, zen_grid_deg, pcv_noazi_m)}} from ANTEX."""
    if _ATX_CACHE:
        return _ATX_CACHE
    if not ATX_PATH.exists():
        return _ATX_CACHE
    lines = ATX_PATH.read_text(errors="replace").splitlines()
    i, ant, zen = 0, None, None
    while i < len(lines):
        ln = lines[i]
        if "TYPE / SERIAL NO" in ln:
            ant = ln[:20].strip()
            _ATX_CACHE.setdefault(ant, {})
        elif "ZEN1 / ZEN2 / DZEN" in ln:
            z1, z2, dz = (float(x) for x in ln[:20].split())
            zen = np.arange(z1, z2 + 1e-9, dz)
        elif "START OF FREQUENCY" in ln and ant is not None:
            freq = ln[3:6].strip()
            pco = None; pcv = None
            j = i + 1
            while j < len(lines) and "END OF FREQUENCY" not in lines[j]:
                if "NORTH / EAST / UP" in lines[j]:
                    v = [float(x) for x in lines[j][:60].split()]
                    pco = np.array(v[:3]) / 1000.0            # mm -> m (N,E,U)
                elif lines[j][:8].strip() == "NOAZI":
                    pcv = np.array([float(x) for x in lines[j][8:].split()]) / 1000.0
                j += 1
            if pco is not None and pcv is not None and zen is not None and len(pcv) == len(zen):
                _ATX_CACHE[ant][freq] = (pco, zen, pcv)
            i = j
        i += 1
    return _ATX_CACHE


def antenna_info(rnx):
    """(antenna_type, delta_NEU_m) from the RINEX header."""
    typ, dneu = None, np.zeros(3)
    for ln in pathlib.Path(rnx).read_text(errors="replace").splitlines()[:80]:
        if "ANT # / TYPE" in ln:
            typ = ln[20:40].strip()
        elif "ANTENNA: DELTA H/E/N" in ln:
            v = [float(x) for x in ln[:42].split()]
            dneu = np.array([v[2], v[1], v[0]])               # file order H,E,N -> N,E,U
        if "END OF HEADER" in ln:
            break
    return typ, dneu


def pcv_correction(atx, typ, dneu, e_neu, zen_deg):
    """Ionosphere-free antenna phase-centre correction (metres).

    The measurement refers to the frequency-dependent phase centre; the
    coordinates refer to the marker.  Without this, the residual carries an
    elevation/azimuth-dependent, ANTENNA-MODEL-specific term -- exactly the
    signature that would otherwise be mistaken for synchronization curvature.
    """
    tot = 0.0
    for freq, w in (("G01", A_IF), ("G02", B_IF)):
        ent = (atx.get(typ) or {}).get(freq)
        if ent is None:
            return None
        pco, zen, pcv = ent
        off = dneu + pco
        d = float(np.dot(off, e_neu)) - float(np.interp(zen_deg, zen, pcv))
        tot += w * d
    return tot


def read_phase(rnx):
    lines = Path(rnx).read_text(errors="replace").splitlines()
    types, i = None, 0
    while i < len(lines):
        ln = lines[i]
        if "SYS / # / OBS TYPES" in ln and ln[:1] == "G":
            n = int(ln[3:6]); tk = ln[7:60].split(); j = i
            while len(tk) < n:
                j += 1; tk += lines[j][7:60].split()
            types = tk[:n]; i = j
        if "END OF HEADER" in ln:
            i += 1; break
        i += 1
    if not types:
        return {}
    def ix(*nm):
        for x in nm:
            if x in types:
                return types.index(x)
        return None
    i1, i2 = ix("L1C", "L1W", "L1P"), ix("L2W", "L2L", "L2C", "L2X", "L2S")
    if i1 is None or i2 is None:
        return {}
    out = {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            tk = ln.split()
            if len(tk) < 9:
                i += 1; continue
            try:
                sod = int(tk[4]) * 3600 + int(tk[5]) * 60 + int(round(float(tk[6])))
                ns = int(tk[8])
            except Exception:
                i += 1; continue
            rec = {}
            for _ in range(ns):
                i += 1
                if i >= len(lines):
                    break
                sl = lines[i]
                if not sl or sl[0] != "G":
                    continue
                prn = sl[:3].replace(" ", "0")
                def g(k):
                    s = sl[3 + k * 16: 3 + k * 16 + 14].strip()
                    return float(s) if s else np.nan
                L1, L2 = g(i1), g(i2)
                if np.isfinite(L1) and np.isfinite(L2):
                    rec[prn] = A_IF * (L1 * LAM1) + B_IF * (L2 * LAM2)
            if rec:
                out[sod] = rec
        i += 1
    return out


def sat_geom(dense, prn, sod, rx):
    """Light-time-corrected range, unit LOS (rx->sat), elevation deg, wet map."""
    t0, dt, G = dense
    if prn not in G:
        return None
    P, _, _ = G[prn]
    tau = 0.075
    psr = None
    for _ in range(3):
        k = (sod - tau - t0) / dt
        ki = int(k)
        if ki < 0 or ki + 1 >= len(P):
            return None
        fr = k - ki
        ps = P[ki] * (1 - fr) + P[ki + 1] * fr
        th = OMEGA * tau
        rot = np.array([[np.cos(th), np.sin(th), 0.], [-np.sin(th), np.cos(th), 0.], [0., 0., 1.]])
        psr = rot @ ps
        rng = float(np.linalg.norm(psr - rx))
        tau = rng / C
    e = (psr - rx) / rng
    up = rx / np.linalg.norm(rx)
    el = float(np.degrees(np.arcsin(np.clip(np.dot(e, up), -1, 1))))
    se = np.sin(np.radians(max(el, 3.0)))
    wm = 1.0 / (se + 0.00035 / (np.tan(np.radians(max(el, 3.0))) + 0.017))
    # local ENU azimuth, for the horizontal tropospheric-gradient regressors
    lat = np.arctan2(rx[2], np.hypot(rx[0], rx[1]))
    lon = np.arctan2(rx[1], rx[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    az = float(np.arctan2(np.dot(e, east), np.dot(e, north)))
    gfac = wm / np.tan(np.radians(max(el, 5.0)))     # Chen-Herring gradient mapping
    e_neu = np.array([np.dot(e, north), np.dot(e, east), np.dot(e, up)])
    return rng, e, el, wm, az, gfac, e_neu, 90.0 - el


def load_station(site, doy):
    f = glob.glob(str(ROOT / f"data/rinex/obs/{YEAR}/{doy:03d}/{site}*.crx.gz"))
    if not f:
        return None
    rnx = dyn._decompress_crx(Path(f[0]), WORK / f"{site}_{doy}")
    if rnx is None:
        return None
    xyz = dyn._rinex_header_xyz(rnx)
    if xyz is None:
        return None
    typ, dneu = antenna_info(rnx)
    return read_phase(rnx), xyz, typ, dneu


def grid_for(doy):
    tag = f"{YEAR}{doy:03d}"
    gc = CACHE / f"grid_{tag}.npz"
    if gc.exists():
        try:
            z = np.load(gc, allow_pickle=True)
            return (float(z["t0"]), float(z["step"]), z["G"].item())
        except Exception:
            pass
    sp3, clk = dyn.find_sp3_clk(tag)
    if not (sp3 and clk):
        return None
    dense = spp.dense_grid(spp.read_sp3_gps(sp3), spp.read_clk_gps(clk), step=1.0)
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gc, t0=dense[0], step=dense[1], G=np.array(dense[2], dtype=object))
    except Exception:
        pass
    return dense


def pair_holonomy(a, b, doy, dense):
    """Double-difference residuals with the signed loop invariant T."""
    da, db = load_station(a, doy), load_station(b, doy)
    if not da or not db:
        return None
    oa, xa, ta_, na_ = da
    ob, xb, tb_, nb_ = db
    bvec = xb - xa
    blen = float(np.linalg.norm(bvec)) / 1000.0
    sods = sorted(set(oa) & set(ob))[::EPOCH_STRIDE]
    if len(sods) < 100:
        return None

    ATX = load_atx()
    n_nopcv = [0]
    recs = []          # (sod, prn, dd, T, dwm, gN, gE)
    for s in sods:
        common = sorted(set(oa[s]) & set(ob[s]))
        if len(common) < 4:
            continue
        geo = {}
        for p in common:
            ga = sat_geom(dense, p, s, xa)
            gb = sat_geom(dense, p, s, xb)
            if ga is None or gb is None:
                continue
            if min(ga[2], gb[2]) < ELEV_MIN:
                continue
            geo[p] = (ga, gb)
        if len(geo) < 4:
            continue
        ref = max(geo, key=lambda p: geo[p][0][2])          # highest elevation
        ga_r, gb_r = geo[ref]
        car = pcv_correction(ATX, ta_, na_, ga_r[6], ga_r[7])
        cbr = pcv_correction(ATX, tb_, nb_, gb_r[6], gb_r[7])
        if car is None or cbr is None:
            car = cbr = 0.0
        sd_ref = ((oa[s][ref] - car) - (ob[s][ref] - cbr)) - (ga_r[0] - gb_r[0])
        for p, (ga, gb) in geo.items():
            if p == ref:
                continue
            ca = pcv_correction(ATX, ta_, na_, ga[6], ga[7])
            cb = pcv_correction(ATX, tb_, nb_, gb[6], gb[7])
            if ca is None or cb is None:
                n_nopcv[0] += 1
                ca = cb = 0.0
            sd = ((oa[s][p] - ca) - (ob[s][p] - cb)) - (ga[0] - gb[0])
            dd = sd - sd_ref
            # signed loop invariant: (e_k x e_ref) . baseline_unit
            T = float(np.dot(np.cross(ga[1], ga_r[1]), bvec / max(np.linalg.norm(bvec), 1e-9)))
            dwm = abs(ga[3] - ga_r[3])                       # zenith-delay proxy
            # HORIZONTAL tropospheric gradient: delay = m_w cot(el) (G_N cos az
            # + G_E sin az).  Differenced across satellites this is exactly a
            # direction-dependent term that projects onto T, so it MUST be
            # fitted or it will be mistaken for synchronization curvature.
            gN = ga[5] * np.cos(ga[4]) - ga_r[5] * np.cos(ga_r[4])
            gE = ga[5] * np.sin(ga[4]) - ga_r[5] * np.sin(ga_r[4])
            recs.append((s, p, dd, T, dwm, gN, gE))
    if len(recs) < 300:
        return None

    # per-arc ambiguity removal: each (satellite, ref) arc has a constant
    by = {}
    for s, p, dd, T, dwm, gN, gE in recs:
        by.setdefault(p, []).append((s, dd, T, dwm, gN, gE))
    DD, TT, WW, GN, GE, AR = [], [], [], [], [], []
    for p, v in by.items():
        v.sort()
        s = np.array([x[0] for x in v], float)
        d = np.array([x[1] for x in v], float)
        t = np.array([x[2] for x in v], float)
        w = np.array([x[3] for x in v], float)
        gn = np.array([x[4] for x in v], float)
        ge = np.array([x[5] for x in v], float)
        # split on data gaps AND on cycle slips, rather than discarding a whole
        # arc for a single slip -- discarding was silently removing most data,
        # including every control pair.
        brk = set(np.where(np.diff(s) > 400)[0])
        brk |= set(np.where(np.abs(np.diff(d)) > 0.05)[0])
        segs = np.split(np.arange(len(s)), sorted(x + 1 for x in brk))
        for sg in segs:
            if len(sg) < MIN_ARC:
                continue
            dv = d[sg]
            if not np.isfinite(dv).all() or np.std(dv) > 0.5:
                continue
            DD.append(dv - np.median(dv))                    # ambiguity out
            TT.append(t[sg] - np.median(t[sg]))
            WW.append(w[sg] - np.median(w[sg]))
            GN.append(gn[sg] - np.median(gn[sg]))
            GE.append(ge[sg] - np.median(ge[sg]))
            AR.append(np.full(len(sg), len(DD)))             # arc id, for the null
    if not DD:
        return None
    DD = np.concatenate(DD); TT = np.concatenate(TT); WW = np.concatenate(WW)
    GN = np.concatenate(GN); GE = np.concatenate(GE); AR = np.concatenate(AR)
    m = (np.isfinite(DD) & np.isfinite(TT) & np.isfinite(WW)
         & np.isfinite(GN) & np.isfinite(GE) & (np.abs(DD) < 0.3))
    DD, TT, WW, GN, GE, AR = DD[m], TT[m], WW[m], GN[m], GE[m], AR[m]
    if len(DD) < 300:
        return None

    # joint fit: DD = alpha*T (signed holonomy) + beta*dwm (unsigned atmosphere)
    X = np.column_stack([TT, WW, GN, GE])
    try:
        coef, *_ = np.linalg.lstsq(X, DD, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = X @ coef
    resid = DD - pred
    # bootstrap the holonomy slope
    bs = []
    for _ in range(600):
        idx = RNG.integers(0, len(DD), len(DD))
        try:
            cb, *_ = np.linalg.lstsq(X[idx], DD[idx], rcond=None)
            bs.append(cb[0])
        except np.linalg.LinAlgError:
            pass
    # NULL: a plain permutation of T destroys its temporal autocorrelation and
    # therefore badly UNDERSTATES the null variance, inflating z.  Instead
    # circularly shift T within each arc by a random lag: this preserves the
    # autocorrelation of both series while destroying their alignment.
    nulls = []
    arcs = np.unique(AR)
    for _ in range(600):
        Ts = TT.copy()
        for aid in arcs:
            sel = np.where(AR == aid)[0]
            if len(sel) > 3:
                Ts[sel] = np.roll(TT[sel], int(RNG.integers(1, len(sel))))
        try:
            cn, *_ = np.linalg.lstsq(np.column_stack([Ts, WW, GN, GE]), DD, rcond=None)
            nulls.append(cn[0])
        except np.linalg.LinAlgError:
            pass
    bs, nulls = np.array(bs), np.array(nulls)
    z = float((coef[0] - np.mean(nulls)) / (np.std(nulls) + 1e-30))
    return {
        "pair": f"{a}-{b}", "baseline_km": blen, "n_obs": int(len(DD)),
        "antennas": [ta_, tb_], "n_obs_without_pcv": int(n_nopcv[0]),
        "dd_residual_rms_m": float(np.std(DD)),
        "holonomy_slope": float(coef[0]),
        "holonomy_slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "atmos_slope": float(coef[1]),
        "tropo_gradient_N": float(coef[2]),
        "tropo_gradient_E": float(coef[3]),
        "circshift_null_mean": float(np.mean(nulls)), "circshift_null_sd": float(np.std(nulls)),
        "holonomy_z": z,
        "post_fit_rms_m": float(np.std(resid)),
        "variance_explained_by_holonomy": float(1 - np.var(DD - coef[0] * TT) / np.var(DD)),
        "detected": bool(abs(z) > 5.0 and
                         np.sign(np.percentile(bs, 2.5)) == np.sign(np.percentile(bs, 97.5))),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for doy in DOYS:
        dense = grid_for(doy)
        if dense is None:
            continue
        for a, b in PAIRS:
            try:
                r = pair_holonomy(a, b, doy, dense)
            except Exception as e:
                r = None
                print(f"  {a}-{b} doy {doy}: {e}")
            if r:
                r["doy"] = doy
                rows.append(r)
                print(f"  [{doy}] {r['pair']:<12} {r['baseline_km']:8.3f} km  "
                      f"DD rms {r['dd_residual_rms_m']*1000:6.1f} mm  "
                      f"holonomy slope {r['holonomy_slope']:+.4e}  z={r['holonomy_z']:+7.1f}  "
                      f"{'DETECTED' if r['detected'] else ''}", flush=True)
    # aggregate per pair across days
    agg = {}
    for r in rows:
        agg.setdefault(r["pair"], []).append(r)
    summary = []
    for p, v in agg.items():
        zs = [x["holonomy_z"] for x in v]
        sl = [x["holonomy_slope"] for x in v]
        summary.append({
            "pair": p, "baseline_km": v[0]["baseline_km"], "n_days": len(v),
            "dd_residual_rms_mm": float(np.median([x["dd_residual_rms_m"] for x in v]) * 1000),
            "holonomy_slope_median": float(np.median(sl)),
            "holonomy_z_median": float(np.median(zs)),
            "slope_sign_consistent": bool(len(set(np.sign(sl))) == 1),
            "detected_all_days": bool(all(x["detected"] for x in v)),
        })
    summary.sort(key=lambda r: r["baseline_km"])
    controls = [s for s in summary if s["baseline_km"] < 0.05]
    tests = [s for s in summary if s["baseline_km"] >= 1.0]
    verdict = {
        "control_pairs_null": bool(all(abs(s["holonomy_z_median"]) < 5 for s in controls)) if controls else None,
        "n_test_pairs_detected": sum(1 for s in tests if s["detected_all_days"]),
        "n_test_pairs": len(tests),
        "slope_grows_with_baseline": None,
    }
    if len(tests) >= 3:
        bl = np.array([s["baseline_km"] for s in tests])
        sl = np.abs([s["holonomy_slope_median"] for s in tests])
        if np.all(np.isfinite(sl)) and sl.std() > 0:
            verdict["slope_baseline_spearman"] = float(
                np.corrcoef(np.argsort(np.argsort(bl)), np.argsort(np.argsort(sl)))[0, 1])
    OUT.write_text(json.dumps({
        "step": 68,
        "title": "Synchronization holonomy -- the non-exact (disformal) TEP sector",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observable": ("double-difference carrier phase = closed synchronization loop "
                       "A->sat1->B->sat2->A; cancels all clocks, hence removes the exact "
                       "(conformal/scalar) sector by construction; regressed on the signed "
                       "loop invariant T=(e_k x e_ref).b against an unsigned atmospheric term"),
        "per_day": rows, "summary": summary, "verdict": verdict}, indent=2))
    print("\nSUMMARY (median over days)")
    for s in summary:
        tag = "CONTROL" if s["baseline_km"] < 0.05 else ""
        print(f"  {s['pair']:<12} {s['baseline_km']:8.3f} km  DD {s['dd_residual_rms_mm']:6.1f} mm  "
              f"slope {s['holonomy_slope_median']:+.4e}  z {s['holonomy_z_median']:+7.1f}  "
              f"{'DETECTED' if s['detected_all_days'] else ''} {tag}")
    print(f"\nverdict: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
