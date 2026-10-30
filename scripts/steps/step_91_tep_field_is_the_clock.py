#!/usr/bin/env python3
"""TEP field IS the receiver clock.  k=3 position from that field.

No GR free-clock.  No stale SPP anchor.  No 40-arm kitchen sink.

At each outage epoch the CORS network reads ln A(phi) at the rover:
    clk = causal_trend(t) + krige(live reference residuals)
with the physics-fixed Gaussian (lambda = 4200 km, nugget = 0.35).
That clock is hard-fixed in a 3-satellite nonlinear SPP solve.

Outputs: results/outputs/step_91_tep_field_is_the_clock.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_sinex as sx
from scripts.lib import tep_spp as spp

OUT = ROOT / "results" / "outputs" / "step_91_tep_field_is_the_clock.json"
CACHE = ROOT / "data" / "step48_cache"
C_M_PER_NS = 0.299792458
LAMBDA_KM = 4200.0
NUGGET = 0.35
PRE_S = 7200.0
GAPS = [3600.0, 14400.0]
DOY = int(os.environ.get("TEP_STEP91_DOY", "100"))
MAX_STN = int(os.environ.get("TEP_STEP91_MAX_STATIONS", "16"))
# Dense EUREF/EPN: this is the network where the field is spatially resolved
# (median nearest reference ~130 km).  A global alphabetical cache slice is
# a sparse IGS set and is the wrong experiment.
DENSE = (
    "ZIMM", "ZIM2", "GRAS", "WTZR", "WTZZ", "BRUX", "BRUS",
    "PTBB", "POTS", "TLSE", "GENO", "IENG", "LROC", "MEDI",
    "GRAZ", "OBE4", "WAB2", "ZIM3", "FFMJ", "REDU",
)
RNG = np.random.default_rng(20260903)


def _rho(d):
    return np.exp(-0.5 * (d / LAMBDA_KM) ** 2)


def _repair_jumps(y, mad_mult=20.0):
    """Remove discrete oscillator steering steps (typically 1 ms)."""
    y = np.asarray(y, float).copy()
    d = np.diff(y)
    finite = np.isfinite(d)
    if finite.sum() < 8:
        return y
    scale = 1.4826 * np.median(np.abs(d[finite] - np.median(d[finite]))) + 1e-6
    step = np.zeros_like(d)
    step[finite] = np.where(np.abs(d[finite]) > mad_mult * scale, d[finite], 0.0)
    y[1:] -= np.cumsum(step)
    return y


def _krige(d_rr, d_t, resid):
    K = _rho(d_rr) + (NUGGET + 1e-9) * np.eye(len(resid))
    try:
        w = np.linalg.solve(K, _rho(d_t))
    except np.linalg.LinAlgError:
        w = np.full(len(resid), 1.0 / len(resid))
    return float(w @ resid)


def _linear_fit(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 4:
        return None
    t0 = t[ok].mean()
    coef = np.polyfit(t[ok] - t0, y[ok], 1)
    return coef, t0


def _pred(fit, t):
    coef, t0 = fit
    return float(np.polyval(coef, float(t) - t0))


def _blind_init(truth):
    az = RNG.uniform(0, 2 * np.pi)
    d = 100_000.0
    n = np.linalg.norm(truth)
    east = np.array([-truth[1], truth[0], 0.0])
    east /= np.linalg.norm(east) + 1e-12
    north = np.cross(truth / n, east)
    return truth + d * (np.cos(az) * north + np.sin(az) * east)


def main():
    # The field is read in the SAME clock the k=3 solver uses: the cached
    # IF-code receiver clock.  Mixing a precise-CLK datum into those
    # observations is a unit error, not a TEP result.
    names = sorted(p.stem.split("_")[0] for p in CACHE.glob(f"*_{DOY:03d}.pkl"))
    xyz, cache = {}, {}
    for n in names:
        pos = sx.station_xyz(n, 2024, DOY)
        if pos is None:
            continue
        with open(CACHE / f"{n}_{DOY:03d}.pkl", "rb") as f:
            blob = pickle.load(f)
        sods = [k for k in blob if isinstance(k, (int, np.integer)) and "clk" in blob[k]]
        if len(sods) < 400:
            continue
        xyz[n] = np.asarray(pos, float)
        cache[n] = blob
    names = [n for n in DENSE if n in cache][:MAX_STN]
    if len(names) < 8:
        names = [n for n in cache][:MAX_STN]
    if len(names) < 8:
        raise SystemExit(f"need 8 cached stations, have {len(names)}")
    print(f"DOY {DOY}: {len(names)} stations  field=precise residual on SPP trend  λ={LAMBDA_KM:.0f} km",
          flush=True)

    grid_t = np.arange(1800.0, 86400.0, 300.0)
    raw = np.full((len(grid_t), len(names)), np.nan)
    for j, n in enumerate(names):
        for i, sod in enumerate(grid_t):
            rec = cache[n].get(int(sod))
            if rec is not None and np.isfinite(rec.get("clk", np.nan)):
                raw[i, j] = float(rec["clk"])
        raw[:, j] = _repair_jumps(raw[:, j])
    t_ref = grid_t
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            ga, gb = td.ecef_to_geodetic(*xyz[a]), td.ecef_to_geodetic(*xyz[b])
            D[i, j] = D[j, i] = td.great_circle_km(
                td.Station(a, xyz[a] / 1000.0, ga[0], ga[1]),
                td.Station(b, xyz[b] / 1000.0, gb[0], gb[1]))

    # Precise-product residual = the field.  SPP trend = the solver datum.
    dt = f"2024{DOY:03d}"
    series = td.load_series("precise", dt, min_valid_frac=0.0, max_jumps=10_000) or {}
    field = np.full_like(raw, np.nan)
    if series:
        for j, n in enumerate(names):
            if n not in series:
                continue
            pt = series[n]["t"].astype(float)
            py = np.asarray(series[n]["raw"], float) * C_M_PER_NS
            py = _repair_jumps(py)
            field[:, j] = np.interp(t_ref, pt, py, left=np.nan, right=np.nan)
        n_field = int(np.isfinite(field).any(axis=0).sum())
        print(f"precise field on {n_field}/{len(names)} stations", flush=True)

    tag = f"2024{DOY:03d}"
    sp3p = next((ROOT / "data" / "rinex" / "sp3").glob(f"*{tag}*SP3*"))
    clkp = next((ROOT / "data" / "rinex" / "clk").glob(f"*{tag}*CLK*"), None)
    print(f"orbits {sp3p.name}", flush=True)
    grid = spp.dense_grid(spp.read_sp3_gps(sp3p),
                          spp.read_clk_gps(clkp) if clkp else {},
                          step=5.0)

    acc = {g: {"tep": [], "pers": [], "ar1": [],
               "tep_clk": [], "pers_clk": [], "ar1_clk": [],
               "dfield": [], "tep_vs_precise": [], "pers_vs_precise": []}
           for g in GAPS}

    dt_s = float(np.median(np.diff(t_ref)))
    npre = int(PRE_S / dt_s)
    for gi, gap in enumerate(GAPS):
        kstep = int(round(gap / dt_s))
        print(f"\n--- gap {gap/3600:.0f} h ---", flush=True)
        for b, tgt in enumerate(names):
            truth = xyz[tgt]
            if truth is None:
                continue
            tdat = cache[tgt]
            n_ok = 0
            for ti in range(npre, len(t_ref) - kstep, 1):
                t0 = float(t_ref[ti])
                t1 = float(t_ref[ti + kstep])
                sod = int(round(t1))
                rec = tdat.get(sod) or tdat.get(sod - sod % 30)
                if rec is None or "raw_obs" not in rec or len(rec["raw_obs"]) < 3:
                    continue
                y_tgt = raw[:ti + 1, b]
                tt = t_ref[:ti + 1]
                fit = _linear_fit(tt[-npre:], y_tgt[-npre:])
                if fit is None:
                    continue
                last = float(y_tgt[ti])
                if not np.isfinite(last):
                    continue
                xr = y_tgt[-npre:] - np.array([_pred(fit, x) for x in tt[-npre:]])
                xr = xr[np.isfinite(xr)]
                if len(xr) < 4:
                    continue
                phi = float(np.clip(
                    np.sum(xr[1:] * xr[:-1]) / (np.sum(xr[:-1] ** 2) + 1e-9),
                    -0.999, 0.999))
                n_ar = max(1, kstep)

                def _field_at(idx):
                    refs, d_t, resid = [], [], []
                    for j in range(len(names)):
                        if j == b:
                            continue
                        yj = field[idx, j]
                        pre_f = field[ti - npre + 1:ti + 1, j]
                        pre_t = t_ref[ti - npre + 1:ti + 1]
                        fitj = _linear_fit(pre_t, pre_f)
                        if fitj is None or not np.isfinite(yj):
                            continue
                        refs.append(j)
                        d_t.append(D[b, j])
                        resid.append(yj - _pred(fitj, t_ref[idx]))
                    if len(resid) < 6:
                        return None
                    resid = np.asarray(resid, float)
                    d_t = np.asarray(d_t, float)
                    d_rr = D[np.ix_(refs, refs)]
                    val = _krige(d_rr, d_t, resid)
                    rmed = float(np.median(resid))
                    rmad = 1.4826 * float(np.median(np.abs(resid - rmed))) + 1e-6
                    return float(np.clip(val, rmed - 6 * rmad, rmed + 6 * rmad))

                f1 = _field_at(ti + kstep)
                pre_f = field[ti - npre + 1:ti + 1, b]
                fit_tgt_f = _linear_fit(t_ref[ti - npre + 1:ti + 1], pre_f)
                if f1 is None or fit_tgt_f is None or not np.isfinite(field[ti, b]):
                    continue
                # t0 residual is known (pre-outage).  Only t1 is predicted.
                # Two kriges was stacking error; this is the Step 55 increment.
                f0 = float(field[ti, b] - _pred(fit_tgt_f, t0))
                pers_clk = last
                tep_clk = last + (f1 - f0)
                last_res = float(y_tgt[ti] - _pred(fit, t0))
                ar1_clk = last + last_res * ((phi ** n_ar) - 1.0)
                truth_clk = float(raw[ti + kstep, b])
                acc[gap]["dfield"].append(abs(f1 - f0))
                if np.isfinite(field[ti + kstep, b]):
                    resid_t1 = float(field[ti + kstep, b] - _pred(fit_tgt_f, t1))
                    acc[gap]["tep_vs_precise"].append(abs(f1 - resid_t1))
                    acc[gap]["pers_vs_precise"].append(abs(f0 - resid_t1))
                obs = list(rec["raw_obs"].items())
                init = _blind_init(truth)
                zwd = rec.get("zwd", 0.0)
                row = {}
                for name, clk in (("tep", tep_clk), ("pers", pers_clk), ("ar1", ar1_clk)):
                    try:
                        ret = spp.solve_spp_nonlinear(
                            obs, grid, float(sod), init, clock=clk,
                            clock_sigma=None, zwd=zwd, k=3,
                            elev_mask=10.0, pdop_max=20.0)
                    except (ValueError, FloatingPointError):
                        ret = None
                    if ret is None:
                        continue
                    err = float(np.linalg.norm(ret[0] - truth))
                    if np.isfinite(err) and err < 500:
                        acc[gap][name].append(err)
                        row[name] = err
                    if truth_clk is not None and np.isfinite(truth_clk):
                        acc[gap][f"{name}_clk"].append(abs(clk - float(truth_clk)))
                if "tep" in row:
                    n_ok += 1
            print(f"  {tgt}: {n_ok} epochs", flush=True)

    summary = {}
    print("\n==== TEP field is the clock ====", flush=True)
    for gap in GAPS:
        summary[str(int(gap))] = {}
        print(f"\ngap {gap/3600:.0f} h", flush=True)
        for name in ("tep", "pers", "ar1"):
            e = np.asarray(acc[gap][name], float)
            c = np.asarray(acc[gap][f"{name}_clk"], float)
            if len(e) < 10:
                summary[str(int(gap))][name] = {"n": int(len(e))}
                continue
            rec = {
                "n": int(len(e)),
                "rms_3d_m": float(np.sqrt(np.mean(e ** 2))),
                "p50_m": float(np.percentile(e, 50)),
                "p95_m": float(np.percentile(e, 95)),
                "fail_25m": float(np.mean(e > 25)),
                "clock_rms_m": float(np.sqrt(np.mean(c ** 2))) if len(c) else None,
            }
            summary[str(int(gap))][name] = rec
            print(f"  {name:5s}  n={rec['n']:5d}  RMS={rec['rms_3d_m']:.2f} m  "
                  f"p50={rec['p50_m']:.2f}  p95={rec['p95_m']:.2f}  "
                  f"clkRMS={rec['clock_rms_m']}", flush=True)
        tep = summary[str(int(gap))].get("tep", {})
        for base in ("pers", "ar1"):
            b = summary[str(int(gap))].get(base, {})
            if tep.get("rms_3d_m") and b.get("rms_3d_m"):
                gain = 100.0 * (1.0 - tep["rms_3d_m"] / b["rms_3d_m"])
                summary[str(int(gap))][f"tep_vs_{base}_pct"] = gain
                print(f"  TEP vs {base}: {gain:+.1f}%", flush=True)
        df = np.asarray(acc[gap]["dfield"], float)
        tp = np.asarray(acc[gap]["tep_vs_precise"], float)
        pp = np.asarray(acc[gap]["pers_vs_precise"], float)
        if len(df):
            print(f"  |Δφ| p50={np.percentile(df,50):.2f} m  p95={np.percentile(df,95):.2f} m",
                  flush=True)
        if len(tp) > 10 and len(pp) > 10:
            tr = float(np.sqrt(np.mean(tp ** 2)))
            pr = float(np.sqrt(np.mean(pp ** 2)))
            print(f"  precise residual at t1: TEP-krige {tr:.2f} m  hold-t0 {pr:.2f} m  "
                  f"skill {100*(1-tr/pr):+.1f}%", flush=True)
            summary[str(int(gap))]["field_increment"] = {
                "dphi_p50_m": float(np.percentile(df, 50)),
                "dphi_p95_m": float(np.percentile(df, 95)),
                "tep_vs_precise_rms_m": tr,
                "persist0_vs_precise_rms_m": pr,
                "skill_pct": 100.0 * (1.0 - tr / pr) if pr else None,
            }

    result = {
        "step": 91,
        "title": "TEP field is the receiver clock",
        "doy": DOY,
        "stations": names,
        "lambda_km": LAMBDA_KM,
        "nugget": NUGGET,
        "architecture": "clk = causal_trend + krige(live precise residuals); hard clock; k=3",
        "by_gap": summary,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
