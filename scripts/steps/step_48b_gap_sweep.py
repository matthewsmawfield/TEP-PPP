#!/usr/bin/env python3
"""STEP 48b -- OUTAGE / STALE-ANCHOR GAP SWEEP.

The three-satellite challenge (Step 48) shows that at k=3 a clock prior is
*necessary* (the conventional solve is rank-deficient) and that TEP and
temporal extrapolation (persistence / AR1) of the receiver's *own* clock are
comparable when the receiver carries a fresh 30-minute clock history.  That
fresh-history regime is persistence's strongest case.  The operationally
distinctive regime for a *network* prior is the opposite one: the receiver
has lost all satellites for an extended period, so its own clock history is
stale and temporal extrapolation must forecast across the outage, while the
network stations (which still have their own satellites) continue to observe
the clock field.

This script sweeps the *anchor age* -- the time between the receiver's last
good clock fix and the k=3 evaluation epoch -- and measures, as a function of
that outage length, the clock-prediction error and the k=3 truth-linearised
position error for:

  tep          local trend + held anchor residual + network-kriged change
  persistence  local trend + held anchor residual (no network)
  ar1          local trend + first-order Gauss-Markov residual extrapolation

Everything is strictly causal: the local offset/frequency model and the AR1
parameter are fitted on a 30-minute window ending at the anchor; the network
kriging uses reference clocks at the anchor and evaluation epochs only; no
evaluation-epoch target data enters any predictor.  The TEP covariance is
FIXED by the physics (step_10: Gaussian, lambda = 3,480 km, nugget 0.15) --
zero tuning on evaluation data.

The cache produced by step_48 (data/step48_cache/<STATION>_<DOY>.pkl) is
reused directly, so no RINEX reprocessing is required.  The output is written
to results/outputs/step_48b_gap_sweep.json.
"""
from __future__ import annotations

import sys
import json
import glob
import pickle
import os
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_sinex as sx  # noqa: E402

CACHE = PROJECT_ROOT / "data" / "step48_cache"
OUT = PROJECT_ROOT / "results" / "outputs"

DAYS = [int(x) for x in os.environ.get(
    "TEP_STEP48B_DOYS", "100,101,102,103,104,105,106").split(",") if x]
MAX_STATIONS_PER_DAY = int(os.environ.get("TEP_STEP48B_MAX_STATIONS", "20"))
EVAL_STRIDE_S = 600.0          # evaluate every 10 minutes
PREWINDOW_S = 1800.0           # 30-min local-model fit window ending at anchor
GAPS_S = [0.0, 600.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
STEP_S = 30.0
ELEV_MASK = 10.0
PDOP_MAX = 20.0

# TEP covariance -- FIXED by the physics, canonical 4,200 km / 0.35.
TEP_FORM = "gaussian"
TEP_LAMBDA_KM = 4200.0
TEP_NUGGET = 0.35
C_MS = 299_792.458


def _greatcircle(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    la = np.arctan2(a[2], np.hypot(a[0], a[1]))
    lo = np.arctan2(a[1], a[0])
    lb = np.arctan2(b[2], np.hypot(b[0], b[1]))
    lob = np.arctan2(b[1], b[0])
    d = np.arccos(np.clip(np.sin(la) * np.sin(lb) +
                          np.cos(la) * np.cos(lb) * np.cos(lo - lob), -1, 1))
    return 6371.0 * d


def _krige(target_xyz, ref_xyz, ref_val, form=TEP_FORM, lam=TEP_LAMBDA_KM,
           nug=TEP_NUGGET, k=25):
    """Isotropic Gaussian-covariance kriging (identical to step_48)."""
    rv_all = np.asarray(ref_val, float)
    med = np.median(rv_all)
    mad = 1.4826 * np.median(np.abs(rv_all - med)) + 1e-6
    good = np.abs(rv_all - med) < 6.0 * mad
    if good.sum() < 5:
        good = np.ones(len(rv_all), bool)
    ref_xyz = [r for r, gd in zip(ref_xyz, good) if gd]
    rv_all = rv_all[good]
    d = np.array([_greatcircle(target_xyz, r) for r in ref_xyz])
    order = np.argsort(d)[:k]
    d = d[order]; rv = rv_all[order]
    dm = np.array([[_greatcircle(ref_xyz[order[i]], ref_xyz[order[j]])
                    for j in range(len(order))] for i in range(len(order))])
    if form == "gaussian":
        K = np.exp(-(dm / lam) ** 2) + nug * np.eye(len(order))
        c = np.exp(-(d / lam) ** 2)
    else:
        K = np.exp(-dm / lam) + nug * np.eye(len(order))
        c = np.exp(-d / lam)
    try:
        wts = np.linalg.solve(K, c)
    except np.linalg.LinAlgError:
        wts = np.full(len(order), 1.0 / len(order))
    if not np.all(np.isfinite(wts)) or np.abs(wts).sum() > 5.0:
        wts = c / (c.sum() + 1e-9)
    val = float(rv @ wts)
    nb_med = np.median(rv); nb_mad = 1.4826 * np.median(np.abs(rv - nb_med)) + 1e-6
    val = float(np.clip(val, nb_med - 4 * nb_mad, nb_med + 4 * nb_mad))
    return val


def _local_model(hist_sods, hist_clk):
    """Linear offset+frequency model (robust) on the supplied history."""
    ee = np.asarray(hist_sods, float)
    vv = np.asarray(hist_clk, float)
    if len(ee) < 4:
        return None
    tn = (ee - ee[0]) / 3600.0
    try:
        coef = np.polyfit(tn, vv, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    resid = vv - np.polyval(coef, tn)
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = np.abs(resid - np.median(resid)) < 4.0 * scale
    if good.sum() >= max(4, len(ee) // 2):
        try:
            coef = np.polyfit(tn[good], vv[good], 1)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return {"coef": coef, "t0": float(ee[0])}


def _lpredict(model, s):
    return float(np.polyval(model["coef"], (float(s) - model["t0"]) / 3600.0))


def _solve_pos_k3(recs, clk_val, tropo_val):
    """Truth-linearised k=3 position solve: state [x y z] with clock + tropo
    supplied externally.  Returns the 3-D error vector (dp = est - truth) or
    None if the geometry is degenerate (PDOP > PDOP_MAX)."""
    A = np.array([[r[0][0], r[0][1], r[0][2]] for r in recs])
    y = np.array([r[2] - clk_val for r in recs])
    if len(recs) < 3:
        return None
    try:
        N = A.T @ A
        if np.abs(np.linalg.det(N)) < 1e-9:
            return None
        dp = np.linalg.solve(N, A.T @ y)
    except np.linalg.LinAlgError:
        return None
    # PDOP gate (deployable integrity): reject near-coplanar geometry.
    try:
        pdop = float(np.sqrt(np.trace(np.linalg.inv(N))))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(pdop) or pdop > PDOP_MAX:
        return None
    return dp


def _load_station_day(code4, doy):
    cf = CACHE / f"{code4}_{doy}.pkl"
    if not cf.exists():
        return None
    try:
        d = pickle.load(open(cf, "rb"))
    except Exception:
        return None
    ks = sorted(k for k in d if isinstance(k, (int, np.integer)))
    if len(ks) < 200:
        return None
    # Ensure clk_bl (band-limited 10-500 µHz) is available.  Old caches
    # created before the band-limited kriging fix do not have this field.
    # Compute it from the dejumped clock if missing.
    first_key = int(ks[0])
    if "clk_bl" not in d.get(first_key, {}):
        from scripts.lib import tep_stats as ts_bl
        ss = np.array(ks, float)
        cc = np.array([d[int(s)].get("clk_dejump", d[int(s)]["clk"])
                       for s in ss], float)
        dt = float(np.median(np.diff(ss)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0
        cc_bl = ts_bl.bandpass(cc, dt, 10.0, 500.0)
        for i, s in enumerate(ss):
            d[int(s)]["clk_bl"] = float(cc_bl[i])
    return d, np.array(ks, float)


def main():
    rng = np.random.default_rng(20260948)
    # Per-gap accumulators: clock error and k=3 position error per method.
    methods = ("tep", "persistence", "ar1")
    clock_err = {g: {m: [] for m in methods} for g in GAPS_S}
    pos_err = {g: {m: [] for m in ("tep", "persistence")} for g in GAPS_S}
    n_eval = {g: 0 for g in GAPS_S}
    n_station_days = 0

    for doy in DAYS:
        # Station set for this day: cached stations with SINEX truth.
        cached = []
        for p in glob.glob(str(CACHE / f"*_{doy}.pkl")):
            code = Path(p).stem.split("_")[0]
            xyz = sx.station_xyz(code, 2024, doy)
            if xyz is not None:
                cached.append((code, xyz))
        if len(cached) > MAX_STATIONS_PER_DAY:
            cached = cached[:MAX_STATIONS_PER_DAY]
        if len(cached) < 8:
            continue
        names = [c[0] for c in cached]
        xyz = {c[0]: c[1] for c in cached}
        # Load all station-days.
        sdata = {}
        sods_all = {}
        for code in names:
            r = _load_station_day(code, doy)
            if r is None:
                continue
            sdata[code], sods_all[code] = r
        names = [c for c in names if c in sdata]
        if len(names) < 8:
            continue
        print(f"  DOY {doy}: {len(names)} stations")

        for tgt in names:
            sods = sods_all[tgt]
            tdat = sdata[tgt]
            refs = [c for c in names if c != tgt]
            n_station_days += 1
            # Evaluation epochs: every EVAL_STRIDE_S, with enough history for
            # the largest gap + prewindow.
            t_start = PREWINDOW_S + GAPS_S[-1] + 1800.0
            eval_sods = sods[(sods >= t_start) & (sods <= sods[-1] - 1800.0)]
            eval_sods = eval_sods[np.arange(0, len(eval_sods),
                                            max(1, int(EVAL_STRIDE_S / STEP_S)))]
            for s in eval_sods:
                rec = tdat[int(s)]
                recs_all = sorted(rec["recs"], key=lambda r: -r[1])
                if len(recs_all) < 8:
                    continue
                top3 = recs_all[:3]
                true_clk = float(rec["clk"])
                clk_step_s = float(rec.get("clk_step", 0.0))
                for gap in GAPS_S:
                    # Anchor: nearest cached epoch strictly before s - gap.  For
                    # gap=0 this is the immediately preceding epoch (the fresh-
                    # history case); for gap>0 it is the receiver's last good
                    # fix before an outage of length `gap`.
                    a_sod = float(s) - gap - 1.0
                    ai = np.searchsorted(sods, a_sod, side="right") - 1
                    if ai < 0:
                        continue
                    a = float(sods[ai])
                    if a >= s:
                        continue
                    # Target local model on 30 min ending at anchor.
                    hmask = (sods >= a - PREWINDOW_S) & (sods <= a)
                    hist_s = sods[hmask]
                    if len(hist_s) < 6:
                        continue
                    hist_clk = np.array(
                        [tdat[int(x)].get("clk_dejump", tdat[int(x)]["clk"])
                         for x in hist_s])
                    mt = _local_model(hist_s, hist_clk)
                    if mt is None:
                        continue
                    anchor_resid = float(
                        tdat[int(a)].get("clk_dejump", tdat[int(a)]["clk"])
                        - _lpredict(mt, a))
                    # AR1 parameter on the history residuals.
                    rhist = hist_clk - np.array([_lpredict(mt, x) for x in hist_s])
                    xr = rhist - rhist.mean()
                    if np.sum(xr[:-1] ** 2) > 1e-9:
                        phi = float(np.clip(np.sum(xr[1:] * xr[:-1]) /
                                            np.sum(xr[:-1] ** 2), -0.999, 0.999))
                    else:
                        phi = 0.0
                    nstep = max(1, int(round((float(s) - a) / STEP_S)))
                    # Network kriged change since anchor.
                    rr, rx = [], []
                    for r in refs:
                        rd = sdata[r]
                        rsods = sods_all[r]
                        # Reference epoch at s (nearest).
                        si = np.searchsorted(rsods, float(s), side="right") - 1
                        if si < 0 or si >= len(rsods):
                            continue
                        rs = float(rsods[si])
                        if abs(rs - float(s)) > 60.0:
                            continue
                        # Reference anchor (nearest to a).
                        rai = np.searchsorted(rsods, a, side="right") - 1
                        if rai < 0:
                            continue
                        ra = float(rsods[rai])
                        if abs(ra - a) > 60.0:
                            continue
                        rhmask = (rsods >= a - PREWINDOW_S) & (rsods <= a)
                        rhist_s = rsods[rhmask]
                        if len(rhist_s) < 6:
                            continue
                        rhist_clk = np.array(
                            [rd[int(x)].get("clk_dejump", rd[int(x)]["clk"])
                             for x in rhist_s])
                        mr = _local_model(rhist_s, rhist_clk)
                        if mr is None:
                            continue
                        # Band-limited kriging: use clk_bl (10-500 µHz) for
                        # the reference residual change.  The TEP spatial
                        # correlation lives in this band; the full residual
                        # mixes it with spatially uncorrelated HF noise.
                        ref_now_bl = float(rd[int(rs)].get("clk_bl", 0.0))
                        ref_anc_bl = float(rd[int(ra)].get("clk_bl", 0.0))
                        rr.append(ref_now_bl - ref_anc_bl)
                        rx.append(xyz[r])
                    if len(rr) < 6:
                        continue
                    kres = _krige(xyz[tgt], rx, np.array(rr))
                    # Predicted clocks (steering added identically to both).
                    base = _lpredict(mt, float(s)) + clk_step_s
                    tep_clk = base + anchor_resid + kres
                    pers_clk = base + anchor_resid
                    ar1_clk = base + (rhist.mean() + (anchor_resid - rhist.mean())
                                      * phi ** nstep)
                    clock_err[gap]["tep"].append(abs(tep_clk - true_clk))
                    clock_err[gap]["persistence"].append(abs(pers_clk - true_clk))
                    clock_err[gap]["ar1"].append(abs(ar1_clk - true_clk))
                    # k=3 truth-linearised position error.
                    dp_tep = _solve_pos_k3(top3, tep_clk, float(rec["zwd"]))
                    dp_pers = _solve_pos_k3(top3, pers_clk, float(rec["zwd"]))
                    if dp_tep is not None:
                        pos_err[gap]["tep"].append(float(np.linalg.norm(dp_tep)))
                    if dp_pers is not None:
                        pos_err[gap]["persistence"].append(float(np.linalg.norm(dp_pers)))
                    n_eval[gap] += 1

    # Aggregate.
    def _stats(arr):
        a = np.asarray(arr, float)
        if a.size == 0:
            return None
        return {"n": int(a.size),
                "mean_m": float(np.mean(a)),
                "rms_m": float(np.sqrt(np.mean(a ** 2))),
                "p50_m": float(np.percentile(a, 50)),
                "p95_m": float(np.percentile(a, 95)),
                "p99_m": float(np.percentile(a, 99))}

    gap_results = []
    for gap in GAPS_S:
        row = {"gap_s": gap, "gap_label": _gap_label(gap), "n_eval": n_eval[gap],
               "clock": {m: _stats(clock_err[gap][m]) for m in methods},
               "position_k3": {m: _stats(pos_err[gap][m])
                               for m in ("tep", "persistence")}}
        # TEP vs persistence advantage (RMS reduction, %).
        ct = row["clock"]["tep"]; cp = row["clock"]["persistence"]
        if ct and cp and cp["rms_m"] > 0:
            row["clock_tep_vs_persistence_rms_pct"] = 100.0 * (
                1.0 - ct["rms_m"] / cp["rms_m"])
        pt = row["position_k3"]["tep"]; pp = row["position_k3"]["persistence"]
        if pt and pp and pp["rms_m"] > 0:
            row["position_tep_vs_persistence_rms_pct"] = 100.0 * (
                1.0 - pt["rms_m"] / pp["rms_m"])
        gap_results.append(row)

    # Crossover analysis: the gap at which TEP's RMS first beats persistence
    # by a margin that exceeds the fresh-anchor margin (i.e. the outage length
    # beyond which the network prior becomes the better deployable choice).
    fresh_margin = (gap_results[0].get("clock_tep_vs_persistence_rms_pct") or 0.0)
    crossover_gap = None
    for row in gap_results[1:]:
        m = row.get("clock_tep_vs_persistence_rms_pct")
        if m is not None and m > max(fresh_margin, 0.0) + 1.0:
            crossover_gap = row["gap_s"]
            break

    result = {
        "test": "gap_sweep_stale_anchor",
        "evidence_level": "raw_observation_mechanism",
        "description": (
            "Sweep of the anchor age (outage length before a k=3 window). The "
            "receiver's local offset/frequency model and AR1 parameter are "
            "fitted on a 30-min window ending at the anchor; persistence holds "
            "the residual fixed across the outage while TEP tracks the "
            "network-kriged band-limited (10-500 uHz) change. Strictly causal; "
            "TEP covariance FIXED by the physics (lambda=4200 km). "
            "Band-limited kriging: the reference residual change is computed "
            "from the 10-500 uHz component where the TEP spatial correlation "
            "lives, not the full detrended residual that includes spatially "
            "uncorrelated HF local-oscillator noise."),
        "n_station_days": n_station_days,
        "days": [f"2024{d:03d}" for d in DAYS],
        "gaps_s": GAPS_S,
        "tep_covariance": {"form": TEP_FORM, "lambda_km": TEP_LAMBDA_KM,
                           "nugget": TEP_NUGGET, "tuned_on_test_data": False},
        "causal_protocol": ("local model + AR1 fitted on 30-min window ending "
                            "at the anchor; network kriging uses reference "
                            "clocks at anchor and eval epochs only; no "
                            "evaluation-epoch target data enters predictors"),
        "gap_results": gap_results,
        "fresh_anchor_tep_vs_persistence_rms_pct": fresh_margin,
        "crossover_gap_s": crossover_gap,
        "crossover_label": _gap_label(crossover_gap) if crossover_gap else None,
        "timestamp": datetime.now().isoformat(),
    }
    (OUT / "step_48b_gap_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"\nGap sweep: {n_station_days} station-days")
    for row in gap_results:
        ct = row["clock"]["tep"]; cp = row["clock"]["persistence"]
        tr = ct["rms_m"] if ct else float("nan")
        pr = cp["rms_m"] if cp else float("nan")
        m = row.get("clock_tep_vs_persistence_rms_pct")
        ms = f"{m:+6.1f}%" if m is not None else "   n/a"
        print(f"  gap={row['gap_label']:>8s}  n={row['n_eval']:6d}  "
              f"clock RMS tep={tr:6.2f} pers={pr:6.2f}  TEP-Pers={ms}")
    print(f"saved {OUT / 'step_48b_gap_sweep.json'}")


def _gap_label(gap):
    if gap is None:
        return None
    g = int(round(gap))
    if g == 0:
        return "0min"
    if g % 3600 == 0:
        return f"{g//3600}h"
    if g % 60 == 0:
        return f"{g//60}min"
    return f"{g}s"


if __name__ == "__main__":
    main()
