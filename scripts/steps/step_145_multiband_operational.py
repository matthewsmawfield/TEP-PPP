#!/usr/bin/env python3
"""
STEP 145  --  Multi-band TEP prior in the degraded-sky positioning flagship.

OPERATIONALISE the multi-band frequency-dependent TEP kernel (Step 134) in
the Step 106 degraded-sky positioning scenario.  The single-lambda Step 96
TEP predictor is replaced by the multi-band predictor that uses Paper 1's
frequency-dependent lambdas:

  * Tidal band    (10-35 uHz):   lambda = 4,677 km
  * Post-tidal    (35-100 uHz):  lambda = 1,502 km
  * Intermediate  (100-500 uHz): lambda = 1,459 km

The multi-band predictor computes a separate phase Gram for each frequency
band, uses the physics-fixed lambda for that band, and averages the per-band
kriging predictions.  This is the same construction as Step 134, but applied
to the SPP receiver-clock accumulated offset in the degraded-sky positioning
test (Step 106).

ARMS per outage epoch, on the k lowest-elevation visible satellites (k = 4, 5):
  free_k       : clock free                          (standard, floor-limited)
  persist_k    : clock = last SPP + drift*gap, soft   (deployable baseline)
  tep_k        : Step 96 single-lambda phase-Gram prior, soft
  tep_multi_k  : multi-band frequency-dependent prior, soft  (the claim)
  tep_scr_k    : as tep_k, spatial term scaled by screening
  netmean_k    : unweighted network mean (coarse TEP)
  distscr_k    : distance-scrambled null
  phasescr_k   : phase-scrambled null

The decisive comparison is tep_multi vs tep (does the multi-band kernel
improve on the single-lambda kernel in the POSITION domain?) and tep_multi
vs persist (does it beat the deployable baseline?).

Output: results/outputs/step_145_multiband_operational.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td      # noqa: E402
from scripts.lib import tep_sinex as sx     # noqa: E402
from scripts.lib import tep_spp as spp      # noqa: E402
from scripts.lib.tep_phase import spectral_phase_gram  # noqa: E402

_s99 = importlib.util.spec_from_file_location(
    "s99", ROOT / "scripts" / "steps" / "step_99_precise_clock_position.py")
s99 = importlib.util.module_from_spec(_s99)
_s99.loader.exec_module(s99)
s91 = s99.s91

# Import Step 106 itself to reuse its infrastructure
_s106 = importlib.util.spec_from_file_location(
    "s106", ROOT / "scripts" / "steps" / "step_106_degraded_sky_position_flagship.py")
s106 = importlib.util.module_from_spec(_s106)
_s106.loader.exec_module(s106)

OUT = ROOT / "results" / "outputs" / "step_145_multiband_operational.json"
CACHE = ROOT / "data" / "step48_cache"
C_M_PER_NS = 0.299792458
DT_S = s99.DT_S
PRE_S = s99.PRE_S
GRAM_DAYS = s99.GRAM_DAYS
SIGMA_FLOOR_M = s99.SIGMA_FLOOR_M
SIGMA_CAP_M = s99.SIGMA_CAP_M

DOYS = [int(x) for x in os.environ.get("TEP_S145_DOYS",
        "99,100,101,102,103,104,105,106").split(",") if x]
GAPS = [float(x) for x in os.environ.get("TEP_S145_GAPS", "3600,14400").split(",")]
MAX_STN = int(os.environ.get("TEP_S145_MAX_STN", "20"))
KS = (4, 5)
RNG = np.random.default_rng(20260915)

# Paper 1 frequency-dependent lambdas (frozen from Step 134)
BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

LAMBDA_KM = s99.LAMBDA_KM
NUGGET = s99.NUGGET
DNR_SWITCH = s99.DNR_SWITCH


def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _band_phase_grams(x, dt_s):
    """Compute a phase Gram for each frequency band."""
    grams = []
    for band in BANDS:
        try:
            result = spectral_phase_gram(
                x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append((result["predictive_gram"], band))
        except ValueError:
            continue
    if not grams:
        return None
    return grams


def _gram_multiband(tag, names, ndays):
    """Frozen multi-band phase Grams from `ndays` strictly-prior days."""
    mats = []
    common = None
    for k in range(ndays + 3, 0, -1):
        if len(mats) >= ndays:
            break
        ptag = (datetime.strptime(tag, "%Y%j") - timedelta(days=k)).strftime("%Y%j")
        r, _ = s106._field_day(ptag, names)
        col = {names[i]: r[:, i] for i in range(len(names)) if np.isfinite(r[:, i]).any()}
        if len(col) < 8:
            continue
        common = set(col) if common is None else (common & set(col))
        mats.append(col)
    if not common or len(common) < 8 or len(mats) < 3:
        return None
    order = sorted(common)
    Y = np.vstack([np.column_stack([c[n] for n in order]) for c in mats])
    cols, kept = [], []
    for jj, nm in enumerate(order):
        y, frac = s99._interp_gaps(Y[:, jj])
        if y is None or frac < 0.9 or not (1e-4 < np.std(y) < 1e6):
            continue
        cols.append(y); kept.append(nm)
    if len(kept) < 8:
        return None
    band_grams = _band_phase_grams(np.column_stack(cols), DT_S)
    if band_grams is None:
        return None
    return kept, [g for g, _ in band_grams]


# Per-day state for the multi-band predictor
_MB_GRAM_NAMES: list = []
_MB_BAND_GRAMS: list = []
_MB_NAME2GRAM: dict = {}
_D_SCR = None
_G_SCR = None


def _tep_multi_accum(ff, raw_ff, D, mb_gram, b, ti, kstep, npre):
    """Multi-band TEP prediction of the target's SPP clock accumulated offset.

    Same structure as s106._tep96_accum but uses per-band phase Grams and
    per-band lambdas, averaging the per-band kriging predictions.
    """
    mb_names, band_grams = mb_gram
    if b not in _MB_NAME2GRAM:
        return None
    tg = _MB_NAME2GRAM[b]
    h = kstep
    rate_b, span_b, dnr_b, amp_b = s106._prewin_rate(raw_ff[:, b], ti, npre)
    g_b = 1.0 if dnr_b > DNR_SWITCH else 0.0
    lr_b = float(np.clip(rate_b * h, -8.0 * span_b, 8.0 * span_b))

    pool, pool_g, cs, rate_p, g_p, amp_p = [], [], [], [], [], []
    for j, pg in _MB_NAME2GRAM.items():
        if j == b:
            continue
        a0 = raw_ff[ti - 1, j]
        if not np.isfinite(a0):
            continue
        # CAUSAL: predict reference change from past data, not future R[ti+kstep, j]
        rj, sj, dj, aj = s106._prewin_rate(raw_ff[:, j], ti, npre)
        a1_pred = a0 + rj * h  # AR(1)-style linear extrapolation
        pool.append(j); pool_g.append(pg)
        cs.append(a1_pred - a0)
        rate_p.append(rj)
        g_p.append(1.0 if dj > DNR_SWITCH else 0.0)
        amp_p.append(aj)
    if len(pool) < 6:
        return None

    cs = np.asarray(cs); rate_p = np.asarray(rate_p); g_p = np.asarray(g_p)
    cs_gated = cs - g_p * rate_p * h
    d_rr = D[np.ix_(pool, pool)]
    d_rt = D[b, pool]

    # Per-band kriging, then average
    band_preds = []
    for G_band, band_info in zip(band_grams, BANDS):
        scales = _multi_scales(band_info["lambda_km"])
        rho_rr_b = _rho_multi(d_rr, scales)
        rho_rt_b = _rho_multi(d_rt, scales)
        G_pp = G_band[np.ix_(pool_g, pool_g)]
        G_tp = G_band[tg, pool_g]
        try:
            w_band = _reg_solve(
                rho_rr_b * G_pp + NUGGET * np.eye(len(pool)),
                rho_rt_b * G_tp)
            band_preds.append(float(w_band @ cs_gated))
        except np.linalg.LinAlgError:
            continue

    if not band_preds:
        return None
    spatial = float(np.mean(band_preds))
    tep_multi_accum = g_b * lr_b + spatial

    # Screening variant
    resp_b = float(np.clip(amp_b / max(np.median(amp_p), 1e-6), 0.25, 3.0))
    tep_multi_scr_accum = g_b * lr_b + resp_b * spatial

    return {"tep_multi": float(tep_multi_accum),
            "tep_multi_scr": float(tep_multi_scr_accum)}


ACCUM_KINDS = ("tep", "tep_scr", "tep_multi", "tep_multi_scr",
               "persist", "netmean", "distscr", "phasescr")


def _score_day(day, mb_gram):
    global _MB_GRAM_NAMES, _MB_BAND_GRAMS, _MB_NAME2GRAM, _D_SCR, _G_SCR
    names, cache, xyz = day["names"], day["cache"], day["xyz"]
    raw, field, D, gram, grid, t_ref = (day["raw"], day["field"],
                                        day["D"], day["gram"],
                                        day["grid"], day["t_ref"])
    gram_names, Gmat = gram
    gidx = {n: i for i, n in enumerate(gram_names)}

    # Multi-band gram setup
    mb_names, mb_band_grams = mb_gram
    mb_gidx = {n: i for i, n in enumerate(mb_names)}
    _MB_GRAM_NAMES = mb_names
    _MB_BAND_GRAMS = mb_band_grams
    _MB_NAME2GRAM = {b: mb_gidx[n] for b, n in enumerate(names) if n in mb_gidx}

    # Single-lambda gram setup (from Step 106)
    s106._NAME2GRAM = {b: gidx[n] for b, n in enumerate(names) if n in gidx}

    # Per-day frozen permutations for nulls
    dperm = RNG.permutation(len(names))
    s106._D_SCR = D[np.ix_(dperm, dperm)]
    _D_SCR = s106._D_SCR
    gperm = RNG.permutation(len(gram_names))
    s106._G_SCR = Gmat[np.ix_(gperm, gperm)]
    _G_SCR = s106._G_SCR

    dt_s = float(np.median(np.diff(t_ref)))
    npre = int(PRE_S / dt_s)
    ff = np.column_stack([s99._causal_fill(field[:, j]) for j in range(len(names))])
    rff = np.column_stack([s99._causal_fill(raw[:, j]) for j in range(len(names))])
    arms = [f"{a}_k{k}" for k in KS for a in ("free",) + ACCUM_KINDS]
    out = {g: {a: [] for a in arms} for g in GAPS}
    clk_err = {g: {a: [] for a in ACCUM_KINDS} for g in GAPS}
    SIG_CAL = int(os.environ.get("TEP_S145_SIGCAL", "4"))

    for gap in GAPS:
        kstep = int(round(gap / dt_s))
        for b, tgt in enumerate(names):
            if b not in s106._NAME2GRAM or b not in _MB_NAME2GRAM:
                continue
            truth = xyz[tgt]
            tdat = cache[tgt]
            for ti in range(npre, len(t_ref) - kstep,
                            int(os.environ.get("TEP_S145_STRIDE", "5"))):
                sod = int(round(t_ref[ti + kstep]))
                rec = tdat.get(sod) or tdat.get(sod - sod % 30)
                if rec is None or "raw_obs" not in rec or len(rec["raw_obs"]) < 6:
                    continue
                if not np.isfinite(raw[ti, b]):
                    continue
                recs = rec.get("recs", [])
                obs = list(rec["raw_obs"].items())
                if len(recs) != len(obs):
                    continue

                # Single-lambda TEP (Step 106)
                acc_single = s106._tep96_accum(ff, rff, D, gram, b, ti, kstep, npre)
                # Multi-band TEP
                acc_multi = _tep_multi_accum(ff, rff, D, mb_gram, b, ti, kstep, npre)
                if acc_single is None or acc_multi is None:
                    continue

                spp0 = float(rff[ti, b])
                cl = lambda a: float(np.clip(spp0 + a, spp0 - 50.0, spp0 + 50.0))
                clk = {
                    "tep": cl(acc_single["tep"]),
                    "tep_scr": cl(acc_single["tep_scr"]),
                    "tep_multi": cl(acc_multi["tep_multi"]),
                    "tep_multi_scr": cl(acc_multi["tep_multi_scr"]),
                    "persist": cl(acc_single["persist"]),
                    "netmean": cl(acc_single["netmean"]),
                    "distscr": cl(acc_single["distscr"]),
                    "phasescr": cl(acc_single["phasescr"]),
                }

                # Matched rolling-origin sigma
                errs = {kd: [] for kd in ACCUM_KINDS}
                lo, hi = npre + 1, ti - kstep
                if hi > lo:
                    stride = max(1, (hi - lo) // SIG_CAL)
                    for tp in range(lo, hi, stride):
                        tru = rff[tp + kstep, b]
                        if not np.isfinite(tru):
                            continue
                        a2s = s106._tep96_accum(ff, rff, D, gram, b, tp, kstep, npre)
                        a2m = _tep_multi_accum(ff, rff, D, mb_gram, b, tp, kstep, npre)
                        if a2s is None or a2m is None:
                            continue
                        s0 = float(rff[tp, b])
                        for kd in ("tep", "tep_scr", "persist", "netmean",
                                   "distscr", "phasescr"):
                            errs[kd].append(
                                (float(np.clip(s0 + a2s[kd], s0 - 50, s0 + 50)) - tru) ** 2)
                        for kd in ("tep_multi", "tep_multi_scr"):
                            errs[kd].append(
                                (float(np.clip(s0 + a2m[kd], s0 - 50, s0 + 50)) - tru) ** 2)
                sig = {kd: (max(SIGMA_FLOOR_M, min(float(np.sqrt(np.mean(errs[kd]))), SIGMA_CAP_M))
                            if len(errs[kd]) >= 4 else 5.0) for kd in ACCUM_KINDS}

                true_clk = float(rff[ti + kstep, b])
                if np.isfinite(true_clk):
                    for kd in ACCUM_KINDS:
                        clk_err[gap][kd].append(abs(clk[kd] - true_clk))

                init = s91._blind_init(truth)
                obs_e = sorted(zip(obs, recs), key=lambda x: x[1][1])
                for k in KS:
                    if len(obs_e) < k:
                        continue
                    ok = [o for o, r in obs_e[:k]]
                    ef = s99._solve(ok, grid, sod, init, truth, None, None, None)
                    if ef is not None:
                        out[gap][f"free_k{k}"].append(ef)
                    for kd in ACCUM_KINDS:
                        e = s99._solve(ok, grid, sod, init, truth,
                                       clk[kd], sig[kd], None)
                        if e is not None:
                            out[gap][f"{kd}_k{k}"].append(e)
    return out, clk_err


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else None


def _boot_pct(per_day, num, den):
    days = [d for d in per_day if per_day[d][num] and per_day[d][den]]
    if len(days) < 3:
        return None
    vals = []
    for _ in range(4000):
        sel = list(RNG.choice(days, len(days), replace=True))
        n = np.concatenate([per_day[d][num] for d in sel])
        de = np.concatenate([per_day[d][den] for d in sel])
        vals.append(100.0 * (1.0 - _rms(n) / _rms(de)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> int:
    print("Step 145: Multi-band TEP prior in degraded-sky positioning", flush=True)
    print(f"  DOYS: {DOYS}", flush=True)
    print(f"  Gaps: {GAPS}s, K={KS}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, "
              f"lambda={band['lambda_km']:.0f} km", flush=True)
    print(flush=True)

    per_day = {}
    clk_per_day = {}
    used = []
    for doy in DOYS:
        day = s106._prepare_day(doy)
        if day is None:
            print(f"  DOY {doy}: skip (setup)")
            continue
        mb_gram = _gram_multiband(day["tag"], day["names"], GRAM_DAYS)
        if mb_gram is None:
            print(f"  DOY {doy}: skip (multi-band gram)")
            continue
        scored, ce = _score_day(day, mb_gram)
        per_day[day["tag"]] = scored
        clk_per_day[day["tag"]] = ce
        used.append(day["tag"])
        ex = scored[GAPS[0]]
        print(f"DOY {doy}: {len(day['names'])} stns  "
              f"k4 n(free/persist/tep/tep_multi)="
              f"{len(ex['free_k4'])}/{len(ex['persist_k4'])}/"
              f"{len(ex['tep_k4'])}/{len(ex['tep_multi_k4'])}",
              flush=True)

    if len(used) < 3:
        OUT.write_text(json.dumps({"step": 145, "status": "insufficient_days",
                                   "days": used}, indent=2))
        print("insufficient days"); return 0

    result = {
        "step": 145,
        "title": "Multi-band TEP prior in degraded-sky positioning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days": used, "n_days": len(used),
        "station_class": "Step 91 dense regional network",
        "gaps_s": GAPS, "k_degraded": list(KS),
        "frozen_spec": {
            "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6,
                        "f_hi_uhz": b["f_hi"]*1e6,
                        "lambda_km": b["lambda_km"]} for b in BANDS],
            "nugget": NUGGET, "gram_days": GRAM_DAYS,
            "dnr_switch": DNR_SWITCH,
            "single_lambda_km": LAMBDA_KM,
            "note": "Multi-band lambdas from Paper 1 (Step 134), "
                    "single-lambda from Step 96. Not re-tuned.",
        },
        "by_gap": {},
    }
    KINDS = ("free",) + ACCUM_KINDS
    BASELINES = ("free", "persist", "netmean", "distscr", "phasescr", "tep")
    for gap in GAPS:
        gs = str(int(gap))
        pd = {d: per_day[d][gap] for d in used}
        row = {"n_days": len(used)}
        for k in KS:
            for a in KINDS:
                pool = (np.concatenate([pd[d][f"{a}_k{k}"] for d in used
                                        if pd[d][f"{a}_k{k}"]])
                        if any(pd[d][f"{a}_k{k}"] for d in used) else np.array([]))
                row[f"{a}_k{k}"] = {
                    "n": int(len(pool)), "rms_3d_m": _rms(pool),
                    "p50_m": float(np.percentile(pool, 50)) if len(pool) else None,
                    "p95_m": float(np.percentile(pool, 95)) if len(pool) else None,
                    "gt25m_rate": float(np.mean(pool > 25)) if len(pool) else None,
                }
            for tep_ in ("tep", "tep_scr", "tep_multi", "tep_multi_scr"):
                tv = row[f"{tep_}_k{k}"]
                for base in BASELINES:
                    bv = row[f"{base}_k{k}"]
                    if tv["rms_3d_m"] and bv["rms_3d_m"]:
                        row[f"{tep_}_vs_{base}_k{k}_pct"] = 100.0 * (1 - tv["rms_3d_m"] / bv["rms_3d_m"])
                        row[f"{tep_}_vs_{base}_k{k}_ci95"] = _boot_pct(
                            {d: {"n": pd[d][f"{tep_}_k{k}"], "d": pd[d][f"{base}_k{k}"]} for d in used},
                            "n", "d")
        ce = {d: clk_per_day[d][gap] for d in used}
        for a in ACCUM_KINDS:
            allc = np.concatenate([np.asarray(ce[d][a]) for d in used if ce[d][a]]) \
                if any(ce[d][a] for d in used) else np.array([])
            if len(allc):
                row[f"clk_err_{a}_rms_m"] = float(np.sqrt(np.mean(allc ** 2)))
        result["by_gap"][gs] = row

    # Verdict: does multi-band beat single-lambda in position domain?
    REQ_BASE = ("free", "persist")
    cells = {}
    for gap in GAPS:
        gs = str(int(gap))
        r = result["by_gap"][gs]
        for k in KS:
            cell = {}
            for base in REQ_BASE:
                ci = r.get(f"tep_multi_vs_{base}_k{k}_ci95")
                cell[f"multi_beats_{base}"] = ci is not None and ci[0] > 0
            ci_tep = r.get(f"tep_multi_vs_tep_k{k}_ci95")
            cell["multi_beats_single"] = ci_tep is not None and ci_tep[0] > 0
            ci_phase = r.get(f"tep_multi_vs_phasescr_k{k}_ci95")
            cell["multi_beats_phasescr"] = ci_phase is not None and ci_phase[0] > 0
            cells[f"k{k}_{gs}s"] = cell
    result["cells"] = cells

    n_base = sum(any(c[f"multi_beats_{b}"] for b in REQ_BASE) for c in cells.values())
    n_single = sum(c["multi_beats_single"] for c in cells.values())
    result["verdict"] = {
        "multi_band_beats_persistence_or_free": n_base >= 2,
        "multi_band_beats_single_lambda": n_single >= 2,
        "n_cells_beats_baseline": n_base,
        "n_cells_beats_single": n_single,
        "n_cells": len(cells),
        "criterion": ("Multi-band TEP prior beats free-clock AND/OR persistence "
                      "in >=2 (k,gap) cells (operational claim), AND beats the "
                      "single-lambda TEP prior in >=2 cells (multi-band improvement)."),
    }

    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nsaved {OUT}")
    for gs, r in result["by_gap"].items():
        print(f"\ngap {gs}s  clk_err: " + "  ".join(
            f"{a} {r.get(f'clk_err_{a}_rms_m', 0):.2f}" for a in ACCUM_KINDS))
        for k in KS:
            print(f"  k={k}: " + "  ".join(
                f"{a} {r[f'{a}_k{k}']['rms_3d_m'] or 0:.2f}" for a in KINDS))
            for tep_ in ("tep", "tep_multi"):
                for base in ("free", "persist", "tep"):
                    if tep_ == "tep" and base == "tep":
                        continue
                    print(f"     {tep_} vs {base}: "
                          f"{r.get(f'{tep_}_vs_{base}_k{k}_pct', float('nan')):+.1f}%  "
                          f"CI {r.get(f'{tep_}_vs_{base}_k{k}_ci95')}")
    print(f"\nVERDICT: {json.dumps(result['verdict'], indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
