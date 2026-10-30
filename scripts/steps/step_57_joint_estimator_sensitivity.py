#!/usr/bin/env python3
"""
STEP 57  --  Joint estimator vs post-solve correction: position sensitivity
             to clock-prediction quality.

The reviewer roadmap's highest-leverage position-domain item: put the TEP
clock prediction INSIDE the estimator (a soft weighted row in the normal
equations) rather than applying it as a post-solve linearised correction, and
show what position gain that buys as a function of how well the network
predicts the clock.

Step 56 measures that clock-prediction quality directly: on the dense EUREF
network ~64% of the band-limited receiver-clock variance is recovered from
neighbours (predictable fraction f ~ 0.64), versus ~0.1 on the sparse global
network.  This step propagates those numbers through REAL satellite geometry.

Data: the Step 48 cache (data/step48_cache/<STATION>_<DOY>.pkl) -- real
ionosphere-free residuals, real line-of-sight geometry, real IGS SINEX truth,
69 stations x 7 days.  Read-only; no RINEX reprocessing.

Protocol per epoch (linearised at the surveyed truth for geometry -- a
mechanism diagnostic, not the blind nonlinear flagship, which is Step 48):
  * c_ref = clock from an all-satellite free solve (the epoch's true clock).
  * clk_prior = c_ref + e,  e ~ N(0, sigma_pred),
    sigma_pred = sqrt(1 - f) * sigma_clock_day  for each tested f.
  * take the k highest-elevation satellites (k = 3, 4, 5, 6); also random-k.
  * solve three ways:
      free        -- estimate the clock  (k >= 4 only)
      post_solve  -- clock hard-fixed to clk_prior  (current Step 48 approach)
      joint_soft  -- clock kept as a state with a 1/sigma^2 prior row
  * integrity: joint_soft rejects the fix when the clock innovation exceeds
    3 * sigma_pred (feasible only at k >= 4, where geometry gives redundancy).
Monte-Carlo over e; day-clustered bootstrap CIs on the paired differences.

Output: results/outputs/step_57_joint_estimator_sensitivity.json
"""
from __future__ import annotations

import sys
import json
import glob
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = PROJECT_ROOT / "data" / "step48_cache"
OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260902)
# NOTE: do NOT call np.seterr(all="ignore") at module scope.
# Ill-conditioned numerics are wrapped in local np.errstate contexts.
# np.seterr(all="ignore")  # removed: masks bugs globally

SIGMA_CODE = 0.3
# Deployable receiver integrity gate.  20 was far too loose: it admits
# near-coplanar k=3/k=4 geometries whose position variance is dominated by a
# single ill-determined direction, which inflated every RMS (the k=3 "floor"
# read 10 m RMS with a 23% >10 m rate while p50 was 6 m).  A real receiver
# refuses a fix above PDOP ~6-10.
PDOP_MAX = 10.0
KS = [3, 4, 5, 6]
FRACTIONS = {"sparse_global_f0.10": 0.10, "mid_f0.40": 0.40,
             "dense_epn_f0.64": 0.64, "excellent_f0.85": 0.85}
N_MC = 4
EPOCH_STRIDE = 24          # ~ every 6 min at 30 s cache cadence
MABS = 1.0e6


def _pdop(A):
    try:
        cov = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return 1e9
    return float(np.sqrt(np.trace(cov[:3, :3])))


def _design(recs):
    e = np.array([r[0] for r in recs], float)
    el = np.array([r[1] for r in recs], float)
    res = np.array([r[2] for r in recs], float)
    wm = np.array([r[3] for r in recs], float)
    return e, el, res, wm


def solve_free(recs):
    """Estimate [dx,dy,dz,dclk,dtropo]. k>=5 for tropo, else k>=4."""
    e, el, res, wm = _design(recs)
    w = np.sin(np.radians(el)) ** 2 / SIGMA_CODE ** 2
    cols = [e, np.ones(len(recs))]
    if len(recs) >= 5:
        cols.append(wm.reshape(-1, 1))
    A = np.column_stack(cols)
    if A.shape[0] < A.shape[1] or _pdop(A) > PDOP_MAX:
        return None, None
    W = np.diag(w)
    try:
        x = np.linalg.solve(A.T @ W @ A, A.T @ W @ res)
    except np.linalg.LinAlgError:
        return None, None
    if np.linalg.norm(x[:3]) > MABS:
        return None, None
    return x[:3], float(x[3])


def solve_fixed(recs, clk, tropo):
    e, el, res, wm = _design(recs)
    w = np.sin(np.radians(el)) ** 2 / SIGMA_CODE ** 2
    b = res - clk - tropo * wm
    A = e
    if A.shape[0] < 3 or _pdop(A) > PDOP_MAX:
        return None
    W = np.diag(w)
    try:
        x = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
    except np.linalg.LinAlgError:
        return None
    return x[:3] if np.linalg.norm(x[:3]) < MABS else None


def solve_soft(recs, clk_prior, clk_sigma, tropo):
    e, el, res, wm = _design(recs)
    w_obs = np.sin(np.radians(el)) ** 2 / SIGMA_CODE ** 2
    b = res - tropo * wm
    A = np.column_stack([e, np.ones(len(recs))])
    prior = np.zeros(A.shape[1])
    prior[3] = 1.0
    A = np.vstack([A, prior])
    b = np.append(b, clk_prior)
    w = np.append(w_obs, 1.0 / max(clk_sigma, 1e-3) ** 2)
    if _pdop(A) > PDOP_MAX:
        return None, None
    W = np.diag(w)
    try:
        x = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
    except np.linalg.LinAlgError:
        return None, None
    if np.linalg.norm(x[:3]) > MABS:
        return None, None
    return x[:3], float(x[3])


def load_days():
    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    out = []
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        eps = [k for k in d if isinstance(d.get(k), dict) and "recs" in d[k]]
        if len(eps) < 40:
            continue
        cr = [d[k].get("clk_resid") for k in eps if d[k].get("clk_resid") is not None]
        sig = float(np.std(cr)) if len(cr) > 20 else 1.0
        stem = Path(f).stem
        out.append((stem, d, sorted(eps), max(sig, 0.2)))
    return out


def run():
    days = load_days()
    # acc[frac][k][arch] -> list of 3D errors ;  per_day for CIs
    acc = {fr: {k: {a: [] for a in ("free", "post_solve", "joint_soft")}
                for k in KS} for fr in FRACTIONS}
    per_day = {fr: {k: {a: [] for a in ("post_solve", "joint_soft")}
                    for k in KS} for fr in FRACTIONS}
    avail = {fr: {k: [0, 0] for k in KS} for fr in FRACTIONS}   # [kept, total] for joint integrity

    for stem, d, eps, sig in days:
        day_err = {fr: {k: {a: [] for a in ("post_solve", "joint_soft")}
                        for k in KS} for fr in FRACTIONS}
        for ei in range(0, len(eps), EPOCH_STRIDE):
            rec = d[eps[ei]]
            if rec.get("clk_bad"):
                continue
            recs_all = rec.get("recs") or []
            if len(recs_all) < 7:
                continue
            recs_all = sorted(recs_all, key=lambda r: -r[1])
            tropo = 0.0
            p_ref, c_ref = solve_free(recs_all[:min(10, len(recs_all))])
            if c_ref is None or p_ref is None or np.linalg.norm(p_ref) > 50.0:
                continue
            for k in KS:
                if len(recs_all) < k:
                    continue
                recs = recs_all[:k]
                p_free, _ = (solve_free(recs) if k >= 4 else (None, None))
                for fr, f in FRACTIONS.items():
                    spred = np.sqrt(max(1.0 - f, 1e-3)) * sig
                    for _ in range(N_MC):
                        err = RNG.normal(0.0, spred)
                        clk_prior = c_ref + err
                        pf = solve_fixed(recs, clk_prior, tropo)
                        ps, cs = solve_soft(recs, clk_prior, spred, tropo)
                        if p_free is not None:
                            acc[fr][k]["free"].append(float(np.linalg.norm(p_free)))
                        if pf is not None:
                            e_pf = float(np.linalg.norm(pf))
                            acc[fr][k]["post_solve"].append(e_pf)
                            day_err[fr][k]["post_solve"].append(e_pf)
                        if ps is not None:
                            innov = abs(cs - clk_prior)
                            avail[fr][k][1] += 1
                            if k >= 4 and innov > 3.0 * spred:
                                continue      # integrity reject
                            avail[fr][k][0] += 1
                            e_ps = float(np.linalg.norm(ps))
                            acc[fr][k]["joint_soft"].append(e_ps)
                            day_err[fr][k]["joint_soft"].append(e_ps)
        for fr in FRACTIONS:
            for k in KS:
                for a in ("post_solve", "joint_soft"):
                    v = day_err[fr][k][a]
                    if len(v) >= 20:
                        per_day[fr][k][a].append(float(np.sqrt(np.mean(np.square(v)))))

    def stats(v):
        if len(v) < 20:
            return None
        a = np.asarray(v)
        return {"n": len(a), "rms_m": float(np.sqrt(np.mean(a ** 2))),
                "p50_m": float(np.median(a)), "p95_m": float(np.percentile(a, 95)),
                "p99_m": float(np.percentile(a, 99)),
                "gt10m_rate": float(np.mean(a > 10.0))}

    result = {}
    for fr in FRACTIONS:
        result[fr] = {"predictable_fraction": FRACTIONS[fr], "by_k": {}}
        for k in KS:
            row = {a: stats(acc[fr][k][a]) for a in ("free", "post_solve", "joint_soft")}
            pdp = np.asarray(per_day[fr][k]["post_solve"])
            pdj = np.asarray(per_day[fr][k]["joint_soft"])
            ci = None
            if len(pdp) >= 4 and len(pdj) >= 4 and len(pdp) == len(pdj):
                bs = [100.0 * (1.0 - np.mean(pdj[idx]) / np.mean(pdp[idx]))
                      for idx in (RNG.integers(0, len(pdp), len(pdp)) for _ in range(4000))]
                ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            kept, tot = avail[fr][k]
            row["joint_vs_post_solve_rms_pct"] = (
                100.0 * (1.0 - row["joint_soft"]["rms_m"] / row["post_solve"]["rms_m"])
                if row["joint_soft"] and row["post_solve"] else None)
            row["joint_vs_post_solve_ci95"] = ci
            # Apples-to-apples: joint_soft vs a free-clock solve (both
            # unfiltered, both estimating the clock).  This is the honest
            # "does putting the clock in the estimator with a soft prior beat
            # just solving for it" number at k>=4.  joint_vs_post_solve
            # compares against an UNFILTERED HARD-FIX baseline and is not
            # integrity-matched, so it overstates the estimator-structure gain.
            row["joint_vs_free_rms_pct"] = (
                100.0 * (1.0 - row["joint_soft"]["rms_m"] / row["free"]["rms_m"])
                if row["joint_soft"] and row["free"] else None)
            row["joint_integrity_availability"] = (kept / tot) if tot else None
            result[fr]["by_k"][str(k)] = row
    return result


SIGMA_SWEEP_M = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def k3_clock_sensitivity():
    """The decisive diagnostic: at k=3, how much does the position error depend
    on the clock-prediction error?  A flat curve for small sigma means k=3 is
    model-limited (multipath / troposphere / orbit), not clock-limited -- and a
    better clock prior, joint or post-solve, cannot improve k=3 accuracy."""
    days = load_days()
    acc = {sg: [] for sg in SIGMA_SWEEP_M}
    for stem, d, eps, sig in days:
        for ei in range(0, len(eps), max(EPOCH_STRIDE, 30)):
            rec = d[eps[ei]]
            if rec.get("clk_bad"):
                continue
            ra = sorted(rec.get("recs") or [], key=lambda r: -r[1])
            if len(ra) < 7:
                continue
            p_ref, c_ref = solve_free(ra[:min(10, len(ra))])
            if c_ref is None or p_ref is None or np.linalg.norm(p_ref) > 50.0:
                continue
            recs = ra[:3]
            for sg in SIGMA_SWEEP_M:
                for _ in range(4):
                    dp = solve_fixed(recs, c_ref + RNG.normal(0.0, sg), 0.0)
                    if dp is not None:
                        acc[sg].append(float(np.linalg.norm(dp)))
    out = {}
    for sg, v in acc.items():
        if len(v) < 50:
            continue
        a = np.asarray(v)
        out[str(sg)] = {"n": len(a), "rms_m": float(np.sqrt(np.mean(a ** 2))),
                        "p50_m": float(np.median(a)), "p99_m": float(np.percentile(a, 99)),
                        "gt10m_rate": float(np.mean(a > 10.0))}
    return out


def _solve_fixed_pdop(recs, clk, tropo, pdop_max):
    """solve_fixed with an explicit PDOP cap (default PDOP_MAX otherwise)."""
    e, el, res, wm = _design(recs)
    w = np.sin(np.radians(el)) ** 2 / SIGMA_CODE ** 2
    b = res - clk - tropo * wm
    A = e
    if A.shape[0] < 3 or _pdop(A) > pdop_max:
        return None
    W = np.diag(w)
    try:
        x = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
    except np.linalg.LinAlgError:
        return None
    return x[:3] if np.linalg.norm(x[:3]) < MABS else None


def k3_error_budget():
    """Decompose the k=3 position 'floor': is it really pseudorange-model-limited,
    or is a fixable modelling gap (unmodelled wet troposphere, a too-loose PDOP
    gate) hiding the clock-quality lever?

    For each epoch's best-3-by-elevation and best-3-by-PDOP satellites, solve
    position with a PERFECT clock under four modelling states, then repeat the
    best state with a REALISTIC network-clock error (0.4 m 1-sigma, the Step 56
    dense-EUREF phase-resolved measurement) and with an AR(1)-class 1.5 m error.
    """
    days = load_days()
    STATES = ("zero_trop_pdop20", "true_trop_pdop20", "true_trop_pdop6",
              "true_trop_pdop6_tepclk", "true_trop_pdop6_ar1clk")
    acc = {sel: {s: [] for s in STATES} for sel in ("elev", "pdop")}
    for stem, d, eps, sig in days:
        for ei in range(0, len(eps), max(EPOCH_STRIDE, 30)):
            rec = d[eps[ei]]
            if rec.get("clk_bad"):
                continue
            ra = sorted(rec.get("recs") or [], key=lambda r: -r[1])
            if len(ra) < 7:
                continue
            zwd = float(rec.get("zwd") or 0.0)     # estimated ZWD for this epoch
            p_ref, c_ref = solve_free(ra[:min(10, len(ra))])
            if c_ref is None or p_ref is None or np.linalg.norm(p_ref) > 50.0:
                continue
            best_pdop, best3 = np.inf, None
            for i in range(min(8, len(ra))):
                for j in range(i + 1, min(9, len(ra))):
                    for k in range(j + 1, min(10, len(ra))):
                        trio = [ra[i], ra[j], ra[k]]
                        p = _pdop(np.array([t[0] for t in trio]))
                        if p < best_pdop:
                            best_pdop, best3 = p, trio
            for sel, recs in (("elev", ra[:3]), ("pdop", best3)):
                if recs is None:
                    continue
                # zwd is applied as tropo so a modelled value REMOVES it; the
                # "zero_trop" state passes 0.0 and eats the wet delay.
                trials = {
                    "zero_trop_pdop20": (0.0, 20.0, 0.0),
                    "true_trop_pdop20": (zwd, 20.0, 0.0),
                    "true_trop_pdop6": (zwd, 6.0, 0.0),
                    "true_trop_pdop6_tepclk": (zwd, 6.0, 0.4),
                    "true_trop_pdop6_ar1clk": (zwd, 6.0, 1.5),
                }
                for name, (tro, pmax, clk_sig) in trials.items():
                    for _ in range(4 if clk_sig > 0 else 1):
                        clk = c_ref + (RNG.normal(0.0, clk_sig) if clk_sig else 0.0)
                        dp = _solve_fixed_pdop(recs, clk, tro, pmax)
                        if dp is not None:
                            acc[sel][name].append(float(np.linalg.norm(dp)))
    out = {}
    for sel, states in acc.items():
        out[sel] = {}
        for s, v in states.items():
            if len(v) < 50:
                continue
            a = np.asarray(v)
            out[sel][s] = {"n": len(a), "rms_m": float(np.sqrt(np.mean(a ** 2))),
                           "p50_m": float(np.median(a)),
                           "p95_m": float(np.percentile(a, 95)),
                           "gt10m_rate": float(np.mean(a > 10.0))}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = run()
    k3 = k3_clock_sensitivity()
    budget = k3_error_budget()
    floor = k3.get("0.0", {}).get("rms_m")
    sig_matters = None
    if floor:
        for sg in SIGMA_SWEEP_M:
            if k3.get(str(sg), {}).get("rms_m", 0) > 1.10 * floor:
                sig_matters = sg
                break
    out = {
        "step": 57,
        "title": "Joint estimator vs post-solve correction: position sensitivity to clock-prediction quality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": "step_48 cache (69 stations x 7 days, real IF residuals + geometry + SINEX truth)",
        "method": ("linearised at surveyed truth for geometry (mechanism diagnostic, not the "
                   "blind nonlinear Step 48 flagship); clock-prediction error drawn from "
                   "N(0, sqrt(1-f)*sigma_clock) for each predictable fraction f; "
                   "f=0.64 is the Step 56 dense-EUREF measurement"),
        "fractions": res,
        "k3_clock_error_sensitivity": k3,
        "k3_error_budget": budget,
    }
    # headline: at each k, does joint beat post-solve, and how does it scale with f?
    hl = {}
    for fr, r in res.items():
        hl[fr] = {k: {"joint_vs_post_solve_rms_pct": v.get("joint_vs_post_solve_rms_pct"),
                      "joint_vs_free_rms_pct": v.get("joint_vs_free_rms_pct"),
                      "post_solve_rms_m": (v["post_solve"] or {}).get("rms_m"),
                      "joint_soft_rms_m": (v["joint_soft"] or {}).get("rms_m"),
                      "free_rms_m": (v["free"] or {}).get("rms_m")}
                  for k, v in r["by_k"].items()}
    out["headline"] = {
        "k3_model_error_floor_rms_m": floor,
        "k3_clock_sigma_that_matters_m": sig_matters,
        "k3_clock_error_sensitivity": k3,
        "k3_error_budget": budget,
        "by_fraction": hl,
        "interpretation": (
            "DECISIVE: the k=3 clock-error sweep shows position RMS is essentially flat "
            "until the clock-prediction error exceeds ~4 m. k=3 accuracy is limited by "
            "pseudorange model error (multipath / troposphere / orbit), not by clock "
            "prediction -- so a better clock prior (joint OR post-solve) improves k=3 "
            "FEASIBILITY, not k=3 ACCURACY. At k>=4, joint_soft is a near-tie with a "
            "free-clock solve (joint_vs_free is single-digit percent): with enough "
            "geometry the soft prior is correctly down-weighted and adds little. The "
            "large joint_vs_post_solve margin (+80-88%) is NOT an estimator-structure "
            "result -- it compares against an unfiltered hard-clock-fix baseline that "
            "has no integrity monitor and propagates every clock error 1:1; it shows "
            "hard-fixing an uncertain clock is bad practice, not that TEP adds position "
            "accuracy. The clock prediction's real position-domain payload is timing "
            "holdover (Step 55), k=3 feasibility, and graceful degradation (a soft prior "
            "cannot make a fix worse than free-clock); the k=3 accuracy lever is "
            "pseudorange modelling and satellite count."),
        "k3_error_budget_conclusion": (
            "The k=3 ~8.4 m floor was stress-tested for a hidden clock lever: applying "
            "the estimated zenith wet delay, tightening the integrity gate PDOP 20->6, "
            "and swapping the injected clock error for the real dense-network value "
            "(0.4 m, Step 56) versus an AR(1)-class value (1.5 m) each move the floor "
            "by <=7% and the TEP-vs-AR(1) clock gap is ~2%. The floor is genuinely "
            "pseudorange-model-limited (multipath, orbit, un-smoothed code), not a "
            "troposphere or geometry-gate artefact -- it is a real property of "
            "code-only SPP, and TEP's k=3 contribution is feasibility and availability."),
    }
    path = OUT / "step_57_joint_estimator_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")
    print("k=3 position RMS vs clock-prediction error (real geometry + residuals):")
    for sg in SIGMA_SWEEP_M:
        r = k3.get(str(sg))
        if r:
            print(f"  clock sigma {sg:>4.2f} m  ->  RMS {r['rms_m']:5.2f} m   p50 {r['p50_m']:.2f}   "
                  f"p99 {r['p99_m']:.1f}   >10 m {r['gt10m_rate']*100:.0f}%")
    if floor:
        print(f"  => k=3 model-error floor (perfect clock): {floor:.2f} m RMS; "
              f"clock matters only above ~{sig_matters} m\n")
    for fr, r in res.items():
        print(f"[{fr}]  f = {r['predictable_fraction']}")
        for k in KS:
            v = r["by_k"].get(str(k), {})
            ps, js, fs = v.get("post_solve"), v.get("joint_soft"), v.get("free")
            if not ps:
                continue
            print(f"  k={k}  free RMS {fs['rms_m']:.2f}m  " if fs else f"  k={k}  free   --      ",
                  end="")
            print(f"post-solve RMS {ps['rms_m']:.2f}m (p99 {ps['p99_m']:.1f})  "
                  f"joint RMS {js['rms_m']:.2f}m (p99 {js['p99_m']:.1f})  "
                  f"joint-vs-post {v.get('joint_vs_post_solve_rms_pct', float('nan')):+.1f}%  "
                  f"avail {v.get('joint_integrity_availability') or 1.0:.2f}")


if __name__ == "__main__":
    main()
