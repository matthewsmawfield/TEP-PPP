#!/usr/bin/env python3
"""
STEP 75c  --  Does the correlation length stratify by oscillator screening?
              Done properly.

Step 75b reported lambda_QQ = 2,617 km > lambda_QA = 1,073 > lambda_AA = 546 and
read it as the screening prediction.  Four things invalidate that as it stands:

  1. It used a Butterworth band-pass and a PEARSON correlation
     (np.corrcoef of band-passed series).  Paper 1 states band-averaged
     coherency FAILS to detect the signal, and Section 3.7b of this paper
     documents that exact statistic as the error it corrected.  The estimator
     must be the phase-alignment index cos(arg(sum_f S_ij(f))).
  2. It fitted A*exp(-d/lam) with NO offset.  Paper 2 and Step 53 fit
     A*exp(-d/lam) + C0.  The offset matters for a specific physical reason
     given below: it is what absorbs the network datum.
  3. One day.
  4. The class sizes are 3,916 / 2,969 / 496 pairs, and lambda from a few
     hundred pairs is unconstrained.  Subsampling QQ to n(AA) gives a lambda
     IQR of [1,422, 29,759] km, and a clock-label permutation produces the
     monotone ordering QQ>QA>AA in 35% of RANDOM label assignments (expected
     1/6 = 17% if the ordering were informative).  The observed ordering was
     therefore consistent with a pure sample-size artefact.

This step redoes it with every one of those fixed, and adds the control that
matters most for the accompanying "the global common mode is the TEP field"
claim.

THE DATUM ARGUMENT.  Analysis centres estimate station clocks against a network
datum, so every station shares a common reference-clock error BY CONSTRUCTION.
A globally correlated common mode is therefore the null expectation, not
evidence.  But a rank-1 uniform mode adds the SAME signal to every station, so
it produces distance-INDEPENDENT phase alignment: it can raise the whole curve
but cannot create a decay.  That is precisely what the offset C0 in
A*exp(-d/lam) + C0 absorbs.  Fitting without C0 (as 75b did) lets the datum
leak into lambda.  As a second, explicit control this step also re-runs the
whole analysis after projecting the per-epoch cross-station mean out of every
station's residual, which removes the rank-1 datum mode exactly.

Controls reported:
  * sample-size-matched lambda: every class subsampled to the SAME number of
    unique pairs, many draws, full distribution -- removes the mechanical bias
  * clock-label permutation null on the ordering AND on the lambda_QQ-lambda_AA
    gap: preserves network geometry and class sizes, destroys only the
    clock-type association
  * datum-projected arm: same analysis with the rank-1 uniform mode removed
  * unique-pair counts reported next to every number

Output: results/outputs/step_75c_screening_lambda_proper.json
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
from scripts.lib import tep_data as td

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_75c_screening_lambda_proper.json"
CACHE = ROOT / "results" / "outputs" / "step_75c_cache.npz"
CLOCKS = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
RNG = np.random.default_rng(20260904)

PRODUCT = "clk_cod"
N_DAYS = 366
N_SUB = 250          # subsample draws per class
N_PERM = 300         # clock-label permutations
MIN_PAIRS_FIT = 120  # never fit a decay on fewer unique pairs than this


def demean_network(ser):
    """Project out the rank-1 uniform mode: subtract the per-epoch cross-station
    mean.  This is the network datum, which is common to every station by
    construction and cannot itself produce distance structure."""
    names = list(ser)
    if len(names) < 8:
        return ser
    t0 = min(np.asarray(ser[n]["t"], float).min() for n in names)
    t1 = max(np.asarray(ser[n]["t"], float).max() for n in names)
    grid = np.arange(t0, t1, 30.0)
    if len(grid) < 64:
        return ser
    M, keep = [], []
    for n in names:
        t = np.asarray(ser[n]["t"], float); x = np.asarray(ser[n]["resid"], float)
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 0.5 * len(grid):
            continue
        M.append(np.interp(grid, t[ok], x[ok])); keep.append(n)
    if len(keep) < 8:
        return ser
    M = np.array(M)
    M = M - np.nanmean(M, axis=0, keepdims=True)     # remove the datum
    return {n: {"t": grid, "resid": M[i], "station": ser[n]["station"]}
            for i, n in enumerate(keep)}


def klass(a, b):
    ca = (CLOCKS.get(a) or ["UNKNOWN"])[0]
    cb = (CLOCKS.get(b) or ["UNKNOWN"])[0]
    if "UNKNOWN" in (ca, cb):
        return ""
    if ca == "QUARTZ" and cb == "QUARTZ":
        return "QQ"
    if ca == "ATOMIC" and cb == "ATOMIC":
        return "AA"
    return "QA"


def collect(dates, datum_removed):
    rows = []
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        if datum_removed:
            ser = demean_network(ser)
        RF, STA = s53._spectra(ser)
        if len(RF) < 10:
            continue
        nm = list(RF)
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            cl = klass(A.name, B.name)
            if not cl:
                continue
            d = td.great_circle_km(A, B)
            if not (s53.DMIN_KM <= d <= s53.DMAX_KM):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((d, float(np.cos(np.angle(S))), cl,
                         A.name if A.name < B.name else B.name,
                         B.name if A.name < B.name else A.name))
        if (k + 1) % 90 == 0:
            print(f"    {k+1}/{len(dates)}  {len(rows):,} pair-days", flush=True)
    return rows


def lam_from(dist, val, pid, pick_pairs=None):
    """3-parameter fit A*exp(-d/lam)+C0 to distance-bin means (Step 53 machinery).
    C0 absorbs the uniform network datum; without it the datum leaks into lam."""
    if pick_pairs is not None:
        keep = np.isin(pid, pick_pairs)
        dist, val = dist[keep], val[keep]
    if len(dist) < 200:
        return None
    ctr, v, cnt = s53._curve(list(zip(dist, val)))
    if len(ctr) < 6:
        return None
    f = s53._fit(ctr, v, cnt)
    if not f:
        return None
    return (f.get("exponential") or {}).get("lambda_km")


def analyse(rows, label):
    dist = np.array([r[0] for r in rows], float)
    val = np.array([r[1] for r in rows], float)
    cls = np.array([r[2] for r in rows])
    pid = np.array([f"{r[3]}-{r[4]}" for r in rows])
    upairs = {c: np.unique(pid[cls == c]) for c in ("QQ", "QA", "AA")}
    n_unique = {c: len(v) for c, v in upairs.items()}
    print(f"\n  [{label}] {len(val):,} pair-days | unique pairs "
          f"QQ {n_unique['QQ']}  QA {n_unique['QA']}  AA {n_unique['AA']}")

    raw = {c: lam_from(dist, val, pid, upairs[c]) for c in ("QQ", "QA", "AA")}
    print(f"    unmatched lambda:  QQ {raw['QQ']}  QA {raw['QA']}  AA {raw['AA']}")

    # ---- sample-size-matched: every class subsampled to the SAME pair count ----
    nmin = min(n_unique.values())
    matched = {}
    if nmin >= MIN_PAIRS_FIT:
        for c in ("QQ", "QA", "AA"):
            draws = []
            for _ in range(N_SUB):
                pk = RNG.choice(upairs[c], nmin, replace=False)
                L = lam_from(dist, val, pid, pk)
                if L:
                    draws.append(L)
            draws = np.array(draws)
            matched[c] = {
                "n_pairs": int(nmin), "n_draws": int(len(draws)),
                "median_km": float(np.median(draws)) if len(draws) else None,
                "iqr_km": [float(np.percentile(draws, 25)), float(np.percentile(draws, 75))] if len(draws) else None,
                "ci95_km": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) else None,
            }
        print(f"    size-matched to n={nmin} pairs:")
        for c in ("QQ", "QA", "AA"):
            m = matched[c]
            print(f"      {c}  median {m['median_km']:8.0f} km   IQR [{m['iqr_km'][0]:.0f}, {m['iqr_km'][1]:.0f}]"
                  f"   CI95 [{m['ci95_km'][0]:.0f}, {m['ci95_km'][1]:.0f}]")
        qq = matched["QQ"]["median_km"]; aa = matched["AA"]["median_km"]
        sep = bool(qq and aa and matched["QQ"]["ci95_km"][0] > matched["AA"]["ci95_km"][1])
    else:
        sep = False
        print(f"    size-matched fit skipped: smallest class has {nmin} pairs (< {MIN_PAIRS_FIT})")

    # ---- clock-label permutation null on the ordering and on the QQ-AA gap ----
    stations = sorted({s for p in pid for s in p.split("-")})
    known = [s for s in stations if (CLOCKS.get(s) or ["U"])[0] in ("QUARTZ", "ATOMIC")]
    nq = sum(1 for s in known if (CLOCKS.get(s) or ["U"])[0] == "QUARTZ")
    sa = np.array([p.split("-")[0] for p in pid]); sb = np.array([p.split("-")[1] for p in pid])
    mono = 0; tot = 0; gaps = []
    for _ in range(N_PERM):
        perm = list(known); RNG.shuffle(perm)
        lb = {s: ("QUARTZ" if i < nq else "ATOMIC") for i, s in enumerate(perm)}
        ca = np.array([lb.get(x, "U") for x in sa]); cb = np.array([lb.get(x, "U") for x in sb])
        mQ = (ca == "QUARTZ") & (cb == "QUARTZ")
        mA = (ca == "ATOMIC") & (cb == "ATOMIC")
        mX = ~mQ & ~mA & (ca != "U") & (cb != "U")
        lq = lam_from(dist, val, pid, np.unique(pid[mQ]))
        lx = lam_from(dist, val, pid, np.unique(pid[mX]))
        la = lam_from(dist, val, pid, np.unique(pid[mA]))
        if None in (lq, lx, la):
            continue
        tot += 1
        if lq > lx > la:
            mono += 1
        gaps.append(lq - la)
    gaps = np.array(gaps)
    obs_gap = (raw["QQ"] - raw["AA"]) if (raw["QQ"] and raw["AA"]) else None
    p_mono = mono / tot if tot else None
    p_gap = float(np.mean(gaps >= obs_gap)) if (obs_gap is not None and len(gaps)) else None
    print(f"    clock-label permutation: monotone in {mono}/{tot} = "
          f"{100*(p_mono or 0):.0f}% of RANDOM assignments (17% = uninformative)")
    if obs_gap is not None:
        print(f"    observed lambda_QQ-lambda_AA = {obs_gap:+.0f} km, null p = {p_gap:.3f}")

    return {
        "n_pair_days": int(len(val)), "n_unique_pairs": n_unique,
        "lambda_unmatched_km": raw,
        "lambda_size_matched": matched,
        "size_matched_QQ_above_AA": sep,
        "perm_monotone_fraction": p_mono,
        "perm_n": tot,
        "observed_lambda_gap_km": obs_gap,
        "perm_p_gap": p_gap,
        "ordering_is_informative": bool(p_mono is not None and p_mono < 0.05),
        "gap_significant": bool(p_gap is not None and p_gap < 0.05),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
    print(f"STEP 75c  |  {PRODUCT}, {len(dates)} days, phase-cos + 3-param fit")
    res = {}
    for datum_removed, label in ((False, "as_published"), (True, "datum_projected_out")):
        print(f"\n=== collecting ({label}) ===")
        rows = collect(dates, datum_removed)
        if len(rows) < 5000:
            print("  insufficient"); continue
        res[label] = analyse(rows, label)

    a = res.get("as_published", {})
    verdict = {
        "estimator": "phase-alignment index cos(arg(sum_f S_ij(f))), Step 53 (NOT band-passed Pearson)",
        "model": "A*exp(-d/lambda) + C0 (the offset absorbs the rank-1 network datum)",
        "n_days": len(dates),
        "step_75b_claim": "lambda_QQ 2617 > lambda_QA 1073 > lambda_AA 546 = screening",
        "ordering_informative": a.get("ordering_is_informative"),
        "gap_significant": a.get("gap_significant"),
        "size_matched_separation": a.get("size_matched_QQ_above_AA"),
        "conclusion": (
            "screening-stratified lambda CONFIRMED" if (a.get("ordering_is_informative")
                                                        and a.get("size_matched_QQ_above_AA"))
            else "NOT confirmed: the ordering is consistent with a class-size artefact "
                 "once the estimator, the nugget, the sample sizes and the label "
                 "permutation are all handled correctly"),
    }
    OUT.write_text(json.dumps({
        "step": "75c",
        "title": "Screening-stratified correlation length, with sample-size and datum controls",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": res, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
