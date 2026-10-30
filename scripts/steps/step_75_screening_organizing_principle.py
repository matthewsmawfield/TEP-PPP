#!/usr/bin/env python3
"""
STEP 75  --  Oscillator screening as the organizing principle: does it order
             EVERY signature at once?

Step 65 established that quartz oscillators carry ~1.9x more network-predictable
clock power than atomic standards, and that the screening is frequency-dependent
(3.8x at low frequency, 0.9x at high frequency where true oscillator noise
lives).  The atomic transition is partially isolated from the proper-time field.

That has a consequence the corpus has never exploited.  Papers 1-2 measured every
one of their signatures -- distance structure, anisotropy, orbital coupling,
diurnal, planetary, Chandler -- on MIXED clock populations.  If screening is
real and continuous, each of those measurements is DILUTED by the screened
atomic stations in the sample, and each should sharpen when the population is
stratified by oscillator type.

This step tests that as a single unified prediction.  Station pairs are split
into three classes by the pair's oscillator composition:

    QQ  both quartz      (least screened)
    QA  mixed
    AA  both atomic      (most screened)

Screening is continuous, so the prediction is not merely "QQ > AA" but a
MONOTONE ORDERING, QQ > QA > AA, holding independently for every signature.
A mixed pair contains one unscreened clock, so it should sit between.

Four signatures are measured, each on the Step 53 phase-alignment index
cos(arg(sum_f S_ij(f))) -- the corrected Papers 1-2 estimator, which is the key
to all of this: Paper 1 states that band-averaged coherency FAILS to detect the
signal, and Paper 2's secondary signatures were nonetheless derived from it.

    A  distance-structured amplitude, compared WITHIN distance bins so the
       comparison cannot be confounded by the three classes having different
       spatial distributions
    B  correlation length lambda from the decay fit
    C  EW-NS anisotropy in the 500-800 km band (Step 74's replicated signal)
    D  annual modulation amplitude of the daily mean phase alignment

Why this is a strong test: an instrumental or processing systematic can produce
one spurious ordering, but it has no reason to produce the SAME monotone
ordering across four statistically independent signatures, on three products
with different processing chains.  Under the null of no screening, each
signature independently has a 1/6 chance of landing in the predicted strict
order by chance.

All inference is clustered on UNIQUE PAIRS, not pair-days: the same pairs recur
every day, so day-clustering overstates significance (see Step 74b).

Output: results/outputs/step_75_screening_organizing_principle.json
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

OUT = ROOT / "results" / "outputs" / "step_75_screening_organizing_principle.json"
CLOCKS = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
RNG = np.random.default_rng(20260903)

PRODUCTS = ["clk_cod", "broadcast", "precise"]
N_DAYS = 366
BINS = np.array([50, 300, 800, 1500, 3000, 6000, 13000], float)
ANISO_BAND = (500.0, 800.0)
HALF = 22.5
N_BOOT = 1500


def klass(a, b):
    ca = (CLOCKS.get(a) or ["UNKNOWN"])[0]
    cb = (CLOCKS.get(b) or ["UNKNOWN"])[0]
    if "UNKNOWN" in (ca, cb):
        return None
    if ca == "QUARTZ" and cb == "QUARTZ":
        return "QQ"
    if ca == "ATOMIC" and cb == "ATOMIC":
        return "AA"
    return "QA"


def collect(product, dates):
    rows = []
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(product, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        if len(RF) < 6:
            continue
        nm = list(RF)
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            cl = klass(A.name, B.name)
            if cl is None:
                continue
            d = td.great_circle_km(A, B)
            if not (BINS[0] <= d <= BINS[-1]):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((k, d, td.baseline_azimuth_midpoint_deg(A, B),
                         float(np.cos(np.angle(S))), cl, f"{A.name}-{B.name}"))
        if (k + 1) % 80 == 0:
            print(f"    {k+1}/{len(dates)} days, {len(rows):,} pair-days", flush=True)
    return rows


def pair_boot(vals, pids, stat, n=N_BOOT):
    """Bootstrap clustered on unique pairs."""
    by = defaultdict(list)
    for i, p in enumerate(pids):
        by[p].append(i)
    up = np.array(list(by))
    out = []
    for _ in range(n):
        pk = RNG.choice(up, len(up), replace=True)
        ii = np.concatenate([by[p] for p in pk])
        v = stat(ii)
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.array(out)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in PRODUCTS:
        print(f"\n=== {product} ===")
        dates = td.available_dates(product, 2024)[:N_DAYS]
        rows = collect(product, dates)
        if len(rows) < 3000:
            print("  insufficient"); continue
        day = np.array([r[0] for r in rows]); dist = np.array([r[1] for r in rows], float)
        az = np.array([r[2] for r in rows], float); val = np.array([r[3] for r in rows], float)
        cls = np.array([r[4] for r in rows]); pid = np.array([r[5] for r in rows])
        npairs = {c: len(np.unique(pid[cls == c])) for c in ("QQ", "QA", "AA")}
        print(f"  {len(val):,} pair-days | unique pairs QQ {npairs['QQ']} "
              f"QA {npairs['QA']} AA {npairs['AA']}")

        sig = {}

        # ---- A: distance-matched amplitude (equal weight per bin) ----
        def amp_of(cl, idx=None):
            m = (cls == cl) if idx is None else (cls[idx] == cl)
            d_ = dist if idx is None else dist[idx]
            v_ = val if idx is None else val[idx]
            vals_ = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                sel = m & (d_ >= lo) & (d_ < hi)
                if sel.sum() >= 20:
                    vals_.append(v_[sel].mean())
            return float(np.mean(vals_)) if vals_ else None
        A = {c: amp_of(c) for c in ("QQ", "QA", "AA")}
        sig["A_distance_matched_amplitude"] = A

        # ---- B: correlation length ----
        B = {}
        for c in ("QQ", "QA", "AA"):
            m = cls == c
            if m.sum() < 200:
                B[c] = None; continue
            ctr, v_, cnt = s53._curve(list(zip(dist[m], val[m])))
            f = s53._fit(ctr, v_, cnt) if len(ctr) >= 6 else {}
            B[c] = (f.get("exponential") or {}).get("lambda_km") if f else None
        sig["B_lambda_km_exponential"] = B

        # ---- C: EW-NS anisotropy in 500-800 km ----
        C = {}
        lo, hi = ANISO_BAND
        band = (dist >= lo) & (dist < hi)
        ew = np.abs(az - 90.0) <= HALF
        ns = (az <= HALF) | (az >= 180.0 - HALF)
        for c in ("QQ", "QA", "AA"):
            e = band & (cls == c) & ew; n = band & (cls == c) & ns
            C[c] = float(val[e].mean() - val[n].mean()) if (e.sum() >= 20 and n.sum() >= 20) else None
        sig["C_anisotropy_500_800km"] = C

        # ---- D: annual modulation amplitude of the daily mean ----
        D = {}
        for c in ("QQ", "QA", "AA"):
            m = cls == c
            if m.sum() < 500:
                D[c] = None; continue
            dd = day[m]; vv = val[m]
            ud = np.unique(dd)
            series = np.array([vv[dd == u].mean() for u in ud])
            if len(ud) < 120:
                D[c] = None; continue
            t = 2 * np.pi * ud / 365.25
            X = np.column_stack([np.ones(len(t)), np.cos(t), np.sin(t)])
            coef, *_ = np.linalg.lstsq(X, series, rcond=None)
            D[c] = float(np.hypot(coef[1], coef[2]))
        sig["D_annual_modulation_amplitude"] = D

        ordered = {}
        for key, dd in sig.items():
            vals3 = [dd.get("QQ"), dd.get("QA"), dd.get("AA")]
            ordered[key] = bool(all(v is not None for v in vals3)
                                and vals3[0] > vals3[1] > vals3[2])

        # significance of the QQ-AA gap on signature A, pair-clustered
        selQ = np.where(cls == "QQ")[0]; selA = np.where(cls == "AA")[0]
        gap = None
        if len(selQ) > 100 and len(selA) > 100:
            def statQ(ii): return amp_of("QQ", ii)
            def statA(ii): return amp_of("AA", ii)
            bq = pair_boot(val[selQ], pid[selQ], lambda ii: (
                np.mean([val[selQ][ii][(dist[selQ][ii] >= lo2) & (dist[selQ][ii] < hi2)].mean()
                         for lo2, hi2 in zip(BINS[:-1], BINS[1:])
                         if ((dist[selQ][ii] >= lo2) & (dist[selQ][ii] < hi2)).sum() >= 20])))
            ba = pair_boot(val[selA], pid[selA], lambda ii: (
                np.mean([val[selA][ii][(dist[selA][ii] >= lo2) & (dist[selA][ii] < hi2)].mean()
                         for lo2, hi2 in zip(BINS[:-1], BINS[1:])
                         if ((dist[selA][ii] >= lo2) & (dist[selA][ii] < hi2)).sum() >= 20])))
            if len(bq) > 50 and len(ba) > 50:
                n = min(len(bq), len(ba))
                diff = bq[:n] - ba[:n]
                gap = {"qq_minus_aa": float(np.mean(diff)),
                       "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                       "p_gt_0": float(np.mean(diff > 0))}

        arms[product] = {"n_pair_days": int(len(val)), "unique_pairs": npairs,
                         "signatures": sig, "monotone_QQ_gt_QA_gt_AA": ordered,
                         "amplitude_gap_pair_clustered": gap}
        print("\n  signature                        QQ         QA         AA    monotone")
        for key, dd in sig.items():
            f = lambda x: f"{x:9.4f}" if isinstance(x, float) else "        -"
            print(f"  {key:<30} {f(dd.get('QQ'))} {f(dd.get('QA'))} {f(dd.get('AA'))}    {ordered[key]}")
        if gap:
            print(f"\n  amplitude gap QQ-AA = {gap['qq_minus_aa']:+.4f}  "
                  f"CI95 [{gap['ci95'][0]:+.4f}, {gap['ci95'][1]:+.4f}]  P(>0)={gap['p_gt_0']:.3f}")

    tot = sum(sum(1 for v in a["monotone_QQ_gt_QA_gt_AA"].values() if v) for a in arms.values())
    tests = sum(len(a["monotone_QQ_gt_QA_gt_AA"]) for a in arms.values())
    from math import comb
    p_binom = sum(comb(tests, k) * (1/6)**k * (5/6)**(tests-k) for k in range(tot, tests+1)) if tests else None
    verdict = {
        "prediction": "screening is continuous, so QQ > QA > AA for every signature",
        "n_tests": tests, "n_monotone": tot,
        "p_binomial_1_in_6_per_test": p_binom,
        "note": "each signature has a 1/6 chance of the predicted strict order under the null",
    }
    OUT.write_text(json.dumps({
        "step": 75,
        "title": "Oscillator screening as the organizing principle across TEP signatures",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nMONOTONE ORDERINGS: {tot}/{tests}   binomial p = {p_binom}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
