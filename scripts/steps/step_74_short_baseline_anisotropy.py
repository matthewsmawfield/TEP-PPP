#!/usr/bin/env python3
"""
STEP 74  --  Paper 3's short-baseline EW/NS anisotropy, on the networks that
             actually have short baselines.

Paper 3 reports the anisotropy as an AMPLITUDE contrast at SHORT range:
"East-West correlations are 2-5% (MSC) to 22% (Phase Alignment) stronger than
North-South at short distances (< 500 km)".  Paper 2 reports a ratio of fitted
correlation lengths (2.16) over 1-20,000 km.

Step 73 tested this on clk_cod and found nothing.  That was a coverage error:

    product      stations   pairs   pairs < 500 km
    clk_cod         128      6079         34
    precise          30       432        178
    broadcast        78      2288        175

clk_cod is a globally distributed network; below 500 km it holds 34 of 6,079
pairs, so the regime Paper 3 describes is essentially unsampled.  `precise` is
the dense European set -- 41% of its pairs are under 500 km -- and `broadcast`
also has real short-range coverage.  Those are the correct networks for this
test, and this step uses them.

Two further corrections carried over:
  * Azimuth is taken at the great-circle MIDPOINT, not at station A.  (Below
    2,000 km the two agree to a few degrees, so this matters little here, but it
    is the correct convention and Step 73 showed it reclassifies 36% of pairs on
    a global network.)
  * The statistic is the binned amplitude contrast, not a ratio of fitted
    lambdas.  A regional network spans ~2,000 km and cannot constrain a ~4,000 km
    decay length at all; Step 73's lambda ratios swung over [0.17, 2.05] between
    30-day windows, which is fit instability, not physics.  Comparing the
    phase-alignment index between sectors WITHIN a distance bin is
    distance-matched by construction and well conditioned.

Estimator is imported from Step 53 so it cannot drift from Section 3.7b.

Output: results/outputs/step_74_short_baseline_anisotropy.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
import itertools
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

OUT = ROOT / "results" / "outputs" / "step_74_short_baseline_anisotropy.json"
RNG = np.random.default_rng(20260903)

PRODUCTS = ["precise", "broadcast"]
N_DAYS = 366
BINS = np.array([50, 150, 300, 500, 800, 1300, 2000], float)
HALF = 22.5          # sector half-width about the cardinal direction
N_NULL = 400
N_BOOT = 2000


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
            d = td.great_circle_km(A, B)
            if not (BINS[0] <= d <= BINS[-1]):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((k, d, td.baseline_azimuth_midpoint_deg(A, B),
                         float(np.cos(np.angle(S)))))
        if (k + 1) % 60 == 0:
            print(f"    {k+1}/{len(dates)} days, {len(rows):,} pair-days", flush=True)
    return rows


def sectors(az):
    ew = np.abs(az - 90.0) <= HALF
    ns = (az <= HALF) | (az >= 180.0 - HALF)
    return ew, ns


def contrast(dist, az, val, bins):
    """Equal-weight mean over bins of the within-bin EW-NS difference."""
    ew, ns = sectors(az)
    diffs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dist >= lo) & (dist < hi)
        e, n = m & ew, m & ns
        if e.sum() < 20 or n.sum() < 20:
            continue
        diffs.append(val[e].mean() - val[n].mean())
    return float(np.mean(diffs)) if diffs else None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in PRODUCTS:
        print(f"\n=== {product} ===")
        dates = td.available_dates(product, 2024)[:N_DAYS]
        rows = collect(product, dates)
        if len(rows) < 2000:
            print(f"  insufficient ({len(rows)})"); continue
        day = np.array([r[0] for r in rows]); dist = np.array([r[1] for r in rows], float)
        az = np.array([r[2] for r in rows], float); val = np.array([r[3] for r in rows], float)
        ew, ns = sectors(az)
        print(f"  {len(val):,} pair-days | EW {ew.sum():,}  NS {ns.sum():,} | {len(np.unique(day))} days")

        table = {}
        print("\n  bin (km)      n_EW    n_NS   phase_EW  phase_NS     EW-NS   rel%")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (dist >= lo) & (dist < hi)
            e, n = m & ew, m & ns
            if e.sum() < 20 or n.sum() < 20:
                continue
            me, mn = val[e].mean(), val[n].mean()
            rel = 100.0 * (me - mn) / abs(mn) if abs(mn) > 1e-9 else float("nan")
            table[f"{lo:.0f}-{hi:.0f}"] = {
                "n_ew": int(e.sum()), "n_ns": int(n.sum()),
                "phase_ew": float(me), "phase_ns": float(mn),
                "diff": float(me - mn), "relative_pct": float(rel),
                "mean_dist_ew": float(dist[e].mean()), "mean_dist_ns": float(dist[n].mean())}
            print("  %-11s %7d %7d %10.4f %9.4f %+9.4f %+6.1f" % (
                f"{lo:.0f}-{hi:.0f}", e.sum(), n.sum(), me, mn, me - mn, rel))

        obs = contrast(dist, az, val, BINS)
        udays = np.unique(day)
        boot = []
        for _ in range(N_BOOT):
            pick = RNG.choice(udays, len(udays), replace=True)
            idx = np.concatenate([np.where(day == d)[0] for d in pick])
            c = contrast(dist[idx], az[idx], val[idx], BINS)
            if c is not None:
                boot.append(c)
        boot = np.array(boot)
        nulls = []
        for _ in range(N_NULL):
            ap = az.copy()
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = np.where((dist >= lo) & (dist < hi))[0]
                if len(m) > 1:
                    ap[m] = RNG.permutation(az[m])
            c = contrast(dist, ap, val, BINS)
            if c is not None:
                nulls.append(c)
        nulls = np.array(nulls)
        z = float((obs - nulls.mean()) / (nulls.std() + 1e-30)) if len(nulls) else None
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) > 20 else None
        sub = BINS[BINS <= 500]
        obs500 = contrast(dist, az, val, sub) if len(sub) >= 2 else None

        # per-bin day-clustered CI and azimuth-shuffle z: the pooled contrast
        # averages over bins where the effect is absent, diluting a localised one.
        perbin = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (dist >= lo) & (dist < hi)
            e, n = m & ew, m & ns
            if e.sum() < 20 or n.sum() < 20:
                continue
            key = f"{lo:.0f}-{hi:.0f}"
            bb = []
            for _ in range(N_BOOT):
                pk = RNG.choice(udays, len(udays), replace=True)
                ix = np.concatenate([np.where(day == d)[0] for d in pk])
                mm = (dist[ix] >= lo) & (dist[ix] < hi)
                ee, nn = mm & ew[ix], mm & ns[ix]
                if ee.sum() >= 20 and nn.sum() >= 20:
                    bb.append(val[ix][ee].mean() - val[ix][nn].mean())
            bb = np.array(bb)
            nl = []
            for _ in range(N_NULL):
                idxm = np.where(m)[0]
                sp = RNG.permutation(az[idxm])
                ee = np.abs(sp - 90.0) <= HALF
                nn = (sp <= HALF) | (sp >= 180.0 - HALF)
                if ee.sum() >= 20 and nn.sum() >= 20:
                    nl.append(val[idxm][ee].mean() - val[idxm][nn].mean())
            nl = np.array(nl)
            obs_b = float(val[e].mean() - val[n].mean())
            perbin[key] = {
                "diff": obs_b,
                "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))] if len(bb) > 20 else None,
                "p_gt_0": float(np.mean(bb > 0)) if len(bb) else None,
                "shuffle_z": float((obs_b - nl.mean()) / (nl.std() + 1e-30)) if len(nl) else None,
            }
        print("\n  per-bin day-clustered significance:")
        for k2, v2 in perbin.items():
            print("    %-11s %+8.4f  CI95 [%+.4f, %+.4f]  P(>0)=%.3f  z=%+.2f" % (
                k2, v2["diff"], v2["ci95"][0], v2["ci95"][1], v2["p_gt_0"], v2["shuffle_z"]))

        arms[product] = {
            "per_bin_significance": perbin,
            "n_pair_days": int(len(val)), "n_days": int(len(udays)),
            "n_ew": int(ew.sum()), "n_ns": int(ns.sum()),
            "binned": table,
            "contrast_ew_minus_ns": obs,
            "contrast_under_500km": obs500,
            "ci95_day_clustered": ci,
            "azimuth_shuffle_z": z,
            "bootstrap_p_gt_0": float(np.mean(boot > 0)) if len(boot) else None,
            "ew_exceeds_ns": bool(obs is not None and obs > 0),
            "significant": bool(ci and ci[0] > 0),
        }
        print(f"\n  EW-NS contrast = {obs:+.4f}   CI95 {ci}")
        print(f"  under 500 km   = {obs500}")
        print(f"  azimuth-shuffle z = {z:+.2f}   P(>0) = {arms[product]['bootstrap_p_gt_0']:.3f}")

    agree = [a for a in arms.values() if a["ew_exceeds_ns"]]
    verdict = {
        "papers_2_3_prediction": "EW > NS, strongest below 500 km "
                                 "(Paper 3: +22% in phase alignment)",
        "n_arms": len(arms), "n_arms_ew_gt_ns": len(agree),
        "arms_significant": [k for k, v in arms.items() if v["significant"]],
        "reproduces_paper3": bool(len(arms) > 0 and len(agree) == len(arms)
                                  and any(v["significant"] for v in arms.values())),
    }
    OUT.write_text(json.dumps({
        "step": 74,
        "title": "Short-baseline EW/NS anisotropy in the phase-alignment index",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
