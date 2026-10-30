#!/usr/bin/env python3
"""
STEP 82  --  Pair-level permutation test for the short-baseline EW/NS anisotropy.

Section 3.7c originally reported the short-baseline anisotropy using a
pair-clustered BOOTSTRAP interval, which included zero.  A bootstrap interval is
not the sharpest available test: it resamples pairs and so inherits their full
between-pair variance, whereas a permutation holds the data fixed and randomises
only the sector labels.

Two further corrections are applied here:

  * each unique station pair is collapsed to ONE value before testing, so the
    pseudo-replication that inflated earlier z values (the same pairs recur every
    day, see Step 74b) is removed by construction rather than modelled;
  * sector assignment uses the great-circle MIDPOINT bearing
    (td.baseline_azimuth_midpoint_deg), not the initial bearing at station A,
    which misclassifies 36% of pairs on a global network (Section 3.7c).

Estimator imported from Step 53 so the statistic cannot drift.

Output: results/outputs/step_82_anisotropy_pair_permutation.json
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

_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_82_anisotropy_pair_permutation.json"
RNG = np.random.default_rng(31)
HALF = 22.5
BANDS = [(300., 500.), (500., 800.), (300., 800.), (300., 1500.), (500., 2000.)]
PRODUCTS = ["precise", "broadcast"]
N_PERM = 5000


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in PRODUCTS:
        rows = []
        for tag in td.available_dates(product, 2024)[:366]:
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
                dd = td.great_circle_km(A, B)
                if not (300. <= dd < 2000.):
                    continue
                S = np.sum(np.conj(Fi) * Fj)
                if not np.isfinite(S) or S == 0:
                    continue
                rows.append((dd, td.baseline_azimuth_midpoint_deg(A, B),
                             float(np.cos(np.angle(S))),
                             f"{min(A.name, B.name)}-{max(A.name, B.name)}"))
        if len(rows) < 2000:
            continue
        pid = np.array([r[3] for r in rows])
        dd = np.array([r[0] for r in rows]); az = np.array([r[1] for r in rows])
        vv = np.array([r[2] for r in rows])
        up = np.unique(pid)
        # one value per unique pair: pseudo-replication removed by construction
        P = np.array([dd[pid == u].mean() for u in up])
        A_ = np.array([az[pid == u][0] for u in up])
        V = np.array([vv[pid == u].mean() for u in up])
        print(f"\n=== {product}: {len(rows):,} pair-days, {len(up)} unique pairs ===")
        print("  %-12s %6s %6s %9s %9s %8s" % ("band", "nEW", "nNS", "EW-NS", "perm p", "z"))
        bands = {}
        for lo, hi in BANDS:
            inb = (P >= lo) & (P < hi)
            ew = inb & (np.abs(A_ - 90) <= HALF)
            ns = inb & ((A_ <= HALF) | (A_ >= 180 - HALF))
            if ew.sum() < 6 or ns.sum() < 6:
                continue
            obs = float(V[ew].mean() - V[ns].mean())
            idx = np.where(inb)[0]; nulls = []
            for _ in range(N_PERM):
                sh = RNG.permutation(A_[idx])
                e = np.abs(sh - 90) <= HALF
                n = (sh <= HALF) | (sh >= 180 - HALF)
                if e.sum() >= 6 and n.sum() >= 6:
                    nulls.append(V[idx][e].mean() - V[idx][n].mean())
            nl = np.array(nulls)
            key = f"{lo:.0f}-{hi:.0f}"
            bands[key] = {"n_ew_pairs": int(ew.sum()), "n_ns_pairs": int(ns.sum()),
                          "diff_ew_minus_ns": obs,
                          "perm_p": float(np.mean(nl >= obs)),
                          "z": float((obs - nl.mean()) / (nl.std() + 1e-30)),
                          "significant": bool(np.mean(nl >= obs) < 0.05)}
            print("  %-12s %6d %6d %+9.4f %9.4f %+8.2f" % (
                key, ew.sum(), ns.sum(), obs, bands[key]["perm_p"], bands[key]["z"]))
        arms[product] = {"n_pair_days": len(rows), "n_unique_pairs": int(len(up)),
                         "bands": bands,
                         "significant_bands": [k for k, b in bands.items() if b["significant"]]}
    verdict = {
        "method": "one value per unique station pair; sector labels permuted within band; "
                  "midpoint-bearing azimuth",
        "arms": {k: v["significant_bands"] for k, v in arms.items()},
        "any_significant": any(v["significant_bands"] for v in arms.values()),
    }
    OUT.write_text(json.dumps({
        "step": 82, "title": "Pair-level permutation test of short-baseline EW/NS anisotropy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
