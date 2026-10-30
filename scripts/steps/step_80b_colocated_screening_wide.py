#!/usr/bin/env python3
"""
STEP 80  --  The co-located screening differential across SIX independent sites.

Step 77 established the design on the one co-located quartz/atomic pair the
COD multi-GNSS product contains (THTI/THTG, Tahiti): the differential of two
clocks at one site cancels the proper-time field, the troposphere, the satellite
errors and the network datum, so any residual coupling to DISTANT stations can
only be a difference in how the two oscillators couple to the field.  It gave
lambda = 4,053 km at R2 = 0.852, against a null distribution with median
R2 = 0.051 -- but from a single site.

Step 78 acquired COD0OPSFIN (317 stations against the MGX product's 128), which
contains SIX co-located quartz/atomic sites:

    MATE/MAT1 (Matera)   STR1/STR2 (Stromlo)   THTG/THTI, THTG/FAA1 (Tahiti)
    WUHN/JFNG, WUHN/WUH2 (Wuhan)

Six sites on four continents turn a single-site existence result into a
replication test, and let the screening factor be compared across independent
oscillator pairs rather than asserted from one.

Efficiency: days are the outer loop and every co-located pair is processed from
the same loaded series, so the product is read once rather than once per pair.

Nulls are DISTRIBUTIONS, not single fits (the error that made Step 77 look
marginal): each realisation phase-randomises the differential over its own block
of days and is fitted independently.

Output: results/outputs/step_80b_colocated_screening_wide.json
"""
from __future__ import annotations

import os as _os
import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_s = importlib.util.spec_from_file_location(
    "s77", ROOT / "scripts" / "steps" / "step_77_colocated_screening_differential.py")
s77 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s77)
s53 = s77.s53

OUT = ROOT / "results" / "outputs" / ("step_80b_colocated_screening_wide%s.json" % ("" if _os.environ.get("TEP_DMAX_KM","13000")=="13000" else "_dmax20000"))
COLOC = json.loads((ROOT / "data" / "meta" / "colocated_pairs.json").read_text())
# Earth's maximum great-circle separation is ~20,000 km.  Step 53 caps the fit at
# 13,000 km, which discards 21.9% of all station pairs -- and 44% of Tahiti's,
# because a mid-Pacific site has most of the world beyond 13,000 km.  Papers 1-2
# bin to 20,000 km.  Set TEP_DMAX_KM=20000 to use the full range.
DMAX_KM = float(_os.environ.get("TEP_DMAX_KM", "13000"))
PRODUCT = "clk_cod_ops"
N_DAYS = 361
N_NULL_REAL = 25
DAYS_PER_REAL = 40


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s53.DMAX_KM = DMAX_KM
    s53.EDGES = np.geomspace(s53.DMIN_KM, DMAX_KM, s53.N_BINS + 1)
    print(f"distance range {s53.DMIN_KM:.0f}-{DMAX_KM:.0f} km")
    dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
    probe = td.load_series(PRODUCT, dates[99], min_valid_frac=0.0, max_jumps=10_000)
    pairs = sorted({(c["atomic"], c["quartz"]) for c in COLOC
                    if c["atomic"] in probe and c["quartz"] in probe})
    print(f"STEP 80  |  {PRODUCT}, {len(dates)} days, {len(probe)} stations, "
          f"{len(pairs)} co-located sites")
    for a, q in pairs:
        print(f"    {q}(quartz) - {a}(atomic)")

    acc = {p: {"delta": [], "atomic": [], "quartz": []} for p in pairs}
    ndays = {p: 0 for p in pairs}
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        for p in pairs:
            r = s77.day_rows(ser, p[0], p[1])
            if r:
                ndays[p] += 1
                for t, _n, d, v in r:
                    acc[p][t].append((d, v))
        if (k + 1) % 60 == 0:
            print(f"    {k+1}/{len(dates)} days", flush=True)

    # null distributions, again reading each day once
    nulls = {p: {"lam": [], "r2": []} for p in pairs}
    for i in range(N_NULL_REAL):
        rng = np.random.default_rng(4000 + i)
        blk = {p: [] for p in pairs}
        for tag in dates[i::9][:DAYS_PER_REAL]:
            try:
                ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
            except Exception:
                continue
            for p in pairs:
                r = s77.day_rows(ser, p[0], p[1], phase_rng=rng)
                blk[p].extend([(d, v) for t, _n, d, v in r if t == "delta"])
        for p in pairs:
            f = s77.fit_curve(blk[p])
            if f and f["lambda_km"]:
                nulls[p]["lam"].append(f["lambda_km"]); nulls[p]["r2"].append(f["R2_exponential"])
        print(f"    null realisation {i+1}/{N_NULL_REAL}", flush=True)

    arms = {}
    print("\n  site                lambda_delta   R2      null R2 med/max    p     n_days")
    for p in pairs:
        if ndays[p] < 30:
            continue
        fits = {t: s77.fit_curve(acc[p][t]) for t in ("delta", "atomic", "quartz")}
        obs = fits["delta"]
        nr = np.array(nulls[p]["r2"]); nl = np.array(nulls[p]["lam"])
        pv = float(np.mean(nr >= obs["R2_exponential"])) if (obs and len(nr)) else None
        key = f"{p[1]}-{p[0]}"
        arms[key] = {
            "quartz": p[1], "atomic": p[0], "n_days": ndays[p], "fits": fits,
            "null": {"n": int(len(nr)),
                     "R2_median": float(np.median(nr)) if len(nr) else None,
                     "R2_max": float(nr.max()) if len(nr) else None,
                     "lambda_median_km": float(np.median(nl)) if len(nl) else None},
            "p_R2_ge_observed": pv,
            "lambda_in_TEP_band": bool(obs and obs["lambda_km"]
                                       and 1000 <= obs["lambda_km"] <= 10000),
            "significant": bool(pv is not None and pv < 0.05),
        }
        if obs:
            print("  %-18s %9.0f km  %.3f    %.3f / %.3f    %.3f   %d" % (
                key, obs["lambda_km"] or -1, obs["R2_exponential"] or 0,
                np.median(nr) if len(nr) else -1, nr.max() if len(nr) else -1,
                pv if pv is not None else -1, ndays[p]))

    sig = [k for k, v in arms.items() if v["significant"] and v["lambda_in_TEP_band"]]
    lams = [v["fits"]["delta"]["lambda_km"] for v in arms.values()
            if v["fits"]["delta"] and v["fits"]["delta"]["lambda_km"]]
    verdict = {
        "n_sites": len(arms),
        "n_sites_significant_and_in_TEP_band": len(sig),
        "sites": sig,
        "lambda_delta_median_km": float(np.median(lams)) if lams else None,
        "lambda_delta_range_km": [float(min(lams)), float(max(lams))] if lams else None,
        "corpus_lambda_km": "4201 +/- 1967 (Paper 2)",
        "replicates_across_sites": bool(len(sig) >= 3),
    }
    OUT.write_text(json.dumps({
        "step": 80,
        "title": "Co-located quartz/atomic screening differential across six sites",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
