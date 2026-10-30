#!/usr/bin/env python3
"""
STEP 14  (Phase 1 closure) -- T1/T3/T4 on a GLOBAL PRECISE-CLOCK network.

This is the pre-registered step that lets T4 be decided.  It uses the IGS/MGEX
analysis-centre receiver-clock products in `data/rinex/clk/` (RINEX CLK, "AR"
records): satellite-clock error is already removed by the analysis centre, and
the network is genuinely global (~118 stations, baselines to 20,000 km).  On
this product the shared-constellation footprint no longer explains a
distance-structured correlation, so the T4 degeneracy can be broken.

  T1  correlation length + null controls        (reuses step_10)
  T3  resolvability vs aperture                 (reuses step_12)
  T4  distance vs satellite co-visibility       (reuses step_13)
  MA  multi-analysis-centre consistency of lambda (COD / GFZ / GRG / JPL)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td                       # noqa: E402
from scripts.lib import tep_stats as ts                      # noqa: E402
from scripts.steps.step_10_global_correlation import run_arm as t1_run          # noqa: E402
from scripts.steps.step_12_resolvability import run as t3_run                   # noqa: E402
from scripts.steps.step_13_commonview import run as t4_run                      # noqa: E402
import scripts.steps.step_10_global_correlation as s10                          # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
MAX_AC_DAYS = 90          # evenly-spaced subset for multi-AC; bounds memory
                          # while giving a real multi-day cross-centre and
                          # within-centre consistency check


def _even_subset(days, n_max):
    """Pick an evenly-spaced subset of at most n_max days."""
    if len(days) <= n_max:
        return list(days)
    idx = np.linspace(0, len(days) - 1, n_max).round().astype(int)
    return [days[i] for i in idx]


COD_DAYS = td.available_dates("clk_cod", 2024)   # full year (step_09)
COD_DAYS_SUBSET = _even_subset(COD_DAYS, MAX_AC_DAYS)
# Smaller subset for T1/T3/T4 (which run 200 null realisations per day-pool);
# the multi-AC consistency uses the larger MAX_AC_DAYS subset.
T1_T3_T4_DAYS = _even_subset(COD_DAYS, 30)


def multi_ac_lambda():
    """Fit lambda for each analysis centre and check they agree; also report the
    within-centre day-to-day lambda scatter.

    step_09 provides full-year COD (366 days) + GFZ (319 days); an evenly-spaced
    ~90-day subset per centre bounds memory while still giving a real multi-day
    cross-centre and within-centre consistency check (was single-day for most
    centres in v0.1).
    """
    res = {}
    for ac in ("clk_cod", "clk_gfz", "clk_grg", "clk_jpl"):
        ac_days = td.available_dates(ac, 2024)
        ac_subset = _even_subset(ac_days, MAX_AC_DAYS)
        loaded = [(d, s) for d, s in td.load_multi_day(ac, ac_subset)
                  if len(s) >= 8]
        if not loaded:
            continue
        days = [s for _, s in loaded]
        used_days = [d for d, _ in loaded]
        fit, _ = s10.pooled_fit(days, dmax=19000.0)
        # Guard against non-finite fit outputs (insufficient bins / failed
        # curve_fit) — round(np.nan) raises ValueError.
        lam_val = int(round(fit["lam"])) if np.isfinite(fit["lam"]) else None
        r2_val = float(round(fit["R2"], 3)) if np.isfinite(fit["R2"]) else None
        res[ac] = {"lambda_km": lam_val, "R2": r2_val,
                   "n_stations": len(days[0]),
                   "n_days": len(days),
                   "days_used": used_days}
        print(f"  {ac:8s}: lambda {fit['lam']:5.0f} km   R2 {fit['R2']:.3f}   "
              f"({len(days[0])} stn, {len(days)} days)")
        # Within-centre day-to-day variability for centres with >1 day
        if len(days) > 1:
            per_day_lams = []
            for d, s in loaded:
                if len(s) < 8:
                    continue
                fd, _ = s10.pooled_fit([s], dmax=19000.0)
                if np.isfinite(fd["lam"]):
                    per_day_lams.append(fd["lam"])
            if len(per_day_lams) >= 2:
                cv = float(np.std(per_day_lams) / np.mean(per_day_lams))
                res[ac]["per_day_lambda_km"] = [float(x) for x in per_day_lams]
                res[ac]["within_centre_cv"] = cv
                print(f"           within-centre day-to-day CV: {cv:.2f}  "
                      f"(per-day: {[f'{x:.0f}' for x in per_day_lams]})")
    # Include ALL analysis centres that produced a fit — do not exclude
    # low-R² centres, as that would be cherry-picking.  Report each
    # centre's R² transparently so the reader can judge quality.
    lams = [v["lambda_km"] for v in res.values() if v["lambda_km"] is not None]
    consistent = bool(len(lams) >= 3 and (np.std(lams) / np.mean(lams) < 0.35))
    cv = float(np.std(lams) / np.mean(lams)) if lams else None
    print(f"  cross-AC lambda CV = {cv:.2f}   consistent: {consistent}")
    return {"per_ac": res, "lambda_cv": cv,
            "consistent": consistent}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Use a 30-day evenly-spaced subset for T1/T3/T4 to keep runtime
    # bounded (T1 runs 200 null realisations); the full-year multi-AC
    # consistency is handled separately with a 90-day subset.
    t1_t3_t4_days = T1_T3_T4_DAYS
    print("=" * 72)
    print("STEP 14 -- Phase 1 on the global precise-clock (CLK) network")
    print(f"  product: COD0MGXFIN,  days: {len(COD_DAYS)} available, "
          f"{len(t1_t3_t4_days)} used (evenly-spaced subset)")
    print("=" * 72)

    print("\n[T1] correlation length + nulls (global precise)")
    t1 = t1_run("clk_cod", t1_t3_t4_days, dmax=19000.0, label="global_precise")

    print("\n[T3] resolvability vs aperture (global precise)")
    t3 = t3_run("clk_cod", t1_t3_t4_days, "global_precise")

    print("\n[T4] distance vs satellite co-visibility (global precise)")
    t4 = t4_run("clk_cod", t1_t3_t4_days, "global_precise")

    print("\n[MA] multi-analysis-centre lambda consistency (full-year subset)")
    ma = multi_ac_lambda()

    result = {
        "test": "phase1_global_precise_clock",
        "product": "COD0MGXFIN receiver clocks (RINEX CLK AR records)",
        "n_days": len(COD_DAYS),
        "T1": t1, "T3": t3, "T4": t4, "multi_ac": ma,
        "T1_pass": bool(t1["T1_pass"]),
        "T3_pass": bool(t3["T3_pass"]),
        "T4_pass": bool(t4["T4_pass"]),
        "timestamp": datetime.now().isoformat(),
    }
    (OUT / "step_14_precise_global.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 72)
    print(f"  T1 (lambda recovers) ............ {result['T1_pass']}")
    print(f"  T3 (resolvable vs aperture) ..... {result['T3_pass']}")
    print(f"  T4 (distance beats co-visibility) {result['T4_pass']}")
    print(f"  multi-AC lambda consistent ...... {ma['consistent']}")
    print("=" * 72)
    print(f"saved {OUT / 'step_14_precise_global.json'}")


if __name__ == "__main__":
    main()
