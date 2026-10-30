#!/usr/bin/env python3
"""Step 104: Regional x seasonal skill sweep (P5 last requirement).

Reuses the frozen Step-96 leftover-clock protocol unchanged: for every
usable day of 2024 (development year, DOY 7-366 -- full seasonal cycle)
and 2025 (independent year, DOY 7-90) and every coherent region
(EUROPE, NORTH_AMERICA, EAST_ASIA), the per-day predictor scores are
persisted by step_96's patched run() and re-aggregated by season.

Season bins (meteorological, Northern Hemisphere):
  DJF = DOY < 60 or DOY >= 335; MAM = 60 <= DOY < 152;
  JJA = 152 <= DOY < 244; SON = 244 <= DOY < 335.

P5 evidence: the frozen prior must stay positive vs linrate and the
empirical-covariance map in every season with non-trivial coverage.
Day-clustered bootstrap CIs as in step_96.

Output: results/outputs/step_104b_seasonal_regional_sweep.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "s96", ROOT / "scripts" / "steps" / "step_96_leftover_clock_supremacy.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["s96"] = _spec.loader and m
_spec.loader.exec_module(m)

OUT = ROOT / "results" / "outputs" / "step_104b_seasonal_regional_sweep.json"
PRODUCT = os.environ.get("TEP_S104_PRODUCT", "clk_cod_ops")
STRIDE_2024 = int(os.environ.get("TEP_S104_STRIDE2024", "2"))
REGIONS = m.REGIONS
TEP_ARMS = m.TEP_ARMS
BASELINES = m.BASELINES
RNG = np.random.default_rng(20261016)


def _season(tag):
    doy = int(tag[4:])
    if doy < 60 or doy >= 335:
        return "DJF"
    if doy < 152:
        return "MAM"
    if doy < 244:
        return "JJA"
    return "SON"


def _ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return {"median": float(np.median(v)) if len(v) else None,
                "ci95": None, "n": int(len(v))}
    bs = [float(np.median(RNG.choice(v, len(v), replace=True)))
          for _ in range(4000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": int(len(v)),
            "frac_positive": float(np.mean(v > 0))}


def main():
    m.PRODUCT = PRODUCT
    seasons = ["DJF", "MAM", "JJA", "SON"]
    out = {"step": 104,
           "title": "Regional x seasonal sweep of the frozen leftover-clock "
                    "predictor (P5)",
           "product": PRODUCT,
           "season_bins_doy": {"DJF": "<60 or >=335", "MAM": "60-151",
                               "JJA": "152-243", "SON": "244-334"},
           "cells": {}}

    for year, stride in ((2024, STRIDE_2024), (2025, 1)):
        dates = m.td.available_dates(PRODUCT, year)
        if not dates:
            continue
        usable = dates[m.GRAM_DAYS + 1:]
        days = usable[::stride]
        for region, rfn in REGIONS.items():
            t0 = time.time()
            r = m.run(days, rfn)
            pd = r.get("per_day", {})
            cell = {}
            for hb in ("mid", "long"):
                tags = pd.get(f"{hb}:__days", [])
                per_season = {s: [] for s in seasons}
                idx = {s: [] for s in seasons}
                for i, tg in enumerate(tags):
                    idx[_season(tg)].append(i)
                hb_out = {}
                for s in seasons:
                    ii = idx[s]
                    if not ii:
                        hb_out[s] = {"n_days": 0}
                        continue
                    row = {"n_days": len(ii)}
                    # skill vs the two deployable-hardest baselines
                    for bl in ("linrate", "empirical_cov"):
                        for arm in ("tep_lfmf_gated", "tep_bandpass"):
                            key = f"{hb}:{arm}_vs_{bl}"
                            vec = pd.get(key, [])
                            row[f"{arm}_vs_{bl}"] = _ci(
                                [vec[i] for i in ii if i < len(vec)])
                    for arm in ("tep_lfmf_gated",):
                        key = f"{hb}:{arm}_rms"
                        vec = pd.get(key, [])
                        row[f"{arm}_rms_median_m"] = (
                            float(np.median([vec[i] for i in ii
                                             if i < len(vec)]))
                            if ii else None)
                    hb_out[s] = row
                cell[hb] = hb_out
            out["cells"][f"{PRODUCT}_{year}_{region}"] = cell
            print(f"{PRODUCT}/{year}/{region}: {r['n_days_used']} days "
                  f"in {time.time()-t0:.0f}s", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(out, indent=2))

    # verdict: seasons with n>=5 days whose gated-arm CI vs linrate is
    # positive at the mid horizon
    summary = {}
    for ck, cell in out["cells"].items():
        hits, tot = 0, 0
        for s, row in cell.get("mid", {}).items():
            if row.get("n_days", 0) < 5:
                continue
            tot += 1
            ci = (row.get("tep_lfmf_gated_vs_linrate") or {}).get("ci95")
            if ci and ci[0] > 0:
                hits += 1
        summary[ck] = {"seasons_evaluated": tot,
                       "seasons_positive_ci": hits}
    out["summary"] = summary
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
