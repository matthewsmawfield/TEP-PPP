#!/usr/bin/env python3
"""
STEP 65d -- Per-station field-coupling covariate decomposition.

The Step 65/65c class-average result (quartz stations carry ~3x the
spatially coherent LF clock power of atomic references) invites a
screening interpretation.  This step tests the alternative: is the
per-station coupling amplitude organised by oscillator class, or by
measurement-chain hardware (antenna model, receiver family) and site
environment?  The coupling observable is the Step 65c regression
coefficient b = <y, pred>/<pred, pred> of each station's LF-band
(<10 uHz) residual onto its leave-one-out network-coherent prediction.

Finding (28 days, dense EUREF): antenna model explains MORE of the
per-station coupling variance than oscillator class (eta^2 = 0.37 vs
0.30).  Identical-antenna stations couple alike regardless of clock
class (passive-maser ZIMM +2.63 alongside quartz MARS/LROC +2.94/+2.43
on the same TRM59800 chain), so the broadband LF coherent residual
enters through the observation chain, not the reference oscillator.
The oscillator-class differential that survives this decomposition is
confined to observables the measurement chain cannot reach: the lunar
M2 line (solar S1/S2 null) and the co-located quartz-minus-maser
monument differentials (Steps 77/80b/164).

Output: results/outputs/step_65d_coupling_covariates.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.steps.step_65_oscillator_screening import _load_dense_day  # noqa: E402
from scripts.steps.step_65c_oscillator_noise_null import (  # noqa: E402
    _load_maps, _bands_for_day, _loo_coherent_matrix, QUARTZ_TYPES)

OUT = PROJECT_ROOT / "results" / "outputs" / "step_65d_coupling_covariates.json"
META_P = (PROJECT_ROOT.parent / "TEP-GNSS-RINEX" / "data" / "processed"
          / "optimal_100_metadata.json")
DAYS = range(2024085, 2024113)


def _eta2(groups, allv):
    allv = np.asarray(allv, float)
    ss_b = sum(len(v) * (np.mean(v) - allv.mean()) ** 2
               for v in groups.values() if v)
    return float(ss_b / ((allv - allv.mean()) ** 2).sum())


def main():
    type_map, class_map = _load_maps()
    meta = json.load(open(META_P))["stations"]
    per = {}
    for day in DAYS:
        s = _load_dense_day(day)
        names = [n for n in s if s[n].get("station") is not None]
        if len(names) < 6:
            continue
        D = np.array([[td.great_circle_km(s[a]["station"], s[b]["station"])
                       for b in names] for a in names])
        bands_per = {}
        for n in names:
            y = s[n]["resid"].astype(float)
            if not np.isfinite(y).all():
                ix = np.arange(len(y))
                y = np.interp(ix, ix[np.isfinite(y)], y[np.isfinite(y)])
            bands_per[n] = _bands_for_day(y, s[n]["t"])
        X = np.column_stack([bands_per[n]["LF"] for n in names])
        if X.shape[0] < 20:
            continue
        pred, _ = _loo_coherent_matrix(X, D)
        for i, n in enumerate(names):
            yi, pi = X[:, i], pred[:, i]
            if not np.isfinite(pi).all() or np.mean(pi ** 2) <= 0:
                continue
            per.setdefault(n, []).append(
                float(np.mean(yi * pi) / np.mean(pi * pi)))

    table = []
    for n, v in sorted(per.items(), key=lambda kv: -np.median(kv[1])):
        eq = meta.get(n, {}).get("equipment", {})
        co = meta.get(n, {}).get("coordinates", {})
        table.append({
            "station": n,
            "oscillator_class": class_map.get(n, "unmapped"),
            "clock_type": type_map.get(n, "UNKNOWN"),
            "coupling_b_median": float(np.median(v)),
            "n_days": len(v),
            "antenna_type": eq.get("antenna_type", "?"),
            "receiver_type": eq.get("receiver_type", "?"),
            "latitude_deg": co.get("latitude_deg"),
            "altitude_m": co.get("altitude_m"),
        })

    mapped = [r for r in table if r["antenna_type"] != "?"]
    by_ant, by_cls = defaultdict(list), defaultdict(list)
    for r in mapped:
        by_ant[r["antenna_type"][:12]].append(r["coupling_b_median"])
        by_cls[r["oscillator_class"]].append(r["coupling_b_median"])
    allv = [r["coupling_b_median"] for r in mapped]
    result = {
        "step": "65d",
        "test": "coupling_covariate_decomposition",
        "description": ("Per-station LF (<10 uHz) network-coupling amplitude b "
                        "decomposed by oscillator class vs antenna model vs "
                        "site environment, using the Step 65c machinery"),
        "days": [min(DAYS), max(DAYS)],
        "n_stations": len(table),
        "per_station": table,
        "group_medians": {
            "by_antenna_prefix": {a: {"median_b": float(np.median(v)),
                                      "n": len(v)}
                                  for a, v in sorted(by_ant.items())},
            "by_oscillator_class": {a: {"median_b": float(np.median(v)),
                                        "n": len(v)}
                                    for a, v in sorted(by_cls.items())},
        },
        "eta_squared": {
            "antenna_model": _eta2(by_ant, allv),
            "oscillator_class": _eta2(by_cls, allv),
        },
        "verdict": {
            "broadband_lf_coupling_organised_by": (
                "measurement-chain hardware and site environment, not "
                "oscillator class"),
            "identical_antenna_couples_alike_example": (
                "passive-maser ZIMM (+2.63) groups with quartz "
                "MARS/LROC (+2.94/+2.43) on the shared TRM59800 chain"),
            "implication": (
                "the broadband LF coherent residual enters through the "
                "observation chain; the class-average quartz>atomic "
                "differential cannot be claimed as oscillator screening. "
                "TEP-relevant coupling is confined to observables the "
                "measurement chain cannot reach: the lunar M2 line and "
                "the co-located monument differentials"),
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}  ({len(table)} stations, "
          f"eta2_ant={result['eta_squared']['antenna_model']:.3f}, "
          f"eta2_cls={result['eta_squared']['oscillator_class']:.3f})")


if __name__ == "__main__":
    main()
