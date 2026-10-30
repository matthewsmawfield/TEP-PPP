#!/usr/bin/env python3
"""
STEP 122  --  Latitude-controlled quartz-vs-atomic screening.

THE STRONGEST TEP EVIDENCE UNDER SCRUTINY.

Step 103 found that quartz pairs show clear exponential distance decay
(lambda=2454 km, R²=0.936) while atomic pairs do not (R²=0.309).  This is
the strongest single TEP-supportive result.  But the interpretation depends
on whether the screening effect is itself atmospheric.

If the screening effect is driven by ionospheric common-mode:
  - It should be stronger at high latitudes (auroral ionosphere)
  - It should be stronger in summer (higher TEC)
  - It should weaken or disappear at low latitudes near the magnetic equator

If the screening effect is a proper-time field (TEP):
  - It should be latitude-independent
  - It should not depend on season
  - It should be present at all latitudes

Protocol:
  * Use 30 days of CODE MGEX clocks.
  * Classify stations as quartz or atomic using igs_clock_type_full.json.
  * Split each group into three latitude bands:
      high (|lat| > 50), mid (30 <= |lat| <= 50), low (|lat| < 30)
  * Compute the TEP-band mean phase alignment for each subgroup.
  * Compute the screening ratio (QQ/AA) for each latitude band.
  * Test whether the screening ratio varies with latitude.

Output: results/outputs/step_122_lat_controlled_screening.json
"""
from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_122_lat_controlled_screening.json"
PRODUCT = "clk_cod"
N_DAYS = 30
TEP_BAND_KM = (1000.0, 5000.0)
LAT_BANDS = [
    ("high", 50.0, 90.0),
    ("mid", 30.0, 50.0),
    ("low", 0.0, 30.0),
]


def _load_clock_types():
    ct_path = ROOT / "data" / "meta" / "igs_clock_type_full.json"
    if not ct_path.exists():
        return {}
    ct = json.loads(ct_path.read_text())
    return {n: v[0] for n, v in ct.items()}


def _oscillator_type(sta_name, clock_types):
    return clock_types.get(sta_name)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 122: Latitude-controlled quartz-vs-atomic screening", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print()

    clock_types = _load_clock_types()
    print(f"  Clock types loaded: {len(clock_types)} stations", flush=True)

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Collect pairs by (oscillator_type, latitude_band)
    pairs_by_group = {}
    for osc in ["quartz", "atomic"]:
        for lat_name, _, _ in LAT_BANDS:
            pairs_by_group[(osc, lat_name)] = []

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        RF, STA = s53._spectra(ser)
        nm = list(RF)
        if len(RF) < 8:
            continue

        # Classify stations
        station_info = {}
        for n in nm:
            sta = STA[n]
            osc = _oscillator_type(sta.name, clock_types)
            lat = abs(sta.lat_deg)
            lat_band = None
            for name, lo, hi in LAT_BANDS:
                if lo <= lat < hi:
                    lat_band = name
                    break
            if osc in ("QUARTZ", "ATOMIC") and lat_band is not None:
                station_info[n] = (osc.lower(), lat_band)

        for i, j in itertools.combinations(range(len(nm)), 2):
            ni, nj = nm[i], nm[j]
            if ni not in station_info or nj not in station_info:
                continue
            osc_i, lat_i = station_info[ni]
            osc_j, lat_j = station_info[nj]
            # Only same-oscillator, same-latitude-band pairs
            if osc_i != osc_j or lat_i != lat_j:
                continue
            Fi, Fj = RF[ni], RF[nj]
            if Fi.shape != Fj.shape:
                continue
            d = td.great_circle_km(STA[ni], STA[nj])
            if not (50.0 <= d <= 13000.0):
                continue
            Ssum = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(Ssum) or Ssum == 0:
                continue
            pairs_by_group[(osc_i, lat_i)].append(
                (d, float(np.cos(np.angle(Ssum)))))

        if (k + 1) % 10 == 0:
            total = sum(len(v) for v in pairs_by_group.values())
            print(f"  [{k+1}/{len(selected)}] {total} pairs so far",
                  flush=True)

    # Compute TEP-band alignment for each group
    results = {}
    print()
    for (osc, lat), pairs in sorted(pairs_by_group.items()):
        tep = [v for d, v in pairs if TEP_BAND_KM[0] <= d <= TEP_BAND_KM[1]]
        align = float(np.mean(tep)) if len(tep) >= 20 else None

        # Fit lambda
        exp_lambda = None
        r2 = None
        if len(pairs) >= 200:
            ctr, val, cnt = s53._curve(pairs)
            fit = s53._fit(ctr, val, cnt)
            exp_fit = fit.get("exponential") or {}
            exp_lambda = exp_fit.get("lambda_km")
            r2 = fit.get("R2")

        key = f"{osc}_{lat}"
        results[key] = {
            "oscillator": osc,
            "latitude_band": lat,
            "n_pairs": len(pairs),
            "n_tep_band": len(tep),
            "tep_band_alignment": align,
            "exponential_lambda_km": exp_lambda,
            "R2": r2,
        }
        print(f"  {key:15s}: n={len(pairs):6d}, align={align}, "
              f"lambda={exp_lambda}, R2={r2}", flush=True)

    # Compute screening ratios per latitude band
    print()
    screening = {}
    for lat_name, _, _ in LAT_BANDS:
        qq = results.get(f"quartz_{lat_name}", {}).get("tep_band_alignment")
        aa = results.get(f"atomic_{lat_name}", {}).get("tep_band_alignment")
        ratio = None
        diff = None
        if qq is not None and aa is not None and abs(aa) > 1e-6:
            ratio = qq / aa
        if qq is not None and aa is not None:
            diff = qq - aa
        screening[lat_name] = {
            "qq_alignment": qq,
            "aa_alignment": aa,
            "screening_ratio": ratio,
            "screening_diff": diff,
        }
        print(f"  Screening {lat_name:5s}: QQ={qq}, AA={aa}, "
              f"ratio={ratio}, diff={diff}", flush=True)

    # Interpretation
    interpretations = []

    # Check if screening varies with latitude
    ratios = {lat: screening[lat]["screening_ratio"]
              for lat in ["high", "mid", "low"] if screening[lat]["screening_ratio"] is not None}
    diffs = {lat: screening[lat]["screening_diff"]
             for lat in ["high", "mid", "low"] if screening[lat]["screening_diff"] is not None}

    if len(diffs) >= 2:
        diff_values = list(diffs.values())
        diff_range = max(diff_values) - min(diff_values)
        if diff_range < 0.02:
            interpretations.append(
                f"The screening difference is approximately latitude-"
                f"independent (range {diff_range:.4f} across latitude bands). "
                f"This is consistent with TEP (a proper-time field should be "
                f"latitude-independent) and inconsistent with an ionospheric "
                f"interpretation (which predicts stronger screening at high "
                f"latitudes).")
        else:
            # Check direction
            high_diff = diffs.get("high")
            low_diff = diffs.get("low")
            if high_diff is not None and low_diff is not None:
                if high_diff > low_diff:
                    interpretations.append(
                        f"The screening difference is stronger at high "
                        f"latitudes ({high_diff:.4f}) than at low latitudes "
                        f"({low_diff:.4f}), range {diff_range:.4f}. This is "
                        f"consistent with an auroral ionospheric origin and "
                        f"inconsistent with a latitude-independent TEP field.")
                else:
                    interpretations.append(
                        f"The screening difference is stronger at low "
                        f"latitudes ({low_diff:.4f}) than at high latitudes "
                        f"({high_diff:.4f}), range {diff_range:.4f}. This "
                        f"pattern is not the simple ionospheric expectation "
                        f"(which predicts high-lat dominance) and may reflect "
                        f"equatorial ionospheric effects or network geometry.")

    # Check if quartz alignment varies with latitude
    qq_aligns = {lat: results.get(f"quartz_{lat}", {}).get("tep_band_alignment")
                 for lat in ["high", "mid", "low"]}
    qq_aligns = {k: v for k, v in qq_aligns.items() if v is not None}
    if len(qq_aligns) >= 2:
        qq_range = max(qq_aligns.values()) - min(qq_aligns.values())
        interpretations.append(
            f"Quartz TEP-band alignment by latitude: "
            + ", ".join(f"{k}={v:.4f}" for k, v in qq_aligns.items())
            + f" (range {qq_range:.4f}).")

    verdict = {
        "step": 122,
        "title": "Latitude-controlled quartz-vs-atomic screening",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "latitude_bands": [(n, lo, hi) for n, lo, hi in LAT_BANDS],
        "results": results,
        "screening_by_latitude": screening,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the quartz-vs-atomic screening effect should be "
            "latitude-independent.  Under an ionospheric interpretation, "
            "the screening should be stronger at high latitudes (auroral "
            "ionosphere) or at low latitudes (equatorial ionosphere). "
            "A latitude-independent screening supports TEP; a latitude-"
            "dependent screening supports atmospheric/ionospheric."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
