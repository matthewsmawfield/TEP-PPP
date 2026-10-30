#!/usr/bin/env python3
"""
STEP 116  --  April 8 2024 total solar eclipse: ionospheric perturbation
               vs TEP clock field stability.

THE ECLISE IS A NATURAL IONOSPHERIC PERTURBATION EXPERIMENT.

On 2024-04-08 (DOY 099) a total solar eclipse crossed North America from
Mexico through Texas to Maine.  The ionosphere was directly perturbed along
the shadow path — electron density dropped, TEC changed, and the ionospheric
delay changed for all GNSS signals passing through the shadow.

Under TEP (a proper-time field), the eclipse does NOT change the clock
correlation field — the Moon's shadow blocks sunlight but does not change
the gravitational/proper-time field.  Under an ionospheric common-mode
interpretation, the clock correlation SHOULD change along the shadow path.

This is a clean discriminating test:
  - TEP predicts: no change in lambda or phase-alignment on eclipse day
  - Ionospheric predicts: change along the shadow path, no change elsewhere

Protocol:
  * Compare phase-alignment structure on DOY 099 (eclipse) vs DOY 097-098
    (control) and DOY 100-101 (recovery).
  * Split stations into "near shadow path" (North America, lat 25-50,
    lon -130 to -60) and "far from shadow path" (Europe, Asia, Southern
    Hemisphere).
  * Compute lambda and mean TEP-band alignment for each group on each day.
  * The discriminating signature: if TEP, both groups are stable; if
    ionospheric, the near-path group changes on DOY 099.

Output: results/outputs/step_116_eclipse_ionospheric_test.json
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

OUT = ROOT / "results" / "outputs" / "step_116_eclipse_ionospheric_test.json"
PRODUCT = "clk_cod"

ECLIPSE_DOY = "2024099"
CONTROL_DOYS = ["2024097", "2024098"]
RECOVERY_DOYS = ["2024100", "2024101"]

# Shadow path: Mexico -> Texas -> Oklahoma -> Arkansas -> Missouri ->
# Illinois -> Indiana -> Ohio -> Pennsylvania -> New York -> Vermont -> Maine
# Roughly lat 25-50, lon -130 to -60
NEAR_PATH_LAT = (25.0, 55.0)
NEAR_PATH_LON = (-130.0, -60.0)


def _in_near_path(sta):
    lat, lon = sta.lat_deg, sta.lon_deg
    return (NEAR_PATH_LAT[0] <= lat <= NEAR_PATH_LAT[1] and
            NEAR_PATH_LON[0] <= lon <= NEAR_PATH_LON[1])


def _in_europe(sta):
    return 35.0 <= sta.lat_deg <= 70.0 and -10.0 <= sta.lon_deg <= 40.0


def _in_southern(sta):
    return sta.lat_deg < -20.0


def analyze_day_grouped(tag, label):
    """Compute phase-alignment for near-path, far-field, and cross pairs.

    The key statistic is the mean phase alignment of pairs that involve at
    least one near-path station (near-near and near-far), compared to pairs
    that are entirely far-field.  Under TEP, both should be stable across
    eclipse and control days.  Under ionospheric, the near-involving pairs
    should change on eclipse day.
    """
    try:
        ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                             max_jumps=10_000)
    except Exception:
        return {"label": label, "error": "no data"}

    if len(ser) < 8:
        return {"label": label, "error": f"only {len(ser)} stations"}

    RF, STA = s53._spectra(ser)
    if len(RF) < 8:
        return {"label": label, "error": "too few spectra"}

    nm = list(RF)
    near_names = {n for n in nm if _in_near_path(STA[n])}
    europe_names = {n for n in nm if _in_europe(STA[n])}
    southern_names = {n for n in nm if _in_southern(STA[n])}

    near_near, near_far, far_far, eur_eur, south_south = [], [], [], [], []

    for i, j in itertools.combinations(range(len(nm)), 2):
        Fi, Fj = RF[nm[i]], RF[nm[j]]
        if Fi.shape != Fj.shape:
            continue
        d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
        if not (50.0 <= d <= 13000.0):
            continue
        Ssum = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(Ssum) or Ssum == 0:
            continue
        cpsi = float(np.cos(np.angle(Ssum)))
        ni, nj = nm[i] in near_names, nm[j] in near_names
        ei, ej = nm[i] in europe_names, nm[j] in europe_names
        si, sj = nm[i] in southern_names, nm[j] in southern_names

        if ni and nj:
            near_near.append((d, cpsi))
        elif ni or nj:
            near_far.append((d, cpsi))
        else:
            far_far.append((d, cpsi))

        if ei and ej:
            eur_eur.append((d, cpsi))
        if si and sj:
            south_south.append((d, cpsi))

    def _stats(pairs, label):
        if len(pairs) < 20:
            return {"n_pairs": len(pairs), "mean_alignment": None,
                    "tep_band_mean": None}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        return {
            "n_pairs": len(pairs),
            "mean_alignment": float(np.mean([v for _, v in pairs])),
            "tep_band_mean": (float(np.mean(tep_band)) if tep_band else None),
            "n_tep_band": len(tep_band),
        }

    # Also try fitting for far_far (should have enough pairs)
    ff_fit = None
    if len(far_far) >= 200:
        ctr, val, cnt = s53._curve(far_far)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        ff_fit = {
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "R2": fit.get("R2"),
            "best_model": fit.get("best_model"),
        }

    return {
        "label": label, "doy": tag, "n_all": len(ser),
        "n_near": len(near_names),
        "n_europe": len(europe_names),
        "n_southern": len(southern_names),
        "near_near": _stats(near_near, "near_near"),
        "near_far": _stats(near_far, "near_far"),
        "far_far": _stats(far_far, "far_far"),
        "europe_europe": _stats(eur_eur, "eur_eur"),
        "southern_southern": _stats(south_south, "south_south"),
        "far_far_fit": ff_fit,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 116: April 8 2024 eclipse — ionospheric perturbation vs TEP stability",
          flush=True)
    print(f"  Eclipse: DOY 099 (2024-04-08), North America", flush=True)
    print(f"  Control: DOY 097-098, Recovery: DOY 100-101", flush=True)
    print()

    results = {}
    for tag, label in [(ECLIPSE_DOY, "eclipse")] + \
                     [(t, f"control_{t}") for t in CONTROL_DOYS] + \
                     [(t, f"recovery_{t}") for t in RECOVERY_DOYS]:
        r = analyze_day_grouped(tag, label)
        results[label] = r
        nn = r.get("near_near") or {}
        nf = r.get("near_far") or {}
        ff = r.get("far_far") or {}
        print(f"  {label}: n_near={r.get('n_near')}, "
              f"near_near={nn.get('mean_alignment')}, "
              f"near_far={nf.get('mean_alignment')}, "
              f"far_far={ff.get('mean_alignment')}", flush=True)

    # Summary: compare eclipse day vs control for each pair group
    def _get(label, group, key):
        r = results.get(label, {}).get(group) or {}
        return r.get(key)

    def _mean(labels, group, key):
        vals = [_get(l, group, key) for l in labels
                if _get(l, group, key) is not None]
        return float(np.mean(vals)) if vals else None

    control_labels = [f"control_{t}" for t in CONTROL_DOYS]
    recovery_labels = [f"recovery_{t}" for t in RECOVERY_DOYS]

    summary = {}
    for group in ["near_near", "near_far", "far_far", "europe_europe",
                  "southern_southern"]:
        ecl_align = _get("eclipse", group, "mean_alignment")
        ctrl_align = _mean(control_labels, group, "mean_alignment")
        rec_align = _mean(recovery_labels, group, "mean_alignment")

        delta_align = None
        if ecl_align is not None and ctrl_align is not None:
            delta_align = float(ecl_align - ctrl_align)

        summary[group] = {
            "eclipse_mean_alignment": ecl_align,
            "control_mean_alignment": ctrl_align,
            "recovery_mean_alignment": rec_align,
            "delta_alignment_abs": delta_align,
        }

    # Interpretation
    near_far_delta = summary["near_far"]["delta_alignment_abs"]
    far_far_delta = summary["far_far"]["delta_alignment_abs"]
    eur_delta = summary["europe_europe"]["delta_alignment_abs"]

    interpretations = []
    if near_far_delta is not None and far_far_delta is not None:
        if abs(near_far_delta) > abs(far_far_delta) * 2 and abs(near_far_delta) > 0.02:
            interpretations.append(
                f"Near-path-involving pairs changed by {near_far_delta:+.4f} "
                f"while far-field pairs changed by {far_far_delta:+.4f} — the "
                f"shadow-path-specific change favours an IONOSPHERIC "
                f"interpretation over TEP.")
        elif abs(near_far_delta) < 0.02 and abs(far_far_delta) < 0.02:
            interpretations.append(
                f"Both near-path-involving ({near_far_delta:+.4f}) and far-field "
                f"({far_far_delta:+.4f}) alignment are stable on eclipse day — "
                f"consistent with TEP (the eclipse does not perturb the "
                f"proper-time field) and inconsistent with an ionospheric "
                f"common-mode interpretation.")
        else:
            interpretations.append(
                f"Near-path-involving pairs changed by {near_far_delta:+.4f}, "
                f"far-field by {far_far_delta:+.4f} — the pattern is ambiguous. "
                f"More control days would help.")

    verdict = {
        "step": 116,
        "title": "April 8 2024 total solar eclipse: ionospheric perturbation vs TEP clock field stability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eclipse": {
            "date": "2024-04-08",
            "doy": "2024099",
            "path": "Mexico -> Texas -> Oklahoma -> Arkansas -> Missouri -> Illinois -> Indiana -> Ohio -> Pennsylvania -> New York -> Vermont -> Maine",
            "near_path_box": {"lat": list(NEAR_PATH_LAT), "lon": list(NEAR_PATH_LON)},
        },
        "product": PRODUCT,
        "results": results,
        "summary": summary,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the eclipse does not change the clock correlation "
            "field — the Moon's shadow blocks sunlight but does not change "
            "the proper-time field.  Under an ionospheric common-mode "
            "interpretation, the clock correlation SHOULD change along the "
            "shadow path.  The discriminating signature: if TEP, both "
            "near-path and far-field groups are stable; if ionospheric, "
            "only the near-path group changes."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Summary:", flush=True)
    for g, s in summary.items():
        print(f"    {g}: delta_align={s.get('delta_alignment_abs')}", flush=True)
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
