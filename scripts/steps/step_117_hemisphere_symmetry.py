#!/usr/bin/env python3
"""
STEP 117  --  Hemisphere symmetry of the TEP clock correlation field.

If the GNSS clock correlation field is a proper-time field (TEP), it should
be symmetric between the northern and southern hemispheres — the same
correlation length scale, the same phase-alignment amplitude, the same
distance decay shape.  If it is an atmospheric/ionospheric common-mode
artefact, it should be hemisphere-asymmetric: the ionosphere, troposphere,
and weather systems are fundamentally different between hemispheres
(different land/ocean distribution, different auroral zones, different
tropospheric humidity).

Protocol:
  * Use the full 2024 CODE MGEX global network.
  * Split station pairs into NH-NH (both lat > 20), SH-SH (both lat < -20),
    and cross-equatorial (one each side).
  * Compute the phase-alignment distance decay for each group.
  * Compare exponential lambda, R^2, and TEP-band mean alignment.
  * TEP predicts: same lambda for all three groups.
  * Atmospheric predicts: different lambda for NH-NH vs SH-SH.

Output: results/outputs/step_117_hemisphere_symmetry.json
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

OUT = ROOT / "results" / "outputs" / "step_117_hemisphere_symmetry.json"
PRODUCT = "clk_cod"
N_DAYS = 30  # sample 30 days across 2024
LAT_CUT = 20.0  # degrees — exclude near-equatorial stations for clean NH/SH split


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 117: Hemisphere symmetry of the TEP clock correlation field",
          flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days across 2024", flush=True)
    print(f"  Lat cut: +-{LAT_CUT} deg", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    if len(dates) < N_DAYS:
        print("  Not enough dates", flush=True)
        return
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    nh_pairs, sh_pairs, cross_pairs, all_pairs = [], [], [], []

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        nm = list(RF)
        if len(RF) < 8:
            continue
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
            lat_i, lat_j = STA[nm[i]].lat_deg, STA[nm[j]].lat_deg
            all_pairs.append((d, cpsi))
            if lat_i > LAT_CUT and lat_j > LAT_CUT:
                nh_pairs.append((d, cpsi))
            elif lat_i < -LAT_CUT and lat_j < -LAT_CUT:
                sh_pairs.append((d, cpsi))
            elif (lat_i > LAT_CUT and lat_j < -LAT_CUT) or \
                 (lat_i < -LAT_CUT and lat_j > LAT_CUT):
                cross_pairs.append((d, cpsi))
        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(selected)}] NH={len(nh_pairs)}, "
                  f"SH={len(sh_pairs)}, cross={len(cross_pairs)}", flush=True)

    print(f"\n  Total: NH={len(nh_pairs)}, SH={len(sh_pairs)}, "
          f"cross={len(cross_pairs)}, all={len(all_pairs)}", flush=True)

    def analyze(pairs, label):
        if len(pairs) < 200:
            return {"label": label, "n_pairs": len(pairs), "error": "too few pairs"}
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        return {
            "label": label,
            "n_pairs": len(pairs),
            "n_bins": int(len(ctr)),
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "exponential_R2": exp_fit.get("R2"),
            "best_model": fit.get("best_model"),
            "R2": fit.get("R2"),
            "tep_band_mean_alignment": (float(np.mean(tep_band)) if tep_band else None),
            "n_tep_band_pairs": len(tep_band),
            "binned_curve": {
                "distance_km": ctr.tolist(),
                "phase_alignment_index": val.tolist(),
                "n_pairs": cnt.tolist(),
            },
        }

    nh_res = analyze(nh_pairs, "northern_hemisphere")
    sh_res = analyze(sh_pairs, "southern_hemisphere")
    cross_res = analyze(cross_pairs, "cross_equatorial")
    all_res = analyze(all_pairs, "all_pairs")

    print(f"\n  NH: lambda={nh_res.get('exponential_lambda_km')}, "
          f"R2={nh_res.get('R2')}, align={nh_res.get('tep_band_mean_alignment')}",
          flush=True)
    print(f"  SH: lambda={sh_res.get('exponential_lambda_km')}, "
          f"R2={sh_res.get('R2')}, align={sh_res.get('tep_band_mean_alignment')}",
          flush=True)
    print(f"  Cross: lambda={cross_res.get('exponential_lambda_km')}, "
          f"R2={cross_res.get('R2')}", flush=True)

    # Comparison
    nh_lam = nh_res.get("exponential_lambda_km")
    sh_lam = sh_res.get("exponential_lambda_km")
    nh_align = nh_res.get("tep_band_mean_alignment")
    sh_align = sh_res.get("tep_band_mean_alignment")

    lambda_ratio = None
    if nh_lam is not None and sh_lam is not None and sh_lam > 0:
        lambda_ratio = float(nh_lam / sh_lam)

    align_diff = None
    if nh_align is not None and sh_align is not None:
        align_diff = float(nh_align - sh_align)

    interpretations = []
    if lambda_ratio is not None:
        if 0.7 <= lambda_ratio <= 1.4:
            interpretations.append(
                f"NH/SH lambda ratio = {lambda_ratio:.3f} — close to unity, "
                f"consistent with TEP (hemisphere-symmetric proper-time field). "
                f"Atmospheric effects would predict a different ratio due to "
                f"the asymmetric land/ocean distribution and different "
                f"ionospheric conditions.")
        else:
            interpretations.append(
                f"NH/SH lambda ratio = {lambda_ratio:.3f} — substantially "
                f"different from unity, which could indicate a hemisphere-"
                f"asymmetric atmospheric/ionospheric confound or a network "
                f"geometry effect (the SH network is much sparser).")

    if align_diff is not None:
        if abs(align_diff) < 0.03:
            interpretations.append(
                f"NH-SH alignment difference = {align_diff:+.4f} — small, "
                f"consistent with hemisphere symmetry.")
        else:
            interpretations.append(
                f"NH-SH alignment difference = {align_diff:+.4f} — the "
                f"hemispheres differ, which could be atmospheric or a "
                f"network-density artefact.")

    # Note the network density asymmetry
    interpretations.append(
        f"Network density: NH has {len(nh_pairs)} pairs, SH has "
        f"{len(sh_pairs)} pairs — the SH network is much sparser, which "
        f"could affect the lambda fit even if the underlying field is "
        f"symmetric.")

    verdict = {
        "step": 117,
        "title": "Hemisphere symmetry of the TEP clock correlation field",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "lat_cut_deg": LAT_CUT,
        "results": {
            "northern_hemisphere": nh_res,
            "southern_hemisphere": sh_res,
            "cross_equatorial": cross_res,
            "all_pairs": all_res,
        },
        "comparison": {
            "nh_sh_lambda_ratio": lambda_ratio,
            "nh_sh_alignment_diff": align_diff,
        },
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the clock correlation field should be hemisphere-"
            "symmetric: same lambda, same alignment amplitude for NH-NH and "
            "SH-SH pairs.  Under an atmospheric/ionospheric interpretation, "
            "the hemispheres should differ due to different land/ocean "
            "distribution, different ionospheric conditions, and different "
            "tropospheric humidity.  The SH network is much sparser, so "
            "network density must be considered."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
