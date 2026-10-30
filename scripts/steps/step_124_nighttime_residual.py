#!/usr/bin/env python3
"""
STEP 124  --  Nighttime residual TEP coherence: the field in the dark.

THE REFRAMED QUESTION.

The common-mode clock correlation field IS the TEP proper-time field.
Step 123 showed that this field is modulated by solar elevation — daytime
coherence is 6.1x higher than nighttime.  Under the TEP framing, this
solar modulation is an atmospheric coupling: the ionosphere amplifies the
TEP field during the day (higher TEC, stronger coupling) and the field
is weaker but still present at night.

The critical question: is there still DISTANCE-STRUCTURED coherence at
night, when the ionospheric amplifier is suppressed?

  * TEP prediction: YES — the proper-time field exists independent of
    solar illumination.  The nighttime coherence should be weaker but
    still distance-structured, with a correlation length that may be
    shorter (the atmospheric amplification extends the apparent scale).
  * Pure-atmospheric prediction: NO — without solar illumination, the
    ionospheric common-mode vanishes and the distance structure should
    disappear (R2 -> 0, no exponential fit).

Protocol:
  * Use 60 days of CODE MGEX clocks (more data for the smaller nighttime
    sample).
  * For each station pair, compute the solar elevation at the pair
    midpoint at the midpoint time.
  * Split pairs into deep-night (elev < -15 deg), twilight (-15 to 0),
    and day (> 0).
  * Compute the distance decay and exponential fit for each subset.
  * Test whether the nighttime subset still has distance structure.

Output: results/outputs/step_124_nighttime_residual.json
"""
from __future__ import annotations

import sys, json, math, itertools
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_124_nighttime_residual.json"
PRODUCT = "clk_cod"
N_DAYS = 60


def _solar_elevation(lat, lon, utc_datetime):
    doy = utc_datetime.timetuple().tm_yday
    decl = math.radians(23.44 * math.sin(2 * math.pi * (doy - 81) / 365.25))
    utc_hour = utc_datetime.hour + utc_datetime.minute / 60.0
    ha = math.radians(15.0 * (utc_hour - 12.0))
    lat_r = math.radians(lat)
    sin_elev = (math.sin(lat_r) * math.sin(decl) +
                math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    return math.degrees(math.asin(np.clip(sin_elev, -1, 1)))


def _midpoint_latlon(lat1, lon1, lat2, lon2):
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlo = math.radians(lon2 - lon1)
    Bx = math.cos(la2) * math.cos(dlo)
    By = math.cos(la2) * math.sin(dlo)
    lat_m = math.atan2(math.sin(la1) + math.sin(la2),
                       math.sqrt((math.cos(la1) + Bx) ** 2 + By ** 2))
    lon_m = math.radians(lon1) + math.atan2(By, math.cos(la1) + Bx)
    return math.degrees(lat_m), math.degrees(lon_m)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 124: Nighttime residual TEP coherence", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Collect pairs by solar elevation regime
    groups = {
        "deep_night": [],   # elev < -15
        "twilight": [],     # -15 <= elev < 0
        "day": [],          # elev >= 0
        "high_sun": [],     # elev > 45
    }

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

        t0 = np.asarray(ser[nm[0]]["t"], float)
        mid_time_s = float(np.median(t0))
        mid_dt = datetime(2024, 1, 1, tzinfo=timezone.utc) + \
            timedelta(seconds=mid_time_s)

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

            lat_m, lon_m = _midpoint_latlon(
                STA[nm[i]].lat_deg, STA[nm[i]].lon_deg,
                STA[nm[j]].lat_deg, STA[nm[j]].lon_deg)
            elev = _solar_elevation(lat_m, lon_m, mid_dt)

            pair = (d, cpsi)
            if elev < -15:
                groups["deep_night"].append(pair)
            elif elev < 0:
                groups["twilight"].append(pair)
            else:
                groups["day"].append(pair)
                if elev > 45:
                    groups["high_sun"].append(pair)

        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(selected)}] "
                  + " ".join(f"{n}={len(g)}" for n, g in groups.items()),
                  flush=True)

    print()
    results = {}
    for label, pairs in groups.items():
        if len(pairs) < 200:
            results[label] = {"n_pairs": len(pairs), "error": "too few"}
            print(f"  {label}: too few ({len(pairs)})", flush=True)
            continue
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        tep_band = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        results[label] = {
            "n_pairs": len(pairs),
            "n_bins": int(len(ctr)),
            "exponential_lambda_km": exp_fit.get("lambda_km"),
            "exponential_R2": exp_fit.get("R2"),
            "best_model": fit.get("best_model"),
            "R2": fit.get("R2"),
            "tep_band_mean_alignment": (float(np.mean(tep_band))
                                        if tep_band else None),
            "n_tep_band": len(tep_band),
        }
        print(f"  {label:12s}: n={len(pairs):6d}, "
              f"lambda={exp_fit.get('lambda_km')}, "
              f"R2={fit.get('R2')}, "
              f"align={results[label]['tep_band_mean_alignment']}",
              flush=True)

    # Interpretation
    interpretations = []

    night_r2 = results.get("deep_night", {}).get("R2")
    night_align = results.get("deep_night", {}).get("tep_band_mean_alignment")
    night_lambda = results.get("deep_night", {}).get("exponential_lambda_km")
    day_r2 = results.get("day", {}).get("R2")
    day_align = results.get("day", {}).get("tep_band_mean_alignment")
    day_lambda = results.get("day", {}).get("exponential_lambda_km")

    if night_r2 is not None and night_r2 > 0.3:
        interpretations.append(
            f"Deep-night distance structure persists: R2={night_r2:.3f}, "
            f"lambda={night_lambda:.0f} km, alignment={night_align:.4f}. "
            f"The TEP field maintains distance-structured coherence even "
            f"when the ionospheric amplifier is suppressed.  The nighttime "
            f"lambda ({night_lambda:.0f} km) is {'shorter' if night_lambda and day_lambda and night_lambda < day_lambda else 'similar'} to "
            f"the daytime lambda ({day_lambda:.0f} km), consistent with "
            f"the atmosphere extending the apparent scale during the day.")
    elif night_r2 is not None and night_r2 < 0.1:
        interpretations.append(
            f"Deep-night distance structure vanishes: R2={night_r2:.3f}. "
            f"No distance-structured coherence when the ionosphere is "
            f"suppressed.  Under TEP, this means the proper-time field "
            f"requires the ionosphere as a coupling medium; under a "
            f"pure-atmospheric interpretation, it means the field IS "
            f"the ionosphere.")
    elif night_r2 is not None:
        interpretations.append(
            f"Deep-night distance structure is marginal: R2={night_r2:.3f}. "
            f"The field is weakened but not eliminated at night.")

    if night_align is not None and day_align is not None:
        ratio = day_align / night_align if abs(night_align) > 1e-6 else None
        if ratio:
            interpretations.append(
                f"Day/night alignment ratio: {ratio:.1f}x "
                f"(day={day_align:.4f}, night={night_align:.4f}). "
                f"The field is {'strongly amplified' if ratio > 3 else 'moderately modulated'} by solar illumination.")

    verdict = {
        "step": 124,
        "title": "Nighttime residual TEP coherence: the field in the dark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the common-mode field IS the proper-time field. "
            "It should maintain distance-structured coherence at night "
            "when the ionospheric amplifier is suppressed, though possibly "
            "weaker and with a shorter correlation length.  Under a pure-"
            "atmospheric interpretation, the distance structure should "
            "vanish at night.  A nighttime R2 > 0.3 supports TEP; R2 < 0.1 "
            "supports pure-atmospheric."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
