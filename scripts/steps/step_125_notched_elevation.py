#!/usr/bin/env python3
"""
STEP 125  --  Notched-residual solar elevation independence.

THE KEY TEST OF THE TEP CONTINUUM.

Step 121 showed that after removing solar tidal lines (24h, 12h, 8h, 6h),
the distance structure persists with lambda=1877 km, R2=0.844.  Under the
TEP framing, this notched residual IS the TEP proper-time continuum —
the solar tidal lines are an atmospheric modulation overlaid on the
fundamental field.

The critical question: is this notched residual solar-elevation-
independent?

  * TEP prediction: the notched residual should be solar-elevation-
    INDEPENDENT, because the TEP continuum is a proper-time field that
    does not depend on where the Sun is.  The solar elevation dependence
    seen in Step 123 was carried by the solar tidal lines, which we have
    now removed.
  * Atmospheric prediction: the notched residual should STILL be solar-
    elevation-dependent, because the ionospheric continuum (not just the
    discrete tidal lines) is solar-driven.

This is the most direct test of whether the TEP continuum exists as a
solar-independent component of the common-mode field.

Protocol:
  * Use 60 days of CODE MGEX clocks.
  * For each station pair, compute the notched spectrum (solar tidal
    lines removed) and the solar elevation at the pair midpoint.
  * Bin the notched-residual pairs by solar elevation.
  * Test for solar elevation dependence in the notched coherence.

Output: results/outputs/step_125_notched_elevation.json
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

# Reuse the notched spectrum function from Step 121
_s121 = importlib.util.spec_from_file_location(
    "s121", ROOT / "scripts" / "steps" / "step_121_peak_filtered_coherence.py")
s121 = importlib.util.module_from_spec(_s121)
_s121.loader.exec_module(s121)

OUT = ROOT / "results" / "outputs" / "step_125_notched_elevation.json"
PRODUCT = "clk_cod"
N_DAYS = 60
F_LO, F_HI = 10e-6, 500e-6
TEP_BAND_KM = (1000.0, 5000.0)

NOTCH_PERIODS = [86400.0, 43200.0, 28800.0, 21600.0, 86164.0, 44712.0]
NOTCH_WIDTH_FRAC = 0.05


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

    print("Step 125: Notched-residual solar elevation independence", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print(f"  Notch periods: {[f'{p/3600:.1f}h' for p in NOTCH_PERIODS]}",
          flush=True)
    print()

    notch_freqs = [1.0 / p for p in NOTCH_PERIODS if p > 0]

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Collect notched pairs with solar elevation
    full_pairs = []       # (distance, alignment, elev) — full band
    notched_pairs = []    # (distance, alignment, elev) — notched

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        # Full band
        RF, STA = s53._spectra(ser)
        nm = list(RF)

        # Notched
        RFn, STAn = s121._spectra_notched(ser, notch_freqs=notch_freqs)
        nmn = list(RFn)

        if len(RF) < 8:
            continue

        t0 = np.asarray(ser[nm[0]]["t"], float)
        mid_time_s = float(np.median(t0))
        mid_dt = datetime(2024, 1, 1, tzinfo=timezone.utc) + \
            timedelta(seconds=mid_time_s)

        # Full band pairs
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
            full_pairs.append((d, cpsi, elev))

        # Notched pairs
        for i, j in itertools.combinations(range(len(nmn)), 2):
            Fi, Fj = RFn[nmn[i]], RFn[nmn[j]]
            if Fi.shape != Fj.shape:
                continue
            d = td.great_circle_km(STAn[nmn[i]], STAn[nmn[j]])
            if not (50.0 <= d <= 13000.0):
                continue
            Ssum = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(Ssum) or Ssum == 0:
                continue
            cpsi = float(np.cos(np.angle(Ssum)))
            lat_m, lon_m = _midpoint_latlon(
                STAn[nmn[i]].lat_deg, STAn[nmn[i]].lon_deg,
                STAn[nmn[j]].lat_deg, STAn[nmn[j]].lon_deg)
            elev = _solar_elevation(lat_m, lon_m, mid_dt)
            notched_pairs.append((d, cpsi, elev))

        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(selected)}] full={len(full_pairs)}, "
                  f"notched={len(notched_pairs)}", flush=True)

    print(f"\n  Total: full={len(full_pairs)}, notched={len(notched_pairs)}",
          flush=True)

    def _analyze_elevation(pairs, label):
        """Bin by solar elevation and compute alignment per bin."""
        d = np.array([p[0] for p in pairs], float)
        a = np.array([p[1] for p in pairs], float)
        e = np.array([p[2] for p in pairs], float)
        tep_mask = (d >= TEP_BAND_KM[0]) & (d <= TEP_BAND_KM[1])

        bins = np.arange(-90, 91, 15)
        centers, aligns, counts, sems = [], [], [], []
        for ie in range(len(bins) - 1):
            mask = tep_mask & (e >= bins[ie]) & (e < bins[ie + 1])
            n = mask.sum()
            if n < 50:
                continue
            centers.append(float((bins[ie] + bins[ie + 1]) / 2))
            aligns.append(float(np.mean(a[mask])))
            counts.append(int(n))
            sems.append(float(np.std(a[mask]) / np.sqrt(n)))

        centers = np.array(centers)
        aligns = np.array(aligns)

        r = float(np.corrcoef(centers, aligns)[0, 1]) if len(centers) > 2 else None

        day_mask = tep_mask & (e > 0)
        night_mask = tep_mask & (e < 0)
        day_a = float(np.mean(a[day_mask])) if day_mask.sum() > 0 else None
        night_a = float(np.mean(a[night_mask])) if night_mask.sum() > 0 else None

        print(f"\n  {label}:", flush=True)
        for c, al, n, s in zip(centers, aligns, counts, sems):
            print(f"    elev={c:+6.1f}: align={al:+.4f} ± {s:.4f}, n={n}",
                  flush=True)
        print(f"    r_elev = {r}", flush=True)
        if day_a is not None and night_a is not None:
            print(f"    day={day_a:.4f}, night={night_a:.4f}, "
                  f"ratio={day_a/night_a:.1f}x" if abs(night_a) > 1e-6 else
                  f"    day={day_a:.4f}, night={night_a:.4f}", flush=True)

        return {
            "elevation_bins": [
                {"elevation_center": float(c), "mean_alignment": float(al),
                 "n_pairs": int(n), "sem": float(s)}
                for c, al, n, s in zip(centers, aligns, counts, sems)],
            "r_elevation": r,
            "day_alignment": day_a,
            "night_alignment": night_a,
            "day_night_ratio": (day_a / night_a if day_a and night_a
                                and abs(night_a) > 1e-6 else None),
            "n_day": int(day_mask.sum()),
            "n_night": int(night_mask.sum()),
        }

    full_res = _analyze_elevation(full_pairs, "Full band")
    notched_res = _analyze_elevation(notched_pairs, "Notched (TEP continuum)")

    # Interpretation
    interpretations = []

    full_r = full_res.get("r_elevation")
    notched_r = notched_res.get("r_elevation")

    if full_r is not None and notched_r is not None:
        delta_r = notched_r - full_r
        if abs(notched_r) < 0.3 and abs(full_r) > 0.5:
            interpretations.append(
                f"The notched residual is solar-elevation-INDEPENDENT "
                f"(r={notched_r:+.3f}) while the full band is strongly "
                f"solar-dependent (r={full_r:+.3f}).  The solar elevation "
                f"dependence was carried entirely by the solar tidal lines. "
                f"The TEP continuum (notched residual) is a solar-"
                f"illumination-independent proper-time field.  This is "
                f"strong evidence for TEP.")
        elif abs(notched_r) > 0.5:
            interpretations.append(
                f"The notched residual is STILL solar-elevation-dependent "
                f"(r={notched_r:+.3f}, full band r={full_r:+.3f}).  The "
                f"solar dependence is not just in the tidal lines but in "
                f"the continuum.  The ionospheric common-mode is a broad-"
                f"band solar-driven process, not just discrete tides.")
        else:
            interpretations.append(
                f"The notched residual shows reduced solar-elevation "
                f"dependence (r={notched_r:+.3f}) compared to the full "
                f"band (r={full_r:+.3f}), delta={delta_r:+.3f}.  The solar "
                f"tidal lines carried part of the solar dependence, but "
                f"the continuum also has a solar component.")

    full_ratio = full_res.get("day_night_ratio")
    notched_ratio = notched_res.get("day_night_ratio")
    if full_ratio and notched_ratio:
        interpretations.append(
            f"Day/night ratio: full band {full_ratio:.1f}x, notched "
            f"{notched_ratio:.1f}x.  The notched residual shows "
            f"{'much less' if notched_ratio < full_ratio * 0.5 else 'similar'} "
            f"day/night contrast, suggesting the solar tidal lines carried "
            f"most of the day/night effect.")

    verdict = {
        "step": 125,
        "title": "Notched-residual solar elevation independence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "n_full_pairs": len(full_pairs),
        "n_notched_pairs": len(notched_pairs),
        "full_band": full_res,
        "notched": notched_res,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the notched residual (solar tidal lines removed) "
            "should be solar-elevation-INDEPENDENT, because the TEP "
            "continuum is a proper-time field.  The solar elevation "
            "dependence seen in the full band was carried by the solar "
            "tidal lines.  Under a pure-atmospheric interpretation, the "
            "notched residual should still be solar-elevation-dependent, "
            "because the ionospheric continuum is solar-driven.  "
            "r_notched < 0.3 supports TEP; r_notched > 0.5 supports "
            "atmospheric."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
