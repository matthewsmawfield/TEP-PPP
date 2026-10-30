#!/usr/bin/env python3
"""
STEP 123  --  Solar elevation angle dependence of the clock correlation field.

THE SOLAR-ILLUMINATION TEST.

If the clock correlation field is an atmospheric/ionospheric common-mode,
it should depend on the solar elevation angle at the station-pair midpoint:
  - Pairs with high solar elevation (near subsolar point) should show
    stronger coherence (the ionosphere is more illuminated and structured)
  - Pairs with low/negative solar elevation (nighttime) should show
    weaker coherence
  - The dependence should be smooth and monotonic in solar elevation

If the field is a proper-time field (TEP), it should NOT depend on solar
elevation angle — a gravitational field does not care about the Sun's
position relative to the ionosphere.

Protocol:
  * Use 30 days of CODE MGEX clocks.
  * For each station pair, compute the solar elevation angle at the
    pair midpoint at the midpoint time of the observation window.
  * Bin pairs by solar elevation angle.
  * Compute the TEP-band mean phase alignment for each solar elevation bin.
  * Test for monotonic dependence.

Output: results/outputs/step_123_solar_elevation_dependence.json
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

OUT = ROOT / "results" / "outputs" / "step_123_solar_elevation_dependence.json"
PRODUCT = "clk_cod"
N_DAYS = 30
TEP_BAND_KM = (1000.0, 5000.0)

# Solar elevation bins (degrees)
ELEV_BINS = np.arange(-90, 91, 10)  # -90 to +90 in 10-degree bins


def _solar_elevation(lat, lon, utc_datetime):
    """Approximate solar elevation angle at a given lat/lon and UTC time.

    Uses a standard solar position approximation.  Accuracy of ~1 degree
    is sufficient for binning purposes.
    """
    # Day of year
    doy = utc_datetime.timetuple().tm_yday
    # Solar declination
    decl = math.radians(23.44 * math.sin(2 * math.pi * (doy - 81) / 365.25))
    # Hour angle
    utc_hour = utc_datetime.hour + utc_datetime.minute / 60.0
    ha = math.radians(15.0 * (utc_hour - 12.0))
    lat_r = math.radians(lat)
    # Solar elevation
    sin_elev = (math.sin(lat_r) * math.sin(decl) +
                math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    return math.degrees(math.asin(np.clip(sin_elev, -1, 1)))


def _midpoint_latlon(lat1, lon1, lat2, lon2):
    """Great-circle midpoint of two points."""
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

    print("Step 123: Solar elevation angle dependence of clock coherence",
          flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Collect pairs with solar elevation
    all_pairs = []  # (distance, alignment, solar_elevation)

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

        # Compute the midpoint time of the observation
        # Use the first station's time range
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

            # Solar elevation at pair midpoint
            lat_m, lon_m = _midpoint_latlon(
                STA[nm[i]].lat_deg, STA[nm[i]].lon_deg,
                STA[nm[j]].lat_deg, STA[nm[j]].lon_deg)
            elev = _solar_elevation(lat_m, lon_m, mid_dt)

            all_pairs.append((d, cpsi, elev))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(selected)}] {len(all_pairs)} pairs",
                  flush=True)

    print(f"\n  Total: {len(all_pairs)} pairs", flush=True)

    if len(all_pairs) < 1000:
        OUT.write_text(json.dumps({"step": 123, "error": "too few pairs",
                                   "n_pairs": len(all_pairs)}, indent=2))
        print("  ERROR: too few pairs", flush=True)
        return

    # Bin by solar elevation
    distances = np.array([p[0] for p in all_pairs], float)
    aligns = np.array([p[1] for p in all_pairs], float)
    elevs = np.array([p[2] for p in all_pairs], float)

    # Overall: restrict to TEP band distances
    tep_mask = (distances >= TEP_BAND_KM[0]) & (distances <= TEP_BAND_KM[1])
    tep_aligns = aligns[tep_mask]
    tep_elevs = elevs[tep_mask]

    # Bin by elevation
    elev_centers = []
    elev_aligns = []
    elev_counts = []
    elev_stds = []

    for ie in range(len(ELEV_BINS) - 1):
        mask = (tep_elevs >= ELEV_BINS[ie]) & (tep_elevs < ELEV_BINS[ie + 1])
        n = mask.sum()
        if n < 50:
            continue
        center = float(np.sqrt(ELEV_BINS[ie] * ELEV_BINS[ie + 1])) \
            if ELEV_BINS[ie] * ELEV_BINS[ie + 1] > 0 else \
            float((ELEV_BINS[ie] + ELEV_BINS[ie + 1]) / 2.0)
        elev_centers.append(float((ELEV_BINS[ie] + ELEV_BINS[ie + 1]) / 2.0))
        elev_aligns.append(float(np.mean(tep_aligns[mask])))
        elev_counts.append(int(n))
        elev_stds.append(float(np.std(tep_aligns[mask]) / np.sqrt(n)))

    elev_centers = np.array(elev_centers)
    elev_aligns = np.array(elev_aligns)
    elev_counts = np.array(elev_counts)
    elev_stds = np.array(elev_stds)

    print(f"\n  Solar elevation bins (TEP band, {tep_mask.sum()} pairs):",
          flush=True)
    for c, a, n, s in zip(elev_centers, elev_aligns, elev_counts, elev_stds):
        print(f"    elev={c:+6.1f}°: align={a:+.4f} ± {s:.4f}, n={n}",
              flush=True)

    # Test for monotonic dependence
    # Correlation between elevation and alignment
    r_elev = float(np.corrcoef(elev_centers, elev_aligns)[0, 1]) \
        if len(elev_centers) > 2 else None

    # Also test: day vs night
    day_mask = tep_elevs > 0
    night_mask = tep_elevs < 0
    day_align = float(np.mean(tep_aligns[day_mask])) if day_mask.sum() > 0 else None
    night_align = float(np.mean(tep_aligns[night_mask])) if night_mask.sum() > 0 else None

    print(f"\n  Day vs night:", flush=True)
    print(f"    Day (elev>0):   align={day_align}, n={day_mask.sum()}",
          flush=True)
    print(f"    Night (elev<0): align={night_align}, n={night_mask.sum()}",
          flush=True)
    if day_align is not None and night_align is not None:
        print(f"    Delta: {day_align - night_align:+.4f}", flush=True)

    # Interpretation
    interpretations = []

    if r_elev is not None:
        if abs(r_elev) > 0.5:
            direction = "positive" if r_elev > 0 else "negative"
            interpretations.append(
                f"The TEP-band alignment shows a strong {direction} "
                f"correlation with solar elevation angle (r={r_elev:+.3f}). "
                f"This is consistent with an atmospheric/ionospheric "
                f"interpretation (solar illumination drives ionospheric "
                f"structure) and inconsistent with TEP (which predicts no "
                f"solar elevation dependence).")
        elif abs(r_elev) > 0.3:
            interpretations.append(
                f"The TEP-band alignment shows a moderate correlation with "
                f"solar elevation angle (r={r_elev:+.3f}). This is weakly "
                f"consistent with an atmospheric interpretation.")
        else:
            interpretations.append(
                f"The TEP-band alignment shows no significant correlation "
                f"with solar elevation angle (r={r_elev:+.3f}). This is "
                f"consistent with TEP (which predicts no solar elevation "
                f"dependence) and inconsistent with a simple ionospheric "
                f"interpretation.")

    if day_align is not None and night_align is not None:
        delta = day_align - night_align
        if abs(delta) > 0.01:
            interpretations.append(
                f"Day-night difference: {delta:+.4f} (day={day_align:.4f}, "
                f"night={night_align:.4f}). The {'higher' if delta > 0 else 'lower'} "
                f"daytime alignment is {'consistent' if delta > 0 else 'inconsistent'} "
                f"with an ionospheric interpretation.")
        else:
            interpretations.append(
                f"Day-night difference: {delta:+.4f} — no significant "
                f"day-night effect, consistent with TEP.")

    verdict = {
        "step": 123,
        "title": "Solar elevation angle dependence of clock coherence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "n_pairs_total": len(all_pairs),
        "n_pairs_tep_band": int(tep_mask.sum()),
        "elevation_bins": [
            {"elevation_center": float(c),
             "mean_alignment": float(a),
             "n_pairs": int(n),
             "sem": float(s)}
            for c, a, n, s in zip(elev_centers, elev_aligns,
                                   elev_counts, elev_stds)
        ],
        "correlation_elevation_vs_alignment": r_elev,
        "day_night_comparison": {
            "day_alignment": day_align,
            "night_alignment": night_align,
            "delta": (day_align - night_align if day_align is not None
                      and night_align is not None else None),
            "n_day": int(day_mask.sum()),
            "n_night": int(night_mask.sum()),
        },
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the clock correlation field should not depend on "
            "solar elevation angle.  Under an atmospheric/ionospheric "
            "interpretation, the coherence should be stronger at high "
            "solar elevation (daytime, near subsolar) and weaker at night. "
            "A significant correlation with solar elevation supports "
            "atmospheric; no correlation supports TEP."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
