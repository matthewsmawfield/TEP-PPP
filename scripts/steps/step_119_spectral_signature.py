#!/usr/bin/env python3
"""
STEP 119  --  Spectral signature of the network mean clock: TEP vs atmospheric.

THE SPECTRAL DISCRIMINATOR.

If the GNSS clock common-mode field is a proper-time field (TEP), its power
spectrum should be:
  - Scale-free (1/f or 1/f^2) — a fundamental field has no preferred period
  - NO peaks at 24 h (solar day) or 12 h (semidiurnal tide)
  - NO peaks at 23h 56m (sidereal day = satellite geometry repeat)
  - NO peaks at orbital periods (GPS: 11h 58m, Galileo: 14h 5m)

If it is an atmospheric/ionospheric common-mode, its spectrum should show:
  - Peaks at 24 h (solar-driven diurnal)
  - Peaks at 12 h (semidiurnal tide)
  - Possible peaks at 23h 56m (sidereal multipath)

This is a clean spectral discriminator.  A scale-free spectrum with no solar
or sidereal peaks is strong evidence for TEP.  Peaks at solar/sidereal
periods are evidence for atmospheric/multipath.

Protocol:
  * Compute the daily network mean clock residual across 60 days of CODE
    MGEX data.
  * Concatenate into a single time series.
  * Compute the power spectrum.
  * Check for peaks at known solar, sidereal, and orbital periods.
  - Fit 1/f^alpha and report alpha.
  - Compute the suppression/enhancement at each known period relative to
    the local 1/f background.

Output: results/outputs/step_119_spectral_signature.json
"""
from __future__ import annotations

import sys, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_119_spectral_signature.json"
PRODUCT = "clk_cod"
N_DAYS = 60  # 60 days for good low-frequency resolution

# Known periods to check (in seconds)
KNOWN_PERIODS = {
    "solar_diurnal_24h":     86400.0,
    "semidiurnal_12h":       43200.0,
    "sidereal_23h56m":       86164.0,
    "gps_orbital_11h58m":    43082.0,
    "galileo_orbital_14h05m": 50670.0,
    "lunar_semidiurnal_12h25m": 44712.0,  # M2 tide
    "lunar_diurnal_24h50m":  89424.0,     # K1/lunar
    "8h_tertiary":           28800.0,     # S3
    "6h_quaternary":         21600.0,     # S4
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 119: Spectral signature of the network mean clock", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    if len(dates) < N_DAYS:
        print("  Not enough dates", flush=True)
        return
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    selected = [dates[i] for i in idx]

    # Build the network mean clock time series
    # For each day, compute the network mean residual at each timestamp
    all_t = []
    all_mean = []

    for k, tag in enumerate(selected):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) < 8:
            continue

        # Find common timestamps across stations
        all_ts = set()
        for n, v in ser.items():
            t = np.asarray(v["t"], float)
            for ti in t:
                all_ts.add(round(ti, 1))
        common_t = sorted(all_ts)

        if len(common_t) < 100:
            continue

        # Compute network mean at each common timestamp
        t_arr = np.array(common_t)
        mean_arr = np.full(len(common_t), np.nan)

        for i, ti in enumerate(common_t):
            vals = []
            for n, v in ser.items():
                t = np.asarray(v["t"], float)
                r = np.asarray(v["resid"], float)
                ok = np.isfinite(t) & np.isfinite(r)
                if ok.sum() == 0:
                    continue
                # Find nearest timestamp
                j = np.argmin(np.abs(t[ok] - ti))
                if abs(t[ok][j] - ti) < 35.0:  # within 35s
                    vals.append(r[ok][j])
            if len(vals) >= 5:
                mean_arr[i] = np.mean(vals)

        # Detrend each day's network mean before concatenation to remove
        # day-boundary discontinuities from daily-mean clock drift.
        # Receiver clocks drift by ~us/day; the daily mean of residuals
        # can shift by ns between days, creating step discontinuities that
        # inject spurious low-frequency power.  Removing the per-day linear
        # trend eliminates this artefact while preserving intra-day structure.
        ok_day = np.isfinite(mean_arr)
        if ok_day.sum() > 10:
            t_day = t_arr[ok_day]
            m_day = mean_arr[ok_day]
            tt_day = t_day - t_day[0]
            m_day = m_day - np.polyval(np.polyfit(tt_day, m_day, 1), tt_day)
            mean_arr[ok_day] = m_day

        # Offset time to start of day
        if len(t_arr) > 0:
            t_arr = t_arr - t_arr[0] + (k * 86400.0)

        ok = np.isfinite(mean_arr)
        if ok.sum() > 50:
            all_t.extend(t_arr[ok].tolist())
            all_mean.extend(mean_arr[ok].tolist())

        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(selected)}] {len(all_t)} points so far",
                  flush=True)

    if len(all_t) < 1000:
        OUT.write_text(json.dumps({"step": 119, "error": "too few points",
                                   "n_points": len(all_t)}, indent=2))
        print("  ERROR: too few points", flush=True)
        return

    t = np.array(all_t, float)
    x = np.array(all_mean, float)

    # Remove linear trend
    tt = t - t[0]
    x = x - np.polyval(np.polyfit(tt, x, 1), tt)

    # Compute median dt
    dt = float(np.median(np.diff(np.sort(tt))))
    if not np.isfinite(dt) or dt <= 0:
        dt = 300.0  # 5 min default for CODE receiver clocks

    print(f"\n  {len(t)} points, dt={dt:.1f}s, duration={tt[-1]/86400:.1f} days",
          flush=True)

    # Interpolate to regular grid
    n_grid = int(np.ceil(tt[-1] / dt)) + 1
    grid = np.arange(n_grid) * dt
    x_reg = np.interp(grid, tt, x, left=x[0], right=x[-1])

    # Remove mean and linear trend
    x_reg = x_reg - np.mean(x_reg)
    x_reg = x_reg - np.polyval(np.polyfit(np.arange(n_grid, dtype=float), x_reg, 1),
                                np.arange(n_grid, dtype=float))

    # Apply Hann window
    window = np.hanning(n_grid)
    x_w = x_reg * window

    # Compute power spectrum
    F = np.fft.rfft(x_w)
    freq = np.fft.rfftfreq(n_grid, d=dt)
    power = np.abs(F) ** 2

    # Normalize power
    power_norm = power / np.sum(power[1:])  # exclude DC

    # Fit 1/f^alpha in log-log space, excluding DC, Nyquist, and known
    # peak regions (24h, 12h, 8h, 6h, sidereal, lunar M2) so that the
    # continuum slope is not biased by discrete spectral lines.
    f_fit = freq[1:-1]
    p_fit = power_norm[1:-1]
    ok = (f_fit > 1e-6) & (p_fit > 0) & np.isfinite(p_fit)

    # Mask ±10% around each known period
    for _name, period_s in KNOWN_PERIODS.items():
        f_target = 1.0 / period_s
        mask_peak = (f_fit >= f_target * 0.9) & (f_fit <= f_target * 1.1)
        ok &= ~mask_peak

    if ok.sum() > 10:
        log_f = np.log10(f_fit[ok])
        log_p = np.log10(p_fit[ok])
        alpha, log_const = np.polyfit(log_f, log_p, 1)
        alpha = float(alpha)
    else:
        alpha = None

    print(f"  1/f^alpha fit: alpha = {alpha}", flush=True)

    # Check for peaks at known periods
    peak_results = {}
    for name, period_s in KNOWN_PERIODS.items():
        f_target = 1.0 / period_s
        # Find the nearest frequency bin
        if f_target < freq[1] or f_target > freq[-1]:
            peak_results[name] = {"frequency_hz": f_target, "in_band": False}
            continue

        # Find bins within +/- 20% of the target frequency
        f_lo = f_target * 0.8
        f_hi = f_target * 1.2
        mask = (freq >= f_lo) & (freq <= f_hi)

        if mask.sum() == 0:
            peak_results[name] = {"frequency_hz": f_target, "in_band": False}
            continue

        # Local power at the target frequency
        local_power = float(np.max(power_norm[mask]))
        local_freq = float(freq[mask][np.argmax(power_norm[mask])])

        # Background power: median of surrounding bins (excluding the peak region)
        bg_mask = (freq >= f_lo * 0.5) & (freq <= f_hi * 2.0) & ~mask
        if bg_mask.sum() > 3:
            bg_power = float(np.median(power_norm[bg_mask]))
        else:
            bg_power = float(np.median(power_norm[1:-1]))

        # Enhancement ratio
        if bg_power > 0:
            enhancement = local_power / bg_power
        else:
            enhancement = None

        # Also compute the 1/f-predicted power at this frequency
        if alpha is not None:
            f_pred = 10 ** (alpha * np.log10(f_target) + log_const)
            ratio_to_1f = local_power / f_pred if f_pred > 0 else None
        else:
            ratio_to_1f = None

        peak_results[name] = {
            "frequency_hz": f_target,
            "period_s": period_s,
            "in_band": True,
            "local_max_power": local_power,
            "local_max_freq_hz": local_freq,
            "background_median_power": bg_power,
            "enhancement_ratio": enhancement,
            "ratio_to_1f_prediction": ratio_to_1f,
        }

        print(f"  {name}: enhancement={enhancement:.2f}x, "
              f"ratio_to_1f={ratio_to_1f:.2f}" if ratio_to_1f else
              f"  {name}: enhancement={enhancement:.2f}x", flush=True)

    # Interpretation
    interpretations = []

    # Check 1/f fit quality
    if alpha is not None:
        if -2.5 <= alpha <= -0.5:
            interpretations.append(
                f"The network mean clock spectrum follows 1/f^{abs(alpha):.2f}, "
                f"consistent with a scale-free fundamental field (TEP "
                f"prediction).  Atmospheric processes typically show peaks "
                f"at solar periods rather than a smooth power law.")
        else:
            interpretations.append(
                f"The network mean clock spectrum follows 1/f^{abs(alpha):.2f}, "
                f"which is {'steeper' if alpha < -2 else 'flatter'} than "
                f"typical 1/f noise.")

    # Check for solar/sidereal peaks
    solar_24h = peak_results.get("solar_diurnal_24h", {}).get("enhancement_ratio")
    semidiurnal_12h = peak_results.get("semidiurnal_12h", {}).get("enhancement_ratio")
    sidereal = peak_results.get("sidereal_23h56m", {}).get("enhancement_ratio")
    lunar_m2 = peak_results.get("lunar_semidiurnal_12h25m", {}).get("enhancement_ratio")

    if solar_24h is not None and semidiurnal_12h is not None:
        if solar_24h < 2.0 and semidiurnal_12h < 2.0:
            interpretations.append(
                f"No significant peaks at 24h (enhancement {solar_24h:.2f}x) "
                f"or 12h (enhancement {semidiurnal_12h:.2f}x) — the spectrum "
                f"is NOT dominated by solar-driven atmospheric tides.  This "
                f"is consistent with TEP and inconsistent with a "
                f"tropospheric/ionospheric common-mode interpretation.")
        else:
            interpretations.append(
                f"Peaks detected at 24h ({solar_24h:.2f}x) and/or 12h "
                f"({semidiurnal_12h:.2f}x) — the spectrum shows "
                f"solar-driven atmospheric tidal signatures.")

    if sidereal is not None:
        # Resolution caveat: at 60-day duration, the frequency resolution is
        # 1/(60*86400) = 1.93e-7 Hz.  The 24h-sidereal separation is
        # 1/86400 - 1/86164 = 3.17e-7 Hz — barely resolved.  The 24h and
        # sidereal peaks are likely the same spectral feature.
        freq_res = 1.0 / (60.0 * 86400.0)
        sidereal_solar_sep = abs(1.0 / 86400.0 - 1.0 / 86164.0)
        resolvable = sidereal_solar_sep > freq_res
        if sidereal > 2.0:
            interpretations.append(
                f"Sidereal peak detected ({sidereal:.2f}x) — satellite "
                f"geometry/multipath contributes to the common-mode.  "
                f"Note: at 60-day duration the 24h and sidereal peaks are "
                f"{'barely resolved' if resolvable else 'NOT resolved'} "
                f"(separation {sidereal_solar_sep:.2e} Hz vs resolution "
                f"{freq_res:.2e} Hz), so this peak may be the solar diurnal "
                f"rather than the sidereal.")
        else:
            interpretations.append(
                f"No significant sidereal peak ({sidereal:.2f}x) — "
                f"satellite geometry/multipath is NOT the dominant "
                f"common-mode source.")

    if lunar_m2 is not None:
        if lunar_m2 > 2.0:
            interpretations.append(
                f"Lunar M2 tidal peak detected ({lunar_m2:.2f}x) — the "
                f"lunar semidiurnal tide contributes to the common-mode. "
                f"This could be atmospheric (lunar atmospheric tide) or "
                f"TEP (gravitational coupling).")
        else:
            interpretations.append(
                f"No significant lunar M2 peak ({lunar_m2:.2f}x) — "
                f"the lunar semidiurnal tide is not a dominant "
                f"common-mode source.")

    verdict = {
        "step": 119,
        "title": "Spectral signature of the network mean clock: TEP vs atmospheric",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "n_points": len(t),
        "dt_s": dt,
        "duration_days": float(tt[-1] / 86400.0),
        "spectral_fit": {
            "alpha": alpha,
            "description": f"1/f^{abs(alpha) if alpha else '?'}" ,
        },
        "known_period_peaks": peak_results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "Under TEP, the network mean clock spectrum should be scale-free "
            "(1/f^alpha) with NO peaks at 24h (solar), 12h (semidiurnal), "
            "or 23h56m (sidereal).  Under an atmospheric/ionospheric "
            "interpretation, peaks at 24h and 12h are expected.  Under a "
            "multipath interpretation, a sidereal peak is expected.  "
            "Enhancement ratio > 2.0 at a known period indicates a "
            "significant peak."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
