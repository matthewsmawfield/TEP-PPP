#!/usr/bin/env python3
"""
STEP 165  --  Gravitational phase prediction: do observed M2 clock phases
              match the lunar gravitational potential?

THE DEFINITIVE TEP TEST.

The M2 lunar semidiurnal tide (22.365 uHz, 12.42h) is purely gravitational.
The tide-generating potential at a station is:

    V_M2(t) ~ sin^2(colat) * cos(2*(H_moon(t) + lambda))

where H_moon = GMST - alpha_moon is the moon's hour angle at Greenwich,
and lambda is the station longitude.  The phase of this potential at t=0 is:

    phi_grav = 2 * (GMST(t0) - alpha_moon(t0) + lambda)

The phase DIFFERENCE between two stations is:

    dphi_grav = 2 * (lambda1 - lambda2)

This is purely a function of longitude difference.  No atmospheric or
processing effect can produce this specific longitude-dependent phase
pattern.  If the observed M2 clock phases match this prediction, the
signal is gravitational.  Full stop.

TEST:
  1. Harmonic fit M2 at each station (simultaneous with O1, S2, S1, K1, P1).
  2. Compute expected M2 phase from station longitude and lunar ephemeris.
  3. Compare observed vs expected phase across all stations.
  4. Cross-station: does cos(2*(lon1-lon2)) predict the bandpass correlation?

The test uses a simplified lunar ephemeris (Meeus algorithm) for the
absolute phase, and the longitude-only prediction for the cross-station
test (which does not require ephemeris knowledge).

Output: results/outputs/step_165_gravitational_phase.json
"""
from __future__ import annotations
import sys, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_165_gravitational_phase.json"

DT = 300.0
N_DAYS = 30

# Tidal frequencies (Hz)
M2_FREQ  = 1.0 / (12.42 * 3600.0)
O1_FREQ  = 1.0 / (25.82 * 3600.0)
S2_FREQ  = 2.0 / 86400.0
S1_FREQ  = 1.0 / 86400.0
K1_FREQ  = 1.0 / (23.93 * 3600.0)
P1_FREQ  = 1.0 / (24.07 * 3600.0)

ALL_FREQS = [M2_FREQ, O1_FREQ, S2_FREQ, S1_FREQ, K1_FREQ, P1_FREQ]
ALL_NAMES = ["M2", "O1", "S2", "S1", "K1", "P1"]


def moon_position_jd(jd):
    """Simplified lunar position (Meeus, Ch. 47).

    Returns (right_ascension_deg, declination_deg) of the moon.
    Accurate to ~0.3 deg — sufficient for tidal phase prediction.
    """
    T = (jd - 2451545.0) / 36525.0

    # Mean longitude
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T**2
    # Mean elongation
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2
    # Mean anomaly of moon
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2
    # Mean anomaly of sun
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2
    # Argument of latitude
    F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T**2

    # Convert to radians
    Lp_r = np.radians(Lp % 360)
    D_r = np.radians(D % 360)
    M_r = np.radians(M % 360)
    Mp_r = np.radians(Mp % 360)
    F_r = np.radians(F % 360)

    # Longitude of the moon (simplified — main terms only)
    lon = Lp + 6.289 * np.sin(Mp_r) - 1.274 * np.sin(Mp_r - 2*D_r) \
          + 0.658 * np.sin(2*D_r) - 0.186 * np.sin(M_r) \
          - 0.059 * np.sin(2*Mp_r - 2*D_r) - 0.057 * np.sin(Mp_r - 2*D_r + M_r) \
          + 0.053 * np.sin(Mp_r + 2*D_r) + 0.046 * np.sin(2*D_r - M_r) \
          + 0.041 * np.sin(Mp_r - M_r) - 0.035 * np.sin(D_r) \
          - 0.031 * np.sin(Mp_r + M_r) - 0.015 * np.sin(2*F_r - 2*D_r)

    # Declination (simplified)
    lat = 5.128 * np.sin(F_r) + 0.281 * np.sin(Mp_r + F_r) \
          - 0.278 * np.sin(Mp_r - F_r) + 0.173 * np.sin(2*D_r - F_r)

    # Nutation in longitude (simplified)
    dpsi = -17.2 * np.sin(Mp_r) / 3600.0  # degrees

    # Apparent longitude
    lon_app = lon + dpsi

    # Right ascension and declination
    lon_r = np.radians(lon_app % 360)
    lat_r = np.radians(lat)
    eps = np.radians(23.4393 - 0.01300 * T)

    ra = np.degrees(np.arctan2(
        np.sin(lon_r) * np.cos(eps) - np.tan(lat_r) * np.sin(eps),
        np.cos(lon_r))) % 360
    dec = np.degrees(np.arcsin(
        np.sin(lat_r) * np.cos(eps) + np.cos(lat_r) * np.sin(eps) * np.sin(lon_r)))

    return ra, dec


def gmst_degrees(jd):
    """Greenwich Mean Sidereal Time in degrees (Meeus, Ch. 12)."""
    T = (jd - 2451545.0) / 36525.0
    # GMST at 0h UT
    gmst0 = 280.46061837 + 360.98564736629 * (jd - 2451545.0) \
            + 0.000387933 * T**2 - T**3 / 38710000.0
    return gmst0 % 360.0


def datetime_to_jd(dt_utc):
    """Convert datetime to Julian Day."""
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.replace(tzinfo=None)
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jd = dt_utc.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd += (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    return jd


def expected_m2_phase_deg(lon_deg, jd_t0):
    """Expected M2 phase at a station for the harmonic fit.

    The harmonic fit models:
        resid ~ A*cos(2*pi*M2_freq*t) + B*sin(2*pi*M2_freq*t)
        = amp * cos(2*pi*M2_freq*t - phase)

    The gravitational potential is:
        V ~ cos(2*(H_moon(t) + lambda))
        = cos(2*(GMST(t) - RA_moon(t) + lambda))

    At t=0:
        V(0) ~ cos(2*(GMST(0) - RA_moon(0) + lambda))

    The phase of this cosine (in the convention cos(omega*t - phase)) is:
        phase_grav = -2*(GMST(0) - RA_moon(0) + lambda)  [in radians]

    But we need to be careful about the convention. The harmonic fit gives:
        A*cos(omega*t) + B*sin(omega*t) = amp*cos(omega*t - arctan2(B,A))

    So phase = arctan2(B, A) is the phase lag.

    The gravitational signal is:
        V(t) = cos(omega*t + phi_grav)
        = cos(phi_grav)*cos(omega*t) - sin(phi_grav)*sin(omega*t)

    So A = cos(phi_grav), B = -sin(phi_grav), and:
        phase_obs = arctan2(B, A) = arctan2(-sin(phi_grav), cos(phi_grav)) = -phi_grav

    Therefore:
        phase_obs = -phi_grav = 2*(GMST(0) - RA_moon(0) + lambda)

    In degrees:
        phase_obs_deg = 2*(GMST_deg - RA_moon_deg + lon_deg) mod 360
    """
    gmst = gmst_degrees(jd_t0)
    ra_moon, dec_moon = moon_position_jd(jd_t0)
    # Phase in the harmonic fit convention:
    # V(t) = cos(omega*t + phi_grav) where phi_grav = 2*(H+lambda)
    # Harmonic fit: A*cos(omega*t) + B*sin(omega*t) = amp*cos(omega*t - phase)
    # A = cos(phi_grav), B = -sin(phi_grav) => phase = -phi_grav
    phase_deg = (-2.0 * (gmst - ra_moon + lon_deg)) % 360.0
    return phase_deg


def harmonic_fit(resid, t_sec, freqs):
    """Simultaneous harmonic fit."""
    t0 = t_sec[0]
    t_c = t_sec - t0 - (t_sec[-1] - t_sec[0]) / 2.0
    t_s = (t_c[-1] - t_c[0]) / 2.0 if len(t_c) > 1 else 1.0
    t_n = t_c / t_s
    cols = [np.ones_like(t_n), t_n, t_n ** 2]
    for f in freqs:
        omega = 2.0 * np.pi * f * (t_sec - t0)
        cols.append(np.cos(omega))
        cols.append(np.sin(omega))
    design = np.column_stack(cols)
    coef, _, _, _ = lstsq(design, resid, cond=1e-12)
    resid_fit = resid - design @ coef
    noise_std = float(np.std(resid_fit))
    n = len(resid)
    results = []
    for i, f in enumerate(freqs):
        A = float(coef[3 + 2 * i])
        B = float(coef[3 + 2 * i + 1])
        amp = float(np.sqrt(A ** 2 + B ** 2))
        phase_rad = float(np.arctan2(B, A))
        phase_deg = float(np.degrees(phase_rad)) % 360.0
        snr = amp / (noise_std / np.sqrt(n / 2.0)) if noise_std > 0 else 0.0
        results.append({"amp": amp, "phase_rad": phase_rad, "phase_deg": phase_deg,
                        "snr": snr, "A": A, "B": B})
    return results, noise_std


def find_best_window(product, year, n_days):
    """Find the best n-day contiguous window."""
    available = []
    for doy in range(1, 367):
        tag = f'{year}{doy:03d}'
        s = td.load_series(product, tag, min_valid_frac=0.0)
        if s:
            available.append(doy)
    if not available:
        return None, 0
    # Find contiguous runs
    runs = []
    start = available[0]
    prev = available[0]
    for d in available[1:]:
        if d == prev + 1:
            prev = d
        else:
            runs.append((start, prev))
            start = d
            prev = d
    runs.append((start, prev))

    best_count = 0
    best_start = None
    for r_start, r_end in runs:
        if r_end - r_start + 1 < n_days:
            continue
        for start in range(r_start, r_end - n_days + 2):
            common = None
            for doy in range(start, start + n_days):
                tag = f'{year}{doy:03d}'
                s = td.load_series(product, tag, min_valid_frac=0.0)
                if not s:
                    common = None
                    break
                stations = set(s.keys())
                common = stations if common is None else common & stations
            n = len(common) if common else 0
            if n > best_count:
                best_count = n
                best_start = start
    return best_start, best_count


def load_station(st, doys, year):
    """Load clock residual for a station across multiple days."""
    all_resid = []
    all_t = []
    station_info = None
    for doy in doys:
        tag = f'{year}{doy:03d}'
        s = td.load_series('clk_cod', tag, min_valid_frac=0.0)
        if not s or st not in s:
            return None
        d = s[st]
        r = np.asarray(d['resid'], float)
        t = np.asarray(d['t'], float) + (doy - 1) * 86400.0
        all_resid.append(r)
        all_t.append(t)
        if station_info is None:
            station_info = d['station']
    resid = np.concatenate(all_resid)
    t = np.concatenate(all_t)
    good = np.isfinite(resid)
    if good.mean() < 0.8:
        return None
    if not good.all():
        ix = np.arange(len(resid))
        resid = np.interp(ix, ix[good], resid[good])
    # Detrend
    tc = t - t[0]
    c = np.polyfit(tc, resid, 2)
    resid = resid - np.polyval(c, tc)
    return resid, t, station_info


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 165: Gravitational phase prediction — M2 clock phase vs lunar potential", flush=True)
    print(f"  Product: clk_cod, Year: 2025, Window: {N_DAYS} days", flush=True)
    print(flush=True)

    # Find best window
    print("  Finding best window...", flush=True)
    best_start, best_count = find_best_window('clk_cod', 2025, N_DAYS)
    if best_start is None:
        OUT.write_text(json.dumps({"step": 165, "error": "no_window"}))
        print("  ERROR: no suitable window")
        return
    doys = list(range(best_start, best_start + N_DAYS))
    print(f"  Best window: DOY {best_start}-{best_start + N_DAYS - 1}, {best_count} stations", flush=True)

    # Get common stations
    common = None
    for doy in doys:
        tag = f'2025{doy:03d}'
        s = td.load_series('clk_cod', tag, min_valid_frac=0.0)
        if not s: continue
        stations = set(s.keys())
        common = stations if common is None else common & stations
    common = sorted(common)
    print(f"  Common stations: {len(common)}", flush=True)

    # Load all stations
    print("  Loading station data...", flush=True)
    stations_data = {}
    for st in common:
        result = load_station(st, doys, 2025)
        if result is None:
            continue
        resid, t, info = result
        if np.std(resid) > 50:
            continue
        stations_data[st] = (resid, t, info)
    print(f"  Loaded: {len(stations_data)} stations", flush=True)

    # Compute JD at t=0 (start of observation)
    # DOY best_start, 2025, t=0 is 00:00 UTC
    t0_utc = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=best_start - 1)
    jd_t0 = datetime_to_jd(t0_utc)
    gmst0 = gmst_degrees(jd_t0)
    ra_moon, dec_moon = moon_position_jd(jd_t0)
    print(f"  Observation start: {t0_utc.isoformat()}", flush=True)
    print(f"  JD: {jd_t0:.6f}", flush=True)
    print(f"  GMST: {gmst0:.4f} deg", flush=True)
    print(f"  Moon RA: {ra_moon:.4f} deg, Dec: {dec_moon:.4f} deg", flush=True)
    print(flush=True)

    # Harmonic fit
    print("  Harmonic fit: M2, O1, S2, S1, K1, P1", flush=True)
    all_results = {}
    for st in sorted(stations_data.keys()):
        resid, t, info = stations_data[st]
        results, noise = harmonic_fit(resid, t, ALL_FREQS)
        lon = info.lon_deg
        lat = info.lat_deg
        # Expected M2 phase from gravitational potential
        expected_m2 = expected_m2_phase_deg(lon, jd_t0)
        all_results[st] = {
            "results": results,
            "lon": lon,
            "lat": lat,
            "noise": noise,
            "expected_m2_phase_deg": expected_m2,
        }

    # === M2 absolute phase test ===
    print("\n  === M2 ABSOLUTE PHASE TEST ===", flush=True)
    observed_m2_phases = []
    expected_m2_phases = []
    m2_amps = []
    m2_snrs = []
    station_names = []
    for st in sorted(all_results.keys()):
        r = all_results[st]
        obs_phase = r["results"][0]["phase_deg"]  # M2 is index 0
        exp_phase = r["expected_m2_phase_deg"]
        amp = r["results"][0]["amp"]
        snr = r["results"][0]["snr"]
        observed_m2_phases.append(obs_phase)
        expected_m2_phases.append(exp_phase)
        m2_amps.append(amp)
        m2_snrs.append(snr)
        station_names.append(st)

    observed_m2_phases = np.array(observed_m2_phases)
    expected_m2_phases = np.array(expected_m2_phases)
    m2_amps = np.array(m2_amps)
    m2_snrs = np.array(m2_snrs)

    # Phase residual (circular)
    phase_resid = ((observed_m2_phases - expected_m2_phases + 180) % 360) - 180

    # Circular statistics
    phase_resid_rad = np.radians(phase_resid)
    mean_resid_rad = np.angle(np.mean(np.exp(1j * phase_resid_rad)))
    mean_resid_deg = np.degrees(mean_resid_rad)
    # Concentration parameter
    R = np.abs(np.mean(np.exp(1j * phase_resid_rad)))
    n = len(phase_resid_rad)
    # Rayleigh test for circular uniformity
    rayleigh_z = n * R**2
    rayleigh_p = np.exp(-rayleigh_z) * (1 + (2 * n - rayleigh_z) / (4 * n))

    print(f"  N stations: {n}", flush=True)
    print(f"  Mean phase residual: {mean_resid_deg:+.2f} deg", flush=True)
    print(f"  Concentration R: {R:.4f}", flush=True)
    print(f"  Rayleigh z: {rayleigh_z:.2f}", flush=True)
    print(f"  Rayleigh p: {rayleigh_p:.6f}", flush=True)
    print(f"  Phase residual std: {np.std(phase_resid):.2f} deg", flush=True)
    print(f"  Median |phase residual|: {np.median(np.abs(phase_resid)):.2f} deg", flush=True)
    print(flush=True)

    # Filter to high-SNR stations
    high_snr = m2_snrs > 3.0
    if high_snr.sum() >= 10:
        phase_resid_hsnr = phase_resid[high_snr]
        phase_resid_hsnr_rad = np.radians(phase_resid_hsnr)
        R_hsnr = np.abs(np.mean(np.exp(1j * phase_resid_hsnr_rad)))
        n_hsnr = high_snr.sum()
        rayleigh_z_hsnr = n_hsnr * R_hsnr**2
        rayleigh_p_hsnr = np.exp(-rayleigh_z_hsnr)
        print(f"  High-SNR (SNR>3): {n_hsnr} stations", flush=True)
        print(f"  Mean phase residual: {np.degrees(np.angle(np.mean(np.exp(1j * phase_resid_hsnr_rad)))):+.2f} deg", flush=True)
        print(f"  Concentration R: {R_hsnr:.4f}", flush=True)
        print(f"  Rayleigh p: {rayleigh_p_hsnr:.6f}", flush=True)
        print(f"  Median |phase residual|: {np.median(np.abs(phase_resid_hsnr)):.2f} deg", flush=True)
        print(flush=True)

    # === M2 cross-station phase prediction (longitude-only) ===
    print("  === M2 CROSS-STATION PHASE PREDICTION (longitude-only) ===", flush=True)
    st_list = sorted(all_results.keys())
    expected_corrs_lon = []
    actual_corrs_lon = []
    n_pairs = 0
    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            r1, r2 = all_results[s1], all_results[s2]
            d1, d2 = stations_data[s1][0], stations_data[s2][0]
            n = min(len(d1), len(d2))
            if n < 100:
                continue
            # Expected phase difference from longitude only
            dlon = r1["lon"] - r2["lon"]
            expected_dphase = 2.0 * dlon  # degrees
            expected_corr = np.cos(np.radians(expected_dphase))

            # Bandpass filter at M2
            freqs = np.fft.fftfreq(n, DT)
            X1 = np.fft.fft(d1[:n])
            X2 = np.fft.fft(d2[:n])
            mask = (np.abs(freqs) >= M2_FREQ - 0.5e-6) & (np.abs(freqs) <= M2_FREQ + 0.5e-6)
            X1[~mask] = 0
            X2[~mask] = 0
            bp1 = np.real(np.fft.ifft(X1))
            bp2 = np.real(np.fft.ifft(X2))
            v1 = bp1 - bp1.mean()
            v2 = bp2 - bp2.mean()
            if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
                continue
            actual_corr = float(np.corrcoef(v1, v2)[0, 1])
            if np.isfinite(actual_corr):
                expected_corrs_lon.append(expected_corr)
                actual_corrs_lon.append(actual_corr)
                n_pairs += 1

    if n_pairs > 5:
        r_lon, p_lon = stats.pearsonr(expected_corrs_lon, actual_corrs_lon)
        print(f"  N pairs: {n_pairs}", flush=True)
        print(f"  r (longitude prediction): {r_lon:.4f}", flush=True)
        print(f"  p: {p_lon:.6f}", flush=True)
        print(f"  Mean actual corr: {np.mean(actual_corrs_lon):.4f}", flush=True)
    else:
        r_lon, p_lon = None, None
        print(f"  Insufficient pairs: {n_pairs}", flush=True)
    print(flush=True)

    # === Same test for S2 (solar — should also show longitude dependence) ===
    print("  === S2 CROSS-STATION PHASE PREDICTION (longitude-only) ===", flush=True)
    # S2 is solar: phase = 2*(GMST_solar + lon) = 2*(0 + lon) = 2*lon
    # (at t=0, GMST for solar is 0 by convention — the sun is at Greenwich)
    # Actually, S2 phase = 2*(hour_angle_sun + lon) = 2*(GMST - RA_sun + lon)
    # RA_sun at DOY best_start: approximately (best_start/365)*360
    ra_sun = (best_start / 365.25) * 360.0  # approximate solar RA
    expected_corrs_s2 = []
    actual_corrs_s2 = []
    n_pairs_s2 = 0
    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            r1, r2 = all_results[s1], all_results[s2]
            d1, d2 = stations_data[s1][0], stations_data[s2][0]
            n = min(len(d1), len(d2))
            if n < 100:
                continue
            dlon = r1["lon"] - r2["lon"]
            expected_corr = np.cos(np.radians(2.0 * dlon))

            freqs = np.fft.fftfreq(n, DT)
            X1 = np.fft.fft(d1[:n])
            X2 = np.fft.fft(d2[:n])
            mask = (np.abs(freqs) >= S2_FREQ - 0.5e-6) & (np.abs(freqs) <= S2_FREQ + 0.5e-6)
            X1[~mask] = 0
            X2[~mask] = 0
            bp1 = np.real(np.fft.ifft(X1))
            bp2 = np.real(np.fft.ifft(X2))
            v1 = bp1 - bp1.mean()
            v2 = bp2 - bp2.mean()
            if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
                continue
            actual_corr = float(np.corrcoef(v1, v2)[0, 1])
            if np.isfinite(actual_corr):
                expected_corrs_s2.append(expected_corr)
                actual_corrs_s2.append(actual_corr)
                n_pairs_s2 += 1

    if n_pairs_s2 > 5:
        r_s2, p_s2 = stats.pearsonr(expected_corrs_s2, actual_corrs_s2)
        print(f"  N pairs: {n_pairs_s2}", flush=True)
        print(f"  r (longitude prediction): {r_s2:.4f}", flush=True)
        print(f"  p: {p_s2:.6f}", flush=True)
    else:
        r_s2, p_s2 = None, None
    print(flush=True)

    # === Control: off-tide frequency should show NO longitude dependence ===
    print("  === CONTROL: off-tide frequency (15 uHz) ===", flush=True)
    control_freq = 15.0e-6  # 15 uHz, not a tidal frequency
    expected_corrs_ctrl = []
    actual_corrs_ctrl = []
    n_pairs_ctrl = 0
    for i in range(len(st_list)):
        for j in range(i + 1, len(st_list)):
            s1, s2 = st_list[i], st_list[j]
            r1, r2 = all_results[s1], all_results[s2]
            d1, d2 = stations_data[s1][0], stations_data[s2][0]
            n = min(len(d1), len(d2))
            if n < 100:
                continue
            dlon = r1["lon"] - r2["lon"]
            expected_corr = np.cos(np.radians(2.0 * dlon))

            freqs = np.fft.fftfreq(n, DT)
            X1 = np.fft.fft(d1[:n])
            X2 = np.fft.fft(d2[:n])
            mask = (np.abs(freqs) >= control_freq - 0.5e-6) & (np.abs(freqs) <= control_freq + 0.5e-6)
            X1[~mask] = 0
            X2[~mask] = 0
            bp1 = np.real(np.fft.ifft(X1))
            bp2 = np.real(np.fft.ifft(X2))
            v1 = bp1 - bp1.mean()
            v2 = bp2 - bp2.mean()
            if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
                continue
            actual_corr = float(np.corrcoef(v1, v2)[0, 1])
            if np.isfinite(actual_corr):
                expected_corrs_ctrl.append(expected_corr)
                actual_corrs_ctrl.append(actual_corr)
                n_pairs_ctrl += 1

    if n_pairs_ctrl > 5:
        r_ctrl, p_ctrl = stats.pearsonr(expected_corrs_ctrl, actual_corrs_ctrl)
        print(f"  N pairs: {n_pairs_ctrl}", flush=True)
        print(f"  r (longitude prediction): {r_ctrl:.4f}", flush=True)
        print(f"  p: {p_ctrl:.6f}", flush=True)
    else:
        r_ctrl, p_ctrl = None, None
    print(flush=True)

    # === Build output ===
    per_station = {}
    for st in sorted(all_results.keys()):
        r = all_results[st]
        per_station[st] = {
            "lon_deg": r["lon"],
            "lat_deg": r["lat"],
            "M2_observed_phase_deg": r["results"][0]["phase_deg"],
            "M2_expected_phase_deg": r["expected_m2_phase_deg"],
            "M2_phase_residual_deg": ((r["results"][0]["phase_deg"] - r["expected_m2_phase_deg"] + 180) % 360) - 180,
            "M2_amp_ns": r["results"][0]["amp"],
            "M2_snr": r["results"][0]["snr"],
        }

    verdict = {
        "step": 165,
        "title": "Gravitational phase prediction: M2 clock phase vs lunar potential",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": "clk_cod",
        "year": 2025,
        "window_doy": [best_start, best_start + N_DAYS - 1],
        "n_days": N_DAYS,
        "n_stations": len(all_results),
        "jd_t0": jd_t0,
        "gmst_deg": gmst0,
        "moon_ra_deg": ra_moon,
        "moon_dec_deg": dec_moon,
        "per_station": per_station,
        "m2_absolute_phase": {
            "n_stations": n,
            "mean_phase_residual_deg": mean_resid_deg,
            "concentration_R": R,
            "rayleigh_z": rayleigh_z,
            "rayleigh_p": rayleigh_p,
            "phase_residual_std_deg": float(np.std(phase_resid)),
            "median_abs_phase_residual_deg": float(np.median(np.abs(phase_resid))),
            "n_high_snr": int(high_snr.sum()),
            "high_snr_concentration_R": float(R_hsnr) if high_snr.sum() >= 10 else None,
            "high_snr_rayleigh_p": float(rayleigh_p_hsnr) if high_snr.sum() >= 10 else None,
            "high_snr_median_abs_residual": float(np.median(np.abs(phase_resid_hsnr))) if high_snr.sum() >= 10 else None,
        },
        "m2_cross_station_longitude": {
            "n_pairs": n_pairs,
            "r": float(r_lon) if r_lon else None,
            "p": float(p_lon) if p_lon else None,
            "mean_actual_corr": float(np.mean(actual_corrs_lon)) if actual_corrs_lon else None,
        },
        "s2_cross_station_longitude": {
            "n_pairs": n_pairs_s2,
            "r": float(r_s2) if r_s2 else None,
            "p": float(p_s2) if p_s2 else None,
        },
        "control_15uhz": {
            "n_pairs": n_pairs_ctrl,
            "r": float(r_ctrl) if r_ctrl else None,
            "p": float(p_ctrl) if p_ctrl else None,
        },
        "interpretation": [
            f"M2 absolute phase: {n} stations, mean residual {mean_resid_deg:+.2f} deg, "
            f"R={R:.4f}, Rayleigh p={rayleigh_p:.6f}.",
            f"M2 cross-station longitude prediction: r={r_lon:.4f}, p={p_lon:.6f}, n={n_pairs} pairs."
            if r_lon else "M2 cross-station: insufficient pairs.",
            f"S2 cross-station longitude prediction: r={r_s2:.4f}, p={p_s2:.6f}, n={n_pairs_s2} pairs."
            if r_s2 else "S2 cross-station: insufficient pairs.",
            f"Control (15 uHz): r={r_ctrl:.4f}, p={p_ctrl:.6f}, n={n_pairs_ctrl} pairs."
            if r_ctrl else "Control: insufficient pairs.",
        ],
        "pre_registered_prediction": (
            "TEP (gravitational): M2 clock phase matches the lunar gravitational potential phase. "
            "The phase difference between stations is 2*(lon1-lon2), purely longitude-dependent. "
            "The absolute phase matches the lunar ephemeris. "
            "The control frequency (15 uHz) shows NO longitude dependence. "
            "S2 also shows longitude dependence (solar gravitational + atmospheric). "
            "If the M2 phase matches the gravitational prediction and the control does not, "
            "the signal is definitively gravitational."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
