#!/usr/bin/env python3
"""
Deterministic astronomical and geophysical feature library for TEP-PPP.

All features computed here are PREDETERMINED — their times are known in
advance from JPL ephemeris or known orbital mechanics.  This module provides
the forward-prediction covariates that TEP-GNSS (Paper 1) and TEP-GNSS-II
(Paper 2) established as real signal in the GNSS clock correlation field:

  - Planetary alignment events (oppositions / conjunctions)
    Paper 2: 56/156 significant, 19 Bonferroni survivors, 5.5x effect vs null
  - Semiannual nutation (182.625 d, R^2 = 0.904, p = 2.7e-5)
  - 18.6-year lunar nodal nutation (R^2 = 0.641, p = 0.010)
  - Chandler wobble (~433 d, R^2 = 0.377-0.471)
  - Lunar phase (synodic 29.53 d), anomalistic (27.55 d), draconic (27.21 d)
  - Orbital velocity direction (r = -0.888, surrogate p < 2e-7)
  - CMB dipole direction (r = 0.747, 10300x variance ratio vs Solar Apex)
  - Eclipse windows (Paper 1: type hierarchy Partial > Annular > Total)

Every function takes a UTC timestamp (epoch seconds or datetime) and returns
a dimensionless phase, direction, or proximity measure.  No fitting is done
here — the periods and directions are fixed by astronomy.  The caller
decides how to use them (regressor, anisotropy axis, integrity flag, etc.).

Ephemeris: JPL DE432s via jplephem.  The file is expected at the project
root (de432s.bsp), symlinked from TEP-GNSS.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants — all from astronomy, not fitted
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EPHEMERIS_PATH = _PROJECT_ROOT / "de432s.bsp"

# JPL body codes
_SUN = 10
_MERCURY = 1
_VENUS = 2
_EARTH = 3
_MARS = 4
_JUPITER = 5
_SATURN = 6
_MOON = 301

# Periods (days) — astronomical constants
SOLAR_YEAR_D = 365.25
SEMIANNUAL_D = SOLAR_YEAR_D / 2.0          # 182.625
LUNAR_NODAL_D = 6798.38                     # 18.6 years
CHANDLER_D = 433.0                          # ~14 months (Paper 1: 433 d)
SYNODIC_MONTH_D = 29.530589                  # lunar phase
ANOMALISTIC_MONTH_D = 27.55455               # perigee-apogee
DRACONIC_MONTH_D = 27.212221                 # nodal crossing
SIDEREAL_DAY_D = 0.9972695663                # 23h 56m 4s (Earth rotation vs stars)
SOLAR_DAY_D = 1.0                            # 24h (Earth rotation vs Sun)

# Gravitational constants for tidal / planetary gravity computations
_G = 6.67430e-11                             # m^3 kg^-1 s^-2 (CODATA 2018)
_M_SUN_KG = 1.98892e30
_M_MERCURY_KG = 3.30104e23
_M_VENUS_KG = 4.86732e24
_M_EARTH_KG = 5.97217e24
_M_MARS_KG = 6.41693e23
_M_JUPITER_KG = 1.89813e27
_M_SATURN_KG = 5.68317e26
_M_MOON_KG = 7.342e22
_AU_M = 1.495978707e11

# Planet codes + masses for stacked gravity
_PLANETS = [
    (_MERCURY, "Mercury", _M_MERCURY_KG),
    (_VENUS, "Venus", _M_VENUS_KG),
    (_MARS, "Mars", _M_MARS_KG),
    (_JUPITER, "Jupiter", _M_JUPITER_KG),
    (_SATURN, "Saturn", _M_SATURN_KG),
]

# CMB dipole direction (Planck 2018)
CMB_DIPOLE_RA_DEG = 167.94
CMB_DIPOLE_DEC_DEG = -6.94

# Obliquity of the ecliptic
OBLIQUITY_DEG = 23.44

# J2000 epoch
_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_J2000_JD = 2451545.0

# Planetary event significance from Paper 2 (Bonferroni survivors)
# These are the 19 events that survived family-wide correction.
# We compute proximity to ALL events dynamically; the significance list
# is used to weight the integrity flag.
_SIGNIFICANT_PLANETS = ("Mars", "Jupiter", "Saturn", "Venus", "Mercury")

# Ephemeris cache
_ephem = None


def _load_ephem():
    global _ephem
    if _ephem is None:
        from jplephem.spk import SPK
        if not _EPHEMERIS_PATH.exists():
            raise FileNotFoundError(f"Ephemeris not found: {_EPHEMERIS_PATH}")
        _ephem = SPK.open(str(_EPHEMERIS_PATH))
    return _ephem


def _to_jd(t):
    """Convert epoch seconds or datetime to Julian Date."""
    if isinstance(t, (int, float)):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
    elif isinstance(t, datetime):
        dt = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    else:
        dt = t
    delta = dt - _J2000
    return _J2000_JD + delta.total_seconds() / 86400.0


def _to_datetime(t):
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(t, tz=timezone.utc)
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    return t


# ---------------------------------------------------------------------------
# Phase computations — all deterministic
# ---------------------------------------------------------------------------

def nutation_phase_semiannual(t) -> float:
    """Phase of the semiannual nutation cycle (radians, [0, 2pi)).

    Period = 182.625 days.  Phase 0 at J2000.
    Paper 2: R^2 = 0.904, p = 2.7e-5.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) / SEMIANNUAL_D)
    return phase % (2.0 * math.pi)


def nutation_phase_lunar_nodal(t) -> float:
    """Phase of the 18.6-year lunar nodal nutation (radians, [0, 2pi)).

    Period = 6798.38 days.  Phase 0 at J2000.
    Paper 2: R^2 = 0.641, p = 0.010.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) / LUNAR_NODAL_D)
    return phase % (2.0 * math.pi)


def chandler_phase(t) -> float:
    """Phase of the Chandler wobble (radians, [0, 2pi)).

    Period = 433 days.  Phase 0 at J2000.
    Paper 1: R^2 = 0.377-0.471, p < 0.01.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) / CHANDLER_D)
    return phase % (2.0 * math.pi)


def lunar_phase_synodic(t) -> float:
    """Synodic lunar phase (radians, [0, 2pi)).

    0 = new moon, pi = full moon.
    Period = 29.530589 days.  Reference: new moon at J2000.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) % SYNODIC_MONTH_D / SYNODIC_MONTH_D)
    return phase % (2.0 * math.pi)


def lunar_phase_anomalistic(t) -> float:
    """Anomalistic lunar phase — perigee/apogee cycle (radians, [0, 2pi)).

    0 = perigee, pi = apogee.
    Period = 27.55455 days.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) % ANOMALISTIC_MONTH_D / ANOMALISTIC_MONTH_D)
    return phase % (2.0 * math.pi)


def lunar_phase_draconic(t) -> float:
    """Draconic lunar phase — nodal crossing cycle (radians, [0, 2pi)).

    0 = ascending node crossing.
    Period = 27.212221 days.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) % DRACONIC_MONTH_D / DRACONIC_MONTH_D)
    return phase % (2.0 * math.pi)


def diurnal_phase_sidereal(t) -> float:
    """Sidereal day phase — Earth rotation relative to the fixed stars
    (radians, [0, 2pi)).

    Period = 0.9972695663 days (23h 56m 4s).  Phase 0 at J2000.
    This is the fundamental diurnal cycle for GNSS: the satellite geometry
    repeats every sidereal day, and any diurnal modulation of the clock
    correlation field (e.g. temperature, ionosphere, Earth-rotation-induced
    proper-time effects) is locked to this period.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) / SIDEREAL_DAY_D)
    return phase % (2.0 * math.pi)


def diurnal_phase_solar(t) -> float:
    """Solar day phase — Earth rotation relative to the Sun (radians, [0, 2pi)).

    Period = 1.0 day (24h).  Phase 0 at J2000 noon.
    This captures the day/night cycle: temperature, ionospheric diurnal
    variation, and solar radiation pressure effects all lock to the solar
    day rather than the sidereal day.
    """
    jd = _to_jd(t)
    phase = 2.0 * math.pi * ((jd - _J2000_JD) / SOLAR_DAY_D)
    return phase % (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Direction computations — for anisotropic kriging
# ---------------------------------------------------------------------------

def _gmst_deg(jd: float) -> float:
    """Greenwich Mean Sidereal Time in degrees (IAU 1982 polynomial)."""
    t = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t - t ** 3 / 38710000.0)
    return gmst % 360.0


def orbital_velocity_azimuth_deg(t, lat_deg: float = 0.0) -> float:
    """Geographic azimuth (deg from north) of Earth's orbital velocity
    projected onto the equatorial plane, at time *t*.

    The orbital velocity is perpendicular to the Sun-Earth direction in the
    ecliptic.  Paper 2: r = -0.888 between EW/NS anisotropy ratio and
    heliocentric velocity, surrogate p < 2e-7.

    Returns the equatorial projection azimuth at the equator.
    At higher latitudes the effective azimuth differs, but for a global
    anisotropy axis (one stretch factor for the entire network) the
    equatorial projection is the correct effective direction.
    """
    jd = _to_jd(t)
    # Sun's approximate ecliptic longitude: spring equinox ~ DOY 80
    dt = datetime.fromtimestamp(
        (jd - _J2000_JD) * 86400.0 + _J2000.timestamp(),
        tz=timezone.utc)
    doy = dt.timetuple().tm_yday
    lam_sun = ((doy - 80) / SOLAR_YEAR_D) * 360.0
    lam_vel = (lam_sun + 90.0) % 360.0  # velocity ⊥ Sun-Earth in ecliptic
    # Convert ecliptic longitude to equatorial RA
    ra = math.degrees(math.atan2(
        math.sin(math.radians(lam_vel)) * math.cos(math.radians(OBLIQUITY_DEG)),
        math.cos(math.radians(lam_vel))
    )) % 360.0
    gmst = _gmst_deg(jd)
    return (ra - gmst) % 360.0


def cmb_azimuth_deg(t) -> float:
    """Geographic azimuth (deg from north) of the CMB dipole direction
    projected onto the equatorial plane, at time *t*.

    Paper 2: r = 0.747, 10300x variance ratio vs Solar Apex.
    CMB dipole: RA = 167.94°, Dec = -6.94° (Planck 2018).
    """
    jd = _to_jd(t)
    gmst = _gmst_deg(jd)
    return (CMB_DIPOLE_RA_DEG - gmst) % 360.0


def orbital_velocity_vector_ecef(t) -> np.ndarray:
    """Unit vector of Earth's heliocentric orbital velocity in ECEF coordinates.

    Computed from JPL ephemeris: v = d/dt[Earth_bary - Sun_bary].
    Uses a finite-difference over 1 hour for the velocity.
    """
    jd = _to_jd(t)
    ephem = _load_ephem()
    dt = 1.0 / 24.0  # 1 hour
    # Earth-Moon barycenter + Earth relative to EMB
    emb1 = ephem[0, 3].compute(jd - dt)
    earth_rel1 = ephem[3, 399].compute(jd - dt)
    earth1 = emb1 + earth_rel1
    emb2 = ephem[0, 3].compute(jd + dt)
    earth_rel2 = ephem[3, 399].compute(jd + dt)
    earth2 = emb2 + earth_rel2
    sun1 = ephem[0, 10].compute(jd - dt)
    sun2 = ephem[0, 10].compute(jd + dt)
    v_helio = (earth2 - earth1) - (sun2 - sun1)
    # Rotate from ICRS to ITRS (ECEF) using GMST
    gmst_rad = math.radians(_gmst_deg(jd))
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    # Simple Z-rotation by GMST (approximate, neglects polar motion/nutation)
    R = np.array([[cos_g, sin_g, 0], [-sin_g, cos_g, 0], [0, 0, 1]])
    v_ecef = R @ v_helio
    norm = np.linalg.norm(v_ecef)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return v_ecef / norm


def cmb_dipole_vector_ecef(t) -> np.ndarray:
    """Unit vector of the CMB dipole direction in ECEF coordinates.

    CMB dipole: RA = 167.94°, Dec = -6.94° (ICRS).
    Rotated to ITRS by GMST.
    """
    jd = _to_jd(t)
    ra_rad = math.radians(CMB_DIPOLE_RA_DEG)
    dec_rad = math.radians(CMB_DIPOLE_DEC_DEG)
    v_icrs = np.array([
        math.cos(dec_rad) * math.cos(ra_rad),
        math.cos(dec_rad) * math.sin(ra_rad),
        math.sin(dec_rad)
    ])
    gmst_rad = math.radians(_gmst_deg(jd))
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    R = np.array([[cos_g, sin_g, 0], [-sin_g, cos_g, 0], [0, 0, 1]])
    v_ecef = R @ v_icrs
    norm = np.linalg.norm(v_ecef)
    return v_ecef / norm if norm > 1e-12 else v_icrs


# ---------------------------------------------------------------------------
# Planetary event proximity — from JPL ephemeris
# ---------------------------------------------------------------------------

def _planet_heliocentric(planet_code: int, jd: float) -> np.ndarray:
    """Heliocentric position (km) of a planet at JD."""
    ephem = _load_ephem()
    if planet_code == _EARTH:
        emb = ephem[0, 3].compute(jd)
        earth_rel = ephem[3, 399].compute(jd)
        planet_bary = emb + earth_rel
    else:
        planet_bary = ephem[0, planet_code].compute(jd)
    sun_bary = ephem[0, _SUN].compute(jd)
    return planet_bary - sun_bary


def _elongation_deg(planet_code: int, jd: float) -> float:
    """Sun-Earth-Planet elongation angle in degrees (0-180)."""
    earth_pos = _planet_heliocentric(_EARTH, jd)
    planet_pos = _planet_heliocentric(planet_code, jd)
    sun_from_earth = -earth_pos
    planet_from_earth = planet_pos - earth_pos
    sn = sun_from_earth / np.linalg.norm(sun_from_earth)
    pn = planet_from_earth / np.linalg.norm(planet_from_earth)
    cos_a = np.clip(np.dot(sn, pn), -1, 1)
    return math.degrees(math.acos(cos_a))


def _earth_planet_dist_au(planet_code: int, jd: float) -> float:
    """Earth-planet distance in AU."""
    earth_pos = _planet_heliocentric(_EARTH, jd)
    planet_pos = _planet_heliocentric(planet_code, jd)
    AU_KM = 149597870.7
    return np.linalg.norm(planet_pos - earth_pos) / AU_KM


def planetary_event_proximity(t, window_days: float = 120.0) -> dict:
    """Proximity to the nearest planetary alignment event for each planet.

    Returns a dict mapping planet name to:
        {
            "days_to_event": float,       # signed: negative = before, positive = after
            "in_window": bool,            # True if within ±window_days
            "event_type": str,            # "opposition" or "inferior_conjunction"
            "elongation_deg": float,      # current elongation
        }

    The event is found by scanning ±window_days around *t* for the
    minimum (inferior planets) or maximum (superior planets) elongation.

    Paper 2: 56/156 events significant, 19 Bonferroni survivors.
    Planetary events produce 5.5x larger coherence changes than random dates.
    """
    jd = _to_jd(t)
    result = {}

    # Inferior planets: conjunction = min elongation (inferior conjunction
    # when planet is between Earth and Sun, superior conjunction when behind
    # the Sun).  Both are astronomically significant — the elongation goes
    # to near zero at either conjunction.  We scan for the minimum elongation
    # without the dist < 1.0 restriction (which caused a bug when Venus was
    # near superior conjunction at ~1.7 AU, making the condition never true
    # and returning days_to_event = 0.0 for every day).
    for code, name in [(_MERCURY, "Mercury"), (_VENUS, "Venus")]:
        best_jd = jd
        best_elong = _elongation_deg(code, jd)
        # Scan ±window_days at 1-day resolution
        for d in np.arange(-window_days, window_days + 1, 1.0):
            test_jd = jd + d
            try:
                el = _elongation_deg(code, test_jd)
            except Exception:
                continue
            if el < best_elong:
                best_elong = el
                best_jd = test_jd
        # Classify: inferior conjunction if planet is closer than Earth-Sun,
        # superior conjunction if farther
        try:
            dist = _earth_planet_dist_au(code, best_jd)
            etype = "inferior_conjunction" if dist < 1.0 else "superior_conjunction"
        except Exception:
            etype = "conjunction"
        result[name] = {
            "days_to_event": float(best_jd - jd),
            "in_window": abs(best_jd - jd) <= window_days,
            "event_type": etype,
            "elongation_deg": float(_elongation_deg(code, jd)),
        }

    # Superior planets: opposition = max elongation (~180°)
    for code, name in [(_MARS, "Mars"), (_JUPITER, "Jupiter"), (_SATURN, "Saturn")]:
        best_jd = jd
        best_elong = _elongation_deg(code, jd)
        for d in np.arange(-window_days, window_days + 1, 1.0):
            test_jd = jd + d
            try:
                el = _elongation_deg(code, test_jd)
            except Exception:
                continue
            if el > best_elong:
                best_elong = el
                best_jd = test_jd
        result[name] = {
            "days_to_event": float(best_jd - jd),
            "in_window": abs(best_jd - jd) <= window_days,
            "event_type": "opposition",
            "elongation_deg": float(_elongation_deg(code, jd)),
        }

    return result


def nearest_significant_event_days(t, window_days: float = 30.0) -> float:
    """Days to the nearest planetary event (any planet), signed.

    Returns the minimum |days_to_event| across all planets, with the sign
    of the nearest one.  This is the scalar proximity used as an integrity
    flag: epochs near events (Paper 2: 5.5x effect) may have elevated
    prediction error.
    """
    prox = planetary_event_proximity(t, window_days=window_days)
    days = [v["days_to_event"] for v in prox.values()]
    if not days:
        return 999.0
    # Return the one with smallest absolute value
    return min(days, key=abs)


# ---------------------------------------------------------------------------
# Gravitational / tidal computations — from JPL ephemeris
# ---------------------------------------------------------------------------

def _earth_bary_pos(jd: float) -> np.ndarray:
    """Solar-system-barycentric position of Earth (km) at JD."""
    ephem = _load_ephem()
    emb = ephem[0, 3].compute(jd)
    earth_rel = ephem[3, 399].compute(jd)
    return emb + earth_rel


def _moon_earth_pos(jd: float) -> np.ndarray:
    """Geocentric position of the Moon (km) at JD."""
    ephem = _load_ephem()
    return ephem[3, 301].compute(jd)


def _sun_bary_pos(jd: float) -> np.ndarray:
    """Solar-system-barycentric position of the Sun (km) at JD."""
    ephem = _load_ephem()
    return ephem[0, 10].compute(jd)


def earth_sun_distance_au(t) -> float:
    """Earth-Sun distance in AU from JPL ephemeris.

    Modulates the solar tidal force (1/r^3) and the annual cycle.
    Perihelion ~0.983 AU (early January), aphelion ~1.017 AU (early July).
    """
    jd = _to_jd(t)
    earth = _earth_bary_pos(jd)
    sun = _sun_bary_pos(jd)
    return float(np.linalg.norm(earth - sun) / (_AU_M / 1000.0))


def earth_moon_distance_km(t) -> float:
    """Earth-Moon distance in km from JPL ephemeris.

    Modulates the lunar tidal force (1/r^3).  Perigee ~363,300 km,
    apogee ~405,500 km.  The tidal acceleration varies by ~40% over the
    anomalistic month.
    """
    jd = _to_jd(t)
    moon = _moon_earth_pos(jd)
    return float(np.linalg.norm(moon))


def solar_tidal_accel(t) -> float:
    """Solar tidal acceleration at Earth (m/s^2).

    The tidal acceleration is the differential gravitational field across
    the Earth: a_tidal = 2 * G * M_sun * R_earth / r^3, where R_earth is
    the Earth radius.  This is the stretching force that drives the
    solid-Earth tide and the dominant diurnal/semiannual modulation of
    the geoid.

    Returns the magnitude of the tidal acceleration (always positive).
    """
    jd = _to_jd(t)
    earth = _earth_bary_pos(jd) * 1000.0  # km -> m
    sun = _sun_bary_pos(jd) * 1000.0
    r = np.linalg.norm(earth - sun)
    if r < 1.0:
        return 0.0
    R_EARTH = 6.371e6  # m
    return float(2.0 * _G * _M_SUN_KG * R_EARTH / r ** 3)


def lunar_tidal_accel(t) -> float:
    """Lunar tidal acceleration at Earth (m/s^2).

    a_tidal = 2 * G * M_moon * R_earth / r^3.

    The lunar tidal force is ~2.2x the solar tidal force on average,
    but it varies by ~40% over the anomalistic month as the Moon moves
    between perigee and apogee.
    """
    jd = _to_jd(t)
    moon = _moon_earth_pos(jd) * 1000.0  # km -> m
    r = np.linalg.norm(moon)
    if r < 1.0:
        return 0.0
    R_EARTH = 6.371e6  # m
    return float(2.0 * _G * _M_MOON_KG * R_EARTH / r ** 3)


def total_tidal_accel(t) -> float:
    """Total (solar + lunar) tidal acceleration magnitude (m/s^2).

    This is the combined tidal stretching force on the Earth.  When Sun
    and Moon are aligned (new/full moon), the tides add constructively
    (spring tides); when they are perpendicular (quarter moon), they
    partially cancel (neap tides).  The total varies with both the
    synodic phase and the anomalistic distance.
    """
    return solar_tidal_accel(t) + lunar_tidal_accel(t)


def stacked_planetary_gravity(t) -> dict:
    """Stacked gravitational acceleration from all major planets on Earth.

    Computes the vector sum of gravitational accelerations from Mercury,
    Venus, Mars, Jupiter, and Saturn at the Earth's position, using JPL
    ephemeris positions and Newtonian gravity:

        a_i = G * M_i * (r_i - r_earth) / |r_i - r_earth|^3

    The "stacked" gravity is the vector sum of all five planetary pulls.
    Jupiter dominates (~3e-7 m/s^2), Venus is second (~2e-7 at closest),
    and the rest are smaller.  When multiple planets align (Paper 2:
    5.5x effect at planetary events), the stacked gravity vector sum
    can be significantly larger than the time-averaged value.

    Returns:
        {
            "accel_m_s2": float,         # magnitude of stacked acceleration
            "vector_m_s2": np.ndarray,   # 3D vector in ICRS (m/s^2)
            "azimuth_deg": float,        # geographic azimuth of projection
            "jupiter_frac": float,       # fraction from Jupiter alone
            "n_planets_above_mean": int, # how many exceed their time-average
        }
    """
    jd = _to_jd(t)
    earth = _earth_bary_pos(jd) * 1000.0  # km -> m
    total_vec = np.zeros(3)
    jupiter_vec = np.zeros(3)
    planet_mags = []

    for code, name, mass in _PLANETS:
        ephem = _load_ephem()
        planet_bary = ephem[0, code].compute(jd) * 1000.0  # km -> m
        dr = planet_bary - earth
        r = np.linalg.norm(dr)
        if r < 1.0:
            continue
        a_vec = _G * mass * dr / r ** 3
        total_vec += a_vec
        planet_mags.append(np.linalg.norm(a_vec))
        if name == "Jupiter":
            jupiter_vec = a_vec

    total_mag = float(np.linalg.norm(total_vec))
    jupiter_mag = float(np.linalg.norm(jupiter_vec))
    jupiter_frac = jupiter_mag / total_mag if total_mag > 0 else 0.0

    # Azimuth of the stacked gravity vector projected onto the equatorial
    # plane, rotated to geographic coordinates by GMST.
    gmst_rad = math.radians(_gmst_deg(jd))
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    R = np.array([[cos_g, sin_g, 0], [-sin_g, cos_g, 0], [0, 0, 1]])
    v_ecef = R @ total_vec
    # Project onto equatorial plane and compute azimuth
    vx, vy, vz = v_ecef
    if abs(vx) + abs(vy) < 1e-20:
        az = 0.0
    else:
        az = math.degrees(math.atan2(vy, vx)) % 360.0

    # How many planets are above their time-averaged contribution?
    # (rough thresholds from mean orbital distances)
    mean_mags = {
        "Mercury": 1.0e-9, "Venus": 2.0e-7, "Mars": 1.0e-9,
        "Jupiter": 3.0e-7, "Saturn": 2.0e-8,
    }
    n_above = sum(1 for (code, name, _), m in zip(_PLANETS, planet_mags)
                  if m > mean_mags.get(name, 0))

    return {
        "accel_m_s2": total_mag,
        "vector_m_s2": total_vec,
        "azimuth_deg": float(az),
        "jupiter_frac": float(jupiter_frac),
        "n_planets_above_mean": int(n_above),
    }


def stacked_gravity_azimuth_deg(t) -> float:
    """Geographic azimuth of the stacked planetary gravity vector (deg from north).

    This is the direction in which the combined gravitational pull of all
    major planets is strongest.  When used as a kriging anisotropy axis,
    it encodes the hypothesis that the TEP field is modulated by the
    stacked planetary gravitational tide — the "stacked gravity" signal
    from Papers 1-2.
    """
    return stacked_planetary_gravity(t)["azimuth_deg"]


def earth_bary_accel(t) -> np.ndarray:
    """Earth's acceleration toward the solar system barycenter (m/s^2).

    Computed by finite differencing the Earth's barycentric position
    over a 2-hour interval.  This is the total gravitational acceleration
    from all solar system bodies, dominated by the Sun (~0.006 m/s^2)
    with planetary perturbations (~1e-7 m/s^2).

    The magnitude and direction of this acceleration modulate the
    proper-time rate (TEP prediction: dt/dtau = 1 + Phi/c^2), and the
    direction can be used as an anisotropy axis.
    """
    jd = _to_jd(t)
    dt = 2.0 / 24.0  # 2 hours
    p1 = _earth_bary_pos(jd - dt) * 1000.0  # km -> m
    p2 = _earth_bary_pos(jd + dt) * 1000.0
    v = (p2 - p1) / (2 * dt * 86400.0)  # m/s
    p0 = _earth_bary_pos(jd) * 1000.0
    # Second derivative: a = (p(t+dt) - 2*p(t) + p(t-dt)) / dt^2
    a = (p2 - 2 * p0 + p1) / (dt * 86400.0) ** 2
    return a  # m/s^2


def bary_accel_azimuth_deg(t) -> float:
    """Geographic azimuth of Earth's barycentric acceleration (deg from north).

    This is the direction of the total gravitational force on Earth,
    dominated by the Sun but with planetary perturbations.  It is the
    physically correct anisotropy axis for the TEP proper-time field:
    the field should be most strongly modulated along the direction of
    the gravitational acceleration.
    """
    jd = _to_jd(t)
    a = earth_bary_accel(t)
    gmst_rad = math.radians(_gmst_deg(jd))
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    R = np.array([[cos_g, sin_g, 0], [-sin_g, cos_g, 0], [0, 0, 1]])
    a_ecef = R @ a
    vx, vy, vz = a_ecef
    if abs(vx) + abs(vy) < 1e-20:
        return 0.0
    return float(math.degrees(math.atan2(vy, vx)) % 360.0)


def tidal_azimuth_deg(t) -> float:
    """Geographic azimuth of the combined solar+lunar tidal axis (deg from north).

    The tidal axis is the direction of the combined Sun-Moon tidal
    stretching force projected onto the equatorial plane.  This is the
    direction along which the solid-Earth tide is strongest, and
    therefore the direction along which the TEP field should show the
    strongest tidal modulation.
    """
    jd = _to_jd(t)
    # Tidal axis is along the Sun-Earth + Moon-Earth direction
    earth = _earth_bary_pos(jd)
    sun = _sun_bary_pos(jd)
    moon = _moon_earth_pos(jd)
    # Combined tidal direction: weighted by tidal strength (M/r^3)
    sun_dir = (sun - earth)
    sun_r = np.linalg.norm(sun_dir)
    sun_w = _M_SUN_KG / (sun_r * 1000.0) ** 3  # weight ~ M/r^3
    moon_w = _M_MOON_KG / (np.linalg.norm(moon) * 1000.0) ** 3
    tidal_vec = sun_w * sun_dir / (sun_r + 1e-6) + moon_w * moon / (np.linalg.norm(moon) + 1e-6)
    # Rotate to ECEF
    gmst_rad = math.radians(_gmst_deg(jd))
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    R = np.array([[cos_g, sin_g, 0], [-sin_g, cos_g, 0], [0, 0, 1]])
    v_ecef = R @ tidal_vec
    vx, vy, vz = v_ecef
    if abs(vx) + abs(vy) < 1e-20:
        return 0.0
    return float(math.degrees(math.atan2(vy, vx)) % 360.0)


# ---------------------------------------------------------------------------
# Eclipse proximity — approximate (no separate eclipse catalog needed)
# ---------------------------------------------------------------------------

def eclipse_proximity(t, window_days: float = 15.0) -> dict:
    """Approximate eclipse proximity.

    Solar eclipses occur near new moon (synodic phase ~0) when the Moon
    is near a node (draconic phase ~0 or ~pi).  Lunar eclipses occur near
    full moon (synodic phase ~pi) near a node.

    This is an approximate predictor — exact eclipse times require a
    dedicated catalog (e.g. NASA GSFC).  The approximation is sufficient
    for an integrity flag: the coincidence of new/full moon with nodal
    crossing is the necessary condition for an eclipse.

    Returns:
        {
            "solar_eclipse_proximity": float,   # 0 = at solar eclipse, 1 = far
            "lunar_eclipse_proximity": float,
            "near_eclipse": bool,               # True if either < 0.1
        }
    """
    syn = lunar_phase_synodic(t)
    drac = lunar_phase_draconic(t)

    # Solar eclipse: synodic ~0 AND draconic ~0 or ~pi
    syn_new = min(syn, 2 * math.pi - syn) / math.pi  # 0 at new moon
    drac_node = min(drac, 2 * math.pi - drac,
                    abs(drac - math.pi), 2 * math.pi - abs(drac - math.pi)) / math.pi
    solar_prox = syn_new * drac_node

    # Lunar eclipse: synodic ~pi AND draconic ~0 or ~pi
    syn_full = abs(syn - math.pi) / math.pi  # 0 at full moon
    lunar_prox = syn_full * drac_node

    return {
        "solar_eclipse_proximity": float(solar_prox),
        "lunar_eclipse_proximity": float(lunar_prox),
        "near_eclipse": bool(solar_prox < 0.05 or lunar_prox < 0.05),
    }


# ---------------------------------------------------------------------------
# Feature vector — all predetermined features for one timestamp
# ---------------------------------------------------------------------------

def feature_vector(t) -> dict:
    """Complete predetermined astronomical feature vector for timestamp *t*.

    Returns a dict with all features.  Every value is deterministic —
    computable from ephemeris for any past or future date.
    """
    return {
        "nutation_semiannual_phase": nutation_phase_semiannual(t),
        "nutation_lunar_nodal_phase": nutation_phase_lunar_nodal(t),
        "chandler_phase": chandler_phase(t),
        "lunar_synodic_phase": lunar_phase_synodic(t),
        "lunar_anomalistic_phase": lunar_phase_anomalistic(t),
        "lunar_draconic_phase": lunar_phase_draconic(t),
        "diurnal_sidereal_phase": diurnal_phase_sidereal(t),
        "diurnal_solar_phase": diurnal_phase_solar(t),
        "orbital_velocity_azimuth_deg": orbital_velocity_azimuth_deg(t),
        "cmb_azimuth_deg": cmb_azimuth_deg(t),
        "stacked_gravity_azimuth_deg": stacked_gravity_azimuth_deg(t),
        "tidal_azimuth_deg": tidal_azimuth_deg(t),
        "bary_accel_azimuth_deg": bary_accel_azimuth_deg(t),
        "solar_tidal_accel": solar_tidal_accel(t),
        "lunar_tidal_accel": lunar_tidal_accel(t),
        "total_tidal_accel": total_tidal_accel(t),
        "earth_sun_distance_au": earth_sun_distance_au(t),
        "earth_moon_distance_km": earth_moon_distance_km(t),
        "nearest_event_days": nearest_significant_event_days(t),
        "eclipse": eclipse_proximity(t),
    }


def harmonic_regressors(t) -> np.ndarray:
    """Sin/cos harmonic regressors for the predetermined periodic features.

    Returns a (20,) array:
        [sin(semiannual),    cos(semiannual),
         sin(lunar_nodal),   cos(lunar_nodal),
         sin(chandler),      cos(chandler),
         sin(synodic),       cos(synodic),
         sin(anomalistic),   cos(anomalistic),
         sin(draconic),      cos(draconic),
         sin(sidereal_day),  cos(sidereal_day),
         sin(solar_day),     cos(solar_day),
         sin(semidiurnal),   cos(semidiurnal),     # 12h (S2 tide)
         sin(terdiurnal),    cos(terdiurnal)]       # 8h (S3 tide)

    The diurnal and semi-diurnal harmonics are the S1 (24h) and S2 (12h)
    atmospheric/oceanic tidal harmonics that dominate the GNSS clock
    residual at sub-daily periods.  The terdiurnal S3 (8h) is weaker but
    is a known harmonic in geodetic residuals.

    These are the exogenous regressors that can be added to the clock
    prediction model.  The amplitudes are fitted on the pre-window
    (training) data only — the phases and periods are fixed by astronomy.
    """
    sa = nutation_phase_semiannual(t)
    ln = nutation_phase_lunar_nodal(t)
    ch = chandler_phase(t)
    sy = lunar_phase_synodic(t)
    an = lunar_phase_anomalistic(t)
    dr = lunar_phase_draconic(t)
    sd = diurnal_phase_sidereal(t)
    so = diurnal_phase_solar(t)
    # Semi-diurnal: 2x the solar day frequency (S2 atmospheric tide, 12h)
    semi = 2.0 * so
    # Terdiurnal: 3x the solar day frequency (S3, 8h)
    ter = 3.0 * so
    return np.array([
        math.sin(sa), math.cos(sa),
        math.sin(ln), math.cos(ln),
        math.sin(ch), math.cos(ch),
        math.sin(sy), math.cos(sy),
        math.sin(an), math.cos(an),
        math.sin(dr), math.cos(dr),
        math.sin(sd), math.cos(sd),
        math.sin(so), math.cos(so),
        math.sin(semi), math.cos(semi),
        math.sin(ter), math.cos(ter),
    ])


# ---------------------------------------------------------------------------
# Anisotropic distance for kriging — using predetermined directions
# ---------------------------------------------------------------------------

def anisotropic_distance_matrix(D: np.ndarray, az: np.ndarray,
                                 preferred_az_deg: float,
                                 ratio: float) -> np.ndarray:
    """Stretch the distance matrix to encode geometric anisotropy.

    For a preferred axis at azimuth *preferred_az_deg*, a pair whose
    bearing is aligned with the axis has its distance divided by *ratio*
    (longer correlation), and a pair perpendicular is unstretched.

        d_eff = d * sqrt( cos^2(delta) / ratio^2 + sin^2(delta) )

    where delta is the acute angle between the pair bearing and the
    preferred axis.

    This is the same formula as tep_kernel.anisotropic_distance, exposed
    here so step_48 can use predetermined directions (velocity, CMB)
    without going through the KernelParams interface.
    """
    if ratio == 1.0:
        return D
    delta = (az - preferred_az_deg) % 180.0
    delta = np.minimum(delta, 180.0 - delta)
    delta_rad = np.radians(delta)
    stretch = np.sqrt(np.cos(delta_rad) ** 2 / ratio ** 2
                      + np.sin(delta_rad) ** 2)
    return D * stretch


def anisotropic_distance_vector(d_vec: np.ndarray, az_vec: np.ndarray,
                                preferred_az_deg: float,
                                ratio: float) -> np.ndarray:
    """Same as anisotropic_distance_matrix but for a target-to-refs vector."""
    if ratio == 1.0:
        return d_vec
    delta = (az_vec - preferred_az_deg) % 180.0
    delta = np.minimum(delta, 180.0 - delta)
    delta_rad = np.radians(delta)
    stretch = np.sqrt(np.cos(delta_rad) ** 2 / ratio ** 2
                      + np.sin(delta_rad) ** 2)
    return d_vec * stretch


def baseline_azimuth_deg(xyz_a: np.ndarray, xyz_b: np.ndarray) -> float:
    """Geographic azimuth (deg from north) of the baseline from a to b.

    Both inputs are ECEF (m or km).  Returns the bearing at point a.
    """
    # Convert ECEF to lat/lon
    x, y, z = xyz_a
    lat_a = math.atan2(z, math.hypot(x, y))
    lon_a = math.atan2(y, x)
    dx, dy, dz = xyz_b - xyz_a
    # ENU at point a
    sl, cl = math.sin(lon_a), math.cos(lon_a)
    sa, ca = math.sin(lat_a), math.cos(lat_a)
    e = -sl * dx + cl * dy
    n = -sa * cl * dx - sa * sl * dy + ca * dz
    az = math.degrees(math.atan2(e, n)) % 180.0
    return az


# ---------------------------------------------------------------------------
# Time-varying lambda — orbital velocity modulation
# ---------------------------------------------------------------------------

def time_varying_lambda(t, lambda_base_km: float = 4200.0,
                        modulation_amplitude: float = 0.095) -> float:
    """Time-varying correlation length modulated by the annual orbital cycle.

    Paper 2 found ~19% annual modulation of the EW/NS anisotropy ratio
    (r = -0.888 with heliocentric velocity).  The correlation length
    itself may modulate with the same period.  We apply a ±9.5%
    sinusoidal modulation (half the 19% anisotropy modulation, conservatively)
    locked to the solar year.

    This is a hypothesis to be tested, not an assumption: the step_48
    A/B test compares fixed-lambda vs time-varying-lambda on held-out
    data.  The phase is locked to the vernal equinox (DOY 80).
    """
    jd = _to_jd(t)
    dt = datetime.fromtimestamp(
        (jd - _J2000_JD) * 86400.0 + _J2000.timestamp(),
        tz=timezone.utc)
    doy = dt.timetuple().tm_yday
    annual_phase = 2.0 * math.pi * ((doy - 80) / SOLAR_YEAR_D)
    modulation = 1.0 + modulation_amplitude * math.cos(annual_phase)
    return lambda_base_km * modulation


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = datetime(2024, 4, 9, 12, 0, 0, tzinfo=timezone.utc)  # DOY 100
    print(f"Features for {t} (DOY 100, 2024):")
    fv = feature_vector(t)
    for k, v in fv.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v:.4f}")

    print(f"\nHarmonic regressors ({len(harmonic_regressors(t))}-dim): {harmonic_regressors(t)}")
    print(f"Orbital velocity azimuth: {orbital_velocity_azimuth_deg(t):.1f}°")
    print(f"CMB azimuth: {cmb_azimuth_deg(t):.1f}°")
    print(f"Stacked gravity azimuth: {stacked_gravity_azimuth_deg(t):.1f}°")
    print(f"Tidal azimuth: {tidal_azimuth_deg(t):.1f}°")
    print(f"Bary accel azimuth: {bary_accel_azimuth_deg(t):.1f}°")
    print(f"Time-varying lambda: {time_varying_lambda(t):.1f} km")
    print(f"Solar tidal accel: {solar_tidal_accel(t):.3e} m/s^2")
    print(f"Lunar tidal accel: {lunar_tidal_accel(t):.3e} m/s^2")
    print(f"Total tidal accel: {total_tidal_accel(t):.3e} m/s^2")
    print(f"Earth-Sun distance: {earth_sun_distance_au(t):.6f} AU")
    print(f"Earth-Moon distance: {earth_moon_distance_km(t):.0f} km")

    # Test stacked planetary gravity
    print("\nStacked planetary gravity:")
    try:
        sg = stacked_planetary_gravity(t)
        print(f"  magnitude: {sg['accel_m_s2']:.3e} m/s^2")
        print(f"  azimuth: {sg['azimuth_deg']:.1f}°")
        print(f"  Jupiter fraction: {sg['jupiter_frac']:.3f}")
        print(f"  planets above mean: {sg['n_planets_above_mean']}")
    except Exception as e:
        print(f"  (ephemeris error: {e})")

    # Test planetary events
    print("\nPlanetary event proximity:")
    try:
        prox = planetary_event_proximity(t)
        for planet, info in prox.items():
            print(f"  {planet}: {info['event_type']}, "
                  f"days_to_event={info['days_to_event']:.1f}, "
                  f"in_window={info['in_window']}")
    except Exception as e:
        print(f"  (ephemeris error: {e})")
