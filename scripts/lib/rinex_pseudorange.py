"""Parse RINEX 2.11 and 3.0x observation + navigation files for raw pseudorange EKF.

Extracts per-epoch, per-satellite pseudoranges and carrier phases, computes
satellite positions from broadcast ephemeris (or precise SP3 products when
available), and forms ionosphere-free combinations for dual-frequency
observations so the EKF can use real measurement geometry instead of SPP
position pseudo-measurements.

RINEX 3.x support is essential for the LC29H(BS) receiver, which tracks
L1+L5 (GPS/Galileo) and B2I+B2a (BeiDou).  The standard RTKLIB convbin
outputs RINEX 2.11, which only supports L1/L2 codes and silently drops L5;
convbin_demo5 with RINEX 3.x output preserves the second frequency.
"""
import numpy as np
from pathlib import Path
import math
import re

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_GM = 3.986005e14
WGS84_OMEGA_E = 7.2921151467e-5
C = 299792458.0  # speed of light m/s

# GNSS carrier frequencies (Hz) and wavelengths (m) for dual-frequency
# ionosphere-free combination.  The LC29H(BS) tracks L1+L5; legacy
# geodetic receivers track L1+L2.  Both pairs are supported below.
L1_FREQ = 1.57542e9
L2_FREQ = 1.22760e9
L5_FREQ = 1.17645e9
L1_WAVELEN = C / L1_FREQ   # 0.1903 m
L2_WAVELEN = C / L2_FREQ   # 0.2442 m
L5_WAVELEN = C / L5_FREQ   # 0.2548 m

# BeiDou frequencies (Hz)
B1I_FREQ = 1.561098e9
B2I_FREQ = 1.207140e9
B2A_FREQ = 1.176450e9      # same as GPS L5 / Galileo E5a
B3I_FREQ = 1.268520e9

# Galileo frequencies
E1_FREQ = 1.575420e9       # same as GPS L1
E5A_FREQ = 1.176450e9      # same as GPS L5
E5B_FREQ = 1.207140e9      # same as BeiDou B2I

# GLONASS frequencies (L1 ~1598.0625 + k*0.5625 MHz, L2 ~1242.9375 + k*0.4375 MHz)
# Channel k is satellite-specific; handled per-satellite in the nav parser.

# Map RINEX 3.x observation codes to carrier frequencies (Hz) per constellation.
# Key = (system_char, obs_code), value = frequency in Hz.
# Only code (C*) and phase (L*) types are mapped; the frequency is the same
# for code and phase on the same signal.
RINEX3_FREQ_MAP: dict[tuple[str, str], float] = {
    # GPS
    ("G", "C1C"): L1_FREQ, ("G", "L1C"): L1_FREQ,
    ("G", "C1W"): L1_FREQ, ("G", "L1W"): L1_FREQ,
    ("G", "C1Y"): L1_FREQ, ("G", "L1Y"): L1_FREQ,
    ("G", "C2W"): L2_FREQ, ("G", "L2W"): L2_FREQ,
    ("G", "C2L"): L2_FREQ, ("G", "L2L"): L2_FREQ,
    ("G", "C2X"): L2_FREQ, ("G", "L2X"): L2_FREQ,
    ("G", "C5I"): L5_FREQ, ("G", "L5I"): L5_FREQ,
    ("G", "C5Q"): L5_FREQ, ("G", "L5Q"): L5_FREQ,
    ("G", "C5X"): L5_FREQ, ("G", "L5X"): L5_FREQ,
    # Galileo
    ("E", "C1C"): E1_FREQ, ("E", "L1C"): E1_FREQ,
    ("E", "C1X"): E1_FREQ, ("E", "L1X"): E1_FREQ,
    ("E", "C5I"): E5A_FREQ, ("E", "L5I"): E5A_FREQ,
    ("E", "C5Q"): E5A_FREQ, ("E", "L5Q"): E5A_FREQ,
    ("E", "C5X"): E5A_FREQ, ("E", "L5X"): E5A_FREQ,
    ("E", "C7I"): E5B_FREQ, ("E", "L7I"): E5B_FREQ,
    ("E", "C7Q"): E5B_FREQ, ("E", "L7Q"): E5B_FREQ,
    ("E", "C7X"): E5B_FREQ, ("E", "L7X"): E5B_FREQ,
    # BeiDou
    # Per RINEX 3.04: attribute "2I" = B1I at 1561.098 MHz,
    # "7I" = B2I at 1207.14 MHz, "5P"/"5X" = B2a at 1176.45 MHz,
    # "6I" = B3I at 1268.52 MHz, "1C"/"1X" = B1C at 1575.42 MHz.
    ("C", "C1I"): B1I_FREQ, ("C", "L1I"): B1I_FREQ,
    ("C", "C1X"): 1575.42e6, ("C", "L1X"): 1575.42e6,  # B1C
    ("C", "C1C"): 1575.42e6, ("C", "L1C"): 1575.42e6,  # B1C (alt)
    ("C", "C2I"): B1I_FREQ, ("C", "L2I"): B1I_FREQ,   # B1I at 1561.098 MHz
    ("C", "C7I"): B2I_FREQ, ("C", "L7I"): B2I_FREQ,   # B2I at 1207.14 MHz
    ("C", "C5P"): B2A_FREQ, ("C", "L5P"): B2A_FREQ,   # B2a at 1176.45 MHz
    ("C", "C5X"): B2A_FREQ, ("C", "L5X"): B2A_FREQ,   # B2a at 1176.45 MHz
    ("C", "C6I"): 1268.52e6, ("C", "L6I"): 1268.52e6, # B3I
    # GLONASS base frequencies (channel-specific offset applied in _glonass_freq)
    ("R", "C1C"): 1602e6, ("R", "L1C"): 1602e6,
    ("R", "C1P"): 1602e6, ("R", "L1P"): 1602e6,
    ("R", "C2C"): 1246e6, ("R", "L2C"): 1246e6,
    ("R", "C2P"): 1246e6, ("R", "L2P"): 1246e6,
}


def _glonass_freq(base_freq: float, channel: int) -> float:
    """GLONASS FDMA frequency: L1 = 1602 + k*0.5625 MHz, L2 = 1246 + k*0.4375 MHz."""
    if base_freq > 1.4e9:  # L1
        return base_freq + channel * 0.5625e6
    else:  # L2
        return base_freq + channel * 0.4375e6


def _freq_for_rinex3_code(sys_char: str, obs_code: str) -> float | None:
    """Carrier frequency (Hz) for a RINEX 3.x observation code."""
    return RINEX3_FREQ_MAP.get((sys_char, obs_code))


def _klobuchar_iono(lat_deg, lon_deg, az_deg, elev_deg, gps_sow,
                     alpha, beta):
    """Klobuchar ionospheric delay model (GPS broadcast).

    Returns delay in metres on L1 frequency.
    alpha: (a0,a1,a2,a3) — vertical delay coefficients.
    beta:  (b0,b1,b2,b3) — period coefficients.
    """
    # Semi-circles (Klobuchar uses semi-circles, not degrees)
    lat_sc = lat_deg / 180.0
    lon_sc = lon_deg / 180.0
    az_sc = az_deg / 180.0
    el_sc = elev_deg / 180.0

    # Earth-centred angle (semi-circles)
    psi = 0.0137 / (el_sc + 0.11) - 0.022

    # Sub-ionospheric latitude (semi-circles)
    lat_i = lat_sc + psi * math.cos(az_sc * math.pi)
    if lat_i > 0.416:
        lat_i = 0.416
    elif lat_i < -0.416:
        lat_i = -0.416

    # Sub-ionospheric longitude (semi-circles)
    lon_i = lon_sc + psi * math.sin(az_sc * math.pi) / math.cos(lat_i * math.pi)

    # Geomagnetic latitude (semi-circles)
    lat_m = lat_i + 0.064 * math.cos((lon_i - 1.617) * math.pi)

    # Local time (sec)
    t = 4.32e4 * lon_i + gps_sow
    t = t % 86400.0
    if t < 0:
        t += 86400.0

    # Slant factor
    F = 1.0 + 16.0 * (0.53 - el_sc) ** 3

    # Period
    PER = beta[0] + beta[1] * lat_m + beta[2] * lat_m ** 2 + beta[3] * lat_m ** 3
    if PER < 72000.0:
        PER = 72000.0

    # Phase
    AMP = alpha[0] + alpha[1] * lat_m + alpha[2] * lat_m ** 2 + alpha[3] * lat_m ** 3
    if AMP < 0.0:
        AMP = 0.0

    # Delay (seconds)
    x = 2.0 * math.pi * (t - 50400.0) / PER
    if abs(x) < 1.57:
        T_iono = F * (5e-9 + AMP * (1 - x * x / 2 + x ** 4 / 24))
    else:
        T_iono = F * 5e-9

    return T_iono * C  # convert to metres


def _iono_free_code(p1: float, p2: float, f1: float, f2: float) -> float:
    """Dual-frequency ionosphere-free pseudorange combination.

    P_IF = (f1^2 * P1 - f2^2 * P2) / (f1^2 - f2^2)

    Removes the first-order ionospheric delay (~99% of it), which is the
    dominant error source after the satellite clock.  Code noise is
    amplified by the IF factor (~2.5-3x for L1/L2, ~3.3x for L1/L5) but
    carrier-phase ambiguity resolution absorbs the constant offset, so
    the converged solution is limited by phase noise (mm) and ephemeris,
    not by the amplified code noise.
    """
    denom = f1 ** 2 - f2 ** 2
    if abs(denom) < 1e6:
        raise ValueError(f"Near-singular ionosphere-free combination: f1={f1}, f2={f2}")
    r = f1 ** 2 / denom
    return r * p1 - (r - 1.0) * p2


def _iono_free_phase(l1_m: float, l2_m: float, f1: float, f2: float) -> float:
    """Dual-frequency ionosphere-free carrier phase (in metres).

    L_IF = (f1^2 * L1 - f2^2 * L2) / (f1^2 - f2 ** 2)

    Inputs are already converted from cycles to metres.  The IF phase
    noise is amplified ~3x but remains at the millimetre-to-centimetre
    level, which is what makes cm-level positioning reachable.
    """
    denom = f1 ** 2 - f2 ** 2
    if abs(denom) < 1e6:
        raise ValueError(f"Near-singular ionosphere-free combination: f1={f1}, f2={f2}")
    r = f1 ** 2 / denom
    return r * l1_m - (r - 1.0) * l2_m


def _graphic_combination(code_m: float, phase_m: float) -> float:
    """GRAPHIC (Group and Phase Ionospheric Correction) observable.

    P_GRAPHIC = (P + L) / 2

    The code pseudorange P and carrier phase L (in metres) have
    ionospheric delays of equal magnitude but opposite sign:

        P = rho + c*dt + I + T + eps_code
        L = rho + c*dt - I + T + N*lambda + eps_phase

    Averaging them cancels the first-order ionospheric delay I entirely,
    leaving:

        P_GRAPHIC = rho + c*dt + T + N*lambda/2 + (eps_code + eps_phase)/2

    The effective ambiguity N*lambda/2 is a constant per satellite-arc
    and is absorbed by the EKF's float ambiguity state.  Code noise is
    reduced by a factor of sqrt(2) (~0.7x) relative to raw code, and the
    ionosphere — which Klobuchar only removes to ~50-70% — is gone
    completely.  This is the standard single-frequency PPP observable
    (e.g. RTKLIB's `-PF -L GRAPHIC`) and is what enables decimetre-level
    positioning on the LC29H's L1-only RINEX 2.11 output, where the
    standard convbin drops L5.

    The GRAPHIC noise is dominated by the code component (phase noise is
    mm-level), so sigma_GRAPHIC ≈ sigma_code / sqrt(2).  The EKF weights
    it accordingly via the if_factor / freq_mode path.
    """
    return 0.5 * (code_m + phase_m)


def _tropo_correction(elev_deg):
    """Simple tropospheric delay model (Saastamoinen).

    Returns delay in metres for a satellite at given elevation.
    """
    if elev_deg < 5.0:
        elev_deg = 5.0
    e = math.radians(elev_deg)
    # Zenith delay: hydrostatic (well modelled by a standard atmosphere at
    # sea level) plus a nominal wet term.  Whatever this gets wrong is the
    # residual the EKF's ZTD state is there to estimate.
    trop_zenith = 2.3 + 0.1
    # Mapping function (Black & Eisner / simplified Niell), better than 1%
    # above about 5 deg:
    #     mf(e) = 1.001 / sqrt(0.002001 + sin^2(e))
    #
    # The previous form was  2.3/sin(e) + 0.085/sin(e)^2.  Both parts were
    # wrong: 1/sin(e) is the flat-Earth approximation, and the 1/sin^2(e)
    # term has no physical basis and explodes near the horizon.  Together
    # they OVER-corrected the slant delay by +2.7 m at 10 deg elevation
    # (the mask boundary) and +13 m at 5 deg, while agreeing at the zenith.
    # An elevation-dependent bias of that shape is indistinguishable from a
    # vertical position error, so it drove the EKF's height and pinned the
    # ZTD state against its clip trying to absorb the excess.
    #
    # It must match tropo_mapping() in ppp_estimator_kinematic, which is the
    # partial derivative the ZTD state is estimated through: if the a-priori
    # is removed with one mapping and the residual estimated with another,
    # the ZTD absorbs the difference between the two mappings instead of the
    # actual wet delay.
    mf = 1.001 / math.sqrt(0.002001 + math.sin(e) ** 2)
    return min(trop_zenith * mf, 30.0)  # cap at 30m for very low elevations


def _windup_correction(sat_pos, rx_pos, sat_vel=None):
    """Carrier-phase wind-up correction (Wu et al. 1993).

    The wind-up effect is a rotation of the phase due to the changing
    relative orientation of the transmitting and receiving antenna
    polarisation patterns.  It can reach several centimetres for
    rotating satellites and must be corrected for cm-level PPP.

    Returns the wind-up correction in cycles (multiply by wavelength
    for metres).  The sign convention follows RTKLIB: the correction
    is ADDED to the phase observation.
    """
    # Line-of-sight vector from receiver to satellite
    los = sat_pos - rx_pos
    norm = np.linalg.norm(los)
    if norm < 1.0:
        return 0.0
    los = los / norm

    # Satellite body-fixed coordinate system.
    # For GPS/Galileo/BeiDou MEO: the satellite antenna points toward
    # Earth centre, and the solar panels rotate about the Sun axis.
    # The effective dipole is along the cross product of the Sun direction
    # and the LOS.
    earth_center = -sat_pos / np.linalg.norm(sat_pos)

    # If satellite velocity is available, use it to estimate the
    # orbital plane normal; otherwise use a simplified model.
    if sat_vel is not None and np.linalg.norm(sat_vel) > 1.0:
        # Sun direction: approximate as the direction from satellite to Sun.
        # The Sun is at ~1 AU from Earth; the satellite is at ~20,000 km.
        # A rough Sun position in ECEF at the given time would be ideal,
        # but for cm-level wind-up the approximation is adequate.
        # Use the direction from Earth centre to satellite as a proxy for
        # the Sun direction when no solar ephemeris is available.
        # This is a known limitation; for precise PPP a solar ephemeris
        # should be used.
        sun_dir = -earth_center  # Sun is roughly opposite to Earth-centre direction
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
    else:
        # Approximate Sun direction (very rough, but wind-up is only cm-level)
        sun_dir = np.array([1.0, 0.0, 0.0])

    # Satellite dipole: along the cross product of Sun direction and LOS
    sat_dipole = np.cross(sun_dir, los)
    norm_d = np.linalg.norm(sat_dipole)
    if norm_d < 1e-10:
        return 0.0
    sat_dipole = sat_dipole / norm_d

    # Receiver dipole: typically aligned with the local vertical
    rx_dir = -earth_center  # pointing up from the ground
    rx_dipole = np.cross(los, rx_dir)
    norm_r = np.linalg.norm(rx_dipole)
    if norm_r < 1e-10:
        return 0.0
    rx_dipole = rx_dipole / norm_r

    # Wind-up in cycles
    dot = np.dot(sat_dipole, rx_dipole)
    cross = np.cross(sat_dipole, rx_dipole)
    windup = math.atan2(np.dot(cross, los), dot) / (2.0 * math.pi)

    return windup


def _solid_earth_tide(rx_ecef, gps_t):
    """Solid Earth tide site displacement (simplified).

    ``gps_t`` is absolute GPS seconds since the GPS epoch (1980-01-06),
    i.e. week*604800 + seconds-of-week — the Sun/Moon ephemeris needs a
    real Julian date, so seconds-of-week alone is NOT sufficient (that
    was a live bug: JD stayed pinned to GPS week 0).

    Returns the displacement vector (metres) to ADD to the receiver
    ECEF position.  The dominant effect is the degree-2 lunar and
    solar tide, which can reach ~0.3m vertically and ~0.05m
    horizontally.  This simplified model uses the IERS Conventions
    degree-2 Love numbers (h2=0.609, l2=0.085).

    For cm-level PPP this correction is essential — without it the
    receiver position appears to oscillate by up to 30cm with
    semidiurnal and diurnal periods.
    """
    # Love numbers
    h2 = 0.609
    l2 = 0.085

    # Receiver position in spherical coordinates
    r = np.linalg.norm(rx_ecef)
    if r < 1.0:
        return np.zeros(3)
    lat = math.asin(rx_ecef[2] / r)
    lon = math.atan2(rx_ecef[1], rx_ecef[0])

    # Approximate Sun and Moon positions.
    # Sun: simplified ephemeris (good to ~0.1° for tide computation)
    # GPS time in seconds; convert to Julian centuries
    # GPS epoch: 1980-01-06 = JD 2444244.5
    jd = 2444244.5 + gps_t / 86400.0
    t = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Sun mean longitude (radians)
    sun_L = math.radians(280.460 + 36000.770 * t)
    # Sun mean anomaly
    sun_M = math.radians(357.5277 + 35999.0503 * t)
    # Sun ecliptic longitude
    sun_lambda = sun_L + math.radians(1.9147) * math.sin(sun_M) + \
                 math.radians(0.02) * math.sin(2 * sun_M)
    # Sun obliquity
    eps = math.radians(23.4393 - 0.01300 * t)
    # Sun position in ECEF (approximate, ignoring equation of equinoxes)
    sun_r = 1.0  # in AU; we only need direction
    sun_x = sun_r * math.cos(sun_lambda)
    sun_y = sun_r * math.sin(sun_lambda) * math.cos(eps)
    sun_z = sun_r * math.sin(sun_lambda) * math.sin(eps)

    # Moon mean longitude
    moon_L = math.radians(218.316 + 481267.8813 * t)
    # Moon mean anomaly
    moon_M = math.radians(134.963 + 477198.8676 * t)
    # Moon argument of latitude
    moon_F = math.radians(93.272 + 483202.0175 * t)
    # Moon ecliptic longitude (simplified)
    moon_lambda = moon_L + math.radians(6.289) * math.sin(moon_M)
    # Moon inclination to ecliptic
    moon_beta = math.radians(5.145) * math.sin(moon_F)
    # Moon position in the ECLIPTIC frame — needs the obliquity rotation
    # to equatorial (the Sun block above already applies it) before the
    # sidereal rotation below can take it to ECEF.
    moon_r = 0.00257  # in AU (384400 km / 149.6e6 km)
    moon_xe = moon_r * math.cos(moon_lambda) * math.cos(moon_beta)
    moon_ye = moon_r * math.sin(moon_lambda) * math.cos(moon_beta)
    moon_ze = moon_r * math.sin(moon_beta)
    moon_x = moon_xe
    moon_y = moon_ye * math.cos(eps) - moon_ze * math.sin(eps)
    moon_z = moon_ye * math.sin(eps) + moon_ze * math.cos(eps)

    # Rotate the equatorial-inertial body positions into the Earth-fixed
    # frame by the Greenwich mean sidereal angle.  Without this the body
    # direction is off by the Earth's current spin angle (hours of RA),
    # making the displacement vector essentially arbitrary.
    gmst = math.radians(
        (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)
    cg, sg = math.cos(gmst), math.sin(gmst)

    def _to_ecef(x, y, z):
        return np.array([cg * x + sg * y, -sg * x + cg * y, z])

    # Compute tide displacement for each body (Sun and Moon)
    disp = np.zeros(3)
    for body_pos, body_mass_ratio in [
        (_to_ecef(moon_x, moon_y, moon_z), 0.0123),      # Moon
        (_to_ecef(sun_x, sun_y, sun_z), 332946.0),       # Sun
    ]:
        # Direction to body from receiver
        body_dir = body_pos - rx_ecef / 149.6e9  # convert rx to AU
        body_dist = np.linalg.norm(body_dir)
        if body_dist < 1e-10:
            continue
        body_dir = body_dir / body_dist

        # Legendre P2(cos z) where z is zenith angle
        cos_z = (rx_ecef @ body_dir) / r
        p2 = 0.5 * (3.0 * cos_z ** 2 - 1.0)

        # Dimensionally correct scale: (R_earth / R_body)^3
        scale = (r / 149.6e9 / body_dist) ** 3

        # Radial displacement
        disp_radial = h2 * body_mass_ratio * scale * p2 * r

        # Horizontal displacement (toward the body)
        # Tangent direction toward the body projected onto local horizontal
        radial_dir = rx_ecef / r
        tangent = body_dir - cos_z * radial_dir
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
            disp_horiz = l2 * body_mass_ratio * scale * \
                         math.sqrt(3.0) * cos_z * math.sqrt(max(0.0, 1.0 - cos_z ** 2)) * r
            disp += disp_horiz * tangent

        disp += disp_radial * radial_dir

    return disp



def _parse_rinex_obs(obs_path: Path) -> dict:
    """Parse RINEX 2.11 observation file.

    Returns dict: {sod: {sat_id: {'C1': float, 'L1': float}}} where
    sod is absolute GPS seconds (week*604800 + seconds-of-week), so the
    axis stays monotonic across the GPS week boundary (SOW wraps at
    Sunday 00:00 UTC).
    """
    with open(obs_path, "r", errors="ignore") as f:
        lines = f.readlines()

    hdr_end = 0
    obs_types = []
    for i, line in enumerate(lines):
        if "# / TYPES OF OBSERV" in line:
            parts = line.split()
            n = int(parts[0])
            obs_types = parts[1:1 + n]
        if "END OF HEADER" in line:
            hdr_end = i + 1
            break

    if not obs_types:
        obs_types = ["C1", "L1"]

    c1_idx = obs_types.index("C1") if "C1" in obs_types else 0
    l1_idx = obs_types.index("L1") if "L1" in obs_types else None
    # Map all observation types to their indices for later extraction
    obs_type_map = {label: idx for idx, label in enumerate(obs_types)}
    n_obs = len(obs_types)
    # In RINEX 2.11, each observation value is 16 chars wide (f14.3 + 2 space)
    # With 5 values per line (80 chars / 16)
    obs_per_line = 5
    lines_per_sat = (n_obs + obs_per_line - 1) // obs_per_line

    epochs = {}
    i = hdr_end
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # Epoch line: ' 26  9  4 18 53 57.0000000  0 28G 6G10G12...'
        # Parse fixed-width: year(1-3), month(4-6), day(7-9), hour(10-12),
        # minute(13-15), sec(16-26), flag(27-29), n_sat(30-32)
        try:
            year = int(line[1:3])
            month = int(line[4:6])
            day = int(line[7:9])
            hour = int(line[10:12])
            minute = int(line[13:15])
            sec = float(line[16:26])
        except (ValueError, IndexError):
            i += 1
            continue

        if year < 80:
            year += 2000
        elif year < 100:
            year += 1900
        # Compute absolute GPS seconds (week*604800 + SOW).  SOW alone
        # wraps to 0 at the week boundary, which sorted a capture
        # crossing Sunday 00:00 UTC into "new week first, old week's
        # tail last" and produced a ~6.8 d fake epoch gap.
        _wk, sod = _gps_week_seconds(year, month, day, hour, minute, sec)
        sod = _wk * 604800.0 + sod

        # n_sat is at position 29-32, but may be followed by sat list on same line
        rest = line[29:]
        rest_stripped = rest.strip()
        # Parse satellite list: starts after the count number
        # Format: '28G 6G10G12G13...' — the count is 28, then sats follow
        # The count may be 1-3 digits, then satellite IDs follow immediately
        sat_section = line[29:]
        # Extract the number at the start
        m = re.match(r'\s*(\d+)', sat_section)
        if not m:
            i += 1
            continue
        n_sat = int(m.group(1))
        sat_list_str = sat_section[m.end():]

        # Parse satellite IDs from the rest of this line + continuation lines
        # Sats are like 'G 6', 'G10', 'R 1', 'E 2', 'C12' etc.
        # They can be packed without spaces: 'G 6G10G12' = G06, G10, G12
        # Or with spaces: 'G 6 G10 G12'
        # The pattern is: letter + 1-2 digits, possibly with space between letter and digits
        sat_ids = []
        # First, remove spaces between system letter and number
        # e.g. 'G 6G10G12' -> 'G06G10G12'
        cleaned = re.sub(r'([GREJC])\s+', r'\1', sat_list_str)
        # Now extract sat IDs: letter + 1-2 digits
        for sm in re.finditer(r'([GREJC])(\d{1,2})', cleaned):
            sat_ids.append(f"{sm.group(1)}{int(sm.group(2)):02d}")
            if len(sat_ids) >= n_sat:
                break

        i += 1
        # Read continuation lines for satellite list if needed
        while len(sat_ids) < n_sat and i < len(lines):
            cont = lines[i].strip()
            if not cont:
                i += 1
                continue
            # Check if this looks like a continuation (starts with spaces in original)
            # or an observation line (starts with a digit/number)
            if cont[0].isdigit() or cont[0] == '.':
                break
            cleaned = re.sub(r'([GREJC])\s+', r'\1', cont)
            for sm in re.finditer(r'([GREJC])(\d{1,2})', cleaned):
                sat_ids.append(f"{sm.group(1)}{int(sm.group(2)):02d}")
                if len(sat_ids) >= n_sat:
                    break
            i += 1

        if len(sat_ids) < n_sat:
            # Skip remaining lines for this epoch
            n_remaining = n_sat - len(sat_ids)
            i += n_remaining * lines_per_sat
            continue

        # Read observation records (n_sat satellites, lines_per_sat lines each)
        epoch_sats = {}
        for si in range(n_sat):
            if i >= len(lines):
                break
            obs_vals = []
            for _ in range(lines_per_sat):
                if i >= len(lines):
                    break
                obs_line = lines[i]
                for j in range(0, len(obs_line), 16):
                    val_str = obs_line[j:j + 14].strip()
                    if val_str:
                        try:
                            obs_vals.append(float(val_str))
                        except ValueError:
                            obs_vals.append(np.nan)
                    else:
                        obs_vals.append(np.nan)
                i += 1
            if len(obs_vals) > c1_idx:
                c1 = obs_vals[c1_idx]
                if np.isfinite(c1) and c1 > 0:
                    rec = {"C1": c1}
                    # Store all available observation types
                    for label, idx in obs_type_map.items():
                        if idx < len(obs_vals):
                            v = obs_vals[idx]
                            if np.isfinite(v) and abs(v) > 0:
                                rec[label] = v
                    epoch_sats[sat_ids[si]] = rec

        if epoch_sats:
            epochs[sod] = epoch_sats

    return epochs


def _parse_rinex3_obs(obs_path: Path) -> dict:
    """Parse a RINEX 3.0x observation file.

    Returns dict: {sod: {sat_id: {obs_code: float, ...}}} where sod is
    absolute GPS seconds (week*604800 + seconds-of-week), monotonic
    across the GPS week boundary.

    RINEX 3.x uses per-constellation ``SYS / # / OBS TYPES`` header lines
    to declare which observation codes are present, and ``>`` epoch records
    with 3-character satellite IDs (e.g. ``G15``, ``E07``, ``C12``).  This
    parser preserves all observation codes so the caller can form
    ionosphere-free combinations from whichever dual-frequency pair the
    receiver actually tracks (L1+L5 for the LC29H(BS), not L1+L2).
    """
    with open(obs_path, "r", errors="ignore") as f:
        lines = f.readlines()

    # Parse header: per-constellation observation types
    sys_obs_types: dict[str, list[str]] = {}
    obs_date: tuple[int, int, int] | None = None
    hdr_end = 0
    for i, line in enumerate(lines):
        if "SYS / # / OBS TYPES" in line:
            parts = line.split()
            # A continuation line for SYS / # / OBS TYPES (when n_types
            # exceeds 13) also carries the label but does NOT repeat the
            # system character or count — its first token is an obs code
            # like 'L7X', not 'G' or 'E'.  Skip those here; they are
            # handled by the continuation loop below.
            if not (len(parts) >= 2 and parts[0] in "GRECJ"
                    and parts[1].isdigit()):
                continue
            sys_char = parts[0]
            n_types = int(parts[1])
            # Take tokens after the count, filtering out the trailing
            # "SYS / # / OBS TYPES" label that appears on every line.
            raw_types = [t for t in parts[2:]
                         if t not in ("SYS", "/", "#", "OBS", "TYPES")]
            types = raw_types[:n_types]
            # Continuation lines: in RINEX 3.x these also contain the
            # "SYS / # / OBS TYPES" label, so we cannot use its absence
            # as a stop condition.  Instead, stop when we have enough
            # types or hit a line that starts with a new system char.
            j = i + 1
            while len(types) < n_types and j < len(lines):
                cont = lines[j]
                if "END OF HEADER" in cont:
                    break
                cont_parts = cont.split()
                # Stop if this is a new SYS / # / OBS TYPES record
                if ("SYS / # / OBS TYPES" in cont
                        and len(cont_parts) >= 2
                        and cont_parts[0] in "GRECJ"
                        and cont_parts[1].isdigit()):
                    break
                # Stop if this is any other header record with a label
                if any(lbl in cont for lbl in (
                        "TIME OF FIRST OBS", "TIME OF LAST OBS",
                        "APPROX POSITION XYZ", "ANTENNA DELTA",
                        "COMMENT", "MARKER NAME", "OBSERVER",
                        "REC # / TYPE", "ANT # / TYPE",
                        "WAVELENGTH FACT", "RCV CLOCK OFFS APPL",
                        "LEAP SECONDS", "# / TYPES OF OBS",
                        "INTERVAL", "GLONASS SLOT")):
                    break
                # Filter out the trailing label tokens that appear on
                # every SYS / # / OBS TYPES line in RINEX 3.x.  Without
                # this, 'SYS', '/', '#', 'OBS', 'TYPES' get added as
                # observation codes, inflating n_types and causing the
                # data parser to read past the end of each satellite's
                # observation line into the next one.
                obs_tokens = [t for t in cont_parts
                              if t not in ("SYS", "/", "#", "OBS", "TYPES")]
                types.extend(obs_tokens)
                j += 1
            # Trim to the declared count in case label tokens slipped in
            types = types[:n_types]
            sys_obs_types[sys_char] = types
        if "TIME OF FIRST OBS" in line:
            parts = line.split()
            obs_date = (int(parts[1]), int(parts[2]), int(parts[3]))
        if "END OF HEADER" in line:
            hdr_end = i + 1
            break

    if not sys_obs_types:
        # Fall back to a sensible default
        sys_obs_types = {"G": ["C1C", "L1C"], "E": ["C1C", "L1C"],
                         "C": ["C2I", "L2I"], "R": ["C1C", "L1C"]}

    epochs: dict[int, dict[str, dict[str, float]]] = {}
    i = hdr_end
    while i < len(lines):
        line = lines[i]
        if not line.startswith(">"):
            i += 1
            continue
        # Epoch line: > YYYY MM DD HH MM SS.SSSS  flag n_sat
        parts = line[1:].split()
        if len(parts) < 7:
            i += 1
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            hour, minute = int(parts[3]), int(parts[4])
            sec = float(parts[5])
        except (ValueError, IndexError):
            i += 1
            continue
        # Compute absolute GPS seconds (week*604800 + SOW), NOT seconds
        # of day and NOT plain SOW.  Seconds-of-day creates a multi-day
        # time mismatch with the nav file's toe; plain SOW wraps to 0 at
        # the week boundary and sorted a capture crossing Sunday 00:00
        # UTC into "new week first, old week's tail last" — a ~6.8 d
        # fake epoch gap.
        _wk, sod = _gps_week_seconds(year, month, day, hour, minute, sec)
        sod = _wk * 604800.0 + sod
        n_sat = int(parts[7]) if len(parts) >= 8 else 0

        i += 1
        epoch_sats: dict[str, dict[str, float]] = {}
        for _ in range(n_sat if n_sat > 0 else 99):
            if i >= len(lines):
                break
            obs_line = lines[i]
            if obs_line.startswith(">") or not obs_line.strip():
                break
            sat_id = obs_line[:3].strip()
            if not sat_id or sat_id[0] not in "GRECJ":
                i += 1
                continue
            sys_char = sat_id[0]
            types = sys_obs_types.get(sys_char, [])
            n_types = len(types)
            # RINEX 3.x: each observation is 16 chars (f14.3 + 2 flag chars)
            # Up to 5 observations per line; continuation lines follow.
            # The LLI (Loss-of-Lock Indicator) is character 15 (index 14):
            #   bit 0 (value 1 or 3) = loss of lock → cycle slip
            #   bit 1 (value 2 or 3) = half-cycle ambiguity (not a slip)
            obs_vals: list[float] = []
            lli_flags: list[str] = []
            obs_line_data = obs_line[3:]
            for j in range(0, len(obs_line_data), 16):
                val_str = obs_line_data[j:j + 14].strip()
                lli_char = obs_line_data[j + 14:j + 15].strip() if j + 14 < len(obs_line_data) else ""
                if val_str:
                    try:
                        obs_vals.append(float(val_str))
                    except ValueError:
                        obs_vals.append(np.nan)
                else:
                    obs_vals.append(np.nan)
                lli_flags.append(lli_char)
                if len(obs_vals) >= n_types:
                    break
            # Read continuation lines if needed
            while len(obs_vals) < n_types and i + 1 < len(lines):
                i += 1
                cont = lines[i]
                if cont.startswith(">") or not cont.strip():
                    i -= 1  # will be incremented below
                    break
                for j in range(0, len(cont), 16):
                    val_str = cont[j:j + 14].strip()
                    lli_char = cont[j + 14:j + 15].strip() if j + 14 < len(cont) else ""
                    if val_str:
                        try:
                            obs_vals.append(float(val_str))
                        except ValueError:
                            obs_vals.append(np.nan)
                    else:
                        obs_vals.append(np.nan)
                    lli_flags.append(lli_char)
                    if len(obs_vals) >= n_types:
                        break
            i += 1
            # Map observation codes to values
            rec: dict[str, float] = {}
            for idx, otype in enumerate(types):
                if idx < len(obs_vals):
                    v = obs_vals[idx]
                    if np.isfinite(v) and abs(v) > 0:
                        rec[otype] = v
                    # LLI is only meaningful for phase observations (L*).
                    # Bit 0 (LLI value 1 or 3) indicates loss of lock,
                    # which is a cycle slip.  Bit 1 (value 2 or 3) is
                    # half-cycle ambiguity, not a slip.
                    if otype.startswith("L") and idx < len(lli_flags):
                        lli_str = lli_flags[idx]
                        if lli_str:
                            try:
                                lli_val = int(lli_str)
                                if lli_val & 1:
                                    rec[f"{otype}_LLI"] = 1.0
                            except ValueError:
                                pass
            if rec:
                epoch_sats[sat_id] = rec
        if epoch_sats:
            epochs[sod] = epoch_sats

    return epochs


def _detect_rinex_version(obs_path: Path) -> int:
    """Return 2 or 3 based on the RINEX version in the file header."""
    try:
        with open(obs_path, "r", errors="ignore") as f:
            first_line = f.readline()
        if "3.0" in first_line or "3.0" in first_line:
            return 3
        # Check for RINEX 3.x marker
        if first_line.strip().startswith("3.0"):
            return 3
        # RINEX 2.11 has "2.11" in the first line
        if "2.1" in first_line:
            return 2
        # Check for SYS / # / OBS TYPES (RINEX 3.x only)
        with open(obs_path, "r", errors="ignore") as f:
            for line in f:
                if "END OF HEADER" in line:
                    break
                if "SYS / # / OBS TYPES" in line:
                    return 3
        return 2
    except Exception as e:
        import warnings
        warnings.warn(f"RINEX version detection failed for {obs_path}: {e}; defaulting to 2")
        return 2


def parse_rinex_obs(obs_path: Path) -> dict:
    """Auto-detect RINEX version and parse accordingly.

    Returns dict: {sod: {sat_id: {obs_code: float}}}.
    For RINEX 2.11 the obs codes are C1/L1/P1/P2/L2/C2.
    For RINEX 3.x the obs codes are per-constellation (C1C/L1C/C5Q/L5Q/...).
    """
    version = _detect_rinex_version(obs_path)
    if version == 3:
        return _parse_rinex3_obs(obs_path)
    return _parse_rinex_obs(obs_path)


def _parse_rinex_nav(nav_path: Path, system: str = "G") -> dict:
    """Parse a RINEX 2.11 broadcast navigation file.

    Returns dict: {sat_id: [ephemeris_dict, ...]}
    In RINEX 2.11 nav, the first record line is:
    'PRN year month day hour min sec af0 af1 af2'
    where PRN is just the number (no constellation prefix) -- the
    constellation is implied by the FILE, which is why *system* is passed
    in by the caller (convbin writes .nav=GPS, .lnav=Galileo, .gnav=GLONASS).

    Galileo uses the same Keplerian broadcast model as GPS and Galileo
    System Time is aligned to GPS time to within nanoseconds, so the GPS
    orbit maths applies unchanged.  GLONASS does NOT (PZ-90 with
    position/velocity/acceleration integration) and is not handled here.
    """
    with open(nav_path, "r", errors="ignore") as f:
        lines = f.readlines()

    hdr_end = 0
    iono_alpha = None
    iono_beta = None
    for i, line in enumerate(lines):
        if "ION ALPHA" in line:
            parts = _split_dnot(line)
            try:
                iono_alpha = [_parse_d(p) for p in parts[:4]]
            except (ValueError, IndexError):
                pass
        if "ION BETA" in line:
            parts = _split_dnot(line)
            try:
                iono_beta = [_parse_d(p) for p in parts[:4]]
            except (ValueError, IndexError):
                pass
        if "END OF HEADER" in line:
            hdr_end = i + 1
            break

    ephemeris = {}
    i = hdr_end
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # The PRN field is FIXED-COLUMN (chars 0-2), and must not be parsed by
        # whitespace splitting.  convbin writes GPS records bare ("15", " 5")
        # but Galileo records prefixed ("E30", "E 7") -- and "E 7" splits into
        # two tokens, which silently misaligns every field after it.  That is
        # why the Galileo file parsed to zero satellites.
        head, rest = line[:3], line[3:]
        if head[:1].isalpha():
            sysc = head[0]
            prn_s = head[1:]
        else:
            sysc = system
            prn_s = head[:2]
            rest = line[2:]
        try:
            prn = int(prn_s.strip())
        except ValueError:
            i += 1
            continue
        # RINEX 2.11 nav files use D-notation (e.g. 0.123456D-03) for
        # floating-point values, and these can run together without
        # spaces when negative: '0.178D-03-0.909D-11'.  Plain split()
        # treats that as one token, producing too few fields and causing
        # every record to be silently skipped.  Use _split_dnot which
        # inserts a space at D-notation boundaries.
        parts = _split_dnot(rest)
        if len(parts) < 9:
            i += 1
            continue
        # Re-index: parts[0..5] = y m d h m s, parts[6..8] = af0 af1 af2
        parts = [str(prn)] + parts
        sat_id = f"{sysc}{prn:02d}"
        try:
            year = int(parts[1])
            if year < 80:
                year += 2000
            elif year < 100:
                year += 1900
            month = int(parts[2])
            day = int(parts[3])
            hour = int(parts[4])
            minute = int(parts[5])
            sec = float(parts[6])
            # Convert toc to seconds-of-week (SOW) to match toe
            toc_week, toc_sow = _gps_week_seconds(year, month, day,
                                                   hour, minute, sec)
            af0 = _parse_d(parts[7])
            af1 = _parse_d(parts[8])
            af2 = _parse_d(parts[9])
        except (ValueError, IndexError):
            i += 1
            continue

        # Lines 2-8: broadcast orbit parameters (4 values per line, f19.12 format)
        if i + 7 >= len(lines):
            break
        try:
            l1 = _split_dnot(lines[i + 1])
            l2 = _split_dnot(lines[i + 2])
            l3 = _split_dnot(lines[i + 3])
            l4 = _split_dnot(lines[i + 4])
            l5 = _split_dnot(lines[i + 5])
            l6 = _split_dnot(lines[i + 6])
            l7 = _split_dnot(lines[i + 7])

            # Line 1: IODE, Crs, Delta_n, M0
            # Line 2: Cuc, e, Cus, sqrt(A)
            # Line 3: Toe, Cic, Omega0, Cis
            # Line 4: i0, Crc, omega, Omega_dot
            # Line 5: IDOT, L2_codes, L2_P_flag, L2_week
            # Line 6: L2_code_phase, accuracy, health, TGD
            # Line 7: IOC, fit_interval
            iode = _parse_d(l1[0])
            crs = _parse_d(l1[1])
            delta_n = _parse_d(l1[2])
            m0 = _parse_d(l1[3])

            cuc = _parse_d(l2[0])
            e = _parse_d(l2[1])
            cus = _parse_d(l2[2])
            sqrt_a = _parse_d(l2[3])

            toe_sod = _parse_d(l3[0])
            cic = _parse_d(l3[1])
            omega0 = _parse_d(l3[2])
            cis = _parse_d(l3[3])

            i0 = _parse_d(l4[0])
            crc = _parse_d(l4[1])
            omega = _parse_d(l4[2])
            omega_dot = _parse_d(l4[3])

            idot = _parse_d(l5[0])
            toe_week = _parse_d(l5[3]) if len(l5) > 3 else 0.0
            # Line 6 (RINEX 2.11 GPS nav orbit 6):
            #   SV accuracy (m), SV health (0-63), TGD (s), IODC
            # TGD is at index 2, NOT 3 — l6[3] is IODC.  Using l6[3]
            # gave 85.0 (an IODC value) instead of ~1e-9 s (the TGD),
            # which is a ~25 million metre error on the satellite clock.
            tgd = 0.0
            try:
                if len(l6) >= 3:
                    tgd = _parse_d(l6[2])
            except (ValueError, IndexError):
                pass
        except (ValueError, IndexError):
            i += 8
            continue

        ephemeris.setdefault(sat_id, []).append({
            "toc_sow": toc_sow, "af0": af0, "af1": af1, "af2": af2,
            "sqrt_a": sqrt_a, "e": e, "m0": m0,
            "omega0": omega0, "i0": i0, "omega": omega,
            "delta_n": delta_n, "cuc": cuc, "cus": cus,
            "crc": crc, "crs": crs, "cic": cic, "cis": cis,
            "idot": idot, "toe_sod": toe_sod, "toe_week": toe_week,
            "omega_dot": omega_dot, "tgd": tgd,
        })
        i += 8

    return ephemeris, iono_alpha, iono_beta


def _parse_d(val_str: str) -> float:
    """Parse a Fortran D-notation float (e.g. -.123D-02) -> float."""
    return float(val_str.replace("D", "E").replace("d", "e"))


def _split_dnot(line: str) -> list[str]:
    """Split a RINEX nav line, handling D-notation values that run together.

    RINEX nav files use Fortran D-notation (e.g. ``0.178D-03``) for
    floating-point values.  When a negative value follows another, the
    minus sign runs into the preceding exponent: ``0.178D-03-0.909D-11``.
    Plain ``split()`` treats that as one token, producing too few fields
    and silently dropping every record.
    """
    import re as _re
    fixed = _re.sub(r'(D[+-]\d\d)([-+])', r'\1 \2', line)
    return [p for p in fixed.split() if p]


def _parse_rinex3_nav(nav_path: Path) -> dict:
    """Parse a RINEX 3.0x mixed broadcast navigation file.

    Handles GPS, Galileo, and BeiDou (all Keplerian, same structure as
    RINEX 2.11 GPS nav but with system-prefixed satellite IDs and
    D-notation exponents).  GLONASS is skipped (PZ-90 Cartesian model).

    Returns (ephemeris_dict, iono_alpha, iono_beta).
    """
    with open(nav_path, "r", errors="ignore") as f:
        lines = f.readlines()

    hdr_end = 0
    iono_alpha = None
    iono_beta = None
    for i, line in enumerate(lines):
        if "IONOSPHERIC CORR" in line or "ION ALPHA" in line:
            parts = line.replace("D", "E").split()
            try:
                iono_alpha = [float(p) for p in parts[1:5] if _is_float(p)]
            except (ValueError, IndexError):
                pass
        if "ION BETA" in line:
            parts = line.replace("D", "E").split()
            try:
                iono_beta = [float(p) for p in parts[1:5] if _is_float(p)]
            except (ValueError, IndexError):
                pass
        if "END OF HEADER" in line:
            hdr_end = i + 1
            break

    ephemeris: dict[str, list[dict]] = {}
    i = hdr_end
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # RINEX 3.x nav: satellite ID is first 3 chars (e.g. "G30", "E07", "C20")
        sat_id = line[:3].strip()
        if len(sat_id) < 2 or sat_id[0] not in "GRECJ":
            i += 1
            continue
        sys_char = sat_id[0]
        # GLONASS uses Cartesian state vectors, not Keplerian — skip
        if sys_char == "R":
            # GLONASS records are 4 lines (not 8 like Keplerian)
            i += 4
            continue
        # Parse the epoch line: SAT year month day hour min sec af0 af1 af2
        rest = line[3:].strip()
        parts = _split_dnot(rest)
        if len(parts) < 9:
            i += 1
            continue
        try:
            year = int(parts[0])
            if year < 80:
                year += 2000
            elif year < 100:
                year += 1900
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            sec = float(parts[5])
            toc_week, toc_sow = _gps_week_seconds(year, month, day, hour, minute, sec)
            af0 = _parse_d(parts[6])
            af1 = _parse_d(parts[7])
            af2 = _parse_d(parts[8])
        except (ValueError, IndexError):
            i += 1
            continue

        # Lines 2-8: broadcast orbit parameters (4 per line, D-notation)
        if i + 7 >= len(lines):
            break
        try:
            l1 = _split_dnot(lines[i + 1])
            l2 = _split_dnot(lines[i + 2])
            l3 = _split_dnot(lines[i + 3])
            l4 = _split_dnot(lines[i + 4])
            l5 = _split_dnot(lines[i + 5])
            l6 = _split_dnot(lines[i + 6])
            l7 = _split_dnot(lines[i + 7])

            iode = _parse_d(l1[0])
            crs = _parse_d(l1[1])
            delta_n = _parse_d(l1[2])
            m0 = _parse_d(l1[3])

            cuc = _parse_d(l2[0])
            e = _parse_d(l2[1])
            cus = _parse_d(l2[2])
            sqrt_a = _parse_d(l2[3])

            toe_sod = _parse_d(l3[0])
            cic = _parse_d(l3[1])
            omega0 = _parse_d(l3[2])
            cis = _parse_d(l3[3])

            i0 = _parse_d(l4[0])
            crc = _parse_d(l4[1])
            omega = _parse_d(l4[2])
            omega_dot = _parse_d(l4[3])

            idot = _parse_d(l5[0])
            toe_week = _parse_d(l5[3]) if len(l5) > 3 else 0.0
            # TGD: line 6, index 2 (same as RINEX 2.11)
            tgd = 0.0
            try:
                if len(l6) >= 3:
                    tgd = _parse_d(l6[2])
            except (ValueError, IndexError):
                pass
        except (ValueError, IndexError):
            i += 8
            continue

        # BeiDou time system offset: BDT = GPST - 14s.
        # The RINEX 3.04 spec stores BeiDou toe/toc in BDT.  Convert to
        # GPST so the rest of the pipeline (which uses GPST throughout)
        # computes the correct tk = t_sow - toe_sod.
        # Without this, a 14-second error in tk causes ~50 km along-track
        # satellite position error, producing 3-7 km range innovations.
        if sys_char == "C":
            toc_sow += 14.0
            toe_sod += 14.0

        ephemeris.setdefault(sat_id, []).append({
            "toc_sow": toc_sow, "af0": af0, "af1": af1, "af2": af2,
            "sqrt_a": sqrt_a, "e": e, "m0": m0,
            "omega0": omega0, "i0": i0, "omega": omega,
            "delta_n": delta_n, "cuc": cuc, "cus": cus,
            "crc": crc, "crs": crs, "cic": cic, "cis": cis,
            "idot": idot, "toe_sod": toe_sod, "toe_week": toe_week,
            "omega_dot": omega_dot, "tgd": tgd,
        })
        i += 8

    return ephemeris, iono_alpha, iono_beta


def _is_float(s: str) -> bool:
    try:
        float(s.replace("D", "E").replace("d", "e"))
        return True
    except (ValueError, TypeError):
        return False


def _gps_week_seconds(year, month, day, hour, minute, sec):
    """Convert calendar date to GPS week and seconds of week."""
    import datetime as dt
    gps_epoch = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime(year, month, day, hour, minute,
                        int(sec), int((sec % 1) * 1e6),
                        tzinfo=dt.timezone.utc)
    except ValueError:
        t = dt.datetime(year, month, day, hour, minute, 0,
                        tzinfo=dt.timezone.utc)
    delta = t - gps_epoch
    total_sec = delta.total_seconds()
    week = int(total_sec // 604800)
    sow = total_sec % 604800
    return week, sow


def _doy_from_gps_abs(sod: float) -> int | None:
    """Day-of-year for an absolute GPS second count (week*604800 + SOW)."""
    try:
        import datetime as dt
        t = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc) \
            + dt.timedelta(seconds=float(sod))
        return int(t.timetuple().tm_yday)
    except Exception:
        return None


def _sow_delta(a: float, b: float) -> float:
    """Signed difference a - b on the circular seconds-of-week axis.

    Returns a value in [-302400, 302400), so a TOE at the end of the
    previous week compares correctly against an epoch at the start of
    the next week (plain ``a - b`` would read ~604800 s).
    """
    return (a - b + 302400.0) % 604800.0 - 302400.0


def _satellite_position(eph: dict, gps_week: int, t_sow: float) -> np.ndarray:
    """Compute ECEF satellite position from broadcast ephemeris.

    Uses standard GPS Keplerian orbit propagation.
    """
    a = eph["sqrt_a"] ** 2
    n0 = math.sqrt(WGS84_GM / a ** 3)
    tk = t_sow - eph["toe_sod"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800

    n = n0 + eph["delta_n"]
    mk = eph["m0"] + n * tk

    ek = mk
    for _ in range(10):
        ek_new = mk + eph["e"] * math.sin(ek)
        if abs(ek_new - ek) < 1e-12:
            break
        ek = ek_new

    cos_ek = math.cos(ek)
    sin_ek = math.sin(ek)

    true_anom = math.atan2(
        math.sqrt(1 - eph["e"] ** 2) * sin_ek,
        cos_ek - eph["e"]
    )
    phi = true_anom + eph["omega"]

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    du = eph["cuc"] * cos_phi + eph["cus"] * sin_phi
    dr = eph["crc"] * cos_phi + eph["crs"] * sin_phi
    di = eph["cic"] * cos_phi + eph["cis"] * sin_phi

    u = phi + du
    r = a * (1 - eph["e"] * cos_ek) + dr
    i = eph["i0"] + di + eph["idot"] * tk

    cos_u = math.cos(u)
    sin_u = math.sin(u)

    omega_k = eph["omega0"] + (eph.get("omega_dot", 0.0) - WGS84_OMEGA_E) * tk - \
        WGS84_OMEGA_E * eph["toe_sod"]
    cos_ok = math.cos(omega_k)
    sin_ok = math.sin(omega_k)

    cos_i = math.cos(i)
    sin_i = math.sin(i)

    x = r * (cos_u * cos_ok - sin_u * cos_i * sin_ok)
    y = r * (cos_u * sin_ok + sin_u * cos_i * cos_ok)
    z = r * sin_u * sin_i

    return np.array([x, y, z])


def _sat_clock_correction(eph: dict, t_sow: float) -> float:
    """Compute satellite clock correction (seconds).

    Includes the broadcast polynomial (af0/af1/af2), the GPS relativistic
    eccentricity correction (-2*sqrt(a*mu)/c^2 * e*sin(E)), and the TGD
    (Timing Group Delay) for the L1 C/A signal.

    The relativistic term can reach ~13 m (45 ns) and varies sinusoidally
    over the orbit; omitting it is a ~10 m systematic on every satellite
    clock, which directly maps into range error.  The _compute_ek helper
    existed but was never wired in -- that was a defect.
    """
    tk = t_sow - eph["toc_sow"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    dt_poly = eph["af0"] + eph["af1"] * tk + eph["af2"] * tk ** 2
    # Relativistic eccentricity correction (IS-GPS-200 §20.3.3.3.3.1)
    ek = _compute_ek(eph, t_sow)
    a = eph["sqrt_a"] ** 2
    dt_rel = -2.0 * math.sqrt(a * WGS84_GM) / C**2 * eph["e"] * math.sin(ek)
    # TGD: L1/L2 differential delay.  The broadcast clock polynomial is
    # referenced to the ionosphere-free combination; a single-frequency L1
    # C/A user must subtract TGD (IS-GPS-200 §20.3.3.3.3.2).  For the IF
    # combination TGD is already absorbed and must NOT be re-applied; the
    # caller uses _sat_clock_correction_if for that path.
    tgd = float(eph.get("tgd", 0.0))
    return dt_poly + dt_rel - tgd


def _sat_clock_correction_if(eph: dict, t_sow: float) -> float:
    """Satellite clock correction for the ionosphere-free combination.

    The broadcast clock polynomial already includes the TGD offset that
    aligns the L1/L2 signals; the IF combination cancels it, so TGD is
    NOT applied here.  The relativistic correction is still needed.
    """
    tk = t_sow - eph["toc_sow"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    dt_poly = eph["af0"] + eph["af1"] * tk + eph["af2"] * tk ** 2
    ek = _compute_ek(eph, t_sow)
    a = eph["sqrt_a"] ** 2
    dt_rel = -2.0 * math.sqrt(a * WGS84_GM) / C**2 * eph["e"] * math.sin(ek)
    return dt_poly + dt_rel


def _compute_ek(eph: dict, t_sow: float) -> float:
    """Compute eccentric anomaly for relativistic correction."""
    a = eph["sqrt_a"] ** 2
    n0 = math.sqrt(WGS84_GM / a ** 3)
    tk = t_sow - eph["toe_sod"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    n = n0 + eph["delta_n"]
    mk = eph["m0"] + n * tk
    ek = mk
    for _ in range(10):
        ek_new = mk + eph["e"] * math.sin(ek)
        if abs(ek_new - ek) < 1e-12:
            break
        ek = ek_new
    return ek


def _ecef_to_latlon(ecef: np.ndarray) -> tuple[float, float]:
    """Convert ECEF to geodetic latitude and longitude (degrees)."""
    x, y, z = ecef
    a = WGS84_A
    f = WGS84_F
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat ** 2)
        lat = math.atan2(z + e2 * N * sin_lat, p)
    return math.degrees(lat), math.degrees(lon)


def _los_to_elev_az(los: np.ndarray, lat_deg: float,
                     lon_deg: float) -> tuple[float, float]:
    """Convert ECEF LOS vector to elevation and azimuth (degrees).

    los: unit vector from receiver to satellite in ECEF.
    Returns (elevation_deg, azimuth_deg).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # ECEF -> ENU rotation.
    #
    # Rows are the local East, North and Up unit vectors expressed in ECEF:
    #     E = [-sin(lon),            cos(lon),           0       ]
    #     N = [-sin(lat)*cos(lon),  -sin(lat)*sin(lon),  cos(lat)]
    #     U = [ cos(lat)*cos(lon),   cos(lat)*sin(lon),  sin(lat)]
    #
    # This matrix was previously written TRANSPOSED (the ENU -> ECEF form),
    # so every elevation and azimuth derived from it was wrong: a satellite
    # at the zenith came out at 40 deg elevation and one on the horizon at
    # 28 deg.  That fed the a-priori troposphere (mapped as 1/sin(elev)),
    # the Klobuchar ionosphere, the elevation mask and the elevation-
    # dependent measurement weighting, biasing every pseudorange by metres
    # in a way that varied as satellites rose and set.
    R = np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])
    enu = R @ los
    e, n, u = enu
    horiz = math.sqrt(e * e + n * n)
    elev = math.degrees(math.atan2(u, horiz))
    az = math.degrees(math.atan2(e, n)) % 360.0
    return elev, az


def _parse_obs_date(obs_path: Path) -> tuple[int, int, int]:
    """Parse TIME OF FIRST OBS from RINEX obs header."""
    with open(obs_path, "r", errors="ignore") as f:
        for line in f:
            if "TIME OF FIRST OBS" in line:
                parts = line.split()
                year = int(parts[0])
                if year < 80:
                    year += 2000
                elif year < 100:
                    year += 1900
                return year, int(parts[1]), int(parts[2])
    # Fallback to today
    import datetime as dt
    t = dt.datetime.utcnow()
    return t.year, t.month, t.day


def parse_rinex_header_position(obs_path: Path) -> np.ndarray | None:
    """Parse APPROX POSITION XYZ from a RINEX 2.11 or 3.x obs header.

    Returns the ECEF position as a (3,) array, or None if the header
    line is absent.  This is the receiver's last known position and is
    typically more stable than a cold-start SPP solution (which can be
    100+ m off on the first epoch before the Kalman filter in rnx2rtkp
    converges).
    """
    with open(obs_path, "r", errors="ignore") as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if "APPROX POSITION XYZ" in line:
                parts = line.split()
                try:
                    return np.array([float(parts[0]), float(parts[1]),
                                     float(parts[2])])
                except (ValueError, IndexError):
                    return None
    return None


def _nav_system_for(path: Path) -> str:
    """Constellation implied by a convbin nav filename extension."""
    return {".nav": "G", ".lnav": "E", ".qnav": "J",
            ".cnav": "C", ".gnav": "R", ".hnav": "S"}.get(path.suffix.lower(), "G")


def _load_nav_files(nav_paths) -> tuple[dict, list | None, list | None]:
    """Merge broadcast ephemerides from one or more nav files.

    A multi-GNSS receiver tracks 19-32 satellites, but convbin splits the
    ephemerides across one file per constellation.  Loading only the GPS
    file (which is what this used to do) throws away most of the sky: on
    this rig's own capture it left 5 usable satellites out of 19, at a
    PDOP of 4.9, which no clock prior can rescue.

    Handles both RINEX 2.11 (per-constellation files: .nav, .lnav, .cnav)
    and RINEX 3.x (single mixed file with all constellations).
    """
    if isinstance(nav_paths, (str, Path)):
        nav_paths = [nav_paths]
    eph: dict = {}
    alpha = beta = None
    for p in nav_paths:
        p = Path(p)
        if not p.exists():
            continue
        # Detect RINEX version of nav file
        is_rinex3 = False
        try:
            with open(p, "r", errors="ignore") as f:
                first = f.readline()
            if "3.0" in first:
                is_rinex3 = True
        except Exception:
            pass
        if is_rinex3:
            try:
                e, a, b = _parse_rinex3_nav(p)
            except Exception:
                continue
            for k, v in e.items():
                eph.setdefault(k, []).extend(v)
            if alpha is None and a is not None:
                alpha, beta = a, b
            continue
        # RINEX 2.11 path
        sysc = _nav_system_for(p)
        if sysc == "R":
            # GLONASS needs the PZ-90 integration model, not Keplerian.
            continue
        try:
            e, a, b = _parse_rinex_nav(p, system=sysc)
        except Exception:
            continue
        for k, v in e.items():
            eph.setdefault(k, []).extend(v)
        if alpha is None and a is not None:
            alpha, beta = a, b
    return eph, alpha, beta


def get_measurements(obs_path: Path, nav_path: Path,
                     receiver_ecef: np.ndarray,
                     precise_products=None,
                     allow_gps_broadcast_fallback: bool = False,
                     only_sods: set | None = None) -> list[dict]:
    """Extract per-epoch measurement data for EKF.

    Returns list of dicts per epoch:
        {
            "sod": int,
            "pseudoranges": {sat_id: float},  # corrected for sat clock
            "sat_positions": {sat_id: np.ndarray(3)},  # ECEF
            "sat_los": np.ndarray(n, 3),  # unit LOS vectors
            "sat_ids": list[str],
            "n_sat": int,
            "freq_mode": "dual_l1l2" | "dual_l1l5" | "single_l1",
            "if_factor": float,  # IF noise amplification factor
        }

    Dual-frequency ionosphere-free combination is used automatically when
    both L1 and a second frequency (L2 or L5) code+phase are present.
    This removes ~99% of the ionospheric delay — the dominant error after
    the satellite clock — and is what makes cm-level carrier-phase
    positioning reachable on this hardware.

    Handles both RINEX 2.11 (C1/L1/P1/P2/L2) and RINEX 3.x (C1C/L1C/C5Q/
    L5Q/...).  RINEX 3.x is essential for the LC29H(BS) receiver, which
    tracks L1+L5 (GPS/Galileo) and B2I+B2a (BeiDou); the standard convbin
    outputs RINEX 2.11 which silently drops L5.
    """
    rinex_version = _detect_rinex_version(obs_path)
    if rinex_version == 3:
        return _get_measurements_rinex3(obs_path, nav_path, receiver_ecef,
                                        precise_products,
                                        allow_gps_broadcast_fallback,
                                        only_sods=only_sods)
    return _get_measurements_rinex2(obs_path, nav_path, receiver_ecef,
                                    precise_products,
                                    allow_gps_broadcast_fallback)


def _get_measurements_rinex2(obs_path: Path, nav_path: Path,
                             receiver_ecef: np.ndarray,
                             precise_products=None,
                             allow_gps_broadcast_fallback: bool = False) -> list[dict]:
    """RINEX 2.11 path (original implementation, preserved for archived data)."""
    obs_epochs = _parse_rinex_obs(obs_path)
    # nav_path may be a single file or a list, one per constellation.
    nav_eph, iono_alpha, iono_beta = _load_nav_files(nav_path)

    if not obs_epochs or not nav_eph:
        return []

    # Parse date from RINEX obs header (TIME OF FIRST OBS)
    obs_year, obs_month, obs_day = _parse_obs_date(obs_path)

    # Convert receiver ECEF to lat/lon for ionospheric model
    rx_lat, rx_lon = _ecef_to_latlon(receiver_ecef)

    # Detect the best dual-frequency pair available across the whole file.
    # The LC29H(BS) is L1+L5; legacy geodetic receivers are L1+L2.  We
    # check which second-frequency observations actually appear in the
    # data, not just the header — convbin lists all types the receiver
    # *can* output even when the actual obs values are blank.
    has_l2 = has_l5 = False
    for sats in obs_epochs.values():
        for obs in sats.values():
            if not has_l2:
                v = obs.get("P2") or obs.get("C2")
                if v is not None and np.isfinite(v) and v > 0:
                    has_l2 = True
            if not has_l5:
                v = obs.get("C5")
                if v is not None and np.isfinite(v) and v > 0:
                    has_l5 = True
            if has_l2 and has_l5:
                break
        if has_l2 and has_l5:
            break

    if has_l2:
        freq_mode, f2, f2_wavelen = "dual_l1l2", L2_FREQ, L2_WAVELEN
        code2_label = "P2"   # RINEX 2.11 L2 code (P-code or C/A on L2)
        code2_alt = "C2"
    elif has_l5:
        freq_mode, f2, f2_wavelen = "dual_l1l5", L5_FREQ, L5_WAVELEN
        code2_label = "C5"
        code2_alt = None
    else:
        # Single-frequency L1: use the GRAPHIC (code+phase)/2 combination
        # to remove the ionosphere.  This is the standard single-frequency
        # PPP observable and is essential for the LC29H receiver, whose
        # L1+L5 signal loses L5 through the standard RTKLIB convbin
        # (RINEX 2.11 has no L5 field).  Without GRAPHIC the residual
        # ionosphere after Klobuchar (~30-50%, 1-5 m) dominates and caps
        # accuracy at metre level; with GRAPHIC the first-order
        # ionosphere is removed entirely and float ambiguities can
        # converge to decimetre level.  The EKF's ambiguity state
        # absorbs the GRAPHIC effective ambiguity N*lambda/2.
        freq_mode, f2, f2_wavelen = "single_l1_graphic", None, None
        code2_label, code2_alt = None, None

    # IF noise amplification factor (for the EKF's measurement weighting).
    if f2 is not None:
        if_factor = L1_FREQ ** 2 / (L1_FREQ ** 2 - f2 ** 2)
    elif freq_mode == "single_l1_graphic":
        # GRAPHIC noise ≈ code_noise / sqrt(2), so the effective
        # amplification factor is 1/sqrt(2) ≈ 0.707 (noise REDUCTION,
        # not amplification).  The EKF multiplies code_std by if_factor,
        # so 0.707 correctly downweights the GRAPHIC observable's noise.
        if_factor = 1.0 / math.sqrt(2.0)
    else:
        if_factor = 1.0

    measurements = []
    _prev_sod_of_day = None
    _epoch_doy = getattr(precise_products, '_epoch_doy', None) if precise_products else None
    for sod in sorted(obs_epochs.keys()):
        sats = obs_epochs[sod]
        prs = {}
        sat_pos = {}
        sat_clk_corr = {}  # satellite clock correction (m) per sat
        tropo_corr = {}    # troposphere correction (m) per sat
        sat_ids = []

        # sod is absolute GPS seconds (week*604800 + SOW) from the
        # parser.  Broadcast ephemeris uses true SOW; daily SP3/CLK use
        # seconds-of-day; cached multi-day products use the absolute
        # axis directly.
        gps_sow = sod % 604800.0
        sod_of_day = sod % 86400

        # Detect UTC midnight crossing: when sod_of_day wraps backward,
        # increment the epoch DOY so precise products look up the correct
        # day's satellite clocks.  Without this, post-midnight epochs
        # silently receive the previous day's clocks (24h stale), which
        # drives the derived receiver clock from metres to hundreds of km.
        if _prev_sod_of_day is not None and sod_of_day < _prev_sod_of_day:
            _epoch_doy = (_epoch_doy + 1) if _epoch_doy is not None else None
            if precise_products is not None and hasattr(precise_products, 'set_epoch_doy'):
                precise_products.set_epoch_doy(_epoch_doy)
        _prev_sod_of_day = sod_of_day

        # Solid Earth tide correction.  Needs absolute GPS time for the
        # Sun/Moon ephemeris Julian date — plain SOW pinned it to Jan
        # 1980, so the tidal displacement was astronomically wrong.
        tide_disp = _solid_earth_tide(receiver_ecef, sod)
        rx_ecef_tide = receiver_ecef + tide_disp

        for sat_id, obs in sats.items():
            c1 = obs.get("C1")
            if c1 is None or not np.isfinite(c1) or c1 <= 0:
                continue

            # --- Dual-frequency ionosphere-free or single-frequency L1 ---
            # Pseudorange = rho + c*(dtr - dts) + iono + trop
            # IF corrected  = P_IF + c*dts_if - trop   (no iono, no TGD)
            # L1 corrected  = C1  + c*dts_l1 - trop - iono_klobuchar
            use_if = False
            if f2 is not None:
                p2_val = obs.get(code2_label)
                if p2_val is None and code2_alt:
                    p2_val = obs.get(code2_alt)
                if (p2_val is not None and np.isfinite(p2_val)
                        and p2_val > 0):
                    pr_raw = _iono_free_code(c1, p2_val, L1_FREQ, f2)
                    use_if = True

            if use_if:
                travel_time = pr_raw / C
            else:
                travel_time = c1 / C
            tx_sow = gps_sow - travel_time
            tx_sod = sod_of_day - travel_time  # seconds of day for SP3 lookup
            # Cached multi-day products use absolute GPS seconds as the
            # time index so they can serve a rolling capture crossing UTC
            # midnight AND the GPS week boundary; daily archive products
            # use seconds-of-day.  Select the index the PreciseProducts
            # instance expects.
            _pp_t = (sod - travel_time) if (precise_products is not None
                               and getattr(precise_products, "_use_sow", False)
                               ) else tx_sod

            # Use precise products when available; fall back to broadcast.
            # IMPORTANT: when the SP3 is extrapolated beyond its bounds
            # (common on the live rig: SP3 ends at midnight, observations
            # continue to 05:00+), the cubic spline extrapolation produces
            # positions with 1-3 km errors.  Broadcast ephemeris (~1-2 m)
            # is far more accurate in this case.  So we only use SP3
            # positions when within the interpolation bounds, and fall
            # back to broadcast for extrapolated epochs.
            use_precise_pos = (precise_products is not None
                               and precise_products.has_satellite(sat_id)
                               and precise_products.is_within_bounds(sat_id, _pp_t))
            if use_precise_pos:
                sat_pos_ecef = precise_products.satellite_position(sat_id, _pp_t)
                if sat_pos_ecef is None:
                    use_precise_pos = False

            if not use_precise_pos:
                if sat_id not in nav_eph:
                    continue
                eph_list = nav_eph[sat_id]
                eph = min(eph_list,
                          key=lambda e: abs(_sow_delta(e["toe_sod"],
                                                       gps_sow)))
                # Ephemeris validity check: GPS broadcast ephemeris is
                # nominally valid for ±2 h around TOE; reject satellites
                # whose closest TOE is >4 h away to prevent the EKF from
                # diverging on stale ephemeris (a known issue with RTCM3
                # streams that carry the next TOE, not the current one).
                # _sow_delta wraps at the week boundary so a TOE from the
                # tail of last week still matches an early-Sunday epoch.
                if abs(_sow_delta(eph["toe_sod"], gps_sow)) > 4 * 3600:
                    continue
                sat_pos_ecef = _satellite_position(eph, 0, tx_sow)

            # Satellite clock: prefer precise CLK, fall back to SP3 clock,
            # then broadcast.
            # IMPORTANT: when the CLK/SP3 is extrapolated beyond its bounds
            # (common on the live rig), the SP3 clock extrapolation has
            # ~50-100 m per-satellite errors.  GPS broadcast clocks (~1.5-3 m)
            # are far more accurate.  So for GPS, use broadcast clocks when
            # the precise clock is extrapolated.  For Galileo/BeiDou, the
            # SP3 clock extrapolation is still better than broadcast (which
            # has GGTO/DCB issues), so we keep the SP3 clock.
            use_precise_clk = (precise_products is not None
                               and (precise_products._clk is not None
                                    or getattr(precise_products, "_clk_splines", None)))
            if use_precise_clk:
                # Check if the clock is within bounds (not extrapolated)
                _clk_in_bounds = precise_products.is_clk_within_bounds(sat_id, _pp_t)
                if not _clk_in_bounds and sat_id[0] in ('G', 'E'):
                    # GPS/Galileo: use broadcast clock when precise clock
                    # is extrapolated.  GPS broadcast clocks are ~1.5-3m,
                    # Galileo broadcast clocks have GGTO (~4.5m, common-
                    # mode, absorbed by EKF receiver clock).  Both are far
                    # better than ~50-100m SP3 clock extrapolation errors.
                    use_precise_clk = False
                else:
                    sat_clk = precise_products.satellite_clock(sat_id, _pp_t)
                    if sat_clk is None:
                        use_precise_clk = False

            # When precise products are available but a satellite lacks
            # a precise clock, skip it rather than falling back to
            # broadcast.  Broadcast clocks for BeiDou (no precise CLK
            # in the GRG product) produce 2000-3500 m innovations
            # because the broadcast clock is referenced to a different
            # signal combination (B1I/B3I) than our IF (B1I/B2a), and
            # the DCB is not corrected.  GPS satellites without precise
            # clocks (e.g. G25 if missing from the CLK file) also have
            # 100-200 m broadcast clock errors.  Skipping these
            # satellites is better than corrupting the EKF with
            # metre-level satellite clock errors.
            # GPS is exempted: when the cached CLK table ends before the
            # current epoch (common on the live rig), skipping ALL GPS
            # satellites leaves the EKF with no measurements at all,
            # which is strictly worse than the 100-200 m per-satellite
            # broadcast clock error the median/MAD statistics absorb.
            # The exemption is opt-in: the EKF (default) keeps the skip
            # to avoid 100-200 m innovations; the OS Net direct-clock
            # path enables it because the median over many satellites
            # averages out per-satellite broadcast clock errors.
            if not use_precise_clk and precise_products is not None:
                # Allow GPS and Galileo broadcast clocks when the precise
                # clock is extrapolated (common on the live rig).
                # GPS broadcast clocks are accurate to ~1.5-3 m.
                # Galileo broadcast clocks have a GGTO offset (~4.5m) but
                # this is common-mode and absorbed by the EKF's receiver
                # clock state.  Both are far better than the ~50-100m
                # SP3 clock extrapolation errors.
                _clk_extrapolated = not precise_products.is_clk_within_bounds(
                    sat_id, _pp_t) if (precise_products._clk is not None
                                        or getattr(precise_products, "_clk_splines", None)) else True
                if _clk_extrapolated and sat_id[0] in ('G', 'E'):
                    pass  # Use broadcast clock
                elif not (allow_gps_broadcast_fallback and sat_id[0] == 'G'):
                    continue

            # Also skip satellites without precise positions when
            # precise products are available AND the SP3 covers the
            # observation time.  A satellite with a precise clock but
            # only broadcast position (e.g. BeiDou present in CLK but
            # absent from SP3) produces 1500-4000 m innovations from
            # broadcast orbit errors, polluting the EKF's median/MAD
            # statistics and driving clock divergence.
            # GPS broadcast positions (~1-2 m) are an acceptable fallback
            # for the same reason as the clock fallback above.
            # BUT: when the SP3 is extrapolated beyond its bounds (common
            # on the live rig), broadcast positions (~1-5 m) are far
            # better than SP3 extrapolation (~1-3 km), so we allow
            # broadcast positions for ALL satellites in that case.
            if not use_precise_pos and precise_products is not None:
                _sp3_extrapolated = not precise_products.is_within_bounds(
                    sat_id, _pp_t) if precise_products.has_satellite(sat_id) else True
                if _sp3_extrapolated and sat_id[0] in ('G', 'E'):
                    pass  # Use broadcast positions for GPS/Galileo
                elif _sp3_extrapolated and sat_id[0] == 'C':
                    continue  # BeiDou broadcast positions have ~2km errors
                elif not (allow_gps_broadcast_fallback and sat_id[0] == 'G'):
                    continue

            if not use_precise_clk:
                if sat_id not in nav_eph:
                    continue
                if not use_precise_pos:
                    # eph already selected above
                    pass
                else:
                    eph_list = nav_eph[sat_id]
                    eph = min(eph_list,
                              key=lambda e: abs(_sow_delta(e["toe_sod"],
                                                           gps_sow)))
                    if abs(_sow_delta(eph["toe_sod"], gps_sow)) > 4 * 3600:
                        continue
                if use_if:
                    sat_clk = _sat_clock_correction_if(eph, tx_sow)
                else:
                    sat_clk = _sat_clock_correction(eph, tx_sow)

            # Sagnac (Earth rotation) correction
            omega_earth = 7.2921151467e-5
            theta = omega_earth * travel_time
            R_sagnac = np.array([
                [np.cos(theta),  np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0,              0,             1],
            ])
            sat_pos_ecef = R_sagnac @ sat_pos_ecef
            pr_corrected = pr_raw + C * sat_clk if use_if else c1 + C * sat_clk

            # Tropospheric correction (always needed)
            rx_to_sat = sat_pos_ecef - rx_ecef_tide
            geom_range = np.linalg.norm(rx_to_sat)
            los = rx_to_sat / geom_range
            elev_deg, az_deg = _los_to_elev_az(los, rx_lat, rx_lon)
            tropo = _tropo_correction(elev_deg)
            pr_corrected -= tropo

            # Ionospheric correction: only for single-frequency L1 code.
            # The IF combination has already removed it.  The GRAPHIC
            # phase observable (built below) is ionosphere-free by
            # construction (code+phase cancels iono), so no Klobuchar
            # is applied to it; the code observation still gets
            # Klobuchar (partial iono removal), and the EKF's ambiguity
            # state absorbs the Klobuchar residual so the position
            # converges to the ionosphere-free solution.
            # Note: Klobuchar is a GPS broadcast model; applying it to
            # Galileo/BeiDou is an approximation. The iono delay is
            # dispersive and common across L-band, so the model is
            # physically applicable, but the broadcast coefficients are
            # GPS-specific. For non-GPS constellations without a system-
            # specific iono model, this is the best available option.
            if not use_if and iono_alpha is not None and iono_beta is not None:
                iono = _klobuchar_iono(rx_lat, rx_lon, az_deg, elev_deg,
                                        gps_sow, iono_alpha, iono_beta)
                pr_corrected -= iono

            prs[sat_id] = pr_corrected
            sat_pos[sat_id] = sat_pos_ecef
            # Store corrections needed for the phase observable.
            # The phase needs the same satellite clock and troposphere
            # corrections as the pseudorange, otherwise the EKF's phase
            # innovation includes uncorrected satellite clock errors
            # (~10-20 m with ultra-rapid products) and troposphere
            # (~2-20 m), preventing ambiguity convergence.
            if not hasattr(measurements, '_sat_clk_corr'):
                pass  # will use local dicts below
            sat_clk_corr[sat_id] = C * sat_clk
            tropo_corr[sat_id] = tropo
            sat_ids.append(sat_id)

        if len(sat_ids) < 4:
            continue

        los_vectors = []
        for sid in sat_ids:
            sp = sat_pos[sid]
            rx_to_sat = sp - rx_ecef_tide
            norm = np.linalg.norm(rx_to_sat)
            if norm < 1e-3:
                los_vectors.append(np.zeros(3))
            else:
                los_vectors.append(rx_to_sat / norm)

        # Build phase observations.
        # Dual-frequency: ionosphere-free phase L_IF (in metres).
        # Single-frequency GRAPHIC: (C1 + L1*lambda) / 2 (ionosphere-free,
        #   with effective ambiguity N*lambda/2 absorbed by the EKF's
        #   float ambiguity state).  This is the standard single-frequency
        #   PPP observable and is what enables decimetre-level positioning
        #   on L1-only data where the ionosphere would otherwise dominate.
        #   The GRAPHIC phase noise is code-level (~0.3-3 m / sqrt(2)),
        #   NOT millimetre-level like dual-frequency IF phase — the EKF
        #   must weight it accordingly (handled in gps_web.py via
        #   freq_mode detection).
        phases = {}
        for sid in sat_ids:
            obs = sats.get(sid, {})
            l1 = obs.get("L1")
            l1_ok = (l1 is not None and np.isfinite(l1) and abs(l1) > 0)
            if f2 is not None and l1_ok:
                l2 = obs.get("L2") if freq_mode == "dual_l1l2" else None
                l5 = obs.get("L5") if freq_mode == "dual_l1l5" else None
                l2_val = l2 if freq_mode == "dual_l1l2" else l5
                if (l2_val is not None and np.isfinite(l2_val)
                        and abs(l2_val) > 0):
                    l1_m = l1 * L1_WAVELEN
                    l2_m = l2_val * f2_wavelen
                    ph_if = _iono_free_phase(l1_m, l2_m, L1_FREQ, f2)
                    # Apply the same satellite clock and troposphere
                    # corrections as the pseudorange.  Without these
                    # the phase innovation includes ~10-20 m of
                    # uncorrected satellite clock error and ~2-20 m
                    # of troposphere, preventing ambiguity convergence.
                    ph_if += sat_clk_corr.get(sid, 0.0)
                    ph_if -= tropo_corr.get(sid, 0.0)
                    phases[sid] = ph_if
                    continue
            if freq_mode == "single_l1_graphic" and l1_ok:
                # GRAPHIC phase observable: (C1 + L1*lambda) / 2 with sat
                # clock and tropo corrections.  Built from the already-
                # corrected code pseudorange plus half the code-phase
                # difference:  prs[sid] + (L1_m - C1)/2.  The Klobuchar
                # iono residual remaining in prs[sid] is a slowly-varying
                # constant absorbed by the EKF's float ambiguity state, so
                # the position converges to the ionosphere-free solution.
                c1_raw = obs.get("C1")
                if c1_raw is not None and np.isfinite(c1_raw) and c1_raw > 0:
                    l1_m = l1 * L1_WAVELEN
                    phases[sid] = prs[sid] + 0.5 * (l1_m - c1_raw)
                    continue
            if l1_ok:
                phases[sid] = l1 * L1_WAVELEN

        measurements.append({
            "sod": sod,
            "pseudoranges": prs,
            "phases": phases,
            "sat_positions": sat_pos,
            "sat_los": np.array(los_vectors),
            "sat_ids": sat_ids,
            "n_sat": len(sat_ids),
            "freq_mode": freq_mode,
            "if_factor": round(if_factor, 4),
            "slips": set(),
        })

    return measurements


# ---------------------------------------------------------------------------
# RINEX 3.x dual-frequency measurement extraction
# ---------------------------------------------------------------------------

# Per-constellation dual-frequency strategy.  For each system, the list is
# ordered by preference: the first pair for which both code and phase are
# present in the actual data is used.  BeiDou B2I+B2a is excluded — the
# frequencies are only 2.6% apart, giving a ~27x noise amplification in the
# iono-free combination, which is worse than single-frequency + Klobuchar.
# BeiDou B1I+B2I would be good but the LC29H(BS) does not track B1I.
_DUAL_FREQ_PAIRS: dict[str, list[tuple[str, str, float, float, str, str]]] = {
    # system: [(code1, phase1, f1, f1_wavelen, code2, phase2, f2, f2_wavelen), ...]
    # GPS: L1+L5 preferred, L1+L2 as fallback
    "G": [("C1C", "L1C", L1_FREQ, L1_WAVELEN, "C5Q", "L5Q", L5_FREQ, L5_WAVELEN),
          ("C1C", "L1C", L1_FREQ, L1_WAVELEN, "C5I", "L5I", L5_FREQ, L5_WAVELEN),
          ("C1C", "L1C", L1_FREQ, L1_WAVELEN, "C5X", "L5X", L5_FREQ, L5_WAVELEN),
          ("C1W", "L1W", L1_FREQ, L1_WAVELEN, "C2W", "L2W", L2_FREQ, L2_WAVELEN),
          ("C1C", "L1C", L1_FREQ, L1_WAVELEN, "C2L", "L2L", L2_FREQ, L2_WAVELEN),
          ("C1C", "L1C", L1_FREQ, L1_WAVELEN, "C2X", "L2X", L2_FREQ, L2_WAVELEN)],
    # Galileo: E1+E5a preferred, E1+E5b as fallback
    "E": [("C1C", "L1C", E1_FREQ, C / E1_FREQ, "C5Q", "L5Q", E5A_FREQ, C / E5A_FREQ),
          ("C1C", "L1C", E1_FREQ, C / E1_FREQ, "C5I", "L5I", E5A_FREQ, C / E5A_FREQ),
          ("C1C", "L1C", E1_FREQ, C / E1_FREQ, "C5X", "L5X", E5A_FREQ, C / E5A_FREQ),
          ("C1X", "L1X", E1_FREQ, C / E1_FREQ, "C7Q", "L7Q", E5B_FREQ, C / E5B_FREQ),
          ("C1C", "L1C", E1_FREQ, C / E1_FREQ, "C7I", "L7I", E5B_FREQ, C / E5B_FREQ)],
    # BeiDou: B1I (1561.098 MHz, RINEX code "2I") + B2a (1176.45 MHz,
    # RINEX code "5P").  The IF noise amplification is ~2.66, comparable
    # to GPS L1/L5 (2.59) and better than L1/L2 (3.0).
    # Note: RINEX attribute "2I" for BeiDou is B1I at 1561.098 MHz, NOT
    # B2I at 1207.14 MHz (which is attribute "7I").
    "C": [("C2I", "L2I", B1I_FREQ, C / B1I_FREQ,
           "C5P", "L5P", B2A_FREQ, C / B2A_FREQ),
          ("C2I", "L2I", B1I_FREQ, C / B1I_FREQ,
           "C5X", "L5X", B2A_FREQ, C / B2A_FREQ),
          ("C1I", "L1I", B1I_FREQ, C / B1I_FREQ,
           "C5P", "L5P", B2A_FREQ, C / B2A_FREQ)],
    # GLONASS: L1 only on this receiver
    "R": [],
}


def _select_freq_pair(sys_char: str, obs: dict
                      ) -> tuple[str, str, float, float, str, str, float, float] | None:
    """Select the best dual-frequency pair for a satellite's observations.

    Returns (code1, phase1, f1, f1_wavelen, code2, phase2, f2, f2_wavelen)
    or None if no dual-frequency pair is available.
    """
    pairs = _DUAL_FREQ_PAIRS.get(sys_char, [])
    for p in pairs:
        code1, phase1 = p[0], p[1]
        code2, phase2 = p[4], p[5]
        c1v = obs.get(code1)
        c2v = obs.get(code2)
        if (c1v is not None and np.isfinite(c1v) and c1v > 0
                and c2v is not None and np.isfinite(c2v) and c2v > 0):
            return p
    return None


def _get_measurements_rinex3(obs_path: Path, nav_path: Path,
                             receiver_ecef: np.ndarray,
                             precise_products=None,
                             allow_gps_broadcast_fallback: bool = False,
                             only_sods: set | None = None) -> list[dict]:
    """RINEX 3.x dual-frequency measurement extraction.

    This is the path for the LC29H(BS) receiver, which outputs L1+L5
    (GPS/Galileo) and B2I+B2a (BeiDou) via RTCM3 MSM7 messages.  The
    standard convbin drops L5 by outputting RINEX 2.11; convbin_demo5
    with RINEX 3.x preserves it, and this function uses it to form the
    ionosphere-free combination that removes ~99% of ionospheric delay.

    When ``precise_products`` (a PreciseProducts instance from
    ``scripts.lib.precise_products``) is provided, satellite positions
    and clock corrections are taken from SP3+CLK files instead of
    broadcast ephemeris, reducing orbit error from ~1-2 m to ~2-5 cm
    and clock error from ~5-10 ns to ~0.1-0.3 ns.
    """
    obs_epochs = _parse_rinex3_obs(obs_path)
    nav_eph, iono_alpha, iono_beta = _load_nav_files(nav_path)

    if only_sods is not None:
        # Compute only the epochs the caller lacks (per-epoch extraction
        # cache): each epoch's result is deterministic in (sod, obs,
        # nav, products, receiver_ecef) so skipping is safe.  The whole-
        # file parse still runs — that is the cheap half; the per-epoch
        # geometry/clock evaluation below is the ~20min part.
        obs_epochs = {k: v for k, v in obs_epochs.items()
                      if k in only_sods}

    if not obs_epochs:
        return []
    # Broadcast ephemeris is still needed as a fallback for satellites
    # not in the precise products (e.g. newly launched SVs).
    if not nav_eph and precise_products is None:
        return []

    obs_year, obs_month, obs_day = _parse_obs_date(obs_path)
    rx_lat, rx_lon = _ecef_to_latlon(receiver_ecef)

    # Detect the dominant dual-frequency mode across the whole file for
    # reporting purposes.  Per-satellite selection still happens in the loop.
    global_mode = "single_l1"
    global_if_factor = 1.0
    for sats in list(obs_epochs.values())[:20]:
        for sid, obs in sats.items():
            pair = _select_freq_pair(sid[0], obs)
            if pair is not None:
                f1, f2 = pair[2], pair[6]
                global_mode = "dual_l1l5" if abs(f2 - L5_FREQ) < 1e6 else "dual_l1l2"
                global_if_factor = f1 ** 2 / (f1 ** 2 - f2 ** 2)
                break
        if global_mode != "single_l1":
            break
    if global_mode == "single_l1":
        # No dual-frequency pair found: use GRAPHIC (code+phase)/2 to
        # remove the ionosphere on single-frequency observations.
        global_mode = "single_l1_graphic"
        global_if_factor = 1.0 / math.sqrt(2.0)

    measurements = []
    _prev_sod_of_day = None
    _epoch_doy = getattr(precise_products, '_epoch_doy', None) if precise_products else None
    for sod in sorted(obs_epochs.keys()):
        sats = obs_epochs[sod]
        prs = {}
        sat_pos = {}
        sat_ids = []
        sat_freqs = {}  # per-satellite frequency info for phase processing

        # sod is absolute GPS seconds (week*604800 + SOW) from the
        # parser.  Broadcast ephemeris uses true SOW; daily SP3/CLK use
        # seconds-of-day; cached multi-day products use the absolute
        # axis directly.
        gps_sow = sod % 604800.0
        sod_of_day = sod % 86400

        # Detect UTC midnight crossing (see comment in the first path).
        if only_sods is not None:
            # With a filtered epoch set the crossing heuristic can miss a
            # midnight that fell inside the skipped range — derive the day
            # straight from absolute GPS seconds instead (sod carries the
            # week number, so the calendar date is exact).
            _epoch_doy = _doy_from_gps_abs(sod)
            if precise_products is not None and hasattr(precise_products, 'set_epoch_doy'):
                precise_products.set_epoch_doy(_epoch_doy)
        elif _prev_sod_of_day is not None and sod_of_day < _prev_sod_of_day:
            _epoch_doy = (_epoch_doy + 1) if _epoch_doy is not None else None
            if precise_products is not None and hasattr(precise_products, 'set_epoch_doy'):
                precise_products.set_epoch_doy(_epoch_doy)
        _prev_sod_of_day = sod_of_day

        # Solid Earth tide correction: the receiver position is displaced
        # by up to 30cm by the lunar/solar tide.  Apply the correction to
        # the reference position so all geometric ranges are computed
        # from the tide-corrected position.  Needs absolute GPS time —
        # plain SOW pinned the ephemeris Julian date to Jan 1980.
        tide_disp = _solid_earth_tide(receiver_ecef, sod)
        rx_ecef_tide = receiver_ecef + tide_disp

        for sat_id, obs in sats.items():
            sys_char = sat_id[0]

            # Select the primary code: prefer the code from the best
            # dual-frequency pair; fall back to any available code.
            pair = _select_freq_pair(sys_char, obs)
            if pair is not None:
                code1, phase1, f1, f1_wl, code2, phase2, f2, f2_wl = pair
                c1_val = obs.get(code1)
                c2_val = obs.get(code2)
                use_if = True
            else:
                # Single-frequency: use the first available code
                code1 = None
                for cc in ("C1C", "C1W", "C1Y", "C1X", "C1I", "C2I", "C1P"):
                    v = obs.get(cc)
                    if v is not None and np.isfinite(v) and v > 0:
                        code1 = cc
                        c1_val = v
                        break
                if code1 is None:
                    continue
                c2_val = None
                use_if = False
                f1 = _freq_for_rinex3_code(sys_char, code1) or L1_FREQ
                f1_wl = C / f1
                phase1 = "L" + code1[1:]  # C1C -> L1C
                f2, f2_wl = None, None

            if use_if:
                pr_raw = _iono_free_code(c1_val, c2_val, f1, f2)
                travel_time = pr_raw / C
            else:
                travel_time = c1_val / C
            tx_sow = gps_sow - travel_time
            tx_sod = sod_of_day - travel_time  # seconds of day for SP3 lookup
            # Cached multi-day products use absolute GPS seconds as the
            # time index so they can serve a rolling capture crossing UTC
            # midnight AND the GPS week boundary; daily archive products
            # use seconds-of-day.  Select the index the PreciseProducts
            # instance expects.
            _pp_t = (sod - travel_time) if (precise_products is not None
                               and getattr(precise_products, "_use_sow", False)
                               ) else tx_sod

            # Use precise products when available; fall back to broadcast.
            # IMPORTANT: when the SP3 is extrapolated beyond its bounds
            # (common on the live rig: SP3 ends at midnight, observations
            # continue to 05:00+), the cubic spline extrapolation produces
            # positions with 1-3 km errors.  Broadcast ephemeris (~1-2 m)
            # is far more accurate in this case.  So we only use SP3
            # positions when within the interpolation bounds, and fall
            # back to broadcast for extrapolated epochs.
            use_precise_pos = (precise_products is not None
                               and precise_products.has_satellite(sat_id)
                               and precise_products.is_within_bounds(sat_id, _pp_t))
            if use_precise_pos:
                sat_pos_ecef = precise_products.satellite_position(sat_id, _pp_t)
                if sat_pos_ecef is None:
                    use_precise_pos = False

            # Satellite clock: prefer precise CLK, fall back to SP3 clock,
            # then broadcast.
            # IMPORTANT: when the CLK/SP3 is extrapolated beyond its bounds
            # (common on the live rig), the SP3 clock extrapolation has
            # ~50-100 m per-satellite errors.  GPS broadcast clocks (~1.5-3 m)
            # are far more accurate.  So for GPS, use broadcast clocks when
            # the precise clock is extrapolated.  For Galileo/BeiDou, the
            # SP3 clock extrapolation is still better than broadcast (which
            # has GGTO/DCB issues), so we keep the SP3 clock.
            use_precise_clk = (precise_products is not None
                               and (precise_products._clk is not None
                                    or getattr(precise_products, "_clk_splines", None)))
            if use_precise_clk:
                # Check if the clock is within bounds (not extrapolated)
                _clk_in_bounds = precise_products.is_clk_within_bounds(sat_id, _pp_t)
                if not _clk_in_bounds and sat_id[0] in ('G', 'E'):
                    # GPS/Galileo: use broadcast clock when precise clock
                    # is extrapolated.  GPS broadcast clocks are ~1.5-3m,
                    # Galileo broadcast clocks have GGTO (~4.5m, common-
                    # mode, absorbed by EKF receiver clock).  Both are far
                    # better than ~50-100m SP3 clock extrapolation errors.
                    use_precise_clk = False
                else:
                    sat_clk = precise_products.satellite_clock(sat_id, _pp_t)
                    if sat_clk is None:
                        use_precise_clk = False

            # Skip satellites without precise clocks when precise products
            # are available (broadcast clocks for BeiDou produce 2000+ m
            # errors due to signal-combination mismatch; Galileo broadcast
            # clocks are referenced to GST not GPST and the GGTO handling
            # is incomplete, causing the EKF clock to diverge to ~-91 km).
            # GPS broadcast clocks are accurate enough (~5-10 ns, ~1.5-3 m)
            # to use as a fallback when the precise CLK table doesn't cover
            # this epoch (e.g. cached products end before the current hour).
            # The exemption is opt-in: the EKF keeps the skip to avoid
            # 100-200 m innovations; the OS Net direct-clock path enables
            # it because the median over many satellites averages out
            # per-satellite broadcast clock errors.
            if not use_precise_clk and precise_products is not None:
                # Allow GPS and Galileo broadcast clocks when the precise
                # clock is extrapolated (common on the live rig).
                # GPS broadcast clocks are accurate to ~1.5-3 m.
                # Galileo broadcast clocks have a GGTO offset (~4.5m) but
                # this is common-mode and absorbed by the EKF's receiver
                # clock state.  Both are far better than the ~50-100m
                # SP3 clock extrapolation errors.
                _clk_extrapolated = not precise_products.is_clk_within_bounds(
                    sat_id, _pp_t) if (precise_products._clk is not None
                                        or getattr(precise_products, "_clk_splines", None)) else True
                if _clk_extrapolated and sat_id[0] in ('G', 'E'):
                    pass  # Use broadcast clock
                elif not (allow_gps_broadcast_fallback and sat_id[0] == 'G'):
                    continue

            # Skip satellites without precise positions when precise
            # products are available AND the SP3 covers the observation
            # time.  BeiDou satellites (C-prefix) have precise clocks in
            # the CLK file but are absent from the SP3 file, so they fall
            # back to broadcast ephemeris for position while using a
            # precise clock.  This mixed precise-clock / broadcast-position
            # combination produces 1500-4000 m innovations (broadcast
            # BeiDou orbit errors are ~2 km) that pollute the EKF's
            # median/MAD statistics and drive clock divergence.
            # Requiring BOTH precise position AND precise clock ensures
            # only fully-precise satellites enter the EKF.
            # BUT: when the SP3 is extrapolated beyond its bounds (common
            # on the live rig), broadcast positions (~1-5 m) are far
            # better than SP3 extrapolation (~1-3 km), so we allow
            # broadcast positions for ALL satellites in that case.
            if not use_precise_pos and precise_products is not None:
                _sp3_extrapolated = not precise_products.is_within_bounds(
                    sat_id, _pp_t) if precise_products.has_satellite(sat_id) else True
                if _sp3_extrapolated and sat_id[0] in ('G', 'E'):
                    pass  # Use broadcast positions for GPS/Galileo
                elif _sp3_extrapolated and sat_id[0] == 'C':
                    continue  # BeiDou broadcast positions have ~2km errors
                elif not (allow_gps_broadcast_fallback and sat_id[0] == 'G'):
                    continue

            # If not using precise, need broadcast ephemeris
            eph = None
            if not use_precise_pos or not use_precise_clk:
                if sat_id not in nav_eph:
                    continue
                eph_list = nav_eph[sat_id]
                eph = min(eph_list,
                          key=lambda e: abs(_sow_delta(e["toe_sod"],
                                                       gps_sow)))
                # Ephemeris validity check (same as RINEX 2.11 path):
                # reject satellites whose closest TOE is >4 h away.
                # _sow_delta wraps at the week boundary so a TOE from the
                # tail of last week still matches an early-Sunday epoch.
                if abs(_sow_delta(eph["toe_sod"], gps_sow)) > 4 * 3600:
                    continue

            if not use_precise_pos:
                sat_pos_ecef = _satellite_position(eph, 0, tx_sow)

            if not use_precise_clk:
                if use_if:
                    sat_clk = _sat_clock_correction_if(eph, tx_sow)
                else:
                    sat_clk = _sat_clock_correction(eph, tx_sow)

            if use_if:
                pr_corrected = pr_raw + C * sat_clk
            else:
                pr_corrected = c1_val + C * sat_clk

            # Sagnac correction
            omega_earth = 7.2921151467e-5
            theta = omega_earth * travel_time
            R_sagnac = np.array([
                [np.cos(theta),  np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0,              0,             1],
            ])
            sat_pos_ecef = R_sagnac @ sat_pos_ecef

            # Tropospheric correction
            rx_to_sat = sat_pos_ecef - rx_ecef_tide
            geom_range = np.linalg.norm(rx_to_sat)
            if geom_range < 1e-3:
                continue
            los = rx_to_sat / geom_range
            elev_deg, az_deg = _los_to_elev_az(los, rx_lat, rx_lon)
            tropo = _tropo_correction(elev_deg)
            pr_corrected -= tropo

            # Ionospheric correction: only for single-frequency
            if not use_if and iono_alpha is not None and iono_beta is not None:
                iono = _klobuchar_iono(rx_lat, rx_lon, az_deg, elev_deg,
                                        gps_sow, iono_alpha, iono_beta)
                pr_corrected -= iono

            prs[sat_id] = pr_corrected
            sat_pos[sat_id] = sat_pos_ecef
            sat_ids.append(sat_id)
            # Store corrections to apply to phase observations.
            # The phase must receive the SAME satellite clock and troposphere
            # corrections as the code; otherwise the float ambiguity state
            # must absorb a ~100-300 km satellite clock bias that drifts
            # over time, making convergence impossible.
            # For single-frequency, the iono correction has OPPOSITE sign
            # on phase (phase is advanced, code is delayed).
            phase_corr = C * sat_clk - tropo
            if not use_if and iono_alpha is not None and iono_beta is not None:
                phase_corr += iono  # add iono back for phase

            # Carrier-phase wind-up correction (Wu et al. 1993).
            # The wind-up is a rotation of the phase due to changing
            # antenna polarisation orientation.  It reaches several cm
            # for rotating satellites and must be corrected for cm-level
            # PPP.  The correction is in cycles; convert to metres using
            # the IF wavelength.
            windup_cycles = _windup_correction(sat_pos_ecef, rx_ecef_tide)
            if use_if and f1 is not None and f2 is not None:
                # Ionosphere-free wavelength.  The IF phase is
                #   L_IF = (f1^2 * L1 - f2^2 * L2) / (f1^2 - f2^2)
                # so one cycle of the IF observable corresponds to
                #   lambda_IF = c * (f1^2 - f2^2) / (f1^2 * f2 - f2^2 * f1)
                # The earlier approximation c/(f1+f2) was 4x too small
                # (0.109 m vs 0.445 m for L1/L5), under-correcting the
                # wind-up by the same factor — a several-cm error at
                # cm-level convergence.
                wl_if = C * (f1**2 - f2**2) / (f1**2 * f2 - f2**2 * f1)
                phase_corr += windup_cycles * wl_if
            elif f1 is not None:
                phase_corr += windup_cycles * f1_wl

            sat_freqs[sat_id] = {
                "use_if": use_if, "f1": f1, "f1_wl": f1_wl,
                "f2": f2, "f2_wl": f2_wl,
                "phase1": phase1, "phase2": phase2 if pair else None,
                "phase_corr": phase_corr,
            }

        if len(sat_ids) < 4:
            continue

        los_vectors = []
        for sid in sat_ids:
            sp = sat_pos[sid]
            rx_to_sat = sp - rx_ecef_tide
            norm = np.linalg.norm(rx_to_sat)
            if norm < 1e-3:
                los_vectors.append(np.zeros(3))
            else:
                los_vectors.append(rx_to_sat / norm)

        # Build phase observations (iono-free where dual-frequency).
        # Apply the SAME satellite clock and troposphere corrections as
        # the code, so the float ambiguity only needs to absorb the
        # constant integer ambiguity + hardware delay (not the drifting
        # satellite clock bias of ~100-300 km).
        phases = {}
        for sid in sat_ids:
            obs = sats.get(sid, {})
            fi = sat_freqs.get(sid, {})
            p1_code = fi.get("phase1")
            l1 = obs.get(p1_code) if p1_code else None
            l1_ok = (l1 is not None and np.isfinite(l1) and abs(l1) > 0)
            phase_corr = fi.get("phase_corr", 0.0)

            if fi.get("use_if") and l1_ok:
                p2_code = fi.get("phase2")
                l2 = obs.get(p2_code) if p2_code else None
                if (l2 is not None and np.isfinite(l2) and abs(l2) > 0):
                    f1, f2 = fi["f1"], fi["f2"]
                    l1_m = l1 * fi["f1_wl"]
                    l2_m = l2 * fi["f2_wl"]
                    ph_if = _iono_free_phase(l1_m, l2_m, f1, f2)
                    phases[sid] = ph_if + phase_corr
                    continue
            if (global_mode == "single_l1_graphic" or not fi.get("use_if")) and l1_ok:
                # GRAPHIC: (code + phase) / 2, ionosphere-free.
                # Built from the corrected code pseudorange plus half the
                # code-phase difference; the Klobuchar residual is absorbed
                # by the float ambiguity state.
                c1_raw = obs.get(fi.get("phase1", "").replace("L", "C"))
                if c1_raw is None:
                    # Fall back to any code obs
                    for cc in ("C1C", "C1W", "C1Y", "C1X", "C1I", "C2I", "C1P"):
                        v = obs.get(cc)
                        if v is not None and np.isfinite(v) and v > 0:
                            c1_raw = v
                            break
                if c1_raw is not None and np.isfinite(c1_raw) and c1_raw > 0:
                    l1_m = l1 * fi.get("f1_wl", L1_WAVELEN)
                    phases[sid] = prs[sid] + 0.5 * (l1_m - c1_raw)
                    continue
            if l1_ok:
                phases[sid] = l1 * fi.get("f1_wl", L1_WAVELEN) + phase_corr

        # Collect LLI-based cycle slips for this epoch.  The RINEX parser
        # stores {phase_code}_LLI = 1.0 when bit 0 (loss of lock) is set.
        # Bit 1 (half-cycle ambiguity) is not a slip and is not recorded.
        epoch_slips = set()
        for sat_id, obs in sats.items():
            for k, v in obs.items():
                if k.endswith("_LLI") and v == 1.0:
                    epoch_slips.add(sat_id)
                    break

        measurements.append({
            "sod": sod,
            "pseudoranges": prs,
            "phases": phases,
            "sat_positions": sat_pos,
            "sat_los": np.array(los_vectors),
            "sat_ids": sat_ids,
            "n_sat": len(sat_ids),
            "freq_mode": global_mode,
            "if_factor": round(global_if_factor, 4),
            # Per-satellite frequency info: the EKF uses this to apply
            # correct noise weighting for IF vs GRAPHIC observations when
            # the global mode is dual-frequency but some satellites only
            # have L1 (tracked by GRAPHIC).  Without this, the EKF would
            # apply IF noise amplification to GRAPHIC observations,
            # underweighting them by ~3x.
            "sat_use_if": {sid: fi.get("use_if", False)
                           for sid, fi in sat_freqs.items()},
            # LLI-flagged cycle slips for this epoch (bit 0 only).
            "slips": epoch_slips,
        })

    return measurements
