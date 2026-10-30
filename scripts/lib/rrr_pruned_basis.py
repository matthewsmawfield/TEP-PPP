"""Pruned low-frequency spatial feature basis for the clean-origin RRR arm.

Replaces the ~5,990-column multi-resolution per-station basis with a
two-tier, physics-projected basis (v0.2 review proposal):

  Tier 1 (16 macro-modes): station innovations projected through the
  validated physical operators -- three-band kriging scales
  (4,677 / 1,502 / 1,459 km), Legendre distance rings, azimuthal
  dipole/quadrupole modes, the instantaneous CMB-dipole axis
  projection, oscillator-screening split, macro velocity, sidereal
  echo, and a causal drift-gate residual.

  Tier 2 (2 x N_sta): per-station causal LF/MF innovation residuals
  orthogonalised against the Tier-1 kriging modes, so individual
  station columns model local departures only.

All features are strictly causal (single-pole exponential low-passes,
anchored innovation levels) and restricted to the validated TEP band
(characteristic smoothing >= 900 s): >200 uHz columns are eliminated
by construction.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
SIDEREAL_DAY_S = 86164.0905

# CMB dipole apex (COBE/WMAP): l = 263.99 deg, b = 48.26 deg galactic,
# RA ~167.94 deg, Dec ~-6.94 deg equatorial (J2000).
_CMB_RA_DEG = 167.942
_CMB_DEC_DEG = -6.943


def causal_exp_smooth(series: np.ndarray, tau_sec: float,
                      dt: float = 1.0) -> np.ndarray:
    """Strictly causal single-pole exponential low-pass filter.

    Anchored to zero at h = 0 so all features represent accumulated
    innovation relative to the window boundary.  `series` may be
    (T,) or (T, N).
    """
    from scipy.signal import lfilter
    x = np.asarray(series, dtype=np.float64)
    dc = x - x[0:1]
    alpha = dt / (tau_sec + dt)
    # y[t] = (1-alpha)*y[t-1] + alpha*x[t], y[0]=0 — identical to the
    # explicit loop, vectorised over columns by lfilter.
    return lfilter([alpha], [1.0, -(1.0 - alpha)], dc, axis=0)


def haversine_km(lat1: float, lon1: float, lat2: float,
                 lon2: float) -> float:
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dla, dlo = la2 - la1, lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def midpoint_bearing_rad(lat1: float, lon1: float, lat2: float,
                       lon2: float) -> float:
    """Bearing of the great-circle path measured at its midpoint --
    the Step 74/82-corrected convention."""
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    # Midpoint of the great-circle segment
    bx = np.cos(la2) * np.cos(lo2 - lo1)
    by = np.cos(la2) * np.sin(lo2 - lo1)
    lam = np.arctan2(np.sin(la1) + np.sin(la2),
                     np.sqrt((np.cos(la1) + bx) ** 2 + by ** 2))
    l_m = lo1 + np.arctan2(by, np.cos(la1) + bx)
    # Bearing from rover to midpoint evaluated AT the midpoint
    # (equivalent to initial bearing rover->midpoint).
    dlo = l_m - lo1
    return float(np.arctan2(np.sin(dlo) * np.cos(lam),
                            np.cos(la1) * np.sin(lam)
                            - np.sin(la1) * np.cos(lam) * np.cos(dlo)))


def gst_rad(unix_s: float) -> float:
    """Greenwich Sidereal Time (radians) from Unix seconds."""
    jd = unix_s / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * T * T)
    return float(np.radians(theta % 360.0))


def cmb_proj_cos2(rover_lat: float, rover_lon: float,
                  sta_lat: np.ndarray, sta_lon: np.ndarray,
                  unix_t: np.ndarray) -> np.ndarray:
    """(b_hat . n_CMB)^2 per station per epoch.

    b_hat is the ECEF rover->station unit vector; n_CMB is the dipole
    apex direction rotated into ECEF at epoch GST.  Returns (T, N).
    """
    la_r, lo_r = np.radians(rover_lat), np.radians(rover_lon)
    r_ecef = np.array([np.cos(la_r) * np.cos(lo_r),
                       np.cos(la_r) * np.sin(lo_r), np.sin(la_r)])
    la_s, lo_s = np.radians(sta_lat), np.radians(sta_lon)
    s_ecef = np.stack([np.cos(la_s) * np.cos(lo_s),
                       np.cos(la_s) * np.sin(lo_s),
                       np.sin(la_s)], axis=0)  # (3, N)
    b = s_ecef - r_ecef[:, None]
    b = b / np.maximum(np.linalg.norm(b, axis=0), 1e-12)
    ra, dec = np.radians(_CMB_RA_DEG), np.radians(_CMB_DEC_DEG)
    theta = np.asarray([gst_rad(t) for t in unix_t])  # (T,)
    # n_CMB in ECEF: rotate the inertial apex by -GST about z.
    nx = np.cos(dec) * np.cos(ra - theta)[:, None]  # (T,1)
    ny = np.cos(dec) * np.sin(ra - theta)[:, None]
    nz = np.full((len(theta), 1), np.sin(dec))
    n = np.concatenate([nx, ny, nz], axis=1)  # (T,3) each row unit-ish
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    proj = n @ b  # (T,N)
    return proj ** 2


def build_pruned_lf_spatial_basis(
    sta_innov: np.ndarray,        # (T, N_sta) detrended reference
                                  # clock innovations, metres
    dists_km: np.ndarray,         # (N_sta,) rover->station distance
    mid_bearings_rad: np.ndarray, # (N_sta,) midpoint bearing
    cmb_proj: np.ndarray,         # (T, N_sta) (b.n_CMB)^2 per epoch
    sta_lf_rms: np.ndarray,       # (N_sta,) trailing LF RMS (screening
                                # proxy)
    sidereal_echo: np.ndarray,    # (T,) network sidereal template
    pre_drift_resid: np.ndarray,  # (T,) causal drift-gate residual
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build the (16 + 2*N_sta)-column causal basis.

    Returns (X, w0_prior, feature_names).  Columns are in physical
    metres -- the prior vector w0 (3-band kriging blend) is meaningful
    only in that scale.
    """
    T, N_sta = sta_innov.shape

    mad = np.median(
        np.abs(sta_innov - np.median(sta_innov, axis=0, keepdims=True)),
        axis=0)
    scale = np.maximum(1.4826 * mad, 0.05)
    sta_clean = np.clip(sta_innov, -3.5 * scale, 3.5 * scale)

    lp_900 = causal_exp_smooth(sta_clean, 900.0, dt)
    lp_1800 = causal_exp_smooth(sta_clean, 1800.0, dt)
    lp_2700 = causal_exp_smooth(sta_clean, 2700.0, dt)
    lp_3600 = causal_exp_smooth(sta_clean, 3600.0, dt)

    def _krige_weights(lam_km, mask=None):
        w = np.exp(-0.5 * (dists_km / lam_km) ** 2)
        if mask is not None:
            w = w * mask.astype(np.float64)
        s = np.sum(w)
        return w / s if s > 1e-12 else np.zeros_like(w)

    w_tidal = _krige_weights(4677.0)
    w_post = _krige_weights(1502.0)
    w_mid = _krige_weights(1459.0)
    w_base = _krige_weights(4200.0)

    m1_tidal = lp_3600 @ w_tidal
    m2_post = lp_1800 @ w_post
    m3_mid = lp_900 @ w_mid

    m4_near = lp_900 @ _krige_weights(4200.0, dists_km <= 360.0)
    m5_cont = lp_900 @ _krige_weights(
        4200.0, (dists_km > 360.0) & (dists_km <= 1000.0))
    m6_far = lp_900 @ _krige_weights(4200.0, dists_km > 1000.0)

    m7_dip_c = lp_900 @ (w_base * np.cos(mid_bearings_rad))
    m8_dip_s = lp_900 @ (w_base * np.sin(mid_bearings_rad))
    m9_quad_c = lp_900 @ (w_base * np.cos(2.0 * mid_bearings_rad))
    m10_quad_s = lp_900 @ (w_base * np.sin(2.0 * mid_bearings_rad))
    m11_cmb = np.sum(lp_1800 * (cmb_proj * w_base[None, :]), axis=1)

    med_rms = float(np.median(sta_lf_rms))
    m12_screened = lp_1800 @ _krige_weights(4200.0, sta_lf_rms <= med_rms)
    m13_unscreened = lp_1800 @ _krige_weights(4200.0, sta_lf_rms > med_rms)

    lag_idx = max(1, int(round(1800.0 / dt)))
    m14_vel = np.zeros(T, dtype=np.float64)
    m14_vel[lag_idx:] = (m1_tidal[lag_idx:] - m1_tidal[:-lag_idx]) / 1800.0
    if lag_idx > 1:
        m14_vel[:lag_idx] = (m1_tidal[:lag_idx]
                             / np.maximum(np.arange(1, lag_idx + 1) * dt,
                                          1.0))

    m15_sid = causal_exp_smooth(
        np.asarray(sidereal_echo, float), 1800.0, dt)
    m16_drift = causal_exp_smooth(
        np.asarray(pre_drift_resid, float), 900.0, dt)

    X_macro = np.column_stack([
        m1_tidal, m2_post, m3_mid,
        m4_near, m5_cont, m6_far,
        m7_dip_c, m8_dip_s, m9_quad_c, m10_quad_s, m11_cmb,
        m12_screened, m13_unscreened,
        m14_vel, m15_sid, m16_drift,
    ])

    # Tier 2: per-station residuals orthogonalised against the macro
    # kriging modes.
    sta_res_900 = lp_900 - m3_mid[:, None]
    sta_res_2700 = lp_2700 - m1_tidal[:, None]
    X = np.hstack([X_macro, sta_res_900, sta_res_2700])

    w0 = np.zeros(X.shape[1], dtype=np.float64)
    w0[0] = 0.50
    w0[1] = 0.30
    w0[2] = 0.20

    names = (
        ["M1_tidal", "M2_post", "M3_mid", "M4_near", "M5_cont", "M6_far",
         "M7_dip_cos", "M8_dip_sin", "M9_quad_cos", "M10_quad_sin",
         "M11_cmb", "M12_screened", "M13_unscreened", "M14_vel",
         "M15_sid", "M16_drift"]
        + [f"sta900_{j}" for j in range(N_sta)]
        + [f"sta2700_{j}" for j in range(N_sta)]
    )
    return X, w0, names


def prior_centered_solve(Sxx: np.ndarray, Sxr: np.ndarray,
                         w0: np.ndarray, diag_pen: np.ndarray) -> np.ndarray:
    """w = (Sxx + diag(Lambda))^-1 (-Sxr + Lambda @ w0).

    Shrinks toward w0 -- in the high-penalty limit the solution IS the
    physics-fixed 3-band kriging blend, not zero."""
    L = np.diag(diag_pen)
    return np.linalg.solve(Sxx + L, -Sxr + L @ w0)
