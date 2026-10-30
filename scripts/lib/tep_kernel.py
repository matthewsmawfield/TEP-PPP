#!/usr/bin/env python3
"""
The operational TEP clock kernel -- ONE definition, used by every step.

Before this module the kernel was re-declared per step with inconsistent
hyperparameters (lambda 4000 vs 4200 km, nugget 0.35 vs 1.27, K 12 vs 15),
and none of them were ever *fitted*.  Everything now goes through here, and
the hyperparameters are selected by nested cross-validation on held-out
stations (`fit_hyperparameters`), never hand-picked.

Model
-----
Band-limited receiver-clock residuals of a station network are treated as a
Gaussian process over great-circle distance with a unit-sill exponential
covariance plus a nugget:

    C(r) = exp(-r / lambda)          (sill normalised to 1)
    K    = C(r_ij) + nugget * I      (nugget = local / spatially-correlated variance)

The prediction of a target station from its K nearest references is the
simple-kriging mean  w = K^-1 c,  pred = w . y_refs .
(Simple kriging: zero-mean assumption, no Lagrange multiplier.  Ordinary
kriging with a sum-to-one constraint is not used because the band-limited
residuals are already de-meaned by the bandpass.)

Baselines the kernel must beat
------------------------------
    raw         -- predict zero (i.e. the station's own band-limited variability).
                  This is the honest "no network at all" reference: it measures
                  how much information the network actually supplies.
    net_mean    -- uniform average of the reference stations (bulk common-mode
                  removal, the conventional operational competitor).
    uniform_krig -- analytic lambda->inf kriging limit: w_i = 1/(n + nugget).
                  This is a SCALED network mean with the same nugget as the
                  kernel but NO distance weighting.  Comparing the kernel
                  against this baseline isolates the geometry-specific
                  contribution from nugget-induced rescaling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

C_M_PER_NS = 0.299792458
C_M_PER_S = 299_792_458.0

# Canonical TEP correlation length (Papers 1-2). Used only as the *centre* of the
# search grid -- never assumed. `fit_hyperparameters` selects on held-out data.
LAMBDA_CANONICAL_KM = 4200.0

LAMBDA_GRID_KM = (300.0, 700.0, 1500.0, 2500.0, 4200.0, 7000.0, 12000.0, 25000.0, 50000.0, 100000.0)
NUGGET_GRID = (0.05, 0.15, 0.35, 0.70, 1.27)
K_GRID = (8, 15, 25, 40)

# ------------------------------------------------------------------ kernel family
# Extended kernel forms and anisotropy modes, reverse-engineered from the
# distance-structured correlations found in Papers 1 (TEP-GNSS) and 2
# (TEP-GNSS-II).  All are tested within the same leakage-free nested-CV
# framework as the baseline exponential isotropic kernel.
#
# Paper 2 findings that motivate each extension:
#   - Gaussian/squared-exponential preferred by AIC (ΔAIC = -12.8 vs exponential)
#   - Matérn(ν=1.5) competitive with exponential
#   - EW/NS anisotropy ratio = 2.16 (p < 10⁻¹⁵)
#   - Orbital-velocity coupling r = -0.888 (surrogate p < 2×10⁻⁷)
#   - CMB dipole-direction alignment r = 0.747 (10,300× variance ratio vs Solar Apex)
FORM_GRID = ("exponential", "gaussian", "matern15", "legendre")

# Measured angular correlation of the phase-alignment index on the sphere
# (Step 81 / Step 86, COD0OPSFIN, 317 stations).  Every other form in this module
# is monotone decaying and therefore POSITIVE at all separations, but the
# measured C(theta) crosses zero near 47 deg, is negative from ~5,250 to
# ~15,750 km (22 of 40 distance bins) and rebounds toward the antipode.  At
# 8,000 km the fitted Gaussian gives +0.03 where the measurement gives -0.09:
# the wrong SIGN, so kriging with it mis-weights distant references.
#
# Coefficients are the Legendre expansion of the measured profile with negative
# terms clipped to zero.  By Schoenberg's theorem C(theta) = sum_l a_l P_l(cos)
# is positive semi-definite on the sphere IF AND ONLY IF every a_l >= 0, so the
# clip is what makes this usable as a covariance at all -- the raw profile has
# a0 = -0.007 and produces an indefinite kriging system.
LEGENDRE_COEFFS = (0.0, 0.03654, 0.06679, 0.02886, 0.01713, 0.02158, 0.0)
EARTH_R_KM = 6371.0
ANISO_MODE_GRID = ("none", "ew_ns", "velocity", "cmb")
ANISO_RATIO_GRID = (1.5, 2.0, 2.16, 3.0)

# CMB dipole direction (Planck 2018): RA = 167.94°, Dec = -6.94°
CMB_DIPOLE_RA_DEG = 167.94
CMB_DIPOLE_DEC_DEG = -6.94
# Obliquity of the ecliptic
OBLIQUITY_DEG = 23.44


@dataclass(frozen=True)
class KernelParams:
    lambda_km: float = LAMBDA_CANONICAL_KM
    nugget: float = 0.35
    k_neighbours: int = 25
    alpha_v: float = 0.35  # TEP Velocity Coupling parameter
    # Kernel family extensions ( Papers 1-2 reverse-engineering )
    form: str = "exponential"       # "exponential", "gaussian", "matern15"
    aniso_mode: str = "none"        # "none", "ew_ns", "velocity", "cmb"
    aniso_ratio: float = 1.0        # λ_parallel / λ_perpendicular  (1.0 = isotropic)

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------- covariance functions
def _cov(d: np.ndarray, form: str, lam: float) -> np.ndarray:
    """Unit-sill covariance as a function of distance, kernel-form dependent.

    exponential : C(r) = exp(-r / λ)                  -- Papers 1-2 baseline
    gaussian    : C(r) = exp(-r² / (2λ²))             -- Paper 2 best AIC
    matern15    : C(r) = (1 + √3 r/λ) exp(-√3 r/λ)    -- Paper 1 competitive
    legendre    : the MEASURED angular correlation, sum_l a_l P_l(cos(r/R)),
                  normalised to unit sill.  Unlike the three parametric forms it
                  is not monotone: it carries the negative lobe and antipodal
                  rebound that the measurement shows and the others cannot
                  represent.  lam is ignored -- the shape is measured, not fitted.
    """
    if form == "legendre":
        x = np.cos(np.asarray(d, float) / EARTH_R_KM)
        Pm1 = np.ones_like(x)
        Pc = x
        out = LEGENDRE_COEFFS[0] * Pm1 + (LEGENDRE_COEFFS[1] * Pc
                                          if len(LEGENDRE_COEFFS) > 1 else 0.0)
        for l in range(2, len(LEGENDRE_COEFFS)):
            Pm1, Pc = Pc, ((2 * l - 1) * x * Pc - (l - 1) * Pm1) / l
            out = out + LEGENDRE_COEFFS[l] * Pc
        c0 = float(sum(LEGENDRE_COEFFS))          # C(0): all P_l(1) = 1
        return out / c0 if abs(c0) > 1e-12 else out
    if form == "gaussian":
        return np.exp(-0.5 * (d / lam) ** 2)
    if form == "matern15":
        r = d / lam * math.sqrt(3.0)
        return (1.0 + r) * np.exp(-r)
    # default: exponential
    return np.exp(-d / lam)


def _gmst_deg(year: int, month: int, day: int, hour: float = 12.0) -> float:
    """Greenwich Mean Sidereal Time in degrees (IAU 1982 polynomial)."""
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jd = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1))
          + day + B - 1524.5 + hour / 24.0)
    t = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t - t ** 3 / 38710000.0)
    return gmst % 360.0


def _doy_to_month_day(year: int, doy: int):
    """Convert day-of-year to (month, day)."""
    from datetime import datetime, timedelta
    d = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return d.month, d.day


def preferred_azimuth_deg(aniso_mode: str, date_tag: str) -> float:
    """Geographic azimuth (degrees from north) of the anisotropy preferred axis.

    For 'ew_ns' the preferred axis is geographic East (90°), constant.
    For 'velocity' the axis is the projection of Earth's orbital velocity
        onto the equatorial plane, evaluated at noon UTC of *date_tag*.
    For 'cmb' the axis is the projection of the CMB dipole direction
        onto the equatorial plane, evaluated at noon UTC of *date_tag*.

    The projection onto the equator gives a direction whose geographic
    azimuth at the equator is  (RA - GMST) mod 360°.  At higher latitudes
    the azimuth of the projected vector differs, but for a *global*
    anisotropy axis (one stretch factor for the entire network) the
    equatorial projection is the correct effective direction.
    """
    if aniso_mode == "ew_ns":
        return 90.0  # East

    year = int(date_tag[:4])
    doy = int(date_tag[4:])
    month, day = _doy_to_month_day(year, doy)
    gmst = _gmst_deg(year, month, day)

    if aniso_mode == "cmb":
        ra = CMB_DIPOLE_RA_DEG
    elif aniso_mode == "velocity":
        # Sun's approximate ecliptic longitude: spring equinox ≈ DOY 80
        n = doy - 80
        lam_sun = (n / 365.25) * 360.0
        lam_vel = (lam_sun + 90.0) % 360.0  # velocity ⊥ Sun-Earth in ecliptic
        # Convert ecliptic longitude to equatorial RA
        ra = math.degrees(math.atan2(
            math.sin(math.radians(lam_vel)) * math.cos(math.radians(OBLIQUITY_DEG)),
            math.cos(math.radians(lam_vel))
        )) % 360.0
    else:
        return 90.0

    return (ra - gmst) % 360.0


def anisotropic_distance(D: np.ndarray, az: np.ndarray, p: KernelParams,
                         date_tag: str | None = None) -> np.ndarray:
    """Stretch the distance matrix to encode geometric anisotropy.

    If the anisotropy ratio is 1.0 or the mode is 'none', returns D unchanged.

    For a preferred axis at azimuth *pref* (degrees from north), a pair whose
    bearing is aligned with *pref* has its distance divided by *ratio*
    (longer correlation), and a pair perpendicular to *pref* is unstretched.
    The smooth interpolation is:

        d_eff = d * sqrt( cos²(Δ) / ratio² + sin²(Δ) )

    where Δ is the acute angle between the pair bearing and the preferred axis.
    """
    if p.aniso_mode == "none" or p.aniso_ratio == 1.0:
        return D
    if az is None:
        return D
    pref = preferred_azimuth_deg(p.aniso_mode, date_tag) if date_tag else 90.0
    # Acute angle between bearing and preferred axis (0-90°)
    delta = (az - pref) % 180.0
    delta = np.minimum(delta, 180.0 - delta)
    delta_rad = np.radians(delta)
    stretch = np.sqrt(np.cos(delta_rad) ** 2 / p.aniso_ratio ** 2
                      + np.sin(delta_rad) ** 2)
    return D * stretch


# --------------------------------------------------------------------- weights
def kriging_weights(dist_refs_refs, dist_refs_target, p: KernelParams, V_refs_refs=None, V_refs_target=None):
    """Simple-kriging weights for one target from its reference set.

    Optional velocity-direction coupling for kinematic / LEO receivers:
    V_refs_refs and V_refs_target should be the cosine similarity of velocity
    directions,  v_hat_i . v_hat_j  (range [-1, +1], dimensionless).
    The coupling multiplies the covariance by exp(alpha_v * (cos_theta - 1)),
    which is normalised to 1.0 when velocities are parallel (cos_theta = +1)
    and is < 1.0 otherwise, keeping the off-diagonal covariance ≤ the diagonal
    and preserving positive semidefiniteness.  This is the PSD-preserving
    form: exp(alpha_v * cos_theta) (without the -1 offset) would inflate the
    diagonal and break PSD.

    Note: the form (1 + alpha_v * v_i.v_j / c^2) is NOT used because for
    any achievable velocity (v << c) the correction is ~10^-10 and
    physically negligible.  The direction-based form is the meaningful
    operational coupling, consistent with DynamicTEPKernel.
    """
    n = len(dist_refs_target)

    # Standard spatial correlation (form-dependent covariance)
    C_rr = _cov(dist_refs_refs, p.form, p.lambda_km)
    c_rt = _cov(dist_refs_target, p.form, p.lambda_km)

    # Apply velocity-direction coupling if provided.
    # The coupling multiplies only the OFF-DIAGONAL covariance: a receiver's
    # self-variance is the sill (1.0), not exp(alpha_v), so the diagonal of
    # C_rr must remain 1.0 before the nugget is added.
    #
    # The multiplier exp(alpha_v * (cos_theta - 1)) is normalised so that it
    # equals 1.0 only when velocities are parallel (cos_theta = +1) and is
    # < 1.0 otherwise, keeping the off-diagonal covariance ≤ the diagonal
    # and preserving positive semidefiniteness of the covariance matrix.
    if V_refs_refs is not None and V_refs_target is not None:
        offdiag = np.exp(p.alpha_v * (np.clip(V_refs_refs, -1.0, 1.0) - 1.0))
        np.fill_diagonal(offdiag, 1.0)
        C_rr *= offdiag
        c_rt *= np.exp(p.alpha_v * (np.clip(V_refs_target, -1.0, 1.0) - 1.0))

    K = C_rr + (p.nugget + 1e-9) * np.eye(n)
    c = c_rt
    with np.errstate(all="ignore"):
        try:
            # PSD guard: if velocity coupling made K indefinite, add a
            # ridge to restore positive definiteness before solving.
            eigvals = np.linalg.eigvalsh(K)
            if eigvals[0] < 1e-12:
                K = K + (1e-12 - eigvals[0]) * np.eye(n)
            w = np.linalg.solve(K, c)
        except np.linalg.LinAlgError:
            # Ill-conditioned: add a small ridge rather than falling back to
            # the uniform mean, so the predictor degrades gracefully.
            ridge = 1e-6 * np.trace(K) / n
            w = np.linalg.solve(K + ridge * np.eye(n), c)
    if not np.all(np.isfinite(w)):
        w = np.full(n, 1.0 / (n + p.nugget))
    # Return the raw simple-kriging weights.  The previous tanh limiter
    # (5*tanh(w/5)) broke the normal-equation solution: after solve(K, c)
    # the post-hoc non-linear squash means the weights no longer reproduce
    # the target covariance c, and the predictor is no longer the simple-
    # kriging BLUP.  If weight bounding is needed for numerical stability,
    # it should be applied via constrained optimisation or a Tikhonov
    # ridge in K, not by distorting the closed-form solution.
    return w


def neighbours(D, target, pool, k):
    """Indices of the k nearest members of `pool` to `target` (target excluded)."""
    cand = [j for j in pool if j != target]
    cand.sort(key=lambda j: D[target, j])
    return cand[:k]


def predict(X, D, target, pool, p: KernelParams, AZ=None, date_tag=None):
    """Kriging prediction of column `target` from columns in `pool`.

    Returns the weighted sum of reference clock residuals, where the weights
    are determined by the distance matrix (not the data).  The prediction
    includes the per-epoch common-mode component, which is the
    operationally-relevant signal for a real-time PPP user: the network
    prior should predict the full clock residual, not just the differential
    anomaly.  Centering the references epoch-wise would remove the common
    mode from the kriging prediction but not from the mean predictor,
    creating an inconsistent comparison; the common mode is therefore
    retained.

    If *AZ* (azimuth matrix) and *date_tag* are provided, the effective
    distance matrix is stretched by the anisotropy specified in *p* before
    computing kriging weights.  The neighbour selection uses the isotropic
    distance D (geometric nearest neighbours), while the covariance uses
    the anisotropically stretched distance -- this preserves the neighbour
    graph while allowing the correlation structure to be anisotropic.
    """
    nn = neighbours(D, target, pool, p.k_neighbours)
    if len(nn) < 3:
        return None
    if AZ is not None and p.aniso_mode != "none" and p.aniso_ratio != 1.0:
        D_eff = anisotropic_distance(D, AZ, p, date_tag)
    else:
        D_eff = D
    w = kriging_weights(D_eff[np.ix_(nn, nn)], D_eff[nn, target], p)
    with np.errstate(all="ignore"):
        pred = X[:, nn] @ w
    return pred


def prediction_error(X, D, target, pool, p: KernelParams, AZ=None, date_tag=None):
    """RMS of (truth - kriging prediction), metres. NaN if not computable."""
    pred = predict(X, D, target, pool, p, AZ=AZ, date_tag=date_tag)
    if pred is None:
        return np.nan
    e = float(np.sqrt(np.mean((X[:, target] - pred) ** 2)))
    return e if np.isfinite(e) else np.nan


def predict_with_variance(X, D, target, pool, p: KernelParams,
                          AZ=None, date_tag=None):
    """Kriging prediction and simple-kriging variance.

    Returns (pred, variance) where variance is the simple-kriging
    prediction variance:  sigma^2 = sill * (1 - w . c_rt)
    where sill is the unit-sill covariance (1.0) and w are the
    simple-kriging weights.  The variance is per-epoch (scalar),
    not a vector — it depends only on the geometry (distances),
    not on the data values.
    """
    nn = neighbours(D, target, pool, p.k_neighbours)
    if len(nn) < 3:
        return None, None
    if AZ is not None and p.aniso_mode != "none" and p.aniso_ratio != 1.0:
        D_eff = anisotropic_distance(D, AZ, p, date_tag)
    else:
        D_eff = D
    w = kriging_weights(D_eff[np.ix_(nn, nn)], D_eff[nn, target], p)
    c_rt = _cov(D_eff[nn, target], p.form, p.lambda_km)
    with np.errstate(all="ignore"):
        pred = X[:, nn] @ w
    # Simple-kriging variance: sigma^2 = 1 - w . c  (unit sill)
    sk_var = float(max(1.0 - np.dot(w, c_rt), 0.0))
    return pred, sk_var


# ----------------------------------------------------------- multi-band
# Paper 1 frequency-dependent lambdas (Step 134/145)
MULTIBAND_BANDS = [
    {"name": "tidal",       "f_lo_hz": 9.84e-6, "f_hi_hz": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo_hz": 35e-6,   "f_hi_hz": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo_hz": 100e-6,  "f_hi_hz": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]


def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _band_phase_grams(x, dt_s, bands=None, exclude=None):
    """Compute a phase Gram for each frequency band.

    Returns a list of (gram, band_info) or None if all bands fail.

    If ``exclude`` is given (a column index or list of indices), those
    columns are removed from the data before the FFT/Gram computation so
    that the target station's own signal does not leak into its
    prediction.  The returned Gram is sized to ``x.shape[1] -
    len(exclude)`` and the caller must index it with the adjusted
    indices.
    """
    from scripts.lib.tep_phase import spectral_phase_gram
    if bands is None:
        bands = MULTIBAND_BANDS
    if exclude is not None:
        exclude_set = set(exclude) if isinstance(exclude, (list, tuple, set, np.ndarray)) else {exclude}
        keep_cols = [i for i in range(x.shape[1]) if i not in exclude_set]
        x_use = x[:, keep_cols]
    else:
        x_use = x
    grams = []
    for band in bands:
        try:
            result = spectral_phase_gram(
                x_use, dt_s, f_lo_hz=band["f_lo_hz"], f_hi_hz=band["f_hi_hz"],
                min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
            grams.append((result["predictive_gram"], band))
        except ValueError:
            continue
    if not grams:
        return None
    return grams


def predict_multiband(X, D, target, pool, p: KernelParams, dt_s=30.0,
                      bands=None, AZ=None, date_tag=None):
    """Multi-band frequency-dependent TEP kriging prediction.

    Computes a separate phase Gram for each frequency band, uses the
    physics-fixed lambda for that band (from Paper 1), and averages the
    per-band kriging predictions.  The distance kernel uses multi-scale
    expansion (lambda/1.6, lambda, lambda*1.6) for each band.

    Returns the averaged prediction vector (same length as X[:, target]),
    or None if no band produces a valid prediction.
    """
    if bands is None:
        bands = MULTIBAND_BANDS
    nn = neighbours(D, target, pool, p.k_neighbours)
    if len(nn) < 3:
        return None
    if AZ is not None and p.aniso_mode != "none" and p.aniso_ratio != 1.0:
        D_eff = anisotropic_distance(D, AZ, p, date_tag)
    else:
        D_eff = D

    # Build per-band phase Grams from the data window INCLUDING the
    # target.  The Gram is a second-order statistic (pairwise band
    # coherence), not a value channel: including the target row supplies
    # the target/reference cross-coherence g_rt that the kriging system
    # needs, without letting the target's own value into X[:, nn] @ w.
    # Excluding it forced a distance-only rho_rt as cross-covariance
    # against a phase-modulated K -- the joint [K, c; c', 1] was then not
    # guaranteed PSD, which is why the SK variance saturated at 0.
    band_grams = _band_phase_grams(X, dt_s, bands)
    if band_grams is None:
        # Fall back to single-lambda prediction
        return predict(X, D, target, pool, p, AZ=AZ, date_tag=date_tag)

    # Align grams to the neighbour set
    d_rr = D_eff[np.ix_(nn, nn)]
    d_rt = D_eff[nn, target]

    band_preds = []
    for G_band, band_info in band_grams:
        # G_band is (n_sta, n_sta) over the original column order
        scales = _multi_scales(band_info["lambda_km"])
        rho_rr = _rho_multi(d_rr, scales)
        rho_rt = _rho_multi(d_rt, scales)
        G_pp = G_band[np.ix_(nn, nn)]
        g_rt = G_band[target, nn]
        # Consistent Schur-product covariance: rho(d) * coherence for
        # both blocks keeps [[K, c_rt], [c_rt', 1]] PSD (Hadamard of two
        # valid joint covariances), so the SK variance stays in [0, 1].
        c_rt = rho_rt * g_rt
        K = rho_rr * G_pp + (p.nugget + 1e-9) * np.eye(len(nn))
        K = 0.5 * (K + K.T)
        try:
            ev = np.linalg.eigvalsh(K)
            if ev[0] < 1e-6:
                K = K + (1e-6 - ev[0]) * np.eye(len(nn))
            w = np.linalg.solve(K, c_rt)
            pred = X[:, nn] @ w
            if not np.isfinite(pred).all():
                continue
            band_preds.append(pred)
        except np.linalg.LinAlgError:
            continue

    if not band_preds:
        return predict(X, D, target, pool, p, AZ=AZ, date_tag=date_tag)
    return np.mean(band_preds, axis=0)


def predict_multiband_with_variance(X, D, target, pool, p: KernelParams,
                                     dt_s=30.0, bands=None,
                                     AZ=None, date_tag=None):
    """Multi-band prediction with simple-kriging variance (averaged across bands).

    Returns (pred, variance, band_details) or (None, None, []).
    The target participates in the phase Gram so its band coherence with
    each reference (g_rt) can modulate the cross-covariance; only a
    second-order statistic is used, never the target's own values.
    """
    if bands is None:
        bands = MULTIBAND_BANDS
    nn = neighbours(D, target, pool, p.k_neighbours)
    if len(nn) < 3:
        return None, None, []
    if AZ is not None and p.aniso_mode != "none" and p.aniso_ratio != 1.0:
        D_eff = anisotropic_distance(D, AZ, p, date_tag)
    else:
        D_eff = D

    # Gram includes the target (see predict_multiband): the pairwise
    # band coherence is a second-order statistic, and the target row
    # supplies g_rt for the cross-covariance the kriging system needs.
    band_grams = _band_phase_grams(X, dt_s, bands)
    if band_grams is None:
        fb_pred, fb_var = predict_with_variance(X, D, target, pool, p, AZ=AZ, date_tag=date_tag)
        return fb_pred, fb_var, []

    d_rr = D_eff[np.ix_(nn, nn)]
    d_rt = D_eff[nn, target]

    band_preds = []
    band_vars = []
    band_details = []
    for G_band, band_info in band_grams:
        scales = _multi_scales(band_info["lambda_km"])
        rho_rr = _rho_multi(d_rr, scales)
        rho_rt = _rho_multi(d_rt, scales)
        G_pp = G_band[np.ix_(nn, nn)]
        g_rt = G_band[target, nn]
        c_rt = rho_rt * g_rt
        K = rho_rr * G_pp + (p.nugget + 1e-9) * np.eye(len(nn))
        K = 0.5 * (K + K.T)
        try:
            ev = np.linalg.eigvalsh(K)
            if ev[0] < 1e-6:
                K = K + (1e-6 - ev[0]) * np.eye(len(nn))
            w = np.linalg.solve(K, c_rt)
            pred_band = X[:, nn] @ w
            if not np.isfinite(pred_band).all():
                band_details.append({
                    "name": band_info.get("name", "?"),
                    "valid": False,
                    "reason": "non-finite prediction",
                })
                continue
            band_preds.append(pred_band)
            _raw_var = 1.0 - float(np.dot(w, c_rt))
            _saturated = _raw_var < 0.0
            sk_var = float(max(_raw_var, 0.0))
            band_vars.append(sk_var)
            band_details.append({
                "name": band_info.get("name", "?"),
                "lambda_km": band_info.get("lambda_km"),
                "f_lo_uhz": band_info.get("f_lo_hz", 0) * 1e6,
                "f_hi_uhz": band_info.get("f_hi_hz", 0) * 1e6,
                "predicted_m": float(pred_band[-1]) if pred_band is not None else None,
                "variance_m2": sk_var,
                "variance_raw": float(_raw_var),
                "variance_saturated": bool(_saturated),
                "n_neighbours": len(nn),
                "valid": True,
            })
        except np.linalg.LinAlgError:
            band_details.append({
                "name": band_info.get("name", "?"),
                "valid": False,
                "reason": "LinAlgError",
            })
            continue

    if not band_preds:
        fb_pred, fb_var = predict_with_variance(X, D, target, pool, p, AZ=AZ, date_tag=date_tag)
        return fb_pred, fb_var, band_details
    # Detect uninformative weights: when every band's simple-kriging
    # variance saturates (negative before clamping), the clamp maps them
    # all to zero, and 1/var weights collapse to 1/n -- encoding no
    # evidence at all.  The full_band entry uses a different routine
    # and legitimately reports a non-zero variance, so it is excluded
    # from the saturation test.
    _all_saturated = all(
        bd.get("variance_saturated", False)
        for bd in band_details
        if bd.get("valid") and bd.get("name") != "full_band"
    )
    if _all_saturated:
        for bd in band_details:
            bd["weights_uninformative"] = True
    pred = np.mean(band_preds, axis=0)
    var = float(np.mean(band_vars))
    return pred, var, band_details


def net_mean_error(X, target, pool):
    """RMS of (truth - uniform mean of pool), metres. Scale=1 (raw mean)."""
    others = [j for j in pool if j != target]
    if len(others) < 2:
        return np.nan
    with np.errstate(all="ignore"):
        e = float(np.sqrt(np.mean((X[:, target] - X[:, others].mean(1)) ** 2)))
    return e if np.isfinite(e) else np.nan


def uniform_kriging_error(X, target, pool, p: KernelParams):
    """RMS of (truth - scaled mean), metres.

    Analytic lambda->inf kriging limit: w_i = 1/(n + nugget) for all i.
    This is a SCALED network mean with the same nugget as the kernel but
    NO distance weighting.  Comparing the kernel against this baseline
    isolates the geometry-specific contribution from nugget rescaling.
    """
    others = [j for j in pool if j != target]
    if len(others) < 2:
        return np.nan
    n = len(others)
    w = np.full(n, 1.0 / (n + p.nugget))
    with np.errstate(all="ignore"):
        pred = X[:, others] @ w
        if not np.all(np.isfinite(pred)):
            return np.nan
        e = float(np.sqrt(np.mean((X[:, target] - pred) ** 2)))
    return e if np.isfinite(e) else np.nan


def raw_error(X, target):
    """RMS of the station's own band-limited residual = 'predict zero'."""
    return float(np.sqrt(np.mean(X[:, target] ** 2)))


# ------------------------------------------------------------------- selection
def fit_hyperparameters(X, D, train_idx, lam_grid=LAMBDA_GRID_KM,
                        nug_grid=NUGGET_GRID, k_grid=K_GRID,
                        AZ=None, date_tag=None,
                        form_grid=None, aniso_mode_grid=None, aniso_ratio_grid=None):
    """
    Select (lambda, nugget, K) by leave-one-out **within the training stations
    only**.  Both the targets and the references are drawn from `train_idx`, so
    a later evaluation on held-out stations is leakage-free.

    If form_grid / aniso_mode_grid / aniso_ratio_grid are provided, the search
    is extended over the full kernel family (exponential, Gaussian, Matérn-1.5)
    and anisotropy modes (none, ew_ns, velocity, cmb).  All parameters are
    selected jointly on the training LOO, so the held-out evaluation remains
    leakage-free in kernel-form space as well.
    """
    forms = form_grid or ("exponential",)
    modes = aniso_mode_grid or ("none",)
    ratios = aniso_ratio_grid or (1.0,)
    best = (np.inf, None)
    for form in forms:
        for mode in modes:
            for ratio in ratios:
                if mode == "none" and ratio != 1.0:
                    continue
                for lam in lam_grid:
                    for nug in nug_grid:
                        for k in k_grid:
                            p = KernelParams(lam, nug, k, form=form,
                                             aniso_mode=mode, aniso_ratio=ratio)
                            errs = [prediction_error(X, D, t, train_idx, p,
                                                     AZ=AZ, date_tag=date_tag)
                                    for t in train_idx]
                            m = np.nanmedian(errs) if np.any(np.isfinite(errs)) else np.inf
                            if np.isfinite(m) and m < best[0]:
                                best = (m, p)
    # Guard: if no grid point produced a finite score, return None to signal
    # failure.  The caller (step_25, step_26) skips splits where this happens,
    # so no hand-picked constants ever enter the held-out evaluation.
    if best[1] is None:
        return None, np.inf
    return best[1], float(best[0])


def score_heldout(X, D, test_idx, ref_idx, p: KernelParams, AZ=None, date_tag=None):
    """
    Score a fitted kernel on held-out target stations, predicting each from the
    reference (training) stations only.

    Returns medians over the held-out targets:
        raw          -- predict-zero error (m)
        kernel       -- kriging error (m)
        net_mean     -- bulk-average error (m, scale=1)
        uniform_krig -- scaled-mean error (m, same nugget, no geometry)
        skill_vs_raw_pct, skill_vs_mean_pct, kernel_vs_uniform_krig_pct
    """
    raw, ker, mean, ukrig = [], [], [], []
    for t in test_idx:
        r = raw_error(X, t)
        e = prediction_error(X, D, t, ref_idx, p, AZ=AZ, date_tag=date_tag)
        m = net_mean_error(X, t, ref_idx)
        u = uniform_kriging_error(X, t, ref_idx, p)
        if np.isfinite(r) and np.isfinite(e) and np.isfinite(m):
            raw.append(r); ker.append(e); mean.append(m)
            ukrig.append(u if np.isfinite(u) else np.nan)
    if not raw:
        return None
    R = float(np.median(raw))
    E = float(np.median(ker))
    M = float(np.median(mean))
    U = float(np.nanmedian(ukrig)) if np.any(np.isfinite(ukrig)) else np.nan
    return {
        "n_targets": len(raw),
        "raw_m": R, "kernel_m": E, "net_mean_m": M,
        "uniform_krig_m": U,
        "skill_vs_raw_pct": 100.0 * (1.0 - E / R) if R > 0 else np.nan,
        "mean_skill_vs_raw_pct": 100.0 * (1.0 - M / R) if R > 0 else np.nan,
        "kernel_vs_mean_pct": 100.0 * (1.0 - E / M) if M > 0 else np.nan,
        "kernel_vs_uniform_krig_pct": (100.0 * (1.0 - E / U) if np.isfinite(U) and U > 0 else np.nan),
        "per_target": {"raw": raw, "kernel": ker, "net_mean": mean, "uniform_krig": ukrig},
    }


# ------------------------------------------------------------------ data prep
def build_matrix(series, band_lo_uhz=10.0, band_hi_uhz=500.0,
                 min_std_m=0.01, max_std_m=60.0, filter_by_std=True,
                 compute_azimuth=False):
    """
    series (from tep_data.load_series) -> (names, X, D) or (names, X, D, AZ)

    X  : (n_epoch, n_station) band-limited clock residuals in METRES. Pass
         ``band_lo_uhz=None, band_hi_uhz=None`` for robust linear detrending
         without a spectral band-limit.
    D  : (n_station, n_station) great-circle distance in km
    AZ : (n_station, n_station) bearing matrix in degrees from north (0-180),
         only returned if *compute_azimuth* is True.  Needed for anisotropic
         kernel variants (ew_ns, velocity, cmb).

    Stations whose band-limited residual is degenerate or wildly outlying are
    dropped (bandpass ringing on a series with an undetected jump).

    NOTE: filter_by_std defaults to True for backward compatibility with
    descriptive steps (step_06, step_03c).  For leakage-free held-out
    evaluation (step_25, step_26) pass filter_by_std=False and filter
    using only training-station statistics after the train/test split.
    """
    from scripts.lib import tep_data as td
    from scripts.lib import tep_stats as ts

    names = [n for n in series if np.isfinite(series[n]["resid"]).all()]
    if len(names) < 6:
        return ([], None, None, None) if compute_azimuth else ([], None, None)
    grid = series[names[0]]["t"]
    cols = []
    for n in names:
        dt = float(np.nanmedian(np.diff(series[n]["t"])))
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0
        if band_lo_uhz is None and band_hi_uhz is None:
            b = td.robust_detrend(series[n]["resid"])
        elif band_lo_uhz is None or band_hi_uhz is None:
            raise ValueError("band edges must both be numeric or both be None")
        else:
            b = ts.bandpass(series[n]["resid"], dt, band_lo_uhz, band_hi_uhz)
        cols.append(np.interp(grid, series[n]["t"], b) * C_M_PER_NS)
    X = np.nan_to_num(np.column_stack(cols))
    if filter_by_std:
        keep = [i for i in range(len(names)) if min_std_m < X[:, i].std() < max_std_m]
        names = [names[i] for i in keep]
        X = X[:, keep]
    # Great-circle distance matrix (canonical TEP distance metric).
    # A 3D distance incorporating altitude differences is implemented for
    # future LEO applications but is NOT used for ground stations, because
    # (a) the gravitational potential difference at ground-station altitude
    #     differences (~1 km) is ~10^-13 in dimensionless units and physically
    #     negligible, and (b) the arbitrary scaling factor S=100 previously
    #     used here had no physical derivation.
    sts = [series[n]["station"] for n in names]
    n = len(sts)
    D = np.zeros((n, n))
    AZ = np.zeros((n, n)) if compute_azimuth else None
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = td.great_circle_km(sts[i], sts[j])
            if compute_azimuth:
                az = td.baseline_azimuth_deg(sts[i], sts[j])
                AZ[i, j] = AZ[j, i] = az
    if compute_azimuth:
        return names, X, D, AZ
    return names, X, D
