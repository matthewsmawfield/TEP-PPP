"""Causal cross-spectral quantities for phase-resolved TEP prediction.

The paper's phase-cos statistic is ``cos(arg(sum_f S_ij(f)))``.  For
prediction we retain the corresponding coherence magnitude as well:

    G_ij = Re(sum_f conj(X_i) X_j) / sqrt(E_i E_j)
         = coherence_ij * phase_cos_ij.

``G`` is the real part of a normalized complex Gram matrix and is therefore
positive semidefinite.  Its Hadamard product with a valid distance covariance
is also positive semidefinite (Schur product theorem).  This supplies the
missing safe bridge from the descriptive phase-cos result to kriging.
"""
from __future__ import annotations

import numpy as np


def spectral_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                        min_bins=6, min_low_cycles=0.85):
    """Return phase/coherence matrices from one strictly prior data window.

    ``x`` is time-by-station on a common uniform grid.  Columns are linearly
    detrended and Hann-windowed.  A ValueError is raised rather than silently
    changing the registered band when the window cannot resolve it.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"dt_s must be finite and positive, got {dt_s}")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history is too short to resolve the registered low frequency")

    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    # Use sum-of-squares instead of q @ q to avoid spurious macOS Accelerate
    # BLAS warnings (divide-by-zero / overflow in matmul) on 1-D arrays.
    den = float(np.sum(q * q))
    # Bring arbitrary clock datums into a numerically safe range before any
    # centring/subtraction.  The later coherence normalisation makes this
    # transformation exactly immaterial to the statistic.
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero station scale in phase window")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    # Coherence is invariant to a positive per-station scale factor.  Normalise
    # before the FFT so valid clock products with very different absolute clock
    # units/ranges cannot overflow F^H F and silently manufacture NaNs.  This is
    # especially important for mixed analysis-centre CLK files, where an
    # otherwise usable station can carry a very large arbitrary clock datum.
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero detrended scale in phase window")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    band = (freq >= f_lo_hz) & (freq <= f_hi_hz)
    if int(band.sum()) < int(min_bins):
        raise ValueError("history provides too few Fourier bins in the registered band")
    F = F[band]

    # ``einsum`` avoids spurious Accelerate/BLAS overflow warnings observed for
    # finite complex matrices on macOS while producing the same Gram matrix.
    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero spectral energy in at least one station")
    # Use sqrt(energy_i) * sqrt(energy_j) instead of sqrt(energy_i * energy_j)
    # to avoid underflow when energy values are small but above tiny.
    sqrt_e = np.sqrt(energy)
    norm = np.outer(sqrt_e, sqrt_e)
    complex_coherence = cross / norm
    predictive_gram = np.real(complex_coherence)
    predictive_gram = 0.5 * (predictive_gram + predictive_gram.T)
    np.fill_diagonal(predictive_gram, 1.0)

    phase_cos = np.ones_like(predictive_gram)
    off = np.abs(cross) > np.finfo(float).tiny
    phase_cos[off] = np.cos(np.angle(cross[off]))
    coherence = np.abs(complex_coherence)
    return {
        "predictive_gram": predictive_gram,
        "phase_cos": phase_cos,
        "coherence": coherence,
        "complex_coherence": complex_coherence,
        "n_bins": int(band.sum()),
        "duration_s": float(duration),
        "lowest_resolved_hz": float(freq[band][0]),
    }


def phase_distance_weights(d_rr, d_rt, gram_rr, gram_rt, lambda_km=4200.0,
                           nugget=0.35, form="gaussian"):
    """PSD-safe phase-coherent TEP kriging weights.

    The reference covariance is the Hadamard product of the TEP distance
    covariance and a spectral Gram matrix.  Unlike elementwise transformations
    of pairwise phase angles, this construction has a PSD guarantee.
    """
    d_rr = np.asarray(d_rr, float)
    d_rt = np.asarray(d_rt, float)
    g_rr = np.asarray(gram_rr, float)
    g_rt = np.asarray(gram_rt, float)
    if form == "gaussian":
        rho_rr = np.exp(-0.5 * (d_rr / lambda_km) ** 2)
        rho_rt = np.exp(-0.5 * (d_rt / lambda_km) ** 2)
    elif form == "exponential":
        rho_rr = np.exp(-d_rr / lambda_km)
        rho_rt = np.exp(-d_rt / lambda_km)
    else:
        raise ValueError(f"unknown covariance form: {form}")
    K = rho_rr * g_rr + (nugget + 1e-9) * np.eye(len(d_rt))
    c = rho_rt * g_rt
    try:
        return np.linalg.solve(K, c)
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.trace(K) / len(d_rt)
        return np.linalg.solve(K + ridge * np.eye(len(d_rt)), c)


def phase_randomised_real_gram(complex_coherence, theta):
    """Rotate station phases while preserving coherence magnitudes and PSD."""
    C = np.asarray(complex_coherence, complex)
    theta = np.asarray(theta, float)
    if C.shape != (len(theta), len(theta)):
        raise ValueError("phase vector does not match coherence matrix")
    rot = np.exp(1j * theta)
    out = np.real(rot.conj()[:, None] * C * rot[None, :])
    return 0.5 * (out + out.T)


def spectral_phase_blocks(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                          bins_per_block=3, min_low_cycles=0.85):
    """Return consecutive, non-overlapping spectral-coherence blocks.

    This is the safe input representation for a phase-resolved predictor.
    A block is deliberately several Fourier bins rather than one DFT bin: a
    single-bin coherence matrix is rank one and its apparent spatial detail is
    too unstable to use as an operational covariance.  With a 24 h, 300 s CLK
    window the registered 10--500 microhertz range has 43 bins, yielding 15
    blocks at the default three-bin resolution.

    The returned quantities use *only* the supplied history.  They are a
    structural prior, not a licence to use a future-day Fourier transform when
    predicting a live current-day clock.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    usable = freq[(freq >= f_lo_hz) & (freq <= f_hi_hz)]
    if (x.shape[0] - 1) * float(dt_s) * f_lo_hz < min_low_cycles:
        raise ValueError("history is too short to resolve the registered low frequency")
    if len(usable) < bins_per_block:
        raise ValueError("history provides too few Fourier bins for phase blocks")
    half = 0.49 / ((x.shape[0]) * float(dt_s))
    out = []
    for start in range(0, len(usable), int(bins_per_block)):
        f = usable[start:start + int(bins_per_block)]
        # Pad by less than half a bin so the inclusive frequency selection in
        # spectral_phase_gram admits exactly this block despite binary rounding.
        result = spectral_phase_gram(
            x, dt_s, max(f_lo_hz, float(f[0] - half)),
            min(f_hi_hz, float(f[-1] + half)), min_bins=len(f),
            min_low_cycles=min_low_cycles)
        result["f_lo_hz"] = float(f[0])
        result["f_hi_hz"] = float(f[-1])
        result["f_center_hz"] = float(np.mean(f))
        out.append(result)
    return out


def complex_wiener_weights(cov_rr, cov_rt, nugget=0.0):
    """Solve a Hermitian PSD complex Wiener system without discarding phase sign.

    ``cov_rr`` must be a Hermitian reference covariance and ``cov_rt`` its
    target/reference cross-covariance in matching orientation.  A caller must
    supply causally available complex/quadrature observations; applying these
    weights directly to real instantaneous data is not a valid lead/lag test.
    """
    K = np.asarray(cov_rr, complex)
    c = np.asarray(cov_rt, complex)
    if K.shape != (len(c), len(c)):
        raise ValueError("complex covariance dimensions do not match")
    K = 0.5 * (K + K.conj().T)
    eig = np.linalg.eigvalsh(K)
    if eig[0] < -1e-8 * max(1.0, float(np.max(np.abs(eig)))):
        raise ValueError("complex covariance is not positive semidefinite")
    return np.linalg.solve(K + (float(nugget) + 1e-9) * np.eye(len(c)), c)
