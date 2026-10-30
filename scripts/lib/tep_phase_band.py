"""Causal band-pass and phase-resolved kernel utilities.

Extracted from the step_95 phase-resolved v2 development module so that
active pipeline steps (22, 96) can import them without depending on an
archived development script.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from scripts.lib import tep_kernel as tk

DT = 300.0
FLO, FHI = 10e-6, 500e-6
BINS_PER_BLOCK = 3
NUGGET = 0.35


def causal_band(prior, current, flo, fhi):
    """Causal band-pass on D-1 + D; return D-1 and D outputs.

    ``sosfilt`` is one-directional, so the first current-day output has only
    historical input.  No filtfilt, full-day detrend or future current samples
    are used.
    """
    fs = 1.0 / DT
    sos = signal.butter(2, [flo, fhi], btype="bandpass", fs=fs, output="sos")
    x = np.vstack((prior, current)).astype(float, copy=True)
    anchor = x[0].copy()
    if not np.isfinite(anchor).all():
        raise ValueError("non-finite prior anchor")
    x -= anchor
    z = signal.sosfilt(sos, x, axis=0)
    return z[:len(prior)], z[len(prior):]


def legendre_cov(D):
    """Legendre covariance matrix from the shared kernel."""
    C = tk._cov(D, "legendre", 4200.0)
    C = 0.5 * (C + C.T)
    if np.linalg.eigvalsh(C).min() < -1e-8:
        raise ValueError("shared Legendre covariance is not PSD")
    return C


def conditional_weights(latent, field_fraction):
    """LOO conditional weights for total-variance-normalised observations.

    Screening is represented correctly as signal variance fraction: spatial
    covariance is scaled by sqrt(f_i f_j), while independent local oscillator
    variance 1-f_i stays on the diagonal.  Merely multiplying a correlation by
    an outer screening factor and renormalising its diagonal would cancel the
    screening effect completely.
    """
    f = np.asarray(field_fraction, float)
    K = latent * np.sqrt(np.outer(f, f))
    K[np.diag_indices_from(K)] += 1.0 - f + NUGGET
    K = 0.5 * (K + K.T)
    Q = np.linalg.inv(K)
    W = np.zeros_like(K)
    for b in range(len(K)):
        o = np.arange(len(K)) != b
        W[b, o] = -Q[b, o] / Q[b, b]
    return W
