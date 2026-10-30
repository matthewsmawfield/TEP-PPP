"""Causal GNSS clock-state unwrapping.

Receiver clocks in this pipeline appear in metres of range (c * dt).  Two
modulo conventions show up in the products:

  * ppp_dense stores the estimator state modulo 1,000,000 m
  * RINEX / steered receivers wrap at 1 ms = 299,792.458 m

A sequential unwrap of both periods, plus an optional per-epoch alignment
onto the live-network median branch, is a representation change: it does
not use future samples of the target and does not consult any predictor.
"""
from __future__ import annotations

import numpy as np

C_M_PER_NS = 0.299792458
C_M_PER_S = 299_792_458.0
MS_WRAP_M = C_M_PER_S * 1.0e-3          # 299_792.458 m
PRODUCT_WRAP_M = 1_000_000.0
DEFAULT_PERIODS = (PRODUCT_WRAP_M, MS_WRAP_M)


def unwrap_series(x, periods=DEFAULT_PERIODS):
    """Causally unwrap a 1-D clock series for each known modulo period.

    Each correction uses only the current and preceding sample.  A prefix
    of the input therefore produces the identical prefix of the output.
    """
    out = np.asarray(x, float).copy()
    if out.size == 0:
        return out
    for period in periods:
        if not period or period <= 0:
            continue
        corr = 0.0
        prev = np.nan
        half = 0.5 * period
        for i, value in enumerate(out):
            if not np.isfinite(value):
                continue
            adjusted = value + corr
            if np.isfinite(prev):
                jump = adjusted - prev
                nwrap = int(np.rint(jump / period))
                if nwrap and abs(jump) > half:
                    corr -= nwrap * period
                    adjusted = value + corr
            out[i] = adjusted
            prev = adjusted
    return out


def align_epoch(values, periods=DEFAULT_PERIODS, reference=None):
    """Unwrap each station at one epoch onto a common branch.

    If *reference* is None the finite-sample median is used.  Passing the
    live-reference median (excluding the target) is the causal scoring
    convention for an outage epoch.
    """
    v = np.asarray(values, float).copy()
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return v
    if reference is None or not np.isfinite(reference):
        reference = float(np.median(v[finite]))
    for period in periods:
        if not period or period <= 0:
            continue
        nwrap = np.rint((v - reference) / period)
        nwrap = np.where(finite, nwrap, 0.0)
        v = v - nwrap * period
    return v


def align_network_epochs(X, periods=DEFAULT_PERIODS):
    """Per-epoch network-median branch alignment of an (epoch, station) matrix."""
    X = np.asarray(X, float).copy()
    if X.ndim != 2:
        raise ValueError("align_network_epochs expects a 2-D array")
    for i in range(X.shape[0]):
        X[i] = align_epoch(X[i], periods=periods)
    return X


def percentile_abs(abs_errors, q):
    """Percentile of absolute errors.  Empty -> nan."""
    a = np.asarray(abs_errors, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))
