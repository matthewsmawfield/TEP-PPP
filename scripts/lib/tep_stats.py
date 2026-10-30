#!/usr/bin/env python3
"""
Shared statistics for the Phase 1 evidence battery.

- exponential correlation-length fit on distance-binned means
- block / station-clustered bootstrap
- null-control generators (temporal shuffle, spatial shuffle, phase randomisation)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# Width (in FFT frequency bins) of the raised-cosine transition at each edge of
# the spectral band-limit in bandpass().  ~2 bins suppresses Gibbs ringing
# without materially narrowing the 10-500 uHz band (43 bins on a 24 h record).
_TAPER_BINS = 2.0


def bandpass(x, dt_s, f_lo_uhz=10.0, f_hi_uhz=500.0):
    """Zero-phase band-limit a clock residual to the TEP-sensitive band.

    Defaults 10-500 uHz  ->  periods ~28 h down to ~33 min.  Removes residual
    drift and white measurement noise that swamp full-band correlation.

    IMPLEMENTATION: exact spectral (FFT) band-limit with a raised-cosine
    transition, NOT an IIR Butterworth.

    Why this changed.  A 4th-order Butterworth high-pass at f_lo = 10 uHz has
    a period of 27.8 h and a settling time of ~1/(2 pi f_lo) ~ 4.4 h; the
    correct `sosfiltfilt` padlen is ~3x that (~13 h, i.e. ~1600 samples at
    30 s).  A 24 h record is SHORTER than the settling time, so the filter can
    never settle and the previous implementation clamped padlen to 36 samples
    (18 min).  Measured against an exact band-limit on real station-days this
    lost ~40% of in-band amplitude and, critically, degraded the short-baseline
    pair correlation from +0.91 to +0.71 (0-1000 km) and +0.78 to +0.45
    (1000-3000 km) -- i.e. it was destroying the distance contrast that carries
    the TEP signal, before any analysis saw the data.

    The spectral band-limit has no settling requirement, is exactly zero-phase,
    and is the same operation the phase-coherent cross-spectral statistic
    (Papers 1-2) is defined by.  A linear detrend is applied first so the
    record's endpoints match and the implicit circular wrap introduces no
    step; the band edges use a raised-cosine taper over `_TAPER_BINS` bins to
    suppress Gibbs ringing.

    If the band-limit cannot be applied (too few samples, invalid parameters,
    NaN/inf input), the mean-removed signal is returned and a warning is
    raised so callers know the data is unfiltered.
    """
    x = np.asarray(x, float)
    # Guard against NaN / inf / constant inputs that would propagate through
    # the filter and produce all-NaN output.  Return the mean-removed signal
    # unfiltered, as documented.
    if not np.all(np.isfinite(x)):
        import warnings
        warnings.warn("bandpass: input contains NaN or inf; "
                      "returning mean-removed signal (unfiltered)", RuntimeWarning)
        # Replace inf with nan so nan_to_num handles both consistently,
        # then fill NaN positions with zero after mean removal.
        x = np.where(np.isinf(x), np.nan, x)
        return np.nan_to_num(x - np.nanmean(x), nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(x) < 1e-15:
        import warnings
        warnings.warn("bandpass: input is constant (std < 1e-15); "
                      "returning zeros", RuntimeWarning)
        return x - np.mean(x)
    if not np.isfinite(dt_s) or dt_s <= 0:
        import warnings
        warnings.warn(f"bandpass: invalid dt_s={dt_s}; "
                      "returning mean-removed signal (unfiltered)", RuntimeWarning)
        return x - np.mean(x)
    nyq = 0.5 / dt_s
    f_lo = max(f_lo_uhz * 1e-6, 0.0)
    f_hi = min(f_hi_uhz * 1e-6, 0.98 * nyq)
    if not (0 <= f_lo < f_hi <= nyq):
        import warnings
        warnings.warn(f"bandpass: invalid band {f_lo:.3e}-{f_hi:.3e} Hz for "
                      f"dt={dt_s}; returning mean-removed signal (unfiltered)",
                      RuntimeWarning)
        return x - np.mean(x)
    n = len(x)
    if n < 30:
        import warnings
        warnings.warn(f"bandpass: only {n} samples (< 30); "
                      "returning mean-removed signal (unfiltered)", RuntimeWarning)
        return x - np.mean(x)
    try:
        # Linear detrend first: the spectral filter treats the record as
        # circular, so matching the endpoints removes the artificial step that
        # would otherwise leak broadband power across the whole spectrum.
        t = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t, x, 1)
        xd = x - (intercept + slope * t)

        F = np.fft.rfft(xd)
        f = np.fft.rfftfreq(n, d=dt_s)
        df = f[1] - f[0] if n > 1 else 1.0
        H = ((f >= f_lo) & (f <= f_hi)).astype(float)
        # Raised-cosine transition over _TAPER_BINS bins at each edge to
        # suppress Gibbs ringing from a hard brick wall.
        w_edge = _TAPER_BINS * df
        if w_edge > 0:
            lo_ramp = (f >= f_lo - w_edge) & (f < f_lo)
            hi_ramp = (f > f_hi) & (f <= f_hi + w_edge)
            H[lo_ramp] = 0.5 * (1.0 - np.cos(
                np.pi * (f[lo_ramp] - (f_lo - w_edge)) / w_edge))
            H[hi_ramp] = 0.5 * (1.0 + np.cos(
                np.pi * (f[hi_ramp] - f_hi) / w_edge))
        out = np.fft.irfft(F * H, n)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite spectral band-limit output")
        return out
    except Exception as exc:
        import warnings
        warnings.warn(f"bandpass: spectral band-limit failed ({exc}); "
                      "returning mean-removed signal (unfiltered)", RuntimeWarning)
        return x - np.mean(x)


def causal_bandpass(x, dt_s, f_lo_uhz=10.0, f_hi_uhz=500.0):
    """Causal (forward-only) band-limit for real-time / look-ahead-free use.

    Uses a Butterworth IIR filter with ``sosfilt`` (forward-only, no
    ``filtfilt``).  This is strictly causal: the output at epoch *t* depends
    only on inputs at epochs <= *t*.  The zero-phase ``bandpass`` above is
    appropriate for offline spectral analysis but introduces look-ahead when
    used on reference stations whose band-limited residuals feed a causal
    prediction loop.

    The group delay of a 4th-order Butterworth high-pass at 10 uHz is
    ~1/(2*pi*f_lo) ~ 4.4 h, concentrated near the band edges.  For the
    kriging input we compute *differences* (ref_now - ref_anchor): a pure
    delay does not cancel in the difference -- it yields the change over a
    delay-shifted interval -- but because every reference is filtered
    identically the delay is common-mode and the differential signal is
    preserved to within the field's coherence time after the initial
    transient settles.

    Falls back to the mean-removed signal on failure and raises a
    RuntimeWarning so callers know the data is unfiltered.
    """
    import warnings
    from scipy.signal import butter, sosfilt
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        warnings.warn("causal_bandpass: input contains NaN or inf; "
                      "returning mean-removed signal (unfiltered)",
                      RuntimeWarning)
        x = np.where(np.isinf(x), np.nan, x)
        x = np.nan_to_num(x - np.nanmean(x), nan=0.0)
        return x
    if np.std(x) < 1e-15:
        warnings.warn("causal_bandpass: input is constant (std < 1e-15); "
                      "returning zeros", RuntimeWarning)
        return x - np.mean(x)
    if not np.isfinite(dt_s) or dt_s <= 0:
        warnings.warn(f"causal_bandpass: invalid dt_s={dt_s}; "
                      "returning mean-removed signal (unfiltered)",
                      RuntimeWarning)
        return x - np.mean(x)
    nyq = 0.5 / dt_s
    f_lo = max(f_lo_uhz * 1e-6, 1e-12)
    f_hi = min(f_hi_uhz * 1e-6, 0.98 * nyq)
    if not (0 < f_lo < f_hi <= nyq):
        warnings.warn(f"causal_bandpass: invalid band {f_lo:.3e}-{f_hi:.3e} Hz "
                      f"for dt={dt_s}; returning mean-removed signal "
                      "(unfiltered)", RuntimeWarning)
        return x - np.mean(x)
    n = len(x)
    if n < 30:
        warnings.warn(f"causal_bandpass: only {n} samples (< 30); "
                      "returning mean-removed signal (unfiltered)",
                      RuntimeWarning)
        return x - np.mean(x)
    try:
        sos = butter(4, [f_lo, f_hi], btype='band',
                     output='sos', fs=1.0 / dt_s)
        # Causal detrend: fit the line only to samples up to the current
        # point.  We use a recursive mean removal instead of a per-sample
        # polyfit (which would be O(n^2)).  The high-pass at f_lo removes
        # any residual linear trend, so a causal mean removal is sufficient
        # for IIR stability.
        cumsum = np.cumsum(x)
        counts = np.arange(1, n + 1, dtype=float)
        causal_mean = cumsum / counts
        xd = x - causal_mean
        out = sosfilt(sos, xd)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite causal bandpass output")
        return out
    except Exception as exc:
        import warnings
        warnings.warn(f"causal_bandpass: Butterworth band-limit failed "
                      f"({exc}); returning mean-removed signal (unfiltered)",
                      RuntimeWarning)
        return x - np.mean(x)


def exp_model(r, A, lam, C0):
    return A * np.exp(-r / lam) + C0


def fit_correlation_length(dist_km, value, weight=None, lam_bounds=(50.0, 40000.0)):
    """
    Fit C(r) = A exp(-r/lam) + C0 to already-binned (dist, value) points.
    Returns dict with lam, sigma_lam, A, C0, R2, n.
    """
    d = np.asarray(dist_km, float)
    v = np.asarray(value, float)
    ok = np.isfinite(d) & np.isfinite(v)
    d, v = d[ok], v[ok]
    if weight is not None:
        w = np.asarray(weight, float)[ok]
    else:
        w = np.ones_like(d)
    if len(d) < 4:
        return {"lam": np.nan, "sigma_lam": np.nan, "A": np.nan, "C0": np.nan, "R2": np.nan, "n": len(d)}
    sigma = 1.0 / np.sqrt(np.clip(w, 1e-9, None))
    p0 = [max(v.max() - v.min(), 1e-3), 3000.0, float(np.median(v))]
    lo = [0.0, lam_bounds[0], -abs(v).max() - 1]
    hi = [10 * (abs(v).max() + 1), lam_bounds[1], abs(v).max() + 1]
    try:
        popt, pcov = curve_fit(exp_model, d, v, p0=p0, sigma=sigma,
                               absolute_sigma=True, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return {"lam": np.nan, "sigma_lam": np.nan, "A": np.nan, "C0": np.nan, "R2": np.nan, "n": len(d)}
    pred = exp_model(d, *popt)
    ss_res = np.sum(w * (v - pred) ** 2)
    ss_tot = np.sum(w * (v - np.average(v, weights=w)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"lam": float(popt[1]), "sigma_lam": float(np.sqrt(np.diag(pcov))[1]),
            "A": float(popt[0]), "C0": float(popt[2]), "R2": float(r2), "n": int(len(d)),
            "lam_at_upper_bound": bool(abs(popt[1] - lam_bounds[1]) < 1.0),
            "lam_at_lower_bound": bool(abs(popt[1] - lam_bounds[0]) < 1.0)}


def bin_by_distance(dist_km, value, n_bins=30, dmin=50.0, dmax=None, min_count=10, log=True):
    d = np.asarray(dist_km, float)
    v = np.asarray(value, float)
    ok = np.isfinite(d) & np.isfinite(v) & (d >= dmin)
    d, v = d[ok], v[ok]
    if len(d) < min_count:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    if dmax is None:
        dmax = np.percentile(d, 99.5)
    # Exclude (do not clip) distances above dmax so the last bin is not
    # contaminated by the truncated tail.
    ok = d <= dmax
    d, v = d[ok], v[ok]
    if len(d) < min_count:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    edges = np.geomspace(dmin, dmax, n_bins + 1) if log else np.linspace(dmin, dmax, n_bins + 1)
    # Digitise with right=False (default): a value equal to an edge goes into
    # the bin to the right.  Values equal to dmax would get index n_bins
    # (out of range), so clip the index to the last valid bin.
    idx = np.digitize(d, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers, means, sems, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_count:
            continue
        centers.append(np.sqrt(edges[b] * edges[b + 1]) if log else 0.5 * (edges[b] + edges[b + 1]))
        means.append(float(np.mean(v[m])))
        sems.append(float(np.std(v[m], ddof=1) / np.sqrt(m.sum())))
        counts.append(int(m.sum()))
    return (np.array(centers), np.array(means), np.array(sems), np.array(counts))


def station_clustered_bootstrap(stat_fn, station_names, n_boot=500, rng=None,
                                n_jobs=8, backend="threading"):
    """
    stat_fn(resample_list) -> scalar, where resample_list is a list of station
    names drawn WITH replacement.  Returns (point, lo95, hi95, samples).

    This resamples station names (spatial clustering) to account for non-
    independence of baselines sharing a station.  It does NOT resample day
    blocks, so day-to-day temporal correlation is not captured; the CI may
    be anti-conservative if day-level variation dominates.

    IMPORTANT: stat_fn must honour the multiplicity of the resample list.
    If a station appears k times, every pair involving it must be weighted
    by k (or the pair multiplicity k_i * k_j).  Collapsing the resample to
    a set (e.g. ``set(sel)``) silently discards the with-replacement
    structure and produces a random-subset CI, not a bootstrap CI.
    """
    from joblib import Parallel, delayed
    rng = rng or np.random.default_rng(0)
    names = list(station_names)
    point = stat_fn(names)
    
    resamples = [list(rng.choice(names, size=len(names), replace=True)) for _ in range(n_boot)]
    
    def _run_stat(rs):
        try:
            return stat_fn(rs)
        except Exception:
            return np.nan

    # threading backend by default: the bootstrap stat closure often captures a
    # large read-only pair table (millions of records for a full-year global
    # network) that loky would pickle per worker and OOM on.  stat_fn is a pure
    # read of that table plus GIL-releasing curve_fit, so threads are safe and
    # do not copy it.
    samples = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_stat)(rs) for rs in resamples)
    
    samples = np.array([s for s in samples if np.isfinite(s)])
    if len(samples) < 10:
        return point, np.nan, np.nan, samples
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)), samples


# --------------------------------------------------------------- null generators
def null_temporal_shuffle(series: dict, rng):
    """Circularly shift each station's residual series by a random lag."""
    out = {}
    for nm, v in series.items():
        n = len(v["resid"])
        k = int(rng.integers(n // 20, n - n // 20))
        w = dict(v)
        w["resid"] = np.roll(v["resid"], k)
        out[nm] = w
    return out


def null_spatial_shuffle(series: dict, rng):
    """Permute the station->position assignment (keep the time series)."""
    names = list(series)
    perm = list(rng.permutation(names))
    while perm == names and len(names) > 1:
        perm = list(rng.permutation(names))
    out = {}
    for nm, src in zip(names, perm):
        w = dict(series[nm])
        w["station"] = series[src]["station"]
        out[nm] = w
    return out


def null_phase_randomise(series: dict, rng):
    """Randomise Fourier phases, preserving each station's power spectrum.

    For even-length series the Nyquist bin must have zero phase (it is purely
    real in the DFT of a real signal); failing to force this produces a
    complex-valued reconstruction with an incorrect power spectrum.  The DC
    and Nyquist bins are kept at their original signed real values (not
    abs()) so the mean and fastest-oscillating component are preserved
    exactly.
    """
    out = {}
    for nm, v in series.items():
        x = v["resid"]
        n = len(x)
        F = np.fft.rfft(x)
        ph = rng.uniform(0, 2 * np.pi, len(F))
        ph[0] = 0.0  # DC component must be real
        if n % 2 == 0:
            ph[-1] = 0.0  # Nyquist bin must be real for even-length series
        Fr = np.abs(F) * np.exp(1j * ph)
        # Preserve signed real values for DC and Nyquist bins
        Fr[0] = F[0].real
        if n % 2 == 0:
            Fr[-1] = F[-1].real
        w = dict(v)
        w["resid"] = np.fft.irfft(Fr, n)
        out[nm] = w
    return out


# --------------------------------------------------------------------- CSD
def pair_correlations_csd(series: dict, f_lo_uhz=10.0, f_hi_uhz=500.0):
    """Cross-spectral-density (CSD) phase-alignment pair correlations.

    For each station pair, compute the Papers 1-2 phase-alignment index
    cos(arg(Σ_f S_ij(f))) in the 10-500 uHz band, where S_ij(f) is the
    complex cross-spectrum.  Earlier versions of this function used
    band-averaged coherence times |mean(magnitude-weighted phasor)|, which
    is not the statistic Papers 1-2 report; that amplitude-domain product
    fails for the same reason Paper 1 identifies.

    NOTE: the returned value is the real number cos(arg(Σ S)) and can be
    negative.  It is the correct signed phase-alignment index, not a
    non-negative coherence strength.

    Returns (distances_km, correlations) arrays.
    """
    import itertools
    from scripts.lib import tep_data as td

    names = list(series)
    if len(names) < 2:
        return np.array([]), np.array([])

    # Band-pass each station's residual and record per-station dt
    cols = {}
    dts = {}
    for n in names:
        x = np.asarray(series[n]["resid"], float)
        dt = float(np.nanmedian(np.diff(series[n]["t"])))
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0
        cols[n] = bandpass(x, dt, f_lo_uhz, f_hi_uhz)
        dts[n] = dt

    d_list, c_list = [], []
    for i, j in itertools.combinations(range(len(names)), 2):
        ni, nj = names[i], names[j]
        xi, xj = cols[ni], cols[nj]
        dt_i, dt_j = dts[ni], dts[nj]
        # Resample to common cadence if mixed.  Using the slower cadence
        # (larger dt) as the common grid; decimate the faster series with
        # an anti-aliasing filter so high-frequency content does not alias
        # into the 10-500 uHz band.
        if abs(dt_i - dt_j) > 0.5 * min(dt_i, dt_j):
            dt_pair = max(dt_i, dt_j)
            if dt_i < dt_pair:
                q = int(round(dt_pair / dt_i))
                if q > 1 and len(xi) >= q * 10:
                    from scipy.signal import decimate
                    xi = decimate(xi, q, ftype='fir', zero_phase=True)
            if dt_j < dt_pair:
                q = int(round(dt_pair / dt_j))
                if q > 1 and len(xj) >= q * 10:
                    from scipy.signal import decimate
                    xj = decimate(xj, q, ftype='fir', zero_phase=True)
        else:
            dt_pair = 0.5 * (dt_i + dt_j)
        n = min(len(xi), len(xj))
        if n < 30:
            continue
        xi, xj = xi[:n], xj[:n]
        # Cross-spectral density via Welch — use the pair's common dt
        from scipy.signal import csd, welch
        fs = 1.0 / dt_pair
        f, Pxy = csd(xi, xj, fs=fs, nperseg=min(n, 256))
        _, Pxx = welch(xi, fs=fs, nperseg=min(n, 256))
        _, Pyy = welch(xj, fs=fs, nperseg=min(n, 256))
        # Band-limit
        mask = (f >= f_lo_uhz * 1e-6) & (f <= f_hi_uhz * 1e-6)
        if mask.sum() < 3:
            continue
        Pxy_b = Pxy[mask]
        Pxx_b = Pxx[mask]
        Pyy_b = Pyy[mask]
        # Mask frequency bins where either station has near-zero power,
        # to keep only well-conditioned cross-spectral bins.
        power_ok = (Pxx_b > 1e-30) & (Pyy_b > 1e-30)
        if power_ok.sum() < 3:
            continue
        Pxy_b = Pxy_b[power_ok]
        # Papers 1-2 phase-alignment index: cosine of the argument of the
        # complex cross-spectrum summed over the band.
        S = np.sum(Pxy_b)
        if not np.isfinite(S) or S == 0:
            continue
        cij = float(np.cos(np.angle(S)))
        d_list.append(td.great_circle_km(series[ni]["station"], series[nj]["station"]))
        c_list.append(cij)
    return np.array(d_list), np.array(c_list)


def fit_all_models(dist_km, value, weight=None):
    """Fit multiple candidate models to binned (dist, value) data and rank by AIC.

    Models: exponential, power-law, gaussian, constant.
    Returns dict with per-model R2/AIC/BIC and the best model name.
    """
    from scipy.optimize import curve_fit

    d = np.asarray(dist_km, float)
    v = np.asarray(value, float)
    ok = np.isfinite(d) & np.isfinite(v)
    d, v = d[ok], v[ok]
    if weight is not None:
        w = np.asarray(weight, float)[ok]
    else:
        w = np.ones_like(d)
    if len(d) < 5:
        return {"best_model": "none", "models": {}, "delta_aic_exponential_vs_best": np.nan}

    sigma = 1.0 / np.sqrt(np.clip(w, 1e-9, None))
    n = len(d)

    def _information_criteria(ss_res, k):
        """Profile-likelihood AIC/AICc/BIC for weighted binned data.

        The weighted RSS ``ss_res = sum(w * residual^2)`` is profiled over
        the unknown common noise scale, giving the standard Gaussian
        profile-likelihood ``n * log(ss_res / sum(w)) + 2k``.  This is
        scale-invariant, unlike the raw ``ss_res + 2k`` form which is only
        valid when the noise variance is known to be 1 in the weight units.
        AICc is used for ranking because these curves can contain as few
        as 6--15 independent station bins.
        """
        scale = max(float(ss_res) / max(float(np.sum(w)), 1e-300), 1e-300)
        neg2loglike = n * np.log(scale)
        aic = neg2loglike + 2 * k
        aicc = (aic + (2 * k * (k + 1)) / (n - k - 1)
                if n > k + 1 else np.inf)
        bic = neg2loglike + k * np.log(n)
        return float(aic), float(aicc), float(bic)

    models = {}

    # Exponential: A * exp(-r/lam) + C0
    try:
        p0 = [max(v.max() - v.min(), 1e-3), 3000.0, float(np.median(v))]
        lo = [0.0, 50.0, -abs(v).max() - 1]
        hi = [10 * (abs(v).max() + 1), 40000.0, abs(v).max() + 1]
        popt, pcov = curve_fit(exp_model, d, v, p0=p0, sigma=sigma,
                               absolute_sigma=True, bounds=(lo, hi), maxfev=20000)
        pred = exp_model(d, *popt)
        ss_res = np.sum(w * (v - pred) ** 2)
        ss_tot = np.sum(w * (v - np.average(v, weights=w)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        aic, aicc, bic = _information_criteria(ss_res, 3)
        sig = np.sqrt(np.diag(pcov))
        models["exponential"] = {"R2": float(r2), "AIC": aic, "AICc": aicc, "BIC": bic,
                                 "params": popt.tolist(), "sigma_params": sig.tolist()}
    except Exception:
        models["exponential"] = {"R2": np.nan, "AIC": np.inf, "AICc": np.inf,
                                 "BIC": np.inf, "params": []}

    # Power-law: A * (r/r0)^(-alpha) + C0
    def power_model(r, A, alpha, C0):
        return A * np.power(np.clip(r / 1000.0, 1e-6, None), -alpha) + C0
    try:
        popt, pcov = curve_fit(power_model, d, v, p0=[1.0, 1.0, float(np.median(v))],
                            sigma=sigma, absolute_sigma=True, maxfev=20000)
        pred = power_model(d, *popt)
        ss_res = np.sum(w * (v - pred) ** 2)
        ss_tot = np.sum(w * (v - np.average(v, weights=w)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        aic, aicc, bic = _information_criteria(ss_res, 3)
        sig = np.sqrt(np.diag(pcov))
        models["power_law"] = {"R2": float(r2), "AIC": aic, "AICc": aicc, "BIC": bic,
                               "params": popt.tolist(), "sigma_params": sig.tolist()}
    except Exception:
        models["power_law"] = {"R2": np.nan, "AIC": np.inf, "AICc": np.inf,
                               "BIC": np.inf, "params": []}

    # Gaussian: A * exp(-(r/lam)^2) + C0
    def gauss_model(r, A, lam, C0):
        return A * np.exp(-(r / lam) ** 2) + C0
    try:
        # lam MUST be bounded positive: gauss_model is even in lam, so an
        # unbounded fit can converge to a negative "correlation length"
        # (observed: -630 km on the EU precise arm).  Bound it to the same
        # range fit_correlation_length uses.
        popt, pcov = curve_fit(gauss_model, d, v, p0=[1.0, 3000.0, float(np.median(v))],
                            sigma=sigma, absolute_sigma=True, maxfev=20000,
                            bounds=([0.0, 50.0, -abs(v).max() - 1],
                                    [10 * (abs(v).max() + 1), 40000.0,
                                     abs(v).max() + 1]))
        pred = gauss_model(d, *popt)
        ss_res = np.sum(w * (v - pred) ** 2)
        ss_tot = np.sum(w * (v - np.average(v, weights=w)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        aic, aicc, bic = _information_criteria(ss_res, 3)
        sig = np.sqrt(np.diag(pcov))
        models["gaussian"] = {"R2": float(r2), "AIC": aic, "AICc": aicc, "BIC": bic,
                              "params": popt.tolist(), "sigma_params": sig.tolist()}
    except Exception:
        models["gaussian"] = {"R2": np.nan, "AIC": np.inf, "AICc": np.inf,
                              "BIC": np.inf, "params": []}

    # Constant: C0
    pred = np.full_like(v, np.average(v, weights=w))
    ss_res = np.sum(w * (v - pred) ** 2)
    ss_tot = np.sum(w * (v - np.average(v, weights=w)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    aic, aicc, bic = _information_criteria(ss_res, 1)
    models["constant"] = {"R2": float(r2), "AIC": aic, "AICc": aicc,
                          "BIC": bic, "params": [float(np.average(v, weights=w))]}

    ranked = sorted(models, key=lambda m: models[m]["AICc"])
    best = ranked[0]
    delta_aic = models["exponential"]["AIC"] - models[best]["AIC"]
    delta_aicc = models["exponential"]["AICc"] - models[best]["AICc"]
    return {"best_model": best, "selection_criterion": "AICc",
            "models": models, "ranked_by_aicc": ranked,
            "delta_aic_exponential_vs_best": float(delta_aic),
            "delta_aicc_exponential_vs_best": float(delta_aicc)}
