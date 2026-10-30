#!/usr/bin/env python3
"""
STEP 03b: Control battery for the LOO-CV clock-variance-reduction claim.

Headline result (Step 03): predicting one European station's detrended SPP
receiver-clock series from 4 neighbours removes ~51% of the residual RMSE.

Question the manuscript's null-vector test was meant to answer but does not:
is that reduction specific to the TEP spatial kernel and its geometry, or is it
the generic behaviour of any spatial / common-mode predictor on a small cluster
of stations that share satellite-product (orbit/clock) errors?

All numbers are honest out-of-sample (no beta rescaling):
    reduction% = 100 * (1 - RMSE(obs - pred) / RMSE(obs))

Isotropic exponential kriging with a fixed covariance is a *constant* linear
combination of the neighbour series, so it is computed analytically and is fast.
The full DynamicTEPKernel is also run for direct comparison.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from core.ppp_kernel import DynamicTEPKernel  # noqa: E402
from scripts.lib import tep_data as td  # noqa: E402

STATIONS = ["ZIMM", "GRAS", "GENO", "BRUX", "WTZR"]
DOYS = [100, 101, 102, 103, 104, 105, 106]
YEAR = 2024
DATA_DIRS = [PROJECT_ROOT / "data" / "ppp_processed",
             PROJECT_ROOT / "data" / "ppp_longitudinal"]
SPF = 0.44  # spatial fraction used by Step 03
RNG = np.random.default_rng(20260831)


def load_day(sta, doy):
    """Return the raw clock series (NaNs retained) so the caller can align
    epochs across stations; independent per-station NaN removal would shift
    the series relative to each other."""
    for d in DATA_DIRS:
        f = d / f"{sta}_{YEAR}{doy:03d}.npz"
        if f.exists():
            try:
                z = np.load(f, allow_pickle=False)
            except Exception:
                # Corrupt / truncated download (e.g. 0-byte file)
                continue
            clk = z["clock_bias_ns"]  # broadcast SPP control battery
            clk = np.asarray(clk, float)
            pos = np.array([float(z["x_mean"]), float(z["y_mean"]), float(z["z_mean"])]) / 1000.0
            return clk, pos
    return None, None


def detrend(v):
    t = np.arange(len(v))
    return v - np.polyval(np.polyfit(t, v, 1), t)


def load_all():
    days = {}
    for doy in DOYS:
        series, pos = {}, {}
        for s in STATIONS:
            clk, p = load_day(s, doy)
            if clk is None or len(clk) < 100:
                break
            series[s], pos[s] = clk, p
        if len(series) < len(STATIONS):
            continue
        # Truncate to the minimum length so all series are epoch-aligned
        # (same time index = same physical epoch), then keep only epochs
        # where every station is finite.
        n_min = min(len(series[s]) for s in STATIONS)
        truncated = {s: series[s][:n_min] for s in STATIONS}
        m = np.logical_and.reduce([np.isfinite(truncated[s]) for s in STATIONS])
        if int(m.sum()) < 100:
            continue
        S = np.column_stack([detrend(truncated[s][m]) for s in STATIONS])   # (n, 5)
        P = np.array([pos[s] for s in STATIONS])                          # (5, 3)
        days[doy] = {"S": S, "P": P, "n": int(m.sum())}
    return days


def reduction(obs, pred):
    base = np.sqrt(np.mean(obs ** 2))
    resid = np.sqrt(np.mean((obs - pred) ** 2))
    if not np.isfinite(resid) or not np.isfinite(base) or base <= 0:
        return float("nan")
    return 100.0 * (1.0 - resid / base)


def _ecef_to_stations(P):
    """Convert (n, 3) ECEF km array to list of Station objects for great-circle distance."""
    stations = []
    for p in P:
        lat, lon = td.ecef_to_geodetic(p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0)
        stations.append(td.Station("", p, lat, lon))
    return stations


def _gc_dist_matrix(stations_a, stations_b):
    """Great-circle distance matrix (km) between two lists of Station objects."""
    n_a, n_b = len(stations_a), len(stations_b)
    D = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            D[i, j] = td.great_circle_km(stations_a[i], stations_b[j])
    return D


def iso_weights(known_P, target_p, lam, nugget_frac):
    """Constant kriging weights for exp(-r/lam) covariance, unit sill + nugget.

    Uses great-circle distance (consistent with tep_kernel.py, step_05b,
    step_12, step_25).  The previous implementation used Euclidean chord
    distance in ECEF, which underestimates great-circle distance and
    introduces a methodological inconsistency with the rest of the pipeline.

    Falls back to uniform weights when the system is ill-conditioned (which
    happens at very large lambda where all covariances collapse to ~1 and
    K_nn becomes near-singular).  Previously this raised divide-by-zero /
    overflow RuntimeWarnings and produced non-finite reductions.
    """
    known_sts = _ecef_to_stations(known_P)
    target_st = _ecef_to_stations(target_p.reshape(1, -1))
    d_nn = _gc_dist_matrix(known_sts, known_sts)
    K_nn = np.exp(-d_nn / lam) + (nugget_frac + 1e-10) * np.eye(len(known_P))
    d_nt = _gc_dist_matrix(known_sts, target_st)[:, 0]
    K_nt = np.exp(-d_nt / lam)
    try:
        with np.errstate(all="ignore"):
            w = np.linalg.solve(K_nn, K_nt)
        if not np.all(np.isfinite(w)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        w = np.full(len(known_P), 1.0 / len(known_P))
    return w


def loo_iso(days, lam, P_override=None):
    per_day = []
    for dd in days.values():
        S, P = dd["S"], dd["P"]
        Pg = P if P_override is None else P_override
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            sig2 = np.var(S[:, t])
            # nugget as fraction of unit sill: (1-SPF)/SPF  (matches Step 03 ratio)
            w = iso_weights(Pg[others], Pg[t], lam, (1.0 - SPF) / SPF)
            with np.errstate(all="ignore"):
                pred = S[:, others] @ w
            reds.append(reduction(S[:, t], pred))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_network_mean(days):
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            reds.append(reduction(S[:, t], S[:, others].mean(axis=1)))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def _uniform_kriging_weights(n, nugget_frac):
    """Analytic lambda->infinity kriging weights (geometry-free baseline).

    With K_nn = ones + nugget*I and K_nt = ones, symmetry gives
    w_i = 1/(n + nugget) for every i, i.e. a SCALED network mean with
    scaling n/(n+nugget).  This is the geometry-free predictor that shares
    the iso kriging nugget scaling, so comparing iso(lam) against it isolates
    the *geometry-specific* contribution from the nugget-induced rescaling.
    """
    w = np.full(n, 1.0 / (n + nugget_frac))
    return w


def loo_uniform_kriging(days):
    """Geometry-free kriging (lambda=inf, same nugget as iso) -- scaled mean."""
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            n = len(others)
            w = _uniform_kriging_weights(n, (1.0 - SPF) / SPF)
            with np.errstate(all="ignore"):
                pred = S[:, others] @ w
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(reduction(S[:, t], pred))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_mean_optimal_beta(days):
    """Network mean with per-fold optimal beta computed from REFERENCES only.

    beta = <target_ref_proxy, mean_ref> / <mean_ref, mean_ref>, where
    target_ref_proxy is the leave-one-out-within-references prediction of the
    held-out reference.  This is an honest (no held-out-target leakage) self-
    scaled mean -- the strongest geometry-free common-mode predictor.  It
    brackets how much of iso4200's edge over the raw mean is just rescaling.
    """
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            mean_ref = S[:, others].mean(axis=1)
            # optimal beta from references only: LOO within the reference set
            betas = []
            for r in others:
                r_others = [j for j in others if j != r]
                if len(r_others) == 0:
                    continue
                m = S[:, r_others].mean(axis=1)
                denom = float(np.dot(m, m))
                if denom > 1e-12:
                    betas.append(float(np.dot(S[:, r], m) / denom))
            beta = float(np.mean(betas)) if betas else 1.0
            if not np.isfinite(beta):
                beta = 1.0
            reds.append(reduction(S[:, t], beta * mean_ref))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_ols(days):
    """In-sample OLS upper bound: regress target on 4 others (no intercept)."""
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            A = S[:, others]
            with np.errstate(all="ignore"):
                coef, *_ = np.linalg.lstsq(A, S[:, t], rcond=1e-5)
                pred = A @ coef
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(reduction(S[:, t], pred))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_tep_dynamic(days):
    ref_epoch = datetime(YEAR, 1, 1)
    per_day = []
    for doy, dd in days.items():
        S, P, n = dd["S"], dd["P"], dd["n"]
        start = ref_epoch + timedelta(days=doy - 1)
        reds = []
        for t in range(5):
            others = [j for j in range(5) if j != t]
            # Use reference-station std (not the held-out target) to avoid
            # leaking target information into the kernel parameterisation.
            sig = float(np.mean([np.std(S[:, j]) for j in others]))
            k = DynamicTEPKernel(
                base_lambda_km=4200.0, base_sigma_ns=sig * np.sqrt(SPF),
                enable_rotation_effects=False, enable_velocity_coupling=True,
                enable_local_time=True, velocity_coupling_factor=0.35,
                diurnal_coupling_factor=0.1, seasonal_coupling_factor=-0.2,
                enable_chandler_wobble=True, chandler_coupling_factor=0.15,
                enable_semiannual_nutation=True,
                semiannual_nutation_coupling_factor=0.12,
                semiannual_nutation_phase_offset_days=60.0)
            nv = sig ** 2 * (1 - SPF)
            pred = np.empty(n)
            for i in range(n):
                mu, _, _ = k.predict(known_coords=P[others], known_values=S[i, others],
                                     target_coords=P[t], noise_variance=nv,
                                     epoch_utc=start + timedelta(seconds=i * 30),
                                     reference_epoch=start)
                pred[i] = np.atleast_1d(mu)[0]
            reds.append(reduction(S[:, t], pred))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def main():
    days = load_all()
    print(f"Loaded {len(days)} complete days: {sorted(days)}")
    R = {}

    R["network_mean"] = loo_network_mean(days)
    R["uniform_kriging_scaled_mean"] = loo_uniform_kriging(days)
    R["mean_optimal_beta_refsonly"] = loo_mean_optimal_beta(days)
    R["ols_insample_upper_bound"] = loo_ols(days)
    for lam in [100, 300, 1000, 2000, 4200, 10000, 40000, 200000]:
        R[f"iso_kriging_lambda_{lam}km"] = loo_iso(days, lam)

    # scrambled-geometry null: iso kriging lambda=4200 with permuted station->position map.
    # Exact enumeration of ALL 44 derangements of 5 stations (every station
    # receives a wrong position; no fixed points, no sampling noise).
    scr = []
    base_P = days[next(iter(days))]["P"]
    from itertools import permutations
    ders = [p for p in permutations(range(5)) if all(p[i] != i for i in range(5))]
    scr = np.array([loo_iso(days, 4200.0, P_override=base_P[list(p)])[0] for p in ders])
    R["scrambled_geometry_null"] = [float(scr.mean()), float(scr.std()),
                                    float(np.percentile(scr, 2.5)), float(np.percentile(scr, 97.5))]

    # far-field null: replace station positions with 5 points spread >8000 km apart
    far_P = np.array([[6371, 0, 0], [-6371, 0, 0], [0, 6371, 0], [0, -6371, 0], [0, 0, 6371]], float)
    R["far_field_geometry_iso4200"] = loo_iso(days, 4200.0, P_override=far_P)

    R["tep_dynamic_kernel"] = loo_tep_dynamic(days)

    print("\n=== LOO-CV honest out-of-sample RMSE reduction (%) ===")
    for name, val in R.items():
        if isinstance(val, list):
            print(f"  {name:34s}: {val[0]:6.2f}   null 95% CI [{val[2]:.2f}, {val[3]:.2f}]")
        else:
            print(f"  {name:34s}: {val[0]:6.2f} +/- {val[1]:.2f}")

    out = PROJECT_ROOT / "results" / "outputs" / "step_03b_controls.json"
    out.write_text(json.dumps({"n_days": len(days), "results": R,
                               "timestamp": datetime.now().isoformat()}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
