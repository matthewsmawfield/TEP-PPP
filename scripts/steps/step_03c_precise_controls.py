#!/usr/bin/env python3
"""
TEP-PPP Step 03c: Precise-clock control battery
================================================

The broadcast-SPP control battery (Step 03b / Step 06) shows that the ~51%
LOO-CV clock reduction on the 5-station European cluster is consistent with
generic common-mode satellite-product error removal: the TEP kernel does not
beat the network mean, the reduction is flat for any lambda >= 2000 km, and
scrambled geometry does not collapse it.

The discriminating test is to repeat the *same* control battery on precise
clock residuals, from which satellite clock and orbit errors have been removed
using IGS final products.  If a TEP-specific spatial field is present, the
residual reduction should (A) outperform a plain network mean, (B) depend on
the correlation length, and (D) collapse under temporal shuffling.  Test C
(scrambled geometry) is expected to be inconclusive at this 500-1200 km
aperture, which is small relative to the 4200 km correlation length.

The precise receiver-clock residuals live in ``data/ppp_longitudinal`` as the
``precise_clock_bias_ns`` field (300 s cadence, 288 epochs/day).  This step
uses the first 20 complete common days for the 5 European stations, matching
the cadence stated in the manuscript.  NOTE: the first complete common days in
the dataset fall in January 2022 (dates are recorded in the output JSON);
2024 longitudinal files only carry the broadcast field.  All numbers are
honest out-of-sample LOO-CV reductions with no post-hoc beta rescaling; the
kriging nugget (spatial fraction) is estimated per fold from the reference
stations only, and epochs are aligned across stations by common-finite-epoch
intersection.

Output: ``results/outputs/step_03c_precise_controls.json``
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td  # noqa: E402

STATIONS = ["ZIMM", "GRAS", "GENO", "BRUX", "WTZR"]
SPF = 0.44          # spatial fraction (mean cross-station correlation), matches Step 03
N_DAYS = 20         # precise-clock control battery sample size
RNG = np.random.default_rng(20260831)


def _load(sta, date, field="precise_clock_bias_ns"):
    f = PROJECT_ROOT / "data" / "ppp_longitudinal" / f"{sta}_{date}.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f, allow_pickle=False)
    except Exception:
        return None
    if field not in z:
        return None
    clk = np.asarray(z[field], float)
    if int(np.isfinite(clk).sum()) < 100:
        return None
    pos = np.array([float(z["x_mean"]), float(z["y_mean"]), float(z["z_mean"])]) / 1000.0
    return clk, pos   # raw series (NaNs retained; caller aligns epochs)


def _detrend_clip(v):
    t = np.arange(len(v))
    r = v - np.polyval(np.polyfit(t, v, 1), t)
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    return np.clip(r, -8 * s, 8 * s)


def _usable_dates(sta):
    ds = set()
    for f in (PROJECT_ROOT / "data" / "ppp_longitudinal").glob(f"{sta}_*.npz"):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:
            continue
        if "precise_clock_bias_ns" not in z:
            continue
        pc = np.asarray(z["precise_clock_bias_ns"], float)
        if len(pc) >= 100 and np.isfinite(pc).mean() > 0.95:
            ds.add(f.stem.split("_")[1])
    return ds


def load_days(n_days=N_DAYS):
    common = None
    for s in STATIONS:
        d = _usable_dates(s)
        common = d if common is None else common & d
    common = sorted(common)
    days = []
    for date in common:
        series, pos = {}, {}
        ok = True
        for s in STATIONS:
            r = _load(s, date)
            if r is None:
                ok = False
                break
            series[s], pos[s] = r
        if not ok:
            continue
        n_min = min(len(series[s]) for s in STATIONS)
        truncated = {s: series[s][:n_min] for s in STATIONS}
        m = np.logical_and.reduce([np.isfinite(truncated[s]) for s in STATIONS])
        if int(m.sum()) < 100:
            continue
        S = np.column_stack([_detrend_clip(truncated[s][m]) for s in STATIONS])
        P = np.array([pos[s] for s in STATIONS])
        days.append({"date": date, "S": S, "P": P, "n": int(m.sum())})
        if len(days) >= n_days:
            break
    return days


def fold_spatial_fraction(S, cols):
    """LOO-honest spatial fraction: mean cross-correlation of the REFERENCE
    stations only (the held-out station never contributes)."""
    C = np.corrcoef(S[:, cols], rowvar=False)
    iu = np.triu_indices_from(C, 1)
    return float(np.clip(C[iu].mean(), 1e-3, 0.999))


def reduction(obs, pred):
    base = np.sqrt(np.mean(obs ** 2))
    if base <= 1e-15:
        return 0.0
    return 100.0 * (1.0 - np.sqrt(np.mean((obs - pred) ** 2)) / base)


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
    # Use great-circle distance (consistent with tep_kernel.py, step_05b,
    # step_12, step_25).  The previous implementation used Euclidean chord
    # distance in ECEF, which underestimates great-circle distance and
    # introduces a methodological inconsistency with the rest of the pipeline.
    known_sts = _ecef_to_stations(known_P)
    target_st = _ecef_to_stations(target_p.reshape(1, -1))
    d_nn = _gc_dist_matrix(known_sts, known_sts)
    # nugget as fraction of unit sill + small jitter for numerical stability
    K_nn = np.exp(-d_nn / lam) + nugget_frac * np.eye(len(known_P)) + 1e-10 * np.eye(len(known_P))
    d_nt = _gc_dist_matrix(known_sts, target_st)[:, 0]
    K_nt = np.exp(-d_nt / lam)
    return np.linalg.solve(K_nn, K_nt)


def loo_iso(days, lam, P_override=None):
    per_day = []
    for dd in days:
        S, P = dd["S"], (dd["P"] if P_override is None else P_override)
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            # nugget from the per-fold spatial fraction of the reference
            # stations only -- no held-out-station leakage
            spf = fold_spatial_fraction(S, o)
            w = iso_weights(P[o], P[t], lam, (1.0 - spf) / spf)
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                pred = S[:, o] @ w
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(reduction(S[:, t], pred))
        per_day.append(float(np.nanmean(reds)))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_network_mean(days):
    per_day = []
    for dd in days:
        S = dd["S"]
        reds = [reduction(S[:, t], np.delete(S, t, axis=1).mean(axis=1)) for t in range(5)]
        per_day.append(float(np.nanmean(reds)))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def _uniform_kriging_weights(n, nugget_frac):
    """Analytic lambda->infinity kriging weights (geometry-free baseline).

    K_nn = ones + nugget*I, K_nt = ones -> w_i = 1/(n + nugget) for all i,
    i.e. a SCALED network mean with scaling n/(n+nugget).  Comparing iso(lam)
    against this isolates the geometry-specific contribution from the
    nugget-induced rescaling (the raw network mean uses scaling=1 and is not
    a fair comparator for test A).
    """
    return np.full(n, 1.0 / (n + nugget_frac))


def loo_uniform_kriging(days):
    """Geometry-free kriging (lambda=inf, same per-fold nugget as iso)."""
    per_day = []
    for dd in days:
        S = dd["S"]
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            spf = fold_spatial_fraction(S, o)
            w = _uniform_kriging_weights(len(o), (1.0 - spf) / spf)
            with np.errstate(all="ignore"):
                pred = S[:, o] @ w
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(reduction(S[:, t], pred))
        per_day.append(float(np.nanmean(reds)))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_mean_optimal_beta(days):
    """Network mean with per-fold optimal beta from REFERENCES only (no leakage).

    The strongest geometry-free common-mode predictor; brackets how much of
    iso4200's edge over the raw mean is just rescaling.
    """
    per_day = []
    for dd in days:
        S = dd["S"]
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            mean_ref = S[:, o].mean(axis=1)
            betas = []
            for r in o:
                r_o = [j for j in o if j != r]
                if not r_o:
                    continue
                m = S[:, r_o].mean(axis=1)
                denom = float(np.dot(m, m))
                if denom > 1e-12:
                    betas.append(float(np.dot(S[:, r], m) / denom))
            beta = float(np.mean(betas)) if betas else 1.0
            if not np.isfinite(beta):
                beta = 1.0
            reds.append(reduction(S[:, t], beta * mean_ref))
        per_day.append(float(np.nanmean(reds)))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


N_SHUFFLE = 200


def loo_time_shuffled(days, lam=4200.0, n_shuffle=N_SHUFFLE):
    """Temporal-shuffle null distribution.

    Each reference station's clock series is circularly shifted by an
    independent random lag per shuffle trial, destroying epoch-to-epoch
    spatial correlation while preserving each station's marginal statistics.
    Returns (mean, std, p2.5, p97.5) over the n_shuffle trials so the
    collapse can be assessed as a distribution, not a single random draw.
    """
    trials = []
    for _ in range(n_shuffle):
        per_day = []
        for dd in days:
            S, P, n = dd["S"], dd["P"], dd["n"]
            reds = []
            for t in range(5):
                o = [j for j in range(5) if j != t]
                spf = fold_spatial_fraction(S, o)
                w = iso_weights(P[o], P[t], lam, (1.0 - spf) / spf)
                # Independent random lag per reference station per trial
                pred = np.zeros(n)
                for ji, j in enumerate(o):
                    lag = int(RNG.integers(n // 20, n - n // 20))
                    pred += w[ji] * np.roll(S[:, j], lag)
                if not np.all(np.isfinite(pred)):
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                reds.append(reduction(S[:, t], pred))
            per_day.append(float(np.nanmean(reds)))
        trials.append(float(np.nanmean(per_day)))
    trials = np.array(trials)
    trials = trials[np.isfinite(trials)]
    if len(trials) == 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    return (float(np.mean(trials)), float(np.std(trials)),
            float(np.percentile(trials, 2.5)), float(np.percentile(trials, 97.5)))


def main():
    days = load_days()
    if len(days) < 2:
        print(f"Insufficient precise-clock days: {len(days)}")
        return
    print(f"Loaded {len(days)} precise-clock days: {[d['date'] for d in days]}")

    netmean = loo_network_mean(days)
    uniform_krig = loo_uniform_kriging(days)
    mean_optbeta = loo_mean_optimal_beta(days)
    lam_sweep = {lam: loo_iso(days, lam) for lam in [100, 300, 1000, 2000, 4200, 10000, 40000, 200000]}

    base_P = days[0]["P"]
    # exact enumeration of all 44 derangements (no fixed points, no sampling noise)
    from itertools import permutations
    ders = [p for p in permutations(range(5)) if all(p[i] != i for i in range(5))]
    scr = np.array([loo_iso(days, 4200.0, P_override=base_P[list(p)])[0] for p in ders])
    tshuf = loo_time_shuffled(days)

    # Plateau spread (>= 2000 km): does the canonical 4200 km stand out vs longer scales?
    plateau = np.array([lam_sweep[l][0] for l in (2000, 4200, 10000, 40000, 200000)])
    lam_spread = float(plateau.max() - plateau.min())
    scr_lo, scr_hi = float(np.percentile(scr, 2.5)), float(np.percentile(scr, 97.5))

    iso4200 = lam_sweep[4200][0]
    iso100 = lam_sweep[100][0]
    # Geometry-specific gain vs the leakage-free scaled-mean baseline (test A').
    # The raw network mean uses scaling=1; iso kriging self-scales via the
    # nugget, so iso4200-vs-raw-mean conflates geometry with rescaling.  The
    # fair geometry test is iso4200 vs uniform-kriging (same nugget, no geometry).
    geometry_gain_pp = float(iso4200 - uniform_krig[0])
    beats_mean = iso4200 > netmean[0] + 1.0
    beats_scaled_mean = iso4200 > uniform_krig[0] + 1.0
    lambda_matters = (iso4200 - iso100) > 3.0
    geometry_matters = not (scr_lo <= iso4200 <= scr_hi)
    time_coherent = tshuf[0] < iso4200 - 10.0

    print("\n=== PRECISE-CLOCK LOO-CV RMSE REDUCTION (%) ===")
    print(f"  Network mean (raw, scale=1)   : {netmean[0]:6.2f} +/- {netmean[1]:.2f}")
    print(f"  Uniform kriging (scaled mean) : {uniform_krig[0]:6.2f} +/- {uniform_krig[1]:.2f}")
    print(f"  Mean, optimal beta (refs only): {mean_optbeta[0]:6.2f} +/- {mean_optbeta[1]:.2f}")
    for lam, v in lam_sweep.items():
        print(f"  iso kriging lambda={lam:>7d} km : {v[0]:6.2f}")
    print(f"  Scrambled-geometry null   : {scr.mean():6.2f}  95% CI [{scr_lo:.2f}, {scr_hi:.2f}]")
    print(f"  Time-shuffled ({N_SHUFFLE} trials) : {tshuf[0]:6.2f} +/- {tshuf[1]:.2f}  "
          f"95% CI [{tshuf[2]:.2f}, {tshuf[3]:.2f}]")
    print(f"  (A)  iso4200 beats raw network mean : {beats_mean} ({iso4200:.1f} vs {netmean[0]:.1f})")
    print(f"  (A') iso4200 beats scaled mean      : {beats_scaled_mean} "
          f"(geometry gain {geometry_gain_pp:+.2f}pp vs {uniform_krig[0]:.1f})")
    print(f"  (B)  lambda dependence              : {lambda_matters} ({iso100:.1f} -> {iso4200:.1f})")
    print(f"  (C)  geometry-dependent             : {geometry_matters} (inconclusive expected at this aperture)")
    print(f"  (D)  time-coherent                  : {time_coherent} ({tshuf[0]:.1f})")

    summary = {
        "test": "precise_clock_control_battery",
        "clock_product": "precise_clock_bias_ns (IGS final satellite clock removed)",
        "n_days": len(days),
        "n_stations": len(STATIONS),
        "cadence_s": 300,
        "spatial_fraction": "per-fold mean cross-correlation of the 4 reference stations (LOO-honest)",
        "epoch_alignment": "common-finite-epoch intersection across stations",
        "years_used": sorted({d["date"][:4] for d in days}),
        "network_mean_pct": netmean[0],
        "network_mean_std": netmean[1],
        "uniform_kriging_scaled_mean_pct": uniform_krig[0],
        "mean_optimal_beta_refsonly_pct": mean_optbeta[0],
        "geometry_gain_vs_scaled_mean_pp": geometry_gain_pp,
        "iso_lambda_sweep_pct": {str(k): v[0] for k, v in lam_sweep.items()},
        "iso_100km_pct": iso100,
        "iso_4200km_pct": iso4200,
        "lambda_sweep_plateau_spread_pp": lam_spread,
        "scrambled_geometry_null_pct_mean": float(scr.mean()),
        "scrambled_geometry_null_95ci": [scr_lo, scr_hi],
        "time_shuffled_pct": tshuf[0],
        "time_shuffled_std": tshuf[1],
        "time_shuffled_percentile_range": [tshuf[2], tshuf[3]],
        "time_shuffled_n_trials": N_SHUFFLE,
        "test_A_beats_network_mean": bool(beats_mean),
        "test_Aprime_beats_scaled_mean": bool(beats_scaled_mean),
        "test_B_lambda_dependent": bool(lambda_matters),
        "test_C_geometry_dependent": bool(geometry_matters),
        "test_D_time_coherent": bool(time_coherent),
        "days_used": [d["date"] for d in days],
        "timestamp": datetime.now().isoformat(),
    }
    out = PROJECT_ROOT / "results" / "outputs" / "step_03c_precise_controls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
