#!/usr/bin/env python3
"""
STEP 12  (Phase 1 / T3) -- Operational resolvability vs network aperture.

Question: does a distance-structured predictor beat a bulk common-mode
(network-mean) predictor out-of-sample, and does the margin GROW as the
network aperture increases?  A flat margin means the structure the series
measured is not operationally resolvable here; a growing margin is the
fingerprint of resolvable differential structure.

Method: LOO-CV clock-residual prediction on nested sub-networks of increasing
radius.  Two predictors on identical folds:
    mean     -- uniform average of the other stations   (bulk common mode)
    kriging  -- exp(-r/lambda) GP weights                (distance structure)
Metric per station: out-of-sample RMSE reduction vs the raw residual.
Report mean(kriging) - mean(mean) as the margin, block-bootstrapped, per aperture.
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402

RNG = np.random.default_rng(20260901)
OUT = PROJECT_ROOT / "results" / "outputs"
# Lambda is now selected by leave-one-out CV within each aperture, not fixed.
# The canonical 4200 km is the centre of the search grid, not imposed.
LAMBDA_GRID = (300.0, 700.0, 1500.0, 2500.0, 4200.0, 7000.0, 12000.0, 25000.0)
NUGGET = 0.35
APERTURES = [1500, 3000, 6000, 12000, 20000]


def band_matrix(series, names):
    """Column-stack the band-passed residuals for `names`.

    The band-pass of a station is memoised on its own series dict ("_bp"):
    select_lambda() and loo_scores() are each called for two station halves
    per day per aperture, so without the cache every station's residual is
    band-passed ~4x per aperture (and the aperture loop repeats it 5x).
    """
    cols = []
    for n in names:
        rec = series[n]
        bp = rec.get("_bp")
        if bp is None:
            x = rec["resid"]
            dt = float(np.nanmedian(np.diff(rec["t"])))
            if not np.isfinite(dt) or dt <= 0:
                dt = 30.0
            bp = ts.bandpass(x, dt)
            rec["_bp"] = bp
        cols.append(bp)
    return np.column_stack(cols)


def reduction(obs, pred):
    r = 100.0 * (1.0 - np.sqrt(np.mean((obs - pred) ** 2)) / np.sqrt(np.mean(obs ** 2)))
    return float(np.clip(r, -50.0, 100.0))


def loo_scores(series, names, D, lam_km=LAMBDA_GRID[4]):
    """Return (median_mean, median_kriging, median_uniform_kriging) reductions.

    Each value is the median over stations of the per-station out-of-sample
    RMSE reduction.  Three predictors on identical folds:
        mean          -- uniform average (scale=1, no nugget)
        kriging       -- exp(-r/lambda) GP weights with nugget (distance + scaling)
        uniform_krig  -- analytic lambda->inf weights (same nugget, NO distance)

    The geometry-specific margin is kriging - uniform_krig.  The raw margin
    (kriging - mean) conflates distance structure with nugget-induced
    rescaling; the geometry-specific margin isolates the distance contribution.
    Columns are standardised to unit variance so the linear predictors are not
    dominated by a few high-noise stations; the fractional-RMSE metric is
    scale-free per target either way.

    lam_km is selected by CV in select_lambda(); the caller passes the
    chosen value here.
    """
    X = band_matrix(series, names)
    # NOTE: standardisation is done per-fold inside the loop (excluding the
    # target) so the left-out station's own mean/std do not influence the
    # reference-column scaling — a strict leave-one-out.
    red_mean, red_krig, red_ukrig = [], [], []
    for t in range(len(names)):
        o = [j for j in range(len(names)) if j != t]
        n = len(o)
        # Standardise reference columns using reference columns only
        Xo = X[:, o] - X[:, o].mean(0)
        sdo = Xo.std(0); sdo[sdo < 1e-9] = 1.0
        Xo = Xo / sdo
        # Target is centred/scaled by its own statistics
        obs = X[:, t] - X[:, t].mean()
        obs = obs / (X[:, t].std() + 1e-12)
        # raw network mean (scale=1)
        red_mean.append(reduction(obs, Xo.mean(1)))
        # kriging with distance structure + nugget
        d_nn = D[np.ix_(o, o)]
        d_nt = D[o, t]
        K = np.exp(-d_nn / lam_km) + (NUGGET + 1e-6) * np.eye(n)
        k = np.exp(-d_nt / lam_km)
        try:
            w = np.linalg.solve(K, k)
            if not np.all(np.isfinite(w)) or np.abs(w).sum() > 50:
                w = np.linalg.lstsq(K, k, rcond=1e-6)[0]
        except np.linalg.LinAlgError:
            w = np.full(n, 1.0 / n)
        # Return raw simple-kriging weights — no tanh limiter, which
        # breaks the normal-equation solution (see tep_kernel.py).
        with np.errstate(all="ignore"):
            pred_krig = Xo @ w
        if not np.all(np.isfinite(pred_krig)):
            pred_krig = np.nan_to_num(pred_krig, nan=0.0, posinf=0.0, neginf=0.0)
        red_krig.append(reduction(obs, pred_krig))
        # uniform kriging: lambda->inf, same nugget, NO distance weighting
        # K_nn = ones + NUGGET*I, K_nt = ones -> w_i = 1/(n + NUGGET)
        wu = np.full(n, 1.0 / (n + NUGGET))
        with np.errstate(all="ignore"):
            pred_ukrig = Xo @ wu
        if not np.all(np.isfinite(pred_ukrig)):
            pred_ukrig = np.nan_to_num(pred_ukrig, nan=0.0, posinf=0.0, neginf=0.0)
        red_ukrig.append(reduction(obs, pred_ukrig))
    return (float(np.median(red_mean)), float(np.median(red_krig)),
            float(np.median(red_ukrig)))


def select_lambda(series, names, D):
    """Select lambda by leave-one-out CV on the kriging reduction.

    For each candidate lambda, compute the LOO kriging reduction (median over
    stations).  Return the lambda that maximises it.  This is done per day
    per aperture, so the kernel is never misspecified by a fixed lengthscale.
    """
    X = band_matrix(series, names)
    # Per-target standardisation depends only on t, not lambda -- precompute
    # it once instead of inside the 8-point lambda grid loop.
    nnames = len(names)
    pre = []
    for t in range(nnames):
        o = [j for j in range(nnames) if j != t]
        Xo = X[:, o] - X[:, o].mean(0)
        sdo = Xo.std(0); sdo[sdo < 1e-9] = 1.0
        Xo = Xo / sdo
        obs = (X[:, t] - X[:, t].mean()) / (X[:, t].std() + 1e-12)
        pre.append((o, Xo, obs, D[np.ix_(o, o)], D[o, t]))
    best_lam, best_score = None, -np.inf
    for lam in LAMBDA_GRID:
        reds = []
        for t in range(nnames):
            o, Xo, obs, d_nn, d_nt = pre[t]
            n = len(o)
            K = np.exp(-d_nn / lam) + (NUGGET + 1e-6) * np.eye(n)
            k = np.exp(-d_nt / lam)
            try:
                w = np.linalg.solve(K, k)
                if not np.all(np.isfinite(w)) or np.abs(w).sum() > 50:
                    w = np.linalg.lstsq(K, k, rcond=1e-6)[0]
            except np.linalg.LinAlgError:
                w = np.full(n, 1.0 / n)
            # Return raw simple-kriging weights — no tanh limiter.
            with np.errstate(all="ignore"):
                pred = Xo @ w
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(reduction(obs, pred))
        score = float(np.median(reds))
        if score > best_score:
            best_score, best_lam = score, lam
    if best_lam is None:
        best_lam = LAMBDA_GRID[4]  # fall back to grid centre only if all fail
    return best_lam, best_score


def dist_matrix(stations):
    n = len(stations)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = td.great_circle_km(stations[i], stations[j])
    return D


def run(product, date_tags, label):
    days = [s for _, s in td.load_multi_day(product, date_tags) if len(s) >= 5]
    all_names = sorted({n for s in days for n in s})
    # centroid of the densest region (Europe) as the aperture origin
    lat0 = np.median([days[0][n]["station"].lat_deg for n in days[0] if 35 < days[0][n]["station"].lat_deg < 60] or [46])
    lon0 = np.median([days[0][n]["station"].lon_deg for n in days[0] if -10 < days[0][n]["station"].lon_deg < 25] or [8])
    origin = td.Station("origin", np.array([0, 0, 0.0]), float(lat0), float(lon0))

    rows = []
    for R in APERTURES:
        # stations within R km of the origin
        sel = [n for n in all_names
               if any(n in s and td.great_circle_km(s[n]["station"], origin) <= R for s in days)]
        if len(sel) < 5:
            continue
        per_day_mean, per_day_krig, per_day_ukrig = [], [], []
        per_day_lambda = []
        per_day_margins = []   # list of (margin, geom_margin) per day
        for s in days:
            nm = [n for n in sel if n in s]
            if len(nm) < 8:
                continue
            sts = [s[n]["station"] for n in nm]
            D = dist_matrix(sts)
            # Nested design to avoid circularity: split stations into two
            # RANDOM halves, select lambda on one half (LOO within that half),
            # evaluate loo_scores on the other half, then swap.  This ensures
            # the reported kriging reduction is not the same objective that
            # was optimised to choose lambda.
            idx = RNG.permutation(len(nm))
            a, b = sorted(idx[: len(nm) // 2].tolist()), sorted(idx[len(nm) // 2:].tolist())
            D_a = D[np.ix_(a, a)]; D_b = D[np.ix_(b, b)]
            nm_a = [nm[i] for i in a]; nm_b = [nm[i] for i in b]
            # Select lambda on half A, evaluate on half B
            lam_a, _ = select_lambda(s, nm_a, D_a)
            rm_b, rk_b, ru_b = loo_scores(s, nm_b, D_b, lam_km=lam_a)
            # Select lambda on half B, evaluate on half A
            lam_b, _ = select_lambda(s, nm_b, D_b)
            rm_a, rk_a, ru_a = loo_scores(s, nm_a, D_a, lam_km=lam_b)
            per_day_lambda.extend([lam_a, lam_b])
            per_day_mean.extend([rm_b, rm_a])
            per_day_krig.extend([rk_b, rk_a])
            per_day_ukrig.extend([ru_b, ru_a])
            # Store per-day margins for day-block bootstrap
            day_margins = [(rk_b - rm_b, rk_b - ru_b), (rk_a - rm_a, rk_a - ru_a)]
            per_day_margins.append(day_margins)
        if not per_day_mean:
            continue

        # Day-block bootstrap: resample whole days (not individual half-day
        # margins, which are correlated within a day).  Each drawn day
        # contributes both its half-day margins.
        margins = np.array(per_day_krig) - np.array(per_day_mean)
        geom_margins = np.array(per_day_krig) - np.array(per_day_ukrig)
        n_days = len(per_day_margins)
        bs, bs_g = [], []
        for _ in range(2000):
            drawn = RNG.integers(0, n_days, n_days)
            dm = [m for d in drawn for m in per_day_margins[d]]
            dm = np.array(dm)
            bs.append(float(np.mean(dm[:, 0])))
            bs_g.append(float(np.mean(dm[:, 1])))
        # Compute max baseline from the UNION of all selected stations across
        # all days, not just days[0] which may lack some stations.
        all_sel_stations = []
        seen_names = set()
        for s in days:
            for n in sel:
                if n in s and n not in seen_names:
                    all_sel_stations.append(s[n]["station"])
                    seen_names.add(n)
        maxbl = max((td.great_circle_km(a, b) for a, b in itertools.combinations(
            all_sel_stations, 2)), default=0.0)
        rows.append({
            "aperture_km": R, "n_stations": len(sel), "max_baseline_km": round(maxbl),
            "lambda_cv_median_km": float(np.median(per_day_lambda)),
            "lambda_cv_values": [float(x) for x in per_day_lambda],
            "reduction_mean_pct": float(np.mean(per_day_mean)),
            "reduction_kriging_pct": float(np.mean(per_day_krig)),
            "reduction_uniform_kriging_pct": float(np.mean(per_day_ukrig)),
            "margin_pct": float(np.mean(margins)),
            "margin_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "margin_sigma": float(np.mean(margins) / (np.std(margins, ddof=1) / np.sqrt(len(margins)) + 1e-9)),
            "geometry_specific_margin_pct": float(np.mean(geom_margins)),
            "geometry_specific_margin_ci95": [float(np.percentile(bs_g, 2.5)),
                                               float(np.percentile(bs_g, 97.5))],
            "geometry_specific_margin_sigma": float(np.mean(geom_margins) /
                (np.std(geom_margins, ddof=1) / np.sqrt(len(geom_margins)) + 1e-9)),
        })
        print(f"  R<={R:5d} km  n={len(sel):2d}  bl<={maxbl:5.0f}  "
              f"mean {rows[-1]['reduction_mean_pct']:5.1f}%  ukrig {rows[-1]['reduction_uniform_kriging_pct']:5.1f}%  "
              f"krig {rows[-1]['reduction_kriging_pct']:5.1f}%  "
              f"margin {rows[-1]['margin_pct']:+5.2f}%  geom {rows[-1]['geometry_specific_margin_pct']:+5.2f}%"
              f"  ({rows[-1]['geometry_specific_margin_sigma']:+.1f} sigma)")

    # monotonic growth test: Spearman(aperture, margin) > 0 and full-net margin > 3 sigma
    ap = np.array([r["aperture_km"] for r in rows], float)
    mg = np.array([r["margin_pct"] for r in rows], float)
    mg_geom = np.array([r["geometry_specific_margin_pct"] for r in rows], float)
    if len(rows) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(ap, mg)
        rho_geom, p_geom = spearmanr(ap, mg_geom)
    else:
        rho, p = np.nan, np.nan
        rho_geom, p_geom = np.nan, np.nan
    grows = bool(np.isfinite(rho) and rho > 0.5 and rows and rows[-1]["margin_sigma"] > 3.0)
    # Geometry-specific pass: the margin over the scaled mean (not just the raw
    # mean) must also grow with aperture and be significant at full network.
    geom_grows = bool(np.isfinite(rho_geom) and rho_geom > 0.5 and rows and
                      rows[-1]["geometry_specific_margin_sigma"] > 3.0)
    print(f"  [{label}] raw margin: Spearman rho = {rho:+.2f} (p={p:.3f})   grows: {grows}")
    print(f"  [{label}] geom margin: Spearman rho = {rho_geom:+.2f} (p={p_geom:.3f})   grows: {geom_grows}")
    return {"label": label, "rows": rows,
            "spearman_rho": float(rho) if np.isfinite(rho) else None,
            "spearman_p": float(p) if np.isfinite(p) else None,
            "geometry_spearman_rho": float(rho_geom) if np.isfinite(rho_geom) else None,
            "geometry_spearman_p": float(p_geom) if np.isfinite(p_geom) else None,
            "T3_pass": grows, "T3_geometry_pass": geom_grows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[T3] operational resolvability vs aperture")
    res = {
        "test": "T3_resolvability_vs_aperture",
        "lambda_grid_km": LAMBDA_GRID, "nugget": NUGGET, "apertures_km": APERTURES,
        "global_broadcast": run("broadcast", [f"2024{d:03d}" for d in range(100, 107)], "global_broadcast"),
        "timestamp": datetime.now().isoformat(),
    }
    # Full-year global-precise T3 aperture sweep is run by step_14_precise_global.
    res["T3_pass"] = res["global_broadcast"]["T3_pass"]
    (OUT / "step_12_resolvability.json").write_text(json.dumps(res, indent=2))
    print(f"\nT3 pass: {res['T3_pass']}\nsaved {OUT / 'step_12_resolvability.json'}")


if __name__ == "__main__":
    main()
