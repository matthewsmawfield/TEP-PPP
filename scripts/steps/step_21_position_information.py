#!/usr/bin/env python3
"""
STEP 21  (Phase 2, analytical) -- Position-domain benefit of a network clock prior.

*** SUPERSEDED BY STEP 26 -- read that one for the headline numbers. ***
Two defects, both fixed in step_26:
  (1) BASELINE.  This step compares "prior of the kernel's quality" against
      "no clock prior at all".  That conflates the gain from constraining the
      clock AT ALL (which any prior gives, including a useless one) with the
      gain the NETWORK supplies.  On precise products the kernel prediction
      error (0.87 m) is no better than the station's own variability (0.85 m),
      so the 26-27% reported here is almost entirely baseline artefact.
  (2) HYPERPARAMETERS.  lambda/nugget/K were hand-picked, not fitted; step_25
      selects them by nested CV on held-out stations.
Retained for provenance and because the geometry machinery is reused.

The Step 20 EKF did not converge well enough to measure a cm-level effect.  This
step answers the same question analytically and without filter-tuning:

  Given the REAL satellite geometry of a station over a day, and a clock prior
  of a MEASURED quality sigma_p, by how much does the prior shrink the
  position covariance?

Per epoch: normal matrix  N = A^T W A  with A = [-los, 1, map]  (rows per sat),
W elevation-dependent.  Add a clock pseudo-observation -> N' = N + e_clk e_clk^T / sigma_p^2.
Position 3D sigma = sqrt(trace([N'^-1]_{pos})).  Compare:
    none      -- no prior
    mean      -- sigma_p = network-mean clock-prediction error   (measured, LOO)
    kriging   -- sigma_p = distance-kernel clock-prediction error (measured, LOO)

Aggregated over real SP3 geometry, many stations, several days.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402
from scripts.lib.tep_geometry import load_sp3, sat_grid, elev_los, ELEV_MASK  # noqa: E402

C = 299_792_458.0
LAMBDA_KM = 4200.0          # canonical TEP correlation length (core/constants.py)
SPF = 0.44                   # canonical spatial fraction (Papers 1-2).  NOTE: the
                             # control battery (03b/03c) estimates the spatial
                             # fraction per fold (LOO-honest); this step keeps the
                             # fixed canonical kernel so the kriging arm is a
                             # pre-set a-priori model, not fitted here.
NUGGET = (1.0 - SPF) / SPF  # = 1.273
K_NEIGHBOURS = 15
OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260902)


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


def clock_prediction_skill(product, date_tag):
    """LOO band-limited clock-prediction error std (m) per station:
    network mean vs nearest-K distance kernel."""
    s = td.load_series(product, date_tag)
    # Derive dt per station from data (broadcast=30s, precise=300s).
    def _bp(n):
        dt = float(np.nanmedian(np.diff(s[n]["t"])))
        if not np.isfinite(dt) or dt <= 0:
            dt = 30.0
        return ts.bandpass(s[n]["resid"], dt)
    names = [n for n in s
             if np.std(_bp(n)) * C * 1e-9 < 5.0]     # QC
    if len(names) < 8:
        return {}
    grid = s[names[0]]["t"]
    X = np.column_stack([np.interp(grid, s[n]["t"], _bp(n)) for n in names]) * (C * 1e-9)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.array([s[n]["station"].ecef_km for n in names])
    all_sts = _ecef_to_stations(P)
    out = {}
    for t, nm in enumerate(names):
        # Great-circle distances from target to all stations (consistent
        # with tep_kernel.py).  The previous implementation used Euclidean
        # chord distance in ECEF, which underestimates great-circle distance
        # and introduces a methodological inconsistency with the rest of
        # the pipeline.
        d = _gc_dist_matrix(all_sts, _ecef_to_stations(P[t].reshape(1, -1)))[:, 0]
        nn = [j for j in np.argsort(d) if j != t][:K_NEIGHBOURS]
        sts_nn = [all_sts[j] for j in nn]
        dnn = _gc_dist_matrix(sts_nn, sts_nn)
        Knn = np.exp(-dnn / LAMBDA_KM) + (NUGGET + 1e-6) * np.eye(len(nn))
        knt = np.exp(-d[nn] / LAMBDA_KM)
        with np.errstate(all="ignore"):
            w = np.clip(np.linalg.solve(Knn, knt), -3.0, 3.0)
            # Use RMSE (not std) to include any predictor bias in the
            # error estimate — std removes the mean and can understate
            # the true prediction error for biased predictors.
            e_mean = float(np.sqrt(np.mean((X[:, t] - X[:, nn].mean(1)) ** 2)))
            e_krig = float(np.sqrt(np.mean((X[:, t] - X[:, nn] @ w) ** 2)))
        if not (np.isfinite(e_krig) and np.isfinite(e_mean)) or e_krig > 20 or e_mean > 20:
            continue
        out[nm] = {"station": s[nm]["station"], "err_mean_m": float(e_mean),
                   "err_krig_m": float(e_krig), "sig_m": float(np.sqrt(np.mean(X[:, t] ** 2)))}
    return out


def position_sigma(A, W, sig_p):
    """3D position 1-sigma (m) from normal matrix with optional clock prior."""
    N = A.T @ (W[:, None] * A)
    if sig_p is not None and np.isfinite(sig_p) and sig_p > 0:
        e = np.zeros(A.shape[1]); e[3] = 1.0
        N = N + np.outer(e, e) / sig_p ** 2
    try:
        Cov = np.linalg.inv(N + 1e-9 * np.eye(A.shape[1]))
    except np.linalg.LinAlgError:
        return np.nan
    return float(np.sqrt(np.trace(Cov[:3, :3])))


def run(product, date_tags, label):
    rows = []
    for dtag in date_tags:
        sp3 = load_sp3(dtag)
        if sp3 is None:
            continue
        grid = np.arange(0, 86400.0, 900.0)          # every 15 min
        sp3g = sat_grid(sp3, grid)
        skill = clock_prediction_skill(product, dtag)
        for nm, sk in skill.items():
            sta = sk["station"].ecef_m
            sn, sm, sk_ = [], [], []
            for gi in range(len(grid)):
                rows_A, w = [], []
                for s, pg in sp3g.items():
                    p = pg[gi]
                    if not np.isfinite(p[0]):
                        continue
                    e, u, _ = elev_los(p, sta)
                    if e < ELEV_MASK:
                        continue
                    mapf = 1.0 / (np.sin(e) + 0.00143 / (np.tan(e) + 0.0445))
                    rows_A.append([-u[0], -u[1], -u[2], 1.0, mapf])
                    w.append(np.clip(np.sin(e), 0.15, 1.0) ** 2 / 0.15 ** 2)   # rel. weight
                if len(rows_A) < 5:
                    continue
                A = np.array(rows_A); W = np.array(w)
                s0 = position_sigma(A, W, None)
                sn.append(s0)
                sm.append(position_sigma(A, W, sk["err_mean_m"]))
                sk_.append(position_sigma(A, W, sk["err_krig_m"]))
            if len(sn) < 10:
                continue
            sn, sm, sk_ = map(lambda a: np.array(a), (sn, sm, sk_))
            rows.append({
                "day": dtag, "station": nm,
                "sig_none_cm": float(np.median(sn) * 100),
                "sig_mean_cm": float(np.median(sm) * 100),
                "sig_krig_cm": float(np.median(sk_) * 100),
                "imp_mean_pct": float(100 * (1 - np.median(sm) / np.median(sn))),
                "imp_krig_pct": float(100 * (1 - np.median(sk_) / np.median(sn))),
                "krig_vs_mean_pct": float(100 * (1 - np.median(sk_) / np.median(sm))),
                "clk_pred_err_mean_m": skill[nm]["err_mean_m"],
                "clk_pred_err_krig_m": skill[nm]["err_krig_m"],
            })
    if not rows:
        return {"label": label, "n": 0}
    imk = np.array([r["imp_krig_pct"] for r in rows])
    imm = np.array([r["imp_mean_pct"] for r in rows])
    kvm = np.array([r["krig_vs_mean_pct"] for r in rows])
    bs = [np.median(RNG.choice(kvm, len(kvm), replace=True)) for _ in range(3000)]
    print(f"[{label}] {len(rows)} station-days")
    print(f"  position 3D-sigma reduction vs no prior:  network-mean {np.median(imm):+.1f}%   "
          f"kriging {np.median(imk):+.1f}%")
    print(f"  kriging vs network-mean:  {np.median(kvm):+.1f}%  "
          f"(95% CI [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}], "
          f"kriging better in {np.mean(kvm>0)*100:.0f}% of station-days)")
    return {
        "label": label, "n_station_days": len(rows),
        "median_reduction_pct": {"network_mean": float(np.median(imm)), "kriging": float(np.median(imk))},
        "kriging_vs_mean_median_pct": float(np.median(kvm)),
        "kriging_vs_mean_ci95_pct": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "kriging_better_frac": float(np.mean(kvm > 0)),
        "significant": bool(np.percentile(bs, 2.5) > 0),
        "rows": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "test": "phase2_position_information_analysis",
        "method": "analytical position covariance with a measured-quality clock prior",
        "global_precise": run("clk_cod", td.available_dates("clk_cod", 2024), "global_precise"),
        "global_broadcast": run("broadcast", [f"2024{d:03d}" for d in range(100, 107)], "global_broadcast"),
        "timestamp": datetime.now().isoformat(),
    }
    gp = res["global_precise"]
    # a network clock prior of measured quality shrinks position; the distance
    # kernel adds a small but consistently-significant increment over a bulk mean
    prior_helps = gp.get("median_reduction_pct", {}).get("kriging", 0) > 10
    kernel_increment_sig = all(r.get("significant") for r in
                               (res["global_precise"], res["global_broadcast"]) if r.get("n_station_days"))
    res["network_clock_prior_shrinks_position"] = bool(prior_helps)
    res["distance_kernel_beats_bulk_mean_in_position"] = bool(kernel_increment_sig)
    (OUT / "step_21_position_information.json").write_text(json.dumps(res, indent=2))
    print(f"\n  network clock prior shrinks position sigma (>10%): {res['network_clock_prior_shrinks_position']}"
          f"  ({gp.get('median_reduction_pct', {}).get('kriging', 0):.0f}% on precise)")
    print(f"  distance kernel beats bulk mean in position (significant, both arms): "
          f"{res['distance_kernel_beats_bulk_mean_in_position']}")
    print(f"saved {OUT / 'step_21_position_information.json'}")


if __name__ == "__main__":
    main()
