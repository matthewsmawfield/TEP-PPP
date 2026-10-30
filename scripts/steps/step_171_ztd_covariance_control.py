#!/usr/bin/env python3
"""Step 171: Real tropospheric (ZWD) covariance control.

Implements the residual-troposphere control requested by corpus issue
1-3: rather than citing a literature correlation range, measure the
tropospheric pair-correlation function directly from estimated zenith
wet delay at real GNSS stations.

For each station-day in the step-67 short-baseline campaign (35
European stations, DOY 100-106 2024), the ionosphere-free pseudorange
prefit residuals are solved jointly for [clock, zenith wet delay] per
epoch — the same per-epoch weighted solve as step_67, but retaining the
wet-delay coefficient that step_67 discarded.  The per-station ZWD
fluctuation series are then correlated pairwise against baseline
distance, giving an empirical tropospheric covariance kernel on exactly
the data class (GNSS pseudorange prefits) where a tropospheric confound
would act.

The tropospheric correlation length so measured is compared with the
corpus's observed clock-coherence scales (MGEX ~1.9e3 km; Papers 1-2
~3-5e3 km).  The claim being tested: if tropospheric ZWD covariance
supplies the observed clock coherence, their distance-ordered
correlation functions must overlap in both scale and shape.

Outputs:
    results/outputs/step_171_ztd_covariance_control.json
    data/processed/step_171_zwd_series.npz
"""

from __future__ import annotations

import sys
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "steps" / "archive"))

from scripts.lib import tep_dynamic as dyn          # noqa: E402
from scripts.lib import tep_spp as spp              # noqa: E402
import step_67_short_baseline_field as s67          # noqa: E402

YEAR = 2024
DOYS = [100, 101, 102, 103, 104, 105, 106]
DECIMATE_S = 300          # 5-min epochs, matching step_67 cadence
GRID_STEP_S = 300         # common interpolation grid within each day
MIN_EPOCHS = 150          # min epochs per station-day to use the series

CACHE = ROOT / "data" / "step67_cache"
WORK = ROOT / "data" / "tmp" / "ztd171"


def zwd_series(site: str, doy: int, dense):
    """Per-epoch [clock, ZWD] solve; returns (t, zwd, xyz) or None."""
    hits = glob.glob(str(
        ROOT / f"data/rinex/obs/{YEAR}/{doy:03d}/{site}*.crx.gz"))
    if not hits:
        return None
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        rnx = dyn._decompress_crx(Path(hits[0]), WORK / site)
        if rnx is None:
            return None
        xyz = dyn._rinex_header_xyz(rnx)
        if xyz is None:
            return None
        obs = s67.read_rinex3_gps_robust(dyn.read_rinex_text(rnx))
    except Exception:
        return None
    ts, zw = [], []
    for sod in sorted(obs):
        if int(round(sod)) % int(DECIMATE_S):
            continue
        pf = spp.prefit_fast(obs[sod], dense, xyz, sod)
        if len(pf) < 6:
            continue
        el = np.array([r[2] for r in pf], float)
        res = np.array([r[3] for r in pf], float)
        wm = np.array([r[4] for r in pf], float)
        w = np.sin(np.radians(el)) ** 2
        if w.sum() <= 0:
            continue
        A = np.column_stack([np.ones(len(res)), wm])
        try:
            sol = np.linalg.solve(A.T @ np.diag(w) @ A, A.T @ np.diag(w) @ res)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(sol).all():
            ts.append(float(sod))
            zw.append(float(sol[1]))
    if len(ts) < MIN_EPOCHS:
        return None
    return np.array(ts), np.array(zw), xyz


def detrend(t, y):
    """Remove day-mean and linear trend from a ZWD series."""
    q = (t - t.mean()) / (t.std() or 1.0)
    return y - np.polyval(np.polyfit(q, y, 2), q)


def main():
    print("[step_171] Real ZWD covariance control (issue 1-3)")

    # ---- extract ZWD series for all station-days -------------------------
    series = {}     # site -> dict(doy -> (t_grid, zwd_detrended))
    xyzs = {}
    for doy in DOYS:
        tag = f"{YEAR}{doy:03d}"
        gz = CACHE / f"grid_{tag}.npz"
        if not gz.exists():
            print(f"[{tag}] no cached satellite grid, skipped")
            continue
        z = np.load(gz, allow_pickle=True)
        dense = (float(z["t0"]), float(z["step"]), z["G"].item())
        for site in s67.STATIONS:
            r = zwd_series(site, doy, dense)
            if r is None:
                continue
            t, zw, xyz = r
            xyzs[site] = xyz
            # resample onto a common 300-s grid for cross-station alignment
            grid = np.arange(0.0, 86400.0, GRID_STEP_S)
            zg = np.interp(grid, t, zw, left=np.nan, right=np.nan)
            m = np.isfinite(zg)
            if m.sum() < MIN_EPOCHS:
                continue
            zg_d = np.full_like(zg, np.nan)
            zg_d[m] = detrend(grid[m], zg[m])
            series.setdefault(site, {})[doy] = zg_d
        print(f"[{tag}] stations solved: "
              f"{sum(1 for s in series if doy in series[s])}")

    # ---- pairwise ZWD correlation vs distance ----------------------------
    sites = sorted(s for s, d in series.items()
                   if sum(1 for k in d if isinstance(k, int)) >= 3)
    pairs = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            a, b = sites[i], sites[j]
            dist = float(np.linalg.norm(xyzs[a] - xyzs[b])) / 1e3  # km
            cors = []
            for doy in DOYS:
                if doy not in series[a] or doy not in series[b]:
                    continue
                ya, yb = series[a][doy], series[b][doy]
                m = np.isfinite(ya) & np.isfinite(yb)
                if m.sum() < MIN_EPOCHS:
                    continue
                if ya[m].std() < 1e-4 or yb[m].std() < 1e-4:
                    continue
                cors.append(float(np.corrcoef(ya[m], yb[m])[0, 1]))
            if len(cors) >= 3:
                pairs.append((dist, float(np.mean(cors)), len(cors)))

    pairs = np.array(pairs) if pairs else np.zeros((0, 3))
    print(f"[step_171] station pairs with >=3 common days: {len(pairs)}")

    # ---- binned correlation profile + exponential fit --------------------
    bins = np.array([0, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000])
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        if m.sum() >= 5:
            binned.append({
                "bin_km": [float(lo), float(hi)],
                "n_pairs": int(m.sum()),
                "mean_zwd_corr": float(np.mean(pairs[m, 1])),
                "sem": float(pairs[m, 1].std(ddof=1) / np.sqrt(m.sum())),
            })
    for b in binned:
        print(f"  {b['bin_km'][0]:5.0f}-{b['bin_km'][1]:5.0f} km "
              f"({b['n_pairs']:3d} pairs): r = {b['mean_zwd_corr']:+.3f} "
              f"+/- {b['sem']:.3f}")

    # exponential fit to the binned profile
    lam_fit = None
    if len(binned) >= 3:
        xc = np.array([(b["bin_km"][0] + b["bin_km"][1]) / 2 for b in binned])
        yc = np.array([b["mean_zwd_corr"] for b in binned])
        wc = np.array([1.0 / max(b["sem"], 1e-3) ** 2 for b in binned])
        pos = yc > 0
        if pos.sum() >= 3:
            X = np.column_stack([np.ones(pos.sum()), -xc[pos]])
            W = np.diag(wc[pos])
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ np.log(yc[pos]))
            lam_fit = 1.0 / beta[1]
            print(f"[step_171] ZWD correlation length (exp fit, r>0 bins): "
                  f"{lam_fit:.0f} km")

    out = {
        "step": "171_ztd_covariance_control",
        "issue": "1-3",
        "description": ("Measured tropospheric pair-correlation function "
                        "from estimated zenith wet delay at real GNSS "
                        "stations (DOY 100-106 2024, 5-min epochs, "
                        "ionosphere-free pseudorange prefits)"),
        "data": {
            "stations": sites,
            "n_stations": len(sites),
            "doys": DOYS,
            "n_pairs_used": int(len(pairs)),
            "method": ("per-epoch joint [clock, ZWD] weighted solve on "
                       "ionosphere-free prefit residuals; per-day "
                       "detrend; pairwise correlation averaged over "
                       "days"),
        },
        "correlation_profile": binned,
        "zwd_corr_length_km": (float(lam_fit) if lam_fit else None),
        "comparison": {
            "observed_clock_lambda_km": {"MGEX": 1862, "GNSS_PPP_range": [3000, 5000]},
            "note": ("If tropospheric delay were the coherence source, "
                     "the ZWD correlation function would need to match "
                     "the observed clock-coherence scale and shape."),
        },
        "pairs_raw": [
            {"dist_km": round(float(p[0]), 1),
             "zwd_corr": round(float(p[1]), 4),
             "n_days": int(p[2])} for p in pairs
        ],
    }

    outdir = ROOT / "results" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "step_171_ztd_covariance_control.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step_171] wrote {outdir / 'step_171_ztd_covariance_control.json'}")


if __name__ == "__main__":
    main()
