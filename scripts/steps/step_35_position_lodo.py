#!/usr/bin/env python3
"""
STEP 35  -- Position-domain benefit, leave-one-day-out, full year 2024.

Step 34 established the clock-domain skill over the full year: the distance kernel
predicts an unseen day's receiver clocks +47% better than no network and +10%
better than a bulk mean.  Step 26 propagated the clock skill to the position
domain analytically (pos_sigma from the geometry matrix), but only on 7 days.

This step closes that gap.  For each of the days in Step 34:
  * the per-day clock-prediction errors (raw, kernel, net_mean) are loaded;
  * representative satellite geometry is computed from the available SP3
    files (DOY 1-52, 2024; GPS geometry repeats with a ~10-day constellation
    cycle, so the median geometry over 52 days is representative of the full
    year);
  * the position-domain gain is computed analytically via the same
    pos_sigma model as Step 26: pos_sigma(los, sig_clk) = sqrt(trace(inv(
    A^T A + diag(0,0,0,1/sig_clk^2))[:3,:3])).

The result is a distribution of position-domain gains over the full year -- the
year-long operational evidence that the clock skill translates to a position
benefit under obstruction.

The clock sigmas used are the per-day MEDIANS of the held-out prediction
errors from Step 34 (raw_m, kernel_m, net_mean_m).  These are the honest
out-of-sample errors: hyperparameters were selected on a different training
day, and every station on the held-out day was predicted from the others.
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
from scripts.lib import tep_kernel as tk     # noqa: E402
from scripts.lib.tep_geometry import (  # noqa: E402
    load_sp3, sat_grid, elev_los, ELEV_MASK,
)

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260935)

K_VALUES = [3, 4, 5, 6, 8, 99]
GRID_STEP_S = 1800.0  # sample geometry every 30 min


def pos_sigma(los, sig_clk):
    """3D position RSS uncertainty for state [x y z clk]; sig_clk None -> free.

    Same model as Step 26: sqrt(trace(Cov[:3,:3])).

    With n >= 3 satellites and a clock prior, the system is solvable
    (3 range equations + 1 clock prior = 4 equations for 4 unknowns).
    Without a clock prior, n < 4 is rank-deficient and the SVD check
    returns inf.  The previous `if n < 4: return inf` rejected the
    3-satellite + clock-prior case, which is the most operationally
    relevant scenario (urban canyon with heavy obstruction).
    """
    n = len(los)
    if n < 3:
        return np.inf
    A = np.column_stack([-los, np.ones(n)])
    N = A.T @ A
    if sig_clk is not None and np.isfinite(sig_clk) and sig_clk > 0:
        N = N + np.diag([0.0, 0.0, 0.0, 1.0 / sig_clk ** 2])
    sv = np.linalg.svd(N, compute_uv=False)
    if sv[-1] < 1e-10 * sv[0] or sv[-1] < 1e-12:
        return np.inf
    try:
        Cov = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        return np.inf
    v = np.trace(Cov[:3, :3])
    return float(np.sqrt(v)) if v > 0 else np.inf


def stack_normals(los_list):
    """Pre-build the (m, 4, 4) stack of per-epoch normal matrices A^T A for a
    list of line-of-sight arrays.  Geometry is fixed across days, so this is
    computed once; each day then only adds its clock-prior term and batch-
    inverts -- ~4 orders of magnitude fewer np.linalg.inv calls than the
    per-(day, k, prior, epoch) loop it replaces."""
    N = np.zeros((len(los_list), 4, 4))
    for i, los in enumerate(los_list):
        los = np.asarray(los, float)
        if len(los) < 3:                       # 3 ranges + clock prior = solvable
            N[i] = np.nan
            continue
        A = np.column_stack([-los, np.ones(len(los))])
        N[i] = A.T @ A
    return N


def median_pos_sigma(Nstack, sig_clk):
    """Vectorised median over epochs of the 3D position RSS uncertainty, given
    the pre-built normal-matrix stack and a scalar clock-prior sigma
    (None/inf -> clock free)."""
    M = Nstack.copy()
    if sig_clk is not None:
        if not np.isfinite(sig_clk) or sig_clk <= 0:
            return np.nan
        M[:, 3, 3] += 1.0 / sig_clk ** 2
    ok = np.isfinite(M).all(axis=(1, 2))
    # rank / conditioning guard, vectorised
    sv = np.linalg.svd(M[ok], compute_uv=False)
    good = (sv[:, -1] > 1e-12) & (sv[:, -1] > 1e-10 * sv[:, 0])
    idx = np.where(ok)[0][good]
    if idx.size == 0:
        return np.inf
    try:
        Cov = np.linalg.inv(M[idx])
    except np.linalg.LinAlgError:
        return np.inf
    v = Cov[:, 0, 0] + Cov[:, 1, 1] + Cov[:, 2, 2]
    v = v[v > 0]
    return float(np.median(np.sqrt(v))) if v.size else np.inf


_SP3G_CACHE: dict = {}


def _gridded_sp3(dtag, grid):
    """load_sp3 + sat_grid for one day, cached -- it is station-independent, so
    without the cache it was recomputed once per representative station."""
    if dtag not in _SP3G_CACHE:
        sp3 = load_sp3(dtag)
        _SP3G_CACHE[dtag] = sat_grid(sp3, grid) if sp3 is not None else None
    return _SP3G_CACHE[dtag]


def representative_geometry(station_ecef_m):
    """Compute representative satellite LOS vectors from all available SP3 days.

    Returns a dict: k_nsat -> list of LOS arrays (one per sampled epoch).
    The GPS constellation repeats with a ~10-day cycle, so 52 days of SP3
    give a representative sample of the geometry distribution.
    """
    sp3_days = sorted({f.stem.split("_")[1][:7]
                       for f in (PROJECT_ROOT / "data" / "rinex" / "sp3").glob(
                           "COD0MGXFIN_*_ORB.SP3.gz")})
    # The GPS constellation repeats with a ~10-day cycle, so 52 days of SP3
    # give a representative sample of the full geometry distribution.
    sp3_days = sp3_days[:52]
    grid = np.arange(0, 86400.0, GRID_STEP_S)
    geometries = {k: [] for k in K_VALUES}

    for dtag in sp3_days:
        sp3g = _gridded_sp3(dtag, grid)
        if sp3g is None:
            continue
        for gi in range(len(grid)):
            vis = []
            for sname, pg in sp3g.items():
                pp = pg[gi]
                if not np.all(np.isfinite(pp)):
                    continue
                e, u, _ = elev_los(pp, station_ecef_m)
                if e > ELEV_MASK:
                    vis.append((e, u))
            if len(vis) < 3:
                continue
            vis.sort(key=lambda z: -z[0])
            for k in K_VALUES:
                kk = min(k, len(vis))
                if kk < 3:
                    continue
                los = np.array([u for _, u in vis[:kk]])
                geometries[k].append(los)

    return geometries


def multi_station_geometry(stations_ecef):
    """Compute representative geometry averaged over multiple station locations.

    Returns a dict: k_nsat -> list of LOS arrays (one per sampled epoch per
    station).  Using multiple stations at different latitudes makes the
    position-domain claim robust to latitude-dependent geometry variation.
    """
    # Disk cache: the geometry computation (~52 SP3 days x 6 stations x 48
    # epochs) takes several minutes and is deterministic given the station
    # set and SP3 files.  Cache to disk so re-runs after code changes to
    # the per-day propagation logic are instant.
    import pickle, hashlib
    cache_dir = PROJECT_ROOT / "data" / "step35_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Hash the station ECEF coordinates to invalidate the cache if the
    # representative station set changes.
    station_hash = hashlib.md5(
        np.asarray(stations_ecef).tobytes()).hexdigest()[:12]
    cache_file = cache_dir / f"geom_{station_hash}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    combined = {k: [] for k in K_VALUES}
    for st_ecef in stations_ecef:
        g = representative_geometry(st_ecef)
        for k in K_VALUES:
            combined[k].extend(g.get(k, []))
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(combined, f)
    except Exception:
        pass
    return combined


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load Step 34 per-day results
    s34 = json.load(open(OUT / "step_34_heldout_lodo.json"))
    b34 = s34["broadcast_spp"]
    per_day = b34["per_day"]
    n_days = len(per_day)
    print(f"Step 35: position-domain benefit over {n_days} days (leave-one-day-out)")

    # Use multiple representative stations at different latitudes to make
    # the position-domain claim robust to latitude-dependent geometry
    # variation.  The position-domain gain is a function of the ratio
    # sig_kernel/sig_self and the geometry; averaging over stations at
    # different latitudes (equatorial, mid-latitude, high-latitude) samples
    # the full geometry distribution a global user would experience.
    reg = td.registry("broadcast", 2024, min_days=5)

    # Select representative stations at different latitude bands
    target_lats = [0.0, 30.0, 45.0, 60.0, -30.0, -45.0]
    rep_stations = []
    used_names = set()
    for tlat in target_lats:
        best_st = None
        best_dist = 999
        for nm, st in reg.items():
            if nm in used_names:
                continue
            d = abs(st.lat_deg - tlat)
            if d < best_dist:
                best_dist = d
                best_st = st
        if best_st is not None and best_dist < 20.0:
            rep_stations.append(best_st)
            used_names.add(best_st.name)

    if not rep_stations:
        # Fallback: use the first station
        best_st = next(iter(reg.values()))
        rep_stations = [best_st]

    print(f"  Representative stations ({len(rep_stations)}):")
    for st in rep_stations:
        print(f"    {st.name} ({st.lat_deg:.1f}N, {st.lon_deg:.1f}E)")

    # Compute representative geometry from multiple stations
    print("  Computing representative satellite geometry from SP3 files...")
    stations_ecef = [st.ecef_m for st in rep_stations]
    geom = multi_station_geometry(stations_ecef)
    # Pre-build the per-epoch normal matrices ONCE (geometry is day-independent).
    normals = {}
    for k in K_VALUES:
        print(f"    k={k}: {len(geom[k])} epoch-geometries")
        normals[k] = stack_normals(geom[k]) if len(geom[k]) >= 10 else None

    # For each day in Step 34, compute the position-domain gain
    # using the per-day clock sigmas and the representative geometry.
    per_day_pos = []
    for pd in per_day:
        sig_raw = pd["raw_m"]
        sig_kernel = pd["kernel_m"]
        sig_mean = pd["net_mean_m"]
        if any(v is None or not (np.isfinite(v) and v > 0) for v in (sig_raw, sig_kernel, sig_mean)):
            continue

        day_result = {
            "day": pd["day"],
            "n_targets": pd["n_targets"],
            "sig_raw_m": sig_raw,
            "sig_kernel_m": sig_kernel,
            "sig_mean_m": sig_mean,
            "skill_vs_raw_pct": pd["skill_vs_raw_pct"],
            "kernel_vs_mean_pct": pd["kernel_vs_mean_pct"],
            "by_k": {}
        }

        for k in K_VALUES:
            Nstack = normals.get(k)
            if Nstack is None:
                continue
            sigmas = {
                "none": None,
                "self": sig_raw,
                "net_mean": sig_mean,
                "kernel": sig_kernel,
            }
            med_pos = {pn: median_pos_sigma(Nstack, sc) for pn, sc in sigmas.items()}

            # Gains
            gain_self = (100.0 * (1 - med_pos["kernel"] / med_pos["self"])
                         if np.isfinite(med_pos["self"]) and med_pos["self"] > 0
                         else np.nan)
            gain_mean = (100.0 * (1 - med_pos["kernel"] / med_pos["net_mean"])
                         if np.isfinite(med_pos["net_mean"]) and med_pos["net_mean"] > 0
                         else np.nan)
            gain_mean_vs_self = (100.0 * (1 - med_pos["net_mean"] / med_pos["self"])
                                 if np.isfinite(med_pos["self"]) and med_pos["self"] > 0
                                 else np.nan)

            day_result["by_k"][str(k)] = {
                "pos_sigma_m": med_pos,
                "kernel_vs_self_pct": gain_self,
                "kernel_vs_mean_pct": gain_mean,
                "mean_vs_self_pct": gain_mean_vs_self,
            }

        per_day_pos.append(day_result)

    n_valid = len(per_day_pos)
    print(f"  {n_valid} days with valid position-domain results")

    # Aggregate: median and bootstrap CI over days
    def ci95(vals):
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) < 3:
            return [float("nan"), float("nan")]
        bs = [float(np.median(RNG.choice(vals, len(vals), replace=True)))
              for _ in range(5000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    by_k_agg = {}
    for k in K_VALUES:
        ks = str(k)
        gains_self = [d["by_k"][ks]["kernel_vs_self_pct"] for d in per_day_pos
                      if ks in d["by_k"] and np.isfinite(d["by_k"][ks]["kernel_vs_self_pct"])]
        gains_mean = [d["by_k"][ks]["kernel_vs_mean_pct"] for d in per_day_pos
                      if ks in d["by_k"] and np.isfinite(d["by_k"][ks]["kernel_vs_mean_pct"])]
        gains_mean_vs_self = [d["by_k"][ks]["mean_vs_self_pct"] for d in per_day_pos
                              if ks in d["by_k"] and np.isfinite(d["by_k"][ks]["mean_vs_self_pct"])]
        if not gains_self:
            continue
        by_k_agg[ks] = {
            "n_days": len(gains_self),
            "kernel_vs_self_pct": {
                "median": float(np.median(gains_self)),
                "ci95": ci95(gains_self),
                "frac_days_positive": float(np.mean(np.array(gains_self) > 0)),
            },
            "kernel_vs_mean_pct": {
                "median": float(np.median(gains_mean)),
                "ci95": ci95(gains_mean),
                "frac_days_positive": float(np.mean(np.array(gains_mean) > 0)),
            },
            "mean_vs_self_pct": {
                "median": float(np.median(gains_mean_vs_self)),
                "ci95": ci95(gains_mean_vs_self),
            },
        }

    result = {
        "test": "position_domain_leave_one_day_out",
        "method": ("For each of the days in Step 34, the per-day clock-"
                   "prediction errors (raw, kernel, net_mean) are propagated "
                   "through the analytical position-uncertainty model of Step 26 "
                   "(pos_sigma = sqrt(trace(inv(A^T A + diag(0,0,0,1/sig^2))[:3,:3]))) "
                   "using representative satellite geometry computed from 52 days "
                   "of COD0MGXFIN SP3 files, averaged over multiple representative "
                   "stations at different latitudes (equatorial, mid-latitude, "
                   "high-latitude, and southern hemisphere) to sample the full "
                   "geometry distribution a global user would experience.  The GPS "
                   "constellation repeats with a ~10-day cycle, so the median "
                   "geometry over 52 days is representative of the full year.  The "
                   "clock sigmas are the per-day medians of the held-out prediction "
                   "errors from Step 34 (hyperparameters selected on a different "
                   "training day)."),
        "n_days": n_valid,
        "representative_stations": [
            {"name": st.name, "lat": st.lat_deg, "lon": st.lon_deg}
            for st in rep_stations
        ],
        "n_geometry_epochs": {str(k): len(geom.get(k, [])) for k in K_VALUES},
        "by_satellite_count": by_k_agg,
        "per_day": per_day_pos,
        "timestamp": datetime.now().isoformat(),
    }

    # Headline
    k4 = by_k_agg.get("4", {})
    k8 = by_k_agg.get("8", {})
    k3 = by_k_agg.get("3", {})
    if k4:
        result["headline"] = (
            f"Position-domain benefit over {n_valid} days (leave-one-day-out, "
            f"full year 2024): the network kernel reduces 3D position uncertainty "
            f"by {k4['kernel_vs_self_pct']['median']:+.0f}% at 4 visible satellites "
            f"(95% CI [{k4['kernel_vs_self_pct']['ci95'][0]:+.0f}, "
            f"{k4['kernel_vs_self_pct']['ci95'][1]:+.0f}]), "
            f"{k8.get('kernel_vs_self_pct', {}).get('median', 0):+.0f}% at 8, and "
            f"{k3.get('kernel_vs_self_pct', {}).get('median', 0):+.0f}% at 3 "
            f"versus a no-network prior.")
        print(f"\n{result['headline']}")

    (OUT / "step_35_position_lodo.json").write_text(json.dumps(result, indent=2))
    print(f"saved {OUT / 'step_35_position_lodo.json'}")


if __name__ == "__main__":
    main()
