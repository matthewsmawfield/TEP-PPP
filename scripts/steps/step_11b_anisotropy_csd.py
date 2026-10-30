#!/usr/bin/env python3
"""
STEP 11b  (Phase 1 / T2, phase-coherent) -- EW vs NS anisotropy via the
          corrected Papers 1-2 phase-alignment index.

This step now uses the corrected Papers 1-2 statistic:

    phase_alignment_index = cos(arg(sum_f S_ij(f))),

where S_ij(f) = conj(X_i(f)) X_j(f) over the 10-500 uHz band.  Earlier
versions of this script computed <coherency> x |magnitude-weighted mean
phasor|, i.e. band-averaged coherency (which Paper 1 reports as failing)
times a phase-concentration measure, instead of the cosine of the mean phase
angle that Papers 1-2 define.

Method:
  * per day: band-pass each receiver-clock residual to 10-500 uHz, rfft once;
  * per station pair: compute the complex scalar S = sum_f conj(F_i) F_j
    over the band;
  * tag each pair with (great-circle distance, baseline azimuth);
  * split into an EW sector (around azimuth 90 deg) and an NS sector
    (around azimuth 0/180 deg);
  * within each sector, bin by distance and compute the mean phase-alignment
    index, cos(arg(S)), per bin;
  * fit C(r) with AIC model selection (exponential, power-law, gaussian,
    constant) per sector -> lambda_EW, lambda_NS;
  * the robust statistic is the per-distance-bin EW-minus-NS phase-alignment
    difference (no fit), station-clustered bootstrapped.

Nulls: phase_randomise (per-clock Fourier phases) and azimuth_shuffle (permute
which sector each pair belongs to, preserving the phase-alignment marginal).
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

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260911)

F_LO_UHZ, F_HI_UHZ = 10.0, 500.0
N_BINS = 18
DMIN_KM, DMAX_KM = 150.0, 18000.0
MIN_PAIRS = 20
SECT_HALFWIDTH_DEG = 25.0
_EDGES = np.geomspace(DMIN_KM, DMAX_KM, N_BINS + 1)

# Canonical intra-day time grid so every station-day shares ONE rfft frequency
# grid (required for coherent stacking across days).  The grid cadence is
# adaptive: broadcast SPP data is at 30 s (2880 pts) or 60 s (1440 pts),
# while precise CLK products are at 300 s (288 pts).  Using a fixed 1440-pt
# grid on 300 s data would require 720 valid samples (0.5*N_GRID), but only
# 288 exist — silently producing zero stations.  The grid is now chosen per
# product family from the median cadence of the first valid station-day.
#
# Nyquist at 300 s = 1667 uHz, well above the 500 uHz band upper edge.
_GRID_CACHE: dict[int, dict] = {}


def _grid_for_cadence(dt: float) -> dict:
    """Build the rfft grid for a given cadence (seconds). Cached."""
    n_grid = int(round(86400.0 / dt))
    if n_grid in _GRID_CACHE:
        return _GRID_CACHE[n_grid]
    grid = np.arange(n_grid) * dt
    freq = np.fft.rfftfreq(n_grid, d=dt)
    fband = (freq >= F_LO_UHZ * 1e-6) & (freq <= F_HI_UHZ * 1e-6)
    hann = np.hanning(n_grid)
    hann /= np.sqrt(np.mean(hann ** 2))
    g = {"n_grid": n_grid, "dt": dt, "grid": grid, "freq": freq,
         "fband": fband, "hann": hann}
    _GRID_CACHE[n_grid] = g
    return g


def _detect_cadence(series) -> float:
    """Detect the median time step (seconds) from the first valid station."""
    for n in series:
        t = np.asarray(series[n]["t"], float)
        g = np.isfinite(t)
        if g.sum() >= 10:
            dt = float(np.median(np.diff(t[g])))
            if np.isfinite(dt) and dt > 0:
                return dt
    return 60.0  # fallback


def _is_ew(az):
    a = az % 180.0
    return abs(a - 90.0) <= SECT_HALFWIDTH_DEG


def _is_ns(az):
    a = az % 180.0
    return a <= SECT_HALFWIDTH_DEG or a >= 180.0 - SECT_HALFWIDTH_DEG


def _day_records(series, phase_rng=None, grid=None):
    """[(dist_km, az_deg, name_i, name_j, S)].

    S = sum_f conj(F_i(f)) F_j(f) is the complex scalar cross-spectrum
    summed over the selected band.  *grid* is the dict from
    _grid_for_cadence(); if None it is auto-detected from the series
    cadence so that 300 s CLK products use a 288-pt grid rather than the
    1440-pt broadcast grid (which would silently filter out every station).
    """
    names = list(series)
    if len(names) < 4:
        return []
    if grid is None:
        dt = _detect_cadence(series)
        grid = _grid_for_cadence(dt)
    n_grid = grid["n_grid"]
    g_grid = grid["grid"]
    dt = grid["dt"]
    fband = grid["fband"]
    hann = grid["hann"]
    RF, STA = {}, {}
    for n in names:
        t = np.asarray(series[n]["t"], float)
        x = np.asarray(series[n]["resid"], float)
        g = np.isfinite(t) & np.isfinite(x)
        if g.sum() < 0.5 * n_grid:
            continue
        t, x = t[g], x[g]
        t0 = t[0]
        # resample onto the canonical grid (seconds-of-day origin at t0)
        xr = np.interp(g_grid, t - t0, x, left=x[0], right=x[-1])
        b = ts.bandpass(xr, dt, F_LO_UHZ, F_HI_UHZ)
        if not np.all(np.isfinite(b)) or np.std(b) < 1e-12:
            continue
        F = np.fft.rfft(b * hann)[fband]
        if F.size < 4:
            continue
        if phase_rng is not None:
            F = np.abs(F) * np.exp(1j * phase_rng.uniform(-np.pi, np.pi, F.shape))
        RF[n] = F
        STA[n] = series[n]["station"]
    ok = list(RF)
    recs = []
    for i, j in itertools.combinations(range(len(ok)), 2):
        ni, nj = ok[i], ok[j]
        Fi, Fj = RF[ni], RF[nj]
        if len(Fi) != len(Fj):
            continue
        a, b = STA[ni], STA[nj]
        dist = td.great_circle_km(a, b)
        if not (DMIN_KM <= dist <= DMAX_KM):
            continue
        az = td.baseline_azimuth_deg(a, b)
        S = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(S) or S == 0:
            continue
        recs.append((dist, az, ni, nj, S))
    return recs


def _phase_align(recs):
    """Mean Papers 1-2 phase-alignment index, cos(arg(S)), over records."""
    vals = [np.cos(np.angle(r[4])) for r in recs
            if np.isfinite(r[4]) and r[4] != 0]
    if len(vals) < 3:
        return np.nan
    v = float(np.mean(vals))
    return v if np.isfinite(v) else np.nan


def _binned_curve(recs):
    """(centers, phase_alignment, counts) for a set of pair records."""
    if not recs:
        return np.array([]), np.array([]), np.array([])
    seps = np.array([r[0] for r in recs])
    idx = np.clip(np.digitize(seps, _EDGES) - 1, 0, N_BINS - 1)
    ctr, val, cnt = [], [], []
    for b in range(N_BINS):
        sel = [recs[i] for i in np.where(idx == b)[0]]
        if len(sel) < MIN_PAIRS:
            continue
        v = _phase_align(sel)
        if np.isfinite(v):
            ctr.append(np.sqrt(_EDGES[b] * _EDGES[b + 1]))
            val.append(v)
            cnt.append(len(sel))
    return np.array(ctr), np.array(val), np.array(cnt)


def _binned_curve_from_sums(sums, wgts):
    """(centers, phase_alignment, counts) from pre-accumulated bin sums."""
    ctr, val, cnt = [], [], []
    for b in range(N_BINS):
        if wgts[b] < MIN_PAIRS:
            continue
        v = float(sums[b] / wgts[b])
        if np.isfinite(v):
            ctr.append(np.sqrt(_EDGES[b] * _EDGES[b + 1]))
            val.append(v)
            cnt.append(int(wgts[b]))
    return np.array(ctr), np.array(val), np.array(cnt)


def _fit(ctr, val, cnt):
    """AIC model selection; return (lambda_km, R2) for the best model.

    A correlation length is reported only when the best model is exponential
    or Gaussian, the fit explains at least 10% of the weighted variance, the
    amplitude is positive, and the estimated scale is not at the search
    boundary.  With fewer than 6 distance bins the scale is not resolvable.
    """
    if len(ctr) < 6 or len(val) == 0:
        return np.nan, np.nan
    w = np.asarray(cnt, float) if (cnt is not None and len(cnt)) else np.ones_like(ctr)
    m = ts.fit_all_models(ctr, val, weight=w)
    best = m.get("best_model", "none")
    if best == "none":
        return np.nan, np.nan
    mm = m.get("models", {}).get(best, {})
    R2 = float(mm.get("R2", np.nan))
    p = mm.get("params", [])
    sig = mm.get("sigma_params") or [np.nan] * len(p)
    lam = np.nan
    if best in ("exponential", "gaussian"):
        if (len(p) > 1 and R2 > 0.10 and float(p[0]) > 0 and
                float(p[1]) < 39900.0 and float(p[1]) > 100.0):
            lam = float(p[1])
    elif best == "power_law":
        lam = np.nan
    elif best == "constant":
        lam = np.nan
    return lam, R2


def run(product, date_tags, label):
    days = [s for _, s in td.load_multi_day(product, date_tags) if len(s) >= 6]
    if not days:
        print(f"[{label}] WARNING: no days with >= 6 stations loaded from "
              f"{len(date_tags)} date tags — skipping arm (cannot produce a "
              f"meaningful result). This is a LOUD failure, not a negative result.")
        return None
    # Detect the cadence from the first valid day and build ONE shared grid
    # so all days share the same rfft frequency grid (required for coherent
    # stacking).  This fixes the silent-zero-station bug where a 1440-pt
    # grid was applied to 300 s CLK data (288 pts).
    cadence = _detect_cadence(days[0])
    grid = _grid_for_cadence(cadence)
    print(f"[{label}] cadence {cadence:.0f} s -> {grid['n_grid']} pt grid, "
          f"Nyquist {1e6/(2*cadence):.0f} uHz")

    # Collect pair-day records directly.  Each record now holds the single
    # complex scalar S for that pair and day, so per-pair aggregation is not
    # needed and there is no OOM risk.
    all_recs = []
    n_pair_days = 0
    for s in days:
        recs = _day_records(s, grid=grid)
        n_pair_days += len(recs)
        all_recs.extend(recs)
    n_unique_pairs = len({tuple(sorted((r[2], r[3]))) for r in all_recs})
    names = sorted({r[2] for r in all_recs} | {r[3] for r in all_recs})
    ew = [r for r in all_recs if _is_ew(r[1])]
    ns = [r for r in all_recs if _is_ns(r[1])]
    print(f"[{label}] {len(days)} days, {len(names)} stations, "
          f"{n_pair_days:,} pair-days ({n_unique_pairs:,} unique pairs) "
          f"-> {len(all_recs):,} records (EW {len(ew):,} / NS {len(ns):,})")

    if len(all_recs) == 0:
        print(f"[{label}] WARNING: 0 pair-days produced from {len(days)} days. "
              f"This is a LOUD failure — the arm is excluded, not reported as "
              f"a negative result. Check data loading and grid cadence.")
        return None

    ce, ve, cte = _binned_curve(ew)
    cn, vn, ctn = _binned_curve(ns)
    lam_ew, r2_ew = _fit(ce, ve, cte)
    lam_ns, r2_ns = _fit(cn, vn, ctn)

    # per-distance-bin EW-minus-NS phase-alignment difference (robust, no fit),
    # on the common set of bin centres
    common = sorted(set(np.round(ce).astype(int)) & set(np.round(cn).astype(int)))
    diff_by_bin = {}
    for c in common:
        ie = int(np.argmin(np.abs(ce - c)))
        jn = int(np.argmin(np.abs(cn - c)))
        diff_by_bin[int(c)] = float(ve[ie] - vn[jn])

    # station-clustered bootstrap on lambda_EW/lambda_NS and mean EW-NS diff
    def _stat(sel_names):
        from collections import Counter
        cnt = Counter(sel_names)
        # accumulate weighted cos(arg S) per distance bin directly
        ew_sums = [0.0] * N_BINS
        ew_wgts = [0.0] * N_BINS
        ns_sums = [0.0] * N_BINS
        ns_wgts = [0.0] * N_BINS
        for r in all_recs:
            k = cnt.get(r[2], 0) * cnt.get(r[3], 0)
            if k == 0:
                continue
            b = int(np.clip(np.digitize(r[0], _EDGES) - 1, 0, N_BINS - 1))
            v = float(np.cos(np.angle(r[4]))) * k
            if _is_ew(r[1]):
                ew_sums[b] += v
                ew_wgts[b] += k
            elif _is_ns(r[1]):
                ns_sums[b] += v
                ns_wgts[b] += k
        c1, v1, w1 = _binned_curve_from_sums(ew_sums, ew_wgts)
        c2, v2, w2 = _binned_curve_from_sums(ns_sums, ns_wgts)
        l1, _ = _fit(c1, v1, w1)
        l2, _ = _fit(c2, v2, w2)
        if not (np.isfinite(l1) and np.isfinite(l2) and l2 > 1):
            return np.nan
        return l1 / l2

    pt, lo, hi, samp = ts.station_clustered_bootstrap(
        _stat, names, n_boot=100, n_jobs=1, rng=RNG)
    p_ratio_le1 = float(np.mean(samp <= 1.0)) if len(samp) else np.nan

    # Nulls, operating on the pair-day records.
    #
    #   phase_randomise : replace each S by |S| with an independent random
    #     phase.  This destroys the phase-alignment index entirely, so BY
    #     DESIGN neither sector fits a correlation length -- the ratio l1/l2
    #     is then undefined and cannot be built into a null distribution
    #     (the previous version silently produced zero null samples and a
    #     None z-score).  The right statistic for this null is an EXISTENCE
    #     test: how often does the null reproduce a fittable distance decay
    #     (best model exponential/Gaussian, R^2 > 0.30) in EITHER sector?
    #     If the real data fits both sectors and the null fits neither, the
    #     distance structure is real; the ANISOTROPY question is then
    #     answered separately by azimuth_shuffle.
    #
    #   azimuth_shuffle : permute which sector each pair belongs to,
    #     preserving the phase-alignment marginal.  This is the meaningful
    #     null for the EW/NS ratio -- it asks whether the sector split
    #     carries information beyond a random azimuth partition.
    null_ratio = {}
    # -- phase_randomise: existence test on the fit --
    real_fits_both = bool(np.isfinite(lam_ew) and np.isfinite(lam_ns))
    pr_hits = 0
    pr_n = 0
    for _k in range(50):
        rng = np.random.default_rng(int(RNG.integers(0, 2**31)))
        surro = [(r[0], r[1], r[2], r[3],
                  np.abs(r[4]) * np.exp(1j * rng.uniform(-np.pi, np.pi)))
                 for r in all_recs]
        e2 = [r for r in surro if _is_ew(r[1])]
        n2 = [r for r in surro if _is_ns(r[1])]
        c1, v1, w1 = _binned_curve(e2); c2, v2, w2 = _binned_curve(n2)
        l1, r1 = _fit(c1, v1, w1); l2, r2 = _fit(c2, v2, w2)
        pr_n += 1
        if (np.isfinite(l1) and r1 > 0.30) or (np.isfinite(l2) and r2 > 0.30):
            pr_hits += 1
    null_ratio["phase_randomise"] = {
        "statistic": "existence: fraction of nulls with a fittable decay (R2>0.30) in either sector",
        "real_fits_both_sectors": real_fits_both,
        "real_R2_ew": float(r2_ew) if np.isfinite(r2_ew) else None,
        "real_R2_ns": float(r2_ns) if np.isfinite(r2_ns) else None,
        "null_n": pr_n, "null_hits": pr_hits,
        "p_value": float((pr_hits + 1) / (pr_n + 1)),
    }
    # -- azimuth_shuffle: the anisotropy-ratio null --
    rr = []
    for _k in range(50):
        rng = np.random.default_rng(int(RNG.integers(0, 2**31)))
        az_perm = rng.permutation([r[1] for r in all_recs])
        surro = [(r[0], az_perm[i], r[2], r[3], r[4])
                 for i, r in enumerate(all_recs)]
        e2 = [r for r in surro if _is_ew(r[1])]
        n2 = [r for r in surro if _is_ns(r[1])]
        c1, v1, w1 = _binned_curve(e2); c2, v2, w2 = _binned_curve(n2)
        l1, _ = _fit(c1, v1, w1); l2, _ = _fit(c2, v2, w2)
        if np.isfinite(l1) and np.isfinite(l2) and l2 > 1:
            rr.append(l1 / l2)
    rr = np.array(rr)
    null_ratio["azimuth_shuffle"] = {
        "statistic": "ratio l_EW/l_NS under random azimuth partition",
        "mean": float(np.nanmean(rr)) if len(rr) else None,
        "sd": float(np.nanstd(rr)) if len(rr) else None,
        "n": int(len(rr)),
        "p_ratio_ge_real": float(np.mean(rr >= pt)) if (len(rr) and np.isfinite(pt)) else None,
        "p_ratio_le_real": float(np.mean(rr <= pt)) if (len(rr) and np.isfinite(pt)) else None,
    }

    # z of the observed ratio vs the azimuth-shuffle null (the meaningful
    # anisotropy null; the phase-randomise null cannot produce a ratio).
    az = null_ratio["azimuth_shuffle"]
    ratio_z = None
    if np.isfinite(pt) and az["sd"] and az["sd"] > 1e-9:
        ratio_z = float((pt - az["mean"]) / az["sd"])

    out = {
        "label": label, "n_days": len(days), "n_stations": len(names),
        "n_pair_days": n_pair_days, "n_unique_pairs": n_unique_pairs,
        "n_ew": len(ew), "n_ns": len(ns),
        "lambda_ew_km": float(lam_ew) if np.isfinite(lam_ew) else None,
        "lambda_ns_km": float(lam_ns) if np.isfinite(lam_ns) else None,
        "R2_ew": float(r2_ew) if np.isfinite(r2_ew) else None,
        "R2_ns": float(r2_ns) if np.isfinite(r2_ns) else None,
        "lambda_ratio_ew_ns": float(pt) if np.isfinite(pt) else None,
        "ratio_ci95": [float(lo) if np.isfinite(lo) else None,
                       float(hi) if np.isfinite(hi) else None],
        "p_ratio_le_1": p_ratio_le1,
        "ratio_z_vs_azimuth_null": ratio_z,
        "distance_structure_exists": bool(
            null_ratio["phase_randomise"]["real_fits_both_sectors"]
            and null_ratio["phase_randomise"]["null_hits"] == 0),
        "nulls": null_ratio,
        "phase_alignment_diff_by_bin_km": diff_by_bin,
        "ew_curve": {"dist_km": ce.tolist(),
                     "phase_alignment_index": ve.tolist()},
        "ns_curve": {"dist_km": cn.tolist(),
                     "phase_alignment_index": vn.tolist()},
        # T2 (anisotropy) passes only if EW clearly exceeds NS: ratio > 1, its
        # lower CI > 1, AND the ratio sits >2 null-sd ABOVE the azimuth-shuffle
        # mean.  Isotropic recovery of lambda_T on both sectors is reported
        # separately and does NOT constitute a T2 pass.
        "T2_pass": bool(np.isfinite(pt) and pt > 1.0 and np.isfinite(lo) and lo > 1.0
                        and ratio_z is not None and ratio_z > 2.0),
        "isotropic_lambda_recovered_both_sectors": bool(
            np.isfinite(lam_ew) and np.isfinite(lam_ns)
            and 1000.0 <= lam_ew <= 6000.0 and 1000.0 <= lam_ns <= 6000.0),
    }
    prz = null_ratio["phase_randomise"]
    print(f"  lambda_EW = {lam_ew:.0f} km (R2 {r2_ew:.2f}) / lambda_NS = {lam_ns:.0f} km "
          f"(R2 {r2_ns:.2f})   ratio {pt:.2f}  CI [{lo:.2f}, {hi:.2f}]  "
          f"p(<=1) {p_ratio_le1:.3f}  z_vs_azimuth_null {ratio_z if ratio_z is None else round(ratio_z,1)}")
    print(f"  distance structure: real fits both sectors={prz['real_fits_both_sectors']}, "
          f"phase-randomised null reproduces a fittable decay in {prz['null_hits']}/{prz['null_n']} "
          f"trials (p={prz['p_value']:.3f})")
    print(f"  EW-minus-NS phase alignment by distance bin: "
          + "  ".join(f"{k}km:{v:+.3f}" for k, v in list(diff_by_bin.items())[:8]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"test": "T2_anisotropy_phase_coherent_CSD",
           "method": ("Papers 1-2 phase-alignment index cos(arg(sum_f S_ij(f))) "
                      "split by baseline azimuth"),
           "timestamp": datetime.now().isoformat(), "arms": {}}

    res["arms"]["global_broadcast"] = run(
        "broadcast", [f"2024{d:03d}" for d in range(100, 107)], "global_broadcast")

    clk_days = td.available_dates("clk_cod", 2024)
    if len(clk_days) >= 60:
        precise = run("clk_cod", clk_days, "global_precise_clk")
        if precise is not None:
            res["arms"]["global_precise_clk"] = precise
        else:
            print("\n*** global_precise_clk arm returned None (loud failure) — "
                  "NOT included in output. This is NOT a negative result. ***")

    a = res["arms"].get("global_precise_clk") or res["arms"]["global_broadcast"]
    if a is None:
        res["T2_pass"] = False
        res["headline"] = ("T2 anisotropy: both arms failed to produce data. "
                           "This is a pipeline failure, not a negative result.")
        (OUT / "step_11b_anisotropy_csd.json").write_text(json.dumps(res, indent=2))
        print(f"\n{res['headline']}\nsaved {OUT / 'step_11b_anisotropy_csd.json'}")
        return
    res["T2_pass"] = a["T2_pass"]
    iso = a.get("isotropic_lambda_recovered_both_sectors")
    res["headline"] = (
        f"Phase-alignment anisotropy ({a['label']}, {a['n_stations']} stations, "
        f"{a['n_pair_days']:,} pair-days): lambda_EW = {a['lambda_ew_km']:.0f} km, "
        f"lambda_NS = {a['lambda_ns_km']:.0f} km"
        f"{' (both in the TEP band -- isotropic lambda_T recovered on two orientation subsets)' if iso else ''}; "
        f"ratio {a['lambda_ratio_ew_ns']:.2f}, 95% CI {a['ratio_ci95']}, "
        f"z vs azimuth-shuffle null {a['ratio_z_vs_azimuth_null']}. "
        f"{'EW > NS anisotropy confirmed' if a['T2_pass'] else 'EW/NS anisotropy not resolved at this network scale'}.")
    (OUT / "step_11b_anisotropy_csd.json").write_text(json.dumps(res, indent=2))
    print(f"\n{res['headline']}\nsaved {OUT / 'step_11b_anisotropy_csd.json'}")


if __name__ == "__main__":
    main()
