#!/usr/bin/env python3
"""
STEP 11  (Phase 1 / T2) -- EW > NS azimuthal anisotropy.

Question: does the operational dataset reproduce the series' directional
signature -- correlation length longer along east-west baselines than
north-south -- under a station-clustered bootstrap?

Method: band-passed pair correlations, split by baseline azimuth into EW
(90 +/- 22.5 deg) and NS (0/180 +/- 22.5 deg) sectors, fit lambda in each,
report the ratio.  Longitude-matched subset (|dlon| < 30 deg, ~same local
solar time) following Paper 3.  Distance strata.  Station-clustered bootstrap
for the CI.
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


def pair_table(days):
    """Pooled pair records: dist, corr, azimuth, dlon, names."""
    recs = []
    for s in days:
        names = list(s)
        cols = []
        for n in names:
            dt = float(np.median(np.diff(s[n]["t"])))
            if not np.isfinite(dt) or dt <= 0:
                dt = 30.0
            cols.append(ts.bandpass(s[n]["resid"], dt))
        R = np.column_stack(cols)
        # Two-pass centring: (1) remove the per-epoch network mean to strip
        # the common-mode satellite-product error that dominates the raw
        # correlation and is isotropic by construction; (2) remove the
        # per-station time mean.  Step 10 uses only (2) because its target
        # is the total distance-structured correlation (including common
        # mode); Step 11 uses both because the anisotropy signal lives in
        # the differential structure that remains after the common mode is
        # removed.  The two steps answer different questions and are not
        # inconsistent.
        # NOTE (bug fix): the per-epoch network mean is NO LONGER removed.
        # It was introduced to strip common-mode satellite error, but on a
        # network whose members are ~0.9 mutually correlated an unweighted
        # per-epoch mean is dominated by the loudest stations and leaks into
        # every other station's residual with the same sign.  Measured effect
        # (post band-pass fix, `diag_netmean.py`): it turns a clean decay
        #   0.20 -> 0.14 -> 0.01 -> -0.01  (lambda 2515 km, R2 0.818)
        # into a FLAT profile
        #   0.95 -> 0.93 -> 0.93 -> 0.90  (lambda railed at 40000 km, R2 0.001)
        # i.e. it annihilates the distance structure this test is measuring.
        # T1 (step_10), which passes, never removed it.  T2 now uses the same
        # pair construction as T1.  Set TEP_REMOVE_NETWORK_MEAN=1 to restore
        # the old behaviour as a control arm.
        import os as _os
        if _os.environ.get("TEP_REMOVE_NETWORK_MEAN", "0") == "1":
            R = R - R.mean(1, keepdims=True)
        R = R - R.mean(0)                 # Remove time mean
        sd = R.std(0); sd[sd < 1e-9] = np.nan
        with np.errstate(all="ignore"):
            C = (R.T @ R) / len(R) / np.outer(sd, sd)
        for i, j in itertools.combinations(range(len(names)), 2):
            if not np.isfinite(C[i, j]):
                continue
            a, b = s[names[i]]["station"], s[names[j]]["station"]
            recs.append((td.great_circle_km(a, b), float(C[i, j]),
                         td.baseline_azimuth_deg(a, b), abs(a.lon_deg - b.lon_deg),
                         names[i], names[j]))
    return recs


def lam_for(recs, dmax):
    d = np.array([r[0] for r in recs]); c = np.array([r[1] for r in recs])
    ctr, mean, sem, cnt = ts.bin_by_distance(d, c, n_bins=22, dmin=100.0, dmax=dmax, min_count=8)
    f = ts.fit_correlation_length(ctr, mean, weight=1.0 / np.clip(sem, 1e-4, None) ** 2)
    return f


def ratio_stat(recs, dmax, lon_match=None):
    ew = [r for r in recs if td.is_ew(r[2]) and (lon_match is None or r[3] < lon_match)]
    ns = [r for r in recs if td.is_ns(r[2]) and (lon_match is None or r[3] < lon_match)]
    if len(ew) < 20 or len(ns) < 20:
        return np.nan, np.nan, np.nan, False
    fe, fn = lam_for(ew, dmax), lam_for(ns, dmax)
    if not (np.isfinite(fe["lam"]) and np.isfinite(fn["lam"]) and fn["lam"] > 1):
        return np.nan, fe["lam"], fn["lam"], False
    # Flag non-identifiable ratio when either lambda hits the optimization bound
    at_bound = fe.get("lam_at_upper_bound", False) or fn.get("lam_at_upper_bound", False) \
        or fe.get("lam_at_lower_bound", False) or fn.get("lam_at_lower_bound", False)
    return fe["lam"] / fn["lam"], fe["lam"], fn["lam"], at_bound


def run(product, date_tags, dmax, label):
    days = [s for _, s in td.load_multi_day(product, date_tags) if len(s) >= 5]
    names = sorted({n for s in days for n in s})
    recs = pair_table(days)
    print(f"[{label}] {len(days)} days, {len(names)} stations, {len(recs):,} pair-days")

    out = {"label": label, "n_stations": len(names), "n_pair_days": len(recs)}
    for tag, lm in (("full_range", None), ("longitude_matched", 30.0)):
        r, le, ln, at_bound = ratio_stat(recs, dmax, lon_match=lm)

        def stat(sel):
            from collections import Counter
            counts = Counter(sel)
            # Weight each pair by the product of its endpoint multiplicities
            # to honour the with-replacement bootstrap structure.
            weighted = []
            for x in recs:
                ci, cj = counts.get(x[4], 0), counts.get(x[5], 0)
                if ci and cj:
                    weighted.extend([x] * (ci * cj))
            rr, _, _, _ = ratio_stat(weighted, dmax, lon_match=lm)
            return rr
        pt, lo, hi, samp = ts.station_clustered_bootstrap(stat, names, n_boot=300, rng=RNG)
        p_gt1 = float(np.mean(samp <= 1.0)) if len(samp) else np.nan
        out[tag] = {"ratio": float(r) if np.isfinite(r) else None,
                    "lam_ew": float(le) if np.isfinite(le) else None,
                    "lam_ns": float(ln) if np.isfinite(ln) else None,
                    "ci95": [float(lo) if np.isfinite(lo) else None,
                             float(hi) if np.isfinite(hi) else None],
                    "p_ratio_le_1": float(p_gt1) if np.isfinite(p_gt1) else None,
                    "lam_at_boundary": bool(at_bound),
                    "ratio_identifiable": bool(not at_bound and np.isfinite(r))}
        bound_tag = " [NON-IDENTIFIABLE: lambda at boundary]" if at_bound else ""
        print(f"  {tag:18s}: ratio {r:.2f}  (EW {le:.0f} / NS {ln:.0f} km)  "
              f"95% CI [{lo:.2f}, {hi:.2f}]  p(ratio<=1)={p_gt1:.3f}{bound_tag}")

    # distance strata (full range) with station-clustered bootstrap on the
    # direct EW-NS correlation difference.  This is the robust statistic: it
    # does not depend on an ill-conditioned exponential fit and is the most
    # direct test of whether EW baselines are more correlated than NS baselines
    # at the same distance scale.
    strata = {}
    for lo_d, hi_d in [(100, 500), (500, 1000), (1000, 3000), (3000, 20000)]:
        sub = [r for r in recs if lo_d <= r[0] < hi_d]
        ew_vals = [r[1] for r in sub if td.is_ew(r[2])]
        ns_vals = [r[1] for r in sub if td.is_ns(r[2])]
        ew = float(np.mean(ew_vals)) if ew_vals else np.nan
        ns = float(np.mean(ns_vals)) if ns_vals else np.nan
        diff = float(ew - ns) if np.isfinite(ew) and np.isfinite(ns) else np.nan

        # Station-clustered bootstrap on the EW-NS difference: resample
        # station names with replacement, recompute the EW and NS means over
        # the pairs whose BOTH endpoints are in the resampled set.
        def diff_stat(sel, _sub=sub, _names=names):
            from collections import Counter
            counts = Counter(sel)
            # Weight each pair by the product of its endpoint multiplicities.
            ew_s, ns_s = [], []
            for r in _sub:
                ci, cj = counts.get(r[4], 0), counts.get(r[5], 0)
                if ci and cj:
                    w = ci * cj
                    if td.is_ew(r[2]):
                        ew_s.extend([r[1]] * w)
                    if td.is_ns(r[2]):
                        ns_s.extend([r[1]] * w)
            if len(ew_s) < 5 or len(ns_s) < 5:
                return np.nan
            return float(np.mean(ew_s) - np.mean(ns_s))

        _, dlo, dhi, dsamp = ts.station_clustered_bootstrap(
            diff_stat, names, n_boot=500, rng=RNG)
        d_p = float(np.mean(np.array(dsamp) <= 0.0)) if len(dsamp) else np.nan
        strata[f"{lo_d}_{hi_d}km"] = {"corr_ew": ew, "corr_ns": ns,
                                      "ew_minus_ns": diff,
                                      "ew_minus_ns_ci95": [float(dlo), float(dhi)],
                                      "ew_minus_ns_p_le_0": d_p,
                                      "n_ew": len(ew_vals), "n_ns": len(ns_vals)}
        print(f"  {lo_d:5d}-{hi_d:5d} km: corr EW {ew:+.3f}  NS {ns:+.3f}  "
              f"diff {diff:+.3f}  CI [{dlo:+.3f}, {dhi:+.3f}]  p(diff<=0)={d_p:.3f}  "
              f"(n_ew={len(ew_vals)}, n_ns={len(ns_vals)})")
    out["distance_strata"] = strata

    lm = out["longitude_matched"]
    p500 = float(strata["500_1000km"]["ew_minus_ns"])
    p500_sig = bool(strata["500_1000km"].get("ew_minus_ns_p_le_0", 1.0) < 0.05)
    identifiable = bool(lm.get("ratio_identifiable", True))
    # Pre-registered T2 pass: longitude-matched lambda ratio > 1 with p < 0.05,
    # AND EW-NS correlation difference > 0 in the 500-1000 km stratum.
    ratio_pass = bool(identifiable and lm["ratio"] is not None
                      and float(lm["ratio"]) > 1.0
                      and lm["p_ratio_le_1"] is not None
                      and float(lm["p_ratio_le_1"]) < 0.05)
    stratum_pass = bool(p500 > 0 and p500_sig)
    passes = bool(ratio_pass and stratum_pass)
    out["T2_pass"] = passes
    out["T2_ratio_pass"] = ratio_pass
    out["T2_stratum_pass"] = stratum_pass
    if not identifiable:
        out["T2_note"] = ("Ratio is non-identifiable: one or both correlation "
                          "lengths hit the optimization boundary. The lambda-ratio "
                          "statistic is ill-conditioned at this aperture; the direct "
                          "EW-NS correlation difference in the distance strata above "
                          "is the robust alternative.")
    print(f"  T2 pass: {passes}  (ratio: {ratio_pass}, stratum: {stratum_pass})"
          + ("" if identifiable else " [non-identifiable]"))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "test": "T2_ew_ns_anisotropy",
        "global_broadcast": run("broadcast", [f"2024{d:03d}" for d in range(100, 107)], 17000.0, "global_broadcast"),
        "timestamp": datetime.now().isoformat(),
    }
    # Global analysis-centre network -- the station count and orientation
    # diversity that a single-region network cannot supply (T2 has been
    # sign-only, not significant, for exactly that reason).  step_09 downloads
    # ~128 globally distributed stations x ~360 days; the anisotropy is driven
    # by station geometry, not day count, so the full year is used to maximise
    # the pair count.  The station-clustered bootstrap uses a threading backend
    # to avoid pickling the large pair table per worker.
    clk_days = td.available_dates("clk_cod", 2024)
    res["T2_pass_global"] = None
    if len(clk_days) >= 60:
        res["global_precise_clk"] = run(
            "clk_cod", clk_days, 20000.0, "global_precise_clk")
        res["T2_pass_global"] = res["global_precise_clk"]["T2_pass"]
    res["T2_pass"] = res["global_broadcast"]["T2_pass"]
    (OUT / "step_11_anisotropy.json").write_text(json.dumps(res, indent=2))
    print(f"\nT2 pass: {res['T2_pass']}\nsaved {OUT / 'step_11_anisotropy.json'}")


if __name__ == "__main__":
    main()
