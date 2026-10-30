#!/usr/bin/env python3
"""
STEP 41 -- Global anisotropy test (T2) on the IGS/MGEX precise receiver-clock network.

The step_27 D3 anisotropy test runs on the 30-station European dense network,
which has limited orientation diversity (all baselines within a single region).
This step runs the same EW/NS distance-matched anisotropy test on the global
IGS/MGEX precise receiver-clock product (COD, ~160 stations, baselines to
20,000 km), where the orientation diversity is far greater: baselines span
all azimuths (N-S, E-W, NE-SW, NW-SE) and all distances up to 20,000 km.

A proper-time field with a preferred axis (the TEP prediction) should produce
an EW-NS anisotropy that grows with aperture as more orientations are sampled.
If the anisotropy is significant on the global network but not on the regional
network, that is direct evidence that the directional signature is real but
below the noise floor at small aperture -- not absent.
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
RNG = np.random.default_rng(20260904)
C_M_PER_NS = 0.299792458


def _clean_clk(v, tt):
    """Clean a receiver clock series: detrend, clip, band-pass, interpolate."""
    n = min(len(v), len(tt))
    v, tt = np.asarray(v, float)[:n], np.asarray(tt, float)[:n]
    if np.mean(np.isfinite(v)) < 0.9:
        return None
    g = np.isfinite(v)
    v = np.interp(np.arange(n), np.flatnonzero(g), v[g])
    r = v - np.polyval(np.polyfit(tt, v, 1), tt)
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    dt = float(np.nanmedian(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 30.0
    b = ts.bandpass(np.clip(r, -8 * s, 8 * s), dt) * C_M_PER_NS
    return b


def corr_matrix(series_list):
    """Compute correlation matrix from a list of cleaned clock series."""
    M = np.array(series_list)
    n = M.shape[0]
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = M[i], M[j]
            g = np.isfinite(a) & np.isfinite(b)
            if g.sum() < 100:
                continue
            c = np.corrcoef(a[g], b[g])[0, 1]
            C[i, j] = c
            C[j, i] = c
    return C


def load_global_day(day, product="clk_cod"):
    """Load a single day of global precise receiver clocks.

    Returns (cleaned_dict, station_dict) where cleaned_dict maps name->series
    and station_dict maps name->Station.
    """
    loaded = list(td.load_multi_day(product, [day]))
    if not loaded:
        return {}, {}
    _, series = loaded[0]
    cleaned = {}
    stations = {}
    for nm, v in series.items():
        c = _clean_clk(v["resid"], v["t"])
        if c is not None:
            cleaned[nm] = c
            stations[nm] = v["station"]
    return cleaned, stations


def global_anisotropy(days, product="clk_cod", half_width_deg=25.0,
                      bin_km=500.0, n_boot=400, dmax=19000.0, min_n=30):
    """EW/NS distance-matched anisotropy on the global precise network."""
    recs = []
    nd = 0
    for day in days:
        D, S = load_global_day(day, product)
        if len(D) < 20:
            continue
        nd += 1
        nms = list(D)
        Cm = corr_matrix([D[n] for n in nms])
        for i, j in itertools.combinations(range(len(nms)), 2):
            if not np.isfinite(Cm[i, j]):
                continue
            a, b = nms[i], nms[j]
            sa, sb = S[a], S[b]
            if sa is None or sb is None:
                continue
            dist = td.great_circle_km(sa, sb)
            if dist > dmax or dist < 100:
                continue
            az = td.baseline_azimuth_deg(sa, sb)
            recs.append((dist, float(Cm[i, j]), az, a, b))

    if len(recs) == 0:
        print(f"\n[Global T2] {nd} days, 0 pair-days -- insufficient data")
        return {"n_days": nd, "n_pair_days": 0, "n_stations": 0,
                "n_ew_pairs": 0, "n_ns_pairs": 0,
                "ew_minus_ns": np.nan, "ci95": [np.nan, np.nan],
                "p_ew_le_ns": np.nan,
                "sign_positive": False, "significant": False,
                "limiting_factor": "no pairs survived QC"}

    d = np.array([r[0] for r in recs])
    c = np.array([r[1] for r in recs])
    az = np.array([r[2] for r in recs])
    bins = np.arange(500, dmax + bin_km, bin_km)

    def matched_diff(dd, cc, aa, mn):
        e = np.abs(aa - 90) <= half_width_deg
        n_ = (aa <= half_width_deg) | (aa >= 180 - half_width_deg)
        sub, w = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            a1 = e & (dd >= lo) & (dd < hi)
            a2 = n_ & (dd >= lo) & (dd < hi)
            if a1.sum() < mn or a2.sum() < mn:
                continue
            sub.append(cc[a1].mean() - cc[a2].mean())
            w.append(min(a1.sum(), a2.sum()))
        return (float(np.average(sub, weights=w)) if sub else np.nan), len(sub)

    point, nbins = matched_diff(d, c, az, min_n)
    stn = sorted({r[3] for r in recs} | {r[4] for r in recs})
    bs = []
    for _ in range(n_boot):
        draws = RNG.choice(stn, len(stn), replace=True)
        draw_count = {}
        for s in draws:
            draw_count[s] = draw_count.get(s, 0) + 1
        idx_list = []
        for k, r in enumerate(recs):
            s1, s2 = r[3], r[4]
            if s1 in draw_count and s2 in draw_count:
                idx_list.extend([k] * (draw_count[s1] * draw_count[s2]))
        if len(idx_list) < 2000:
            continue
        idx_arr = np.array(idx_list)
        v, _ = matched_diff(d[idx_arr], c[idx_arr], az[idx_arr], min_n)
        if np.isfinite(v):
            bs.append(v)
    bs = np.array(bs)
    if len(bs) == 0:
        lo, hi, p = np.nan, np.nan, np.nan
    else:
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        p = float(np.mean(bs <= 0))
    ew_n = int((np.abs(az - 90) <= half_width_deg).sum())
    ns_n = int(((az <= half_width_deg) | (az >= 180 - half_width_deg)).sum())
    print(f"\n[Global T2] {nd} days, {len(recs):,} pair-days  (EW {ew_n:,} / NS {ns_n:,})")
    print(f"  {len(stn)} stations, baselines to {d.max():.0f} km")
    print(f"  distance-matched EW-NS = {point:+.4f} over {nbins} bins")
    print(f"  station-clustered 95% CI [{lo:+.4f}, {hi:+.4f}]   p(EW<=NS) = {p:.3f}")
    print(f"  sign is {'POSITIVE (EW>NS, as the series reports)' if point > 0 else 'negative'}; "
          f"significant: {lo > 0}")
    return {"n_days": nd, "n_pair_days": len(recs), "n_stations": len(stn),
            "n_ew_pairs": ew_n, "n_ns_pairs": ns_n,
            "ew_minus_ns": point, "ci95": [lo, hi], "p_ew_le_ns": p,
            "sign_positive": bool(point > 0), "significant": bool(lo > 0),
            "dmax_km": float(d.max()),
            "limiting_factor": "none" if lo > 0 else "orientation diversity or station count"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cod_days = td.available_dates("clk_cod", 2024)
    # Use a 30-day evenly-spaced subset (same as step_14 T1)
    idx = np.linspace(0, len(cod_days) - 1, 30).round().astype(int)
    days = [cod_days[i] for i in idx]
    print(f"Global anisotropy test on COD precise receiver clocks ({len(days)} days)\n")
    res = {
        "test": "global_anisotropy_t2",
        "product": "COD0MGXFIN precise receiver clocks (RINEX CLK AR)",
        "T2_global": global_anisotropy(days, "clk_cod"),
        "timestamp": datetime.now().isoformat(),
    }
    # Also run on aperture-restricted subsets to see if anisotropy grows
    print("\n--- Aperture-restricted subsets ---")
    for dmax in [5000, 10000, 19000]:
        print(f"\n  dmax = {dmax} km:")
        res[f"T2_global_dmax{dmax}"] = global_anisotropy(
            days, "clk_cod", dmax=float(dmax))

    outpath = OUT / "step_41_global_anisotropy.json"
    outpath.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved {outpath}")

    # Summary
    t2g = res["T2_global"]
    print("\n" + "=" * 72)
    print("GLOBAL T2 ANISOTROPY RESULT")
    print("=" * 72)
    if np.isfinite(t2g["ew_minus_ns"]):
        sig = "SIGNIFICANT" if t2g["significant"] else "not significant"
        print(f"  EW-NS = {t2g['ew_minus_ns']:+.4f}  CI [{t2g['ci95'][0]:+.4f}, {t2g['ci95'][1]:+.4f}]  {sig}")
        print(f"  {t2g['n_stations']} stations, {t2g['n_pair_days']:,} pair-days, {t2g['n_days']} days")
    else:
        print("  Insufficient data")


if __name__ == "__main__":
    main()
