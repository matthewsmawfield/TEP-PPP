#!/usr/bin/env python3
"""
STEP 73  --  Reproducing the Paper 2 EW/NS anisotropy with the CORRECTED
             (phase-cos) observable.

Paper 2 (TEP-GNSS-II, 25.3 years, 165,189,605 station-pair observations) reports
the corpus's strongest secondary signature -- and its methods section is
explicit about how (Section "Spatial anisotropy"):

    "spatial anisotropy was tested by fitting exponential decay models
     C(d) = A*exp(-d/lambda) + B to station pairs BINNED BY AZIMUTH ...
     pronounced directional variation in correlation DECAY LENGTHS"
    "E-W oriented sectors consistently show longer lambda_T than N-S oriented"

so the quoted ratio 2.16 is lambda_EW / lambda_NS -- a ratio of fitted
correlation LENGTHS, not of amplitudes.  Paper 2 obtained it from band-averaged
coherence.  This paper's Step 53 establishes that the correct Papers 1-2
estimator is the phase-alignment index

    cos(arg(sum_f S_ij(f)))            10-500 uHz, linear detrend only

and that band-averaged coherence is the statistic Paper 1 reports as failing.
This step therefore applies PAPER 2's GEOMETRY (azimuth sectors x log distance
bins x exponential fit to bin means) to PAPER 53's OBSERVABLE (phase-cos), which
is the like-for-like corpus comparison that has been missing.

Construction, all inherited from Step 53 so the statistic cannot drift:
  * _spectra : per-station in-band complex spectrum, linear detrend
  * _curve   : 40 log-spaced distance bins, 50-13,000 km, count-weighted means
  * _fit     : AIC model selection; the exponential lambda is reported
               separately for direct comparability with Paper 2's 2.16
Azimuth is folded to [0,180) because a station pair has an orientation but not
a direction, then split into 8 sectors of 22.5 deg.

Nulls: azimuth shuffle (permute sector labels, preserving the joint
distance/phase-alignment distribution) and a day-clustered bootstrap on the
ratio, since the same pair recurs every day and a pair-level bootstrap would
badly overstate significance.

Output: results/outputs/step_73_phase_anisotropy.json
"""
from __future__ import annotations

import os as _os
import sys
import json
import importlib.util
import itertools
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_73_phase_anisotropy.json"
CACHE = ROOT / "results" / "outputs" / f"step_73_cache_{_os.environ.get('TEP_ANISO_DAYS','120')}.npz"
RNG = np.random.default_rng(20260903)

PRODUCT = "clk_cod"
N_DAYS = int(_os.environ.get("TEP_ANISO_DAYS", "120"))
N_BOOT = 300
N_NULL = 200
SECTORS = [(0.0, 22.5, "N/S"), (22.5, 45.0, "NNE"), (45.0, 67.5, "NE/SW"),
           (67.5, 90.0, "ENE"), (90.0, 112.5, "E/W"), (112.5, 135.0, "ESE"),
           (135.0, 157.5, "SE/NW"), (157.5, 180.0, "SSE")]
EW_RANGE = (67.5, 112.5)              # E-W oriented
NS_RANGE = [(0.0, 22.5), (157.5, 180.0)]   # N-S oriented


def bearing_initial(a, b):
    """Bearing at station A.  BIASED for this purpose: a great circle between two
    mid-latitude stations bends poleward, so a true east-west 6,000 km pair can
    leave A on a bearing tens of degrees off 90.  Measured on this network the
    initial-vs-midpoint bearing differs by 20.5 deg on average and reclassifies
    35.8% of pairs, with the error growing from 2 deg below 1,000 km to 26 deg
    beyond 10,000 km -- i.e. it strips long EW pairs out of the EW sector exactly
    where the decay tail determines lambda.  Retained only for comparison."""
    la1, lo1 = np.radians(a.lat_deg), np.radians(a.lon_deg)
    la2, lo2 = np.radians(b.lat_deg), np.radians(b.lon_deg)
    dlo = lo2 - lo1
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return float(np.degrees(np.arctan2(y, x)) % 180.0)


def bearing_deg(a, b):
    """Orientation at the great-circle MIDPOINT: the symmetric, unbiased measure
    of whether a baseline is east-west or north-south."""
    la1, lo1 = np.radians(a.lat_deg), np.radians(a.lon_deg)
    la2, lo2 = np.radians(b.lat_deg), np.radians(b.lon_deg)
    Bx = np.cos(la2) * np.cos(lo2 - lo1)
    By = np.cos(la2) * np.sin(lo2 - lo1)
    lam = np.arctan2(np.sin(la1) + np.sin(la2),
                     np.sqrt((np.cos(la1) + Bx) ** 2 + By ** 2))
    lom = lo1 + np.arctan2(By, np.cos(la1) + Bx)
    dlo = lo2 - lom
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(lam) * np.sin(la2) - np.sin(lam) * np.cos(la2) * np.cos(dlo)
    return float(np.degrees(np.arctan2(y, x)) % 180.0)


def collect(dates):
    rows = []
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        if len(RF) < 10:
            continue
        nm = list(RF)
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            d = td.great_circle_km(A, B)
            if not (s53.DMIN_KM <= d <= s53.DMAX_KM):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((k, d, bearing_deg(A, B), float(np.cos(np.angle(S))),
                         bearing_initial(A, B)))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(dates)} days, {len(rows):,} pair-days", flush=True)
    return rows


def in_ew(az):
    return (az >= EW_RANGE[0]) & (az < EW_RANGE[1])


def in_ns(az):
    return ((az >= NS_RANGE[0][0]) & (az < NS_RANGE[0][1])) | \
           ((az >= NS_RANGE[1][0]) & (az <= NS_RANGE[1][1]))


def lam_of(d, v, want_exponential=True):
    """Paper 2 geometry: distance-bin the pairs, fit decay to the bin means."""
    ctr, val, cnt = s53._curve(list(zip(d, v)))
    if len(ctr) < 6:
        return None, None, None
    f = s53._fit(ctr, val, cnt)
    if not f:
        return None, None, None
    exp_lam = (f.get("exponential") or {}).get("lambda_km")
    return f.get("lambda_km"), exp_lam, f


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        z = np.load(CACHE)
        day, dist, az, val = z["day"], z["dist"], z["az"], z["val"]
        az0 = z["az_initial"] if "az_initial" in z.files else None
        print(f"loaded cache: {len(val):,} pair-days")
    else:
        dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
        print(f"STEP 73  |  {PRODUCT}, {len(dates)} days, phase-cos + Paper 2 binning")
        rows = collect(dates)
        if len(rows) < 5000:
            OUT.write_text(json.dumps({"step": 73, "status": "insufficient"}, indent=2)); return
        day = np.array([r[0] for r in rows]); dist = np.array([r[1] for r in rows], float)
        az = np.array([r[2] for r in rows], float); val = np.array([r[3] for r in rows], float)
        az0 = np.array([r[4] for r in rows], float)
        np.savez_compressed(CACHE, day=day, dist=dist, az=az, val=val, az_initial=az0)
    print(f"{len(val):,} pair-days over {len(np.unique(day))} days\n")

    # full azimuthal profile
    profile = {}
    print("  sector      n_pair_days   lambda(best)  lambda(exp)   R2     best model")
    for lo, hi, name in SECTORS:
        m = (az >= lo) & (az < hi)
        if m.sum() < 2000:
            continue
        lb, le, f = lam_of(dist[m], val[m])
        profile[name] = {"az_range": [lo, hi], "n_pair_days": int(m.sum()),
                         "lambda_best_km": lb, "lambda_exponential_km": le,
                         "best_model": f.get("best_model") if f else None,
                         "R2": f.get("R2") if f else None}
        print("  %-9s %12d   %11s  %11s  %6s  %s" % (
            name, m.sum(),
            f"{lb:.0f}" if lb else "-", f"{le:.0f}" if le else "-",
            f"{f.get('R2'):.3f}" if f and f.get("R2") else "-",
            f.get("best_model") if f else "-"))

    mew, mns = in_ew(az), in_ns(az)
    lb_ew, le_ew, f_ew = lam_of(dist[mew], val[mew])
    lb_ns, le_ns, f_ns = lam_of(dist[mns], val[mns])
    ratio_best = (lb_ew / lb_ns) if (lb_ew and lb_ns) else None
    ratio_exp = (le_ew / le_ns) if (le_ew and le_ns) else None
    print(f"\n  EW  n={mew.sum():,}  lambda_best={lb_ew}  lambda_exp={le_ew}")
    print(f"  NS  n={mns.sum():,}  lambda_best={lb_ns}  lambda_exp={le_ns}")
    print(f"  ratio (best-model) = {ratio_best}   ratio (exponential) = {ratio_exp}")
    print(f"  Paper 2 reports lambda_EW/lambda_NS = 2.16")

    # day-clustered bootstrap on the ratio
    udays = np.unique(day); boot_b, boot_e = [], []
    for _ in range(N_BOOT):
        pick = RNG.choice(udays, len(udays), replace=True)
        idx = np.concatenate([np.where(day == d)[0] for d in pick])
        de, ve = dist[idx], val[idx]; ae = az[idx]
        b1, e1, _ = lam_of(de[in_ew(ae)], ve[in_ew(ae)])
        b2, e2, _ = lam_of(de[in_ns(ae)], ve[in_ns(ae)])
        if b1 and b2:
            boot_b.append(b1 / b2)
        if e1 and e2:
            boot_e.append(e1 / e2)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) > 20 else None

    # azimuth-shuffle null on the ratio
    nulls = []
    for _ in range(N_NULL):
        ap = RNG.permutation(az)
        _, e1, _ = lam_of(dist[in_ew(ap)], val[in_ew(ap)])
        _, e2, _ = lam_of(dist[in_ns(ap)], val[in_ns(ap)])
        if e1 and e2:
            nulls.append(e1 / e2)
    nulls = np.array(nulls)
    # null and z are on the EXPONENTIAL ratio: Paper 2 fits C(d)=A*exp(-d/lambda)+B,
    # and best-AIC selection can return a degenerate power-law "lambda" in one
    # sector, which makes a best-model ratio uninterpretable.
    z = float((ratio_exp - nulls.mean()) / (nulls.std() + 1e-30)) if (ratio_exp and len(nulls)) else None

    win = []
    udl = np.unique(day)
    for w0 in range(0, len(udl), 30):
        sel = np.isin(day, udl[w0:w0 + 30])
        if sel.sum() < 20000:
            continue
        _, e1, _ = lam_of(dist[sel & in_ew(az)], val[sel & in_ew(az)])
        _, e2, _ = lam_of(dist[sel & in_ns(az)], val[sel & in_ns(az)])
        if e1 and e2:
            win.append({"window_start_day": int(udl[w0]), "lambda_ew": e1,
                        "lambda_ns": e2, "ratio": e1 / e2})
    if win:
        wr = np.array([w["ratio"] for w in win])
        print(f"\n  30-day windows: n={len(win)}  ratio mean {wr.mean():.2f} "
              f"median {np.median(wr):.2f}  range [{wr.min():.2f}, {wr.max():.2f}]  "
              f"modulation {100*(wr.max()-wr.min())/wr.mean():.0f}%")

    # same contrast under the BIASED initial-bearing convention, to size the bug
    ratio_initial = None
    if az0 is not None:
        _, ei, _ = lam_of(dist[in_ew(az0)], val[in_ew(az0)])
        _, en, _ = lam_of(dist[in_ns(az0)], val[in_ns(az0)])
        if ei and en:
            ratio_initial = ei / en
            print(f"  same data, initial-bearing sectors (biased): ratio = {ratio_initial:.3f}")

    verdict = {
        "product": PRODUCT,
        "azimuth_convention": "great-circle midpoint bearing (initial-bearing is biased: "
                              "20.5 deg mean error, reclassifies 35.8% of pairs, error grows "
                              "with baseline and strips long EW pairs from the EW sector)",
        "ratio_exponential_initial_bearing_biased": ratio_initial,
        "windows_30day": win,
        "window_ratio_mean": float(np.mean([w["ratio"] for w in win])) if win else None,
        "window_ratio_median": float(np.median([w["ratio"] for w in win])) if win else None, "n_days": int(len(np.unique(day))),
        "n_pair_days": int(len(val)), "n_ew": int(mew.sum()), "n_ns": int(mns.sum()),
        "method": "Paper 2 geometry (azimuth sector x log distance bins x decay fit to bin means) "
                  "applied to the Step 53 phase-cos observable",
        "lambda_ew_km": lb_ew, "lambda_ns_km": lb_ns,
        "lambda_ew_exponential_km": le_ew, "lambda_ns_exponential_km": le_ns,
        "ratio_best_model": ratio_best, "ratio_exponential": ratio_exp,
        "ratio_ci95_day_clustered": ci(np.array(boot_b)),
        "ratio_exponential_ci95_day_clustered": ci(np.array(boot_e)),
        "azimuth_shuffle_z": z,
        "azimuth_shuffle_mean": float(nulls.mean()) if len(nulls) else None,
        "paper2_reported_ratio": 2.16,
        "ew_exceeds_ns": bool(ratio_best and ratio_best > 1.0),
        "primary_statistic": "ratio_exponential (Paper 2 fits an exponential decay)",
        "ew_exceeds_ns_exponential": bool(ratio_exp and ratio_exp > 1.0),
        "consistent_with_paper2": bool(
            ratio_exp and ci(np.array(boot_e)) and ci(np.array(boot_e))[0] > 1.0),
        "matches_paper2_magnitude": bool(
            ratio_exp and ci(np.array(boot_e)) and
            ci(np.array(boot_e))[0] <= 2.16 <= ci(np.array(boot_e))[1]),
    }
    OUT.write_text(json.dumps({
        "step": 73,
        "title": "EW/NS correlation-length anisotropy in the phase-alignment index",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "azimuth_profile": profile, "verdict": verdict}, indent=2))
    print(f"\n  ratio CI95 (day-clustered) = {verdict['ratio_ci95_day_clustered']}")
    print(f"  azimuth-shuffle z = {z}")
    print(f"  EW > NS: {verdict['ew_exceeds_ns']}   consistent with Paper 2: {verdict['consistent_with_paper2']}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
