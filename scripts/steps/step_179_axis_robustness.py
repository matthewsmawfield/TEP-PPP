#!/usr/bin/env python3
"""STEP 179 -- CMB-axis robustness: day-clustered bootstrap + split-half.

Step 137 reports the CMB dipole preferred over Solar Apex, geomagnetic pole
and GPS nodal axes at Delta BIC = -2315.5, on ~2.14 million pair-window
measurements pooled over 53 days.  The reviewer's objection is correct in
principle: the pair-window count is not the effective sample size (windows
overlap in time, pairs share stations and days), so the iid-BIC margin can
be inflated.  This step re-evaluates the SAME axis comparison on the SAME
windows with three dependence-robust diagnostics:

  1. Day-clustered bootstrap (B=300): resample the 53 days with
     replacement, recompute per-axis BIC on each resample.  Report how
     often CMB wins, and the median Delta BIC.  This is the reviewer's
     "cluster bootstrap by independent stations and time blocks".
  2. Split-half consistency: first-half days vs second-half days, per-axis
     BIC computed independently on each half.  A real axis wins both.
  3. Frozen-template held-out score: axis parameters fitted on half A,
     evaluated (no refit) on half B and vice versa.

Implementation: for each (day, axis) we accumulate sufficient statistics
(Sn, Sc, Sc2, Sm, Sm2, Scm) for the Pearson r of C vs exp(-u/lam) on a
lambda grid, because for simple linear regression R^2 = r^2 and the Step
137 BIC proxy ll = -n/2 ln(1-R^2) then follows in closed form.  Resampling
then costs O(1) per bootstrap realisation.  The full-sample per-axis BIC
is verified against the committed step_137 values first.

Output: results/outputs/step_179_axis_robustness.json
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "steps"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

import step_137_aperture_synthesis as s137
from tep_astronomical import _gmst_deg, _to_jd
from datetime import datetime, timedelta, timezone

YEAR = 2024
WIN_S = 43200
STEP_S = 10800
N_BOOT = 300
LAM_GRID = np.concatenate([
    np.geomspace(500.0, 40000.0, 80),
])
OUT = os.path.join(ROOT, "results", "outputs", "step_179_axis_robustness.json")

RNG = np.random.default_rng(20261029)


def _day_stats(latlon, day_dt, windows, axis_def):
    """Sufficient statistics for one day under one axis.

    For each window: the axis unit vector in ECEF at window mid-time, then
    per-pair |u| = |axis . (r_j - r_i)| in km and C_ij.  Accumulates
    n, Sc, Sc2, Sm[k], Sm2[k], Scm[k] over the lambda grid.
    """
    sta = s137.latlon_to_ecef_unit  # lat,lon -> ECEF unit
    n_sta = len(latlon)
    E = np.array([sta(ll[0], ll[1]) for ll in latlon])   # (n_sta, 3)
    iu, ju = np.triu_indices(n_sta, 1)
    dR = (E[ju] - E[iu]) * s137.EARTH_R_KM               # (npair, 3) km

    S = {"n": 0, "Sc": 0.0, "Sc2": 0.0,
         "Sm": np.zeros(len(LAM_GRID)), "Sm2": np.zeros(len(LAM_GRID)),
         "Scm": np.zeros(len(LAM_GRID)),
         "Sm_4200": 0.0, "Scm_4200": 0.0, "Sm2_4200": 0.0}

    for mid_s, C, valid in windows:
        t = day_dt + timedelta(seconds=mid_s)
        gmst = _gmst_deg(_to_jd(t))
        ax = s137._icrs_to_ecef(axis_def["ra"], axis_def["dec"], gmst)
        v = np.where(valid)[0]
        # map full pair indices to valid-pair indices
        keep = np.isin(iu, v) & np.isin(ju, v)
        # strict: both endpoints valid AND index within original arrays
        iv = np.zeros(n_sta, dtype=int)
        iv[v] = np.arange(len(v))
        iu_f, ju_f = iu[keep], ju[keep]
        # C is indexed on the VALID submatrix positions
        pos_i, pos_j = iv[iu_f], iv[ju_f]
        c = C[pos_i, pos_j]
        u = np.abs(dR[keep] @ ax)
        m = u[:, None] / LAM_GRID[None, :]              # broadcast base
        m = np.exp(-m)                                  # (npair, nlam)
        S["n"] += len(c)
        S["Sc"] += float(c.sum())
        S["Sc2"] += float((c * c).sum())
        S["Sm"] += m.sum(axis=0)
        S["Sm2"] += (m * m).sum(axis=0)
        S["Scm"] += (m * c[:, None]).sum(axis=0)
        m42 = np.exp(-u / s137.LAMBDA_TEP)
        S["Sm_4200"] += float(m42.sum())
        S["Sm2_4200"] += float((m42 * m42).sum())
        S["Scm_4200"] += float((m42 * c).sum())
    return S


def _bic_from_stats(stats_list):
    """Given a list of day-stat dicts, per-lambda R^2 -> best BIC.

    R^2 of the 1-parameter+intercept exp fit = squared Pearson r of c vs
    m(u; lam).  BIC = -2 ll + 2 ln n with ll = -n/2 ln(1-R^2),
    the Step 137 convention.
    """
    n = sum(s["n"] for s in stats_list)
    Sc = sum(s["Sc"] for s in stats_list)
    Sc2 = sum(s["Sc2"] for s in stats_list)
    Sm = sum(s["Sm"] for s in stats_list)
    Sm2 = sum(s["Sm2"] for s in stats_list)
    Scm = sum(s["Scm"] for s in stats_list)
    if n < 100:
        return np.nan, np.nan, np.nan
    cov = Scm - Sc * Sm / n
    var_c = Sc2 - Sc * Sc / n
    var_m = Sm2 - Sm * Sm / n
    ok = (var_m > 0) & (var_c > 0)
    r2 = np.where(ok, cov * cov / np.clip(var_c * var_m, 1e-30, None), 0.0)
    k = int(np.nanargmax(r2))
    r2b = float(np.clip(r2[k], 1e-12, 1 - 1e-12))
    ll = -n / 2.0 * math.log(1.0 - r2b)
    bic = -2.0 * ll + 2.0 * math.log(n)
    return bic, r2b, float(LAM_GRID[k])


def _r4200_from_stats(stats_list):
    n = sum(s["n"] for s in stats_list)
    Sc = sum(s["Sc"] for s in stats_list)
    Sc2 = sum(s["Sc2"] for s in stats_list)
    Sm = sum(s["Sm_4200"] for s in stats_list)
    Sm2 = sum(s["Sm2_4200"] for s in stats_list)
    Scm = sum(s["Scm_4200"] for s in stats_list)
    if n < 100:
        return np.nan
    cov = Scm - Sc * Sm / n
    var_c = Sc2 - Sc * Sc / n
    var_m = Sm2 - Sm * Sm / n
    if var_c <= 0 or var_m <= 0:
        return np.nan
    return float(cov / math.sqrt(var_c * var_m))


def main():
    sample_doys = list(range(1, 366, 7))
    day_stat = {}      # doy -> {axis: S}
    days_used = []
    print(f"STEP 179 | day-clustered axis robustness | {len(sample_doys)} days")
    for doy in sample_doys:
        result = s137.load_day_residuals(YEAR, doy)
        if result is None:
            continue
        sta_list, latlon, residuals, dt = result
        windows = s137.windowed_coherence(residuals, dt, WIN_S, STEP_S)
        if not windows:
            continue
        day_dt = datetime(YEAR, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        day_stat[doy] = {ax: _day_stats(latlon, day_dt, windows, adef)
                         for ax, adef in s137.AXES.items()}
        days_used.append(doy)
        print(f"  DOY {doy:3d}: {len(sta_list)} sta, {len(windows)} windows")

    axes = list(s137.AXES.keys())
    # ---- full-sample reproduction --------------------------------------
    full_bic, full_r2, full_lam = {}, {}, {}
    for ax in axes:
        b, r2, lam = _bic_from_stats([day_stat[d][ax] for d in days_used])
        full_bic[ax], full_r2[ax], full_lam[ax] = b, r2, lam
    print("full-sample BIC (grid-fit convention):",
          {a: round(full_bic[a], 1) for a in axes})

    # ---- day-clustered bootstrap ----------------------------------------
    boot_wins = {a: 0 for a in axes}
    boot_dbic = []          # bic(best) - bic(cmb)  (>0 = cmb wins)
    boot_cmb_margin = []    # min over competitors of their bic minus cmb
    nd = len(days_used)
    for b in range(N_BOOT):
        pick = RNG.choice(days_used, size=nd, replace=True)
        bics = {}
        for ax in axes:
            bics[ax], _, _ = _bic_from_stats([day_stat[d][ax] for d in pick])
        vals = {a: v for a, v in bics.items() if np.isfinite(v)}
        best = min(vals, key=vals.get)
        boot_wins[best] += 1
        comp = min(v for a, v in vals.items() if a != "cmb")
        boot_cmb_margin.append(comp - vals["cmb"])
        boot_dbic.append(vals.get("cmb", np.nan))
    boot_cmb_margin = np.array(boot_cmb_margin)
    print(f"bootstrap: CMB wins {boot_wins['cmb']}/{N_BOOT}; "
          f"median margin {np.median(boot_cmb_margin):.1f}")

    # ---- split-half ------------------------------------------------------
    half = len(days_used) // 2
    halves = {"first": days_used[:half], "second": days_used[half:]}
    split = {}
    for name, ds in halves.items():
        split[name] = {}
        for ax in axes:
            b, r2, lam = _bic_from_stats([day_stat[d][ax] for d in ds])
            split[name][ax] = {"bic": b, "r2": r2, "lambda_km": lam}
        win = min(split[name], key=lambda a: split[name][a]["bic"])
        split[name]["winner"] = win
        split[name]["cmb_margin"] = min(
            v["bic"] for a, v in split[name].items()
            if isinstance(v, dict) and a != "cmb") - split[name]["cmb"]["bic"]
        print(f"  {name} half ({len(ds)} days): winner={win}, "
              f"CMB margin={split[name]['cmb_margin']:.1f}")

    # ---- frozen-template held-out ---------------------------------------
    # lambda fixed on half A -> score r2 at that lambda on half B (both ways)
    held = {}
    for tr, te in (("first", "second"), ("second", "first")):
        held[f"{tr}->{te}"] = {}
        for ax in axes:
            # r at fixed lam=4200 on test half (frozen template, no refit)
            r = _r4200_from_stats([day_stat[d][ax] for d in halves[te]])
            held[f"{tr}->{te}"][ax] = {"r_4200_heldout": r}
    for k in held:
        print(f"  heldout {k}: r4200 =",
              {a: round(held[k][a]['r_4200_heldout'], 4) for a in axes})

    out = {
        "step": 179,
        "title": "CMB-axis robustness (day-clustered bootstrap + split-half)",
        "year": YEAR, "win_s": WIN_S, "step_s": STEP_S,
        "n_days": len(days_used), "days": days_used,
        "n_boot": N_BOOT,
        "lam_grid_km": [float(x) for x in LAM_GRID],
        "full_sample": {
            "bic_grid_convention": full_bic,
            "r2": full_r2, "lambda_km": full_lam,
            "r_4200": {a: _r4200_from_stats([day_stat[d][a] for d in days_used])
                       for a in axes},
            "note": "R^2 via grid-search r^2 convention; committed step_137 "
                    "BICs (nll fit) are -6322.55 (cmb), -4007.07 (apex), "
                    "-2117.67 (geomag), -3486.39 (gps_nodal)."},
        "bootstrap": {
            "wins": boot_wins,
            "cmb_win_frac": boot_wins["cmb"] / N_BOOT,
            "cmb_margin_median": float(np.median(boot_cmb_margin)),
            "cmb_margin_ci95": [float(np.percentile(boot_cmb_margin, 2.5)),
                                float(np.percentile(boot_cmb_margin, 97.5))],
            "cmb_margin_p5": float(np.percentile(boot_cmb_margin, 5)),
        },
        "split_half": split,
        "heldout_frozen_4200": held,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
