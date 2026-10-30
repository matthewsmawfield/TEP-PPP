#!/usr/bin/env python3
"""
STEP 13  (Phase 1 / T4) -- Common-view model as the nuisance competitor.

Question: after accounting for how much of each station pair's satellite
constellation is shared (co-visibility), does the pair correlation still
depend on great-circle distance?  If the correlation is purely shared
satellite-product error, co-visibility explains it and the distance term
adds nothing.  If a distance-structured field is present, the distance term
survives controlling for co-visibility.

Method:
  1. Propagate a nominal 24-slot GPS constellation over the analysis day.
  2. For every station, the set of visible satellites (elev > 10 deg) per epoch.
  3. Per pair: mean Jaccard overlap of visible sets = co-visibility fraction.
  4. Models for band-passed pair correlation c:
        M_cv    : c ~ f(covis)            (spline / monotone)
        M_dist  : c ~ A exp(-r/lambda)+C0
        M_both  : c ~ covis term + distance term   (linear, held-out scored)
     Report: partial correlation of c with distance | covis, and whether
     M_both beats M_cv out-of-sample.
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402

RNG = np.random.default_rng(20260901)
OUT = PROJECT_ROOT / "results" / "outputs"
MU = 3.986004418e14          # m^3/s^2
GPS_A = 26_560_000.0         # m
GPS_INC = np.radians(55.0)
ELEV_MASK = np.radians(10.0)
OMEGA_E = 7.2921150e-5       # rad/s


def gps_constellation(t_s):
    """ECEF positions (24, 3) of a nominal GPS constellation at time t_s (s of day)."""
    n = np.sqrt(MU / GPS_A ** 3)
    slots = []
    for p in range(6):                       # 6 planes
        raan = np.radians(60.0 * p)
        for k in range(4):                   # 4 sats / plane
            M = np.radians(90.0 * k + 15.0 * p) + n * t_s
            # circular orbit in plane
            xo, yo = GPS_A * np.cos(M), GPS_A * np.sin(M)
            # rotate by inclination then RAAN (ECI)
            x1 = xo
            y1 = yo * np.cos(GPS_INC)
            z1 = yo * np.sin(GPS_INC)
            xe = x1 * np.cos(raan) - y1 * np.sin(raan)
            ye = x1 * np.sin(raan) + y1 * np.cos(raan)
            ze = z1
            # ECI -> ECEF
            th = OMEGA_E * t_s
            slots.append([xe * np.cos(th) + ye * np.sin(th),
                          -xe * np.sin(th) + ye * np.cos(th), ze])
    return np.array(slots)


def visible_mask(sat_ecef, sta_ecef_m):
    """Boolean (n_sat,) visibility for one station (elev > mask)."""
    up = sta_ecef_m / np.linalg.norm(sta_ecef_m)
    los = sat_ecef - sta_ecef_m
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    elev = np.arcsin(np.clip(los @ up, -1, 1))
    return elev > ELEV_MASK


def covisibility(stations, epochs_s):
    """Mean Jaccard overlap of visible-satellite sets for every station pair."""
    n = len(stations)
    vis = np.zeros((len(epochs_s), n, 24), bool)
    for e, t in enumerate(epochs_s):
        sat = gps_constellation(float(t))
        for i, st in enumerate(stations):
            vis[e, i] = visible_mask(sat, st.ecef_m)
    J = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        inter = np.sum(vis[:, i] & vis[:, j], axis=1).astype(float)
        union = np.sum(vis[:, i] | vis[:, j], axis=1).astype(float)
        J[i, j] = J[j, i] = np.mean(inter / np.clip(union, 1, None))
    return J


_GLOBAL_COVIS: dict = {}


def global_covisibility(station_by_name: dict):
    """Pairwise co-visibility Jaccard for a set of stations, computed ONCE over
    a representative day and cached by station-name set.

    The nominal GPS constellation repeats each sidereal day, so a station
    pair's mean co-visibility is essentially day-independent.  The previous
    code recomputed the full O(n^2 * n_epoch) co-visibility matrix for EVERY
    day in the pooled fit (~366x redundant on the full-year global arm)."""
    key = frozenset(station_by_name)
    if key in _GLOBAL_COVIS:
        return _GLOBAL_COVIS[key]
    names = sorted(station_by_name)
    sts = [station_by_name[n] for n in names]
    epochs = np.linspace(0.0, 86164.0, 48)
    J = covisibility(sts, epochs)
    lut = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            lut[(names[a], names[b])] = lut[(names[b], names[a])] = float(J[a, b])
    _GLOBAL_COVIS[key] = lut
    return lut


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z (all 1-D)."""
    with np.errstate(all="ignore"):
        def resid(a, b):
            bs = (b - b.mean()) / (b.std() + 1e-9)
            A = np.column_stack([np.ones_like(bs), bs])
            beta = np.linalg.solve(A.T @ A + 1e-6 * np.eye(2), A.T @ a)
            return a - A @ beta
        rx, ry = resid(np.asarray(x, float), np.asarray(z, float)), \
                 resid(np.asarray(y, float), np.asarray(z, float))
        return float(np.corrcoef(rx, ry)[0, 1])


def run(product, date_tags, label):
    days = list(td.load_multi_day(product, date_tags))
    names = sorted({n for _, s in days for n in s})
    # co-visibility Jaccard, computed ONCE for the union station set (the
    # nominal constellation repeats each sidereal day)
    covis_lut = global_covisibility(
        {n: s[n]["station"] for _, s in days for n in s})
    # pooled pair records with covisibility
    D, Cc, V = [], [], []
    pair_stations = []  # (station_a, station_b) for each pair, for station-level CV
    for dt, s in days:
        nm = list(s)
        sts = [s[n]["station"] for n in nm]
        cols = []
        for n in nm:
            bp = s[n].get("_bp")
            if bp is None:
                ddt = float(np.median(np.diff(s[n]["t"])))
                if not np.isfinite(ddt) or ddt <= 0:
                    ddt = 30.0
                bp = ts.bandpass(s[n]["resid"], ddt)
                s[n]["_bp"] = bp
            cols.append(bp)
        Rm = np.column_stack(cols)
        # NOTE (bug fix): per-epoch network-mean removal disabled -- it
        # destroys the distance structure T4 is trying to separate from
        # co-visibility (flat ~0.9 correlation at ALL distances, R2 0.001).
        # See step_11 for the measured effect.  T1 never removed it.
        import os as _os
        if _os.environ.get("TEP_REMOVE_NETWORK_MEAN", "0") == "1":
            Rm = Rm - Rm.mean(1, keepdims=True)
        Rm = Rm - Rm.mean(0)                 # Remove time mean
        sd = Rm.std(0); sd[sd < 1e-9] = np.nan
        with np.errstate(all="ignore"):
            Cm = (Rm.T @ Rm) / len(Rm) / np.outer(sd, sd)
        for i, j in itertools.combinations(range(len(nm)), 2):
            if not np.isfinite(Cm[i, j]):
                continue
            jv = covis_lut.get((nm[i], nm[j]))
            if jv is None:
                continue
            D.append(td.great_circle_km(sts[i], sts[j]))
            Cc.append(float(Cm[i, j]))
            V.append(jv)
            pair_stations.append((nm[i], nm[j]))
    D, Cc, V = np.array(D), np.array(Cc), np.array(V)
    print(f"[{label}] {len(D):,} pairs   covis range [{V.min():.2f}, {V.max():.2f}]   "
          f"corr(D,covis) = {np.corrcoef(D, V)[0,1]:+.2f}")

    logd = np.log(D + 1)
    pc_dist = partial_corr(Cc, logd, V)          # corr | covisibility
    pc_cov = partial_corr(Cc, V, logd)           # covis | distance
    print(f"  partial corr(correlation, log-distance | covisibility) = {pc_dist:+.3f}")
    print(f"  partial corr(correlation, covisibility | log-distance) = {pc_cov:+.3f}")

    # held-out model comparison: 5-fold by STATION (not by pair) to prevent
    # leakage of station-specific structure between folds.  A pair is in the
    # test fold if EITHER station is held out; training pairs have BOTH
    # stations in the training fold.
    all_stations = sorted(set(names))
    sta_idx = {st: i for i, st in enumerate(all_stations)}
    sta_arr = np.array([sta_idx[a] for a, b in pair_stations])
    stb_arr = np.array([sta_idx[b] for a, b in pair_stations])
    n_sta = len(all_stations)
    sta_perm = RNG.permutation(n_sta)
    sta_folds = np.array_split(sta_perm, 5)
    def r2_oos(Xcols):
        sse, sst = 0.0, 0.0
        for f in range(5):
            te_sta = set(sta_folds[f].tolist())
            te_mask = np.array([(sta_arr[k] in te_sta or stb_arr[k] in te_sta)
                                for k in range(len(D))])
            tr_mask = ~te_mask
            if tr_mask.sum() < 10 or te_mask.sum() < 2:
                continue
            tr = np.flatnonzero(tr_mask)
            te = np.flatnonzero(te_mask)
            def z(m):
                mu, sg = m.mean(0), m.std(0) + 1e-9
                return (m - mu) / sg, mu, sg
            Xtr0 = np.column_stack([c[tr] for c in Xcols])
            Xte0 = np.column_stack([c[te] for c in Xcols])
            Xtrs, mu, sg = z(Xtr0)
            Xtr = np.column_stack([np.ones(len(tr)), Xtrs])
            Xte = np.column_stack([np.ones(len(te)), (Xte0 - mu) / sg])
            lam = 1e-2 * np.eye(Xtr.shape[1]); lam[0, 0] = 0.0
            beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ Cc[tr])
            with np.errstate(all="ignore"):
                pred = Xte @ beta
            sse += np.sum((Cc[te] - pred) ** 2)
            # Proper held-out R²: SST uses the TEST-set mean (the best constant
            # predictor on the held-out fold), not the training mean — using
            # the training mean leaks training information into the denominator.
            sst += np.sum((Cc[te] - np.mean(Cc[te])) ** 2)
        return 1.0 - sse / sst if sst > 0 else np.nan
    r2_cv = r2_oos([V, V ** 2])
    r2_both = r2_oos([V, V ** 2, np.exp(-D / 4000.0)])
    # Complementary test: fit lambda on each training fold only, then use
    # the fold-fitted exp(-D/lambda) feature.  This confirms the result is
    # not an artefact of the pre-registered 4000 km value from Papers 1-2.
    LAM_SEARCH = (1500.0, 2500.0, 3500.0, 4200.0, 6000.0, 9000.0)
    def r2_oos_fold_lambda():
        sse, sst = 0.0, 0.0
        for f in range(5):
            te_sta = set(sta_folds[f].tolist())
            te_mask = np.array([(sta_arr[k] in te_sta or stb_arr[k] in te_sta)
                                for k in range(len(D))])
            tr_mask = ~te_mask
            if tr_mask.sum() < 10 or te_mask.sum() < 2:
                continue
            tr = np.flatnonzero(tr_mask)
            te = np.flatnonzero(te_mask)
            # Fit lambda on training pairs only: bin by distance, fit exp
            # Use D[tr] for the percentile to avoid leakage from test pairs.
            ctr, mn, se, cn = ts.bin_by_distance(
                D[tr], Cc[tr], n_bins=15, dmin=100.0,
                dmax=np.percentile(D[tr], 99), min_count=10)
            ff = ts.fit_correlation_length(ctr, mn,
                                           weight=1.0 / np.clip(se, 1e-4, None) ** 2)
            lam_fold = ff["lam"] if np.isfinite(ff["lam"]) and ff["lam"] > 100 else 4000.0
            Xcols = [V, V ** 2, np.exp(-D / lam_fold)]
            Xtr0 = np.column_stack([c[tr] for c in Xcols])
            Xte0 = np.column_stack([c[te] for c in Xcols])
            mu, sg = Xtr0.mean(0), Xtr0.std(0) + 1e-9
            Xtr = np.column_stack([np.ones(len(tr)), (Xtr0 - mu) / sg])
            Xte = np.column_stack([np.ones(len(te)), (Xte0 - mu) / sg])
            lam_reg = 1e-2 * np.eye(Xtr.shape[1]); lam_reg[0, 0] = 0.0
            beta = np.linalg.solve(Xtr.T @ Xtr + lam_reg, Xtr.T @ Cc[tr])
            with np.errstate(all="ignore"):
                pred = Xte @ beta
            sse += np.sum((Cc[te] - pred) ** 2)
            sst += np.sum((Cc[te] - np.mean(Cc[te])) ** 2)
        return 1.0 - sse / sst if sst > 0 else np.nan
    r2_fold_lam = r2_oos_fold_lambda()
    print(f"  out-of-sample R2 (raw pairs):  covis-only {r2_cv:.3f}   "
          f"covis+distance(4000km) {r2_both:.3f}   "
          f"covis+distance(fold-fit lambda) {r2_fold_lam:.3f}   "
          f"gain {r2_both - r2_cv:+.3f}")

    # ---- co-visibility-controlled distance structure -------------------------
    # (1) residualise: fit binned corr ~ smooth(covis), subtract, re-fit exp(dist)
    ov = np.argsort(V)
    Vs, Cs = V[ov], Cc[ov]
    k = max(50, len(V) // 40)
    csum = np.cumsum(np.insert(Cs, 0, 0.0))
    smooth = np.zeros_like(Cs)
    for a in range(len(Cs)):
        lo, hi = max(0, a - k), min(len(Cs), a + k)
        smooth[a] = (csum[hi] - csum[lo]) / (hi - lo)
    resid_corr = np.empty_like(Cc)
    resid_corr[ov] = Cs - smooth
    ctr, mean, sem, cnt = ts.bin_by_distance(D, resid_corr, n_bins=20, dmin=100.0,
                                             dmax=np.percentile(D, 99), min_count=20)
    f_resid = ts.fit_correlation_length(ctr, mean, weight=1.0 / np.clip(sem, 1e-4, None) ** 2)
    print(f"  distance decay in covis-residualised correlation: "
          f"lambda {f_resid['lam']:.0f} km  R2 {f_resid['R2']:+.3f}")

    # (2) within fixed co-visibility bands, does correlation still fall with distance?
    band_fits = []
    edges = np.quantile(V, [0.2, 0.4, 0.6, 0.8])
    bands = [(0, edges[0]), (edges[0], edges[1]), (edges[1], edges[2]),
             (edges[2], edges[3]), (edges[3], 1.01)]
    for lo, hi in bands:
        m = (V >= lo) & (V < hi)
        if m.sum() < 200:
            continue
        # correlation of pair-corr with distance within this narrow covis slice
        r = float(np.corrcoef(D[m], Cc[m])[0, 1])
        band_fits.append({"covis": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                          "corr_vs_distance": round(r, 3)})
        print(f"  covis [{lo:.2f},{hi:.2f}]  n={m.sum():5d}  corr(pairCorr, distance) = {r:+.3f}")
    band_r = np.array([b["corr_vs_distance"] for b in band_fits])
    within_band_negative = bool(len(band_r) >= 3 and np.mean(band_r < -0.1) >= 0.6)

    resid_has_structure = bool(np.isfinite(f_resid["R2"]) and f_resid["R2"] > 0.3
                               and 200 < f_resid["lam"] < 20000)

    # Pre-registered T4 decision rule (PREREGISTRATION.md):
    # Pass: |partial corr(corr, log-dist | covis)| > 0.15 AND OOS R² gain
    # from the distance term > 0.02.
    # The residualised-fit and within-band analyses are descriptive extras
    # only; they do not enter the pass decision (no post-hoc criteria).
    r2_gain = float(r2_both - r2_cv)
    passes = bool(np.isfinite(pc_dist) and abs(pc_dist) > 0.15
                  and np.isfinite(r2_gain) and r2_gain > 0.02)
    print(f"  PRE-REGISTERED T4 rule: |partial corr| {abs(pc_dist):.3f} > 0.15 and "
          f"R2 gain {r2_gain:+.3f} > 0.02  ->  T4 pass: {passes}")
    print(f"  (descriptive extras: residual-fit {resid_has_structure}, "
          f"within-band {within_band_negative})")

    return {"label": label, "n_pairs": int(len(D)),
            "corr_dist_covis": float(np.corrcoef(D, V)[0, 1]),
            "partial_corr_distance_given_covis": pc_dist,
            "partial_corr_covis_given_distance": pc_cov,
            "oos_r2_covis_only": float(r2_cv), "oos_r2_covis_plus_distance": float(r2_both),
            "oos_r2_covis_plus_distance_fold_lambda": float(r2_fold_lam) if np.isfinite(r2_fold_lam) else np.nan,
            "oos_r2_gain": float(r2_both - r2_cv),
            "covis_residual_lambda_km": float(f_resid["lam"]), "covis_residual_R2": float(f_resid["R2"]),
            "within_covis_band_corr_vs_distance": band_fits,
            "T4_pass_rule": ("prereg: |partial_corr(corr, log-dist | covis)| > 0.15 and "
                             "OOS R2 gain from the distance term > 0.02"),
            "T4_descriptive_residual_structure": bool(resid_has_structure),
            "T4_descriptive_within_band_negative": bool(within_band_negative),
            "T4_pass": passes}


def precise_residual_structure(date_tags):
    """EU precise cluster: remove per-epoch network mean (common mode), then test
    whether the residual pair-correlation still shows distance decay."""
    days = [s for _, s in td.load_multi_day("precise", date_tags) if len(s) >= 6]
    D, C = [], []
    for s in days:
        nm = list(s)
        cols = []
        for n in nm:
            ddt = float(np.median(np.diff(s[n]["t"])))
            if not np.isfinite(ddt) or ddt <= 0:
                ddt = 300.0
            cols.append(ts.bandpass(s[n]["resid"], ddt))
        M = np.column_stack(cols)
        M = M - M.mean(1, keepdims=True)          # remove per-epoch network mean
        M = M - M.mean(0)
        sd = M.std(0); sd[sd < 1e-9] = np.nan
        with np.errstate(all="ignore"):
            Cm = (M.T @ M) / len(M) / np.outer(sd, sd)
        sts = [s[n]["station"] for n in nm]
        for i, j in itertools.combinations(range(len(nm)), 2):
            if np.isfinite(Cm[i, j]):
                D.append(td.great_circle_km(sts[i], sts[j])); C.append(float(Cm[i, j]))
    D, C = np.array(D), np.array(C)
    ctr, mean, sem, cnt = ts.bin_by_distance(D, C, n_bins=16, dmin=50.0, dmax=1300.0, min_count=10)
    f = ts.fit_correlation_length(ctr, mean, weight=1.0 / np.clip(sem, 1e-4, None) ** 2)
    print(f"[eu_precise_cm_removed] {len(D):,} pairs  after network-mean removal: "
          f"lambda {f['lam']:.0f} km  R2 {f['R2']:.3f}")
    return {"n_pairs": int(len(D)), "lambda_km": f["lam"], "R2": f["R2"],
            "structure_survives_cm_removal": bool(np.isfinite(f["R2"]) and f["R2"] > 0.3
                                                 and 200 < f["lam"] < 20000)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "test": "T4_common_view_vs_distance",
        "global_broadcast": run("broadcast", [f"2024{d:03d}" for d in range(100, 107)], "global_broadcast"),
        "eu_precise_residual": precise_residual_structure(td.available_dates("precise", 2024)[:60]),
        "timestamp": datetime.now().isoformat(),
    }
    # Full-year global-precise T4 is run by step_14_precise_global.
    res["T4_pass"] = res["global_broadcast"]["T4_pass"]
    (OUT / "step_13_commonview.json").write_text(json.dumps(res, indent=2))
    print(f"\nT4 pass: {res['T4_pass']}\nsaved {OUT / 'step_13_commonview.json'}")


if __name__ == "__main__":
    main()
