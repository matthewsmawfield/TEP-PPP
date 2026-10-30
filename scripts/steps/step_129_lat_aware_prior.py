#!/usr/bin/env python3
"""
STEP 129  --  Latitude-aware screening prior: exploiting the equatorial
              coupling modulation for improved prediction.

EXPLOITING THE LATITUDE-DEPENDENT SCREENING (Step 122) AS A STRENGTH.

Step 122 showed that the TEP field's coupling to quartz oscillators is
3.6x stronger at low latitudes than at mid latitudes (screening diff:
low-lat=0.339, mid-lat=0.093), consistent with the equatorial ionization
anomaly amplifying the TEP field's coupling to less stable oscillators.

A latitude-aware prior that uses different lambdas for different latitude
bands should beat a uniform prior, especially for low-latitude stations.

PROTOCOL:
  * Run the Step 64 P1 gate on 2024 development data.
  * Compare three kernels:
    1. uniform:     lambda=4200 km for all station pairs
    2. lat_aware:   lambda=5627 km for low-lat pairs, 1450 km for mid-lat
                    (from Step 122 quartz fits)
    3. empirical:   data-driven covariance

  * TEP exploitation prediction: the latitude-aware kernel beats the
    uniform kernel, especially for low-latitude target stations, because
    it captures the latitude-dependent coupling.

Output: results/outputs/step_129_lat_aware_prior.json
"""
from __future__ import annotations

import sys, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_129_lat_aware_prior.json"
PRODUCT = "clk_cod"
N_DAYS = 30

# Lambda values from Step 122 (quartz fits)
LAMBDA_LOW_LAT = 5627.0   # equatorial
LAMBDA_MID_LAT = 1450.0   # mid-latitude
LAMBDA_HIGH_LAT = 1920.0  # high-latitude
LAMBDA_UNIFORM = s64.LAMBDA_BASE_KM

LAT_BANDS = [(-90, -30, LAMBDA_HIGH_LAT), (-30, 30, LAMBDA_LOW_LAT),
             (30, 90, LAMBDA_MID_LAT)]


def _lat_band_lambda(lat_a, lat_b):
    """Get the appropriate lambda for a station pair based on latitude."""
    mid_lat = (lat_a + lat_b) / 2.0
    for lo, hi, lam in LAT_BANDS:
        if lo <= mid_lat < hi:
            return lam
    return LAMBDA_MID_LAT


def _rho_lat_aware(d_rr, lats):
    """Latitude-aware multi-scale covariance.

    Each pair gets a lambda based on the midpoint latitude of the pair.
    The covariance matrix is built element-by-element.
    """
    n = len(lats)
    rho_rr = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            lam = _lat_band_lambda(lats[i], lats[j])
            scales = (lam / 1.6, lam, lam * 1.6)
            rho_rr[i, j] = np.mean([np.exp(-d_rr[i, j] / L) for L in scales])
    return rho_rr


def _rho_lat_aware_rt(d_rt, lat_target, lats_ref):
    """Latitude-aware cross-covariance (target to references)."""
    n = len(lats_ref)
    rho_rt = np.ones(n)
    for j in range(n):
        lam = _lat_band_lambda(lat_target, lats_ref[j])
        scales = (lam / 1.6, lam, lam * 1.6)
        rho_rt[j] = np.mean([np.exp(-d_rt[j] / L) for L in scales])
    return rho_rt


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260902)

    print("Step 129: Latitude-aware screening prior", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print(f"  Low-lat lambda:  {LAMBDA_LOW_LAT} km", flush=True)
    print(f"  Mid-lat lambda:  {LAMBDA_MID_LAT} km", flush=True)
    print(f"  High-lat lambda: {LAMBDA_HIGH_LAT} km", flush=True)
    print(f"  Uniform lambda:  {LAMBDA_UNIFORM} km", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_DAYS).astype(int)]

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "uniform_rms", "lat_aware_rms", "emp_rms",
        "lat_aware_vs_uniform", "lat_aware_vs_emp",
        "low_lat_lat_aware_vs_uniform", "low_lat_lat_aware_vs_emp")}

    for k, tag in enumerate(days):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                               max_jumps=10_000)
        if not s_cur:
            continue
        pg = s64.prior_gram(prev, names)
        if pg is None:
            continue
        gnames, G, P = pg
        idx = [names.index(n) for n in gnames]
        Dg = D[np.ix_(idx, idx)]

        # Get station latitudes
        lats = []
        for n in gnames:
            st = s_cur[n]["station"]
            lats.append(st.lat_deg)
        lats = np.array(lats)

        # Causal detrend
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            t_s = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for ki in range(len(clk_ns)):
                tk = t_s[ki]
                mask = (t_s <= tk) & (t_s >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(t_s[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS

        P = P.astype(float, copy=True)
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]),
                                    np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        lats = lats[keep]
        R = R[:, keep]; P = P[:, keep]
        G = G[np.ix_(keep, keep)]; Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("uniform", "lat_aware", "emp")}
        day_sq_ll = {k: [] for k in ("uniform", "lat_aware", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Uniform kernel
            rho_rr_u = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_u = s64._rho_multi(Dg[b, pool])
            try:
                w_uniform = _reg_solve(
                    rho_rr_u * G[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_u * G[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Latitude-aware kernel
            lats_pool = lats[pool]
            rho_rr_la = _rho_lat_aware(Dg[np.ix_(pool, pool)], lats_pool)
            rho_rt_la = _rho_lat_aware_rt(Dg[b, pool], lats[b], lats_pool)
            try:
                w_la = _reg_solve(
                    rho_rr_la * G[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_la * G[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Empirical
            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu_p, ju_p = np.triu_indices(len(pool), 1)
            dv = Dg[np.ix_(pool, pool)][iu_p, ju_p]
            cv = corr_emp[iu_p, ju_p]
            best_e = (np.inf, 4200.0)
            for Le in (200, 400, 800, 1500, 3000, 6000, 12000, 25000):
                pr = np.exp(-0.5 * (dv / Le) ** 2)
                sse = np.sum((cv - pr) ** 2)
                if sse < best_e[0]:
                    best_e = (sse, Le)
            Le = best_e[1]
            ce_rr = np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / Le) ** 2)
            ce_rt = np.exp(-0.5 * (Dg[b, pool] / Le) ** 2)
            try:
                w_emp = _reg_solve(ce_rr + s64.NUGGET * np.eye(len(pool)), ce_rt)
            except np.linalg.LinAlgError:
                w_emp = w_uniform

            B = R[:, b]; A = R[:, pool]
            is_low_lat = abs(lats[b]) < 30.0
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                p_u = float(w_uniform @ rr)
                p_la = float(w_la @ rr)
                p_e = float(w_emp @ rr)
                day_sq["uniform"].append((truth - p_u) ** 2)
                day_sq["lat_aware"].append((truth - p_la) ** 2)
                day_sq["emp"].append((truth - p_e) ** 2)
                if is_low_lat:
                    day_sq_ll["uniform"].append((truth - p_u) ** 2)
                    day_sq_ll["lat_aware"].append((truth - p_la) ** 2)
                    day_sq_ll["emp"].append((truth - p_e) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["uniform_rms"].append(rms["uniform"])
        per_day["lat_aware_rms"].append(rms["lat_aware"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["lat_aware_vs_uniform"].append(
            100.0 * (1.0 - rms["lat_aware"] / rms["uniform"]))
        per_day["lat_aware_vs_emp"].append(
            100.0 * (1.0 - rms["lat_aware"] / rms["emp"]))

        if all(len(day_sq_ll[k]) > 10 for k in day_sq_ll):
            rms_ll = {k: float(np.sqrt(np.mean(v)))
                      for k, v in day_sq_ll.items()}
            per_day["low_lat_lat_aware_vs_uniform"].append(
                100.0 * (1.0 - rms_ll["lat_aware"] / rms_ll["uniform"]))
            per_day["low_lat_lat_aware_vs_emp"].append(
                100.0 * (1.0 - rms_ll["lat_aware"] / rms_ll["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(days)}] "
                  f"uniform={np.median(per_day['uniform_rms']):.1f}, "
                  f"lat_aware={np.median(per_day['lat_aware_rms']):.1f} m",
                  flush=True)

    n = len(per_day["uniform_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "lat_aware_vs_uniform_pct": _ci(per_day["lat_aware_vs_uniform"], rng),
        "lat_aware_vs_emp_pct": _ci(per_day["lat_aware_vs_emp"], rng),
        "low_lat_lat_aware_vs_uniform_pct": _ci(
            per_day["low_lat_lat_aware_vs_uniform"], rng),
        "low_lat_lat_aware_vs_emp_pct": _ci(
            per_day["low_lat_lat_aware_vs_emp"], rng),
        "median_rms_m": {
            "uniform": float(np.median(per_day["uniform_rms"])) if n else None,
            "lat_aware": float(np.median(per_day["lat_aware_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    lavu = results["lat_aware_vs_uniform_pct"]
    llavu = results["low_lat_lat_aware_vs_uniform_pct"]

    if lavu.get("median") is not None:
        if lavu["median"] > 0:
            interpretations.append(
                f"The latitude-aware TEP kernel beats the uniform kernel "
                f"(median {lavu['median']:+.2f}%, "
                f"CI95 [{lavu['ci95'][0]:+.2f}, {lavu['ci95'][1]:+.2f}]). "
                f"The latitude-dependent lambdas from Step 122 carry "
                f"predictive information that the uniform kernel misses. "
                f"This exploits the equatorial coupling modulation as a "
                f"predictive feature.")
        else:
            interpretations.append(
                f"The latitude-aware kernel does not beat the uniform "
                f"kernel ({lavu['median']:+.2f}%).  The latitude-dependent "
                f"lambdas from Step 122 do not improve prediction.")

    if llavu.get("median") is not None and llavu.get("n", 0) > 0:
        interpretations.append(
            f"For low-latitude targets specifically, latitude-aware vs "
            f"uniform: {llavu['median']:+.2f}% "
            f"(CI95 [{llavu['ci95'][0]:+.2f}, {llavu['ci95'][1]:+.2f}], "
            f"n={llavu['n']}).")

    verdict = {
        "step": 129,
        "title": "Latitude-aware screening prior: exploiting equatorial coupling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "lambdas": {
            "low_lat": LAMBDA_LOW_LAT,
            "mid_lat": LAMBDA_MID_LAT,
            "high_lat": LAMBDA_HIGH_LAT,
            "uniform": LAMBDA_UNIFORM,
        },
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The latitude-aware TEP kernel (low-lat lambda=5627 km, "
            "mid-lat lambda=1450 km, from Step 122) should beat the "
            "uniform kernel (lambda=4200 km), especially for low-latitude "
            "targets, because the latitude-dependent coupling strength "
            "carries predictive information.  This exploits the equatorial "
            "ionization anomaly modulation as a predictive feature."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
