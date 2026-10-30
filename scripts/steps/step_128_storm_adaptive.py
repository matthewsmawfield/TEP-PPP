#!/usr/bin/env python3
"""
STEP 128  --  Storm-adaptive TEP prior: exploiting geomagnetic storm
              modulation for improved holdover during disturbances.

EXPLOITING THE STORM MODULATION (Step 114) AS A PREDICTIVE STRENGTH.

Step 114 showed that the TEP field's correlation length increases
during geomagnetic storms (lambda: 2009 -> 3097 km, +54%).  Under the
TEP framing, this is the atmosphere amplifying the TEP field's coupling
during disturbances.  A storm-adaptive prior that uses a longer lambda
during storm periods should beat a fixed-lambda prior.

PROTOCOL:
  * Identify storm days from the Step 114 analysis (2024-05-10 G4 storm).
  * Run the Step 64 P1 gate on storm vs control vs recovery days.
  * Compare three kernels:
    1. fixed:     lambda=4200 km (registered, no adaptation)
    2. adaptive:  lambda=3097 km during storm, 2009 km during control
    3. empirical: data-driven covariance

  * TEP exploitation prediction: the adaptive kernel beats the fixed
    kernel during storm periods, because it captures the storm-modulated
    field structure.

Output: results/outputs/step_128_storm_adaptive.json
"""
from __future__ import annotations

import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_128_storm_adaptive.json"
PRODUCT = "clk_cod"

# Storm periods from Step 114 (2024-05-10 G4 geomagnetic storm)
# Control: 2024-05-07 (DOY 128), Storm: 2024-05-10 (DOY 131),
# Recovery: 2024-05-12 (DOY 133)
STORM_DOY = 131
CONTROL_DOYS = [126, 127, 128, 129, 130]
STORM_DOYS = [131, 132]
RECOVERY_DOYS = [133, 134, 135, 136, 137]

# Lambda values from Step 114
LAMBDA_CONTROL = 2009.0
LAMBDA_STORM = 3097.0
LAMBDA_RECOVERY = 5190.0
LAMBDA_FIXED = s64.LAMBDA_BASE_KM  # 4200 km


def _rho_multi_adaptive(d, lambda_km):
    """Multi-scale exponential covariance with adaptive lambda."""
    scales = (lambda_km / 1.6, lambda_km, lambda_km * 1.6)
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 3:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(4000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def _run_period(doys, year, rng, lambda_adaptive=None):
    """Run P1 gate for a set of DOYs with optional adaptive lambda."""
    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in ("fixed_rms", "adaptive_rms", "emp_rms",
                               "adaptive_vs_fixed", "adaptive_vs_emp")}

    for doy in doys:
        tag = f"{year}{doy:03d}"
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
        R = R[:, keep]; P = P[:, keep]
        G = G[np.ix_(keep, keep)]; Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("fixed", "adaptive", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        lam_adapt = lambda_adaptive or LAMBDA_FIXED

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Fixed kernel
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_fixed = _reg_solve(
                    rho_rr_f * G[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Adaptive kernel
            rho_rr_a = _rho_multi_adaptive(Dg[np.ix_(pool, pool)], lam_adapt)
            rho_rt_a = _rho_multi_adaptive(Dg[b, pool], lam_adapt)
            try:
                w_adapt = _reg_solve(
                    rho_rr_a * G[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_a * G[b, pool])
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
                w_emp = w_fixed

            B = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                day_sq["fixed"].append((truth - float(w_fixed @ rr)) ** 2)
                day_sq["adaptive"].append((truth - float(w_adapt @ rr)) ** 2)
                day_sq["emp"].append((truth - float(w_emp @ rr)) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["fixed_rms"].append(rms["fixed"])
        per_day["adaptive_rms"].append(rms["adaptive"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["adaptive_vs_fixed"].append(
            100.0 * (1.0 - rms["adaptive"] / rms["fixed"]))
        per_day["adaptive_vs_emp"].append(
            100.0 * (1.0 - rms["adaptive"] / rms["emp"]))

    return per_day


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260902)

    print("Step 128: Storm-adaptive TEP prior", flush=True)
    print(f"  Product: {PRODUCT}", flush=True)
    print(f"  Control lambda: {LAMBDA_CONTROL} km", flush=True)
    print(f"  Storm lambda:   {LAMBDA_STORM} km", flush=True)
    print(f"  Recovery lambda: {LAMBDA_RECOVERY} km", flush=True)
    print(f"  Fixed lambda:   {LAMBDA_FIXED} km", flush=True)
    print()

    periods = {
        "control": (CONTROL_DOYS, LAMBDA_CONTROL),
        "storm": (STORM_DOYS, LAMBDA_STORM),
        "recovery": (RECOVERY_DOYS, LAMBDA_RECOVERY),
    }

    all_results = {}
    for label, (doys, lam) in periods.items():
        print(f"  Running {label} ({len(doys)} days, lambda={lam})...",
              flush=True)
        per_day = _run_period(doys, 2024, rng, lambda_adaptive=lam)
        n = len(per_day["fixed_rms"])
        if n == 0:
            all_results[label] = {"n_days": 0, "error": "no valid days"}
            print(f"    {label}: no valid days", flush=True)
            continue
        all_results[label] = {
            "n_days": n,
            "adaptive_vs_fixed_pct": _ci(per_day["adaptive_vs_fixed"], rng),
            "adaptive_vs_emp_pct": _ci(per_day["adaptive_vs_emp"], rng),
            "median_rms_m": {
                "fixed": float(np.median(per_day["fixed_rms"])),
                "adaptive": float(np.median(per_day["adaptive_rms"])),
                "empirical": float(np.median(per_day["emp_rms"])),
            },
        }
        avf = all_results[label]["adaptive_vs_fixed_pct"]
        print(f"    {label}: n={n}, adaptive_vs_fixed={avf.get('median')}",
              flush=True)

    # Interpretation
    interpretations = []
    storm_res = all_results.get("storm", {})
    storm_avf = storm_res.get("adaptive_vs_fixed_pct", {})
    if storm_avf.get("median") is not None:
        if storm_avf["median"] > 0:
            interpretations.append(
                f"During the geomagnetic storm, the storm-adaptive TEP kernel "
                f"(lambda={LAMBDA_STORM} km) beats the fixed kernel "
                f"(lambda={LAMBDA_FIXED} km) by {storm_avf['median']:+.2f}% "
                f"(CI95 [{storm_avf['ci95'][0]:+.2f}, {storm_avf['ci95'][1]:+.2f}]). "
                f"The storm-modulated lambda from Step 114 carries predictive "
                f"information that the fixed kernel misses.  This exploits "
                f"the atmospheric storm modulation as a predictive feature.")
        else:
            interpretations.append(
                f"During the storm, the adaptive kernel does not beat the "
                f"fixed kernel ({storm_avf['median']:+.2f}%).  The storm "
                f"lambda from Step 114 does not carry useful predictive "
                f"information for holdover.")

    control_avf = all_results.get("control", {}).get("adaptive_vs_fixed_pct", {})
    if control_avf.get("median") is not None:
        interpretations.append(
            f"During control periods, adaptive vs fixed: "
            f"{control_avf.get('median'):+.2f}% "
            f"(CI95 [{control_avf['ci95'][0]:+.2f}, {control_avf['ci95'][1]:+.2f}]).")

    verdict = {
        "step": 128,
        "title": "Storm-adaptive TEP prior: exploiting geomagnetic storm modulation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "lambdas": {
            "control": LAMBDA_CONTROL,
            "storm": LAMBDA_STORM,
            "recovery": LAMBDA_RECOVERY,
            "fixed": LAMBDA_FIXED,
        },
        "results": all_results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The storm-adaptive TEP kernel (lambda=3097 km during storm, "
            "from Step 114) should beat the fixed kernel (lambda=4200 km) "
            "during storm periods, because the storm-modulated correlation "
            "length captures the amplified TEP field structure.  This "
            "exploits the atmospheric storm modulation as a predictive "
            "strength."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
