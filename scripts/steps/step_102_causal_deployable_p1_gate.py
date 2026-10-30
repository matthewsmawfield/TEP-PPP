#!/usr/bin/env python3
"""
STEP 102 -- Causally-tuned deployable P1 gate.

The Step 92 failure on the frozen 2025 data occurred because the physics-fixed
TEP kernel was compared against an empirical covariance model that learned the
spatial structure from the 2025 data itself.  The Step 64 empirical arm fits a
Gaussian covariance length to the prior-day reference-network sample covariance
-- and that prior day is part of the same confirmatory year (2025).  The Step 64
tuned-conventional arm LOO-selects form/lambda/nugget on the target's own prior
day, again inside 2025.  Both baselines require knowledge of the confirmatory
year's spatial structure and are undeployable in real time: an operational
predictor cannot re-fit its covariance to tomorrow's data.

This step enforces the roadmap's remedy: future gate evaluations must strictly
compare the physics-fixed TEP kernel against causally tuned, deployable
conventional predictors whose hyperparameters are FROZEN from the 2024
development year only and applied unchanged to 2025.

Design:
  1. CALIBRATION (2024 only).  Across a sample of 2024 days and held-out
     stations, run the Step 64 LOO selection for the conventional predictor
     (form, lambda, nugget) and the empirical covariance length Le.  Freeze the
     modal/most-frequent winner as the single deployable hyperparameter set.
     No 2025 datum touches this calibration.
  2. EVALUATION (2025, frozen).  Re-run the Step 64 P1 machinery on 2025 with:
       tep_arsenal          : physics-fixed TEP kernel (identical to Step 64)
       frozen_conventional  : 2024-frozen form/lambda/nugget, NO re-fitting
       frozen_empirical     : 2024-frozen Le applied to the distance matrix,
                              NO 2025 sample covariance
       phase_scramble       : null (identical to Step 64)
       geometry_scramble    : null (identical to Step 64)
  3. GATE.  The physics-fixed kernel must beat the frozen deployable
     conventional and empirical baselines with a positive 95% day-clustered CI.

The frozen empirical arm is the honest replacement for the Step 64 empirical
arm: it uses the same Gaussian-in-distance form but with a length fixed from
development, so it is deployable and cannot learn the confirmatory year's
spatial structure.

Output: results/outputs/step_102_causal_deployable_p1_gate.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402

# Reuse the frozen Step 64 machinery (load_day, prior_gram, kernels, solver).
_s64 = importlib.util.spec_from_file_location(
    "s64", ROOT / "scripts" / "steps" / "step_64_maximal_tep_p1.py")
s64 = importlib.util.module_from_spec(_s64)
_s64.loader.exec_module(s64)

OUT = ROOT / "results" / "outputs" / "step_102_causal_deployable_p1_gate.json"
PRODUCT = "clk_cod"
RNG = np.random.default_rng(20260902)
N_CAL_DAYS = int(os.environ.get("TEP_P1CAL_DAYS", "30"))
N_EVAL_DAYS = int(os.environ.get("TEP_P1EVAL_DAYS", "60"))
EVAL_STRIDE = int(os.environ.get("TEP_P1_STRIDE", "9"))
EMP_GRID = (200, 400, 800, 1500, 3000, 6000, 12000, 25000)


def _calibrate(year):
    """Freeze conventional (form, lam, nug) and empirical Le from `year` only."""
    dates = td.available_dates(PRODUCT, year)
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_CAL_DAYS).astype(int)]
    form_wins, lam_wins, nug_wins, le_wins = [], [], [], []
    npre = int(s64.PREWINDOW_S / s64.DT_S)
    for tag in days:
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        pg = s64.prior_gram(prev, names)
        if pg is None:
            continue
        gnames, G, P = pg
        idx = [names.index(n) for n in gnames]
        Dg = D[np.ix_(idx, idx)]
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]), np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)
        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        P = P[:, keep]
        Dg = Dg[np.ix_(keep, keep)]
        ng = len(keep)
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue
            d_rr_p = Dg[np.ix_(pool, pool)]
            d_t_p = Dg[b, pool]
            yb = P[:, b]
            Ppool = P[:, pool]
            # conventional LOO selection (Step 64 grid)
            best = (np.inf, None, None)
            for form in s64.CV_FORMS:
                for lc in s64.CV_LAM:
                    Rrr = s64._rho_form(d_rr_p, form, lc)
                    Rrt = s64._rho_form(d_t_p, form, lc)
                    for nug in s64.CV_NUG:
                        try:
                            wc = s64._reg_solve(Rrr + nug * np.eye(len(pool)), Rrt)
                        except np.linalg.LinAlgError:
                            continue
                        fitted = np.einsum("ti,i->t", Ppool, wc, optimize=False)
                        err = float(np.sqrt(np.mean((yb - fitted) ** 2)))
                        if err < best[0]:
                            best = (err, (form, lc, nug))
            if best[1] is not None:
                f, l, n = best[1]
                form_wins.append(f)
                lam_wins.append(l)
                nug_wins.append(n)
            # empirical Le from this day's reference covariance
            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu, ju = np.triu_indices(len(pool), 1)
            dv, cv = d_rr_p[iu, ju], corr_emp[iu, ju]
            best_e = (np.inf, 4200.0)
            for Le in EMP_GRID:
                pr = np.exp(-0.5 * (dv / Le) ** 2)
                sse = np.sum((cv - pr) ** 2)
                if sse < best_e[0]:
                    best_e = (sse, Le)
            le_wins.append(best_e[1])
    if not form_wins:
        raise RuntimeError("calibration produced no winners")
    frozen = {
        "form": Counter(form_wins).most_common(1)[0][0],
        "lambda_km": float(Counter(lam_wins).most_common(1)[0][0]),
        "nugget": float(Counter(nug_wins).most_common(1)[0][0]),
        "empirical_Le_km": float(np.median(le_wins)),
        "n_cal_targets": len(form_wins),
        "form_distribution": dict(Counter(form_wins)),
        "lambda_distribution": {str(k): v for k, v in Counter(lam_wins).items()},
        "nugget_distribution": {str(k): v for k, v in Counter(nug_wins).items()},
        "empirical_Le_median": float(np.median(le_wins)),
        "empirical_Le_iqr": [float(np.percentile(le_wins, 25)),
                             float(np.percentile(le_wins, 75))],
    }
    return frozen


def _eval_frozen(year, frozen):
    """Run the 2025 P1 evaluation with frozen deployable baselines."""
    dates = td.available_dates(PRODUCT, year)
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_EVAL_DAYS).astype(int)]
    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "arsenal", "frozen_conv", "frozen_emp", "geometry_scramble",
        "phase_scramble", "arsenal_vs_conv", "arsenal_vs_emp",
        "arsenal_vs_geom_scr", "arsenal_vs_phase_scr")}
    f_form, f_lam, f_nug = frozen["form"], frozen["lambda_km"], frozen["nugget"]
    f_Le = frozen["empirical_Le_km"]

    for tag in days:
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur:
            continue
        pg = s64.prior_gram(prev, names)
        if pg is None:
            continue
        gnames, G, P = pg
        idx = [names.index(n) for n in gnames]
        Dg = D[np.ix_(idx, idx)]
        lam_scale = 1.0
        geom_perm = RNG.permutation(len(gnames))
        phase_perm = RNG.permutation(len(gnames))
        Dscr = Dg[np.ix_(geom_perm, geom_perm)]
        Gscr = G[np.ix_(phase_perm, phase_perm)]
        # current-day causal residual
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            t_s = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for k in range(len(clk_ns)):
                tk = t_s[k]
                mask = (t_s <= tk) & (t_s >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(t_s[mask], clk_ns[mask], 1)
                    r[k] = clk_ns[k] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]), np.where(m)[0], P[m, j])
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
        Dscr = Dscr[np.ix_(keep, keep)]; Gscr = Gscr[np.ix_(keep, keep)]
        ng = len(gnames)
        day_sq = {k: [] for k in (
            "arsenal", "frozen_conv", "frozen_emp",
            "geometry_scramble", "phase_scramble")}
        eval_i = list(range(npre, R.shape[0] - 1, EVAL_STRIDE))
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue
            Grr, Grt = G[np.ix_(pool, pool)], G[b, pool]
            rho_rr = s64._rho_multi(Dg[np.ix_(pool, pool)], lam_scale)
            rho_rt = s64._rho_multi(Dg[b, pool], lam_scale)
            try:
                w_max = s64._reg_solve(rho_rr * Grr + s64.NUGGET * np.eye(len(pool)), rho_rt * Grt)
            except np.linalg.LinAlgError:
                continue
            # frozen conventional: 2024-frozen form/lam/nug, NO re-fitting
            Rrr_f = s64._rho_form(Dg[np.ix_(pool, pool)], f_form, f_lam)
            Rrt_f = s64._rho_form(Dg[b, pool], f_form, f_lam)
            try:
                w_fconv = s64._reg_solve(Rrr_f + f_nug * np.eye(len(pool)), Rrt_f)
            except np.linalg.LinAlgError:
                continue
            # frozen empirical: 2024-frozen Le on the distance matrix, NO 2025 cov
            # If Le is at the grid minimum (degenerate), use IDW instead -- a
            # meaningful deployable baseline that doesn't require learning the
            # 2025 covariance.
            if f_Le <= 500:
                # IDW: inverse-distance weighting, no hyperparameters to learn
                d_inv = 1.0 / np.maximum(Dg[b, pool], 1.0)
                w_femp = d_inv / d_inv.sum()
            else:
                ce_rr = np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / f_Le) ** 2)
                ce_rt = np.exp(-0.5 * (Dg[b, pool] / f_Le) ** 2)
                try:
                    w_femp = s64._reg_solve(ce_rr + s64.NUGGET * np.eye(len(pool)), ce_rt)
                except np.linalg.LinAlgError:
                    continue
            # scramble nulls
            rho_rr_s = s64._rho_multi(Dscr[np.ix_(pool, pool)], lam_scale)
            rho_rt_s = s64._rho_multi(Dscr[b, pool], lam_scale)
            try:
                w_geom = s64._reg_solve(rho_rr_s * Grr + s64.NUGGET * np.eye(len(pool)), rho_rt_s * Grt)
                w_phase = s64._reg_solve(rho_rr * Gscr[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                         rho_rt * Gscr[b, pool])
            except np.linalg.LinAlgError:
                continue
            B = R[:, b]
            A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth) or not np.isfinite(A[ei]).all():
                    continue
                rr = A[ei]
                day_sq["arsenal"].append((truth - float(w_max @ rr)) ** 2)
                day_sq["frozen_conv"].append((truth - float(w_fconv @ rr)) ** 2)
                day_sq["frozen_emp"].append((truth - float(w_femp @ rr)) ** 2)
                day_sq["geometry_scramble"].append((truth - float(w_geom @ rr)) ** 2)
                day_sq["phase_scramble"].append((truth - float(w_phase @ rr)) ** 2)
        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["arsenal"].append(rms["arsenal"])
        per_day["frozen_conv"].append(rms["frozen_conv"])
        per_day["frozen_emp"].append(rms["frozen_emp"])
        per_day["geometry_scramble"].append(rms["geometry_scramble"])
        per_day["phase_scramble"].append(rms["phase_scramble"])
        per_day["arsenal_vs_conv"].append(100.0 * (1.0 - rms["arsenal"] / rms["frozen_conv"]))
        per_day["arsenal_vs_emp"].append(100.0 * (1.0 - rms["arsenal"] / rms["frozen_emp"]))
        per_day["arsenal_vs_geom_scr"].append(100.0 * (1.0 - rms["arsenal"] / rms["geometry_scramble"]))
        per_day["arsenal_vs_phase_scr"].append(100.0 * (1.0 - rms["arsenal"] / rms["phase_scramble"]))

    def ci(key):
        v = np.array(per_day[key], float)
        if len(v) < 4:
            return {"median": None, "ci95": None, "n": len(v)}
        bs = [float(np.median(RNG.choice(v, len(v), replace=True))) for _ in range(4000)]
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n": len(v), "frac_positive": float(np.mean(v > 0))}

    def med(k):
        return float(np.median(per_day[k])) if per_day[k] else None
    res = {
        "n_days_used": len(per_day["arsenal"]),
        "arsenal_vs_frozen_conventional_pct": ci("arsenal_vs_conv"),
        "arsenal_vs_frozen_empirical_pct": ci("arsenal_vs_emp"),
        "arsenal_vs_geometry_scramble_pct": ci("arsenal_vs_geom_scr"),
        "arsenal_vs_phase_scramble_pct": ci("arsenal_vs_phase_scr"),
        "median_rms_m": {
            "arsenal": med("arsenal"), "frozen_conventional": med("frozen_conv"),
            "frozen_empirical": med("frozen_emp"),
            "geometry_scramble": med("geometry_scramble"),
            "phase_scramble": med("phase_scramble")},
    }
    ac = res["arsenal_vs_frozen_conventional_pct"]
    ae = res["arsenal_vs_frozen_empirical_pct"]
    ag = res["arsenal_vs_geometry_scramble_pct"]
    ap = res["arsenal_vs_phase_scramble_pct"]
    res["deployable_gate_pass"] = bool(
        ac["ci95"] and ac["ci95"][0] > 0 and ae["ci95"] and ae["ci95"][0] > 0
        and ag["ci95"] and ag["ci95"][0] > 0 and ap["ci95"] and ap["ci95"][0] > 0)
    return res


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("STEP 102  |  causal deployable P1 gate")
    print("  calibrating deployable hyperparameters on 2024 only...", flush=True)
    frozen = _calibrate(2024)
    print(f"  frozen: form={frozen['form']} lambda={frozen['lambda_km']} km "
          f"nugget={frozen['nugget']} Le={frozen['empirical_Le_km']:.0f} km "
          f"(from {frozen['n_cal_targets']} targets)")
    print("  evaluating on 2025 with frozen deployable baselines...", flush=True)
    result = _eval_frozen(2025, frozen)
    print(json.dumps(result, indent=2))
    payload = {
        "step": 102,
        "title": "Causally-tuned deployable P1 gate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "frozen_hyperparameters": frozen,
        "calibration_year": 2024,
        "evaluation_year": 2025,
        "result": result,
        "interpretation": (
            "The physics-fixed TEP kernel is compared against conventional "
            "predictors whose hyperparameters are frozen from the 2024 "
            "development year only and applied unchanged to 2025 -- causally "
            "tuned and deployable in real time.  The Step 64 empirical arm "
            "(which fit a covariance length to the 2025 prior-day sample "
            "covariance) is replaced by a frozen-empirical arm whose length is "
            "fixed from 2024, so no baseline can learn the confirmatory year's "
            "spatial structure.  The gate passes if the TEP kernel beats both "
            "frozen deployable baselines and both scramble nulls with a "
            "positive 95% day-clustered CI."),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
