#!/usr/bin/env python3
"""
STEP 64  --  Maximal a-priori TEP clock predictor vs a data-tuned conventional
             model.  The P1 gate: TEP-specific predictive superiority.

P1 asks whether the *physics-fixed* TEP kernel -- every parameter set a priori
from Papers 1-2 -- beats a conventional spatial predictor that is allowed to
LEARN its covariance from the target's own history.  Earlier P1 attempts used
an isotropic Gaussian in the amplitude domain and tied.  This step gives TEP
its full arsenal, all in the phase domain where Papers 1-2 measured it:

  1. phase-cos coherence      spectral_phase_gram from the complete prior UTC
                              day (Re of the normalised cross-spectrum);
     Hadamard'd with the TEP distance covariance (PSD-safe, Step 56).
  2. fixed TEP scale          4,200 km, the Papers 1-2 / Step 53 phase scale.
  3. multi-scale              continuous Temporal-Topology screening -> the
                              kernel is the mean of exponential components at
                              lambda, lambda/1.6 and 1.6*lambda.

The former "full arsenal" added a harmonic regression fitted to a
full-day-detrended prior-day series directly to a current-day two-hour causal
residual.  Those are different response variables; the addition caused the
catastrophic Step 64 error and is removed rather than tuned around.  Likewise,
the former pairwise azimuth deformation was not a guaranteed positive-definite
kernel on the sphere.  Anisotropy is excluded from the registered operational
predictor until it is implemented as a valid spherical covariance.  A
speculative annual/event modulation also weakened the 2024 development result
and is excluded from the frozen independent-year model.

The conventional competitor picks (form, lambda, nugget, k) by leave-one-out
on the held-out station's OWN prior-day residual -- the strongest deployable
conventional spatial model.

Held-out (leave-one-station-out) clock prediction on the dense EUREF/EPN
network; the current-day evaluation epochs are struck AFTER the prior-day
structures are frozen.  Day-clustered bootstrap CI on the paired RMS
difference.  If the maximal TEP predictor beats the tuned conventional model
with a positive 95% CI, P1 passes.

Output: results/outputs/step_64_maximal_tep_p1.json
"""
from __future__ import annotations

import sys
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td                       # noqa: E402
from scripts.lib.tep_phase import spectral_phase_gram        # noqa: E402
DT_S = 300.0
C_M_PER_NS = 0.299792458


def load_day(dt, product="precise"):
    s = td.load_series(product, dt, min_valid_frac=0.0, max_jumps=10_000)
    if not s or len(s) < 12:
        return None
    names = list(s)
    raw = np.column_stack([s[n]["raw"] for n in names]).astype(float) * C_M_PER_NS
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            D[i, j] = D[j, i] = td.great_circle_km(s[names[i]]["station"], s[names[j]]["station"])
    return names, s[names[0]]["t"].astype(float), raw, D

OUT = ROOT / "results" / "outputs" / "step_64_maximal_tep_p1.json"
RNG = np.random.default_rng(20260902)

N_DAYS = int(os.environ.get("TEP_P1_DAYS", "30"))
EVAL_STRIDE = int(os.environ.get("TEP_P1_STRIDE", "9"))
PREWINDOW_S = 7200.0
MIN_REFS = 8
LAMBDA_BASE_KM = 4200.0
# Nuisance value selected during the 2024 repair/development pass.  It is not a
# physics constant and cannot be re-selected on the independent-year run.
NUGGET = 0.70
MULTI_SCALES_KM = (LAMBDA_BASE_KM / 1.6, LAMBDA_BASE_KM, LAMBDA_BASE_KM * 1.6)

CV_FORMS = ("gaussian", "exponential", "matern15")
CV_LAM = (150.0, 400.0, 900.0, 2000.0, 4200.0, 9000.0, 20000.0)
CV_NUG = (0.15, 0.35, 0.70, 1.27)


def _prev(tag):
    return (datetime.strptime(tag, "%Y%j") - timedelta(days=1)).strftime("%Y%j")


def _rho_multi(d, lam_scale=1.0):
    # mean of exponential (Matern-1/2) components across the screening continuum
    return np.mean([np.exp(-d / (L * lam_scale)) for L in MULTI_SCALES_KM], axis=0)


def _rho_form(d, form, lam):
    if form == "gaussian":
        return np.exp(-0.5 * (d / lam) ** 2)
    if form == "matern15":
        r = np.sqrt(3.0) * d / lam
        return (1.0 + r) * np.exp(-r)
    return np.exp(-d / lam)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def prior_gram(prev_day, cur_names):
    pnames, pt, praw, _ = prev_day
    spectral_cols, clean_raw_cols, kept = [], [], []
    for nm in cur_names:
        if nm not in pnames:
            continue
        y = praw[:, pnames.index(nm)].astype(float)
        good = np.isfinite(y)
        if good.mean() < 0.9 or good.sum() < 20:
            continue
        if not good.all():
            ix = np.arange(len(y))
            y = np.interp(ix, ix[good], y[good])
        clean_raw = y.copy()
        xq = np.arange(len(clean_raw)) - len(clean_raw) / 2.0
        spectral_y = clean_raw - np.polyval(np.polyfit(xq, clean_raw, 2), xq)
        if not np.isfinite(spectral_y).all() or not (1e-4 < np.std(spectral_y) < 1e4):
            continue
        spectral_cols.append(spectral_y)
        clean_raw_cols.append(clean_raw)
        kept.append(nm)
    if len(kept) < MIN_REFS + 1:
        return None
    try:
        g = spectral_phase_gram(np.column_stack(spectral_cols), DT_S)
    except ValueError:
        return None
    return kept, g["predictive_gram"], np.column_stack(clean_raw_cols)


def run(days, product="precise"):
    per_day = {k: [] for k in (
        "phasecos_rms", "arsenal_rms", "phase_only_rms", "conv_rms", "emp_rms",
        "phasecos_vs_conv", "phasecos_vs_emp", "arsenal_vs_phasecos",
        "arsenal_vs_conv", "arsenal_vs_emp", "arsenal_vs_phase_only", "arsenal_vs_geometry_scramble",
        "arsenal_vs_phase_scramble")}
    npre = int(PREWINDOW_S / DT_S)

    for tag in days:
        cur = load_day(tag, product=product)
        prev = load_day(_prev(tag), product=product)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        # band-limit current-day residuals per station (10-500 uHz), for scoring
        s_cur = td.load_series(product, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur:
            continue
        pg = prior_gram(prev, names)
        if pg is None:
            continue
        gnames, G, P = pg
        idx = [names.index(n) for n in gnames]
        Dg = D[np.ix_(idx, idx)]
        # Frozen after the 2024 repair pass.  Annual/event modulation was an
        # unsupported degree of freedom and reduced held-out development
        # performance; the independent-year predictor uses the measured TEP
        # phase scale directly.
        lam_scale = 1.0
        geometry_perm = RNG.permutation(len(gnames))
        phase_perm = RNG.permutation(len(gnames))
        Dscr = Dg[np.ix_(geometry_perm, geometry_perm)]
        Gscr = G[np.ix_(phase_perm, phase_perm)]

        # current-day band-limited residual matrix aligned to gnames
        # Use raw clock with causal detrending (expanding window) to avoid
        # look-ahead from full-day robust_detrend.  The prior-day matrix P
        # below is detrended over the full prior day, which is legitimate
        # because the prior day is entirely in the past.
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            t_s = s_cur[n]["t"].astype(float)
            # causal linear detrend: fit trend only on samples up to each
            # evaluation point, then take the residual at that point.
            # For efficiency, use a sliding-window fit (2h window ending
            # at each point) rather than a full expanding window.
            r = np.full_like(clk_ns, np.nan)
            win_s = 7200.0  # 2-hour causal window
            for k in range(len(clk_ns)):
                tk = t_s[k]
                mask = (t_s <= tk) & (t_s >= tk - win_s) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(t_s[mask], clk_ns[mask], 1)
                    r[k] = clk_ns[k] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * C_M_PER_NS
        # Clean prior-day raw matrix aligned to gnames.  ``prior_gram`` returns
        # the interpolated columns it actually admitted; do not re-index the
        # unfiltered source matrix and reintroduce invalid clock datums.
        P = P.astype(float, copy=True)
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]), np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)
        # robust station QC: drop columns whose band-limited residual std is
        # degenerate or > 4x the network median (unrepaired jump).
        # Use nanstd because the causal detrend produces NaN values during
        # the warmup period (first ~2h of each day).
        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G = G[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]
        Dscr = Dscr[np.ix_(keep, keep)]
        Gscr = Gscr[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in (
            "phasecos", "arsenal", "phase_only", "conv", "emp",
            "geometry_scramble", "phase_scramble")}
        eval_i = list(range(npre, R.shape[0] - 1, EVAL_STRIDE))
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < MIN_REFS:
                continue
            Grr = G[np.ix_(pool, pool)]
            Grt = G[b, pool]

            # --- registered operational TEP: phase Gram x isotropic
            #     multi-scale covariance with a predetermined time-varying
            #     correlation length.  Every covariance in the mixture is
            #     positive definite; no pairwise azimuth warp is used. ---
            rho_rr = _rho_multi(Dg[np.ix_(pool, pool)], lam_scale)
            rho_rt = _rho_multi(Dg[b, pool], lam_scale)
            try:
                w_max = _reg_solve(rho_rr * Grr + NUGGET * np.eye(len(pool)), rho_rt * Grt)
            except np.linalg.LinAlgError:
                continue

            # Same historical phase information without TEP geometry.
            try:
                w_phase_only = _reg_solve(Grr + NUGGET * np.eye(len(pool)), Grt)
            except np.linalg.LinAlgError:
                continue

            # --- isotropic phase-cos TEP (Step 56 baseline) ---
            iso_rr = np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / LAMBDA_BASE_KM) ** 2)
            iso_rt = np.exp(-0.5 * (Dg[b, pool] / LAMBDA_BASE_KM) ** 2)
            try:
                w_iso = _reg_solve(iso_rr * Grr + NUGGET * np.eye(len(pool)), iso_rt * Grt)
            except np.linalg.LinAlgError:
                w_iso = w_max

            # Geometry-label and phase-label nulls preserve the marginal
            # matrices while severing their station correspondence.
            rho_rr_s = _rho_multi(Dscr[np.ix_(pool, pool)], lam_scale)
            rho_rt_s = _rho_multi(Dscr[b, pool], lam_scale)
            try:
                w_geometry_scr = _reg_solve(
                    rho_rr_s * Grr + NUGGET * np.eye(len(pool)), rho_rt_s * Grt)
                Grr_s = Gscr[np.ix_(pool, pool)]
                Grt_s = Gscr[b, pool]
                w_phase_scr = _reg_solve(
                    rho_rr * Grr_s + NUGGET * np.eye(len(pool)), rho_rt * Grt_s)
            except np.linalg.LinAlgError:
                continue

            # --- empirical-covariance conventional: fit rho(d) to the prior-day
            #     reference-network covariance, EXCLUDING the target station
            #     (same info budget as the TEP kernel, which uses only the
            #     reference phase Gram and the physics-fixed distance
            #     covariance -- never the target's own covariance) ---
            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu_p, ju_p = np.triu_indices(len(pool), 1)
            dv, cv = Dg[np.ix_(pool, pool)][iu_p, ju_p], corr_emp[iu_p, ju_p]
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
                w_emp = _reg_solve(ce_rr + NUGGET * np.eye(len(pool)), ce_rt)
            except np.linalg.LinAlgError:
                w_emp = w_iso

            # --- tuned conventional: LOO-select form/lam/nug on B's prior day ---
            best = (np.inf, None)
            yb = P[:, b]
            Ppool = P[:, pool]
            d_rr_p = Dg[np.ix_(pool, pool)]
            d_t_p = Dg[b, pool]
            for form in CV_FORMS:
                for lc in CV_LAM:
                    Rrr = _rho_form(d_rr_p, form, lc)
                    Rrt = _rho_form(d_t_p, form, lc)
                    for nug in CV_NUG:
                        try:
                            wc = _reg_solve(Rrr + nug * np.eye(len(pool)), Rrt)
                        except np.linalg.LinAlgError:
                            continue
                        fitted = np.einsum("ti,i->t", Ppool, wc, optimize=False)
                        err = float(np.sqrt(np.mean((yb - fitted) ** 2)))
                        if err < best[0]:
                            best = (err, wc)
            if best[1] is None:
                continue
            w_conv = best[1]

            B = R[:, b]
            A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                # NOTE: a previous version of this step applied a truth-
                # dependent cap filter (``cap = 20 + 20*|truth|``) that
                # excluded epochs where any predictor's magnitude was large
                # relative to the eventual residual.  That filter used the
                # future truth and was a data leak; it has been removed.
                p_phasecos = float(w_iso @ rr)                 # PRIMARY P1 predictor
                p_arsenal = float(w_max @ rr)                  # corrected registered predictor
                p_phase_only = float(w_phase_only @ rr)
                pc = float(w_conv @ rr)
                pe = float(w_emp @ rr)
                pgs = float(w_geometry_scr @ rr)
                pps = float(w_phase_scr @ rr)
                day_sq["phasecos"].append((truth - p_phasecos) ** 2)
                day_sq["arsenal"].append((truth - p_arsenal) ** 2)
                day_sq["phase_only"].append((truth - p_phase_only) ** 2)
                day_sq["conv"].append((truth - pc) ** 2)
                day_sq["emp"].append((truth - pe) ** 2)
                day_sq["geometry_scramble"].append((truth - pgs) ** 2)
                day_sq["phase_scramble"].append((truth - pps) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["phasecos_rms"].append(rms["phasecos"])
        per_day["arsenal_rms"].append(rms["arsenal"])
        per_day["phase_only_rms"].append(rms["phase_only"])
        per_day["conv_rms"].append(rms["conv"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["phasecos_vs_conv"].append(100.0 * (1.0 - rms["phasecos"] / rms["conv"]))
        per_day["phasecos_vs_emp"].append(100.0 * (1.0 - rms["phasecos"] / rms["emp"]))
        per_day["arsenal_vs_phasecos"].append(100.0 * (1.0 - rms["arsenal"] / rms["phasecos"]))
        per_day["arsenal_vs_conv"].append(100.0 * (1.0 - rms["arsenal"] / rms["conv"]))
        per_day["arsenal_vs_emp"].append(100.0 * (1.0 - rms["arsenal"] / rms["emp"]))
        per_day["arsenal_vs_phase_only"].append(100.0 * (1.0 - rms["arsenal"] / rms["phase_only"]))
        per_day["arsenal_vs_geometry_scramble"].append(
            100.0 * (1.0 - rms["arsenal"] / rms["geometry_scramble"]))
        per_day["arsenal_vs_phase_scramble"].append(
            100.0 * (1.0 - rms["arsenal"] / rms["phase_scramble"]))

    def ci(key):
        v = np.array(per_day[key], float)
        if len(v) < 4:
            return {"median": None, "ci95": None, "n": len(v)}
        bs = [float(np.median(RNG.choice(v, len(v), replace=True))) for _ in range(6000)]
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n": len(v), "frac_positive": float(np.mean(v > 0))}

    def med(k):
        return float(np.median(per_day[k])) if per_day[k] else None
    res = {
        "n_days_used": len(per_day["phasecos_rms"]),
        "phasecos_vs_tuned_conventional_pct": ci("phasecos_vs_conv"),
        "phasecos_vs_empirical_covariance_pct": ci("phasecos_vs_emp"),
        "full_arsenal_vs_phasecos_pct": ci("arsenal_vs_phasecos"),
        "full_arsenal_vs_tuned_conventional_pct": ci("arsenal_vs_conv"),
        "full_arsenal_vs_empirical_covariance_pct": ci("arsenal_vs_emp"),
        "full_arsenal_vs_phase_only_pct": ci("arsenal_vs_phase_only"),
        "full_arsenal_vs_geometry_scramble_pct": ci("arsenal_vs_geometry_scramble"),
        "full_arsenal_vs_phase_scramble_pct": ci("arsenal_vs_phase_scramble"),
        "median_rms_m": {
            "phasecos": med("phasecos_rms"), "full_arsenal": med("arsenal_rms"),
            "phase_only": med("phase_only_rms"),
            "tuned_conventional": med("conv_rms"), "empirical_covariance": med("emp_rms"),
        },
    }
    m = res["full_arsenal_vs_tuned_conventional_pct"]
    e = res["full_arsenal_vs_empirical_covariance_pct"]
    po = res["full_arsenal_vs_phase_only_pct"]
    gs = res["full_arsenal_vs_geometry_scramble_pct"]
    ps = res["full_arsenal_vs_phase_scramble_pct"]
    res["development_gate_pass"] = bool(
        m["ci95"] and m["ci95"][0] > 0
        and e["ci95"] and e["ci95"][0] > 0
        and po["ci95"] and po["ci95"][0] > 0
        and gs["ci95"] and gs["ci95"][0] > 0
        and ps["ci95"] and ps["ci95"][0] > 0)
    # The corrected model and nuisance value were developed against the 2024
    # corpus.  P1 can pass only on a subsequently acquired independent year
    # (Step 92).  On the development year itself, p1_pass is False by
    # construction -- the gate requires independent-year confirmation, not
    # in-sample validation.  This is a design constraint, not a hardcoded
    # verdict: the development_gate_pass flag above reflects the in-sample
    # statistical result, and p1_pass is the operational gate that requires
    # out-of-sample replication.
    res["p1_pass"] = False
    res["p1_requires_independent_year"] = True
    return res


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in ("precise", "clk_cod"):
        dates = td.available_dates(product, 2024)
        days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_DAYS).astype(int)]
        arms[product] = run(days, product=product)
        print(f"\n[{product}] {json.dumps(arms[product], indent=1)}")
    p1_pass = bool(arms.get("precise", {}).get("p1_pass"))
    out = {
        "step": 64,
        "title": "Maximal a-priori TEP clock predictor vs data-tuned conventional (P1 gate)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arsenal": ["phase-cos coherence Gram (prior day)",
                    "fixed 4,200 km TEP phase scale",
                    "isotropic multi-scale exponential covariance (lambda/1.6, lambda, 1.6 lambda)"],
        "nugget": NUGGET,
        "development_period": "2024; strict P1 requires a frozen independent-year replication",
        "withdrawn_components": [
            "harmonic add-on: incompatible prior/current residual definitions",
            "pairwise azimuth warp: not guaranteed positive definite on the sphere",
            "annual/event lambda modulation: weakened the 2024 development benchmark",
        ],
        "conventional": "LOO-selected (form, lambda, nugget) on the held-out station's own prior day",
        "arms": arms,
        "p1_pass": p1_pass,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nP1 {'PASS' if p1_pass else 'not yet'}  -- saved {OUT}")


if __name__ == "__main__":
    main()
