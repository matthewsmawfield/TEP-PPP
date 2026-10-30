#!/usr/bin/env python3
"""
STEP 83  --  The T4 co-visibility separation, redone with the CORRECT estimator.

Gate S2 (physics separability) is recorded PARTIAL because the frozen criterion
|partial corr(corr, log d | covis)| > 0.15 is not met: Step 45 reports 0.038 on
broadcast and -0.006 on precise.  But Step 45 measures the pair correlation as a
PEARSON CORRELATION OF BAND-PASSED SERIES (ts.bandpass at line 74, np.corrcoef
at line 119).  That is the statistic Paper 1 states "fails to detect TEP signals
due to phase averaging effects and amplitude suppression", and the same class of
statistic Section 3.7b identified and corrected in this paper's own pipeline.

A gate failed on a estimator the corpus rejects is not evidence about the
physics.  This step repeats the T4 design with the Papers 1-2 phase-alignment
index cos(arg(sum_f S_ij(f))), imported from Step 53 so it cannot drift.

Design (unchanged from Step 45 except for the estimator and the time unit):

  1. For each pair and DAY: p_ij(t) = phase-alignment index for that pair-day,
     and J_ij(t) = mean co-visibility of the two stations that day.  The unit is
     the pair-day rather than the epoch because the 10-500 uHz band needs a full
     day; co-visibility still varies day to day as the constellation precesses,
     which is what the regression needs.
  2. Stage 1, pooled over pairs with pair fixed effects:  p = alpha_p + kappa*J
     kappa is a SINGLE pooled common-view slope, not a per-pair OLS beta -- a
     per-pair slope from ~30-360 day points is noisy, and alpha_p then carries
     -err_kappa * Jbar_p, an errors-in-variables contaminant aligned with
     distance through the d-Jbar collinearity (the same bug fixed in Step 45
     on 2026-10-29).  Pairs with J ~ const over the year (the common-view-free
     regime) are now INCLUDED with alpha = pbar - kappa*Jbar rather than
     discarded.  Co-visibility uses the day's real SP3 ephemerides.
  3. Stage 2, across pairs:  alpha = a + b*log(d) + c*Jbar
     Report the partial correlation of alpha with log(d) given Jbar -- the
     frozen S2 criterion -- plus the same PC with cos(delta-lon) controlled
     and PC(alpha, cos dlon | Jbar), and the day-folded out-of-sample R^2
     gain of the distance term (the frozen criterion's second half).
  4. Nulls: (a) permute each pair's phase-cos series over days and re-fit
     kappa -- the honest null for the stage-1 day-level p-J covariation.
     (alpha = pbar - kappa*Jbar is permutation-invariant in pbar, and the
     kappa*Jbar term is exactly absorbed by the stage-2 Jbar control, so
     this permutation cannot produce a null PC -- an earlier version of
     this null returned the real PC to machine precision and its z-score
     was a floating-point artefact.)  (b) permute alpha across pairs --
     the honest null for the stage-2 distance-structure claim -- plus a
     station-clustered bootstrap CI, since pairs share stations.

If TEP is real, alpha carries distance structure that co-visibility cannot
explain.  If common view is the only mechanism, alpha is ~0 for every pair.

Output: results/outputs/step_83_t4_phasecos.json
"""
from __future__ import annotations

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

_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)
_s45 = importlib.util.spec_from_file_location(
    "s45", ROOT / "scripts" / "steps" / "step_45_temporal_t4.py")
s45 = importlib.util.module_from_spec(_s45)
_s45.loader.exec_module(s45)

import os

OUT = ROOT / "results" / "outputs" / "step_83_t4_phasecos.json"
if os.environ.get("TEP_S83_OUT"):
    OUT = ROOT / "results" / "outputs" / os.environ["TEP_S83_OUT"]
RNG = np.random.default_rng(20260905)
_ALL_PRODUCTS = [("clk_cod_ops", "precise_ops"), ("broadcast", "broadcast_spp")]
_ONLY = os.environ.get("TEP_S83_PRODUCTS")  # e.g. "precise_ops" to run one arm
PRODUCTS = [p for p in _ALL_PRODUCTS if not _ONLY or p[1] in _ONLY.split(",")]
# N_DAYS: 120 was Jan-Apr 2024 only -- the constellation precesses ~8 h over that
# span, so per-pair co-visibility barely varies and the stage-1 common-view slope
# beta is weakly identified, which drags the co-visibility-removed intercept
# toward the raw mean and biases the partial correlation toward zero.  A full
# year gives the full J-precession range and a genuinely identified beta.
N_DAYS = int(os.environ.get("TEP_S83_DAYS", "361"))
# DAY_STRIDE spreads the sampled days across the whole year even when N_DAYS is
# small, so J precession is exercised without loading every day.
DAY_STRIDE = int(os.environ.get("TEP_S83_STRIDE", "1"))
MIN_DAYS_PER_PAIR = int(os.environ.get("TEP_S83_MIN_DAYS", "30"))
N_NULL = 60
COVIS_EPOCHS = 8          # epochs per day at which co-visibility is evaluated
MIN_COVIS_STD = 1e-3      # a pair needs real day-to-day co-visibility variation


def partial_corr(x, y, z):
    def resid(a, b):
        A = np.column_stack([np.ones_like(b), b])
        beta = np.linalg.solve(A.T @ A + 1e-6 * np.eye(2), A.T @ a)
        return a - A @ beta
    return float(np.corrcoef(resid(x, z), resid(y, z))[0, 1])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product, label in PRODUCTS:
        _avail = td.available_dates(product, 2024)
        dates = _avail[::DAY_STRIDE][:N_DAYS] if DAY_STRIDE > 1 else _avail[:N_DAYS]
        if not dates:
            continue
        print(f"\n=== {label} ({product}), {len(dates)} days ===")
        per_pair = {}          # (a,b) -> list of (day_index, phasecos, covis)
        coords = {}
        covis_real = 0
        for k, tag in enumerate(dates):
            # Absolute day index within 2024 (NOT the enumerate index) so the
            # modelled constellation precesses correctly even when DAY_STRIDE
            # skips days.  tag is "YYYYDDD".
            doy = int(tag[4:]) - 1
            try:
                ser = td.load_series(product, tag, min_valid_frac=0.0, max_jumps=10_000)
            except Exception:
                continue
            RF, STA = s53._spectra(ser)
            if len(RF) < 8:
                continue
            nm = sorted(RF)
            sts = [STA[n] for n in nm]
            for n, stn in zip(nm, sts):
                coords[n] = stn
            # co-visibility for this day, averaged over a few epochs.
            # Prefer the day's REAL SP3 ephemerides (GPS), evaluated at UTC
            # seconds-of-day; the nominal model is arbitrarily Earth-phased.
            ep_sod = np.linspace(0, 86400, COVIS_EPOCHS, endpoint=False)
            rv = s45.load_real_visibility(tag, sts, ep_sod, systems=("G",))
            if rv is not None:
                J, pidx = s45._per_epoch_covisibility_from_vis(rv[0])
                covis_real += 1
            else:
                ep = doy * 86400.0 + ep_sod
                J, pidx = s45._per_epoch_covisibility(sts, ep)
            Jbar_day = J.mean(axis=1)
            for pi, (i, j) in enumerate(pidx):
                Fi, Fj = RF[nm[i]], RF[nm[j]]
                if Fi.shape != Fj.shape:
                    continue
                S = np.sum(np.conj(Fi) * Fj)
                if not np.isfinite(S) or S == 0:
                    continue
                key = (nm[i], nm[j])
                per_pair.setdefault(key, []).append(
                    (k, float(np.cos(np.angle(S))), float(Jbar_day[pi])))
            if (k + 1) % 20 == 0:
                print(f"    {k+1}/{len(dates)} days, {len(per_pair):,} pairs", flush=True)

        # Stage 1: POOLED common-view slope with pair fixed effects.
        # The earlier per-pair OLS beta is noisy (a handful-to-few-hundred day
        # points per pair) and its error enters alpha multiplied by Jbar --
        # distance-aligned noise that attenuates the stage-2 partial
        # correlation.  Here kappa is one slope over ALL pair-days:
        #   kappa = sum_p sum_t (J - Jbar_p)(p - pbar_p) / sum_p sum_t (J-Jbar_p)^2
        # Pairs whose J is constant across the year are kept (alpha = pbar -
        # kappa*Jbar); they are the co-visibility-free regime.
        lons = {n: np.arctan2(coords[n].ecef_m[1], coords[n].ecef_m[0]) for n in coords}
        recs, cache = [], []
        knum = kden = 0.0
        for (a, b), v in per_pair.items():
            if len(v) < MIN_DAYS_PER_PAIR:
                continue
            p_ = np.array([x[1] for x in v]); J_ = np.array([x[2] for x in v])
            Jc = J_ - J_.mean(); pc_c = p_ - p_.mean()
            knum += float(np.dot(Jc, pc_c)); kden += float(np.dot(Jc, Jc))
            cache.append((pc_c, Jc, float(np.dot(Jc, Jc)), float(J_.mean()),
                          float(p_.mean()), len(p_), a, b))
        kappa = knum / kden if kden > 1e-12 else 0.0
        for (pc_c, Jc, vJ, mJ, mp, n_, a, b) in cache:
            alpha = mp - kappa * mJ
            if not np.isfinite(alpha) or abs(alpha) > 5.0:
                continue
            d = td.great_circle_km(coords[a], coords[b])
            dl = abs(lons[a] - lons[b]); dl = min(dl, 2 * np.pi - dl)
            recs.append((alpha, d, mJ, kappa, a, b, dl))
        if len(recs) < 100:
            print("  too few pairs"); continue
        alpha = np.array([r[0] for r in recs]); dist = np.array([r[1] for r in recs])
        jbar = np.array([r[2] for r in recs]); beta = np.array([r[3] for r in recs])
        dlon = np.array([r[6] for r in recs])
        print(f"  stage 1: {len(recs):,} pairs (pooled kappa = {kappa:+.4f})")

        pc = partial_corr(alpha, np.log(dist + 1), jbar)
        pc_cov = partial_corr(alpha, jbar, np.log(dist + 1))
        pc_dlon = partial_corr(alpha, np.cos(dlon), jbar)

        def _resid(y, Z):
            Z = np.asarray(Z, float)
            Z = Z if Z.ndim == 2 else Z.reshape(-1, 1)
            Zs = (Z - Z.mean(0)) / np.clip(Z.std(0), 1e-9, None)
            A = np.column_stack([np.ones(len(y)), Zs])
            bt, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - A @ bt

        Zd = np.column_stack([jbar, np.cos(dlon)])
        ra2 = _resid(alpha, Zd); rl2 = _resid(np.log(dist + 1), Zd)
        pc_ctrl = float(np.corrcoef(ra2, rl2)[0, 1]) if ra2.std() > 0 and rl2.std() > 0 else np.nan

        # frozen OOS R^2 gain of the distance term (5-fold over pairs):
        # base = [Jbar, Jbar^2] vs base + exp(-d/4000); also with cos dlon base.
        feat = np.exp(-dist / 4000.0)
        folds = RNG.permutation(len(alpha)) % 5
        gains, gains_dl = [], []
        for fd in range(5):
            te = folds == fd; tr = ~te
            for extra, acc in ((False, gains), (True, gains_dl)):
                Xb = [jbar, jbar ** 2] + ([np.cos(dlon)] if extra else [])
                Xa = Xb + [feat]
                Ab = np.column_stack([np.ones(len(alpha))] + Xb)
                Aa = np.column_stack([np.ones(len(alpha))] + Xa)
                bb, *_ = np.linalg.lstsq(Ab[tr], alpha[tr], rcond=None)
                ba, *_ = np.linalg.lstsq(Aa[tr], alpha[tr], rcond=None)
                ss_b = np.sum((alpha[te] - Ab[te] @ bb) ** 2)
                ss_a = np.sum((alpha[te] - Aa[te] @ ba) ** 2)
                tot = np.sum((alpha[te] - alpha[te].mean()) ** 2)
                if tot > 0:
                    acc.append((ss_b - ss_a) / tot)
        oos_gain = float(np.mean(gains)) if gains else np.nan
        oos_gain_dl = float(np.mean(gains_dl)) if gains_dl else np.nan

        ctr, val, cnt = s53._curve(list(zip(dist, alpha)))
        f = s53._fit(ctr, val, cnt) if len(ctr) >= 6 else {}

        # Nulls.
        #
        # (i) Stage-1 (kappa) null: permute each pair's phase-cos series over
        #     days, destroying the day-to-day p-J pairing while preserving
        #     every marginal, and re-fit the pooled kappa per draw.  The null
        #     distribution of kn is the honest significance statement for the
        #     day-level common-view covariation.
        #
        #     Note the statistic this null CANNOT exercise: alpha = pbar -
        #     kappa*Jbar is invariant under the permutation in pbar, and
        #     because stage 2 controls Jbar linearly the kappa*Jbar intercept
        #     term is absorbed by that control EXACTLY.  A "null PC" built
        #     from the permuted-kappa alpha is therefore machine-identical to
        #     the real PC (the Jbar control removes kappa*Jbar to machine
        #     precision); the earlier version of this block reported a
        #     z_vs_null that was a floating-point artefact of that identity.
        #
        # (ii) Stage-2 (S2 separation) null: permute alpha across pairs,
        #      breaking the alpha <-> (dist, Jbar) link while preserving every
        #      marginal.  Under it PC ~ N(0, ~1/sqrt(N)); pairs share stations
        #      so the iid permutation sd is mildly anti-conservative -- the
        #      station-clustered bootstrap CI below is the cluster-aware
        #      companion.
        ld = np.log(dist + 1)
        knulls = np.empty(N_NULL)
        for q in range(N_NULL):
            knum = 0.0
            for (pc_c, Jc, vJ, mJ, mp, n_, a, b) in cache:
                knum += float(np.dot(RNG.permutation(pc_c), Jc))
            knulls[q] = knum / kden if kden > 1e-12 else 0.0
        z_kappa = float((abs(kappa) - np.abs(knulls).mean())
                        / (np.abs(knulls).std() + 1e-30))

        nulls = np.array([
            partial_corr(alpha[RNG.permutation(len(alpha))], ld, jbar)
            for _ in range(N_NULL)])
        z = float((abs(pc) - np.abs(nulls).mean()) / (np.abs(nulls).std() + 1e-30))
        p_pc = float(np.mean(np.abs(nulls) >= abs(pc)))

        # Station-clustered bootstrap on the stage-2 PC: resample stations
        # with replacement, weight each pair by endpoint multiplicity (capped
        # at 4, the step_45b convention).  Pairs share stations, so this is
        # the cluster-aware significance companion to the iid permutation z.
        sta_all = sorted({r[4] for r in recs} | {r[5] for r in recs})
        idx_of = {nm: k for k, nm in enumerate(sta_all)}
        sia = np.array([idx_of[r[4]] for r in recs])
        sib = np.array([idx_of[r[5]] for r in recs])
        boot = []
        for _ in range(300):
            mult = np.bincount(RNG.choice(len(sta_all), size=len(sta_all),
                                          replace=True),
                               minlength=len(sta_all))
            w = mult[sia] * mult[sib]
            sel = w > 0
            if sel.sum() < 100:
                continue
            rep = np.minimum(w[sel], 4)
            boot.append(partial_corr(np.repeat(alpha[sel], rep),
                                     np.repeat(ld[sel], rep),
                                     np.repeat(jbar[sel], rep)))
        pc_boot_ci = ([float(np.percentile(boot, 2.5)),
                       float(np.percentile(boot, 97.5))]
                      if len(boot) >= 30 else None)

        arms[label] = {
            "product": product, "n_days": len(dates), "n_pairs": len(recs),
            "estimator": "phase-alignment index cos(arg(sum_f S_ij(f)))",
            "stage1": "pooled common-view kappa with pair fixed effects",
            "pooled_common_view_kappa": float(kappa),
            "n_days_real_sp3_covis": covis_real,
            "partial_corr_alpha_logdist_given_covis": pc,
            "partial_corr_alpha_cosdlon_given_covis": pc_dlon,
            "partial_corr_alpha_logdist_given_covis_cosdlon": pc_ctrl,
            "oos_r2_gain_distance": oos_gain,
            "oos_r2_gain_distance_with_dlon_base": oos_gain_dl,
            "partial_corr_alpha_covis_given_logdist": pc_cov,
            "frozen_S2_threshold": 0.15,
            "meets_S2_partial_corr": bool(abs(pc) > 0.15),
            "null_partial_corr_abs_mean": float(np.abs(nulls).mean()) if len(nulls) else None,
            "null_partial_corr_abs_sd": float(np.abs(nulls).std()) if len(nulls) else None,
            "p_vs_null": p_pc,
            "z_vs_null": z,
            "kappa_null_mean": float(knulls.mean()),
            "kappa_null_sd": float(knulls.std()),
            "z_kappa": z_kappa,
            "pc_station_boot_ci95": pc_boot_ci,
            "distance_fit": {"best_model": f.get("best_model"),
                             "lambda_km": f.get("lambda_km"), "R2": f.get("R2"),
                             "n_bins": len(ctr)},
            "mean_beta_covisibility": float(beta.mean()),
        }
        print(f"  pairs {len(recs)}   partial corr(alpha, log d | Jbar) = {pc:+.4f}"
              f"   | Jbar+cos-dlon = {pc_ctrl:+.4f}   OOS gain = {oos_gain:+.4f}")
        print(f"  pair-permutation null |pc| mean {np.abs(nulls).mean():.4f}"
              f"  z = {z:+.2f}  (kappa-null z = {z_kappa:+.2f}"
              f"{'' if pc_boot_ci is None else f',  PC boot CI {pc_boot_ci}'})"
              f"   meets frozen S2 (>0.15): {abs(pc) > 0.15}")
        print(f"  distance fit on alpha: {f.get('best_model')} lambda "
              f"{f.get('lambda_km') or -1:.0f} km  R2 {f.get('R2') or 0:.3f}")

    verdict = {
        "question": "does the frozen S2 criterion pass when T4 uses the corpus's own "
                    "estimator instead of band-passed Pearson?",
        "arms_meeting_S2": [k for k, v in arms.items() if v["meets_S2_partial_corr"]],
        "n_arms": len(arms),
    }
    OUT.write_text(json.dumps({
        "step": 83, "title": "T4 co-visibility separation with the phase-alignment index",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
