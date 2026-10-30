#!/usr/bin/env python3
"""Step 56: causal phase-cos -> operational outage-prediction bridge.

The phase prior is frozen from the complete *preceding* UTC day.  During the
current-day simulated receiver outage, only live reference clocks are used.
The predictive phase matrix is coherence*cos(phase), a normalized spectral
Gram matrix; its Hadamard product with the fixed TEP distance covariance is
PSD by construction.  Phase-label and distance-label scrambles are null arms.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.lib.tep_phase import (  # noqa: E402
    spectral_phase_gram, phase_distance_weights, phase_randomised_real_gram,
)
from scripts.steps.step_55_dense_outage_prediction import (  # noqa: E402
    DT_S, PREWINDOW_S, MIN_REFS, NUGGET as DISTANCE_NUGGET, _solve_weights,
    _causal_spatial_weights, load_day,
)

OUT = ROOT / "results" / "outputs" / "step_56_phase_predictive_bridge.json"
GAPS_S = [1800.0, 3600.0, 14400.0, 28800.0]
SUPPLEMENTAL_GAPS_S = [600.0, 7200.0]
ALL_GAPS_S = sorted(GAPS_S + SUPPLEMENTAL_GAPS_S)
N_DAYS = int(os.environ.get("TEP_PHASE_DAYS", "45"))
STRIDE = int(os.environ.get("TEP_PHASE_STRIDE", "6"))
RNG = np.random.default_rng(20260902)
RNG_SUPP = np.random.default_rng(20260903)
# Phase-domain nuisance fixed during 2024 method development.  It is distinct
# from the 0.05 distance-only nugget selected by Step 25 and must remain frozen
# for an independent-year validation.
PHASE_NUGGET = 0.35
KEYS = ("best_temporal", "net_mean", "idw2", "causal_cv_spatial", "tep",
        "phase_tep", "coherence_only_tep", "phase_only", "phase_scramble",
        "phase_angle_scramble", "phase_nugget_015", "phase_nugget_070",
        "phase_nugget_127", "phase_1e4348", "phase_1e4200", "distance_scramble")


def _previous_tag(tag):
    return (datetime.strptime(tag, "%Y%j") - timedelta(days=1)).strftime("%Y%j")


def _phase_prior(prev, current_names):
    pnames, _, praw, _ = prev
    common = [n for n in current_names if n in pnames]
    cols, kept = [], []
    for name in common:
        y = praw[:, pnames.index(name)].astype(float)
        good = np.isfinite(y)
        if good.mean() < 0.95 or good.sum() < 8:
            continue
        if not good.all():
            ix = np.arange(len(y))
            y = np.interp(ix, ix[good], y[good])
        cols.append(y)
        kept.append(name)
    if len(kept) < MIN_REFS + 1:
        return None
    result = spectral_phase_gram(np.column_stack(cols), DT_S)
    return kept, result


def _weights(D, b, pool, G, coherence2, angle_scrambled_G, gi, phase_perm,
             distance_perm):
    drr, dt = D[np.ix_(pool, pool)], D[b, pool]
    base = _solve_weights(drr, dt)
    gpool = [gi[p] for p in pool]
    gb = gi[b]
    phase = phase_distance_weights(
        drr, dt, G[np.ix_(gpool, gpool)], G[gb, gpool], nugget=PHASE_NUGGET)
    phase015 = phase_distance_weights(drr, dt, G[np.ix_(gpool, gpool)],
                                      G[gb, gpool], nugget=0.15)
    phase070 = phase_distance_weights(drr, dt, G[np.ix_(gpool, gpool)],
                                      G[gb, gpool], nugget=0.70)
    phase127 = phase_distance_weights(drr, dt, G[np.ix_(gpool, gpool)],
                                      G[gb, gpool], nugget=1.27)
    phase_1e4348 = phase_distance_weights(
        drr, dt, G[np.ix_(gpool, gpool)], G[gb, gpool],
        lambda_km=4347.65 / np.sqrt(2.0), nugget=PHASE_NUGGET)
    phase_1e4200 = phase_distance_weights(
        drr, dt, G[np.ix_(gpool, gpool)], G[gb, gpool],
        lambda_km=4200.0 / np.sqrt(2.0), nugget=PHASE_NUGGET)
    coh = phase_distance_weights(drr, dt, coherence2[np.ix_(gpool, gpool)],
                                 coherence2[gb, gpool])
    only = phase_distance_weights(drr * 0.0, dt * 0.0,
                                  G[np.ix_(gpool, gpool)], G[gb, gpool],
                                  nugget=PHASE_NUGGET)
    ps = phase_distance_weights(
        drr, dt, G[np.ix_(phase_perm[gpool], phase_perm[gpool])],
        G[phase_perm[gb], phase_perm[gpool]])
    pas = phase_distance_weights(
        drr, dt, angle_scrambled_G[np.ix_(gpool, gpool)],
        angle_scrambled_G[gb, gpool])
    ds = _solve_weights(D[np.ix_(distance_perm[pool], distance_perm[pool])],
                        D[distance_perm[b], distance_perm[pool]])
    return dict(tep=base, phase_tep=phase, coherence_only_tep=coh, phase_only=only,
                phase_scramble=ps, phase_angle_scramble=pas,
                phase_nugget_015=phase015, phase_nugget_070=phase070,
                phase_nugget_127=phase127,
                phase_1e4348=phase_1e4348, phase_1e4200=phase_1e4200,
                distance_scramble=ds)


def run(days):
    npre = int(PREWINDOW_S / DT_S)
    all_sq = {g: {k: [] for k in KEYS} for g in ALL_GAPS_S}
    day_mse = {g: {k: [] for k in KEYS} for g in ALL_GAPS_S}
    day_q95 = {g: {k: [] for k in KEYS} for g in ALL_GAPS_S}
    day_q99 = {g: {k: [] for k in KEYS} for g in ALL_GAPS_S}
    day_n = {g: [] for g in ALL_GAPS_S}
    target_reset_like_events = {g: 0 for g in ALL_GAPS_S}
    target_reset_like_examples = {g: [] for g in ALL_GAPS_S}
    phase_summaries = []

    for tag in days:
        cur, prev = load_day(tag), load_day(_previous_tag(tag))
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        prior = _phase_prior(prev, names)
        if prior is None:
            continue
        phase_names, pr = prior
        eligible = [i for i, n in enumerate(names) if n in phase_names]
        if len(eligible) < MIN_REFS + 1:
            continue
        gi = {i: phase_names.index(names[i]) for i in eligible}
        G = pr["predictive_gram"]
        coherence2 = pr["coherence"] ** 2
        # coherence**2 is not guaranteed PSD (coherence is a magnitude, not a
        # Gram).  Project to nearest PSD matrix for the coherence_only_tep arm
        # so the kriging system is well-conditioned.  The phase_tep arm uses
        # the predictive_gram which IS a normalised spectral Gram (PSD by
        # construction), so no projection is needed there.
        coh_evals, coh_evecs = np.linalg.eigh(coherence2)
        coh_evals = np.maximum(coh_evals, 0.0)
        coherence2 = (coh_evecs * coh_evals) @ coh_evecs.T
        theta = RNG.uniform(-np.pi, np.pi, len(phase_names))
        angle_scrambled_G = phase_randomised_real_gram(
            pr["complex_coherence"], theta)
        phase_perm = RNG.permutation(len(phase_names))
        distance_perm = RNG.permutation(len(names))
        tri = np.triu_indices(len(phase_names), 1)
        phase_summaries.append({
            "day": tag, "stations": len(phase_names), "n_bins": pr["n_bins"],
            "history_h": pr["duration_s"] / 3600.0,
            "mean_phase_cos": float(np.mean(pr["phase_cos"][tri])),
            "mean_coherence_cos": float(np.mean(G[tri])),
        })
        local = {g: {k: [] for k in KEYS} for g in ALL_GAPS_S}

        for ti in range(npre, len(t) - 1, STRIDE):
            sl = slice(ti - npre, ti + 1)
            tt = t[sl] - t[ti]
            A = np.column_stack((tt, np.ones_like(tt)))
            pre = raw[sl]
            sd = np.full(len(names), np.inf)
            for q in eligible:
                if np.isfinite(pre[:, q]).all():
                    fit = np.polyfit(tt, pre[:, q], 1)
                    sd[q] = np.std(pre[:, q] - np.polyval(fit, tt))
            finite = sd[np.isfinite(sd) & (sd > 0.01)]
            if len(finite) < MIN_REFS + 1:
                continue
            med = np.median(finite)
            good = (sd > 0.01) & (sd < 60.0) & (sd < 4.0 * med)
            for b in eligible:
                if not good[b]:
                    continue
                pool = [j for j in eligible if j != b and good[j]]
                if len(pool) < MIN_REFS:
                    continue
                y, Y = raw[sl, b], raw[sl][:, pool]
                if not np.isfinite(y).all() or not np.isfinite(Y).all():
                    continue
                bt = np.polyfit(tt, y, 1)
                coef = np.linalg.lstsq(A, Y, rcond=None)[0]
                rok = np.abs(coef[0]) <= 0.20
                if abs(bt[0]) > 0.20 or rok.sum() < MIN_REFS:
                    continue
                pool = [p for p, ok in zip(pool, rok) if ok]
                Y = raw[sl][:, pool]
                coef = np.linalg.lstsq(A, Y, rcond=None)[0]
                rb = y - np.polyval(bt, tt)
                ref_hist = Y - A @ coef
                r0, r1 = rb[:-1], rb[1:]
                phi = np.clip(np.sum(r0 * r1) / max(np.sum(r0 * r0), 1e-12),
                              -0.999, 0.999)
                # Horizon-matched temporal selection (see Step 55 for the full
                # rationale): one-step CV systematically favours AR(1) because
                # phi<1 shrinks the one-step prediction, but at the actual
                # prediction horizon (30 min - 8 h) the phi**k decay accumulates
                # and persistence is often better.  Select per-gap using a
                # horizon-matched CV on the pre-window residuals.
                npre_rb = len(rb)

                def _horizon_use_ar1(k_step):
                    best_h = min(k_step, npre_rb - 3)
                    if best_h < 1:
                        return False
                    origins = np.arange(0, npre_rb - best_h)
                    if len(origins) < 3:
                        return False
                    pe = rb[origins + best_h] - rb[origins]
                    ae = rb[origins + best_h] - (phi ** best_h) * rb[origins]
                    return float(np.mean(ae ** 2)) < float(np.mean(pe ** 2))
                try:
                    weights = _weights(D, b, pool, G, coherence2,
                                       angle_scrambled_G, gi,
                                       phase_perm, distance_perm)
                except np.linalg.LinAlgError:
                    continue
                wcv, _ = _causal_spatial_weights(
                    ref_hist, rb, D[np.ix_(pool, pool)], D[b, pool])
                if wcv is None:
                    continue
                inv2 = 1.0 / np.maximum(D[b, pool], 1.0) ** 2
                weights["idw2"] = inv2 / inv2.sum()
                weights["causal_cv_spatial"] = wcv
                for gap in ALL_GAPS_S:
                    k = int(round(gap / DT_S))
                    if ti + k >= len(t) or not np.isfinite(raw[ti + k, b]):
                        continue
                    rr = raw[ti + k, pool] - (coef[0] * gap + coef[1])
                    if not np.isfinite(rr).all():
                        continue
                    rmed = np.median(rr)
                    rmad = 1.4826 * np.median(np.abs(rr - rmed))
                    ok = np.abs(rr - rmed) <= max(10.0, 8.0 * rmad)
                    if ok.sum() < MIN_REFS:
                        continue
                    weights_g = weights
                    if not ok.all():
                        pool2 = [p for p, z in zip(pool, ok) if z]
                        rr = rr[ok]
                        try:
                            weights_g = _weights(D, b, pool2, G, coherence2,
                                                 angle_scrambled_G, gi,
                                                 phase_perm, distance_perm)
                        except np.linalg.LinAlgError:
                            continue
                        wcv_g, _ = _causal_spatial_weights(
                            ref_hist[:, ok], rb, D[np.ix_(pool2, pool2)], D[b, pool2])
                        if wcv_g is None:
                            continue
                        inv2_g = 1.0 / np.maximum(D[b, pool2], 1.0) ** 2
                        weights_g["idw2"] = inv2_g / inv2_g.sum()
                        weights_g["causal_cv_spatial"] = wcv_g
                    truth = raw[ti + k, b]
                    base = np.polyval(bt, gap)
                    preds = {name: base + float(w @ rr) for name, w in weights_g.items()}
                    persistence = base + rb[-1]
                    ar1 = base + rb[-1] * phi ** k
                    preds["best_temporal"] = ar1 if _horizon_use_ar1(k) else persistence
                    preds["net_mean"] = base + float(np.mean(rr))
                    # Predictor-independent causal screen from pre-outage target
                    # data.  A reset-like future truth is a DATA-VALIDITY problem
                    # on the observable, not a performance filter: clock
                    # PREDICTION skill across a gap is undefined when the clock
                    # is discontinuous, and every arm fails by the full jump
                    # magnitude, which then dominates the pooled sum-of-squares
                    # and collapses every relative comparison to ~0% (this is
                    # exactly the step_55 MAX_PLAUSIBLE_CLOCK_M / reset-exclusion
                    # bug -- 5 km ~ 17 us is far beyond the p99.9 holdover error
                    # ~520 m and far below reset magnitudes ~3-5e5 m).  Counted
                    # transparently as target_reset_like_events, then EXCLUDED.
                    pre_jump = float(abs(rb[-1] - rb[-max(2, int(600.0 / DT_S))]))
                    if pre_jump > 200.0:
                        continue
                    if abs(truth - base) > 5000.0:   # empty band: legit p99.9 ~520 m, resets ~1e6 m
                        target_reset_like_events[gap] += 1
                        path = raw[ti:ti + k + 1, b]
                        path_diff = np.diff(path)
                        if np.isfinite(path_diff).any():
                            jump_ix = int(np.nanargmax(np.abs(path_diff)))
                            max_step = float(path_diff[jump_ix])
                            max_step_s = float(t[ti + jump_ix + 1])
                        else:
                            max_step = None
                            max_step_s = None
                        target_reset_like_examples[gap].append({
                            "day": str(tag), "station": names[b],
                            "anchor_s": float(t[ti]),
                            "prediction_s": float(t[ti + k]),
                            "gap_s": float(gap),
                            "pretrend_slope_mps": float(bt[0]),
                            "truth_m": float(truth),
                            "pretrend_prediction_m": float(base),
                            "endpoint_deviation_m": float(truth - base),
                            "max_observed_step_m": max_step,
                            "max_observed_step_epoch_s": max_step_s,
                        })
                        continue
                    for name, pred in preds.items():
                        local[gap][name].append(float((truth - pred) ** 2))

        for gap in ALL_GAPS_S:
            if len(local[gap]["tep"]) < 25:
                continue
            day_n[gap].append(len(local[gap]["tep"]))
            for key in KEYS:
                values = np.asarray(local[gap][key], float)
                all_sq[gap][key].extend(values.tolist())
                day_mse[gap][key].append(float(np.mean(values)))
                abs_err = np.sqrt(values)
                day_q95[gap][key].append(float(np.percentile(abs_err, 95)))
                day_q99[gap][key].append(float(np.percentile(abs_err, 99)))

    def _summarize_gaps(gap_list, rng):
        out = {}
        for gap in gap_list:
            if not all_sq[gap]["tep"]:
                continue
            rms = {k: float(np.sqrt(np.mean(all_sq[gap][k]))) for k in KEYS}
            p95 = {k: float(np.sqrt(np.percentile(all_sq[gap][k], 95))) for k in KEYS}
            p99 = {k: float(np.sqrt(np.percentile(all_sq[gap][k], 99))) for k in KEYS}
            counts = np.asarray(day_n[gap], float)
            def ci(num, den):
                a, b = np.asarray(day_mse[gap][num]), np.asarray(day_mse[gap][den])
                vals = []
                for ix in (rng.integers(0, len(a), len(a)) for _ in range(4000)):
                    vals.append(100 * (1 - np.sqrt(np.average(a[ix], weights=counts[ix])) /
                                       np.sqrt(np.average(b[ix], weights=counts[ix]))))
                return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

            def quantile_ci(num, den, quantile):
                source = day_q95 if quantile == 95 else day_q99
                a = np.asarray(source[gap][num], float)
                b = np.asarray(source[gap][den], float)
                gain = 100.0 * (1.0 - a / b)
                vals = [float(np.median(rng.choice(gain, len(gain), replace=True)))
                        for _ in range(4000)]
                return {
                    "estimand": f"median per-day p{quantile} relative improvement",
                    "median_pct": float(np.median(gain)),
                    "ci95": [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))],
                    "n_days": int(len(gain)),
                }
            out[str(int(gap))] = {
                "n_eval": len(all_sq[gap]["tep"]), "n_days": len(counts),
                "rms_m": rms, "p95_m": p95, "p99_m": p99,
                "phase_tep_vs_tep_pct": 100 * (1 - rms["phase_tep"] / rms["tep"]),
                "phase_tep_vs_tep_ci95": ci("phase_tep", "tep"),
                "phase_tep_vs_phase_scramble_pct": 100 * (1 - rms["phase_tep"] / rms["phase_scramble"]),
                "phase_tep_vs_phase_scramble_ci95": ci("phase_tep", "phase_scramble"),
                "phase_tep_vs_coherence_only_pct": 100 * (1 - rms["phase_tep"] / rms["coherence_only_tep"]),
                "phase_tep_vs_coherence_only_ci95": ci("phase_tep", "coherence_only_tep"),
                "phase_tep_vs_phase_only_pct": 100 * (1 - rms["phase_tep"] / rms["phase_only"]),
                "phase_tep_vs_phase_only_ci95": ci("phase_tep", "phase_only"),
                "phase_tep_vs_phase_angle_scramble_pct": 100 * (1 - rms["phase_tep"] / rms["phase_angle_scramble"]),
                "phase_tep_vs_phase_angle_scramble_ci95": ci("phase_tep", "phase_angle_scramble"),
                "phase_tep_vs_best_temporal_pct": 100 * (1 - rms["phase_tep"] / rms["best_temporal"]),
                "phase_tep_vs_best_temporal_ci95": ci("phase_tep", "best_temporal"),
                "phase_tep_vs_best_temporal_p95_pct": 100 * (
                    1 - p95["phase_tep"] / p95["best_temporal"]),
                "phase_tep_vs_best_temporal_p99_pct": 100 * (
                    1 - p99["phase_tep"] / p99["best_temporal"]),
                "phase_tep_vs_best_temporal_p95_daily_cluster_ci": quantile_ci(
                    "phase_tep", "best_temporal", 95),
                "phase_tep_vs_best_temporal_p99_daily_cluster_ci": quantile_ci(
                    "phase_tep", "best_temporal", 99),
                "phase_tep_vs_net_mean_pct": 100 * (1 - rms["phase_tep"] / rms["net_mean"]),
                "phase_tep_vs_net_mean_ci95": ci("phase_tep", "net_mean"),
                "phase_tep_vs_causal_cv_spatial_pct": 100 * (1 - rms["phase_tep"] / rms["causal_cv_spatial"]),
                "phase_tep_vs_causal_cv_spatial_ci95": ci("phase_tep", "causal_cv_spatial"),
                "nugget_sensitivity_rms_m": {
                    "0.15": rms["phase_nugget_015"], "0.35": rms["phase_tep"],
                    "0.70": rms["phase_nugget_070"], "1.27": rms["phase_nugget_127"],
                },
                "target_reset_like_events_scored": int(target_reset_like_events[gap]),
                "target_reset_like_event_rate": float(
                    target_reset_like_events[gap] / len(all_sq[gap]["tep"])),
                "target_reset_like_examples": target_reset_like_examples[gap],
            }
        return out

    by_gap = _summarize_gaps(GAPS_S, RNG)
    supplemental_by_gap = _summarize_gaps(SUPPLEMENTAL_GAPS_S, RNG_SUPP)
    return by_gap, supplemental_by_gap, phase_summaries


def main():
    dates = td.available_dates("precise", 2024)
    available = set(dates)
    candidates = [d for d in dates if _previous_tag(d) in available]
    days = [candidates[i] for i in np.linspace(0, len(candidates) - 1, N_DAYS).astype(int)]
    by_gap, supplemental_by_gap, priors = run(days)
    payload = {
        "step": 56, "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "2024_development_benchmark_requires_independent_year",
        "design": "preceding-day frozen 10-500 uHz phase/coherence Gram x fixed TEP distance covariance; current-day causal outages",
        "nugget_provenance": ("phase nugget 0.35 was frozen during 2024 method development; "
                               "the distance-only Step 25-selected nugget is 0.05. Neither is a "
                               "TEP physics prediction. Fixed sensitivity arms 0.15, 0.70 and "
                               "1.27 are scored on identical support."),
        "phase_nugget": PHASE_NUGGET,
        "distance_nugget": DISTANCE_NUGGET,
        "target_history_mode": ("calibrated-client: the preceding-day Gram includes the target "
                                "receiver; this is not a target-blind predictor"),
        "days_requested": days, "phase_prior_summaries": priors,
        "by_gap": by_gap,
        "supplemental_by_gap": supplemental_by_gap,
        "lambda_convention_sensitivity_rms_m": {
            "note": "Comparison of exp(-0.5*(d/lambda_op)^2) at lambda_op=4200 km (lambda_1/e=5940 km) vs 1/e-matched lambda_1/e=4348 km (lambda_op=3074 km) and lambda_1/e=4200 km (lambda_op=2970 km) across all 6 horizons on the identical 45-day support.",
            **{k: {"lam_op_4200_1e_5940": round(v["rms_m"]["phase_tep"], 4),
                   "lam_1e_4348_op_3074": round(v["rms_m"]["phase_1e4348"], 4),
                   "lam_1e_4200_op_2970": round(v["rms_m"]["phase_1e4200"], 4)}
               for k, v in sorted({**by_gap, **supplemental_by_gap}.items(), key=lambda kv: int(kv[0]))}
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
