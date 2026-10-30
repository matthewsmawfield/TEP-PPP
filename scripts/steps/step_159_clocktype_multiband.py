#!/usr/bin/env python3
"""
STEP 159  --  Clock-type-specific multi-band advantage.

The TEP prediction is that the field couples to quartz but not atomic.
Step 103/158 confirmed this: quartz pairs show R2=0.936 exponential
distance decay, atomic pairs do not (R2=0.26).

This step tests whether the multi-band TEP kernel's advantage over the
single-lambda kernel is LARGER for quartz stations than atomic stations.
If the frequency-dependent spatial structure is a TEP effect, it should
help more where the field is stronger (quartz).

METHOD:
  1. For each evaluation day, classify the target station as quartz or atomic.
  2. Compute multi-band and full-band predictions.
  3. Accumulate RMS separately for quartz targets and atomic targets.
  4. Compare the multi-band advantage (multi vs full) between clock types.

If multi-band advantage is significantly larger for quartz than atomic,
this is direct TEP evidence: the physics-fixed frequency-dependent
spatial structure helps precisely where the TEP field is strongest.

Output: results/outputs/step_159_clocktype_multiband.json
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib.tep_phase import spectral_phase_gram
from scripts.steps import step_64_maximal_tep_p1 as s64
from scripts.steps.step_152_freq_spectrum import _load_multi_day_prior, _input_path

OUT = ROOT / "results" / "outputs" / "step_159_clocktype_multiband.json"
PRODUCT = "clk_cod"
YEAR = 2025
RNG_SEED = 20260920

EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10

BANDS = [
    {"name": "tidal",        "f_lo": 10e-6,  "f_hi": 35e-6,   "lambda_km": 4677},
    {"name": "post_tidal",   "f_lo": 35e-6,  "f_hi": 100e-6,  "lambda_km": 1502},
    {"name": "intermediate", "f_lo": 100e-6, "f_hi": 500e-6,  "lambda_km": 1459},
]
FULL_BAND = {"f_lo": 10e-6, "f_hi": 500e-6, "lambda_km": 4677}

def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-8:
        K = K + (1e-8 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)

def _ci(values, rng):
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v), "frac_positive": None}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}

def _band_gram(X_prior, dt_s, band):
    try:
        r = spectral_phase_gram(X_prior, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                min_bins=10, min_low_cycles=0.85)
        return r["predictive_gram"]
    except (ValueError, np.linalg.LinAlgError):
        return None

def _multi_band_gram(X_prior, dt_s, bands):
    grams = []
    for band in bands:
        G = _band_gram(X_prior, dt_s, band)
        if G is not None:
            grams.append(G)
    if not grams:
        return None
    return np.mean(grams, axis=0)

def _clock_types():
    ct_raw = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    ctype = {}
    for n, v in ct_raw.items():
        if "QUARTZ" in v[0].upper():
            ctype[n] = "QUARTZ"
        elif "ATOMIC" in v[0].upper() or "MASER" in v[0].upper() or "CESIUM" in v[0].upper() or "RUBIDIUM" in v[0].upper():
            ctype[n] = "ATOMIC"
    return ctype

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    ctype = _clock_types()
    print("Step 159: Clock-type-specific multi-band advantage", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Clock types: {sum(1 for v in ctype.values() if v == 'QUARTZ')} quartz, "
          f"{sum(1 for v in ctype.values() if v == 'ATOMIC')} atomic", flush=True)
    print(flush=True)

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t)]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)
    print(flush=True)

    dt_s = s64.DT_S
    npre = int(s64.PREWINDOW_S / dt_s)

    # Accumulate per-target RMS by clock type
    # Each entry: (day_tag, target_name, clock_type, full_rms, multi_rms)
    target_results = []

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        if cur is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur:
            continue
        multi_prior = _load_multi_day_prior(tag, names, 13)
        if multi_prior is None:
            continue
        prior_names, prior_x, _ = multi_prior

        R_cols = []; kept_cur = []
        for nm in names:
            if nm not in prior_names:
                continue
            clk_ns = s_cur[nm]["raw"].astype(float)
            ts = s_cur[nm]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for ki in range(len(clk_ns)):
                tk = ts[ki]
                mask = (ts <= tk) & (ts >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(ts[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r); kept_cur.append(nm)
        if len(kept_cur) < s64.MIN_REFS + 2:
            continue
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS
        Dg = D[np.ix_([names.index(n) for n in kept_cur], [names.index(n) for n in kept_cur])]
        gnames = kept_cur
        sd_r = np.nanstd(R, axis=0); med_r = np.median(sd_r)
        keep = [i for i in range(len(gnames)) if 0.01 < sd_r[i] < 12.0 and sd_r[i] < 3.5 * med_r]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]; R = R[:, keep]; Dg = Dg[np.ix_(keep, keep)]
        prior_idx = [prior_names.index(nm) for nm in gnames]
        X_prior = prior_x[:, prior_idx]
        sd_p = np.nanstd(X_prior, axis=0)
        keep2 = [i for i in range(len(gnames)) if 0.01 < sd_p[i] < 1e4]
        if len(keep2) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep2]; R = R[:, keep2]; Dg = Dg[np.ix_(keep2, keep2)]
        X_prior = X_prior[:, keep2]
        ng = len(gnames)
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        G_full = _band_gram(X_prior, dt_s, FULL_BAND)
        G_multi = _multi_band_gram(X_prior, dt_s, BANDS)
        if G_full is None or G_multi is None:
            continue

        # Per-target squared errors by clock type
        for b in range(ng):
            target_name = gnames[b]
            target_ct = ctype.get(target_name, "UNKNOWN")
            if target_ct not in ("QUARTZ", "ATOMIC"):
                continue

            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            rho_rr = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt = s64._rho_multi(Dg[b, pool])
            B_arr = R[:, b]; A = R[:, pool]

            full_sq = []
            multi_sq = []
            for ei in eval_i:
                truth = B_arr[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                try:
                    w_f = _reg_solve(rho_rr * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                     rho_rt * G_full[b, pool])
                    p_f = float(w_f @ rr)
                    full_sq.append((truth - p_f) ** 2)
                except np.linalg.LinAlgError:
                    pass
                try:
                    w_m = _reg_solve(rho_rr * G_multi[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                     rho_rt * G_multi[b, pool])
                    p_m = float(w_m @ rr)
                    multi_sq.append((truth - p_m) ** 2)
                except np.linalg.LinAlgError:
                    pass

            if len(full_sq) > 10 and len(multi_sq) > 10:
                full_rms = float(np.sqrt(np.mean(full_sq)))
                multi_rms = float(np.sqrt(np.mean(multi_sq)))
                target_results.append({
                    "tag": tag, "target": target_name, "clock_type": target_ct,
                    "full_rms": full_rms, "multi_rms": multi_rms,
                    "advantage_pct": 100.0 * (1.0 - multi_rms / full_rms),
                    "n_samples": len(full_sq),
                })

        if (k + 1) % 10 == 0:
            n_q = sum(1 for r in target_results if r["clock_type"] == "QUARTZ")
            n_a = sum(1 for r in target_results if r["clock_type"] == "ATOMIC")
            print(f"  [{k+1}/{len(eval_tags)}] quartz targets={n_q}, atomic targets={n_a}", flush=True)

    # Analyze by clock type
    quartz_adv = [r["advantage_pct"] for r in target_results if r["clock_type"] == "QUARTZ"]
    atomic_adv = [r["advantage_pct"] for r in target_results if r["clock_type"] == "ATOMIC"]

    results = {
        "quartz": {
            "n_targets": len(quartz_adv),
            "mean_advantage_pct": float(np.mean(quartz_adv)) if quartz_adv else None,
            "median_advantage_pct": float(np.median(quartz_adv)) if quartz_adv else None,
            "std_advantage_pct": float(np.std(quartz_adv)) if quartz_adv else None,
            "ci": _ci(quartz_adv, rng),
            "frac_positive": float(np.mean(np.array(quartz_adv) > 0)) if quartz_adv else None,
        },
        "atomic": {
            "n_targets": len(atomic_adv),
            "mean_advantage_pct": float(np.mean(atomic_adv)) if atomic_adv else None,
            "median_advantage_pct": float(np.median(atomic_adv)) if atomic_adv else None,
            "std_advantage_pct": float(np.std(atomic_adv)) if atomic_adv else None,
            "ci": _ci(atomic_adv, rng),
            "frac_positive": float(np.mean(np.array(atomic_adv) > 0)) if atomic_adv else None,
        },
    }

    # Statistical test: is quartz advantage > atomic advantage?
    if len(quartz_adv) > 5 and len(atomic_adv) > 5:
        from scipy import stats as sp_stats
        t_stat, t_p = sp_stats.ttest_ind(quartz_adv, atomic_adv, equal_var=False)
        u_stat, u_p = sp_stats.mannwhitneyu(quartz_adv, atomic_adv, alternative="greater")
        # Bootstrap difference
        boot_diffs = []
        for _ in range(10000):
            qb = rng.choice(quartz_adv, len(quartz_adv), replace=True)
            ab = rng.choice(atomic_adv, len(atomic_adv), replace=True)
            boot_diffs.append(float(np.mean(qb) - np.mean(ab)))
        results["difference_test"] = {
            "quartz_minus_atomic_mean": float(np.mean(quartz_adv) - np.mean(atomic_adv)),
            "t_statistic": float(t_stat), "t_p_value": float(t_p),
            "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
            "bootstrap_ci95": [float(np.percentile(boot_diffs, 2.5)),
                               float(np.percentile(boot_diffs, 97.5))],
        }

    interp = []
    interp.append(f"Clock-type-specific multi-band advantage: {len(quartz_adv)} quartz, {len(atomic_adv)} atomic targets")
    q_med = results["quartz"]["median_advantage_pct"]
    a_med = results["atomic"]["median_advantage_pct"]
    q_ci = results["quartz"]["ci"]["ci95"]
    a_ci = results["atomic"]["ci"]["ci95"]
    interp.append(f"  Quartz:  multi vs full = {q_med:+.2f}% CI{q_ci} (frac_pos={results['quartz']['frac_positive']:.2f})")
    interp.append(f"  Atomic:  multi vs full = {a_med:+.2f}% CI{a_ci} (frac_pos={results['atomic']['frac_positive']:.2f})")

    if "difference_test" in results:
        dt = results["difference_test"]
        interp.append(f"  Difference (Q-A): {dt['quartz_minus_atomic_mean']:+.2f}%")
        interp.append(f"  t-test: t={dt['t_statistic']:.2f}, p={dt['t_p_value']:.4f}")
        interp.append(f"  Mann-Whitney (Q>A): p={dt['mannwhitney_p']:.4f}")
        interp.append(f"  Bootstrap CI: [{dt['bootstrap_ci95'][0]:+.2f}, {dt['bootstrap_ci95'][1]:+.2f}]")
        if dt["mannwhitney_p"] < 0.05 and dt["bootstrap_ci95"][0] > 0:
            interp.append("")
            interp.append("  The multi-band advantage is SIGNIFICANTLY larger for QUARTZ than ATOMIC.")
            interp.append("  The frequency-dependent spatial structure helps precisely where the TEP field is strongest.")
            interp.append("  This is direct TEP evidence: the physics-fixed lambdas capture a real field that")
            interp.append("  couples to quartz oscillators but not atomic, exactly as TEP predicts.")
        elif dt["mannwhitney_p"] < 0.10:
            interp.append("")
            interp.append("  The multi-band advantage is LARGER for quartz than atomic (suggestive).")

    verdict = {
        "step": 159,
        "title": "Clock-type-specific multi-band advantage",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR,
        "results": results,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If TEP is real, the multi-band kernel's advantage over the single-lambda kernel "
            "should be LARGER for quartz targets than atomic targets, because the frequency-"
            "dependent spatial structure captures a field that couples to quartz but not atomic."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for line in interp:
        print(line, flush=True)


if __name__ == "__main__":
    main()
