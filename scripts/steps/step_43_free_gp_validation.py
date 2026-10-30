#!/usr/bin/env python3
"""
STEP 43 -- Free-GP scale-discovery validation: injection/recovery,
           baseline-restricted selection, and null controls.

Red-team concern N2 [PARTIAL]: the unconstrained-GP scale-discovery result
is framed as a "key physics finding" but is post-hoc, the precise-product
free GP selects lambda ~ 700 km (below the TEP-predicted 1,000-5,000 km
band), and "a free GP selecting a finite length scale is what GPs do on
any spatially correlated data."

This step provides three direct tests that turn the free-GP observation
from a post-hoc reframing into a validated scale-discovery result:

  1. INJECTION/RECOVERY.  A synthetic TEP signal with a known lambda is
     injected into the actual precise-product residual stream (which has
     the satellite common-mode removed and therefore has the smallest
     natural spatial signal).  The free GP is then run with no TEP prior
     and asked to recover the scale.  If the free GP recovers the injected
     lambda across the 1,000-5,000 km band, it is a validated
     scale-discovery tool; the 700 km selection on the un-injected precise
     product is then interpretable as the scale of the residual receiver
     noise that dominates when the TEP signal is weak, not as a
     falsification of the TEP scale.

  2. BASELINE-RESTRICTED FREE GP.  The 700 km selection on the precise
     product is driven by the many short baselines (< 1,000 km) in the
     European dense network, where receiver-specific noise and local
     effects dominate the small precise residual.  Restricting the free
     GP to baselines > 1,000 km and > 2,000 km (where the TEP signal
     should dominate over short-baseline noise) tests whether the scale
     selection moves into the TEP-predicted band.

  3. NULL CONTROL.  The free GP is run on temporally shuffled data (no
     spatial signal) and on geometry-shuffled data (spatial signal
     destroyed).  If the free GP selects a consistent finite scale on
     these nulls, the red-team objection ("any spatially correlated data
     gives a finite scale") stands.  If it does not -- if the null
     selections are uniformly distributed across the grid or collapse to
     the grid edges -- then the free GP's finite-scale selection on the
     real data is informative and not a generic property of GPs.

The TEP-predicted 1,000-5,000 km band was specified in the prior series
(Papers 1-2) BEFORE this analysis.  The free GP is the unconstrained
estimator; finding that it lands in the pre-specified band on the
broadcast product, and recovers injected TEP scales on the precise
product, is confirmatory -- not post-hoc.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td                       # noqa: E402
from scripts.lib import tep_stats as ts                      # noqa: E402
from scripts.lib import tep_kernel as tk                     # noqa: E402
from scripts.steps.step_34_heldout_lodo import day_matrix    # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"

# Free-GP grid -- identical to step_38 so the comparison is apples-to-apples.
FREE_FORMS = ("exponential", "gaussian", "matern15")
FREE_LAMBDA_KM = (100.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0)
_NUG = (0.05, 0.35, 1.27)
_KG = (8, 20, 40)

# TEP-predicted band (specified in Papers 1-2, not chosen here).
TEP_BAND_LO = 1000.0
TEP_BAND_HI = 5000.0

# Injection scales to test (span the TEP band and below).
INJECT_LAMBDA_KM = (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)

# Days to use for the precise product (evenly-spaced 30-day subset, as in
# step_40).
PRECISE_PRODUCT = "clk_cod"


def _select_free_gp(X, D, train_idx):
    """Run the unconstrained GP on the training set and return
    (KernelParams, train_LOO_median_RMS).  Identical to step_38's free arm."""
    p, err = tk.fit_hyperparameters(
        X, D, train_idx, lam_grid=FREE_LAMBDA_KM, form_grid=FREE_FORMS,
        nug_grid=_NUG, k_grid=_KG)
    return p, err


def _free_gp_scale_on_day(product, day, baseline_min_km=0.0):
    """Load one day, optionally restrict to baselines > baseline_min_km,
    and return the free-GP selected (form, lambda) on the within-day LOO.

    For the baseline-restricted variant the distance matrix is masked so
    that stations closer than `baseline_min_km` to every other station
    are dropped -- this keeps only stations that have at least one
    reference beyond the cutoff, which is the regime where the TEP scale
    should dominate over short-baseline receiver noise.
    """
    test = day_matrix(product, day)
    if test is None:
        return None
    names, X, D, _ = test
    n = X.shape[1]
    if n < 16:
        return None
    if baseline_min_km > 0:
        # Keep only stations whose nearest neighbour is >= baseline_min_km.
        keep = []
        for i in range(n):
            nn_dist = np.min([D[i, j] for j in range(n) if j != i])
            if nn_dist >= baseline_min_km:
                keep.append(i)
        if len(keep) < 16:
            return None
        X = X[:, keep]
        D = D[np.ix_(keep, keep)]
        n = len(keep)
    idx = list(range(n))
    p, err = _select_free_gp(X, D, idx)
    if p is None:
        return None
    return {
        "form": p.form,
        "lambda_km": float(p.lambda_km),
        "nugget": float(p.nugget),
        "k": int(p.k_neighbours),
        "n_stations": n,
        "train_loo_rms": float(err) if np.isfinite(err) else None,
        "in_tep_band": TEP_BAND_LO <= p.lambda_km <= TEP_BAND_HI,
    }


def injection_recovery(product, day, inject_lambda_km, rng):
    """Inject a synthetic TEP signal with known lambda into the actual
    residual stream and test whether the free GP recovers it.

    The injection is a zero-mean Gaussian process with a Gaussian
    covariance C(r) = A * exp(-r^2 / (2 * lambda^2)).  At each epoch an
    independent spatial draw is taken from N(0, C), so the injected
    signal has the correct distance-dependent correlation structure at
    every time step and varies in time (avoiding removal by the
    band-pass).

    The amplitude A is set PER DAY to dominate the actual per-station
    band-limited residual variance of that day's matrix.  The earlier
    fixed A = 0.3 m^2 was calibrated against a mis-stated ~0.05 m spatial
    std; the real precise-product per-station residual is ~1.6 m std
    (~2.6 m^2 variance, local-oscillator dominated), so a fixed A = 0.3
    left the injected signal buried ~3:1 below the noise and the recovery
    test was measuring SNR, not resolving power.  Setting
    A = SNR_TARGET * median(var(X)) makes this a true resolving-power
    test: it answers "can the free GP + this network geometry recover a
    known TEP-band lambda when the signal is unambiguously present",
    which is the question the S3 gate asks.  The separate low-SNR
    question ("is the real un-injected amplitude signal above the LO
    noise floor") is answered NO on precise products -- which is exactly
    why the operative unconstrained-discovery evidence is the phase
    observable (S4 / step_53), not band-passed amplitude.

    Returns the free-GP selected (form, lambda) on the injected stream.
    """
    test = day_matrix(product, day)
    if test is None:
        return None
    names, X, D, _ = test
    n = X.shape[1]
    n_t = X.shape[0]
    if n < 16:
        return None

    # Build the spatial covariance matrix.  Amplitude dominates the actual
    # per-station residual variance so the test measures resolving power,
    # not SNR (see docstring).
    SNR_TARGET = 4.0
    resid_var = float(np.nanmedian(np.nanvar(X, axis=0)))
    if not np.isfinite(resid_var) or resid_var <= 0.0:
        resid_var = 0.3
    A = SNR_TARGET * resid_var
    C = A * np.exp(-0.5 * (D / inject_lambda_km) ** 2)
    # Cholesky for the spatial draw; add a ridge proportional to the
    # diagonal so large-lambda (near-rank-1) covariances remain PSD.
    ridge = 1e-4 * np.trace(C) / n
    C = C + ridge * np.eye(n)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(C)
        C = C + (1e-3 * np.trace(C) / n - eigvals[0]) * np.eye(n)
        L = np.linalg.cholesky(C)

    # Per-epoch independent spatial draws: each epoch gets a fresh sample
    # from N(0, C), so the signal is spatially correlated with the
    # injected scale at every time step and varies in time.  This is the
    # correct way to inject a space-time GP with a spatial covariance.
    # To keep the computation tractable (n_t * n^2 can be large), draw
    # in blocks.  NaN-safe: replace any non-finite entries in L with 0.
    L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    X_inj = X.copy()
    block = 256
    for start in range(0, n_t, block):
        end = min(start + block, n_t)
        draws = rng.standard_normal((end - start, n))
        with np.errstate(all="ignore"):
            X_inj[start:end, :] += draws @ L.T
    X_inj = np.nan_to_num(X_inj, nan=0.0, posinf=0.0, neginf=0.0)

    idx = list(range(n))
    p, err = _select_free_gp(X_inj, D, idx)
    if p is None:
        return None
    return {
        "injected_lambda_km": float(inject_lambda_km),
        "recovered_form": p.form,
        "recovered_lambda_km": float(p.lambda_km),
        "recovery_ratio": float(p.lambda_km / inject_lambda_km),
        "in_tep_band": TEP_BAND_LO <= p.lambda_km <= TEP_BAND_HI,
        "within_2x_of_injected": 0.5 * inject_lambda_km <= p.lambda_km <= 2.0 * inject_lambda_km,
        "train_loo_rms": float(err) if np.isfinite(err) else None,
    }


def null_control(product, day, rng, mode="temporal_shuffle"):
    """Run the free GP on a null where the spatial signal is destroyed.

    temporal_shuffle: each station's residual time series is independently
        shuffled in time, destroying any cross-station spatial correlation
        while preserving each station's marginal distribution.
    geometry_shuffle: the station labels (and therefore the distance
        matrix) are randomly permuted, so any real spatial structure is
        assigned to wrong distances.
    """
    test = day_matrix(product, day)
    if test is None:
        return None
    names, X, D, _ = test
    n = X.shape[1]
    if n < 16:
        return None

    if mode == "temporal_shuffle":
        X_null = X.copy()
        for j in range(n):
            rng.shuffle(X_null[:, j])
    elif mode == "geometry_shuffle":
        perm = rng.permutation(n)
        D_null = D[np.ix_(perm, perm)]
        X_null = X
    else:
        return None

    D_use = D_null if mode == "geometry_shuffle" else D
    idx = list(range(n))
    p, err = _select_free_gp(X_null, D_use, idx)
    if p is None:
        return None
    return {
        "null_mode": mode,
        "selected_form": p.form,
        "selected_lambda_km": float(p.lambda_km),
        "in_tep_band": TEP_BAND_LO <= p.lambda_km <= TEP_BAND_HI,
        "at_grid_edge": p.lambda_km <= FREE_LAMBDA_KM[0] or p.lambda_km >= FREE_LAMBDA_KM[-1],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("STEP 43 -- Free-GP scale-discovery validation")
    print("=" * 72)

    rng = np.random.default_rng(20260943)

    # ---- 1. Baseline-restricted free GP on the precise product ----
    print("\n--- 1. Baseline-restricted free GP (precise product) ---")
    cdays = td.available_dates(PRECISE_PRODUCT, 2024)
    idx = np.linspace(0, len(cdays) - 1, 20).round().astype(int)
    cdays = [cdays[i] for i in idx if i < len(cdays)]
    print(f"  Days: {len(cdays)} (evenly-spaced subset of {PRECISE_PRODUCT})")

    baseline_results = {}
    for bmin, label in [(0.0, "all_baselines"),
                        (500.0, "gt_500km"),
                        (1000.0, "gt_1000km"),
                        (2000.0, "gt_2000km")]:
        scales = []
        forms = []
        n_stations = []
        for day in cdays:
            r = _free_gp_scale_on_day(PRECISE_PRODUCT, day, baseline_min_km=bmin)
            if r is not None:
                scales.append(r["lambda_km"])
                forms.append(r["form"])
                n_stations.append(r["n_stations"])
        if not scales:
            baseline_results[label] = None
            print(f"  {label}: (no valid days)")
            continue
        med = float(np.median(scales))
        in_band = sum(1 for s in scales if TEP_BAND_LO <= s <= TEP_BAND_HI)
        baseline_results[label] = {
            "baseline_min_km": bmin,
            "n_days": len(scales),
            "lambda_median_km": med,
            "lambda_iqr_km": [float(np.percentile(scales, 25)),
                              float(np.percentile(scales, 75))],
            "frac_in_tep_band": in_band / len(scales),
            "form_counts": {f: forms.count(f) for f in set(forms)},
            "median_n_stations": float(np.median(n_stations)),
        }
        print(f"  {label}: median lambda={med:.0f} km, "
              f"in TEP band {in_band}/{len(scales)} ({in_band/len(scales)*100:.0f}%), "
              f"median n_stations={np.median(n_stations):.0f}")

    # ---- 2. Injection/recovery on the precise product ----
    print("\n--- 2. Injection/recovery (precise product) ---")
    # Use a few days for the injection test.
    inj_days = cdays[:5] if len(cdays) >= 5 else cdays
    injection_results = {}
    for inj_lam in INJECT_LAMBDA_KM:
        recoveries = []
        for day in inj_days:
            for trial in range(3):  # 3 random trials per day per scale
                r = injection_recovery(PRECISE_PRODUCT, day, inj_lam, rng)
                if r is not None:
                    recoveries.append(r)
        if not recoveries:
            injection_results[str(inj_lam)] = None
            continue
        rec_lams = [r["recovered_lambda_km"] for r in recoveries]
        within_2x = sum(r["within_2x_of_injected"] for r in recoveries)
        in_band = sum(r["in_tep_band"] for r in recoveries)
        injection_results[str(inj_lam)] = {
            "injected_lambda_km": float(inj_lam),
            "n_trials": len(recoveries),
            "recovered_lambda_median_km": float(np.median(rec_lams)),
            "recovered_lambda_iqr_km": [float(np.percentile(rec_lams, 25)),
                                        float(np.percentile(rec_lams, 75))],
            "recovery_ratio_median": float(np.median(rec_lams) / inj_lam),
            "frac_within_2x": within_2x / len(recoveries),
            "frac_in_tep_band": in_band / len(recoveries),
            "in_tep_band": TEP_BAND_LO <= inj_lam <= TEP_BAND_HI,
        }
        r = injection_results[str(inj_lam)]
        print(f"  injected={inj_lam:.0f} km -> recovered median={r['recovered_lambda_median_km']:.0f} km "
              f"(ratio={r['recovery_ratio_median']:.2f}, within 2x: {r['frac_within_2x']*100:.0f}%)")

    # ---- 3. Null controls ----
    print("\n--- 3. Null controls (free GP on destroyed spatial signal) ---")
    null_days = cdays[:10] if len(cdays) >= 10 else cdays
    null_results = {}
    for mode in ("temporal_shuffle", "geometry_shuffle"):
        null_scales = []
        null_in_band = 0
        null_at_edge = 0
        for day in null_days:
            for trial in range(3):
                r = null_control(PRECISE_PRODUCT, day, rng, mode=mode)
                if r is not None:
                    null_scales.append(r["selected_lambda_km"])
                    if r["in_tep_band"]:
                        null_in_band += 1
                    if r["at_grid_edge"]:
                        null_at_edge += 1
        if not null_scales:
            null_results[mode] = None
            continue
        null_results[mode] = {
            "null_mode": mode,
            "n_trials": len(null_scales),
            "lambda_median_km": float(np.median(null_scales)),
            "frac_in_tep_band": null_in_band / len(null_scales),
            "frac_at_grid_edge": null_at_edge / len(null_scales),
            "lambda_distribution": {
                str(l): null_scales.count(l)
                for l in sorted(set(null_scales))
            },
        }
        r = null_results[mode]
        print(f"  {mode}: median lambda={r['lambda_median_km']:.0f} km, "
              f"in TEP band {r['frac_in_tep_band']*100:.0f}%, "
              f"at grid edge {r['frac_at_grid_edge']*100:.0f}%")

    # ---- Summary ----
    # Diagnostic subset: how well does the estimator recover the shorter
    # scales that this particular station geometry resolves most readily?
    inj_resolvable = [injection_results[k] for k in injection_results
                      if injection_results[k] and
                      500.0 <= injection_results[k]["injected_lambda_km"] <= 2000.0]
    inj_recovery_rate = (np.mean([r["frac_within_2x"] for r in inj_resolvable])
                         if inj_resolvable else 0.0)

    # Confirmatory gate: evaluate recovery across the whole pre-specified TEP
    # band.  The weak 3,000--5,000 km recoveries are a resolution debug flag;
    # they cannot be excluded merely by calling only 500--2,000 km resolvable.
    inj_tep_band = [injection_results[k] for k in injection_results
                    if injection_results[k] and
                    TEP_BAND_LO <= injection_results[k]["injected_lambda_km"] <= TEP_BAND_HI]
    inj_tep_band_rate = (np.mean([r["frac_within_2x"] for r in inj_tep_band])
                         if inj_tep_band else 0.0)
    inj_tep_band_scales_passing = sum(
        r["frac_within_2x"] >= 0.5 for r in inj_tep_band)
    injection_validates_full_band = bool(
        inj_tep_band and inj_tep_band_rate > 0.5 and
        inj_tep_band_scales_passing >= int(np.ceil(0.8 * len(inj_tep_band))))

    # Does the baseline-restricted precise free GP move into the TEP band?
    b1000 = baseline_results.get("gt_1000km")
    baseline_restricted_in_band = (
        b1000 is not None and
        b1000["lambda_median_km"] >= TEP_BAND_LO and
        b1000["frac_in_tep_band"] >= 0.5
    )

    # Do the nulls fail to select a consistent TEP-band scale?
    nulls_fail = all(
        null_results[m] is None or null_results[m]["frac_in_tep_band"] < 0.3
        for m in null_results if null_results[m] is not None
    )

    print(f"\n{'=' * 72}")
    print("STEP 43 SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Injection recovery rate (resolvable 500-2000 km, within 2x): "
          f"{inj_recovery_rate*100:.0f}%")
    print(f"  Injection recovery rate (full TEP band, within 2x): "
          f"{inj_tep_band_rate*100:.0f}% "
          f"({inj_tep_band_scales_passing}/{len(inj_tep_band)} scales >=50%)")
    print(f"  Baseline-restricted (>1000 km) precise free GP in TEP band: "
          f"{baseline_restricted_in_band}")
    print(f"  Nulls fail to select TEP-band scale: {nulls_fail}")

    result = {
        "test": "free_gp_scale_validation",
        "description": (
            "Validates the free-GP scale-discovery result via injection/recovery, "
            "baseline-restricted selection, and null controls on the precise product."
        ),
        "tep_band_km": [TEP_BAND_LO, TEP_BAND_HI],
        "baseline_restricted": baseline_results,
        "injection_recovery": injection_results,
        "null_controls": null_results,
        "summary": {
            "injection_recovery_rate_resolvable_within_2x": float(inj_recovery_rate),
            "injection_recovery_rate_tep_band_within_2x": float(inj_tep_band_rate),
            "injection_tep_band_scales_passing": int(inj_tep_band_scales_passing),
            "injection_tep_band_n_scales": int(len(inj_tep_band)),
            "injection_validates_full_tep_band": injection_validates_full_band,
            "baseline_restricted_in_tep_band": bool(baseline_restricted_in_band),
            "nulls_fail_tep_band": bool(nulls_fail),
            "free_gp_is_validated_scale_discovery": bool(
                injection_validates_full_band and baseline_restricted_in_band and nulls_fail
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }
    outpath = OUT / "step_43_free_gp_validation.json"
    outpath.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nsaved {outpath}")


if __name__ == "__main__":
    main()
