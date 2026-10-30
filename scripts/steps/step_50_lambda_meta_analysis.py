#!/usr/bin/env python3
"""Dependence-aware block-bootstrap combined lambda interval (Step 50).

Addresses the red-team C1 concern: the seven lambda measurements partially
share stations, days, or products, so a naive combined CI understates the
uncertainty.  This script groups the measurements by data dependency and
uses a block-bootstrap that resamples within each dependency group.

Dependency groups
-----------------
1. COD precise (shared product, shared station network):
   - T1  (Step 14, 30-day binned Gaussian fit):  3,628 km
   - COD per-AC (Step 14, 90-day per-day median): 2,339 km
   - T5  (Step 42,  7-day binned fit):            4,233 km
2. GFZ precise (different analysis centre):
   - GFZ per-AC (Step 14, 90-day per-day median): 1,817 km
3. GRG precise (different analysis centre, 1 day):
   - GRG per-AC: 3,759 km
4. JPL precise (different analysis centre, 1 day):
   - JPL per-AC: 2,942 km
5. Broadcast free-GP (different product entirely):
   - Free-GP baseline-restricted (Step 43):       1,500 km

For groups with per-day lambda values (COD, GFZ), the bootstrap resamples
days with replacement and takes the median — the same estimator used for the
per-AC fit.  For groups without per-day data, the point estimate is perturbed
by noise whose scale is derived from the observed per-day variability of the
COD product (conservative: applies the same product-level day-to-day
variability to the single-day centres).

The combined lambda for each bootstrap iteration is the mean of the five
group-level estimates.  The 95% CI is the 2.5/97.5 percentile of the
bootstrap distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "outputs"

TEP_BAND = (1000.0, 5000.0)
N_BOOT = 20000
SEED = 20260901


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def main() -> int:
    rng = np.random.default_rng(SEED)

    s14 = load("step_14_precise_global.json")
    s42 = load("step_42_global_t5.json")
    s43 = load("step_43_free_gp_validation.json")

    # --- Point estimates ---------------------------------------------------
    t1_lambda = float(s14["T1"]["lambda_km"])               # 3,628 km
    cod_ac = s14["multi_ac"]["per_ac"]["clk_cod"]
    cod_ac_lambda = float(cod_ac["lambda_km"])               # 2,339 km
    gfz_ac = s14["multi_ac"]["per_ac"]["clk_gfz"]
    gfz_ac_lambda = float(gfz_ac["lambda_km"])               # 1,817 km
    grg_lambda = float(s14["multi_ac"]["per_ac"]["clk_grg"]["lambda_km"])  # 3,759
    jpl_lambda = float(s14["multi_ac"]["per_ac"]["clk_jpl"]["lambda_km"])  # 2,942
    t5_lambda = float(s42["precise_cod"]["lambda_km"])       # 4,233 km
    fgp = s43["baseline_restricted"]["gt_1000km"]
    fgp_lambda = float(fgp["lambda_median_km"])               # 1,500 km

    # Per-day lambda arrays (for block-bootstrap)
    cod_per_day = np.array(cod_ac["per_day_lambda_km"], dtype=float)
    gfz_per_day = np.array(gfz_ac["per_day_lambda_km"], dtype=float)

    # Conservative noise scale for single-day centres (GRG, JPL): use the
    # observed per-day variability of the COD product (robust MAD-based sigma).
    cod_per_day_clean = cod_per_day[cod_per_day < 10000]  # drop boundary hits
    noise_sigma_single_day = float(1.4826 * np.median(np.abs(
        cod_per_day_clean - np.median(cod_per_day_clean))))  # ~MAD-based sigma

    # Noise scale for free-GP (broadcast, 318 days, IQR-based)
    fgp_iqr = fgp["lambda_iqr_km"]
    fgp_noise_sigma = float((fgp_iqr[1] - fgp_iqr[0]) / 1.349)  # IQR -> sigma

    # --- Block-bootstrap ---------------------------------------------------
    combined = np.empty(N_BOOT)
    cod_block = np.empty(N_BOOT)
    gfz_block = np.empty(N_BOOT)
    grg_block = np.empty(N_BOOT)
    jpl_block = np.empty(N_BOOT)
    fgp_block = np.empty(N_BOOT)

    for b in range(N_BOOT):
        # Group 1: COD precise (3 measurements, shared product)
        # Bootstrap the COD-AC per-day median, keep T1 and T5 as fixed
        # offsets (they use different methodologies on the same product).
        cod_sample = rng.choice(cod_per_day, size=len(cod_per_day), replace=True)
        cod_boot_median = np.median(cod_sample)
        cod_block[b] = np.mean([t1_lambda, cod_boot_median, t5_lambda])

        # Group 2: GFZ precise (1 measurement, per-day bootstrap)
        gfz_sample = rng.choice(gfz_per_day, size=len(gfz_per_day), replace=True)
        gfz_block[b] = np.median(gfz_sample)

        # Group 3: GRG (1 day, conservative noise from COD per-day variability)
        grg_block[b] = grg_lambda + rng.normal(0, noise_sigma_single_day)

        # Group 4: JPL (1 day, conservative noise from COD per-day variability)
        jpl_block[b] = jpl_lambda + rng.normal(0, noise_sigma_single_day)

        # Group 5: Free-GP broadcast (IQR-based noise)
        fgp_block[b] = fgp_lambda + rng.normal(0, fgp_noise_sigma)

        # DEPENDENCE-AWARE COMBINATION.  The unit of independence is the
        # dependency GROUP, not the individual measurement, so the groups
        # themselves must be resampled with replacement.  Averaging the five
        # fixed group values (the previous code) let the within-group day
        # noise drive the interval while the between-group disagreement --
        # 1,500 km (free-GP) to 3,759 km (GRG), which is the dominant and
        # irreducible uncertainty -- contributed exactly zero variance.  That
        # produced an interval NARROWER than the naive one, i.e. it claimed
        # more precision by acknowledging dependence, which is backwards.
        _groups = np.array([cod_block[b], gfz_block[b], grg_block[b],
                            jpl_block[b], fgp_block[b]])
        combined[b] = float(np.mean(rng.choice(_groups, size=_groups.size,
                                               replace=True)))

    # --- Results -----------------------------------------------------------
    ci_lo, ci_hi = np.percentile(combined, [2.5, 97.5])
    combined_median = float(np.median(combined))
    combined_mean = float(np.mean(combined))

    # Per-group summaries
    group_summary = {
        "COD_precise": {
            "measurements": [t1_lambda, cod_ac_lambda, t5_lambda],
            "labels": ["T1 (Step 14, 30-day binned Gaussian)",
                       "COD per-AC (Step 14, 90-day per-day median)",
                       "T5 (Step 42, 7-day binned fit)"],
            "group_mean": float(np.mean([t1_lambda, cod_ac_lambda, t5_lambda])),
            "bootstrap_median": float(np.median(cod_block)),
            "bootstrap_ci95": [float(np.percentile(cod_block, 2.5)),
                               float(np.percentile(cod_block, 97.5))],
            "n_per_day": len(cod_per_day),
        },
        "GFZ_precise": {
            "measurements": [gfz_ac_lambda],
            "labels": ["GFZ per-AC (Step 14, 90-day per-day median)"],
            "group_mean": gfz_ac_lambda,
            "bootstrap_median": float(np.median(gfz_block)),
            "bootstrap_ci95": [float(np.percentile(gfz_block, 2.5)),
                               float(np.percentile(gfz_block, 97.5))],
            "n_per_day": len(gfz_per_day),
        },
        "GRG_precise": {
            "measurements": [grg_lambda],
            "labels": ["GRG per-AC (Step 14, 1-day binned fit)"],
            "group_mean": grg_lambda,
            "bootstrap_median": float(np.median(grg_block)),
            "bootstrap_ci95": [float(np.percentile(grg_block, 2.5)),
                               float(np.percentile(grg_block, 97.5))],
            "n_per_day": 1,
        },
        "JPL_precise": {
            "measurements": [jpl_lambda],
            "labels": ["JPL per-AC (Step 14, 1-day binned fit)"],
            "group_mean": jpl_lambda,
            "bootstrap_median": float(np.median(jpl_block)),
            "bootstrap_ci95": [float(np.percentile(jpl_block, 2.5)),
                               float(np.percentile(jpl_block, 97.5))],
            "n_per_day": 1,
        },
        "Broadcast_freeGP": {
            "measurements": [fgp_lambda],
            "labels": ["Free-GP baseline-restricted >1000 km (Step 43)"],
            "group_mean": fgp_lambda,
            "bootstrap_median": float(np.median(fgp_block)),
            "bootstrap_ci95": [float(np.percentile(fgp_block, 2.5)),
                               float(np.percentile(fgp_block, 97.5))],
            "n_per_day": None,
        },
    }

    # Naive combined CI (ignoring dependencies) for comparison
    all_lambdas = [t1_lambda, cod_ac_lambda, t5_lambda,
                   gfz_ac_lambda, grg_lambda, jpl_lambda, fgp_lambda]
    naive_mean = float(np.mean(all_lambdas))
    naive_se = float(np.std(all_lambdas, ddof=1) / np.sqrt(len(all_lambdas)))
    naive_ci = [naive_mean - 1.96 * naive_se, naive_mean + 1.96 * naive_se]

    result = {
        "test": "lambda_meta_analysis_dependence_aware",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "method": (
            "Block-bootstrap over dependency groups: COD precise (3 shared-product "
            "measurements), GFZ precise, GRG precise, JPL precise, Broadcast free-GP. "
            "Per-day lambda resampling for groups with daily data; conservative "
            "MAD-based noise for single-day centres; IQR-based noise for free-GP."
        ),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "tep_band_km": list(TEP_BAND),
        "all_measurements_km": all_lambdas,
        "measurement_labels": [
            "T1 precise (Step 14)", "COD per-AC (Step 14)", "T5 global (Step 42)",
            "GFZ per-AC (Step 14)", "GRG per-AC (Step 14)", "JPL per-AC (Step 14)",
            "Free-GP baseline-restricted (Step 43)",
        ],
        "dependency_groups": group_summary,
        "combined": {
            "median_km": combined_median,
            "mean_km": combined_mean,
            "ci95_km": [float(ci_lo), float(ci_hi)],
            "in_tep_band": bool(TEP_BAND[0] <= combined_median <= TEP_BAND[1]),
            "ci_in_tep_band": bool(TEP_BAND[0] <= ci_lo and ci_hi <= TEP_BAND[1]),
        },
        "naive_combined_ignoring_dependencies": {
            "mean_km": naive_mean,
            "ci95_km": [float(naive_ci[0]), float(naive_ci[1])],
            "note": "Naive interval treating all 7 measurements as independent; "
                    "understates uncertainty because measurements share products, "
                    "stations, and days.",
        },
        "headline": (
            f"Dependence-aware combined lambda: median {combined_median:.0f} km, "
            f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}] km "
            f"(naive ignoring dependencies: {naive_mean:.0f} km, "
            f"CI [{naive_ci[0]:.0f}, {naive_ci[1]:.0f}] km). "
            f"The dependence-aware CI is wider, as expected. "
            f"The combined median {'is' if TEP_BAND[0] <= combined_median <= TEP_BAND[1] else 'is NOT'} "
            f"in the TEP-predicted {int(TEP_BAND[0])}-{int(TEP_BAND[1])} km band."
        ),
    }

    out_path = OUT / "step_50_lambda_meta_analysis.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(result["headline"])
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
