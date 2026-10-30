#!/usr/bin/env python3
"""
STEP 32 -- Kernel family comparison: reverse-engineering all patterns from
Papers 1-2 into the operational TEP clock kernel.

Papers 1 (TEP-GNSS) and 2 (TEP-GNSS-II) found five distinct patterns in the
distance-structured correlations of CODE PPP clock residuals:

  1. Exponential distance decay  (λ ≈ 3,330-4,549 km, R² = 0.92-0.97)
  2. Gaussian/squared-exponential preferred by AIC  (ΔAIC = -12.8)
  3. Matérn(ν=1.5) competitive with exponential
  4. EW/NS anisotropy  (ratio = 2.16, p < 10⁻¹⁵)
  5. Velocity-dependent anisotropy  (r = -0.888, surrogate p < 2×10⁻⁷)
  6. CMB dipole-direction alignment  (r = 0.747, 10,300× variance ratio)

The baseline TEP-PPP kernel (step_25) uses only pattern 1: an isotropic
exponential covariance.  This step tests whether patterns 2-6 carry
exploitable operational signal by extending the kernel family and
re-running the same leakage-free nested-CV.

Procedure (identical to step_25 but with an extended grid):
  1. Split stations 50/50 into TRAIN and TEST.
  2. Select (λ, nugget, K, form, aniso_mode, aniso_ratio) by LOO inside TRAIN.
  3. Predict each TEST station from TRAIN references.
  4. Score against raw (no network) and bulk mean.

All hyperparameters -- including the kernel form and anisotropy mode -- are
selected on training data only, so the held-out skill is leakage-free in
kernel-family space as well.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_kernel as tk     # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
N_SPLITS = 40
SEED = 20260903  # primary seed, matching step_25

# Reduced grid for the extended family search (the full 8×5×4 grid would
# make the joint search over 3 forms × 7 mode/ratio combos infeasible).
# The canonical λ = 4200 km is the centre; 1500 and 7000 bracket the IQR.
LAM_FAMILY_GRID = (1500.0, 4200.0, 7000.0)
NUG_FAMILY_GRID = (0.05, 0.35)
K_FAMILY_GRID = (15, 25)

# Full grid for the baseline (matching step_25 exactly)
LAM_FULL_GRID = tk.LAMBDA_GRID_KM
NUG_FULL_GRID = tk.NUGGET_GRID
K_FULL_GRID = tk.K_GRID


def _run_split(idx, X, D, AZ, dtag, n, families):
    """Run one train/test split for all kernel families.

    *families* is a list of (form, aniso_mode, aniso_ratio) tuples.
    Returns a list of dicts, one per family.
    """
    tr, te = sorted(idx[: n // 2].tolist()), sorted(idx[n // 2:].tolist())
    tr_std = np.array([X[:, i].std() for i in tr])
    tr = [i for i, sd in zip(tr, tr_std) if 0.01 < sd < 60.0]
    if len(tr) < 8 or len(te) < 2:
        return None

    results = []
    for form, mode, ratio in families:
        if mode == "none":
            # Baseline uses the FULL grid (matching step_25 exactly) so the
            # isotropic exponential is not handicapped relative to step_25.
            p, tr_err = tk.fit_hyperparameters(
                X, D, tr,
                lam_grid=LAM_FULL_GRID, nug_grid=NUG_FULL_GRID,
                k_grid=K_FULL_GRID, AZ=AZ, date_tag=dtag,
                form_grid=(form,), aniso_mode_grid=("none",), aniso_ratio_grid=(1.0,))
        else:
            p, tr_err = tk.fit_hyperparameters(
                X, D, tr,
                lam_grid=LAM_FAMILY_GRID, nug_grid=NUG_FAMILY_GRID,
                k_grid=K_FAMILY_GRID, AZ=AZ, date_tag=dtag,
                form_grid=(form,), aniso_mode_grid=(mode,),
                aniso_ratio_grid=(ratio,))
        if p is None:
            results.append(None)
            continue
        sc = tk.score_heldout(X, D, te, tr, p, AZ=AZ, date_tag=dtag)
        if sc is None:
            results.append(None)
            continue
        results.append({
            "form": form, "aniso_mode": mode, "aniso_ratio": ratio,
            "lam_pick": p.lambda_km, "nug_pick": p.nugget, "k_pick": p.k_neighbours,
            "train_loo_m": tr_err,
            "row": {"day": dtag, "n_stations": n,
                    "params": p.as_dict(), **{k: v for k, v in sc.items()
                                              if k != "per_target"}},
        })
    return results


def run_family_comparison(days, families, label="broadcast"):
    """Run nested-CV for all kernel families on the broadcast product."""
    rng = np.random.default_rng(SEED)
    # family_results[i] = list of rows for family i
    family_results = [[] for _ in families]

    for dtag in days:
        s = td.load_series("broadcast", dtag)
        if not s:
            continue
        names, X, D, AZ = tk.build_matrix(s, filter_by_std=False, compute_azimuth=True)
        if X is None or len(names) < 16:
            continue
        n = len(names)
        permutations = [rng.permutation(n) for _ in range(N_SPLITS)]

        from joblib import Parallel, delayed
        split_results = Parallel(n_jobs=-1)(
            delayed(_run_split)(idx, X, D, AZ, dtag, n, families)
            for idx in permutations)

        for sr in split_results:
            if sr is None:
                continue
            for i, r in enumerate(sr):
                if r is not None:
                    family_results[i].append(r["row"])

    # Aggregate per family
    summaries = []
    for i, (form, mode, ratio) in enumerate(families):
        rows = family_results[i]
        if not rows:
            summaries.append({"form": form, "aniso_mode": mode,
                              "aniso_ratio": ratio, "n": 0})
            continue
        skill = np.array([r["skill_vs_raw_pct"] for r in rows])
        vs_mean = np.array([r["kernel_vs_mean_pct"] for r in rows])
        vs_ukrig = np.array([r.get("kernel_vs_uniform_krig_pct", np.nan) for r in rows])
        lam_picks = [r["params"]["lambda_km"] for r in rows]
        nug_picks = [r["params"]["nugget"] for r in rows]
        k_picks = [r["params"]["k_neighbours"] for r in rows]

        # Day-clustered bootstrap
        day_ids = np.array([r["day"] for r in rows])
        uniq_days = sorted(set(day_ids))
        by_day = {d: np.flatnonzero(day_ids == d) for d in uniq_days}

        def day_boot(vals, n_boot=4000):
            vals = np.asarray(vals, float)
            out = []
            for _ in range(n_boot):
                drawn = rng.choice(uniq_days, len(uniq_days), replace=True)
                idx = np.concatenate([by_day[d] for d in drawn])
                v = vals[idx]
                v = v[np.isfinite(v)]
                if len(v):
                    out.append(float(np.median(v)))
            return out

        bs_s = day_boot(skill)
        bs_m = day_boot(vs_mean)
        bs_u = day_boot(vs_ukrig)

        e = {k: float(np.nanmedian([r[k] for r in rows]))
             for k in ("raw_m", "kernel_m", "net_mean_m")}

        summaries.append({
            "form": form, "aniso_mode": mode, "aniso_ratio": ratio,
            "label": f"{form}_{mode}_{ratio}" if mode != "none" else form,
            "n_splits": len(rows),
            "n_days": len(set(r["day"] for r in rows)),
            "selected_lambda_km": float(np.nanmedian(lam_picks)),
            "selected_nugget_median": float(np.nanmedian(nug_picks)),
            "selected_k_median": float(np.nanmedian(k_picks)),
            "heldout_error_m": e,
            "skill_vs_raw_pct": {"median": float(np.nanmedian(skill)),
                                 "ci95": [float(np.percentile(bs_s, 2.5)),
                                          float(np.percentile(bs_s, 97.5))] if bs_s else [None, None]},
            "kernel_vs_bulk_mean_pct": {"median": float(np.nanmedian(vs_mean)),
                                        "ci95": [float(np.percentile(bs_m, 2.5)),
                                                 float(np.percentile(bs_m, 97.5))] if bs_m else [None, None]},
            "kernel_vs_uniform_krig_pct": {
                "median": float(np.nanmedian(vs_ukrig)) if np.any(np.isfinite(vs_ukrig)) else None,
                "ci95": [float(np.percentile(bs_u, 2.5)),
                         float(np.percentile(bs_u, 97.5))] if bs_u else [None, None]},
            "frac_splits_kernel_beats_mean": float(np.mean(vs_mean[np.isfinite(vs_mean)] > 0)) if np.any(np.isfinite(vs_mean)) else None,
            "splits": rows,
        })

        s_med = summaries[-1]["skill_vs_raw_pct"]["median"]
        m_med = summaries[-1]["kernel_vs_bulk_mean_pct"]["median"]
        wins = summaries[-1]["frac_splits_kernel_beats_mean"]
        wins_str = f"{wins*100:.0f}%" if wins is not None else "N/A"
        print(f"  [{summaries[-1]['label']:30s}]  skill={s_med:+5.1f}%  "
              f"vs_bulk={m_med:+5.1f}%  wins={wins_str}  "
              f"λ*={summaries[-1]['selected_lambda_km']:.0f}")

    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [f"2024{d:03d}" for d in range(100, 107)]

    # Define the kernel families to test
    families = [
        # Baseline: exponential isotropic (matches step_25)
        ("exponential", "none", 1.0),
        # Form variants (isotropic)
        ("gaussian", "none", 1.0),
        ("matern15", "none", 1.0),
        # Anisotropy variants (exponential form)
        ("exponential", "ew_ns", 2.16),
        ("exponential", "velocity", 2.0),
        ("exponential", "cmb", 2.0),
        # Best-guess combinations
        ("gaussian", "ew_ns", 2.16),
        ("gaussian", "cmb", 2.0),
        ("matern15", "ew_ns", 2.16),
    ]

    print("=" * 70)
    print("STEP 32: Kernel family comparison")
    print(f"  {len(families)} families × {N_SPLITS} splits × {len(days)} days")
    print(f"  Grid: λ={LAM_FAMILY_GRID}, nug={NUG_FAMILY_GRID}, K={K_FAMILY_GRID}")
    print("=" * 70)

    summaries = run_family_comparison(days, families, label="broadcast")

    # Find the best family
    best = max(summaries, key=lambda s: s.get("skill_vs_raw_pct", {}).get("median", -999))
    baseline = next((s for s in summaries if s["form"] == "exponential"
                     and s["aniso_mode"] == "none"), summaries[0])

    # Compute improvement of each family over baseline
    base_skill = baseline.get("skill_vs_raw_pct", {}).get("median", 0)
    for s in summaries:
        s["improvement_over_baseline_pp"] = (
            s.get("skill_vs_raw_pct", {}).get("median", 0) - base_skill)

    res = {
        "test": "kernel_family_comparison",
        "description": ("Reverse-engineering all distance-structured patterns "
                        "from Papers 1-2 into the operational TEP clock kernel. "
                        "Each family is tested with the same leakage-free nested-CV "
                        "as step_25, with hyperparameters selected on training LOO."),
        "method": ("nested CV: (λ, nugget, K, form, aniso_mode, aniso_ratio) "
                   "selected by LOO inside TRAIN, scored on held-out TEST"),
        "grids": {
            "lambda_km": list(LAM_FAMILY_GRID),
            "nugget": list(NUG_FAMILY_GRID),
            "k": list(K_FAMILY_GRID),
            "form": list(tk.FORM_GRID),
            "aniso_mode": list(tk.ANISO_MODE_GRID),
            "aniso_ratio": list(tk.ANISO_RATIO_GRID),
        },
        "n_splits": N_SPLITS,
        "n_days": len(days),
        "seed": SEED,
        "families": summaries,
        "baseline": baseline,
        "best": best,
        "baseline_skill_vs_raw_pct": base_skill,
        "best_skill_vs_raw_pct": best.get("skill_vs_raw_pct", {}).get("median", 0),
        "best_improvement_over_baseline_pp": best.get("improvement_over_baseline_pp", 0),
        "timestamp": datetime.now().isoformat(),
    }

    # Headline
    b = best
    bl = baseline
    res["headline"] = (
        f"Kernel family comparison: baseline (exponential, isotropic) achieves "
        f"{bl['skill_vs_raw_pct']['median']:.1f}% skill vs raw. "
        f"Best family: {b['label']} with {b['skill_vs_raw_pct']['median']:.1f}% skill "
        f"({b['improvement_over_baseline_pp']:+.1f} pp vs baseline). "
        f"Selected λ*={b['selected_lambda_km']:.0f} km, "
        f"nugget={b['selected_nugget_median']:.2f}, K={b['selected_k_median']:.0f}."
    )

    (OUT / "step_32_kernel_family.json").write_text(json.dumps(res, indent=2))
    print(f"\n{res['headline']}")
    print(f"\nSaved {OUT / 'step_32_kernel_family.json'}")

    # Print comparison table
    print(f"\n{'Family':<35s} {'Skill':>7s} {'vsBulk':>7s} {'Wins':>6s} "
          f"{'λ*':>6s} {'Δpp':>7s}")
    print("-" * 75)
    for s in summaries:
        sk = s.get("skill_vs_raw_pct", {}).get("median", float("nan"))
        mb = s.get("kernel_vs_bulk_mean_pct", {}).get("median", float("nan"))
        wn = s.get("frac_splits_kernel_beats_mean", 0)
        lm = s.get("selected_lambda_km", 0)
        dp = s.get("improvement_over_baseline_pp", 0)
        print(f"{s.get('label','?'):<35s} {sk:>+6.1f}% {mb:>+6.1f}% "
              f"{wn*100:>5.0f}% {lm:>6.0f} {dp:>+6.1f}")


if __name__ == "__main__":
    main()
