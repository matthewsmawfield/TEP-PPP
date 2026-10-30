#!/usr/bin/env python3
"""
STEP 25  -- Nested cross-validated kernel selection and held-out prediction skill.

This is the operational core of the paper, done without any hand-picked
constants and without leakage.

Procedure, repeated per day and over repeated random station splits:
  1. Split the network's stations 50/50 into TRAIN and TEST.
  2. Select (lambda, nugget, K) by leave-one-out **inside TRAIN only** --
     both targets and references come from TRAIN.
  3. Predict each held-out TEST station's clock using **TRAIN stations only**.
  4. Score against two baselines:
        raw       -- predict zero (no network at all)
        net_mean  -- uniform average of the TRAIN stations (bulk common mode)

Nothing about a TEST station -- neither its own data nor its use as a
reference -- enters the selection, so the reported skill is honest
out-of-sample in both station space and hyperparameter space.

Two product arms, because they answer different questions:
  broadcast  -- SPP receiver clock (real-time / single-frequency user regime)
  clk_cod    -- IGS/MGEX precise receiver clock (post-processed regime)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_kernel as tk     # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
N_SPLITS = 40          # random 50/50 station splits per day; deterministic via fixed RNG.
#                        With 8 splits the median skill was unstable to the RNG realisation
#                        (probe: +26% at NS>=16, but individual 8-split seeds ranged +26..+41%);
#                        40 splits (280 evaluations) gives a stable median with a day-clustered CI.
SEEDS = [20260903, 20260904, 20260905, 20260906, 20260907, 20260908, 20260909]


def run_arm(product, date_tags, label, seed=20260903):
    rng = np.random.default_rng(seed)
    rows = []
    lam_picks, nug_picks, k_picks = [], [], []
    for dtag in date_tags:
        s = td.load_series(product, dtag)
        if not s:
            continue
        # Build matrix WITHOUT std filtering — the train/test split must not
        # be influenced by the held-out stations' own data.  Degenerate or
        # jumped stations are removed from the TRAINING set only, inside
        # each split, so the test set is never selected based on its own
        # residual statistics (leakage-free station QC).
        names, X, D = tk.build_matrix(s, filter_by_std=False)
        if X is None or len(names) < 16:
            continue
        n = len(names)
        # Generate permutations upfront
        permutations = [rng.permutation(n) for _ in range(N_SPLITS)]
        
        def _run_split(idx):
            tr, te = sorted(idx[: n // 2].tolist()), sorted(idx[n // 2:].tolist())
            # Training-side QC: compute per-station std from the TRAINING
            # columns only, AFTER the split, so the test set is never
            # selected based on its own residual statistics (leakage-free
            # station QC).  The previous implementation computed col_std
            # over the full matrix before the split, which — while not a
            # cross-station data leak — did not satisfy the strict
            # "filter on training stations only after the split" constraint.
            tr_std = np.array([X[:, i].std() for i in tr])
            tr = [i for i, sd in zip(tr, tr_std) if 0.01 < sd < 60.0]
            if len(tr) < 8 or len(te) < 2:
                return None
            p, tr_err = tk.fit_hyperparameters(X, D, tr)
            if p is None:
                return None
            sc = tk.score_heldout(X, D, te, tr, p)
            if sc is None:
                return None
            return {
                "lam_pick": p.lambda_km, "nug_pick": p.nugget, "k_pick": p.k_neighbours,
                "row": {"day": dtag, "n_stations": n, "params": p.as_dict(),
                        "train_loo_m": tr_err, **{k: v for k, v in sc.items() if k != "per_target"}}
            }

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(delayed(_run_split)(idx) for idx in permutations)
        
        for res in results:
            if res is not None:
                lam_picks.append(res["lam_pick"])
                nug_picks.append(res["nug_pick"])
                k_picks.append(res["k_pick"])
                rows.append(res["row"])
    if not rows:
        return {"label": label, "n": 0}

    def med(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else float('nan')

    skill = np.array([r["skill_vs_raw_pct"] for r in rows])
    vs_mean = np.array([r["kernel_vs_mean_pct"] for r in rows])
    vs_ukrig_all = np.array([r.get("kernel_vs_uniform_krig_pct", np.nan) for r in rows])
    has_ukrig = bool(np.any(np.isfinite(vs_ukrig_all)))
    vs_ukrig = vs_ukrig_all[np.isfinite(vs_ukrig_all)] if has_ukrig else np.array([])

    # Day-clustered bootstrap: the 7 processing days are the independent unit
    # (same stations, repeating satellite geometry), so resampling the 56 rows
    # i.i.d. would give an anti-conservative interval.  Resample days with
    # replacement, take all rows from the drawn days, drop NaNs, take the median.
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
    bs_u = day_boot(vs_ukrig_all) if has_ukrig else []

    res = {
        "label": label, "product": product, "n_day_splits": len(rows),
        "n_days": len(set(r["day"] for r in rows)),
        "selected_lambda_km": {"median": float(np.median(lam_picks)),
                               "iqr": [float(np.percentile(lam_picks, 25)),
                                       float(np.percentile(lam_picks, 75))],
                               "counts": {str(int(v)): int(np.sum(np.array(lam_picks) == v))
                                          for v in sorted(set(lam_picks))}},
        "selected_nugget_median": float(np.median(nug_picks)),
        "selected_k_median": float(np.median(k_picks)),
        "heldout_error_m": {"raw": med("raw_m"), "kernel": med("kernel_m"),
                            "net_mean": med("net_mean_m"),
                            "uniform_krig": med("uniform_krig_m") if "uniform_krig_m" in rows[0] else None},
        "skill_vs_raw_pct": {"median": float(np.nanmedian(skill)) if np.any(np.isfinite(skill)) else float('nan'),
                             "ci95": [float(np.percentile(bs_s, 2.5)),
                                      float(np.percentile(bs_s, 97.5))] if bs_s else [None, None]},
        "bulk_mean_skill_vs_raw_pct": float(np.nanmedian([r["mean_skill_vs_raw_pct"] for r in rows])) if np.any(np.isfinite([r["mean_skill_vs_raw_pct"] for r in rows])) else float('nan'),
        "kernel_vs_bulk_mean_pct": {"median": float(np.nanmedian(vs_mean)) if np.any(np.isfinite(vs_mean)) else float('nan'),
                                    "ci95": [float(np.percentile(bs_m, 2.5)),
                                             float(np.percentile(bs_m, 97.5))] if bs_m else [None, None]},
        "kernel_vs_uniform_krig_pct": {"median": float(np.nanmedian(vs_ukrig)) if len(vs_ukrig) else None,
                                       "ci95": [float(np.percentile(bs_u, 2.5)),
                                                float(np.percentile(bs_u, 97.5))] if bs_u else None},
        "frac_splits_kernel_beats_mean": float(np.mean(vs_mean[np.isfinite(vs_mean)] > 0)) if np.any(np.isfinite(vs_mean)) else None,
        "significant": bool(bs_m and np.percentile(bs_m, 2.5) > 0),
        "splits": rows,
    }
    print(f"[{label}]  {res['n_days']} days x {N_SPLITS} random station splits "
          f"= {len(rows)} held-out evaluations")
    lam = res["selected_lambda_km"]
    print(f"  selected lambda*: median {lam['median']:.0f} km  IQR [{lam['iqr'][0]:.0f}, {lam['iqr'][1]:.0f}]  "
          f"nugget {res['selected_nugget_median']:.2f}  K {res['selected_k_median']:.0f}")
    e = res["heldout_error_m"]
    print(f"  held-out clock prediction error:  raw {e['raw']:.2f}  "
          f"bulk-mean {e['net_mean']:.2f}  KERNEL {e['kernel']:.2f} m")
    s95 = res["skill_vs_raw_pct"]; m95 = res["kernel_vs_bulk_mean_pct"]
    print(f"  skill vs raw    : {s95['median']:+.1f}%  95% CI [{s95['ci95'][0]:+.1f}, {s95['ci95'][1]:+.1f}]  "
          f"(bulk mean alone: {res['bulk_mean_skill_vs_raw_pct']:+.1f}%)")
    print(f"  kernel vs bulk  : {m95['median']:+.1f}%  95% CI [{m95['ci95'][0]:+.1f}, {m95['ci95'][1]:+.1f}]  "
          f"wins {res['frac_splits_kernel_beats_mean']*100:.0f}% of splits  significant: {res['significant']}")
    u = res.get("kernel_vs_uniform_krig_pct", {})
    if u.get("median") is not None:
        print(f"  kernel vs ukrig : {u['median']:+.1f}%  95% CI [{u['ci95'][0]:+.1f}, {u['ci95'][1]:+.1f}]  "
              f"(geometry-specific margin)")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [f"2024{d:03d}" for d in range(100, 107)]

    # Multi-seed robustness sweep for the broadcast arm: run the full
    # nested-CV with 7 independent seeds and report the range of median
    # skills.  The primary result is the first seed (matching step_26).
    seed_results = []
    primary_result = None
    for i, seed in enumerate(SEEDS):
        r = run_arm("broadcast", days, f"broadcast_spp_seed{i}", seed=seed)
        if r.get("n_day_splits", r.get("n", 0)) > 0:
            seed_results.append(r)
            print(f"  [seed {seed}] skill_vs_raw={r['skill_vs_raw_pct']['median']:+.1f}%  "
                  f"vs_bulk={r['kernel_vs_bulk_mean_pct']['median']:+.1f}%  "
                  f"wins={r['frac_splits_kernel_beats_mean']*100:.0f}%")
            if seed == SEEDS[0]:
                primary_result = r

    seed_skills = [r["skill_vs_raw_pct"]["median"] for r in seed_results]
    seed_vs_bulk = [r["kernel_vs_bulk_mean_pct"]["median"] for r in seed_results]

    res = {
        "test": "kernel_selection_and_heldout_skill",
        "method": ("nested CV: hyperparameters chosen by LOO inside TRAIN stations, "
                   "scored on held-out TEST stations predicted from TRAIN references only"),
        "grids": {"lambda_km": list(tk.LAMBDA_GRID_KM), "nugget": list(tk.NUGGET_GRID),
                  "k": list(tk.K_GRID)},
        "n_seeds": len(seed_results),
        "seed_skill_range": {"skill_vs_raw_pct": [float(min(seed_skills)), float(max(seed_skills))],
                             "kernel_vs_bulk_mean_pct": [float(min(seed_vs_bulk)), float(max(seed_vs_bulk))]},
        # Explicitly fetch the primary-seed result rather than indexing the
        # filtered list, so the headline always corresponds to SEEDS[0]
        # (matching step_26).  If the primary seed failed, report it.
        "broadcast_spp": primary_result if primary_result is not None else {"label": "broadcast_spp", "n": 0, "primary_seed_failed": True},
        "precise_clk": run_arm("clk_cod", td.available_dates("clk_cod", 2024), "precise_clk"),
        "timestamp": datetime.now().isoformat(),
    }
    b = res["broadcast_spp"]
    res["headline"] = (
        f"On broadcast SPP receiver clocks -- the real-time / single-frequency user regime -- "
        f"a distance kernel whose correlation length is SELECTED on held-out data "
        f"(median lambda* = {b['selected_lambda_km']['median']:.0f} km) predicts an unseen "
        f"station's clock {b['skill_vs_raw_pct']['median']:.0f}% better than no network and "
        f"{b['kernel_vs_bulk_mean_pct']['median']:.0f}% better than a bulk network average "
        f"(wins {b['frac_splits_kernel_beats_mean']*100:.0f}% of splits; "
        f"across {len(seed_results)} seeds the median skill ranges "
        f"{min(seed_skills):+.0f}% to {max(seed_skills):+.0f}%). On precise "
        f"post-processed clocks the effect is absent "
        f"({res['precise_clk']['skill_vs_raw_pct']['median']:.0f}% skill) -- satellite-side "
        f"errors have been removed and receiver oscillators are independent."
    )
    (OUT / "step_25_kernel_selection.json").write_text(json.dumps(res, indent=2))
    print(f"\n{res['headline']}")
    print(f"saved {OUT / 'step_25_kernel_selection.json'}")


if __name__ == "__main__":
    main()
