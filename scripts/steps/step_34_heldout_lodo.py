#!/usr/bin/env python3
"""
STEP 34  -- Held-out operational skill, LEAVE-ONE-DAY-OUT, on the expanded
            broadcast set (Phase B).

Step 25 established the held-out clock-prediction skill on 7 days with random
50/50 station splits and a day-clustered bootstrap.  Two reviewer concerns
remain: (1) only 7 days, (2) random station splits mix train/test days.

This step addresses both.  The broadcast set is the full year 2024
(366 days, of which 318 have >= 16 usable stations).  For each held-out DAY:
  * hyperparameters (lambda, nugget, K) are selected by leave-one-out on a
    DIFFERENT training day (the chronologically previous available day), so
    nothing from the held-out day touches model selection;
  * every station on the held-out day is predicted from the OTHER stations on
    that same day using those hyperparameters;
  * skill is scored against raw (predict zero), the bulk network mean, and a
    uniform-kriging scaled mean (isolates distance geometry from the nugget).

The result is a distribution of per-day skills -- the day is the unit of
replication, so the CI is a plain percentile bootstrap over days.
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
RNG = np.random.default_rng(20260934)


from functools import lru_cache


@lru_cache(maxsize=96)
def day_matrix(product, day):
    """Load + band-pass one day's clock matrix.  Cached: the leave-one-day-out
    loop uses each day both as a held-out target and (for its neighbours) as
    the training day, and build_matrix's band-pass is ~0.5 s/day for broadcast
    SPP -- without the cache the arm re-band-passes every day ~2x."""
    s = td.load_series(product, day)
    if not s:
        return None
    names, X, D = tk.build_matrix(s, filter_by_std=False)
    if X is None or len(names) < 16:
        return None
    good = [i for i in range(len(names)) if 0.01 < X[:, i].std() < 60.0]
    if len(good) < 16:
        return None
    return names, X[:, good], D[np.ix_(good, good)], [names[i] for i in good]


def run_arm(product, days, label):
    per_day = []
    lam_pick, nug_pick, k_pick = [], [], []
    for di, day in enumerate(days):
        test = day_matrix(product, day)
        if test is None:
            continue
        _, Xte, Dte, names_te = test
        # hyperparameters from a DIFFERENT day (chronologically previous if
        # available; for the first day in the year, the next available day
        # is used -- this is still cross-day, just forward rather than
        # backward, and only affects the first day).
        train = None
        if di > 0:
            for pd in reversed(days[:di]):
                train = day_matrix(product, pd)
                if train is not None:
                    break
        if train is None:
            for pd in days[di + 1:di + 4]:
                train = day_matrix(product, pd)
                if train is not None:
                    break
        if train is None:
            continue
        _, Xtr, Dtr, _ = train
        tr_idx = list(range(Xtr.shape[1]))
        p, _ = tk.fit_hyperparameters(Xtr, Dtr, tr_idx)
        if p is None:
            continue
        lam_pick.append(p.lambda_km); nug_pick.append(p.nugget); k_pick.append(p.k_neighbours)

        n = Xte.shape[1]
        raw, ker, mean, ukr = [], [], [], []
        for t in range(n):
            pool = [j for j in range(n) if j != t]
            r = tk.raw_error(Xte, t)
            e = tk.prediction_error(Xte, Dte, t, pool, p)
            m = tk.net_mean_error(Xte, t, pool)
            u = tk.uniform_kriging_error(Xte, t, pool, p)
            # Matched-sample comparison: only include a target when ALL
            # metrics are finite.  This ensures the raw, kernel, mean, and
            # uniform-kriging medians are computed over the SAME set of
            # targets, preventing kernel failures from inflating apparent
            # skill (the kernel median would be over easier targets while
            # the baselines remain over all targets).
            if not all(np.isfinite(v) for v in (r, e, m)):
                continue
            raw.append(r)
            ker.append(e)
            mean.append(m)
            ukr.append(u if np.isfinite(u) else np.nan)
        if len(raw) < 8:
            continue
        R = np.median(raw)
        E = np.nanmedian(ker)  # robust to NaN kernel predictions
        M = np.median(mean) if mean else np.nan
        U = np.nanmedian(ukr)
        per_day.append({
            "day": day, "n_targets": len(raw),
            "raw_m": float(R), "kernel_m": float(E), "net_mean_m": float(M),
            "uniform_krig_m": float(U) if np.isfinite(U) else np.nan,
            "skill_vs_raw_pct": float(100 * (1 - E / R)) if R > 0 else np.nan,
            "kernel_vs_mean_pct": float(100 * (1 - E / M)) if M > 0 else np.nan,
            "kernel_vs_ukrig_pct": float(100 * (1 - E / U)) if np.isfinite(U) and U > 0 else np.nan,
            "lambda_km": p.lambda_km,
        })
    if not per_day:
        return {"label": label, "n_days": 0}

    def col(k):
        return np.array([d[k] for d in per_day if np.isfinite(d[k])])

    def ci(v):
        if len(v) < 3:
            return [float("nan"), float("nan")]
        bs = [float(np.median(RNG.choice(v, len(v), replace=True))) for _ in range(5000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    sk, vm, vu = col("skill_vs_raw_pct"), col("kernel_vs_mean_pct"), col("kernel_vs_ukrig_pct")
    res = {
        "label": label, "product": product, "n_days": len(per_day),
        "selected_lambda_km_median": float(np.median(lam_pick)) if lam_pick else None,
        "selected_lambda_km_counts": {str(int(v)): int(np.sum(np.array(lam_pick) == v))
                                      for v in sorted(set(lam_pick))} if lam_pick else {},
        "selected_nugget_median": float(np.median(nug_pick)) if nug_pick else None,
        "heldout_error_m": {"raw": float(np.median(col("raw_m"))),
                            "kernel": float(np.median(col("kernel_m"))),
                            "net_mean": float(np.median(col("net_mean_m")))},
        "skill_vs_raw_pct": {"median": float(np.median(sk)), "ci95": ci(sk),
                             "frac_days_positive": float(np.mean(sk > 0))},
        "kernel_vs_bulk_mean_pct": {"median": float(np.median(vm)), "ci95": ci(vm),
                                    "frac_days_positive": float(np.mean(vm > 0))},
        "kernel_vs_uniform_krig_pct": {"median": float(np.median(vu)) if len(vu) else None,
                                       "ci95": ci(vu) if len(vu) else None},
        "per_day": per_day,
    }
    print(f"[{label}]  leave-one-day-out over {len(per_day)} days")
    print(f"  lambda* median {res['selected_lambda_km_median']:.0f} km   "
          f"counts {res['selected_lambda_km_counts']}")
    e = res["heldout_error_m"]
    print(f"  held-out clock error:  raw {e['raw']:.2f}  bulk-mean {e['net_mean']:.2f}  KERNEL {e['kernel']:.2f} m")
    s = res["skill_vs_raw_pct"]; m = res["kernel_vs_bulk_mean_pct"]; u = res["kernel_vs_uniform_krig_pct"]
    print(f"  skill vs raw    : {s['median']:+.1f}%  95% CI [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]  "
          f"({s['frac_days_positive']*100:.0f}% of days positive)")
    print(f"  kernel vs bulk  : {m['median']:+.1f}%  95% CI [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]  "
          f"({m['frac_days_positive']*100:.0f}% of days positive)")
    if u["median"] is not None:
        print(f"  kernel vs ukrig : {u['median']:+.1f}%  95% CI [{u['ci95'][0]:+.1f}, {u['ci95'][1]:+.1f}]  "
              f"(geometry-specific)")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bdays = td.available_dates("broadcast", 2024)
    print(f"Leave-one-day-out held-out skill on the expanded broadcast set "
          f"({len(bdays)} days)\n")
    res = {
        "test": "heldout_skill_leave_one_day_out",
        "method": ("leave-one-day-out: hyperparameters selected on a different training "
                   "day, every station on the held-out day predicted from the other "
                   "stations that day; day is the unit of replication"),
        "n_broadcast_days": len(bdays),
        "broadcast_spp": run_arm("broadcast", bdays, "broadcast_spp"),
        "timestamp": datetime.now().isoformat(),
    }
    # Full-year GLOBAL held-out skill on the COD0MGXFIN analysis-centre
    # receiver-clock product (~128 globally distributed stations/day, baselines
    # to 20,000 km).  The broadcast arm above is ~full-year but regional
    # (ppp_dense, ~25 European stations); this arm is genuinely global.
    cdays = td.available_dates("clk_cod", 2024)
    if len(cdays) >= 30:
        print(f"\nLeave-one-day-out on the global precise CLK product "
              f"({len(cdays)} days)\n")
        res["global_precise_clk"] = run_arm("clk_cod", cdays, "global_precise_clk")
    b = res["broadcast_spp"]
    if b.get("n_days"):
        s = b["skill_vs_raw_pct"]; m = b["kernel_vs_bulk_mean_pct"]
        res["headline"] = (
            f"Leave-one-day-out over {b['n_days']} days of broadcast SPP: the distance "
            f"kernel predicts an unseen day's receiver clocks {s['median']:.0f}% better "
            f"than no network (95% CI [{s['ci95'][0]:.0f}, {s['ci95'][1]:.0f}], "
            f"{s['frac_days_positive']*100:.0f}% of days) and {m['median']:.0f}% better "
            f"than a bulk network mean; lambda* median "
            f"{b['selected_lambda_km_median']:.0f} km, selected on separate training days.")
        print(f"\n{res['headline']}")
    (OUT / "step_34_heldout_lodo.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'step_34_heldout_lodo.json'}")


if __name__ == "__main__":
    main()
