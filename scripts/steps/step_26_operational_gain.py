#!/usr/bin/env python3
"""
STEP 26  -- Position-domain operational gain, with an HONEST baseline decomposition.

Steps 21 and 24 compared "clock prior of the kernel's quality" against "no clock
prior at all".  That conflates two very different things:

    (i)  the gain from constraining the clock AT ALL, which any prior delivers,
         including a useless one; and
   (ii)  the gain the NETWORK actually supplies, i.e. the kernel prediction
         being better than what the station already knows about itself.

This step separates them with three priors, all of MEASURED quality, all
scored on held-out stations with the nested-CV kernel from Step 25:

    none      -- clock free (standard per-epoch PPP)
    self      -- prior at the station's own band-limited variability (raw).
                 "No network at all": you only know the clock is a decent
                 oscillator near its own trend.
    net_mean  -- prior at the bulk network-average prediction error
    kernel    -- prior at the distance-kernel prediction error

The operational claim the paper should make is
        kernel  vs  net_mean   (the kernel's marginal value), and
        kernel  vs  self       (the network's marginal value),
NOT kernel vs none.

Geometry: real SP3, per epoch, top-k highest-elevation satellites,
state [x y z clk] (ZTD held from a model, as in a short obstruction interval).
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
from scripts.lib.tep_geometry import load_sp3, sat_grid, elev_los, ELEV_MASK  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
K_VALUES = [3, 4, 5, 6, 8, 99]
RNG = np.random.default_rng(20260903)
PRIORS = ("none", "self", "net_mean", "uniform_krig", "kernel")


def pos_sigma(los, sig_clk):
    """3D position RSS uncertainty for state [x y z clk]; sig_clk None -> clock free.

    Returns sqrt(trace(Cov[:3,:3])) -- the root-sum-square of the three
    coordinate standard deviations.  This is NOT the conventional 3D 1-sigma
    (semi-major axis of the error ellipsoid, sqrt(max(eigenvalues))); it is
    the DRMS-like quantity that aggregates all three axes.  We report it
    because it is what the manuscript tables use and it is a monotonic
    function of the clock prior quality.  The conventional 1-sigma would be
    smaller by a factor of ~1.2-1.5 for typical GNSS geometry.

    For k < 4 satellites the normal matrix is rank-deficient; we detect this
    via condition number rather than relying on np.linalg.inv to raise
    (it returns a huge finite value for near-singular matrices).
    """
    n = len(los)
    A = np.column_stack([-los, np.ones(n)])
    N = A.T @ A
    if sig_clk is not None:
        if not np.isfinite(sig_clk) or sig_clk <= 0:
            # A non-finite or non-positive prior width is invalid; return NaN
            # so the caller can skip it rather than silently falling back to
            # the free-clock (no-prior) solution, which would bias the
            # comparison by making invalid priors look identical to no prior.
            return np.nan
        N = N + np.diag([0.0, 0.0, 0.0, 1.0 / sig_clk ** 2])
    # Detect rank deficiency via condition number
    sv = np.linalg.svd(N, compute_uv=False)
    if sv[-1] < 1e-10 * sv[0] or sv[-1] < 1e-12:
        return np.inf
    try:
        Cov = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        return np.inf
    v = np.trace(Cov[:3, :3])
    return float(np.sqrt(v)) if v > 0 else np.inf


N_SPLITS_PER_DAY = 40      # Must match step_25's N_SPLITS so the position-domain
#                          # table uses the same held-out evaluations as the
#                          # clock-domain skill reported in step_25.  Both scripts
#                          # seed their RNG with the same constant, so the 40
#                          # random permutations are identical.


def heldout_prior_quality(product, date_tag, split_rng=None):
    """
    Per held-out station: the MEASURED quality (m) of each prior.
    Kernel hyperparameters are fitted on the train half; the held-out station is
    predicted from train references only -- no leakage.

    Two sigma variants are returned:
      actual     -- the station's ACTUAL prediction error. This gives the HONEST
                    position-domain gain (the gain achievable when the prior sigma
                    perfectly matches the true prediction quality). Requires
                    knowledge of the true target error, unavailable in deployment.
      train_est  -- the training-set LOO error for kernel and net_mean priors
                    (deployable: estimated from training stations only, before
                    seeing the target). The 'self' sigma uses the median raw
                    error across training stations as a network-wide estimate
                    of typical clock variability — a conservative proxy for the
                    station's own history that would be available in a real
                    deployment. The kernel train_est is the selected model's
                    in-sample LOO, which is optimistically biased relative to
                    the true test error because the hyperparameters were chosen
                    to minimise it; this bias is reported transparently by
                    comparing the oracle and deployable variants.
    """
    s = td.load_series(product, date_tag)
    if not s:
        return {}
    # Build matrix WITHOUT std filtering — the train/test split must not
    # be influenced by the held-out stations' own data.  Degenerate or
    # jumped stations are removed from the TRAINING set only, inside
    # each split, so the test set is never selected based on its own
    # residual statistics (leakage-free station QC).
    names, X, D = tk.build_matrix(s, filter_by_std=False)
    if X is None or len(names) < 16:
        return {}
    n = len(names)
    rng = split_rng if split_rng is not None else RNG
    out = {}

    permutations = [rng.permutation(n) for _ in range(N_SPLITS_PER_DAY)]

    def _run_split(idx):
        tr, te = sorted(idx[: n // 2].tolist()), sorted(idx[n // 2:].tolist())
        # Training-side QC: compute per-station std from the TRAINING
        # columns only, AFTER the split, so the test set is never
        # selected based on its own residual statistics (leakage-free
        # station QC, consistent with step_25).
        tr_std = np.array([X[:, i].std() for i in tr])
        tr = [i for i, sd in zip(tr, tr_std) if 0.01 < sd < 60.0]
        if len(tr) < 8 or len(te) < 2:
            return None
        p, tr_err = tk.fit_hyperparameters(X, D, tr)
        if p is None:
            return None
        # Training-estimated sigmas: deployable estimates from the training set only
        tr_self_errs = [tk.raw_error(X, t) for t in tr]
        tr_mean_errs = [tk.net_mean_error(X, t, tr) for t in tr]
        tr_ukrig_errs = [tk.uniform_kriging_error(X, t, tr, p) for t in tr]
        tr_self = float(np.nanmedian(tr_self_errs)) if tr_self_errs else np.nan
        tr_mean = float(np.nanmedian(tr_mean_errs)) if tr_mean_errs else np.nan
        tr_ukrig = float(np.nanmedian(tr_ukrig_errs)) if tr_ukrig_errs else np.nan
        tr_kernel = float(tr_err) if np.isfinite(tr_err) else np.nan
        
        res = []
        for t in te:
            r = tk.raw_error(X, t)
            e = tk.prediction_error(X, D, t, tr, p)
            m = tk.net_mean_error(X, t, tr)
            u = tk.uniform_kriging_error(X, t, tr, p)
            if not all(np.isfinite(v) for v in (r, e, m, u)):
                continue
            res.append({
                "name": names[t],
                "station": s[names[t]]["station"],
                "self": r, "net_mean": m,
                "uniform_krig": u if np.isfinite(u) else np.nan,
                "kernel": e,
                "train_est_self": tr_self,
                "train_est_net_mean": tr_mean,
                "train_est_uniform_krig": tr_ukrig,
                "train_est_kernel": tr_kernel,
                "params": p.as_dict()
            })
        return res

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(_run_split)(idx) for idx in permutations)
    
    for res_list in results:
        if res_list is not None:
            for item in res_list:
                # keep every (split, station) evaluation as a distinct key
                out[f"{item['name']}#{len(out)}"] = item
                del out[f"{item['name']}#{len(out)-1}"]["name"] # cleanup
    
    return out

def run(product, date_tags, label):
    # Fresh RNG per arm, seeded with the same constant as step_25 so the
    # 40 random station permutations are identical across the two scripts.
    # The previous code used the module-level RNG, which was consumed by
    # the broadcast arm before the precise arm started — so the precise-arm
    # permutations diverged from step_25, breaking the correspondence
    # between the clock-domain skill (step_25) and the position-domain
    # table (step_26) for the precise product.
    arm_rng = np.random.default_rng(20260903)
    # Two sigma variants: 'actual' (upper bound) and 'train_est' (deployable)
    variants = ("actual", "train_est")
    acc = {v: {k: {p: [] for p in PRIORS} for k in K_VALUES} for v in variants}
    singular = {k: [] for k in K_VALUES}
    quality = {p: [] for p in ("self", "net_mean", "uniform_krig", "kernel")}
    grid = np.arange(0, 86400.0, 1800.0)

    for dtag in date_tags:
        sp3 = load_sp3(dtag)
        if sp3 is None:
            continue
        sp3g = sat_grid(sp3, grid)
        skill = heldout_prior_quality(product, dtag, split_rng=arm_rng)
        for nm, sk in skill.items():
            for q in quality:
                if q == "uniform_krig":
                    uv = sk.get("uniform_krig", np.nan)
                    if np.isfinite(uv):
                        quality[q].append(uv)
                else:
                    quality[q].append(sk[q])
            sta = sk["station"].ecef_m
            uval = sk.get("uniform_krig", np.nan)
            sig_actual = {"none": None, "self": sk["self"],
                          "net_mean": sk["net_mean"],
                          "uniform_krig": uval if np.isfinite(uval) else np.nan,
                          "kernel": sk["kernel"]}
            # Deployable variant: all sigmas estimated from the training set
            # only (the only data available in real-time deployment).  The
            # previous code used sk["self"] (the held-out station's actual
            # error) for the self baseline, making the deployable
            # kernel_vs_self gains optimistic.  Use tr_self instead.
            tu = sk.get("train_est_uniform_krig", np.nan)
            sig_train = {"none": None, "self": sk.get("train_est_self", sk["self"]),
                         "net_mean": sk["train_est_net_mean"],
                         "uniform_krig": tu if np.isfinite(tu) else np.nan,
                         "kernel": sk["train_est_kernel"]}
            sig_by_var = {"actual": sig_actual, "train_est": sig_train}
            for gi in range(len(grid)):
                vis = []
                for sname, pg in sp3g.items():
                    pp = pg[gi]
                    if not np.all(np.isfinite(pp)):
                        continue
                    e, u, _ = elev_los(pp, sta)
                    if e > ELEV_MASK:
                        vis.append((e, u))
                if len(vis) < 3:
                    continue
                vis.sort(key=lambda z: -z[0])
                for k in K_VALUES:
                    kk = min(k, len(vis))
                    if kk < 3:
                        continue
                    los = np.array([u for _, u in vis[:kk]])
                    vals_none = pos_sigma(los, None)
                    singular[k].append(not np.isfinite(vals_none))
                    for v in variants:
                        for pr in PRIORS:
                            acc[v][k][pr].append(pos_sigma(los, sig_by_var[v][pr]))

    if not quality["kernel"]:
        return {"label": label, "n": 0}

    qmed = {q: float(np.median(v)) for q, v in quality.items() if v}
    has_ukrig = "uniform_krig" in qmed
    print(f"[{label}]  held-out prior quality (m):  self {qmed.get('self',0):.2f}   "
          f"bulk-mean {qmed.get('net_mean',0):.2f}   ukrig {qmed.get('uniform_krig',0):.2f}   "
          f"kernel {qmed.get('kernel',0):.2f}")
    skill_parts = {
        "bulk_mean_vs_self": 100 * (1 - qmed["net_mean"] / qmed["self"]),
        "kernel_vs_self": 100 * (1 - qmed["kernel"] / qmed["self"]),
        "kernel_vs_bulk_mean": 100 * (1 - qmed["kernel"] / qmed["net_mean"]),
    }
    if has_ukrig:
        skill_parts["uniform_krig_vs_self"] = 100 * (1 - qmed["uniform_krig"] / qmed["self"])
        skill_parts["kernel_vs_uniform_krig"] = 100 * (1 - qmed["kernel"] / qmed["uniform_krig"])
    if has_ukrig:
        ukrig_str = f"ukrig {skill_parts['uniform_krig_vs_self']:+.1f}%   "
        ukrig_over_str = f"kernel over ukrig {skill_parts['kernel_vs_uniform_krig']:+.1f}%"
    else:
        ukrig_str = ""
        ukrig_over_str = ""
    print(f"    -> network's marginal skill: bulk {skill_parts['bulk_mean_vs_self']:+.1f}%   "
          f"{ukrig_str}"
          f"kernel {skill_parts['kernel_vs_self']:+.1f}%   "
          f"kernel over bulk {skill_parts['kernel_vs_bulk_mean']:+.1f}%   "
          f"{ukrig_over_str}")

    tables = {}
    for v in variants:
        table = {}
        vlabel = "honest (actual sigma)" if v == "actual" else "deployable (train-est sigma)"
        print(f"  [{vlabel} sigma]  sats |  none     self  ukrig  bulk   kernel  | "
              f"kvSelf  kvBulk  kvUkrig")
        for k in K_VALUES:
            med = {}
            for pr in PRIORS:
                a = np.array(acc[v][k][pr]); a = a[np.isfinite(a)]
                med[pr] = float(np.median(a)) if len(a) else float("inf")
            frac_sing = float(np.mean(singular[k])) if singular[k] else 0.0
            gain_self = 100.0 * (1 - med["kernel"] / med["self"]) if np.isfinite(med["self"]) and med["self"] > 0 else np.nan
            gain_bulk = 100.0 * (1 - med["kernel"] / med["net_mean"]) if np.isfinite(med["net_mean"]) and med["net_mean"] > 0 else np.nan
            gain_ukrig = 100.0 * (1 - med["kernel"] / med["uniform_krig"]) if np.isfinite(med["uniform_krig"]) and med["uniform_krig"] > 0 else np.nan
            table[str(k)] = {"n_epochs": len(acc[v][k]["kernel"]),
                             "frac_singular_no_prior": frac_sing,
                             "pos_sigma_m": med,
                             "kernel_vs_self_pct": gain_self,
                             "kernel_vs_bulk_mean_pct": gain_bulk,
                             "kernel_vs_uniform_krig_pct": gain_ukrig}
            tag = "all" if k == 99 else str(k)
            nn = f"{med['none']:8.1f}" if np.isfinite(med["none"]) else "  SINGULAR"
            print(f"    {tag:>4} | {nn} {med['self']:7.2f} {med['uniform_krig']:6.2f} "
                  f"{med['net_mean']:6.2f} {med['kernel']:8.2f}  |"
                  f" {gain_self:+6.1f}% {gain_bulk:+6.1f}% {gain_ukrig:+7.1f}%")
        tables[v] = table

    return {"label": label, "product": product,
            "heldout_prior_quality_m": qmed,
            "network_marginal_skill_pct": skill_parts,
            "by_satellite_count": tables["actual"],
            "by_satellite_count_train_est": tables["train_est"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [f"2024{d:03d}" for d in range(100, 107)]
    print("Position-domain gain with an honest baseline decomposition")
    print("  (priors of MEASURED, held-out quality; 'self' = no network at all)\n")
    res = {
        "test": "operational_gain_honest_baselines",
        "baselines": {
            "none": "clock free -- standard per-epoch PPP",
            "self": "prior at the station's own band-limited variability (NO NETWORK)",
            "net_mean": "prior at the bulk network-average prediction error (scale=1)",
            "uniform_krig": "prior at the scaled-mean prediction error (same nugget, NO geometry)",
            "kernel": "prior at the nested-CV distance-kernel prediction error"},
        "broadcast_spp": run("broadcast", days, "broadcast_spp"),
        "precise_clk": run("clk_cod", td.available_dates("clk_cod", 2024), "precise_clk"),
        "timestamp": datetime.now().isoformat(),
    }
    b = res["broadcast_spp"]
    if b.get("by_satellite_count"):
        k4 = b["by_satellite_count"].get("4", {})
        k8 = b["by_satellite_count"].get("8", {})
        k4t = b.get("by_satellite_count_train_est", {}).get("4", {})
        k8t = b.get("by_satellite_count_train_est", {}).get("8", {})
        res["headline"] = (
            f"Broadcast SPP, held-out stations: the network kernel is "
            f"{b['network_marginal_skill_pct']['kernel_vs_self']:.0f}% better than knowing nothing "
            f"but the station's own clock variability, and "
            f"{b['network_marginal_skill_pct']['kernel_vs_bulk_mean']:.0f}% better than a bulk "
            f"network average. In the position domain under obstruction (honest, "
            f"prior sigma = actual error) this is worth "
            f"{k4.get('kernel_vs_self_pct', float('nan')):.0f}% at 4 visible satellites and "
            f"{k8.get('kernel_vs_self_pct', float('nan')):.0f}% at 8, versus a no-network prior. "
            f"With a deployable training-estimated sigma the corresponding gains are "
            f"{k4t.get('kernel_vs_self_pct', float('nan')):.0f}% (4 sats) and "
            f"{k8t.get('kernel_vs_self_pct', float('nan')):.0f}% (8 sats).")
        print(f"\n{res['headline']}")
    (OUT / "step_26_operational_gain.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'step_26_operational_gain.json'}")


if __name__ == "__main__":
    main()
