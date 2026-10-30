#!/usr/bin/env python3
"""
STEP 156  --  Weighted leave-one-band-out with per-band spatial kernels.

Step 155 used simple averaging of per-band predictions.  Step 135 found
the optimal fusion weights are tidal=2.25, post_tidal=1.75, intermediate=1.00.

This step re-runs the leave-one-band-out with the PROPER weights and
per-band spatial kernels (Step 134 methodology).  Each band gets its own
lambda_km for the distance kernel, and the per-band predictions are
weighted-averaged.

If the tidal band is essential with correct weighting, removing it should
hurt.

Output: results/outputs/step_156_weighted_loob.json
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

OUT = ROOT / "results" / "outputs" / "step_156_weighted_loob.json"
PRODUCT = "clk_cod"
YEAR = 2025
RNG_SEED = 20260920

EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10

# Fusion weights from Step 135 (frozen, tuned on 2024, applied to 2025)
WEIGHTS = {"tidal": 2.25, "post_tidal": 1.75, "intermediate": 1.00}

# Band definitions — per-band lambdas from Paper 1
BANDS = [
    {"name": "tidal",        "f_lo": 10e-6,  "f_hi": 35e-6,   "lambda_km": 4677},
    {"name": "post_tidal",   "f_lo": 35e-6,  "f_hi": 100e-6,  "lambda_km": 1502},
    {"name": "intermediate", "f_lo": 100e-6, "f_hi": 500e-6,  "lambda_km": 1459},
]

FULL_BAND = {"f_lo": 10e-6, "f_hi": 500e-6, "lambda_km": 4677}


def _multi_scales(lam):
    """Multi-scale expansion: lambda/1.6, lambda, lambda*1.6."""
    return (lam / 1.6, lam, lam * 1.6)


def _rho_multi(d, scales):
    """Mean of exp(-d/L) over the multi-scale set."""
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-8 * max(1.0, float(np.max(np.abs(ev)))):
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
    """Compute phase Gram for a single band."""
    try:
        r = spectral_phase_gram(X_prior, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                min_bins=5, min_low_cycles=0.85)
        return r["predictive_gram"]
    except (ValueError, np.linalg.LinAlgError):
        return None


def _band_prediction(G_band, band_info, Dg, R, pool, b, ei, nugget):
    """Compute a single per-band kriging prediction using band-specific lambda.

    Returns (prediction, success).  prediction is a float; success is False
    if the band Gram or solve failed.
    """
    if G_band is None:
        return None, False
    scales = _multi_scales(band_info["lambda_km"])
    rho_rr = _rho_multi(Dg[np.ix_(pool, pool)], scales)
    rho_rt = _rho_multi(Dg[b, pool], scales)
    G_pp = G_band[np.ix_(pool, pool)]
    K = rho_rr * G_pp + nugget * np.eye(len(pool))
    try:
        w = _reg_solve(K, rho_rt * G_band[b, pool])
        rr = R[ei, pool]
        if not np.isfinite(rr).all():
            return None, False
        pred = float(w @ rr)
        if not np.isfinite(pred):
            return None, False
        return pred, True
    except np.linalg.LinAlgError:
        return None, False


def _fused_prediction(band_grams, band_infos, weights, Dg, R, pool, b, ei, nugget):
    """Weighted average of per-band predictions using per-band lambdas.

    Returns (prediction, n_valid_bands, n_total_bands).
    """
    preds = []
    total_w = 0.0
    for G_band, band_info in zip(band_grams, band_infos):
        p, ok = _band_prediction(G_band, band_info, Dg, R, pool, b, ei, nugget)
        if ok:
            w = weights.get(band_info["name"], 1.0)
            preds.append(w * p)
            total_w += w
    if not preds or total_w < 1e-12:
        return None, 0, len(band_grams)
    return sum(preds) / total_w, len(preds), len(band_grams)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("Step 156: Weighted leave-one-band-out (per-band lambdas)", flush=True)
    print(f"  Weights: {WEIGHTS}", flush=True)
    print(f"  Per-band lambdas: {[(b['name'], b['lambda_km']) for b in BANDS]}", flush=True)
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

    # Configurations: each uses per-band predictions with per-band lambdas
    # "weighted_full": all 3 bands with weights 2.25/1.75/1.0
    # "weighted_no_X": remove band X, renormalise remaining weights
    # "simple_avg": all 3 bands with equal weights
    # "simple_no_tidal": simple average without tidal
    # "tidal_only": tidal band alone (weight cancels for single band)
    # "post_tidal_only": post-tidal band alone
    # "intermediate_only": intermediate band alone
    # "full_band_baseline": single-lambda full-band kernel
    configs = {
        "weighted_full": [],
        "weighted_no_tidal": [],
        "weighted_no_post": [],
        "weighted_no_inter": [],
        "simple_avg": [],
        "simple_no_tidal": [],
        "tidal_only": [],
        "post_tidal_only": [],
        "intermediate_only": [],
        "full_band_baseline": [],
    }
    # Track band failures per config
    config_failures = {k: 0 for k in configs}
    # Track day tags for day-aligned comparison
    config_day_tags = {k: [] for k in configs}

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

        # Compute per-band Grams
        band_grams_all = []
        for band in BANDS:
            band_grams_all.append(_band_gram(X_prior, dt_s, band))
        G_full = _band_gram(X_prior, dt_s, FULL_BAND)

        # Define which bands and weights each config uses
        config_defs = {
            "weighted_full":       (BANDS, WEIGHTS),
            "weighted_no_tidal":   ([b for b in BANDS if b["name"] != "tidal"], WEIGHTS),
            "weighted_no_post":    ([b for b in BANDS if b["name"] != "post_tidal"], WEIGHTS),
            "weighted_no_inter":   ([b for b in BANDS if b["name"] != "intermediate"], WEIGHTS),
            "simple_avg":          (BANDS, {n: 1.0 for n in WEIGHTS}),
            "simple_no_tidal":     ([b for b in BANDS if b["name"] != "tidal"], {n: 1.0 for n in WEIGHTS}),
            "tidal_only":          ([BANDS[0]], {"tidal": 1.0}),
            "post_tidal_only":     ([BANDS[1]], {"post_tidal": 1.0}),
            "intermediate_only":   ([BANDS[2]], {"intermediate": 1.0}),
        }

        # Pre-compute Grams for each config's band set
        config_grams = {}
        for cfg_name, (cfg_bands, cfg_weights) in config_defs.items():
            grams = []
            for band in cfg_bands:
                idx = BANDS.index(band)
                grams.append(band_grams_all[idx])
            config_grams[cfg_name] = (grams, cfg_bands, cfg_weights)

        day_sq = {k: [] for k in configs}
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue
            for ei in eval_i:
                truth = R[ei, b]
                if not np.isfinite(truth):
                    continue
                # Per-band configs
                for cfg_name, (grams, cfg_bands, cfg_weights) in config_grams.items():
                    pred, n_valid, n_total = _fused_prediction(
                        grams, cfg_bands, cfg_weights, Dg, R, pool, b, ei, s64.NUGGET)
                    if pred is not None:
                        day_sq[cfg_name].append((truth - pred) ** 2)
                    else:
                        day_sq[cfg_name].append(np.nan)
                        config_failures[cfg_name] += 1
                # Full-band baseline (single lambda)
                if G_full is not None:
                    scales = _multi_scales(FULL_BAND["lambda_km"])
                    rho_rr = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                    rho_rt = _rho_multi(Dg[b, pool], scales)
                    K = rho_rr * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool))
                    try:
                        w = _reg_solve(K, rho_rt * G_full[b, pool])
                        rr = R[ei, pool]
                        if np.isfinite(rr).all():
                            pred = float(w @ rr)
                            if np.isfinite(pred):
                                day_sq["full_band_baseline"].append((truth - pred) ** 2)
                            else:
                                day_sq["full_band_baseline"].append(np.nan)
                        else:
                            day_sq["full_band_baseline"].append(np.nan)
                    except np.linalg.LinAlgError:
                        day_sq["full_band_baseline"].append(np.nan)
                else:
                    day_sq["full_band_baseline"].append(np.nan)

        for cfg in configs:
            vals = np.array(day_sq[cfg])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                configs[cfg].append(float(np.sqrt(np.mean(vals))))
                config_day_tags[cfg].append(tag)

        if (k + 1) % 10 == 0:
            wf = np.median(configs['weighted_full']) if configs['weighted_full'] else 0
            nt = np.median(configs['weighted_no_tidal']) if configs['weighted_no_tidal'] else 0
            print(f"  [{k+1}/{len(eval_tags)}] weighted={wf:.4f}, "
                  f"no_tidal={nt:.4f} m", flush=True)

    # Compute results — day-aligned comparisons
    results = {}
    # Find common days across all configs for fair comparison
    all_tags = set(config_day_tags.get("weighted_full", []))
    for cfg in configs:
        all_tags &= set(config_day_tags.get(cfg, []))
    common_tags = sorted(all_tags)

    for cfg_name, rms_list in configs.items():
        rms = np.array(rms_list)
        tags = config_day_tags[cfg_name]
        if len(rms) < 4:
            results[cfg_name] = {"n": len(rms), "band_failures": config_failures[cfg_name]}
            continue
        # Day-aligned comparison against weighted_full and full_band
        results[cfg_name] = {
            "median_rms_m": float(np.median(rms)),
            "n": len(rms),
            "band_failures": config_failures[cfg_name],
        }
        # vs weighted_full (day-aligned)
        if cfg_name != "weighted_full" and "weighted_full" in configs:
            w_tags = config_day_tags["weighted_full"]
            w_rms = np.array(configs["weighted_full"])
            # Align by tag
            tag_to_w = dict(zip(w_tags, w_rms))
            tag_to_cfg = dict(zip(tags, rms))
            shared = [t for t in common_tags if t in tag_to_w and t in tag_to_cfg]
            if len(shared) >= 4:
                vs_w = 100.0 * (1.0 - np.array([tag_to_cfg[t] for t in shared]) /
                                np.array([tag_to_w[t] for t in shared]))
                results[cfg_name]["vs_weighted_pct"] = _ci(vs_w, rng)
        # vs full_band (day-aligned)
        if cfg_name != "full_band_baseline" and "full_band_baseline" in configs:
            f_tags = config_day_tags["full_band_baseline"]
            f_rms = np.array(configs["full_band_baseline"])
            tag_to_f = dict(zip(f_tags, f_rms))
            tag_to_cfg = dict(zip(tags, rms))
            shared = [t for t in common_tags if t in tag_to_f and t in tag_to_cfg]
            if len(shared) >= 4:
                vs_f = 100.0 * (1.0 - np.array([tag_to_cfg[t] for t in shared]) /
                                np.array([tag_to_f[t] for t in shared]))
                results[cfg_name]["vs_full_band_pct"] = _ci(vs_f, rng)
        # Self-comparison (weighted_full vs itself = 0)
        if cfg_name == "weighted_full":
            results[cfg_name]["vs_weighted_pct"] = {"median": 0.0, "ci95": [0.0, 0.0], "n": len(rms), "frac_positive": None}
        if cfg_name == "full_band_baseline":
            results[cfg_name]["vs_full_band_pct"] = {"median": 0.0, "ci95": [0.0, 0.0], "n": len(rms), "frac_positive": None}

    interp = []
    interp.append(f"Weighted leave-one-band-out (per-band lambdas): "
                  f"{len(configs['weighted_full'])} eval days, 13-day window.")
    interp.append(f"  Weights: tidal={WEIGHTS['tidal']}, post_tidal={WEIGHTS['post_tidal']}, "
                  f"intermediate={WEIGHTS['intermediate']}")
    interp.append(f"  Per-band lambdas: tidal=4677, post_tidal=1502, intermediate=1459 km")
    interp.append(f"  Band failures: " + ", ".join(f"{k}={v}" for k, v in config_failures.items() if v > 0))

    for cfg in ["weighted_full", "weighted_no_tidal", "weighted_no_post", "weighted_no_inter",
                "simple_avg", "simple_no_tidal", "tidal_only", "post_tidal_only",
                "intermediate_only", "full_band_baseline"]:
        r = results.get(cfg, {})
        vw = r.get("vs_weighted_pct", {})
        vf = r.get("vs_full_band_pct", {})
        med_w = vw.get("median")
        ci_w = vw.get("ci95")
        med_f = vf.get("median")
        ci_f = vf.get("ci95")
        if med_w is not None or med_f is not None:
            w_str = f"vs weighted {med_w:+.2f}% CI{ci_w}" if med_w is not None else "vs weighted N/A"
            f_str = f"vs full-band {med_f:+.2f}% CI{ci_f}" if med_f is not None else "vs full-band N/A"
            interp.append(f"  {cfg:25s}: {w_str}, {f_str}, RMS={r.get('median_rms_m', 0):.4f}, n={r.get('n', 0)}")

    # Key test: does removing tidal band hurt with proper weights?
    no_tidal_vs_weighted = results.get("weighted_no_tidal", {}).get("vs_weighted_pct", {})
    if no_tidal_vs_weighted.get("median") is not None:
        hurt = no_tidal_vs_weighted["median"]
        ci = no_tidal_vs_weighted["ci95"]
        if hurt < 0 and ci[1] < 0:
            interp.append(f"\n  REMOVING THE TIDAL BAND HURTS ({hurt:+.2f}% CI{ci}).")
            interp.append("  With proper fusion weights (tidal=2.25x) and per-band lambdas, "
                          "the gravitational tidal band IS essential.")
        elif hurt < 0:
            interp.append(f"\n  Removing the tidal band slightly hurts ({hurt:+.2f}% CI{ci}).")
        else:
            interp.append(f"\n  Removing the tidal band does NOT hurt ({hurt:+.2f}% CI{ci}).")
            interp.append("  Even with proper weights and per-band lambdas, the tidal band is not essential.")

    # Compare weighted vs simple
    weighted_vs_full = results.get("weighted_full", {}).get("vs_full_band_pct", {})
    simple_vs_full = results.get("simple_avg", {}).get("vs_full_band_pct", {})
    if weighted_vs_full.get("median") is not None and simple_vs_full.get("median") is not None:
        interp.append(f"\n  Weighted multi-band vs full-band: {weighted_vs_full['median']:+.2f}%")
        interp.append(f"  Simple multi-band vs full-band: {simple_vs_full['median']:+.2f}%")
        interp.append(f"  Weighted advantage over simple: "
                      f"{weighted_vs_full['median'] - simple_vs_full['median']:+.2f}%")

    verdict = {
        "step": 156,
        "title": "Weighted leave-one-band-out with per-band spatial kernels",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR,
        "weights": WEIGHTS,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "method": "per-band predictions with per-band lambdas, weighted average",
        "results": results,
        "interpretation": interp,
        "pre_registered_prediction": (
            "Step 135 found the optimal fusion weight for the tidal band is 2.25x (highest of all bands). "
            "Step 155 used simple averaging (1x for all), which under-weights the tidal band. "
            "With proper weights and per-band spatial kernels, removing the tidal band should hurt "
            "performance, confirming the gravitational tidal structure is essential."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"\n  Output: {OUT}", flush=True)
    for line in interp:
        print(line, flush=True)


if __name__ == "__main__":
    main()
