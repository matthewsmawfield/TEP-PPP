#!/usr/bin/env python3
"""
STEP 155  --  Leave-one-band-out: which frequency band drives the multi-band advantage?

The multi-band TEP prior (Step 134) uses three bands with physics-fixed lambdas:
  - Tidal (10-35 uHz, lambda=4677 km)  -- where gravitational tides live
  - Post-tidal (35-100 uHz, lambda=1502 km)
  - Intermediate (100-500 uHz, lambda=1459 km)

If the tidal band is essential, removing it should hurt performance significantly.
If the continuum bands drive the advantage, removing the tidal band should not hurt.

This directly tests whether the gravitational tidal frequencies carry the
predictive signal — the core TEP prediction.

METHOD: Per-band predictions are computed using each band's own lambda_km
(matching Step 134), then averaged. Leave-one-band-out removes one band from
the average. Single-band-only uses just one band's prediction.

Output: results/outputs/step_155_leave_one_band_out.json
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

OUT = ROOT / "results" / "outputs" / "step_155_leave_one_band_out.json"
PRODUCT = "clk_cod"
YEAR = 2025
RNG_SEED = 20260920

EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10

# Band definitions (from Step 134)
BANDS = [
    {"name": "tidal",        "f_lo": 10e-6,  "f_hi": 35e-6,   "lambda_km": 4677},
    {"name": "post_tidal",   "f_lo": 35e-6,  "f_hi": 100e-6,  "lambda_km": 1502},
    {"name": "intermediate", "f_lo": 100e-6, "f_hi": 500e-6,  "lambda_km": 1459},
]
FULL_BAND = {"f_lo": 10e-6, "f_hi": 500e-6, "lambda_km": 4677}

def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)

def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)

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

def _input_path(tag):
    for cadence in ("30S", "05M"):
        p = ROOT / "data" / "rinex" / "clk" / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None

def _band_gram(X_prior, dt_s, band):
    """Compute phase Gram for a frequency band."""
    try:
        r = spectral_phase_gram(X_prior, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                min_bins=2, min_low_cycles=0.85)
        return r["predictive_gram"]
    except (ValueError, np.linalg.LinAlgError):
        return None

def _band_weights(G_band, band_info, Dg, pool, b):
    """Compute kriging weights for one band using its own lambda_km."""
    if G_band is None:
        return None
    scales = _multi_scales(band_info["lambda_km"])
    rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
    rho_rt_b = _rho_multi(Dg[b, pool], scales)
    try:
        w = _reg_solve(
            rho_rr_b * G_band[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
            rho_rt_b * G_band[b, pool])
        return w
    except np.linalg.LinAlgError:
        return None

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("Step 155: Leave-one-band-out analysis", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Bands: {[b['name'] for b in BANDS]}", flush=True)
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

    # Per-day RMS for each configuration, keyed by day index
    # configs maps config_name -> list of (day_idx, rms)
    day_results = {}  # day_idx -> {config_name: rms}

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        if cur is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur:
            continue

        # Load 13-day prior (same as Step 134)
        from scripts.steps.step_152_freq_spectrum import _load_multi_day_prior
        multi_prior = _load_multi_day_prior(tag, names, 13)
        if multi_prior is None:
            continue
        prior_names, prior_x, n_days_loaded = multi_prior

        # Current day residuals
        R_cols = []
        kept_cur = []
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
            R_cols.append(r)
            kept_cur.append(nm)

        if len(kept_cur) < s64.MIN_REFS + 2:
            continue

        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS
        Dg = D[np.ix_([names.index(n) for n in kept_cur], [names.index(n) for n in kept_cur])]
        gnames = kept_cur

        # QC
        sd_r = np.nanstd(R, axis=0)
        med_r = np.median(sd_r)
        keep = [i for i in range(len(gnames)) if 0.01 < sd_r[i] < 12.0 and sd_r[i] < 3.5 * med_r]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]
        Dg = Dg[np.ix_(keep, keep)]

        prior_idx = [prior_names.index(nm) for nm in gnames]
        X_prior = prior_x[:, prior_idx]

        sd_p = np.nanstd(X_prior, axis=0)
        keep2 = [i for i in range(len(gnames)) if 0.01 < sd_p[i] < 1e4]
        if len(keep2) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep2]
        R = R[:, keep2]
        Dg = Dg[np.ix_(keep2, keep2)]
        X_prior = X_prior[:, keep2]

        ng = len(gnames)
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        # Compute all phase Grams
        G_full = _band_gram(X_prior, dt_s, FULL_BAND)
        G_tidal = _band_gram(X_prior, dt_s, BANDS[0])
        G_post = _band_gram(X_prior, dt_s, BANDS[1])
        G_inter = _band_gram(X_prior, dt_s, BANDS[2])

        # Track band availability
        band_avail = {
            "tidal": G_tidal is not None,
            "post_tidal": G_post is not None,
            "intermediate": G_inter is not None,
        }
        n_bands_avail = sum(band_avail.values())

        # Skip if full-band gram failed (needed for baseline)
        if G_full is None:
            continue

        # Skip if fewer than 2 bands available (can't do meaningful leave-one-out)
        if n_bands_avail < 2:
            continue

        # Per-day squared errors keyed by config name
        day_sq = {
            "full_band": [],
            "multi_band": [],
            "no_tidal": [],
            "no_post_tidal": [],
            "no_intermediate": [],
            "tidal_only": [],
            "post_tidal_only": [],
            "intermediate_only": [],
        }
        # Track band failures per day
        band_fail_count = {b: 0 for b in band_avail}

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band weights (single lambda=4677 km for full band)
            scales_full = _multi_scales(FULL_BAND["lambda_km"])
            rho_rr_f = _rho_multi(Dg[np.ix_(pool, pool)], scales_full)
            rho_rt_f = _rho_multi(Dg[b, pool], scales_full)
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Per-band weights
            w_tidal = _band_weights(G_tidal, BANDS[0], Dg, pool, b)
            w_post = _band_weights(G_post, BANDS[1], Dg, pool, b)
            w_inter = _band_weights(G_inter, BANDS[2], Dg, pool, b)

            band_weights_map = {
                "tidal": w_tidal,
                "post_tidal": w_post,
                "intermediate": w_inter,
            }

            B_arr = R[:, b]
            A = R[:, pool]

            for ei in eval_i:
                truth = B_arr[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue

                # Full-band prediction
                p_full = float(w_full @ rr)
                day_sq["full_band"].append((truth - p_full) ** 2)

                # Per-band predictions
                band_preds = {}
                for bname, w in band_weights_map.items():
                    if w is not None:
                        band_preds[bname] = float(w @ rr)
                    else:
                        band_fail_count[bname] += 1

                # Multi-band: average of all available band predictions
                if band_preds:
                    p_multi = float(np.mean(list(band_preds.values())))
                else:
                    p_multi = p_full  # fallback only if ALL bands fail
                day_sq["multi_band"].append((truth - p_multi) ** 2)

                # Leave-one-band-out: average excluding one band
                for exclude in ["tidal", "post_tidal", "intermediate"]:
                    remaining = {k: v for k, v in band_preds.items() if k != exclude}
                    if remaining:
                        p_loo = float(np.mean(list(remaining.values())))
                    else:
                        p_loo = p_full
                    day_sq[f"no_{exclude}"].append((truth - p_loo) ** 2)

                # Single-band-only
                for bname in ["tidal", "post_tidal", "intermediate"]:
                    if bname in band_preds:
                        day_sq[f"{bname}_only"].append((truth - band_preds[bname]) ** 2)
                    else:
                        day_sq[f"{bname}_only"].append(np.nan)

        # Compute per-day RMS for each config
        day_rms = {}
        for cfg_name in day_sq:
            vals = np.array(day_sq[cfg_name])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                day_rms[cfg_name] = float(np.sqrt(np.mean(vals)))
            else:
                day_rms[cfg_name] = np.nan

        day_results[k] = {"rms": day_rms, "band_avail": band_avail, "band_fails": band_fail_count}

        if (k + 1) % 10 == 0:
            multi_med = np.median([d["rms"]["multi_band"] for d in day_results.values()
                                   if np.isfinite(d["rms"].get("multi_band", np.nan))])
            full_med = np.median([d["rms"]["full_band"] for d in day_results.values()
                                  if np.isfinite(d["rms"].get("full_band", np.nan))])
            no_tidal_med = np.median([d["rms"]["no_tidal"] for d in day_results.values()
                                      if np.isfinite(d["rms"].get("no_tidal", np.nan))])
            print(f"  [{k+1}/{len(eval_tags)}] multi={multi_med:.3f}, "
                  f"full={full_med:.3f}, "
                  f"no_tidal={no_tidal_med:.3f} m", flush=True)

    # Compute improvements — aligned by day index
    valid_days = sorted(day_results.keys())
    configs = {
        "full_band": [],
        "multi_band": [],
        "no_tidal": [],
        "no_post_tidal": [],
        "no_intermediate": [],
        "tidal_only": [],
        "post_tidal_only": [],
        "intermediate_only": [],
    }

    for di in valid_days:
        rms_di = day_results[di]["rms"]
        for cfg in configs:
            configs[cfg].append(rms_di.get(cfg, np.nan))

    full_rms = np.array(configs["full_band"])
    multi_rms = np.array(configs["multi_band"])

    results = {}
    for cfg_name, rms_list in configs.items():
        rms = np.array(rms_list)
        # Align with full and multi by day index (same length since from same days)
        vs_full = 100.0 * (1.0 - rms / full_rms)
        vs_multi = 100.0 * (1.0 - rms / multi_rms)
        # Guard against zero denominators
        vs_full[~np.isfinite(vs_full)] = np.nan
        vs_multi[~np.isfinite(vs_multi)] = np.nan
        results[cfg_name] = {
            "median_rms_m": float(np.nanmedian(rms)) if np.isfinite(rms).any() else None,
            "vs_full_band_pct": _ci(vs_full, rng),
            "vs_multi_band_pct": _ci(vs_multi, rng),
            "n": int(np.isfinite(rms).sum()),
        }

    # Band availability summary
    band_avail_counts = {"tidal": 0, "post_tidal": 0, "intermediate": 0}
    for di in valid_days:
        for bname in band_avail_counts:
            if day_results[di]["band_avail"].get(bname, False):
                band_avail_counts[bname] += 1

    # Key comparisons
    interp = []
    interp.append(f"Leave-one-band-out: {len(valid_days)} eval days, 13-day window.")
    interp.append(f"Band availability: tidal {band_avail_counts['tidal']}/{len(valid_days)}, "
                  f"post_tidal {band_avail_counts['post_tidal']}/{len(valid_days)}, "
                  f"intermediate {band_avail_counts['intermediate']}/{len(valid_days)} days.")

    for cfg_name in ["multi_band", "no_tidal", "no_post_tidal", "no_intermediate",
                     "tidal_only", "post_tidal_only", "intermediate_only"]:
        r = results.get(cfg_name, {})
        vf = r.get("vs_full_band_pct", {})
        vm = r.get("vs_multi_band_pct", {})
        med = vf.get("median")
        ci = vf.get("ci95")
        med_m = vm.get("median")
        ci_m = vm.get("ci95")
        if med is not None:
            interp.append(
                f"  {cfg_name}: vs full-band {med:+.2f}% CI{ci}, "
                f"vs multi-band {med_m:+.2f}% CI{ci_m}, "
                f"RMS={r.get('median_rms_m', 0):.3f} m"
            )

    # The key test: does removing the tidal band hurt?
    no_tidal_vs_multi = results.get("no_tidal", {}).get("vs_multi_band_pct", {})
    if no_tidal_vs_multi.get("median") is not None:
        hurt = no_tidal_vs_multi["median"]
        ci = no_tidal_vs_multi["ci95"]
        if hurt < 0 and ci[1] < 0:
            interp.append(
                f"Removing the tidal band HURTS performance ({hurt:+.2f}% CI{ci}). "
                f"The gravitational tidal frequencies are essential for the multi-band advantage. "
                f"This is direct evidence for TEP gravitational coupling."
            )
        elif hurt < 0:
            interp.append(
                f"Removing the tidal band slightly hurts ({hurt:+.2f}% CI{ci}). "
                f"The tidal band contributes but is not the sole driver."
            )
        else:
            interp.append(
                f"Removing the tidal band does NOT hurt ({hurt:+.2f}% CI{ci}). "
                f"The tidal band is not essential; the advantage comes from the continuum."
            )

    verdict = {
        "step": 155,
        "title": "Leave-one-band-out: which frequency band drives the multi-band advantage?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "n_days_scored": len(valid_days),
        "band_availability": band_avail_counts,
        "results": results,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If TEP is real, the tidal band (10-35 uHz, lambda=4677 km) carries the "
            "gravitational tidal signal. Removing it should hurt performance more than "
            "removing the post-tidal or intermediate bands. The tidal-only kernel should "
            "also carry predictive signal, even if it doesn't beat the full-band kernel "
            "(because the continuum is also needed)."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"\n  Output: {OUT}", flush=True)
    if results.get("multi_band", {}).get("vs_full_band_pct", {}).get("median") is not None:
        print(f"  Multi-band vs full: {results['multi_band']['vs_full_band_pct']['median']:+.2f}%", flush=True)
    if results.get("no_tidal", {}).get("vs_multi_band_pct", {}).get("median") is not None:
        print(f"  No-tidal vs multi: {results['no_tidal']['vs_multi_band_pct']['median']:+.2f}%", flush=True)
    if results.get("no_post_tidal", {}).get("vs_multi_band_pct", {}).get("median") is not None:
        print(f"  No-post vs multi: {results['no_post_tidal']['vs_multi_band_pct']['median']:+.2f}%", flush=True)
    if results.get("no_intermediate", {}).get("vs_multi_band_pct", {}).get("median") is not None:
        print(f"  No-inter vs multi: {results['no_intermediate']['vs_multi_band_pct']['median']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
