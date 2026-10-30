#!/usr/bin/env python3
"""
STEP 157  --  Tidal-line notching: do specific gravitational tidal frequencies
              carry predictive signal that cannot be replaced by nearby frequencies?

Previous tests (Step 155/156) removed the entire tidal BAND (10-35 uHz),
which has too few spectral bins for stable Gram estimation.  That tests
whether the band is useful as a PREDICTOR, not whether the tidal LINES
carry signal.

This step takes the opposite approach: start with the full band (10-500 uHz)
and NOTCH OUT specific tidal frequencies.  If the tidal lines carry unique
predictive signal, notching them should HURT.  If they are noise, notching
should HELP.

Test frequencies:
  - M2 lunar (22.365 uHz, 12.42h) — purely gravitational
  - O1 lunar (10.758 uHz, 25.82h) — purely gravitational
  - S2 solar (23.148 uHz, 12.00h) — has atmospheric component
  - K1 mixed (11.608 uHz, 23.93h) — mixed gravitational + atmospheric
  - 24h solar (11.574 uHz) — solar diurnal
  - Control: fake tides at non-physical frequencies (25h, 13h, 11h, 7h)

If notching M2 and O1 (lunar) hurts MORE than notching fake tides,
the gravitational tidal frequencies carry unique predictive signal.

Output: results/outputs/step_157_tidal_notching.json
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

OUT = ROOT / "results" / "outputs" / "step_157_tidal_notching.json"
PRODUCT = "clk_cod"
YEAR = 2025
RNG_SEED = 20260920

EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10

# Notch width: ±0.5 uHz around each target frequency
NOTCH_HALF_WIDTH_HZ = 0.5e-6

# Tidal frequencies to notch
NOTCH_TARGETS = {
    "M2_lunar":      22.365e-6,
    "O1_lunar":      10.758e-6,
    "S2_solar":      23.148e-6,
    "K1_mixed":      11.608e-6,
    "solar_24h":     11.574e-6,
    "N2_lunar":      21.422e-6,
    # Fake tides at non-physical frequencies, chosen to be >2 uHz from
    # any real tidal frequency so the notch windows do not overlap.
    "fake_25h":      15.000e-6,  # 1/18.5h, far from any real tide
    "fake_13h":      27.000e-6,  # 1/10.3h, far from any real tide
    "fake_11h":      33.000e-6,  # 1/8.4h, far from any real tide
    "fake_7h":       45.000e-6,  # 1/6.2h, far from any real tide
}

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

def _notched_gram(X_prior, dt_s, full_band, notch_freqs, notch_half_width):
    """Compute phase Gram with specific frequencies notched out."""
    # Get the full-band frequency bins
    n_t = X_prior.shape[0]
    freqs = np.fft.rfftfreq(n_t, d=dt_s)

    # Compute the windowed FFT per station
    window = np.hanning(n_t)
    Xw = np.fft.rfft(X_prior * window[:, None], axis=0)

    # Notch out the target frequencies
    for f_notch in notch_freqs:
        mask = np.abs(freqs - f_notch) < notch_half_width
        Xw[mask, :] = 0

    # Reconstruct the notched signal
    x_notched = np.fft.irfft(Xw, n=n_t, axis=0)

    # Compute the phase Gram on the notched signal
    try:
        r = spectral_phase_gram(x_notched, dt_s,
                                f_lo_hz=full_band["f_lo"], f_hi_hz=full_band["f_hi"],
                                min_bins=10, min_low_cycles=0.85)
        return r["predictive_gram"]
    except (ValueError, np.linalg.LinAlgError):
        return None

def _full_band_gram(X_prior, dt_s, full_band):
    try:
        r = spectral_phase_gram(X_prior, dt_s,
                                f_lo_hz=full_band["f_lo"], f_hi_hz=full_band["f_hi"],
                                min_bins=10, min_low_cycles=0.85)
        return r["predictive_gram"]
    except (ValueError, np.linalg.LinAlgError):
        return None

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("Step 157: Tidal-line notching", flush=True)
    print(f"  Notch width: ±{NOTCH_HALF_WIDTH_HZ*1e6:.1f} uHz", flush=True)
    print(f"  Targets: {list(NOTCH_TARGETS.keys())}", flush=True)
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

    # Configurations: full-band (baseline) + each notched version
    configs = {"full_band": []}
    for name in NOTCH_TARGETS:
        configs[f"notch_{name}"] = []
    # Combined notches
    configs["notch_all_lunar"] = []     # M2 + O1 + N2
    configs["notch_all_solar"] = []     # S2 + K1 + 24h
    configs["notch_all_real"] = []      # all real tides
    configs["notch_all_fake"] = []      # all fake tides
    configs["notch_lunar_and_solar"] = []  # everything

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

        # Compute all Grams
        G_full = _full_band_gram(X_prior, dt_s, FULL_BAND)
        if G_full is None:
            continue

        gram_map = {"full_band": G_full}
        for name, freq in NOTCH_TARGETS.items():
            gram_map[f"notch_{name}"] = _notched_gram(X_prior, dt_s, FULL_BAND, [freq], NOTCH_HALF_WIDTH_HZ)

        # Combined notches
        lunar_freqs = [NOTCH_TARGETS["M2_lunar"], NOTCH_TARGETS["O1_lunar"], NOTCH_TARGETS["N2_lunar"]]
        solar_freqs = [NOTCH_TARGETS["S2_solar"], NOTCH_TARGETS["K1_mixed"], NOTCH_TARGETS["solar_24h"]]
        fake_freqs = [NOTCH_TARGETS["fake_25h"], NOTCH_TARGETS["fake_13h"], NOTCH_TARGETS["fake_11h"], NOTCH_TARGETS["fake_7h"]]
        all_real = lunar_freqs + solar_freqs

        gram_map["notch_all_lunar"] = _notched_gram(X_prior, dt_s, FULL_BAND, lunar_freqs, NOTCH_HALF_WIDTH_HZ)
        gram_map["notch_all_solar"] = _notched_gram(X_prior, dt_s, FULL_BAND, solar_freqs, NOTCH_HALF_WIDTH_HZ)
        gram_map["notch_all_real"] = _notched_gram(X_prior, dt_s, FULL_BAND, all_real, NOTCH_HALF_WIDTH_HZ)
        gram_map["notch_all_fake"] = _notched_gram(X_prior, dt_s, FULL_BAND, fake_freqs, NOTCH_HALF_WIDTH_HZ)
        gram_map["notch_lunar_and_solar"] = _notched_gram(X_prior, dt_s, FULL_BAND, all_real + fake_freqs, NOTCH_HALF_WIDTH_HZ)

        day_sq = {k: [] for k in configs}
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue
            rho_rr = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt = s64._rho_multi(Dg[b, pool])
            B_arr = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B_arr[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                for cfg, G in gram_map.items():
                    if G is None:
                        day_sq[cfg].append(np.nan)
                        continue
                    try:
                        w = _reg_solve(rho_rr * G[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                       rho_rt * G[b, pool])
                        pred = float(w @ rr)
                        day_sq[cfg].append((truth - pred) ** 2)
                    except np.linalg.LinAlgError:
                        day_sq[cfg].append(np.nan)

        for cfg in configs:
            vals = np.array(day_sq[cfg])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                configs[cfg].append(float(np.sqrt(np.mean(vals))))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] full={np.median(configs['full_band']):.4f}, "
                  f"notch_M2={np.median(configs['notch_M2_lunar']):.4f} m", flush=True)

    # Compute results
    results = {}
    full_rms = np.array(configs["full_band"])

    for cfg_name, rms_list in configs.items():
        rms = np.array(rms_list)
        if len(rms) < 4:
            results[cfg_name] = {"n": len(rms)}
            continue
        n = min(len(rms), len(full_rms))
        # vs full_band: positive = notching HURTS (rms went up), negative = notching HELPS
        vs_full = 100.0 * (rms[:n] / full_rms[:n] - 1.0)
        results[cfg_name] = {
            "median_rms_m": float(np.median(rms)),
            "notch_effect_pct": _ci(vs_full, rng),
            "n": len(rms),
        }

    interp = []
    interp.append(f"Tidal-line notching: {len(configs['full_band'])} eval days, 13-day window.")
    interp.append(f"  Positive % = notching HURTS (the frequency carries unique signal)")
    interp.append(f"  Negative % = notching HELPS (the frequency is noise)")
    interp.append("")

    for cfg in ["full_band", "notch_M2_lunar", "notch_O1_lunar", "notch_S2_solar",
                "notch_K1_mixed", "notch_solar_24h", "notch_N2_lunar",
                "notch_fake_25h", "notch_fake_13h", "notch_fake_11h", "notch_fake_7h",
                "notch_all_lunar", "notch_all_solar", "notch_all_real", "notch_all_fake",
                "notch_lunar_and_solar"]:
        r = results.get(cfg, {})
        ne = r.get("notch_effect_pct", {})
        med = ne.get("median")
        ci = ne.get("ci95")
        if med is not None:
            tag = "HURTS" if med > 0 else "HELPS"
            sig = "***" if ci and ci[0] > 0 else "**" if ci and ci[1] > 0 else ""
            interp.append(f"  {cfg:30s}: {med:+.3f}% CI{ci} {tag} {sig}")

    # Key comparison: lunar notching vs fake notching
    lunar_hurt = results.get("notch_all_lunar", {}).get("notch_effect_pct", {})
    fake_hurt = results.get("notch_all_fake", {}).get("notch_effect_pct", {})
    if lunar_hurt.get("median") is not None and fake_hurt.get("median") is not None:
        diff = lunar_hurt["median"] - fake_hurt["median"]
        interp.append(f"\n  Lunar notching hurt: {lunar_hurt['median']:+.3f}%")
        interp.append(f"  Fake notching hurt:  {fake_hurt['median']:+.3f}%")
        interp.append(f"  Difference (lunar - fake): {diff:+.3f}%")
        if diff > 0:
            interp.append("  Notching LUNAR tides hurts MORE than notching FAKE tides.")
            interp.append("  The gravitational lunar frequencies carry unique predictive signal.")
        else:
            interp.append("  Notching fake tides hurts as much as or more than notching lunar tides.")
            interp.append("  The lunar frequencies do NOT carry unique signal beyond nearby noise.")

    # M2 specifically
    m2_hurt = results.get("notch_M2_lunar", {}).get("notch_effect_pct", {})
    if m2_hurt.get("median") is not None:
        interp.append(f"\n  M2 lunar notching: {m2_hurt['median']:+.3f}% CI{m2_hurt.get('ci95')}")
        if m2_hurt["median"] > 0 and m2_hurt.get("ci95", [0])[1] > 0:
            interp.append("  Notching M2 HURTS — the M2 lunar tide carries unique predictive signal.")

    verdict = {
        "step": 157,
        "title": "Tidal-line notching: do specific gravitational frequencies carry unique signal?",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR,
        "notch_half_width_uhz": NOTCH_HALF_WIDTH_HZ * 1e6,
        "results": results,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If TEP is real, notching the M2 and O1 lunar tidal frequencies should HURT "
            "prediction more than notching fake non-physical frequencies. "
            "The gravitational tidal lines carry unique predictive signal that "
            "cannot be replaced by nearby frequencies."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for line in interp:
        print(line, flush=True)


if __name__ == "__main__":
    main()
