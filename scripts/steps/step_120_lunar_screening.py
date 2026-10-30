#!/usr/bin/env python3
"""
STEP 120  --  Lunar phase dependence of the oscillator screening effect.

THE MOST DIRECT TEP TEST: does the quartz-vs-atomic clock screening effect
vary with the lunar tidal gravitational potential?

Under TEP, the clock correlation field is a proper-time field, and the
oscillator screening effect (quartz clocks show stronger distance-structured
coherence than atomic clocks) reflects the coupling of the oscillator to
this field.  If the field is gravitational in origin, it should be modulated
by the lunar tidal acceleration, which varies by ~40% over the anomalistic
month (27.55 d) as the Moon moves between perigee and apogee.

The specific TEP prediction is:
  - The quartz-atomic screening ratio should correlate with lunar tidal
    acceleration
  - The correlation should be positive (stronger screening at perigee,
    when the lunar tidal field is stronger)
  - The modulation should be at the anomalistic month period

Under an atmospheric/ionospheric interpretation:
  - The screening ratio should NOT correlate with lunar tidal acceleration
  - Any modulation should be at solar periods (24h, 365.25d), not lunar

This is a variant of Step 108 (lunar tidal modulation of lambda) but applied
to the SCREENING RATIO rather than the overall lambda.  The screening ratio
is the most TEP-specific observable: it isolates the oscillator-coupling
component from the signal-propagation component.

Protocol:
  * Use 60 days of CODE MGEX clocks across 2024.
  * For each 15-day window, compute:
      - quartz-pair phase-alignment distance decay (lambda, R^2)
      - atomic-pair phase-alignment distance decay (lambda, R^2)
      - screening ratio = quartz_tep_band_alignment / atomic_tep_band_alignment
  * Correlate the screening ratio with the lunar tidal acceleration.
  * Block permutation test for significance.

Output: results/outputs/step_120_lunar_screening.json
"""
from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib import tep_astronomical as ta

import importlib.util
_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_120_lunar_screening.json"
PRODUCT = "clk_cod"
YEAR = 2024
WINDOW_DAYS = 15
STEP_DAYS = 10
RNG = np.random.default_rng(20260906)


def _date_tag(year, doy):
    return f"{year}{doy:03d}"


def _doy_to_jd(year, doy):
    Y, M, D = year, 1, 0
    jd0 = (367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4)
           + np.floor(275 * M / 9) + D + 1721013.5)
    return jd0 + doy + 0.5


def _load_clock_types():
    """Load the IGS clock-type metadata used by Steps 65/103.

    Returns a dict mapping station name -> 'QUARTZ' or 'ATOMIC'.
    """
    ct_path = ROOT / "data" / "meta" / "igs_clock_type_full.json"
    if not ct_path.exists():
        return {}
    ct = json.loads(ct_path.read_text())
    return {n: v[0] for n, v in ct.items()}


_CLOCK_TYPES = None


def _oscillator_type(sta_name):
    """Return 'QUARTZ', 'ATOMIC', or None for a station name."""
    global _CLOCK_TYPES
    if _CLOCK_TYPES is None:
        _CLOCK_TYPES = _load_clock_types()
    return _CLOCK_TYPES.get(sta_name)


def _is_atomic(sta_name):
    return _oscillator_type(sta_name) == "ATOMIC"


def _is_quartz(sta_name):
    return _oscillator_type(sta_name) == "QUARTZ"


def analyze_window_screening(date_tags, label):
    """Compute screening ratio for a window.

    Splits station pairs into quartz-quartz, atomic-atomic, and mixed.
    Computes the TEP-band mean phase alignment for each group.
    The screening ratio is quartz_alignment / atomic_alignment.
    """
    days = []
    for tag in date_tags:
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                                 max_jumps=10_000)
        except Exception:
            continue
        if len(ser) >= 8:
            days.append(ser)

    if not days:
        return None

    qq_pairs, aa_pairs, mixed_pairs = [], [], []

    for ser in days:
        RF, STA = s53._spectra(ser)
        nm = list(RF)
        if len(RF) < 8:
            continue

        # Classify stations by oscillator type
        quartz_names = set()
        atomic_names = set()
        for n in nm:
            sta_name = STA[n].name
            if _is_quartz(sta_name):
                quartz_names.add(n)
            elif _is_atomic(sta_name):
                atomic_names.add(n)

        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
            if not (50.0 <= d <= 13000.0):
                continue
            Ssum = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(Ssum) or Ssum == 0:
                continue
            cpsi = float(np.cos(np.angle(Ssum)))

            qi, qj = nm[i] in quartz_names, nm[j] in quartz_names
            ai, aj = nm[i] in atomic_names, nm[j] in atomic_names

            if qi and qj:
                qq_pairs.append((d, cpsi))
            elif ai and aj:
                aa_pairs.append((d, cpsi))
            elif (qi and aj) or (ai and qj):
                mixed_pairs.append((d, cpsi))

    # Compute TEP-band mean alignment for each group
    def _tep_mean(pairs):
        tep = [v for d, v in pairs if 1000.0 <= d <= 5000.0]
        return float(np.mean(tep)) if len(tep) >= 20 else None

    qq_align = _tep_mean(qq_pairs)
    aa_align = _tep_mean(aa_pairs)
    mixed_align = _tep_mean(mixed_pairs)

    # Screening ratio (unstable when aa_align is near zero)
    screening_ratio = None
    if qq_align is not None and aa_align is not None and abs(aa_align) > 1e-6:
        screening_ratio = qq_align / aa_align

    # Screening difference (robust — does not blow up when aa_align is near zero)
    screening_diff = None
    if qq_align is not None and aa_align is not None:
        screening_diff = qq_align - aa_align

    # Also fit lambdas
    def _fit_lambda(pairs):
        if len(pairs) < 200:
            return None, None
        ctr, val, cnt = s53._curve(pairs)
        fit = s53._fit(ctr, val, cnt)
        exp_fit = fit.get("exponential") or {}
        return exp_fit.get("lambda_km"), fit.get("R2")

    qq_lambda, qq_r2 = _fit_lambda(qq_pairs)
    aa_lambda, aa_r2 = _fit_lambda(aa_pairs)

    return {
        "label": label,
        "n_qq_pairs": len(qq_pairs),
        "n_aa_pairs": len(aa_pairs),
        "n_mixed_pairs": len(mixed_pairs),
        "qq_tep_alignment": qq_align,
        "aa_tep_alignment": aa_align,
        "mixed_tep_alignment": mixed_align,
        "screening_ratio": screening_ratio,
        "screening_diff": screening_diff,
        "qq_lambda_km": qq_lambda,
        "qq_R2": qq_r2,
        "aa_lambda_km": aa_lambda,
        "aa_R2": aa_r2,
    }


def block_permutation_p(x, y, n_perm=500):
    """Block permutation p-value."""
    r_obs = float(np.corrcoef(x, y)[0, 1])
    rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
    if abs(rho1) < 0.99:
        n_eff = len(x) * (1 - rho1) / (1 + rho1)
    else:
        n_eff = 2.0
    block_size = max(1, int(np.ceil(len(x) / max(n_eff, 2))))

    n_exc = 0
    for _ in range(n_perm):
        n_blocks = len(x) // block_size
        if n_blocks < 2:
            yp = RNG.permutation(y)
        else:
            idx = np.arange(n_blocks) * block_size
            perm = RNG.permutation(n_blocks)
            yp = np.empty_like(y)
            for j, p in enumerate(perm):
                yp[j * block_size:(j + 1) * block_size] = \
                    y[p * block_size:(p + 1) * block_size]
            rem = len(x) % block_size
            if rem > 0:
                yp[-rem:] = y[-rem:]
        r_perm = float(np.corrcoef(x, yp)[0, 1])
        if abs(r_perm) >= abs(r_obs):
            n_exc += 1
    return float((n_exc + 1) / (n_perm + 1)), r_obs, rho1, n_eff


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Step 120: Lunar phase dependence of the oscillator screening effect",
          flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Window: {WINDOW_DAYS} days, step: {STEP_DAYS} days", flush=True)
    print()

    # Check if oscillator classification works
    dates = td.available_dates(PRODUCT, 2024)
    if dates:
        try:
            ser = td.load_series(PRODUCT, dates[0], min_valid_frac=0.0,
                                 max_jumps=10_000)
            n_q = sum(1 for v in ser.values() if _is_quartz(v["station"].name))
            n_a = sum(1 for v in ser.values() if _is_atomic(v["station"].name))
            n_u = len(ser) - n_q - n_a
            print(f"  Oscillator classification (first day): "
                  f"quartz={n_q}, atomic={n_a}, unknown={n_u}", flush=True)
            if n_q < 5 or n_a < 5:
                print("  WARNING: too few classified stations.", flush=True)
        except Exception as e:
            print(f"  Error checking classification: {e}", flush=True)

    # Build windows
    all_doys = list(range(1, 367))
    windows = []
    start = 1
    while start + WINDOW_DAYS <= 367:
        win_doys = all_doys[start:start + WINDOW_DAYS]
        tags = [_date_tag(YEAR, d) for d in win_doys]
        jd_center = _doy_to_jd(YEAR, win_doys[WINDOW_DAYS // 2])
        epoch = (jd_center - 2440587.5) * 86400.0
        try:
            lunar_tidal = ta.lunar_tidal_accel(epoch)
            em_dist = ta.earth_moon_distance_km(epoch)
        except Exception:
            start += STEP_DAYS
            continue
        windows.append({
            "tags": tags, "jd_center": jd_center,
            "doy_center": win_doys[WINDOW_DAYS // 2],
            "lunar_tidal_accel": lunar_tidal,
            "earth_moon_dist_km": em_dist,
        })
        start += STEP_DAYS

    print(f"  {len(windows)} windows", flush=True)
    print()

    # Analyze each window
    results = []
    for i, w in enumerate(windows):
        res = analyze_window_screening(w["tags"], f"window_{i}")
        if res is not None and res.get("screening_ratio") is not None:
            res["jd_center"] = w["jd_center"]
            res["doy_center"] = w["doy_center"]
            res["lunar_tidal_accel"] = w["lunar_tidal_accel"]
            res["earth_moon_dist_km"] = w["earth_moon_dist_km"]
            results.append(res)
            print(f"  [{i+1}/{len(windows)}] DOY {res['doy_center']:3d}: "
                  f"QQ={res['n_qq_pairs']}, AA={res['n_aa_pairs']}, "
                  f"ratio={res['screening_ratio']:.3f}, "
                  f"lunar={res['lunar_tidal_accel']:.4e}", flush=True)
        elif res is not None:
            print(f"  [{i+1}/{len(windows)}] DOY {w['doy_center']:3d}: "
                  f"QQ={res['n_qq_pairs']}, AA={res['n_aa_pairs']}, "
                  f"ratio=None (insufficient data)", flush=True)

    print(f"\n  {len(results)} windows with valid screening ratios", flush=True)

    if len(results) < 10:
        OUT.write_text(json.dumps({
            "step": 120, "error": "too few valid windows",
            "n_windows": len(results),
            "note": "Oscillator classification may not be available for this product. "
                    "The tep_data station metadata may not include oscillator type."},
            indent=2))
        print("  ERROR: too few valid windows", flush=True)
        print("  The oscillator classification may not be available.", flush=True)
        return

    # Extract time series
    ratios = np.array([r["screening_ratio"] for r in results], float)
    diffs = np.array([r["screening_diff"] for r in results
                      if r["screening_diff"] is not None], float)
    lunar_tidals = np.array([r["lunar_tidal_accel"] for r in results], float)
    em_dists = np.array([r["earth_moon_dist_km"] for r in results], float)
    doys = np.array([r["doy_center"] for r in results], float)
    qq_aligns = np.array([r["qq_tep_alignment"] for r in results], float)
    aa_aligns = np.array([r["aa_tep_alignment"] for r in results], float)

    # Align diffs with other arrays (diffs may be shorter if some were None)
    diffs_full = np.array([r["screening_diff"] if r["screening_diff"] is not None
                           else np.nan for r in results], float)
    ok_diff = np.isfinite(diffs_full)

    # Correlations
    correlations = {}

    # 1. Screening ratio vs lunar tidal (KEY TEST)
    p, r, rho1, n_eff = block_permutation_p(ratios, lunar_tidals)
    correlations["screening_ratio_vs_lunar_tidal"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"\n  Screening ratio vs lunar tidal: r={r:+.4f}, "
          f"p_block={p:.4f}, n_eff={n_eff:.1f}", flush=True)

    # 1b. Screening DIFFERENCE vs lunar tidal (ROBUST — primary observable)
    if ok_diff.sum() >= 10:
        p, r, rho1, n_eff = block_permutation_p(
            diffs_full[ok_diff], lunar_tidals[ok_diff])
        correlations["screening_diff_vs_lunar_tidal"] = {
            "r": r, "p_block_permutation": p,
            "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
            "n_windows": int(ok_diff.sum()),
            "note": "Robust alternative to the ratio — does not blow up "
                    "when atomic alignment is near zero.",
        }
        print(f"  Screening diff vs lunar tidal: r={r:+.4f}, "
              f"p_block={p:.4f}, n_eff={n_eff:.1f}", flush=True)

    # 2. Screening ratio vs EM distance (CONTROL)
    p, r, rho1, n_eff = block_permutation_p(ratios, em_dists)
    correlations["screening_ratio_vs_em_dist"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  Screening ratio vs EM distance: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 3. QQ alignment vs lunar tidal
    p, r, rho1, n_eff = block_permutation_p(qq_aligns, lunar_tidals)
    correlations["qq_align_vs_lunar_tidal"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  QQ alignment vs lunar tidal: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 4. AA alignment vs lunar tidal
    p, r, rho1, n_eff = block_permutation_p(aa_aligns, lunar_tidals)
    correlations["aa_align_vs_lunar_tidal"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  AA alignment vs lunar tidal: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 5. Screening ratio vs DOY (seasonal control)
    p, r, rho1, n_eff = block_permutation_p(ratios, doys)
    correlations["screening_ratio_vs_doy"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  Screening ratio vs DOY: r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # 6. Partial: screening ratio vs lunar tidal, controlling for DOY
    from numpy.linalg import lstsq
    A = np.column_stack([doys, np.ones_like(doys)])
    resid_ratios = ratios - A @ lstsq(A, ratios, rcond=None)[0]
    resid_lunar = lunar_tidals - A @ lstsq(A, lunar_tidals, rcond=None)[0]
    p, r, rho1, n_eff = block_permutation_p(resid_ratios, resid_lunar)
    correlations["screening_ratio_vs_lunar_partial_doy"] = {
        "r": r, "p_block_permutation": p,
        "lag1_autocorrelation": rho1, "n_eff_bartlett": n_eff,
    }
    print(f"  Screening ratio vs lunar (partial DOY): r={r:+.4f}, "
          f"p_block={p:.4f}", flush=True)

    # Interpretation — prefer the robust screening difference over the ratio
    if "screening_diff_vs_lunar_tidal" in correlations:
        key_r = correlations["screening_diff_vs_lunar_tidal"]["r"]
        key_p = correlations["screening_diff_vs_lunar_tidal"]["p_block_permutation"]
        key_label = "screening difference (QQ - AA)"
    else:
        key_r = correlations["screening_ratio_vs_lunar_tidal"]["r"]
        key_p = correlations["screening_ratio_vs_lunar_tidal"]["p_block_permutation"]
        key_label = "screening ratio"
    partial_r = correlations["screening_ratio_vs_lunar_partial_doy"]["r"]
    partial_p = correlations["screening_ratio_vs_lunar_partial_doy"]["p_block_permutation"]

    if key_p < 0.05 and partial_p < 0.10:
        interpretation = (
            f"The oscillator {key_label} shows a significant lunar tidal "
            f"modulation (r={key_r:+.3f}, p={key_p:.3f}), remaining after "
            f"removing the seasonal trend (partial r={partial_r:+.3f}, "
            f"p={partial_p:.3f}).  This is direct evidence that the clock "
            f"correlation field couples to the lunar gravitational potential, "
            f"as predicted by TEP.  No atmospheric or ionospheric process "
            f"produces a lunar-month modulation of the quartz-vs-atomic "
            f"screening effect.")
    elif key_p < 0.05:
        interpretation = (
            f"The {key_label} shows a lunar tidal modulation "
            f"(r={key_r:+.3f}, p={key_p:.3f}), but the partial correlation "
            f"after removing seasonal trend is not significant (partial "
            f"r={partial_r:+.3f}, p={partial_p:.3f}).  The lunar signal may "
            f"share a seasonal trend with DOY.")
    else:
        interpretation = (
            f"No significant lunar tidal modulation of the screening ratio "
            f"(r={key_r:+.3f}, p={key_p:.3f}).  Under TEP this is a null "
            f"result; under an atmospheric interpretation it is expected.  "
            f"Note: with one year of data and high autocorrelation, the "
            f"effective sample size is small.")

    verdict = {
        "step": 120,
        "title": "Lunar phase dependence of the oscillator screening effect",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "window_days": WINDOW_DAYS,
        "step_days": STEP_DAYS,
        "n_windows": len(results),
        "windows": [{"doy_center": r["doy_center"],
                      "n_qq_pairs": r["n_qq_pairs"],
                      "n_aa_pairs": r["n_aa_pairs"],
                      "qq_tep_alignment": r["qq_tep_alignment"],
                      "aa_tep_alignment": r["aa_tep_alignment"],
                      "screening_ratio": r["screening_ratio"],
                      "screening_diff": r["screening_diff"],
                      "lunar_tidal_accel": r["lunar_tidal_accel"],
                      "earth_moon_dist_km": r["earth_moon_dist_km"]}
                     for r in results],
        "correlations": correlations,
        "interpretation": interpretation,
        "pre_registered_prediction": (
            "Under TEP, the quartz-atomic screening ratio should correlate "
            "with the lunar tidal acceleration (stronger screening at perigee). "
            "Under an atmospheric interpretation, the screening ratio should "
            "not correlate with lunar tidal acceleration.  This is the most "
            "direct test of whether the clock field is gravitational (TEP) "
            "or atmospheric (conventional)."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Interpretation: {interpretation}", flush=True)
    print(f"  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
