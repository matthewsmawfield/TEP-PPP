#!/usr/bin/env python3
"""
STEP 144  --  Fake-tides control for the multi-band TEP predictor.

TEST whether the real gravitational tidal frequencies specifically carry
the predictive signal, or whether any low-frequency structure would work
equally well.

The multi-band predictor (Step 134) uses Paper 1's frequency-dependent
lambdas with three bands: tidal (10-35 uHz), post-tidal (35-100 uHz),
and intermediate (100-500 uHz).  The tidal band contains the real
gravitational tidal lines:

  * 24h solar:   11.574 uHz
  * 12.4h M2:    22.355 uHz
  * 12h solar:   23.148 uHz
  * 8h solar:    34.722 uHz

This step replaces the real tidal lines with FAKE lines at non-physical
periods that have no gravitational counterpart:

  * 25h:  11.111 uHz
  * 13h:  21.368 uHz
  * 11h:  25.253 uHz
  * 7h:   39.683 uHz

METHOD:
  1. Compute the normal spectral phase Gram for the tidal band.
  2. Notch out the real tidal frequency bins (±1 bin around each).
  3. Inject synthetic line components at the fake frequencies with the
     same average amplitude as the notched real lines.
  4. Use this fake-tides Gram in the multi-band predictor with the SAME
     lambda=4677 km.

If the real gravitational tides carry the signal, the fake-tides variant
should be WORSE than the real-tides variant.

PROTOCOL:
  * Run on 2025 held-out data (same as Step 134).
  * Compare:
    1. real_tides_multi:  multi-band with real tidal Gram (Step 134 baseline)
    2. fake_tides_multi:  multi-band with fake-tides Gram in the tidal band
    3. full_band:         single-lambda full-band (Step 64 baseline)

Output: results/outputs/step_144_fake_tides.json
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

OUT = ROOT / "results" / "outputs" / "step_144_fake_tides.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2025

EVALUATION_DOYS = (
    3, 9, 15, 21, 28, 34, 40, 46, 52, 58, 64, 70, 77, 83, 89,
    95, 101, 107, 113, 120, 126, 132, 138, 144, 150, 156, 163,
    169, 175, 181, 187, 193, 199, 205, 212, 218, 224, 230, 236,
    242, 248, 255, 261, 267, 273, 279, 285, 291, 298, 304, 310,
    316, 322, 328, 334, 340, 347, 353, 359, 365,
)
MIN_VALID_DAYS = 50
REGISTERED_RNG_SEED = 20260914

# Paper 1 frequency-dependent lambdas
BANDS = [
    {"name": "tidal",       "f_lo": 9.84e-6, "f_hi": 35e-6,  "lambda_km": 4677.0, "min_bins": 2},
    {"name": "post_tidal",  "f_lo": 35e-6,   "f_hi": 100e-6, "lambda_km": 1502.0, "min_bins": 3},
    {"name": "intermediate","f_lo": 100e-6,  "f_hi": 500e-6, "lambda_km": 1459.0, "min_bins": 6},
]

# Real gravitational tidal frequencies (Hz)
REAL_TIDAL_FREQS_HZ = np.array([
    11.574e-6,   # 24h solar
    22.355e-6,   # 12.4h M2 lunar
    23.148e-6,   # 12h solar
    34.722e-6,   # 8h solar
])

# Fake non-physical frequencies (Hz) — no gravitational counterpart
FAKE_TIDAL_FREQS_HZ = np.array([
    11.111e-6,   # 25h
    21.368e-6,   # 13h
    25.253e-6,   # 11h
    39.683e-6,   # 7h
])


def _multi_scales(lam):
    return (lam / 1.6, lam, lam * 1.6)


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _rho_multi(d, scales):
    return np.mean([np.exp(-d / L) for L in scales], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _band_phase_grams_with_fake_tides(x, dt_s, fake=False):
    """Compute per-band phase Grams.  If fake=True, the tidal band Gram
    is constructed with real tidal lines notched out and fake lines injected.

    For the tidal band:
      1. Compute the full cross-spectral matrix F (all bins in 10-35 uHz).
      2. Identify bins near real tidal frequencies (±1 bin).
      3. Record the average amplitude of those bins.
      4. Zero out (notch) the real tidal bins.
      5. Inject synthetic components at fake frequencies with the same
         average amplitude, using random phases (frozen by RNG).
      6. Reconstruct the Gram from the modified cross-spectral matrix.

    The post_tidal and intermediate bands are computed normally.
    """
    grams = []
    for band in BANDS:
        if band["name"] == "tidal" and fake:
            gram = _fake_tides_gram(x, dt_s, band)
            if gram is not None:
                grams.append((gram, band))
        else:
            try:
                result = spectral_phase_gram(
                    x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                    min_bins=band.get("min_bins", 3), min_low_cycles=0.85)
                grams.append((result["predictive_gram"], band))
            except ValueError:
                continue
    if not grams:
        return None
    return grams


def _fake_tides_gram(x, dt_s, band):
    """Build a tidal-band Gram with real tides notched and fake tides injected."""
    x = np.asarray(x, float)
    n, n_sta = x.shape
    if n < 8 or n_sta < 2:
        return None

    # Replicate the spectral_phase_gram preprocessing
    q = np.arange(n, dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        return None
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        return None
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(n)[:, None], axis=0)
    freq = np.fft.rfftfreq(n, d=float(dt_s))

    # Band mask
    band_mask = (freq >= band["f_lo"]) & (freq <= band["f_hi"])
    if int(band_mask.sum()) < int(band.get("min_bins", 3)):
        return None

    # Find bins near real tidal frequencies (±1 bin)
    df = freq[1] - freq[0]
    notch_mask = np.zeros(len(freq), dtype=bool)
    for f_real in REAL_TIDAL_FREQS_HZ:
        for i in range(len(freq)):
            if band_mask[i] and abs(freq[i] - f_real) <= 1.5 * df:
                notch_mask[i] = True

    # Record the average amplitude of notched bins
    if notch_mask.sum() > 0:
        avg_amp = float(np.mean(np.abs(F[notch_mask, :])))
    else:
        avg_amp = float(np.mean(np.abs(F[band_mask, :])))

    # Notch out real tidal bins
    F_mod = F.copy()
    F_mod[notch_mask, :] = 0.0

    # Inject fake tidal components at fake frequencies
    rng = np.random.default_rng(REGISTERED_RNG_SEED)
    for f_fake in FAKE_TIDAL_FREQS_HZ:
        # Find nearest bin in the band
        idx_fake = np.argmin(np.abs(freq - f_fake))
        if not band_mask[idx_fake]:
            # Find nearest in-band bin
            in_band = np.where(band_mask)[0]
            if len(in_band) == 0:
                continue
            idx_fake = in_band[np.argmin(np.abs(freq[in_band] - f_fake))]
        # Inject synthetic line with random per-station phase, same amplitude
        phases = rng.uniform(0, 2 * np.pi, size=n_sta)
        F_mod[idx_fake, :] = avg_amp * np.exp(1j * phases)

    # Reconstruct Gram from modified cross-spectral matrix
    F_band = F_mod[band_mask]
    cross = np.einsum("fi,fj->ij", F_band.conj(), F_band, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        # Add small floor to avoid division by zero
        energy = np.maximum(energy, np.finfo(float).tiny)
    norm = np.sqrt(np.outer(energy, energy))
    complex_coherence = cross / norm
    gram = np.real(complex_coherence)
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    return gram


def _prior_gram_bands(prev_day, cur_names, fake_tides=False):
    """Build per-band phase Grams from the prior day."""
    pnames, pt, praw, _ = prev_day
    spectral_cols, clean_raw_cols, kept = [], [], []
    for nm in cur_names:
        if nm not in pnames:
            continue
        y = praw[:, pnames.index(nm)].astype(float)
        good = np.isfinite(y)
        if good.mean() < 0.9 or good.sum() < 20:
            continue
        if not good.all():
            ix = np.arange(len(y))
            y = np.interp(ix, ix[good], y[good])
        clean_raw = y.copy()
        xq = np.arange(len(clean_raw)) - len(clean_raw) / 2.0
        spectral_y = clean_raw - np.polyval(np.polyfit(xq, clean_raw, 2), xq)
        if not np.isfinite(spectral_y).all() or not (1e-4 < np.std(spectral_y) < 1e4):
            continue
        spectral_cols.append(spectral_y)
        clean_raw_cols.append(clean_raw)
        kept.append(nm)
    if len(kept) < s64.MIN_REFS + 1:
        return None

    band_grams = _band_phase_grams_with_fake_tides(
        np.column_stack(spectral_cols), s64.DT_S, fake=fake_tides)
    if band_grams is None:
        return None

    return kept, band_grams, np.column_stack(clean_raw_cols)


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)

    print("Step 144: Fake-tides control for multi-band TEP prior", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Real tidal freqs (uHz): {REAL_TIDAL_FREQS_HZ*1e6}", flush=True)
    print(f"  Fake tidal freqs (uHz): {FAKE_TIDAL_FREQS_HZ*1e6}", flush=True)
    for band in BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.0f}-{band['f_hi']*1e6:.0f} uHz, "
              f"lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 144, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "real_tides_rms", "fake_tides_rms",
        "real_vs_full", "fake_vs_full", "real_vs_fake")}

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                               max_jumps=10_000)
        if not s_cur:
            continue

        # Full-band Gram (registered)
        pg_full = s64.prior_gram(prev, names)
        # Real-tides multi-band Grams
        pg_real = _prior_gram_bands(prev, names, fake_tides=False)
        # Fake-tides multi-band Grams
        pg_fake = _prior_gram_bands(prev, names, fake_tides=True)

        if pg_full is None or pg_real is None or pg_fake is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_r, band_grams_real, _ = pg_real
        gnames_k, band_grams_fake, _ = pg_fake

        common = sorted(set(gnames_f) & set(gnames_r) & set(gnames_k))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_r = [gnames_r.index(n) for n in common]
        idx_k = [gnames_k.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        band_grams_real = [(g[np.ix_(idx_r, idx_r)], band)
                           for g, band in band_grams_real]
        band_grams_fake = [(g[np.ix_(idx_k, idx_k)], band)
                           for g, band in band_grams_fake]
        P = P_full[:, idx_f]
        Dg = D[np.ix_(idx_cur, idx_cur)]
        gnames = common

        # Causal detrend
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            ts = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for ki in range(len(clk_ns)):
                tk = ts[ki]
                mask = (ts <= tk) & (ts >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(ts[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS

        P = P.astype(float, copy=True)
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]),
                                    np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        band_grams_real = [(g[np.ix_(keep, keep)], band)
                           for g, band in band_grams_real]
        band_grams_fake = [(g[np.ix_(keep, keep)], band)
                           for g, band in band_grams_fake]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "real_tides", "fake_tides")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band TEP kernel (single lambda)
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Real-tides multi-band weights
            real_weights = []
            for G_band, band_info in band_grams_real:
                scales = _multi_scales(band_info["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    real_weights.append(w_band)
                except np.linalg.LinAlgError:
                    real_weights.append(None)

            # Fake-tides multi-band weights
            fake_weights = []
            for G_band, band_info in band_grams_fake:
                scales = _multi_scales(band_info["lambda_km"])
                rho_rr_b = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt_b = _rho_multi(Dg[b, pool], scales)
                try:
                    w_band = _reg_solve(
                        rho_rr_b * G_band[np.ix_(pool, pool)] +
                        s64.NUGGET * np.eye(len(pool)),
                        rho_rt_b * G_band[b, pool])
                    fake_weights.append(w_band)
                except np.linalg.LinAlgError:
                    fake_weights.append(None)

            B = R[:, b]; A = R[:, pool]

            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue

                p_full = float(w_full @ rr)

                real_preds = [float(w @ rr) for w in real_weights if w is not None]
                p_real = float(np.mean(real_preds)) if real_preds else p_full

                fake_preds = [float(w @ rr) for w in fake_weights if w is not None]
                p_fake = float(np.mean(fake_preds)) if fake_preds else p_full

                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["real_tides"].append((truth - p_real) ** 2)
                day_sq["fake_tides"].append((truth - p_fake) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["real_tides_rms"].append(rms["real_tides"])
        per_day["fake_tides_rms"].append(rms["fake_tides"])
        per_day["real_vs_full"].append(
            100.0 * (1.0 - rms["real_tides"] / rms["full_band"]))
        per_day["fake_vs_full"].append(
            100.0 * (1.0 - rms["fake_tides"] / rms["full_band"]))
        per_day["real_vs_fake"].append(
            100.0 * (1.0 - rms["real_tides"] / rms["fake_tides"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"real={np.median(per_day['real_tides_rms']):.1f}, "
                  f"fake={np.median(per_day['fake_tides_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "real_tides_vs_full_band_pct": _ci(per_day["real_vs_full"], rng),
        "fake_tides_vs_full_band_pct": _ci(per_day["fake_vs_full"], rng),
        "real_tides_vs_fake_tides_pct": _ci(per_day["real_vs_fake"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "real_tides": float(np.median(per_day["real_tides_rms"])) if n else None,
            "fake_tides": float(np.median(per_day["fake_tides_rms"])) if n else None,
        },
    }

    interpretations = []
    rvf = results["real_tides_vs_full_band_pct"]
    fvf = results["fake_tides_vs_full_band_pct"]
    rvr = results["real_tides_vs_fake_tides_pct"]

    if rvf.get("median") is not None:
        interpretations.append(
            f"Real-tides multi-band vs full-band: median {rvf['median']:+.2f}%, "
            f"CI95 [{rvf['ci95'][0]:+.2f}, {rvf['ci95'][1]:+.2f}].")
    if fvf.get("median") is not None:
        interpretations.append(
            f"Fake-tides multi-band vs full-band: median {fvf['median']:+.2f}%, "
            f"CI95 [{fvf['ci95'][0]:+.2f}, {fvf['ci95'][1]:+.2f}].")
    if rvr.get("median") is not None:
        if rvr["median"] > 0 and rvr["ci95"] and rvr["ci95"][0] > 0:
            interpretations.append(
                f"Real tides BEAT fake tides (median {rvr['median']:+.2f}%, "
                f"CI95 [{rvr['ci95'][0]:+.2f}, {rvr['ci95'][1]:+.2f}]). "
                f"The real gravitational tidal frequencies carry predictive "
                f"signal that fake non-physical frequencies do not. This is "
                f"direct evidence that the gravitational tidal structure is "
                f"operationally relevant, not just any low-frequency coherence.")
        elif rvr["median"] > 0:
            interpretations.append(
                f"Real tides are better than fake tides (median {rvr['median']:+.2f}%, "
                f"CI includes zero). The gravitational tidal structure may carry "
                f"predictive signal but the effect is not significant at 95%.")
        else:
            interpretations.append(
                f"Real tides do NOT beat fake tides (median {rvr['median']:+.2f}%). "
                f"The tidal band's contribution does not depend on the specific "
                f"gravitational tidal frequencies.")

    verdict = {
        "step": 144,
        "title": "Fake-tides control",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days_scored": n,
        "real_tidal_frequencies_uhz": [float(f*1e6) for f in REAL_TIDAL_FREQS_HZ],
        "fake_tidal_frequencies_uhz": [float(f*1e6) for f in FAKE_TIDAL_FREQS_HZ],
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6,
                    "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in BANDS],
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "If the real gravitational tidal frequencies carry the predictive "
            "signal, the real-tides multi-band predictor should beat the "
            "fake-tides variant. If the tidal band's contribution is just "
            "generic low-frequency coherence, fake tides should work equally "
            "well."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    if rvf.get("median") is not None:
        print(f"  Real vs full: {rvf['median']:+.2f}%", flush=True)
    if fvf.get("median") is not None:
        print(f"  Fake vs full: {fvf['median']:+.2f}%", flush=True)
    if rvr.get("median") is not None:
        print(f"  Real vs fake: {rvr['median']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
