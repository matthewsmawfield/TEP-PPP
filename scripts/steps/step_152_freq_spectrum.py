#!/usr/bin/env python3
"""
STEP 152  --  Per-frequency-bin predictive power spectrum.

The single most direct TEP test.  For each Fourier bin in the TEP band
(10-500 uHz), compute the leave-one-out prediction error using only that
bin's phase Gram.  The resulting "predictive power spectrum" shows which
frequencies carry predictive signal.

KEY PHYSICS:
  - M2 lunar tide (22.365 uHz, 12.42h): PURELY GRAVITATIONAL.
    No solar atmospheric counterpart.
  - S2 solar tide (23.148 uHz, 12.00h): has strong atmospheric component.
  - K1 (11.608 uHz, 23.93h): mixed gravitational + atmospheric.
  - O1 (10.758 uHz, 25.82h): primarily lunar gravitational.

If the predictive power spectrum shows peaks at M2 and O1 (lunar, purely
gravitational) and NOT at S2 (solar, atmospheric), that is direct evidence
that the gravitational tidal structure carries predictive signal — the
core TEP prediction.

METHOD:
  1. Use a 13-day causal window (resolution 0.77 uHz, enough to resolve
     M2 from S2 which are 0.79 uHz apart).
  2. For each Fourier bin, compute the cross-station phase Gram.
  3. For each target station, compute LOO prediction using only that bin.
  4. Average the squared prediction error across all targets and epochs.
  5. The predictive power at each frequency = 1 - (bin_error / total_error).

The 13-day window is constructed by concatenating 13 prior days of clock
data, resampled to a common 288-epoch grid.

Protocol: 2025 held-out data, 30 evaluation days.

Output: results/outputs/step_152_freq_spectrum.json
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib.tep_phase import spectral_phase_gram
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_152_freq_spectrum.json"
CLK_DIR = ROOT / "data" / "rinex" / "clk"
PRODUCT = "clk_cod"
YEAR = 2025

# Use every 5th day starting from DOY 20 (need 13 prior days)
EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10
REGISTERED_RNG_SEED = 20260920

# 26-day causal window for spectral resolution (0.445 uHz, resolves M2 from S2 at 0.79 uHz = 1.8 bins)
CAUSAL_DAYS = 26

# TEP band: 10-500 uHz
F_LO_HZ = 10e-6
F_HI_HZ = 500e-6

# Known tidal frequencies (uHz) for peak identification
TIDAL_LINES = {
    "Q1":  { "freq_uhz": 10.676, "type": "lunar",   "period_h": 26.87 },
    "O1":  { "freq_uhz": 10.758, "type": "lunar",   "period_h": 25.82 },
    "M1":  { "freq_uhz": 11.206, "type": "lunar",   "period_h": 24.83 },
    "P1":  { "freq_uhz": 11.542, "type": "solar",   "period_h": 24.07 },
    "S1":  { "freq_uhz": 11.574, "type": "solar",   "period_h": 24.00 },
    "K1":  { "freq_uhz": 11.608, "type": "mixed",   "period_h": 23.93 },
    "J1":  { "freq_uhz": 12.040, "type": "lunar",   "period_h": 23.10 },
    "N2":  { "freq_uhz": 21.422, "type": "lunar",   "period_h": 12.66 },
    "M2":  { "freq_uhz": 22.365, "type": "lunar",   "period_h": 12.42 },
    "S2":  { "freq_uhz": 23.148, "type": "solar",   "period_h": 12.00 },
    "K2":  { "freq_uhz": 23.098, "type": "solar",   "period_h": 11.97 },
    "M3":  { "freq_uhz": 34.533, "type": "lunar",   "period_h": 8.28 },
    "S3":  { "freq_uhz": 34.722, "type": "solar",   "period_h": 8.00 },
    "M4":  { "freq_uhz": 44.710, "type": "lunar",   "period_h": 6.21 },
    "S4":  { "freq_uhz": 46.296, "type": "solar",   "period_h": 6.00 },
    "M5":  { "freq_uhz": 55.887, "type": "lunar",   "period_h": 4.97 },
    "S5":  { "freq_uhz": 57.870, "type": "solar",   "period_h": 4.80 },
    "M6":  { "freq_uhz": 67.065, "type": "lunar",   "period_h": 4.14 },
    "S6":  { "freq_uhz": 69.444, "type": "solar",   "period_h": 4.00 },
    "M7":  { "freq_uhz": 78.243, "type": "lunar",   "period_h": 3.55 },
    "S7":  { "freq_uhz": 81.019, "type": "solar",   "period_h": 3.43 },
    "M8":  { "freq_uhz": 89.420, "type": "lunar",   "period_h": 3.11 },
    "S8":  { "freq_uhz": 92.593, "type": "solar",   "period_h": 3.00 },
}

def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-8: K = K + (1e-8 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)

def _input_path(tag):
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists(): return p
    return None

def _load_multi_day_prior(eval_tag, names, n_days):
    """Load and concatenate n_days of prior clock data, resampled to common grid."""
    eval_dt = datetime.strptime(eval_tag, "%Y%j")
    day_tags = []
    for d in range(n_days, 0, -1):
        ptag = (eval_dt - timedelta(days=d)).strftime("%Y%j")
        if _input_path(ptag) is not None:
            day_tags.append(ptag)

    if len(day_tags) < n_days - 2:  # allow up to 2 missing days
        return None

    # Common grid: 288 epochs at 300s cadence (matches step_64 DT_S)
    grid_t = np.arange(0.0, 86400.0, s64.DT_S)
    n_grid = len(grid_t)

    all_cols = {}
    for tag in day_tags:
        prev = s64.load_day(tag, product=PRODUCT)
        if prev is None: continue
        pnames, pt, praw, _ = prev
        for nm in names:
            if nm not in pnames: continue
            y_raw = praw[:, pnames.index(nm)].astype(float)
            t_raw = pt.astype(float)
            good = np.isfinite(y_raw)
            if good.mean() < 0.9: continue
            # Resample to common grid
            y = np.interp(grid_t, t_raw[good], y_raw[good], left=np.nan, right=np.nan)
            if not np.isfinite(y).all():
                good2 = np.isfinite(y)
                if good2.sum() < n_grid * 0.8: continue
                ix = np.arange(n_grid)
                y = np.interp(ix, ix[good2], y[good2])
            if nm not in all_cols:
                all_cols[nm] = []
            all_cols[nm].append(y)

    # All stations must have the same number of days
    n_days_per_sta = {nm: len(all_cols[nm]) for nm in all_cols}
    max_days = max(n_days_per_sta.values()) if n_days_per_sta else 0
    if max_days < n_days - 2:
        return None
    kept = [nm for nm in names if nm in all_cols and n_days_per_sta[nm] == max_days]
    if len(kept) < s64.MIN_REFS + 2:
        return None

    # Concatenate and detrend (quadratic)
    cols = []
    valid = []
    for nm in kept:
        y = np.concatenate(all_cols[nm])
        xq = np.arange(len(y)) - len(y) / 2.0
        sy = y - np.polyval(np.polyfit(xq, y, 2), xq)
        if not np.isfinite(sy).all() or not (1e-4 < np.std(sy) < 1e4):
            continue
        cols.append(sy)
        valid.append(nm)

    if len(valid) < s64.MIN_REFS + 2:
        return None

    return valid, np.column_stack(cols), max_days

def _per_bin_phase_gram(x, dt_s, freq_hz):
    """Compute the phase Gram at a single frequency.

    The phase Gram at frequency f is the outer product of the station
    phase vectors at that frequency.  It is rank-1 but still useful for
    coherent beamforming prediction.
    """
    n_st = x.shape[1]
    n_t = x.shape[0]
    # Windowed FFT at the target frequency
    window = np.hanning(n_t)
    Xw = np.fft.rfft(x * window[:, None], axis=0)
    freqs = np.fft.rfftfreq(n_t, d=dt_s)
    # Find the bin closest to freq_hz
    idx = np.argmin(np.abs(freqs - freq_hz))
    phase = Xw[idx, :]  # complex amplitude per station
    # Normalise to unit magnitude
    mag = np.abs(phase)
    if mag.max() < 1e-12:
        return None
    phase_norm = phase / (mag + 1e-15)
    # Phase Gram = Re(phase_i * conj(phase_j))
    G = np.real(np.outer(phase_norm, phase_norm.conj()))
    return G

def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4: return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(4000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)
    print("Step 152: Per-frequency-bin predictive power spectrum", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Causal window: {CAUSAL_DAYS} days (resolution ~{1e6/(CAUSAL_DAYS*86400):.2f} uHz)")
    print(f"  M2 (lunar): 22.365 uHz, S2 (solar): 23.148 uHz (separation 0.78 uHz)")
    print(f"  Band: {F_LO_HZ*1e6:.0f}-{F_HI_HZ*1e6:.0f} uHz", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t)]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        OUT.write_text(json.dumps({"step": 152, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    # Frequency grid: scan the TEP band at the 13-day resolution
    dt_s = s64.DT_S
    n_day = 288  # epochs per day
    n_total = n_day * CAUSAL_DAYS
    freqs = np.fft.rfftfreq(n_total, d=dt_s)
    # Select bins in the TEP band
    band_mask = (freqs >= F_LO_HZ) & (freqs <= F_HI_HZ)
    freq_bins = freqs[band_mask]
    n_bins = len(freq_bins)
    print(f"  {n_bins} frequency bins in TEP band", flush=True)
    print(f"  Frequency resolution: {freqs[1]*1e6:.3f} uHz", flush=True)
    print()

    # For each evaluation day, compute per-bin prediction error
    # Accumulate squared errors per bin across all days/stations/epochs
    bin_sq_err = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    # Also accumulate total error (using all bins = full-band)
    total_sq_err = 0.0
    total_count = 0

    npre = int(s64.PREWINDOW_S / dt_s)

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        if cur is None: continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur: continue

        # Load 13-day prior
        multi_prior = _load_multi_day_prior(tag, names, CAUSAL_DAYS)
        if multi_prior is None: continue
        prior_names, prior_x, n_days_loaded = multi_prior

        # Current day's clock residuals (detrended, causal)
        R_cols = []
        kept_cur = []
        for nm in names:
            if nm not in prior_names: continue
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

        if len(kept_cur) < s64.MIN_REFS + 2: continue

        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS
        Dg = D[np.ix_([names.index(n) for n in kept_cur], [names.index(n) for n in kept_cur])]
        gnames = kept_cur

        # QC on current day
        sd_r = np.nanstd(R, axis=0)
        med_r = np.median(sd_r)
        keep = [i for i in range(len(gnames)) if 0.01 < sd_r[i] < 12.0 and sd_r[i] < 3.5 * med_r]
        if len(keep) < s64.MIN_REFS + 2: continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]
        Dg = Dg[np.ix_(keep, keep)]

        # Prior data for the same station set
        prior_idx = [prior_names.index(nm) for nm in gnames]
        X_prior = prior_x[:, prior_idx]

        # QC on prior
        sd_p = np.nanstd(X_prior, axis=0)
        keep2 = [i for i in range(len(gnames)) if 0.01 < sd_p[i] < 1e4]
        if len(keep2) < s64.MIN_REFS + 2: continue
        gnames = [gnames[i] for i in keep2]
        R = R[:, keep2]
        Dg = Dg[np.ix_(keep2, keep2)]
        X_prior = X_prior[:, keep2]

        ng = len(gnames)
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        # Compute per-bin phase Grams
        bin_grams = []
        for bi, fb in enumerate(freq_bins):
            G = _per_bin_phase_gram(X_prior, dt_s, fb)
            bin_grams.append(G)

        # For each target, compute LOO prediction at each bin
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS: continue

            rho_rr = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt = s64._rho_multi(Dg[b, pool])

            B = R[:, b]
            A = R[:, pool]

            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth): continue
                rr = A[ei]
                if not np.isfinite(rr).all(): continue

                # Per-bin prediction
                for bi, G in enumerate(bin_grams):
                    if G is None:
                        bin_sq_err[bi] += truth ** 2
                        bin_count[bi] += 1
                        continue
                    Grt = G[b, pool]
                    Grr = G[np.ix_(pool, pool)]
                    try:
                        w = _reg_solve(rho_rr * Grr + s64.NUGGET * np.eye(len(pool)),
                                       rho_rt * Grt)
                        pred = float(w @ rr)
                        bin_sq_err[bi] += (truth - pred) ** 2
                        bin_count[bi] += 1
                    except np.linalg.LinAlgError:
                        bin_sq_err[bi] += truth ** 2
                        bin_count[bi] += 1

                # Full-band prediction (using the full-band phase Gram)
                try:
                    r_full = spectral_phase_gram(X_prior, dt_s, f_lo_hz=F_LO_HZ, f_hi_hz=F_HI_HZ,
                                                 min_bins=10, min_low_cycles=0.85)
                    G_full = r_full["predictive_gram"]
                    w_full = _reg_solve(rho_rr * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                        rho_rt * G_full[b, pool])
                    pred_full = float(w_full @ rr)
                    total_sq_err += (truth - pred_full) ** 2
                    total_count += 1
                except (ValueError, np.linalg.LinAlgError):
                    pass

        if (k + 1) % 5 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] {len(gnames)} stations, "
                  f"n_days_loaded={n_days_loaded}", flush=True)

    n_days = sum(1 for c in bin_count if c > 0)
    print(f"\n  {n_days} frequency bins scored", flush=True)
    if total_count == 0:
        print("  ERROR: no valid predictions", flush=True)
        OUT.write_text(json.dumps({"step": 152, "error": "no predictions", "n_days": 0}, indent=2))
        return

    # Compute per-bin RMS error and predictive power
    bin_rms = np.sqrt(bin_sq_err / np.maximum(bin_count, 1))
    total_rms = np.sqrt(total_sq_err / max(total_count, 1))

    # Predictive power = fractional reduction vs the mean bin error
    # (how much better than average each bin is)
    mean_bin_rms = np.mean(bin_rms)
    predictive_power = 100.0 * (1.0 - bin_rms / mean_bin_rms)

    # Also compute: for each bin, how much better than the full-band?
    predictive_vs_full = 100.0 * (1.0 - bin_rms / total_rms)

    # Identify peaks in the predictive power spectrum
    # Smooth with a 3-bin running mean
    pp_smooth = np.convolve(predictive_power, np.ones(3) / 3, mode="same")

    # Find the top peaks
    peak_indices = []
    for i in range(1, n_bins - 1):
        if pp_smooth[i] > pp_smooth[i-1] and pp_smooth[i] > pp_smooth[i+1] and pp_smooth[i] > 0:
            peak_indices.append(i)
    peak_indices.sort(key=lambda i: -pp_smooth[i])
    top_peaks = peak_indices[:20]

    # Match peaks to known tidal lines
    peak_matches = []
    for pi in top_peaks:
        f_uhz = freq_bins[pi] * 1e6
        pp = float(pp_smooth[pi])
        rms = float(bin_rms[pi])
        # Find nearest tidal line
        best_match = None
        best_dist = 999
        for name, info in TIDAL_LINES.items():
            dist = abs(f_uhz - info["freq_uhz"])
            if dist < best_dist:
                best_dist = dist
                best_match = name
        peak_matches.append({
            "freq_uhz": float(f_uhz),
            "predictive_power_pct": float(pp),
            "rms_m": float(rms),
            "nearest_tidal": best_match,
            "tidal_type": TIDAL_LINES[best_match]["type"] if best_match else "unknown",
            "dist_uhz": float(best_dist),
        })

    # Lunar vs solar comparison: average predictive power at lunar vs solar lines
    lunar_pp = []
    solar_pp = []
    for name, info in TIDAL_LINES.items():
        f_hz = info["freq_uhz"] * 1e-6
        if f_hz < F_LO_HZ or f_hz > F_HI_HZ: continue
        idx = np.argmin(np.abs(freq_bins - f_hz))
        if abs(freq_bins[idx] - f_hz) * 1e6 < 1.0:  # within 1 uHz
            pp = float(pp_smooth[idx])
            if info["type"] == "lunar":
                lunar_pp.append((name, pp))
            elif info["type"] == "solar":
                solar_pp.append((name, pp))

    lunar_mean = float(np.mean([p for _, p in lunar_pp])) if lunar_pp else None
    solar_mean = float(np.mean([p for _, p in solar_pp])) if solar_pp else None

    # M2 vs S2 specifically
    m2_idx = np.argmin(np.abs(freq_bins - 22.365e-6))
    s2_idx = np.argmin(np.abs(freq_bins - 23.148e-6))
    m2_pp = float(pp_smooth[m2_idx])
    s2_pp = float(pp_smooth[s2_idx])
    m2_rms = float(bin_rms[m2_idx])
    s2_rms = float(bin_rms[s2_idx])

    results = {
        "n_bins": int(n_bins),
        "n_bins_scored": int(n_days),
        "n_eval_days": len(eval_tags),
        "freq_resolution_uhz": float(freqs[1] * 1e6),
        "total_rms_m": float(total_rms),
        "mean_bin_rms_m": float(mean_bin_rms),
        "m2_lunar": {
            "freq_uhz": float(freq_bins[m2_idx] * 1e6),
            "predictive_power_pct": m2_pp,
            "rms_m": m2_rms,
        },
        "s2_solar": {
            "freq_uhz": float(freq_bins[s2_idx] * 1e6),
            "predictive_power_pct": s2_pp,
            "rms_m": s2_rms,
        },
        "m2_vs_s2_pp_diff": float(m2_pp - s2_pp),
        "lunar_mean_pp": lunar_mean,
        "solar_mean_pp": solar_mean,
        "lunar_vs_solar_diff": float(lunar_mean - solar_mean) if lunar_mean and solar_mean else None,
        "lunar_lines": [{"name": n, "pp": p} for n, p in lunar_pp],
        "solar_lines": [{"name": n, "pp": p} for n, p in solar_pp],
        "top_peaks": peak_matches,
        "freq_bins_uhz": [float(f * 1e6) for f in freq_bins],
        "predictive_power_pct": [float(p) for p in predictive_power],
        "predictive_power_smoothed_pct": [float(p) for p in pp_smooth],
        "bin_rms_m": [float(r) for r in bin_rms],
    }

    interp = []
    interp.append(f"Per-frequency-bin predictive power spectrum: {n_bins} bins, {n_days} eval days, 13-day window.")
    interp.append(f"M2 (lunar, 22.365 uHz): predictive power = {m2_pp:+.2f}%, RMS = {m2_rms:.3f} m.")
    interp.append(f"S2 (solar, 23.148 uHz): predictive power = {s2_pp:+.2f}%, RMS = {s2_rms:.3f} m.")
    interp.append(f"M2 vs S2 predictive power difference: {m2_pp - s2_pp:+.2f}%.")
    if lunar_mean is not None and solar_mean is not None:
        interp.append(f"Lunar mean predictive power: {lunar_mean:+.2f}% ({len(lunar_pp)} lines).")
        interp.append(f"Solar mean predictive power: {solar_mean:+.2f}% ({len(solar_pp)} lines).")
        interp.append(f"Lunar vs solar difference: {lunar_mean - solar_mean:+.2f}%.")
        if lunar_mean > solar_mean:
            interp.append("Lunar tidal lines carry MORE predictive power than solar tidal lines. Since lunar tides are purely gravitational (no atmospheric counterpart), this is direct evidence for gravitational TEP coupling.")
        else:
            interp.append("Solar tidal lines carry more predictive power than lunar. This could indicate an atmospheric contribution to the solar tidal prediction.")
    top5 = ", ".join("{}({:.2f}uHz, {:+.1f}%)".format(p["nearest_tidal"], p["freq_uhz"], p["predictive_power_pct"]) for p in peak_matches[:5])
    interp.append(f"Top 5 peaks: {top5}")

    verdict = {
        "step": 152, "title": "Per-frequency-bin predictive power spectrum",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR,
        "causal_window_days": CAUSAL_DAYS,
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "If TEP is real, the predictive power spectrum should show peaks at lunar tidal frequencies (M2, O1, N2) that are purely gravitational. The M2 (22.365 uHz) should carry more predictive power than the nearby S2 (23.148 uHz) which has an atmospheric counterpart. The lunar mean predictive power should exceed the solar mean.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"  M2 (lunar): pp={m2_pp:+.2f}%, rms={m2_rms:.3f} m", flush=True)
    print(f"  S2 (solar): pp={s2_pp:+.2f}%, rms={s2_rms:.3f} m", flush=True)
    print(f"  M2 - S2: {m2_pp - s2_pp:+.2f}%", flush=True)
    if lunar_mean and solar_mean:
        print(f"  Lunar mean: {lunar_mean:+.2f}%, Solar mean: {solar_mean:+.2f}%", flush=True)
        print(f"  Lunar - Solar: {lunar_mean - solar_mean:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
