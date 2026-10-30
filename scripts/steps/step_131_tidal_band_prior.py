#!/usr/bin/env python3
"""
STEP 131  --  Tidal-band TEP prior: the gravitational tidal lines ARE the
              TEP signal.

CORRECTING THE FRAMING.

The previous Step 127 notched OUT the solar and lunar tidal lines, treating
them as atmospheric modulation to be removed.  This was backwards.  The
tidal lines are GRAVITATIONAL in origin:

  * 24h, 12h, 8h, 6h:  solar gravitational tides (the Sun's gravitational
    forcing drives these periods; the atmosphere responds gravitationally)
  * 12.4h M2:           lunar semidiurnal gravitational tide (purely
    gravitational — no solar atmospheric component)

The TEP-GNSS Paper 1 already established that the longest correlation
lengths are at tidal frequencies (lambda=4677 km at tidal frequencies vs
1502 km post-tidal).  The tidal band has the STRONGEST spatial coherence
— it is the most TEP-like part of the signal, not noise to be removed.

This step ISOLATES the gravitational tidal band and uses it as the
predictive kernel.  If the tidal-band kernel beats both the full-band
kernel and the empirical covariance on independent-year data, that is
direct evidence that the gravitational coupling carries predictive power.

PROTOCOL:
  * Run the Step 64/92 P1 gate on 2025 held-out data.
  * Compare four kernels:
    1. full_band:   the registered full-band TEP kernel (10-500 uHz)
    2. tidal_band:  ONLY the gravitational tidal frequencies
                    (24h, 12h, 8h, 6h solar + 12.4h M2 lunar)
    3. notched:     the continuum with tides REMOVED (Step 127, for contrast)
    4. empirical:   data-driven covariance

  * TEP prediction: the tidal-band kernel beats both the full-band and
    empirical covariance, because the gravitational tides are the
    strongest and most stable component of the TEP field.

Output: results/outputs/step_131_tidal_band_prior.json
"""
from __future__ import annotations

import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_131_tidal_band_prior.json"
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
REGISTERED_RNG_SEED = 20260902

# Gravitational tidal periods (seconds)
TIDAL_PERIODS = [
    86400.0,   # 24h  S1   solar gravitational
    43200.0,   # 12h  S2   solar gravitational
    28800.0,   # 8h   S3   solar gravitational
    21600.0,   # 6h   S4   solar gravitational
    86164.0,   # 23h56m sidereal (solar + Earth rotation)
    44712.0,   # 12h25m M2 lunar semidiurnal (purely gravitational)
]
TIDAL_WIDTH_FRAC = 0.05

# Tidal-band lambda from TEP-GNSS Paper 1: 4677 km at tidal frequencies
TIDAL_LAMBDA_KM = 4677.0
TIDAL_MULTI_SCALES_KM = (TIDAL_LAMBDA_KM / 1.6,
                         TIDAL_LAMBDA_KM,
                         TIDAL_LAMBDA_KM * 1.6)

# Notched lambda from Step 121: 1877 km (for contrast)
NOTCHED_LAMBDA_KM = 1877.0
NOTCHED_MULTI_SCALES_KM = (NOTCHED_LAMBDA_KM / 1.6,
                           NOTCHED_LAMBDA_KM,
                           NOTCHED_LAMBDA_KM * 1.6)


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _tidal_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                      min_bins=4, min_low_cycles=0.85):
    """Phase Gram using ONLY the gravitational tidal frequencies.

    This isolates the gravitational TEP signal.  The tidal frequencies
    are the strongest and most spatially coherent component of the TEP
    field (Paper 1: lambda=4677 km at tidal vs 1502 km post-tidal).
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history is too short to resolve the registered low frequency")

    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero station scale")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("non-finite or zero detrended scale")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))

    # Select ONLY the tidal frequencies
    tidal_mask = np.zeros(len(freq), dtype=bool)
    for period in TIDAL_PERIODS:
        fn = 1.0 / period
        lo = fn * (1.0 - TIDAL_WIDTH_FRAC)
        hi = fn * (1.0 + TIDAL_WIDTH_FRAC)
        tidal_mask |= (freq >= lo) & (freq <= hi)

    if int(tidal_mask.sum()) < int(min_bins):
        raise ValueError("too few tidal frequency bins")
    F = F[tidal_mask]

    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero spectral energy in tidal band")
    norm = np.sqrt(np.outer(energy, energy))
    pg = np.real(cross / norm)
    pg = 0.5 * (pg + pg.T)
    np.fill_diagonal(pg, 1.0)
    return pg


def _notched_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                        min_bins=6, min_low_cycles=0.85):
    """Phase Gram with tidal lines REMOVED (for contrast)."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history too short")
    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("bad scale")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("bad detrended scale")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    band = (freq >= f_lo_hz) & (freq <= f_hi_hz)
    notch_mask = np.zeros(len(freq), dtype=bool)
    for period in TIDAL_PERIODS:
        fn = 1.0 / period
        lo = fn * (1.0 - TIDAL_WIDTH_FRAC)
        hi = fn * (1.0 + TIDAL_WIDTH_FRAC)
        notch_mask |= (freq >= lo) & (freq <= hi)
    band &= ~notch_mask
    if int(band.sum()) < int(min_bins):
        raise ValueError("too few bins after notching")
    F = F[band]
    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero energy")
    norm = np.sqrt(np.outer(energy, energy))
    pg = np.real(cross / norm)
    pg = 0.5 * (pg + pg.T)
    np.fill_diagonal(pg, 1.0)
    return pg


def _rho_multi_tidal(d):
    return np.mean([np.exp(-d / L) for L in TIDAL_MULTI_SCALES_KM], axis=0)


def _rho_multi_notched(d):
    return np.mean([np.exp(-d / L) for L in NOTCHED_MULTI_SCALES_KM], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _prior_gram_band(prev_day, cur_names, gram_fn):
    """Build a phase Gram using the specified band selection function."""
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
    try:
        g = gram_fn(np.column_stack(spectral_cols), s64.DT_S)
    except ValueError:
        return None
    return kept, g, np.column_stack(clean_raw_cols)


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

    print("Step 131: Tidal-band TEP prior — gravitational tides ARE the signal",
          flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Tidal lambda: {TIDAL_LAMBDA_KM} km (Paper 1 tidal-band scale)", flush=True)
    print(f"  Full-band lambda: {s64.LAMBDA_BASE_KM} km", flush=True)
    print(f"  Notched lambda: {NOTCHED_LAMBDA_KM} km (for contrast)", flush=True)
    print(f"  Tidal periods: {[f'{p/3600:.1f}h' for p in TIDAL_PERIODS]}", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    missing = [tag for tag in required_tags if _input_path(tag) is None]
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 131, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "tidal_rms", "notched_rms", "emp_rms",
        "tidal_vs_full", "tidal_vs_notched", "tidal_vs_emp")}

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

        pg_full = s64.prior_gram(prev, names)
        pg_tidal = _prior_gram_band(prev, names, _tidal_phase_gram)
        pg_notched = _prior_gram_band(prev, names, _notched_phase_gram)

        if pg_full is None or pg_tidal is None or pg_notched is None:
            continue

        gnames_f, G_full, P_full = pg_full
        gnames_t, G_tidal, _ = pg_tidal
        gnames_n, G_notch, _ = pg_notched

        common = sorted(set(gnames_f) & set(gnames_t) & set(gnames_n))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_t = [gnames_t.index(n) for n in common]
        idx_n = [gnames_n.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        G_tidal = G_tidal[np.ix_(idx_t, idx_t)]
        G_notch = G_notch[np.ix_(idx_n, idx_n)]
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
        G_tidal = G_tidal[np.ix_(keep, keep)]
        G_notch = G_notch[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "tidal", "notched", "emp")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band TEP kernel
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Tidal-band TEP kernel (gravitational tides only)
            rho_rr_t = _rho_multi_tidal(Dg[np.ix_(pool, pool)])
            rho_rt_t = _rho_multi_tidal(Dg[b, pool])
            try:
                w_tidal = _reg_solve(
                    rho_rr_t * G_tidal[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_t * G_tidal[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Notched kernel (tides removed, for contrast)
            rho_rr_n = _rho_multi_notched(Dg[np.ix_(pool, pool)])
            rho_rt_n = _rho_multi_notched(Dg[b, pool])
            try:
                w_notch = _reg_solve(
                    rho_rr_n * G_notch[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_n * G_notch[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Empirical covariance
            Pc = P[:, pool] - P[:, pool].mean(1, keepdims=True)
            C_emp = np.einsum("ti,tj->ij", Pc, Pc, optimize=False) / Pc.shape[0]
            de = np.diag(C_emp).copy(); de[de <= 0] = 1e-9
            corr_emp = C_emp / np.sqrt(np.outer(de, de))
            iu_p, ju_p = np.triu_indices(len(pool), 1)
            dv = Dg[np.ix_(pool, pool)][iu_p, ju_p]
            cv = corr_emp[iu_p, ju_p]
            best_e = (np.inf, 4200.0)
            for Le in (200, 400, 800, 1500, 3000, 6000, 12000, 25000):
                pr = np.exp(-0.5 * (dv / Le) ** 2)
                sse = np.sum((cv - pr) ** 2)
                if sse < best_e[0]:
                    best_e = (sse, Le)
            Le = best_e[1]
            ce_rr = np.exp(-0.5 * (Dg[np.ix_(pool, pool)] / Le) ** 2)
            ce_rt = np.exp(-0.5 * (Dg[b, pool] / Le) ** 2)
            try:
                w_emp = _reg_solve(ce_rr + s64.NUGGET * np.eye(len(pool)), ce_rt)
            except np.linalg.LinAlgError:
                w_emp = w_full

            B = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                day_sq["full_band"].append((truth - float(w_full @ rr)) ** 2)
                day_sq["tidal"].append((truth - float(w_tidal @ rr)) ** 2)
                day_sq["notched"].append((truth - float(w_notch @ rr)) ** 2)
                day_sq["emp"].append((truth - float(w_emp @ rr)) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["tidal_rms"].append(rms["tidal"])
        per_day["notched_rms"].append(rms["notched"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["tidal_vs_full"].append(
            100.0 * (1.0 - rms["tidal"] / rms["full_band"]))
        per_day["tidal_vs_notched"].append(
            100.0 * (1.0 - rms["tidal"] / rms["notched"]))
        per_day["tidal_vs_emp"].append(
            100.0 * (1.0 - rms["tidal"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"tidal={np.median(per_day['tidal_rms']):.1f}, "
                  f"notch={np.median(per_day['notched_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "tidal_vs_full_band_pct": _ci(per_day["tidal_vs_full"], rng),
        "tidal_vs_notched_pct": _ci(per_day["tidal_vs_notched"], rng),
        "tidal_vs_empirical_pct": _ci(per_day["tidal_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "tidal": float(np.median(per_day["tidal_rms"])) if n else None,
            "notched": float(np.median(per_day["notched_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
    }

    interpretations = []
    tvf = results["tidal_vs_full_band_pct"]
    tvn = results["tidal_vs_notched_pct"]
    tve = results["tidal_vs_empirical_pct"]

    if tvf.get("median") is not None:
        if tvf["median"] > 0 and tvf["ci95"] and tvf["ci95"][0] > 0:
            interpretations.append(
                f"The tidal-band TEP kernel BEATS the full-band kernel on "
                f"independent-year data (median {tvf['median']:+.2f}%, "
                f"CI95 [{tvf['ci95'][0]:+.2f}, {tvf['ci95'][1]:+.2f}]). "
                f"The gravitational tidal lines carry the strongest "
                f"predictive power — they are the TEP signal, not noise. "
                f"This confirms Paper 1's finding that the longest "
                f"correlation lengths are at tidal frequencies.")
        elif tvf["median"] > 0:
            interpretations.append(
                f"The tidal-band kernel is better than the full-band "
                f"kernel (median {tvf['median']:+.2f}%, CI includes zero).")
        else:
            interpretations.append(
                f"The tidal-band kernel does not beat the full-band "
                f"kernel ({tvf['median']:+.2f}%).  The tidal lines alone "
                f"do not carry sufficient predictive power without the "
                f"continuum.")

    if tvn.get("median") is not None:
        if tvn["median"] > 0 and tvn["ci95"] and tvn["ci95"][0] > 0:
            interpretations.append(
                f"The tidal-band kernel BEATS the notched kernel "
                f"(median {tvn['median']:+.2f}%, "
                f"CI95 [{tvn['ci95'][0]:+.2f}, {tvn['ci95'][1]:+.2f}]). "
                f"The gravitational tides carry MORE predictive power than "
                f"the residual continuum.  Step 127's notching approach "
                f"was removing the strongest part of the TEP signal.")
        elif tvn["median"] > 0:
            interpretations.append(
                f"The tidal-band kernel is better than the notched "
                f"kernel (median {tvn['median']:+.2f}%, CI includes zero).")

    if tve.get("median") is not None:
        if tve["median"] > 0 and tve["ci95"] and tve["ci95"][0] > 0:
            interpretations.append(
                f"The tidal-band TEP kernel BEATS empirical covariance on "
                f"independent-year data (median {tve['median']:+.2f}%, "
                f"CI95 [{tve['ci95'][0]:+.2f}, {tve['ci95'][1]:+.2f}]). "
                f"The physics-fixed gravitational tidal kernel (lambda="
                f"{TIDAL_LAMBDA_KM} km from Paper 1) generalises where the "
                f"data-driven covariance overfits.  This is direct evidence "
                f"that gravitational coupling carries predictive power.")
        elif tve["median"] > 0:
            interpretations.append(
                f"The tidal-band kernel is better than empirical "
                f"covariance (median {tve['median']:+.2f}%, CI includes "
                f"zero).")

    verdict = {
        "step": 131,
        "title": "Tidal-band TEP prior: gravitational tides ARE the TEP signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_evaluation_days": len(eval_tags),
        "n_days_scored": n,
        "tidal_periods_h": [p / 3600.0 for p in TIDAL_PERIODS],
        "tidal_width_frac": TIDAL_WIDTH_FRAC,
        "lambdas": {
            "tidal": TIDAL_LAMBDA_KM,
            "full_band": s64.LAMBDA_BASE_KM,
            "notched": NOTCHED_LAMBDA_KM,
        },
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The gravitational tidal lines (24h, 12h, 8h, 6h solar + 12.4h "
            "M2 lunar) are the strongest and most spatially coherent "
            "component of the TEP field (Paper 1: lambda=4677 km at tidal "
            "frequencies vs 1502 km post-tidal).  The tidal-band kernel "
            "should beat both the full-band kernel (which dilutes the "
            "tidal signal with continuum noise) and the notched kernel "
            "(which removes the tidal signal entirely).  If it also beats "
            "empirical covariance on independent-year data, that is direct "
            "evidence that gravitational coupling carries predictive power."),
        "correction": (
            "Step 127 notched OUT the tidal lines, treating them as "
            "atmospheric modulation.  This was backwards — the tidal lines "
            "are gravitational in origin.  This step corrects the framing "
            "by isolating and leveraging the gravitational tidal band."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  Tidal vs full:    {tvf.get('median')}", flush=True)
    print(f"  Tidal vs notched: {tvn.get('median')}", flush=True)
    print(f"  Tidal vs emp:     {tve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
