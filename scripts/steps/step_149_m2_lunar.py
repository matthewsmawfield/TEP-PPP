#!/usr/bin/env python3
"""
STEP 149  --  M2 lunar tidal test: the cleanest gravitational signal.

The M2 lunar tide (12.42h, 22.365 uHz) is the cleanest gravitational
tidal signal in the TEP band because it has NO solar atmospheric
counterpart.  Solar tides (24h, 12h, 8h) could conceivably be confused
with solar-driven atmospheric effects, but the M2 is purely gravitational
— the atmosphere has negligible M2 response.

This test asks: does the M2 lunar frequency specifically carry
predictive signal in the TEP clock field?

METHOD:
  1. Use a 5-day causal window (120h) to resolve the M2 frequency
     (resolution ~2.31 uHz).  M2 and S2 are 0.79 uHz apart, so they
     cannot be fully separated at this resolution; the bands overlap
     but centre on different frequencies.
  2. Build a narrow-band phase Gram centred on M2 (±1.5 uHz).
  3. Compare predictions using:
     a. M2 band: 20.86-23.86 uHz (centred on M2 at 22.365, ±1.5 uHz)
     b. S2 solar band: 21.65-24.65 uHz (centred on S2 at 23.148, ±1.5 uHz)
        — same width, but centred on the solar semidiurnal tide
     c. Off-tide band: 15-19 uHz (no major tidal lines)
     d. Full tidal band: 9.84-35 uHz (Step 134 baseline)
  4. All use the same lambda=4677 km (tidal band scale).
  5. If M2 carries gravitational TEP signal, it should beat the off-tide
     band and be comparable to or better than the S2 solar band.

The 5-day causal window is constructed by concatenating the prior 5
days of clock data for the common station set, with proper handling of
day boundaries.

Protocol: 2025 held-out data.

Output: results/outputs/step_149_m2_lunar.json
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

OUT = ROOT / "results" / "outputs" / "step_149_m2_lunar.json"
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
MIN_VALID_DAYS = 40
REGISTERED_RNG_SEED = 20260918

# M2 lunar tide: 12.42h = 22.365 uHz
# S2 solar tide: 12.00h = 23.148 uHz
# These are 0.78 uHz apart; a 5-day window gives 2.31 uHz resolution.
# Use ±1.5 uHz bands → 3 uHz wide, giving ≥1 bin at 5-day resolution.
# M2: 20.865–23.865, S2: 21.648–24.648 — they overlap but centre on
# different frequencies. The phase Gram captures the central frequency.
M2_FREQ_HZ = 22.365e-6
S2_FREQ_HZ = 23.148e-6
BAND_HALF_WIDTH_HZ = 1.5e-6  # ±1.5 uHz

# Test bands (all use lambda=4677 km, the tidal band scale)
TEST_BANDS = [
    {"name": "m2_lunar",    "f_lo": M2_FREQ_HZ - BAND_HALF_WIDTH_HZ, "f_hi": M2_FREQ_HZ + BAND_HALF_WIDTH_HZ, "lambda_km": 4677.0, "min_bins": 1},
    {"name": "s2_solar",    "f_lo": S2_FREQ_HZ - BAND_HALF_WIDTH_HZ, "f_hi": S2_FREQ_HZ + BAND_HALF_WIDTH_HZ, "lambda_km": 4677.0, "min_bins": 1},
    {"name": "off_tide",    "f_lo": 15e-6, "f_hi": 19e-6, "lambda_km": 4677.0, "min_bins": 2},
    {"name": "full_tidal",  "f_lo": 9.84e-6, "f_hi": 35e-6, "lambda_km": 4677.0, "min_bins": 2},
]

# 5-day causal window for spectral resolution (2.31 uHz)
CAUSAL_DAYS = 5

def _multi_scales(lam): return (lam / 1.6, lam, lam * 1.6)
def _rho_multi(d, scales): return np.mean([np.exp(-d / L) for L in scales], axis=0)

def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6: K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)

def _input_path(tag):
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists(): return p
    return None

def _load_multi_day_prior(eval_tag, names, n_days):
    """Load and concatenate n_days of prior clock data for the common station set.

    Each day is resampled to a common 288-epoch grid (30s cadence, 86400s day)
    before concatenation to handle varying epoch counts.
    """
    eval_dt = datetime.strptime(eval_tag, "%Y%j")
    day_tags = []
    for d in range(n_days, 0, -1):
        ptag = (eval_dt - timedelta(days=d)).strftime("%Y%j")
        if _input_path(ptag) is not None:
            day_tags.append(ptag)

    if len(day_tags) < 2:  # need at least 2 days for M2 resolution
        return None

    # Common grid: 288 epochs at 30s cadence
    grid_t = np.arange(1800.0, 86400.0, s64.DT_S)
    n_grid = len(grid_t)

    # Load each day and resample to common grid
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
                # Fill gaps
                good2 = np.isfinite(y)
                if good2.sum() < n_grid * 0.8: continue
                ix = np.arange(n_grid)
                y = np.interp(ix, ix[good2], y[good2])
            if nm not in all_cols:
                all_cols[nm] = []
            all_cols[nm].append(y)

    # Must have data from at least 2 days for all stations
    kept = [nm for nm in names if nm in all_cols and len(all_cols[nm]) >= 2]
    if len(kept) < s64.MIN_REFS + 1:
        return None

    # All stations must have the same number of days; find the common set
    n_days_per_sta = {nm: len(all_cols[nm]) for nm in kept}
    max_days = max(n_days_per_sta.values())
    kept = [nm for nm in kept if n_days_per_sta[nm] == max_days]
    if len(kept) < s64.MIN_REFS + 1:
        return None

    # Concatenate and detrend
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

    if len(valid) < s64.MIN_REFS + 1:
        return None

    return valid, np.column_stack(cols)

def _band_phase_grams(x, dt_s, bands):
    grams = []
    for band in bands:
        try:
            r = spectral_phase_gram(x, dt_s, f_lo_hz=band["f_lo"], f_hi_hz=band["f_hi"],
                                    min_bins=band.get("min_bins", 2), min_low_cycles=0.85)
            grams.append((r["predictive_gram"], band))
        except ValueError: continue
    return grams if grams else None

def _ci(values, rng):
    v = np.array(values, float)
    v = v[np.isfinite(v)]  # drop NaN from failed bands
    if len(v) < 4: return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(REGISTERED_RNG_SEED)
    print("Step 149: M2 lunar tidal test", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  M2: {M2_FREQ_HZ*1e6:.3f} uHz (12.42h lunar)")
    print(f"  S2: {S2_FREQ_HZ*1e6:.3f} uHz (12.00h solar)")
    print(f"  Causal window: {CAUSAL_DAYS} days (resolution ~{1e6/(CAUSAL_DAYS*86400):.2f} uHz)")
    for band in TEST_BANDS:
        print(f"  {band['name']:14s}: {band['f_lo']*1e6:.2f}-{band['f_hi']*1e6:.2f} uHz, lambda={band['lambda_km']:.0f} km", flush=True)
    print()

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t) and _input_path(s64._prev(t))]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        OUT.write_text(json.dumps({"step": 149, "error": "too few days", "n_days": len(eval_tags)}, indent=2))
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "m2_rms", "s2_rms", "off_tide_rms", "full_tidal_rms",
        "m2_vs_off", "s2_vs_off", "m2_vs_s2", "full_tidal_vs_off")}

    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        if cur is None: continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s_cur: continue

        # Full-band Gram (single-lambda, 1-day prior)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if prev is None: continue
        pg_full = s64.prior_gram(prev, names)
        if pg_full is None: continue

        # Multi-day prior for M2/S2/off-tide bands
        multi_prior = _load_multi_day_prior(tag, names, CAUSAL_DAYS)
        if multi_prior is None: continue
        prior_names, prior_x = multi_prior

        # Build per-band Grams from the 3-day window
        band_grams = _band_phase_grams(prior_x, s64.DT_S, TEST_BANDS)
        if band_grams is None: continue

        gnames_f, G_full, P_full = pg_full
        bg_names = prior_names
        common = sorted(set(gnames_f) & set(bg_names))
        if len(common) < s64.MIN_REFS + 2: continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_bg = [bg_names.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]
        G_full = G_full[np.ix_(idx_f, idx_f)]
        band_grams = [(g[np.ix_(idx_bg, idx_bg)], band) for g, band in band_grams]
        P = P_full[:, idx_f]; Dg = D[np.ix_(idx_cur, idx_cur)]; gnames = common

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
            if not m.all(): P[:, j] = np.interp(np.arange(P.shape[0]), np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0); med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames)) if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2: continue
        gnames = [gnames[i] for i in keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        band_grams = [(g[np.ix_(keep, keep)], band) for g, band in band_grams]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "m2", "s2", "off_tide", "full_tidal")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS: continue

            # Full-band single-lambda
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(rho_rr_f * G_full[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError: continue

            # Per-band weights
            band_ws = {}
            for G_band, band_info in band_grams:
                scales = _multi_scales(band_info["lambda_km"])
                rho_rr = _rho_multi(Dg[np.ix_(pool, pool)], scales)
                rho_rt = _rho_multi(Dg[b, pool], scales)
                try:
                    w = _reg_solve(rho_rr * G_band[np.ix_(pool, pool)] + s64.NUGGET * np.eye(len(pool)),
                                   rho_rt * G_band[b, pool])
                    band_ws[band_info["name"]] = w
                except np.linalg.LinAlgError:
                    band_ws[band_info["name"]] = None

            B = R[:, b]; A = R[:, pool]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth): continue
                rr = A[ei]
                if not np.isfinite(rr).all(): continue
                p_full = float(w_full @ rr)
                day_sq["full_band"].append((truth - p_full) ** 2)
                for bname, sq_key in (("m2_lunar", "m2"), ("s2_solar", "s2"),
                                       ("off_tide", "off_tide"), ("full_tidal", "full_tidal")):
                    w = band_ws.get(bname)
                    if w is not None:
                        p = float(w @ rr)
                        day_sq[sq_key].append((truth - p) ** 2)
                    else:
                        # Band failed — skip rather than silently using p_full,
                        # which would make the band identical to full_band.
                        day_sq[sq_key].append(np.nan)

        if any(len(day_sq[kk]) < 30 for kk in day_sq): continue
        # Use only finite (non-NaN) samples for each band's RMS
        rms = {}
        for kk, v in day_sq.items():
            fv = np.array(v)[np.isfinite(v)]
            rms[kk] = float(np.sqrt(np.mean(fv))) if len(fv) > 0 else np.nan
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["m2_rms"].append(rms["m2"])
        per_day["s2_rms"].append(rms["s2"])
        per_day["off_tide_rms"].append(rms["off_tide"])
        per_day["full_tidal_rms"].append(rms["full_tidal"])
        # Guard against NaN or zero denominators
        off = rms["off_tide"]; s2 = rms["s2"]; m2 = rms["m2"]
        per_day["m2_vs_off"].append(100.0 * (1.0 - m2 / off) if np.isfinite(off) and off > 1e-12 else np.nan)
        per_day["s2_vs_off"].append(100.0 * (1.0 - rms["s2"] / off) if np.isfinite(off) and off > 1e-12 else np.nan)
        per_day["m2_vs_s2"].append(100.0 * (1.0 - m2 / s2) if np.isfinite(s2) and s2 > 1e-12 else np.nan)
        per_day["full_tidal_vs_off"].append(100.0 * (1.0 - rms["full_tidal"] / off) if np.isfinite(off) and off > 1e-12 else np.nan)

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] m2={np.median(per_day['m2_rms']):.1f}, "
                  f"s2={np.median(per_day['s2_rms']):.1f}, "
                  f"off={np.median(per_day['off_tide_rms']):.1f} m", flush=True)

    n = len(per_day["m2_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "m2_vs_off_tide_pct": _ci(per_day["m2_vs_off"], rng),
        "s2_vs_off_tide_pct": _ci(per_day["s2_vs_off"], rng),
        "m2_vs_s2_pct": _ci(per_day["m2_vs_s2"], rng),
        "full_tidal_vs_off_tide_pct": _ci(per_day["full_tidal_vs_off"], rng),
        "median_rms_m": {
            "full_band": float(np.nanmedian(per_day["full_band_rms"])) if n else None,
            "m2_lunar": float(np.nanmedian(per_day["m2_rms"])) if n else None,
            "s2_solar": float(np.nanmedian(per_day["s2_rms"])) if n else None,
            "off_tide": float(np.nanmedian(per_day["off_tide_rms"])) if n else None,
            "full_tidal": float(np.nanmedian(per_day["full_tidal_rms"])) if n else None,
        },
    }

    interp = []
    m2vo = results["m2_vs_off_tide_pct"]
    s2vo = results["s2_vs_off_tide_pct"]
    m2vs = results["m2_vs_s2_pct"]
    ftvo = results["full_tidal_vs_off_tide_pct"]

    for label, v in [("M2 vs off-tide", m2vo), ("S2 vs off-tide", s2vo),
                     ("M2 vs S2", m2vs), ("Full tidal vs off-tide", ftvo)]:
        if v.get("median") is not None:
            interp.append(f"{label}: median {v['median']:+.2f}%, CI95 [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}].")

    if m2vo.get("median") is not None and m2vo["median"] > 0 and m2vo.get("ci95", [0])[0] > 0:
        interp.append("M2 lunar beats off-tide with positive CI: the M2 gravitational tidal frequency carries predictive signal. Since M2 has no solar atmospheric counterpart, this is direct gravitational TEP evidence.")
    elif m2vo.get("median") is not None and m2vo["median"] > 0:
        interp.append("M2 lunar is better than off-tide but CI includes zero.")

    verdict = {
        "step": 149, "title": "M2 lunar tidal test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR, "n_days_scored": n,
        "m2_freq_uhz": M2_FREQ_HZ * 1e6,
        "s2_freq_uhz": S2_FREQ_HZ * 1e6,
        "causal_window_days": CAUSAL_DAYS,
        "bands": [{"name": b["name"], "f_lo_uhz": b["f_lo"]*1e6, "f_hi_uhz": b["f_hi"]*1e6,
                    "lambda_km": b["lambda_km"]} for b in TEST_BANDS],
        "results": results, "interpretation": interp,
        "pre_registered_prediction": "The M2 lunar frequency (22.365 uHz, 12.42h) should carry predictive signal because it is purely gravitational with no solar atmospheric counterpart. If M2 beats the off-tide band, the gravitational tidal structure is directly operationally relevant.",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for label, v in [("M2 vs off", m2vo), ("S2 vs off", s2vo), ("M2 vs S2", m2vs)]:
        if v.get("median") is not None: print(f"  {label}: {v['median']:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
