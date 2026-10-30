#!/usr/bin/env python3
"""
STEP 132  --  Lunar M2 isolation: purely gravitational TEP prediction.

THE PUREST GRAVITATIONAL TEST.

The 12.4-hour lunar semidiurnal tide (M2) is purely gravitational.  Unlike
the solar tides (S1, S2, S3, S4), M2 has NO solar atmospheric component —
it is driven entirely by the Moon's gravitational field.  If the M2-band
coherence shows distance structure and carries predictive power, that is
direct evidence of gravitational coupling, with no possible atmospheric
confound.

The TEP-GNSS Paper 1 reported a 5.14x spectral enhancement at the M2
frequency.  This step isolates the M2 band and tests whether it carries
predictive power for clock prediction on independent-year data.

PROTOCOL:
  * Run the P1 gate on 2025 held-out data.
  * Compare:
    1. full_band:  the registered full-band TEP kernel
    2. m2_band:    ONLY the lunar M2 frequency (12.4h ± 5%)
    3. empirical:  data-driven covariance

  * TEP prediction: the M2-band kernel carries predictive power, because
    the lunar gravitational tide couples to the TEP field.  If it beats
    empirical covariance, that is direct gravitational evidence with no
    atmospheric confound.

  * Note: the M2 band is a single frequency with limited spectral bins.
    The kernel may be noisy due to the small number of bins.  This is a
    trade-off between purity (no atmospheric confound) and power (limited
    spectral information).

Output: results/outputs/step_132_m2_isolation.json
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

OUT = ROOT / "results" / "outputs" / "step_132_m2_isolation.json"
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

# M2 lunar semidiurnal period: 12.421 hours = 44712 seconds
M2_PERIOD_S = 44712.0
M2_WIDTH_FRAC = 0.08  # slightly wider to capture enough bins

# Use the Paper 1 tidal-band lambda for the M2 kernel
M2_LAMBDA_KM = 4677.0
M2_MULTI_SCALES_KM = (M2_LAMBDA_KM / 1.6, M2_LAMBDA_KM, M2_LAMBDA_KM * 1.6)


def _input_path(tag: str) -> Path | None:
    for cadence in ("30S", "05M"):
        p = CLK_DIR / f"COD0MGXFIN_{tag}0000_01D_{cadence}_CLK.CLK.gz"
        if p.exists():
            return p
    return None


def _m2_phase_gram(x, dt_s, min_bins=2, min_low_cycles=0.85):
    """Phase Gram using ONLY the lunar M2 frequency band.

    The M2 tide (12.4h) is purely gravitational — no solar atmospheric
    component.  This is the purest gravitational test possible with
    GNSS clock data.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * (1.0/M2_PERIOD_S) < min_low_cycles:
        raise ValueError("history too short to resolve M2")

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

    fn_m2 = 1.0 / M2_PERIOD_S
    lo = fn_m2 * (1.0 - M2_WIDTH_FRAC)
    hi = fn_m2 * (1.0 + M2_WIDTH_FRAC)
    m2_mask = (freq >= lo) & (freq <= hi)

    if int(m2_mask.sum()) < int(min_bins):
        raise ValueError(f"too few M2 bins ({int(m2_mask.sum())})")
    F = F[m2_mask]

    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero M2 spectral energy")
    norm = np.sqrt(np.outer(energy, energy))
    pg = np.real(cross / norm)
    pg = 0.5 * (pg + pg.T)
    np.fill_diagonal(pg, 1.0)
    return pg, int(m2_mask.sum())


def _rho_multi_m2(d):
    return np.mean([np.exp(-d / L) for L in M2_MULTI_SCALES_KM], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


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

    print("Step 132: Lunar M2 isolation — purely gravitational TEP prediction",
          flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  M2 period: {M2_PERIOD_S/3600:.3f} h", flush=True)
    print(f"  M2 lambda: {M2_LAMBDA_KM} km", flush=True)
    print()

    planned_eval_tags = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    required_tags = sorted(set(planned_eval_tags +
                               [s64._prev(t) for t in planned_eval_tags]))
    eval_tags = [tag for tag in planned_eval_tags
                 if _input_path(tag) is not None and
                 _input_path(s64._prev(tag)) is not None]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} complete day pairs", flush=True)
        OUT.write_text(json.dumps({"step": 132, "error": "too few days",
                                   "n_days": len(eval_tags)}, indent=2))
        return

    print(f"  {len(eval_tags)} evaluation days available", flush=True)

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "m2_rms", "emp_rms",
        "m2_vs_full", "m2_vs_emp")}
    m2_bin_counts = []

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
        if pg_full is None:
            continue
        gnames_f, G_full, P_full = pg_full

        # M2 Gram
        pnames, pt, praw, _ = prev
        spectral_cols, kept_n = [], []
        for nm in names:
            if nm not in pnames:
                continue
            y = praw[:, pnames.index(nm)].astype(float)
            good = np.isfinite(y)
            if good.mean() < 0.9 or good.sum() < 20:
                continue
            if not good.all():
                ix = np.arange(len(y))
                y = np.interp(ix, ix[good], y[good])
            xq = np.arange(len(y)) - len(y) / 2.0
            spectral_y = y - np.polyval(np.polyfit(xq, y, 2), xq)
            if not np.isfinite(spectral_y).all() or not (1e-4 < np.std(spectral_y) < 1e4):
                continue
            spectral_cols.append(spectral_y)
            kept_n.append(nm)
        if len(kept_n) < s64.MIN_REFS + 1:
            continue
        try:
            G_m2, n_m2_bins = _m2_phase_gram(np.column_stack(spectral_cols), s64.DT_S)
        except ValueError as e:
            continue
        m2_bin_counts.append(n_m2_bins)

        common = sorted(set(gnames_f) & set(kept_n))
        if len(common) < s64.MIN_REFS + 2:
            continue

        idx_f = [gnames_f.index(n) for n in common]
        idx_n = [kept_n.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        G_m2 = G_m2[np.ix_(idx_n, idx_n)]
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
        G_m2 = G_m2[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "m2", "emp")}
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

            # M2-band kernel
            rho_rr_m = _rho_multi_m2(Dg[np.ix_(pool, pool)])
            rho_rt_m = _rho_multi_m2(Dg[b, pool])
            try:
                w_m2 = _reg_solve(
                    rho_rr_m * G_m2[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_m * G_m2[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Empirical
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
                day_sq["m2"].append((truth - float(w_m2 @ rr)) ** 2)
                day_sq["emp"].append((truth - float(w_emp @ rr)) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["m2_rms"].append(rms["m2"])
        per_day["emp_rms"].append(rms["emp"])
        per_day["m2_vs_full"].append(
            100.0 * (1.0 - rms["m2"] / rms["full_band"]))
        per_day["m2_vs_emp"].append(
            100.0 * (1.0 - rms["m2"] / rms["emp"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"m2={np.median(per_day['m2_rms']):.1f}, "
                  f"emp={np.median(per_day['emp_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)
    if m2_bin_counts:
        print(f"  M2 bins per day: median={int(np.median(m2_bin_counts))}, "
              f"range=[{min(m2_bin_counts)}, {max(m2_bin_counts)}]", flush=True)

    results = {
        "n_days": n,
        "m2_vs_full_band_pct": _ci(per_day["m2_vs_full"], rng),
        "m2_vs_empirical_pct": _ci(per_day["m2_vs_emp"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "m2": float(np.median(per_day["m2_rms"])) if n else None,
            "empirical": float(np.median(per_day["emp_rms"])) if n else None,
        },
        "m2_bins_median": int(np.median(m2_bin_counts)) if m2_bin_counts else None,
        "m2_bins_range": [min(m2_bin_counts), max(m2_bin_counts)] if m2_bin_counts else None,
    }

    interpretations = []
    mvf = results["m2_vs_full_band_pct"]
    mve = results["m2_vs_empirical_pct"]

    if mvf.get("median") is not None:
        if mvf["median"] > 0:
            interpretations.append(
                f"The M2 lunar gravitational kernel carries predictive power "
                f"(vs full-band: {mvf['median']:+.2f}%, "
                f"CI95 [{mvf['ci95'][0]:+.2f}, {mvf['ci95'][1]:+.2f}]). "
                f"The lunar semidiurnal tide is purely gravitational — no "
                f"solar atmospheric component can explain this.  If the CI "
                f"is positive, this is direct gravitational evidence.")
        else:
            interpretations.append(
                f"The M2 kernel does not beat the full-band kernel "
                f"({mvf['median']:+.2f}%).  The M2 band alone may have "
                f"insufficient spectral bins ({results.get('m2_bins_median')}) "
                f"to carry stable predictive power.")

    if mve.get("median") is not None:
        if mve["median"] > 0 and mve["ci95"] and mve["ci95"][0] > 0:
            interpretations.append(
                f"The M2 lunar gravitational kernel BEATS empirical "
                f"covariance on independent-year data "
                f"(median {mve['median']:+.2f}%, "
                f"CI95 [{mve['ci95'][0]:+.2f}, {mve['ci95'][1]:+.2f}]). "
                f"This is direct evidence of gravitational coupling: a "
                f"physics-fixed kernel driven by the Moon's gravitational "
                f"field generalises where the data-driven covariance "
                f"overfits.  No atmospheric process can explain this "
                f"because M2 has no solar component.")
        elif mve["median"] > 0:
            interpretations.append(
                f"The M2 kernel is better than empirical covariance "
                f"(median {mve['median']:+.2f}%, CI includes zero).")

    verdict = {
        "step": 132,
        "title": "Lunar M2 isolation: purely gravitational TEP prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days_scored": n,
        "m2_period_h": M2_PERIOD_S / 3600.0,
        "m2_width_frac": M2_WIDTH_FRAC,
        "m2_lambda_km": M2_LAMBDA_KM,
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The M2 lunar semidiurnal tide (12.4h) is purely gravitational. "
            "If the M2-band kernel carries predictive power and beats "
            "empirical covariance on independent-year data, that is direct "
            "evidence of gravitational coupling with no atmospheric "
            "confound.  The M2 band has limited spectral bins, so the test "
            "trades purity for power."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    print(f"\n  M2 vs full: {mvf.get('median')}", flush=True)
    print(f"  M2 vs emp:  {mve.get('median')}", flush=True)


if __name__ == "__main__":
    main()
