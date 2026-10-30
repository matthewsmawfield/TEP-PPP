#!/usr/bin/env python3
"""
STEP 103 -- Phase-aligned oscillator dependence and independent gradients.

The TEP signal is in the PHASE, not the amplitude.  Step 53 showed that the
phase alignment index cos(arg(sum_f conj(Fi) * Fj)) gives lambda ~ 4200 km
with R^2 ~ 0.90 and z ~ 27 vs the phase-randomised null on independent COD
and GFZ products.  Amplitude-based metrics (resid correlation, excess power)
fail because GNSS processing suppresses spatially correlated amplitudes when
phases are aligned (Paper 1, Sec 2.3).

This step applies the phase alignment index to the oscillator-dependence test:
if TEP couples to quartz but not atomic (Step 65 screening), quartz clock
pairs should show higher phase alignment at the TEP scale than atomic clock
pairs.  This is the correct observable for the TEP prediction that
conventional physics did not make.

Protocol (follows Step 53 / Paper 1):
  * Linear detrend (degree 1) only; band selected in frequency domain
  * Phase alignment index = cos(arg(sum_f conj(Fi) * Fj)) over 10-500 uHz
  * 40 logarithmic distance bins over 50-13,000 km
  * Fit weighted by pair count; AIC model selection
  * Nulls: per-station Fourier phase randomisation

The test: compute the phase alignment decay curve separately for quartz-quartz
and atomic-atomic pairs.  The TEP prediction is that quartz shows a clear
Gaussian decay at ~4200 km (high R^2, large z vs null) while atomic does not
(low R^2, small z vs null).  The difference is the differential field coupling
that conventional physics does not predict.

Output: results/outputs/step_103_atomic_quartz_gradients.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.lib import tep_stats as ts  # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_103_atomic_quartz_gradients.json"
PRODUCT = "clk_cod"
C_M_PER_NS = 0.299792458
F_LO_HZ, F_HI_HZ = 10e-6, 500e-6
RESID_STD_MAX_NS = 20.0
DMIN_KM, DMAX_KM = 50.0, 13000.0
N_BINS = 40
MIN_PAIRS_PER_BIN = 20
EDGES = np.geomspace(DMIN_KM, DMAX_KM, N_BINS + 1)
TEP_BAND_KM = (1000.0, 5000.0)
N_DAYS = int(os.environ.get("TEP_AQ_DAYS", "30"))
N_NULL = 200
RNG = np.random.default_rng(20260903)


def _clock_types():
    ct = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    return {n: v[0] for n, v in ct.items()}


def _spectra(series, phase_rng=None):
    """Per-station in-band complex spectrum on a shared frequency grid.

    Paper 1: linear detrend (degree 1) only; band selected in frequency domain.
    """
    dts = []
    for n in list(series)[:8]:
        t = np.asarray(series[n]["t"], float)
        if len(t) > 4:
            d = np.median(np.diff(t))
            if np.isfinite(d) and d > 0:
                dts.append(d)
    dt = float(np.median(dts)) if dts else 30.0
    n_grid = max(64, min(int(round(86400.0 / dt)), 2880))
    grid = np.arange(n_grid) * dt
    freq = np.fft.rfftfreq(n_grid, d=dt)
    band = (freq >= F_LO_HZ) & (freq <= F_HI_HZ)
    if band.sum() < 6:
        return {}, {}

    RF, STA = {}, {}
    for n in series:
        # Station QC: reject clock-jump stations
        x_raw = np.asarray(series[n]["resid"], float)
        if np.nanstd(x_raw) > RESID_STD_MAX_NS:
            continue
        t = np.asarray(series[n]["t"], float)
        x = x_raw
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 0.5 * n_grid:
            continue
        t, x = t[ok], x[ok]
        xr = np.interp(grid, t - t[0], x, left=x[0], right=x[-1])
        tt = np.arange(len(xr), dtype=float)
        xr = xr - np.polyval(np.polyfit(tt, xr, 1), tt)
        if not np.all(np.isfinite(xr)) or np.std(xr) < 1e-12:
            continue
        F = np.fft.rfft(xr)[band]
        if F.size < 6 or not np.all(np.isfinite(F)):
            continue
        if phase_rng is not None:
            F = np.abs(F) * np.exp(1j * phase_rng.uniform(-np.pi, np.pi, F.shape))
        RF[n] = F
        STA[n] = series[n]["station"]
    return RF, STA


def _pairs_by_type(RF, STA, ctype):
    """Phase alignment index per station pair, split by clock type.

    Returns (quartz_pairs, atomic_pairs) where each is a list of
    (distance_km, phase_alignment_index).
    """
    nm = list(RF)
    quartz_pairs = []
    atomic_pairs = []
    for i, j in itertools.combinations(range(len(nm)), 2):
        ni, nj = nm[i], nm[j]
        ti = ctype.get(ni, "?")
        tj = ctype.get(nj, "?")
        Fi, Fj = RF[ni], RF[nj]
        if Fi.shape != Fj.shape:
            continue
        d = td.great_circle_km(STA[ni], STA[nj])
        if not (DMIN_KM <= d <= DMAX_KM):
            continue
        Ssum = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(Ssum) or Ssum == 0:
            continue
        pai = float(np.cos(np.angle(Ssum)))
        if ti == "QUARTZ" and tj == "QUARTZ":
            quartz_pairs.append((d, pai))
        elif ti == "ATOMIC" and tj == "ATOMIC":
            atomic_pairs.append((d, pai))
    return quartz_pairs, atomic_pairs


def _curve(pairs):
    if not pairs:
        return np.array([]), np.array([]), np.array([])
    d = np.array([p[0] for p in pairs], float)
    v = np.array([p[1] for p in pairs], float)
    idx = np.clip(np.digitize(d, EDGES) - 1, 0, N_BINS - 1)
    ctr, val, cnt = [], [], []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() < MIN_PAIRS_PER_BIN:
            continue
        ctr.append(float(np.sqrt(EDGES[b] * EDGES[b + 1])))
        val.append(float(np.mean(v[m])))
        cnt.append(int(m.sum()))
    return np.array(ctr), np.array(val), np.array(cnt)


def _fit(ctr, val, cnt):
    """Fit the model family and report each model's parameters HONESTLY.

    Only the exponential and Gaussian models have a length scale ``lambda``;
    the power-law model's second parameter is the exponent ``alpha`` (no
    characteristic length scale), and the constant model has only an
    intercept.  Reporting ``p[1]`` as ``lambda_km`` for those models
    published the power-law exponent (alpha ~ 2.6e-4) as a bogus "0.00026 km
    length scale" for the atomic arm.  Each model now reports only the
    parameters it actually has.
    """
    if len(ctr) < 5:
        return {}
    m = ts.fit_all_models(ctr, val, weight=cnt.astype(float))
    best = m.get("best_model")
    models = m.get("models") or {}
    out = {"best_model": best,
           "selection_criterion": m.get("selection_criterion"),
           "delta_aic_exponential_vs_best": m.get("delta_aic_exponential_vs_best")}
    for k, mm in models.items():
        p = mm.get("params") or []
        entry = {"R2": float(mm.get("R2", np.nan)),
                 "AIC": float(mm.get("AIC", np.nan))}
        if k in ("exponential", "gaussian") and len(p) >= 2:
            # params = [A, lambda, C0] -- a true length scale in km
            entry["lambda_km"] = float(p[1])
        elif k == "power_law" and len(p) >= 2:
            # params = [A, alpha, C0] -- exponent, NOT a length scale
            entry["alpha"] = float(p[1])
            entry["lambda_km"] = None
        else:
            entry["lambda_km"] = None
        out[k] = entry
    bm = out.get(best) or {}
    out["lambda_km"] = bm.get("lambda_km")
    out["R2"] = bm.get("R2")
    # Report the power-law exponent at the top level too when it is the
    # selected model, so the atomic arm's "no characteristic length scale"
    # result is explicit rather than a missing field.
    if best == "power_law":
        out["alpha"] = bm.get("alpha")
    return out


def _run_null(series, ctype, clock_type, n_null=N_NULL):
    """Phase-randomised null for one clock type."""
    null_R2 = []
    null_lambda = []
    for _ in range(n_null):
        rng = np.random.default_rng(RNG.integers(0, 2**31))
        RF, STA = _spectra(series, phase_rng=rng)
        if len(RF) < 8:
            continue
        q, a = _pairs_by_type(RF, STA, ctype)
        pairs = q if clock_type == "QUARTZ" else a
        if len(pairs) < 100:
            continue
        ctr, val, cnt = _curve(pairs)
        fit = _fit(ctr, val, cnt)
        if "R2" in fit and fit["R2"] is not None and np.isfinite(fit["R2"]):
            null_R2.append(fit["R2"])
            if fit.get("lambda_km") and np.isfinite(fit["lambda_km"]):
                null_lambda.append(fit["lambda_km"])
    return np.array(null_R2), np.array(null_lambda)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctype = _clock_types()
    dates = td.available_dates(PRODUCT, 2024)
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_DAYS).astype(int)]
    print(f"STEP 103  |  phase-aligned oscillator dependence | {len(days)} days")

    all_q_pairs = []
    all_a_pairs = []
    for k, tag in enumerate(days):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        if not ser or len(ser) < 20:
            continue
        RF, STA = _spectra(ser)
        if len(RF) < 12:
            continue
        q, a = _pairs_by_type(RF, STA, ctype)
        all_q_pairs.extend(q)
        all_a_pairs.extend(a)
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{len(days)} days  Q pairs={len(all_q_pairs)}  A pairs={len(all_a_pairs)}",
                  flush=True)

    # ---- quartz phase alignment curve ----
    q_ctr, q_val, q_cnt = _curve(all_q_pairs)
    q_fit = _fit(q_ctr, q_val, q_cnt)

    # ---- atomic phase alignment curve ----
    a_ctr, a_val, a_cnt = _curve(all_a_pairs)
    a_fit = _fit(a_ctr, a_val, a_cnt)

    # ---- null: phase randomisation (pooled across ALL days, like Step 53) ----
    # Pool all null pairs across all days before fitting, so the null has
    # the same pair count as the real data.  This gives a stable null R2
    # distribution (per-day nulls are noisy because each day has only
    # ~2000 quartz pairs and ~5-15 distance bins).
    N_NULL_POOL = 200
    q_null_R2 = []
    a_null_R2 = []
    for i_null in range(N_NULL_POOL):
        all_q_null = []
        all_a_null = []
        for tag in days:
            try:
                ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
            except Exception:
                continue
            if not ser:
                continue
            rng = np.random.default_rng(RNG.integers(0, 2**31))
            RF, STA = _spectra(ser, phase_rng=rng)
            if len(RF) < 12:
                continue
            q, a = _pairs_by_type(RF, STA, ctype)
            all_q_null.extend(q)
            all_a_null.extend(a)
        if len(all_q_null) > 500:
            ctr, val, cnt = _curve(all_q_null)
            fit = _fit(ctr, val, cnt)
            if fit.get("R2") is not None and np.isfinite(fit["R2"]):
                q_null_R2.append(fit["R2"])
        if len(all_a_null) > 100:
            ctr, val, cnt = _curve(all_a_null)
            fit = _fit(ctr, val, cnt)
            if fit.get("R2") is not None and np.isfinite(fit["R2"]):
                a_null_R2.append(fit["R2"])
        if (i_null + 1) % 50 == 0:
            print(f"    null {i_null+1}/{N_NULL_POOL}", flush=True)

    q_null_R2 = np.array(q_null_R2)
    a_null_R2 = np.array(a_null_R2)

    # z-scores.  When the null R2 distribution is degenerate (std ~ 0, i.e.
    # every phase-randomised realisation gives the same R2 -- typically 0 for
    # the atomic arm whose binned phase-alignment means collapse to ~0 under
    # the null), the z-score is UNDEFINED, not "infinite".  Capping it at 100
    # published a misleading "100 sigma" that was really a divide-by-tiny-
    # constant.  Report z = None in that case and let the p-value (which is
    # well-defined: no null realisation reached the observed R2) carry the
    # significance statement.
    q_R2 = q_fit.get("R2")
    a_R2 = a_fit.get("R2")
    q_null_std = float(np.std(q_null_R2)) if len(q_null_R2) > 5 else 0.0
    a_null_std = float(np.std(a_null_R2)) if len(a_null_R2) > 5 else 0.0
    NULL_STD_FLOOR = 1e-6   # below this the null distribution is degenerate

    def _zscore(obs, null_arr, null_std):
        if len(null_arr) <= 5 or obs is None or not np.isfinite(obs):
            return None
        if null_std < NULL_STD_FLOOR:
            # Degenerate null: every realisation identical.  The z-score is
            # undefined; significance is carried by the p-value alone.
            return None
        return float((obs - np.mean(null_arr)) / null_std)

    q_z = _zscore(q_R2, q_null_R2, q_null_std)
    a_z = _zscore(a_R2, a_null_R2, a_null_std)
    q_p = float((np.sum(q_null_R2 >= q_R2) + 1) / (len(q_null_R2) + 1)) if len(q_null_R2) > 5 and q_R2 else None
    a_p = float((np.sum(a_null_R2 >= a_R2) + 1) / (len(a_null_R2) + 1)) if len(a_null_R2) > 5 and a_R2 else None

    # ---- TEP-scale phase alignment comparison ----
    # Average phase alignment in the TEP distance band (1000-5000 km)
    q_tep = [pai for d, pai in all_q_pairs if TEP_BAND_KM[0] <= d <= TEP_BAND_KM[1]]
    a_tep = [pai for d, pai in all_a_pairs if TEP_BAND_KM[0] <= d <= TEP_BAND_KM[1]]

    result = {
        "quartz": {
            "n_pairs": len(all_q_pairs),
            "lambda_km": q_fit.get("lambda_km"),
            "R2": q_R2,
            "best_model": q_fit.get("best_model"),
            "z_vs_phase_null": q_z,
            "p_vs_phase_null": q_p,
            "null_R2_mean": float(np.mean(q_null_R2)) if len(q_null_R2) else None,
            "null_R2_std": float(np.std(q_null_R2)) if len(q_null_R2) else None,
            "n_null": len(q_null_R2),
            "mean_phase_alignment_tep_band": float(np.mean(q_tep)) if q_tep else None,
            "n_pairs_tep_band": len(q_tep),
        },
        "atomic": {
            "n_pairs": len(all_a_pairs),
            "lambda_km": a_fit.get("lambda_km"),
            "R2": a_R2,
            "best_model": a_fit.get("best_model"),
            "z_vs_phase_null": a_z,
            "p_vs_phase_null": a_p,
            "null_R2_mean": float(np.mean(a_null_R2)) if len(a_null_R2) else None,
            "null_R2_std": float(np.std(a_null_R2)) if len(a_null_R2) else None,
            "n_null": len(a_null_R2),
            "mean_phase_alignment_tep_band": float(np.mean(a_tep)) if a_tep else None,
            "n_pairs_tep_band": len(a_tep),
        },
        "tep_scale_comparison": {
            "quartz_mean_pa_tep": float(np.mean(q_tep)) if q_tep else None,
            "atomic_mean_pa_tep": float(np.mean(a_tep)) if a_tep else None,
            "difference_q_minus_a": (float(np.mean(q_tep) - np.mean(a_tep))
                                     if q_tep and a_tep else None),
        },
    }

    # ---- independent gradients (from band power) ----
    # (kept from previous version for the localization ambiguity section)
    grads = []
    n_tri = 0
    for tag in days[:min(10, len(days))]:
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        if not ser:
            continue
        # Compute band powers for gradient estimation
        dts = []
        for n in list(ser)[:8]:
            t = np.asarray(ser[n]["t"], float)
            if len(t) > 4:
                d = np.median(np.diff(t))
                if np.isfinite(d) and d > 0:
                    dts.append(d)
        dt = float(np.median(dts)) if dts else 300.0
        n_grid = max(64, min(int(round(86400.0 / dt)), 2880))
        grid = np.arange(n_grid) * dt
        freq = np.fft.rfftfreq(n_grid, d=dt)
        df = freq[1] - freq[0] if len(freq) > 1 else 1.0 / (n_grid * dt)
        tep_mask = (freq >= F_LO_HZ) & (freq <= F_HI_HZ)
        if tep_mask.sum() < 6:
            continue
        names = list(ser)
        sta = {n: ser[n]["station"] for n in names}
        powers = {}
        for n in names:
            x_raw = np.asarray(ser[n]["resid"], float)
            if np.nanstd(x_raw) > RESID_STD_MAX_NS:
                continue
            t = np.asarray(ser[n]["t"], float)
            x = x_raw * C_M_PER_NS
            ok = np.isfinite(t) & np.isfinite(x)
            if ok.sum() < 0.5 * n_grid:
                continue
            t, x = t[ok], x[ok]
            xr = np.interp(grid, t - t[0], x, left=x[0], right=x[-1])
            tt = np.arange(len(xr), dtype=float)
            xr = xr - np.polyval(np.polyfit(tt, xr, 1), tt)
            if np.std(xr) < 1e-12:
                continue
            F = np.fft.rfft(xr)
            psd = (np.abs(F) ** 2) / (n_grid * dt)
            powers[n] = float(np.sum(psd[tep_mask]) * df)
        cand = [n for n in names if n in powers and ctype.get(n, "?") in ("ATOMIC", "QUARTZ")]
        if len(cand) < 3:
            continue
        tri_count = 0
        for a, b, c in itertools.combinations(cand, 3):
            d_ab = td.great_circle_km(sta[a], sta[b])
            d_ac = td.great_circle_km(sta[a], sta[c])
            d_bc = td.great_circle_km(sta[b], sta[c])
            if not (100 <= d_ab < 2000 and 100 <= d_ac < 2000 and 100 <= d_bc < 2000):
                continue
            # Simple gradient from band powers
            def enu(s_ref, s):
                d = td.great_circle_km(s_ref, s)
                az = np.radians(td.baseline_azimuth_midpoint_deg(s_ref, s))
                return d * np.cos(az), d * np.sin(az)
            e_b, n_b = enu(sta[a], sta[b])
            e_c, n_c = enu(sta[a], sta[c])
            Aa = float(np.sqrt(max(powers[a], 0)))
            Ab = float(np.sqrt(max(powers[b], 0)))
            Ac = float(np.sqrt(max(powers[c], 0)))
            if min(Aa, Ab, Ac) < 1e-9:
                continue
            M = np.array([[e_b, n_b], [e_c, n_c]])
            rhs = np.array([Ab - Aa, Ac - Aa])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < 1e-6:
                continue
            ge, gn = np.linalg.solve(M, rhs)
            grads.append(float(np.hypot(ge, gn)))
            tri_count += 1
            n_tri += 1
            if tri_count >= 8:
                break

    gmag = np.array(grads) if grads else np.array([])
    independent_gradients = {
        "n_triplets": int(n_tri),
        "gradient_magnitude_m_per_km": {
            "n": len(gmag),
            "median": float(np.median(gmag)) if len(gmag) else None,
        } if len(gmag) else {"n": 0},
    }

    localization = {
        "single_scalar_constraints": 1,
        "single_scalar_ambiguity_dim": 2,
        "independent_gradients_available": int(n_tri),
        "gradients_needed_for_3d": 2,
        "sufficient_for_3d": bool(n_tri >= 2),
    }

    payload = {
        "step": 103,
        "title": "Phase-aligned oscillator dependence and independent gradients",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": len(days),
        "observable": (
            "Phase alignment index cos(arg(sum_f conj(Fi) * Fj)) over "
            "10-500 uHz (Paper 1 / Step 53).  This is the corrected phase "
            "observable that gives lambda ~ 4200 km with R^2 ~ 0.90 and "
            "z ~ 27 vs the phase-randomised null on independent products.  "
            "Amplitude-based metrics fail because GNSS processing suppresses "
            "spatially correlated amplitudes when phases are aligned."),
        "result": result,
        "independent_gradients": independent_gradients,
        "localization": localization,
        "interpretation": (
            "The phase alignment index is computed separately for quartz-"
            "quartz and atomic-atomic pairs.  If TEP couples to quartz but "
            "not atomic (Step 65 screening), quartz pairs should show a "
            "clear Gaussian decay at ~4200 km (high R^2, large z vs the "
            "phase-randomised null) while atomic pairs should not (low R^2, "
            "small z).  The difference is the differential field coupling "
            "that conventional physics does not predict: the atomic "
            "transition screens the field, so atomic clocks do not develop "
            "the phase-coherent spatial structure that quartz clocks do."),
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"\n  QUARTZ: n={len(all_q_pairs)}, lambda={q_fit.get('lambda_km')}, "
          f"R2={q_R2}, z={q_z}")
    print(f"  ATOMIC: n={len(all_a_pairs)}, lambda={a_fit.get('lambda_km')}, "
          f"R2={a_R2}, z={a_z}")
    if q_tep and a_tep:
        print(f"  TEP band (1000-5000 km): Q PA={np.mean(q_tep):+.4f}, "
              f"A PA={np.mean(a_tep):+.4f}, diff={np.mean(q_tep)-np.mean(a_tep):+.4f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
