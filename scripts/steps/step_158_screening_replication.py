#!/usr/bin/env python3
"""
STEP 158  --  Clock-type screening replication on 2025 independent data.

Step 103 showed that quartz clock pairs exhibit exponential distance decay
in phase alignment (R2=0.936, lambda~2454 km) while atomic pairs do not
(R2=0.309).  This is the TEP screening signature: the field couples to
quartz oscillators but not atomic.

This step replicates Step 103 on independent 2025 data.  If TEP is real,
the quartz exponential decay should replicate (high R2, lambda in the
TEP-predicted 2000-5000 km range) while atomic should not.

Output: results/outputs/step_158_screening_replication.json
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

OUT = ROOT / "results" / "outputs" / "step_158_screening_replication.json"
PRODUCT = "clk_cod"
YEAR = 2025
F_LO_HZ, F_HI_HZ = 10e-6, 500e-6
RESID_STD_MAX_NS = 20.0
DMIN_KM, DMAX_KM = 50.0, 13000.0
N_BINS = 40
MIN_PAIRS_PER_BIN = 20
EDGES = np.geomspace(DMIN_KM, DMAX_KM, N_BINS + 1)
N_DAYS = int(os.environ.get("TEP_S158_DAYS", "30"))
RNG = np.random.default_rng(20260907)


def _clock_types():
    ct = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    return {n: v[0] for n, v in ct.items()}


def _spectra(series, phase_rng=None):
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
    if len(ctr) < 5:
        return {}
    m = ts.fit_all_models(ctr, val, weight=cnt.astype(float))
    best = m.get("best_model")
    models = m.get("models") or {}
    out = {"best_model": best,
           "selection_criterion": m.get("selection_criterion")}
    for k, mm in models.items():
        p = mm.get("params") or []
        entry = {"R2": float(mm.get("R2", np.nan)),
                 "AIC": float(mm.get("AIC", np.nan))}
        if k in ("exponential", "gaussian") and len(p) >= 2:
            entry["lambda_km"] = float(p[1])
        elif k == "power_law" and len(p) >= 2:
            entry["alpha"] = float(p[1])
            entry["lambda_km"] = None
        else:
            entry["lambda_km"] = None
        out[k] = entry
    bm = out.get(best) or {}
    out["lambda_km"] = bm.get("lambda_km")
    out["R2"] = bm.get("R2")
    if best == "power_law":
        out["alpha"] = bm.get("alpha")
    return out


def _tep_band_pa(pairs):
    """Mean phase alignment for pairs in the TEP distance range (1000-5000 km)."""
    if not pairs:
        return None
    d = np.array([p[0] for p in pairs], float)
    v = np.array([p[1] for p in pairs], float)
    m = (d >= 1000.0) & (d <= 5000.0)
    if m.sum() < 10:
        return None
    return float(np.mean(v[m]))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctype = _clock_types()
    dates = td.available_dates(PRODUCT, YEAR)
    if not dates:
        print(f"  ERROR: no dates available for {PRODUCT} {YEAR}")
        return
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_DAYS).astype(int)]
    print(f"STEP 158  |  clock-type screening replication | {YEAR} | {len(days)} days")

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
            print(f"    {k+1}/{len(days)} days  Q pairs={len(all_q_pairs)}  "
                  f"A pairs={len(all_a_pairs)}", flush=True)

    q_ctr, q_val, q_cnt = _curve(all_q_pairs)
    q_fit = _fit(q_ctr, q_val, q_cnt)
    a_ctr, a_val, a_cnt = _curve(all_a_pairs)
    a_fit = _fit(a_ctr, a_val, a_cnt)

    q_pa_tep = _tep_band_pa(all_q_pairs)
    a_pa_tep = _tep_band_pa(all_a_pairs)

    # Load 2024 comparison from Step 103 output
    s103_path = ROOT / "results" / "outputs" / "step_103_atomic_quartz_gradients.json"
    comparison_2024 = {}
    if s103_path.exists():
        s103 = json.loads(s103_path.read_text())
        # Step 103 nests results under "result"
        s103_result = s103.get("result", s103)
        q103 = s103_result.get("quartz", {})
        a103 = s103_result.get("atomic", {})
        comparison_2024 = {
            "quartz": {
                "R2": q103.get("R2"),
                "lambda_km": q103.get("lambda_km"),
                "mean_pa_tep": q103.get("mean_phase_alignment_tep_band",
                                         q103.get("mean_pa_tep")),
                "n_pairs": q103.get("n_pairs"),
            },
            "atomic": {
                "R2": a103.get("R2"),
                "lambda_km": a103.get("lambda_km"),
                "mean_pa_tep": a103.get("mean_phase_alignment_tep_band",
                                         a103.get("mean_pa_tep")),
                "n_pairs": a103.get("n_pairs"),
            },
        }
        q103_pa = q103.get("mean_phase_alignment_tep_band",
                            q103.get("mean_pa_tep", 0)) or 0
        a103_pa = a103.get("mean_phase_alignment_tep_band",
                            a103.get("mean_pa_tep", 0)) or 0
        comparison_2024["quartz_atomic_ratio"] = (q103_pa / a103_pa
                                                   if a103_pa > 0 else None)

    q_R2 = q_fit.get("R2")
    a_R2 = a_fit.get("R2")
    q_ratio = (q_pa_tep / a_pa_tep) if (q_pa_tep and a_pa_tep and a_pa_tep > 0) else None

    # Phase-randomised null (reduced N_NULL for tractability)
    N_NULL_S158 = int(os.environ.get("TEP_S158_NULL", "50"))
    q_null_R2 = []
    a_null_R2 = []
    for i_null in range(N_NULL_S158):
        all_q_null = []
        all_a_null = []
        for tag in days:
            try:
                ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
            except Exception:
                continue
            if not ser:
                continue
            rng_null = np.random.default_rng(RNG.integers(0, 2**31))
            RF, STA = _spectra(ser, phase_rng=rng_null)
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
        if (i_null + 1) % 10 == 0:
            print(f"    null {i_null+1}/{N_NULL_S158}", flush=True)

    q_z = None; a_z = None
    q_p_null = None; a_p_null = None
    if q_R2 is not None and len(q_null_R2) > 5:
        q_null_arr = np.array(q_null_R2)
        q_z = float((q_R2 - q_null_arr.mean()) / q_null_arr.std()) if q_null_arr.std() > 0 else None
        q_p_null = float(np.mean(q_null_arr >= q_R2))
    if a_R2 is not None and len(a_null_R2) > 5:
        a_null_arr = np.array(a_null_R2)
        a_z = float((a_R2 - a_null_arr.mean()) / a_null_arr.std()) if a_null_arr.std() > 0 else None
        a_p_null = float(np.mean(a_null_arr >= a_R2))

    result = {
        "quartz": {
            "R2": q_R2,
            "lambda_km": q_fit.get("lambda_km"),
            "best_model": q_fit.get("best_model"),
            "mean_pa_tep": q_pa_tep,
            "n_pairs": len(all_q_pairs),
            "z_vs_phase_null": q_z,
            "p_vs_phase_null": q_p_null,
            "null_R2_mean": float(np.mean(q_null_R2)) if q_null_R2 else None,
            "null_R2_std": float(np.std(q_null_R2)) if q_null_R2 else None,
            "n_null": len(q_null_R2),
        },
        "atomic": {
            "R2": a_R2,
            "lambda_km": a_fit.get("lambda_km"),
            "best_model": a_fit.get("best_model"),
            "mean_pa_tep": a_pa_tep,
            "n_pairs": len(all_a_pairs),
            "z_vs_phase_null": a_z,
            "p_vs_phase_null": a_p_null,
            "null_R2_mean": float(np.mean(a_null_R2)) if a_null_R2 else None,
            "null_R2_std": float(np.std(a_null_R2)) if a_null_R2 else None,
            "n_null": len(a_null_R2),
        },
        "quartz_atomic_ratio": q_ratio,
    }

    interp = []
    interp.append(f"Clock-type screening replication on {YEAR} independent data.")
    if q_R2 is not None and comparison_2024.get("quartz", {}).get("R2") is not None:
        interp.append(f"Quartz R2={q_R2:.3f} (2024: {comparison_2024['quartz']['R2']:.3f}).")
    if q_fit.get("lambda_km") is not None and comparison_2024.get("quartz", {}).get("lambda_km"):
        interp.append(f"Quartz lambda={q_fit['lambda_km']:.0f} km "
                      f"(2024: {comparison_2024['quartz']['lambda_km']} km).")
    if a_R2 is not None and comparison_2024.get("atomic", {}).get("R2") is not None:
        interp.append(f"Atomic R2={a_R2:.3f} (2024: {comparison_2024['atomic']['R2']:.3f}).")
    if q_ratio is not None and comparison_2024.get("quartz_atomic_ratio") is not None:
        interp.append(f"Quartz/Atomic ratio={q_ratio:.2f}x "
                      f"(2024: {comparison_2024['quartz_atomic_ratio']:.2f}x).")
    interp.append(f"Quartz n_pairs={len(all_q_pairs)}, Atomic n_pairs={len(all_a_pairs)}.")
    if q_z is not None:
        interp.append(f"Quartz z vs phase-null: {q_z:.2f} (p={q_p_null:.4f}, n_null={len(q_null_R2)}).")
    if a_z is not None:
        interp.append(f"Atomic z vs phase-null: {a_z:.2f} (p={a_p_null:.4f}, n_null={len(a_null_R2)}).")

    if q_R2 is not None and a_R2 is not None:
        if q_R2 > 0.7 and a_R2 < 0.4:
            interp.append("The clock-type screening REPLICATES: quartz shows exponential "
                          "decay, atomic does not.")
        else:
            interp.append("The clock-type screening does NOT clearly replicate.")

    verdict = {
        "step": 158,
        "title": f"Step 103 replication on {YEAR} independent data",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "year": YEAR,
        "n_days": len(days),
        "result": result,
        "comparison_2024": comparison_2024,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If TEP is real, quartz clock pairs should show exponential distance decay "
            "(high R2) while atomic pairs should not (low R2). This should replicate "
            "on independent-year data."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"\n  Output: {OUT}")
    for line in interp:
        print(line)


if __name__ == "__main__":
    main()
