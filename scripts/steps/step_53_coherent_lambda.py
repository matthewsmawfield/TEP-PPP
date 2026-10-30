#!/usr/bin/env python3
"""
STEP 53 -- THE PAPERS 1-2 PHASE-ALIGNMENT INDEX, APPLIED CORRECTLY.

This step resolves what looked like a hard contradiction in this pipeline:

    Step 40 identifies lambda ~ 2,300 km on precise receiver clocks at
    15-32 sigma, while Steps 34/38/51/52 find the length scale that
    minimises held-out PREDICTION error on the same precise data is
    ~700 km, and that fixing lambda at the TEP value costs ~31%
    relative to predicting zero.

There is no contradiction.  Two separate estimator errors produced it.

ERROR 1 -- the wrong observable.
Every kriging step predicts the band-passed clock residual AMPLITUDE.  A
station's in-band residual is the coherent proper-time component PLUS local
oscillator/site noise, which is spatially uncorrelated.  On precise products
the satellite common-mode has been removed, so the local noise dominates the
variance; least-squares prediction of the total amplitude is therefore driven
almost entirely by the incoherent part, and the scale that minimises it is not
lambda_T but the nearest-neighbour amplitude-similarity scale.  Paper 1
(Sec. 2.3) states this directly: GNSS processing "suppresses spatially
correlated amplitudes precisely when phases are aligned", so amplitude-based
metrics "cannot detect the distance-structured phase relationships central
to TEP".

ERROR 2 -- the wrong statistic.
This repository's CSD steps (11b, 36) compute

    <coherence> x |magnitude-weighted mean phasor|,
    coherence = |Pxy|^2 / (Pxx Pyy)

which is (a) band-averaged coherency, the metric Paper 1 explicitly reports
as FAILING, times (b) the MAGNITUDE of the mean phasor -- a phase-CONCENTRATION
measure -- where Papers 1-2 use the COSINE OF ITS ANGLE.  Those are different
quantities: |wm| says how tightly phases cluster, cos(arg wm) says whether they
cluster around zero, which is what TEP predicts.  Summing Pxy across different
pairs in a bin also cancels, since two pairs at equal separation share no
cross-spectral phase.

Papers 1-2 define the phase-alignment index (Paper 1, "Key Terminology"):

    weighted_phase = np.angle(np.average(np.exp(1j*phases),
                                         weights=magnitudes))
    phase_alignment_index = np.cos(weighted_phase)

Since average(exp(i*phase), weights=|S|) = sum(S)/sum(|S|), this is exactly
cos(arg(sum_f S_ij(f))) over the 10-500 uHz band: one real number per station
pair, averaged over pairs within a distance bin, then fitted.

Protocol follows Paper 1:
  * linear detrend only (np.polyfit degree 1) before the CSD; the band is
    selected in the frequency domain, not by filtering the time series;
  * 40 logarithmic distance bins over 50-13,000 km;
  * fit weighted by pair count per bin; AIC model selection.

Nulls: per-station Fourier phase randomisation (preserves each station's power
spectrum exactly, destroys cross-station phase relations) and distance
scrambling (destroys the geometry-coherence link).

Output: results/outputs/step_53_coherent_lambda.json
"""
from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td            # noqa: E402
from scripts.lib import tep_stats as ts           # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"

F_LO_HZ, F_HI_HZ = 10e-6, 500e-6          # Papers 1-2 band
DMIN_KM, DMAX_KM = 50.0, 13000.0          # Paper 1 Step 2.0
N_BINS = 40                                # Paper 1 Step 2.0
MIN_PAIRS_PER_BIN = 20
EDGES = np.geomspace(DMIN_KM, DMAX_KM, N_BINS + 1)
TEP_BAND_KM = (1000.0, 5000.0)


def _spectra(series, phase_rng=None):
    """Per-station in-band complex spectrum on a shared frequency grid.

    Paper 1: linear detrend (degree 1) only; the 10-500 uHz restriction is
    applied by selecting rfft bins, NOT by filtering the time series.
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
        t = np.asarray(series[n]["t"], float)
        x = np.asarray(series[n]["resid"], float)
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


def _pairs(RF, STA):
    """(distance_km, phase_alignment_index) per station pair.

    phase_alignment_index = cos(arg(sum_f S_ij(f)))   [Papers 1-2]
    """
    nm = list(RF)
    out = []
    for i, j in itertools.combinations(range(len(nm)), 2):
        Fi, Fj = RF[nm[i]], RF[nm[j]]
        if Fi.shape != Fj.shape:
            continue
        d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
        if not (DMIN_KM <= d <= DMAX_KM):
            continue
        Ssum = np.sum(np.conj(Fi) * Fj)
        if not np.isfinite(Ssum) or Ssum == 0:
            continue
        out.append((d, float(np.cos(np.angle(Ssum)))))
    return out


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
    """AIC model selection, fit weighted by pair count (Paper 1)."""
    if len(ctr) < 5:
        return {}
    m = ts.fit_all_models(ctr, val, weight=cnt.astype(float))
    best = m.get("best_model")
    models = m.get("models") or {}
    out = {"best_model": best,
           "selection_criterion": m.get("selection_criterion"),
           "delta_aic_exponential_vs_best": m.get("delta_aic_exponential_vs_best"),
           "delta_aicc_exponential_vs_best": m.get("delta_aicc_exponential_vs_best")}
    for k, mm in models.items():
        p = mm.get("params") or []
        out[k] = {"lambda_km": (float(p[1]) if len(p) > 1 else None),
                  "R2": float(mm.get("R2", np.nan)),
                  "AIC": float(mm.get("AIC", np.nan)),
                  "AICc": float(mm.get("AICc", np.nan))}
    bm = out.get(best) or {}
    out["lambda_km"] = bm.get("lambda_km")
    out["R2"] = bm.get("R2")
    return out


def run(product, date_tags, label, n_null=60):
    days = [s for _, s in td.load_multi_day(product, date_tags) if len(s) >= 8]
    if not days:
        return {"label": label, "error": "no days with >= 8 stations"}

    pairs = []
    for s in days:
        RF, STA = _spectra(s)
        if len(RF) >= 8:
            pairs.extend(_pairs(RF, STA))
    if len(pairs) < 500:
        return {"label": label, "error": f"only {len(pairs)} pairs"}

    ctr, val, cnt = _curve(pairs)
    real = _fit(ctr, val, cnt)

    # --- null 1: per-station Fourier phase randomisation -------------------
    null_r2, null_lam = [], []
    for i in range(n_null):
        rng = np.random.default_rng(70_000 + i)
        pn = []
        for s in days:
            RFn, STAn = _spectra(s, phase_rng=rng)
            if len(RFn) >= 8:
                pn.extend(_pairs(RFn, STAn))
        if len(pn) < 500:
            continue
        f2 = _fit(*_curve(pn))
        # NB: a null whose best model is "constant" has R2 EXACTLY 0.0, which is
        # the correct null outcome.  `x or np.nan` would silently discard it
        # (0.0 is falsy) and leave only the high-R2 nulls, biasing the null pool
        # upward and destroying the significance estimate.  Test for None.
        r2 = f2.get("R2")
        if r2 is not None and np.isfinite(r2):
            null_r2.append(float(r2))
        lm = f2.get("lambda_km")
        if lm is not None and np.isfinite(lm):
            null_lam.append(float(lm))

    # --- null 2: distance scrambling --------------------------------------
    scr_r2 = []
    d_all = np.array([p[0] for p in pairs], float)
    v_all = np.array([p[1] for p in pairs], float)
    for i in range(n_null):
        rng = np.random.default_rng(80_000 + i)
        f3 = _fit(*_curve(list(zip(d_all, v_all[rng.permutation(len(d_all))]))))
        r3 = f3.get("R2")
        if r3 is not None and np.isfinite(r3):
            scr_r2.append(float(r3))

    def _sig(nulls):
        """Sigma separation, plus an exceedance count that stays meaningful when
        the null pool is degenerate.

        A scrambled curve is usually fitted best by the CONSTANT model, whose
        R2 is exactly 0.  When every null returns 0 the pool has zero spread and
        (real - mean)/std explodes to ~1e11, which is not a significance -- it is
        a divide-by-zero.  Report sigma only when the null pool actually has
        spread, and always report how many nulls reached the observed R2."""
        r = real.get("R2")
        if not nulls or r is None:
            return {"sigma": None, "n_null": len(nulls), "n_exceeding": None,
                    "p_value": None, "null_R2_std": None}
        a = np.asarray(nulls, float)
        sd = float(np.std(a))
        n_exc = int(np.sum(a >= r))
        return {"sigma": (float((r - a.mean()) / sd) if sd > 1e-6 else None),
                "n_null": len(a), "n_exceeding": n_exc,
                "p_value": float((n_exc + 1) / (len(a) + 1)),
                "null_R2_std": sd,
                "note": (None if sd > 1e-6 else
                         "null pool degenerate (every null fitted best by the "
                         "constant model, R2=0); sigma undefined, use p_value")}

    lam = real.get("lambda_km")
    return {
        "label": label, "product": product, "n_days": len(days),
        "n_pairs": len(pairs), "n_bins": int(len(ctr)),
        "statistic": "cos(arg(sum_f S_ij(f))) -- Papers 1-2 phase-alignment index",
        "fit": real, "lambda_km": lam, "R2": real.get("R2"),
        "best_model": real.get("best_model"),
        "lambda_in_TEP_band": bool(lam is not None and
                                   TEP_BAND_KM[0] <= lam <= TEP_BAND_KM[1]),
        "null_phase_randomise": {
            "R2_mean": (float(np.mean(null_r2)) if null_r2 else None),
            "lambda_median": (float(np.median(null_lam)) if null_lam else None),
            **_sig(null_r2)},
        "null_distance_scramble": {
            "R2_mean": (float(np.mean(scr_r2)) if scr_r2 else None),
            **_sig(scr_r2)},
        "binned_curve": {"distance_km": ctr.tolist(),
                         "phase_alignment_index": val.tolist(),
                         "n_pairs": cnt.tolist()},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"test": "papers_1_2_phase_alignment_index",
           "definition": ("phase_alignment_index = cos(angle(average(exp(i*phase), "
                          "weights=|S|))) = cos(arg(sum_f S_ij(f))); 10-500 uHz; "
                          "linear detrend only; 40 log bins 50-13,000 km; "
                          "count-weighted AIC model selection"),
           "corrects": ("steps 11b/36 used <coherency> x |mean phasor| -- band-averaged "
                        "coherency (which Paper 1 reports as failing) times a phase-"
                        "concentration measure -- instead of the cosine of the mean "
                        "phase angle, and stacked Pxy across pairs, which cancels"),
           "timestamp": datetime.now().isoformat(), "arms": {}}

    # Two independent analysis centres (CODE, GFZ) with different clock-
    # combination software and station networks.  Paper 1 requires "replication
    # across independent processing chains ... with different systematic
    # vulnerabilities"; an AC-specific systematic would not reproduce the same
    # lambda across both.  200 nulls per test so the exceedance p-value is a
    # real bound, not the 1/(N+1) floor.
    cod = td.available_dates("clk_cod", 2024)
    if len(cod) >= 30:
        idx = np.linspace(0, len(cod) - 1, 30).round().astype(int)
        res["arms"]["precise_cod"] = run("clk_cod", [cod[i] for i in idx],
                                         "precise_cod", n_null=200)
    gfz = td.available_dates("clk_gfz", 2024)
    if len(gfz) >= 30:
        idx = np.linspace(0, len(gfz) - 1, 30).round().astype(int)
        res["arms"]["precise_gfz"] = run("clk_gfz", [gfz[i] for i in idx],
                                         "precise_gfz", n_null=200)
    res["arms"]["broadcast_spp"] = run(
        "broadcast", [f"2024{d:03d}" for d in range(100, 107)], "broadcast_spp",
        n_null=200)

    # Cross-AC consistency: an AC systematic would not give the same lambda.
    lams = {k: v.get("lambda_km") for k, v in res["arms"].items()
            if isinstance(v, dict) and v.get("lambda_km")}
    if len(lams) >= 2:
        vals = np.array(list(lams.values()), float)
        res["cross_arm_consistency"] = {
            "lambda_km_by_arm": lams,
            "spread_cv": float(np.std(vals) / np.mean(vals)),
            "all_in_TEP_band": bool(np.all((vals >= 1000.0) & (vals <= 5000.0))),
        }

    for name, a in res["arms"].items():
        if "error" in a:
            print(f"[{name}] {a['error']}")
            continue
        f = a["fit"]
        print(f"\n[{name}] {a['n_days']} days, {a['n_pairs']:,} pairs, {a['n_bins']} bins")
        print(f"  BEST: {a['best_model']}  lambda = {a['lambda_km']:.0f} km  "
              f"R2 = {a['R2']:.3f}   in TEP band: {a['lambda_in_TEP_band']}")
        for k in ("exponential", "gaussian"):
            if k in f and f[k].get("lambda_km") is not None:
                print(f"     {k:12s} lambda={f[k]['lambda_km']:8.0f} km  "
                      f"R2={f[k]['R2']:.3f}  AIC={f[k]['AIC']:.1f}")
        pr, ds = a["null_phase_randomise"], a["null_distance_scramble"]
        for tag, nn in (("phase-randomise  ", pr), ("distance-scramble", ds)):
            if nn.get("R2_mean") is None:
                continue
            sg = (f"{nn['sigma']:.1f} sigma" if nn.get("sigma") is not None
                  else "sigma undefined (degenerate null)")
            print(f"  null {tag}: R2 {nn['R2_mean']:.4f} -> {sg}, "
                  f"{nn['n_exceeding']}/{nn['n_null']} nulls reached the observed R2 "
                  f"(p = {nn['p_value']:.4f})")

    (OUT / "step_53_coherent_lambda.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved {OUT / 'step_53_coherent_lambda.json'}")


if __name__ == "__main__":
    main()
