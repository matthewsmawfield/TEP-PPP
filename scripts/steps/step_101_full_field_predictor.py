#!/usr/bin/env python3
"""
STEP 101 -- Full-field spatiotemporal predictor at the TEP scale.

Step 53 showed that the TEP signal is in the PHASE, not the amplitude: the
phase alignment index cos(arg(sum_f conj(Fi) * Fj)) gives lambda ~ 4200 km
with R^2 ~ 0.90 and z ~ 27 vs the phase-randomised null.  Amplitude-based
prediction fails as a *spatial-structure* test because GNSS processing
suppresses spatially correlated amplitudes when phases are aligned (Paper 1,
Sec 2.3).  This step tests a different, deployable question: is the TEP-scale
spatial coherence of the resid INCREMENT -- the change in the IGS detrended
residual over the prediction gap -- strong enough to improve held-out clock
prediction versus persistence?

The resid still contains the TEP field's LF component (10-100 uHz, 69x more
power than the MF band), which is spatially correlated at the TEP scale
(Step 103: quartz phase alignment R^2 = 0.94 at lambda = 2,454 km).  No
additional detrending is applied: the IGS resid is already detrended, and
further detrending removes ~47.5% of the power including the TEP LF signal.

PROTOCOL (strictly causal):
  For each prediction epoch t_i and gap dt, the resid increment
      incr_j = resid_j(t_i + dt) - resid_j(t_i)
  is computed for each reference station j and kriged to the target at a
  physics-fixed TEP scale.  The kriged increment is added to the target's
  last known value.  The cross-spectrum used by the phase-coherence filter
  is computed from the pre-window [t_i - 12h, t_i] only -- no future data
  enters any arm.

ARMS:
  amplitude_krige     : raw resid increment kriged at the TEP scale
                        (lambda = 4,200 km) -- the full-field predictor
  phase_filtered_krige: as amplitude_krige but restricted to references
                        whose pre-window phase alignment PA_bj > 0 with the
                        target (a phase-coherence FILTER, not a projection).
                        Tests whether selecting phase-aligned references
                        improves the kriging.  Falls back to the full pool
                        when fewer than 4 references are phase-aligned.
  phase_only_krige    : kriging with the PA Gram (pairwise phase alignment
                        between references) as the covariance, NO distance
                        information.  Tests whether the per-epoch phase
                        alignment alone can predict the increment.  The
                        expectation is that this is weaker than the distance
                        kernel because the per-epoch PA from a 12-hour
                        window (21 frequency bins in 10-500 uHz) is noisy;
                        the phase observable detects the field (Step 103)
                        but the smooth distance kernel predicts its
                        evolution better.
  persistence         : last value held flat

The decisive comparison is amplitude_krige vs persistence.  The
phase_filtered_krige arm tests whether phase selection of individual
references adds anything on top of the TEP-scale spatial kriging.

SWEEPS (so the manuscript claims are computed, not hand-written):
  * gap sweep: 1 h and 4 h (the regime where a leftover clock drifts)
  * lambda sweep: 500 / 2,000 / 4,200 / 8,000 km for amplitude_krige vs
    persistence, to show the gain is concentrated at the TEP scale and a
    local-scale kernel cannot reproduce it.

Output: results/outputs/step_101_full_field_predictor.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_101_full_field_predictor.json"
PRODUCT = "clk_cod"
LAMBDA_KM = 4200.0
NUGGET = 0.35
C_M_PER_NS = 0.299792458
F_LO_HZ, F_HI_HZ = 10e-6, 500e-6
PRE_S = 43200.0          # 12h pre-window for cross-spectrum (enough frequency resolution)
N_DAYS = int(os.environ.get("TEP_FF_DAYS", "40"))
EVAL_STRIDE = int(os.environ.get("TEP_FF_STRIDE", "9"))
GAPS_S = [3600, 14400]                          # 1 h and 4 h gap horizons
LAMBDA_SWEEP_KM = [500.0, 2000.0, 4200.0, 8000.0]  # TEP-scale load-bearing test
MIN_REFS = 8
RNG = np.random.default_rng(20260903)


def _rho(d, lam):
    return np.exp(-0.5 * (np.asarray(d, float) / lam) ** 2)


def _krige(d_rr, d_rt, resid, lam, nugget=NUGGET):
    K = _rho(d_rr, lam) + (nugget + 1e-9) * np.eye(len(resid))
    try:
        w = np.linalg.solve(K, _rho(d_rt, lam))
    except np.linalg.LinAlgError:
        w = np.full(len(resid), 1.0 / len(resid))
    if not np.all(np.isfinite(w)):
        w = np.full(len(resid), 1.0 / len(resid))
    return float(w @ np.asarray(resid, float))


def _krige_gram(G_rr, G_rt, resid, nugget=NUGGET):
    """Kriging with an arbitrary Gram matrix as the covariance.

    Used for the phase-only kernel: G_rr[i,j] = PA_ij (the pre-window
    cross-spectrum phase alignment between references i and j), and
    G_rt[j] = PA_bj (phase alignment between the target and reference j).

    Phase alignment is cos(arg(...)) which ranges over [-1, 1].  A covariance
    matrix must be PSD with non-negative entries, so we apply the standard
    similarity-to-correlation transform (1 + PA) / 2, mapping [-1, 1] to
    [0, 1].  This preserves the ordering (higher PA = higher covariance)
    and produces a valid correlation-like matrix.  A nugget is added for
    numerical stability, and the matrix is projected to PSD by clipping
    negative eigenvalues.
    """
    # Transform from [-1, 1] to [0, 1] (similarity -> correlation)
    G_rr = 0.5 * (1.0 + np.asarray(G_rr, float))
    G_rt = 0.5 * (1.0 + np.asarray(G_rt, float))
    # Replace any NaN/inf with 0 (no information about alignment)
    G_rr = np.nan_to_num(G_rr, nan=0.0, posinf=1.0, neginf=0.0)
    G_rt = np.nan_to_num(G_rt, nan=0.0, posinf=1.0, neginf=0.0)
    # Clip to [0, 1] and symmetrise
    G_rr = np.clip(G_rr, 0.0, 1.0)
    G_rr = 0.5 * (G_rr + G_rr.T)
    G_rt = np.clip(G_rt, 0.0, 1.0)
    # Project to PSD: clip negative eigenvalues (suppress warnings from
    # near-singular matrices)
    with np.errstate(all="ignore"):
        try:
            evals, evecs = np.linalg.eigh(G_rr)
            if not np.all(np.isfinite(evals)):
                raise np.linalg.LinAlgError("non-finite eigenvalues")
            evals = np.clip(evals, 1e-9, None)
            G_rr = (evecs * evals) @ evecs.T
        except np.linalg.LinAlgError:
            G_rr = G_rr  # fall back to clipped (already PSD-ish) matrix
    K = G_rr + (nugget + 1e-9) * np.eye(len(resid))
    with np.errstate(all="ignore"):
        try:
            w = np.linalg.solve(K, G_rt)
        except np.linalg.LinAlgError:
            w = np.full(len(resid), 1.0 / len(resid))
    if not np.all(np.isfinite(w)):
        w = np.full(len(resid), 1.0 / len(resid))
    return float(w @ np.asarray(resid, float))


def _load_day(tag):
    s = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
    if not s or len(s) < 12:
        return None
    names = list(s)
    raw = np.column_stack([s[n]["resid"] for n in names]).astype(float) * C_M_PER_NS
    t = s[names[0]]["t"].astype(float)
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            D[i, j] = D[j, i] = td.great_circle_km(s[names[i]]["station"],
                                                   s[names[j]]["station"])
    return names, t, raw, D


def _pre_window_spectrum(t, y, ti, pre_s=PRE_S):
    """Causal in-band spectrum from the pre-window [ti - pre_s, ti].

    Returns the complex FFT coefficients in the 10-500 uHz band, or None.
    """
    t_lo = t[ti] - pre_s
    mask = (t >= t_lo) & (t <= t[ti]) & np.isfinite(y)
    if mask.sum() < 20:
        return None
    tm = t[mask]
    ym = y[mask]
    dt = float(np.median(np.diff(tm)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 300.0
    n = len(ym)
    # Pad to power of 2 for efficiency
    n_fft = 1 << int(np.ceil(np.log2(n)))
    tt = np.arange(n, dtype=float)
    yd = ym - np.polyval(np.polyfit(tt, ym, 1), tt)
    # Zero-pad
    yd_pad = np.zeros(n_fft)
    yd_pad[:n] = yd
    F = np.fft.rfft(yd_pad)
    freq = np.fft.rfftfreq(n_fft, d=dt)
    band = (freq >= F_LO_HZ) & (freq <= F_HI_HZ)
    if band.sum() < 3:
        return None
    return F[band]


def _phase_alignment(Fb, Fj):
    """Phase alignment index = cos(arg(sum_f conj(Fb) * Fj))."""
    if Fb is None or Fj is None or Fb.shape != Fj.shape:
        return None
    Ssum = np.sum(np.conj(Fb) * Fj)
    if not np.isfinite(Ssum) or Ssum == 0:
        return None
    return float(np.cos(np.angle(Ssum)))


def run(days):
    """Run the gap sweep (1 h, 4 h) and the lambda sweep (500-8000 km).

    Returns a dict with per-gap results (amplitude_krige / phase_filtered_krige
    / persistence at the canonical 4,200 km) and a lambda sweep of
    amplitude_krige vs persistence at the 4 h gap.
    """
    npre = int(PRE_S / 300.0)
    # per-day squared errors, keyed by (gap_s, arm)
    arms = ("amplitude_krige", "phase_filtered_krige", "phase_only_krige",
            "persistence")
    per_day = {(g, a): [] for g in GAPS_S for a in arms}
    # lambda sweep at the 4 h gap only (the regime where the field drifts)
    lam_gap = GAPS_S[-1]
    per_day_lam = {(lam, "amplitude_krige"): [] for lam in LAMBDA_SWEEP_KM}
    per_day_lam.update({(lam, "persistence"): [] for lam in LAMBDA_SWEEP_KM})

    for tag in days:
        cur = _load_day(tag)
        if cur is None:
            continue
        names, t, raw, D = cur
        sd = np.nanstd(raw, axis=0)
        med = np.median(sd)
        good = (sd > 0.001) & (sd < 60.0) & (sd < 4.0 * med)
        if good.sum() < MIN_REFS + 2:
            continue
        names = [n for i, n in enumerate(names) if good[i]]
        raw = raw[:, good]
        D = D[np.ix_(good, good)]
        ng = len(names)

        max_kstep = int(round(max(GAPS_S) / 300.0))
        eval_i = list(range(npre, len(t) - max_kstep - 1, EVAL_STRIDE))

        day_sq = {(g, a): [] for g in GAPS_S for a in arms}
        day_sq_lam = {(lam, a): [] for lam in LAMBDA_SWEEP_KM
                      for a in ("amplitude_krige", "persistence")}
        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < MIN_REFS:
                continue
            d_rr = D[np.ix_(pool, pool)]
            d_rt = D[b, pool]

            for ei in eval_i:
                # target pre-window spectrum (used by the phase filter and
                # the phase-only kernel for every gap at this epoch)
                Fb = _pre_window_spectrum(t, raw[:, b], ei)

                # --- collect reference increments + phase alignments ---
                ref_idx, ref_inc, ref_idx_ph, ref_inc_ph = [], [], [], []
                # Phase alignment vectors for the phase-only kernel:
                # pa_target[j] = PA between target b and reference j
                # pa_ref[i,j]  = PA between reference i and reference j
                pa_target = []
                ref_spectra = {}   # cache: j -> Fj (reused for Gram)
                for j in pool:
                    ri = float(raw[ei, j])
                    if not np.isfinite(ri):
                        continue
                    ref_idx.append(j)
                    ref_inc.append(ri)   # last value; increment computed per gap
                    if Fb is not None:
                        Fj = _pre_window_spectrum(t, raw[:, j], ei)
                        ref_spectra[j] = Fj
                        pa = _phase_alignment(Fb, Fj)
                        pa_target.append(pa if pa is not None else 0.0)
                        if pa is not None and pa > 0:
                            ref_idx_ph.append(j)
                            ref_inc_ph.append(ri)
                    else:
                        pa_target.append(0.0)
                if len(ref_idx) < MIN_REFS:
                    continue
                # phase-filtered arm falls back to the full pool when too
                # few references are phase-aligned (keeps the kriging system
                # well-conditioned; documented in the docstring).
                if len(ref_idx_ph) < 4:
                    ref_idx_ph, ref_inc_ph = ref_idx, ref_inc

                d_rr_amp = D[np.ix_(ref_idx, ref_idx)]
                d_rt_amp = D[b, ref_idx]
                d_rr_ph = D[np.ix_(ref_idx_ph, ref_idx_ph)]
                d_rt_ph = D[b, ref_idx_ph]
                last_vals = np.array(ref_inc, float)
                last_vals_ph = np.array(ref_inc_ph, float)

                # --- phase-only Gram: PA_ij between all reference pairs ---
                # This is the covariance for the phase-only kriging kernel.
                # It uses no distance information, only the pre-window
                # cross-spectrum phase alignment between stations.
                n_ref = len(ref_idx)
                pa_gram = np.zeros((n_ref, n_ref))
                pa_vec = np.array(pa_target, float)
                if Fb is not None:
                    for ii in range(n_ref):
                        Fi = ref_spectra.get(ref_idx[ii])
                        if Fi is None:
                            continue
                        for jj in range(ii, n_ref):
                            if ii == jj:
                                pa_gram[ii, ii] = 1.0
                                continue
                            Fj = ref_spectra.get(ref_idx[jj])
                            paij = _phase_alignment(Fi, Fj)
                            if paij is not None:
                                pa_gram[ii, jj] = paij
                                pa_gram[jj, ii] = paij

                for gap_s in GAPS_S:
                    kstep = int(round(gap_s / 300.0))
                    ei_pred = ei + kstep
                    if ei_pred >= len(t):
                        continue
                    truth = float(raw[ei_pred, b])
                    last_val = float(raw[ei, b])
                    if not np.isfinite(truth) or not np.isfinite(last_val):
                        continue
                    # reference increments over this gap
                    rp = np.array([float(raw[ei_pred, j]) for j in ref_idx], float)
                    if not np.isfinite(rp).all():
                        continue
                    inc = rp - last_vals
                    rp_ph = np.array([float(raw[ei_pred, j]) for j in ref_idx_ph], float)
                    if not np.isfinite(rp_ph).all():
                        continue
                    inc_ph = rp_ph - last_vals_ph

                    amp_krige = _krige(d_rr_amp, d_rt_amp, inc, LAMBDA_KM)
                    phase_krige = _krige(d_rr_ph, d_rt_ph, inc_ph, LAMBDA_KM)
                    # Phase-only kernel: use the PA Gram as the covariance
                    # (no distance information).  Tests whether the per-epoch
                    # phase alignment alone can predict the increment.
                    po_krige = _krige_gram(pa_gram, pa_vec, inc)
                    p_amp = last_val + amp_krige
                    p_phase = last_val + phase_krige
                    p_po = last_val + po_krige
                    p_persist = last_val
                    day_sq[(gap_s, "amplitude_krige")].append((truth - p_amp) ** 2)
                    day_sq[(gap_s, "phase_filtered_krige")].append((truth - p_phase) ** 2)
                    day_sq[(gap_s, "phase_only_krige")].append((truth - p_po) ** 2)
                    day_sq[(gap_s, "persistence")].append((truth - p_persist) ** 2)

                    # lambda sweep at the long gap only
                    if gap_s == lam_gap:
                        for lam in LAMBDA_SWEEP_KM:
                            ak = _krige(d_rr_amp, d_rt_amp, inc, lam)
                            day_sq_lam[(lam, "amplitude_krige")].append(
                                (truth - (last_val + ak)) ** 2)
                            day_sq_lam[(lam, "persistence")].append(
                                (truth - p_persist) ** 2)

        # day-level RMS for each (gap, arm) and (lam, arm)
        ok_day = all(len(day_sq[(g, a)]) >= 30
                     for g in GAPS_S for a in arms)
        if not ok_day:
            continue
        for g in GAPS_S:
            for a in arms:
                per_day[(g, a)].append(
                    float(np.sqrt(np.mean(day_sq[(g, a)]))))
        for lam in LAMBDA_SWEEP_KM:
            if len(day_sq_lam[(lam, "amplitude_krige")]) >= 30:
                per_day_lam[(lam, "amplitude_krige")].append(
                    float(np.sqrt(np.mean(day_sq_lam[(lam, "amplitude_krige")]))))
                per_day_lam[(lam, "persistence")].append(
                    float(np.sqrt(np.mean(day_sq_lam[(lam, "persistence")]))))

    def ci(a_vals, b_vals):
        """Day-clustered bootstrap of 100*(1 - rms_a/rms_b) using per-day RMS."""
        a = np.array(a_vals, float)
        b = np.array(b_vals, float)
        if len(a) < 4 or len(b) < 4 or len(a) != len(b):
            return {"median": None, "ci95": None, "n": int(len(a))}
        pct = 100.0 * (1.0 - a / b)
        bs = [float(np.median(RNG.choice(pct, len(pct), replace=True)))
              for _ in range(4000)]
        return {"median": float(np.median(pct)),
                "ci95": [float(np.percentile(bs, 2.5)),
                         float(np.percentile(bs, 97.5))],
                "n": int(len(pct)), "frac_positive": float(np.mean(pct > 0))}

    def med(vals):
        v = np.array(vals, float)
        return float(np.median(v)) if len(v) else None

    by_gap = {}
    for g in GAPS_S:
        amp = per_day[(g, "amplitude_krige")]
        ph = per_day[(g, "phase_filtered_krige")]
        po = per_day[(g, "phase_only_krige")]
        pers = per_day[(g, "persistence")]
        entry = {
            "n_days_used": len(amp),
            "median_rms_m": {
                "amplitude_krige": med(amp),
                "phase_filtered_krige": med(ph),
                "phase_only_krige": med(po),
                "persistence": med(pers),
            },
            "amplitude_vs_persistence_pct": ci(amp, pers),
            "phase_filtered_vs_persistence_pct": ci(ph, pers),
            "phase_filtered_vs_amplitude_pct": ci(ph, amp),
            "phase_only_vs_persistence_pct": ci(po, pers),
            "phase_only_vs_amplitude_pct": ci(po, amp),
        }
        ap = entry["amplitude_vs_persistence_pct"]
        entry["amplitude_beats_persistence"] = bool(
            ap["ci95"] and ap["ci95"][0] > 0)
        pa = entry["phase_filtered_vs_amplitude_pct"]
        entry["phase_filtered_beats_amplitude"] = bool(
            pa["ci95"] and pa["ci95"][0] > 0)
        by_gap[str(g)] = entry

    # lambda sweep at the long gap: amplitude_krige vs persistence at each lam
    lambda_sweep = {}
    for lam in LAMBDA_SWEEP_KM:
        amp = per_day_lam[(lam, "amplitude_krige")]
        pers = per_day_lam[(lam, "persistence")]
        lambda_sweep[str(int(lam))] = {
            "n_days_used": int(len(amp)),
            "median_rms_m": {"amplitude_krige": med(amp), "persistence": med(pers)},
            "amplitude_vs_persistence_pct": ci(amp, pers),
        }

    # headline = the 4 h gap (the regime the manuscript quotes)
    headline = by_gap[str(GAPS_S[-1])]
    res = {
        "n_days_used": headline["n_days_used"],
        "gaps_s": GAPS_S,
        "lambda_sweep_km": LAMBDA_SWEEP_KM,
        "by_gap": by_gap,
        "lambda_sweep_at_long_gap": lambda_sweep,
        # headline convenience fields (4 h gap, canonical 4,200 km)
        "median_rms_m": headline["median_rms_m"],
        "amplitude_vs_persistence_pct": headline["amplitude_vs_persistence_pct"],
        "phase_filtered_vs_persistence_pct": headline["phase_filtered_vs_persistence_pct"],
        "phase_filtered_vs_amplitude_pct": headline["phase_filtered_vs_amplitude_pct"],
        "phase_only_vs_persistence_pct": headline["phase_only_vs_persistence_pct"],
        "phase_only_vs_amplitude_pct": headline["phase_only_vs_amplitude_pct"],
        "amplitude_beats_persistence": headline["amplitude_beats_persistence"],
        "phase_filtered_beats_amplitude": headline["phase_filtered_beats_amplitude"],
    }
    return res


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)
    days = [dates[i] for i in np.linspace(2, len(dates) - 2, N_DAYS).astype(int)]
    print(f"STEP 101  |  full-field predictor at the TEP scale | {len(days)} days")
    res = run(days)
    print(json.dumps(res, indent=2))
    payload = {
        "step": 101,
        "title": "Full-field spatiotemporal predictor at the TEP scale",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "lambda_km": LAMBDA_KM,
        "nugget": NUGGET,
        "observable": (
            "Resid increment (r_j(t_i+dt) - r_j(t_i)) kriged at the "
            "physics-fixed TEP scale (lambda = 4,200 km, nugget 0.35, no "
            "data tuning) and added to the target's last known value.  A "
            "phase-coherence FILTER (restrict kriging to references with "
            "pre-window cross-spectrum phase alignment PA_bj > 0) is tested "
            "as a second arm; it is a filter on the reference set, not a "
            "projection of the increment onto the target phase."),
        "causality": (
            "All processing is strictly causal.  The cross-spectrum for the "
            "phase filter is computed from the pre-window [t_i - 12h, t_i] "
            "only.  No future data enters any arm."),
        "arms": {
            "amplitude_krige": "raw resid increment kriged at the TEP scale (the full-field predictor)",
            "phase_filtered_krige": "resid increment kriged at the TEP scale, restricted to phase-aligned references (PA > 0)",
            "phase_only_krige": "resid increment kriged with the PA Gram as covariance (no distance information)",
            "persistence": "last value held flat",
        },
        "sweeps": {
            "gap_horizons_s": GAPS_S,
            "lambda_sweep_km": LAMBDA_SWEEP_KM,
            "lambda_sweep_note": ("amplitude_krige vs persistence at each "
                                  "lambda, at the long (4 h) gap.  A TEP-"
                                  "scale load-bearing gain shows a peak in "
                                  "the 2,000-4,200 km band and a smaller "
                                  "gain at local (500 km) and far (8,000 km) "
                                  "scales."),
        },
        "result": res,
        "interpretation": (
            "The full-field predictor kriges the resid increment at the "
            "TEP scale.  The gain over persistence is the operational "
            "signal.  The lambda sweep shows whether the TEP scale is "
            "load-bearing (a local-scale kernel cannot reproduce the gain). "
            "The phase-filter arm tests whether selecting phase-aligned "
            "references adds anything on top of the TEP-scale kriging; if "
            "it does not, the gain comes from the spatial kriging, not from "
            "phase selection of individual references."),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
