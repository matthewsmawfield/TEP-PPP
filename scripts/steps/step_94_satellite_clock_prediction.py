#!/usr/bin/env python3
"""
Step 94 -- Satellite-clock prediction from inter-satellite TEP structure.

Every GPS positioning solution -- from a survey receiver to a phone -- subtracts
a satellite-clock correction.  Precise services (IGS, MADOCA, regional SBAS)
deliver these in real time and must bridge gaps: a tracking-station outage, a
satellite event, or an editing hole leaves a satellite's clock unobserved for a
few epochs, and the standard fill is that satellite's own polynomial
extrapolation.  Step 36 detected distance-structured phase correlation BETWEEN
GPS satellite clocks at 13.5 sigma.  If that structure is real, the rest of the
constellation -- observed during the gap -- predicts the missing satellite
better than its own extrapolation.

This is the satellite-side analogue of Step 55 (ground-clock outage bridging),
and it targets the correction term with the widest operational reach.

Design (all causal, all held-out):
  * 31 GPS satellites, CODE final SP3 clocks, 5-min epochs.
  * Discrete clock jumps (product resets) are repaired with the same
    excess-difference rule as Step 36's _clean_clock before detrending;
    the former version passed raw clock series straight into the detrend,
    so a single jump contaminated a satellite-day's residual and every
    innovation derived from it.
  * Per-satellite cubic detrend removes the deterministic drift; the
    residual is the fluctuation Step 36 found to be spatially structured.
  * TRAIN (first N_TRAIN days): fit C(psi), the residual correlation as a
    function of geocentric angular separation psi between two satellites, as a
    non-negative Legendre expansion (Schoenberg-PSD, same construction as
    Step 86 for the ground network).
  * TEST (next N_TEST days): for gap lengths 1..8 epochs (5..40 min), remove one
    satellite's clock over the gap and predict each missing epoch.  All
    spatial methods use the honest univariate baseline on a detrended
    residual -- the last pre-gap value -- and the innovation of every
    other satellite is measured against ITS OWN last value.  (The former
    version extrapolated a quadratic fitted to a 24-epoch window of an
    already-detrended series, which both inflated the baselines and
    injected every reference satellite's extrapolation error into the
    kriged innovations.)
      conventional : quadratic extrapolation of that satellite's own
                     pre-gap residual (documented for comparison; on a
                     detrended series this is a weak baseline and
                     const_hold is the meaningful univariate reference)
      const_hold   : last pre-gap value held flat
      const_mean   : pre-gap value + constellation-mean residual change
      tep_krige    : pre-gap value + GLS combination of the other
                     satellites' observed residual changes during the gap,
                     with covariances C(psi)*sigma_i*sigma_j off-diagonal
                     and the empirical pre-gap innovation variances on the
                     diagonal.  (The former version renormalised C(psi) to
                     a unit diagonal and added a 0.10 nugget, inflating
                     the assumed signal-to-noise ratio ~5-fold -- true
                     nugget is 1 - C(0) ~ 0.81 -- which drove tep_krige
                     and tep_scramble to the same noise-dominated RMS.)
      tep_scramble : same, with the angular-separation assignments
                     shuffled relative to the innovations (null)
  * Metric: RMS of (prediction - truth) over all held-out satellite-epochs,
    per gap length, with a day-clustered bootstrap CI on the tep_krige vs
    conventional and tep_krige vs const_mean relative improvement.

Output: results/outputs/step_94_satellite_clock_prediction.json
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_94_satellite_clock_prediction.json"
RNG = np.random.default_rng(20260904)

N_TRAIN = int(os.environ.get("TEP_SAT_TRAIN", "40"))
N_TEST = int(os.environ.get("TEP_SAT_TEST", "40"))
GAPS = [1, 2, 3, 4, 6, 8]           # epochs (5 min each)
LMAX = 6
PRNS = [f"G{i:02d}" for i in range(1, 33)]
NS_PER_M = 1.0 / 0.299792458        # ns per metre


def _repair_jumps(r, max_jump_diffs=12):
    """Repair discrete clock jumps (product resets) rather than discarding
    the satellite-day -- the same excess-difference rule as Step 36's
    _clean_clock.  Each flagged first difference is replaced by the median
    per-epoch drift, i.e. only the excess over the smooth clock wander is
    subtracted.  Returns the repaired series, or None if the clock product
    is genuinely broken (>max_jump_diffs flagged differences)."""
    r = np.asarray(r, float).copy()
    dr = np.diff(r)
    mad = 1.4826 * np.median(np.abs(dr - np.median(dr)))
    if not np.isfinite(mad) or mad <= 0:
        return None
    med = np.median(dr)
    jidx = np.flatnonzero(np.abs(dr - med) > 30.0 * mad)
    if len(jidx) > max_jump_diffs:
        return None
    for j in jidx:
        r[j + 1:] -= dr[j] - med
    return r


def legP(l, x):
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x
    Pm1, Pc = np.ones_like(x), x
    for k in range(2, l + 1):
        Pm1, Pc = Pc, ((2 * k - 1) * x * Pc - (k - 1) * Pm1) / k
    return Pc


def load_day(doy):
    """Return (names, R[n_epoch,n_sat] detrended residual ns, POS[n_epoch,n_sat,3])
    on the common epoch grid, or None."""
    s = td.load_meo_series(2024, doy, PRNS)
    if not s or len(s) < 12:
        return None
    sats = sorted(s)
    common = set(np.asarray(s[sats[0]]["t"], float).tolist())
    for k in sats[1:]:
        common &= set(np.asarray(s[k]["t"], float).tolist())
    common = np.array(sorted(common))
    if len(common) < 60:
        return None
    R = np.full((len(common), len(sats)), np.nan)
    POS = np.full((len(common), len(sats), 3), np.nan)
    for j, k in enumerate(sats):
        t = np.asarray(s[k]["t"], float)
        c = np.asarray(s[k]["resid"], float)
        traj = np.asarray(s[k]["station"].trajectory_km, float)
        idx = {tt: i for i, tt in enumerate(t)}
        rows = np.array([idx[tt] for tt in common])
        ck = c[rows]
        ok = np.isfinite(ck)
        if ok.sum() < 40:
            continue
        tt = common[ok]
        ckr = _repair_jumps(ck[ok])
        if ckr is None:
            continue                    # genuinely broken clock product
        p = np.polyfit(tt - tt.mean(), ckr, 3)
        R[ok, j] = ckr - np.polyval(p, tt - tt.mean())
        POS[:, j, :] = traj[rows]
    return sats, R, POS, common


def ang_sep(POS_epoch):
    """Geocentric angular separation matrix (deg) for one epoch's positions."""
    r = np.linalg.norm(POS_epoch, axis=1)
    c = (POS_epoch @ POS_epoch.T) / np.outer(r, r)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fit_cpsi(pairs_psi, pairs_corr):
    """Non-negative Legendre fit of C(psi); psi in degrees."""
    edges = np.linspace(0, 180, 31)
    ctr, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pairs_psi >= lo) & (pairs_psi < hi)
        if m.sum() >= 30:
            ctr.append(0.5 * (lo + hi))
            val.append(float(np.mean(pairs_corr[m])))
    if len(ctr) < 6:
        return None
    ctr = np.radians(np.array(ctr))
    val = np.array(val)
    A = np.column_stack([legP(l, np.cos(ctr)) for l in range(LMAX + 1)])
    a, *_ = np.linalg.lstsq(A, val, rcond=None)
    return np.clip(a, 0.0, None)


def cov_from(a, psi_deg):
    """Normalised C(psi)/C(0), for reporting the fitted curve's shape only.
    Kriging must use cov_raw -- C(psi) is a correlation function whose
    zero-separation value C(0) < 1 already excludes each satellite's local
    noise floor."""
    x = np.cos(np.radians(psi_deg))
    K = sum(a[l] * legP(l, x) for l in range(len(a)))
    c0 = float(sum(a))
    return K / c0 if abs(c0) > 1e-12 else K


def cov_raw(a, psi_deg):
    """Raw fitted correlation C(psi) (Schoenberg-PSD Legendre sum)."""
    x = np.cos(np.radians(psi_deg))
    return sum(a[l] * legP(l, x) for l in range(len(a)))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_doys = list(range(1, 367))
    train_doys = all_doys[:N_TRAIN]
    test_doys = all_doys[N_TRAIN:N_TRAIN + N_TEST]

    # ---- TRAIN: fit C(psi) ----
    psi_all, cor_all = [], []
    for doy in train_doys:
        d = load_day(doy)
        if d is None:
            continue
        sats, R, POS, _ = d
        n = len(sats)
        # correlation over the day between each satellite pair
        Rc = R - np.nanmean(R, axis=0, keepdims=True)
        for i in range(n):
            for j in range(i + 1, n):
                a_, b_ = Rc[:, i], Rc[:, j]
                m = np.isfinite(a_) & np.isfinite(b_)
                if m.sum() < 60:
                    continue
                sa, sb = a_[m].std(), b_[m].std()
                if sa < 1e-9 or sb < 1e-9:
                    continue
                cc = float(np.mean(a_[m] * b_[m]) / (sa * sb))
                # mean angular separation over the day
                psi = np.nanmean([ang_sep(POS[e])[i, j] for e in range(0, len(POS), 12)
                                  if np.isfinite(POS[e, i]).all() and np.isfinite(POS[e, j]).all()])
                if np.isfinite(psi):
                    psi_all.append(psi)
                    cor_all.append(cc)
        print(f"  train {doy}: {len(psi_all)} pairs", flush=True)
    psi_all = np.array(psi_all)
    cor_all = np.array(cor_all)
    a_fit = fit_cpsi(psi_all, cor_all)
    if a_fit is None:
        OUT.write_text(json.dumps({"step": 94, "status": "train_failed"}, indent=2))
        return
    # summary of the fitted curve
    near = cov_from(a_fit, np.array([25.0]))[0]
    far = cov_from(a_fit, np.array([160.0]))[0]
    print(f"\n  C(psi): Legendre coeffs {np.round(a_fit, 4)}  near(25deg)={near:.3f} far(160deg)={far:.3f}")

    # ---- TEST: gap-fill ----
    METHODS = ["conventional", "const_hold", "const_mean", "tep_krige", "tep_scramble"]
    sq = {g: {m: [] for m in METHODS} for g in GAPS}
    per_day = {g: {m: [] for m in METHODS} for g in GAPS}
    for doy in test_doys:
        d = load_day(doy)
        if d is None:
            continue
        sats, R, POS, common = d
        n = len(sats)
        nep = len(common)
        dsq = {g: {m: [] for m in METHODS} for g in GAPS}
        PRE = 24                       # 2 h pre-gap window (24 x 5 min)
        for g in GAPS:
            for t0 in range(PRE, nep - g, 6):
                em = t0 + g // 2
                if not np.isfinite(POS[em]).all():
                    continue
                PSI = ang_sep(POS[em])
                tt = np.arange(PRE, dtype=float)
                win = R[t0 - PRE:t0, :]                       # (PRE, n)
                wok = np.isfinite(win).all(axis=0)            # sats with a clean pre-gap window
                QC = np.full((3, n), np.nan)
                for j in np.flatnonzero(wok):
                    QC[:, j] = np.polyfit(tt, win[:, j], 2)
                last_v = win[-1]                              # per-sat last pre-gap residual
                iv = np.maximum(np.var(win - last_v, axis=0), 1e-12)
                for b in range(n):
                    if not wok[b]:
                        continue
                    q = QC[:, b]
                    last = float(last_v[b])
                    for gi in range(1, g + 1):
                        truth = R[t0 + gi - 1, b]
                        if not np.isfinite(truth):
                            continue
                        # innovation of every OTHER satellite at this epoch:
                        # observed level minus that satellite's own last
                        # pre-gap value -- the same const-hold baseline the
                        # target uses, so the kriged field is scored against
                        # an honest univariate reference and no reference
                        # satellite's extrapolation error leaks into inno.
                        others = np.array([j for j in range(n) if j != b])
                        obs = R[t0 + gi - 1, others]
                        inno = obs - last_v[others]
                        okm = np.isfinite(inno) & wok[others]
                        if okm.sum() < 8:
                            continue
                        oj = np.array(others)[okm]
                        inno = inno[okm]
                        te2 = PRE + gi - 1
                        conv = float(q[0]*te2*te2 + q[1]*te2 + q[2])
                        preds = {
                            "conventional": conv,
                            "const_hold": last,
                            "const_mean": float(last + np.mean(inno)),
                        }
                        # GLS kriging weights from C(psi): off-diagonal
                        # covariances C(psi)*s_i*s_j, diagonal the empirical
                        # pre-gap innovation variance (each satellite's own
                        # noise floor).  The former version used a
                        # C(psi)/C(0)-renormalised kernel with a 0.10 nugget,
                        # inflating the assumed S/N ~5-fold so the weights
                        # amplified innovation noise instead of the field.
                        s_o = np.sqrt(iv[oj])
                        psi_oo = PSI[np.ix_(oj, oj)]
                        psi_ob = PSI[oj, b]
                        Koo = cov_raw(a_fit, psi_oo) * np.outer(s_o, s_o)
                        np.fill_diagonal(Koo, iv[oj])
                        kob = cov_raw(a_fit, psi_ob) * s_o * float(np.sqrt(iv[b]))
                        try:
                            w = np.linalg.solve(Koo, kob)
                            preds["tep_krige"] = float(last + w @ inno)
                        except np.linalg.LinAlgError:
                            preds["tep_krige"] = float(last + np.mean(inno))
                        # scramble null: shuffle the psi assignments relative
                        # to the innovations, then refit weights under the
                        # broken pairing
                        perm = RNG.permutation(len(oj))
                        Koo_s = Koo[np.ix_(perm, perm)]
                        kob_s = kob[perm]
                        try:
                            ws = np.linalg.solve(Koo_s, kob_s)
                            preds["tep_scramble"] = float(last + ws @ inno)
                        except np.linalg.LinAlgError:
                            preds["tep_scramble"] = float(last + np.mean(inno))
                        for m in METHODS:
                            dsq[g][m].append((preds[m] - truth) ** 2)
        for g in GAPS:
            if len(dsq[g]["conventional"]) >= 50:
                for m in METHODS:
                    arr = np.asarray(dsq[g][m], float)
                    sq[g][m].extend(arr.tolist())
                    per_day[g][m].append(float(np.sqrt(np.mean(arr))))
        print(f"  test {doy}: n(1-epoch)={len(dsq[1]['conventional'])}", flush=True)

    # ---- aggregate ----
    by_gap = {}
    for g in GAPS:
        if not sq[g]["conventional"]:
            continue
        rms = {m: float(np.sqrt(np.mean(sq[g][m]))) for m in METHODS}
        rms_m = {m: rms[m] * 0.299792458 for m in METHODS}   # ns -> m
        def boot(num, den):
            pn = np.asarray(per_day[g][num], float)
            pd = np.asarray(per_day[g][den], float)
            if len(pn) < 4:
                return None
            gain = 100.0 * (1.0 - pn / pd)
            bs = [float(np.median(RNG.choice(gain, len(gain), replace=True))) for _ in range(4000)]
            return {"median_pct": float(np.median(gain)),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "n_days": int(len(gain))}
        by_gap[str(g * 5)] = {
            "gap_min": g * 5,
            "n_eval": len(sq[g]["conventional"]),
            "rms_ns": rms,
            "rms_m": rms_m,
            "tep_vs_conventional_pct": 100.0 * (1.0 - rms["tep_krige"] / rms["conventional"]),
            "tep_vs_const_mean_pct": 100.0 * (1.0 - rms["tep_krige"] / rms["const_mean"]),
            "tep_vs_scramble_pct": 100.0 * (1.0 - rms["tep_krige"] / rms["tep_scramble"]),
            "tep_vs_conventional_ci": boot("tep_krige", "conventional"),
            "tep_vs_const_mean_ci": boot("tep_krige", "const_mean"),
        }
        print(f"  gap {g*5:>3} min: conv {rms['conventional']:.3f} ns  tep {rms['tep_krige']:.3f} ns  "
              f"tep/conv {by_gap[str(g*5)]['tep_vs_conventional_pct']:+.1f}%  "
              f"tep/mean {by_gap[str(g*5)]['tep_vs_const_mean_pct']:+.1f}%  "
              f"tep/scr {by_gap[str(g*5)]['tep_vs_scramble_pct']:+.1f}%", flush=True)

    out = {
        "step": 94,
        "title": "Satellite-clock gap prediction from inter-satellite TEP structure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_days": N_TRAIN, "n_test_days": N_TEST,
        "cpsi_legendre_coeffs": [float(x) for x in a_fit],
        "cpsi_near_25deg": float(near), "cpsi_far_160deg": float(far),
        "krige_covariance": ("off-diag C(psi)*s_i*s_j, diag empirical "
                             "pre-gap innovation variance"),
        "by_gap": by_gap,
        "operational_context": (
            "satellite-clock correction gap-fill for real-time precise services; "
            "conventional = the missing satellite's own quadratic extrapolation, "
            "which is current practice"),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
