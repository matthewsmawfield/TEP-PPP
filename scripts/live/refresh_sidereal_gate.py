#!/usr/bin/env python3
"""Same-day self-validation of the deployed sidereal map.

The map's build-time ``corr`` table is learned on the (prev-day, day)
residual pair — a lag day stale when the previous pair can't be built
(missing products) and always measured one scene earlier than the day
the map serves.  Multipath transfer is scene-dependent (antenna moves,
wet surfaces), so the build-time corr does not reliably predict which
sats transfer TODAY.  Measured 2026-10-14 on the Oct-13 map serving
Oct-14: gate corr vs same-day transfer uncorrelated — G12 gated at 0.83
transferred at 0.11 and G19 at 0.65 -> -0.04 (unit-gain apply injected
noise, dRMSE -18..-74%), while G17 transferred at 0.975 (+78% optimal)
below its 0.40 gate and went unused.

This script scores each sat's deployed template against TODAY'S residual
table (the incremental update_sidereal_resid capture), per sat:

    rho_s = corr( resid_today(t), template(t - lag_s) )
    a_s   = rho_s * std(resid) / std(template)     (Wiener gain)

A small lag rescan around the stored lag keeps the pairing honest when
the per-sat repeat period drifts.  Results are written atomically as
map["live"] with a UTC-day stamp; _apply_sidereal_correction prefers
same-day live entries and applies the measured gain instead of the
harmful unit gain (a template with rho < ~0.5*sig_t/sig_r subtracted at
gain 1 ADDS variance — sqrt(2-2rho) for equal stds — which is how the
marginal-corr sats hurt).

Runs hourly through tep-sidereal-resid.sh after the resid update so the
score grows with the day's coverage; sats whose overlap has not arrived
yet simply stay unscored until a later run.

Usage: refresh_sidereal_gate.py [--day YYYY-MM-DD]
"""
import argparse
import datetime as dt
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "rpi"))
MAP = ROOT / "data" / "ppp_live" / "sidereal_map.pkl"
RESID_DIR = ROOT / "data" / "ppp_live" / "sidereal_resid"

MIN_PAIRS = 100          # ~5-8 min of 5s-stride epochs for a pass
MAX_GAP_S = 120.0        # contiguous-coverage guard (mirrors apply side)
LAG_SCAN = (240.0, 60.0)  # half-span, step around stored lag
LAG_SWITCH_MARGIN = 0.05  # rescan must beat stored-lag corr by this


def _per_sat_today(res):
    out = {}
    for g, dd in res.items():
        for s, v in dd.items():
            out.setdefault(s, []).append((g, v))
    return out


def _score(ut, rr, gt, rv, lag, rr_amp=None):
    """Transfer rho + Wiener gain of template(ut,rr) predicting today's
    residual rv at gt, paired at t-lag inside contiguous template spans.
    rr_amp (raw, unsmoothed rr) keeps the envelope-corr measurement on
    the unsmoothed template when the main fit runs on a smoothed one."""
    u = gt - lag
    m = (u >= ut[0]) & (u <= ut[-1])
    j = np.searchsorted(ut, u)
    ok = m.copy()
    ok[m] = (j[m] > 0) & (
        (ut[np.minimum(j[m], len(ut) - 1)] - ut[np.maximum(j[m] - 1, 0)])
        <= MAX_GAP_S)
    if int(ok.sum()) < MIN_PAIRS:
        return None
    pred = np.interp(u[ok], ut, rr)
    obs = rv[ok]
    pv = pred - pred.mean()
    ov = obs - obs.mean()
    if pv.std() < 1e-9 or ov.std() < 1e-9:
        return None
    rho = float(np.corrcoef(pv, ov)[0, 1])
    gain = rho * float(ov.std() / pv.std())
    # Envelope correlation (2026-10-18, tmp/var_echo.py): |r| repeats
    # far more repeatably than r itself (0.83-0.97 on G sats vs
    # 0.27-0.84) — the sigma-map channel gates on this.
    if rr_amp is not None:
        pv = np.interp(u[ok], ut, rr_amp)
        pv = pv - pv.mean()
        av = np.abs(pv); ao = np.abs(obs)
    else:
        av = np.abs(pred); ao = np.abs(obs)
    av = av - av.mean(); ao = ao - ao.mean()
    amp = float(np.corrcoef(av, ao)[0, 1]) if av.std() > 1e-9 else 0.0
    return rho, gain, int(ok.sum()), amp


def _smooth_rr(ut, rr, sig):
    """Gaussian-smooth template values on their (irregular) axis.
    tmp/persat_smooth.py (2026-10-18, Oct-18 corpus): sigma=60 s lifts
    single-lag transfer +19.8 -> +27.9 % mean, all-positive, flat
    30-120 s plateau — the raw 5 s resid axis rides per-epoch noise
    into the correction at full vertex.  Same mechanism as the scalar
    channel's TEP_SID_SMOOTH_S=30."""
    if sig <= 0 or len(rr) < 30:
        return rr
    # Winsorise before smoothing (matches _sid_rr_smooth apply side):
    # heavy-tail arc-edge pollution at the +-300 m clip convolves ~25
    # min into the template otherwise.  MAD x5: +0.9 pt jt eval mean.
    wk = float(os.environ.get("TEP_SID_WINSOR_MAD", "5.0"))
    if wk > 0.0:
        med = float(np.median(rr))
        mad = float(np.median(np.abs(rr - med))) * 1.4826 + 1e-9
        rr = np.clip(rr, med - wk * mad, med + wk * mad)
    dt = float(np.median(np.diff(ut)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 5.0
    w = int(np.ceil(5 * sig / dt))
    k = np.exp(-0.5 * (np.arange(-w, w + 1) / (sig / dt)) ** 2)
    k /= float(k.sum())
    rp = np.concatenate([np.full(w, rr[0]), rr, np.full(w, rr[-1])])
    return np.convolve(rp, k, mode="same")[w:w + len(rr)]


SID_SMOOTH_S = float(os.environ.get("TEP_SID_SMOOTH_PERSAT_S", "60.0"))


def _cov_mask(ut, u):
    """Dense-coverage mask over query positions u on axis ut
    (the _joint_fir/_score contiguous-span rule, hoisted)."""
    m = (u >= ut[0]) & (u <= ut[-1])
    j = np.searchsorted(ut, u)
    m &= j > 0
    jm = np.minimum(j, len(ut) - 1)
    jp = np.maximum(j - 1, 0)
    m &= (ut[jm] - ut[jp]) <= MAX_GAP_S
    return m


def _dis_corr(ut, rr, gt, rv, sat):
    """Correlation between the day-column realisation spread and
    today's |residual| (tmp/persat_disagree.py, 2026-10-18): the
    disagreement across independent day-offset templates is a cross-
    validated reliability measure — mean corr +0.55 vs the deployed
    |template| envelope's +0.30 on 32 gated sats (better on 78%).
    """
    dpos = DAY_COLS_R if sat.startswith("R") else DAY_COLS_GE
    cols = []
    okm = np.ones(len(gt), bool)
    for p in dpos:
        uu = gt - p
        dm = _cov_mask(ut, uu)
        okm &= dm
        c = np.full(len(gt), np.nan)
        c[dm] = np.interp(uu[dm], ut, rr)
        cols.append(c)
    if int(okm.sum()) < MIN_PAIRS:
        return 0.0
    P = np.stack([c[okm] for c in cols], 1)
    d = np.std(P, axis=1)
    a = np.abs(rv[okm])
    if d.std() < 1e-9 or a.std() < 1e-9:
        return 0.0
    # same ~300 s smoothing the screen applied for corr stability
    w = min(len(d), 61) | 1
    ds = np.convolve(d, np.ones(w) / w, mode="same")
    as_ = np.convolve(a, np.ones(w) / w, mode="same")
    return float(np.corrcoef(ds, as_)[0, 1])


FIR_TAPS = (150.0, 300.0, 600.0, 900.0)

# Multi-day template columns (2026-10-18, tmp/persat_multiday*.py):
# each day-offset is an independent realisation of the same sky-
# anchored multipath — a joint lstsq over [T(u0), taps, T(t - k*C)
# k=1..5] lifted transfer +8.0 pt mean over the deployed 9-tap basis
# (23 gated sats, worst -0.5).  Positions equal the family's deployed
# lag centres; the column within +-900 s of the sat's own lag is
# redundant with T(u0)+taps and is zeroed (coef lands at 0).
# Day-offset realisation columns.  R 10-orbit (~405500 s) and G/E
# 6-day added 2026-10-19 (tmp/r10_g6_screen.py: R10orb +0.24 mean /
# R06 +0.45 on the 18<-14 pair; G/E 6d G31/G05/G29 +0.78 on 18<-12).
DAY_COLS_R = (81100.0, 162200.0, 243300.0, 324400.0, 364950.0,
              405500.0)
DAY_COLS_GE = (86164.0, 172328.0, 258492.0, 344656.0, 430820.0,
               516984.0)
DAY_COL_OVERLAP_S = 900.0


def _joint_basis(ut, rr, gt, lag, sat=""):
    """Regressor matrix for the joint sidereal-echo fit:
    [T(u), e/o taps at +-150,300,600,900, day-cols T(t - p) for p in
    the family DAY_COLS] over the epochs where all lag positions sit
    inside contiguous template spans.
    e_fd = (T(u+fd)+T(u-fd))/2 - T(u)  (dispersion/curvature),
    o_fd = (T(u+fd)-T(u-fd))/2        (lag drift),
    D_p  = T(t - p)                    (older-day realisation).
    A day position within DAY_COL_OVERLAP_S of `lag` or with <MIN_PAIRS
    coverage gets a zero column (coef 0 — contributes nothing at apply).
    Returns (A, ok) — rows of A correspond to epochs of gt where ok.
    """
    def _cov(u):
        m = (u >= ut[0]) & (u <= ut[-1])
        j = np.searchsorted(ut, u)
        m &= j > 0
        jm = np.minimum(j, len(ut) - 1)
        jp = np.maximum(j - 1, 0)
        m &= (ut[jm] - ut[jp]) <= MAX_GAP_S
        return m

    dpos = DAY_COLS_R if sat.startswith("R") else DAY_COLS_GE
    duk = [gt - p for p in dpos]
    dok = []
    for p, uu in zip(dpos, duk):
        if abs(p - lag) < DAY_COL_OVERLAP_S:
            dok.append(None)  # redundant with the T(u0) column
            continue
        dm = _cov(uu)
        dok.append(dm if int(dm.sum()) >= MIN_PAIRS else None)

    u0 = gt - lag
    ok = _cov(u0)
    for fd in FIR_TAPS:
        ok &= _cov(u0 + fd)
        ok &= _cov(u0 - fd)
    # Day columns do NOT shrink ok: an uncovered epoch just reads 0 in
    # the column (matching the apply side, which contributes 0 where
    # the position has no dense coverage).  Shrinking by all five
    # positions cost more fit epochs than the columns are worth on
    # sparse-coverage sats (24/65 lost the jt entirely in the strict
    # variant, 2026-10-18).
    if int(ok.sum()) < MIN_PAIRS:
        return None, None
    T = np.interp(u0[ok], ut, rr)
    cols = [T]
    for fd in FIR_TAPS:
        up = np.interp(u0[ok] + fd, ut, rr)
        um = np.interp(u0[ok] - fd, ut, rr)
        cols.append(0.5 * (up + um) - T)
        cols.append(0.5 * (up - um))
    for uu, dm in zip(duk, dok):
        if dm is None:
            cols.append(np.zeros(int(ok.sum())))
            continue
        col = np.zeros(int(ok.sum()))
        cov = dm[ok]
        col[cov] = np.interp(uu[ok][cov], ut, rr)
        cols.append(col)
    A = np.stack(cols, 1)
    return A, ok


def _ridge_solve(A, y):
    """Split-lambda ridge (2026-10-19, tmp/persat_ridge.py): plain
    lstsq blows up out-of-fit on thin-coverage day columns (R03 -200%,
    R18 -236%, G23 -43% eval-half transfer on Oct-18; mean -5.9% on
    Oct-17).  Ridge scaled to mean diag(A^T A) with T(u0) protected
    (0.05), FIR taps moderate (0.5), sparse/leveraged day-cols hard
    (1.2) fixes the blow-ups: Oct-18 mean +2.3 -> +13.5, Oct-17
    -5.9 -> +15.1 with the best median (+9.7) of all variants."""
    _rl = float(os.environ.get("TEP_SID_JT_RIDGE", "1.0"))
    if _rl <= 0.0:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.asarray(c, float)
    _d = float(np.diag(A.T @ A).mean())
    _L = np.full(A.shape[1], 0.5 * _rl * _d)
    _L[0] = 0.05 * _rl * _d
    _L[9:] = 1.2 * _rl * _d
    return np.linalg.solve(A.T @ A + np.diag(_L), A.T @ y)


def _apply_jt(B, jt):
    """Deployed apply semantics: a0 clip [0,1.2], aux clip [-1,1]."""
    return (float(np.clip(jt[0], 0.0, 1.2)) * B[:, 0]
            + B[:, 1:] @ np.clip(np.asarray(jt[1:], float), -1.0, 1.0))


def _joint_fir(ut, rr, gt, rv, lag, sat=""):
    """Joint ridge fit on [T(u), e/o taps at +-150,300,600,900,
    day-cols T(t - p) for p in the family DAY_COLS] over the epochs
    where all lag positions sit inside contiguous template spans.
    e_fd = (T(u+fd)+T(u-fd))/2 - T(u)  (dispersion/curvature),
    o_fd = (T(u+fd)-T(u-fd))/2        (lag drift),
    D_p  = T(t - p)                    (older-day realisation).
    A day position within DAY_COL_OVERLAP_S of `lag` or with <MIN_PAIRS
    coverage gets a zero column (coef 0 — contributes nothing at apply).
    Returns the 14/15-coefficient vector multiplying the basis so the
    apply side SUBTRACTS jt.basis — i.e. jt = lstsq(A, +resid),
    matching the screen basis in tmp/persat_fir.py + persat_multiday*.py
    (+8.0 pt mean over the 9-tap basis on 23 gated sats, worst -0.5,
    2026-10-18)."""
    A, ok = _joint_basis(ut, rr, gt, lag, sat)
    if A is None:
        return None
    return [float(x) for x in _ridge_solve(A, rv[ok])]


def _forward_single(ut, rr, gt, rv, lag):
    """Forward-transfer check on the single-lag APPLY set — cov(u0)
    coverage only, which is exactly the epoch set a jt-less entry
    serves.  _forward_check can only see jt-basis-covered epochs;
    when jt is dropped the served set WIDENS and goes unvalidated
    (live 2026-10-19: R04 ph stored fwd_1=+99.4 measured on the
    jt-covered subset, yet anti-transfers -68% on the wider cov(u0)
    apply set).  Fit the a0-only arm on the first 55% of the apply
    set, evaluate the last 45%."""
    u0 = np.asarray(gt, float) - float(lag)
    ok = _cov_mask(ut, u0)
    if int(ok.sum()) < MIN_PAIRS:
        return None
    gi = np.argsort(u0[ok])
    nf = int(len(gi) * 0.55)
    fi, ei = gi[:nf], gi[nf:]
    if len(fi) < MIN_PAIRS or len(ei) < 50:
        return None
    x = np.interp(u0[ok], ut, rr)
    y = np.asarray(rv, float)[ok]
    ye = y[ei]
    var0 = ye.std() ** 2
    if var0 < 1e-9:
        return None
    a0 = float(np.clip(
        float(x[fi] @ y[fi]) / max(float(x[fi] @ x[fi]), 1e-9),
        0.0, 1.2))
    return 100.0 * (1.0 - (ye - a0 * x[ei]).std() ** 2 / var0)


def _forward_check(ut, rr, gt, rv, lag, sat):
    """Temporal-split honesty check on the jt fit: the gate refits
    hourly and applies to later same-day epochs, so fit on the first
    55 % of covered epochs and evaluate residual-variance reduction on
    the last 45 % — jt basis vs the a0-only (single-lag) arm.
    Returns (pct_jt, pct_single) or None when the eval set is thin."""
    A, ok = _joint_basis(ut, rr, gt, lag, sat)
    if A is None:
        return None
    gi = np.argsort(gt[ok])
    nf = int(len(gi) * 0.55)
    fi, ei = gi[:nf], gi[nf:]
    if len(fi) < MIN_PAIRS or len(ei) < 50:
        return None
    y = rv[ok]
    ye = y[ei]
    var0 = ye.std() ** 2
    if var0 < 1e-9:
        return None
    jt = _ridge_solve(A[fi], y[fi])
    pj = 100.0 * (1.0 - (ye - _apply_jt(A[ei], jt)).std() ** 2 / var0)
    a0 = float(np.clip(
        float(A[fi, 0] @ y[fi]) / max(float(A[fi, 0] @ A[fi, 0]), 1e-9),
        0.0, 1.2))
    p1 = 100.0 * (1.0 - (ye - a0 * A[ei, 0]).std() ** 2 / var0)
    return pj, p1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None,
                    help="UTC date YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    day = (dt.date.fromisoformat(args.day) if args.day
           else dt.datetime.now(dt.timezone.utc).date())

    rp = RESID_DIR / f"{day.isoformat()}.pkl"
    if not rp.exists():
        print(f"no resid table for {day}")
        return 2
    blob = pickle.loads(rp.read_bytes())
    res = blob.get("res", {}) if isinstance(blob, dict) else blob
    if not res:
        print(f"empty resid table for {day}")
        return 2
    try:
        mp = pickle.loads(MAP.read_bytes())
    except Exception as e:
        print(f"map unreadable: {e}")
        return 2
    sats = mp.get("sats") or {}
    lags = mp.get("lag") or {}
    if not sats:
        print("map has no sats")
        return 2

    today = _per_sat_today(res)
    live = {}
    span, step = LAG_SCAN
    for s, ent in sats.items():
        obs = today.get(s)
        if not obs or len(obs) < MIN_PAIRS:
            continue
        ut = np.asarray(ent["u"], float)
        rr = np.asarray(ent["r"], float)
        rr_s = _smooth_rr(ut, rr, SID_SMOOTH_S)
        gt = np.array([g for g, _ in obs])
        rv = np.array([v for _, v in obs])
        lag0 = float(lags.get(s, 86164.0))
        best = _score(ut, rr_s, gt, rv, lag0, rr_amp=rr)
        best_lag = lag0
        for dl in np.arange(-span, span + 1, step):
            if dl == 0.0:
                continue
            r = _score(ut, rr_s, gt, rv, lag0 + float(dl), rr_amp=rr)
            if r is not None and (best is None or
                                  r[0] > best[0] + LAG_SWITCH_MARGIN):
                best, best_lag = r, lag0 + float(dl)
        is_ladder = False
        if best is None:
            # Lag-ladder coverage fallback (2026-10-19): the stored lag
            # can land inside today's tracking gap — deep-lag sats are
            # dead all day even when another echo centre IS covered
            # (phase screen: 16 sats recoverable, R08 lag=407120 cov=0
            # -> 535 pairs at the 81100 centre).  Try each family
            # centre with its own +-rescan; the entry's lag then serves
            # the apply side transparently, and the same measured-rho /
            # forward gates below decide honestly whether it applies.
            cands = DAY_COLS_R if s.startswith("R") else DAY_COLS_GE
            for c in dict.fromkeys([86164.0] + list(cands)):
                if abs(c - lag0) < 1.0:
                    continue
                r = _score(ut, rr_s, gt, rv, c, rr_amp=rr)
                bc = c
                for dl in np.arange(-span, span + 1, step):
                    if dl == 0.0:
                        continue
                    r2 = _score(ut, rr_s, gt, rv, c + float(dl),
                                rr_amp=rr)
                    if r2 is not None and (r is None or
                                           r2[0] > r[0]
                                           + LAG_SWITCH_MARGIN):
                        r, bc = r2, c + float(dl)
                if r is not None and (best is None or
                                      r[0] > best[0]):
                    best, best_lag = r, bc
                    is_ladder = True
        if best is None:
            continue
        rho, gain, n, amp = best
        if is_ladder and rho < 0.35:
            # Ladder entries are weak evidence by construction — the
            # stored lag failed coverage — so they clear a harder
            # measured-rho bar than the 0.25 live-min floor.
            continue
        if n < 250 and rho < 0.30:
            # Thin-count floor: too few pairs for the forward check to
            # run, so nothing validates that today's transfer persists
            # to later hours — observed harm: R08 (n=171, rho 0.29)
            # anti-transferred -47% on the eval half.  Strong-but-thin
            # entries (G19 0.58, n=148) pass on rho.
            continue
        ent_l = {"corr": rho, "gain": float(np.clip(gain, 0.0, 1.2)),
                 "n": n, "lag": best_lag, "amp_corr": amp,
                 "dis_corr": _dis_corr(ut, rr_s, gt, rv, s),
                 # Correction magnitude cap (2026-10-19): a correction
                 # larger than ~1.5x today's measured |resid| p95 can
                 # only inject — the bound scales with the scene so a
                 # quiet window shrinks the cap automatically.  2 m
                 # floor keeps it non-binding in normal operation.
                 "corr_cap": float(max(2.0, 1.5 * float(
                     np.percentile(np.abs(np.asarray(rv, float)), 95))
                     )) if len(rv) else 300.0}
        jt = _joint_fir(ut, rr_s, gt, rv, best_lag, s)
        if jt is not None:
            ent_l["jt"] = jt
        # Forward-transfer honesty check (2026-10-19, applied via the
        # temporal split in _forward_check): when even the single-lag
        # arm anti-transfers on the eval half the sat is gated OFF for
        # the rest of the day (corr->0 keeps it under the live-min
        # gate; the measured rho is kept under rho_raw), and when jt
        # adds less than ~1 pt over single-lag the entry falls back to
        # the cheap path.  Kills the R03/R18/G23-class silent harms
        # that the full-day rho cannot see.
        fc = _forward_check(ut, rr_s, gt, rv, best_lag, s)
        if fc is not None:
            ent_l["fwd_jt"], ent_l["fwd_1"] = (round(float(fc[0]), 1),
                                             round(float(fc[1]), 1))
            if fc[0] < 0.0 and fc[1] < 0.0:
                ent_l["rho_raw"] = ent_l["corr"]
                ent_l["corr"] = 0.0
                ent_l.pop("jt", None)
            elif jt is not None and fc[0] < fc[1] - 1.0:
                ent_l.pop("jt", None)
        if "jt" not in ent_l and ent_l["corr"] > 0.0:
            # Apply-set forward check: jt-less entries serve the wider
            # cov(u0) set the jt-basis check cannot see.
            fs = _forward_single(ut, rr_s, gt, rv, best_lag)
            if fs is not None:
                ent_l["fwd_a"] = round(float(fs), 1)
                if fs < -3.0:
                    ent_l["rho_raw"] = ent_l["corr"]
                    ent_l["corr"] = 0.0
        live[s] = ent_l

    # Phase-consistency-veto parameters (2026-10-19, deployed apply):
    # fit today's code resid ~ a*ph_dev_anom per sat (arc-demeaned
    # phase deviations from the sidereal_resid_ph table), then store
    # phv_a + phv_thr = the no-fault disagreement bound (4*1.4826*MAD
    # of the regression residual, clipped [6, 30]).  The apply vetoes
    # a served correction when |corr - phv_a*ph_anom| > phv_thr —
    # epochs the template disagrees with the sat's OWN live phase
    # are transient pollution, not the repeatable structure.
    try:
        ph_res_p0 = RESID_DIR.parent / "sidereal_resid_ph" / \
            f"{day.isoformat()}.pkl"
        if ph_res_p0.exists():
            _pb = pickle.loads(ph_res_p0.read_bytes())
            _pres = (_pb.get("res", {}) if isinstance(_pb, dict)
                     else _pb) or {}
            _psl = (_pb.get("slips", {}) if isinstance(_pb, dict)
                    else {}) or {}
            # Build per-sat phase-deviation ANOMALY series exactly as
            # the apply side sees them: dev minus a per-arc level
            # (median), arcs split at >120 s gaps or slip flags —
            # NOT phase_arcs_demean (its heavier processing leaves a
            # ~4x smaller signal, which shrank the fitted slope to
            # ~0.5 vs the true ~0.95 and would have mistuned the veto).
            _pdem = {}
            for _g in sorted(_pres):
                for _s, _v in _pres[_g].items():
                    _pdem.setdefault(_s, []).append((_g, _v))
            for _s, _rows in list(_pdem.items()):
                _t = np.array([g for g, _ in _rows], float)
                _v = np.array([v for _, v in _rows], float)
                _br = [0]
                for _i in range(1, len(_t)):
                    _sl = (_s in (_psl.get(_t[_i]) or ()))
                    if _t[_i] - _t[_i - 1] > 120.0 or _sl:
                        _br.append(_i)
                _br.append(len(_t))
                _tt, _vv = [], []
                for _i in range(len(_br) - 1):
                    _sg = slice(_br[_i], _br[_i + 1])
                    if _br[_i + 1] - _br[_i] < 30:
                        continue
                    _tt.append(_t[_sg])
                    _vv.append(_v[_sg] - np.median(_v[_sg]))
                if _tt:
                    _pdem[_s] = (np.concatenate(_tt),
                                 np.concatenate(_vv))
                else:
                    _pdem.pop(_s)
            # per-sat code series
            _ctoday = _per_sat_today(res)
            _npv = 0
            for s, ent_l in live.items():
                _pc = _ctoday.get(s)
                _phd = _pdem.get(s)
                if not _pc or _phd is None or len(_pc) < 300:
                    continue
                _pu, _pr = _phd
                if len(_pu) < 300:
                    continue
                _gt2 = np.array([g for g, _ in _pc], float)
                _rv2 = np.array([v for _, v in _pc], float)
                _pv = np.interp(_gt2, np.asarray(_pu, float),
                                np.asarray(_pr, float))
                _A = np.stack([_pv, np.ones_like(_pv)], 1)
                _cf, *_ = np.linalg.lstsq(_A, _rv2, rcond=None)
                _rcp = _rv2 - _A @ _cf
                _mad = float(np.median(np.abs(_rcp
                                              - np.median(_rcp))))
                ent_l["phv_a"] = float(np.clip(_cf[0], 0.0, 1.5))
                ent_l["phv_thr"] = float(np.clip(
                    4.0 * 1.4826 * _mad, 6.0, 30.0))
                _npv += 1
            if _npv:
                print(f"phveto params fit on {_npv} sats")
    except Exception as _e:
        print(f"phveto fit skipped: {_e}")

    mp["live"] = live
    mp["live_meta"] = {"day": day.isoformat(),
                       "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
    # Per-day validation history: replays of a recent-day window consult
    # the gate that served that obs day without mutating the deployed
    # "live" entries (apply side keys off the observations' UTC day).
    hist = mp.setdefault("live_hist", {})
    hist[day.isoformat()] = live
    for k in sorted(hist)[:-4]:
        del hist[k]
    tmp = MAP.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(mp))
    tmp.replace(MAP)

    strong = sorted(live.items(), key=lambda kv: -kv[1]["corr"])
    print(f"{day}: scored {len(live)}/{len(sats)} sats on "
          f"{sum(v['n'] for v in live.values())} pairs")
    for s, v in strong:
        if v["corr"] >= 0.25 or v["corr"] <= -0.15:
            print(f"  {s}: rho={v['corr']:+.3f} gain={v['gain']:.2f} "
                  f"n={v['n']} lag={v['lag']:.0f} "
                  f"(build corr={float((mp.get('corr') or {}).get(s, 0)):.2f})")

    # ---- sidereal PHASE map (sidereal_map_ph.pkl) --------------------
    # Same machinery on the carrier-phase channel: arc-demean today's
    # phase-dev table (ambiguity constant out per arc) and score the
    # deployed phase template's same-day transfer.  Phase apply is
    # live-gated ONLY (no build-corr fallback) — this is the newer,
    # less-proven channel.
    try:
        from lib.sidereal_map import phase_arcs_demean
        ph_map_p = MAP.parent / "sidereal_map_ph.pkl"
        ph_res_p = RESID_DIR.parent / "sidereal_resid_ph" / \
            f"{day.isoformat()}.pkl"
        if ph_map_p.exists() and ph_res_p.exists():
            mp_ph = pickle.loads(ph_map_p.read_bytes())
            blob = pickle.loads(ph_res_p.read_bytes())
            pr_ = blob.get("res", {}) if isinstance(blob, dict) else blob
            psl = blob.get("slips", {}) if isinstance(blob, dict) else {}
            td = phase_arcs_demean(pr_, psl)
            sats_ph = mp_ph.get("sats") or {}
            lags_ph = mp_ph.get("lag") or {}
            live_ph = {}
            for s, ent in sats_ph.items():
                obs = td.get(s)
                if obs is None or len(obs[0]) < MIN_PAIRS:
                    continue
                ut = np.asarray(ent["u"], float)
                rr = np.asarray(ent["r"], float)
                rr_s = _smooth_rr(ut, rr, SID_SMOOTH_S)
                gt, rv = obs
                lag0_ph = float(lags_ph.get(s, 86164.0))
                sc = _score(ut, rr_s, gt, rv, lag0_ph, rr_amp=rr)
                best, best_lag = sc, lag0_ph
                for dl in np.arange(-span, span + 1, step):
                    if dl == 0.0:
                        continue
                    r = _score(ut, rr_s, gt, rv, lag0_ph + float(dl),
                               rr_amp=rr)
                    if r is not None and (best is None or
                                          r[0] > best[0]
                                          + LAG_SWITCH_MARGIN):
                        best = r
                        best_lag = float(lags_ph.get(s, 86164.0)) \
                            + float(dl)
                is_ladder = False
                if best is None:
                    # Lag-ladder coverage fallback (same rule as the
                    # code channel): deep-lag phase sats sit in today's
                    # tracking gap — 16/80 recoverable via a covered
                    # family centre (R08 ph lag=407120 cov=0 -> 535
                    # pairs at 81100).  Phase apply reads the entry's
                    # lag, so the gate only needs to score it.
                    cands = (DAY_COLS_R if s.startswith("R")
                             else DAY_COLS_GE)
                    for c in dict.fromkeys([86164.0] + list(cands)):
                        if abs(c - lag0_ph) < 1.0:
                            continue
                        r = _score(ut, rr_s, gt, rv, c, rr_amp=rr)
                        bc = c
                        for dl in np.arange(-span, span + 1, step):
                            if dl == 0.0:
                                continue
                            r2 = _score(ut, rr_s, gt, rv,
                                        c + float(dl), rr_amp=rr)
                            if r2 is not None and (
                                    r is None or
                                    r[0] > r[0] + LAG_SWITCH_MARGIN):
                                r, bc = r2, c + float(dl)
                        if r is not None and (best is None or
                                              r[0] > best[0]):
                            best, best_lag = r, bc
                            is_ladder = True
                if best is None:
                    continue
                rho, gain, n, _amp = best
                if is_ladder and rho < 0.35:
                    # Weak-evidence rule (same as the code channel):
                    # harder measured-rho bar than the live-min floor.
                    continue
                if n < 250 and rho < 0.30:
                    # Thin-count floor (same as the code channel): no
                    # forward-check coverage + weak rho = unvalidated.
                    continue
                ent_ph = {"corr": rho,
                          "gain": float(np.clip(gain, 0.0, 1.2)),
                          "n": n, "lag": best_lag}
                # Phase dispersion taps (tmp/ph_fir.py, 2026-10-18):
                # joint +15.6 pt over single-lag, 10/10 gated sats —
                # carrier-phase echo dispersion is wider than code's
                # (R12 12->43 %, R06 24->45 % variance transfer).
                jtp = _joint_fir(ut, rr_s, gt, rv, best_lag, s)
                if jtp is not None:
                    ent_ph["jt"] = jtp
                fc = _forward_check(ut, rr_s, gt, rv, best_lag, s)
                if fc is not None:
                    ent_ph["fwd_jt"], ent_ph["fwd_1"] = (
                        round(float(fc[0]), 1), round(float(fc[1]), 1))
                    if fc[0] < 0.0 and fc[1] < 0.0:
                        ent_ph["rho_raw"] = ent_ph["corr"]
                        ent_ph["corr"] = 0.0
                        ent_ph.pop("jt", None)
                    elif jtp is not None and fc[0] < fc[1] - 1.0:
                        ent_ph.pop("jt", None)
                if "jt" not in ent_ph and ent_ph["corr"] > 0.0:
                    fs = _forward_single(ut, rr_s, gt, rv, best_lag)
                    if fs is not None:
                        ent_ph["fwd_a"] = round(float(fs), 1)
                        if fs < -3.0:
                            ent_ph["rho_raw"] = ent_ph["corr"]
                            ent_ph["corr"] = 0.0
                live_ph[s] = ent_ph
            mp_ph["live"] = live_ph
            mp_ph["live_meta"] = {
                "day": day.isoformat(),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
            hist_ph = mp_ph.setdefault("live_hist", {})
            hist_ph[day.isoformat()] = live_ph
            for k in sorted(hist_ph)[:-4]:
                del hist_ph[k]
            tmp = ph_map_p.with_suffix(".pkl.tmp")
            tmp.write_bytes(pickle.dumps(mp_ph))
            tmp.replace(ph_map_p)
            print(f"phase: scored {len(live_ph)}/{len(sats_ph)} sats")
            for s, v in sorted(live_ph.items(),
                               key=lambda kv: -kv[1]["corr"]):
                if v["corr"] >= 0.3:
                    print(f"  ph {s}: rho={v['corr']:+.3f} "
                          f"gain={v['gain']:.2f} n={v['n']}")
    except Exception as e:
        print(f"phase refresh skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
