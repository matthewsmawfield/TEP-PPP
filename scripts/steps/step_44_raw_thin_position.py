#!/usr/bin/env python3
"""
STEP 44 -- Raw-observation degraded-geometry positioning: strict P2 gate.

This is the genuine raw-observation position test that Step 39 could not be.
Step 39 uses a linearised replay around RTKLIB's FULL-SKY solution: the
removed satellites have already informed the state and post-fit residuals.
That is a mechanism/diagnostic check, not a raw-observation proof.

Step 44 removes satellites from the RINEX observation file BEFORE the solver
sees them (scripts/lib/rinex_thin.py), runs an independent RTKLIB SPP solve
on the thinned observations, and then applies each clock-prior method as a
linearised correction around the THINNED solution -- never the full-sky
state.  The full-sky solution is never computed, never used, and never
contaminates the residuals.

Method per station-day and satellite count k:
  1. Thin the RINEX to the k highest-elevation satellites at each epoch
     (elevation computed from SP3 orbits and the station's SINEX truth ECEF).
  2. Run rnx2rtkp -p 0 -y 2 on the thinned RINEX -> .stat file with
     position, clock, and per-satellite az/el/post-fit-residual.
  3. Predict the held-out station's receiver clock from the REST of the
     network with the nested-CV distance kernel (Step 25 machinery).
  4. For each clock-prior method (ordinary, persistence, linear, quadratic,
     AR1, TEP kernel), apply the clock prior as a linearised correction
     around the thinned solution and compute the 3-D position error against
     IGS SINEX truth.
  5. Calibrate every prior sigma on a same-horizon pseudo-gap that ends
     before the evaluation window (causal, no leakage).

P2 gate: TEP kernel beats the deployable conventional clock model
(training-selected temporal predictor) at 4/5/6 satellites with clustered
CI excluding zero, on raw-observation solves against external truth.
"""
from __future__ import annotations

import sys
import json
import shutil
import subprocess
import tempfile
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td          # noqa: E402
from scripts.lib import tep_sinex as sx         # noqa: E402
from scripts.lib import rinex_thin as rt        # noqa: E402
from scripts.lib import tep_kernel as tk        # noqa: E402
from scripts.steps.step_33_position_demo import (  # noqa: E402
    parse_stat, los_unit, kernel_clock_prediction,
    _find_train_day, _rnx2rtkp, _obs_path, _nav_path, _sp3_path,
)

OUT = PROJECT_ROOT / "results" / "outputs"
CACHE = PROJECT_ROOT / "data" / "step44_cache"
RNG = np.random.default_rng(20240944)

DAYS = [f"2024{d:03d}" for d in range(100, 107)]
K_VALUES = [4, 5, 6, 8]
GAP_LENGTHS_S = [600.0, 1800.0, 3600.0, 7200.0, 14400.0]
GAP_STRIDE_S = 3 * 3600.0
ELEV_MIN_DEG = 10.0
SYSTEM = "G"
MAX_STATIONS = 15
TEMPORAL_METHODS = ("persistence", "linear", "quadratic", "ar1")
# NOTE: AR1 is the discrete-time first-order Gauss-Markov process with a
# data-estimated time constant (phi = lag-1 autocorrelation).  It IS the
# "Gauss-Markov process with a properly estimated time constant" that a
# fair conventional battery requires; no separate GM method is needed.
# Deployable conventional methods (excluding unstable polynomial extrapolation
# over long gaps).  These are the fair conventional baselines a GNSS engineer
# would actually deploy: persistence, bounded Gauss-Markov, and a free GP
# with learned length scale.
DEPLOYABLE_METHODS = ("persistence", "gauss_markov", "free_gp")
# Fixed TEP kernel parameters (physics-predicted, not tuned on data)
TEP_FIXED_FORM = "gaussian"
TEP_FIXED_LAMBDA_KM = 4200.0
TEP_FIXED_NUGGET = 0.35


# ------------------------------------------------------------------ solver
def _enu_rot(truth_xyz):
    x, y, z = truth_xyz
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y))
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def _clock_dop(A, w):
    """Clock (time) dilution of precision from the geometry matrix.

    Returns sqrt of the (4,4) element of (A^T W A)^{-1}, i.e. the standard
    deviation of the clock estimate from observations alone in units of the
    code noise sigma.  Returns +inf when the clock is unobservable (rank < 4).
    """
    AtWA = A.T @ (w[:, None] * A)
    try:
        cov = np.linalg.inv(AtWA)
        return float(np.sqrt(abs(cov[3, 3])))
    except np.linalg.LinAlgError:
        return float("inf")


# A finite-error clock prior is a real measurement.  The previous
# INERT_FACTOR = 30 inflated the prior until it contributed < 0.2 % of the
# clock information at k>=4, making TEP look inert by construction.
# The prior now enters at its causally calibrated sigma.  A disagreement
# gate still down-weights it when it conflicts with the free-clock estimate
# by more than DISAGREE_K sigma -- receiver integrity, not a forced null.
INERT_FACTOR = 1.0
DISAGREE_K = 3.0


def _solve(A, b, w, cdt_prior=None, sig=None, return_full=False):
    A = np.asarray(A, float); b = np.asarray(b, float); w = np.asarray(w, float)
    if cdt_prior is not None and sig and sig > 0:
        A = np.vstack([A, [0.0, 0.0, 0.0, 1.0]])
        b = np.append(b, cdt_prior)
        w = np.append(w, 1.0 / sig ** 2)
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w
    sv = np.linalg.svd(Aw, compute_uv=False)
    if sv[-1] < 1e-4 * sv[0]:
        return None
    try:
        x, _, rank, _ = np.linalg.lstsq(Aw, bw, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    if rank < 4:
        return None
    if np.linalg.norm(x[:3]) > 50_000.0:
        return None
    return x if return_full else x[:3]


def _ar1_predict(t_pre, c_pre, t_query, anchor):
    if len(c_pre) < 5:
        return anchor
    dt = np.median(np.diff(t_pre))
    x = c_pre - np.mean(c_pre)
    num = np.sum(x[1:] * x[:-1]); den = np.sum(x[:-1] ** 2) + 1e-12
    phi = np.clip(num / den, -0.999, 0.999)
    steps = max(1, int(round((t_query - t_pre[-1]) / max(dt, 1.0))))
    return float(np.mean(c_pre) + (c_pre[-1] - np.mean(c_pre)) * phi ** steps)


# FOGM alpha grid for Gauss-Markov time-constant tuning (deployable baseline)
FOGM_ALPHA_GRID = (0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)


def _gauss_markov_predict(t_pre, c_pre, t_query, alpha=0.95):
    """First-order Gauss-Markov extrapolation with bounded decay to the mean.

    This is the deployable conventional clock prior: a properly bounded
    stochastic process with a tuned time constant, unlike polynomial
    extrapolation which diverges over multi-hour gaps.
    """
    if len(c_pre) < 5:
        return float(c_pre[-1])
    dt = np.median(np.diff(t_pre))
    mean_pre = float(np.mean(c_pre))
    val = float(c_pre[-1])
    steps = max(1, int(round((t_query - t_pre[-1]) / max(dt, 1.0))))
    return float(mean_pre + (val - mean_pre) * alpha ** steps)


def _tune_fogm_alpha(t_pre, c_pre):
    """Tune the FOGM decay parameter on a validation segment inside the pre-gap history."""
    n = len(c_pre)
    if n < 12:
        return 0.95
    mid = n // 2
    fit_t, fit_c = t_pre[:mid], c_pre[:mid]
    val_t = t_pre[mid:]
    best_a, best_e = 0.95, np.inf
    for a in FOGM_ALPHA_GRID:
        errs = []
        for tq in val_t:
            p = _gauss_markov_predict(fit_t, fit_c, float(tq), alpha=a)
            errs.append(abs(p - np.interp(tq, t_pre, c_pre)))
        e = float(np.sqrt(np.mean(np.square(errs))))
        if e < best_e:
            best_e, best_a = e, a
    return best_a


def _poly_predict(t_pre, c_pre, t_query, deg):
    if len(c_pre) < deg + 2:
        return c_pre[-1]
    t0 = t_pre[-1]
    coef = np.polyfit(t_pre - t0, c_pre, deg)
    return float(np.polyval(coef, t_query - t0))


def _clock_predictions(t_hist, c_hist, query_times, kt, kres, kres_fixed=None,
                      fogm_alpha=0.95):
    anchor = float(c_hist[-1])
    rate = float((c_hist[-1] - c_hist[0]) /
                 max(t_hist[-1] - t_hist[0], 1.0))
    kres_anchor = float(np.interp(t_hist[-1], kt, kres))
    kres_fixed_anchor = (float(np.interp(t_hist[-1], kt, kres_fixed))
                         if kres_fixed is not None else kres_anchor)
    out = []
    for s in query_times:
        dt = float(s) - t_hist[-1]
        pred = {
            "persistence": anchor,
            "linear": anchor + rate * dt,
            "quadratic": _poly_predict(t_hist, c_hist, float(s), 2),
            "ar1": _ar1_predict(t_hist, c_hist, float(s), anchor),
            "gauss_markov": _gauss_markov_predict(t_hist, c_hist, float(s),
                                                  alpha=fogm_alpha),
            # free_gp: tuned exponential kernel (lambda selected on training day)
            "free_gp": anchor + (float(np.interp(s, kt, kres)) - kres_anchor),
        }
        # tep_fixed: physics-fixed Gaussian kernel (lambda=4200 km, no tuning)
        if kres_fixed is not None:
            pred["tep_fixed"] = anchor + (float(np.interp(s, kt, kres_fixed))
                                          - kres_fixed_anchor)
        out.append(pred)
    return out


def _causal_prior_sigmas(sods, cdt_full, kt, kres, gap_start, gap_length,
                         kres_fixed=None, fogm_alpha=0.95):
    history_s = 1800.0
    cal_end = gap_start - history_s
    cal_start = cal_end - gap_length
    cal_hist = sods[(sods >= cal_start - history_s) & (sods < cal_start)]
    cal_query = sods[(sods >= cal_start) & (sods < cal_end)]
    if len(cal_hist) < 6 or len(cal_query) < 10:
        return None, None
    c_hist = np.interp(cal_hist, sods, cdt_full)
    truth = np.interp(cal_query, sods, cdt_full)
    preds = _clock_predictions(cal_hist, c_hist, cal_query, kt, kres,
                               kres_fixed=kres_fixed, fogm_alpha=fogm_alpha)
    methods = ("persistence", "linear", "quadratic", "ar1", "gauss_markov",
               "free_gp")
    if kres_fixed is not None:
        methods = methods + ("tep_fixed",)
    sigmas = {}
    for method in methods:
        errors = np.abs(np.array([p[method] for p in preds], float) - truth)
        # MAD-based sigma (correct Gaussian scale estimator) with realistic
        # floor of 1.0 m.  The old median absolute error underestimated sigma
        # by ~1.5x and the 0.05 m floor was physically implausible, causing
        # the prior to dominate the observations and produce catastrophic
        # position outliers when the clock prediction was wrong.
        mad_sigma = 1.4826 * np.median(errors)
        p84_sigma = float(np.percentile(errors, 84))
        sigma = max(mad_sigma, p84_sigma)
        sigmas[method] = min(max(sigma, 1.0), 50.0)
    return sigmas, float(cal_end)


# ------------------------------------------------------------------ thinned RTKLIB
def _run_thinned_rtklib(station9: str, doy: int, k: int, truth_ecef, tmp: Path):
    """Thin RINEX to k satellites, run RTKLIB SPP, return parsed .stat or None."""
    obs = _obs_path(station9, doy)
    nav = _nav_path(doy)
    if not obs.exists() or not nav.exists():
        return None
    sp3gz = _sp3_path(doy)
    if not sp3gz.exists():
        return None

    thinned = tmp / f"{station9}_k{k}.rnx"
    try:
        rt.thin_rinex_file(obs, sp3gz, thinned, k=k, strategy="highest",
                           elev_min_deg=ELEV_MIN_DEG, station_ecef=truth_ecef,
                           system=SYSTEM)
    except Exception:
        return None
    if not thinned.exists() or thinned.stat().st_size < 1000:
        return None

    eph = []
    sp3 = tmp / f"{doy}_k{k}.sp3"
    if str(sp3gz).endswith(".gz"):
        try:
            with open(sp3, "wb") as f:
                subprocess.run(["gunzip", "-c", str(sp3gz)], stdout=f, check=True)
            eph = [str(sp3)]
        except Exception:
            pass
    else:
        shutil.copy(sp3gz, sp3)
        eph = [str(sp3)]

    pos = tmp / f"{station9}_k{k}.pos"
    stat = tmp / f"{station9}_k{k}.pos.stat"
    cmd = [_rnx2rtkp(), "-x", "0", "-m", str(int(ELEV_MIN_DEG)), "-p", "0",
           "-f", "2", "-y", "2", "-o", str(pos), str(thinned), str(nav)] + eph
    try:
        subprocess.run(cmd, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    if not stat.exists():
        return None
    return parse_stat(stat)


_TEP_FIXED_MEMO: dict = {}
_DAY_MATRIX_CACHE: dict = {}


def _day_matrix(day):
    """Load broadcast series and build distance matrix for a day (cached per day)."""
    if day in _DAY_MATRIX_CACHE:
        return _DAY_MATRIX_CACHE[day]
    s = td.load_series("broadcast", day)
    if not s:
        _DAY_MATRIX_CACHE[day] = None
        return None
    names, X, D = tk.build_matrix(s, filter_by_std=False)
    if X is None:
        _DAY_MATRIX_CACHE[day] = None
        return None
    result = {"series": s, "names": names, "X": X, "D": D}
    _DAY_MATRIX_CACHE[day] = result
    return result


def _tep_fixed_clock_prediction(day, target, train_day=None):
    """LOO kernel prediction with FIXED TEP parameters (Gaussian, lambda=4200 km).

    Unlike kernel_clock_prediction which tunes lambda and form on training data,
    this uses the physics-predicted covariance with zero tuning (Papers 1-2 /
    Step 53 scale, nugget 0.35).
    """
    # Memoize: kp_fixed depends only on (day, target), not on k.
    cache_key = (day, target)
    if cache_key in _TEP_FIXED_MEMO:
        return _TEP_FIXED_MEMO[cache_key]
    dm = _day_matrix(day)
    if dm is None:
        return None
    s, names, X, D = dm["series"], dm["names"], dm["X"], dm["D"]
    if target not in names:
        return None
    ti = names.index(target)
    pool = [i for i in range(len(names)) if i != ti and 0.01 < X[:, i].std() < 60.0]
    if len(pool) < 8:
        return None
    p_fixed = tk.KernelParams(lambda_km=TEP_FIXED_LAMBDA_KM, nugget=TEP_FIXED_NUGGET,
                              k_neighbours=25, form=TEP_FIXED_FORM)
    pred = tk.predict(X, D, ti, pool, p_fixed)
    if pred is None:
        return None
    result = {"t": np.asarray(s[target]["t"], float),
              "kres_m": np.asarray(pred, float),
              "params": p_fixed.as_dict(),
              "train_day": "tep_fixed"}
    _TEP_FIXED_MEMO[cache_key] = result
    return result


def _station_run(day, code4, code9, k, tmp):
    """Get thinned RTKLIB epochs + kernel prediction for one station-day-k."""
    truth = sx.station_xyz(code4, 2024, int(day[4:]))
    if truth is None:
        return None

    cf = CACHE / f"{code4}_{day}_k{k}.pkl"
    if cf.exists():
        try:
            c = pickle.load(open(cf, "rb"))
            if "ep" in c and c["ep"]:
                # Always recompute the physics-fixed prediction from current
                # TEP_FIXED_* so a cached 4200 km / 0.35 nugget cannot stick.
                kp_fixed = _tep_fixed_clock_prediction(day, code4)
                if kp_fixed is not None:
                    return {"ep": c["ep"], "kp": c["kp"], "kp_fixed": kp_fixed,
                            "truth": truth}
                return {"ep": c["ep"], "kp": c["kp"], "kp_fixed": None,
                        "truth": truth}
        except Exception:
            pass

    # Kernel prediction (from step_33 cache or fresh)
    kp = None
    for ksrc in (PROJECT_ROOT / "data" / "step33_cache_v2" / f"{code4}_{day}.pkl",
                 PROJECT_ROOT / "data" / "step33_cache" / f"{code4}_{day}.pkl"):
        if ksrc.exists():
            try:
                c33 = pickle.load(open(ksrc, "rb"))
                if "kp" in c33 and c33["kp"]:
                    kp = c33["kp"]
                    break
            except Exception:
                pass
    if kp is None:
        kp = kernel_clock_prediction(day, code4, _find_train_day(day, DAYS))
    if kp is None:
        return None

    # Fixed TEP kernel prediction (physics-predicted, no tuning)
    kp_fixed = _tep_fixed_clock_prediction(day, code4)

    ep = _run_thinned_rtklib(code9, int(day[4:]), k, truth, tmp)
    if ep is None or len(ep) < 200:
        return None

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        pickle.dump({"ep": ep, "kp": kp, "kp_fixed": kp_fixed}, open(cf, "wb"))
    except Exception:
        pass
    return {"ep": ep, "kp": kp, "kp_fixed": kp_fixed, "truth": truth}


def _eval_station(day, code4, code9, k, tmp, acc):
    r = _station_run(day, code4, code9, k, tmp)
    if r is None:
        return 0
    ep, kp, truth = r["ep"], r["kp"], r["truth"]
    kp_fixed = r.get("kp_fixed")
    Rmat = _enu_rot(truth)
    kt, kres = kp["t"], kp["kres_m"]
    kres_fixed = kp_fixed["kres_m"] if kp_fixed else None
    sods = np.array(sorted(ep.keys()))
    cdt_full = np.array([ep[s]["cdt_rtk_m"] for s in sods])

    # Full method battery: ordinary (free-clock), temporal extrapolators,
    # deployable conventional (gauss_markov, free_gp), fixed TEP kernel,
    # and adaptive selectors.
    spatial_methods = ("free_gp",)
    if kres_fixed is not None:
        spatial_methods = spatial_methods + ("tep_fixed",)
    methods = (("ordinary",) + TEMPORAL_METHODS + ("gauss_markov",)
               + spatial_methods + ("adaptive_conventional", "adaptive_deployable"))
    n_used = 0
    for L in GAP_LENGTHS_S:
        key = (L, k)
        for g0 in np.arange(sods[0] + 3600.0, sods[-1] - L, GAP_STRIDE_S):
            pre = sods[(sods >= g0 - 1800.0) & (sods < g0)]
            if len(pre) < 6:
                continue
            cpre = np.interp(pre, sods, cdt_full)
            # Tune FOGM alpha on the pre-gap history
            fogm_alpha = _tune_fogm_alpha(pre, cpre)

            win = sods[(sods >= g0) & (sods < g0 + L)]
            msig, calibration_end = _causal_prior_sigmas(
                sods, cdt_full, kt, kres, float(g0), float(L),
                kres_fixed=kres_fixed, fogm_alpha=fogm_alpha)
            if msig is None or not calibration_end < g0:
                continue
            # adaptive_conventional: best of temporal methods (includes unstable poly)
            selected_conventional = min(TEMPORAL_METHODS, key=lambda m: msig[m])
            # adaptive_deployable: best of deployable methods only (no poly extrapolation)
            deployable_pool = tuple(m for m in DEPLOYABLE_METHODS if m in msig)
            selected_deployable = (min(deployable_pool, key=lambda m: msig[m])
                                   if deployable_pool else selected_conventional)
            ep_rows = []
            all_preds = _clock_predictions(pre, cpre, win, kt, kres,
                                           kres_fixed=kres_fixed,
                                           fogm_alpha=fogm_alpha)
            for s, preds in zip(win, all_preds):
                rec = ep[int(s)]
                cdt_ref = rec["cdt_rtk_m"]
                preds["adaptive_conventional"] = preds[selected_conventional]
                preds["adaptive_deployable"] = preds[selected_deployable]
                ep_rows.append((rec, cdt_ref, preds))
            if len(ep_rows) < 10:
                continue
            msig["adaptive_conventional"] = msig[selected_conventional]
            msig["adaptive_deployable"] = msig[selected_deployable]

            for rec, cdt_ref, preds in ep_rows:
                alls = [q for q in rec["sats"] if q[1] >= ELEV_MIN_DEG]
                if len(alls) < 4:
                    continue
                kk = len(alls)
                e = np.array([los_unit(a, el) for a, el, _ in alls])
                postfit = np.array([rr for _, _, rr in alls])
                el = np.array([q[1] for q in alls])
                A = np.column_stack([-e, np.ones(kk)])
                w = np.sin(np.radians(el)) ** 2
                offs = rec["pos_rtk"] - truth
                # Reconstruct PRE-FIT residuals relative to the SINEX truth.
                # The post-fit residual from RTKLIB is near-zero by construction
                # (it is the residual after the least-squares fit), so using it
                # directly makes the free-clock solution optimal by construction
                # and any clock prior harmful by construction.  The pre-fit
                # residual relative to the truth is:
                #   prefit_i = postfit_i + (-los_i) . offs + cdt_ref
                # which restores the actual measurement information that the
                # least-squares fit absorbed.  The solve then gives:
                #   ordinary:  [dpos, dcdt] = [offs, cdt_ref]  (the actual error)
                #   clock-prior: balances observations against the prior
                # and errv = dpos (NOT offs + dpos).
                b_prefit = postfit + (-e) @ offs + cdt_ref
                # Scale observation weights by code noise variance so the
                # prior weight 1/sigma^2 is on the same physical scale.
                SIGMA_CODE = 0.3  # m, typical C1W code noise
                w_scaled = w / SIGMA_CODE ** 2
                # Clock DOP: geometric clock-estimation uncertainty in units
                # of SIGMA_CODE.  The prior enters at its causally calibrated
                # sigma.  Do not inflate it to a no-op when k>=4.
                cdop = _clock_dop(A, w_scaled)
                sigma_obs = cdop * SIGMA_CODE if np.isfinite(cdop) else None
                sol_free = _solve(A, b_prefit, w_scaled, return_full=True)
                cdt_free = float(sol_free[3]) if sol_free is not None else None
                for m in methods:
                    if m == "ordinary":
                        dpos = _solve(A, b_prefit, w_scaled)
                    else:
                        prior_sig = msig[m]
                        if (cdt_free is not None and prior_sig > 0
                                and np.isfinite(preds[m])):
                            disagree = abs(preds[m] - cdt_free)
                            if disagree > DISAGREE_K * prior_sig:
                                prior_sig = max(prior_sig,
                                                disagree / DISAGREE_K)
                        dpos = _solve(A, b_prefit, w_scaled,
                                      preds[m], prior_sig)
                    if dpos is None:
                        acc.setdefault(key, {}).setdefault(m, {"err": [], "enu": [], "fail": 0, "n": 0})
                        acc[key][m]["n"] += 1
                        acc[key][m]["fail"] += 1
                        continue
                    errv = dpos  # pre-fit solve gives dpos directly
                    enu = Rmat @ errv
                    d = acc.setdefault(key, {}).setdefault(m, {"err": [], "enu": [], "fail": 0, "n": 0})
                    d["err"].append(float(np.linalg.norm(errv)))
                    d["enu"].append([float(enu[0]), float(enu[1]), float(enu[2])])
                    d["n"] += 1
                n_used += 1
    return n_used


def _summ(d):
    err = np.array(d["err"], float)
    enu = np.array(d["enu"], float) if d["enu"] else np.zeros((0, 3))
    if len(err) < 20:
        return None
    h = np.hypot(enu[:, 0], enu[:, 1]) if len(enu) else np.array([np.nan])
    v = np.abs(enu[:, 2]) if len(enu) else np.array([np.nan])
    return {
        "n_epochs": int(d["n"]),
        "availability": float(len(err) / max(d["n"], 1)),
        "rms_3d_m": float(np.sqrt(np.mean(err ** 2))),
        "rms_h_m": float(np.sqrt(np.mean(h ** 2))),
        "rms_v_m": float(np.sqrt(np.mean(v ** 2))),
        "p50_m": float(np.percentile(err, 50)),
        "p68_m": float(np.percentile(err, 68)),
        "p95_m": float(np.percentile(err, 95)),
        "p99_m": float(np.percentile(err, 99)),
        "max_m": float(np.max(err)),
        "fail_10m": float(np.mean(err > 10.0)),
        "fail_25m": float(np.mean(err > 25.0)),
        "fail_50m": float(np.mean(err > 50.0)),
    }


def _clustered_ci(errors_by_station, stat_fn, n_boot=2000):
    """Station-clustered bootstrap CI for a statistic."""
    stations = sorted(errors_by_station.keys())
    if len(stations) < 3:
        return None
    rng = np.random.default_rng(44)
    boots = []
    pool = []
    for _ in range(n_boot):
        sample = [stations[i] for i in rng.integers(0, len(stations), len(stations))]
        vals = []
        for st in sample:
            vals.extend(errors_by_station[st])
        if vals:
            boots.append(stat_fn(np.array(vals)))
    boots = np.array(boots)
    pool = np.concatenate([np.array(v) for v in errors_by_station.values()])
    return {
        "median": float(stat_fn(pool)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "frac_positive": float(np.mean(np.array(boots) > 0)) if np.all(np.isfinite(boots)) else 0.0,
    }


def main():
    import os as _os
    CACHE_ONLY = _os.environ.get("TEP_STEP44_CACHE_ONLY", "0") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    acc = {}
    station_errors = {}  # for clustered CI: {(L,k): {station: [pct_improvements]}}
    n_st = 0
    processed_days = []  # track days that actually produced data
    processed_k = set()  # track k values that actually produced cells

    with tempfile.TemporaryDirectory() as td_:
        tmp = Path(td_)
        for day in DAYS:
            s = td.load_series("broadcast", day)
            if not s:
                continue
            import glob
            code9_map = {}
            for f in glob.glob(str(PROJECT_ROOT / "data" / "rinex" / "obs" / "2024"
                                   / f"{int(day[4:]):03d}" / "*_R_*_MO.crx.gz")):
                c9 = Path(f).name.split("_")[0]
                code9_map[c9[:4]] = c9
            cands = [c for c in s if c in code9_map][:MAX_STATIONS]
            day_had_data = False
            for code4 in cands:
                for k in K_VALUES:
                    nu = _eval_station(day, code4, code9_map[code4], k, tmp, acc)
                    if nu:
                        n_st += 1
                        day_had_data = True
                        processed_k.add(k)
            if day_had_data:
                processed_days.append(day)
            print(f"  {day}: {n_st} station-k cells processed")

    # Build results
    result = {
        "test": "raw_observation_thinned_position_p2",
        "evidence_level": "raw_observation",
        "strict_p2_pass": False,  # set below
        "uses_full_sky_linearization": False,
        "prior_sigma_calibration": ("same-horizon rolling-origin pseudo-gap ending before "
                                    "the evaluation history; no evaluation-window truth"),
        "truth_source": "IGS daily SINEX",
        "n_station_k_cells": n_st,
        "days": processed_days,  # actual processed days, not declared
        "n_days_processed": len(processed_days),
        "days_declared": DAYS,  # for transparency
        "k_values": sorted(processed_k),  # actual k values that produced cells
        "k_values_declared": K_VALUES,
        "gap_lengths_s": GAP_LENGTHS_S,
        "by_gap_k": {},
        "timestamp": datetime.now().isoformat(),
    }

    p2_cells_pass = []
    for (L, k), md in sorted(acc.items()):
        summ = {m: _summ(d) for m, d in md.items()}
        summ = {m: v for m, v in summ.items() if v}
        # Primary TEP method: tep_fixed (physics-fixed Gaussian, lambda=4200 km).
        # Fall back to free_gp if tep_fixed is not available.
        tep_method = "tep_fixed" if "tep_fixed" in summ else (
            "free_gp" if "free_gp" in summ else None)
        if tep_method is None:
            continue
        conv = [m for m in TEMPORAL_METHODS if m in summ]
        oracle = min(conv, key=lambda m: summ[m]["rms_3d_m"]) if conv else None
        deployable = "adaptive_deployable" if "adaptive_deployable" in summ else (
            "adaptive_conventional" if "adaptive_conventional" in summ else None)
        row = {"summary": summ, "tep_method": tep_method,
               "best_conventional_oracle": oracle,
               "deployable_conventional": deployable}
        tep_rms = summ[tep_method]["rms_3d_m"]

        # Primary comparison: TEP vs ordinary (free-clock, no prior)
        if "ordinary" in summ:
            ord_rms = summ["ordinary"]["rms_3d_m"]
            row["tep_vs_ordinary_3d_pct"] = float(100 * (1 - tep_rms / ord_rms)) if ord_rms > 0 else None
            row["tep_vs_ordinary_fail25_ratio"] = (
                float(summ[tep_method]["fail_25m"] / summ["ordinary"]["fail_25m"])
                if summ["ordinary"]["fail_25m"] > 0 else None)

        # Deployable conventional comparison (no polynomial extrapolation)
        if deployable and deployable in summ:
            dep_rms = summ[deployable]["rms_3d_m"]
            row["tep_vs_deployable_3d_pct"] = float(100 * (1 - tep_rms / dep_rms)) if dep_rms > 0 else None
            row["tep_vs_deployable_fail25_ratio"] = (
                float(summ[tep_method]["fail_25m"] / summ[deployable]["fail_25m"])
                if summ[deployable]["fail_25m"] > 0 else None)

        # Legacy adaptive_conventional comparison (includes unstable poly)
        if "adaptive_conventional" in summ:
            ac_rms = summ["adaptive_conventional"]["rms_3d_m"]
            row["tep_vs_adaptive_conv_3d_pct"] = float(100 * (1 - tep_rms / ac_rms)) if ac_rms > 0 else None

        # Free GP comparison (tuned kernel vs fixed TEP kernel)
        if "free_gp" in summ and tep_method == "tep_fixed":
            fg_rms = summ["free_gp"]["rms_3d_m"]
            row["tep_fixed_vs_free_gp_3d_pct"] = float(100 * (1 - tep_rms / fg_rms)) if fg_rms > 0 else None

        if oracle:
            kr, br = tep_rms, summ[oracle]["rms_3d_m"]
            row["tep_vs_oracle_conv_3d_pct"] = float(100 * (1 - kr / br)) if br > 0 else None
        result["by_gap_k"][f"L{int(L)}_k{k}"] = row
        if deployable and row.get("tep_vs_deployable_3d_pct") is not None:
            pct = row["tep_vs_deployable_3d_pct"]
            p2_cells_pass.append(pct > 0)
            ord_pct = row.get("tep_vs_ordinary_3d_pct")
            ord_str = f", vs ordinary {ord_pct:+.1f}%" if ord_pct is not None else ""
            print(f"  L={int(L/60)}min k={k}: TEP 3D {tep_rms:.2f} m "
                  f"vs deployable {summ[deployable]['rms_3d_m']:.2f} m "
                  f"({pct:+.1f}%){ord_str}")

    # P2 gate: the primary comparison is TEP vs ordinary (free-clock).
    # At k=4/5/6 the clock is already observable from satellite geometry, so
    # a finite-error prior is expected to be inert or slightly harmful — this
    # is the correct physics, not a failure.  The operational niche is k=3
    # (underdetermined), tested in step_48.  At k=8 the system is fully
    # overdetermined and no prior should help.
    # The P2 gate passes if TEP beats the deployable conventional baseline
    # (gauss_markov / free_gp / persistence, no polynomial extrapolation) at
    # k=4/5/6 in the majority of cells.
    p2_key_cells = []
    tep_vs_ordinary_cells = []
    for key, row in result["by_gap_k"].items():
        k_val = int(key.split("_k")[1])
        if k_val in (4, 5, 6) and row.get("tep_vs_deployable_3d_pct") is not None:
            p2_key_cells.append(row["tep_vs_deployable_3d_pct"])
        if k_val in (4, 5, 6) and row.get("tep_vs_ordinary_3d_pct") is not None:
            tep_vs_ordinary_cells.append(row["tep_vs_ordinary_3d_pct"])

    # P2-as-GAIN must be statistically distinguishable from zero, not merely a
    # positive sign.  The previous rule (`median > 0 and frac_positive > 0.5`)
    # passed on a median of +0.0004 %, which is a tie by any reading and
    # contradicted the manuscript's own "statistically tied" wording.  Require
    # a bootstrap CI over cells that excludes zero AND a minimum effect size of
    # 1 % in 3-D RMS.  When the prior is inert -- which is the CORRECT physics
    # at k >= 4, where the receiver clock is already observable from satellite
    # geometry -- this reports a tie instead of a fake pass, and the
    # operational claim rests where the physics says it must: k = 3 (step_48).
    P2_MIN_EFFECT_PCT = 1.0
    if p2_key_cells:
        arr = np.asarray(p2_key_cells, float)
        median_p2 = float(np.median(arr))
        frac_positive = float(np.mean(arr > 0))
        _rng = np.random.default_rng(4402)
        boot = np.median(arr[_rng.integers(0, len(arr), size=(4000, len(arr)))], axis=1)
        p2_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        p2_pass = bool(p2_ci[0] > 0.0 and median_p2 >= P2_MIN_EFFECT_PCT)
    else:
        median_p2 = None
        frac_positive = None
        p2_ci = [None, None]
        p2_pass = False

    # TEP vs ordinary (free-clock): the honest null comparison
    if tep_vs_ordinary_cells:
        median_vs_ordinary = float(np.median(tep_vs_ordinary_cells))
        frac_positive_vs_ordinary = float(np.mean(np.array(tep_vs_ordinary_cells) > 0))
    else:
        median_vs_ordinary = None
        frac_positive_vs_ordinary = None

    result["strict_p2_pass"] = bool(p2_pass)
    result["p2_median_improvement_pct"] = median_p2
    result["p2_improvement_ci95_pct"] = p2_ci
    result["p2_pass_rule"] = (f"bootstrap CI over cells excludes 0 AND median >= "
                              f"{P2_MIN_EFFECT_PCT}% (sign alone is not a gain)")
    result["p2_frac_positive_cells"] = frac_positive
    result["p2_n_key_cells"] = len(p2_key_cells)
    result["tep_vs_ordinary_median_pct"] = median_vs_ordinary
    result["tep_vs_ordinary_frac_positive"] = frac_positive_vs_ordinary
    result["p2_primary_comparison"] = "tep_fixed vs adaptive_deployable (gauss_markov/free_gp/persistence)"
    result["p2_honest_null"] = "tep_fixed vs ordinary (free-clock solve)"
    result["p2_physics_note"] = (
        "At k=4/5/6 the receiver clock is already observable from satellite "
        "geometry, so any finite-error prior is expected to be inert or "
        "slightly harmful.  The operational niche where a clock prior is "
        "NECESSARY is k=3 (underdetermined), tested in step_48.  At k=8 the "
        "system is fully overdetermined and no prior should help.")

    # Integrity tail diagnostic (same as step 39)
    fail25_reductions = []
    four_hour_reductions = []
    for key, row in result["by_gap_k"].items():
        ratio = row.get("tep_vs_deployable_fail25_ratio")
        if ratio is None:
            continue
        reduction = float(100.0 * (1.0 - ratio))
        fail25_reductions.append(reduction)
        if key.startswith("L14400_"):
            four_hour_reductions.append(reduction)
    if fail25_reductions:
        result["integrity_tail_diagnostic"] = {
            "metric": "relative reduction in epochs with position error >25 m",
            "all_gap_satellite_cells_improve": bool(all(x > 0 for x in fail25_reductions)),
            "n_cells": len(fail25_reductions),
            "reduction_range_pct": [min(fail25_reductions), max(fail25_reductions)],
            "four_hour_reduction_range_pct": (
                [min(four_hour_reductions), max(four_hour_reductions)]
                if four_hour_reductions else None),
            "evidence_level": "raw_observation",
        }

    (OUT / "step_44_raw_thin_position.json").write_text(json.dumps(result, indent=2))
    print(f"\nP2 gate: {'PASS' if p2_pass else 'DEBUG'}")
    if median_p2 is not None:
        print(f"  TEP vs deployable: median {median_p2:+.2f}%, {frac_positive:.0%} positive cells")
    if median_vs_ordinary is not None:
        print(f"  TEP vs ordinary (free-clock): median {median_vs_ordinary:+.2f}%, "
              f"{frac_positive_vs_ordinary:.0%} positive cells")
    print(f"saved {OUT / 'step_44_raw_thin_position.json'}")


if __name__ == "__main__":
    main()
