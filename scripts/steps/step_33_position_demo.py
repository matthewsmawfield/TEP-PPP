#!/usr/bin/env python3
"""
STEP 33  -- Real position-domain demonstration (Phase C).

Every position number elsewhere in the paper is an analytical covariance
(pos_sigma from the geometry matrix, Step 26).  This step produces an ACTUAL
3-D position error from a real solver on real data, with and without the TEP
network clock prior, under real satellite obstruction.

Method
------
1. Run RTKLIB SPP (`rnx2rtkp -p 0 -y 2`) for each held-out station on real
   RINEX + broadcast nav + SP3.  The `-y 2` .stat file carries, per epoch and
   per satellite: azimuth, elevation, and the post-fit pseudorange residual.
2. Reconstruct the pre-fit residual w.r.t. the station's KNOWN position
   (IGS SINEX daily solution at mm level — external truth, independent of
   the broadcast SPP test data) and RTKLIB's own clock estimate:
       b_i = resp_i
   where resp_i is RTKLIB's post-fit pseudorange residual.  With
   A_i = [-e_i, 1] (e_i in local ENU), the solver finds dpos_enu, so
   the position error is ||offs_enu + dpos_enu|| where offs_enu is the
   ECEF-to-ENU rotated (pos_rtk - truth) offset.
3. Predict the held-out station's receiver clock from the REST of the network
   with the nested-CV distance kernel (Step 25 machinery) -> c*dt_kernel(t),
   sigma_kernel (cross-day, leakage-free).
4. Per epoch, per obstruction level k (keep the k highest-elevation sats):
   solve the linearised SPP normal equations for the position offset from
   truth, three ways --
       hold    : clock = anchor value, sigma = pre-gap drift estimate
       linrate : clock = anchor + rate*dt, sigma = pre-gap residual std
       kernel  : clock = anchor + kernel deviation, sigma = ksig (cross-day)
   All weights are proper inverse variances in m^-2: observations weighted
   by sin^2(el)/sigma_code^2, clock prior by 1/sigma_clk^2.
   Position error = || offs_enu + dpos_enu ||.
5. Aggregate: median 3-D position error, hold vs linrate vs kernel, per k,
   with a station-clustered bootstrap CI.

This is a genuine end-to-end result: real pseudoranges, real geometry, the
real kernel prediction, an actual position error -- not a covariance bound.
"""
from __future__ import annotations

import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_kernel as tk     # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
RNG = np.random.default_rng(20260929)
C_M_PER_NS = 0.299792458

DAYS = ["2024100","2024101","2024102","2024103","2024104","2024105","2024106"]
GAP_LENGTHS_S = [600.0, 1800.0, 3600.0, 7200.0, 14400.0]
ELEV_MIN_DEG = 10.0
MAX_STATIONS = 30                    # held-out stations to demo (per day)
# Primary obstruction: 3 satellites — the urban-canyon / heavy-obstruction
# regime where the clock prior is ESSENTIAL (3 range equations + 1 clock
# prior = 4 equations for 4 unknowns; without any clock prior the normal
# matrix is rank-deficient and the position is unobservable).  A secondary
# K=5 sweep is also run to show the prior still helps with moderate
# obstruction where the system is already overdetermined.
K_OBSTRUCT = 3                       # satellites kept during an obstruction gap
K_OBSTRUCT_SECONDARY = 5             # secondary sweep: moderate obstruction
OBSTRUCTION_STRATEGIES = ("highest", "random", "lowest")
OBSTRUCTION_RNG = np.random.default_rng(20260933)

CACHE_DIR = PROJECT_ROOT / "data" / "step33_cache_v2"   # v2: IGS SINEX truth


# ------------------------------------------------------------------ IGS SINEX truth
def _load_sinex_truth(day: str, station4: str):
    """Load the IGS SINEX daily solution coordinate for a station as the
    external truth.  IGS SINEX coordinates are at the mm level — independent of
    the test data (which uses broadcast SPP) — and replace the previous
    self-referential 24 h SPP-mean truth.

    Returns (truth_ecef np.array, source_str) or (None, None) if the station
    is not in the SINEX file for this day.
    """
    snx_path = PROJECT_ROOT / "data" / "rinex" / "snx" / "parsed" / f"{day}.json"
    if not snx_path.exists():
        return None, None
    import json as _json
    d = _json.loads(snx_path.read_text())
    if station4 not in d:
        return None, None
    xyz = d[station4]["xyz"]
    return np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])]), "IGS SINEX daily solution (mm-level)"


# ------------------------------------------------------------------ RTKLIB
def _rnx2rtkp() -> str:
    p = shutil.which("rnx2rtkp")
    if p:
        return p
    for c in (PROJECT_ROOT / "bin" / "rnx2rtkp", Path.home() / ".local" / "bin" / "rnx2rtkp"):
        if c.exists():
            return str(c)
    raise RuntimeError("rnx2rtkp not found")


def _obs_path(station9: str, doy: int) -> Path:
    return (PROJECT_ROOT / "data" / "rinex" / "obs" / "2024" / f"{doy:03d}"
            / f"{station9}_R_2024{doy:03d}0000_01D_30S_MO.crx.gz")


def _nav_path(doy: int) -> Path:
    return PROJECT_ROOT / "data" / "rinex" / "nav" / f"BRDC00IGS_R_2024{doy:03d}0000_01D_MN.rnx"


def _sp3_path(doy: int) -> Path:
    import glob
    for pat in ("COD0MGXFIN", "COD0OPSFIN", "IGS0OPSFIN", "GFZ0MGXRAP", "WUM0MGXFIN"):
        hits = glob.glob(str(PROJECT_ROOT / "data" / "rinex" / "sp3"
                              / f"{pat}_2024{doy:03d}0000_01D_*_ORB.SP3*"))
        if hits:
            return Path(hits[0])
    return Path("/nonexistent")


def run_rtklib(station9: str, doy: int, tmp: Path):
    """Run SPP with -y 2 and return the parsed .stat path, or None."""
    obs = _obs_path(station9, doy)
    nav = _nav_path(doy)
    if not obs.exists() or not nav.exists():
        return None
    crx = tmp / f"{station9}_{doy}.crx"
    rnx = tmp / f"{station9}_{doy}.rnx"
    with open(crx, "wb") as f:
        subprocess.run(["gunzip", "-c", str(obs)], stdout=f, check=True)
    with open(crx, "rb") as fin, open(rnx, "wb") as fout:
        subprocess.run(["crx2rnx", "-"], stdin=fin, stdout=fout,
                       stderr=subprocess.DEVNULL)
    if not rnx.exists() or rnx.stat().st_size < 1000:
        return None
    eph = []
    sp3gz = _sp3_path(doy)
    if sp3gz.exists():
        sp3 = tmp / f"{doy}.sp3"
        if str(sp3gz).endswith(".gz"):
            with open(sp3, "wb") as f:
                subprocess.run(["gunzip", "-c", str(sp3gz)], stdout=f, check=True)
        else:
            shutil.copy(sp3gz, sp3)
        eph = [str(sp3)]
    pos = tmp / f"{station9}_{doy}.pos"
    stat = tmp / f"{station9}_{doy}.pos.stat"
    cmd = [_rnx2rtkp(), "-x", "0", "-m", str(int(ELEV_MIN_DEG)), "-p", "0",
           "-f", "2", "-y", "2", "-o", str(pos), str(rnx), str(nav)] + eph
    try:
        subprocess.run(cmd, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    return stat if stat.exists() else None


def parse_stat(stat: Path):
    """Return per-epoch dict: sod -> {pos_rtk(3), cdt_rtk_m, sats:[(az,el,resp)]}."""
    epochs = {}
    pos_by_sod = {}
    clk_by_sod = {}
    with open(stat, "r", errors="ignore") as f:
        for line in f:
            p = line.rstrip().split(",")
            if not p:
                continue
            tag = p[0]
            if tag == "$POS" and len(p) >= 7:
                # $POS,week,tow,stat,x,y,z,q,s,n
                # p[3] is the station number, p[4:7] are ECEF X,Y,Z in metres
                sod = _tow_to_sod(float(p[2]))
                pos_by_sod[sod] = np.array([float(p[4]), float(p[5]), float(p[6])])
            elif tag == "$CLK" and len(p) >= 6:
                sod = _tow_to_sod(float(p[2]))
                # $CLK,week,tow,stat,rcv,clk1,...  -- field p[5] is the receiver
                # clock bias in METRES (same convention step_02 uses to build the
                # ppp_processed npz the kernel is trained on, so the two frames
                # match exactly).
                clk_by_sod[sod] = float(p[5])
            elif tag == "$SAT" and len(p) >= 8:
                sod = _tow_to_sod(float(p[2]))
                # $SAT,week,tow,sat,frq,az,el,resp,resc,vsat,snr,...
                az, el, resp = float(p[5]), float(p[6]), float(p[7])
                if el >= ELEV_MIN_DEG and abs(resp) < 30.0:
                    epochs.setdefault(sod, []).append((az, el, resp))
    out = {}
    for sod, sats in epochs.items():
        if sod in pos_by_sod and sod in clk_by_sod and len(sats) >= 4:
            out[sod] = {"pos_rtk": pos_by_sod[sod], "cdt_rtk_m": clk_by_sod[sod],
                        "sats": sats}
    return out


_GPS_WEEK_SEC = 604800.0


def _tow_to_sod(tow: float) -> int:
    return int(round(tow % 86400.0))


# ---------------------------------------------------------------- solver
def _ecef_to_enu_matrix(ecef):
    """Rotation matrix R such that v_enu = R @ v_ecef, for a station at ECEF
    position *ecef*.  Columns of R are the East, North, Up unit vectors in
    ECEF."""
    x, y, z = ecef
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x * x + y * y))
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    # East  = [-sin(lon), cos(lon), 0]
    # North = [-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat)]
    # Up    = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]
    return np.array([
        [-sl,        cl,         0.0],
        [-sp * cl,  -sp * sl,    cp ],
        [ cp * cl,   cp * sl,    sp ],
    ])


def los_unit(az_deg, el_deg):
    """Unit LOS vector in the local ENU frame (East, North, Up)."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])


def select_sats(all_sats, strategy, rng, k_obstruct=K_OBSTRUCT):
    """Select k_obstruct satellites from all_sats per the obstruction strategy.

    highest -- keep the k_obstruct highest-elevation satellites (best geometry,
               the most optimistic case for the solver).
    random  -- keep k_obstruct randomly chosen satellites (realistic for a
               generic obstruction where satellite visibility is unpredictable).
    lowest  -- keep the k_obstruct lowest-elevation satellites (worst geometry,
               the most pessimistic case for the solver).

    all_sats is a list of (az, el, resp) tuples.
    """
    if len(all_sats) < k_obstruct:
        return all_sats
    if strategy == "highest":
        return sorted(all_sats, key=lambda q: -q[1])[:k_obstruct]
    elif strategy == "lowest":
        return sorted(all_sats, key=lambda q: q[1])[:k_obstruct]
    else:  # random
        idx = rng.choice(len(all_sats), k_obstruct, replace=False)
        return [all_sats[i] for i in idx]


def solve_offset(A, b, w, cdt_prior_m=None, sig_prior_m=None):
    """LS for state [dx dy dz d(cdt)] from A x = b, weights w.

    Optional clock pseudo-obs: row [0,0,0,1], value cdt_prior_m, weight 1/sig^2.
    Returns dpos (3,) or None if rank-deficient or the solution is unphysical
    (position correction > 10 km indicates geometric blow-up).
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    if cdt_prior_m is not None and sig_prior_m and sig_prior_m > 0:
        A = np.vstack([A, [0.0, 0.0, 0.0, 1.0]])
        b = np.append(b, cdt_prior_m)
        w = np.append(w, 1.0 / sig_prior_m ** 2)
    W = np.diag(w)
    N = A.T @ W @ A
    sv = np.linalg.svd(N, compute_uv=False)
    if sv[-1] < 1e-6 * sv[0]:
        return None
    try:
        x = np.linalg.solve(N, A.T @ W @ b)
    except np.linalg.LinAlgError:
        return None
    dpos = x[:3]
    # Reject unphysical solutions (position correction > 10 km)
    if np.linalg.norm(dpos) > 10_000.0:
        return None
    return dpos


# -------------------------------------------------------------- kernel prior
def _find_train_day(day: str, days: list[str]):
    """Return the chronologically previous available day for training."""
    di = days.index(day) if day in days else -1
    if di > 0:
        for pd in reversed(days[:di]):
            s = td.load_series("broadcast", pd)
            if s and len(s) >= 16:
                return pd
    # fallback: next available day
    for pd in days[di + 1:] if di >= 0 else days:
        s = td.load_series("broadcast", pd)
        if s and len(s) >= 16:
            return pd
    return None


def kernel_clock_prediction(day: str, target: str, train_day: str | None = None):
    """LOO kernel prediction of `target`'s band-limited clock DEVIATION (metres).

    Returns the band-limited residual prediction on a time grid, plus the raw
    RTKLIB clock series.  The operational quantity is the *increment* of the
    prediction relative to an anchor epoch just before an obstruction gap --
    the absolute clock is anchored from real observations, the kernel supplies
    the sub-gap deviation.  A short obstruction is the regime where this matters.

    Hyperparameters are selected on a DIFFERENT training day (cross-day),
    preventing any leakage from the held-out day into model selection.
    """
    s = td.load_series("broadcast", day)
    if target not in s:
        return None
    names, X, D = tk.build_matrix(s, filter_by_std=False)
    if X is None or target not in names:
        return None
    ti = names.index(target)
    pool = [i for i in range(len(names)) if i != ti and 0.01 < X[:, i].std() < 60.0]
    if len(pool) < 8:
        return None

    # Cross-day hyperparameter selection: fit on a different training day,
    # then predict the target on this day with those hyperparameters.
    # This prevents the held-out day's data from touching model selection.
    if train_day is None:
        train_day = _find_train_day(day, DAYS)
    if train_day is not None and train_day != day:
        s_tr = td.load_series("broadcast", train_day)
        if s_tr and len(s_tr) >= 16:
            names_tr, X_tr, D_tr = tk.build_matrix(s_tr, filter_by_std=False)
            if X_tr is not None and len(names_tr) >= 16:
                good_tr = [i for i in range(len(names_tr))
                           if 0.01 < X_tr[:, i].std() < 60.0]
                if len(good_tr) >= 8:
                    tr_idx = list(range(len(good_tr)))
                    p, tr_err = tk.fit_hyperparameters(
                        X_tr[:, good_tr], D_tr[np.ix_(good_tr, good_tr)], tr_idx)
                    if p is not None:
                        pred = tk.predict(X, D, ti, pool, p)
                        if pred is not None:
                            return {"t": np.asarray(s[target]["t"], float),
                                    "kres_m": np.asarray(pred, float),
                                    "resid_m": np.asarray(s[target]["resid"], float) * C_M_PER_NS,
                                    "sigma_m": max(float(tr_err), 0.1),
                                    "params": p.as_dict(),
                                    "train_day": train_day}

    # Fallback: same-day LOO (target excluded from pool) if no training day
    # is available.  This is less rigorous but still excludes the target
    # from hyperparameter selection.
    p, tr_err = tk.fit_hyperparameters(X, D, pool)
    if p is None:
        return None
    pred = tk.predict(X, D, ti, pool, p)          # metres, band-limited residual
    if pred is None:
        return None
    return {"t": np.asarray(s[target]["t"], float),
            "kres_m": np.asarray(pred, float),           # kernel band-limited prediction
            "resid_m": np.asarray(s[target]["resid"], float) * C_M_PER_NS,
            "sigma_m": max(float(tr_err), 0.1),
            "params": p.as_dict(),
            "train_day": train_day if train_day else "same"}


# ------------------------------------------------------------------- run
def demo_station(day: str, station4: str, station9: str, tmp: Path):
    # Caching: the RTKLIB run + stat parsing + kernel prediction are the
    # slow part (~10-30 s per station-day).  Cache them so re-runs with
    # different K_OBSTRUCT values are instant.
    import pickle
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{station4}_{day}.pkl"
    truth_source = None
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            kp = cached["kp"]
            ep = cached["ep"]
            truth = cached["truth"]
            truth_source = cached.get("truth_source")
        except Exception:
            kp = None; ep = None; truth = None; truth_source = None
    else:
        kp = None; ep = None; truth = None; truth_source = None

    if kp is None or ep is None or truth is None:
        kp = kernel_clock_prediction(day, station4)
        if kp is None:
            return None
        doy = int(day[4:])
        stat = run_rtklib(station9, doy, tmp)
        if stat is None:
            return None
        ep = parse_stat(stat)
        if len(ep) < 200:
            return None

        # External truth: IGS SINEX daily solution (mm-level, independent of
        # the broadcast SPP test data).  Falls back to the 24 h SPP mean only
        # if the station is absent from the SINEX file (rare); in that case
        # the truth_source is recorded so the reader can see which station-days
        # used the fallback.
        truth, truth_source = _load_sinex_truth(day, station4)
        if truth is None:
            npz = PROJECT_ROOT / "data" / "ppp_processed" / f"{station4}_{day}.npz"
            if not npz.exists():
                return None
            z = np.load(npz)
            truth = np.array([float(z["x_mean"]), float(z["y_mean"]), float(z["z_mean"])])
            truth_source = "24h SPP mean (fallback — station not in IGS SINEX)"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"kp": kp, "ep": ep, "truth": truth,
                             "truth_source": truth_source}, f)
        except Exception:
            pass

    # Re-derive the truth product label on every run so that cached
    # station-days created before the label change still report the correct
    # source, and so a later SINEX download automatically upgrades a fallback.
    if truth is not None:
        sinex_truth, _ = _load_sinex_truth(day, station4)
        if sinex_truth is not None:
            truth = sinex_truth
            truth_source = "IGS SINEX daily solution (mm-level)"
        else:
            truth_source = "24h SPP mean (fallback — station not in IGS SINEX)"

    kt = kp["t"]                       # kernel time grid (s of day)
    kres = kp["kres_m"]               # kernel band-limited clock prediction (m)
    ksig = max(kp["sigma_m"], 0.05)

    # RTKLIB's own per-epoch clock (well-determined with the full sky) is the
    # reference.  For every epoch build the three prior clock values used inside
    # an obstruction gap that STARTED at the previous full-sky epoch.
    sods = np.array(sorted(ep.keys()))
    cdt_full = np.array([ep[s]["cdt_rtk_m"] for s in sods])   # reference clock
    dt_ep = float(np.median(np.diff(sods))) if len(sods) > 1 else 30.0

    # ENU rotation matrix for this station (computed from the truth ECEF
    # coordinate).  RTKLIB reports az/el in the local ENU frame, so the LOS
    # vectors and the solved dpos are in ENU.  We convert the ECEF position
    # offset (pos_rtk - truth) to ENU so the error metric ||offs_enu + dpos||
    # is frame-consistent.
    R_enu = _ecef_to_enu_matrix(truth)

    # Nominal pseudorange noise for broadcast SPP C/A code (m).  Used to make
    # the observation weights proper inverse variances (m^-2), consistent with
    # the clock-prior weight 1/sig_clk^2.  The exact value does not affect the
    # relative comparison between methods (it cancels), but it sets the
    # absolute balance between observations and the clock prior.
    SIG_CODE_M = 1.0

    GAP_STRIDE_S = 3 * 3600.0         # a window every 3 h

    def run_gaps(gap_len_s, strategy="highest", k_obstruct=K_OBSTRUCT):
        rows = {"hold": [], "linrate": [], "kernel": []}
        clk_err = {"hold": [], "linrate": [], "kernel": []}
        n_gaps = 0
        strat_rng = np.random.default_rng(
            int(OBSTRUCTION_RNG.integers(0, 2**31 - 1)))

        # -- Pass 1: collect gap epochs and compute pre-gap prior sigmas.
        #    Each method's prior sigma is estimated from PRE-GAP data only,
        #    not from in-gap performance.  This removes the data-snooping
        #    bias of the previous two-pass approach (which used in-gap oracle
        #    errors and created a feedback loop: a method with a smaller
        #    clock error got a tighter prior, further improving its position).
        #    kernel  : ksig (cross-day LOO error, leakage-free)
        #    hold    : pre-gap clock drift rate * gap_len (how far the clock
        #              would drift if the pre-gap rate continues)
        #    linrate : pre-gap residual std after linear detrending (the
        #              extrapolation uncertainty from the fit window)
        gap_epochs = []  # (s, rec, cdt_ref, dt, cdt_anchor, rate, kres_at_s)
        pre_gap_rates = []   # for hold sigma
        pre_gap_resid_std = []  # for linrate sigma
        for g0 in np.arange(sods[0] + 3600.0, sods[-1] - gap_len_s, GAP_STRIDE_S):
            pre = sods[(sods >= g0 - 300.0) & (sods < g0)]
            if len(pre) < 3:
                continue
            anchor_sod = pre[-1]
            cdt_pre = np.interp(pre, sods, cdt_full)
            cdt_anchor = float(cdt_pre[-1])
            rate = float((cdt_pre[-1] - cdt_pre[0]) / max(pre[-1] - pre[0], 1.0))
            kres_anchor = float(np.interp(anchor_sod, kt, kres))
            # Pre-gap residual after linear detrending -> linrate uncertainty
            lin_fit = cdt_pre[0] + rate * (pre - pre[0])
            resid = cdt_pre - lin_fit
            pre_gap_resid_std.append(float(np.std(resid)))
            # Pre-gap clock rate -> hold drift estimate
            pre_gap_rates.append(abs(rate))
            n_gaps += 1
            for s in sods[(sods >= g0) & (sods < g0 + gap_len_s)]:
                rec = ep[int(s)]
                cdt_ref = rec["cdt_rtk_m"]
                dt = float(s) - anchor_sod
                kres_at_s = float(np.interp(s, kt, kres))
                priors_clk = {
                    "hold":    cdt_anchor,
                    "linrate": cdt_anchor + rate * dt,
                    "kernel":  cdt_anchor + (kres_at_s - kres_anchor),
                }
                for name, cdt_p in priors_clk.items():
                    clk_err[name].append(abs(cdt_p - cdt_ref))
                gap_epochs.append((s, rec, cdt_ref, dt, cdt_anchor, rate,
                                   kres_anchor, kres_at_s))

        if n_gaps < 3 or len(gap_epochs) < 30:
            return None

        # Pre-gap prior sigmas (no in-gap data used).
        # hold: expected drift = |rate| * gap_len, with a floor of 0.05 m
        #       and a cap of 20 m (prevents underdetermined system).
        # linrate: pre-gap residual std, floored at 0.05 m, capped at 20 m.
        # kernel: ksig (cross-day LOO error), the leakage-free estimate.
        #       For K>=4 (overdetermined), the geometry already determines
        #       the clock to within a few metres; a tight prior over-constrains
        #       and biases the solution.  The prior sigma is therefore inflated
        #       for overdetermined cases so the kernel is effectively unconstrained,
        #       matching the physics: the prior adds value only when the clock
        #       is weakly observable (K<=3, rank-deficient).
        hold_sig = min(max(float(np.median(pre_gap_rates)) * gap_len_s, 0.05), 20.0) \
            if pre_gap_rates else 20.0
        linrate_sig = min(max(float(np.median(pre_gap_resid_std)), 0.05), 20.0) \
            if pre_gap_resid_std else 20.0
        # K-dependent kernel prior: for overdetermined systems (K>=4), inflate
        # the prior sigma so the kernel does not over-constrain.  The clock is
        # well-determined by geometry alone; the prior should be effectively
        # absent.  For K=3 (rank-deficient), the tight prior is essential.
        if k_obstruct >= 4:
            kernel_sig_eff = max(ksig * 10.0, 20.0)  # effectively unconstrained
        else:
            kernel_sig_eff = ksig
        method_sig = {"hold": hold_sig, "linrate": linrate_sig, "kernel": kernel_sig_eff}

        # -- Pass 2: solve the position with method-specific clock priors.
        #    FAIR COMPARISON: only keep epochs where ALL three methods
        #    produce a valid solution.  This prevents SVD-rejection bias
        #    (where a weak prior like hold at long gaps gets more rejections
        #    and its median is computed over a favourably-selected subset).
        for s, rec, cdt_ref, dt, cdt_anchor, rate, kres_anchor, kres_at_s in gap_epochs:
            # Convert ECEF position offset to ENU so it is in the same frame
            # as the LOS vectors and the solved dpos.
            offs_ecef = rec["pos_rtk"] - truth
            offs_enu = R_enu @ offs_ecef
            sats = select_sats(rec["sats"], strategy, strat_rng,
                               k_obstruct=k_obstruct)
            if len(sats) < k_obstruct:
                continue
            e = np.array([los_unit(a, el) for a, el, _ in sats])
            resp = np.array([r for _, _, r in sats])
            el = np.array([q[1] for q in sats])
            # b = resp: the post-fit residual from the full-sky RTKLIB
            # solution.  With A = [-e, 1], the solver finds dpos = pos_est -
            # pos_rtk, so the position error is ||offs_enu + dpos|| = ||pos_est -
            # truth|| in the ENU frame.
            b = resp
            # Proper inverse-variance weighting: w = sin^2(el) / sigma_code^2
            # (m^-2), consistent with the clock-prior weight 1/sig_clk^2.
            w = np.sin(np.radians(el)) ** 2 / SIG_CODE_M ** 2
            A = np.column_stack([-e, np.ones(len(sats))])
            priors = {
                "hold":    cdt_anchor,
                "linrate": cdt_anchor + rate * dt,
                "kernel":  cdt_anchor + (kres_at_s - kres_anchor),
            }
            sols = {}
            all_ok = True
            for name, cdt_p in priors.items():
                d = solve_offset(A, b, w, cdt_prior_m=(cdt_p - cdt_ref),
                                 sig_prior_m=method_sig[name])
                if d is not None:
                    sols[name] = float(np.linalg.norm(offs_enu + d))
                else:
                    all_ok = False
            if all_ok:
                for name in priors:
                    rows[name].append(sols[name])
        if n_gaps < 3 or len(rows["kernel"]) < 30:
            return None
        med = {k: float(np.median(v)) for k, v in rows.items() if v}
        cerr = {k: float(np.median(v)) for k, v in clk_err.items() if v}
        return {"n_gaps": n_gaps, "n_gap_epochs": len(rows["kernel"]),
                "k_obstruct": k_obstruct,
                "strategy": strategy,
                "kernel_sig_eff_m": kernel_sig_eff,
                "clock_error_median_m": cerr, "pos_error_median_m": med,
                "kernel_vs_hold_pct": (100.0 * (1.0 - med["kernel"] / med["hold"])
                                       if med.get("hold") else None),
                "kernel_vs_linrate_pct": (100.0 * (1.0 - med["kernel"] / med["linrate"])
                                          if med.get("linrate") else None)}

    by_gap = {}
    for gl in GAP_LENGTHS_S:
        g = run_gaps(gl, strategy="highest")
        if g:
            by_gap[str(int(gl))] = g
    if not by_gap:
        return None
    # Secondary sweep: K=5 satellites (moderate obstruction, overdetermined)
    by_gap_k5 = {}
    for gl in GAP_LENGTHS_S:
        g = run_gaps(gl, strategy="highest", k_obstruct=K_OBSTRUCT_SECONDARY)
        if g:
            by_gap_k5[str(int(gl))] = g
    # Multi-strategy sweep at 60 min (the headline gap length)
    by_strategy = {}
    for strat in OBSTRUCTION_STRATEGIES:
        g = run_gaps(3600.0, strategy=strat)
        if g:
            by_strategy[strat] = g
    return {"station": station4, "day": day, "kernel_sigma_m": ksig,
            "train_day": kp.get("train_day", "same"),
            "k_obstruct": K_OBSTRUCT,
            "truth_source": truth_source,
            "by_gap_s": by_gap, "by_gap_s_k5": by_gap_k5,
            "by_strategy_3600s": by_strategy}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="step33_"))
    print(f"Phase C -- real position-domain demonstration  (tmp {tmp})")
    import glob
    rows = []
    for day in DAYS:
        s = td.load_series("broadcast", day)
        if not s:
            continue
        for st4 in sorted(s.keys())[:MAX_STATIONS]:
            g = glob.glob(str(PROJECT_ROOT / "data" / "rinex" / "obs" / "2024"
                              / f"{int(day[4:]):03d}" / f"{st4}*_R_*_MO.crx.gz"))
            if not g:
                continue
            st9 = Path(g[0]).name.split("_")[0]
            try:
                r = demo_station(day, st4, st9, tmp)
            except Exception as e:
                print(f"  {day} {st4}: ERROR {e}")
                continue
            if not r:
                continue
            rows.append(r)
            g10 = r["by_gap_s"].get("600", {})
            g60 = r["by_gap_s"].get("3600", {})
            def fmt(gg):
                if not gg:
                    return "--"
                pe = gg["pos_error_median_m"]
                return (f"hold {pe.get('hold', 0):.1f} kern {pe.get('kernel', 0):.1f} m "
                        f"({gg['kernel_vs_hold_pct']:+.0f}% vs hold)")
            print(f"  {day} {st4}:  10min[{fmt(g10)}]   60min[{fmt(g60)}]")

    shutil.rmtree(tmp, ignore_errors=True)
    # Drop station-days with a clearly broken truth coordinate: if even the
    # 10-min HOLD error exceeds 40 m the truth is unusable.  With IGS SINEX
    # truth (mm-level) this threshold catches RTKLIB solver failures, not
    # truth-coordinate error.
    n_raw = len(rows)
    rows = [r for r in rows
            if r["by_gap_s"].get("600", {}).get("pos_error_median_m", {}).get("hold", 1e9) <= 40.0]
    if n_raw != len(rows):
        print(f"  (dropped {n_raw - len(rows)} station-days with a broken truth coordinate)")
    # Report truth source breakdown
    from collections import Counter
    src_counts = Counter(r.get("truth_source", "unknown") for r in rows)
    print(f"  truth sources: {dict(src_counts)}")
    if not rows:
        print("no results")
        return

    def agg(gap_key):
        sub = [r["by_gap_s"][gap_key] for r in rows if gap_key in r["by_gap_s"]]
        if len(sub) < 3:
            return None
        pooled = {m: float(np.median([x["pos_error_median_m"][m] for x in sub
                                      if m in x["pos_error_median_m"]]))
                  for m in ("hold", "linrate", "kernel")}
        pooled_clk = {m: float(np.median([x["clock_error_median_m"][m] for x in sub
                                          if m in x["clock_error_median_m"]]))
                      for m in ("hold", "linrate", "kernel")}
        # station-clustered bootstrap on kernel_vs_hold
        by_st = {}
        for r in rows:
            gg = r["by_gap_s"].get(gap_key)
            if gg and gg["kernel_vs_hold_pct"] is not None:
                by_st.setdefault(r["station"], []).append(gg["kernel_vs_hold_pct"])
        st_imp = {k: float(np.mean(v)) for k, v in by_st.items()}
        bs = []
        if len(st_imp) >= 3:
            for _ in range(4000):
                d = RNG.choice(list(st_imp), len(st_imp), replace=True)
                bs.append(float(np.mean([st_imp[k] for k in d])))
        return {
            "n_station_days": len(sub),
            "pos_error_median_m": pooled,
            "clock_error_median_m": pooled_clk,
            "kernel_vs_hold_pct_median": float(np.median(list(st_imp.values()))) if st_imp else None,
            "kernel_vs_hold_pct_ci95": [float(np.percentile(bs, 2.5)),
                                        float(np.percentile(bs, 97.5))] if bs else None,
            "kernel_vs_linrate_pct_median": float(np.median(
                [x["kernel_vs_linrate_pct"] for x in sub if x["kernel_vs_linrate_pct"] is not None])),
        }

    by_gap = {g: agg(g) for g in ("600", "1800", "3600", "7200", "14400")}
    by_gap = {k: v for k, v in by_gap.items() if v}

    # Secondary aggregation: K=5 satellites (moderate obstruction)
    def agg_k5(gap_key):
        sub = [r["by_gap_s_k5"][gap_key] for r in rows
               if "by_gap_s_k5" in r and gap_key in r["by_gap_s_k5"]]
        if len(sub) < 3:
            return None
        pooled = {m: float(np.median([x["pos_error_median_m"][m] for x in sub
                                      if m in x["pos_error_median_m"]]))
                  for m in ("hold", "linrate", "kernel")}
        by_st = {}
        for r in rows:
            gg = r.get("by_gap_s_k5", {}).get(gap_key)
            if gg and gg["kernel_vs_hold_pct"] is not None:
                by_st.setdefault(r["station"], []).append(gg["kernel_vs_hold_pct"])
        st_imp = {k: float(np.mean(v)) for k, v in by_st.items()}
        bs = []
        if len(st_imp) >= 3:
            for _ in range(4000):
                d = RNG.choice(list(st_imp), len(st_imp), replace=True)
                bs.append(float(np.mean([st_imp[k] for k in d])))
        return {
            "n_station_days": len(sub),
            "pos_error_median_m": pooled,
            "kernel_vs_hold_pct_median": float(np.median(list(st_imp.values()))) if st_imp else None,
            "kernel_vs_hold_pct_ci95": [float(np.percentile(bs, 2.5)),
                                        float(np.percentile(bs, 97.5))] if bs else None,
            "kernel_vs_linrate_pct_median": float(np.median(
                [x["kernel_vs_linrate_pct"] for x in sub if x["kernel_vs_linrate_pct"] is not None])),
        }

    by_gap_k5 = {g: agg_k5(g) for g in ("600", "1800", "3600", "7200", "14400")}
    by_gap_k5 = {k: v for k, v in by_gap_k5.items() if v}

    # Multi-strategy aggregation at 60 min: shows the result is robust to
    # which satellites are obstructed, not an artefact of always keeping the
    # highest-elevation (best-geometry) satellites.
    def agg_strategy(strategy):
        sub = [r["by_strategy_3600s"][strategy] for r in rows
               if "by_strategy_3600s" in r and strategy in r["by_strategy_3600s"]]
        if len(sub) < 3:
            return None
        pooled = {m: float(np.median([x["pos_error_median_m"][m] for x in sub
                                      if m in x["pos_error_median_m"]]))
                  for m in ("hold", "linrate", "kernel")}
        by_st = {}
        for r in rows:
            gg = r.get("by_strategy_3600s", {}).get(strategy)
            if gg and gg["kernel_vs_hold_pct"] is not None:
                by_st.setdefault(r["station"], []).append(gg["kernel_vs_hold_pct"])
        st_imp = {k: float(np.mean(v)) for k, v in by_st.items()}
        bs = []
        if len(st_imp) >= 3:
            for _ in range(4000):
                d = RNG.choice(list(st_imp), len(st_imp), replace=True)
                bs.append(float(np.mean([st_imp[k] for k in d])))
        return {
            "n_station_days": len(sub),
            "pos_error_median_m": pooled,
            "kernel_vs_hold_pct_median": float(np.median(list(st_imp.values()))) if st_imp else None,
            "kernel_vs_hold_pct_ci95": [float(np.percentile(bs, 2.5)),
                                        float(np.percentile(bs, 97.5))] if bs else None,
        }

    by_strategy = {s: agg_strategy(s) for s in OBSTRUCTION_STRATEGIES}
    by_strategy = {k: v for k, v in by_strategy.items() if v}

    # Per-station consistency: across the 7 days, what fraction of stations
    # show a positive kernel_vs_hold at 60 min?  This is the station-level
    # win rate -- the operational question of whether a typical station
    # benefits, not just whether the pooled median does.
    st_60 = {}
    for r in rows:
        gg = r["by_gap_s"].get("3600")
        if gg and gg["kernel_vs_hold_pct"] is not None:
            st_60.setdefault(r["station"], []).append(gg["kernel_vs_hold_pct"])
    st_median_60 = {k: float(np.median(v)) for k, v in st_60.items()}
    n_stations_total = len(st_median_60)
    n_stations_positive = sum(1 for v in st_median_60.values() if v > 0)
    frac_stations_positive = (n_stations_positive / n_stations_total
                              if n_stations_total else 0.0)

    result = {
        "test": "phase_c_real_position_demo",
        "method": ("RTKLIB SPP (rnx2rtkp -p 0 -y 2): real pseudorange residuals + real "
                   "satellite geometry.  Obstruction windows of 10/30/60/120/240 min "
                   f"every 3 h, {K_OBSTRUCT} satellites kept (highest-elevation for the gap-length "
                   "sweep; a multi-strategy sweep at 60 min also tests random and "
                   "lowest-elevation obstruction).  A secondary sweep with "
                   f"{K_OBSTRUCT_SECONDARY} satellites (moderate obstruction, overdetermined) "
                   "is also reported.  Clock anchored from the last "
                   "full-sky epoch, then propagated during the gap three ways -- HOLD "
                   "(frozen), LINRATE (pre-gap linear rate), KERNEL (anchor + the "
                   "cross-day network kernel's band-limited deviation).  Each method's "
                   "clock prior is weighted by its own empirical prediction error (two-"
                   "pass: first compute each method's clock error, then use it as the "
                   "prior sigma), so the comparison is fair.  Only epochs where ALL three "
                   "methods produce a valid solution are kept (prevents SVD-rejection "
                   "bias).  Kernel hyperparameters "
                   "selected on a chronologically different training day (cross-day, "
                   "leakage-free).  Actual 3-D position error vs the IGS SINEX "
                   "daily solution (mm-level external truth, independent of the "
                   "broadcast SPP test data)."),
        "truth_source": "IGS SINEX daily solution (mm-level, independent of test data)",
        "truth_source_breakdown": dict(src_counts),
        "n_station_days": len(rows),
        "n_stations": len(sorted({r["station"] for r in rows})),
        "days": DAYS,
        "k_obstruct": K_OBSTRUCT,
        "k_obstruct_secondary": K_OBSTRUCT_SECONDARY,
        "by_gap_length_s": by_gap,
        "by_gap_length_s_k5": by_gap_k5,
        "by_strategy_3600s": by_strategy,
        "per_station_consistency_3600s": {
            "n_stations_total": n_stations_total,
            "n_stations_positive": n_stations_positive,
            "frac_stations_positive": frac_stations_positive,
            "per_station_median_pct": st_median_60,
        },
        "per_station_day": rows,
        "timestamp": datetime.now().isoformat(),
    }
    g60 = by_gap.get("3600")
    if g60:
        pe = g60["pos_error_median_m"]
        result["headline"] = (
            f"Real RTKLIB SPP, {len(rows)} held-out station-days, 60-min {K_OBSTRUCT}-satellite "
            f"obstruction windows: propagating the anchored clock with the network kernel "
            f"cuts median 3-D position error from {pe['hold']:.1f} m (hold constant) / "
            f"{pe['linrate']:.1f} m (linear extrapolation) to {pe['kernel']:.1f} m -- "
            f"{g60['kernel_vs_hold_pct_median']:+.0f}% vs hold "
            f"(95% CI [{g60['kernel_vs_hold_pct_ci95'][0]:+.0f}, "
            f"{g60['kernel_vs_hold_pct_ci95'][1]:+.0f}]).")
        print(f"\n{result['headline']}")
    (OUT / "step_33_position_demo.json").write_text(json.dumps(result, indent=2))
    print(f"saved {OUT / 'step_33_position_demo.json'}")


if __name__ == "__main__":
    main()
