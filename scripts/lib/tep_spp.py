#!/usr/bin/env python3
"""
Minimal single-point-positioning front end for the three-satellite challenge.

Given a RINEX3 obs file, an SP3 orbit file, a 30 s CLK satellite-clock file, and
a FIXED reference position (IGS SINEX truth), compute the pre-fit pseudorange
residual for each GPS satellite at each epoch:

    b_i = P_IF_i  -  |r_sat_i(t - tau) - r_rx|  +  c*dt_sat_i
                   -  trop_dry_i  -  sagnac_i  -  rel_i

where P_IF is the ionosphere-free L1/L2 code combination (C1W, C2W).  b_i then
contains only:  c*dt_rx  +  trop_wet  +  multipath  +  noise.

With the receiver clock supplied by an external source (the terrestrial TEP
network) the position is  A [dx dy dz] = b - c*dt_rx  with A_i = -los_i, and
THREE satellites suffice.  With the clock free, four are required.

No iteration around a prior solver solution is involved -- the linearisation
point IS the surveyed truth, so a single least-squares step is the exact SPP
answer to first order (sub-mm for a metre-level residual).
"""
from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import numpy as np

C = 299_792_458.0
MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
F1, F2 = 1575.42e6, 1227.60e6           # GPS L1, L2
IF_A = F1 ** 2 / (F1 ** 2 - F2 ** 2)
IF_B = -F2 ** 2 / (F1 ** 2 - F2 ** 2)


# ------------------------------------------------------------------- RINEX obs
def read_rinex_clock_metadata(rnx_text: str):
    """Return receiver-clock metadata without interpreting it as event truth.

    RINEX 3.04 makes the epoch receiver-clock offset optional.  The
    ``RCV CLOCK OFFS APPL`` header says whether that reported offset was
    applied consistently to epoch, code and phase; it is not a log of
    oscillator-steering events.  Missing policy and missing epoch offsets are
    preserved as ``None`` so callers cannot silently treat them as a causal
    correction channel.
    """
    lines = rnx_text.splitlines()
    policy = None
    offsets: dict[int, float | None] = {}
    in_header = True
    for line in lines:
        if in_header:
            if "RCV CLOCK OFFS APPL" in line:
                try:
                    policy = int(line[:6].strip())
                except ValueError:
                    policy = None
            if "END OF HEADER" in line:
                in_header = False
            continue
        if not line.startswith(">"):
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            sod = (int(fields[4]) * 3600 + int(fields[5]) * 60
                   + int(round(float(fields[6]))))
            offsets[sod] = float(fields[9]) if len(fields) > 9 else None
        except (ValueError, IndexError):
            continue
    n_reported = sum(v is not None for v in offsets.values())
    return {
        "rcv_clock_offs_appl": policy,
        "epoch_clock_offsets_s": offsets,
        "n_epochs": len(offsets),
        "n_offsets_reported": n_reported,
        "offset_coverage": (n_reported / len(offsets)) if offsets else 0.0,
        "is_steering_event_log": False,
    }


def read_gps_code(rnx_text: str):
    """Return {sod: {prn: (C1W_or_C1C, C2W)}} for GPS from a RINEX3 obs string."""
    lines = rnx_text.splitlines()
    i = 0
    gps_types: list[str] = []
    while i < len(lines):
        ln = lines[i]
        if "SYS / # / OBS TYPES" in ln and ln[0] == "G":
            n = int(ln[3:6])
            toks = ln[7:60].split()
            j = i
            while len(toks) < n:
                j += 1
                toks += lines[j][7:60].split()
            gps_types = toks[:n]
            i = j
        if "END OF HEADER" in ln:
            i += 1
            break
        i += 1
    if not gps_types:
        return {}
    # column index of the code observables we want
    def idx(name):
        return gps_types.index(name) if name in gps_types else None
    i_c1 = idx("C1W") if "C1W" in gps_types else idx("C1C")
    i_c2 = (idx("C2W") if "C2W" in gps_types else
            (idx("C2L") if "C2L" in gps_types else idx("C2C")))
    out: dict[int, dict[str, tuple]] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            t = ln.split()
            sod = int(t[4]) * 3600 + int(t[5]) * 60 + int(round(float(t[6])))
            nsat = int(t[8])
            rec = {}
            for k in range(nsat):
                i += 1
                sl = lines[i]
                if not sl or sl[0] != "G":
                    continue
                prn = sl[:3].replace(" ", "0")
                fields = []
                for f in range(len(gps_types)):
                    seg = sl[3 + f * 16: 3 + f * 16 + 14].strip()
                    fields.append(float(seg) if seg else np.nan)
                c1 = fields[i_c1] if i_c1 is not None and i_c1 < len(fields) else np.nan
                c2 = fields[i_c2] if i_c2 is not None and i_c2 < len(fields) else np.nan
                if np.isfinite(c1) and np.isfinite(c2):
                    rec[prn] = (c1, c2)
            if rec:
                out[sod] = rec
        i += 1
    return out


def crx_gz_to_text(crx_gz: Path) -> str:
    raw = subprocess.run(["gunzip", "-c", str(crx_gz)], capture_output=True).stdout
    txt = subprocess.run(["crx2rnx", "-"], input=raw, capture_output=True).stdout
    return txt.decode("latin-1")


# ------------------------------------------------------------------- SP3 orbits
def read_sp3_gps(sp3_gz: Path):
    """{prn: (t_sod (n,), xyz_km (n,3), clk_us (n,))} for GPS."""
    op = gzip.open if str(sp3_gz).endswith(".gz") else open
    epochs, cur = [], None
    pos: dict[str, list] = {}
    clk: dict[str, list] = {}
    with op(sp3_gz, "rt") as f:
        for ln in f:
            if ln.startswith("*  "):
                p = ln.split()
                cur = int(p[4]) * 3600 + int(p[5]) * 60 + int(round(float(p[6])))
                epochs.append(cur)
            elif ln.startswith("PG") and cur is not None:
                prn = ln[1:4]
                x, y, z, c = (float(ln[4:18]), float(ln[18:32]),
                              float(ln[32:46]), float(ln[46:60]))
                if x == 0 and y == 0 and z == 0:
                    continue
                pos.setdefault(prn, []).append((cur, x, y, z))
                clk.setdefault(prn, []).append((cur, c))
    out = {}
    for prn in pos:
        a = np.array(pos[prn])
        cc = np.array(clk[prn])
        # SP3 files repeat SOD=0 at the day boundary (last epoch of the day is
        # written as 00:00:00 of the next day).  Keep only strictly-increasing
        # epochs so interpolation never sees a non-monotonic axis.
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
        a, cc = a[keep], cc[keep]
        order = np.argsort(a[:, 0])
        out[prn] = (a[order, 0], a[order, 1:4], cc[order, 1])
    return out


def read_clk_gps(clk_gz: Path):
    """{prn: (t_sod (n,), bias_s (n,))} GPS satellite clocks (30 s)."""
    op = gzip.open if str(clk_gz).endswith(".gz") else open
    d: dict[str, list] = {}
    with op(clk_gz, "rt") as f:
        for ln in f:
            if ln.startswith("AS G"):
                p = ln.split()
                prn = p[1]
                h, m, s = int(p[5]), int(p[6]), float(p[7])
                sod = h * 3600 + m * 60 + s
                d.setdefault(prn, []).append((sod, float(p[9])))
    out = {}
    for prn, v in d.items():
        a = np.array(v)
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
        a = a[keep]
        order = np.argsort(a[:, 0])
        out[prn] = (a[order, 0], a[order, 1])
    return out


# --------------------------------------------------------------- sat state
def _lagrange(t, tt, yy, order=10):
    j = np.searchsorted(tt, t)
    lo = max(0, j - order // 2)
    hi = min(len(tt), lo + order)
    lo = max(0, hi - order)
    xs, ys = tt[lo:hi], yy[lo:hi]
    val = 0.0
    for i in range(len(xs)):
        li = 1.0
        for k in range(len(xs)):
            if k != i:
                li *= (t - xs[k]) / (xs[i] - xs[k])
        val += ys[i] * li
    return val


def sat_state(sp3, prn, t_tx):
    """ECEF position (m) and velocity (m/s) of `prn` at transmit time t_tx."""
    tt, xyz, _ = sp3[prn]
    if t_tx < tt[0] + 1.0 or t_tx > tt[-1] - 1.0:
        return None, None
    dt = 1.0
    p0 = np.array([_lagrange(t_tx, tt, xyz[:, k]) for k in range(3)]) * 1000.0
    pp = np.array([_lagrange(t_tx + dt, tt, xyz[:, k]) for k in range(3)]) * 1000.0
    pm = np.array([_lagrange(t_tx - dt, tt, xyz[:, k]) for k in range(3)]) * 1000.0
    v = (pp - pm) / (2 * dt)
    return p0, v


def sat_clock(clk, prn, t):
    tt, cc = clk[prn]
    if t < tt[0] or t > tt[-1]:
        return None
    return float(np.interp(t, tt, cc))


# --------------------------------------------------------------- geometry
def _tropo_dry(el_rad, lat_rad, h_m):
    """Saastamoinen dry delay with a simple mapping (good to ~cm at el>10 deg)."""
    # Keep failed/nonlinear trial iterates outside the standard-atmosphere
    # singularity. Valid terrestrial receiver heights are comfortably inside
    # this range; an invalid iterate must never turn the normal equations NaN.
    h_m = float(np.clip(h_m, -500.0, 20_000.0))
    p = 1013.25 * (1 - 2.2557e-5 * h_m) ** 5.2568
    zhd = 0.0022768 * p / (1 - 0.00266 * np.cos(2 * lat_rad) - 2.8e-7 * h_m)
    m = 1.0 / (np.sin(el_rad) + 0.00143 / (np.tan(el_rad) + 0.0445))
    return zhd * m


def prefit(obs_epoch, sp3, clk, truth_ecef, sod):
    """Return list of (prn, los_unit(3), elev_deg, residual_m) for GPS sats
    with a valid dual-frequency code and orbit at this epoch."""
    r_rx = np.asarray(truth_ecef, float)
    lon = np.arctan2(r_rx[1], r_rx[0])
    lat = np.arctan2(r_rx[2], np.hypot(r_rx[0], r_rx[1]))
    h = np.linalg.norm(r_rx) - 6371000.0
    out = []
    for prn, (c1, c2) in obs_epoch.items():
        p_if = IF_A * c1 + IF_B * c2
        if not np.isfinite(p_if) or p_if < 1e7:
            continue
        # light-time iteration
        tau, ps_rx, vs_rx, rng, ok = 0.075, None, None, 0.0, True
        for _ in range(4):
            ps, vs = sat_state(sp3, prn, sod - tau)
            if ps is None:
                ok = False
                break
            th = OMEGA_E * tau
            rot = np.array([[np.cos(th), np.sin(th), 0.0],
                            [-np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
            ps_rx = rot @ ps
            vs_rx = rot @ vs
            rng = np.linalg.norm(ps_rx - r_rx)
            tau = rng / C
        if not ok:
            continue
        dts = sat_clock(clk, prn, sod - tau)
        if dts is None:
            continue
        los = (ps_rx - r_rx) / rng
        up = r_rx / np.linalg.norm(r_rx)
        el = np.arcsin(np.clip(np.dot(los, up), -1.0, 1.0))
        if el < np.radians(5.0):
            continue
        # relativistic eccentricity clock term  dt_rel = -2 (r . v) / c^2
        dt_rel = -2.0 * np.dot(ps_rx, vs_rx) / (C * C)
        trd = _tropo_dry(el, lat, h)
        # Niell-like wet mapping function (adequate above ~10 deg)
        se = np.sin(el)
        wet_map = 1.0 / (se + 0.00035 / (np.tan(el) + 0.017))
        resid = p_if - rng + C * (dts + dt_rel) - trd
        out.append((prn, -los, float(np.degrees(el)), float(resid), float(wet_map)))
    return out



# ------------------------------------------------ dense pre-interpolated grid
def dense_grid(sp3, clk, step=1.0, pad=900.0):
    """Pre-interpolate every GPS satellite onto a fine uniform time grid ONCE.

    Returns (t0, dt, {prn: (pos (n,3) m, vel (n,3) m/s, dts (n,) s)}).
    prefit_fast() then does O(1) linear look-ups instead of an O(order) Lagrange
    evaluation per satellite per epoch (the previous bottleneck)."""
    t_lo = min(v[0][0] for v in sp3.values()) + pad
    t_hi = max(v[0][-1] for v in sp3.values()) - pad
    tg = np.arange(t_lo, t_hi, step)
    out = {}
    for prn, (tt, xyz, _) in sp3.items():
        if prn not in clk:
            continue
        if len(tt) < 4:  # need at least 4 epochs for meaningful interpolation
            continue
        P = np.column_stack([_lagrange_vec(tg, tt, xyz[:, k]) for k in range(3)]) * 1000.0
        V = np.gradient(P, step, axis=0)
        ct, cc = clk[prn]
        D = np.interp(tg, ct, cc)
        out[prn] = (P, V, D)
    return float(t_lo), float(step), out


def _lagrange_vec(t, tt, yy, order=10):
    t = np.atleast_1d(np.asarray(t, float))
    res = np.empty_like(t)
    n = len(tt)
    eff_order = min(order, n)  # handle satellites with few epochs
    for m in range(len(t)):
        j = np.searchsorted(tt, t[m])
        lo = max(0, min(j - eff_order // 2, n - eff_order))
        xs, ys = tt[lo:lo + eff_order], yy[lo:lo + eff_order]
        L = np.ones(eff_order)
        for i in range(eff_order):
            for k in range(eff_order):
                if k != i:
                    L[i] *= (t[m] - xs[k]) / (xs[i] - xs[k])
        res[m] = ys @ L
    return res


def prefit_fast(obs_epoch, grid, truth_ecef, sod):
    """Same output as prefit() but using the dense pre-interpolated grid."""
    t0, dt, G = grid
    r_rx = np.asarray(truth_ecef, float)
    lon = np.arctan2(r_rx[1], r_rx[0])
    lat = np.arctan2(r_rx[2], np.hypot(r_rx[0], r_rx[1]))
    h = np.linalg.norm(r_rx) - 6371000.0
    out = []
    for prn, (c1, c2) in obs_epoch.items():
        if prn not in G:
            continue
        p_if = IF_A * c1 + IF_B * c2
        if not np.isfinite(p_if) or p_if < 1e7:
            continue
        P, V, D = G[prn]
        tau = 0.075
        ps_rx = None
        for _ in range(3):
            k = (sod - tau - t0) / dt
            ki = int(k)
            if ki < 0 or ki + 1 >= len(P):
                ps_rx = None
                break
            fr = k - ki
            ps = P[ki] * (1 - fr) + P[ki + 1] * fr
            vs = V[ki] * (1 - fr) + V[ki + 1] * fr
            th = OMEGA_E * tau
            rot = np.array([[np.cos(th), np.sin(th), 0.0],
                            [-np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
            ps_rx = rot @ ps
            vs_rx = rot @ vs
            rng = np.linalg.norm(ps_rx - r_rx)
            tau = rng / C
        if ps_rx is None:
            continue
        kk = (sod - tau - t0) / dt
        ki = int(kk); fr = kk - ki
        if ki < 0 or ki + 1 >= len(D):
            continue
        dts = D[ki] * (1 - fr) + D[ki + 1] * fr
        los = (ps_rx - r_rx) / rng
        up = r_rx / np.linalg.norm(r_rx)
        el = np.arcsin(np.clip(np.dot(los, up), -1.0, 1.0))
        if el < np.radians(5.0):
            continue
        dt_rel = -2.0 * np.dot(ps_rx, vs_rx) / (C * C)
        trd = _tropo_dry(el, lat, h)
        se = np.sin(el)
        wet_map = 1.0 / (se + 0.00035 / (np.tan(el) + 0.017))
        resid = p_if - rng + C * (dts + dt_rel) - trd
        out.append((prn, -los, float(np.degrees(el)), float(resid), float(wet_map)))
    return out


def prefit_fast_if(obs_epoch: dict, grid: tuple, truth_ecef: np.ndarray,
                   sod: float, elev_mask_deg: float = 5.0):
    """Prefit residuals from pre-computed ionosphere-free pseudoranges.

    Same output as ``prefit_fast`` but ``obs_epoch`` is ``{prn: p_if_m}`` —
    the ionosphere-free combination has already been applied by the caller
    with the correct per-system coefficients.  This is the multi-GNSS entry
    point: the grid is keyed by 'G01', 'E02', 'C06' etc. and the IF
    combination is system-specific.
    """
    t0, dt, G = grid
    r_rx = np.asarray(truth_ecef, float)
    lon = np.arctan2(r_rx[1], r_rx[0])
    lat = np.arctan2(r_rx[2], np.hypot(r_rx[0], r_rx[1]))
    h = np.linalg.norm(r_rx) - 6371000.0
    out = []
    for prn, p_if in obs_epoch.items():
        if prn not in G:
            continue
        if not np.isfinite(p_if) or p_if < 1e7:
            continue
        P, V, D = G[prn]
        tau = 0.075
        ps_rx = None
        for _ in range(3):
            k = (sod - tau - t0) / dt
            ki = int(k)
            if ki < 0 or ki + 1 >= len(P):
                ps_rx = None
                break
            fr = k - ki
            ps = P[ki] * (1 - fr) + P[ki + 1] * fr
            vs = V[ki] * (1 - fr) + V[ki + 1] * fr
            th = OMEGA_E * tau
            rot = np.array([[np.cos(th), np.sin(th), 0.0],
                            [-np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
            ps_rx = rot @ ps
            vs_rx = rot @ vs
            rng = np.linalg.norm(ps_rx - r_rx)
            tau = rng / C
        if ps_rx is None:
            continue
        kk = (sod - tau - t0) / dt
        ki = int(kk); fr = kk - ki
        if ki < 0 or ki + 1 >= len(D):
            continue
        dts = D[ki] * (1 - fr) + D[ki + 1] * fr
        los = (ps_rx - r_rx) / rng
        up = r_rx / np.linalg.norm(r_rx)
        el = np.arcsin(np.clip(np.dot(los, up), -1.0, 1.0))
        if el < np.radians(elev_mask_deg):
            continue
        dt_rel = -2.0 * np.dot(ps_rx, vs_rx) / (C * C)
        trd = _tropo_dry(el, lat, h)
        se = np.sin(el)
        wet_map = 1.0 / (se + 0.00035 / (np.tan(el) + 0.017))
        resid = p_if - rng + C * (dts + dt_rel) - trd
        out.append((prn, -los, float(np.degrees(el)), float(resid), float(wet_map)))
    return out


def pseudorange_model(rx_ecef, prn, grid, sod):
    """Compute the geometry-only predicted pseudorange (m) for a single PRN at
    a single receiver position.  Returns (p_pred, los, wet_map, el_deg, ok),
    where p_pred = range - C*(dts + dt_rel) + trd.  The caller adds the clock,
    ZWD and measurement to obtain the residual.

    This is the same physical model as `prefit_fast` and `prefit`, but split
    out so that a nonlinear solver can re-evaluate it at arbitrary positions.
    """
    r_rx = np.asarray(rx_ecef, float)
    rnorm = float(np.linalg.norm(r_rx))
    if not np.all(np.isfinite(r_rx)) or not (5.5e6 <= rnorm <= 7.5e6):
        return None, None, None, None, False
    lon = np.arctan2(r_rx[1], r_rx[0])
    lat = np.arctan2(r_rx[2], np.hypot(r_rx[0], r_rx[1]))
    h = rnorm - 6371000.0
    t0, dt, G = grid
    if prn not in G:
        return None, None, None, None, False
    P, V, D = G[prn]
    tau = 0.075
    ps_rx, vs_rx, rng = None, None, None
    for _ in range(4):
        k = (sod - tau - t0) / dt
        if not np.isfinite(k):
            return None, None, None, None, False
        ki = int(k)
        if ki < 0 or ki + 1 >= len(P):
            ps_rx = None
            break
        fr = k - ki
        ps = P[ki] * (1 - fr) + P[ki + 1] * fr
        vs = V[ki] * (1 - fr) + V[ki + 1] * fr
        th = OMEGA_E * tau
        rot = np.array([[np.cos(th), np.sin(th), 0.0],
                        [-np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        ps_rx = rot @ ps
        vs_rx = rot @ vs
        rng = np.linalg.norm(ps_rx - r_rx)
        tau = rng / C
    if ps_rx is None:
        return None, None, None, None, False
    kk = (sod - tau - t0) / dt
    ki = int(kk); fr = kk - ki
    if ki < 0 or ki + 1 >= len(D):
        return None, None, None, None, False
    dts = D[ki] * (1 - fr) + D[ki + 1] * fr
    los = (ps_rx - r_rx) / rng
    up = r_rx / (np.linalg.norm(r_rx) + 1e-9)
    el = np.arcsin(np.clip(np.dot(los, up), -1.0, 1.0))
    if el < np.radians(5.0):
        return None, None, None, None, False
    dt_rel = -2.0 * np.dot(ps_rx, vs_rx) / (C * C)
    trd = _tropo_dry(el, lat, h)
    se = np.sin(el)
    wet_map = 1.0 / (se + 0.00035 / (np.tan(el) + 0.017))
    p_pred = rng - C * (dts + dt_rel) + trd
    # Return receiver-to-satellite LOS. The range derivative w.r.t. receiver
    # position is its negative; solve_spp_nonlinear applies that sign once.
    return p_pred, los, wet_map, float(np.degrees(el)), True


def _pdop_from_jac(A):
    """Position DOP from the (possibly weighted) design matrix A.

    PDOP = sqrt(trace((A.T A)^-1)[:3,:3]) on the unweighted normal matrix.
    """
    try:
        N = A.T @ A
        sv = np.linalg.svd(N, compute_uv=False)
        if sv[-1] <= 1e-12 * sv[0]:
            return np.inf
        Q = np.linalg.inv(N)
        tr = float(np.trace(Q[:3, :3]))
        return float(np.sqrt(tr)) if tr > 0 else np.inf
    except np.linalg.LinAlgError:
        return np.inf


def solve_spp_nonlinear(obs, grid, sod, init_pos, clock=None, zwd=None,
                        clock_sigma=None, k=None, elev_mask=10.0,
                        max_iter=20, tol=0.01, pdop_max=20.0,
                        max_update_m=500_000.0, position_prior=None,
                        position_sigma=None):
    """Nonlinear SPP solver from raw pseudoranges and an optional clock prior.

    Parameters
    ----------
    obs : list of (prn, p_if_m)
        Ionosphere-free pseudorange observations.
    grid : tuple from dense_grid()
    sod : seconds of day.
    init_pos : (3,) ECEF starting position (m).
    clock : float or None
        Predicted receiver clock C*dt_rx in meters.  If None the clock is
        solved as a free parameter; if clock_sigma is None the clock is fixed.
    zwd : float or None
        If not None, the wet delay is fixed to this value (m).  If None it is
        estimated jointly (requires at least 5 satellites, or 4 if clock is
        fixed).
    clock_sigma : float or None
        Standard deviation of the clock prior in meters.  If provided, a soft
        clock-prior row is added to the normal equations.
    position_prior : (3,) or None
        Optional causal ECEF motion prediction.  This supports kinematic
        filtering experiments; it must never be populated from scoring truth.
    position_sigma : float, (3,), or None
        One-sigma uncertainty of ``position_prior`` in metres.
    k : int or None
        Use the k highest-elevation satellites from init_pos.  If None, use all
        satellites above elev_mask.
    elev_mask : degrees.
    max_iter : int.
    tol : metres for dx stopping.
    pdop_max : receiver integrity gate.
    max_update_m : largest allowed single-iteration update; larger moves are
        clamped and non-convergence is reported.

    Returns
    -------
    (pos, clk, zwd_out, converged, n_iter, pdop, ok) or None if the solver
    rejects the epoch.  pos is the final ECEF position; the returned error is
    the 3-D norm w.r.t. an external truth.
    """
    rx = np.asarray(init_pos, float, copy=True)
    clock_prior = None if clock is None else float(clock)
    clk = 0.0 if clock_prior is None else clock_prior
    zwd_out = 0.0 if zwd is None else float(zwd)
    pos_prior = (None if position_prior is None else
                 np.asarray(position_prior, float).reshape(3))
    if pos_prior is not None:
        psig_value = 10.0 if position_sigma is None else position_sigma
        psig = np.broadcast_to(np.asarray(psig_value, float), (3,)).copy()
        psig = np.maximum(psig, 1e-3)
    else:
        psig = None

    free_clk = clock is None
    soft_clk = clock is not None and clock_sigma is not None
    est_zwd = zwd is None

    def _build(only_select=False):
        recs = []
        for prn, p_if in obs:
            p_pred, los, wet_map, el, ok = pseudorange_model(rx, prn, grid, sod)
            if not ok or el < elev_mask or not np.isfinite(p_if):
                continue
            recs.append((prn, p_if, los, wet_map, el))
        if not recs:
            return None
        recs.sort(key=lambda r: -r[4])
        if k is not None and k < len(recs):
            recs = recs[:k]
        if only_select:
            return recs
        # form design and residual
        n_obs = len(recs)
        # columns: [dx, dy, dz, dclk?, dzwd?]
        col_names = ["dx", "dy", "dz"]
        if free_clk or soft_clk:
            col_names.append("dclk")
        if est_zwd:
            col_names.append("dzwd")
        n_state = len(col_names)
        A_rows, b_rows, w_rows = [], [], []
        for prn, p_if, los, wet_map, el in recs:
            p_pred, _, _, _, _ = pseudorange_model(rx, prn, grid, sod)
            # residual: observation - prediction - current clock - current zwd*wet
            row = np.zeros(n_state)
            # f(x+dx,clk+dclk,zwd+dzwd) ~= f(x,clk,zwd) + (-los).dx + dclk + wet_map dzwd
            row[:3] = -los
            if free_clk or soft_clk:
                row[3] = 1.0
            if est_zwd:
                row[-1] = wet_map
            A_rows.append(row)
            b_rows.append(p_if - (p_pred + clk + zwd_out * wet_map))
            w_rows.append(np.sin(np.radians(el)) ** 2)
        A = np.asarray(A_rows, float)
        b = np.asarray(b_rows, float)
        W = np.diag(w_rows)
        # Priors constrain the final state around a frozen prediction.  The
        # residual must be recomputed relative to that frozen prediction on
        # every nonlinear iteration; centring it on the current state would
        # merely penalise each increment and is not a valid Bayesian prior.
        prior_rows = []
        prior_rhs = []
        if soft_clk:
            prior_row = np.zeros(n_state)
            prior_row[3] = 1.0
            csig = max(clock_sigma, 1e-3)
            prior_rows.append(prior_row / csig)
            prior_rhs.append((clock_prior - clk) / csig)
        if pos_prior is not None:
            for axis in range(3):
                prior_row = np.zeros(n_state)
                prior_row[axis] = 1.0 / psig[axis]
                prior_rows.append(prior_row)
                prior_rhs.append((pos_prior[axis] - rx[axis]) / psig[axis])
        if prior_rows:
            A = np.vstack([A, np.asarray(prior_rows)])
            b = np.concatenate([b, np.asarray(prior_rhs)])
            W = np.eye(A.shape[0])
            W[:n_obs, :n_obs] = np.diag(w_rows)
        return recs, A, b, W, n_state, col_names

    def _geometric_pdop(recs):
        """PDOP from satellite geometry, independent of prior units.

        The former calculation included the soft clock-prior row after it had
        been divided by ``clock_sigma``.  That mixed metres with a dimensionless
        DOP and caused every three-satellite clock-constrained solution to be
        rejected merely because the prior was conservatively uncertain.
        """
        los_rows = np.array([-np.asarray(r[2], float) for r in recs])
        if free_clk and not (pos_prior is not None and len(recs) == 3):
            geom = np.column_stack((los_rows, np.ones(len(recs))))
        else:
            geom = los_rows
        return _pdop_from_jac(geom)

    for it in range(max_iter):
        ret = _build()
        if ret is None:
            return None
        recs, A, b, W, n_state, col_names = ret
        if len(recs) < 3:
            return None
        if _geometric_pdop(recs) > pdop_max:
            return None
        try:
            N = A.T @ W @ A
            sv = np.linalg.svd(N, compute_uv=False)
            if sv[-1] < 1e-12 * sv[0]:
                return None
            dx = np.linalg.solve(N, A.T @ W @ b)
        except np.linalg.LinAlgError:
            return None
        # clamp huge updates (cold start from very wrong a priori)
        if np.linalg.norm(dx[:3]) > max_update_m:
            scale = max_update_m / np.linalg.norm(dx[:3])
            dx[:3] *= scale
        candidate = rx + dx[:3]
        candidate_norm = float(np.linalg.norm(candidate))
        if (not np.all(np.isfinite(candidate)) or
                not (5.5e6 <= candidate_norm <= 7.5e6)):
            return None
        rx = candidate
        if free_clk or soft_clk:
            clk += dx[3]
        if est_zwd:
            zwd_out += dx[-1]
        if np.linalg.norm(dx) < tol:
            break
    else:
        # reached max_iter; accept if the last update was small-ish
        if np.linalg.norm(dx) > 10 * tol:
            return None

    pdop = _geometric_pdop(recs)
    if pdop > pdop_max:
        return None
    return rx, clk, zwd_out, True, it + 1, pdop, True


def raim_residual(pos, clk, zwd, obs_extra, grid, sod, elev_mask=10.0):
    """Post-fit pseudorange residual to a satellite NOT used in the solve.

    This is the integrity check for the k=3 + clock-prior regime: after
    solving position from 3 satellites and a TEP network clock prior, the
    residual to a 4th (partially visible) satellite tests whether the solved
    position + clock are consistent with an independent measurement.  A large
    residual indicates either a bad clock prior or degenerate geometry, and
    the solution should be rejected.

    Parameters
    ----------
    pos : (3,) ECEF position (m) from the solver.
    clk : float, receiver clock C*dt_rx in metres at the solved position.
    zwd : float or None, wet troposphere delay (m).
    obs_extra : list of (prn, p_if_m) -- the extra satellite(s) to check.
    grid, sod : orbit/clock grid and seconds-of-day.
    elev_mask : minimum elevation for the check satellite (degrees).

    Returns
    -------
    list of (prn, residual_m, elevation_deg) or None if no valid satellite.
    """
    results = []
    for prn, p_if in obs_extra:
        p_pred, _, wet_map, el, ok = pseudorange_model(pos, prn, grid, sod)
        if not ok or el < elev_mask or not np.isfinite(p_if):
            continue
        tr = zwd * wet_map if zwd is not None else 0.0
        residual = float(p_if - (p_pred + clk + tr))
        results.append((prn, residual, float(el)))
    return results if results else None


def solve_spp_multistart(obs, grid, sod, init_positions, clock=None,
                         zwd=None, clock_sigma=None, k=None,
                         elev_mask=10.0, pdop_max=20.0,
                         consistency_m=50.0):
    """Multi-start blind solve with convergence consistency check.

    Runs the nonlinear solver from multiple initial positions and accepts the
    solution only if all starts converge to within ``consistency_m`` of each
    other.  This bounds the catastrophic tail of the k=3 blind solve: when the
    3-satellite geometry is near-degenerate, different starting points converge
    to different basins, and the inconsistency is detected.

    Parameters
    ----------
    init_positions : list of (3,) ECEF arrays -- the different starting points.
    consistency_m : maximum allowed spread between converged solutions.

    Returns
    -------
    (median_pos, clk, zwd, n_converged, spread_m, pdop, ok) or None.
    """
    solutions = []
    for ip in init_positions:
        ret = solve_spp_nonlinear(obs, grid, sod, ip, clock=clock,
                                  zwd=zwd, clock_sigma=clock_sigma,
                                  k=k, elev_mask=elev_mask,
                                  pdop_max=pdop_max)
        if ret is not None:
            solutions.append(ret)
    if len(solutions) < max(2, len(init_positions) // 2):
        return None
    positions = np.array([s[0] for s in solutions], float)
    # Pairwise max distance between converged solutions
    from itertools import combinations
    max_dist = 0.0
    for i, j in combinations(range(len(positions)), 2):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if d > max_dist:
            max_dist = d
    if max_dist > consistency_m:
        return None
    # Accept the median position; use the clock from the closest-to-median solve
    median_pos = np.median(positions, axis=0)
    dists = np.linalg.norm(positions - median_pos, axis=1)
    best = solutions[int(np.argmin(dists))]
    return (median_pos, best[1], best[2], len(solutions), max_dist,
            best[5], True)


# ================================================================ multi-GNSS
# Per-system ionosphere-free coefficients and preferred code pairs.
# Each entry: (F1_Hz, F2_Hz, [primary_code_pairs]).
# The IF combination is F1^2/(F1^2-F2^2) * P1 - F2^2/(F1^2-F2^2) * P2.

GNSS_SYSTEMS: dict[str, dict] = {
    "G": {  # GPS
        "f1": 1575.42e6, "f2": 1227.60e6,
        "codes": [("C1W", "C2W"), ("C1C", "C2W"), ("C1C", "C2L"),
                  ("C1C", "C2C"), ("C1W", "C2L")],
    },
    "E": {  # Galileo — multiple frequency pairs possible
        "f1": 1575.42e6, "f2": 1176.45e6,  # E1 / E5a (default)
        "codes": [("C1C", "C5Q"), ("C1C", "C5X"), ("C1C", "C7Q"),
                  ("C1C", "C7X"), ("C1C", "C8Q")],
        # alternative frequency pairs (f2, code_pair)
        "alt_pairs": [
            (1207.140e6, [("C1C", "C7Q"), ("C1C", "C7X")]),  # E1 / E5b
            (1278.750e6, [("C1C", "C6C"), ("C1C", "C6X")]),  # E1 / E6
        ],
    },
    "C": {  # BeiDou — multiple frequency pairs possible
        "f1": 1561.098e6, "f2": 1207.140e6,  # B1I / B2I (default)
        "codes": [("C1P", "C2I"), ("C1I", "C2I"), ("C1P", "C6I"),
                  ("C1I", "C6I"), ("C1P", "C7I"), ("C1P", "C7D")],
        "alt_pairs": [
            (1575.420e6, [("C1X", "C2X"), ("C1C", "C2X")]),  # B1C / B2a
            (1268.520e6, [("C2I", "C7I"), ("C2P", "C7I")]),  # B2I / B3I
        ],
    },
}


def _if_coeffs(f1: float, f2: float) -> tuple[float, float]:
    """Ionosphere-free coefficients for two frequencies."""
    return f1 ** 2 / (f1 ** 2 - f2 ** 2), -f2 ** 2 / (f1 ** 2 - f2 ** 2)


def read_sp3_multi(sp3_path: Path,
                   systems: tuple[str, ...] = ("G", "E", "C")) -> dict:
    """Multi-GNSS SP3 reader.  Returns {prn: (t_sod, xyz_km (n,3), clk_us (n,))}.

    PRNs are keyed as 'G01', 'E02', 'C06' etc.  Only the requested systems
    are parsed.  Satellite clock values from the SP3 are kept but the
    dedicated CLK file is preferred when available.
    """
    op = gzip.open if str(sp3_path).endswith(".gz") else open
    pos: dict[str, list] = {}
    clk: dict[str, list] = {}
    cur = None
    with op(sp3_path, "rt") as f:
        for ln in f:
            if ln.startswith("*  "):
                p = ln.split()
                cur = int(p[4]) * 3600 + int(p[5]) * 60 + int(round(float(p[6])))
            elif ln.startswith("P") and cur is not None and ln[1] in systems:
                prn = ln[1:4]  # e.g. "G01", "E02", "C06"
                try:
                    x, y, z, c = (float(ln[4:18]), float(ln[18:32]),
                                  float(ln[32:46]), float(ln[46:60]))
                except ValueError:
                    continue
                if x == 0 and y == 0 and z == 0:
                    continue
                pos.setdefault(prn, []).append((cur, x, y, z))
                clk.setdefault(prn, []).append((cur, c))
    out = {}
    for prn in pos:
        a = np.array(pos[prn])
        cc = np.array(clk[prn])
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
        a, cc = a[keep], cc[keep]
        order = np.argsort(a[:, 0])
        out[prn] = (a[order, 0], a[order, 1:4], cc[order, 1])
    return out


def read_clk_multi(clk_path: Path,
                   systems: tuple[str, ...] = ("G", "E", "C")) -> dict:
    """Multi-GNSS CLK reader.  Returns {prn: (t_sod, bias_s)}.

    Parses 'AS G', 'AS E', 'AS C' records.  PRNs keyed as 'G01', 'E02', etc.
    """
    op = gzip.open if str(clk_path).endswith(".gz") else open
    d: dict[str, list] = {}
    with op(clk_path, "rt") as f:
        for ln in f:
            if ln.startswith("AS ") and len(ln) > 6:
                p = ln.split()
                if len(p) < 10:
                    continue
                prn_label = p[1].strip()  # e.g. 'G01', 'E02', 'C06'
                if not prn_label or prn_label[0] not in systems:
                    continue
                prn = prn_label
                try:
                    h, m, s = int(p[5]), int(p[6]), float(p[7])
                    sod = h * 3600 + m * 60 + s
                    d.setdefault(prn, []).append((sod, float(p[9])))
                except (ValueError, IndexError):
                    continue
    out = {}
    for prn, v in d.items():
        a = np.array(v)
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
        a = a[keep]
        order = np.argsort(a[:, 0])
        out[prn] = (a[order, 0], a[order, 1])
    return out


def dense_grid_multi(sp3: dict, clk: dict, step: float = 1.0,
                     pad: float = 900.0) -> tuple:
    """Pre-interpolate multi-GNSS satellites onto a fine uniform time grid.

    Same as dense_grid() but for multi-GNSS sp3/clk dicts keyed by 'G01',
    'E02', 'C06' etc.  Returns (t0, dt, {prn: (pos (n,3) m, vel (n,3) m/s,
    dts (n,) s)}).

    Uses scipy CubicSpline for the orbit interpolation (the SP3 data is at
    5-minute intervals, so cubic spline is more than adequate and ~100x
    faster than Lagrange for 99 satellites on a 1-second grid).
    """
    from scipy.interpolate import CubicSpline
    t_lo = min(v[0][0] for v in sp3.values()) + pad
    t_hi = max(v[0][-1] for v in sp3.values()) - pad
    tg = np.arange(t_lo, t_hi, step)
    out = {}
    for prn, (tt, xyz, _) in sp3.items():
        if prn not in clk:
            continue
        if len(tt) < 4:
            continue
        # ensure strictly increasing and unique
        _, unique_idx = np.unique(tt, return_index=True)
        tt_u = tt[unique_idx]
        xyz_u = xyz[unique_idx]
        if len(tt_u) < 4:
            continue
        splines = [CubicSpline(tt_u, xyz_u[:, k]) for k in range(3)]
        P = np.column_stack([splines[k](tg) for k in range(3)]) * 1000.0
        V = np.gradient(P, step, axis=0)
        ct, cc = clk[prn]
        _, unique_idx2 = np.unique(ct, return_index=True)
        ct_u, cc_u = ct[unique_idx2], cc[unique_idx2]
        D = np.interp(tg, ct_u, cc_u)
        out[prn] = (P, V, D)
    return float(t_lo), float(step), out
