#!/usr/bin/env python3
"""
STEP 22  (Phase 2 end-to-end) -- Generate a network receiver-clock prior file
for the patched PRIDE-PPP-AR `lsq` engine.

Output: plain text, one line per epoch  `<mjd> <sod> <clock_pred_m> <sigma_m>`,
read by `lsq_add_clkcon.f90` when the environment variable TEP_CLK_PRIOR points
at it.

The network prediction of the rover clock is built from a DISJOINT set of
analysis-centre receiver-clock series (the rover's own series is never used):
    arm B  -- network mean
    arm C  -- exp(-r/lambda) distance kriging  (nearest K stations)

Absolute frame (two-pass).  PRIDE's RECCLK parameter is the absolute receiver
clock (~1e2-1e5 m).  The network prediction here is the correlated *variation*
about a smooth trend.  With --from-rck <pass1.rck> the smooth trend from an
unconstrained PRIDE pass is added, so the prior is in PRIDE's absolute frame.
Without it, only the variation is written (use a loose sigma, for testing).
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_data as td       # noqa: E402
from scripts.lib import tep_stats as ts      # noqa: E402
from scripts.lib import tep_kernel as tk     # noqa: E402
from scripts.lib.ppp_estimator import tep_process_noise_scale  # noqa: E402
from scripts.lib.tep_phase_band import (  # noqa: E402
    BINS_PER_BLOCK, DT, FHI, FLO, causal_band, conditional_weights, legendre_cov,
)

C = 299_792_458.0
K_NEIGH = 15
GPS_T0_MJD = 44244            # 1980-01-06
# Default prior sigma for the persistence arm (matches Step 62's PRIOR_SIGMA_M)
PRIOR_SIGMA_M = 0.70

# Operational TEP kernel (Papers 1-2 / Steps 53, 55, 56, 64).
# The previous 2500 km / nugget 0.15 / CMB-anisotropy tuple was a broadcast-SPP
# nested-CV fit (Step 25).  That is a conventional data-tuned kernel, not TEP,
# and CMB anisotropy was withdrawn from the registered predictor.  Production
# PPP must ingest the physics-fixed Gaussian at lambda_T.
FITTED_PARAMS = tk.KernelParams(
    lambda_km=tk.LAMBDA_CANONICAL_KM,
    nugget=0.35,
    k_neighbours=25,
    alpha_v=0.35,
    aniso_mode="none",
    aniso_ratio=1.0,
    form="gaussian",
)


def _previous_tag(tag: str) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(tag, "%Y%j") - timedelta(days=1)).strftime("%Y%j")


def phase_v2_prediction(product, tag, target, refs, current, grid):
    """Experimental causal mode-block TEP variation for a stationary receiver.

    The target's preceding-day history identifies its phase relationship, but
    its current-day series is never used.  Each live reference component is
    filtered forward from D-1 into D, then predicted using a PSD Legendre x
    phase Gram covariance for a three-mode spectral block.
    """
    from scripts.lib.tep_phase import spectral_phase_blocks
    prev = td.load_series(product, _previous_tag(tag), min_valid_frac=0.95,
                          max_jumps=10_000)
    if target not in prev or target not in current:
        raise ValueError("C_v2 requires preceding-day target clock history")
    names = [target] + [n for n in refs if n in prev and n in current]
    if len(names) < 9:
        raise ValueError("C_v2 has fewer than eight live references")
    # Step 95 establishes the clock-product representation must be protected
    # from arbitrary datum overflow before any spectral calculation.
    P = np.column_stack([prev[n]["raw"] for n in names]).astype(float)
    Cday = np.column_stack([current[n]["raw"] for n in names]).astype(float)
    if not (np.isfinite(P).all() and np.isfinite(Cday).all()):
        raise ValueError("C_v2 phase history has missing samples")
    if len(grid) != len(Cday) or abs(np.median(np.diff(grid)) - DT) > 1e-6:
        raise ValueError("C_v2 currently requires the 5-minute CODE common grid")
    stations = [current[n]["station"] for n in names]
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            D[i, j] = D[j, i] = td.great_circle_km(stations[i], stations[j])
    spatial = legendre_cov(D)
    # Static first-half screening fractions from Step 88.  They enter as
    # signal variance fractions in conditional_weights, not a diagonal
    # rescaling that would algebraically cancel the screening information.
    screen = {"LF": {"QUARTZ": .11797542138802575, "ATOMIC": .09226289115336607},
              "MF": {"QUARTZ": .05, "ATOMIC": .06516163300321921},
              "HF": {"QUARTZ": .05, "ATOMIC": .0633173539532593}}
    clock_types = __import__("json").loads(
        (PROJECT_ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    def field_fraction(freq):
        band = "LF" if freq < 60e-6 else ("MF" if freq < 200e-6 else "HF")
        return np.array([screen[band].get((clock_types.get(n) or ["?"])[0], .50)
                         for n in names])
    blocks = spectral_phase_blocks(P, DT, FLO, FHI, BINS_PER_BLOCK)
    pred = np.zeros(len(grid))
    # Make accidental target-current use impossible even though causal_band is
    # columnwise: its current column is a harmless held-last-value placeholder.
    Cday = Cday.copy()
    Cday[:, 0] = P[-1, 0]
    for block in blocks:
        halfbin = .49 / (len(P) * DT)
        fp, fc = causal_band(P, Cday, block["f_lo_hz"] - halfbin,
                             block["f_hi_hz"] + halfbin)
        scale = np.std(fp, axis=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-9):
            continue
        live = fc[:, 1:] / scale[1:]
        live = np.where(np.abs(live) <= 200.0, live, 0.0)
        W = conditional_weights(spatial * block["predictive_gram"],
                                field_fraction(block["f_center_hz"]))
        pred += (live @ W[0, 1:]) * scale[0] * (C * 1e-9)
    return pred


def _ecef_to_stations(P):
    """Convert (n, 3) ECEF km array to list of Station objects for great-circle distance."""
    stations = []
    for p in P:
        lat, lon = td.ecef_to_geodetic(p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0)
        stations.append(td.Station("", p, lat, lon))
    return stations


def _gc_dist_matrix(stations_a, stations_b):
    """Great-circle distance matrix (km) between two lists of Station objects."""
    n_a, n_b = len(stations_a), len(stations_b)
    D = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            D[i, j] = td.great_circle_km(stations_a[i], stations_b[j])
    return D


def _bearing_deg(st_a, st_b):
    """Geographic bearing (deg from north) from station a to station b."""
    lat1, lon1 = np.radians(st_a.lat_deg), np.radians(st_a.lon_deg)
    lat2, lon2 = np.radians(st_b.lat_deg), np.radians(st_b.lon_deg)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360.0


def _azimuth_matrix(stations_a, stations_b):
    """Bearing matrix (deg) from a to b."""
    n_a, n_b = len(stations_a), len(stations_b)
    AZ = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            AZ[i, j] = _bearing_deg(stations_a[i], stations_b[j])
    return AZ


def kriging_weights(others_km, target_km, params=None, date_tag=None):
    # Use the unified tep_kernel module for kriging weights, consistent with
    # step_05b, step_12, step_25.  Falls back to fitted params if none given.
    if params is None:
        params = FITTED_PARAMS
    P = np.asarray(others_km, float); t = np.asarray(target_km, float)
    sts = _ecef_to_stations(P)
    tgt = _ecef_to_stations(t.reshape(1, -1))
    dnn = _gc_dist_matrix(sts, sts)
    dnt = _gc_dist_matrix(sts, tgt)[:, 0]
    # Apply anisotropic distance stretching if configured
    if params.aniso_mode != "none" and params.aniso_ratio != 1.0 and date_tag:
        az_nn = _azimuth_matrix(sts, sts)
        az_nt = _azimuth_matrix(sts, tgt)
        dnn = tk.anisotropic_distance(dnn, az_nn, params, date_tag)
        dnt = tk.anisotropic_distance(dnt.reshape(-1, 1), az_nt, params, date_tag)[:, 0]
    return tk.kriging_weights(dnn, dnt, params)


def smooth_trend_from_rck(rck_path, grid_sod, trend_end_sod=None):
    """Smooth trend (m) of the receiver clock from a PRIDE rck_* file.

    Format:  YYYY MM DD HH MM SS.ssssss  RCK(GPS)  RCK(GLO) ...
    We use the first (GPS) column and a robust running-median + spline-ish fit.

    If *trend_end_sod* is given, only rck rows with sod <= trend_end_sod are
    used to fit the polynomial.  This makes the trend causal: a cold-start
    window beginning at trend_end_sod cannot see target clock data from
    later in the day.  The polynomial order is reduced when fewer points are
    available so the fit remains stable on short arcs.
    """
    t, clk = [], []
    for line in Path(rck_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 7 and p[0].isdigit() and len(p[0]) == 4:
            try:
                sod = int(p[3]) * 3600 + int(p[4]) * 60 + float(p[5])
                v = float(p[6])
            except (ValueError, IndexError):
                continue
            if abs(v) > 1e-6:                       # skip zero-filled rows
                if trend_end_sod is not None and sod > trend_end_sod:
                    continue
                t.append(sod); clk.append(v)
    if len(t) < 20:
        return np.zeros_like(grid_sod)
    t = np.array(t); clk = np.array(clk)
    # robust: MAD-clip, then low-order poly on hourly-scaled time
    m = np.median(clk); s = 1.4826 * np.median(np.abs(clk - m)) + 1e-9
    ok = np.abs(clk - m) < 6 * s
    t, clk = t[ok], clk[ok]
    # Choose polynomial order based on available data: a 4th-order poly
    # needs at least ~120 points to be stable; a linear fit is the safe
    # fallback for short causal arcs.
    n_pts = len(t)
    if n_pts >= 120:
        deg = 4
    elif n_pts >= 40:
        deg = 2
    elif n_pts >= 10:
        deg = 1
    else:
        return np.zeros_like(grid_sod)
    with np.errstate(all="ignore"):
        coef = np.polyfit((t - t.mean()) / 3600.0, clk, deg)
    tr = np.polyval(coef, (grid_sod - t.mean()) / 3600.0)
    return np.clip(tr, clk.min() - 1.0, clk.max() + 1.0)


def persistence_trend_from_rck(rck_path, grid_sod, trend_end_sod=None):
    """Causal persistence (linear extrapolation) trend from a PRIDE rck_* file.

    Fits a linear offset+drift model to the rck data up to *trend_end_sod*
    and extrapolates to *grid_sod*.  This is the deployable conventional
    control: it uses only the target's own past clock, no network info.
    """
    t, clk = [], []
    for line in Path(rck_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 7 and p[0].isdigit() and len(p[0]) == 4:
            try:
                sod = int(p[3]) * 3600 + int(p[4]) * 60 + float(p[5])
                v = float(p[6])
            except (ValueError, IndexError):
                continue
            if abs(v) > 1e-6:
                if trend_end_sod is not None and sod > trend_end_sod:
                    continue
                t.append(sod); clk.append(v)
    if len(t) < 10:
        return np.zeros_like(grid_sod)
    t = np.array(t); clk = np.array(clk)
    m = np.median(clk); s = 1.4826 * np.median(np.abs(clk - m)) + 1e-9
    ok = np.abs(clk - m) < 6 * s
    t, clk = t[ok], clk[ok]
    # Use the last 30 minutes of causal data for the linear fit, matching
    # the _recent_clock_model window in Step 48.
    t_end = t[-1]
    recent = t >= (t_end - 1800)
    if recent.sum() >= 5:
        tf, cf = t[recent], clk[recent]
    else:
        tf, cf = t, clk
    coef = np.polyfit((tf - tf.mean()) / 3600.0, cf, 1)
    tr = np.polyval(coef, (grid_sod - tf.mean()) / 3600.0)
    return np.clip(tr, cf.min() - 1.0, cf.max() + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rover", required=True, help="4-char station code")
    ap.add_argument("--date", required=True, help="yyyyddd, e.g. 2024100")
    ap.add_argument("--product", default="clk_cod")
    ap.add_argument("--arm", choices=["B", "C", "C_v2", "persistence", "PN", "PN_v2"], default="C",
                    help="B=network mean, C=distance kriging, "
                         "persistence=target-only linear extrapolation, "
                         "C_v2=experimental causal phase-resolved mode-block prior; "
                         "PN_v2=persistence trend plus C_v2 variation; "
                         "PN=persistence trend PLUS network kriging (the nested "
                         "control: identical trend to `persistence`, so the only "
                         "difference is the network term)")
    ap.add_argument("--from-rck", default=None, help="pass-1 PRIDE .rck for the absolute trend")
    ap.add_argument("--trend-end-sod", type=float, default=None,
                    help="only use rck data with sod <= this value (causal trend)")
    ap.add_argument("--sigma-m", type=float, default=None, help="override prior sigma (m)")
    ap.add_argument("--rover-xyz", default=None,
                    help="ECEF x,y,z metres when the rover is not in the clock product")
    ap.add_argument("--min-ref-km", type=float, default=0.0,
                    help="exclude reference stations closer than this to the rover. "
                         "A co-located receiver (ZIM2/ZIM3 sit 0.0 km from ZIMM in "
                         "COD0OPSFIN) shares the site, antenna environment and often "
                         "the frequency standard, so including it turns a test of the "
                         "distance-structured field into 'use the receiver next door'. "
                         "Set 1 to drop true co-locations, 50 to drop the immediate "
                         "neighbourhood as well.")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    s = td.load_series(a.product, a.date)
    if a.rover in s:
        grid = s[a.rover]["t"]
        rov = s[a.rover]["station"]
        pool = [n for n in s if n != a.rover]
    elif a.rover_xyz:
        x, y, z = (float(v) for v in a.rover_xyz.replace(",", " ").split())
        lat, lon = td.ecef_to_geodetic(x, y, z)
        rov = td.Station(a.rover, np.array([x, y, z]) / 1000.0, lat, lon)
        grid = next(iter(s.values()))["t"]
        pool = list(s)
    else:
        sys.exit(f"rover {a.rover} not in {a.product} {a.date}; pass --rover-xyz")

    # Persistence arm: target-only causal linear extrapolation, no network.
    if a.arm == "persistence":
        if not a.from_rck:
            sys.exit("persistence arm requires --from-rck")
        trend = persistence_trend_from_rck(a.from_rck, grid, a.trend_end_sod)
        z = trend
        sig = np.full_like(grid, a.sigma_m if a.sigma_m is not None else PRIOR_SIGMA_M)
        doy = int(a.date[4:]); year = int(a.date[:4])
        from datetime import date
        mjd = (date(year, 1, 1).toordinal() + doy - 1) - date(1858, 11, 17).toordinal()
        with open(a.out, "w") as f:
            for sod, zz, ss in zip(grid, z, sig):
                f.write(f"{mjd} {sod:9.2f} {zz:15.6f} {ss:12.6f}\n")
        print(f"wrote {a.out}: {len(grid)} epochs, arm persistence, "
              f"trend std {np.std(trend):.3f} m, sigma {np.median(sig):.3f} m"
              f" (causal end={a.trend_end_sod})")
        return

    if a.min_ref_km > 0:
        n_before = len(pool)
        pool = [n for n in pool
                if td.great_circle_km(rov, s[n]["station"]) >= a.min_ref_km]
        if n_before != len(pool):
            print(f"  excluded {n_before - len(pool)} reference(s) within "
                  f"{a.min_ref_km:g} km of {a.rover}", file=sys.stderr)
    cand = sorted(pool, key=lambda n: td.great_circle_km(rov, s[n]["station"]))
    refs, cols = [], []
    for n in cand:
        dt_n = float(np.median(np.diff(s[n]["t"])))
        if not np.isfinite(dt_n) or dt_n <= 0:
            dt_n = 30.0
        # Live residual in metres, causally band-limited to the TEP band
        # (10-500 uHz) and trend-removed to cancel the causal-bandpass
        # group delay.  The raw residual includes high-frequency local
        # oscillator noise that is uncorrelated between stations and
        # degrades the kriging prediction when added to the trend.  The
        # causal bandpass (sosfilt, forward-only) removes this noise
        # without look-ahead, but introduces a group delay of ~4.4 h at
        # 10 uHz.  Removing each reference's own short-term linear trend
        # before bandpassing cancels the common-mode drift that causes
        # the delay to bias the kriging, preserving the differential TEP
        # spatial signal that Steps 55/56 exploit.
        raw = np.interp(grid, s[n]["t"], np.asarray(s[n]["resid"], float)) * (C * 1e-9)
        # Detrend: remove a linear fit to the last 30 min of *causal* data
        # (sod <= trend_end_sod) before bandpassing.  Fitting on the whole
        # day leaked evening curvature into morning residuals.
        n_raw = len(raw)
        if n_raw > 10:
            t_idx = np.arange(n_raw, dtype=float)
            causal = np.ones(n_raw, dtype=bool)
            if a.trend_end_sod is not None:
                causal = grid <= float(a.trend_end_sod)
            fit_idx = np.where(causal)[0]
            if fit_idx.size >= 5:
                t_end = fit_idx[-1]
                recent = fit_idx[fit_idx >= (t_end - min(60, fit_idx.size))]
                if recent.size >= 5:
                    coef = np.polyfit(t_idx[recent], raw[recent], 1)
                    trend_local = np.polyval(coef, t_idx)
                else:
                    trend_local = np.full(n_raw, float(np.mean(raw[fit_idx])))
            else:
                trend_local = np.full(n_raw, float(np.mean(raw)))
            raw_detrended = raw - trend_local
        else:
            raw_detrended = raw - np.mean(raw)
        b = ts.causal_bandpass(raw_detrended, dt_n, f_lo_uhz=10.0, f_hi_uhz=500.0)
        if np.isfinite(b).all() and np.std(b) < 5.0:      # QC: no bandpass ringing / jumps
            refs.append(n); cols.append(b)
        if len(refs) >= K_NEIGH:
            break
    R = np.nan_to_num(np.column_stack(cols))

    # Physics-fixed TEP kernel.  Do not substitute a nested-CV conventional fit.
    params = tk.KernelParams(
        lambda_km=FITTED_PARAMS.lambda_km,
        nugget=FITTED_PARAMS.nugget,
        k_neighbours=min(FITTED_PARAMS.k_neighbours, len(refs)),
        alpha_v=FITTED_PARAMS.alpha_v,
        aniso_mode=FITTED_PARAMS.aniso_mode,
        aniso_ratio=FITTED_PARAMS.aniso_ratio,
        form=FITTED_PARAMS.form,
    )
    with np.errstate(all="ignore"):
        if a.arm == "B":
            pred_var = R.mean(1)
        elif a.arm in ("C_v2", "PN_v2"):
            if a.rover not in s:
                sys.exit("C_v2 requires rover history in the reference clock product")
            pred_var = phase_v2_prediction(a.product, a.date, a.rover, refs, s, grid)
        else:
            w = kriging_weights([s[n]["station"].ecef_km for n in refs], rov.ecef_km,
                                params=params, date_tag=a.date)
            pred_var = R @ w

    # Per-epoch sigma: base LOO RMSE modulated by TEP astronomical phases.
    # When the phases predict higher clock variability (constructive alignment),
    # the sigma is inflated; when lower, it is deflated.  This feeds TEP
    # temporal physics directly into the prior precision.
    if a.sigma_m is not None:
        sig = np.full_like(grid, a.sigma_m)
    else:
        errs = []
        with np.errstate(all="ignore"):
            for i in range(len(refs)):
                o = [j for j in range(len(refs)) if j != i]
                if a.arm == "B":
                    e = R[:, i] - R[:, o].mean(1)
                else:
                    wi = kriging_weights([s[refs[j]]["station"].ecef_km for j in o],
                                         s[refs[i]]["station"].ecef_km,
                                         params=params, date_tag=a.date)
                    e = R[:, i] - R[:, o] @ wi
                if np.isfinite(e).all() and np.std(e) < 20:
                    errs.append(e)
        base_sig = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)) + 0.02) if errs else 0.5
        # TEP phase modulation of per-epoch sigma
        from datetime import datetime, timedelta
        year = int(a.date[:4]); doy = int(a.date[4:])
        start_dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
        sig = np.array([
            base_sig * tep_process_noise_scale(start_dt + timedelta(seconds=float(sod)))
            for sod in grid
        ])

    # NESTED CONTROL.  Arm C pairs the network term with a polynomial trend over
    # all causal history, while the `persistence` arm uses a linear fit to the
    # last 30 minutes only.  Those trends are not equivalent -- a receiver clock
    # is locally linear but globally curved, so the short linear fit is the
    # better short-horizon predictor regardless of any network information.
    # Comparing arm C against `persistence` therefore confounds the network term
    # with the trend model.  Arm PN uses the SAME trend as `persistence` and adds
    # the network variation, so PN vs persistence isolates the network alone.
    if a.arm in ("PN", "PN_v2"):
        trend = (persistence_trend_from_rck(a.from_rck, grid, a.trend_end_sod)
                 if a.from_rck else np.zeros_like(grid))
    else:
        trend = (smooth_trend_from_rck(a.from_rck, grid, a.trend_end_sod)
                 if a.from_rck else np.zeros_like(grid))
    z = trend + pred_var

    doy = int(a.date[4:]); year = int(a.date[:4])
    from datetime import date
    mjd = (date(year, 1, 1).toordinal() + doy - 1) - date(1858, 11, 17).toordinal()

    # Compute TEP anisotropy position constraint direction (ECEF unit vector)
    # and per-epoch position sigma for the CMB anisotropy preferred axis.
    write_pos_constraint = (a.arm == "C" and params.aniso_mode != "none"
                            and params.aniso_ratio != 1.0)
    if write_pos_constraint:
        from scripts.lib.tep_kernel import preferred_azimuth_deg
        az_pref_deg = preferred_azimuth_deg(params.aniso_mode, a.date)
        az_pref_rad = np.radians(az_pref_deg)
        rover_lat, rover_lon = td.ecef_to_geodetic(
            rov.ecef_km[0] * 1000.0, rov.ecef_km[1] * 1000.0, rov.ecef_km[2] * 1000.0)
        east = np.array([-np.sin(np.radians(rover_lon)), np.cos(np.radians(rover_lon)), 0.0])
        north = np.array([-np.sin(np.radians(rover_lat)) * np.cos(np.radians(rover_lon)),
                          -np.sin(np.radians(rover_lat)) * np.sin(np.radians(rover_lon)),
                          np.cos(np.radians(rover_lat))])
        tep_dir = np.cos(az_pref_rad) * north + np.sin(az_pref_rad) * east
        tep_dir = tep_dir / (np.linalg.norm(tep_dir) + 1e-12)
        # Perpendicular direction (horizontal cross with z-axis)
        perp = np.cross(tep_dir, np.array([0.0, 0.0, 1.0]))
        pn = np.linalg.norm(perp)
        if pn > 1e-6:
            perp = perp / pn
        else:
            perp = np.array([0.0, 1.0, 0.0])
        # Position sigma: clock sigma / aniso_ratio^2 (tighter when anisotropy is stronger)
        pos_sig = sig / (params.aniso_ratio ** 2)

    with open(a.out, "w") as f:
        for idx, (sod, zz, ss) in enumerate(zip(grid, z, sig)):
            if write_pos_constraint:
                f.write(f"{mjd} {sod:9.2f} {zz:15.6f} {ss:12.6f} "
                        f"{perp[0]:.6f} {perp[1]:.6f} {perp[2]:.6f} {pos_sig[idx]:12.6f}\n")
            else:
                f.write(f"{mjd} {sod:9.2f} {zz:15.6f} {ss:12.6f}\n")
    pos_note = " +pos_constraint" if write_pos_constraint else ""
    print(f"wrote {a.out}: {len(grid)} epochs, arm {a.arm}, "
          f"pred-var std {np.std(pred_var):.3f} m, sigma median {np.median(sig):.3f} m"
          f"{' (+trend)' if a.from_rck else ' (variation only)'}{pos_note}")


if __name__ == "__main__":
    main()
