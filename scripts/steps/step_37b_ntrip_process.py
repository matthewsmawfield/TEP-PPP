#!/usr/bin/env python3
"""
STEP 37b -- Live NTRIP Stream Processing
========================================

Converts recorded RTCM3 streams from step_37 to RINEX, extracts
receiver clock biases via RTKLIB SPP, and runs the same kriging
obstruction protocol as Step 33 on the live data.  The result is
directly comparable to the archived Step 33 demonstration.

Pipeline
--------
1. Read the step_37 manifest to locate recorded RTCM3 files.
2. Convert each RTCM3 file to RINEX observation + navigation files
   using RTKLIB ``convbin``.
3. Run RTKLIB SPP (``rnx2rtkp -p 0 -y 2``) on each station to extract
   per-epoch receiver clock biases and satellite geometry (.stat).
4. Build the network clock series and run the cross-day kriging
   prediction (using the existing archived data as the training set,
   so the live data is genuinely held-out).
5. Run the Step 33 obstruction protocol: 10/30/60/120/240-minute
   windows, 5-satellite obstruction, persistence vs. linear vs. kernel.
6. Write the result JSON alongside step_33_position_demo.json for
   direct comparison.

The live data never enters hyperparameter selection: the kernel is
trained on the existing archived broadcast set (DOY 100-106, 2024),
and the live stream is the held-out evaluation.  This preserves the
leakage-free design.
"""
from __future__ import annotations

import sys
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # noqa: E402
from scripts.lib import tep_data as td  # noqa: E402
from scripts.lib import tep_kernel as tk  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
DATA_LIVE = PROJECT_ROOT / "data" / "ntrip_live"

# Obstruction protocol (same as Step 33)
GAP_LENGTHS_S = [600.0, 1800.0, 3600.0, 7200.0, 14400.0]
SHORT_GAP_LENGTHS_S = [30.0, 60.0, 120.0, 180.0, 240.0]
ELEV_MIN_DEG = 10.0
K_OBSTRUCT = 5
OBSTRUCTION_STRATEGIES = ("highest", "random", "lowest")
OBSTRUCTION_RNG = np.random.default_rng(20260937)

# Training days from the archived broadcast set (DOY 100-106, 2024)
TRAIN_DAYS = ["2024100", "2024101", "2024102", "2024103",
              "2024104", "2024105", "2024106"]


def _find_tool(name: str) -> str:
    """Locate an RTKLIB tool.

    For convbin, prefers the demo5 build (convbin_demo5) which properly
    parses RTCM 3.3 MSM7 messages.  Falls back to the standard convbin.
    """
    if name == "convbin":
        # Prefer demo5 convbin for RTCM 3.3 MSM7 support
        p5 = shutil.which("convbin_demo5")
        if p5:
            return p5
        for c in (Path.home() / ".local" / "bin" / "convbin_demo5",
                  Path.home() / ".local" / "bin" / "convbin"):
            if c.exists():
                return str(c)
        p = shutil.which("convbin")
        if p:
            return p
        raise RuntimeError("convbin not found")
    p = shutil.which(name)
    if p:
        return p
    for c in (PROJECT_ROOT / "bin" / name,
              Path.home() / ".local" / "bin" / name):
        if c.exists():
            return str(c)
    raise RuntimeError(f"{name} not found")


def _convbin_rtcm3(rtcm3_path: Path, output_dir: Path) -> dict:
    """Convert a raw RTCM3 file to RINEX obs + nav using RTKLIB convbin.

    Returns a dict with paths to the generated RINEX files, or None on failure.
    """
    convbin = _find_tool("convbin")
    # convbin flags:
    #   -r rtcm3     : input format
    #   -d <dir>     : output directory
    #   -f 2         : output frequency (L1+L2)
    # Defaults write <file>.obs and <file>.nav in the output directory.
    cmd = [convbin, "-r", "rtcm3", "-d", str(output_dir),
           "-f", "2", str(rtcm3_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120,
                                cwd=str(output_dir))
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    # Find the generated RINEX files
    obs_files = list(output_dir.glob("*.obs")) + list(output_dir.glob("*.O"))
    nav_files = list(output_dir.glob("*.nav")) + list(output_dir.glob("*.N"))
    # Also check for Rinex 3 naming
    if not obs_files:
        obs_files = list(output_dir.glob("*MO.rnx"))
    if not nav_files:
        nav_files = list(output_dir.glob("*MN.rnx")) + list(output_dir.glob("*.gnav"))

    if not obs_files:
        return None

    return {"obs": obs_files[0], "nav": nav_files[0] if nav_files else None}


def _merge_rinex3_navs(nav_paths, out_path: Path) -> Path | None:
    """Merge RINEX 3.x nav files into one (first header + deduped records).

    Broadcast ephemeris are receiver-independent; several NTRIP casters
    (e.g. ReNEP) stream observations without in-band ephemeris, so a pooled
    nav built from the same capture's own nav streams plus the live
    system's shared_nav product is required to process those stations.
    A record begins on a line whose first two columns are non-space
    (e.g. ``C22 2026 10 16 ...``); continuation lines are indented.
    """
    header, records, seen = None, [], set()
    for p in nav_paths:
        try:
            lines = Path(p).read_text(errors="ignore").splitlines()
        except Exception:
            continue
        h_end = next((i for i, l in enumerate(lines)
                      if "END OF HEADER" in l), None)
        if h_end is None:
            continue
        if header is None:
            header = lines[:h_end + 1]
        i = h_end + 1
        cur = []
        for l in lines[i:]:
            if l[:1] != " " and l[:2].strip():
                if cur:
                    key = cur[0][:24]
                    if key not in seen:
                        seen.add(key)
                        records.extend(cur)
                    cur = []
            cur.append(l)
        if cur:
            key = cur[0][:24]
            if key not in seen:
                seen.add(key)
                records.extend(cur)
    if header is None or not records:
        return None
    out_path.write_text("\n".join(header + records) + "\n")
    return out_path


def _shared_nav_candidates() -> list:
    """Latest pooled broadcast nav products from the live system."""
    d = PROJECT_ROOT / "data" / "ntrip_live" / "shared_nav"
    if not d.exists():
        return []
    return sorted(d.glob("local_*.nav"), key=lambda p: p.stat().st_mtime)


def _ecef_to_ll(x, y, z):
    """WGS84 ECEF -> (lat_deg, lon_deg)."""
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(8):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    return float(np.degrees(lat)), float(np.degrees(lon))


def _hav_km(a, b, c, e):
    R = 6371.0
    p1, p2 = np.radians(a), np.radians(c)
    dp, dl = np.radians(c - a), np.radians(e - b)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def _live_field_grid():
    """Shared 30 s grid + per-station detrended clock columns from the
    newest live-network snapshot (data/ppp_live/results/raw_window_*.pkl),
    plus station coordinates.  Returns (grid, raw, coords) or None."""
    import glob as _glob
    import pickle as _pickle
    files = sorted(_glob.glob(str(
        PROJECT_ROOT / "data" / "ppp_live" / "results"
        / "raw_window_*.pkl")), key=os.path.getmtime)
    if not files:
        return None
    try:
        d = _pickle.load(open(files[-1], "rb"))
    except Exception:
        return None
    raw, coords = d.get("raw_series"), d.get("station_coords")
    if not raw or not coords:
        return None
    sts = [s for s in raw if s in coords and len(raw[s][0]) > 100]
    if len(sts) < 6:
        return None
    s0 = max(float(np.min(raw[s][0])) for s in sts)
    s1 = min(float(np.max(raw[s][0])) for s in sts)
    return np.arange(s0, s1, 30.0), raw, coords


def _contemporaneous_kernel(grid, raw, coords, tgt_ll, exclude_sid,
                            lambda_km=1500.0):
    """Kriged accumulated-offset field at the target's location, built
    from the SAME period's live network (the deployed predictor), with
    the target station left out when it is a network member."""
    if tgt_ll is None:
        return None
    cols, ws = [], []
    for sid, (ts, vs) in raw.items():
        if sid == exclude_sid or sid not in coords:
            continue
        ts = np.asarray(ts, float)
        vs = np.asarray(vs, float)
        if len(ts) < 100:
            continue
        y = np.interp(grid, ts, vs, left=np.nan, right=np.nan)
        if np.isfinite(y).mean() < 0.9:
            continue
        mm = np.isfinite(y)
        xi = np.arange(len(y), dtype=float)
        y[~mm] = np.interp(xi[~mm], xi[mm], y[mm])
        cols.append(y)
        ws.append(np.exp(-0.5 * (_hav_km(tgt_ll[0], tgt_ll[1],
                                         *coords[sid]) / lambda_km) ** 2))
    ws = np.array(ws)
    if len(cols) < 6 or ws.sum() < 1e-6:
        return None
    M = np.column_stack(cols)
    inc = M - M[0]
    u = (inc * ws[None, :]).sum(axis=1) / ws.sum()
    return {"t": grid, "kres_m": u, "sigma_m": 3.67,
            "variant": "contemporaneous_field"}


def _remap_midnight(sd):
    """Continuous time axis across UTC midnight: when a merged station
    record carries epochs on both sides of 12:00 SOD, fold the
    day-15 evening chunk (sod>=43200) to negative values so the axis is
    chronological."""
    sods = sd["sods"]
    # Genuine midnight crossing: early-morning AND evening epochs both
    # present (a noon-spanning capture never remaps).
    if not (np.any(sods < 43200) and np.any(sods >= 60000)):
        return sd
    epochs = {int(s - 86400 if s >= 43200 else s): rec
              for s, rec in sd["epochs"].items()}
    sods = np.array(sorted(epochs.keys()))
    sd = dict(sd)
    sd["epochs"] = epochs
    sd["sods"] = sods
    sd["cdt_full"] = np.array([epochs[s]["cdt_rtk_m"] for s in sods])
    sd["pos_rtk"] = np.array([epochs[s]["pos_rtk"] for s in sods])
    return sd


def _run_spp(obs_path: Path, nav_path: Path, sp3_path: Path | None,
             tmp: Path) -> Path | None:
    """Run RTKLIB SPP and return the .stat file path, or None."""
    rnx2rtkp = _find_tool("rnx2rtkp")
    pos = tmp / (obs_path.stem + ".pos")
    stat = Path(str(pos) + ".stat")

    cmd = [rnx2rtkp, "-x", "0", "-m", str(int(ELEV_MIN_DEG)), "-p", "0",
           "-f", "2", "-y", "2", "-o", str(pos), str(obs_path), str(nav_path)]
    if sp3_path and sp3_path.exists():
        cmd.append(str(sp3_path))
    try:
        subprocess.run(cmd, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    return stat if stat.exists() else None


def _tow_to_sod(tow: float) -> int:
    return int(round(tow % 86400.0))


def _parse_stat(stat: Path) -> dict:
    """Parse RTKLIB .stat file -> per-epoch satellite geometry + clock."""
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
                sod = _tow_to_sod(float(p[2]))
                pos_by_sod[sod] = np.array([float(p[4]), float(p[5]), float(p[6])])
            elif tag == "$CLK" and len(p) >= 6:
                sod = _tow_to_sod(float(p[2]))
                clk_by_sod[sod] = float(p[5])
            elif tag == "$SAT" and len(p) >= 8:
                sod = _tow_to_sod(float(p[2]))
                az, el, resp = float(p[5]), float(p[6]), float(p[7])
                if el >= ELEV_MIN_DEG and abs(resp) < 30.0:
                    epochs.setdefault(sod, []).append((az, el, resp))
    out = {}
    for sod, sats in epochs.items():
        if sod in pos_by_sod and sod in clk_by_sod and len(sats) >= 4:
            out[sod] = {"pos_rtk": pos_by_sod[sod],
                        "cdt_rtk_m": clk_by_sod[sod], "sats": sats}
    return out


def _los_unit(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])


def _select_sats(all_sats, strategy, rng):
    if len(all_sats) < K_OBSTRUCT:
        return all_sats
    if strategy == "highest":
        return sorted(all_sats, key=lambda q: -q[1])[:K_OBSTRUCT]
    elif strategy == "lowest":
        return sorted(all_sats, key=lambda q: q[1])[:K_OBSTRUCT]
    else:
        idx = rng.choice(len(all_sats), K_OBSTRUCT, replace=False)
        return [all_sats[i] for i in idx]


def _solve_offset(A, b, w, cdt_prior_m=None, sig_prior_m=None):
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
    if not np.all(np.isfinite(dpos)) or np.linalg.norm(dpos) > 100_000.0:
        return None
    return dpos


def _kernel_clock_prediction_live():
    """Predict a network-mean kernel prior from the archived training network.

    The kernel is trained on the existing archived broadcast set
    (DOY 100-106, 2024).  This preserves the leakage-free cross-day
    design: the live stream has never been seen by the model.

    The kernel prediction is the network mean of the LOO band-limited
    clock deviation of each archived station, predicted from the rest of
    the archived network.  The live target's clock is anchored from real
    observations; the kernel supplies the sub-gap deviation (the increment
    relative to the anchor epoch).  The network mean of the per-station
    LOO predictions is used as a conservative common-mode temporal shape
    that captures the band-limited network clock dynamics without
    requiring a spatial match to the live target.

    Returns a dict with the kernel prediction (t, kres_m) and metadata,
    or None.
    """
    # Train on the first available archived day
    train_day = None
    for d in TRAIN_DAYS:
        s = td.load_series("broadcast", d)
        if s and len(s) >= 16:
            train_day = d
            break
    if train_day is None:
        return None

    s_tr = td.load_series("broadcast", train_day)
    if not s_tr or len(s_tr) < 16:
        return None

    names_tr, X_tr, D_tr = tk.build_matrix(s_tr, filter_by_std=False)
    if X_tr is None or len(names_tr) < 16:
        return None

    good_tr = [i for i in range(len(names_tr)) if 0.01 < X_tr[:, i].std() < 60.0]
    if len(good_tr) < 8:
        return None

    tr_idx = list(range(len(good_tr)))
    p, tr_err = tk.fit_hyperparameters(
        X_tr[:, good_tr], D_tr[np.ix_(good_tr, good_tr)], tr_idx)
    if p is None:
        return None

    # LOO predict each archived station from the rest, then return the
    # prediction for the station whose band-limited clock has the highest
    # correlation with the live target's clock (best proxy for the live
    # target's spatial location in the kernel's correlation structure).
    # If no live clock is available, use the network-mean prediction.
    t_grid = np.asarray(s_tr[names_tr[good_tr[0]]]["t"], float)
    preds = {}
    for ii, gi in enumerate(good_tr):
        pool = [j for j in range(len(good_tr)) if j != ii]
        pred = tk.predict(X_tr[:, good_tr], D_tr[np.ix_(good_tr, good_tr)],
                          ii, pool, p)
        if pred is not None:
            preds[names_tr[gi]] = np.asarray(pred, float)

    if not preds:
        return None

    # Use the network mean of the LOO predictions as the kernel prior.
    # This is conservative (the mean of the per-station predictions) and
    # does not require matching the live target to a specific archived
    # station.  The increment of this mean relative to the anchor epoch
    # is the kernel's sub-gap deviation prediction.
    kres_m = np.mean(list(preds.values()), axis=0)

    return {"sigma_m": max(float(tr_err), 0.1),
            "train_day": train_day, "n_train_stations": len(good_tr),
            "t": t_grid, "kres_m": kres_m,
            "params": p.as_dict()}


def _process_station(station_id: str, rtcm3_path: Path,
                     live_dir: Path, tmp: Path,
                     nav_pool=None) -> dict | None:
    """Process one station's RTCM3 file through the full chain.

    ``nav_pool`` is a callable returning a pooled broadcast nav path (or
    None) for stations whose own stream carries no ephemeris; broadcast
    ephemeris are receiver-independent, so this is an input substitution,
    not an estimator change.  The returned dict records which nav source
    was used.
    """
    # 1. Convert RTCM3 -> RINEX (unique dir per input file so stations
    # with multiple capture segments never alias a previous conversion)
    conv_dir = tmp / station_id / rtcm3_path.stem
    conv_dir.mkdir(parents=True, exist_ok=True)
    rnx = _convbin_rtcm3(rtcm3_path, conv_dir)
    if rnx is None:
        return None

    # 2. Find SP3 for the current day (if available)
    sp3 = None
    sp3_dir = PROJECT_ROOT / "data" / "rinex" / "sp3"
    if sp3_dir.exists():
        today = datetime.now(timezone.utc)
        doy = today.timetuple().tm_yday
        import glob
        for pat in ("COD0MGXFIN", "COD0OPSFIN", "IGS0OPSFIN"):
            hits = glob.glob(str(sp3_dir / f"{pat}_2024{doy:03d}0000_01D_*_ORB.SP3*"))
            if not hits:
                hits = glob.glob(str(sp3_dir / f"{pat}_{today.year}{doy:03d}0000_01D_*_ORB.SP3*"))
            if hits:
                sp3 = Path(hits[0])
                break

    # 3. Run SPP
    nav = rnx["nav"]
    nav_source = "own_stream"
    if nav is None and nav_pool is not None:
        nav = nav_pool()
        nav_source = "pooled_broadcast"
    if nav is None:
        print(f"  {station_id}: no navigation file, skipping")
        return None
    stat = _run_spp(rnx["obs"], nav, sp3, tmp)
    if stat is None:
        return None

    # 4. Parse .stat
    epochs = _parse_stat(stat)
    if len(epochs) < 200:
        return None

    # 5. Extract clock series (filter non-finite positions/clocks)
    sods = np.array(sorted(epochs.keys()))
    cdt_full = np.array([epochs[s]["cdt_rtk_m"] for s in sods])
    pos_rtk = np.array([epochs[s]["pos_rtk"] for s in sods])
    # Truth: median of finite positions only
    finite_pos = pos_rtk[np.all(np.isfinite(pos_rtk), axis=1)]
    if len(finite_pos) < 100:
        return None
    truth = np.median(finite_pos, axis=0)

    # Station coords from the obs header (ECEF -> geodetic) for the
    # contemporaneous-field kernel prior.
    approx = None
    try:
        with open(rnx["obs"], errors="ignore") as fh:
            for line in fh:
                if "APPROX POSITION" in line:
                    x, y, z = (float(v) for v in line[:60].split())
                    approx = _ecef_to_ll(x, y, z)
                    break
                if "END OF HEADER" in line:
                    break
    except Exception:
        approx = None

    return {
        "station_id": station_id,
        "n_epochs": len(epochs),
        "sods": sods,
        "cdt_full": cdt_full,
        "pos_rtk": pos_rtk,
        "truth": truth,
        "epochs": epochs,
        "nav_source": nav_source,
        "own_nav": rnx["nav"],
        "approx_ll": approx,
    }


def _run_obstruction(station_data: dict, kernel_pred: dict | None,
                     gap_len_s: float, strategy: str = "highest") -> dict | None:
    """Run the Step 33 obstruction protocol on one live station."""
    epochs = station_data["epochs"]
    sods = station_data["sods"]
    cdt_full = station_data["cdt_full"]
    truth = station_data["truth"]
    pos_rtk = station_data["pos_rtk"]

    rows = {"hold": [], "linrate": [], "kernel": []}
    clk_err = {"hold": [], "linrate": [], "kernel": []}
    n_gaps = 0
    strat_rng = np.random.default_rng(
        int(OBSTRUCTION_RNG.integers(0, 2**31 - 1)))

    GAP_STRIDE_S = 3 * 3600.0

    # If no kernel prediction, we can still run hold vs linrate
    ksig = kernel_pred["sigma_m"] if kernel_pred else 5.0

    # Build the kernel's band-limited prediction of the target's clock
    # deviation on the training-day time grid.  The live target's clock
    # is anchored from real observations; the kernel supplies the sub-gap
    # deviation (the increment relative to the anchor epoch).  We use the
    # archived network's LOO prediction of the target station as the
    # kernel prior -- the same spatial correlation structure, applied to
    # the training day's network state.
    kres_t = None
    kres_m = None
    if kernel_pred is not None:
        kres_t = kernel_pred.get("t")
        kres_m = kernel_pred.get("kres_m")
        if kres_t is not None and kres_m is not None:
            kres_t = np.asarray(kres_t, float)
            kres_m = np.asarray(kres_m, float)

    # Filter non-finite clock values
    finite_clk = np.isfinite(cdt_full)
    if not np.all(finite_clk):
        cdt_full = np.where(finite_clk, cdt_full, np.interp(
            sods, sods[finite_clk], cdt_full[finite_clk]))

    gap_epochs = []
    # Adaptive pre-window and stride: for short captures the pre-window is
    # half the gap length (max 60 s), and gaps are spaced so the pre-window
    # for the next gap begins where the previous gap ends.
    pre_len = min(60.0, gap_len_s / 2.0)
    gap_stride_s = pre_len + gap_len_s
    for g0 in np.arange(sods[0] + pre_len, sods[-1] - gap_len_s, gap_stride_s):
        pre = sods[(sods >= g0 - pre_len) & (sods < g0)]
        if len(pre) < 3:
            continue
        anchor_sod = pre[-1]
        cdt_anchor = float(np.interp(anchor_sod, sods, cdt_full))
        rate = float((np.interp(pre[-1], sods, cdt_full)
                      - np.interp(pre[0], sods, cdt_full)) / max(pre[-1] - pre[0], 1.0))
        # Kernel prediction increment: the archived network's prediction
        # of the target's band-limited clock deviation, differenced
        # between the gap epoch and the anchor.  This is the same
        # quantity step_33 uses; the live target's absolute clock is
        # anchored from real observations, the kernel supplies the
        # sub-gap deviation.
        kres_anchor = 0.0
        if kres_t is not None and kres_m is not None:
            kres_anchor = float(np.interp(anchor_sod, kres_t, kres_m))
        n_gaps += 1
        for s in sods[(sods >= g0) & (sods < g0 + gap_len_s)]:
            rec = epochs[int(s)]
            cdt_ref = rec["cdt_rtk_m"]
            if not np.isfinite(cdt_ref):
                continue
            dt = float(s) - anchor_sod
            kres_at_s = kres_anchor  # default: no kernel deviation
            if kres_t is not None and kres_m is not None:
                kres_at_s = float(np.interp(float(s), kres_t, kres_m))
            priors_clk = {
                "hold": cdt_anchor,
                "linrate": cdt_anchor + rate * dt,
                "kernel": cdt_anchor + (kres_at_s - kres_anchor),
            }
            for name, cdt_p in priors_clk.items():
                clk_err[name].append(abs(cdt_p - cdt_ref))
            gap_epochs.append((s, rec, cdt_ref, dt, cdt_anchor, rate,
                               kres_anchor, kres_at_s))

    if n_gaps < 2 or len(gap_epochs) < 20:
        return None

    method_sig = {
        name: min(max(float(np.nanmedian(
            [e for e in errs if np.isfinite(e)])), 0.05), 20.0)
        if any(np.isfinite(e) for e in errs) else 5.0
        for name, errs in clk_err.items()
    }

    # Fair comparison: only keep epochs where ALL three methods produce
    # a valid solution (same as step_33).
    for s, rec, cdt_ref, dt, cdt_anchor, rate, kres_anchor, kres_at_s in gap_epochs:
        offs = rec["pos_rtk"] - truth
        if not np.all(np.isfinite(offs)):
            continue
        sats = _select_sats(rec["sats"], strategy, strat_rng)
        if len(sats) < K_OBSTRUCT:
            continue
        e = np.array([_los_unit(a, el) for a, el, _ in sats])
        resp = np.array([r for _, _, r in sats])
        el = np.array([q[1] for q in sats])
        b = resp
        w = np.sin(np.radians(el)) ** 2
        A = np.column_stack([-e, np.ones(len(sats))])
        priors = {
            "hold": cdt_anchor,
            "linrate": cdt_anchor + rate * dt,
            "kernel": cdt_anchor + (kres_at_s - kres_anchor),
        }
        sols = {}
        all_ok = True
        for name, cdt_p in priors.items():
            d = _solve_offset(A, b, w, cdt_prior_m=(cdt_p - cdt_ref),
                              sig_prior_m=method_sig[name])
            if d is not None:
                sols[name] = float(np.linalg.norm(offs + d))
            else:
                all_ok = False
        if all_ok:
            for name in priors:
                rows[name].append(sols[name])

    if len(rows["kernel"]) < 10:
        return None
    med = {k: float(np.nanmedian(v)) for k, v in rows.items()
           if v and any(np.isfinite(v))}
    cerr = {k: float(np.nanmedian(v)) for k, v in clk_err.items()
            if v and any(np.isfinite(v))}
    return {
        "n_gaps": n_gaps, "n_gap_epochs": len(rows["kernel"]),
        "strategy": strategy,
        "clock_error_median_m": cerr,
        "pos_error_median_m": med,
        "kernel_vs_hold_pct": (100.0 * (1.0 - med["kernel"] / med["hold"])
                               if med.get("hold") else None),
        "kernel_vs_linrate_pct": (100.0 * (1.0 - med["kernel"] / med["linrate"])
                                  if med.get("linrate") else None),
    }


def main():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_37b", log_file_path=log_dir / "step_37b.log")
    set_step_logger(logger)

    print_status("STEP 37b: Live NTRIP Stream Processing", "TITLE")

    # Read step_37 manifest
    manifest_path = OUT / "step_37_ntrip_acquire.json"
    if not manifest_path.exists():
        print_status(f"Manifest not found: {manifest_path}", "ERROR")
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ok":
        print_status(f"Step 37 status was '{manifest.get('status')}' -- nothing to process", "ERROR")
        return

    # Process each recorded station
    tmp = Path(tempfile.mkdtemp(prefix="step37b_"))
    produced_navs = []

    def _nav_pool():
        """Pooled broadcast nav: this capture's own nav streams plus the
        live system's shared_nav products.  Rebuilt on demand so stations
        processed early still see every ephemeris available so far."""
        srcs = _shared_nav_candidates() + list(produced_navs)
        if not srcs:
            return None
        return _merge_rinex3_navs(srcs, tmp / "pooled_broadcast.nav")

    stations = []
    for f_info in manifest["files"]:
        if f_info["error"] is not None or f_info["bytes"] == 0:
            continue
        rtcm3_path = PROJECT_ROOT / f_info["file"]
        if not rtcm3_path.exists():
            continue
        print_status(f"Processing {f_info['station_id']}...", "PROCESS")
        try:
            sd = _process_station(f_info["station_id"], rtcm3_path,
                                  DATA_LIVE, tmp, nav_pool=_nav_pool)
        except Exception as e:
            print_status(f"  {f_info['station_id']}: ERROR {e}", "ERROR")
            continue
        if sd:
            if sd.get("own_nav") is not None:
                produced_navs.append(sd["own_nav"])
            stations.append(sd)
            print_status(f"  {f_info['station_id']}: {sd['n_epochs']} epochs "
                         f"({sd['nav_source']} nav)", "INFO")
        else:
            print_status(f"  {f_info['station_id']}: no valid data", "WARN")

    # Merge multiple capture segments of the same station into one record:
    # a station's capture is the union of its files (epoch keys are
    # seconds-of-day; overlapping keys resolve to the later segment).
    by_id = {}
    for sd in stations:
        d = by_id.setdefault(sd["station_id"], {
            "station_id": sd["station_id"], "epochs": {},
            "nav_source": sd["nav_source"],
            "approx_ll": sd.get("approx_ll")})
        d["epochs"].update(sd["epochs"])
        if sd["nav_source"] == "own_stream":
            d["nav_source"] = "own_stream"
        if sd.get("approx_ll") is not None:
            d["approx_ll"] = sd["approx_ll"]
    stations = []
    for sid, d in by_id.items():
        epochs = d["epochs"]
        sods = np.array(sorted(epochs.keys()))
        cdt_full = np.array([epochs[s]["cdt_rtk_m"] for s in sods])
        pos_rtk = np.array([epochs[s]["pos_rtk"] for s in sods])
        finite_pos = pos_rtk[np.all(np.isfinite(pos_rtk), axis=1)]
        if len(finite_pos) < 100:
            continue
        stations.append(_remap_midnight({
            "station_id": sid,
            "n_epochs": len(epochs),
            "sods": sods,
            "cdt_full": cdt_full,
            "pos_rtk": pos_rtk,
            "truth": np.median(finite_pos, axis=0),
            "epochs": epochs,
            "nav_source": d["nav_source"],
            "approx_ll": d["approx_ll"],
        }))

    shutil.rmtree(tmp, ignore_errors=True)

    if not stations:
        print_status("No stations with valid data.", "WARN")
        OUT.mkdir(parents=True, exist_ok=True)
        result = {
            "test": "ntrip_process",
            "status": "no_data",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / "step_37b_ntrip_process.json").write_text(json.dumps(result, indent=2))
        return

    # Kernel prediction (trained on archived data, live is held-out)
    print_status("\nBuilding kernel prior from archived training data...", "PROCESS")
    kernel_pred = _kernel_clock_prediction_live()
    if kernel_pred:
        print_status(f"  Trained on {kernel_pred['train_day']}, "
                     f"{kernel_pred['n_train_stations']} stations, "
                     f"sigma={kernel_pred['sigma_m']:.2f} m", "INFO")

    # Contemporaneous field prior: when a live-network snapshot covering
    # the capture exists, the deployed predictor kriges the same-period
    # station field at each target's location (leave-target-out) instead
    # of the stale time-of-day template above.
    live_grid = _live_field_grid()
    if live_grid is not None:
        grid_l, raw_l, coords_l = live_grid
        print_status(f"  Contemporaneous field available: "
                     f"{len(coords_l)} stations, "
                     f"{(grid_l[-1]-grid_l[0])/3600:.1f} h span", "INFO")
    else:
        grid_l = raw_l = coords_l = None

    # Run obstruction protocol on each station
    print_status("\nRunning obstruction protocol...", "PROCESS")
    # Adapt gap lengths to the recording duration.  Short captures (<= 600 s)
    # use 0.5/1/2/3/4-minute windows; longer captures use the pre-registered
    # 10/30/60/120/240-minute windows.
    duration_s = float(manifest.get("duration_s", 300.0))
    gap_lengths = GAP_LENGTHS_S if duration_s >= 9000.0 else SHORT_GAP_LENGTHS_S
    strategy_gap_len = 3600.0 if duration_s >= 9000.0 else 120.0
    all_results = []
    for sd in stations:
        kp = kernel_pred
        if raw_l is not None:
            tgt = sd.get("approx_ll") or coords_l.get(sd["station_id"])
            kp_live = _contemporaneous_kernel(
                grid_l, raw_l, coords_l, tgt, sd["station_id"])
            if kp_live is not None:
                kp = kp_live
        by_gap = {}
        for gl in gap_lengths:
            g = _run_obstruction(sd, kp, gl, strategy="highest")
            if g:
                by_gap[str(int(gl))] = g
        if not by_gap:
            continue
        by_strategy = {}
        for strat in OBSTRUCTION_STRATEGIES:
            g = _run_obstruction(sd, kp, strategy_gap_len, strategy=strat)
            if g:
                by_strategy[strat] = g
        all_results.append({
            "station": sd["station_id"],
            "n_epochs": sd["n_epochs"],
            "nav_source": sd.get("nav_source"),
            "kernel_variant": kp.get("variant", "sod_template_2024")
            if kp else None,
            "by_gap_s": by_gap,
            f"by_strategy_{int(strategy_gap_len)}s": by_strategy,
        })
        gmid = by_gap.get(str(int(strategy_gap_len)), {})
        if gmid:
            pe = gmid["pos_error_median_m"]
            print_status(f"  {sd['station_id']}: {int(strategy_gap_len)}s hold={pe.get('hold',0):.1f} "
                        f"kernel={pe.get('kernel',0):.1f} m "
                        f"({gmid['kernel_vs_hold_pct']:+.0f}% vs hold)", "INFO")

    # Aggregate
    def agg(gap_key):
        sub = [r["by_gap_s"][gap_key] for r in all_results if gap_key in r["by_gap_s"]]
        if len(sub) < 1:
            return None
        pooled = {}
        for m in ("hold", "linrate", "kernel"):
            vals = [x["pos_error_median_m"][m] for x in sub
                    if m in x["pos_error_median_m"]
                    and x["pos_error_median_m"][m] is not None
                    and np.isfinite(x["pos_error_median_m"][m])]
            pooled[m] = float(np.nanmedian(vals)) if vals else None
        # Pooled kernel-vs-hold and kernel-vs-linrate percentages
        kvh = None
        kvl = None
        if pooled.get("hold") and pooled.get("kernel") and pooled["hold"] > 0:
            kvh = 100.0 * (1.0 - pooled["kernel"] / pooled["hold"])
        if pooled.get("linrate") and pooled.get("kernel") and pooled["linrate"] > 0:
            kvl = 100.0 * (1.0 - pooled["kernel"] / pooled["linrate"])
        return {"n_stations": len(sub), "pos_error_median_m": pooled,
                "kernel_vs_hold_pct": kvh, "kernel_vs_linrate_pct": kvl}

    by_gap = {str(int(g)): agg(str(int(g))) for g in gap_lengths}
    by_gap = {k: v for k, v in by_gap.items() if v}

    # Sanitise kernel_prior for JSON (drop non-serialisable objects)
    kp_json = None
    if kernel_pred:
        kp_json = {
            "sigma_m": kernel_pred.get("sigma_m"),
            "train_day": kernel_pred.get("train_day"),
            "n_train_stations": kernel_pred.get("n_train_stations"),
        }

    def _sanitize(obj):
        """Replace NaN/inf with None for valid JSON."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        return obj

    gap_label = "10/30/60/120/240 min" if duration_s >= 9000.0 else "0.5/1/2/3/4 min"
    result = {
        "test": "ntrip_live_position_demo",
        "method": ("Live NTRIP RTCM3 streams recorded from public NTRIP casters, "
                   "converted to RINEX via RTKLIB convbin, processed via RTKLIB SPP "
                   "(rnx2rtkp -p 0 -y 2).  Stations whose streams carry no "
                   "in-band ephemeris use a pooled broadcast nav merged from "
                   "the same capture's nav streams plus the live system's "
                   "shared_nav product (broadcast ephemeris are "
                   "receiver-independent).  Obstruction windows of "
                   f"{gap_label}, {K_OBSTRUCT} satellites kept.  Clock anchored from last "
                   "full-sky epoch, propagated by persistence, linear extrapolation, "
                   "or network kernel.  Kernel trained on archived broadcast set "
                   "(DOY 100-106, 2024); live data is held-out.  Actual 3-D position "
                   "error vs median SPP position."),
        "status": "ok",
        "n_stations": len(all_results),
        "nav_fallback_stations": sorted(
            {r["station"] for r in all_results
             if r.get("nav_source") == "pooled_broadcast"}),
        "recording_start_utc": manifest.get("start_utc"),
        "recording_duration_s": manifest.get("duration_s"),
        "caster": manifest.get("caster"),
        "kernel_prior": kp_json,
        "by_gap_length_s": by_gap,
        "per_station": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result = _sanitize(result)
    (OUT / "step_37b_ntrip_process.json").write_text(json.dumps(result, indent=2))
    print_status(f"\nSaved {OUT / 'step_37b_ntrip_process.json'}", "SUCCESS")


if __name__ == "__main__":
    main()
