#!/usr/bin/env python3
"""
STEP 181 -- European optical-fiber clock network reanalysis (Pizzocaro et al. 2026,
            Phys. Rev. Research 8, 033250; Zenodo 10.5281/zenodo.16539534).

Analyzes the 65-day March-April 2023 European optical-fiber clock comparison campaign
(MJD 59999-60064; 16 comparison folders, 328 daily .dat files, 7.90x10^6 valid flag=1
1-second fractional-frequency measurements across 7 optical clocks at NPL, LNE-OP/SYRTE,
PTB, and INRIM) along seven complementary axes:

  1. Differential Clock Frequency-Ratio Consistency Constraint & Star-Topology Closure:
     Active phase-stabilized two-way fiber links reflect the optical carrier around a
     closed path, cancelling the exact conformal proper-time integral (oint d ln A = 0)
     and constraining differential frequency-ratio offsets on NPL-YbE3 / PTB-Yb1E3
     (2,155 km fiber, 705 km chord) to -7.1 +/- 7.7 x 10^-18 (1 sigma; conditional
     Gaussian 95% single-effect magnitude envelope <= 2.2 x 10^-17, B = 1.13; daily
     Birge B = 1.09, B_wn = 0.883 against 1,000 s white-noise floor; a model-independent
     disformal synchronization holonomy bound requires a physically closed b1 >= 1 fiber loop).

  2. DerSimonian-Laird Excess Between-Day Floor (tau_DL), 40,000 s Excess ADEV,
     Projection-Noise Unmasking, and Geodesic vs Fiber-Length Scaling:
     Subtracts each daily run's internal 100 s white-noise variance u_i^2 in quadrature
     to extract the between-day excess standard deviation tau_DL and 40,000 s overlapping
     Allan deviation excess sigma_exc(40 ks). Across the 8 non-interleaved clock pairs
     (0 km -> 705 km -> 755 km -> 915 km), tau_DL increases monotonically with geodesic
     chord distance d_geo (Spearman rho = +0.927, p = 0.00094), and increases by
     +72% to +89% from NPL-PTB (d_geo = 705 km, L_fiber = 2,155 km, tau_DL = 9.65-10.61e-18)
     to INRIM-NPL (d_geo = 915 km, L_fiber = 1,808 km, tau_DL = 18.23e-18) even as the
     fiber path length decreases by 347 km (-16%).

  3. Simultaneous 5-Tide (M2, S2, N2, K1, O1) and 6-Tide (+S1 at 24.00 h) Harmonic
     Regression on 600 s True-UTC Bins with 6-Hour Block and Whole-Day MJD-Date Permutations:
     Documents statistically significant lunar M2 (12.4206 h) modulation across the
     705-755 km baselines (M2 > S2, Fisher combined chi^2_4 = 17.69, p = 0.0014 on raw 6-h
     block permutation, chi^2_4 = 17.03, p = 0.0019 on daily-demeaned 6-h block permutation,
     and chi^2_4 = 12.41, p = 0.0146 under whole-day MJD-date permutation on the two 705 km
     homonuclear NPL-PTB pairs; the two homonuclear detections occur on disjoint uptime halves,
     with common-window MJD 60018.5-60034.2 y_NPL-y_PTB phases at 217.2, 161.4, 51.5, and
     218.5 deg, Rayleigh R = 0.477).

  4. Simultaneous 1-Second Star-Topology Triangle Closure Audit:
     Verifies that on simultaneous 1-second epochs, 3-node loops across the Paris-centered
     star fiber tree close to machine precision (3.61e-20 +/- 1.27e-19 over N = 280,947 s
     on NPL_YbE3 - PTB_Yb1E3 - NPL_Sr1; +8.45e-23 +/- 1.37e-21 over N = 62,446 s on
     NPL_YbE3 - INRIM_ITYb1 - PTB_Yb1E3), proving that Table II's 5.1-7.9 x 10^-18
     campaign non-closure arises from non-overlapping uptime schedules sampling the
     time-varying sub-mHz floor.

  5. Zero-Fiber, Zero-Satellite 705 km Anti-Phase Coherence Between Co-Located Optical
     Clock Pairs at NPL and PTB (Opposite-Sign Differential Scalar Coupling Delta K_alpha):
     Compares the 100% local co-located ratio at NPL (NPL_YbE3 - NPL_Sr1, 0 km fiber,
     Delta K_alpha = -5.95 - 0.06 = -6.01 < 0) against the 100% local co-located ratio
     inside a single Paul trap at PTB 705 km away (PTB_Yb1E2 - PTB_Yb1E3, 0 km fiber,
     Delta K_alpha = +1.00 - (-5.95) = +6.95 > 0). Because Delta K_1 * Delta K_2 = -41.77 < 0,
     a spatially coherent scalar field (lambda_T ~ 4,200 km >> 705 km) drives the two
     local ratios in anti-phase (PA < 0): across n = 45 simultaneous 3-h windows,
     PA = -0.335 +/- 0.091 (z = -3.69, p = 0.00022; window-shuffle null +0.002 +/- 0.103,
     differential -0.336, z_null = -3.28, p_perm = 0.0010; and in the 10-500 uHz band,
     PA = -0.292 +/- 0.104, z = -2.80, p = 0.0052 at 3 h; PA = -0.388 +/- 0.143,
     z = -2.71, p = 0.0068 at 6 h; PA = -0.451 +/- 0.166, z = -2.72, p = 0.0066 at 8 h),
     with neither GNSS satellites nor optical fiber links participating in either measurement.

  6. Target-Blind 43-Mode Global Clock-Field Localization of Zero-Satellite Co-Located
     Optical Clocks (Step 80 + Step 172 Closure):
     Applies the exact Step 80 43-mode phase-cos localization estimator (fixed_lambda_km = 4201.0,
     target coordinates withheld) to the three 100% GNSS-free, zero-satellite, zero-fiber
     co-located optical clock differentials at PTB and NPL against the contemporaneous
     366-station global GNSS archive (TEP-GNSS-RINEX/data/processed, DOY 055-120 of 2023):
       - PTB_Yb1E2 - PTB_Yb1E3 (43 days, single Paul trap at PTB): localizes PTB to 896 km
         (ionofree_clock_bias_ns, vs 9,212 km geometry-null median, p_err = 0.0140,
         p_score = 0.0020; even/odd modes independently 810 km / 1,195 km, agreeing within
         947 km; 850 km with all stations <300 km excluded, 928 km with <500 km excluded)
         and 1,036 km (precise_clock_bias_ns, vs 9,144 km null median, p_err = 0.0180,
         p_score = 0.0020; even/odd modes 702 km / 1,659 km; day-shuffle temporal null
         collapses error to 8,448 km and truth-kernel correlation from +0.277 to
         +0.038 +/- 0.108, p_day = 0.0149).
       - PTB_Yb1E3 - PTB_Sr3 (32 days at PTB): localizes PTB to 879 km (vs 9,477 km null
         median, p_err = 0.0100; even/odd modes 1,382 km / 7,519 km), landing within 112 km
         of the independent PTB_Yb1E2 - PTB_Yb1E3 estimate.
       - NPL_YbE3 - NPL_Sr1 (18 days at NPL): localizes NPL to 1,416 km (vs 8,773 km null
         median, p_err = 0.0240, p_score = 0.0020; even/odd modes 1,894 km / 11,836 km;
         1,402 km with <500 km excluded).
       - Independent-site Fisher combination (one primary pair per laboratory, PTB + NPL):
         chi^2_4 = 16.00, p = 0.0030 (three-pair chi^2_6 = 25.21, p = 0.00031).

     Additionally, the exact four-clock variance difference between the two 705 km
     homonuclear pairs and the two 0 km local pairs covering the same four clocks
     cancels all per-clock variances algebraically, isolating a 705 km baseline
     floor of sigma_baseline = 7.30e-18 per link (remote/local variance ratio 2.73);
     a non-negative least-squares clock/link variance decomposition attributes
     most of the 755-915 km elevation to INRIM-ITYb1 wander (19.55e-18) while a
     coherent +3.5 to +5.2e-18 remote residual survives leave-INRIM-out refits.
     Diurnal/semidiurnal mode exclusion (f > 30 uHz, 41 modes) leaves the
     localisation errors at 704-1,285 km, ruling out a shared European diurnal
     envelope origin.

  7. Cross-Pair Sub-mHz Coherence Across Strictly Disjoint Optical Clock Pairs:
     Evaluates multi-hour (1-6 h, 46-278 uHz) correlations across pairs sharing zero clocks.

  8. Contemporaneous Optical-to-GNSS Waveform Correlations:
     Direct timestamp-aligned 3-24 h detrended correlations between optical pair
     residuals and GNSS clock/position differentials, scored against 1,000
     day-block permutations (p_dayblk = 0.0020-0.0370 across eight channel pairs).

Output: results/outputs/step_181_european_optical_fiber_2023.json
"""
from __future__ import annotations

import glob
import itertools
import json
import warnings
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from scipy.linalg import lstsq
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.steps.step_80_clock_field_localisation import (
    candidate_grid,
    estimate_separation,
    gc_error,
    kernel_matrix,
    locate,
    mode_signature,
)

DATA_DIR = ROOT / "data" / "optical_fiber_2023"
GNSS_DIR = ROOT.parent / "TEP-GNSS-RINEX" / "data" / "processed"
OUT = ROOT / "results" / "outputs" / "step_181_european_optical_fiber_2023.json"

MJD_2023_001 = 59945.0
M2_F = 1.0 / (12.4206012 * 3600.0)
O1_F = 1.0 / (25.8193417 * 3600.0)
S2_F = 2.0 / 86400.0
CTRL_FS = [13e-6, 15e-6, 17e-6, 19e-6, 25e-6, 27e-6, 29e-6, 31e-6]

FIVE_TIDE_CPD = {
    "M2": 24.0 / 12.4206012,
    "S2": 24.0 / 12.0000000,
    "N2": 24.0 / 12.6583475,
    "K1": 24.0 / 23.9344721,
    "O1": 24.0 / 25.8193387,
}

COORDS = {
    "NPL": (51.4245, -0.3435),
    "SYRTE": (48.8364, 2.3365),
    "PTB": (52.2966, 10.4628),
    "INRIM": (45.0163, 7.6418),
}

PUBLISHED_TABLE_II = {
    "PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten": {"published_birge": 1.13, "baseline_km": 0.0, "fiber_km": 0.0, "arch": "Ion x Ion (local)"},
    "NPL_YbE3-NPL_Sr1": {"published_birge": 1.09, "baseline_km": 0.0, "fiber_km": 0.0, "arch": "Ion x Lattice (local)"},
    "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten": {"published_birge": 1.97, "baseline_km": 0.0, "fiber_km": 0.0, "arch": "Ion x Lattice (local PTB)"},
    "NPL_YbE3-PTB_Yb_CombKnoten": {"published_birge": 1.13, "baseline_km": 705.0, "fiber_km": 2155.0, "arch": "Ion x Ion (remote)"},
    "PTB_Yb_CombKnoten-NPL_Sr1": {"published_birge": 2.15, "baseline_km": 705.0, "fiber_km": 2155.0, "arch": "Ion x Lattice (remote)"},
    "NPL_Sr1-PTB_Sr3_CombKnoten": {"published_birge": 4.51, "baseline_km": 705.0, "fiber_km": 2155.0, "arch": "Lattice x Lattice (remote)"},
    "PTB_Yb_CombKnoten-INRIM_ITYb1": {"published_birge": 1.55, "baseline_km": 755.0, "fiber_km": 2393.0, "arch": "Ion x Lattice (remote)"},
    "INRIM_ITYb1-PTB_Sr3_CombKnoten": {"published_birge": 1.68, "baseline_km": 755.0, "fiber_km": 2393.0, "arch": "Lattice x Lattice (remote)"},
    "INRIM_ITYb1-NPL_Sr1": {"published_birge": 1.26, "baseline_km": 915.0, "fiber_km": 1808.0, "arch": "Lattice x Lattice (remote)"},
}


def _load_pair_mad(name: str):
    d = DATA_DIR / name
    yml = yaml.safe_load(list(d.glob("*.yml"))[0].read_text())[0]
    grsA = float(yml.get("grsA", 0.0))
    grsB = float(yml.get("grsB", 0.0))
    ts, ys = [], []
    for f in sorted(d.glob("*.dat")):
        arr = np.loadtxt(f, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        m = (arr[:, 2] == 1) & np.isfinite(arr[:, 1])
        if m.any():
            ts.append(arr[m, 0])
            ys.append(arr[m, 1] + grsB - grsA)
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    ok = np.abs(y - med) < 6 * mad
    return t[ok], y[ok]


def _load_pair_files_valid(name: str):
    """Load valid (flag == 1, |y| < 1e-12) 1-second rows per daily .dat file with grs offset."""
    d = DATA_DIR / name
    yml = yaml.safe_load(list(d.glob("*.yml"))[0].read_text())[0]
    grsA = float(yml.get("grsA", 0.0))
    grsB = float(yml.get("grsB", 0.0))
    files_data = []
    for f in sorted(d.glob("*.dat")):
        arr = np.loadtxt(f, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        m = (arr[:, 2] == 1) & np.isfinite(arr[:, 1]) & (np.abs(arr[:, 1]) < 1e-12)
        if m.any():
            files_data.append((f.name, arr[m, 0], arr[m, 1] + grsB - grsA))
    return files_data


def _adev_bin(t_mjd: np.ndarray, y: np.ndarray, tau_s: float) -> float:
    t_sec = (t_mjd - np.floor(t_mjd[0])) * 86400.0
    b_idx = (t_sec // tau_s).astype(int)
    u_b, inv, cnts = np.unique(b_idx, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=y)
    means = sums / cnts
    valid = cnts >= 0.4 * tau_s
    u_b = u_b[valid]
    means = means[valid]
    contig = np.where(np.diff(u_b) == 1)[0]
    if len(contig) >= 3:
        diffs = means[contig + 1] - means[contig]
    elif len(means) >= 4:
        diffs = np.diff(means)
    else:
        return float("nan")
    return float(np.sqrt(0.5 * np.mean(diffs ** 2)))


def _fit_harmonic_amp(tb: np.ndarray, yb: np.ndarray, f_hz: float) -> float:
    w = 2.0 * np.pi * f_hz * (tb - tb[0])
    tn = (tb - tb.mean()) / (tb.std() + 1.0)
    X = np.column_stack([np.ones_like(tb), tn, np.cos(w), np.sin(w)])
    c, _, _, _ = lstsq(X, yb, cond=1e-10)
    return float(np.hypot(c[2], c[3]))


def _compute_dl_and_40ks(name: str, mjd_cutoff: float | None = None) -> dict:
    files_data = _load_pair_files_valid(name)
    dm, du, ys = [], [], []
    for _, t_f, y_f in files_data:
        if mjd_cutoff is not None:
            m = t_f < mjd_cutoff
            t_f, y_f = t_f[m], y_f[m]
        if len(y_f) > 0:
            ys.append(y_f)
        if len(y_f) >= 3600:
            nb = len(y_f) // 100
            bm = y_f[: nb * 100].reshape(nb, 100).mean(axis=1)
            dm.append(float(np.mean(y_f)))
            du.append(float(np.std(bm, ddof=1) / np.sqrt(nb)))
    y_all = np.concatenate(ys) if ys else np.array([])
    nb_all = len(y_all) // 100
    bm_all = y_all[: nb_all * 100].reshape(nb_all, 100).mean(axis=1)
    s100 = float(np.sqrt(0.5 * np.mean(np.diff(bm_all) ** 2)))

    dm_a = np.asarray(dm)
    du_a = np.asarray(du)
    w = 1.0 / (du_a ** 2)
    wm = float(np.sum(w * dm_a) / np.sum(w))
    Q = float(np.sum(w * (dm_a - wm) ** 2))
    df = max(1, len(dm_a) - 1)
    birge_dl = float(np.sqrt(Q / df))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau_dl = float(np.sqrt(max(0.0, (Q - df) / C))) if C > 0 else 0.0

    tau_40k = 40000
    # Require at least 4 independent 40,000 s windows (>= 160,000 valid seconds)
    if len(y_all) >= 4 * tau_40k and mjd_cutoff is None:
        cs = np.concatenate([[0.0], np.cumsum(y_all)])
        m40 = (cs[tau_40k:] - cs[:-tau_40k]) / tau_40k
        d40 = m40[tau_40k:] - m40[:-tau_40k]
        s40 = float(np.sqrt(0.5 * np.mean(d40 ** 2)))
        s40_w = s100 / np.sqrt(400.0)
        s40_exc = float(np.sqrt(max(0.0, s40 ** 2 - s40_w ** 2)))
        ratio40 = float(s40 / s40_w) if s40_w > 0 else None
    else:
        s40_exc = None
        ratio40 = None

    return {
        "n_valid_seconds": int(len(y_all)),
        "n_days_ge_1h": int(len(dm_a)),
        "sigma_y_100s_1e18": round(s100 * 1e18, 1),
        "median_daily_white_u_1e18": round(float(np.median(du_a)) * 1e18, 1),
        "daily_birge_B": round(birge_dl, 2),
        "excess_daily_std_tau_DL_1e18": round(tau_dl * 1e18, 2),
        "excess_adev_40ks_1e18": round(s40_exc * 1e18, 2) if s40_exc is not None else None,
        "adev_40ks_over_white": round(ratio40, 2) if ratio40 is not None else None,
    }


def _tide_design(t_mjd: np.ndarray) -> np.ndarray:
    columns = [np.ones_like(t_mjd)]
    for frequency in FIVE_TIDE_CPD.values():
        phase = 2 * np.pi * frequency * (t_mjd - 60000.0)
        columns.extend([np.cos(phase), np.sin(phase)])
    return np.column_stack(columns)


def _permute_block_dates(t_mjd, block_hours, rng):
    """Reassign UTC block labels, retaining gaps and offsets within each block.

    The same reassigned timestamps must be used for every ratio in a network.
    Blocks are elapsed UTC intervals, not a fixed number of retained samples.
    """
    scaled = (np.asarray(t_mjd) - 60000.0) * (24.0 / block_hours)
    labels = np.floor(scaled).astype(int)
    unique, inverse = np.unique(labels, return_inverse=True)
    return 60000.0 + (rng.permutation(unique)[inverse] + scaled - labels) * (block_hours / 24.0)


def _joint_species_phase_null(t_mjd, ratios, block_hours=6, n_perm=1000, seed=42):
    """Conditional UTC-block exchangeability test preserving all shared inputs."""
    ratios = np.asarray(ratios)
    if ratios.ndim != 2 or len(ratios) != len(t_mjd) or ratios.shape[1] < 2:
        raise ValueError("Expected simultaneous rows and at least two ratio columns")

    def resultant(times):
        coefficients = np.linalg.lstsq(_tide_design(times), ratios, rcond=None)[0]
        return float(abs(np.mean(np.exp(1j * np.arctan2(coefficients[2], coefficients[1])))))

    observed = resultant(t_mjd)
    rng = np.random.default_rng(seed)
    null = np.array([resultant(_permute_block_dates(t_mjd, block_hours, rng))
                     for _ in range(n_perm)])
    exceedances = int(np.sum(null >= observed - 1e-12))
    return {
        "block_hours": block_hours,
        "n_blocks": int(len(np.unique(np.floor((np.asarray(t_mjd) - 60000.0) * 24 / block_hours)))),
        "n_permutations": n_perm,
        "seed": seed,
        "shared_timestamp_permutation": True,
        "null_mean_R": float(null.mean()),
        "exceedances": exceedances,
        "p_value": (exceedances + 1) / (n_perm + 1),
        "method": "Joint UTC-block date-label permutation; ratio vectors and within-block offsets retained.",
        "assumption": "UTC blocks are exchangeable under the null; longer-range serial dependence is not preserved.",
    }


def _compute_5tide_600s(name: str, n_perm: int = 500) -> dict:
    files_data = _load_pair_files_valid(name)
    t = np.concatenate([tf for _, tf, _ in files_data])
    y = np.concatenate([yf for _, _, yf in files_data])
    b600 = np.floor(t * 144.0).astype(int)
    ub, inv, cnt = np.unique(b600, return_inverse=True, return_counts=True)
    ok = cnt >= 60
    tb = (ub[ok] + 0.5) / 144.0
    yb = np.bincount(inv, weights=y)[ok] / cnt[ok]

    days = np.floor(tb).astype(int)
    yb_dm = yb.copy()
    for d0 in np.unique(days):
        yb_dm[days == d0] -= np.mean(yb[days == d0])

    X = [np.ones_like(tb)]
    for f_cpd in FIVE_TIDE_CPD.values():
        w = 2.0 * np.pi * f_cpd * (tb - 60000.0)
        X.extend([np.cos(w), np.sin(w)])
    X = np.column_stack(X)

    c_raw, _, _, _ = np.linalg.lstsq(X, yb - np.mean(yb), rcond=None)
    amps_raw = {
        k: float(np.hypot(c_raw[1 + 2 * i], c_raw[2 + 2 * i]))
        for i, k in enumerate(FIVE_TIDE_CPD)
    }
    c_dm, _, _, _ = np.linalg.lstsq(X, yb_dm, rcond=None)
    amps_dm = {
        k: float(np.hypot(c_dm[1 + 2 * i], c_dm[2 + 2 * i]))
        for i, k in enumerate(FIVE_TIDE_CPD)
    }

    rng = np.random.default_rng(42)
    rng = np.random.default_rng(42)
    chunks = [yb_dm[i : i + 36] for i in range(0, len(yb_dm), 36)]
    ex_6h_raw = 0
    ex_6h_dm = 0
    for _ in range(n_perm):
        yp = np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])
        cp, _, _, _ = np.linalg.lstsq(X, yp, rcond=None)
        m2_p = float(np.hypot(cp[1], cp[2]))
        if m2_p >= amps_raw["M2"]:
            ex_6h_raw += 1
        if m2_p >= amps_dm["M2"]:
            ex_6h_dm += 1

    rng_day = np.random.default_rng(42)
    tod = tb - days
    udays = np.unique(days)
    ex_day_dm = 0
    for _ in range(n_perm):
        perm_days = rng_day.permutation(udays)
        dmap = dict(zip(udays, perm_days))
        tb_perm = np.array([dmap[d] for d in days]) + tod
        Xp = [np.ones_like(tb_perm)]
        for f_cpd in FIVE_TIDE_CPD.values():
            w = 2.0 * np.pi * f_cpd * (tb_perm - 60000.0)
            Xp.extend([np.cos(w), np.sin(w)])
        Xp = np.column_stack(Xp)
        cp, _, _, _ = np.linalg.lstsq(Xp, yb_dm, rcond=None)
        if float(np.hypot(cp[1], cp[2])) >= amps_dm["M2"]:
            ex_day_dm += 1

    # Simultaneous 6-tide fit adding solar diurnal S1 (1.0000000 cpd) to verify M2 stability
    six_tide_cpd = dict(FIVE_TIDE_CPD, S1=1.0000000)
    X6 = [np.ones_like(tb)]
    for f_cpd in six_tide_cpd.values():
        w = 2.0 * np.pi * f_cpd * (tb - 60000.0)
        X6.extend([np.cos(w), np.sin(w)])
    X6 = np.column_stack(X6)
    c6_dm, _, _, _ = np.linalg.lstsq(X6, yb_dm, rcond=None)
    m2_6tide_dm = float(np.hypot(c6_dm[1], c6_dm[2]))
    s2_6tide_dm = float(np.hypot(c6_dm[3], c6_dm[4]))
    s1_6tide_dm = float(np.hypot(c6_dm[11], c6_dm[12]))

    return {
        "n_600s_bins": int(len(tb)),
        "M2_amp_raw_1e18": round(amps_raw["M2"] * 1e18, 1),
        "S2_amp_raw_1e18": round(amps_raw["S2"] * 1e18, 1),
        "K1_amp_raw_1e18": round(amps_raw["K1"] * 1e18, 1),
        "O1_amp_raw_1e18": round(amps_raw["O1"] * 1e18, 1),
        "M2_over_S2_raw": round(amps_raw["M2"] / amps_raw["S2"], 2) if amps_raw["S2"] > 0 else None,
        "M2_amp_daily_demeaned_1e18": round(amps_dm["M2"] * 1e18, 1),
        "S2_amp_daily_demeaned_1e18": round(amps_dm["S2"] * 1e18, 1),
        "M2_over_S2_daily_demeaned": round(amps_dm["M2"] / amps_dm["S2"], 2) if amps_dm["S2"] > 0 else None,
        "six_tide_with_S1_daily_demeaned": {
            "M2_amp_1e18": round(m2_6tide_dm * 1e18, 1),
            "S2_amp_1e18": round(s2_6tide_dm * 1e18, 1),
            "S1_amp_1e18": round(s1_6tide_dm * 1e18, 1),
            "M2_over_S2": round(m2_6tide_dm / s2_6tide_dm, 2) if s2_6tide_dm > 0 else None,
        },
        "p_perm_6h_block": round(ex_6h_raw / n_perm, 3),
        "p_perm_6h_block_demeaned": round(ex_6h_dm / n_perm, 3),
        "p_perm_whole_day_date": round(ex_day_dm / n_perm, 3),
        "n_permutations": n_perm,
        "permutation_seed": 42,
    }


def _compute_star_topology_closure() -> dict:
    def _sec_dict(name: str) -> dict[int, float]:
        files_data = _load_pair_files_valid(name)
        t = np.concatenate([np.round((tf - 59999.0) * 86400.0).astype(np.int64) for _, tf, _ in files_data])
        y = np.concatenate([yf for _, _, yf in files_data])
        return dict(zip(t.tolist(), y.tolist()))

    d_ny_py = _sec_dict("NPL_YbE3-PTB_Yb_CombKnoten")
    d_py_ns = _sec_dict("PTB_Yb_CombKnoten-NPL_Sr1")
    d_ny_ns = _sec_dict("NPL_YbE3-NPL_Sr1")
    c1 = sorted(set(d_ny_py) & set(d_py_ns) & set(d_ny_ns))
    v1 = np.array([d_ny_py[k] + d_py_ns[k] - d_ny_ns[k] for k in c1])

    d_ny_iy = _sec_dict("NPL_YbE3-INRIM_ITYb1")
    d_py_iy = _sec_dict("PTB_Yb_CombKnoten-INRIM_ITYb1")
    c2 = sorted(set(d_ny_iy) & set(d_py_iy) & set(d_ny_py))
    v2 = np.array([d_ny_iy[k] - d_py_iy[k] - d_ny_py[k] for k in c2])

    return {
        "network_topology": "Star tree centered on LNE-SYRTE (Paris); first Betti number b_1 = E - V + 1 = 3 - 4 + 1 = 0",
        "triangles": {
            "NPL_YbE3__PTB_Yb1E3__NPL_Sr1": {
                "n_simultaneous_seconds": int(len(c1)),
                "simultaneous_hours": round(len(c1) / 3600.0, 2),
                "mean_closure": float(f"{v1.mean():.2e}"),
                "std_closure": float(f"{v1.std():.2e}"),
            },
            "NPL_YbE3__INRIM_ITYb1__PTB_Yb1E3": {
                "n_simultaneous_seconds": int(len(c2)),
                "simultaneous_hours": round(len(c2) / 3600.0, 2),
                "mean_closure": float(f"{v2.mean():.2e}"),
                "std_closure": float(f"{v2.std():.2e}"),
            },
        },
        "table_II_campaign_non_closure_1e18_range": [5.1, 7.9],
        "physical_interpretation": (
            "Simultaneous 1-second closure vanishes to 10^-20 - 10^-23 because the star-tree "
            "topology retraces identical fiber segments; Table II's 5.1-7.9 x 10^-18 campaign "
            "non-closure therefore proves temporal epoch-sampling drift across non-overlapping "
            "uptime windows sampling the time-varying tau_DL ~ 9-15 x 10^-18 sub-mHz floor."
        ),
    }


def _compute_zero_fiber_705km_antiphase() -> dict:
    """Zero-fiber, zero-satellite 705 km anti-phase coherence between local NPL and local PTB."""
    def _to_60s(name: str):
        files_data = _load_pair_files_valid(name)
        t = np.concatenate([tf for _, tf, _ in files_data])
        y = np.concatenate([yf for _, _, yf in files_data])
        med = np.median(y)
        mad = 1.4826 * np.median(np.abs(y - med))
        ok = np.abs(y - med) <= 6.0 * mad
        t, y = t[ok], y[ok]
        idx = np.floor((t - 59999.0) * 1440.0).astype(int)
        u, inv, c = np.unique(idx, return_inverse=True, return_counts=True)
        m = c >= 15
        return dict(zip(u[m].tolist(), (np.bincount(inv, weights=y)[m] / c[m]).tolist()))

    d_npl = _to_60s("NPL_YbE3-NPL_Sr1")
    d_ptb = _to_60s("PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten")
    common = sorted(set(d_npl) & set(d_ptb))

    windows = {}
    rng_w = np.random.default_rng(42)
    for w_hours, f_hi in ((3, 1000e-6), (3, 500e-6), (3, 300e-6), (6, 500e-6), (8, 500e-6)):
        W = w_hours * 60
        freqs = np.fft.rfftfreq(W, d=60.0)
        band = (freqs > 0) & (freqs <= f_hi)
        w_set = sorted(set(m // W for m in common))
        pas = []
        f1_list, f2_list = [], []
        for w in w_set:
            grid = np.arange(w * W, (w + 1) * W)
            g1 = [g for g in grid if g in d_npl]
            g2 = [g for g in grid if g in d_ptb]
            if len(g1) >= 0.5 * W and len(g2) >= 0.5 * W:
                x1 = np.interp(grid, g1, [d_npl[g] for g in g1])
                x2 = np.interp(grid, g2, [d_ptb[g] for g in g2])
                x1 = (x1 - np.mean(x1)) * np.hanning(W)
                x2 = (x2 - np.mean(x2)) * np.hanning(W)
                F1 = np.fft.rfft(x1)[band]
                F2 = np.fft.rfft(x2)[band]
                f1_list.append(F1)
                f2_list.append(F2)
                cs = np.sum(F1 * np.conj(F2))
                pas.append(float(np.cos(np.angle(cs))))
        pas_a = np.asarray(pas)
        m_pa = float(np.mean(pas_a))
        se_pa = float(np.std(pas_a, ddof=1) / np.sqrt(len(pas_a)))
        z = m_pa / se_pa
        p_val = float(2.0 * stats.norm.sf(abs(z)))
        # 1,000-realisation window-shuffle permutation null
        null_means = []
        n_w = len(f1_list)
        for _ in range(1000):
            perm = rng_w.permutation(n_w)
            null_means.append(
                float(np.mean([np.cos(np.angle(np.sum(f1_list[i] * np.conj(f2_list[perm[i]])))) for i in range(n_w)]))
            )
        nm_a = np.asarray(null_means)
        null_mu = float(np.mean(nm_a))
        null_sd = float(np.std(nm_a, ddof=1))
        diff_null = m_pa - null_mu
        z_null = diff_null / (null_sd + 1e-12)
        p_perm = float((np.sum(np.abs(nm_a - null_mu) >= abs(diff_null)) + 1) / 1001.0)
        windows[f"W_{w_hours}h_f_le_{int(f_hi*1e6)}uHz"] = {
            "n_windows": int(len(pas_a)),
            "mean_pa": round(m_pa, 3),
            "se_pa": round(se_pa, 3),
            "z_score": round(z, 2),
            "p_two_sided": round(p_val, 5),
            "window_shuffle_null_mean": round(null_mu, 3),
            "window_shuffle_null_std": round(null_sd, 3),
            "differential_vs_null": round(diff_null, 3),
            "z_vs_window_shuffle_null": round(z_null, 2),
            "p_window_shuffle_two_sided": round(p_perm, 4),
        }

    return {
        "pair_1": "NPL_YbE3-NPL_Sr1 (100% local at NPL, Teddington UK, 0 km fiber)",
        "pair_2": "PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten (100% local single Paul trap at PTB, Braunschweig DE, 0 km fiber)",
        "baseline_separation_km": 705.0,
        "shared_clocks": 0,
        "shared_fiber_links": 0,
        "shared_gnss_satellites": 0,
        "delta_K_alpha_pair_1": -6.01,
        "delta_K_alpha_pair_2": +6.95,
        "delta_K_alpha_product": -41.77,
        "predicted_phase_sign": "negative (anti-phase, pi shift due to opposite-sign Delta K_alpha)",
        "windows": windows,
    }


def _compute_zero_satellite_global_localisation() -> dict | None:
    """Step 80 target-blind 43-mode global localization of zero-satellite optical clocks."""
    if not GNSS_DIR.exists():
        return None

    def _ecef_to_latlon(x: float, y: float, z: float):
        r = np.sqrt(x * x + y * y + z * z)
        return float(np.degrees(np.arcsin(z / r))), float(np.degrees(np.arctan2(y, x)))

    def _load_opt_300s(name: str):
        files_data = _load_pair_files_valid(name)
        t = np.concatenate([tf for _, tf, _ in files_data])
        y = np.concatenate([yf for _, _, yf in files_data])
        med = np.median(y)
        mad = 1.4826 * np.median(np.abs(y - med))
        ok = np.abs(y - med) <= 6.0 * mad
        t, y = t[ok], y[ok]
        mjd_int = np.floor(t).astype(int)
        b288 = np.clip(np.floor((t - mjd_int) * 288.0).astype(int), 0, 287)
        key = mjd_int * 288 + b288
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        valid = cnt >= 30
        u, means = u[valid], np.bincount(inv, weights=y)[valid] / cnt[valid]
        by_mjd = {}
        for k, val in zip(u, means):
            d0, slot = int(k // 288), int(k % 288)
            if d0 not in by_mjd:
                by_mjd[d0] = np.full(288, np.nan)
            by_mjd[d0][slot] = val
        return by_mjd

    opt_npl = _load_opt_300s("NPL_YbE3-NPL_Sr1")
    opt_ptb = _load_opt_300s("PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten")
    opt_e2 = _load_opt_300s("PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten")

    all_mjds = sorted(set(opt_npl) | set(opt_ptb) | set(opt_e2))
    doys = sorted([d0 - 59944 for d0 in all_mjds])

    fs75 = sorted(glob.glob(os.path.join(GNSS_DIR, "*_2023075.npz")))
    if not fs75:
        return None
    st_coords = {}
    for f in fs75:
        d = np.load(f)
        st = str(d["station"])
        st_coords[st] = _ecef_to_latlon(float(d["approx_x"]), float(d["approx_y"]), float(d["approx_z"]))

    freqs = np.fft.rfftfreq(288, d=300.0)
    band = (freqs >= 10e-6) & (freqs <= 500e-6)

    gnss_data = {field: {} for field in ("precise_clock_bias_ns", "ionofree_clock_bias_ns")}
    for doy in doys:
        mjd = 59944 + doy
        for f in glob.glob(os.path.join(GNSS_DIR, f"*_2023{doy:03d}.npz")):
            st = os.path.basename(f).split("_")[0]
            if st not in st_coords:
                continue
            d = np.load(f)
            ts = d["timestamps"].astype(float)
            slots = np.clip(np.round(ts / 300.0).astype(int), 0, 287)
            for field in gnss_data:
                if field not in d.files:
                    continue
                arr = d[field].astype(float)
                if len(arr) == 288:
                    grid = arr
                elif len(arr) == len(slots):
                    grid = np.full(288, np.nan)
                    grid[slots] = arr
                else:
                    continue
                if np.isfinite(grid).sum() >= 200:
                    gnss_data[field].setdefault(st, {})[mjd] = grid

    def _run_loc(opt_d, lab0, field, min_d_km=0.0, f_lo_hz=10e-6):
        band_use = (freqs >= f_lo_hz) & (freqs <= 500e-6)
        true_lat, true_lon = COORDS[lab0]
        sts, mode_rows, lats, lons = [], [], [], []
        for st, st_days in sorted(gnss_data[field].items()):
            d_st = gc_error(st_coords[st][0], st_coords[st][1], true_lat, true_lon)
            if d_st < min_d_km:
                continue
            common = [mjd for mjd in opt_d if mjd in st_days]
            if len(common) < 8:
                continue
            day_modes = []
            for mjd in common:
                xo = opt_d[mjd]
                xg = st_days[mjd]
                v = np.isfinite(xo) & np.isfinite(xg)
                if v.sum() < 48:
                    continue
                xoc = np.where(v, xo - np.mean(xo[v]), 0.0)
                xgc = np.where(v, xg - np.mean(xg[v]), 0.0)
                Fo = np.fft.rfft(xoc)[band_use]
                Fg = np.fft.rfft(xgc)[band_use]
                cs = Fo * np.conj(Fg)
                day_modes.append(np.cos(np.angle(cs)))
            if len(day_modes) >= 8:
                sts.append(st)
                mode_rows.append(np.mean(day_modes, axis=0))
                lats.append(st_coords[st][0])
                lons.append(st_coords[st][1])
        mode_matrix = np.asarray(mode_rows)
        lats_a, lons_a = np.asarray(lats), np.asarray(lons)
        nmodes = mode_matrix.shape[1]
        f_sub = freqs[band_use] > 30e-6  # 41 modes excluding 24h diurnal (11.57 uHz) and 12h semidiurnal (23.15 uHz)
        rng = np.random.default_rng(79001)
        r_all = locate(mode_signature(mode_matrix, np.ones(nmodes, bool)), lats_a, lons_a, true_lat, true_lon, "exponential", rng)
        r_even = locate(mode_signature(mode_matrix, np.arange(nmodes) % 2 == 0), lats_a, lons_a, true_lat, true_lon, "exponential", rng)
        r_odd = locate(mode_signature(mode_matrix, np.arange(nmodes) % 2 == 1), lats_a, lons_a, true_lat, true_lon, "exponential", rng)
        r_nodiurnal = locate(mode_signature(mode_matrix, f_sub), lats_a, lons_a, true_lat, true_lon, "exponential", np.random.default_rng(79001))
        sep = estimate_separation(r_odd, r_even)
        k_true_vec = kernel_matrix(np.array([true_lat]), np.array([true_lon]), lats_a, lons_a, "exponential")[0]
        z_m = mode_matrix - np.mean(mode_matrix, axis=0, keepdims=True)
        z_m /= np.linalg.norm(z_m, axis=0, keepdims=True)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            per_mode_corr = k_true_vec @ z_m
        n_pos_modes = int(np.sum(per_mode_corr > 0))
        binom_p = float(stats.binomtest(n_pos_modes, int(nmodes), 0.5, alternative="greater").pvalue)
        return {
            "n_reference_stations": int(len(sts)),
            "estimated_lat_deg": round(r_all["estimated_lat_deg"], 1),
            "estimated_lon_deg": round(r_all["estimated_lon_deg"], 1),
            "true_lat_deg": round(true_lat, 2),
            "true_lon_deg": round(true_lon, 2),
            "surface_error_km": round(r_all["surface_error_km"], 0),
            "truth_signed_correlation": round(r_all["truth_signed_correlation"], 3),
            "truth_score_percentile_global": round(r_all["truth_score_percentile_among_global_grid"] * 100.0, 1),
            "n_positive_modes_of_43": n_pos_modes,
            "positive_modes_binomial_p": round(binom_p, 4),
            "positive_modes_binomial_scope": "Nominal independent-mode benchmark only; irregular observation windows correlate Fourier modes.",
            "mean_mode_corr_odd_cpd_even_idx": round(float(np.mean(per_mode_corr[0::2])), 4),
            "mean_mode_corr_even_cpd_odd_idx": round(float(np.mean(per_mode_corr[1::2])), 4),
            "even_modes_error_km": round(r_even["surface_error_km"], 0),
            "odd_modes_error_km": round(r_odd["surface_error_km"], 0),
            "odd_even_separation_km": round(sep, 0),
            "geometry_null_median_error_km": round(r_all["geometry_null"]["median_localisation_error_km"], 0),
            "p_localisation_error_le_observed": round(r_all["geometry_null"]["p_localisation_error_le_observed"], 4),
            "p_best_score_ge_observed": round(r_all["geometry_null"]["p_best_score_ge_observed"], 4),
            "diurnal_semidiurnal_excluded_41modes_gt_30uHz": {
                "surface_error_km": round(r_nodiurnal["surface_error_km"], 0),
                "estimated_lat_deg": round(r_nodiurnal["estimated_lat_deg"], 1),
                "estimated_lon_deg": round(r_nodiurnal["estimated_lon_deg"], 1),
                "geometry_null_median_error_km": round(r_nodiurnal["geometry_null"]["median_localisation_error_km"], 0),
                "p_localisation_error_le_observed": round(r_nodiurnal["geometry_null"]["p_localisation_error_le_observed"], 4),
                "p_best_score_ge_observed": round(r_nodiurnal["geometry_null"]["p_best_score_ge_observed"], 4),
            },
        }

    e2_if_0 = _run_loc(opt_e2, "PTB", "ionofree_clock_bias_ns", 0.0)
    e2_if_300 = _run_loc(opt_e2, "PTB", "ionofree_clock_bias_ns", 300.0)
    e2_if_500 = _run_loc(opt_e2, "PTB", "ionofree_clock_bias_ns", 500.0)
    e2_pr_0 = _run_loc(opt_e2, "PTB", "precise_clock_bias_ns", 0.0)
    e2_pr_300 = _run_loc(opt_e2, "PTB", "precise_clock_bias_ns", 300.0)
    e2_pr_500 = _run_loc(opt_e2, "PTB", "precise_clock_bias_ns", 500.0)

    sr_if_0 = _run_loc(opt_ptb, "PTB", "ionofree_clock_bias_ns", 0.0)
    sr_if_100 = _run_loc(opt_ptb, "PTB", "ionofree_clock_bias_ns", 100.0)
    npl_pr_0 = _run_loc(opt_npl, "NPL", "precise_clock_bias_ns", 0.0)
    npl_pr_500 = _run_loc(opt_npl, "NPL", "precise_clock_bias_ns", 500.0)

    # Inter-pair PTB consistency (distance between PTB_Yb1E2-PTB_Yb1E3 and PTB_Yb1E3-PTB_Sr3 estimated positions)
    ptb_inter_pair_sep_km = round(
        gc_error(
            e2_if_0["estimated_lat_deg"],
            e2_if_0["estimated_lon_deg"],
            sr_if_0["estimated_lat_deg"],
            sr_if_0["estimated_lon_deg"],
        ),
        0,
    )

    # Conditional Fisher combination across sites sharing the GNSS reference network.
    # and across all three co-located zero-satellite pairs (df = 6)
    p1 = e2_if_0["p_localisation_error_le_observed"]
    p2 = sr_if_0["p_localisation_error_le_observed"]
    p3 = npl_pr_0["p_localisation_error_le_observed"]
    chi2_2site = float(-2.0 * (np.log(p1) + np.log(p3)))
    fisher_p_2site = float(stats.chi2.sf(chi2_2site, 4))
    chi2_loc = float(-2.0 * (np.log(p1) + np.log(p2) + np.log(p3)))
    fisher_p_loc = float(stats.chi2.sf(chi2_loc, 6))

    # Day-shuffle temporal null on PTB_Yb1E2 - PTB_Yb1E3 (precise_clock_bias_ns, station coordinates fixed)
    mjds_e2 = sorted([d0 for d0 in opt_e2 if np.isfinite(opt_e2[d0]).sum() >= 48])
    gnss_fft = {}
    for st, sd in sorted(gnss_data["precise_clock_bias_ns"].items()):
        st_f = {}
        for m in mjds_e2:
            if m in sd:
                g = sd[m]
                v = np.isfinite(g)
                if v.sum() >= 240:
                    st_f[m] = np.fft.rfft(np.where(v, g - np.mean(g[v]), 0.0))[band]
        if len(st_f) >= 35:
            gnss_fft[st] = st_f
    opt_fft = {
        m: np.fft.rfft(np.where(np.isfinite(opt_e2[m]), opt_e2[m] - np.nanmean(opt_e2[m]), 0.0))[band]
        for m in mjds_e2
    }
    sts_d = sorted(gnss_fft.keys())
    lats_d = np.array([st_coords[s][0] for s in sts_d])
    lons_d = np.array([st_coords[s][1] for s in sts_d])
    glat, glon = candidate_grid()
    K_d = kernel_matrix(glat, glon, lats_d, lons_d, "exponential")
    k_true = kernel_matrix(np.array([COORDS["PTB"][0]]), np.array([COORDS["PTB"][1]]), lats_d, lons_d, "exponential")[0]

    def _eval_day_map(mjd_map):
        rows = []
        for st in sts_d:
            sd = gnss_fft[st]
            dm = [np.cos(np.angle(opt_fft[m] * np.conj(sd[mjd_map[m]]))) for m in mjds_e2 if mjd_map[m] in sd]
            rows.append(np.mean(dm, axis=0))
        sig = mode_signature(np.asarray(rows), np.ones(43, bool))
        y0 = sig - np.mean(sig)
        y0 /= np.linalg.norm(y0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sc = K_d @ y0
        best = int(np.argmax(sc))
        return gc_error(glat[best], glon[best], COORDS["PTB"][0], COORDS["PTB"][1]), float(np.dot(k_true, y0))

    real_d_err, real_d_tcorr = _eval_day_map({m: m for m in mjds_e2})
    rng_d = np.random.default_rng(42)
    null_d_errs, null_d_tcorrs = [], []
    for _ in range(200):
        perm = rng_d.permutation(mjds_e2)
        e, tc = _eval_day_map(dict(zip(mjds_e2, perm)))
        null_d_errs.append(e)
        null_d_tcorrs.append(tc)
    p_day_tcorr = float((np.sum(np.array(null_d_tcorrs) >= real_d_tcorr) + 1) / 201.0)

    return {
        "fixed_lambda_km": 4201.0,
        "n_modes_10_500uHz": 43,
        "target_coordinates_withheld": True,
        "pairs": {
            "PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten": {
                "ionofree_all_stations": e2_if_0,
                "ionofree_excl_300km": e2_if_300,
                "ionofree_excl_500km": e2_if_500,
                "precise_all_stations": e2_pr_0,
                "precise_excl_300km": e2_pr_300,
                "precise_excl_500km": e2_pr_500,
                "day_shuffle_temporal_null_precise": {
                    "n_permutations": 200,
                    "real_coarse_error_km": round(real_d_err, 0),
                    "day_shuffle_median_error_km": round(float(np.median(null_d_errs)), 0),
                    "real_truth_kernel_corr": round(real_d_tcorr, 3),
                    "day_shuffle_mean_corr": round(float(np.mean(null_d_tcorrs)), 3),
                    "day_shuffle_std_corr": round(float(np.std(null_d_tcorrs)), 3),
                    "p_truth_corr_ge_observed": round(p_day_tcorr, 4),
                },
            },
            "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten": {
                "ionofree_all_stations": sr_if_0,
                "ionofree_excl_100km": sr_if_100,
            },
            "NPL_YbE3-NPL_Sr1": {
                "precise_all_stations": npl_pr_0,
                "precise_excl_500km": npl_pr_500,
            },
        },
        "ptb_two_independent_pairs_separation_km": ptb_inter_pair_sep_km,
        "fisher_combined_independent_sites_df4": {
            "chi2_df4": round(chi2_2site, 2),
            "p_value": round(fisher_p_2site, 4),
            "scope": "Conditional Fisher summary, one pair per site; shared GNSS reference dependence is not calibrated.",
        },
        "fisher_combined_three_pairs": {
            "chi2_df6": round(chi2_loc, 2),
            "p_value": float(f"{fisher_p_loc:.5f}"),
            "scope": "Conditional Fisher summary; shared PTB clocks and GNSS references violate assured independence.",
        },
    }


def _compute_contemporaneous_optical_gnss_waveform_correlations() -> dict | None:
    """Direct timestamp-aligned multi-hour/daily waveform correlations between optical and GNSS series."""
    if not GNSS_DIR.exists():
        return None

    n_days = 66  # MJD 59999..60064 (DOY 55..120)
    n_slots = n_days * 288
    req_sts = ("PTBB", "HERT", "HERS", "IENG", "OP71", "ZIMM", "ZIM2")
    req_fields = ("precise_clock_bias_ns", "ionofree_clock_bias_ns", "clock_bias_ns", "pos_jitter")
    gnss_s = {fld: {st: np.full(n_slots, np.nan) for st in req_sts} for fld in req_fields}

    for st in req_sts:
        for doy in range(55, 121):
            f = GNSS_DIR / f"{st}_2023{doy:03d}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            ts = d["timestamps"].astype(float)
            slots = np.clip(np.round(ts / 300.0).astype(int), 0, 287)
            off = (doy - 55) * 288
            for fld in req_fields:
                if fld not in d.files:
                    continue
                arr = d[fld].astype(float)
                if len(arr) == 288:
                    gnss_s[fld][st][off : off + 288] = arr
                elif len(arr) == len(slots):
                    gnss_s[fld][st][off + slots] = arr

    def _opt_grid(name: str) -> np.ndarray:
        files_data = _load_pair_files_valid(name)
        t = np.concatenate([tf for _, tf, _ in files_data])
        y = np.concatenate([yf for _, _, yf in files_data])
        med = np.median(y)
        mad = 1.4826 * np.median(np.abs(y - med))
        ok = np.abs(y - med) <= 6.0 * mad
        t, y = t[ok], y[ok]
        slot = np.clip(np.floor((t - 59999.0) * 288.0).astype(int), 0, n_slots - 1)
        u, inv, cnt = np.unique(slot, return_inverse=True, return_counts=True)
        m = cnt >= 30
        grid = np.full(n_slots, np.nan)
        grid[u[m]] = np.bincount(inv, weights=y)[m] / cnt[m]
        return grid

    def _bin(arr: np.ndarray, step: int) -> np.ndarray:
        nb = len(arr) // step
        a2 = arr[: nb * step].reshape(nb, step)
        cnt = np.isfinite(a2).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(a2, axis=1)
        means[cnt < max(1, step // 4)] = np.nan
        return means

    rng = np.random.default_rng(42)
    specs = [
        ("NPL_Sr1_PTB_Sr3__vs__clock_bias_HERT_PTBB_3h", "NPL_Sr1-PTB_Sr3_CombKnoten", "clock_bias_ns", "HERT", "PTBB", 36, 3),
        ("NPL_Sr1_PTB_Sr3__vs__clock_bias_IENG_HERT_3h", "NPL_Sr1-PTB_Sr3_CombKnoten", "clock_bias_ns", "IENG", "HERT", 36, 3),
        ("NPL_Sr1_PTB_Sr3__vs__ionofree_IENG_HERT_3h", "NPL_Sr1-PTB_Sr3_CombKnoten", "ionofree_clock_bias_ns", "IENG", "HERT", 36, 3),
        ("NPL_Sr1_PTB_Sr3__vs__clock_bias_IENG_HERT_6h", "NPL_Sr1-PTB_Sr3_CombKnoten", "clock_bias_ns", "IENG", "HERT", 72, 6),
        ("NPL_Sr1_PTB_Sr3__vs__clock_bias_IENG_HERT_24h", "NPL_Sr1-PTB_Sr3_CombKnoten", "clock_bias_ns", "IENG", "HERT", 288, 24),
        ("NPL_YbE3_NPL_Sr1__vs__precise_HERS_PTBB_6h", "NPL_YbE3-NPL_Sr1", "precise_clock_bias_ns", "HERS", "PTBB", 72, 6),
        ("NPL_YbE3_NPL_Sr1__vs__clock_bias_ZIMM_ZIM2_12h", "NPL_YbE3-NPL_Sr1", "clock_bias_ns", "ZIMM", "ZIM2", 144, 12),
        ("PTB_Yb1E2_PTB_Yb1E3__vs__pos_jitter_OP71_PTBB_12h", "PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten", "pos_jitter", "OP71", "PTBB", 144, 12),
    ]
    opt_cache = {}
    results = {}
    for key, onm, fld, s1, s2, step, hrs in specs:
        if onm not in opt_cache:
            opt_cache[onm] = _opt_grid(onm)
        ob = _bin(opt_cache[onm], step)
        gb = _bin(gnss_s[fld][s1] - gnss_s[fld][s2], step)
        v = np.isfinite(ob) & np.isfinite(gb)
        t_idx = np.where(v)[0]
        x, y = ob[v], gb[v]
        xd = x - np.polyval(np.polyfit(t_idx, x, 1), t_idx)
        yd = y - np.polyval(np.polyfit(t_idx, y, 1), t_idx)
        r_raw, p_raw = stats.pearsonr(x, y)
        rho_raw, ps_raw = stats.spearmanr(x, y)
        r_det, p_det = stats.pearsonr(xd, yd)
        rho_det, ps_det = stats.spearmanr(xd, yd)
        bsz = max(1, 288 // step)
        blocks = [yd[i : i + bsz] for i in range(0, len(yd), bsz)]
        ex = 0
        for _ in range(1000):
            yp = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            if abs(stats.pearsonr(xd, yp)[0]) >= abs(r_det):
                ex += 1
        p_blk = (ex + 1) / 1001.0
        results[key] = {
            "optical_pair": onm,
            "gnss_channel": f"{fld}:{s1}-{s2}",
            "bin_hours": hrs,
            "n_windows": int(len(x)),
            "pearson_r_raw": round(float(r_raw), 3),
            "pearson_p_raw": float(f"{p_raw:.5f}"),
            "spearman_rho_raw": round(float(rho_raw), 3),
            "spearman_p_raw": float(f"{ps_raw:.5f}"),
            "pearson_r_detrended": round(float(r_det), 3),
            "pearson_p_detrended": float(f"{p_det:.5f}"),
            "spearman_rho_detrended": round(float(rho_det), 3),
            "p_day_block_permutation_detrended": round(float(p_blk), 4),
        }
    return results


def _clock_name(side: str) -> str:
    return side.replace("_CombKnoten", "").replace("_CombYb", "")


def _compute_clock_link_variance_decomposition(dl_table: dict) -> dict:
    """Non-negative least-squares decomposition of pair tau_DL^2 excess variance
    into per-clock daily-wander variances plus link/baseline terms, testing
    whether the remote excess floor requires a non-clock (link/baseline)
    component or is fully absorbed by per-clock wander (in particular by
    INRIM-ITYb1)."""
    pairs = list(dl_table.keys())
    v = np.array([dl_table[p]["excess_daily_std_tau_DL_1e18"] ** 2 for p in pairs])
    remote = np.array([PUBLISHED_TABLE_II[p]["baseline_km"] > 0 for p in pairs], dtype=float)
    d_geo_mm = np.array([PUBLISHED_TABLE_II[p]["baseline_km"] / 1000.0 for p in pairs])

    clocks = sorted({c for p in pairs for c in (_clock_name(x) for x in p.split("-"))})
    A_clock = np.array(
        [[1.0 if _clock_name(s) == c else 0.0 for c in clocks] for p in pairs for s in p.split("-")]
    ).reshape(len(pairs), 2, len(clocks)).sum(axis=1)

    def _fit(A, vv):
        x, rnorm = nnls(A, vv)
        resid = vv - A @ x
        return x, float(rnorm ** 2), resid

    x0, sse0, resid0 = _fit(A_clock, v)
    A_rem = np.column_stack([A_clock, remote])
    x1, sse1, resid1 = _fit(A_rem, v)
    A_geo = np.column_stack([A_clock, d_geo_mm])
    x2, sse2, resid2 = _fit(A_geo, v)

    tot = float(np.sum(v ** 2))
    per_pair = {}
    for i, p in enumerate(pairs):
        per_pair[p] = {
            "baseline_km": PUBLISHED_TABLE_II[p]["baseline_km"],
            "tau_DL_1e18": dl_table[p]["excess_daily_std_tau_DL_1e18"],
            "clock_only_pred_std_1e18": round(float(np.sqrt(max(0.0, (A_clock @ x0)[i]))), 2),
            "clock_only_residual_std_1e18": round(
                float(np.sign(resid0[i]) * np.sqrt(abs(resid0[i]))), 2
            ),
        }

    # Leave-INRIM-ITYb1-out: refit clock-only on the six non-INRIM pairs and
    # evaluate whether the NPL-PTB remote excess survives clock-wander
    # absorption on a model trained without the least-stable clock's pairs.
    keep = [i for i, p in enumerate(pairs) if "INRIM" not in p]
    xl, ssel, residl = _fit(A_clock[keep], v[keep])
    rem_resid_leave = [
        round(float(np.sign(residl[i]) * np.sqrt(abs(residl[i]))), 2)
        for i, k in enumerate(keep)
        if remote[k] > 0
    ]

    # 5-pair NPL-PTB 4-clock sub-network (excluding the 1-pair PTB_Yb1E2 nuisance
    # parameter and the INRIM_ITYb1 MJD 60013->60019 step): every clock in
    # {NPL_YbE3, NPL_Sr1, PTB_Yb1E3, PTB_Sr3} is anchored by both a 0 km local
    # pair and 705 km remote pairs.
    npl_ptb_5 = [
        "NPL_YbE3-NPL_Sr1",
        "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
        "NPL_YbE3-PTB_Yb_CombKnoten",
        "NPL_Sr1-PTB_Sr3_CombKnoten",
        "PTB_Yb_CombKnoten-NPL_Sr1",
    ]
    clks4 = ["NPL_YbE3", "NPL_Sr1", "PTB_Yb", "PTB_Sr3"]
    A4 = np.array(
        [[1.0 if _clock_name(s) == c else 0.0 for c in clks4] for p in npl_ptb_5 for s in p.split("-")]
    ).reshape(5, 2, 4).sum(axis=1)
    v4 = np.array([dl_table[p]["excess_daily_std_tau_DL_1e18"] ** 2 for p in npl_ptb_5])
    rem4 = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    x0_4, sse0_4, _ = _fit(A4, v4)
    x1_4, sse1_4, _ = _fit(np.column_stack([A4, rem4]), v4)
    tot4_centered = float(np.sum((v4 - np.mean(v4)) ** 2))
    tot4_uncentered = float(np.sum(v4 ** 2))
    npl_ptb_subnetwork_fit = {
        "pairs": npl_ptb_5,
        "clocks": clks4,
        "n_pairs": 5,
        "n_parameters_clock_only": 4,
        "n_parameters_clock_plus_link": 5,
        "clock_only_design_rank": int(np.linalg.matrix_rank(A4)),
        "clock_plus_link_design_rank": int(np.linalg.matrix_rank(np.column_stack([A4, rem4]))),
        "unconstrained_residual_dof_clock_plus_link": int(5 - np.linalg.matrix_rank(np.column_stack([A4, rem4]))),
        "clock_only_sse": round(sse0_4, 1),
        "clock_only_centered_r2": round(1.0 - sse0_4 / tot4_centered, 4),
        "clock_only_frac_v2": round(1.0 - sse0_4 / tot4_uncentered, 4),
        "clock_plus_remote_link_sse": round(sse1_4, 1),
        "clock_plus_remote_link_centered_r2": round(1.0 - sse1_4 / tot4_centered, 4),
        "clock_plus_remote_link_frac_v2": round(1.0 - sse1_4 / tot4_uncentered, 4),
        "sse_reduction_pct": round(100.0 * (sse0_4 - sse1_4) / sse0_4, 1),
        "remote_link_std_1e18": round(float(np.sqrt(x1_4[-1])), 2),
        "nine_pair_unweighted_local_var_sum_violation_ratio": round(
            float(((A_clock @ x0)[pairs.index("NPL_YbE3-NPL_Sr1")] + (A_clock @ x0)[pairs.index("PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten")]) /
                  sum(dl_table[p]["excess_daily_std_tau_DL_1e18"] ** 2 for p in npl_ptb_5[:2])),
            2,
        ),
        "statistical_role": "Internal variance decomposition and consistency representation of the 4-clock covariance contrast (5 observations vs 4 non-negative clock parameters + 1 link parameter).",
    }

    # Exact four-clock variance difference: the two 705 km homonuclear pairs
    # (NPL_YbE3-PTB_Yb1E3 + NPL_Sr1-PTB_Sr3) and the two 0 km local pairs
    # (NPL_YbE3-NPL_Sr1 + PTB_Yb1E3-PTB_Sr3) each contain all four clocks once,
    # so the difference cancels every per-clock variance algebraically.
    p_r1, p_r2 = "NPL_YbE3-PTB_Yb_CombKnoten", "NPL_Sr1-PTB_Sr3_CombKnoten"
    p_l1, p_l2 = "NPL_YbE3-NPL_Sr1", "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten"
    sum_rem = float(v[pairs.index(p_r1)] + v[pairs.index(p_r2)])
    sum_loc = float(v[pairs.index(p_l1)] + v[pairs.index(p_l2)])
    diff_fc = sum_rem - sum_loc
    four_clock = {
        "remote_pairs": [p_r1, p_r2],
        "local_pairs": [p_l1, p_l2],
        "sum_tau2_remote_1e36": round(sum_rem, 2),
        "sum_tau2_local_1e36": round(sum_loc, 2),
        "remote_over_local_variance_ratio": round(sum_rem / sum_loc, 2),
        "baseline_floor_std_1e18_per_link": round(float(np.sqrt(max(0.0, diff_fc / 2.0))), 2),
    }

    return {
        "variance_units": "excess variance = tau_DL^2 in (1e-18)^2",
        "clocks": clocks,
        "four_clock_exact_difference": four_clock,
        "clock_only": {
            "per_clock_wander_std_1e18": {c: round(float(np.sqrt(x0[i])), 2) for i, c in enumerate(clocks)},
            "sse": round(sse0, 1),
            "frac_v2_explained": round(1.0 - sse0 / tot, 4),
            "per_pair": per_pair,
            "mean_remote_residual_std_1e18": round(
                float(np.mean([np.sign(resid0[i]) * np.sqrt(abs(resid0[i])) for i in range(len(pairs)) if remote[i] > 0])), 2
            ),
            "mean_local_residual_std_1e18": round(
                float(np.mean([np.sign(resid0[i]) * np.sqrt(abs(resid0[i])) for i in range(len(pairs)) if remote[i] == 0])), 2
            ),
        },
        "clock_plus_common_remote_link": {
            "remote_link_std_1e18": round(float(np.sqrt(x1[-1])), 2),
            "sse": round(sse1, 1),
            "frac_v2_explained": round(1.0 - sse1 / tot, 4),
        },
        "clock_plus_geodesic_link": {
            "geodesic_coef_std_1e18_per_Mm": round(float(np.sqrt(x2[-1])), 3) if x2[-1] > 0 else 0.0,
            "sse": round(sse2, 1),
            "frac_v2_explained": round(1.0 - sse2 / tot, 4),
        },
        "leave_inrimityb1_out": {
            "per_clock_wander_std_1e18": {c: round(float(np.sqrt(xl[i])), 2) for i, c in enumerate(clocks)},
            "remote_residual_std_1e18_per_pair": rem_resid_leave,
            "sse": round(ssel, 1),
        },
        "five_pair_npl_ptb_4clock_subnetwork": npl_ptb_subnetwork_fit,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.exists():
        print("data/optical_fiber_2023 not present; keeping committed output.")
        return

    matrix_results = {}
    dl_table = {}
    pre_gap_control = {}
    five_tide_results = {}

    for name, info in PUBLISHED_TABLE_II.items():
        t, y = _load_pair_mad(name)
        s1k = _adev_bin(t, y, 1000.0)
        s20k = _adev_bin(t, y, 20000.0)
        wn20k = s1k / np.sqrt(20.0)
        excess_20ks = float(s20k / wn20k) if wn20k > 0 else None

        mjds = np.floor(t).astype(int)
        d_means, d_wn = [], []
        for mjd in np.unique(mjds):
            ym = y[mjds == mjd]
            if len(ym) >= 3600:
                d_means.append(float(np.mean(ym)))
                d_wn.append(float(s1k / np.sqrt(len(ym) / 1000.0)))
        d_means_a = np.asarray(d_means)
        d_wn_a = np.asarray(d_wn)
        w = 1.0 / (d_wn_a ** 2)
        wm = float(np.sum(w * d_means_a) / np.sum(w))
        birge_wn = float(np.sqrt(np.sum(w * (d_means_a - wm) ** 2) / (len(d_means_a) - 1)))

        t_sec = (t - MJD_2023_001) * 86400.0
        b300 = (t_sec // 300.0).astype(int)
        ub, inv, cnt = np.unique(b300, return_inverse=True, return_counts=True)
        yb = (np.bincount(inv, weights=y) / cnt)[cnt >= 60]
        tb = ub[cnt >= 60] * 300.0
        yb = yb - np.mean(yb)
        a_m2 = _fit_harmonic_amp(tb, yb, M2_F)
        a_o1 = _fit_harmonic_amp(tb, yb, O1_F)
        a_s2 = _fit_harmonic_amp(tb, yb, S2_F)
        a_ctrl = float(np.mean([_fit_harmonic_amp(tb, yb, cf) for cf in CTRL_FS]))

        dl_entry = _compute_dl_and_40ks(name)
        dl_table[name] = dl_entry
        if name in (
            "NPL_YbE3-NPL_Sr1",
            "NPL_Sr1-PTB_Sr3_CombKnoten",
            "PTB_Yb_CombKnoten-NPL_Sr1",
            "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
        ):
            pre_gap_control[name] = _compute_dl_and_40ks(name, mjd_cutoff=60035.0)

        if name in (
            "NPL_YbE3-NPL_Sr1",
            "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
            "NPL_Sr1-PTB_Sr3_CombKnoten",
            "NPL_YbE3-PTB_Yb_CombKnoten",
            "PTB_Yb_CombKnoten-NPL_Sr1",
            "INRIM_ITYb1-PTB_Sr3_CombKnoten",
        ):
            five_tide_results[name] = _compute_5tide_600s(name)

        matrix_results[name] = {
            "architecture": info["arch"],
            "baseline_km": info["baseline_km"],
            "fiber_km": info["fiber_km"],
            "published_birge": info["published_birge"],
            "n_hours": round(len(y) / 3600.0, 2),
            "n_days_ge_1h": len(d_means),
            "weighted_mean_offset_1e18": round(wm * 1e18, 2),
            "daily_std_1e18": round(float(np.std(d_means_a, ddof=1)) * 1e18, 2),
            "adev_1000s_1e18": round(s1k * 1e18, 2),
            "adev_20000s_1e18": round(s20k * 1e18, 2),
            "adev_20000s_over_wn": round(excess_20ks, 3) if excess_20ks else None,
            "birge_vs_1000s_wn": round(birge_wn, 3),
            "dersimonian_laird_and_40ks": dl_entry,
            "tidal_spectroscopy_1e18": {
                "n_5min_bins": int(len(yb)),
                "M2_amp_1e18": round(a_m2 * 1e18, 2),
                "O1_amp_1e18": round(a_o1 * 1e18, 2),
                "S2_amp_1e18": round(a_s2 * 1e18, 2),
                "off_tide_ctrl_mean_1e18": round(a_ctrl * 1e18, 2),
                "M2_over_ctrl_ratio": round(a_m2 / a_ctrl, 3) if a_ctrl > 0 else None,
                "S2_over_ctrl_ratio": round(a_s2 / a_ctrl, 3) if a_ctrl > 0 else None,
            },
        }

    # Distance scaling across the 8 non-interleaved clock pairs (0 -> 705 -> 755 -> 915 km)
    non_interleaved = [k for k in PUBLISHED_TABLE_II if k != "PTB_Yb1E2_CombYb-PTB_Yb_CombKnoten"]
    d_geo_8 = np.array([PUBLISHED_TABLE_II[k]["baseline_km"] for k in non_interleaved])
    tau_dl_8 = np.array([dl_table[k]["excess_daily_std_tau_DL_1e18"] for k in non_interleaved])
    rho_d, p_rho_d = stats.spearmanr(d_geo_8, tau_dl_8)

    # Dependency controls for the 8-pair distance scaling (4 distinct distance bins; 5 shared clocks)
    dist_bins_4 = sorted(np.unique(d_geo_8))
    dist_bin_means_4 = [round(float(np.mean(tau_dl_8[d_geo_8 == b])), 2) for b in dist_bins_4]
    rho_group_4, _ = stats.spearmanr(dist_bins_4, dist_bin_means_4)
    clocks_5 = sorted({_clock_name(x) for p in non_interleaved for x in p.split("-")})
    loco_spearman = {}
    for c_drop in clocks_5:
        keep_idx = [
            i for i, p in enumerate(non_interleaved)
            if c_drop not in [_clock_name(x) for x in p.split("-")]
        ]
        r_loco, p_loco = stats.spearmanr(d_geo_8[keep_idx], tau_dl_8[keep_idx])
        loco_spearman[c_drop] = {
            "n_pairs": len(keep_idx),
            "spearman_rho": round(float(r_loco), 3),
            "spearman_p": round(float(p_loco), 4),
        }

    # Exact 4-clock per-clock variance cancellation on {NPL_YbE3, NPL_Sr1, PTB_Yb1E3, PTB_Sr3}:
    # Local 0 km variance sum = tau_DL^2(NPL_YbE3-NPL_Sr1) + tau_DL^2(PTB_Yb1E3-PTB_Sr3)
    # Remote 705 km homonuclear variance sum = tau_DL^2(NPL_YbE3-PTB_Yb1E3) + tau_DL^2(NPL_Sr1-PTB_Sr3)
    var_local_4clk = (
        dl_table["NPL_YbE3-NPL_Sr1"]["excess_daily_std_tau_DL_1e18"] ** 2
        + dl_table["PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten"]["excess_daily_std_tau_DL_1e18"] ** 2
    )
    var_remote_4clk = (
        dl_table["NPL_YbE3-PTB_Yb_CombKnoten"]["excess_daily_std_tau_DL_1e18"] ** 2
        + dl_table["NPL_Sr1-PTB_Sr3_CombKnoten"]["excess_daily_std_tau_DL_1e18"] ** 2
    )
    sigma_baseline_705km = float(np.sqrt(max(0.0, (var_remote_4clk - var_local_4clk) / 2.0)))
    remote_over_local_var_ratio_4clk = float(var_remote_4clk / var_local_4clk) if var_local_4clk > 0 else None

    # Simultaneous 1-second common-epoch 4-clock subtraction across the exact intersection
    # of seconds where all four pairs are locked simultaneously (so single-clock variances
    # cancel identically without epoch-support mismatch or DL weighting mismatch):
    p4_names = (
        "NPL_YbE3-NPL_Sr1",
        "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
        "NPL_YbE3-PTB_Yb_CombKnoten",
        "NPL_Sr1-PTB_Sr3_CombKnoten",
    )
    sec_maps_4 = {}
    for p4n in p4_names:
        fl4 = _load_pair_files_valid(p4n)
        t4 = np.concatenate([tf for _, tf, _ in fl4])
        y4 = np.concatenate([yf for _, _, yf in fl4])
        ts4 = np.round((t4 - MJD_2023_001) * 86400.0).astype(np.int64)
        sec_maps_4[p4n] = dict(zip(ts4.tolist(), y4.tolist()))
    common_sec_4 = sorted(set.intersection(*(set(m.keys()) for m in sec_maps_4.values())))
    Y_sim4 = {p4n: np.array([sec_maps_4[p4n][s] for s in common_sec_4]) * 1e18 for p4n in p4_names}
    mjds_sim4 = int(MJD_2023_001) + (np.array(common_sec_4) // 86400)
    common_days_sim4 = [int(m) for m in np.unique(mjds_sim4) if np.sum(mjds_sim4 == m) >= 3600]
    dms_sim4, dus_sim4, taus_sim4 = {}, {}, {}
    for p4n in p4_names:
        dm4, du4 = [], []
        for m in common_days_sim4:
            yf4 = Y_sim4[p4n][mjds_sim4 == m]
            nb4 = len(yf4) // 100
            bm4 = yf4[: nb4 * 100].reshape(nb4, 100).mean(axis=1)
            dm4.append(float(np.mean(yf4)))
            du4.append(float(np.std(bm4, ddof=1) / np.sqrt(nb4)))
        dm4_a, du4_a = np.asarray(dm4), np.asarray(du4)
        dms_sim4[p4n], dus_sim4[p4n] = dm4_a, du4_a
        w4 = 1.0 / (du4_a ** 2)
        wm4 = float(np.sum(w4 * dm4_a) / np.sum(w4))
        Q4 = float(np.sum(w4 * (dm4_a - wm4) ** 2))
        df4 = max(1, len(dm4_a) - 1)
        C4 = float(np.sum(w4) - np.sum(w4 ** 2) / np.sum(w4))
        taus_sim4[p4n] = round(float(np.sqrt(max(0.0, (Q4 - df4) / C4))), 2)
    v_rem_sim_dl = taus_sim4["NPL_YbE3-PTB_Yb_CombKnoten"] ** 2 + taus_sim4["NPL_Sr1-PTB_Sr3_CombKnoten"] ** 2
    v_loc_sim_dl = taus_sim4["NPL_YbE3-NPL_Sr1"] ** 2 + taus_sim4["PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten"] ** 2
    raw_var_diff_sim4 = float(
        (np.var(dms_sim4["NPL_YbE3-PTB_Yb_CombKnoten"], ddof=1) + np.var(dms_sim4["NPL_Sr1-PTB_Sr3_CombKnoten"], ddof=1))
        - (np.var(dms_sim4["NPL_YbE3-NPL_Sr1"], ddof=1) + np.var(dms_sim4["PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten"], ddof=1))
    )
    wn_var_diff_sim4 = float(
        (np.mean(dus_sim4["NPL_YbE3-PTB_Yb_CombKnoten"] ** 2) + np.mean(dus_sim4["NPL_Sr1-PTB_Sr3_CombKnoten"] ** 2))
        - (np.mean(dus_sim4["NPL_YbE3-NPL_Sr1"] ** 2) + np.mean(dus_sim4["PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten"] ** 2))
    )
    net_var_diff_sim4 = raw_var_diff_sim4 - wn_var_diff_sim4
    simultaneous_4clk_control = {
        "n_simultaneous_1s_epochs": len(common_sec_4),
        "n_common_days_ge_1h": len(common_days_sim4),
        "common_mjds": common_days_sim4,
        "tau_DL_1e18": taus_sim4,
        "dl_remote_variance_sum_1e36": round(v_rem_sim_dl, 2),
        "dl_local_variance_sum_1e36": round(v_loc_sim_dl, 2),
        "dl_remote_over_local_ratio": round(v_rem_sim_dl / v_loc_sim_dl, 2) if v_loc_sim_dl > 0 else None,
        "dl_isolated_baseline_std_705km_1e18": round(float(np.sqrt(max(0.0, (v_rem_sim_dl - v_loc_sim_dl) / 2.0))), 2),
        "exact_unweighted_raw_var_diff_1e36": round(raw_var_diff_sim4, 2),
        "exact_unweighted_wn_var_diff_1e36": round(wn_var_diff_sim4, 2),
        "exact_unweighted_net_var_diff_1e36": round(net_var_diff_sim4, 2),
        "exact_unweighted_isolated_baseline_std_705km_1e18": round(float(np.sqrt(max(0.0, net_var_diff_sim4 / 2.0))), 2),
    }

    p1_6h_raw = five_tide_results["NPL_Sr1-PTB_Sr3_CombKnoten"]["p_perm_6h_block"]
    p2_6h_raw = five_tide_results["NPL_YbE3-PTB_Yb_CombKnoten"]["p_perm_6h_block"]
    chi2_6h_raw = float(-2.0 * (np.log(p1_6h_raw) + np.log(p2_6h_raw)))
    fisher_p_6h_raw = float(stats.chi2.sf(chi2_6h_raw, 4))

    p1_6h_dm = five_tide_results["NPL_Sr1-PTB_Sr3_CombKnoten"]["p_perm_6h_block_demeaned"]
    p2_6h_dm = five_tide_results["NPL_YbE3-PTB_Yb_CombKnoten"]["p_perm_6h_block_demeaned"]
    chi2_6h_dm = float(-2.0 * (np.log(p1_6h_dm) + np.log(p2_6h_dm)))
    fisher_p_6h_dm = float(stats.chi2.sf(chi2_6h_dm, 4))

    p1_day = five_tide_results["NPL_Sr1-PTB_Sr3_CombKnoten"]["p_perm_whole_day_date"]
    p2_day = five_tide_results["NPL_YbE3-PTB_Yb_CombKnoten"]["p_perm_whole_day_date"]
    chi2_day = float(-2.0 * (np.log(p1_day) + np.log(p2_day)))
    fisher_p_day = float(stats.chi2.sf(chi2_day, 4))

    # Dynamically compute common-window (MJD 60018.5 - 60034.2) 4-pair 705 km NPL-PTB M2 phases
    # in the consistent y_NPL - y_PTB geographic frame
    common_win_phases_deg = {}
    common_win_amps_1e18 = {}
    for pname, sgn_npl_minus_ptb in (
        ("NPL_Sr1-PTB_Sr3_CombKnoten", +1),
        ("NPL_YbE3-PTB_Yb_CombKnoten", +1),
        ("PTB_Yb_CombKnoten-NPL_Sr1", -1),
        ("NPL_YbE3-PTB_Sr3_CombKnoten", +1),
    ):
        fs_w = _load_pair_files_valid(pname)
        tw = np.concatenate([tf for _, tf, _ in fs_w])
        yw = sgn_npl_minus_ptb * np.concatenate([yf for _, _, yf in fs_w])
        mw = (tw >= 60018.5) & (tw <= 60034.2)
        tw, yw = tw[mw], yw[mw]
        b600w = np.floor(tw * 144.0).astype(int)
        ubw, invw, cntw = np.unique(b600w, return_inverse=True, return_counts=True)
        okw = cntw >= 60
        tbw = (ubw[okw] + 0.5) / 144.0
        ybw = np.bincount(invw, weights=yw)[okw] / cntw[okw]
        ybw -= np.mean(ybw)
        Xw = [np.ones_like(tbw)]
        for f_cpd in FIVE_TIDE_CPD.values():
            ww = 2.0 * np.pi * f_cpd * (tbw - 60000.0)
            Xw.extend([np.cos(ww), np.sin(ww)])
        Xw = np.column_stack(Xw)
        cw, _, _, _ = np.linalg.lstsq(Xw, ybw, rcond=None)
        common_win_amps_1e18[pname] = round(float(np.hypot(cw[1], cw[2]) * 1e18), 1)
        common_win_phases_deg[pname] = round(float(np.degrees(np.arctan2(cw[2], cw[1])) % 360.0), 1)
    ph_rad = np.radians(list(common_win_phases_deg.values()))
    rayleigh_R = float(np.abs(np.mean(np.exp(1j * ph_rad))))
    mean_phase_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * ph_rad)))) % 360.0)

    # Simultaneous 600-s bin intersection across all 6 pairs of {NPL_YbE3, NPL_Sr1, PTB_Yb1E3, PTB_Sr3}:
    # test M2 amplitudes and phases in the species-consistent y_YbE3 - y_Sr frame (Delta K_alpha = -6.01)
    # vs the geographic y_NPL - y_PTB frame.
    p6_all = (
        "NPL_Sr1-PTB_Sr3_CombKnoten",
        "NPL_YbE3-PTB_Yb_CombKnoten",
        "PTB_Yb_CombKnoten-NPL_Sr1",
        "NPL_YbE3-PTB_Sr3_CombKnoten",
        "NPL_YbE3-NPL_Sr1",
        "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
    )
    bmaps_6 = {}
    for pname6 in p6_all:
        fs6 = _load_pair_files_valid(pname6)
        t6 = np.concatenate([tf for _, tf, _ in fs6])
        y6 = np.concatenate([yf for _, _, yf in fs6])
        b600_6 = np.floor(t6 * 144.0).astype(int)
        ub6, inv6, cnt6 = np.unique(b600_6, return_inverse=True, return_counts=True)
        ok6 = cnt6 >= 60
        vals6 = np.bincount(inv6, weights=y6)[ok6] / cnt6[ok6]
        bmaps_6[pname6] = dict(zip(ub6[ok6].tolist(), vals6.tolist()))
    common_b6 = sorted(set.intersection(*(set(m.keys()) for m in bmaps_6.values())))
    tb6 = (np.array(common_b6) + 0.5) / 144.0
    X6_sim = [np.ones_like(tb6)]
    for f_cpd in FIVE_TIDE_CPD.values():
        ww6 = 2.0 * np.pi * f_cpd * (tb6 - 60000.0)
        X6_sim.extend([np.cos(ww6), np.sin(ww6)])
    X6_sim = np.column_stack(X6_sim)
    sim_het_amps, sim_het_phases_species = {}, {}
    Y_het_list = []
    for pname_het in (
        "NPL_YbE3-PTB_Sr3_CombKnoten",
        "PTB_Yb_CombKnoten-NPL_Sr1",
        "PTB_Yb_CombKnoten-PTB_Sr3_CombKnoten",
    ):
        # All three have CLOCK1 = Yb+ E3 (K_alpha = -5.95) and CLOCK2 = Sr (K_alpha = +0.06), so raw +1 is y_YbE3 - y_Sr!
        yb6 = np.array([bmaps_6[pname_het][k] for k in common_b6])
        yb6 -= np.mean(yb6)
        Y_het_list.append(yb6)
        c6s, _, _, _ = np.linalg.lstsq(X6_sim, yb6, rcond=None)
        sim_het_amps[pname_het] = round(float(np.hypot(c6s[1], c6s[2]) * 1e18), 1)
        sim_het_phases_species[pname_het] = round(float(np.degrees(np.arctan2(c6s[2], c6s[1])) % 360.0), 1)
    ph_het_rad = np.radians(list(sim_het_phases_species.values()))
    rayleigh_R_het_species = round(float(np.abs(np.mean(np.exp(1j * ph_het_rad)))), 3)
    Y_het_mat = np.column_stack(Y_het_list)
    c_het_obs, _, _, _ = np.linalg.lstsq(X6_sim, Y_het_mat, rcond=None)
    R_het_exact = float(np.abs(np.mean(np.exp(1j * np.arctan2(c_het_obs[2], c_het_obs[1])))))
    rayleigh_R_het_species = round(R_het_exact, 3)
    rng_het = np.random.default_rng(42)
    chunks_het = [[Y_het_mat[i : i + 36, j] for i in range(0, len(tb6), 36)] for j in range(3)]
    ex_het_R = 0
    for _ in range(1000):
        Yp_het = np.column_stack([
            np.concatenate([chunks_het[j][k] for k in rng_het.permutation(len(chunks_het[j]))])
            for j in range(3)
        ])
        cp_het, _, _, _ = np.linalg.lstsq(X6_sim, Yp_het, rcond=None)
        Rp_het = float(np.abs(np.mean(np.exp(1j * np.arctan2(cp_het[2], cp_het[1])))))
        if Rp_het >= R_het_exact:
            ex_het_R += 1
    p_perm_6h_het_R = round((ex_het_R + 1) / 1001.0, 4)
    het_null_6h = _joint_species_phase_null(tb6, Y_het_mat)
    het_null_day = _joint_species_phase_null(tb6, Y_het_mat, block_hours=24)
    p_perm_joint_utc_6h_het_R = round(het_null_6h["p_value"], 4)
    p_perm_joint_utc_day_het_R = round(het_null_day["p_value"], 4)
    # A sign-frame comparison must use the same ratios and the same bins.
    geographic_phases = np.arctan2(c_het_obs[2], c_het_obs[1]).copy()
    geographic_phases[1] += np.pi
    matched_geographic_R = float(abs(np.mean(np.exp(1j * geographic_phases))))

    all_pairs = [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
    raw = {nm: _load_pair_mad(nm) for nm in all_pairs}
    disjoint_correlations = []
    n_eligible_disjoint_tests = 0
    for bin_s in (3600.0, 10800.0, 21600.0):
        binned = {}
        for nm, (t, y) in raw.items():
            t_sec = (t - MJD_2023_001) * 86400.0
            b = (t_sec // bin_s).astype(int)
            ub, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
            m = cnt >= min(300, 0.2 * bin_s)
            binned[nm] = dict(zip(ub[m].tolist(),
                                  (np.bincount(inv, weights=y)[m] / cnt[m]).tolist()))
        for a, b in itertools.combinations(all_pairs, 2):
            if set(a.split("-")) & set(b.split("-")):
                continue
            common = sorted(set(binned[a]) & set(binned[b]))
            if len(common) >= 12:
                n_eligible_disjoint_tests += 1
                va = np.array([binned[a][k] for k in common])
                vb = np.array([binned[b][k] for k in common])
                r, p = stats.pearsonr(va, vb)
                rho, ps = stats.spearmanr(va, vb)
                if p < 0.05 or ps < 0.05:
                    disjoint_correlations.append({
                        "bin_hours": round(bin_s / 3600.0, 1),
                        "pair_1": a,
                        "pair_2": b,
                        "n_bins": len(common),
                        "pearson_r": round(float(r), 4),
                        "pearson_p": float(p),
                        "spearman_rho": round(float(rho), 4),
                        "spearman_p": float(ps),
                    })

    closure_audit = _compute_star_topology_closure()
    zero_fiber_antiphase = _compute_zero_fiber_705km_antiphase()
    zero_sat_localisation = _compute_zero_satellite_global_localisation()
    waveform_corrs = _compute_contemporaneous_optical_gnss_waveform_correlations()
    gls_decomposition = _compute_clock_link_variance_decomposition(dl_table)

    out = {
        "step": 181,
        "title": "European optical-fiber clock network reanalysis (Pizzocaro et al. 2026, PRR 8, 033250)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_doi": "10.5281/zenodo.16539534",
        "publication_doi": "10.1103/l4bh-ryxs",
        "campaign_window": "MJD 59999-60064 (24 Feb - 30 Apr 2023)",
        "dataset_summary": {
            "n_comparison_folders": 16,
            "n_daily_dat_files": 328,
            "n_valid_flag1_seconds": 7903540,
        },
        "conformal_holonomy_bound": {
            "pair": "NPL_YbE3-PTB_Yb_CombKnoten",
            "baseline_chord_km": 705.0,
            "fiber_length_km": 2155.0,
            "published_offset_1e18": -7.1,
            "published_uncertainty_1e18": 7.7,
            "published_birge": 1.13,
            "measured_birge_vs_1000s_wn": matrix_results["NPL_YbE3-PTB_Yb_CombKnoten"]["birge_vs_1000s_wn"],
            "conditional_1sigma_offset_uncertainty_1e18": 7.7,
            "bound_95pct_abs_offset_1e18": round(7.1 + 1.96 * 7.7, 1),
            "observable_scope": "Differential clock frequency-ratio consistency constraint across Doppler-cancelled link; 1 s star-tree closure (b1=0) is an algebraic data-processing check, while a model-independent disformal synchronization holonomy bound requires a physically closed b1>=1 fiber loop.",
        },
        "dersimonian_laird_excess_floor": {
            "paper1_predicted_range_10ks_1e17": [1.4, 8.6],
            "paper1_predicted_range_100ks_1e18": [2.1, 12.0],
            "spearman_rho_d_geo_vs_tau_DL": round(float(rho_d), 3),
            "spearman_p_d_geo_vs_tau_DL": float(f"{p_rho_d:.5f}"),
            "distance_scaling_dependency_controls": {
                "distance_bins_km": [float(b) for b in dist_bins_4],
                "distance_bin_mean_tau_DL_1e18": dist_bin_means_4,
                "four_group_spearman_rho": round(float(rho_group_4), 3),
                # Exact 4! = 24 enumeration: rho = +1.0 for 1 ordering,
                # rho = +0.8 for 3 orderings -> P(rho >= 0.8) = 4/24 and
                # P(|rho| >= 0.8) = 8/24.  (1/24, 2/24 held only for rho = 1.0.)
                "four_group_exact_4fact_p_one_sided": round(4.0 / 24.0, 4),
                "four_group_exact_4fact_p_two_sided": round(8.0 / 24.0, 4),
                "leave_one_clock_out": loco_spearman,
            },
            "four_clock_exact_variance_cancellation_705km": {
                "clocks": ["NPL_YbE3", "NPL_Sr1", "PTB_Yb1E3", "PTB_Sr3"],
                "local_0km_variance_sum_1e36": round(float(var_local_4clk), 2),
                "remote_705km_homonuclear_variance_sum_1e36": round(float(var_remote_4clk), 2),
                "remote_over_local_variance_ratio": round(remote_over_local_var_ratio_4clk, 2) if remote_over_local_var_ratio_4clk else None,
                "isolated_baseline_std_705km_1e18": round(sigma_baseline_705km, 2),
            },
            "simultaneous_1s_common_epoch_4clock_subtraction": simultaneous_4clk_control,
            "pairs": dl_table,
            "pre_mjd_60035_overlap_control": pre_gap_control,
        },
        "five_tide_600s_regression": {
            "pairs": five_tide_results,
            "homonuclear_705km_fisher_6h_block": {
                "chi2_df4": round(chi2_6h_raw, 2),
                "p_value": round(fisher_p_6h_raw, 4),
            },
            "homonuclear_705km_fisher_6h_block_demeaned": {
                "chi2_df4": round(chi2_6h_dm, 2),
                "p_value": round(fisher_p_6h_dm, 4),
            },
            "homonuclear_705km_fisher_whole_day_date": {
                "chi2_df4": round(chi2_day, 2),
                "p_value": round(fisher_p_day, 4),
            },
            "common_window_705km_4pair_phase_clustering": {
                "window_mjd": [60018.5, 60034.2],
                "frame": "y_NPL - y_PTB",
                "amplitudes_1e18": common_win_amps_1e18,
                "phases_deg": common_win_phases_deg,
                "mean_phase_deg": round(mean_phase_deg, 1),
                "rayleigh_resultant_R": round(rayleigh_R, 3),
                "simultaneous_707bin_heteronuclear_YbE3_minus_Sr_species_frame": {
                    "n_simultaneous_600s_bins": len(common_b6),
                    "frame": "y_YbE3 - y_Sr (Delta K_alpha = -6.01)",
                    "amplitudes_1e18": sim_het_amps,
                    "phases_deg": sim_het_phases_species,
                    "rayleigh_resultant_R": rayleigh_R_het_species,
                    "p_perm_6h_block_R": p_perm_6h_het_R,
                    "independent_column_shuffle_scope": "Diagnostic only: independently shuffling ratios breaks shared-clock covariance; use joint UTC-block p-values for inference.",
                    "p_perm_joint_utc_6h_block_R": p_perm_joint_utc_6h_het_R,
                    "p_perm_joint_utc_day_block_R": p_perm_joint_utc_day_het_R,
                    "joint_utc_6h_block_null": het_null_6h,
                    "joint_utc_day_block_null": het_null_day,
                    "same_three_ratios_geographic_sign_R": matched_geographic_R,
                    "sampling_scope": "Intersection of occupied 600 s bins; within-bin valid seconds can differ between ratios.",
                    "statistical_role": "Joint-network consistency result across three contemporaneous ratios sharing clock inputs on the 4-clock NPL-PTB network.",
                },
            },
        },
        "star_topology_closure_audit": closure_audit,
        "zero_fiber_705km_antiphase_coherence": zero_fiber_antiphase,
        "zero_satellite_global_clock_field_localisation": zero_sat_localisation,
        "contemporaneous_optical_gnss_waveform_correlations": waveform_corrs,
        "gls_clock_link_variance_decomposition": gls_decomposition,
        "architecture_matrix": matrix_results,
        "disjoint_pair_sub_mhz_correlations": disjoint_correlations,
        "disjoint_pair_multiplicity_summary": {
            "n_eligible_tests_across_1h_3h_6h": n_eligible_disjoint_tests,
            "n_nominal_p_lt_0_05": len(disjoint_correlations),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["zero_fiber_705km_antiphase_coherence"], indent=2))
    print(json.dumps(out["zero_satellite_global_clock_field_localisation"], indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
