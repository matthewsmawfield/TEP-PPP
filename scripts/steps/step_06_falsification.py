#!/usr/bin/env python3
"""
TEP-PPP Step 06: Falsification / Control Battery
===============================================

The Step 03 headline result is that predicting one European station's detrended
SPP receiver-clock series from its 4 neighbours removes ~51% of the residual
RMSE (honest, out-of-sample, no beta rescaling).

A skeptical reviewer's natural alternative explanation is *common-mode error*:
stations in a ~1000 km cluster observe the same GNSS satellites, so they share
the same broadcast/precise ephemeris (orbit + satellite-clock product) error,
which any spatial or averaging predictor will remove regardless of whether a
TEP field exists.

This script runs the controls needed to distinguish the two hypotheses. If the
TEP kernel and its 4200 km correlation length carry real, geometry-specific
information, then:

  (A) the TEP kernel should beat a plain network-mean predictor;
  (B) the reduction should depend on the correlation length lambda_T;
  (C) the reduction should collapse when the station -> position mapping is
      scrambled (the honest "null vector" test);
  (D) the reduction should collapse when the reference epochs are shuffled in
      time (breaking same-epoch common mode).

The previous version of this test rotated only the CMB component of an ad-hoc
linear-gradient model (leaving the larger orbital-velocity component intact) and
tested a different model than Step 03 -- it could not answer any of (A)-(D).
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from core.ppp_kernel import DynamicTEPKernel  # noqa: E402
from scripts.lib import tep_data as td  # noqa: E402
from scripts.utils.logger import TEPLogger, print_status, set_step_logger  # noqa: E402
from scripts.utils.plotting import setup_publication_style, get_color_palette  # noqa: E402

STATIONS = ["ZIMM", "GRAS", "GENO", "BRUX", "WTZR"]
DOYS = [100, 101, 102, 103, 104, 105, 106]
YEAR = 2024
DATA_DIRS = [PROJECT_ROOT / "data" / "ppp_processed",
             PROJECT_ROOT / "data" / "ppp_longitudinal"]
SPF = 0.44
RNG = np.random.default_rng(20260831)


def _load_day(sta, doy):
    for d in DATA_DIRS:
        f = d / f"{sta}_{YEAR}{doy:03d}.npz"
        if f.exists():
            try:
                z = np.load(f, allow_pickle=False)
            except Exception:
                # Corrupt / truncated download (e.g. 0-byte file) -- skip
                # rather than crashing the whole control battery.
                continue
            # Broadcast SPP control battery: always use the broadcast clock
            # field, never the precise-clock field even if present.
            clk = np.asarray(z["clock_bias_ns"], float)
            pos = np.array([float(z["x_mean"]), float(z["y_mean"]), float(z["z_mean"])]) / 1000.0
            return clk, pos   # raw series (NaNs retained; caller aligns epochs)
    return None, None


def _detrend(v):
    t = np.arange(len(v))
    r = v - np.polyval(np.polyfit(t, v, 1), t)
    # clip gross outliers so a single spike cannot dominate a variance ratio
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    return np.clip(r, -8 * s, 8 * s)


def load_all():
    days = {}
    for doy in DOYS:
        series, pos = {}, {}
        for s in STATIONS:
            clk, p = _load_day(s, doy)
            if clk is None or len(clk) < 100:
                break
            series[s], pos[s] = clk, p
        if len(series) < len(STATIONS):
            continue
        # Truncate to the minimum length so all series are epoch-aligned
        # (same time index = same physical epoch), then keep only epochs
        # where every station is finite.
        n_min = min(len(series[s]) for s in STATIONS)
        truncated = {s: series[s][:n_min] for s in STATIONS}
        m = np.logical_and.reduce([np.isfinite(truncated[s]) for s in STATIONS])
        if int(m.sum()) < 100:
            continue
        days[doy] = {
            "S": np.column_stack([_detrend(truncated[s][m]) for s in STATIONS]),
            "P": np.array([pos[s] for s in STATIONS]),
            "n": int(m.sum()),
        }
    return days


def _reduction(obs, pred):
    base = np.sqrt(np.mean(obs ** 2))
    resid = np.sqrt(np.mean((obs - pred) ** 2))
    if not np.isfinite(resid) or not np.isfinite(base) or base <= 0:
        return float("nan")
    return 100.0 * (1.0 - resid / base)


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


def _iso_weights(known_P, target_p, lam):
    # Use great-circle distance (consistent with tep_kernel.py, step_05b,
    # step_12, step_25).  The previous implementation used Euclidean chord
    # distance in ECEF, which underestimates great-circle distance by up
    # to ~30% at global scale and introduces a methodological inconsistency
    # with the rest of the pipeline.
    known_sts = _ecef_to_stations(known_P)
    target_st = _ecef_to_stations(target_p.reshape(1, -1))
    d_nn = _gc_dist_matrix(known_sts, known_sts)
    K_nn = np.exp(-d_nn / lam) + ((1.0 - SPF) / SPF + 1e-10) * np.eye(len(known_P))
    d_nt = _gc_dist_matrix(known_sts, target_st)[:, 0]
    K_nt = np.exp(-d_nt / lam)
    try:
        with np.errstate(all="ignore"):
            w = np.linalg.solve(K_nn, K_nt)
        if not np.all(np.isfinite(w)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        # Ill-conditioned (large-lambda collapse) -> geometry-free fallback
        w = np.full(len(known_P), 1.0 / (len(known_P) + (1.0 - SPF) / SPF))
    return w


def _uniform_kriging_weights(n):
    """Analytic lambda->inf kriging weights (geometry-free, same nugget)."""
    return np.full(n, 1.0 / (n + (1.0 - SPF) / SPF))


def loo_iso(days, lam, P_override=None):
    per_day = []
    for dd in days.values():
        S, P = dd["S"], (dd["P"] if P_override is None else P_override)
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            w = _iso_weights(P[o], P[t], lam)
            with np.errstate(all="ignore"):
                reds.append(_reduction(S[:, t], S[:, o] @ w))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_network_mean(days):
    per_day = []
    for dd in days.values():
        S = dd["S"]
        per_day.append(np.nanmean([_reduction(S[:, t], np.delete(S, t, axis=1).mean(axis=1))
                                   for t in range(5)]))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_uniform_kriging(days):
    """Geometry-free kriging (lambda=inf, same nugget as iso) -- scaled mean.

    Isolates the nugget-induced rescaling from any geometry-specific gain:
    iso(lam) vs this baseline measures the geometry contribution alone.
    """
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            w = _uniform_kriging_weights(len(o))
            with np.errstate(all="ignore"):
                pred = S[:, o] @ w
            if not np.all(np.isfinite(pred)):
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            reds.append(_reduction(S[:, t], pred))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


def loo_mean_optimal_beta(days):
    """Network mean with per-fold optimal beta from REFERENCES only (no leakage)."""
    per_day = []
    for dd in days.values():
        S = dd["S"]
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            mean_ref = S[:, o].mean(axis=1)
            betas = []
            for r in o:
                r_o = [j for j in o if j != r]
                if not r_o:
                    continue
                m = S[:, r_o].mean(axis=1)
                denom = float(np.dot(m, m))
                if denom > 1e-12:
                    betas.append(float(np.dot(S[:, r], m) / denom))
            beta = float(np.mean(betas)) if betas else 1.0
            if not np.isfinite(beta):
                beta = 1.0
            reds.append(_reduction(S[:, t], beta * mean_ref))
        per_day.append(np.nanmean(reds))
    return float(np.nanmean(per_day)), float(np.nanstd(per_day))


N_SHUFFLE = 200


def loo_time_shuffled(days, lam=4200.0, n_shuffle=N_SHUFFLE):
    """Temporal-shuffle null distribution.

    Each reference station's clock series is circularly shifted by an
    independent random lag per shuffle trial, destroying epoch-to-epoch
    spatial correlation while preserving each station's marginal statistics.
    Returns (mean, std, p2.5, p97.5) over the n_shuffle trials.
    """
    trials = []
    for _ in range(n_shuffle):
        per_day = []
        for dd in days.values():
            S, P, n = dd["S"], dd["P"], dd["n"]
            reds = []
            for t in range(5):
                o = [j for j in range(5) if j != t]
                w = _iso_weights(P[o], P[t], lam)
                # Independent random lag per reference station per trial
                pred = np.zeros(n)
                for ji, j in enumerate(o):
                    lag = int(RNG.integers(n // 20, n - n // 20))
                    pred += w[ji] * np.roll(S[:, j], lag)
                reds.append(_reduction(S[:, t], pred))
            per_day.append(np.mean(reds))
        trials.append(float(np.mean(per_day)))
    trials = np.array(trials)
    return (float(np.mean(trials)), float(np.std(trials)),
            float(np.percentile(trials, 2.5)), float(np.percentile(trials, 97.5)))


def loo_tep_dynamic(days):
    ref = datetime(YEAR, 1, 1)
    per_day = []
    for doy, dd in days.items():
        S, P, n = dd["S"], dd["P"], dd["n"]
        start = ref + timedelta(days=doy - 1)
        reds = []
        for t in range(5):
            o = [j for j in range(5) if j != t]
            # Use reference-station std (not the held-out target) to avoid
            # leaking target information into the kernel parameterisation.
            sig = float(np.mean([np.std(S[:, j]) for j in o]))
            k = DynamicTEPKernel(
                base_lambda_km=4200.0, base_sigma_ns=sig * np.sqrt(SPF),
                enable_rotation_effects=False, enable_velocity_coupling=True,
                enable_local_time=True, velocity_coupling_factor=0.35,
                diurnal_coupling_factor=0.1, seasonal_coupling_factor=-0.2,
                enable_chandler_wobble=True, chandler_coupling_factor=0.15,
                enable_semiannual_nutation=True,
                semiannual_nutation_coupling_factor=0.12,
                semiannual_nutation_phase_offset_days=60.0)
            nv = sig ** 2 * (1 - SPF)
            pred = np.empty(n)
            for i in range(n):
                mu, _, _ = k.predict(known_coords=P[o], known_values=S[i, o],
                                     target_coords=P[t], noise_variance=nv,
                                     epoch_utc=start + timedelta(seconds=i * 30),
                                     reference_epoch=start)
                pred[i] = np.atleast_1d(mu)[0]
            reds.append(_reduction(S[:, t], pred))
        per_day.append(np.mean(reds))
    return float(np.mean(per_day)), float(np.std(per_day))


def main():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    set_step_logger(TEPLogger("step_06", log_file_path=log_dir / "step_06_falsification.log"))

    print_status("STEP 06: FALSIFICATION / CONTROL BATTERY", "TITLE")
    days = load_all()
    if len(days) < 2:
        print_status("Insufficient data", "ERROR")
        return
    print_status(f"Loaded {len(days)} complete days: {sorted(days)}", "INFO")

    tep = loo_tep_dynamic(days)
    netmean = loo_network_mean(days)
    uniform_krig = loo_uniform_kriging(days)
    mean_optbeta = loo_mean_optimal_beta(days)
    lam_sweep = {lam: loo_iso(days, lam) for lam in [300, 1000, 2000, 4200, 10000, 40000, 200000]}

    base_P = days[next(iter(days))]["P"]
    # Exact enumeration of all 44 derangements (no fixed points, no sampling noise)
    from itertools import permutations
    ders = [p for p in permutations(range(5)) if all(p[i] != i for i in range(5))]
    scr = np.array([loo_iso(days, 4200.0, P_override=base_P[list(p)])[0] for p in ders])
    tshuf = loo_time_shuffled(days)

    # Only the physically relevant plateau matters: does the canonical 4200 km
    # value stand out against other long correlation lengths (>= 2000 km)?
    plateau = np.array([lam_sweep[l][0] for l in (2000, 4200, 10000, 40000, 200000)])
    lam_spread = float(plateau.max() - plateau.min())
    scr_lo, scr_hi = float(np.percentile(scr, 2.5)), float(np.percentile(scr, 97.5))

    # Geometry-specific gain: does iso4200 beat the geometry-free scaled mean
    # (uniform kriging, same nugget)?  This is the leakage-free version of test A
    # -- the raw network mean uses scaling=1 and is not a fair comparison.
    iso4200 = lam_sweep[4200][0]
    geometry_gain_pp = float(iso4200 - uniform_krig[0])

    beats_mean = tep[0] > netmean[0] + 1.0
    beats_scaled_mean = iso4200 > uniform_krig[0] + 1.0
    lambda_matters = lam_spread > 3.0
    geometry_matters = not (scr_lo <= tep[0] <= scr_hi)
    time_coherent = tshuf[0] < tep[0] - 10.0
    tep_specific = beats_mean and lambda_matters and geometry_matters and time_coherent

    print_status("", "INFO")
    print_status("=== HONEST OUT-OF-SAMPLE RMSE REDUCTION (%) ===", "TITLE")
    print_status(f"  TEP dynamic kernel           : {tep[0]:6.2f} +/- {tep[1]:.2f}", "INFO")
    print_status(f"  Network mean (raw, scale=1)  : {netmean[0]:6.2f} +/- {netmean[1]:.2f}", "INFO")
    print_status(f"  Uniform kriging (scaled mean): {uniform_krig[0]:6.2f} +/- {uniform_krig[1]:.2f}", "INFO")
    print_status(f"  Mean, optimal beta (refs only): {mean_optbeta[0]:6.2f} +/- {mean_optbeta[1]:.2f}", "INFO")
    for lam, v in lam_sweep.items():
        print_status(f"  iso kriging lambda={lam:>7d} km  : {v[0]:6.2f}", "INFO")
    print_status(f"  Scrambled-geometry null       : {scr.mean():6.2f}  95% CI [{scr_lo:.2f}, {scr_hi:.2f}]", "INFO")
    print_status(f"  Time-shuffled references ({N_SHUFFLE} trials): {tshuf[0]:6.2f} +/- {tshuf[1]:.2f}  "
                 f"95% CI [{tshuf[2]:.2f}, {tshuf[3]:.2f}]", "INFO")
    print_status("", "INFO")
    print_status(f"  (A)  TEP kernel beats raw network mean by >1pp .. {beats_mean}", "INFO")
    print_status(f"  (A') iso4200 beats scaled mean (geometry gain) .. {beats_scaled_mean} "
                 f"(gain {geometry_gain_pp:+.2f}pp)", "INFO")
    print_status(f"  (B)  reduction depends on lambda_T ............. {lambda_matters} (spread {lam_spread:.2f}pp)", "INFO")
    print_status(f"  (C)  reduction needs correct geometry ......... {geometry_matters}", "INFO")
    print_status(f"  (D)  reduction needs same-epoch coherence ..... {time_coherent}", "INFO")
    print_status("", "INFO")
    if tep_specific:
        print_status("VERDICT: reduction carries TEP-specific, geometry-dependent information.", "SUCCESS")
    else:
        print_status("VERDICT: reduction is consistent with generic common-mode removal; "
                     "no TEP-specific spatial information demonstrated.", "WARNING")

    summary = {
        "tep_dynamic_pct": tep[0],
        "network_mean_pct": netmean[0],
        "uniform_kriging_scaled_mean_pct": uniform_krig[0],
        "mean_optimal_beta_refsonly_pct": mean_optbeta[0],
        "iso_4200km_pct": iso4200,
        "geometry_gain_vs_scaled_mean_pp": geometry_gain_pp,
        "iso_lambda_sweep_pct": {str(k): v[0] for k, v in lam_sweep.items()},
        "lambda_sweep_spread_pp": lam_spread,
        "scrambled_geometry_null_pct_mean": float(scr.mean()),
        "scrambled_geometry_null_95ci": [scr_lo, scr_hi],
        "time_shuffled_pct": tshuf[0],
        "time_shuffled_std": tshuf[1],
        "time_shuffled_percentile_range": [tshuf[2], tshuf[3]],
        "time_shuffled_n_trials": N_SHUFFLE,
        "test_A_beats_network_mean": bool(beats_mean),
        "test_Aprime_beats_scaled_mean": bool(beats_scaled_mean),
        "test_B_lambda_dependent": bool(lambda_matters),
        "test_C_geometry_dependent": bool(geometry_matters),
        "test_D_time_coherent": bool(time_coherent),
        "tep_specific_information_demonstrated": bool(tep_specific),
        "falsification_success": bool(tep_specific),
        "n_days": len(days),
        "timestamp": datetime.now().isoformat(),
    }
    out = PROJECT_ROOT / "results" / "outputs" / "step_06_falsification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    (PROJECT_ROOT / "results" / "outputs" / "step_05_falsification_test.json").write_text(
        json.dumps(summary, indent=2))
    print_status(f"Saved summary: {out}", "SUCCESS")

    # --- Plot: lambda sweep + control comparison ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    setup_publication_style()
    colors = get_color_palette()
    c_data, c_mean, c_null, c_time = colors[3], colors[4], colors[1], colors[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Left: lambda sweep
    lams = sorted(lam_sweep.keys())
    vals = [lam_sweep[l][0] for l in lams]
    ax1.semilogx(lams, vals, 'o-', color=c_data, linewidth=1.5, markersize=5)
    ax1.axhline(netmean[0], color=c_mean, linestyle='--', linewidth=1.5, label=f'Network mean ({netmean[0]:.1f}%)')
    ax1.axvline(4200, color='gray', linestyle=':', alpha=0.5, label='$\\lambda_T = 4200$ km')
    ax1.set_xlabel('Correlation length $\\lambda$ (km)')
    ax1.set_ylabel('LOO-CV RMSE reduction (%)')
    ax1.set_title('(B) Correlation length sweep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(vals) * 1.15)

    # Right: control battery bar chart
    labels = ['TEP kernel', 'Network mean', 'Scrambled\ngeometry', 'Time\nshuffled']
    bar_vals = [tep[0], netmean[0], scr.mean(), tshuf[0]]
    colors_bar = [c_data, c_mean, c_null, c_time]
    bars = ax2.bar(labels, bar_vals, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('LOO-CV RMSE reduction (%)')
    ax2.set_title('(A/C/D) Control battery')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=11)

    fig_path = PROJECT_ROOT / "results" / "figures" / "step_06_falsification.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_status(f"Saved figure: {fig_path}", "SUCCESS")


if __name__ == "__main__":
    main()
