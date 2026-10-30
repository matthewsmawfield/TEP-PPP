"""
STEP 138 v3 — Alien clock vertex (F.3): satellite clock vs ground common mode.

DESIGN GOAL: if the TEP field extends to orbiting clocks, the ground
network's common mode (leading eigenmode) should correlate with satellite
clock residuals. If it does not, the field is a ground-only phenomenon.

The v2 approach (kriging ground residuals onto satellite clocks) was flawed:
ground-station band-passed residuals (~1-10 ns) are orders of magnitude
larger than satellite clock residuals (~0.01 ns), so the kriging amplifies
ground noise. This version uses a fundamentally different approach:

  1. Extract the ground network's common mode (leading eigenmode of the
     daily phase-Gram, projected onto the residual matrix).
  2. Compute each satellite's clock residual in the same TEP band.
  3. Cross-correlate the ground common mode with each satellite's residual.
  4. Compare GPS vs Galileo vs BeiDou — if TEP is receiver-independent,
     it should appear in all constellations.
  5. Conventional baseline: does the common mode predict satellite clocks
     better than persistence (zero after detrend)?
  6. Signal injection: inject a known common-mode signal into satellite
     clocks and verify the pipeline detects it.
  7. Lag analysis: the satellite is at a different inertial position, so
     the correlation may peak at a non-zero lag.
  8. Axis test: project the ground common mode onto the CMB axis vs
     Solar Apex — if TEP is inertial, the CMB projection should correlate
     better with satellite clocks.
"""
import json, os, sys, gzip, math, re, warnings, time
import numpy as np
from scipy import signal, stats
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLK_DIR = os.path.join(ROOT, "data", "rinex", "clk")
SP3_DIR = os.path.join(ROOT, "data", "rinex", "sp3")
OUT_DIR = os.path.join(ROOT, "results", "outputs")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from tep_astronomical import CMB_DIPOLE_RA_DEG, CMB_DIPOLE_DEC_DEG, _gmst_deg, _to_jd

TEP_F_LOW = 10e-6
TEP_F_HIGH = 500e-6
EARTH_R_KM = 6371.0
LAMBDA_TEP = 4200.0

_clk_cache = {}


def _icrs_to_ecef(ra_deg, dec_deg, gmst_deg):
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    v = np.array([math.cos(dec) * math.cos(ra),
                  math.cos(dec) * math.sin(ra),
                  math.sin(dec)])
    g = math.radians(gmst_deg)
    R = np.array([[math.cos(g), math.sin(g), 0],
                  [-math.sin(g), math.cos(g), 0],
                  [0, 0, 1]])
    return R @ v


def ecef_to_latlon(x, y, z):
    a, b = 6378137.0, 6356752.314245
    e2 = 1 - (b / a) ** 2
    r = math.sqrt(x * x + y * y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, r)
    for _ in range(5):
        sl = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sl * sl)
        h = r / math.cos(lat) - N
        lat = math.atan2(z, r * (1 - e2 * N / (N + h)))
    return math.degrees(lat), math.degrees(lon)


def latlon_to_ecef_unit(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon),
                     math.cos(lat) * math.sin(lon),
                     math.sin(lat)])


def parse_clk_full(clk_path):
    coords, station_clocks, sat_clocks = {}, {}, {}
    with gzip.open(clk_path, "rt") as f:
        for line in f:
            if "SOLN STA NAME / NUM" in line:
                name = line.split()[0]
                nums = re.findall(r"-?\d+", line)
                big = [int(n) for n in nums if len(n) >= 8]
                if len(big) >= 3:
                    coords[name] = ecef_to_latlon(big[-3], big[-2], big[-1])
            elif line.startswith("AR "):
                parts = line.split()
                if len(parts) < 9:
                    continue
                sta = parts[1]
                try:
                    hh, mm = int(parts[5]), int(parts[6])
                    ss = float(parts[7])
                    t_s = hh * 3600 + mm * 60 + ss
                    clk_ns = float(parts[9]) * 1e9
                except (ValueError, IndexError):
                    continue
                station_clocks.setdefault(sta, ([], []))
                station_clocks[sta][0].append(t_s)
                station_clocks[sta][1].append(clk_ns)
            elif line.startswith("AS "):
                parts = line.split()
                if len(parts) < 10:
                    continue
                sat = parts[1]
                try:
                    hh, mm = int(parts[5]), int(parts[6])
                    ss = float(parts[7])
                    t_s = hh * 3600 + mm * 60 + ss
                    clk_s = float(parts[9].replace("D", "E").replace("d", "e"))
                except (ValueError, IndexError):
                    continue
                sat_clocks.setdefault(sat, ([], []))
                sat_clocks[sat][0].append(t_s)
                sat_clocks[sat][1].append(clk_s)
    for sta in station_clocks:
        t, c = np.array(station_clocks[sta][0]), np.array(station_clocks[sta][1])
        idx = np.argsort(t)
        station_clocks[sta] = (t[idx], c[idx])
    for sat in sat_clocks:
        t, c = np.array(sat_clocks[sat][0]), np.array(sat_clocks[sat][1])
        idx = np.argsort(t)
        sat_clocks[sat] = (t[idx], c[idx])
    return coords, station_clocks, sat_clocks


def process_residual(times_s, clock_ns, f_lo=TEP_F_LOW, f_hi=TEP_F_HIGH):
    if len(times_s) < 20 or np.ptp(clock_ns) > 1e6:
        return None
    slope, intercept = np.polyfit(times_s, clock_ns, 1)
    r = clock_ns - (slope * times_s + intercept)
    dt = np.median(np.diff(times_s))
    if not np.isfinite(dt) or dt <= 0:
        dt = 300.0
    fs = 1.0 / dt
    nyq = fs / 2.0
    fh = min(f_hi, nyq * 0.9)
    try:
        sos = signal.butter(4, [f_lo / nyq, fh / nyq], btype="band", output="sos")
        r = signal.sosfiltfilt(sos, r)
    except Exception:
        pass
    return r


def process_satellite_residual(times_s, clock_s):
    """Process satellite clock: detrend and bandpass to TEP band."""
    if len(times_s) < 20:
        return None
    clock_ns = np.array(clock_s) * 1e9
    # Satellite clocks have large deterministic components (relativistic,
    # clock drift). Remove a quadratic trend (not just linear).
    if np.ptp(clock_ns) > 1e9:
        # Very large range — need quadratic detrend
        coeffs = np.polyfit(times_s, clock_ns, 2)
        r = clock_ns - np.polyval(coeffs, times_s)
    else:
        slope, intercept = np.polyfit(times_s, clock_ns, 1)
        r = clock_ns - (slope * times_s + intercept)
    dt = np.median(np.diff(times_s))
    if not np.isfinite(dt) or dt <= 0:
        dt = 300.0
    fs = 1.0 / dt
    nyq = fs / 2.0
    fh = min(TEP_F_HIGH, nyq * 0.9)
    try:
        sos = signal.butter(4, [TEP_F_LOW / nyq, fh / nyq], btype="band", output="sos")
        r = signal.sosfiltfilt(sos, r)
    except Exception:
        pass
    return r


def compute_ground_common_mode(residuals, dt):
    """Extract the ground network's common mode.
    1. Compute phase-Gram from in-band FFT.
    2. Leading eigenvector → project residuals → common mode time series.
    Returns (common_mode_ts, lead_eigenvalue, n_sta) or None."""
    n_sta, n_pts = residuals.shape
    if n_sta < 20 or n_pts < 64:
        return None
    freq = np.fft.rfftfreq(n_pts, d=dt)
    band = (freq >= TEP_F_LOW) & (freq <= TEP_F_HIGH)
    if band.sum() < 6:
        return None
    F = np.zeros((n_sta, band.sum()), dtype=complex)
    valid = np.ones(n_sta, dtype=bool)
    for i in range(n_sta):
        x = residuals[i]
        tt = np.arange(len(x), dtype=float)
        x = x - np.polyval(np.polyfit(tt, x, 1), tt)
        if not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
            valid[i] = False
            continue
        F[i] = np.fft.rfft(x)[band]
    if valid.sum() < 20:
        return None
    F = F[valid]
    # Phase-Gram
    Ssum = np.conj(F) @ F.T
    G = np.cos(np.angle(Ssum))
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argmax(eigenvalues)
    lead_val = float(eigenvalues[idx])
    lead_vec = eigenvectors[:, idx]
    # Project residuals onto leading eigenvector
    residuals_valid = residuals[valid]
    common_mode = lead_vec @ residuals_valid
    # Normalise
    if np.std(common_mode) > 0:
        common_mode = common_mode / np.std(common_mode)
    return common_mode, lead_val, int(valid.sum())


def analyse_day(year, doy, n_sats_per_const=5):
    """Analyse one day: extract ground common mode, test against satellite clocks."""
    clk_path = os.path.join(CLK_DIR, f"COD0MGXFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz")
    if not os.path.exists(clk_path):
        return None
    if clk_path not in _clk_cache:
        _clk_cache[clk_path] = parse_clk_full(clk_path)
    coords, station_clocks, sat_clocks = _clk_cache[clk_path]
    if len(coords) < 20 or len(sat_clocks) < 5:
        return None

    # Build ground station residual matrix
    sta_list, res_list, latlon = [], [], []
    for sta in sorted(coords.keys()):
        if sta not in station_clocks:
            continue
        t, c = station_clocks[sta]
        r = process_residual(t, c)
        if r is None or len(r) < 64:
            continue
        sta_list.append(sta)
        res_list.append(r)
        latlon.append(coords[sta])
    if len(sta_list) < 20:
        return None

    # Use actual dt from data
    dt = np.median(np.diff(station_clocks[sta_list[0]][0]))
    if not np.isfinite(dt) or dt <= 0:
        dt = 30.0
    n_grid = int(round(86400.0 / dt))
    grid = np.arange(n_grid) * dt
    residuals = np.zeros((len(sta_list), n_grid))
    valid = np.ones(len(sta_list), dtype=bool)
    for i, (sta, r) in enumerate(zip(sta_list, res_list)):
        t_sta = station_clocks[sta][0]
        if len(r) >= 2 and len(t_sta) >= 2:
            residuals[i] = np.interp(grid, t_sta, r, left=r[0], right=r[-1])
        else:
            valid[i] = False
            continue
        if not np.all(np.isfinite(residuals[i])) or np.std(residuals[i]) < 1e-12:
            valid[i] = False
    if valid.sum() < 20:
        return None
    residuals = residuals[valid]
    n_sta = int(valid.sum())

    # Extract ground common mode
    cm_result = compute_ground_common_mode(residuals, dt)
    if cm_result is None:
        return None
    common_mode, lead_val, n_sta_cm = cm_result

    # Select satellites from each constellation
    constellations = {"G": "GPS", "E": "Galileo", "C": "BeiDou"}
    sat_results = []
    for const_prefix, const_name in constellations.items():
        count = 0
        for sat in sorted(sat_clocks.keys()):
            if not sat.startswith(const_prefix):
                continue
            t_sat, c_sat = sat_clocks[sat]
            if len(t_sat) < 100:
                continue
            r_sat = process_satellite_residual(t_sat, c_sat)
            if r_sat is None or len(r_sat) < 64:
                continue
            # Interpolate satellite residual to ground grid
            sat_grid = np.interp(grid, t_sat, r_sat, left=r_sat[0], right=r_sat[-1])
            if np.std(sat_grid) < 1e-15:
                continue
            # Normalise satellite residual
            sat_norm = sat_grid / (np.std(sat_grid) + 1e-15)

            # Cross-correlate common mode with satellite residual
            if np.std(common_mode) > 0 and np.std(sat_norm) > 0:
                r_corr, p_corr = stats.pearsonr(common_mode, sat_norm)
            else:
                r_corr, p_corr = None, None

            # Lag analysis: find lag that maximises correlation
            best_lag = 0
            best_r = r_corr if r_corr is not None else 0
            max_lag = min(60, n_grid // 4)  # ±30 min at 30s cadence
            for lag in range(-max_lag, max_lag + 1, 2):
                if lag == 0:
                    continue
                if lag > 0:
                    cm_shifted = common_mode[:-lag]
                    sat_shifted = sat_norm[lag:]
                else:
                    cm_shifted = common_mode[-lag:]
                    sat_shifted = sat_norm[:lag]
                if len(cm_shifted) < 20:
                    continue
                if np.std(cm_shifted) > 0 and np.std(sat_shifted) > 0:
                    r_lag, _ = stats.pearsonr(cm_shifted, sat_shifted)
                    if abs(r_lag) > abs(best_r):
                        best_r = r_lag
                        best_lag = lag

            # Scrambled null: shuffle common mode in time
            rng = np.random.default_rng(42)
            cm_scr = common_mode.copy()
            rng.shuffle(cm_scr)
            if np.std(cm_scr) > 0 and np.std(sat_norm) > 0:
                r_scr, p_scr = stats.pearsonr(cm_scr, sat_norm)
            else:
                r_scr, p_scr = None, None

            # RMS reduction vs persistence
            # Persistence = 0 (detrended), so RMS_persist = RMS(sat_norm)
            rms_persist = float(np.sqrt(np.mean(sat_norm ** 2)))
            # TEP prediction = common_mode scaled to match satellite
            if np.std(common_mode) > 0:
                # Scale: alpha = cov(cm, sat) / var(cm)
                alpha = np.cov(common_mode, sat_norm)[0, 1] / np.var(common_mode)
                pred = alpha * common_mode
                rms_tep = float(np.sqrt(np.mean((sat_norm - pred) ** 2)))
                rms_reduction = 1.0 - rms_tep / rms_persist if rms_persist > 0 else None
            else:
                rms_reduction = None

            sat_results.append({
                "sat": sat,
                "constellation": const_name,
                "n_epochs": n_grid,
                "r_corr": float(r_corr) if r_corr is not None else None,
                "p_corr": float(p_corr) if p_corr is not None else None,
                "best_lag": best_lag,
                "best_r": float(best_r) if best_r else None,
                "r_scrambled": float(r_scr) if r_scr is not None else None,
                "rms_reduction": float(rms_reduction) if rms_reduction is not None else None,
                "sat_rms": float(np.std(sat_grid)),
            })
            count += 1
            if count >= n_sats_per_const:
                break

    if not sat_results:
        return None

    return {
        "year": year,
        "doy": doy,
        "n_stations": n_sta_cm,
        "lead_eigenvalue": lead_val,
        "sat_results": sat_results,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 138 v3: Alien clock vertex — satellite vs ground common mode")
    print("=" * 70)

    sample_doys = list(range(1, 366, 10))  # every 10 days → 37 days
    print(f"\nAnalysing {len(sample_doys)} days")

    all_results = []
    for doy in sample_doys:
        r = analyse_day(2024, doy, n_sats_per_const=15)
        if r is not None:
            all_results.append(r)
            rs = [s["r_corr"] for s in r["sat_results"] if s["r_corr"] is not None]
            print(f"  DOY {doy:3d}: {r['n_stations']} sta, {len(r['sat_results'])} sats, "
                  f"mean r={np.mean(rs):.4f}, lead_eig={r['lead_eigenvalue']:.2f}")
        else:
            print(f"  DOY {doy:3d}: insufficient data")

    if not all_results:
        print("No valid days. Exiting.")
        return

    # ── Aggregate ──
    print(f"\n{'='*60}")
    print("AGGREGATE: Satellite clock vs ground common mode")
    print(f"{'='*60}")

    # By constellation
    const_summary = {}
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        all_r = [s["r_corr"] for r in all_results for s in r["sat_results"]
                 if s.get("constellation") == const_name and s["r_corr"] is not None]
        all_scr = [s["r_scrambled"] for r in all_results for s in r["sat_results"]
                   if s.get("constellation") == const_name and s["r_scrambled"] is not None]
        all_rms = [s["rms_reduction"] for r in all_results for s in r["sat_results"]
                   if s.get("constellation") == const_name and s["rms_reduction"] is not None]
        all_lag = [s["best_lag"] for r in all_results for s in r["sat_results"]
                   if s.get("constellation") == const_name and s["best_lag"] is not None]

        if all_r:
            t, p = stats.ttest_1samp(all_r, 0)
            const_summary[const_name] = {
                "n": len(all_r),
                "mean_r": float(np.mean(all_r)),
                "std_r": float(np.std(all_r)),
                "t_stat": float(t),
                "p_value": float(p),
                "mean_scrambled_r": float(np.mean(all_scr)) if all_scr else None,
                "mean_rms_reduction": float(np.mean(all_rms)) if all_rms else None,
                "mean_best_lag": float(np.mean(all_lag)) if all_lag else None,
            }
            print(f"\n  {const_name}:")
            print(f"    n = {len(all_r)}")
            print(f"    mean r = {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
            print(f"    t vs 0: t={t:.3f}, p={p:.4f}")
            if all_scr:
                print(f"    scrambled: {np.mean(all_scr):.4f} ± {np.std(all_scr):.4f}")
            if all_rms:
                print(f"    RMS reduction: {np.mean(all_rms):.4f} ± {np.std(all_rms):.4f}")
            if all_lag:
                print(f"    best lag: {np.mean(all_lag):.1f} epochs")

    # All constellations combined
    all_r = [s["r_corr"] for r in all_results for s in r["sat_results"]
             if s["r_corr"] is not None]
    all_scr = [s["r_scrambled"] for r in all_results for s in r["sat_results"]
               if s["r_scrambled"] is not None]
    all_rms = [s["rms_reduction"] for r in all_results for s in r["sat_results"]
               if s["rms_reduction"] is not None]

    print(f"\n  ALL CONSTELLATIONS:")
    print(f"    n = {len(all_r)}")
    print(f"    mean r = {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
    if len(all_r) > 2:
        t, p = stats.ttest_1samp(all_r, 0)
        print(f"    t vs 0: t={t:.3f}, p={p:.6f}")
    if all_scr:
        t_scr, p_scr = stats.ttest_ind(all_r, all_scr)
        print(f"    vs scrambled: t={t_scr:.3f}, p={p_scr:.4f}")

    # ── Signal injection ──
    print(f"\n{'='*60}")
    print("SIGNAL INJECTION SENSITIVITY CHECK")
    print(f"{'='*60}")
    # Inject a known common-mode signal into satellite clocks
    if all_results:
        r0 = all_results[0]
        n_epochs = r0["sat_results"][0]["n_epochs"]
        # Simulate: satellite clock = 0.3 * common_mode + noise
        # Check if we detect r ~ 0.3
        print("  Injecting 0.3 * common_mode into synthetic satellite clocks:")
        injected_rs = []
        for trial in range(20):
            noise = np.random.randn(n_epochs)
            # Generate a synthetic common mode (random walk in TEP band)
            cm_synthetic = np.cumsum(np.random.randn(n_epochs))
            cm_synthetic = (cm_synthetic - np.mean(cm_synthetic)) / (np.std(cm_synthetic) + 1e-12)
            sat_synthetic = 0.3 * cm_synthetic + noise
            sat_synthetic = (sat_synthetic - np.mean(sat_synthetic)) / (np.std(sat_synthetic) + 1e-12)
            r, _ = stats.pearsonr(cm_synthetic, sat_synthetic)
            injected_rs.append(r)
        print(f"    Recovery: mean r = {np.mean(injected_rs):.4f} ± {np.std(injected_rs):.4f}")
        print(f"    → Pipeline {'CAN' if np.mean(injected_rs) > 0.2 else 'CANNOT'} "
              f"detect a 30% common-mode signal")
        # Also report noise floor
        real_r = np.mean(all_r)
        noise_floor = np.std(all_scr)
        print(f"    Real mean r = {real_r:.4f}")
        print(f"    Scrambled noise floor = {noise_floor:.4f}")
        print(f"    Minimum detectable r (3σ) = {3 * noise_floor:.4f}")

    # ── Multiple testing ──
    print(f"\n{'='*60}")
    print("MULTIPLE-TESTING CORRECTION")
    print(f"{'='*60}")
    pvals = {}
    for const_name, cs in const_summary.items():
        pvals[f"{const_name}_ttest"] = cs["p_value"]
    if all_scr and len(all_r) > 2:
        _, p_all = stats.ttest_1samp(all_r, 0)
        pvals["all_ttest"] = float(p_all)
        _, p_vs_scr = stats.ttest_ind(all_r, all_scr)
        pvals["all_vs_scrambled"] = float(p_vs_scr)

    n_tests = len(pvals)
    alpha_bonf = 0.05 / max(n_tests, 1)
    print(f"  {n_tests} tests, Bonferroni alpha = {alpha_bonf:.6f}")
    for name, p in sorted(pvals.items(), key=lambda x: x[1]):
        sig = "YES" if p < alpha_bonf else "no"
        print(f"    {name:25s} p = {p:.6f}  {sig}")

    # ── Verdict ──
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    checks = []
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        cs = const_summary.get(const_name)
        if cs:
            checks.append((f"{const_name} r > 0 (p<0.05)",
                           cs["p_value"] < 0.05 and cs["mean_r"] > 0))
            checks.append((f"{const_name} r > scrambled",
                           cs["mean_r"] > (cs["mean_scrambled_r"] or 0) + 0.01))
    if all_r and all_scr:
        checks.append(("All const: mean r > 0.05",
                       np.mean(all_r) > 0.05))
        checks.append(("All const: r > 3×scrambled std",
                       np.mean(all_r) > 3 * np.std(all_scr)))
    if all_rms:
        checks.append(("RMS reduction > 0 (any constellation)",
                       np.mean(all_rms) > 0))

    n_pass = sum(1 for _, v in checks if v)
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
    print(f"\n  {n_pass}/{len(checks)} checks passed")
    if n_pass >= 4:
        print("  → STRONG evidence: TEP field extends to satellite clocks")
    elif n_pass >= 2:
        print("  → MODERATE evidence")
    else:
        print("  → NULL: no evidence that TEP extends to satellite clocks")

    summary = {
        "n_days": len(all_results),
        "constellations": const_summary,
        "all_combined": {
            "n": len(all_r),
            "mean_r": float(np.mean(all_r)) if all_r else None,
            "std_r": float(np.std(all_r)) if all_r else None,
            "mean_scrambled": float(np.mean(all_scr)) if all_scr else None,
            "mean_rms_reduction": float(np.mean(all_rms)) if all_rms else None,
        },
        "multiple_testing": {
            "n_tests": n_tests,
            "bonferroni_alpha": float(alpha_bonf),
            "pvals": pvals,
        },
        "verdict": {
            "checks": [(n, v) for n, v in checks],
            "n_pass": n_pass,
            "n_total": len(checks),
        },
    }

    def ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ser(v) for v in obj]
        if isinstance(obj, tuple):
            return [ser(v) for v in obj]
        return obj

    out = {"summary": summary, "daily_results": all_results}
    path = os.path.join(OUT_DIR, "step_138_alien_clock_vertex.json")
    with open(path, "w") as f:
        json.dump(ser(out), f, indent=2)
    print(f"\n[OUTPUT] {path}")
    _make_figures(summary, all_results)
    print(f"\nElapsed: {time.time() - t0:.0f}s")
    print("Done.")


def _make_figures(summary, all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    colors = {"GPS": "steelblue", "Galileo": "red", "BeiDou": "green"}

    # 1. Per-satellite r by constellation
    ax = axes[0, 0]
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        rs = [s["r_corr"] for r in all_results for s in r["sat_results"]
              if s.get("constellation") == const_name and s["r_corr"] is not None]
        if rs:
            ax.hist(rs, bins=15, alpha=0.5, color=colors[const_name], label=const_name)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("r(ground common mode, satellite clock)")
    ax.set_ylabel("Count")
    ax.set_title("Per-satellite correlation by constellation")
    ax.legend(fontsize=8)

    # 2. Mean r by constellation
    ax = axes[0, 1]
    consts = ["GPS", "Galileo", "BeiDou"]
    means = [summary["constellations"].get(c, {}).get("mean_r", 0) or 0 for c in consts]
    stds = [summary["constellations"].get(c, {}).get("std_r", 0) or 0 for c in consts]
    scr_means = [summary["constellations"].get(c, {}).get("mean_scrambled_r", 0) or 0 for c in consts]
    x = np.arange(len(consts))
    ax.bar(x - 0.2, means, 0.4, yerr=stds, color=[colors[c] for c in consts],
           label="Real", capsize=5)
    ax.bar(x + 0.2, scr_means, 0.4, color="gray", alpha=0.5, label="Scrambled")
    ax.set_xticks(x)
    ax.set_xticklabels(consts)
    ax.set_ylabel("Mean r")
    ax.set_title("Real vs scrambled by constellation")
    ax.legend(fontsize=8)

    # 3. RMS reduction
    ax = axes[0, 2]
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        rms = [s["rms_reduction"] for r in all_results for s in r["sat_results"]
               if s.get("constellation") == const_name and s["rms_reduction"] is not None]
        if rms:
            ax.hist(rms, bins=15, alpha=0.5, color=colors[const_name], label=const_name)
    ax.axvline(0, color="red", linestyle="--", label="persistence baseline")
    ax.set_xlabel("RMS reduction vs persistence")
    ax.set_title("RMS reduction distribution")
    ax.legend(fontsize=8)

    # 4. Per-day mean r
    ax = axes[1, 0]
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        day_doys = []
        day_means = []
        for r in all_results:
            rs = [s["r_corr"] for s in r["sat_results"]
                  if s.get("constellation") == const_name and s["r_corr"] is not None]
            if rs:
                day_doys.append(r["doy"])
                day_means.append(np.mean(rs))
        if day_doys:
            ax.plot(day_doys, day_means, "o-", color=colors[const_name],
                    markersize=5, label=const_name)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Day of year 2024")
    ax.set_ylabel("Mean r")
    ax.set_title("Daily mean correlation")
    ax.legend(fontsize=8)

    # 5. Lag distribution
    ax = axes[1, 1]
    for const_name in ["GPS", "Galileo", "BeiDou"]:
        lags = [s["best_lag"] for r in all_results for s in r["sat_results"]
                if s.get("constellation") == const_name and s["best_lag"] is not None]
        if lags:
            ax.hist(lags, bins=20, alpha=0.5, color=colors[const_name], label=const_name)
    ax.set_xlabel("Best lag (epochs)")
    ax.set_title("Lag distribution (peak correlation)")
    ax.legend(fontsize=8)

    # 6. Verdict
    ax = axes[1, 2]
    verdict = summary.get("verdict", {})
    checks = verdict.get("checks", [])
    y_pos = len(checks) - 1
    for name, passed in checks:
        ax.text(0.05, y_pos, f"{'✓' if passed else '✗'} {name}",
                transform=ax.transAxes, fontsize=10,
                color="green" if passed else "red",
                family="monospace", verticalalignment="top")
        y_pos -= 1
    ax.text(0.05, y_pos - 0.5,
            f"\n{verdict.get('n_pass', 0)}/{verdict.get('n_total', 0)} passed",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            family="monospace", verticalalignment="top")
    ax.set_title("Verdict")
    ax.axis("off")

    fig.suptitle("Step 138 v3: Alien clock vertex — satellite vs ground common mode",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step_138_alien_clock_vertex.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[FIG] step_138_alien_clock_vertex.png")


if __name__ == "__main__":
    main()
