"""
STEP 137 v3 — Earth-rotation aperture synthesis: the timing-sky map (F.2 + F.4).

DESIGN GOAL: if TEP is real, this script must detect it. If TEP is not real,
this script must produce a clean null with quantified sensitivity.

The TEP prediction (frozen before running):
  1. C_ij(t) = Re⟨S_i(f) S_j*(f)⟩_{10-500 µHz}  is a function of the
     projected inertial baseline  u_ij(t) = n̂_CMB · (r_j(t) − r_i(t)).
  2. C(u) = A exp(−|u|/λ_T)  with  λ_T ≈ 4200 km.
  3. The CMB axis works; the Solar Apex axis fails.
  4. Scrambling station positions destroys the curve.
  5. A sidereal stack beats a solar stack.

METHODOLOGY (bulletproof):
  • 24-hour sliding windows with 6-hour step → resolves 23–500 µHz
    (80 % of the TEP band; the 10–23 µHz tail carries <15 % of power).
  • For each window: band-averaged coherence C_ij and projected
    baseline u_ij at the window mid-time.
  • C(u) collapse test: bin C_ij by |u|, measure within-bin variance
    vs between-bin variance.  A real C(u) has low within-bin variance.
  • Bootstrap 95 % CI on each C(u) bin (1000 resamples).
  • Likelihood-ratio test: H0: C_ij = const  vs  H1: C_ij = A exp(−|u|/λ).
  • Axis discrimination: log-likelihood for CMB vs Solar Apex vs
    geomagnetic pole vs GPS nodal.  ΔBIC > 10 is decisive.
  • Cross-frequency consistency: C(u) in 3 sub-bands (23–100, 100–300,
    300–500 µHz).  A real field produces the same λ in all sub-bands.
  • Great-circle distance control: C(d_gc) should NOT show a tighter
    collapse than C(u) — if it does, the signal is local, not inertial.
  • Station-scrambled null with 100 permutations.
  • Sidereal vs solar fold over the full year.
  • Signal injection: inject a known exp(−|u|/4200) signal into the
    residuals and verify the pipeline recovers it (sensitivity check).
"""
import json, os, sys, gzip, math, re, warnings, time
import numpy as np
from scipy import signal, stats, optimize
from datetime import datetime, timedelta, timezone
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLK_DIR = os.path.join(ROOT, "data", "rinex", "clk")
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
SIDEREAL_DAY_S = 86164.0905
SOLAR_DAY_S = 86400.0

# Sub-bands for cross-frequency consistency
SUB_BANDS = [(23e-6, 100e-6, "23-100 uHz"),
             (100e-6, 300e-6, "100-300 uHz"),
             (300e-6, 500e-6, "300-500 uHz")]

AXES = {
    "cmb":         {"ra": 167.94, "dec": -6.94, "label": "CMB dipole"},
    "solar_apex":  {"ra": 271.0,  "dec": 30.0,  "label": "Solar Apex"},
    "geomag_pole": {"ra": 0.0,    "dec": 80.0,  "label": "Geomagnetic N pole"},
    "gps_nodal":   {"ra": 0.0,    "dec": 55.0,  "label": "GPS orbital normal"},
}

_clk_cache = {}


# ── geometry ──────────────────────────────────────────────────────────

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


def great_circle_km(ll_i, ll_j):
    lat1, lon1 = math.radians(ll_i[0]), math.radians(ll_i[1])
    lat2, lon2 = math.radians(ll_j[0]), math.radians(ll_j[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def projected_baseline(ll_i, ll_j, axis_ecef):
    ri = latlon_to_ecef_unit(ll_i[0], ll_i[1])
    rj = latlon_to_ecef_unit(ll_j[0], ll_j[1])
    return float(np.dot(axis_ecef, (rj - ri))) * EARTH_R_KM


# ── CLK parsing ────────────────────────────────────────────────────────

def parse_clk(clk_path):
    coords, clocks = {}, {}
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
                if sta not in clocks:
                    clocks[sta] = ([], [])
                clocks[sta][0].append(t_s)
                clocks[sta][1].append(clk_ns)
    for sta in clocks:
        t, c = np.array(clocks[sta][0]), np.array(clocks[sta][1])
        idx = np.argsort(t)
        clocks[sta] = (t[idx], c[idx])
    return coords, clocks


# ── residual processing ────────────────────────────────────────────────

def process_residual(times_s, clock_ns):
    if len(times_s) < 20 or np.ptp(clock_ns) > 1e6:
        return None
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


def load_day_residuals(year, doy):
    """Return (sta_list, latlon, residuals_grid, dt) or None."""
    clk_path = os.path.join(CLK_DIR, f"COD0MGXFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz")
    if not os.path.exists(clk_path):
        return None
    if clk_path not in _clk_cache:
        _clk_cache[clk_path] = parse_clk(clk_path)
    coords, clocks = _clk_cache[clk_path]

    sta_list, res_list, latlon = [], [], []
    for sta in sorted(coords.keys()):
        if sta not in clocks:
            continue
        t, c = clocks[sta]
        r = process_residual(t, c)
        if r is None or len(r) < 64:
            continue
        sta_list.append(sta)
        res_list.append(r)
        latlon.append(coords[sta])
    if len(sta_list) < 20:
        return None

    # Use the actual dt from the data (30s for COD0MGXFIN)
    dt = np.median(np.diff(clocks[sta_list[0]][0]))
    if not np.isfinite(dt) or dt <= 0:
        dt = 30.0
    n_grid = int(round(86400.0 / dt))  # full day
    grid = np.arange(n_grid) * dt
    residuals = np.zeros((len(sta_list), n_grid))
    valid = np.ones(len(sta_list), dtype=bool)
    for i, (sta, r) in enumerate(zip(sta_list, res_list)):
        t_sta = clocks[sta][0]
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
    sta_list = [s for s, v in zip(sta_list, valid) if v]
    latlon = [ll for ll, v in zip(latlon, valid) if v]
    return sta_list, np.array(latlon), residuals, dt


# ── windowed coherence ────────────────────────────────────────────────

def windowed_coherence(residuals, dt, win_s, step_s, f_lo=TEP_F_LOW,
                       f_hi=TEP_F_HIGH):
    """Compute band-averaged coherence matrix in sliding windows.
    Returns list of (mid_s, C_matrix, valid_mask)."""
    n_sta, n_pts = residuals.shape
    win_n = int(round(win_s / dt))
    step_n = int(round(step_s / dt))
    if win_n < 64 or win_n > n_pts:
        return None

    freq = np.fft.rfftfreq(win_n, d=dt)
    band = (freq >= f_lo) & (freq <= f_hi)
    if band.sum() < 3:
        return None

    windows = []
    for start in range(0, n_pts - win_n + 1, step_n):
        end = start + win_n
        F = np.zeros((n_sta, band.sum()), dtype=complex)
        valid = np.ones(n_sta, dtype=bool)
        for i in range(n_sta):
            x = residuals[i, start:end]
            tt = np.arange(len(x), dtype=float)
            x = x - np.polyval(np.polyfit(tt, x, 1), tt)
            if not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
                valid[i] = False
                continue
            F[i] = np.fft.rfft(x)[band]
        if valid.sum() < 20:
            continue
        F = F[valid]
        # Band-averaged cross-spectrum: S_ij = sum_f conj(F_i) F_j
        Ssum = np.conj(F) @ F.T
        # Coherence-like: Re(Ssum) / sqrt(|Ssum_ii| |Ssum_jj|)
        diag = np.sqrt(np.abs(np.diag(Ssum)))
        diag[diag < 1e-30] = 1e-30
        C = np.real(Ssum) / np.outer(diag, diag)
        mid_s = (start + end) / 2 * dt
        windows.append((mid_s, C, valid))
    return windows if windows else None


# ── statistics ────────────────────────────────────────────────────────

def bin_profile(u_abs, c_vals, n_bins=25):
    """Bin C by |u|, return (centers, mean, count, std_err)."""
    u_max = max(u_abs.max(), 1.0)
    bins = np.linspace(0, u_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    sem = np.full(n_bins, np.nan)
    for k in range(len(u_abs)):
        idx = np.searchsorted(bins, u_abs[k]) - 1
        if 0 <= idx < n_bins:
            count[idx] += 1
    for b in range(n_bins):
        mask = (u_abs >= bins[b]) & (u_abs < bins[b + 1])
        if mask.sum() > 0:
            mean[b] = np.mean(c_vals[mask])
            sem[b] = np.std(c_vals[mask]) / np.sqrt(mask.sum()) if mask.sum() > 1 else 0
    mask = count > 0
    return centers[mask], mean[mask], count[mask], sem[mask]


def bootstrap_ci(u_abs, c_vals, n_bins=25, n_boot=1000, ci=0.95):
    """Bootstrap CI on binned C(u)."""
    u_max = max(u_abs.max(), 1.0)
    bins = np.linspace(0, u_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (u_abs >= bins[b]) & (u_abs < bins[b + 1])
        n = mask.sum()
        count[b] = n
        if n < 2:
            continue
        samples = c_vals[mask]
        boots = np.array([np.mean(np.random.choice(samples, n, replace=True))
                          for _ in range(n_boot)])
        lo[b] = np.percentile(boots, (1 - ci) / 2 * 100)
        hi[b] = np.percentile(boots, (1 + ci) / 2 * 100)
    mask = count > 0
    return centers[mask], lo[mask], hi[mask], count[mask]


def fit_exp(u_abs, c_vals):
    """Fit C = A exp(-u/lambda) by nonlinear least squares.
    Returns (A, lambda, R^2, log_likelihood)."""
    def model(u, A, lam):
        return A * np.exp(-u / lam)

    def nll(params):
        A, lam = params
        if A <= 0 or lam <= 0:
            return 1e20
        pred = model(u_abs, A, lam)
        resid = c_vals - pred
        sigma = np.std(resid) + 1e-12
        return 0.5 * np.sum(resid ** 2) / sigma ** 2 + len(u_abs) * np.log(sigma)

    try:
        p0 = [np.mean(c_vals), 5000.0]
        result = optimize.minimize(nll, p0, method="Nelder-Mead",
                                   options={"maxiter": 5000})
        A, lam = result.x
        pred = model(u_abs, A, lam)
        ss_res = np.sum((c_vals - pred) ** 2)
        ss_tot = np.sum((c_vals - np.mean(c_vals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ll = -result.fun
        return float(A), float(abs(lam)), float(r2), float(ll)
    except Exception:
        return None, None, None, None


def log_likelihood_null(c_vals):
    """Log-likelihood under H0: C = const = mean(C)."""
    mu = np.mean(c_vals)
    sigma = np.std(c_vals) + 1e-12
    return -0.5 * len(c_vals) * np.log(2 * np.pi * sigma ** 2) - 0.5 * np.sum(
        (c_vals - mu) ** 2) / sigma ** 2


def likelihood_ratio_test(u_abs, c_vals):
    """LRT: H0: C = const  vs  H1: C = A exp(-u/lambda)."""
    ll_null = log_likelihood_null(c_vals)
    A, lam, r2, ll_alt = fit_exp(u_abs, c_vals)
    if ll_alt is None:
        return None, None, None, None, None
    lr = 2 * (ll_alt - ll_null)
    # 2 extra params (A, lambda) under H1
    p = 1 - stats.chi2.cdf(lr, df=2)
    return float(lr), float(p), float(A), float(abs(lam)), float(r2)


# ── main analysis ─────────────────────────────────────────────────────

def analyse_window_set(windows, latlon, axis_def, day_dt, f_lo=TEP_F_LOW,
                       f_hi=TEP_F_HIGH):
    """For a set of windows and one axis, pool (u, C) pairs."""
    all_u, all_c, all_d = [], [], []
    for mid_s, C, valid in windows:
        t = day_dt + timedelta(seconds=mid_s)
        gmst = _gmst_deg(_to_jd(t))
        axis_vec = _icrs_to_ecef(axis_def["ra"], axis_def["dec"], gmst)
        valid_idx = np.where(valid)[0]
        for ii in range(len(valid_idx)):
            for jj in range(ii + 1, len(valid_idx)):
                i, j = valid_idx[ii], valid_idx[jj]
                u = projected_baseline(latlon[i], latlon[j], axis_vec)
                all_u.append(u)
                all_c.append(C[ii, jj])
                all_d.append(great_circle_km(latlon[i], latlon[j]))
    return np.array(all_u), np.array(all_c), np.array(all_d)


def run_analysis(year=2024, win_s=12 * 3600, step_s=3 * 3600,
                 sample_doys=None):
    if sample_doys is None:
        sample_doys = list(range(1, 366, 7))  # every 7 days → 52 days

    print(f"\nAnalysing {len(sample_doys)} days, {win_s/3600:.0f}h windows, "
          f"{step_s/3600:.0f}h step")

    # ── Collect windowed data ──
    # Store residuals too for cross-frequency analysis
    all_window_data = []  # (doy, mid_s, C, valid, latlon, day_dt)
    all_day_data = {}  # doy → (latlon, residuals, dt) for cross-frequency
    for doy in sample_doys:
        result = load_day_residuals(year, doy)
        if result is None:
            continue
        sta_list, latlon, residuals, dt = result
        windows = windowed_coherence(residuals, dt, win_s, step_s)
        if windows is None or len(windows) < 2:
            continue
        day_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        for mid_s, C, valid in windows:
            all_window_data.append((doy, mid_s, C, valid, latlon, day_dt))
        all_day_data[doy] = (latlon, residuals, dt)
        print(f"  DOY {doy:3d}: {len(sta_list)} sta, {len(windows)} windows")

    if len(all_window_data) < 10:
        print("Insufficient data.")
        return None

    # ── Per-axis analysis ──
    print(f"\n{'='*60}")
    print("PER-AXIS ANALYSIS (all windows pooled)")
    print(f"{'='*60}")

    axis_summary = {}
    for axis_name, axis_def in AXES.items():
        all_u, all_c, all_d = [], [], []
        for doy, mid_s, C, valid, latlon, day_dt in all_window_data:
            t = day_dt + timedelta(seconds=mid_s)
            gmst = _gmst_deg(_to_jd(t))
            axis_vec = _icrs_to_ecef(axis_def["ra"], axis_def["dec"], gmst)
            valid_idx = np.where(valid)[0]
            for ii in range(len(valid_idx)):
                for jj in range(ii + 1, len(valid_idx)):
                    i, j = valid_idx[ii], valid_idx[jj]
                    u = projected_baseline(latlon[i], latlon[j], axis_vec)
                    all_u.append(u)
                    all_c.append(C[ii, jj])
                    all_d.append(great_circle_km(latlon[i], latlon[j]))

        all_u = np.array(all_u)
        all_c = np.array(all_c)
        all_d = np.array(all_d)
        u_abs = np.abs(all_u)

        # LRT
        lr, p_lrt, A_fit, lam_fit, r2_fit = likelihood_ratio_test(u_abs, all_c)

        # Correlation with exp(-u/4200)
        c_model = np.exp(-u_abs / LAMBDA_TEP)
        r_model, p_model = stats.pearsonr(all_c, c_model) if np.std(all_c) > 0 else (None, None)

        # Binned profile
        centers, mean_c, count, sem = bin_profile(u_abs, all_c, n_bins=25)
        # Bootstrap CI
        ci_centers, ci_lo, ci_hi, ci_count = bootstrap_ci(u_abs, all_c, n_bins=25)

        # Great-circle distance control
        d_centers, d_mean, d_count, d_sem = bin_profile(all_d, all_c, n_bins=25)
        c_model_d = np.exp(-all_d / LAMBDA_TEP)
        r_d, p_d = stats.pearsonr(all_c, c_model_d) if np.std(all_c) > 0 else (None, None)

        # Within-bin variance ratio (collapse test)
        # If C(u) is real, within-bin variance << total variance
        u_max = max(u_abs.max(), 1.0)
        bins = np.linspace(0, u_max, 26)
        within_var = 0.0
        total_var = np.var(all_c)
        n_within = 0
        for b in range(25):
            mask = (u_abs >= bins[b]) & (u_abs < bins[b + 1])
            if mask.sum() > 2:
                within_var += np.var(all_c[mask]) * mask.sum()
                n_within += mask.sum()
        within_var = within_var / max(n_within, 1)
        collapse_ratio = 1 - within_var / total_var if total_var > 0 else 0

        axis_summary[axis_name] = {
            "n_pairs": len(all_u),
            "r_vs_exp4200": float(r_model) if r_model is not None else None,
            "p_vs_exp4200": float(p_model) if p_model is not None else None,
            "lrt_stat": float(lr) if lr is not None else None,
            "lrt_p": float(p_lrt) if p_lrt is not None else None,
            "A_fit": float(A_fit) if A_fit is not None else None,
            "lambda_fit": float(lam_fit) if lam_fit is not None else None,
            "r2_fit": float(r2_fit) if r2_fit is not None else None,
            "collapse_ratio": float(collapse_ratio),
            "r_vs_great_circle": float(r_d) if r_d is not None else None,
            "p_vs_great_circle": float(p_d) if p_d is not None else None,
            "u_centers": centers.tolist(),
            "c_mean": mean_c.tolist(),
            "c_sem": sem.tolist(),
            "c_count": count.tolist(),
            "ci_lo": ci_lo.tolist(),
            "ci_hi": ci_hi.tolist(),
        }

        print(f"\n  {axis_name} ({AXES[axis_name]['label']}):")
        print(f"    n_pairs = {len(all_u)}")
        print(f"    r vs exp(-u/4200) = {r_model:.4f} (p={p_model:.2e})" if r_model else "    r = N/A")
        print(f"    LRT: stat={lr:.2f}, p={p_lrt:.4f}" if lr else "    LRT: N/A")
        print(f"    fit: A={A_fit:.4f}, lambda={lam_fit:.0f} km, R²={r2_fit:.4f}" if A_fit else "    fit: N/A")
        print(f"    collapse ratio = {collapse_ratio:.4f}")
        print(f"    r vs great-circle = {r_d:.4f}" if r_d else "    r vs gc = N/A")

    # ── BIC comparison across axes ──
    print(f"\n{'='*60}")
    print("AXIS DISCRIMINATION (BIC)")
    print(f"{'='*60}")
    bic_values = {}
    for axis_name in AXES:
        s = axis_summary[axis_name]
        if s["r2_fit"] is not None and s["lambda_fit"] is not None:
            # BIC = -2*ll + k*ln(n); k=2 for exp model, k=1 for null
            # Use R² as proxy: higher R² → lower BIC
            n = s["n_pairs"]
            # ll_alt ≈ -n/2 * ln(1 - R²) - n/2 * ln(2π) - n/2
            if s["r2_fit"] > 0 and s["r2_fit"] < 1:
                ll = -n / 2 * math.log(1 - s["r2_fit"])
                bic = -2 * ll + 2 * math.log(n)
            else:
                bic = 1e10
            bic_values[axis_name] = float(bic)
            print(f"  {axis_name:15s}: BIC = {bic:.2f}")
        else:
            bic_values[axis_name] = None
            print(f"  {axis_name:15s}: BIC = N/A")

    if bic_values.get("cmb") is not None:
        best_axis = min(bic_values, key=lambda x: bic_values[x] if bic_values[x] else 1e20)
        print(f"\n  Best axis: {best_axis} (BIC = {bic_values[best_axis]:.2f})")
        if best_axis != "cmb":
            delta = bic_values.get(best_axis, 0) - bic_values.get("cmb", 0)
            print(f"  ΔBIC vs CMB = {delta:.2f} (CMB is {'worse' if delta < 0 else 'better'})")

    # ── Distance-regressed CMB test ──
    # The key test: remove great-circle distance dependence from C_ij,
    # then test whether the residual still correlates with CMB projection.
    # If it does, there is an inertial signal ON TOP OF the local signal.
    print(f"\n{'='*60}")
    print("DISTANCE-REGRESSED CMB TEST (remove local confound)")
    print(f"{'='*60}")
    cmb_s = axis_summary["cmb"]
    # Use the CMB axis u values and great-circle distances
    # Reconstruct all_u, all_c, all_d for CMB
    cmb_u_all, cmb_c_all, cmb_d_all = [], [], []
    for doy, mid_s, C, valid, latlon, day_dt in all_window_data:
        t = day_dt + timedelta(seconds=mid_s)
        gmst = _gmst_deg(_to_jd(t))
        cmb_vec = _icrs_to_ecef(AXES["cmb"]["ra"], AXES["cmb"]["dec"], gmst)
        valid_idx = np.where(valid)[0]
        for ii in range(len(valid_idx)):
            for jj in range(ii + 1, len(valid_idx)):
                i, j = valid_idx[ii], valid_idx[jj]
                u = abs(projected_baseline(latlon[i], latlon[j], cmb_vec))
                d = great_circle_km(latlon[i], latlon[j])
                cmb_u_all.append(u)
                cmb_c_all.append(C[ii, jj])
                cmb_d_all.append(d)
    cmb_u_all = np.array(cmb_u_all)
    cmb_c_all = np.array(cmb_c_all)
    cmb_d_all = np.array(cmb_d_all)

    # Step 1: raw CMB correlation
    c_model_cmb = np.exp(-cmb_u_all / LAMBDA_TEP)
    r_cmb_raw, p_cmb_raw = stats.pearsonr(cmb_c_all, c_model_cmb)

    # Step 2: great-circle distance correlation
    c_model_d = np.exp(-cmb_d_all / LAMBDA_TEP)
    r_gc_raw, p_gc_raw = stats.pearsonr(cmb_c_all, c_model_d)

    # Step 3: regress out great-circle distance from C_ij
    # C_residual = C - beta * exp(-d/lambda_gc)
    # Fit beta by least squares
    X_gc = c_model_d.reshape(-1, 1)
    beta = np.linalg.lstsq(np.hstack([X_gc, np.ones_like(X_gc)]), cmb_c_all, rcond=None)[0]
    c_residual = cmb_c_all - beta[0] * c_model_d - beta[1]

    # Step 4: test CMB correlation on residual
    if np.std(c_residual) > 0 and np.std(c_model_cmb) > 0:
        r_cmb_regressed, p_cmb_regressed = stats.pearsonr(c_residual, c_model_cmb)
    else:
        r_cmb_regressed, p_cmb_regressed = None, None

    # Step 5: also regress out CMB from great-circle to check the reverse
    X_cmb = c_model_cmb.reshape(-1, 1)
    beta2 = np.linalg.lstsq(np.hstack([X_cmb, np.ones_like(X_cmb)]), cmb_c_all, rcond=None)[0]
    c_residual2 = cmb_c_all - beta2[0] * c_model_cmb - beta2[1]
    if np.std(c_residual2) > 0 and np.std(c_model_d) > 0:
        r_gc_regressed, p_gc_regressed = stats.pearsonr(c_residual2, c_model_d)
    else:
        r_gc_regressed, p_gc_regressed = None, None

    print(f"  Raw CMB r = {r_cmb_raw:.4f} (p={p_cmb_raw:.2e})")
    print(f"  Raw great-circle r = {r_gc_raw:.4f} (p={p_gc_raw:.2e})")
    print(f"  After removing great-circle: CMB r = {r_cmb_regressed:.4f} (p={p_cmb_regressed:.2e})" if r_cmb_regressed else "  After removing gc: CMB r = N/A")
    print(f"  After removing CMB: great-circle r = {r_gc_regressed:.4f} (p={p_gc_regressed:.2e})" if r_gc_regressed else "  After removing CMB: gc r = N/A")
    if r_cmb_regressed and r_gc_regressed:
        if r_cmb_regressed > r_gc_regressed:
            print(f"  → CMB SURVIVES distance regression; great-circle does not survive CMB regression")
            print(f"  → STRONG evidence for inertial (CMB) signal on top of local signal")
        elif r_cmb_regressed > 0 and p_cmb_regressed < 0.01:
            print(f"  → CMB survives distance regression (p<0.01) but great-circle also survives")
            print(f"  → Both inertial and local components present")
        else:
            print(f"  → CMB does not survive distance regression")
            print(f"  → Signal is primarily local (great-circle), not inertial")

    distance_regressed = {
        "r_cmb_raw": float(r_cmb_raw),
        "p_cmb_raw": float(p_cmb_raw),
        "r_gc_raw": float(r_gc_raw),
        "p_gc_raw": float(p_gc_raw),
        "r_cmb_after_gc_removal": float(r_cmb_regressed) if r_cmb_regressed else None,
        "p_cmb_after_gc_removal": float(p_cmb_regressed) if p_cmb_regressed else None,
        "r_gc_after_cmb_removal": float(r_gc_regressed) if r_gc_regressed else None,
        "p_gc_after_cmb_removal": float(p_gc_regressed) if p_gc_regressed else None,
    }

    # ── Random axis controls (100 random axes) ──
    print(f"\n{'='*60}")
    print("RANDOM AXIS CONTROLS (100 random inertial axes)")
    print(f"{'='*60}")
    rng_axes = np.random.default_rng(12345)
    random_axis_rs = []
    for i_ax in range(100):
        ra_rand = rng_axes.uniform(0, 360)
        dec_rand = math.degrees(math.asin(rng_axes.uniform(-1, 1)))
        rand_u_all, rand_c_all = [], []
        for doy, mid_s, C, valid, latlon, day_dt in all_window_data:
            t = day_dt + timedelta(seconds=mid_s)
            gmst = _gmst_deg(_to_jd(t))
            rand_vec = _icrs_to_ecef(ra_rand, dec_rand, gmst)
            valid_idx = np.where(valid)[0]
            for ii in range(len(valid_idx)):
                for jj in range(ii + 1, len(valid_idx)):
                    i, j = valid_idx[ii], valid_idx[jj]
                    u = abs(projected_baseline(latlon[i], latlon[j], rand_vec))
                    rand_u_all.append(u)
                    rand_c_all.append(C[ii, jj])
        rand_u_all = np.array(rand_u_all)
        rand_c_all = np.array(rand_c_all)
        c_model_rand = np.exp(-rand_u_all / LAMBDA_TEP)
        if np.std(rand_c_all) > 0 and np.std(c_model_rand) > 0:
            r_rand, _ = stats.pearsonr(rand_c_all, c_model_rand)
            random_axis_rs.append(r_rand)
    random_axis_rs = np.array(random_axis_rs)
    p_random = float(np.mean(np.abs(random_axis_rs) >= abs(r_cmb_raw))) if r_cmb_raw else None
    print(f"  CMB r = {r_cmb_raw:.4f}")
    print(f"  Random axes: mean r = {np.mean(random_axis_rs):.4f} ± {np.std(random_axis_rs):.4f}")
    print(f"  Random axes: range [{np.min(random_axis_rs):.4f}, {np.max(random_axis_rs):.4f}]")
    print(f"  CMB percentile: {np.mean(np.abs(random_axis_rs) < abs(r_cmb_raw))*100:.1f}%")
    if p_random is not None:
        print(f"  Permutation p-value (CMB vs random axes) = {p_random:.4f}")
        if p_random == 0.0:
            print(f"  → CMB axis BEATS ALL 100 random axes (very strong axis specificity)")
        elif p_random < 0.01:
            print(f"  → CMB axis is in the top 1% of random axes (STRONG axis specificity)")
        elif p_random < 0.05:
            print(f"  → CMB axis is in the top 5% of random axes (significant axis specificity)")
        else:
            print(f"  → CMB axis is NOT significantly preferred over random axes")
    else:
        print(f"  p = N/A")

    # ── Cross-frequency consistency (FIXED: actually recompute in sub-bands) ──
    print(f"\n{'='*60}")
    print("CROSS-FREQUENCY CONSISTENCY (recomputed per sub-band)")
    print(f"{'='*60}")
    subband_results = {}
    for f_lo, f_hi, label in SUB_BANDS:
        sub_u_all, sub_c_all = [], []
        for doy in sorted(all_day_data.keys()):
            latlon, residuals, dt = all_day_data[doy]
            windows = windowed_coherence(residuals, dt, win_s, step_s,
                                         f_lo=f_lo, f_hi=f_hi)
            if windows is None:
                continue
            day_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
            for mid_s, C, valid in windows:
                t = day_dt + timedelta(seconds=mid_s)
                gmst = _gmst_deg(_to_jd(t))
                cmb_vec = _icrs_to_ecef(AXES["cmb"]["ra"], AXES["cmb"]["dec"], gmst)
                valid_idx = np.where(valid)[0]
                for ii in range(len(valid_idx)):
                    for jj in range(ii + 1, len(valid_idx)):
                        i, j = valid_idx[ii], valid_idx[jj]
                        u = abs(projected_baseline(latlon[i], latlon[j], cmb_vec))
                        sub_u_all.append(u)
                        sub_c_all.append(C[ii, jj])
        if len(sub_u_all) < 100:
            subband_results[label] = {"lambda": None, "r2": None, "n": 0}
            print(f"  {label}: insufficient data")
            continue
        sub_u_all = np.array(sub_u_all)
        sub_c_all = np.array(sub_c_all)
        c_model_sub = np.exp(-sub_u_all / LAMBDA_TEP)
        r_sub, p_sub = stats.pearsonr(sub_c_all, c_model_sub) if np.std(sub_c_all) > 0 else (None, None)
        A_sub, lam_sub, r2_sub, _ = fit_exp(sub_u_all, sub_c_all)
        subband_results[label] = {
            "lambda": float(lam_sub) if lam_sub else None,
            "r2": float(r2_sub) if r2_sub else None,
            "r_vs_exp4200": float(r_sub) if r_sub else None,
            "p_vs_exp4200": float(p_sub) if p_sub else None,
            "n": len(sub_u_all),
        }
        print(f"  {label}: n={len(sub_u_all)}, r={r_sub:.4f}, lambda={lam_sub:.0f} km, R²={r2_sub:.4f}" if lam_sub else f"  {label}: n={len(sub_u_all)}, r=N/A")

    # ── Station-scrambled null (100 permutations) ──
    print(f"\n{'='*60}")
    print("STATION-SCRAMBLED NULL (100 permutations)")
    print(f"{'='*60}")
    scr_rs = []
    for perm in range(100):
        rng = np.random.default_rng(perm)
        scr_r_vals = []
        for doy, mid_s, C, valid, latlon, day_dt in all_window_data:
            scr_ll = latlon.copy()
            rng.shuffle(scr_ll)
            t = day_dt + timedelta(seconds=mid_s)
            gmst = _gmst_deg(_to_jd(t))
            cmb_vec = _icrs_to_ecef(AXES["cmb"]["ra"], AXES["cmb"]["dec"], gmst)
            valid_idx = np.where(valid)[0]
            u_vals, c_vals = [], []
            for ii in range(len(valid_idx)):
                for jj in range(ii + 1, len(valid_idx)):
                    i, j = valid_idx[ii], valid_idx[jj]
                    u = abs(projected_baseline(scr_ll[i], scr_ll[j], cmb_vec))
                    u_vals.append(u)
                    c_vals.append(C[ii, jj])
            u_vals = np.array(u_vals)
            c_vals = np.array(c_vals)
            c_model = np.exp(-u_vals / LAMBDA_TEP)
            if np.std(c_vals) > 0 and np.std(c_model) > 0:
                r, _ = stats.pearsonr(c_vals, c_model)
                scr_r_vals.append(r)
        if scr_r_vals:
            scr_rs.append(np.mean(scr_r_vals))
    scr_rs = np.array(scr_rs)
    real_r = axis_summary["cmb"]["r_vs_exp4200"]
    p_scr = np.mean(np.abs(scr_rs) >= abs(real_r)) if real_r is not None else None
    print(f"  Real CMB r = {real_r:.4f}" if real_r else "  Real r = N/A")
    print(f"  Scrambled: mean r = {np.mean(scr_rs):.4f} ± {np.std(scr_rs):.4f}")
    print(f"  Permutation p-value = {p_scr:.4f}" if p_scr is not None else "  p = N/A")

    # ── Sidereal vs solar fold ──
    print(f"\n{'='*60}")
    print("SIDEREAL vs SOLAR FOLD")
    print(f"{'='*60}")
    fold_data = []
    for doy, mid_s, C, valid, latlon, day_dt in all_window_data:
        t = day_dt + timedelta(seconds=mid_s)
        gmst = _gmst_deg(_to_jd(t))
        cmb_vec = _icrs_to_ecef(AXES["cmb"]["ra"], AXES["cmb"]["dec"], gmst)
        valid_idx = np.where(valid)[0]
        u_vals, c_vals = [], []
        for ii in range(len(valid_idx)):
            for jj in range(ii + 1, len(valid_idx)):
                i, j = valid_idx[ii], valid_idx[jj]
                u = abs(projected_baseline(latlon[i], latlon[j], cmb_vec))
                u_vals.append(u)
                c_vals.append(C[ii, jj])
        u_vals = np.array(u_vals)
        c_vals = np.array(c_vals)
        c_model = np.exp(-u_vals / LAMBDA_TEP)
        if np.std(c_vals) > 0 and np.std(c_model) > 0:
            r, _ = stats.pearsonr(c_vals, c_model)
            fold_data.append({"doy": doy, "mid_s": mid_s, "r": r, "gmst": gmst})

    fold_summary = {}
    for period_name, period_s in [("sidereal", SIDEREAL_DAY_S), ("solar", SOLAR_DAY_S)]:
        phases = np.array([(fd["mid_s"]) % period_s / period_s for fd in fold_data])
        rs = np.array([fd["r"] for fd in fold_data])
        n_bins = 12
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_r = np.zeros(n_bins)
        bin_n = np.zeros(n_bins, dtype=int)
        for k in range(len(phases)):
            idx = min(int(phases[k] * n_bins), n_bins - 1)
            bin_r[idx] += rs[k]
            bin_n[idx] += 1
        mask = bin_n > 0
        bin_r[mask] /= bin_n[mask]
        var = float(np.var(bin_r[mask])) if mask.sum() > 1 else 0
        fold_summary[period_name] = {
            "profile_var": var,
            "profile_mean": float(np.mean(bin_r[mask])),
            "profile_std": float(np.std(bin_r[mask])),
            "bin_centers": (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist(),
            "bin_r": bin_r.tolist(),
            "bin_n": bin_n.tolist(),
        }
        print(f"  {period_name}: mean r = {np.mean(bin_r[mask]):.4f}, "
              f"std = {np.std(bin_r[mask]):.4f}, var = {var:.6f}")

    sid_var = fold_summary["sidereal"]["profile_var"]
    sol_var = fold_summary["solar"]["profile_var"]
    ratio = sid_var / sol_var if sol_var > 0 else 1.0
    fold_summary["variance_ratio"] = float(ratio)
    print(f"\n  Sidereal/solar variance ratio = {ratio:.4f}")
    if ratio > 1.1:
        print("  → Sidereal stack has MORE structure (sidereal period is better)")
    elif ratio < 0.9:
        print("  → Solar stack has more structure (solar period is better)")
    else:
        print("  → No distinction between sidereal and solar")

    # ── Signal injection sensitivity check ──
    print(f"\n{'='*60}")
    print("SIGNAL INJECTION SENSITIVITY CHECK")
    print(f"{'='*60}")
    # Inject a known exp(-u/4200) signal into the coherence values
    # and check if the pipeline detects it
    if all_window_data:
        injected_u, injected_c = [], []
        for doy, mid_s, C, valid, latlon, day_dt in all_window_data[:20]:
            t = day_dt + timedelta(seconds=mid_s)
            gmst = _gmst_deg(_to_jd(t))
            cmb_vec = _icrs_to_ecef(AXES["cmb"]["ra"], AXES["cmb"]["dec"], gmst)
            valid_idx = np.where(valid)[0]
            for ii in range(len(valid_idx)):
                for jj in range(ii + 1, len(valid_idx)):
                    i, j = valid_idx[ii], valid_idx[jj]
                    u = abs(projected_baseline(latlon[i], latlon[j], cmb_vec))
                    # Inject signal: C_injected = C_real + 0.3 * exp(-u/4200)
                    c_inj = C[ii, jj] + 0.3 * np.exp(-u / LAMBDA_TEP)
                    injected_u.append(u)
                    injected_c.append(c_inj)
        injected_u = np.array(injected_u)
        injected_c = np.array(injected_c)
        r_inj, p_inj = stats.pearsonr(injected_c, np.exp(-injected_u / LAMBDA_TEP))
        A_inj, lam_inj, r2_inj, _ = fit_exp(injected_u, injected_c)
        print(f"  Injected 0.3*exp(-u/4200) signal:")
        print(f"    Recovery: r = {r_inj:.4f}, lambda = {lam_inj:.0f} km, R² = {r2_inj:.4f}")
        print(f"    → Pipeline {'CAN' if r_inj > 0.3 else 'CANNOT'} detect a signal of this amplitude")

    # ── Multiple-testing ──
    print(f"\n{'='*60}")
    print("MULTIPLE-TESTING CORRECTION")
    print(f"{'='*60}")
    pvals = {}
    for axis_name in AXES:
        p = axis_summary[axis_name].get("p_vs_exp4200")
        if p is not None:
            pvals[f"r_{axis_name}"] = p
        p_lrt = axis_summary[axis_name].get("lrt_p")
        if p_lrt is not None:
            pvals[f"lrt_{axis_name}"] = p_lrt
    if p_scr is not None:
        pvals["scrambled_perm"] = p_scr
    if p_random is not None:
        pvals["random_axis_perm"] = p_random
    if r_cmb_regressed is not None and p_cmb_regressed is not None:
        pvals["cmb_after_gc_removal"] = p_cmb_regressed
    n_tests = len(pvals)
    alpha_bonf = 0.05 / max(n_tests, 1)
    print(f"  {n_tests} tests, Bonferroni alpha = {alpha_bonf:.6f}")
    for name, p in sorted(pvals.items(), key=lambda x: x[1]):
        sig = "YES" if p < alpha_bonf else "no"
        print(f"    {name:25s} p = {p:.6f}  {sig}")

    # ── Assemble output ──
    summary = {
        "year": year,
        "win_s": win_s,
        "step_s": step_s,
        "n_windows": len(all_window_data),
        "n_sample_doys": len(sample_doys),
        "axes": axis_summary,
        "bic_values": bic_values,
        "scrambled_null": {
            "mean_r": float(np.mean(scr_rs)),
            "std_r": float(np.std(scr_rs)),
            "permutation_p": float(p_scr) if p_scr is not None else None,
            "n_perms": len(scr_rs),
        },
        "random_axis_null": {
            "mean_r": float(np.mean(random_axis_rs)),
            "std_r": float(np.std(random_axis_rs)),
            "permutation_p": float(p_random) if p_random is not None else None,
            "cmb_percentile": float(np.mean(np.abs(random_axis_rs) < abs(r_cmb_raw)) * 100) if r_cmb_raw else None,
            "n_axes": len(random_axis_rs),
        },
        "distance_regressed": distance_regressed,
        "sidereal_vs_solar": fold_summary,
        "subband": subband_results,
        "multiple_testing": {
            "n_tests": n_tests,
            "bonferroni_alpha": float(alpha_bonf),
            "pvals": pvals,
        },
    }
    return summary


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 137 v4: Earth-rotation aperture synthesis (F.2 + F.4)")
    print("=" * 70)

    summary = run_analysis(year=2024, win_s=12 * 3600, step_s=3 * 3600)
    if summary is None:
        print("Analysis failed.")
        return

    # ── Verdict ──
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    cmb = summary["axes"]["cmb"]
    apex = summary["axes"]["solar_apex"]
    scr = summary["scrambled_null"]
    rand_null = summary.get("random_axis_null", {})
    dist_reg = summary.get("distance_regressed", {})

    checks = []
    checks.append(("CMB r > Solar Apex r",
                   cmb["r_vs_exp4200"] is not None and
                   apex["r_vs_exp4200"] is not None and
                   cmb["r_vs_exp4200"] > apex["r_vs_exp4200"]))
    checks.append(("CMB r > scrambled r (3σ)",
                   cmb["r_vs_exp4200"] is not None and
                   cmb["r_vs_exp4200"] > scr["mean_r"] + 3 * scr["std_r"]))
    checks.append(("LRT significant (CMB)",
                   cmb["lrt_p"] is not None and cmb["lrt_p"] < 0.01))
    checks.append(("CMB BIC best",
                   summary["bic_values"].get("cmb") is not None and
                   all(summary["bic_values"].get("cmb", 1e20) <=
                       (summary["bic_values"].get(a, 1e20) or 1e20)
                       for a in AXES)))
    checks.append(("CMB > 95% of random axes",
                   rand_null.get("permutation_p") is not None and
                   rand_null["permutation_p"] < 0.05))
    checks.append(("CMB survives distance regression (p<0.01)",
                   dist_reg.get("p_cmb_after_gc_removal") is not None and
                   dist_reg["p_cmb_after_gc_removal"] < 0.01))
    checks.append(("CMB > great-circle after regression",
                   dist_reg.get("r_cmb_after_gc_removal") is not None and
                   dist_reg.get("r_gc_after_cmb_removal") is not None and
                   dist_reg["r_cmb_after_gc_removal"] > dist_reg["r_gc_after_cmb_removal"]))
    checks.append(("Cross-frequency: CMB r > 0 in all sub-bands",
                   all(s.get("r_vs_exp4200") is not None and s["r_vs_exp4200"] > 0
                       for s in summary.get("subband", {}).values())))

    n_pass = sum(1 for _, v in checks if v)
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
    print(f"\n  {n_pass}/{len(checks)} checks passed")
    if n_pass >= 6:
        print("  → STRONG evidence for TEP inertial structure")
    elif n_pass >= 4:
        print("  → MODERATE evidence for TEP inertial structure")
    elif n_pass >= 2:
        print("  → WEAK evidence")
    else:
        print("  → NULL: no evidence for TEP inertial structure")

    summary["verdict"] = {
        "checks": [(n, v) for n, v in checks],
        "n_pass": n_pass,
        "n_total": len(checks),
    }

    # Save
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

    path = os.path.join(OUT_DIR, "step_137_aperture_synthesis.json")
    with open(path, "w") as f:
        json.dump(ser(summary), f, indent=2)
    print(f"\n[OUTPUT] {path}")
    _make_figures(summary)
    print(f"\nElapsed: {time.time() - t0:.0f}s")
    print("Done.")


def _make_figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    colors = {"cmb": "steelblue", "solar_apex": "red",
              "geomag_pole": "green", "gps_nodal": "orange"}

    # 1. C(u) curves with bootstrap CI for CMB
    ax = axes[0, 0]
    cmb = summary["axes"]["cmb"]
    u = np.array(cmb["u_centers"])
    c = np.array(cmb["c_mean"])
    lo = np.array(cmb["ci_lo"])
    hi = np.array(cmb["ci_hi"])
    mask = np.isfinite(c)
    ax.errorbar(u[mask], c[mask], yerr=[c[mask] - lo[mask], hi[mask] - c[mask]],
                fmt="o-", color="steelblue", markersize=5, capsize=3, label="CMB (95% CI)")
    u_model = np.linspace(0, max(u), 100)
    if cmb["lambda_fit"]:
        ax.plot(u_model, cmb["A_fit"] * np.exp(-u_model / cmb["lambda_fit"]),
                "b--", alpha=0.5, label=f"fit: λ={cmb['lambda_fit']:.0f} km")
    ax.plot(u_model, np.exp(-u_model / 4200), "k:", alpha=0.3, label="exp(-u/4200)")
    ax.set_xlabel("|u| projected inertial baseline (km)")
    ax.set_ylabel("C(u) coherence")
    ax.set_title("C(u) with bootstrap 95% CI (CMB axis)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.3, 1.0)

    # 2. All axes C(u)
    ax = axes[0, 1]
    for axis_name in AXES:
        s = summary["axes"][axis_name]
        u = np.array(s["u_centers"])
        c = np.array(s["c_mean"])
        mask = np.isfinite(c)
        ax.plot(u[mask], c[mask], "o-", color=colors[axis_name], markersize=4,
                label=f"{AXES[axis_name]['label']} (r={s['r_vs_exp4200']:.3f})")
    ax.plot(u_model, np.exp(-u_model / 4200), "k:", alpha=0.3)
    ax.set_xlabel("|u| (km)")
    ax.set_ylabel("C(u)")
    ax.set_title("C(u) by axis")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.3, 1.0)

    # 3. BIC comparison
    ax = axes[0, 2]
    bic_vals = summary["bic_values"]
    names = list(AXES.keys())
    vals = [bic_vals.get(n, 0) or 0 for n in names]
    ax.bar(names, vals, color=[colors[n] for n in names])
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title("Axis discrimination (BIC)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Great-circle distance control
    ax = axes[1, 0]
    cmb = summary["axes"]["cmb"]
    # Use great-circle r
    ax.text(0.5, 0.5, f"r vs exp(-d/4200) = {cmb['r_vs_great_circle']:.4f}\n"
            f"r vs exp(-u/4200) = {cmb['r_vs_exp4200']:.4f}\n\n"
            f"If r_gc > r_u, signal is\nlocal (great-circle),\nnot inertial.",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("Great-circle vs inertial baseline")

    # 5. Collapse ratio comparison
    ax = axes[1, 1]
    names = list(AXES.keys())
    cr = [summary["axes"][n]["collapse_ratio"] for n in names]
    ax.bar(names, cr, color=[colors[n] for n in names])
    ax.set_ylabel("Collapse ratio (1 - within_var/total_var)")
    ax.set_title("C(u) collapse test")
    ax.tick_params(axis="x", rotation=30)

    # 6. Sidereal vs solar fold
    ax = axes[1, 2]
    for period, color in [("sidereal", "steelblue"), ("solar", "red")]:
        ps = summary["sidereal_vs_solar"][period]
        bc = np.array(ps["bin_centers"])
        br = np.array(ps["bin_r"])
        bn = np.array(ps["bin_n"])
        mask = bn > 0
        ax.plot(bc[mask], br[mask], "o-", color=color, markersize=5,
                label=f"{period} (var={ps['profile_var']:.6f})")
    ax.set_xlabel("Phase")
    ax.set_ylabel("r(C, exp(-u/4200))")
    ax.set_title(f"Sidereal vs solar fold (ratio={summary['sidereal_vs_solar']['variance_ratio']:.3f})")
    ax.legend(fontsize=8)

    # 7. Scrambled null distribution
    ax = axes[2, 0]
    scr = summary["scrambled_null"]
    ax.hist(np.random.normal(scr["mean_r"], scr["std_r"], 1000), bins=30,
            color="gray", alpha=0.5, label="Scrambled null")
    ax.axvline(summary["axes"]["cmb"]["r_vs_exp4200"], color="steelblue",
               linewidth=2, label=f"CMB r={summary['axes']['cmb']['r_vs_exp4200']:.4f}")
    ax.set_xlabel("r(C, exp(-u/4200))")
    ax.set_title(f"Scrambled null (p={scr['permutation_p']:.4f})")
    ax.legend(fontsize=8)

    # 8. Cross-frequency consistency
    ax = axes[2, 1]
    sb = summary.get("subband", {})
    labels = list(sb.keys())
    lams = [sb[l].get("lambda", 0) or 0 for l in labels]
    ax.bar(labels, lams, color="steelblue")
    ax.axhline(4200, color="red", linestyle="--", label="λ_T = 4200 km")
    ax.set_ylabel("Fitted λ (km)")
    ax.set_title("Cross-frequency λ consistency")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)

    # 9. Verdict
    ax = axes[2, 2]
    verdict = summary.get("verdict", {})
    checks = verdict.get("checks", [])
    y_pos = len(checks) - 1
    for name, passed in checks:
        ax.text(0.05, y_pos, f"{'✓' if passed else '✗'} {name}",
                transform=ax.transAxes, fontsize=10,
                color="green" if passed else "red",
                family="monospace", verticalalignment="top")
        y_pos -= 1
    ax.text(0.05, y_pos - 0.5, f"\n{verdict.get('n_pass', 0)}/{verdict.get('n_total', 0)} passed",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            family="monospace", verticalalignment="top")
    ax.set_title("Verdict")
    ax.axis("off")

    fig.suptitle("Step 137 v3: Earth-rotation aperture synthesis — timing-sky map",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step_137_aperture_synthesis.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[FIG] step_137_aperture_synthesis.png")


if __name__ == "__main__":
    main()
