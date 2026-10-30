#!/usr/bin/env python3
"""
STEP 165b  --  Direct gravitational phase test using harmonic fit phases.

The definitive TEP test: do the OBSERVED M2 clock phases, extracted by
simultaneous harmonic fitting, match the gravitational prediction
cos(2*(lon1-lon2)) for station pairs?

This is independent of:
  - ephemeris accuracy (only uses longitude differences)
  - bandpass noise (uses clean harmonic fit phases)
  - processing pipeline (tested on COD, GFZ, 2024, 2025)

TEST:
  1. Harmonic fit M2 (simultaneous with O1, S2, S1, K1, P1) at each station.
  2. For each station pair, compute:
       - observed phase difference = phase1 - phase2 (circular)
       - expected phase difference = 2*(lon1 - lon2) (gravitational)
  3. Correlate cos(observed_dphase) with cos(expected_dphase).
  4. Repeat for S2, O1, and control frequencies.

The gravitational prediction for M2 is:
  dphi = 2*(lon1 - lon2) mod 360

This is purely longitude-dependent. No atmospheric or processing effect
can produce this specific pattern.

Output: results/outputs/step_165b_gravitational_phase_direct.json
"""
from __future__ import annotations
import sys, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_165b_gravitational_phase_direct.json"

DT = 300.0

# Tidal frequencies (Hz)
TIDAL_FREQS = {
    "M2":  1.0 / (12.42 * 3600.0),   # lunar semidiurnal (purely gravitational)
    "O1":  1.0 / (25.82 * 3600.0),   # lunar diurnal
    "S2":  2.0 / 86400.0,             # solar semidiurnal
    "S1":  1.0 / 86400.0,             # solar diurnal
    "K1":  1.0 / (23.93 * 3600.0),   # luni-solar diurnal
    "P1":  1.0 / (24.07 * 3600.0),   # solar diurnal
    "N2":  1.0 / (12.66 * 3600.0),   # lunar semidiurnal
    "Q1":  1.0 / (26.87 * 3600.0),   # lunar diurnal
}

# Control frequencies (Hz) — not tidal
CONTROL_FREQS_HZ = [13e-6, 15e-6, 17e-6, 19e-6, 25e-6, 27e-6, 29e-6, 31e-6]


def harmonic_fit(resid, t_sec, freqs_hz):
    """Simultaneous harmonic fit. Returns phases in degrees and amplitudes."""
    t0 = t_sec[0]
    tc = t_sec - t0 - (t_sec[-1] - t_sec[0]) / 2.0
    ts = (tc[-1] - tc[0]) / 2.0 if len(tc) > 1 else 1.0
    tn = tc / ts
    cols = [np.ones_like(tn), tn, tn ** 2]
    for name, f in freqs_hz.items():
        omega = 2.0 * np.pi * f * (t_sec - t0)
        cols.append(np.cos(omega))
        cols.append(np.sin(omega))
    design = np.column_stack(cols)
    coef, _, _, _ = lstsq(design, resid, cond=1e-12)
    resid_fit = resid - design @ coef
    noise_std = float(np.std(resid_fit))
    n = len(resid)
    results = {}
    for i, (name, f) in enumerate(freqs_hz.items()):
        A = float(coef[3 + 2 * i])
        B = float(coef[3 + 2 * i + 1])
        amp = float(np.sqrt(A ** 2 + B ** 2))
        phase_rad = float(np.arctan2(B, A))
        phase_deg = float(np.degrees(phase_rad)) % 360.0
        snr = amp / (noise_std / np.sqrt(n / 2.0)) if noise_std > 0 else 0.0
        results[name] = {"amp": amp, "phase_deg": phase_deg, "snr": snr}
    return results, noise_std


def load_station_multiday(st, doys, year, product):
    """Load clock residual for a station across multiple days."""
    all_resid = []
    all_t = []
    station_info = None
    for doy in doys:
        tag = f'{year}{doy:03d}'
        s = td.load_series(product, tag, min_valid_frac=0.0)
        if not s or st not in s:
            return None
        d = s[st]
        r = np.asarray(d['resid'], float)
        t = np.asarray(d['t'], float) + (doy - 1) * 86400.0
        all_resid.append(r)
        all_t.append(t)
        if station_info is None:
            station_info = d['station']
    resid = np.concatenate(all_resid)
    t = np.concatenate(all_t)
    good = np.isfinite(resid)
    if good.mean() < 0.8:
        return None
    if not good.all():
        ix = np.arange(len(resid))
        resid = np.interp(ix, ix[good], resid[good])
    # Detrend
    tc = t - t[0]
    c = np.polyfit(tc, resid, 2)
    resid = resid - np.polyval(c, tc)
    return resid, t, station_info


def find_best_window(product, year, n_days):
    """Find the best n-day contiguous window with max common stations."""
    available = []
    for doy in range(1, 367):
        tag = f'{year}{doy:03d}'
        s = td.load_series(product, tag, min_valid_frac=0.0)
        if s:
            available.append(doy)
    if not available:
        return None, 0
    runs = []
    start = available[0]
    prev = available[0]
    for d in available[1:]:
        if d == prev + 1:
            prev = d
        else:
            runs.append((start, prev))
            start = d
            prev = d
    runs.append((start, prev))
    best_count = 0
    best_start = None
    for r_start, r_end in runs:
        if r_end - r_start + 1 < n_days:
            continue
        for start in range(r_start, r_end - n_days + 2):
            common = None
            for doy in range(start, start + n_days):
                tag = f'{year}{doy:03d}'
                s = td.load_series(product, tag, min_valid_frac=0.0)
                if not s:
                    common = None
                    break
                stations = set(s.keys())
                common = stations if common is None else common & stations
            n = len(common) if common else 0
            if n > best_count:
                best_count = n
                best_start = start
    return best_start, best_count


def run_dataset(product, year, n_days, label):
    """Run the gravitational phase test on one dataset."""
    print(f"\n  === {label}: {product}, {year}, {n_days}d ===", flush=True)
    best_start, best_count = find_best_window(product, year, n_days)
    if best_start is None:
        print(f"    No suitable window", flush=True)
        return None
    doys = list(range(best_start, best_start + n_days))
    print(f"    Window: DOY {best_start}-{best_start+n_days-1}, {best_count} common stations", flush=True)

    common = None
    for doy in doys:
        tag = f'{year}{doy:03d}'
        s = td.load_series(product, tag, min_valid_frac=0.0)
        if not s: continue
        stations = set(s.keys())
        common = stations if common is None else common & stations
    common = sorted(common)

    stations_data = {}
    for st in common:
        result = load_station_multiday(st, doys, year, product)
        if result is None: continue
        resid, t, info = result
        if np.std(resid) > 50: continue
        stations_data[st] = (resid, t, info)
    print(f"    Loaded: {len(stations_data)} stations", flush=True)

    # Harmonic fit
    freq_list = list(TIDAL_FREQS.keys())
    freq_hz = {name: TIDAL_FREQS[name] for name in freq_list}
    all_results = {}
    for st in sorted(stations_data.keys()):
        resid, t, info = stations_data[st]
        results, noise = harmonic_fit(resid, t, freq_hz)
        all_results[st] = {
            "results": results,
            "lon": info.lon_deg,
            "lat": info.lat_deg,
            "noise": noise,
        }

    # Also fit control frequencies
    for st in sorted(stations_data.keys()):
        resid, t, info = stations_data[st]
        ctrl_results = {}
        for cf in CONTROL_FREQS_HZ:
            r, _ = harmonic_fit(resid, t, {"ctrl": cf})
            ctrl_results[f"{cf*1e6:.0f}uHz"] = r["ctrl"]
        all_results[st]["control"] = ctrl_results

    st_list = sorted(all_results.keys())
    n_st = len(st_list)

    # === Cross-station phase prediction using harmonic fit phases ===
    # For each tidal line, test if cos(2*(lon1-lon2)) predicts cos(phase1-phase2)
    line_results = {}
    for line_name in freq_list:
        expected = []
        observed = []
        amps = []
        snrs = []
        for i in range(n_st):
            for j in range(i+1, n_st):
                s1, s2 = st_list[i], st_list[j]
                r1, r2 = all_results[s1], all_results[s2]
                dlon = r1["lon"] - r2["lon"]
                # Gravitational prediction: phase diff = 2*dlon
                # (for semidiurnal tides; for diurnal tides it's 1*dlon)
                if "2" in line_name:  # semidiurnal
                    expected_dphase = 2.0 * dlon
                else:  # diurnal
                    expected_dphase = 1.0 * dlon
                expected_corr = np.cos(np.radians(expected_dphase))

                obs_dphase = r1["results"][line_name]["phase_deg"] - r2["results"][line_name]["phase_deg"]
                obs_corr = np.cos(np.radians(obs_dphase))

                expected.append(expected_corr)
                observed.append(obs_corr)
                amps.append(min(r1["results"][line_name]["amp"], r2["results"][line_name]["amp"]))
                snrs.append(min(r1["results"][line_name]["snr"], r2["results"][line_name]["snr"]))

        expected = np.array(expected)
        observed = np.array(observed)
        amps = np.array(amps)
        snrs = np.array(snrs)
        n_pairs = len(expected)

        if n_pairs < 20:
            line_results[line_name] = {"n_pairs": n_pairs, "r": None, "p": None}
            continue

        r, p = stats.pearsonr(expected, observed)

        # Also compute for high-SNR pairs only
        high_snr = snrs > 3.0
        if high_snr.sum() >= 20:
            r_hsnr, p_hsnr = stats.pearsonr(expected[high_snr], observed[high_snr])
        else:
            r_hsnr, p_hsnr = None, None

        # Amplitude-weighted correlation
        if amps.sum() > 0:
            w = amps / amps.sum()
            wm = np.sum(w * expected * observed) - np.sum(w * expected) * np.sum(w * observed)
            wv = np.sqrt(np.sum(w * expected**2) - np.sum(w * expected)**2) * \
                 np.sqrt(np.sum(w * observed**2) - np.sum(w * observed)**2)
            r_weighted = wm / wv if wv > 0 else 0
        else:
            r_weighted = None

        line_results[line_name] = {
            "n_pairs": n_pairs,
            "r": float(r),
            "p": float(p),
            "r_high_snr": float(r_hsnr) if r_hsnr else None,
            "p_high_snr": float(p_hsnr) if p_hsnr else None,
            "n_high_snr": int(high_snr.sum()),
            "r_weighted": float(r_weighted) if r_weighted else None,
            "mean_amp_ns": float(np.mean(amps)),
        }

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {line_name}: r={r:.4f}, p={p:.6f} {sig}, n={n_pairs}, "
              f"high-SNR r={r_hsnr:.4f}" if r_hsnr else
              f"    {line_name}: r={r:.4f}, p={p:.6f} {sig}, n={n_pairs}", flush=True)

    # Control frequencies
    ctrl_results = {}
    for cf_name in [f"{cf*1e6:.0f}uHz" for cf in CONTROL_FREQS_HZ]:
        expected = []
        observed = []
        for i in range(n_st):
            for j in range(i+1, n_st):
                s1, s2 = st_list[i], st_list[j]
                r1, r2 = all_results[s1], all_results[s2]
                dlon = r1["lon"] - r2["lon"]
                expected_corr = np.cos(np.radians(2.0 * dlon))
                obs_dphase = r1["control"][cf_name]["phase_deg"] - r2["control"][cf_name]["phase_deg"]
                obs_corr = np.cos(np.radians(obs_dphase))
                expected.append(expected_corr)
                observed.append(obs_corr)
        expected = np.array(expected)
        observed = np.array(observed)
        if len(expected) >= 20:
            r, p = stats.pearsonr(expected, observed)
            ctrl_results[cf_name] = {"n_pairs": len(expected), "r": float(r), "p": float(p)}
            print(f"    ctrl {cf_name}: r={r:.4f}, p={p:.4f}", flush=True)

    # === Key comparison: M2 vs S2 vs control ===
    m2_r = line_results.get("M2", {}).get("r")
    s2_r = line_results.get("S2", {}).get("r")
    o1_r = line_results.get("O1", {}).get("r")
    ctrl_rs = [v["r"] for v in ctrl_results.values() if v.get("r") is not None]
    ctrl_r_mean = float(np.mean(ctrl_rs)) if ctrl_rs else None

    print(f"\n    SUMMARY: M2 r={m2_r:.4f}, S2 r={s2_r:.4f}, O1 r={o1_r:.4f}, "
          f"ctrl mean r={ctrl_r_mean:.4f}" if ctrl_r_mean is not None else
          f"\n    SUMMARY: M2 r={m2_r:.4f}, S2 r={s2_r:.4f}, O1 r={o1_r:.4f}", flush=True)

    return {
        "label": label,
        "product": product,
        "year": year,
        "n_days": n_days,
        "window_doy": [best_start, best_start + n_days - 1],
        "n_stations": n_st,
        "tidal_lines": line_results,
        "control_freqs": ctrl_results,
        "m2_r": m2_r,
        "s2_r": s2_r,
        "o1_r": o1_r,
        "control_r_mean": ctrl_r_mean,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Step 165b: Direct gravitational phase test (harmonic fit phases)", flush=True)
    print("  Testing: does cos(2*(lon1-lon2)) predict cos(phase1-phase2)?", flush=True)

    datasets = [
        ("clk_cod", 2025, 30, "COD 2025 (30d)"),
        ("clk_gfz", 2025, 30, "GFZ 2025 (30d)"),
        ("clk_cod", 2024, 90, "COD 2024 (90d)"),
        ("clk_gfz", 2024, 90, "GFZ 2024 (90d)"),
    ]

    all_results = []
    for product, year, n_days, label in datasets:
        result = run_dataset(product, year, n_days, label)
        if result:
            all_results.append(result)

    # === Meta-analysis across datasets ===
    print("\n\n  === META-ANALYSIS ===", flush=True)
    m2_rs = [d["m2_r"] for d in all_results if d["m2_r"] is not None]
    s2_rs = [d["s2_r"] for d in all_results if d["s2_r"] is not None]
    o1_rs = [d["o1_r"] for d in all_results if d["o1_r"] is not None]
    ctrl_rs = [d["control_r_mean"] for d in all_results if d["control_r_mean"] is not None]

    print(f"  M2  r across datasets: {m2_rs} -> mean={np.mean(m2_rs):.4f}", flush=True)
    print(f"  S2  r across datasets: {s2_rs} -> mean={np.mean(s2_rs):.4f}", flush=True)
    print(f"  O1  r across datasets: {o1_rs} -> mean={np.mean(o1_rs):.4f}", flush=True)
    print(f"  Ctrl r across datasets: {ctrl_rs} -> mean={np.mean(ctrl_rs):.4f}", flush=True)

    # Fisher combined p-value for M2
    from scipy.stats import combine_pvalues
    m2_ps = [d["tidal_lines"]["M2"]["p"] for d in all_results if "M2" in d["tidal_lines"]]
    if len(m2_ps) >= 2:
        chi2, combined_p = combine_pvalues(m2_ps, method='fisher')
        print(f"  M2 Fisher combined p: {combined_p:.2e} (chi2={chi2:.2f})", flush=True)

    output = {
        "step": "165b",
        "title": "Direct gravitational phase test using harmonic fit phases",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design": (
            "For each station pair, the gravitational prediction for the M2 phase difference "
            "is 2*(lon1-lon2), purely longitude-dependent. The test correlates cos(expected_dphase) "
            "with cos(observed_dphase) from harmonic fitting. Control frequencies should show no "
            "longitude dependence. S2 (solar) has both gravitational and atmospheric components, "
            "so it should also show longitude dependence but with additional scatter from atmospheric "
            "phase variations. O1 (lunar diurnal) has phase difference 1*(lon1-lon2)."
        ),
        "pre_registered_prediction": (
            "TEP (gravitational): M2 r > 0 and significantly exceeds control r. "
            "S2 r > 0 (gravitational component) but may be weaker than M2 due to atmospheric scatter. "
            "Control r ~ 0 (no longitude dependence at non-tidal frequencies). "
            "M2 r should be consistent across datasets (COD, GFZ, 2024, 2025)."
        ),
        "datasets": all_results,
        "meta_analysis": {
            "m2_r_mean": float(np.mean(m2_rs)) if m2_rs else None,
            "m2_r_values": m2_rs,
            "s2_r_mean": float(np.mean(s2_rs)) if s2_rs else None,
            "s2_r_values": s2_rs,
            "o1_r_mean": float(np.mean(o1_rs)) if o1_rs else None,
            "o1_r_values": o1_rs,
            "control_r_mean": float(np.mean(ctrl_rs)) if ctrl_rs else None,
            "control_r_values": ctrl_rs,
            "m2_fisher_combined_p": float(combined_p) if len(m2_ps) >= 2 else None,
        },
    }

    OUT.write_text(json.dumps(output, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
