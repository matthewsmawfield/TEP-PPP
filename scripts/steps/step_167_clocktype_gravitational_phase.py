#!/usr/bin/env python3
"""
STEP 167  --  Clock-type screening of the gravitational phase signature.

Step 165b showed that M2 clock phases match the gravitational prediction
cos(2*(lon1-lon2)) across all stations (Fisher p=3.53e-13).  Step 166b
showed that the co-located differential |r| is 2.8x the control across 97
pairs.

This step asks: does the gravitational phase matching depend on clock type?

TEP predicts that quartz clocks couple MORE to the TEP field than atomic
clocks (screening).  The standard gravitational tide (common-mode) affects
all clocks equally.  If the M2 phase matching is stronger for quartz than
atomic, this is the screening signature.

Stations are classified by clock noise level:
  - Atomic (< 5 ns):  H-maser, Cs, Rb referenced stations
  - Good quartz (5-50 ns):  TCXO with good processing
  - Bad quartz (> 50 ns):  extreme drift, excluded

OUTPUT: results/outputs/step_167_clocktype_gravitational_phase.json
"""
from __future__ import annotations

import sys, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

OUT = ROOT / "results" / "outputs" / "step_167_clocktype_gravitational_phase.json"

M2_FREQ = 1.0 / (12.42 * 3600.0)
O1_FREQ = 1.0 / (25.82 * 3600.0)
S2_FREQ = 2.0 / 86400.0
ALL_FREQS = [M2_FREQ, O1_FREQ, S2_FREQ]
ALL_NAMES = ["M2", "O1", "S2"]
CONTROL_FREQS = [13e-6, 15e-6, 17e-6, 19e-6, 25e-6, 27e-6, 29e-6, 31e-6]

PRODUCT = "clk_cod_ops"
DOYS = list(range(133, 163))
YEAR = 2024
N_PERM = 10000
RINEX_PROCESSED = ROOT.parent / "TEP-GNSS-RINEX" / "data" / "processed"

TIDAL_FREQS_8 = [
    ("M2", 1.0 / (12.42 * 3600.0), 2),
    ("N2", 1.0 / (12.66 * 3600.0), 2),
    ("S2", 2.0 / 86400.0, 2),
    ("O1", 1.0 / (25.82 * 3600.0), 1),
    ("K1", 1.0 / (23.93 * 3600.0), 1),
    ("S1", 1.0 / 86400.0, 1),
    ("P1", 1.0 / (24.07 * 3600.0), 1),
    ("Q1", 1.0 / (26.87 * 3600.0), 1),
]


def _load_verified_clock_types() -> dict[str, str]:
    """Build verified IGS site-log oscillator classification map."""
    verified_ct: dict[str, str] = {}
    full_p = ROOT / "data" / "meta" / "igs_clock_type_full.json"
    if full_p.exists():
        raw = json.loads(full_p.read_text())
        for st, val in raw.items():
            ct = val[0] if isinstance(val, list) else val
            if ct in ("ATOMIC", "QUARTZ"):
                verified_ct[st] = ct
    out_dir = ROOT / "results" / "outputs"
    for fn in ("step_152c_igs_lunar.json", "step_152f_cod_2024.json",
               "step_152d_gfz_lunar.json", "step_152e_gfz_2024.json"):
        p = out_dir / fn
        if p.exists():
            ps = json.loads(p.read_text()).get("per_station", {})
            for st, info in ps.items():
                ct = info.get("clock_type")
                if ct in ("ATOMIC", "QUARTZ"):
                    verified_ct.setdefault(st, ct)
    for fn, key, ckey in (("step_72_housing_screening.json", "stations", "clock_type"),
                          ("step_74c_comprehensive_clocks.json", "ground_stations", "clock_class")):
        p = out_dir / fn
        if p.exists():
            blk = json.loads(p.read_text()).get(key, {})
            for st, info in blk.items():
                ct = info.get(ckey)
                if ct in ("ATOMIC", "QUARTZ"):
                    verified_ct.setdefault(st, ct)
    for meta_p in (ROOT / "data" / "meta" / "optimal_100_metadata.json",
                   RINEX_PROCESSED / "optimal_100_metadata.json"):
        if meta_p.exists():
            meta100 = json.loads(meta_p.read_text()).get("stations", {})
            for st, info in meta100.items():
                if st not in verified_ct:
                    fs = (info.get("equipment") or {}).get("frequency_standard", "")
                    if fs in {"H-MASER", "CESIUM", "RUBIDIUM"}:
                        verified_ct[st] = "ATOMIC"
                    elif fs in {"OCXO", "INTERNAL", "QUARTZ", "TCXO"}:
                        verified_ct[st] = "QUARTZ"
    return verified_ct


def _station_perm_test(lons: np.ndarray, phases_rad: np.ndarray, harm: int,
                       n_perm: int = N_PERM, seed: int = 16700) -> dict:
    """Station-level longitude permutation test (effective dof = n_stations)."""
    n = len(lons)
    iu = np.triu_indices(n, k=1)
    dlon = np.radians(lons[iu[0]] - lons[iu[1]])
    exp = np.cos(harm * dlon)
    obs = np.cos(phases_rad[iu[0]] - phases_rad[iu[1]])
    r_obs, p_pair = stats.pearsonr(exp, obs)
    obs_c = obs - obs.mean()
    obs_n = float(np.linalg.norm(obs_c))
    rng = np.random.default_rng(seed)
    null_r = np.empty(n_perm)
    for k in range(n_perm):
        pl = rng.permutation(lons)
        ne = np.cos(harm * np.radians(pl[iu[0]] - pl[iu[1]]))
        ne_c = ne - ne.mean()
        den = float(np.linalg.norm(ne_c)) * obs_n
        null_r[k] = float(np.dot(ne_c, obs_c) / den) if den > 1e-30 else 0.0
    p_perm = float((1 + np.sum(np.abs(null_r) >= abs(r_obs))) / (1 + n_perm))
    return {
        "r": float(r_obs),
        "p_perm_station": p_perm,
        "p_naive_pairs": float(p_pair),
        "n_stations": int(n),
        "n_pairs": int(len(exp)),
        "null_r_std": float(np.std(null_r)),
    }


def _run_metadata_verified_arm() -> dict:
    """Evaluate S2 vs M2/O1/N2 using verified IGS metadata and station-level
    longitude permutation across the 100-station 2024 precise network and the
    four CODE/GFZ final clock datasets."""
    from scipy.stats import combine_pvalues
    verified_ct = _load_verified_clock_types()
    out: dict = {
        "classification_source": "verified IGS site-log metadata (H-MASER/CESIUM/RUBIDIUM vs INTERNAL/OCXO)",
        "inference": f"station-level longitude permutation (n_perm={N_PERM}, effective dof = n_stations)",
        "n_verified_stations_total": len(verified_ct),
    }

    # 1. Simultaneous 8-line harmonic fit on 2024 30d precise clock series (DOY 133-162)
    if RINEX_PROCESSED.exists():
        doys_24 = list(range(133, 163))
        fits_24 = {}
        for st in sorted(verified_ct):
            all_r, all_t = [], []
            info = None
            ok = True
            for doy in doys_24:
                p = RINEX_PROCESSED / f"{st}_2024{doy:03d}.npz"
                if not p.exists():
                    ok = False
                    break
                z = td._read_npz(p)
                if z is None or "precise_clock_bias_ns" not in z:
                    ok = False
                    break
                clk = np.asarray(z["precise_clock_bias_ns"], float)
                t = (np.asarray(z["timestamps"], float)
                     if "timestamps" in z else np.arange(len(clk)) * 30.0)
                n = min(len(clk), len(t))
                clk, t = clk[:n], t[:n]
                if np.mean(np.isfinite(clk)) < 0.85 or td.detect_clock_jumps(clk, 12.0) > 10000:
                    ok = False
                    break
                if info is None:
                    info = td._station_from_npz(st, z)
                r = td.robust_detrend(clk)
                all_r.append(r)
                all_t.append(t + (doy - 1) * 86400.0)
            if not ok or info is None:
                continue
            resid = np.concatenate(all_r)
            t = np.concatenate(all_t)
            g = np.isfinite(resid)
            if g.mean() < 0.8:
                continue
            if not g.all():
                ix = np.arange(len(resid))
                resid = np.interp(ix, ix[g], resid[g])
            if np.std(resid) > 100:
                continue
            tc = t - t[0]
            resid = resid - np.polyval(np.polyfit(tc, resid, 2), tc)
            ts_half = (tc[-1] - tc[0]) / 2.0 if len(tc) > 1 else 1.0
            tn = tc / ts_half
            cols = [np.ones_like(tn), tn, tn ** 2]
            for _, f, _ in TIDAL_FREQS_8:
                w = 2.0 * np.pi * f * tc
                cols.append(np.cos(w))
                cols.append(np.sin(w))
            design = np.column_stack(cols)
            coef, _, _, _ = lstsq(design, resid, cond=1e-10)
            ph_map = {}
            for i, (nm, _, _) in enumerate(TIDAL_FREQS_8):
                ph_map[nm] = float(np.arctan2(coef[3 + 2 * i + 1], coef[3 + 2 * i]))
            fits_24[st] = (ph_map, info.lon_deg, verified_ct[st])

        precise_2024 = {}
        for grp in ("ATOMIC", "QUARTZ"):
            stas = [s for s, v in fits_24.items() if v[2] == grp]
            if len(stas) < 6:
                continue
            lons = np.array([fits_24[s][1] for s in stas])
            grp_res = {"n_stations": len(stas), "n_pairs": len(stas) * (len(stas) - 1) // 2}
            for nm, _, harm in TIDAL_FREQS_8[:4]:
                ph = np.array([fits_24[s][0][nm] for s in stas])
                grp_res[nm] = _station_perm_test(
                    lons, ph, harm, N_PERM, 16700 + sum(ord(c) for c in grp + nm))
            precise_2024[grp.lower()] = grp_res
        if "atomic" in precise_2024 and "quartz" in precise_2024:
            m2_q = precise_2024["quartz"]["M2"]["r"]
            m2_a = precise_2024["atomic"]["M2"]["r"]
            o1_q = precise_2024["quartz"]["O1"]["r"]
            o1_a = precise_2024["atomic"]["O1"]["r"]
            precise_2024["M2_quartz_over_atomic_ratio"] = float(m2_q / max(m2_a, 0.001))
            precise_2024["O1_quartz_over_atomic_ratio"] = float(o1_q / max(o1_a, 0.001))
        out["precise_2024_simultaneous_8line"] = precise_2024

    # 2. Multi-centre IGS/CODE and GFZ final clock products (step_152c/d/e/f)
    coords_p = RINEX_PROCESSED / "station_coordinates.json"
    coords = {}
    if coords_p.exists():
        raw_c = json.loads(coords_p.read_text())
        coords = {s: td.ecef_to_geodetic(*xyz) for s, xyz in raw_c.items()}
    s165_p = ROOT / "results" / "outputs" / "step_165_gravitational_phase.json"
    if s165_p.exists():
        for s, v in json.loads(s165_p.read_text()).get("per_station", {}).items():
            if s not in coords and "lon_deg" in v:
                coords[s] = (v.get("lat_deg", 0.0), v["lon_deg"])

    datasets_meta = {}
    q_m2_ps, a_m2_ps = [], []
    q_s2_ps, a_s2_ps = [], []
    for tag, fn in (("COD_2025_30d", "step_152c_igs_lunar.json"),
                    ("COD_90d", "step_152f_cod_2024.json"),
                    ("GFZ_2025_30d", "step_152d_gfz_lunar.json"),
                    ("GFZ_90d", "step_152e_gfz_2024.json")):
        p = ROOT / "results" / "outputs" / fn
        if not p.exists():
            continue
        ps = json.loads(p.read_text()).get("per_station", {})
        ds_res = {}
        for grp in ("ATOMIC", "QUARTZ"):
            stas = sorted([s for s in ps if verified_ct.get(s) == grp and s in coords])
            if len(stas) < 6:
                continue
            lons = np.array([coords[s][1] for s in stas])
            g_dict = {"n_stations": len(stas), "n_pairs": len(stas) * (len(stas) - 1) // 2}
            for line, harm in (("M2", 2), ("O1", 1), ("S2", 2)):
                ph = np.array([ps[s][line]["phase"] for s in stas])
                g_dict[line] = _station_perm_test(
                    lons, ph, harm, N_PERM, 16700 + sum(ord(c) for c in fn + grp + line))
            ds_res[grp.lower()] = g_dict
        if "quartz" in ds_res:
            q_m2_ps.append(ds_res["quartz"]["M2"]["p_perm_station"])
            q_s2_ps.append(ds_res["quartz"]["S2"]["p_perm_station"])
        if "atomic" in ds_res:
            a_m2_ps.append(ds_res["atomic"]["M2"]["p_perm_station"])
            a_s2_ps.append(ds_res["atomic"]["S2"]["p_perm_station"])
        datasets_meta[tag] = ds_res

    out["multi_centre_final_products"] = datasets_meta
    # Dependence-preserving joint station-longitude permutation across all 4 CODE/GFZ datasets
    # (permuting each unique station's longitude ONCE per trial across all datasets simultaneously):
    dsets_raw = {}
    for tag, fn in (("COD_2025_30d", "step_152c_igs_lunar.json"),
                    ("COD_90d", "step_152f_cod_2024.json"),
                    ("GFZ_2025_30d", "step_152d_gfz_lunar.json"),
                    ("GFZ_90d", "step_152e_gfz_2024.json")):
        p = ROOT / "results" / "outputs" / fn
        if p.exists():
            dsets_raw[tag] = json.loads(p.read_text()).get("per_station", {})
    joint_perm_summary = {}
    for grp in ("QUARTZ", "ATOMIC"):
        all_stas = sorted({s for ps in dsets_raw.values() for s in ps if verified_ct.get(s) == grp and s in coords})
        sta_to_idx = {s: i for i, s in enumerate(all_stas)}
        base_lons = np.array([coords[s][1] for s in all_stas])
        ds_info = []
        for tag, ps in dsets_raw.items():
            stas = sorted([s for s in ps if verified_ct.get(s) == grp and s in coords])
            if len(stas) < 6:
                continue
            idxs = np.array([sta_to_idx[s] for s in stas])
            ph = np.array([ps[s]["M2"]["phase"] for s in stas])
            ii, jj = np.triu_indices(len(stas), k=1)
            y_pairs = np.cos(ph[ii] - ph[jj])
            ds_info.append((idxs, ii, jj, y_pairs))
        def _eval_joint_m2(lon_vec):
            rs = []
            for idxs, ii, jj, y_pairs in ds_info:
                l_sub = lon_vec[idxs]
                x_pairs = np.cos(np.radians(2 * (l_sub[ii] - l_sub[jj])))
                vx = x_pairs - x_pairs.mean()
                vy = y_pairs - y_pairs.mean()
                rs.append(float((vx @ vy) / np.sqrt((vx @ vx) * (vy @ vy))))
            return float(np.mean(rs))
        obs_mean_r = _eval_joint_m2(base_lons)
        rng_j = np.random.default_rng(16799)
        null_means = np.empty(N_PERM)
        for k in range(N_PERM):
            null_means[k] = _eval_joint_m2(rng_j.permutation(base_lons))
        p_joint = float((np.sum(np.abs(null_means) >= abs(obs_mean_r)) + 1) / (N_PERM + 1))
        z_joint = float((obs_mean_r - np.mean(null_means)) / np.std(null_means))
        joint_perm_summary[grp.lower()] = {
            "n_permutations": N_PERM,
            "seed": 16799,
            "tail": "two-sided absolute mean correlation",
            "null_assumption": "Station longitude labels are exchangeable within oscillator class; shared station assignments are retained across datasets.",
            "z_scope": "Standardized distance from permutation mean, not a Gaussian-tail significance.",
            "n_datasets": len(ds_info),
            "n_unique_stations": len(all_stas),
            "mean_M2_r": round(obs_mean_r, 4),
            "null_mean": round(float(np.mean(null_means)), 4),
            "null_std": round(float(np.std(null_means)), 4),
            "z_joint_station_perm": round(z_joint, 2),
            "p_joint_station_perm": round(p_joint, 4),
        }
    out["multi_centre_fisher_combined"] = {
        "quartz_M2_p_perm_fisher": float(combine_pvalues(q_m2_ps)[1]) if q_m2_ps else None,
        "atomic_M2_p_perm_fisher": float(combine_pvalues(a_m2_ps)[1]) if a_m2_ps else None,
        "quartz_S2_p_perm_fisher": float(combine_pvalues(q_s2_ps)[1]) if q_s2_ps else None,
        "atomic_S2_p_perm_fisher": float(combine_pvalues(a_s2_ps)[1]) if a_s2_ps else None,
        "dependence_preserving_joint_station_permutation_M2": joint_perm_summary,
    }
    return out


def harmonic_fit(resid, t_sec, freqs):
    t0 = t_sec[0]
    tc = t_sec - t0 - (t_sec[-1] - t_sec[0]) / 2.0
    ts = (tc[-1] - tc[0]) / 2.0 if len(tc) > 1 else 1.0
    tn = tc / ts
    cols = [np.ones_like(tn), tn, tn ** 2]
    for f in freqs:
        omega = 2.0 * np.pi * f * (t_sec - t0)
        cols.append(np.cos(omega))
        cols.append(np.sin(omega))
    design = np.column_stack(cols)
    coef, _, _, _ = lstsq(design, resid, cond=1e-12)
    results = []
    for i, f in enumerate(freqs):
        A = float(coef[3 + 2 * i])
        B = float(coef[3 + 2 * i + 1])
        phase_deg = float(np.degrees(np.arctan2(B, A))) % 360.0
        amp = float(np.sqrt(A**2 + B**2))
        results.append({"amp": amp, "phase_deg": phase_deg})
    return results


def load_station(st):
    all_resid, all_t = [], []
    info = None
    for doy in DOYS:
        tag = f'{YEAR}{doy:03d}'
        try:
            s = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10000)
        except Exception:
            return None
        if not s or st not in s:
            return None
        d = s[st]
        all_resid.append(np.asarray(d['resid'], float))
        all_t.append(np.asarray(d['t'], float) + (doy - 1) * 86400.0)
        if info is None:
            info = d['station']
    resid = np.concatenate(all_resid)
    t = np.concatenate(all_t)
    good = np.isfinite(resid)
    if good.mean() < 0.8:
        return None
    if not good.all():
        ix = np.arange(len(resid))
        resid = np.interp(ix, ix[good], resid[good])
    tc = t - t[0]
    c = np.polyfit(tc, resid, 2)
    resid = resid - np.polyval(c, tc)
    return resid, t, info


def run_group(stations, label):
    print(f"\n  {label} ({len(stations)} stations)", flush=True)
    data = {}
    for st in stations:
        r = load_station(st)
        if r is None:
            continue
        resid, t, info = r
        if np.std(resid) > 100:
            continue
        data[st] = (resid, t, info)
    print(f"    Loaded: {len(data)}", flush=True)
    if len(data) < 10:
        return None

    fits = {}
    for st in sorted(data.keys()):
        resid, t, info = data[st]
        results = harmonic_fit(resid, t, ALL_FREQS)
        ctrl = {}
        for cf in CONTROL_FREQS:
            cr = harmonic_fit(resid, t, [cf])
            ctrl[f"{cf*1e6:.0f}uHz"] = cr[0]
        fits[st] = {"results": results, "ctrl": ctrl,
                    "lon": info.lon_deg, "lat": info.lat_deg}

    st_list = sorted(fits.keys())
    n_st = len(st_list)
    out = {"n_stations": n_st}

    for li, ln in enumerate(ALL_NAMES):
        exp, obs = [], []
        for i in range(n_st):
            for j in range(i+1, n_st):
                r1, r2 = fits[st_list[i]], fits[st_list[j]]
                dlon = r1["lon"] - r2["lon"]
                edp = 2.0 * dlon if "2" in ln else 1.0 * dlon
                exp.append(np.cos(np.radians(edp)))
                obs.append(np.cos(np.radians(
                    r1["results"][li]["phase_deg"] - r2["results"][li]["phase_deg"])))
        r, p = stats.pearsonr(exp, obs)
        out[ln] = {"r": float(r), "p": float(p), "n_pairs": len(exp)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {ln}: r={r:.4f}, p={p:.2e} {sig}, n={len(exp)}", flush=True)

    ctrl_rs = []
    for cf_name in [f"{cf*1e6:.0f}uHz" for cf in CONTROL_FREQS]:
        exp, obs = [], []
        for i in range(n_st):
            for j in range(i+1, n_st):
                r1, r2 = fits[st_list[i]], fits[st_list[j]]
                dlon = r1["lon"] - r2["lon"]
                exp.append(np.cos(np.radians(2.0 * dlon)))
                obs.append(np.cos(np.radians(
                    r1["ctrl"][cf_name]["phase_deg"] - r2["ctrl"][cf_name]["phase_deg"])))
        r, _ = stats.pearsonr(exp, obs)
        ctrl_rs.append(r)
    out["control_r_mean"] = float(np.mean(ctrl_rs))
    print(f"    Control mean r: {np.mean(ctrl_rs):.4f}", flush=True)
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Step 167: Clock-type screening of gravitational phase signature", flush=True)

    # Classify by noise (if raw PRODUCT is on disk; otherwise preserve existing
    # noise-stratified COD0OPSFIN block from OUT and compute the metadata-verified
    # station-permutation block).
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    dates = td.available_dates(PRODUCT, YEAR)
    if dates:
        print("  Computing noise levels...", flush=True)
        noise = {}
        for tag in dates[:30]:
            try:
                ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10000)
            except Exception:
                continue
            for st, d in ser.items():
                r = np.asarray(d['resid'], float)
                v = np.isfinite(r)
                if v.sum() > 100:
                    noise.setdefault(st, []).append(float(np.std(r[v])))
        avg = {st: float(np.mean(v)) for st, v in noise.items()}

        atomic = [st for st, n in avg.items() if n < 5.0]
        quartz = [st for st, n in avg.items() if 5.0 <= n <= 50.0]
        print(f"  Atomic (<5 ns): {len(atomic)}, Quartz (5-50 ns): {len(quartz)}", flush=True)

        atomic_res = run_group(atomic, "ATOMIC (< 5 ns)")
        quartz_res = run_group(quartz, "GOOD QUARTZ (5-50 ns)")

        # 2025 replication
        atomic_res25 = quartz_res25 = None
        dates25 = td.available_dates(PRODUCT, 2025)
        if dates25:
            global DOYS
            DOYS = list(range(22, 52))
            atomic_res25 = run_group(atomic, "ATOMIC 2025")
            quartz_res25 = run_group(quartz, "GOOD QUARTZ 2025")
    else:
        atomic_res = (existing.get("results_2024") or {}).get("atomic")
        quartz_res = (existing.get("results_2024") or {}).get("quartz")
        atomic_res25 = (existing.get("results_2025") or {}).get("atomic")
        quartz_res25 = (existing.get("results_2025") or {}).get("quartz")

    meta_verified = _run_metadata_verified_arm()

    # Summary
    print("\n=== SCREENING COMPARISON ===", flush=True)
    summary = {}
    for yr, ar, qr in [("2024", atomic_res, quartz_res),
                        ("2025", atomic_res25, quartz_res25)]:
        if ar and qr:
            m2a = ar.get("M2", {}).get("r", 0)
            m2q = qr.get("M2", {}).get("r", 0)
            s2a = ar.get("S2", {}).get("r", 0)
            s2q = qr.get("S2", {}).get("r", 0)
            ca = ar.get("control_r_mean", 0)
            cq = qr.get("control_r_mean", 0)
            print(f"  {yr}:")
            print(f"    M2: atomic={m2a:.4f}, quartz={m2q:.4f}, ratio={m2q/max(m2a,0.001):.2f}x")
            print(f"    S2: atomic={s2a:.4f}, quartz={s2q:.4f}, ratio={s2q/max(s2a,0.001):.2f}x")
            print(f"    Ctrl: atomic={ca:.4f}, quartz={cq:.4f}")
            print(f"    M2/Ctrl: atomic={m2a/max(ca,0.001):.1f}x, quartz={m2q/max(cq,0.001):.1f}x")
            summary[yr] = {
                "atomic_M2_r": m2a, "quartz_M2_r": m2q,
                "atomic_S2_r": s2a, "quartz_S2_r": s2q,
                "atomic_ctrl_r": ca, "quartz_ctrl_r": cq,
                "M2_quartz_over_atomic": m2q / max(m2a, 0.001),
                "S2_quartz_over_atomic": s2q / max(s2a, 0.001),
            }

    output = {
        "step": 167,
        "title": "Clock-type screening of the gravitational phase signature",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "design": (
            "Stations classified by clock noise: atomic (<5 ns) vs good quartz "
            "(5-50 ns). TEP predicts quartz > atomic for M2 (screening). "
            "S2 should show an even larger contrast because it has an atmospheric "
            "component that only affects quartz."
        ),
        "results_2024": {"atomic": atomic_res, "quartz": quartz_res},
        "results_2025": {"atomic": atomic_res25, "quartz": quartz_res25},
        "summary": summary,
        "metadata_verified_station_permutation": meta_verified,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\n  Output: {OUT}")


if __name__ == "__main__":
    main()
