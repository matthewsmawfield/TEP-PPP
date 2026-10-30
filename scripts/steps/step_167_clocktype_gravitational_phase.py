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

    # Classify by noise
    print("  Computing noise levels...", flush=True)
    noise = {}
    dates = td.available_dates(PRODUCT, YEAR)
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
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\n  Output: {OUT}")


if __name__ == "__main__":
    main()
