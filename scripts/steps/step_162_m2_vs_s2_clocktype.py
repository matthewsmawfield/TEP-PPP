#!/usr/bin/env python3
"""
STEP 162  --  Per-frequency-bin clock-type differentiation spectrum.

Step 160 found Q/A=9x in the tidal band (10-35 uHz) with single-day data.
CRITICAL ISSUE: single-day 30s data has 11.574 uHz resolution. The 3 bins
in the tidal band are at S1 (11.574), S2 (23.148), M3 (34.722) — SOLAR
tides, not the M2 lunar tide (22.365 uHz).

This step computes the quartz/atomic phase alignment ratio at EACH
frequency bin using 13-day multi-day prior data (0.891 uHz resolution).
This produces a Q/A ratio spectrum that shows exactly which frequencies
carry the clock-type differentiation.

TEP predicts: the Q/A ratio should peak at M2 (22.365 uHz, lunar) and
O1 (10.758 uHz, lunar), not at S2 (23.148 uHz, solar) or S1 (11.574 uHz).

Atmospheric origin predicts: the Q/A ratio should peak at S2 and S1
(solar atmospheric tides).

Output: results/outputs/step_162_m2_vs_s2_clocktype.json
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.steps import step_64_maximal_tep_p1 as s64
from scripts.steps.step_152_freq_spectrum import _load_multi_day_prior, _input_path

OUT = ROOT / "results" / "outputs" / "step_162_m2_vs_s2_clocktype.json"
PRODUCT = "clk_cod"
YEAR = 2025
RNG_SEED = 20260920

EVALUATION_DOYS = tuple(range(20, 366, 5))
MIN_VALID_DAYS = 10

# Tidal frequencies (uHz)
TIDAL_LINES = {
    "O1":  10.758,  # lunar diurnal
    "Q1":  10.676,  # lunar diurnal
    "S1":  11.574,  # solar diurnal
    "P1":  11.542,  # solar diurnal
    "K1":  11.608,  # mixed
    "N2":  21.422,  # lunar semidiurnal
    "M2":  22.365,  # lunar semidiurnal
    "S2":  23.148,  # solar semidiurnal
    "K2":  23.098,  # solar semidiurnal
    "M3":  34.533,  # lunar terdiurnal
}

def _clock_types():
    ct_raw = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    ctype = {}
    for n, v in ct_raw.items():
        if "QUARTZ" in v[0].upper():
            ctype[n] = "QUARTZ"
        elif "ATOMIC" in v[0].upper() or "MASER" in v[0].upper() or "CESIUM" in v[0].upper() or "RUBIDIUM" in v[0].upper():
            ctype[n] = "ATOMIC"
    return ctype

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctype = _clock_types()
    print("Step 162: Per-frequency-bin clock-type differentiation spectrum", flush=True)
    print(f"  Product: {PRODUCT}, Year: {YEAR}", flush=True)
    print(f"  Using 13-day multi-day prior", flush=True)
    print(flush=True)

    planned = [f"{YEAR}{d:03d}" for d in EVALUATION_DOYS]
    eval_tags = [t for t in planned if _input_path(t)]
    if len(eval_tags) < MIN_VALID_DAYS:
        print(f"  ERROR: only {len(eval_tags)} days", flush=True)
        return
    print(f"  {len(eval_tags)} evaluation days available", flush=True)
    print(flush=True)

    # Accumulate per-tidal-line quartz and atomic phase alignment
    # across all station pairs and all days
    tidal_accum = {name: {"q_sum": 0.0, "a_sum": 0.0, "q_n": 0, "a_n": 0,
                          "q_vals": [], "a_vals": []}
                   for name in TIDAL_LINES}
    # Also accumulate off-tide background
    bg_accum = {"q_sum": 0.0, "a_sum": 0.0, "q_n": 0, "a_n": 0,
                "q_vals": [], "a_vals": []}

    n_days_used = 0
    for k, tag in enumerate(eval_tags):
        cur = s64.load_day(tag, product=PRODUCT)
        if cur is None:
            continue
        names, t, raw, D = cur
        multi_prior = _load_multi_day_prior(tag, names, 13)
        if multi_prior is None:
            continue
        prior_names, prior_x, _ = multi_prior

        # Keep stations with known clock type
        kept = [n for n in prior_names if n in ctype and ctype[n] in ("QUARTZ", "ATOMIC")]
        if len(kept) < 20:
            continue

        # Distance matrix: map prior names to day names for D indexing
        name_to_day_idx = {nm: i for i, nm in enumerate(names)}
        kept_day_idx = [name_to_day_idx[nm] for nm in kept if nm in name_to_day_idx]
        kept_with_d = [nm for nm in kept if nm in name_to_day_idx]
        if len(kept_with_d) < 20:
            continue

        # Sub-distance matrix for kept stations
        D_sub = D[np.ix_(kept_day_idx, kept_day_idx)]

        kept_idx = [prior_names.index(nm) for nm in kept_with_d]
        X = prior_x[:, kept_idx].copy()
        n_t, n_sta = X.shape

        # Detrend
        for j in range(n_sta):
            X[:, j] = X[:, j] - np.mean(X[:, j])

        # FFT
        dt_s = s64.DT_S
        freqs = np.fft.rfftfreq(n_t, d=dt_s)
        freq_uhz = freqs * 1e6
        XF = np.fft.rfft(X, axis=0)

        # For each tidal line, find the nearest frequency bin
        tidal_bin_idx = {}
        for name, f_uhz in TIDAL_LINES.items():
            idx = np.argmin(np.abs(freq_uhz - f_uhz))
            tidal_bin_idx[name] = idx

        # Off-tide bins: frequencies >2 uHz from any tidal line
        off_tide_bins = []
        for idx in range(len(freqs)):
            f = freq_uhz[idx]
            if f < 5 or f > 50:
                continue
            min_dist = min(abs(f - t) for t in TIDAL_LINES.values())
            if min_dist > 2.0:
                off_tide_bins.append(idx)

        # Compute per-frequency phase alignment for each same-type pair
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                ni, nj = kept_with_d[i], kept_with_d[j]
                ti, tj = ctype[ni], ctype[nj]
                if ti != tj:
                    continue

                d_km = float(D_sub[i, j])
                if not (1000.0 <= d_km <= 5000.0):
                    continue  # TEP distance band only

                # Per-frequency phase alignment
                cross = np.conj(XF[:, i]) * XF[:, j]
                pa = np.cos(np.angle(cross))

                # Accumulate at each tidal line
                for name, idx in tidal_bin_idx.items():
                    val = float(pa[idx])
                    if np.isfinite(val):
                        tidal_accum[name]["q_sum" if ti == "QUARTZ" else "a_sum"] += val
                        tidal_accum[name]["q_n" if ti == "QUARTZ" else "a_n"] += 1
                        (tidal_accum[name]["q_vals" if ti == "QUARTZ" else "a_vals"]).append(val)

                # Accumulate off-tide background
                for idx in off_tide_bins:
                    val = float(pa[idx])
                    if np.isfinite(val):
                        bg_accum["q_sum" if ti == "QUARTZ" else "a_sum"] += val
                        bg_accum["q_n" if ti == "QUARTZ" else "a_n"] += 1
                        (bg_accum["q_vals" if ti == "QUARTZ" else "a_vals"]).append(val)

        n_days_used += 1
        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(eval_tags)}] days_used={n_days_used}, "
                  f"M2 Q n={tidal_accum['M2']['q_n']}, A n={tidal_accum['M2']['a_n']}", flush=True)

    if n_days_used < MIN_VALID_DAYS:
        print(f"  ERROR: only {n_days_used} days used", flush=True)
        return

    # Compute mean phase alignment per tidal line
    tidal_qa = {}
    for name, f_uhz in TIDAL_LINES.items():
        acc = tidal_accum[name]
        q_mean = acc["q_sum"] / acc["q_n"] if acc["q_n"] > 0 else None
        a_mean = acc["a_sum"] / acc["a_n"] if acc["a_n"] > 0 else None
        ratio = q_mean / a_mean if q_mean and a_mean and a_mean > 0 else None
        tidal_qa[name] = {
            "freq_uhz": f_uhz,
            "qa_ratio": ratio,
            "quartz_pa": q_mean,
            "atomic_pa": a_mean,
            "n_q": acc["q_n"],
            "n_a": acc["a_n"],
        }

    # Background
    bg_q_mean = bg_accum["q_sum"] / bg_accum["q_n"] if bg_accum["q_n"] > 0 else None
    bg_a_mean = bg_accum["a_sum"] / bg_accum["a_n"] if bg_accum["a_n"] > 0 else None
    bg_ratio = bg_q_mean / bg_a_mean if bg_q_mean and bg_a_mean and bg_a_mean > 0 else None

    # Bootstrap CI for background
    rng = np.random.default_rng(RNG_SEED)
    bg_q_vals = np.array(bg_accum["q_vals"])
    bg_a_vals = np.array(bg_accum["a_vals"])
    bg_boot = []
    if len(bg_q_vals) > 100 and len(bg_a_vals) > 100:
        for _ in range(2000):
            qb = float(np.mean(rng.choice(bg_q_vals, min(1000, len(bg_q_vals)), replace=True)))
            ab = float(np.mean(rng.choice(bg_a_vals, min(1000, len(bg_a_vals)), replace=True)))
            if ab > 0:
                bg_boot.append(qb / ab)
    bg_ci = [float(np.percentile(bg_boot, 2.5)), float(np.percentile(bg_boot, 97.5))] if bg_boot else None

    # Z-scores for tidal lines vs background
    bg_mean_ratio = float(np.mean(bg_boot)) if bg_boot else bg_ratio
    bg_std_ratio = float(np.std(bg_boot)) if bg_boot else 0.0

    for name in tidal_qa:
        r = tidal_qa[name]["qa_ratio"]
        if r is not None and bg_std_ratio > 0:
            tidal_qa[name]["z_score"] = (r - bg_mean_ratio) / bg_std_ratio

    # Lunar vs solar
    lunar_names = ["O1", "Q1", "N2", "M2", "M3"]
    solar_names = ["S1", "P1", "K1", "S2", "K2"]
    lunar_qa_vals = [tidal_qa[n]["qa_ratio"] for n in lunar_names if n in tidal_qa and tidal_qa[n]["qa_ratio"] is not None]
    solar_qa_vals = [tidal_qa[n]["qa_ratio"] for n in solar_names if n in tidal_qa and tidal_qa[n]["qa_ratio"] is not None]

    # Peak Q/A
    valid_ratios = [(n, tidal_qa[n]["qa_ratio"]) for n in tidal_qa if tidal_qa[n]["qa_ratio"] is not None]
    valid_ratios.sort(key=lambda x: x[1], reverse=True)
    peak_name, peak_ratio = valid_ratios[0] if valid_ratios else (None, None)

    interp = []
    interp.append(f"Per-frequency-bin clock-type differentiation: {n_days_used} days, 13-day window")
    interp.append(f"  Tidal lines analyzed at nearest FFT bin")
    interp.append(f"  Quartz pairs: {tidal_accum['M2']['q_n']}, Atomic pairs: {tidal_accum['M2']['a_n']} (per tidal line)")
    interp.append("")

    if peak_name:
        interp.append(f"  Peak Q/A ratio: {peak_ratio:.2f}x at {peak_name} ({TIDAL_LINES[peak_name]:.3f} uHz)")
    interp.append("")

    if bg_ratio is not None:
        interp.append(f"  Background (off-tide): Q/A={bg_ratio:.2f}x CI{bg_ci}")
        interp.append(f"  Background mean: {bg_mean_ratio:.2f}x, std: {bg_std_ratio:.2f}x")
    interp.append("")

    interp.append("  Tidal lines:")
    for name in ["O1", "Q1", "S1", "P1", "K1", "N2", "M2", "S2", "K2", "M3"]:
        if name in tidal_qa:
            t = tidal_qa[name]
            z_s = f"z={t.get('z_score', 0):.1f}" if "z_score" in t else ""
            tide_type = "LUNAR" if name in lunar_names else "SOLAR" if name in solar_names else "MIXED"
            q_s = f"{t['quartz_pa']:.4f}" if t['quartz_pa'] is not None else "N/A"
            a_s = f"{t['atomic_pa']:.4f}" if t['atomic_pa'] is not None else "N/A"
            r_s = f"{t['qa_ratio']:.2f}x" if t['qa_ratio'] is not None else "N/A"
            interp.append(f"    {name:4s} ({tide_type:5s}) {TIDAL_LINES[name]:7.3f} uHz: "
                         f"Q={q_s}, A={a_s}, Q/A={r_s} {z_s}")

    interp.append("")
    if lunar_qa_vals and solar_qa_vals:
        interp.append(f"  Lunar mean Q/A: {np.mean(lunar_qa_vals):.2f}x (n={len(lunar_qa_vals)})")
        interp.append(f"  Solar mean Q/A: {np.mean(solar_qa_vals):.2f}x (n={len(solar_qa_vals)})")
        if np.mean(lunar_qa_vals) > np.mean(solar_qa_vals):
            interp.append("  LUNAR tides show STRONGER clock-type differentiation than SOLAR tides.")
            interp.append("  This supports TEP: gravitational coupling dominates over atmospheric coupling.")
        else:
            interp.append("  SOLAR tides show STRONGER or equal clock-type differentiation than LUNAR tides.")
            interp.append("  This does NOT support TEP: atmospheric effects dominate.")

    # M2 vs S2 specifically
    if "M2" in tidal_qa and "S2" in tidal_qa:
        m2_r = tidal_qa["M2"]["qa_ratio"]
        s2_r = tidal_qa["S2"]["qa_ratio"]
        if m2_r is not None and s2_r is not None:
            interp.append("")
            interp.append(f"  M2 (lunar) Q/A:      {m2_r:.2f}x")
            interp.append(f"  S2 (solar) Q/A:      {s2_r:.2f}x")
            interp.append(f"  M2 - S2:             {m2_r - s2_r:+.2f}x")
            if m2_r > s2_r:
                interp.append("  M2 (purely gravitational) > S2 (has atmospheric component)")
                interp.append("  This is the TEP signature.")
            else:
                interp.append("  S2 (solar) >= M2 (lunar): atmospheric component dominates")

    verdict = {
        "step": 162,
        "title": "Per-frequency-bin clock-type differentiation spectrum",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT, "year": YEAR,
        "n_days": n_days_used,
        "window_days": 13,
        "peak_qa_ratio": float(peak_ratio) if peak_ratio else None,
        "peak_tidal_line": peak_name,
        "background": {"ratio": bg_ratio, "ci95": bg_ci, "mean": bg_mean_ratio, "std": bg_std_ratio},
        "tidal_lines": tidal_qa,
        "lunar_mean_qa": float(np.mean(lunar_qa_vals)) if lunar_qa_vals else None,
        "solar_mean_qa": float(np.mean(solar_qa_vals)) if solar_qa_vals else None,
        "interpretation": interp,
        "pre_registered_prediction": (
            "If TEP is real, the Q/A ratio should be higher at lunar tidal frequencies "
            "(M2, O1, N2) than at solar tidal frequencies (S2, S1, K1), because lunar "
            "tides are purely gravitational while solar tides have atmospheric components."
        ),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)
    for line in interp:
        print(line, flush=True)


if __name__ == "__main__":
    main()
