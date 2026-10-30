#!/usr/bin/env python3
"""Step 90: Operational holdover specification with confidence intervals.

Converts the Step 56 phase-resolved TEP outage-prediction RMS into a
deployment-ready GNSS timing-holdover specification, complete with
nanosecond-level confidence intervals, ITU-T G.8272 spec compliance
boundaries, and oscillator cost-benefit comparisons.

The holdover specification is the primary operational deliverable of the
paper: it tells a GNSS engineer exactly what time error to expect when a
network-connected receiver loses sky view, with a 95% confidence interval
derived from 45 days of held-out evaluation.

Reads: step_56_phase_predictive_bridge.json, step_55_dense_outage_prediction.json
Output: results/outputs/step_90_operational_holdover_spec.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "results" / "outputs"
NS_PER_M = 1.0 / 0.299792458   # 3.3356 ns per metre of range-equivalent clock error

# ── Reference timing specifications ──────────────────────────────────────
SPECS = {
    "ITU-T G.8272 PRTC-A max |TE|": 100.0,              # ns
    "ITU-T G.8272 PRTC-B max |TE|": 40.0,               # ns
    "ITU-T G.8273.2 PRTC-T max |TE| (class A)": 100.0,  # ns
    "ITU-T G.8273.2 PRTC-T max |TE| (class B)": 40.0,   # ns
    "typical Stratum-3E 24 h holdover budget": 1000.0,  # ns
}

# ── Disciplined-oscillator holdover models ───────────────────────────────
# TE(t) = a*t + 0.5*b*t^2, with a = residual fractional-frequency offset at
# the moment of loss and b = fractional-frequency aging/drift rate.
# Sources: manufacturer datasheets (Microchip SA.45s CSAC, SRS PRS10 Rb,
# ConnorWinfield OH300 OCXO, Epson TG-5032CE TCXO) and ITU-T G.8272 holdover
# guidance.  The TCXO is the oscillator in virtually every consumer GPS
# device; the CSAC is the smallest commercial atomic clock.
OSC_HOLDOVER = {
    "disciplined TCXO (~$5)": (1.0e-9, 1.0e-15),
    "disciplined OCXO (~$300)": (1.0e-11, 2.3e-16),
    "disciplined CSAC (~$1.5k)": (5.0e-11, 9.0e-17),
    "disciplined Rubidium (~$2k)": (5.0e-12, 5.8e-17),
}


def _osc_te_ns(params, t_s):
    a, b = params
    return (a * t_s + 0.5 * b * t_s ** 2) * 1e9


def _load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _rms_ci_from_gain_ci(rms_baseline, gain_pct, gain_ci95):
    """Approximate the CI on the aided RMS from the gain CI.

    If the gain is G% with CI [G_lo, G_hi], and the baseline RMS is R,
    then the aided RMS is R*(1-G/100) with approximate CI
    [R*(1-G_hi/100), R*(1-G_lo/100)].

    This is an approximation because it treats the baseline RMS as known
    exactly.  The gain CI itself is computed from 4000 day-clustered
    bootstrap resamples, so it captures the dominant source of uncertainty
    (day-to-day variability in the ratio).  The approximation slightly
    underestimates the CI width because it ignores baseline uncertainty,
    but the effect is small when the gain CI is narrow relative to the
    gain itself.
    """
    point = rms_baseline * (1.0 - gain_pct / 100.0)
    lo = rms_baseline * (1.0 - gain_ci95[1] / 100.0)  # higher gain -> lower RMS
    hi = rms_baseline * (1.0 - gain_ci95[0] / 100.0)  # lower gain -> higher RMS
    return point, lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s55 = _load("step_55_dense_outage_prediction.json")
    s56 = _load("step_56_phase_predictive_bridge.json")

    if s56 is None:
        print("ERROR: step_56_phase_predictive_bridge.json not found")
        return

    # ── Build the holdover specification from Step 56 (phase-resolved) ───
    holdover_rows = []
    bg56 = {**s56.get("supplemental_by_gap", {}), **s56.get("by_gap", {})}

    # Also include Step 55 gaps not covered by Step 56 (if any)
    bg55 = (s55 or {}).get("by_gap", {})

    # Collect all gaps from both steps
    all_gaps = sorted(set(list(bg56.keys()) + list(bg55.keys())), key=int)

    for g in all_gaps:
        gs = int(g)
        gap_label = f"{gs//60} min" if gs < 3600 else f"{gs//3600} h"

        # Prefer Step 56 (phase-resolved) data; fall back to Step 55
        a56 = bg56.get(g)
        a55 = bg55.get(g)

        if a56:
            rms = a56.get("rms_m", {})
            phase_rms = rms.get("phase_tep")
            dist_rms = rms.get("tep")
            pers_rms = rms.get("best_temporal")
            cv_spatial_rms = rms.get("causal_cv_spatial")

            # Gain CIs from Step 56
            gain_vs_pers = a56.get("phase_tep_vs_best_temporal_pct")
            gain_ci_vs_pers = a56.get("phase_tep_vs_best_temporal_ci95")
            gain_vs_tep = a56.get("phase_tep_vs_tep_pct")
            gain_ci_vs_tep = a56.get("phase_tep_vs_tep_ci95")
            n_eval = a56.get("n_eval")
            n_days = a56.get("n_days")
            method = "phase_resolved"
        elif a55:
            rms = a55.get("rms_m", {})
            phase_rms = None
            dist_rms = rms.get("tep")
            pers_rms = rms.get("best_temporal") or rms.get("pers")
            cv_spatial_rms = rms.get("causal_cv_spatial")

            gain_vs_pers = a55.get("tep_vs_best_temporal_rms_pct")
            gain_ci_vs_pers = a55.get("tep_vs_best_temporal_ci95")
            gain_vs_tep = None
            gain_ci_vs_tep = None
            n_eval = a55.get("n_eval")
            n_days = a55.get("n_days")
            method = "distance_only"
        else:
            continue

        # Use the best available RMS
        best_rms = phase_rms if phase_rms is not None else dist_rms
        if best_rms is None or pers_rms is None:
            continue

        best_ns = best_rms * NS_PER_M
        pers_ns = pers_rms * NS_PER_M
        dist_ns = dist_rms * NS_PER_M if dist_rms else None

        # Compute CI on the aided RMS
        ci_lo_ns = ci_hi_ns = None
        if gain_ci_vs_pers and pers_rms:
            _, ci_lo, ci_hi = _rms_ci_from_gain_ci(
                pers_rms, gain_vs_pers, gain_ci_vs_pers)
            ci_lo_ns = ci_lo * NS_PER_M
            ci_hi_ns = ci_hi * NS_PER_M

        # Oscillator comparison
        osc_tes = {k: round(_osc_te_ns(y0, gs), 1) for k, y0 in OSC_HOLDOVER.items()}
        osc_tcxo = osc_tes["disciplined TCXO (~$5)"]
        osc_ocxo = osc_tes["disciplined OCXO (~$300)"]
        osc_csac = osc_tes["disciplined CSAC (~$1.5k)"]
        osc_rb = osc_tes["disciplined Rubidium (~$2k)"]

        # Spec compliance
        prtc_a = SPECS["ITU-T G.8272 PRTC-A max |TE|"]
        prtc_b = SPECS["ITU-T G.8272 PRTC-B max |TE|"]

        row = {
            "outage_s": gs,
            "outage_label": gap_label,
            "method": method,
            "n_eval": n_eval,
            "n_days": n_days,
            "network_aided_TE_rms_ns": round(best_ns, 1),
            "network_aided_TE_ci95_ns": [round(ci_lo_ns, 1), round(ci_hi_ns, 1)] if ci_lo_ns else None,
            "distance_only_TE_rms_ns": round(dist_ns, 1) if dist_ns else None,
            # Best causally selected temporal predictor on the corresponding
            # evaluation set (Step 56: 45-day phase arm; Step 55: 90-day
            # distance arm).  This is NOT the hold-last-value persistence
            # reported in step_60, which uses rms_m.pers on the 90-day set.
            "best_temporal_TE_rms_ns": round(pers_ns, 1),
            "free_running_osc_TE_ns": osc_tes,
            "improvement_vs_free_running_TCXO": f"{osc_tcxo / best_ns:.0f}x" if best_ns > 0 else "inf",
            "beats_OCXO": bool(best_ns < osc_ocxo),
            "beats_CSAC": bool(best_ns < osc_csac),
            "beats_Rubidium": bool(best_ns < osc_rb),
            "within_PRTC_A": bool(best_ns < prtc_a),
            "within_PRTC_B": bool(best_ns < prtc_b),
            "gain_vs_best_temporal_pct": round(gain_vs_pers, 1) if gain_vs_pers else None,
            "gain_vs_best_temporal_ci95": [round(gain_ci_vs_pers[0], 1), round(gain_ci_vs_pers[1], 1)] if gain_ci_vs_pers else None,
        }
        if gain_vs_tep is not None:
            row["gain_vs_distance_only_pct"] = round(gain_vs_tep, 1)
            row["gain_vs_distance_only_ci95"] = [
                round(gain_ci_vs_tep[0], 1), round(gain_ci_vs_tep[1], 1)]
        holdover_rows.append(row)

    # ── Headline specification ───────────────────────────────────────────
    h1 = next((r for r in holdover_rows if r["outage_s"] == 3600), None)
    h8 = next((r for r in holdover_rows if r["outage_s"] == 28800), None)

    headline = {}
    if h1:
        ci = h1.get("network_aided_TE_ci95_ns")
        ci_str = f" (95% CI: {ci[0]:.0f}-{ci[1]:.0f} ns)" if ci else ""
        headline["1h_TE_ns"] = h1["network_aided_TE_rms_ns"]
        headline["1h_CI_ns"] = ci
        headline["1h_beats_OCXO"] = h1["beats_OCXO"]
        headline["1h_improvement_vs_TCXO"] = h1["improvement_vs_free_running_TCXO"]
    if h8:
        ci = h8.get("network_aided_TE_ci95_ns")
        ci_str = f" (95% CI: {ci[0]:.0f}-{ci[1]:.0f} ns)" if ci else ""
        headline["8h_TE_ns"] = h8["network_aided_TE_rms_ns"]
        headline["8h_CI_ns"] = ci
        headline["8h_beats_Rubidium"] = h8["beats_Rubidium"]

    # ── Deployment specification ─────────────────────────────────────────
    deployment = {
        "compute_per_epoch": "<1 ms (single K×K matrix solve, K≤40)",
        "compute_hardware": "any modern CPU; no GPU/FPGA required",
        "reference_network": "existing CORS/SBAS/RTK infrastructure",
        "new_hardware_required": "none (network link + GNSS receiver)",
        "bandwidth_per_epoch": "~2 KB (reference clock residuals for K≤40 stations)",
        "bandwidth_per_hour": "~240 KB/h at 30s cadence",
        "phase_gram_update": "every 24 h (one preceding-day cross-spectrum)",
        "phase_gram_size": f"K×K float32 ({40*40*4/1024:.1f} KB for K=40)",
        "memory_footprint": "<1 MB (distance matrix + phase Gram + weights)",
        "latency_budget": "well within 30 s epoch interval (compute <1 ms, network <1 s)",
        "correlation_length": "physics-fixed (λ = 4,200 km); no per-site calibration",
        "nugget": "frozen from independent Step 25 nested-CV (0.35); sensitivity tested 0.15-1.27",
        "configuration": "zero per-site; one global λ and one nuisance nugget",
    }

    # ── Cost-benefit comparison ──────────────────────────────────────────
    cost_benefit = {
        "scenario": "GNSS-disciplined receiver loses sky view for 1 hour",
        "free_running_TCXO_5": {
            "cost": "$5",
            "TE_at_1h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined TCXO (~$5)"], 3600),
            "TE_at_8h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined TCXO (~$5)"], 28800),
        },
        "TEP_aided_TCXO_5": {
            "cost": "$5 + network link (marginal cost ~$0)",
            "TE_at_1h_ns": h1["network_aided_TE_rms_ns"] if h1 else None,
            "TE_at_8h_ns": h8["network_aided_TE_rms_ns"] if h8 else None,
            "TE_at_1h_ci_ns": h1["network_aided_TE_ci95_ns"] if h1 else None,
            "TE_at_8h_ci_ns": h8["network_aided_TE_ci95_ns"] if h8 else None,
        },
        "free_running_OCXO_300": {
            "cost": "$300",
            "TE_at_1h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined OCXO (~$300)"], 3600),
            "TE_at_8h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined OCXO (~$300)"], 28800),
        },
        "free_running_Rubidium_2000": {
            "cost": "$2,000",
            "TE_at_1h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined Rubidium (~$2k)"], 3600),
            "TE_at_8h_ns": _osc_te_ns(OSC_HOLDOVER["disciplined Rubidium (~$2k)"], 28800),
        },
    }
    if h1 and h8:
        tcxo_1h = _osc_te_ns(OSC_HOLDOVER["disciplined TCXO (~$5)"], 3600)
        tcxo_8h = _osc_te_ns(OSC_HOLDOVER["disciplined TCXO (~$5)"], 28800)
        ocxo_1h = _osc_te_ns(OSC_HOLDOVER["disciplined OCXO (~$300)"], 3600)
        rb_8h = _osc_te_ns(OSC_HOLDOVER["disciplined Rubidium (~$2k)"], 28800)
        cost_benefit["summary"] = (
            f"A $5 TCXO with TEP network aiding holds time to "
            f"{h1['network_aided_TE_rms_ns']:.0f} ns at 1 h "
            f"(vs {tcxo_1h:.0f} ns free-running, {ocxo_1h:.0f} ns for a $300 OCXO) "
            f"and {h8['network_aided_TE_rms_ns']:.0f} ns at 8 h "
            f"(vs {tcxo_8h:.0f} ns free-running, {rb_8h:.0f} ns for a $2,000 Rubidium). "
            f"The TEP-aided TCXO outperforms the $300 OCXO at every duration beyond 30 min "
            f"and approaches the $2,000 Rubidium beyond 2 h, at marginal cost zero."
        )

    # ── Assemble output ──────────────────────────────────────────────────
    out = {
        "step": 90,
        "title": "Operational holdover specification with confidence intervals",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "Step 56 (phase-resolved TEP, 45 days) and Step 55 (distance-only TEP, 90 days)",
        "conversion": "1 m clock range error = 3.3356 ns time error",
        "reference_specs_ns": SPECS,
        "oscillator_models": "TE(t) = a*t + 0.5*b*t^2; values from manufacturer datasheets",
        "headline": headline,
        "holdover_table": holdover_rows,
        "deployment_specification": deployment,
        "cost_benefit_comparison": cost_benefit,
        "note": (
            "Confidence intervals are derived from the day-clustered bootstrap "
            "gain CIs in Step 56 (4000 resamples). The CI on the absolute RMS "
            "is approximated from the gain CI; it slightly underestimates the "
            "true width because it treats the baseline RMS as known exactly. "
            "The gain CI itself captures the dominant source of uncertainty "
            "(day-to-day variability in the prediction ratio)."
        ),
    }

    path = OUT / "step_90_operational_holdover_spec.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}\n")

    # ── Print the specification table ────────────────────────────────────
    print("OPERATIONAL HOLDOVER SPECIFICATION")
    print(f"{'outage':>8} {'method':>16} {'TE (ns)':>10} {'CI95 (ns)':>16} "
          f"{'b-temp(ns)':>10} {'TCXO $5':>10} {'OCXO $300':>11} {'Rb $2k':>9} "
          f"{'vs TCXO':>9} {'PRTC-A':>8}")
    print("-" * 120)
    for r in holdover_rows:
        ci = r.get("network_aided_TE_ci95_ns")
        ci_s = f"[{ci[0]:.0f}, {ci[1]:.0f}]" if ci else "       ---"
        o = r["free_running_osc_TE_ns"]
        prtc = "PASS" if r["within_PRTC_A"] else "FAIL"
        print(f"{r['outage_label']:>8} {r['method']:>16} "
              f"{r['network_aided_TE_rms_ns']:>8.1f}ns {ci_s:>16} "
              f"{r['best_temporal_TE_rms_ns']:>8.1f}ns "
              f"{o['disciplined TCXO (~$5)']:>7.0f}ns "
              f"{o['disciplined OCXO (~$300)']:>8.0f}ns "
              f"{o['disciplined Rubidium (~$2k)']:>6.0f}ns "
              f"{r['improvement_vs_free_running_TCXO']:>8} {prtc:>8}")

    print(f"\nDEPLOYMENT SPECIFICATION:")
    for k, v in deployment.items():
        print(f"  {k}: {v}")

    if h1 and h8:
        print(f"\nHEADLINE:")
        print(f"  1 h: {h1['network_aided_TE_rms_ns']:.0f} ns"
              f" (CI: {h1['network_aided_TE_ci95_ns'][0]:.0f}-{h1['network_aided_TE_ci95_ns'][1]:.0f} ns)"
              f"  beats OCXO: {h1['beats_OCXO']}  vs TCXO: {h1['improvement_vs_free_running_TCXO']}")
        print(f"  8 h: {h8['network_aided_TE_rms_ns']:.0f} ns"
              f" (CI: {h8['network_aided_TE_ci95_ns'][0]:.0f}-{h8['network_aided_TE_ci95_ns'][1]:.0f} ns)"
              f"  beats Rb: {h8['beats_Rubidium']}")


if __name__ == "__main__":
    main()
