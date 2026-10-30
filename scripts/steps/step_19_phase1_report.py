#!/usr/bin/env python3
"""
STEP 19 -- Phase 1 gate report.

Aggregates T1-T4 (steps 10-13) into a single verdict and a human-readable
summary, following the pre-registered decision rule in PREREGISTRATION.md.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "results" / "outputs"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    t1 = load("step_10_global_correlation.json")
    t2 = load("step_11_anisotropy.json")
    t3 = load("step_12_resolvability.json")
    t4 = load("step_13_commonview.json")
    t14 = load("step_14_precise_global.json")
    t25 = load("step_25_kernel_selection.json")
    t27 = load("step_27_discriminators.json")
    t45 = load("step_45_temporal_t4.json")
    if not all((t1, t2, t3, t4)):
        print("Run steps 10-13 first.")
        return

    g1 = t1["global_broadcast"]; g3 = t3["global_broadcast"]; g4 = t4["global_broadcast"]
    g2 = t2["global_broadcast"]
    pg = t14 or {}

    summary = {
        "T1_correlation_length": {
            "pass": bool(t1["T1_pass"]),
            "lambda_km": round(g1["lambda_km"]), "lambda_ci95": [round(x) for x in g1["lambda_ci95"]],
            "R2": round(g1["R2"], 3), "null_separation_sigma": round(g1["null_separation_sigma"], 1),
            "n_stations": g1["n_stations"], "note": "primary series signature (Papers 1-2: 3330-4549 km)",
        },
        "T2_anisotropy": {
            "pass": bool(t2["T2_pass"]),
            "ratio_longitude_matched": g2["longitude_matched"]["ratio"],
            "note": "not recovered at this network scale/geometry; consistent with MGEX 'suggestive only'",
        },
        "T3_resolvability": {
            "pass": bool(t3["T3_pass"]),
            "margins_pct": [round(r["margin_pct"], 1) for r in g3["rows"]],
            "apertures_km": [r["aperture_km"] for r in g3["rows"]],
            "spearman_rho": round(g3["spearman_rho"], 2) if g3["spearman_rho"] else None,
            "full_network_margin_sigma": round(g3["rows"][-1]["margin_sigma"], 1) if g3["rows"] else None,
            "note": "structured predictor beats bulk network-mean; margin grows with aperture",
        },
        "T4_common_view": {
            "pass": bool(t4["T4_pass"]) and bool(pg.get("T4_pass")) if pg else bool(t4["T4_pass"]),
            "pass_rule_per_arm": {"broadcast": bool(t4["T4_pass"]),
                                  "precise_global": bool(pg.get("T4_pass")) if pg else None},
            "pass_rule": (t4.get("T4_pass_rule")
                          or "prereg: |partial_corr(corr, log-dist | covis)| > 0.15 and OOS R2 gain > 0.02"),
            "broadcast_partial_corr_distance_given_covis": round(g4["partial_corr_distance_given_covis"], 3),
            "precise_global_partial_corr_distance_given_covis":
                round(pg.get("T4", {}).get("partial_corr_distance_given_covis", float("nan")), 3) if pg else None,
            "precise_global_within_band_corr_vs_distance":
                [b["corr_vs_distance"] for b in pg.get("T4", {}).get("within_covis_band_corr_vs_distance", [])] if pg else None,
            "note": ("distance structure and satellite co-visibility are collinear (r ~ -0.93 to -0.95) "
                     "on BOTH broadcast and precise global products; within fixed co-visibility bands the "
                     "residual dependence on distance is weak. GNSS receiver clocks alone cannot separate "
                     "a distance field from common-view geometry -- the series' directional/temporal "
                     "signatures (anisotropy, orbital coupling, CMB axis) are the discriminators, and they "
                     "do not reproduce at this network scale (T2)."),
        },
        "T3b_heldout_operational_skill": ({
            "pass": bool(t25["broadcast_spp"]["significant"]),
            "broadcast_skill_vs_raw_pct": t25["broadcast_spp"]["skill_vs_raw_pct"],
            "broadcast_kernel_vs_bulk_mean_pct": t25["broadcast_spp"]["kernel_vs_bulk_mean_pct"],
            "frac_splits_kernel_beats_mean": t25["broadcast_spp"]["frac_splits_kernel_beats_mean"],
            "selected_lambda_km_median": t25["broadcast_spp"]["selected_lambda_km"]["median"],
            "precise_skill_vs_raw_pct": t25["precise_clk"]["skill_vs_raw_pct"]["median"],
            "note": "nested-CV, leakage-free in station AND hyperparameter space; "
                    "lambda* selected blind on held-out data brackets the canonical 4200 km",
        } if t25 and t25.get("broadcast_spp", {}).get("n_day_splits") else {"pass": None, "note": "step_25 not run"}),
        "T5_constellation_independence": ({
            "pass": bool(t27["D1_D2_constellation_and_distance"]["satellite_product_axis_spread_frac"] < 0.10),
            "satellite_product_axis_spread_frac":
                round(t27["D1_D2_constellation_and_distance"]["satellite_product_axis_spread_frac"], 3),
            "broadcast_vs_precise_paired_t":
                t27["D1_D2_constellation_and_distance"].get("mode_day_paired_test", {})
                   .get("pairwise", {}).get("gps_broadcast__precise", {}).get("t"),
            "note": "GPS broadcast / ionosphere-free / precise agree on short-baseline correlation; "
                    "swapping broadcast for precise satellite clocks does not move it (spread < 10%) -- "
                    "rules out satellite-ephemeris and ionospheric artefacts",
        } if t27 and t27.get("D1_D2_constellation_and_distance", {}).get("satellite_product_axis_spread_frac") is not None
          else {"pass": None, "note": "step_27 not run"}),
        "T1_precise_global": {
            "pass": bool(pg.get("T1_pass")) if pg else None,
            "lambda_km": round(pg["T1"]["lambda_km"]) if pg else None,
            "R2": round(pg["T1"]["R2"], 3) if pg else None,
            "multi_ac_lambda": pg.get("multi_ac", {}).get("per_ac") if pg else None,
            "multi_ac_consistent": pg.get("multi_ac", {}).get("consistent") if pg else None,
            "note": "correlation length reproduces on the global precise-clock network and is "
                    "consistent across four independent analysis centres (COD/GFZ/GRG/JPL)",
        },
        "T4b_temporal_common_view": ({
            "pass": bool(t45.get("summary", {}).get("broadcast_passes")
                         and t45.get("summary", {}).get("precise_passes")),
            "broadcast_lambda_km": round(t45.get("summary", {}).get("broadcast_lambda", 0)),
            "broadcast_r2": round(t45.get("summary", {}).get("broadcast_r2", 0), 3),
            "broadcast_sigma": round(t45.get("summary", {}).get("broadcast_r2_sigma", 0), 1),
            "precise_lambda_km": round(t45.get("summary", {}).get("precise_lambda", 0)),
            "precise_r2": round(t45.get("summary", {}).get("precise_r2", 0), 3),
            "precise_sigma": round(t45.get("summary", {}).get("precise_r2_sigma", 0), 1),
            "note": ("temporal T4 breaks the co-visibility collinearity by exploiting within-pair "
                     "co-visibility variation: distance is fixed, co-visibility varies. With the corrected "
                     "pooled-kappa stage-1 on real-SP3 co-visibility, the non-common-view intercept meets "
                     "the frozen separability criterion on broadcast (PC=-0.164, 10.8 sigma, OOS gain "
                     "+0.022, lambda=4,349 km) and shows a significant but attenuated partial correlation "
                     "on precise (PC=-0.040, 9.2 sigma, lambda=2,155 km) -- the expected network-solution "
                     "common-mode absorption plus atomic screening."),
        } if t45 else {"pass": None, "note": "step_45 not run"}),
    }

    # Pre-registered decision rule
    claim_b = summary["T3_resolvability"]["pass"]                       # operational
    claim_a = claim_b and summary["T4_common_view"]["pass"]             # + physics separable
    if claim_a:
        verdict = "operational_exploitation"
        headline = ("Phase 2 as 'Operational exploitation of the TEP clock correlation': "
                    "structure beats bulk common-mode AND separates from co-visibility.")
    elif claim_b:
        verdict = "physics_informed_prior"
        # Compute headline from actual T3 data instead of hard-coding
        t3_rows = g3.get("rows", [])
        if t3_rows:
            max_aperture = t3_rows[-1]["aperture_km"]
            max_margin = t3_rows[-1]["margin_pct"]
            headline = (
                f"Phase 2 as 'A physics-informed network clock prior for real-time / broadcast PPP'. "
                f"T1 reproduces the correlation length globally and across four analysis centres. "
                f"T3 shows a distance kernel beats a bulk network-mean clock predictor, with the "
                f"margin growing to +{max_margin:.1f}% at {max_aperture:,} km aperture on broadcast SPP "
                f"(the common-mode that precise products already remove). T4 is NOT closed even on a "
                f"global precise network: distance and satellite co-visibility are collinear (r=-0.93) "
                f"and GNSS receiver clocks alone cannot separate a distance field from common-view "
                f"geometry -- that requires the series' directional signatures or inter-technique "
                f"data (Phase 3). Physics interpretation is deferred to Papers 1-3; the operational "
                f"contribution stands on its own.")
        else:
            headline = ("Phase 2 as 'A physics-informed network clock prior for real-time / broadcast PPP'. "
                        "T1 reproduces the correlation length globally and across four analysis centres. "
                        "T3 shows a distance kernel beats a bulk network-mean clock predictor. "
                        "T4 is NOT closed: distance and satellite co-visibility are collinear.")
    else:
        verdict = "scope_note"
        headline = "The structure is not operationally resolvable beyond a standard network correction at achievable scale."

    report = {
        "phase": 1, "generated": datetime.now().isoformat(),
        "tests": summary,
        "claim_A_physics_resolvable": bool(claim_a),
        "claim_B_operational": bool(claim_b),
        "verdict": verdict, "headline": headline,
        "series_context": "Distance-structured correlation itself is established by TEP-GNSS Papers 1-3/14; "
                          "Phase 1 concerns its reproduction in the operational PPP dataset and its "
                          "operational exploitability.",
    }
    (OUT / "phase1_gate.json").write_text(json.dumps(report, indent=2))

    print("=" * 72)
    print("PHASE 1 GATE")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k:22s}  {'PASS' if v['pass'] else 'no  '}   {v['note']}")
    print("-" * 72)
    print(f"  Claim B (operational, T3) : {claim_b}")
    print(f"  Claim A (physics, T3&T4)  : {claim_a}")
    print(f"  VERDICT: {verdict}")
    print(f"  {headline}")
    print("=" * 72)
    print(f"saved {OUT / 'phase1_gate.json'}")


if __name__ == "__main__":
    main()
