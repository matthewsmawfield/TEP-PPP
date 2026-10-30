#!/usr/bin/env python3
"""Fail the build when the manuscript outruns the evidence ledger.

This pins the current load-bearing values and, more importantly, rejects stale
claims that survived earlier pipeline bugs.  The generated manuscript is the
artifact checked because it is what readers see.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "outputs"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())



def _meta_lambda_pattern() -> str:
    """Require the manuscript to quote the CURRENT dependence-aware combined
    lambda, read from the JSON rather than hard-coded.  A literal here goes
    stale the moment the estimator is corrected, which is exactly how a
    superseded number survives in reader-facing text."""
    try:
        v = load("step_50_lambda_meta_analysis.json")["combined"]["median_km"]
    except Exception:
        return r"dependence.aware"
    n = int(round(v))
    return rf"{n // 1000}(?:,|\{{,\}}){n % 1000:03d}"


def main() -> int:
    manuscripts = sorted(ROOT.glob("33-TEP-PPP-*.md"))
    if not manuscripts:
        print("FAIL: no generated manuscript found")
        return 1
    text = manuscripts[-1].read_text()
    problems: list[str] = []

    s10 = load("step_10_global_correlation.json")["global_broadcast"]
    s38 = load("step_38_baseline_battery.json")["arms"]["broadcast_spp"]
    s39 = load("step_39_endtoend_position.json")
    s43 = load("step_43_free_gp_validation.json")
    s44 = load("step_44_raw_thin_position.json")
    s45 = load("step_45_temporal_t4.json")
    s48 = load("step_48_three_satellite_precise.json")
    s52 = load("step_52_drone_kinematic.json")
    s56 = load("step_56_phase_predictive_bridge.json")
    audit = load("evidence_audit.json")

    # Current values must be present in reader-facing text.
    required = {
        "broadcast T1 lambda": r"3(?:,|\{,\})480",
        "Step 38 total skill": r"52\.9\\?%",
        "Step 38 bulk margin": r"8\.7\\?%",
        "Step 38 ensemble skill": r"53\.5\\?%",
        "Step 38 ensemble bulk margin": r"10\.0\\?%",
        "Step 38 deployable near-tie": r"(?:−|-|\\?-)?0\.1\\?%",
        "free-GP scale": r"1(?:,|\{,\})500\s*km",
        "post-fit evidence label": r"post-fit",
        "raw-observation closure": r"raw[- ]observation|raw RINEX",
        "Step 43 resolving-power validation status": r"free_gp_is_validated_scale_discovery = true|resolving[- ]power|88\\?%",
        "dependence-aware lambda meta-analysis": _meta_lambda_pattern(),
        "Step 44 raw-observation P2": r"[Ss]tep\s*44|raw.observation.(?:position|P2|thin)",
        "Step 45 temporal T4": r"[Ss]tep\s*45|temporal T4|T4 temporal|3.?890|4.?179",
        "Step 56 phase prediction": r"[Ss]tep\s*56|phase.resolved TEP",
    }
    for label, pattern in required.items():
        if not re.search(pattern, text, re.IGNORECASE):
            problems.append(f"missing {label}")

    # JSON sanity and provenance boundaries.
    # The corrected Step 38 pipeline (Gaussian TEP form, refined lambda/nugget/K
    # grids) moved the TEP-vs-training-selected deficit from -0.6% to ~-0.1%,
    # and the CI upper bound is now 0.0 (the CI includes zero: a statistical
    # tie, not a barely-significant deficit).
    metric = s38["tep_vs_best_conventional_pct"]
    if not (-0.15 < metric["median"] < 0.0 and metric["ci95"][1] <= 0.0):
        problems.append(f"unexpected Step 38 selector metric: {metric}")
    # The TEP ensemble must be the best single predictor vs bulk mean.
    ens = s38.get("tep_ensemble_vs_best_conventional_pct")
    if ens and not (ens["median"] >= -0.01 and ens["ci95"][0] >= -0.01):
        problems.append(f"unexpected Step 38 ensemble metric: {ens}")
    if not (abs(s10["lambda_km"] - 3480) < 2 and s10["R2"] > 0.9):
        problems.append(f"unexpected Step 10 result: {s10}")
    # The quoted T1 model-selection margin must equal the committed model
    # comparison -- a stale "15.6" literal once claimed ~100x the real
    # amplitude-domain margin.
    daic = (s10.get("model_comparison") or {}).get(
        "delta_aic_exponential_vs_best")
    if daic is not None and f"{daic:.2f}" not in text:
        problems.append(
            f"Step 10 delta-AIC {daic:.2f} not quoted in manuscript")
    # The Rubidium crossover in the holdover claim is read from the Step 90
    # table itself: the earliest gap at which the aided TE beats the
    # disciplined-Rubidium model and keeps beating it at every longer gap.
    # A stale "~2 h" literal survived the Step 55/56 regeneration (the
    # distance-only 2 h arm loses to the Rb model), so the boundary is
    # pinned here rather than trusted.
    try:
        s90 = load("step_90_operational_holdover_spec.json")
        rows = sorted(s90["holdover_table"], key=lambda r: r["outage_s"])
        rb_gap = next(
            (r["outage_s"] for r in rows
             if all(q["beats_Rubidium"] for q in rows
                    if q["outage_s"] >= r["outage_s"])),
            None)
        if rb_gap is not None:
            rb_phrase = (f"{rb_gap / 3600:g} h" if rb_gap >= 3600
                         else f"{rb_gap / 60:g} min")
            if rb_gap >= 3600:
                gap_pat = rb_phrase.replace(" ", r"\s*")
                expected = rf"beyond \$(?:\\sim)?\${gap_pat}"
                if not re.search(expected, text):
                    problems.append(
                        f"manuscript must give the Rubidium crossover as "
                        f"beyond ~{rb_phrase} (from step_90 beats_Rubidium)")
            for m in re.finditer(
                    r"Rubidium[^\n]{0,60}?beyond\s+\$?\\?s?i?m?\$?\s*(\d+)\s*h",
                    text):
                if float(m.group(1)) * 3600 != rb_gap:
                    problems.append(
                        f"stale Rubidium crossover 'beyond {m.group(1)} h' "
                        f"contradicts step_90 (true boundary {rb_phrase})")
    except (KeyError, FileNotFoundError):
        problems.append("step_90 holdover spec missing for Rubidium pin")
    # P4 blind-replication disclosure: DLF1 sits in the DOY-100 broadcast
    # training pool, so the manuscript may not call the capture ten unseen
    # stations and must disclose the pool membership.
    if re.search(
            r"ten\s+(?:previously\s+)?unseen|10\s+unseen\s+(?:live\s+)?stations"
            r"|ten\s+live\s+stations\s+that\s+had\s+never",
            text, re.I):
        problems.append(
            "P4 must not claim ten unseen stations: DLF1 is in the "
            "DOY-100 broadcast training pool")
    if not re.search(r"DLF1[^\n]{0,140}training[- ]pool", text, re.I):
        problems.append(
            "manuscript must disclose DLF1's training-pool membership")
    # The canonical k=3 blind-solve figures quoted in the text come from the
    # dense-arm JSON (step_48_three_satellite.json); pin them so they cannot
    # drift to the precise-arm values or a stale recalibration.
    try:
        s48d = load("step_48_three_satellite.json")
        can = s48d["by_k_window"]["k3_L3600"]["tep_blind_integrity"]
        for label, val in (
            ("canonical k=3 RMS", can["rms_3d_m"]),
            ("canonical k=3 p50", can["p50_m"]),
            ("canonical k=3 p99", can["p99_m"]),
        ):
            if f"{val:.2f}" not in text:
                problems.append(
                    f"Step 48 {label} {val:.2f} m not in manuscript")
        if f"{can['availability'] * 100:.1f}" not in text:
            problems.append(
                "Step 48 canonical k=3 availability "
                f"{can['availability'] * 100:.1f}% not in manuscript")
    except (KeyError, FileNotFoundError):
        problems.append("step_48 dense-arm output missing for canonical pin")
    # Step 39 must remain a post-fit diagnostic (not raw-observation)
    if s39.get("strict_p2_pass") is not False or not s39.get("uses_full_sky_linearization"):
        problems.append("Step 39 must remain diagnostic; raw-observation P2 is step_44")
    # Step 44 is the raw-observation P2 result
    if s44.get("evidence_level") != "raw_observation":
        problems.append("Step 44 must be raw_observation evidence level")
    if s44.get("uses_full_sky_linearization") is not False:
        problems.append("Step 44 must not use full-sky linearization")
    # Step 48 must be judged on the whole sample.  ``excl_steering`` uses a
    # diagnostic flag derived from the target's full-sky truth-linearised
    # clock; the archived RINEX files do not consistently expose causal
    # per-event steering telemetry.  It cannot be the operational headline.
    s48_k3 = s48.get("by_k_window", {}).get("k3_L3600", {})
    s48_cal = s48_k3.get("tep_calibrated", {})
    s48_avail = s48_cal.get("availability", 0)
    s48_p50 = s48_cal.get("p50_m", 0)
    s48_p99 = s48_cal.get("p99_m", 0)
    s48_rms = s48_cal.get("rms_3d_m", 0)
    if not (0 < s48_avail <= 1):
        problems.append(f"invalid Step 48 k=3 availability: {s48_avail}")
    if not (s48_p50 > 0 and s48_p99 >= s48_p50 and s48_rms > 0):
        problems.append("invalid Step 48 k=3 whole-sample metrics")
    # Only the genuinely blind nonlinear arm can support strict P2. RAIM or
    # multistart labels must not promote a truth-linearised fallback.
    s48_summary = s48.get("summary", {})
    s48_blind_method = s48_summary.get("k3_blind_tep_method", "")
    s48_blind_gain = s48_summary.get("k3_blind_tep_vs_best_conventional_rms_pct")
    s48_paired_gain = s48_summary.get("k3_blind_paired_tep_vs_best_conventional_rms_pct")
    if s48.get("strict_p2_pass") is True:
        # Blind solve passes: verify the manuscript reports the blind result
        if not re.search(r"blind.{0,40}(?:nonlinear|navigation|solve|converg)", text, re.I):
            problems.append("manuscript must report the blind nonlinear solve result when strict_p2_pass is True")
        if s48_blind_method != "tep_blind_integrity":
            problems.append(f"strict P2 used non-flagship method {s48_blind_method!r}")
        if s48_blind_gain is None or s48_blind_gain < 5:
            problems.append(f"strict_p2_pass is True but blind gain is {s48_blind_gain}")
        if s48_paired_gain is None or s48_paired_gain < 5:
            problems.append(f"strict_p2_pass is True but paired blind gain is {s48_paired_gain}")
    elif s48.get("strict_p2_pass") is not False:
        problems.append("Step 48 strict_p2_pass must be True or False")
    if not re.search(r"whole[- ]sample.{0,180}(?:tail|RMS|p99)", text, re.I | re.S):
        problems.append("manuscript must report the Step 48 whole-sample tail")
    if not re.search(r"steering[- ]event telemetry|per[- ]event steering", text, re.I):
        problems.append("manuscript must state the missing steering-event telemetry boundary")
    # Stale Step 48 literals must not survive in reader-facing text.
    stale_48 = {
        "Step 48 stale 99.2% availability": r"99\.2\\?%\s*availability",
        "Step 48 stale p50 5.6 m": r"p50\s*=\s*5\.6\s*m",
        "Step 48 stale p95 63.1 m": r"p95\s*=\s*63\.1\s*m",
        "Step 48 stale p99 1,249 m": r"p99\s*=\s*1,249\s*m",
        "Step 48 stale 13.6-21.8 km RMS": r"13\.6.{0,8}21\.8\s*km",
        "Step 48 stale 26% >10m": r"26\\?%\s*of\s*epochs\s*exceed\s*10\s*m",
        "Step 48 stale 348 km p99": r"348[,29]*\s*m",
        "Step 48 stale 44.2 km RMS": r"44\.2\s*km",
        "Step 48 stale 39.9 km RMS": r"39\.9\s*km",
        "Step 48 stale 87% availability": r"87\.0\?%\s*availability",
        "Step 48 stale 92.3% availability": r"92\.3\?%\s*availability",
        "Step 48 stale 5.86 m RMS": r"5\.86\s*m",
        "Step 48 stale 8.28 m RMS": r"8\.28\s*m",
    }
    for label, pattern in stale_48.items():
        if re.search(pattern, text, re.IGNORECASE):
            problems.append(label)
    # The integrity gate cost is dynamically checked against the canonical
    # dense-arm JSON (the referent of the manuscript's claim) on the blind
    # arms (tep_blind -> tep_blind_integrity): the manuscript claims "0.0
    # percentage points availability cost", which holds only if the
    # recalibrated gate rejects zero epochs on that network.  The precise-arm
    # run (a different network/product) is not the claim's referent.
    ig_costs = []
    try:
        _s48_dense = load("step_48_three_satellite.json")
    except FileNotFoundError:
        _s48_dense = s48
    for win in ("k3_L600", "k3_L3600"):
        kw = _s48_dense.get("by_k_window", {}).get(win, {})
        for ungated, gated in (("tep_blind", "tep_blind_integrity"),
                               ("tep_blind_phase", "tep_blind_phase_integrity")):
            if ungated in kw and gated in kw:
                ig_costs.append(
                    (kw[ungated].get("availability", 0)
                     - kw[gated].get("availability", 0)) * 100)
    if ig_costs and max(ig_costs) > 0.05:
        problems.append(
            f"integrity gate costs {max(ig_costs):.2f} pp on the canonical "
            "dense-arm blind arms; the '0.0 pp' claim must be re-checked")
    # The manuscript must mention the integrity gate and its cost.
    if not re.search(r"integrity\s*gate", text, re.I):
        problems.append("manuscript must mention the integrity gate")
    # Stale 8.98 pp / 1.64% must not survive.
    if re.search(r"8\.98\s*(?:pp|percentage|p\.?p\.?)", text, re.I):
        problems.append("stale integrity gate cost 8.98 pp must be updated")
    if re.search(r"500.{0,30}1.?200\s*km.{0,80}aperture sufficient", text, re.I | re.S):
        problems.append("five-station aperture must not claim to resolve lambda_T")
    if not isinstance(audit.get("strict_operational_claim_pass"), bool):
        problems.append("evidence audit strict_operational_claim_pass must be a boolean")

    # Step 52 -- genuine public-UAV evaluation (v3: blind, GPS-only, causal
    # fixed TEP).  Numerical k=3 feasibility must not be promoted as useful
    # navigation, and the earlier post-hoc screening story is not flight
    # evidence.
    # The old "100% TEP bridge rate at k=3 (3/3 epochs)" claim is withdrawn.
    if re.search(r"100\\?%\s*TEP\s*bridge\s*rate|3/3\s*epochs\s*bridged", text, re.I):
        problems.append("Step 52 stale '100% bridge rate' or '3/3 epochs' claim must be removed")
    s52_ds = (s52.get("datasets") or [{}])
    s52_ok = [d for d in s52_ds if d.get("status") == "ok"]
    if s52_ok:
        for ds in s52_ok:
            ds_name = ds.get("name", "?")
            s52s = ds.get("summary", {})
            if not s52s.get("version", "").startswith("v3"):
                problems.append(f"Step 52 {ds_name} must use v3 causal fixed TEP evaluation")
            if not ds.get("doi") or not ds.get("provenance_status", "").startswith("verified"):
                problems.append(f"Step 52 {ds_name} lacks verified public provenance")
            k3 = s52s.get("per_k", {}).get("k3", {})
            if k3.get("n_free_valid") != 0 or k3.get("n_tep_valid", 0) <= 0:
                problems.append(f"Step 52 {ds_name} no longer demonstrates the k=3 feasibility boundary")
            if k3.get("err_tep_p50", 0) <= 10:
                problems.append(f"Step 52 {ds_name} unexpectedly changed operational outcome; audit before promotion")
            if s52s.get("inference_unit", "").split(";")[0] != "one flight":
                problems.append(f"Step 52 {ds_name} must identify one flight as the inference unit")
        # Manuscript must disclose the v3 approach and failed operational result.
        if not re.search(r"simulated\s+jamming|jamming\s+window", text, re.I):
            problems.append("manuscript must mention the Step 52 simulated jamming windows")
        if not re.search(r"blind.{0,40}(?:nonlinear|solver)", text, re.I):
            problems.append("manuscript must mention the Step 52 blind nonlinear solver")
        uav_p50 = int(round(s52_ok[0]["summary"]["per_k"]["k3"]["err_tep_p50"]))
        if not re.search(rf"{uav_p50}\s*m.{{0,100}}(?:not operational|not a usable|receiver.only)", text, re.I | re.S):
            problems.append("manuscript must disclose the unusable Step 52 UAV position error")
        if re.search(r"WUT\s+drone|97\.5\\?%\s+of\s+(?:the\s+)?drone", text, re.I):
            problems.append("manuscript contains withdrawn WUT-drone or unverified UAV spectral claim")

    # Step 70 -- clock-type-aware TEP outage test on quartz CORS.
    # Must report the screening-scaled result honestly: per-station gains
    # but aggregate overcorrection.  Must NOT claim aggregate improvement
    # from the 3.78x screening factor.
    s70_path = ROOT / "results" / "outputs" / "step_70_quartz_outage_clock.json"
    if s70_path.exists():
        s70 = json.load(open(s70_path))
        agg = s70.get("aggregate", {})
        # Manuscript must mention Step 70
        if not re.search(r"Step\s*70|step_70|clock.type.aware", text, re.I):
            problems.append("manuscript must mention the Step 70 clock-type-aware test")
        # Must mention the screening factor
        if not re.search(r"3\.78|3\.8\\?x.*screening|screening.*factor", text, re.I):
            problems.append("manuscript must mention the Step 65/70 screening factor")
        # Must NOT claim aggregate improvement from screening_scaled
        if re.search(r"screening.scaled.{0,200}(?:improves|better than|reduces).{0,50}(?:aggregate|overall|RMS)", text, re.I):
            problems.append("Step 70: must not claim aggregate RMS improvement from screening-scaled mode")
        # Verify the standard TEP prior improves over temporal at long outages
        a28800 = agg.get("28800", {})
        t_rms = a28800.get("temporal", {}).get("median_m")
        n_rms = a28800.get("none", {}).get("median_m")
        if t_rms and n_rms and n_rms > t_rms * 1.5:
            problems.append("Step 70: TEP(none) should not be much worse than temporal at 8 h")

    # Step 56 must remain genuinely prior-day and must beat its strongest
    # controls with day-clustered intervals at every registered horizon.
    s56g = s56.get("by_gap", {})
    if len(s56g) != 4 or any(a.get("n_days", 0) < 40 for a in s56g.values()):
        problems.append("Step 56 must contain the frozen 45-day phase benchmark")
    for gap, a in s56g.items():
        for key in ("phase_tep_vs_tep_ci95",
                    "phase_tep_vs_causal_cv_spatial_ci95",
                    "phase_tep_vs_phase_scramble_ci95",
                    "phase_tep_vs_phase_angle_scramble_ci95"):
            if not a.get(key) or a[key][0] <= 0:
                problems.append(f"Step 56 {gap}s does not pass {key}")

    # Pin the Step 56 headline ranges: the manuscript-quoted min-max spread for
    # each comparator must be derived from the committed JSON, not a stale
    # literal left over from an earlier run.  Format used in the text is
    # "lo&ndash;hi" (1 decimal) or "lo-hi" in generated markdown.
    def _rng56(metric, ndp=1):
        vals = [a.get(metric) for a in s56g.values() if a.get(metric) is not None]
        if len(vals) < 2:
            return None
        return f"{min(vals):.{ndp}f}", f"{max(vals):.{ndp}f}"

    for label, metric in (
        ("phase-TEP vs distance-only TEP", "phase_tep_vs_tep_pct"),
        ("phase-TEP vs temporal", "phase_tep_vs_best_temporal_pct"),
        ("phase-TEP vs phase scramble", "phase_tep_vs_phase_scramble_pct"),
        ("phase-TEP vs causal CV spatial", "phase_tep_vs_causal_cv_spatial_pct"),
    ):
        r = _rng56(metric)
        if r is None:
            continue
        lo, hi = r
        if not re.search(rf"{re.escape(lo)}\D{{0,4}}{re.escape(hi)}\\?%", text):
            problems.append(
                f"Step 56 {label} range {lo}-{hi} not quoted in manuscript "
                "(stale range suspected)")

    # Per-gap table cells: the four horizon RMS values must appear verbatim.
    for gap, a in s56g.items():
        rms = (a.get("rms_m") or {}).get("phase_tep")
        if rms is None:
            continue
        if f"{rms:.2f}" not in text:
            problems.append(
                f"Step 56 {gap}s phase-TEP RMS {rms:.2f} m not in manuscript")

    # Step 43 is a resolving-power validation: with the injection amplitude set
    # to dominate the actual per-station residual variance (not the earlier
    # mis-calibrated fixed 0.3), the free GP must recover the full pre-specified
    # TEP band, the >1,000 km baseline restriction must move into the band, and
    # both spatial-signal nulls must collapse to the grid edge.
    s43_summary = s43.get("summary", {})
    if not s43_summary.get("free_gp_is_validated_scale_discovery"):
        problems.append("Step 43 free GP must validate the TEP scale under a noise-dominating injection")
    if s43_summary.get("injection_validates_full_tep_band") is not True:
        problems.append("Step 43 full-band injection recovery must pass (>=0.5 rate, >=4/5 scales)")
    if s43_summary.get("injection_recovery_rate_tep_band_within_2x", 0) < 0.6:
        problems.append("Step 43 full-band recovery-within-2x must be >= 60%")
    if not s43_summary.get("nulls_fail_tep_band"):
        problems.append("Step 43 null controls must fail to select a TEP-band scale")
    b1000 = s43.get("baseline_restricted", {}).get("gt_1000km", {})
    if not b1000 or b1000.get("lambda_median_km", 0) < 1000:
        problems.append("Step 43 baseline-restricted (>1000 km) free GP must select a TEP-band scale")

    # Step 45 temporal T4.  The PRECISE arm is the load-bearing one and must
    # pass with a TEP-band lambda.  The BROADCAST arm is deliberately NOT
    # required: the Step 45b forensic battery shows it is fragile (every
    # alternative co-visibility definition drives lambda to 10,000-24,000 km
    # and the long-baseline restriction collapses to 1.3 sigma), consistent
    # with broadcast satellite common-mode being itself co-visibility-
    # structured.  Requiring it here would force the build to defend a claim
    # the forensic controls withdrew.
    s45_summary = s45.get("summary", {})
    # Corrected stage-1 (pooled kappa on real-SP3 co-visibility, 2026-10-29):
    # the broadcast arm must meet the frozen S2 criterion and the precise arm
    # must show a significant (attenuated) partial correlation -- if either
    # regresses, the manuscript's S2 text is stale.
    b_lam = s45_summary.get("broadcast_lambda", 0)
    if not (1000 <= b_lam <= 6000):
        problems.append(f"Step 45 broadcast lambda must be in TEP band, got {b_lam}")
    # The separated distance fit (alpha residualised on [Jbar, cos dlon],
    # identically residualised epoch-shuffle null) is the scored distance
    # fit: it removes the kappa_null*Jbar(d) leakage that heavy-tailed the
    # uncontrolled null, and must stay decisive on broadcast and
    # sub-threshold on the attenuated precise arm.
    b_sep_lam = s45_summary.get("broadcast_sep_lambda", 0)
    if not (500 <= b_sep_lam <= 20000):
        problems.append(f"Step 45 broadcast separated lambda must be in band, got {b_sep_lam}")
    if not (s45_summary.get("broadcast_sep_r2_sigma", 0) > 3.0):
        problems.append("Step 45 broadcast separated R2 sigma must exceed 3")
    if not s45_summary.get("broadcast_passes"):
        problems.append("Step 45 broadcast T4 composite must pass with the separated fit")
    if s45_summary.get("precise_passes"):
        problems.append("Step 45 precise T4 composite must stay attenuated (S2 PARTIAL)")
    if not s45_summary.get("broadcast_frozen_S2"):
        problems.append("Step 45 broadcast arm no longer meets frozen S2 -- "
                        "re-check the manuscript's T4 text")
    if abs(s45_summary.get("precise_pc", 0)) < 0.02:
        problems.append("Step 45 precise partial correlation collapsed -- "
                        "re-check the corrected stage-1")
    if s45_summary.get("precise_frozen_S2"):
        problems.append("Step 45 precise arm now meets frozen S2 -- "
                        "update the S2 PARTIAL framing")
    # the manuscript must keep the honest S2 disclosure somewhere
    if not re.search(r"S2[^\n]{0,40}PARTIAL|attenuat", text, re.I):
        problems.append("manuscript no longer discloses the precise-arm "
                        "attenuation (S2 PARTIAL)")

    # Step 45b forensic battery under the corrected stage-1 (2026-10-29):
    # the separation must be robust across specs on broadcast (PC < -0.1),
    # and the collinearity-broken matched-low-covis spec must return a
    # finite lambda rather than railing to the grid edge as the old
    # nominal-covis run did.
    try:
        s45b_all = load("step_45b_t4_forensic.json")["arms"]
        bsb = s45b_all["broadcast_global"]["spec_battery"]
        psb = s45b_all["precise_cod"]["spec_battery"]
        if not (bsb["baseline"]["partial_corr_alpha_logd_given_controls"] < -0.1):
            problems.append("Step 45b broadcast baseline PC must be < -0.1")
        if not (bsb["pooled_distbin"]["partial_corr_alpha_logd_given_controls"] < -0.3):
            problems.append("Step 45b broadcast distance-binned-kappa PC must be < -0.3")
        mlc = bsb.get("matched_lowcovis", {})
        if not (1000 <= (mlc.get("lambda_km") or 0) <= 12000):
            problems.append("Step 45b broadcast matched_lowcovis lambda must be "
                            "finite in-band (rails only under the old nominal covis)")
        # precise arm must be attenuated-but-negative in the baseline spec
        if not (-0.15 < psb["baseline"]["partial_corr_alpha_logd_given_controls"] < 0):
            problems.append("Step 45b precise baseline PC must be small-negative "
                            "(attenuated, not positive)")
    except (KeyError, FileNotFoundError):
        problems.append("Step 45b forensic battery output missing")

    # Step 83 -- full-year phase-cos T4 with the corrected stage-2 null
    # (2026-10-29): the day-permutation was degenerate (alpha = pbar -
    # kappa*Jbar is permutation-invariant in pbar and kappa*Jbar is absorbed
    # by the Jbar control), so z_vs_null is now a pair-label permutation and
    # the day-permutation is reported as a kappa-null.  Pin the corrected
    # statistics so a regression to the artefact (or a re-broken null) fails
    # the build.
    try:
        s83 = load("step_83_t4_phasecos.json")["arms"]
        p83 = s83["precise_ops"]; b83 = s83["broadcast_spp"]
        if not (-0.10 < p83["partial_corr_alpha_logdist_given_covis"] < 0.0):
            problems.append("Step 83 precise PC must be small-negative (attenuated)")
        if not (p83["z_vs_null"] > 2.0):
            problems.append("Step 83 precise pair-permutation z must exceed 2 "
                            "(the old degenerate z~1.8 was an artefact)")
        if not (p83["z_kappa"] > 3.0):
            problems.append("Step 83 precise kappa-null z must be large")
        ci83 = p83.get("pc_station_boot_ci95")
        if not (ci83 and ci83[1] < 0.0):
            problems.append("Step 83 precise cluster-bootstrap CI must exclude zero")
        if not (abs(b83["z_vs_null"]) < 2.0):
            problems.append("Step 83 broadcast sparse arm must stay null")
        if not (2000.0 <= p83["distance_fit"]["lambda_km"] <= 8000.0):
            problems.append("Step 83 precise distance fit must stay in/near TEP band")
    except (KeyError, FileNotFoundError):
        problems.append("Step 83 phase-cos T4 output missing corrected nulls")

    # Step 98 -- satellite-channel flagship (seven arms).  Every arm must be
    # directionally positive vs its nulls; the GPSxGalileo cross arm carries
    # the resolved decay scale, and the GLONASSxothers cross arm must be
    # positive (the intra-GLONASS null is noise-floor dilution, not absence).
    try:
        s98 = load("step_98_satellite_channel_flagship.json")
        a98 = s98["arms"]
        for k in ("gps_only", "galileo_only", "glonass_only", "beidou_only"):
            if not (a98[k]["contrast"] > 0):
                problems.append(f"Step 98 {k} contrast must be positive")
        xg = a98["cross_gps_galileo"]
        if not (4000.0 <= (xg.get("lambda_km") or 0) <= 15000.0):
            problems.append("Step 98 cross GPSxGalileo lambda must be resolved")
        cg = a98.get("cross_glonass", {})
        if not (cg.get("contrast", 0) > 0):
            problems.append("Step 98 cross-GLONASS contrast must be positive")
        brms = s98.get("band_rms_median_per_constellation", {})
        if not (brms.get("R", 0) > 2.0 * max(brms.get("G", 1), brms.get("E", 1),
                                           brms.get("C", 1))):
            problems.append("Step 98 GLONASS band RMS must exceed ~2x the others")
    except (KeyError, FileNotFoundError):
        problems.append("Step 98 satellite-channel output missing arms")

    # Step 53 -- the Papers 1-2 phase-alignment index applied correctly.  This
    # is now the headline physics result: the earlier CSD steps (11b/36) used
    # band-averaged coherency (which Paper 1 reports as failing) times a phase-
    # concentration magnitude, instead of cos(arg(sum_f S_ij(f))).  Pin the
    # corrected precise-arm result so a regression to the wrong statistic, or a
    # drift off the Papers 1-2 scale, fails the build.
    try:
        s53 = load("step_53_coherent_lambda.json")["arms"]["precise_cod"]
        lam53 = s53.get("lambda_km")
        r2_53 = s53.get("R2")
        pr_sig = (s53.get("null_phase_randomise") or {}).get("sigma")
        if not (s53.get("best_model") == "gaussian" and lam53 and
                1000.0 <= lam53 <= 5000.0):
            problems.append(f"Step 53 precise arm must be a TEP-band Gaussian, got "
                            f"{s53.get('best_model')} lambda={lam53}")
        if not (r2_53 and r2_53 > 0.8):
            problems.append(f"Step 53 precise arm R^2 must exceed 0.8, got {r2_53}")
        if not (pr_sig and pr_sig > 8.0):
            problems.append(f"Step 53 precise phase-randomise sigma must exceed 8, got {pr_sig}")
        # the manuscript must name the corrected estimator and the superseded one
        if not re.search(r"cos\(\\?arg", text):
            problems.append("manuscript must state the corrected Step 53 estimator cos(arg ...)")
        if not re.search(r"band[- ]averaged coh", text, re.I):
            problems.append("manuscript must identify the superseded band-averaged-coherency statistic")
        # and must not still present the old 12.6 sigma MEO or the old T2 as final.
        # The guard is dynamic: the superseded figure happened to be "12.6
        # sigma", but the CURRENT committed pair-shuffle sigma may itself
        # round to 12.6 -- a literal ban would then flag the correct number.
        # Flag only when no committed Step 36 GPS-null sigma rounds to the
        # quoted value and the text lacks the superseded-context qualifier.
        try:
            _s36z = load("step_36_meo_csd.json")["arms"]["gps_only"]["nulls"]
            _cur_126 = any(
                round((_s36z[n] or {}).get("z_contrast", 0.0), 1) == 12.6
                for n in ("phase_randomise", "pair_shuffle"))
        except Exception:
            _cur_126 = False
        if (re.search(r"12\.6\s*\\?sigma", text) and not _cur_126
                and not re.search(
                    r"(?:superseded|earlier|previous|band-averaged).{0,80}12\.6",
                    text, re.I | re.S)):
            problems.append("manuscript still presents the pre-correction 12.6 sigma MEO result")
    except (KeyError, FileNotFoundError):
        problems.append("Step 53 phase-alignment output missing")

    # Step 54 -- operational skill scales with reference-station density.  The
    # physics-fixed TEP kernel must beat a flat regional mean on the dense
    # EUREF network (common mode removed from both), the distance-scramble null
    # must be ~0, and the sparse global network must NOT show the same gain --
    # otherwise the "density regime" framing in 3.5c / discussion is unsupported.
    try:
        s54 = load("step_54_dense_operational_skill.json")["headline"]
        dk = s54["dense"]["kernel_vs_mean_pct"]
        dscr = s54["dense"]["kernel_vs_mean_pct_scramble"]
        gk = s54["global"]["kernel_vs_mean_pct"]
        if not (dk > 2.0):
            problems.append(f"Step 54 dense kernel-vs-mean must exceed +2%, got {dk:.2f}")
        if not (abs(dscr) < 2.0):
            problems.append(f"Step 54 dense distance-scramble null must be ~0, got {dscr:.2f}")
        if not (dk - gk > 5.0):
            problems.append(f"Step 54 dense gain must exceed global by >5 pts, got {dk - gk:.2f}")
        if not re.search(r"densit(?:y|ies)", text, re.I) or not re.search(r"Step 54", text):
            problems.append("manuscript must report the Step 54 density-scaling result")
    except (KeyError, FileNotFoundError):
        problems.append("Step 54 density-scaling output missing")

    # Step 55 -- causal outage clock prediction on the dense network (under
    # active development).  Only gate on it once the run is numerically clean
    # (sane persistence RMS); the manuscript does not yet cite its numbers.
    try:
        s55doc = load("step_55_dense_outage_prediction.json")
        by_gap = s55doc.get("by_gap", {})
        clean = bool(by_gap) and all(
            0.0 < a.get("rms_m", {}).get("pers", 1e9) < 100.0 for a in by_gap.values())
        if clean:
            s55 = s55doc["headline"]
            scr = s55.get("distance_scramble_vs_net_mean_rms_pct", {})
            if scr and not all(abs(v) < 3.0 for v in scr.values()):
                problems.append(f"Step 55 distance-scramble null must be ~0, got {scr}")
            if re.search(r"Step 55", text):
                vb = s55.get("tep_vs_best_temporal_posthoc_rms_pct", {})
                if vb and not all(v > 0.0 for v in vb.values()):
                    problems.append(f"manuscript cites Step 55 but it is not net-positive: {vb}")
                vs = s55.get("tep_vs_best_conventional_spatial_posthoc_rms_pct", {})
                if vs and not all(v >= 0.0 for v in vs.values()):
                    problems.append(
                        f"manuscript cites Step 55 but TEP loses to a stronger spatial control: {vs}")
                positive_ci = sum(
                    1 for a in by_gap.values()
                    if (a.get("tep_vs_best_conventional_spatial_posthoc_ci95") or [None])[0]
                    is not None
                    and a["tep_vs_best_conventional_spatial_posthoc_ci95"][0] > 0)
                if positive_ci < 4:
                    problems.append(
                        f"Step 55 needs positive clustered CI vs strongest spatial control "
                        f"at >=4 gaps, got {positive_ci}")
    except (KeyError, FileNotFoundError):
        pass

    # Steps 77/80 -- co-located differential anchors and the first blinded
    # clock-field coordinate inversion.  These gates prevent daily
    # pseudo-replication, GPS-steered WUHN promotion, or a random localisation
    # surface from becoming a headline.
    try:
        s77 = load("step_77_colocated_screening_differential.json")
        v77 = s77["verdict"]
        if v77.get("n_primary_free_atomic_anchors") != 3:
            problems.append("Step 77 must retain exactly three verified free-atomic anchors")
        if set(v77.get("primary_sites_showing_reference_class_contrast",
                       v77.get("primary_sites_showing_screening", []))) != {"MAT1-MATE", "THTI-THTG"}:
            problems.append(f"Step 77 primary replicas changed: {v77.get('primary_sites_showing_reference_class_contrast')}")
        for site in ("MAT1-MATE", "THTI-THTG"):
            arm = s77["arms"][site]
            null = arm["primary_geometry_null"]
            if null.get("unit", "").startswith("distant reference station") is False:
                problems.append(f"Step 77 {site} null is not station-clustered")
            if not (null["n_realisations"] >= 1000 and null["p_R2_ge_observed"] <= 0.0011):
                problems.append(f"Step 77 {site} geometry null no longer passes: {null}")
        if s77["arms"]["WUH2-WUHN"]["anchor_metadata"].get("primary_anchor"):
            problems.append("GPS-steered WUHN must not be promoted to a primary free-atomic anchor")
        if s77["arms"]["STR2-STR1"]["delta_couples_to_distant_field"]:
            problems.append("Step 77 must retain the adverse Stromlo anchor result")

        s80 = load("step_80_clock_field_localisation.json")
        for site, max_error in (("MAT1-MATE", 400.0), ("THTI-THTG", 800.0)):
            fit = s80["sites"][site]["multimode_fits"]["exponential"]["all_modes"]
            if fit["surface_error_km"] > max_error:
                problems.append(f"Step 80 {site} multimode localisation regressed: {fit['surface_error_km']:.1f} km")
            if fit["geometry_null"]["p_localisation_error_le_observed"] >= 0.05:
                problems.append(f"Step 80 {site} does not beat its geometry-null localisation")
        if not re.search(r"339\s*km", text) or not re.search(r"744\s*km", text):
            problems.append("manuscript must report both Step 80 multimode localisation results")
        if not re.search(r"GNSS-derived precise products|GNSS data", text, re.I):
            problems.append("Step 80 must be labelled as a GNSS-derived observability replay")
    except (KeyError, FileNotFoundError):
        problems.append("Step 77/80 anchor-localisation outputs missing or incomplete")

    # Step 174 -- cryptographic sealed-envelope ex-ante predictive proof.
    # If the result file exists, its headline numbers must be internally
    # consistent and any manuscript text invoking the sealed envelope must
    # disclose the frozen-weight skill-decay boundary (the honest limit of
    # the ex-ante claim).
    s174_path = ROOT / "results" / "outputs" / "step_174_sealed_envelope_prediction.json"
    if s174_path.exists():
        s174 = json.load(open(s174_path))
        wf = s174["walk_forward_sealed_envelopes"]["full_causal_sealed"]
        wn = s174["walk_forward_sealed_envelopes"]["network_only_sealed"]
        perm = s174["spatial_falsification_battery"][
            "station_identity_permutation_null"
        ]
        cmr = s174["spatial_falsification_battery"]["common_mode_stripped"]
        ed = s174["live_causal_audit"]["epoch_disjoint_7200s_all"][
            "causal_unscaled"
        ]
        if wf["win_count"] != wf["n"] or wf["ci95_block_boot_pct"][0] <= 0:
            problems.append("Step 174 sealed walk-forward must win every window with CI>0")
        if wn["win_count"] != wn["n"] or wn["mean_pct"] <= 0:
            problems.append("Step 174 network-only sealed arm must stay positive")
        if perm["z_score_mean"] < 5.0:
            problems.append("Step 174 station-permutation null separation collapsed")
        if cmr["skill_retained_pct"] < 50.0:
            problems.append("Step 174 common-mode-stripped retention collapsed")
        if ed["sign_test_p"] >= 0.05 or ed["win_count"] < ed["n"] - 1:
            problems.append("Step 174 epoch-disjoint live audit must stay significant")

        # Value pinning: if the manuscript cites Step 174 it must quote the
        # CURRENT committed numbers, not an intermediate snapshot.  Each entry
        # is (label, json value, decimal places as quoted in the manuscript).
        aud = s174["live_causal_audit"]["causal_unscaled"]
        fb = s174["spatial_falsification_battery"]
        pe = s174["purged_epoch_disjoint_horizons"]
        pinned = [
            ("causal audit mean", aud["mean_pct"], 2),
            ("causal audit median", aud["median_pct"], 2),
            ("sealed 1-step mean", wf["mean_pct"], 2),
            ("sealed 1-step median", wf["median_pct"], 2),
            ("network-only sealed mean", wn["mean_pct"], 2),
            ("station-basis unpermuted mean", perm["unpermuted_sta_mean_pct"], 2),
            ("permutation null mean", perm["null_mean_pct"], 2),
            ("permutation null std", perm["null_std_pct"], 2),
            ("permutation z mean", perm["z_score_mean"], 1),
            ("common-mode-stripped mean", cmr["mean_pct"], 2),
            ("common-mode-stripped retention",
             cmr["skill_retained_pct"], 1),
            ("common-mode-only mean",
             fb["common_mode_only"]["mean_pct"], 2),
            ("beyond-360km band mean",
             fb["distance_bands"]["beyond_troposphere_gt360km"]["mean_pct"], 2),
            ("beyond-1000km band mean",
             fb["distance_bands"]["far_arc_gt1000km"]["mean_pct"], 2),
            ("1h embargo mean",
             pe["embargo_3600s_1h_gap"]["full_causal_sealed"]["mean_pct"], 2),
            ("2h embargo mean",
             pe["embargo_7200s_2h_disjoint"]["full_causal_sealed"]["mean_pct"], 2),
            ("45-min lead mean",
             pe["multi_step_horizon_decay"]["step_4"]["full_causal_mean_pct"], 2),
            ("67-min lead mean",
             pe["multi_step_horizon_decay"]["step_6"]["full_causal_mean_pct"], 2),
            ("fixed-position step-1",
             pe["fixed_position_late_band_h4500_5400_by_lead"]["step_1"][
                 "mean_pct"], 2),
            ("fixed-position step-2",
             pe["fixed_position_late_band_h4500_5400_by_lead"]["step_2"][
                 "mean_pct"], 2),
            ("strict-future slice mean",
             pe["strict_future_epoch_slices"]["full_causal_sealed"]["mean_pct"],
             2),
            ("resubstitution bound",
             pe["resubstitution_bound"]["full_causal_sealed_on_own_member"][
                 "mean_pct"], 2),
            ("disjoint-7200s audit mean", ed["mean_pct"], 2),
            ("post-warmup audit mean",
             s174["live_causal_audit"]["epoch_disjoint_7200s_post_warmup"][
                 "causal_unscaled"]["mean_pct"], 2),
            ("audit span hours",
             s174["live_causal_audit"]["span_hours"], 1),
            ("90-min parity step",
             pe["multi_step_horizon_decay"]["step_8"]["full_causal_mean_pct"], 2),
            ("clean-origin composite mean",
             pe["clean_origin_fit_half"][
                 "one_step_sealed_stack_composite"]["mean_pct"], 2),
            ("clean-origin pure mean",
             pe["clean_origin_fit_half"][
                 "one_step_sealed_pure_correction"]["mean_pct"], 2),
            ("clean-origin freshest pure band",
             pe["clean_origin_fit_half"]["position_bands_1step_pure"][
                 "h_0_1800"]["mean_pct"], 2),
            ("oracle-replay echo ceiling",
             pe["clean_origin_fit_half"][
                 "oracle_replay_echo_ceiling_1step"]["mean_pct"], 2),
            ("inner-ho ex-ante lam selection mean",
             pe["clean_origin_fit_half"]["lambda_robustness_pure"][
                 "inner_cv_lambda_selection"]["inner_ho"]["mean_pct"], 2),
            ("inner-pool ex-ante lam selection mean",
             pe["clean_origin_fit_half"]["lambda_robustness_pure"][
                 "inner_cv_lambda_selection"]["inner_pool"]["mean_pct"], 2),
        ]
        for label, val, ndp in pinned:
            if f"{val:.{ndp}f}" not in text:
                problems.append(
                    f"Step 174 {label} value {val:.{ndp}f} not in manuscript")
        n_w = s174["live_causal_audit"]["n_windows"]
        if f"{aud['win_count']} of {n_w}" not in text:
            problems.append(
                f"Step 174 audit win count {aud['win_count']} of {n_w} missing")

        # Every post-seal frozen score logged in the watcher log for any
        # envelope the manuscript cites must appear in reader-facing text
        # (no silent drops of negative windows).  The log holds the
        # corrected-taper rescore produced by the live watcher.
        wlog = ROOT / "data" / "ppp_live" / "sealed_envelopes" / "watch.log"
        cited_envs = set(re.findall(r"envelope_\d{8}_\d{6}", text))
        if wlog.exists() and cited_envs:
            blocks = wlog.read_text().split("=== SEALED ENVELOPE:")
            for blk in blocks:
                env = re.search(r"envelope_\d{8}_\d{6}", blk)
                if not env or env.group(0) not in cited_envs:
                    continue
                for sc in re.findall(
                    r"^\s*\d{9,}\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+"
                    r"([+-]\d+\.\d+)%",
                    blk, re.M):
                    if sc not in text:
                        problems.append(
                            f"Step 174 post-seal frozen score {sc}% "
                            f"({env.group(0)}) not disclosed in manuscript")

        # Step 174b -- untapered clean-arm companion.  If the manuscript
        # cites the untapered pure-correction arm it must quote the
        # committed per-lambda means, not the probe's rounded recall.
        s174b_path = (ROOT / "results" / "outputs"
                      / "step_174_clean_untapered_arm.json")
        if s174b_path.exists():
            s174b = json.load(open(s174b_path))
            obl = s174b["one_step_pure_by_lambda"]
            for lam_key in ("lam_5e+06", "lam_1e+07"):
                v = obl[lam_key]["untapered"]["mean_pct"]
                if re.search(r"untapered|un.tapered", text, re.I) and \
                        f"{v:.2f}" not in text:
                    problems.append(
                        f"Step 174b untapered {lam_key} mean {v:.2f} "
                        "not in manuscript")

        mentioned = re.search(
            r"sealed.envelope|cryptographic|ex.?ante|SHA.?256|Step\s*174",
            text, re.I)
        if mentioned:
            # If the manuscript cites the sealed-envelope result it must carry
            # the headline 1-step number and the honest decay disclosure.
            if not re.search(r"(?:39|4[0-9])\.\d+\s*\\?%", text):
                problems.append("Step 174 sealed 1-step headline missing")
            if not re.search(
                r"skill.{0,80}(?:decay|half.life|lead)|(?:decay|zero.crossing|stale).{0,120}(?:min|hour|horizon)|~?90\s*min",
                text, re.I):
                problems.append(
                    "Step 174 must disclose the frozen-weight skill-decay boundary")
            if not re.search(r"17\.\d|z\s*=\s*1[5-9]", text):
                problems.append("Step 174 permutation-null z-score missing")
            # The epoch-sharing disclosure is mandatory: the clean-origin
            # (zero-overlap) arm showed the sealed headline is dominated
            # by retrodiction of member-observed epochs, so citing the
            # +39.6% sealed figure without the zero-overlap bound is a
            # misleading claim.
            if not re.search(
                r"clean.origin|zero.overlap|fit.half.gram|oracle.replay",
                text, re.I):
                problems.append(
                    "Step 174 must disclose the clean-origin zero-overlap bound")
            if not re.search(
                r"retrodiction|reproduc.{0,40}observed|never.observed|unseen epoch",
                text, re.I):
                problems.append(
                    "Step 174 must state the retrodiction-vs-forecast decomposition")
            # The guarded/oracle channel must never be presented as the
            # causal sealed result.
            if re.search(
                r"sealed.envelope.{0,200}guard|guard.{0,80}sealed",
                text, re.I | re.DOTALL):
                problems.append("Step 174 must not conflate sealed causal with guarded oracle")

    # Step 80 geometry-null medians must be quoted from the committed JSON.
    # The Tahiti median (10,044 km) was previously typo'd as 10,450 km in
    # reader-facing text -- ban the stale literal outright.
    try:
        s80g = load("step_80_clock_field_localisation.json")["sites"]
        for site in ("MAT1-MATE", "THTI-THTG"):
            med = (s80g[site]["multimode_fits"]["exponential"]["all_modes"]
                   ["geometry_null"]["median_localisation_error_km"])
            med_s = f"{med:,.0f}"
            if med_s not in text:
                problems.append(
                    f"Step 80 {site} geometry-null median {med_s} km not in manuscript")
        if re.search(r"10,450\s*km", text):
            problems.append("stale Step 80 Tahiti null median 10,450 km")
    except (KeyError, FileNotFoundError):
        problems.append("Step 80 localisation output missing for null-median check")

    # Step 72 housing/environment screening: every quoted value must come from
    # the persisted analysis block; the previous paragraph admitted the values
    # were stdout-only, which is precisely how stale numbers survive.
    try:
        s72 = load("step_72_housing_screening.json")
        ana = s72.get("analysis")
        if not isinstance(ana, dict):
            problems.append("Step 72 output must persist its analysis block")
            ana = {}
        else:
            pc = ((ana.get("temperature") or {}).get("combined") or {})
            if pc:
                pr = pc.get("partial_r_temp_tol_lf_given_clock_type", 0)
                pp = pc.get("partial_r_p_value", 0)
                if f"{pr:.3f}" not in text or f"{pp:.2f}" not in text:
                    problems.append(
                        f"Step 72 partial correlation {pr:.3f} (p={pp:.2f}) "
                        "not quoted in manuscript")
            met = ana.get("metrology") or {}
            if met.get("ratio"):
                if f"{met['ratio']:.2f}" not in text:
                    problems.append(
                        f"Step 72 metrology ratio {met['ratio']:.2f} not in manuscript")
            cens = ana.get("temperature_census") or {}
            if cens.get("n_site_logs") and str(cens["n_site_logs"]) not in text:
                problems.append("Step 72 site-log census count not in manuscript")
        for label, pattern in (
            ("stale Step 72 'not persisted' admission",
             r"not persisted in the JSON output|cannot be independently verified"),
            ("stale Step 72 site-log census", r"parsed from 65 site logs"),
            ("stale Step 72 uncontrolled count", r"34 report none"),
            ("stale Step 72 controlled ratio", r"2\.29\\?times.{0,40}temperature"),
            ("stale Step 72 uncontrolled ratio", r"1\.64\\?times.{0,40}uncontrolled"),
            ("stale Step 72 metrology ratio", r"0\.58\\?times"),
        ):
            if re.search(pattern, text, re.I):
                problems.append(label)
    except FileNotFoundError:
        problems.append("Step 72 housing-screening output missing")

    # Stale/invalid reader-facing headlines are build failures.
    forbidden = {
        "invalid temporal headline": r"beats the best temporal predictor by\s*\$?\+57",
        "Step 33 end-to-end claim": r"tested end-to-end.{0,160}Step 33",
        "Step 33 real-solver headline": r"real-solver (?:position )?demonstration",
        "live acquisition promoted to proof": r"real-time NTRIP demonstration confirming live operability",
        "deployment readiness overclaim": r"method is ready for operational trials",
        "invalid free-GP validation claim": r"free.?gp (?:scale.discovery )?is (?:therefore )?a validated|S3 (?:physics-separability )?gate therefore passes",
        "invalid Step 48 lowest-p50 claim": r"TEP kernel has the lowest p50",
        "excluded steering promoted to operations": r"excl(?:uded)?[. -]?steering.{0,180}(?:operationally meaningful|deployable receiver would achieve|represent what a real receiver)",
        "RINEX header misdescribed as event removal": r"every real GNSS processor removes these via the RINEX",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            problems.append(label)

    # Live seal ledger: the quoted accumulating figure must equal the
    # current pinned JSON (rebuilt by scripts/live/live_seal_ledger.py
    # from the watcher logs), not a stale snapshot.
    try:
        led = load("step_174_live_ledger.json")["first_window"]
        if not re.search(rf"{led['win_count']} of {led['n']}", text):
            problems.append("seal ledger win count not quoted at current value")
        if not re.search(re.escape(f"{led['mean_pct']:+.2f}"), text):
            problems.append("seal ledger mean not quoted at current value")
        for label, pat in (
            ("stale 66-envelope ledger", r"66 (?:sealed|committed) envelope|26\.64|27\.18|64 of 66"),
        ):
            if re.search(pat, text):
                problems.append(label)
    except FileNotFoundError:
        problems.append("seal ledger output missing")

    # Live observatory recorded-score audit: the quoted accumulating
    # figure must equal the pinned JSON (rebuilt by
    # scripts/live/live_observatory_audit.py from the recorded
    # last_result_window_*.json archives).
    try:
        obs = load("live_observatory_audit.json")
        allw = obs["all_windows"]
        if not re.search(rf"{allw['n_positive']} of {allw['n_windows']}",
                         text):
            problems.append("observatory audit win count not quoted at "
                            "current value")
        if not re.search(re.escape(f"{allw['mean_pct']:+.2f}"), text):
            problems.append("observatory audit mean not quoted at "
                            "current value")
        for arm in ("epoch_disjoint", "post_warmup"):
            a = obs[arm]
            if not re.search(rf"{a['n_positive']}\$?\s+of\s+\$?"
                             rf"{a['n_windows']}", text):
                problems.append(f"observatory audit {arm} count not "
                                "quoted at current value")
            if not re.search(re.escape(f"{a['mean_pct']:+.2f}"), text):
                problems.append(f"observatory audit {arm} mean not "
                                "quoted at current value")
    except FileNotFoundError:
        problems.append("live observatory audit output missing "
                        "(run scripts/live/live_observatory_audit.py)")

    # Anchor chain: the quoted link count must equal the verified chain
    # status (rebuilt by scripts/live/anchor_chain_status.py), and the
    # chain must verify intact.
    try:
        anc = load("sealed_anchor_chain_status.json")
        if anc.get("failures"):
            problems.append("anchor chain verification has failures")
        if not re.search(rf"{anc['n_links']}-link chain", text):
            problems.append("anchor chain link count not quoted at "
                            "current value")
    except FileNotFoundError:
        problems.append("anchor chain status missing "
                        "(run scripts/live/anchor_chain_status.py)")

    # Step 177 distance-blind geography: the quoted regional-subnet
    # statistics must match the committed JSON.
    try:
        sub = load("step_177_distance_blind_geography.json")["subnet"]
        plain = re.sub(r"[{},\s]", "", text)
        for v in (f"{sub['pearson_r']:.3f}",
                  f"{sub['spearman_rho']:.3f}",
                  f"{sub['median_position_error_km']:.0f}"):
            if v not in plain:
                problems.append(f"Step 177 value {v} not quoted in "
                                "manuscript")
        npr = sub["null_phase_randomise"]
        if not re.search(rf"{npr['n_exceeding']} of {npr['n_null']} "
                         r"exceedances", text):
            problems.append("Step 177 phase-null exceedance count not "
                            "quoted")
    except FileNotFoundError:
        problems.append("step_177 output missing")

    # Step 178 satellite-scale reconciliation: the quoted half-corpus and
    # legacy-screen values must match the committed JSON.
    try:
        r178 = load("step_178_satellite_scale_reconciliation.json")["subsets"]
        plain = re.sub(r"[{},\s]", "", text)
        for key, val in (
            ("step36_90d_repaired", r178["step36_90d_repaired"]["lambda_km"]),
            ("step98_first_90d", r178["step98_first_90d"]["lambda_km"]),
            ("step98_all_180d", r178["step98_all_180d"]["lambda_km"]),
            ("step36_90d_legacy", r178["step36_90d_legacy"]["lambda_km"]),
        ):
            if val is None:
                continue
            if f"{val:,.0f}".replace(",", "") not in plain:
                problems.append(f"Step 178 {key} lambda {val:,.0f} not "
                                "quoted in manuscript")
        if f"{r178['step36_90d_legacy']['n_pair_days']:,}".replace(
                ",", "") not in plain:
            problems.append("Step 178 legacy-screen pair-day count not "
                            "quoted")
    except FileNotFoundError:
        problems.append("step_178 output missing")

    # Step 179 CMB-axis robustness: bootstrap win count, split-half
    # margins and held-out template correlations.
    try:
        r179 = load("step_179_axis_robustness.json")
        boot = r179["bootstrap"]
        if not re.search(
                rf"{boot['wins']['cmb']} of {r179['n_boot']}|"
                rf"{boot['wins']['cmb']}/{r179['n_boot']}", text):
            problems.append("Step 179 bootstrap CMB win count not quoted")
        for half in ("first", "second"):
            m = r179["split_half"][half]["cmb_margin"]
            if f"{m:,.0f}".replace(",", "") not in \
                    re.sub(r"[{},]", "", text):
                problems.append(f"Step 179 {half}-half CMB margin {m:,.0f} "
                                "not quoted")
        if r179["split_half"]["first"]["winner"] != "cmb" or \
                r179["split_half"]["second"]["winner"] != "cmb":
            problems.append("Step 179 split-half winner changed: update "
                            "manuscript")
    except FileNotFoundError:
        problems.append("step_179 output missing")

    # Step 88 per-station and seasonal breakdown: must have 4 quarters
    # (Q1..Q4, never a remainder Q5) and quote the committed quarter gains.
    try:
        r88 = load("step_88_per_station_breakdown.json")
        seas = r88.get("seasonal_stability") or {}
        if set(seas.keys()) != {"Q1", "Q2", "Q3", "Q4"}:
            problems.append(f"Step 88 seasonal_stability must have Q1..Q4, got {sorted(seas)}")
        for q in ("Q1", "Q2", "Q3", "Q4"):
            for g in ("3600", "14400"):
                v = seas[q][g]["tep_vs_pers_pct"]
                if f"+{v:.1f}" not in text:
                    problems.append(f"Step 88 {q} {g}s gain +{v:.1f}% not quoted in manuscript")
        if re.search(r"\bQ5\b", text):
            problems.append("manuscript contains bogus Q5 quarter")
    except (KeyError, FileNotFoundError):
        problems.append("step_88 output missing or incomplete")

    # Step 45 synthetic common-view-only null and Step 56 lambda_1/e sensitivity
    try:
        scv = s45["arms"]["broadcast_global"]["synthetic_common_view_only_null"]
        eq_pc = scv["equal_weight"]["pc_alpha_logd_given_jbar_mean"]
        s2_pc = scv["sin2_elev_weighted"]["pc_alpha_logd_given_jbar_mean"]
        if f"+{eq_pc:.3f}" not in text or f"+{s2_pc:.3f}" not in text:
            problems.append("Step 45 synthetic common-view-only null PCs not quoted in manuscript")
        lsens = s56["lambda_convention_sensitivity_rms_m"]
        if "0.13" not in text or "0.33" not in text or f"{lsens['3600']['lam_1e_4200_op_2970']:.2f}" not in text:
            problems.append("Step 56 lambda_1/e convention sensitivity not quoted in manuscript")
    except (KeyError, FileNotFoundError):
        problems.append("step_45 synthetic common-view null or step_56 lambda sensitivity missing")

    if problems:
        print("MANUSCRIPT / EVIDENCE FAILURES:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("manuscript matches the corrected evidence ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
