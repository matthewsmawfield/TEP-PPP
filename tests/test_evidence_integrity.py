import json
import sys
import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.lib.tep_phase import (spectral_phase_gram, phase_distance_weights,
                                   phase_randomised_real_gram, spectral_phase_blocks,
                                   complex_wiener_weights)


def test_phase_gram_matches_registered_phase_cos_and_is_psd():
    dt = 300.0
    n = 576  # 48 h: resolves the complete registered 10--500 uHz band
    t = np.arange(n) * dt
    a = np.sin(2 * np.pi * 80e-6 * t) + 0.5 * np.sin(2 * np.pi * 210e-6 * t)
    b = a.copy()
    c = -a
    result = spectral_phase_gram(np.column_stack((a, b, c)), dt)
    assert result["phase_cos"][0, 1] > 0.999
    assert result["phase_cos"][0, 2] < -0.999
    assert np.allclose(result["predictive_gram"],
                       result["coherence"] * result["phase_cos"], atol=1e-12)
    assert np.linalg.eigvalsh(result["predictive_gram"]).min() > -1e-10


def test_phase_gram_is_scale_invariant_and_does_not_overflow():
    dt = 300.0
    n = 576
    t = np.arange(n) * dt
    a = np.sin(2 * np.pi * 80e-6 * t) + 0.2 * np.cos(2 * np.pi * 210e-6 * t)
    x = np.column_stack((a, np.roll(a, 3), -a))
    expected = spectral_phase_gram(x, dt)["predictive_gram"]
    scaled = x * np.array([1e140, 1e-140, 1e75])
    actual = spectral_phase_gram(scaled, dt)["predictive_gram"]
    assert np.isfinite(actual).all()
    assert np.allclose(actual, expected, atol=1e-12)


def test_phase_analysis_refuses_unresolved_registered_band():
    x = np.arange(24, dtype=float)[:, None] * np.array([[1.0, 2.0]])
    with np.testing.assert_raises_regex(ValueError, "too short"):
        spectral_phase_gram(x, 300.0)


def test_phase_distance_covariance_solve_is_finite():
    d = np.array([[0.0, 100.0], [100.0, 0.0]])
    gram = np.array([[1.0, 0.5], [0.5, 1.0]])
    w = phase_distance_weights(d, np.array([50.0, 80.0]), gram,
                               np.array([0.7, 0.6]))
    assert np.isfinite(w).all()


def test_phase_randomisation_preserves_magnitude_and_psd():
    C = np.array([[1, 0.4 + 0.2j], [0.4 - 0.2j, 1]], complex)
    theta = np.array([0.3, -1.1])
    rot = np.exp(1j * theta)
    Cr = rot.conj()[:, None] * C * rot[None, :]
    G = phase_randomised_real_gram(C, theta)
    assert np.allclose(np.abs(Cr), np.abs(C))
    assert np.linalg.eigvalsh(G).min() > -1e-12


def test_phase_blocks_preserve_24_hour_mode_resolution_without_single_bin_covariance():
    # 24 h at the CODE common 5-minute cadence has the 43 registered DFT bins.
    t = np.arange(288) * 300.0
    x = np.column_stack([
        np.sin(2 * np.pi * 80e-6 * t),
        np.sin(2 * np.pi * 80e-6 * t + 0.4),
        np.sin(2 * np.pi * 210e-6 * t - 0.6),
    ])
    blocks = spectral_phase_blocks(x, 300.0, bins_per_block=3)
    assert sum(b["n_bins"] for b in blocks) == 43
    assert all(1 <= b["n_bins"] <= 3 for b in blocks)
    assert all(np.linalg.eigvalsh(b["predictive_gram"]).min() > -1e-10 for b in blocks)


def test_complex_wiener_weights_retain_hermitian_phase_covariance():
    z = np.array([1.0 + 0j, 0.6 + 0.8j])
    K = np.outer(z, z.conj()) + 0.1 * np.eye(2)
    w = complex_wiener_weights(K, np.array([0.5 + 0.2j, 0.1 - 0.3j]), nugget=0.2)
    assert np.iscomplexobj(w)
    assert np.isfinite(w).all()


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "outputs"
sys.path.insert(0, str(ROOT))


def load(name):
    return json.loads((OUT / name).read_text())


def test_t1_interval_contains_selected_family_estimate():
    t1 = load("step_10_global_correlation.json")["global_broadcast"]
    lo, hi = t1["lambda_ci95"]
    assert t1["covariance_form"] == "gaussian"
    assert lo <= t1["lambda_km"] <= hi


def test_offline_temporal_result_is_not_promoted_to_causal_evidence():
    arm = load("step_38_baseline_battery.json")["arms"]["broadcast_spp"]
    assert arm["tep_vs_best_conventional_pct"]["ci95"][0] <= 0


def test_raw_thinning_does_not_silently_pass_p2():
    p2 = load("step_44_raw_thin_position.json")
    assert p2["evidence_level"] == "raw_observation"
    assert p2["uses_full_sky_linearization"] is False
    assert p2["strict_p2_pass"] is False


def test_three_satellite_result_keeps_truth_linearization_boundary():
    p = ROOT / "scripts" / "steps" / "step_48_three_satellite.py"
    source = p.read_text()
    assert "linearisation point IS\n    the truth" in source
    assert load("step_48_three_satellite.json")["uses_full_sky_linearization"] is False


def test_free_gp_resolves_tep_band_when_signal_dominates_noise():
    """Resolving-power validation of the unconstrained GP (Step 43).

    The injection amplitude is set per day to dominate the ACTUAL per-station
    band-limited residual variance (the earlier fixed A = 0.3 m^2 was
    calibrated against a mis-stated ~0.05 m spatial std and left the signal
    ~3:1 below the ~1.6 m local-oscillator noise, so the test measured SNR,
    not resolving power).  With that corrected, the free GP recovers injected
    TEP-band scales, the >1,000 km baseline-restricted free GP moves into the
    band, and both spatial-signal nulls collapse to the grid edge.
    """
    s = load("step_43_free_gp_validation.json")["summary"]
    assert s["injection_recovery_rate_tep_band_within_2x"] >= 0.6
    assert s["injection_tep_band_scales_passing"] >= 4
    assert s["injection_validates_full_tep_band"] is True
    assert s["baseline_restricted_in_tep_band"] is True
    assert s["nulls_fail_tep_band"] is True
    assert s["free_gp_is_validated_scale_discovery"] is True


def test_free_gp_injection_amplitude_tracks_actual_residual_variance():
    """Guard the SNR-calibration fix: the injection must scale with the real
    per-station residual variance, not a hard-coded 0.3."""
    src = (ROOT / "scripts" / "steps" / "step_43_free_gp_validation.py").read_text()
    assert "SNR_TARGET" in src
    assert "np.nanvar(X, axis=0)" in src


def test_step96_leftover_clock_passes_and_is_tep_specific():
    """Step 96 (P1 leftover-clock flagship): on the held-out year 2025 the
    frozen target-blind LF+MF field must beat every deployable target-blind
    baseline AND every TEP-specificity null (unweighted common-mode mean,
    distance-scramble, phase-scramble) at the 1-4 h horizon, on Europe +
    North America + East Asia -- via the drift-gated construction where a
    deterministic drift dominates (EU, NA) and the same construction without
    the gate where it does not (East Asia).  The pass is worth nothing if
    the phase-scramble null is not also cleared."""
    d = load("step_96_leftover_clock_supremacy.json")
    v = d["verdict"]
    assert v["level1_pass"] is True
    cells = set(v.get("tep_lfmf_gated_passing_cells_mid") or [])
    assert {"clk_cod_ops/EUROPE", "clk_cod_ops/NORTH_AMERICA"} <= cells
    # The disclosed per-region claim (manuscript 3.9h / P1): the drift-GATED
    # construction carries Europe and North America, where deterministic
    # station drift must be removed before the field is visible; in East Asia
    # the gate is inactive on that network's low-drift atomic clocks and its
    # residual subtraction costs a few percent, so the SAME construction
    # WITHOUT the gate is the winning arm there.
    winning_arm = {"EUROPE": "tep_lfmf_gated",
                   "NORTH_AMERICA": "tep_lfmf_gated",
                   "EAST_ASIA": "tep_lfmf"}
    for reg, arm in winning_arm.items():
        h = d["results"][f"clk_cod_ops_2025_{reg}"]["by_horizon"]["mid"]
        assert h[f"{arm}_beats_all_baselines"] is True, reg
        assert h[f"{arm}_beats_specificity_nulls"] is True, reg
        # the phase Gram must be load-bearing: scrambling it must cost >5%
        assert h[f"{arm}_vs_tep_lfmf_gated_phasescr_pct"]["median"] > 5.0, reg


def test_step96_reports_specificity_nulls():
    """The Step 96 script must carry the network-mean / distance-scramble /
    phase-scramble nulls -- a bare 'beats persistence' is not a TEP claim."""
    src = (ROOT / "scripts" / "steps"
           / "step_96_leftover_clock_supremacy.py").read_text()
    assert "netmean_gated" in src
    assert "tep_lfmf_gated_distscr" in src
    assert "tep_lfmf_gated_phasescr" in src
    assert "beats_specificity_nulls" in src


def test_three_satellite_soft_prior_sigma_cannot_act_as_fde():
    """At k=3, three ranges plus one clock row exactly determine four states.

    Reweighting the clock row therefore cannot protect the solution.  This
    guards against describing sigma inflation as an integrity monitor until an
    independent constraint or an explicit reject/no-fix decision is added.
    """
    from scripts.steps.step_48_three_satellite import _solve_pos_soft

    recs = [
        (np.array([1.0, 0.0, 0.0]), 60.0, 10.0, 1.0),
        (np.array([0.0, 1.0, 0.0]), 55.0, 12.0, 1.0),
        (np.array([0.0, 0.0, 1.0]), 50.0, 14.0, 1.0),
    ]
    tight = _solve_pos_soft(recs, clk_prior=100.0, clk_sigma=1.0, tropo_val=0.0)
    loose = _solve_pos_soft(recs, clk_prior=100.0, clk_sigma=1_000.0, tropo_val=0.0)
    assert tight is not None and loose is not None
    assert np.allclose(tight, loose, atol=1e-8)


def test_iterative_clock_jump_repair_preserves_accumulated_offset():
    from scripts.steps.step_48_three_satellite import _repair_clock_jumps

    # A dominant first jump masks the smaller second jump on the first MAD
    # pass.  The saved correction must still reproduce the original series.
    base = np.arange(20, dtype=float) * 0.2
    observed = base.copy()
    observed[6:] += 300_000.0
    observed[14:] += 1_000.0
    repaired, removed, n_jumps = _repair_clock_jumps(observed)
    assert n_jumps >= 2
    assert np.allclose(repaired + removed, observed)
    assert np.max(np.abs(np.diff(repaired) - 0.2)) < 1e-6


def test_causal_clock_model_recognises_and_forecasts_one_ms_wraps():
    from scripts.steps.step_48_three_satellite import (
        _causal_fit, _recent_clock_model, _recent_clock_predict_raw)

    period = 299_792.458
    sods = np.arange(0.0, 120 * 30.0, 30.0)
    unwrapped = -140_000.0 + 12_000.0 * np.arange(len(sods))
    raw = (unwrapped + period / 2) % period - period / 2
    g0 = sods[100]
    fit = _causal_fit(sods, raw, g0)
    pre = sods[(sods >= g0 - 1800.0) & (sods < g0)]
    model = _recent_clock_model(fit, pre)
    predicted = _recent_clock_predict_raw(fit, model, sods[105])
    assert fit["wrap_mode_ms"] is True
    assert fit["clock_unreliable_causal"] is True
    assert np.isclose(predicted, raw[105], atol=1e-6)


def test_k3_integrity_flag_is_reject_not_sigma_inflation():
    source = (ROOT / "scripts" / "steps" / "step_48_three_satellite.py").read_text()
    assert "if k == 3 and integrity_reject" in source
    assert '"k3_sigma_inflation_effective": False' in source


def test_rinex_clock_policy_is_not_promoted_to_event_telemetry():
    from scripts.lib.tep_spp import read_rinex_clock_metadata

    sample = """     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
     1                                                      RCV CLOCK OFFS APPL
                                                            END OF HEADER
> 2024 04 09 00 00  0.0000000  0  3         0.000000002000
> 2024 04 09 00 00 30.0000000  0  3
"""
    meta = read_rinex_clock_metadata(sample)
    assert meta["rcv_clock_offs_appl"] == 1
    assert meta["n_offsets_reported"] == 1
    assert meta["epoch_clock_offsets_s"][0] == 2e-9
    assert meta["epoch_clock_offsets_s"][30] is None
    assert meta["is_steering_event_log"] is False


def test_causal_clock_monitor_does_not_replace_whole_sample():
    from scripts.steps.step_48_three_satellite import _agg

    errors = np.array([1.0] * 20 + [10_000.0])
    flagged = np.array([False] * 20 + [True])
    result = _agg(errors, flagged)
    assert result["n"] == 21
    assert result["rms_3d_m"] > 1_000
    assert result["excl_clock_monitor"]["n"] == 20
    assert result["excl_clock_monitor"]["evidence_level"] == "causal_receiver_side_clock_monitor"


def test_operational_outage_steps_record_reset_like_endpoints():
    for name in ("step_55_dense_outage_prediction.py", "step_56_phase_predictive_bridge.py"):
        source = (ROOT / "scripts" / "steps" / name).read_text()
        assert "target_reset_like_events" in source
        # Representation wraps are unwrapped causally -- either directly, or via
        # the shared step_55 `load_day` loader that step_56 imports.
        assert ("unwrap_series" in source or "_unwrap_clock_m" in source
                or "load_day" in source)
        # A disclosed |truth-base| > 5000 m cut remains for non-modulo
        # discontinuities (hardware resets) and must be counted, not silent.
        assert "5000.0" in source
        # ... and it must EXCLUDE the epoch, not just log it: a reset day's
        # full-jump squared error (~1e11 m^2) otherwise dominates the pooled
        # sum-of-squares and collapses every relative comparison to ~0%.
        # The `continue` must sit inside the > 5000 m branch.
        blk = source.split("abs(truth - base) > 5000.0", 1)[1][:1500]
        assert "continue" in blk, f"{name}: reset-like epoch is logged but not excluded"


def test_step96_gfz_independence_beats_toughest_baseline_and_nulls():
    """The frozen CODE-calibrated predictor must beat the toughest deployable
    baseline (target-aware shrunk empirical covariance) AND both specificity
    nulls on GFZ0OPSFIN -- an independently processed analysis-centre product
    -- with NO re-tuning.  Guards against the result being a CODE-specific
    network-adjustment artefact."""
    for name, key in (("step_96_gfz_global.json", "clk_gfz_2024_GLOBAL"),):
        try:
            d = load(name)
        except FileNotFoundError:
            continue
        mid = d["results"][key]["by_horizon"]["mid"]
        assert mid["tep_lfmf_gated_vs_empirical_cov_full_pct"]["median"] > 0
        assert (mid["tep_lfmf_gated_vs_empirical_cov_full_pct"]["median"]
                > mid["tep_lfmf_gated_distscr_vs_empirical_cov_full_pct"]["median"])
        assert (mid["tep_lfmf_gated_vs_empirical_cov_full_pct"]["median"]
                > mid["tep_lfmf_gated_phasescr_vs_empirical_cov_full_pct"]["median"])
    try:
        d = load("step_96_gfz_level1.json")
    except FileNotFoundError:
        return
    mid_eu = d["results"]["clk_gfz_2025_EUROPE"]["by_horizon"]["mid"]
    assert mid_eu["tep_lfmf_gated_beats_all_baselines"] is True
    assert mid_eu["tep_lfmf_gated_beats_specificity_nulls"] is True


def test_step106_network_prior_beats_free_and_persistence_with_cis():
    """The bulletproof degraded-sky position result: the frozen TEP network
    clock prior must beat BOTH a free-clock solve AND a deployable
    persistence-trend prior, with day-clustered 95% CIs excluding zero, in
    >=2 (k, gap) cells; and it must beat a phase-scrambled kernel (the
    Papers 1-2 phase alignment doing real work).  The predictor operates on
    the SPP clock (what a positioning receiver holds), not the precise
    product -- a source guard."""
    try:
        d = load("step_106_degraded_sky_position_flagship.json")
    except FileNotFoundError:
        return
    if d.get("status"):
        return
    v = d["verdict"]
    assert v["network_clock_prior_beats_persistence"] is True
    assert v["operational_position_gain_pass"] is True
    assert v["n_cells_network_beats_persistence"] >= 2
    # every network-vs-persistence cell CI lower bound > 0
    for k in (4, 5):
        for gs in d["by_gap"]:
            ci = d["by_gap"][gs].get(f"tep_vs_persist_k{k}_ci95")
            if ci is not None:
                assert ci[0] > 0, f"k{k} {gs}: tep-vs-persist CI includes 0"
    src = (ROOT / "scripts" / "steps"
           / "step_106_degraded_sky_position_flagship.py").read_text()
    assert "operates on the target's OWN (SPP) receiver clock" in src


def test_step104_105_leo_negative_with_validated_control():
    """The moving-platform (LEO) clock-bridging test must (a) show the ground
    network does NOT beat persistence for a fast orbital receiver, and (b) the
    ground-to-ground control on the SAME machinery MUST reproduce the Step 55/56
    positive -- otherwise the negative is a broken pipeline, not physics.
    Holds for both the steered Swarm clock (104) and the free-running GRACE-FO
    USO (105)."""
    for name in ("step_104_leo_moving_platform.json",
                 "step_105_gracefo_moving_platform.json"):
        try:
            d = load(name)
        except FileNotFoundError:
            continue
        if d.get("status") == "no_data":
            continue
        ctl = d.get("ground_control", {})
        leo = d.get("leo") or d.get("grace_c_absolute") or {}
        ctl_pos = [v["tep_vs_persist_ci95"][0] > 0 for v in ctl.values()
                   if "tep_vs_persist_ci95" in v]
        leo_neg = [v["tep_vs_persist_pct"] < 0 for v in leo.values()
                   if "tep_vs_persist_pct" in v]
        assert ctl_pos and all(ctl_pos), f"{name}: ground control does not reproduce Step 56"
        assert leo_neg and all(leo_neg), f"{name}: LEO unexpectedly bridgeable -- re-examine"


def test_step45_temporal_t4_reports_distance_fit_and_separation_separately():
    """Step 45 under the corrected stage-1 (pooled kappa on real-SP3
    co-visibility): the broadcast arm must separate common-view at the frozen
    level; the precise arm must show a significant but attenuated partial
    correlation (S2 PARTIAL).  The scored distance fit is the SEPARATED one:
    alpha is residualised on [Jbar, cos(dlon)] before binning (the same
    control set as the partial correlation), and the epoch-shuffle null is
    residualised identically -- an unresidualised null leaks the random
    kappa_null*Jbar(d) term into a heavy-tailed null R^2 distribution."""
    s = load("step_45_temporal_t4.json")["summary"]
    assert s["broadcast_common_view_separated"] is True
    assert s["broadcast_frozen_S2"] is True
    assert abs(s["broadcast_pc"]) > 0.15
    assert s["precise_common_view_separated"] is False
    assert s["precise_frozen_S2"] is False
    assert -0.15 < s["precise_pc"] < 0.0  # attenuated, decaying direction
    # separated distance fit: broadcast passes decisively (R^2 sigma > 3,
    # TEP-band lambda), so the broadcast composite T4 pass is now True;
    # the precise arm remains attenuated (sub-threshold separated fit).
    assert s["broadcast_sep_r2_sigma"] > 3.0
    assert 500 < s["broadcast_sep_lambda"] < 20000
    assert s["broadcast_sep_r2"] > 0.2
    assert s["broadcast_passes"] is True and s["precise_passes"] is False
    # the source must actually gate the pass on both conditions
    src = (ROOT / "scripts" / "steps" / "step_45_temporal_t4.py").read_text()
    assert "distance_fit_significant and common_view_separated" in src


def test_step83_fullyear_phasecos_t4_partial_corr_still_zero_but_scale_robust():
    """A full year of the phase-cos T4 (not 120 days) must still show the
    Jbar-conditioned partial correlation attenuated below the frozen 0.15
    threshold -- more data does not break the between-pair distance/
    co-visibility collinearity -- while the distance fit to the
    non-common-view intercept stays in/near the TEP band with high R^2.
    The stage-2 significance uses the pair-permutation null (the day-level
    permutation can only exercise kappa: alpha = pbar - kappa*Jbar is
    permutation-invariant in pbar and kappa*Jbar is exactly absorbed by the
    stage-2 Jbar control, so the earlier 'null PC' was machine-identical to
    the real one and its z a floating-point artefact).  The corrected null
    must show the precise arm small but significant -- the same attenuated
    direction Step 45 reports (-0.040, 9.2 sigma) -- while the sparse
    broadcast arm stays null.  If |pc| ever clears 0.15, S2 should be
    revisited."""
    a = load("step_83_t4_phasecos.json")["arms"]["precise_ops"]
    assert a["n_days"] >= 360
    assert -0.10 < a["partial_corr_alpha_logdist_given_covis"] < 0.0
    assert a["z_vs_null"] > 2.0            # pair-permutation null, significant
    assert a["pc_station_boot_ci95"] is not None
    assert a["pc_station_boot_ci95"][1] < 0.0   # cluster-aware CI, decaying direction
    assert "z_kappa" in a and "kappa_null_sd" in a
    b = load("step_83_t4_phasecos.json")["arms"]["broadcast_spp"]
    assert abs(b["z_vs_null"]) < 2.0       # sparse tail arm stays null
    assert 2000.0 <= a["distance_fit"]["lambda_km"] <= 8000.0
    assert a["distance_fit"]["R2"] > 0.6


def test_step45b_matched_lowcovis_does_not_survive_on_precise():
    """Under the corrected stage-1 the collinearity-broken matched-low-covis
    spec returns a FINITE in-band lambda on broadcast (the old nominal-covis
    run railed it to the grid edge) and an attenuated sub-threshold PC on
    precise -- consistent with S2 PARTIAL."""
    arms = load("step_45b_t4_forensic.json")["arms"]
    b = arms["broadcast_global"]["spec_battery"]["matched_lowcovis"]
    p = arms["precise_cod"]["spec_battery"]["matched_lowcovis"]
    assert 1000 <= b["lambda_km"] <= 12000
    assert b["partial_corr_alpha_logd_given_controls"] < -0.1
    assert p["survives"] is False


def test_step57_uses_a_deployable_pdop_gate_and_reports_joint_vs_free():
    """The k=3 'noise floor' is only meaningful under a deployable integrity
    gate; PDOP 20 admitted near-singular geometries. And the honest k>=4
    comparison is joint_soft vs a free-clock solve, not vs an unfiltered
    hard-fix baseline."""
    src = (ROOT / "scripts" / "steps" / "step_57_joint_estimator_sensitivity.py").read_text()
    assert "PDOP_MAX = 10.0" in src
    hl = load("step_57_joint_estimator_sensitivity.json")["headline"]["by_fraction"]
    for fr in hl.values():
        for k, row in fr.items():
            if k == "3":
                continue
            assert "joint_vs_free_rms_pct" in row and row["joint_vs_free_rms_pct"] is not None


def test_step56_phase_bridge_beats_its_controls_after_reset_exclusion():
    """Step 56 must beat distance-only TEP, the phase scramble, and the best
    temporal arm at every registered horizon with day-clustered CIs > 0 on
    the frozen 45-day benchmark.  A collapse to ~0% means reset-like clock
    days are contaminating the pooled RMS (the exclusion `continue` is
    missing) -- see test_operational_outage_steps_record_reset_like_endpoints."""
    s56 = load("step_56_phase_predictive_bridge.json")["by_gap"]
    assert len(s56) == 4
    for gap, a in s56.items():
        assert a["n_days"] >= 40, f"{gap}s: not the frozen 45-day benchmark"
        for key in ("phase_tep_vs_tep_ci95",
                    "phase_tep_vs_phase_scramble_ci95",
                    "phase_tep_vs_best_temporal_ci95",
                    "phase_tep_vs_causal_cv_spatial_ci95"):
            assert a[key][0] > 0, f"{gap}s {key} lower bound {a[key][0]:.3f} <= 0"


def test_step48_flagship_is_genuinely_blind_not_raim_labelled_linear_fallback():
    source = (ROOT / "scripts" / "steps" / "step_48_three_satellite.py").read_text()
    assert 'blind_ref_name = "tep_blind_phase_s56"' in source
    assert '"tep_blind_phase_s56": tep_clk_phase' in source
    assert 'blind_ref_name = ("tep_blind_raim_ms"' not in source
    assert "blind_gain >= 5.0" in source


def test_step48_phase_gram_is_not_collapsed_to_distance():
    """The 4–8 h kernel is the phase Gram (Step 56: phase-only ≈ phase×distance).

    Filling missing stations with G=1, falling back to distance-only when
    |w| is large, or MAD-clipping to the neighbour range all convert the
    predictor into a spatial median and erase anti-phase structure.
    """
    source = (ROOT / "scripts" / "steps" / "step_48_three_satellite.py").read_text()
    # Isolate _krige_phase_gram so distance-only _krige's MAD clip is not
    # mistaken for a phase-path clip.
    start = source.index("def _krige_phase_gram")
    end = source.index("\ndef _", start + 1)
    body = source[start:end]
    assert "g_rt = np.ones" not in body
    assert "np.abs(wts).sum() > 5.0" not in body
    assert "nb_med - 4 * nb_mad" not in body
    assert "in_gram" in body
    # Operational precise clocks are the leftover field, not a full-day
    # zero-phase bandpass of a cubic residual.
    load_start = source.index("def _load_precise_clocks_for_day")
    load_end = source.index("\ndef _", load_start + 1)
    load_body = source[load_start:load_end]
    assert "ts_bl.bandpass" not in load_body
    assert "np.polyfit(tn, y, 3)" not in load_body
    assert "clk_precise_raw" in load_body


def test_step48_summary_does_not_hardcode_1h_window():
    source = (ROOT / "scripts" / "steps" / "step_48_three_satellite.py").read_text()
    assert "acc[(k, 3600.0)]" not in source
    assert "acc[(3, 3600.0)]" not in source
    assert "def _preferred_k3_window_key" in source
    assert "def _acc_k3" in source


def test_phase_gram_kriging_recovers_in_phase_cluster():
    """Distance-only averages in-phase and anti-phase neighbours.

    The Gram must pull the prediction onto the in-phase cluster.  Step 56
    4 h: phase-angle scramble is ~2x worse than phase_tep; that contrast
    is the operational signal.
    """
    from scripts.steps.step_48_three_satellite import _krige, _krige_phase_gram

    def xyz(lat, lon):
        r = 6371e3
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.array([r * np.cos(la) * np.cos(lo),
                         r * np.cos(la) * np.sin(lo),
                         r * np.sin(la)])

    tgt = xyz(47.0, 8.0)
    coords = [(47.8, 8.0), (47.0, 8.8), (46.2, 8.0),
              (47.0, 7.2), (47.5, 8.5), (46.5, 7.5)]
    ref_xyz = [xyz(la, lo) for la, lo in coords]
    names = [f"R{i}" for i in range(6)]
    ref_val = np.array([4.0, 4.0, 4.0, -4.0, -4.0, -4.0])
    phases = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi, 0.0])
    G = np.cos(phases[:, None] - phases[None, :])
    gram = {
        "gram": G,
        "names": names + ["TGT"],
        "name_to_idx": {n: i for i, n in enumerate(names + ["TGT"])},
    }
    phase_val, _, _ = _krige_phase_gram(
        tgt, ref_xyz, ref_val, names, "TGT", gram)
    dist_val, _, _ = _krige(tgt, ref_xyz, ref_val)
    assert phase_val > 1.5, (phase_val, dist_val)
    assert phase_val > dist_val + 1.0, (phase_val, dist_val)


def test_paired_method_comparison_uses_only_common_successful_epochs():
    from scripts.steps.step_48_three_satellite import _paired_metrics

    # The unmatched catastrophic errors must not enter a paired comparison.
    ka = [(i,) for i in range(15)] + [(99,)]
    kb = [(i,) for i in range(15)] + [(100,)]
    a = [1.0, 2.0] * 7 + [1.0, 10_000.0]
    b = [2.0, 4.0] * 7 + [2.0, 20_000.0]
    result = _paired_metrics(a, b, ka, kb)
    assert result["n_common"] == 15
    assert np.isclose(result["a_vs_b_rms_pct"], 50.0)


def test_step48_flagship_kriges_absolute_residuals_not_stale_anchor():
    """The winning Step 55/56 predictor is an absolute residual.

    Change-on-stale-SPP-anchor is retained as a diagnostic, but it must not
    be the flagship: that architecture drowned the TEP signal in the target's
    own noisy SPP residual and inverted the clock-domain win.
    """
    source = (ROOT / "scripts" / "steps" / "step_48_three_satellite.py").read_text()
    assert '"absolute_residual_levels_kriged": True' in source
    # Flagship is SPP-domain absolute residual (same observable as AR1).
    assert "tep_clk = local + kres_abs_spp" in source
    assert "kres_abs_spp = kres_abs_spp_diff + abs_spp_mean" in source
    assert "tep_clk = local + kres_abs_pg" in source


def _synthetic_spp_grid(truth):
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.75, 0.66, 0.0]),
        np.array([0.75, -0.25, 0.61]),
        np.array([0.72, -0.30, -0.62]),
    ]
    grid = {}
    for i, direction in enumerate(directions, 1):
        direction = direction / np.linalg.norm(direction)
        pos = truth + 21_000_000.0 * direction
        grid[f"G{i:02d}"] = (
            np.repeat(pos[None, :], 100, axis=0),
            np.zeros((100, 3)),
            np.zeros(100),
        )
    return (0.0, 5.0, grid)


def test_nonlinear_spp_jacobian_sign_and_blind_convergence():
    from scripts.lib.tep_spp import pseudorange_model, solve_spp_nonlinear

    truth = np.array([6_371_000.0, 0.0, 0.0])
    grid = _synthetic_spp_grid(truth)
    sod = 100.0
    clock = 40.0
    obs = []
    for prn in grid[2]:
        pred, los, _, _, ok = pseudorange_model(truth, prn, grid, sod)
        assert ok
        eps = 0.1
        shifted = truth + np.array([eps, 0.0, 0.0])
        pred_shift, *_ = pseudorange_model(shifted, prn, grid, sod)
        numeric_dx = (pred_shift - pred) / eps
        # The simple troposphere model adds a small height derivative that is
        # intentionally omitted from the geometric Jacobian.
        assert np.isclose(numeric_dx, -los[0], atol=1e-3)
        obs.append((prn, pred + clock))

    init = truth + np.array([0.0, 100_000.0, 0.0])
    solved = solve_spp_nonlinear(
        obs, grid, sod, init, clock=clock, zwd=0.0,
        k=3, elev_mask=5.0, pdop_max=100.0)
    assert solved is not None
    assert np.linalg.norm(solved[0] - truth) < 1.0


def test_causal_motion_prior_makes_free_clock_k3_observable():
    from scripts.lib.tep_spp import pseudorange_model, solve_spp_nonlinear

    truth = np.array([6_371_000.0, 0.0, 0.0])
    grid = _synthetic_spp_grid(truth)
    clock = 40.0
    obs = []
    for prn in list(grid[2])[:3]:
        pred, *_ = pseudorange_model(truth, prn, grid, 100.0)
        obs.append((prn, pred + clock))
    # Three ranges cannot estimate position plus clock by themselves.  A
    # truth-independent kinematic prediction makes the sequential state
    # observable without pretending the TEP clock prior supplied the motion.
    solved = solve_spp_nonlinear(
        obs, grid, 100.0, truth + np.array([50.0, -30.0, 20.0]),
        clock=None, zwd=0.0, position_prior=truth, position_sigma=5.0,
        pdop_max=100.0)
    assert solved is not None
    assert np.linalg.norm(solved[0] - truth) < 1.0
    assert abs(solved[1] - clock) < 1.0


def test_soft_clock_prior_remains_centred_on_frozen_prediction():
    from scripts.lib.tep_spp import pseudorange_model, solve_spp_nonlinear

    truth = np.array([6_371_000.0, 0.0, 0.0])
    grid = _synthetic_spp_grid(truth)
    true_clock = 40.0
    obs = []
    for prn in grid[2]:
        pred, *_ = pseudorange_model(truth, prn, grid, 100.0)
        obs.append((prn, pred + true_clock))
    frozen_prior = 35.0
    solved = solve_spp_nonlinear(
        obs, grid, 100.0, truth + np.array([20.0, 10.0, -10.0]),
        clock=frozen_prior, clock_sigma=0.01, zwd=0.0, pdop_max=100.0)
    assert solved is not None
    assert abs(solved[1] - frozen_prior) < 0.05


def test_dense_outage_million_metre_unwrap_is_causal_and_exact():
    from scripts.steps.step_55_dense_outage_prediction import (
        CLOCK_WRAP_M, _unwrap_clock_m)

    continuous = np.arange(30, dtype=float) * 20_000.0 - 120_000.0
    wrapped = (continuous + CLOCK_WRAP_M / 2.0) % CLOCK_WRAP_M - CLOCK_WRAP_M / 2.0
    repaired = _unwrap_clock_m(wrapped)
    # Absolute branch is conventional; increments carry the physical clock.
    assert np.allclose(np.diff(repaired), np.diff(continuous), atol=1e-9)
    # A prefix gives the identical prefix output: no future samples are used.
    assert np.allclose(_unwrap_clock_m(wrapped[:17]), repaired[:17])


def test_clock_unwrap_repairs_one_millisecond_and_product_wraps():
    from scripts.lib.tep_clock import MS_WRAP_M, PRODUCT_WRAP_M, unwrap_series

    t = np.arange(40, dtype=float)
    continuous = 12.0 * t - 80.0
    ms_wrapped = (continuous + MS_WRAP_M / 2.0) % MS_WRAP_M - MS_WRAP_M / 2.0
    repaired_ms = unwrap_series(ms_wrapped, periods=(MS_WRAP_M,))
    assert np.allclose(np.diff(repaired_ms), np.diff(continuous), atol=1e-6)
    mixed = continuous.copy()
    mixed[20:] += PRODUCT_WRAP_M
    mixed[25:] += MS_WRAP_M
    repaired_both = unwrap_series(mixed, periods=(PRODUCT_WRAP_M, MS_WRAP_M))
    assert np.allclose(np.diff(repaired_both), np.diff(continuous), atol=1e-6)
    assert np.allclose(unwrap_series(ms_wrapped[:11], periods=(MS_WRAP_M,)),
                       repaired_ms[:11])


def test_dense_outage_script_does_not_prewrite_a_success_conclusion():
    source = (ROOT / "scripts" / "steps" /
              "step_55_dense_outage_prediction.py").read_text()
    assert "Result-dependent interpretation intentionally omitted" in source
    assert "beats temporal extrapolation of the receiver's own clock at every" not in source


def test_dense_outage_artifact_reports_all_event_and_tail_diagnostics():
    result = load("step_55_dense_outage_prediction.json")
    assert result["n_days"] >= 90
    gaps = result["by_gap"]
    assert len(gaps) == 6
    # Corrected all-event RMS can be dominated by rare product/reset events.
    # The result must expose those events and its robust p99 alongside RMS;
    # the test must never silently demand the old censored clean-RMS outcome.
    assert all("target_reset_like_events" in a and "p99_m" in a
               for a in gaps.values())
    assert all("p50_m" in a and "p95_m" in a for a in gaps.values())


def test_registered_p1_predictor_excludes_failed_speculative_components():
    source = (ROOT / "scripts" / "steps" /
              "step_64_maximal_tep_p1.py").read_text()
    assert "lam_scale = 1.0" in source
    assert "tep_astronomical" not in source
    assert "harmonic_regressors" not in source
    assert "Dani" not in source
    development = load("step_64_maximal_tep_p1.json")["arms"]["precise"]
    assert development["development_gate_pass"] is True
    for key in (
        "full_arsenal_vs_tuned_conventional_pct",
        "full_arsenal_vs_empirical_covariance_pct",
        "full_arsenal_vs_phase_only_pct",
        "full_arsenal_vs_geometry_scramble_pct",
        "full_arsenal_vs_phase_scramble_pct",
    ):
        assert development[key]["ci95"][0] > 0


def test_drone_manifest_requires_verified_public_provenance():
    p = ROOT / "data" / "dynamic_datasets" / "manifest.json"
    if not p.exists():
        pytest.skip("data/dynamic_datasets not downloaded")
    manifest = json.loads(p.read_text())
    drones = [d for d in manifest if d.get("type") == "drone"]
    assert len(drones) == 1
    drone = drones[0]
    assert drone["name"] == "ign_fixed_wing_uav_flight8"
    assert drone["doi"] == "10.5281/zenodo.4476489"
    assert drone["provenance_status"].startswith("verified_public_archive")
    wut = next(d for d in manifest if d["name"] == "wut_kinematic")
    assert wut["type"] != "drone"


def test_public_uav_rinex2_parser_preserves_real_gps_epochs():
    from scripts.lib.tep_dynamic import read_rinex3_multi

    path = (ROOT / "data" / "dynamic_datasets" / "ign_uav_flight8" /
            "rover" / "rov01840.18o")
    if not path.exists():
        pytest.skip("data/dynamic_datasets not downloaded")
    obs = read_rinex3_multi(path.read_text(encoding="latin-1"), systems=("G",))
    assert len(obs) >= 2500
    assert min(obs) <= 28320 and max(obs) >= 30850
    assert all(prn.startswith("G") for epoch in obs.values() for prn in epoch)
    assert np.median([len(epoch) for epoch in obs.values()]) >= 8


def test_uav_result_keeps_blind_boundary_and_reports_failure_as_failure():
    result = load("step_52_drone_kinematic.json")
    assert result["registered_design"]["systems"] == ["G"]
    assert result["registered_design"]["test_day_hyperparameter_fit"] is False
    assert result["registered_design"]["truth_role"] == "scoring only"
    flight = result["datasets"][0]
    assert flight["doi"] == "10.5281/zenodo.4476489"
    assert flight["provenance_status"].startswith("verified_public_archive")
    summary = flight["summary"]
    assert summary["local_base_dgps_applied"] is True
    assert summary["per_k"]["k4"]["err_free_p50"] < 10.0
    assert summary["per_k"]["k3"]["n_free_valid"] == 0
    assert summary["per_k"]["k3"]["n_tep_valid"] > 0
    # Numerical feasibility is not silently promoted to operational success.
    assert summary["per_k"]["k3"]["err_tep_p50"] > 10.0
    assert summary["inference_unit"].startswith("one flight")


def test_static_site_inserts_components_literally():
    source = (ROOT / "site" / "build.js").read_text()
    assert "() => components" in source


def test_colocated_anchor_inference_is_station_clustered_and_keeps_adverse_site():
    result = load("step_77_colocated_screening_differential.json")
    assert result["verdict"]["n_primary_free_atomic_anchors"] == 3
    assert result["verdict"]["metadata_audit_status"] == "PASS"
    assert result["verdict"]["internal_oscillator_material_confirmed"] is False
    assert set(result["verdict"]["primary_sites_showing_reference_class_contrast"]) == {
        "MAT1-MATE", "THTI-THTG"}
    assert set(result["verdict"]["primary_sites_replicating_in_both_temporal_halves"]) == {
        "MAT1-MATE", "THTI-THTG"}
    for site in ("MAT1-MATE", "THTI-THTG"):
        null = result["arms"][site]["primary_geometry_null"]
        assert null["unit"].startswith("distant reference station")
        assert null["n_realisations"] == 1000
        assert null["p_R2_ge_observed"] <= 1.0 / 1001.0
    assert not result["arms"]["WUH2-WUHN"]["anchor_metadata"]["primary_anchor"]
    assert not result["arms"]["STR2-STR1"]["delta_couples_to_distant_field"]


def test_clock_metadata_taxonomy_does_not_invent_quartz_material():
    audit = load("archive/step_82_clock_metadata_audit.json")
    assert audit["status"] == "PASS"
    assert audit["stations"]["MAT1"]["reference_class"] == "INTERNAL_RECEIVER_STANDARD"
    assert audit["stations"]["MAT1"]["oscillator_material"] == "UNSPECIFIED"
    assert audit["stations"]["MATE"]["reference_class"] == "EXTERNAL_ATOMIC"
    assert audit["stations"]["WUHN"]["reference_class"] == "NAVIGATION_STEERED_ATOMIC"
    assert audit["pairs"]["MAT1-MATE"]["receiver_model_matched_through_2024"] is True
    assert audit["pairs"]["THTI-THTG"]["receiver_model_matched_through_2024"] is False


def test_clock_history_selection_uses_latest_overlapping_start():
    from scripts.utils.parse_igs_site_logs import get_clock_type_on_date

    entries = [
        {"type": "INTERNAL", "start": "1998-01-01", "end": "CCYY-MM-DD"},
        {"type": "EXTERNAL RUBIDIUM", "start": "2001-01-01", "end": "2010-01-01"},
        {"type": "INTERNAL", "start": "2010-01-01", "end": "CCYY-MM-DD"},
    ]
    assert get_clock_type_on_date(entries, "2005-01-01")["standard_type"] == "EXTERNAL RUBIDIUM"
    assert get_clock_type_on_date(entries, "2024-01-01")["classification"] == "INTERNAL_UNSPECIFIED"


def test_information_criterion_recognises_large_exponential_rss_reduction():
    from scripts.lib.tep_stats import fit_all_models

    distance = np.linspace(100.0, 10_000.0, 12)
    value = 0.4 * np.exp(-distance / 3500.0) + 0.1
    fit = fit_all_models(distance, value, np.full(12, 20.0))
    assert fit["selection_criterion"] == "AICc"
    assert fit["best_model"] == "exponential"
    assert fit["models"]["exponential"]["R2"] > 0.99


def test_multimode_clock_field_localisation_beats_geometry_null():
    result = load("step_80_clock_field_localisation.json")
    limits = {"MAT1-MATE": 400.0, "THTI-THTG": 800.0}
    for site, limit in limits.items():
        fit = result["sites"][site]["multimode_fits"]["exponential"]["all_modes"]
        assert fit["surface_error_km"] < limit
        assert fit["geometry_null"]["p_localisation_error_le_observed"] < 0.05
    mate = result["sites"]["MAT1-MATE"]["multimode_fits"]["exponential"]
    assert mate["odd_even_estimate_separation_km"] < 100.0


def test_pride_kin_parser_accepts_windowed_quality_flag():
    from scripts.steps.step_62_ppp_convergence import _kin_xyz
    full = "60409      0.00    4331295.7179   567557.2520  4633132.9488".split()
    flagged = ("60409  51300.00 *  4331295.3691   567555.4384  "
               "4633133.2568").split()
    a, b = _kin_xyz(full), _kin_xyz(flagged)
    assert a is not None and b is not None
    assert abs(a[0] - 4331295.7179) < 1e-4
    assert abs(b[0] - 4331295.3691) < 1e-4


def test_step44_fixed_kernel_is_canonical_tep():
    source = (ROOT / "scripts" / "steps" / "step_44_raw_thin_position.py").read_text()
    assert "TEP_FIXED_LAMBDA_KM = 4200.0" in source
    assert "TEP_FIXED_NUGGET = 0.35" in source
    assert "TEP_FIXED_LAMBDA_KM = 3480.0" not in source


def test_ppp_prior_uses_physics_fixed_tep_kernel():
    """Step 22 must ingest the registered TEP kernel, not a nested-CV conventional fit."""
    from scripts.lib import tep_kernel as tk
    from scripts.steps.step_22_make_clock_prior import FITTED_PARAMS

    assert FITTED_PARAMS.lambda_km == tk.LAMBDA_CANONICAL_KM
    assert FITTED_PARAMS.nugget == 0.35
    assert FITTED_PARAMS.form == "gaussian"
    assert FITTED_PARAMS.aniso_mode == "none"
    source = (ROOT / "scripts" / "steps" / "step_22_make_clock_prior.py").read_text()
    assert "lambda_km=2500" not in source
    assert 'aniso_mode="cmb"' not in source


def test_kge4_clock_prior_is_not_inflated_to_a_noop():
    """A fixed 10 m floor at k>=4 made a ~0.7 m TEP readout a no-op against
    0.3 m code.  The DOP-adaptive inflation (INERT_FACTOR * clock_dop *
    SIGMA_CODE) is principled: it scales with the geometric clock-observation
    weight, not a fixed floor, and makes the prior inert (not a no-op) by
    ensuring the prior weight is a negligible fraction of the geometric
    clock-observation weight."""
    source = (ROOT / "scripts" / "steps" / "step_44_raw_thin_position.py").read_text()
    assert "prior_sig * 5.0" not in source
    # The DOP-adaptive inflation uses max(prior_sig, INERT_FACTOR * sigma_obs)
    # which is principled (scales with clock DOP), not a fixed floor.
    assert "_clock_dop" in source
    assert "INERT_FACTOR" in source
    assert "sigma_obs" in source


def test_p1_audit_scores_arsenal_versus_tuned_conventional():
    """P1 is the arsenal vs LOO-tuned conventional, not the weaker phase-cos-only tie."""
    source = (ROOT / "scripts" / "audit_operational_evidence.py").read_text()
    assert "full_arsenal_vs_tuned_conventional_pct" in source
    assert '-0.3%' not in source
    assert "step_92_independent_year_p1.json" in source
