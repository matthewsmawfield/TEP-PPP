#!/usr/bin/env python3
"""
Pipeline audit generator.

Produces PIPELINE_AUDIT.md in the repository root: a per-step matrix scoring
every pipeline step (active and archived) against a fixed rubric.  Objective
columns are computed from the filesystem and source files; subjective columns
(TEP success, manuscript importance, archive recommendation) are taken from a
curated annotation table below, cross-referenced against EVIDENCE_STATUS.md and
the orchestrator's --list role descriptions.

Rubric (per step):
  in_orch    -- appears in `python scripts/run_pipeline.py --list` (the
                orchestrated reproduction path).  Y/N.
  has_json   -- at least one results/outputs/<step>*.json|.npz file exists. Y/N.
  has_log    -- at least one logs/<step>*.log file exists.  Y/N.
  docstring  -- module docstring quality. 0=none, 1=stub, 2=adequate
                (question+method+I/O), 3=exemplary (full rationale+controls).
  comments   -- inline explanatory comment density (explanatory lines / 100
                source lines).  0=<0.5, 1=0.5-1.5, 2=1.5-3, 3=>3.
  logging    -- runtime logging/progress verbosity. 0=none, 1=print only,
                2=structured print+summary, 3=rich progress/JSON telemetry.
  tep_success-- evidence verdict for TEP.  PASS / PARTIAL / DIAGNOSTIC / DEBUG /
                NULL / N-A / SUPERSEDED / ARCHIVED.
  in_ms      -- referenced in site/components/*.html.  Y/N.
  ms_import  -- manuscript importance. Critical / Supporting / Minor / None.
  archive    -- recommendation. Keep / Archive / Candidate / Already archived.

Re-run:  python scripts/audit_pipeline.py
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"
ARCHIVE_DIR = STEPS_DIR / "archive"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
COMPONENTS_DIR = PROJECT_ROOT / "site" / "components"
AUDIT_PATH = PROJECT_ROOT / "PIPELINE_AUDIT.md"


# --------------------------------------------------------------------------
# Subjective annotations.
#
# These are curated from EVIDENCE_STATUS.md (gate verdicts), the orchestrator
# --list role descriptions, and a read of each step's docstring.  Only steps
# that carry a non-default verdict / importance / recommendation need to be
# listed; everything else falls back to the defaults below.
#
# Keys are the step id without the .py suffix, e.g. "step_10_global_correlation".
# --------------------------------------------------------------------------
DEFAULT_TEP_SUCCESS = "N-A"
DEFAULT_MS_IMPORT = "None"
DEFAULT_ARCHIVE = "Keep"

# (tep_success, ms_import, archive_rec)
ANNOTATIONS: dict[str, tuple[str, str, str]] = {
    # --- Acquisition / processing (foundational, not evidence per se) ---
    # step_01/02/03_kernel/05 live in the untracked
    # archive/early_superseded/ tree, outside this audit's scope, and are
    # retained in the orchestrator as optional (skip-if-absent) steps.
    # step_08_integrity_monitoring was recovered from its orphaned bytecode
    # (the source had been lost while the committed output JSON remained);
    # it produces the Step 08 scorecard result, so it is audited as evidence.
    "step_08_integrity_monitoring": ("PASS", "Supporting", "Keep"),
    "step_09_acquire_global_clk": ("N-A", "Supporting", "Keep"),
    "step_09b_acquire_sinex": ("N-A", "Supporting", "Keep"),
    "step_29_fetch_leo": ("N-A", "Minor", "Candidate"),
    "step_61_acquire_regional_cors": ("N-A", "Supporting", "Keep"),
    "step_63_acquire_kinematic": ("N-A", "Supporting", "Keep"),
    "step_72_acquire_uav_benchmark": ("N-A", "Minor", "Candidate"),
    "step_78_acquire_wide_clock_network": ("N-A", "Supporting", "Keep"),

    # --- Phase-1 core evidence (T1-T4 + falsification) ---
    "step_10_global_correlation": ("PASS", "Critical", "Keep"),
    "step_11_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_11b_anisotropy_csd": ("PASS", "Supporting", "Keep"),
    "step_12_resolvability": ("PASS", "Supporting", "Keep"),
    "step_13_commonview": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_14_precise_global": ("PASS", "Critical", "Keep"),
    "step_15_csd_crosscheck": ("PASS", "Supporting", "Keep"),
    "step_16_draconitic_falsification": ("PASS", "Supporting", "Keep"),
    "step_03b_controls": ("PASS", "Supporting", "Keep"),
    "step_03c_precise_controls": ("PASS", "Supporting", "Keep"),
    "step_05b_longitudinal_both": ("PASS", "Minor", "Candidate"),
    "step_06_falsification": ("PASS", "Supporting", "Keep"),
    "step_07_urban_canyon": ("DIAGNOSTIC", "Minor", "Candidate"),

    # --- Operational core (P1/P2/P3 + kernel selection) ---
    "step_19_phase1_report": ("PASS", "Critical", "Keep"),
    "step_20_operational_ekf": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_21_position_information": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_22_make_clock_prior": ("N-A", "Supporting", "Keep"),
    "step_23_pride_endtoend": ("PASS", "Critical", "Keep"),
    "step_24_degraded_geometry": ("PASS", "Supporting", "Keep"),
    "step_25_kernel_selection": ("PASS", "Critical", "Keep"),
    "step_26_operational_gain": ("PASS", "Critical", "Keep"),
    "step_27_discriminators": ("PASS", "Supporting", "Keep"),
    "step_28_publication_figures": ("N-A", "Critical", "Keep"),
    "step_32_kernel_family": ("PASS", "Supporting", "Keep"),
    "step_34_heldout_lodo": ("PASS", "Critical", "Keep"),
    "step_38_baseline_battery": ("PASS", "Critical", "Keep"),
    "step_33_position_demo": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_39_endtoend_position": ("PASS", "Critical", "Keep"),
    "step_40_bandpass_sensitivity": ("PASS", "Critical", "Keep"),
    "step_41_global_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_42_global_t5": ("PASS", "Supporting", "Keep"),
    "step_43_free_gp_validation": ("PASS", "Critical", "Keep"),
    "step_44_raw_thin_position": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_45_temporal_t4": ("PARTIAL", "Critical", "Keep"),
    "step_45b_t4_forensic": ("PARTIAL", "Critical", "Keep"),
    "step_46_hybrid_tep_knn": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_47_continuous_azimuth_t2": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_48_three_satellite": ("PASS", "Critical", "Keep"),
    "step_48b_gap_sweep": ("PASS", "Supporting", "Keep"),
    "step_48_rejection_forensics": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_49_p1_reframe": ("PASS", "Minor", "Candidate"),
    "step_50_lambda_meta_analysis": ("PASS", "Supporting", "Keep"),
    "step_51_dynamic_urban": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_51_tuning_tax": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_52_drone_kinematic": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_52_extrapolation_regime": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_53_coherent_lambda": ("PASS", "Critical", "Keep"),
    "step_54_dense_operational_skill": ("SUPERSEDED", "Supporting", "Keep"),
    "step_55_dense_outage_prediction": ("PASS", "Critical", "Keep"),
    "step_55_dense_position_gain": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_56_phase_predictive_bridge": ("PASS", "Critical", "Keep"),
    "step_57_joint_estimator_sensitivity": ("PASS", "Critical", "Keep"),
    "step_58_clock_field_lengthscales": ("PASS", "Critical", "Keep"),
    "step_59_common_mode_is_tep": ("PASS", "Critical", "Keep"),
    "step_60_operational_benefits": ("PASS", "Critical", "Keep"),
    "step_62_ppp_convergence": ("PASS", "Critical", "Keep"),
    "step_64_maximal_tep_p1": ("PASS", "Critical", "Keep"),
    "step_65b_drift_decomposition": ("PASS", "Supporting", "Keep"),
    "step_65_oscillator_screening": ("PASS", "Critical", "Keep"),
    "step_66_clock_gradient_holygrail": ("PASS", "Critical", "Keep"),
    "step_67_short_baseline_field": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_68_synchronization_holonomy": ("DIAGNOSTIC", "Critical", "Keep"),
    "step_69_sidereal_multipath_separation": ("PASS", "Supporting", "Keep"),
    "step_70_quartz_outage_clock": ("PASS", "Supporting", "Keep"),
    "step_70_sidereal_short_baseline_field": ("DIAGNOSTIC", "None", "Archived"),
    "step_71b_colocated_tep_anchor": ("PASS", "Critical", "Keep"),
    "step_71_station_field_decomposition": ("PASS", "Supporting", "Keep"),
    "step_72_housing_screening": ("PASS", "Critical", "Keep"),
    "step_72b_sidereal_template_blindness": ("PASS", "Critical", "Keep"),
    "step_73_phase_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_73b_satellite_unwrap_screening": ("PASS", "Supporting", "Keep"),
    "step_73_uav_failure_localisation": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_74c_comprehensive_clocks": ("PASS", "Supporting", "Keep"),
    "step_74_short_baseline_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_74b_anisotropy_pair_clustered": ("PASS", "Supporting", "Keep"),
    "step_75_screening_organizing_principle": ("PASS", "Critical", "Keep"),
    "step_75b_common_mode_analysis": ("PASS", "Supporting", "Keep"),
    "step_75c_screening_lambda_proper": ("PASS", "Supporting", "Keep"),
    "step_76_screening_anisotropy_null": ("PASS", "Supporting", "Keep"),
    "step_77_colocated_screening_differential": ("PASS", "Critical", "Keep"),
    "step_79_azel_holonomy": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_80_clock_field_localisation": ("PASS", "Critical", "Keep"),
    "step_80b_colocated_screening_wide": ("PASS", "Supporting", "Keep"),
    "step_80_plot_clock_field_localisation": ("N-A", "Supporting", "Keep"),
    "step_81_spherical_structure": ("PASS", "Supporting", "Keep"),
    "step_82_anisotropy_pair_permutation": ("PASS", "Supporting", "Keep"),
    "step_82_clock_metadata_audit": ("PASS", "Minor", "Archived"),
    "step_83_t4_phasecos": ("PASS", "Critical", "Keep"),
    "step_84_quadrupole_axis": ("PASS", "Supporting", "Keep"),
    "step_85_network_density_robustness": ("PASS", "Supporting", "Keep"),
    "step_86_angular_kernel_operational": ("PASS", "Supporting", "Keep"),
    "step_87_prewindow_sensitivity": ("PASS", "Supporting", "Keep"),
    "step_88b_band_decomposed_kernel": ("PASS", "Supporting", "Keep"),
    "step_88_per_station_breakdown": ("PASS", "Supporting", "Keep"),
    "step_89_seasonal_axis": ("PASS", "Supporting", "Keep"),
    "step_90_operational_holdover_spec": ("PASS", "Critical", "Keep"),
    "step_90b_physics_kernel_stack": ("PASS", "Supporting", "Keep"),
    "step_91_tep_field_is_the_clock": ("PASS", "Critical", "Keep"),
    "step_92_independent_year_p1": ("PARTIAL", "Critical", "Keep"),
    "step_93_fullsky_ppp_clock": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_94_satellite_clock_prediction": ("PASS", "Supporting", "Keep"),
    "step_95_phase_resolved_v2": ("DIAGNOSTIC", "Minor", "Candidate"),
    "step_96_leftover_clock_supremacy": ("PASS", "Critical", "Keep"),
    "step_97_multi_year_seasonal_lever": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_98_satellite_channel_flagship": ("PASS", "Critical", "Keep"),
    "step_99_precise_clock_position": ("PASS", "Critical", "Keep"),
    "step_100_decisive_drone_experiment": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_101_full_field_predictor": ("PASS", "Critical", "Keep"),
    "step_102_causal_deployable_p1_gate": ("PARTIAL", "Critical", "Keep"),
    "step_103_atomic_quartz_gradients": ("PASS", "Critical", "Keep"),
    "step_104_leo_moving_platform": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_104b_seasonal_regional_sweep": ("PASS", "Critical", "Keep"),
    "step_105_gracefo_moving_platform": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_106_degraded_sky_position_flagship": ("PASS", "Critical", "Keep"),

    # --- Multi-year / multi-centre replication & live-observatory arm ---
    "step_137_aperture_synthesis": ("PASS", "Supporting", "Keep"),
    "step_140_multiband_controls": ("PASS", "Critical", "Keep"),
    "step_142_multi_year": ("PASS", "Supporting", "Keep"),
    "step_146_constellation_8h_clocktype": ("PASS", "Supporting", "Keep"),
    "step_153_cross_product": ("PASS", "Critical", "Keep"),
    "step_164_fullyear_differential_spectrometer": ("PASS", "Critical", "Keep"),
    "step_167_clocktype_gravitational_phase": ("PASS", "Supporting", "Keep"),
    "step_168_live_network_tep_evidence": ("PASS", "Critical", "Keep"),
    "step_169_corrected_gravitational_phase": ("PASS", "Critical", "Keep"),
    "step_170_clock_field_localisation_live": ("PASS", "Critical", "Keep"),
    "step_171_ztd_covariance_control": ("PASS", "Supporting", "Keep"),
    "step_172_satellite_free_holdover": ("N-A", "Supporting", "Keep"),
    "step_173_twstft_screen": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_174_sealed_envelope_prediction": ("PASS", "Critical", "Keep"),
    "step_174b_clean_untapered_arm": ("PASS", "Supporting", "Keep"),
    "step_176_t2_alpha_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_125_notched_elevation": ("PASS", "Supporting", "Keep"),
    "step_127_notched_prior": ("PASS", "Supporting", "Keep"),
    "step_158_screening_replication": ("PASS", "Supporting", "Keep"),
    "step_160_freq_clocktype": ("PASS", "Supporting", "Keep"),
    "step_65c_oscillator_noise_null": ("PASS", "Supporting", "Keep"),

    # --- NTRIP live-stream arm (P4 PASS, 2026-10-16 blind capture) ---
    "step_37_ntrip_acquire": ("PASS", "Supporting", "Keep"),
    "step_37b_ntrip_process": ("PASS", "Supporting", "Keep"),
    "step_37c_ntrip_report": ("PASS", "Supporting", "Keep"),

    # --- MEO/satellite-side exploratory arm ---
    "step_30_meo_kriging": ("PASS", "Supporting", "Keep"),
    "step_31_meo_anisotropy": ("PASS", "Supporting", "Keep"),
    "step_33_meo_signal": ("PASS", "Minor", "Candidate"),
    "step_34b_meo_gradient": ("PASS", "Supporting", "Keep"),
    "step_34_meo_validation": ("PASS", "Minor", "Candidate"),
    "step_35_position_lodo": ("DIAGNOSTIC", "Supporting", "Keep"),
    "step_35_tep_ppp_correction": ("DIAGNOSTIC", "None", "Archived"),
    "step_36_meo_csd": ("PASS", "Critical", "Keep"),

    # --- Superseded / exploratory duplicates (archive candidates) ---
    "step_41_t4_matched": ("SUPERSEDED", "Minor", "Candidate"),
    "step_32_reverse_engineer": ("SUPERSEDED", "Minor", "Candidate"),
}


# Artefact/citation aliases: steps whose outputs or manuscript citations
# use a stem different from the script name (e.g. variant scripts writing
# to the flagship's output, or figure-only artefacts).
ARTEFACT_ALIASES: dict[str, list[str]] = {
    "step_08_integrity_monitoring": ["step_08_integrity",
                                     "step_12_integrity"],
    "step_19_phase1_report": ["phase1_gate"],
    "step_165_gravitational_phase_test": ["step_165_gravitational_phase"],
    "step_174b_clean_untapered_arm": ["step_174_clean_untapered_arm"],
    "step_73b_satellite_unwrap_screening": ["step_73b_satellite_unwrap"],
    "step_80_plot_clock_field_localisation": [
        "step_80_clock_field_localisation"],
}


def orchestrator_steps() -> set[str]:
    out = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--list"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    steps = set()
    for line in out.stdout.splitlines():
        m = re.match(r"(step_\d+[a-z]*_\w+)\s{2,}", line)
        if m:
            steps.add(m.group(1))
    return steps


def output_prefixes() -> set[str]:
    names = {p.name for p in OUTPUTS_DIR.iterdir() if p.is_file()}
    fig_dir = PROJECT_ROOT / "results" / "figures"
    if fig_dir.is_dir():
        names |= {p.name for p in fig_dir.iterdir() if p.is_file()}
    return names


def log_prefixes() -> set[str]:
    return {p.name for p in LOGS_DIR.iterdir() if p.is_file()}


def manuscript_step_refs() -> dict[str, int]:
    """Count references to each step id in site/components/*.html."""
    refs: dict[str, int] = {}
    pattern = re.compile(r"step_\d+[a-z]*_[a-z_]+")
    for comp in COMPONENTS_DIR.glob("*.html"):
        text = comp.read_text(errors="ignore")
        for m in pattern.finditer(text):
            refs[m.group(0)] = refs.get(m.group(0), 0) + 1
    return refs


def score_docstring(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    ds = ast.get_docstring(tree)
    if not ds:
        return 0
    n = len(ds)
    has_qm = bool(re.search(r"(question|method|procedure|strategy)", ds, re.I))
    has_io = bool(re.search(r"(input|output|produces|writes|stores)", ds, re.I))
    has_ctl = bool(re.search(r"(null|control|falsif|baseline|guard|leakage)", ds, re.I))
    score = 1
    if n > 400:
        score = 2
    if n > 1200 and has_qm and has_io:
        score = 3
    elif has_qm and has_io and n > 600:
        score = 2
    if score == 2 and has_ctl and n > 1000:
        score = 3
    return score


def score_comments(source: str) -> int:
    lines = source.splitlines()
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    n_code = max(len(code_lines), 1)
    comment_lines = sum(
        1 for l in lines
        if l.strip().startswith("#") and not l.strip().startswith("#!")
    )
    density = comment_lines / (n_code / 100.0)
    if density < 0.5:
        return 0
    if density < 1.5:
        return 1
    if density < 3.0:
        return 2
    return 3


def score_logging(source: str) -> int:
    has_print = "print(" in source
    has_log = bool(re.search(r"\blogging\b|\bgetLogger\b|log\.(info|debug|warn)", source))
    has_tqdm = "tqdm" in source
    has_json_dump = "json.dump" in source or "json.dumps" in source
    has_summary = bool(re.search(r"summary|ledger|report|telemetry", source, re.I))
    score = 0
    if has_print or has_log:
        score = 1
    if has_json_dump and (has_summary or has_log):
        score = 2
    if has_tqdm or (has_json_dump and has_summary and has_log):
        score = 3
    return score


def first_sentence(docstring: str | None) -> str:
    if not docstring:
        return ""
    # Take the first non-blank line, then trim to first sentence.
    for line in docstring.splitlines():
        line = line.strip()
        if line:
            break
    else:
        return ""
    line = re.sub(r"\s+", " ", line)
    # cut at first period followed by space/end, or first newline-group
    m = re.match(r"(.*?[.:])", line)
    return (m.group(1) if m else line).strip()


def role_description(step_id: str, orch: dict[str, str]) -> str:
    return orch.get(step_id, "")


def load_orchestrator_roles() -> dict[str, str]:
    out = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--list"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    roles: dict[str, str] = {}
    for line in out.stdout.splitlines():
        m = re.match(r"(step_\d+[a-z]*_\w+)\s{2,}(.+)$", line)
        if m:
            roles[m.group(1)] = m.group(2).strip()
    return roles


def gather_steps() -> list[tuple[str, Path, bool]]:
    """Return (step_id, path, is_archived) for every step file."""
    steps = []
    for p in sorted(STEPS_DIR.glob("step_*.py")):
        steps.append((p.stem, p, False))
    for p in sorted(ARCHIVE_DIR.glob("step_*.py")):
        steps.append((p.stem, p, True))
    return steps


def main() -> int:
    orch = orchestrator_steps()
    roles = load_orchestrator_roles()
    outputs = output_prefixes()
    logs = log_prefixes()
    ms_refs = manuscript_step_refs()

    steps = gather_steps()

    # Build rows
    rows = []
    for step_id, path, is_archived in steps:
        source = path.read_text(errors="ignore")
        try:
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)
        except SyntaxError:
            docstring = None

        in_orch = step_id in orch
        # JSON/log presence: match any file whose name starts with step_id
        # or with a declared artefact alias (variant/renamed outputs).
        stems = [step_id, *ARTEFACT_ALIASES.get(step_id, [])]
        has_json = any(
            name.startswith(stem) for name in outputs for stem in stems)
        has_log = any(
            name.startswith(stem) for name in logs for stem in stems)
        doc_score = score_docstring(source)
        com_score = score_comments(source)
        log_score = score_logging(source)

        tep_success, ms_import, archive_rec = ANNOTATIONS.get(
            step_id, (DEFAULT_TEP_SUCCESS, DEFAULT_MS_IMPORT, DEFAULT_ARCHIVE),
        )
        if is_archived:
            tep_success = "ARCHIVED"
            archive_rec = "Already archived"
            # The annotation refers to the active version; archived copies do
            # not feed the manuscript.
            ms_import = "None"

        ms_count = ms_refs.get(step_id, 0) + sum(
            ms_refs.get(stem, 0) for stem in stems[1:])
        in_ms = "Y" if ms_count else "N"
        # Promote ms_import to None if not in manuscript and not annotated critical
        if in_ms == "N" and step_id not in ANNOTATIONS:
            ms_import = "None"

        purpose = role_description(step_id, roles) or first_sentence(docstring)
        purpose = purpose.replace("|", "\\|")
        if len(purpose) > 90:
            purpose = purpose[:87] + "..."

        rows.append({
            "step_id": step_id,
            "purpose": purpose,
            "in_orch": "Y" if in_orch else "N",
            "has_json": "Y" if has_json else "N",
            "has_log": "Y" if has_log else "N",
            "doc": doc_score,
            "com": com_score,
            "log": log_score,
            "tep": tep_success,
            "in_ms": in_ms,
            "ms_n": ms_count,
            "ms_imp": ms_import,
            "archive": archive_rec,
            "is_archived": is_archived,
        })

    # Sort: active first (by numeric step), then archived
    def sort_key(r):
        m = re.match(r"step_(\d+)([a-z]?)_", r["step_id"])
        num = int(m.group(1)) if m else 999
        suffix = m.group(2) if m else ""
        return (r["is_archived"], num, suffix, r["step_id"])

    rows.sort(key=sort_key)

    # Summary stats
    n_active = sum(1 for r in rows if not r["is_archived"])
    n_archived = sum(1 for r in rows if r["is_archived"])
    n_in_orch = sum(1 for r in rows if r["in_orch"] == "Y")
    n_in_ms = sum(1 for r in rows if r["in_ms"] == "Y")
    n_pass = sum(1 for r in rows if r["tep"] == "PASS")
    n_critical = sum(1 for r in rows if r["ms_imp"] == "Critical")
    n_archive_cand = sum(1 for r in rows if r["archive"] == "Candidate")

    # Write markdown
    lines = []
    lines.append("# TEP-PPP pipeline audit")
    lines.append("")
    lines.append(
        "A per-step audit of every script in `scripts/steps/` (active) and "
        "`scripts/steps/archive/` against a fixed rubric.  Objective columns "
        "(`in_orch`, `has_json`, `has_log`, `doc`, `com`, `log`, `in_ms`, "
        "`ms_n`) are computed from the filesystem and source by "
        "`scripts/audit_pipeline.py`; subjective columns (`tep`, `ms_imp`, "
        "`archive`) are curated against `EVIDENCE_STATUS.md` and the "
        "orchestrator role descriptions.  Re-generate with "
        "`python scripts/audit_pipeline.py`."
    )
    lines.append("")
    lines.append("## Rubric")
    lines.append("")
    lines.append("| Column | Meaning |")
    lines.append("|---|---|")
    lines.append(f"| `in_orch` | Listed in `python scripts/run_pipeline.py --list` (the {len(orch)}-step orchestrated path; absent optional scripts skip cleanly). Y/N. |")
    lines.append("| `has_json` | At least one `results/outputs|figures/<step>*` artefact exists (incl. declared aliases). Y/N. |")
    lines.append("| `has_log` | At least one `logs/<step>*.log` file exists. Y/N. |")
    lines.append("| `doc` | Module docstring quality. 0 none, 1 stub, 2 adequate (question+method+I/O), 3 exemplary (+controls/nulls). |")
    lines.append("| `com` | Inline explanatory comment density (comments per 100 source lines). 0 <0.5, 1 0.5-1.5, 2 1.5-3, 3 >3. |")
    lines.append("| `log` | Runtime logging/progress verbosity. 0 none, 1 print only, 2 structured + JSON summary, 3 rich progress/telemetry. |")
    lines.append("| `tep` | Evidence verdict for TEP. PASS / PARTIAL / DIAGNOSTIC / DEBUG / NULL / N-A / SUPERSEDED / ARCHIVED. |")
    lines.append("| `in_ms` | Referenced in `site/components/*.html` (the manuscript source). Y/N. |")
    lines.append("| `ms_n` | Number of manuscript references to this step id. |")
    lines.append("| `ms_imp` | Manuscript importance. Critical / Supporting / Minor / None. |")
    lines.append("| `archive` | Recommendation. Keep / Candidate (archive-eligible) / Already archived. |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Active step scripts: **{n_active}**")
    lines.append(f"- Archived step scripts: **{n_archived}**")
    lines.append(f"- In the active orchestrator (`--list`): **{n_in_orch}**")
    lines.append(f"- Referenced in the manuscript: **{n_in_ms}**")
    lines.append(f"- Verdict PASS: **{n_pass}**")
    lines.append(f"- Manuscript-critical: **{n_critical}**")
    lines.append(f"- Archive candidates: **{n_archive_cand}**")
    lines.append("")
    lines.append("## Active pipeline")
    lines.append("")
    lines.append("| Step | Purpose | orch | json | log | doc | com | log | tep | ms | ms_n | ms_imp | archive |")
    lines.append("|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|:-:|:-:|---|---|")
    for r in rows:
        if r["is_archived"]:
            continue
        lines.append(
            f"| `{r['step_id']}` | {r['purpose']} | {r['in_orch']} | {r['has_json']} | "
            f"{r['has_log']} | {r['doc']} | {r['com']} | {r['log']} | {r['tep']} | "
            f"{r['in_ms']} | {r['ms_n']} | {r['ms_imp']} | {r['archive']} |"
        )
    lines.append("")
    lines.append("## Archived pipeline (`scripts/steps/archive/`)")
    lines.append("")
    lines.append(
        "These scripts are retained for provenance but are not part of the "
        "active pipeline.  They are listed here for completeness; their "
        "outputs (where present) live under `results/outputs/archive/`."
    )
    lines.append("")
    lines.append("| Step | Purpose | json | log | doc | com | log | tep | ms | ms_n | ms_imp | archive |")
    lines.append("|---|---|:-:|:-:|:-:|:-:|:-:|---|:-:|:-:|---|---|")
    for r in rows:
        if not r["is_archived"]:
            continue
        lines.append(
            f"| `{r['step_id']}` | {r['purpose']} | {r['has_json']} | {r['has_log']} | "
            f"{r['doc']} | {r['com']} | {r['log']} | {r['tep']} | {r['in_ms']} | "
            f"{r['ms_n']} | {r['ms_imp']} | {r['archive']} |"
        )
    lines.append("")
    lines.append("## Archive recommendations")
    lines.append("")
    lines.append(
        "Steps marked **Candidate** below are exploratory duplicates, "
        "diagnostic-only arms, or superseded implementations whose evidence "
        "has been absorbed into a flagship step.  They are safe to move into "
        "`scripts/steps/archive/` once their outputs are confirmed captured "
        "by the flagship.  Steps marked **Keep** feed a manuscript claim, a "
        "gate in `EVIDENCE_STATUS.md`, or the orchestrator."
    )
    lines.append("")
    candidates = [r for r in rows if r["archive"] == "Candidate" and not r["is_archived"]]
    if candidates:
        lines.append("| Step | Purpose | TEP verdict | Manuscript | Reason |")
        lines.append("|---|---|---|---|---|")
        for r in candidates:
            reason = (
                "Diagnostic/exploratory arm; evidence absorbed by a flagship step"
                if r["tep"] in ("DIAGNOSTIC", "DEBUG")
                else "Superseded or minor; retained for provenance only"
            )
            lines.append(
                f"| `{r['step_id']}` | {r['purpose']} | {r['tep']} | "
                f"{r['in_ms']} ({r['ms_imp']}) | {reason} |"
            )
    else:
        lines.append("_No archive candidates._")
    lines.append("")

    # ------------------------------------------------------------------
    # Evidence clusters, supersession chains, and impact tiers.
    # These are curated analyses, not auto-computed.  They encode the
    # logical structure of the pipeline: which steps test the same claim,
    # which supersede which, and which are load-bearing for the manuscript.
    # ------------------------------------------------------------------
    lines.append("## Evidence clusters")
    lines.append("")
    lines.append(
        f"The {n_active} active steps cluster into ten logical evidence "
        "groups (the step lists retain archived members where they anchor "
        "a supersession chain).  Within each cluster, steps test the same "
        "physical claim from different angles; across clusters, the "
        "claims are independent.  The cluster structure reveals where the "
        "pipeline is over-built (many attempts at the same gate) and "
        "where it is thin (a single step carrying a claim)."
    )
    lines.append("")
    clusters = [
        ("C1. Phase-1 correlation physics (T1&ndash;T4)",
         "10, 11, 11b, 12, 13, 14, 15, 16, 53, 83",
         "The foundational physics: does the operational clock field reproduce "
         "the Papers 1&ndash;2 distance structure, anisotropy, resolvability, "
         "and common-view separability?  Step 53 (phase-alignment index, 27&sigma;) "
         "and Step 83 (T4 with the correct estimator) are the load-bearing "
         "results; Steps 10&ndash;16 are the amplitude-domain originals, "
         "retained for the broadcast vs precise arms and the falsification "
         "nulls."),
        ("C2. P1 gate: TEP-specific predictive superiority",
         "25, 38, 49, 64, 92, 96, 101, 102",
         "The most contested gate: does the physics-fixed TEP kernel beat a "
         "data-tuned conventional predictor out of sample?  Attempted eight "
         "times.  The current flagships are Step 96 (leftover-clock supremacy, "
         "frozen field beats persistence/linrate/AR1/free-GP/empirical-covar "
         "on held-out 2025) and Step 102 (causal deployable gate).  Steps 25 "
         "and 38 are earlier nested-CV attempts; Step 49 is a read-only "
         "re-aggregation of Step 38 (now archived).  Step 92 is the "
         "independent-year replication.  Step 101 is the full-field predictor "
         "at the TEP scale."),
        ("C3. Position-domain evidence (P2 / P3)",
         "23, 24, 26, 33, 39, 44, 48, 62, 99, 106",
         "Does the clock-domain advantage translate to positions?  A "
         "progressive rigour chain: Step 26 (analytical) &rarr; Step 33 "
         "(post-fit replay, diagnostic) &rarr; Step 39 (causally calibrated) "
         "&rarr; Step 44 (raw-observation) &rarr; Step 99 (precise-clock, "
         "single day) &rarr; Step 106 (multi-day, CI-backed flagship).  "
         "Step 23 shows TEP is correctly inert at full sky (the clock is "
         "already observable from geometry).  Step 48 is the three-satellite "
         "challenge.  Step 62 is the P3 PPP cold-start gate."),
        ("C4. Anisotropy (T2, EW/NS directional signature)",
         "11, 11b, 41, 47, 73, 74, 74b, 76, 82",
         "The most redundant cluster.  Progression: Step 11 (amplitude) "
         "&rarr; Step 11b (phase-cos) &rarr; Step 73 (corrected phase-cos).  "
         "Step 41 extends to the global precise network.  Step 47 (continuous "
         "azimuth) is retained as a diagnostic arm.  Steps 74, 74b, 76, "
         "82 are independent controls on the short-baseline anisotropy "
         "(pair-clustered bootstrap, screening null, pair-level permutation) "
         "&mdash; not redundant, but layered confirmation."),
        ("C5. Holonomy / non-exact sector",
         "68, 70, 72, 79",
         "Tests the distinctive TEP observable that a conformal-only scalar "
         "cannot produce: the closed-loop synchronization holonomy.  All "
         "DIAGNOSTIC: Step 68 (initial) found the holonomy null is dominated "
         "by multipath.  Step 72 proved the sidereal template is blind to "
         "the signal it should leave behind.  Step 79 fits per-station az/el "
         "multipath surfaces jointly with T (injection recovery 1.00 on all 8 "
         "pairs, zero-area controls null) and finds no area-scaling holonomy "
         "above the site-multipath-modelled floor.  Step 70 (short-baseline at "
         "multipath floor) is archived.  The non-exact sector is "
         "bounded by a powered estimator."),
        ("C6. MEO / satellite-side channel",
         "29, 30, 31, 33_meo, 34_meo, 34_meo_val, 36, 94, 98",
         "An independent channel: inter-satellite TEP structure.  Step 36 "
         "(CSD) is the flagship (constellation-independent, in manuscript).  "
         "Step 98 is the satellite-channel flagship (cross-spectral evidence).  "
         "Step 94 predicts satellite clocks from inter-satellite structure.  "
         "Steps 30/31 are earlier kriging/anisotropy attempts superseded by "
         "36.  Steps 33_meo and 34_meo_val have no docstrings (now "
         "archived).  Step 29 (LEO fetch) is archived."),
        ("C7. Dense-network outage / operational skill (S6)",
         "54, 55, 55_pos, 56, 85, 87, 88, 88_per",
         "Step 55 is the flagship (causal outage clock prediction, +5&ndash;19% "
         "vs persistence/temporal).  Step 56 adds the phase-cos causal bridge.  "
         "Steps 85, 87, 88 are robustness controls (density, pre-window, "
         "band-decomposed).  Step 54 is explicitly superseded by Step 55 "
         "but retained in the orchestrator.  Step 55_pos (position gain) "
         "is archived; Step 88_per is the per-station breakdown."),
        ("C8. Clock-field structure / common mode / screening",
         "58, 59, 65_osc, 65_drift, 71_field, 71_anchor, 75, 75b, 75c, 77, 80_loc, 80_wide, 91",
         "A large but mostly non-redundant cluster.  Step 58 (two length "
         "scales: phase-lock ~2,800 km vs coherence ~1,200 km), Step 59 "
         "(common mode IS TEP), Step 65 (oscillator screening: quartz carries "
         "1.9&times; more network-predictable power), Step 75 (screening as "
         "organizing principle), Step 77 (co-located differential), Step 80 "
         "(clock-field localisation: 43 modes localise Matera to 339 km), "
         "Step 91 (TEP field IS the receiver clock).  Steps 75b/75c are "
         "progressive refinements of the screening stratification.  Step "
         "80_wide extends the co-located design to six sites."),
        ("C9. NTRIP live stream (P4)",
         "37, 37b, 37c",
         "P4 PASS: a 9.5 h continuous overnight capture across 10 unseen "
         "Dutch/Portuguese stations (2026-10-16) was processed by the "
         "frozen obstruction protocol at all five registered windows; the "
         "contemporaneous leave-target-out kernel improves 3-D position "
         "by +2.1/+4.0/+5.2/+15.1% over persistence at 30/60/120/240 min."),
        ("C10. Moving platforms / drone / UAV",
         "52, 72_uav, 73_uav, 100, 104, 105",
         "Step 100 (decisive drone projection) is the flagship &mdash; a "
         "projection, not a live test.  Steps 104/105 (LEO/GRACE-FO moving "
         "platforms) are DIAGNOSTIC: the field does not follow a fast "
         "orbital receiver (Swarm is steered; GRACE-FO USO is free-running "
         "but the result is negative).  Step 52 is the exploratory drone "
         "kinematic arm; the 72_uav acquisition and 73_uav failure-"
         "localisation arms are archived."),
    ]
    lines.append("| Cluster | Steps | Assessment |")
    lines.append("|---|---|---|")
    for name, steps, assessment in clusters:
        lines.append(f"| {name} | {steps} | {assessment} |")
    lines.append("")

    # Supersession chains
    lines.append("## Supersession chains and redundancy")
    lines.append("")
    lines.append(
        "Where a later step renders an earlier step's evidence fully "
        "redundant, the chain is listed below.  The superseded step is not "
        "necessarily wrong &mdash; it is the earlier attempt whose result "
        "is absorbed by the flagship.  Steps already marked Candidate in "
        "the archive table are flagged here with the superseding step."
    )
    lines.append("")
    chains = [
        ("step_11_anisotropy", "step_11b_anisotropy_csd &rarr; step_73_phase_anisotropy",
         "Amplitude EW/NS &rarr; phase-cos EW/NS &rarr; corrected phase-cos.  Step 11 is in the orchestrator and manuscript (broadcast arm); Step 73 is the corrected flagship."),
        ("step_13_commonview", "step_83_t4_phasecos",
         "T4 common-view discriminator (DIAGNOSTIC, Pearson correlation) &rarr; T4 with the correct phase-cos estimator (PASS).  Step 13 is in the orchestrator; Step 83 resolves the S2 gate."),
        ("step_33_position_demo", "step_39_endtoend_position &rarr; step_106_degraded_sky_position_flagship",
         "Post-fit replay (DIAGNOSTIC) &rarr; causally calibrated replay (PASS) &rarr; multi-day CI-backed flagship (PASS).  Step 33 is in the orchestrator; Step 106 is the current flagship."),
        ("step_49_p1_reframe", "step_38_baseline_battery (read-only re-aggregation)",
         "Step 49 is explicitly a read-only re-aggregation of Step 38 outputs.  No new computation.  Archived."),
        ("step_54_dense_operational_skill", "step_55_dense_outage_prediction",
         "Retrospective density-scaling association (DIAGNOSTIC) &rarr; causal outage gate (PASS, S6).  The evidence ledger marks S5 as SUPERSEDED by S6.  Step 54 itself is retained in the orchestrator."),
        ("step_30_meo_kriging / step_31_meo_anisotropy", "step_36_meo_csd",
         "Earlier MEO kriging/anisotropy &rarr; phase-coherent CSD flagship.  Steps 30/31 are in the manuscript as the MEO exploration path; Step 36 is the constellation-independent result."),
        ("step_41_t4_matched", "step_83_t4_phasecos",
         "Matched-design T4 (SUPERSEDED) &rarr; correct phase-cos T4.  Archived."),
        ("step_99_precise_clock_position", "step_106_degraded_sky_position_flagship",
         "Single-day, no CI, units mismatch &rarr; multi-day, day-clustered CI, correct SPP construction.  Step 99 is in the manuscript; Step 106 is the bulletproof version.  Both kept."),
        ("step_68_synchronization_holonomy", "step_72b_sidereal_template_blindness &rarr; step_79_azel_holonomy",
         "Initial holonomy test (DIAGNOSTIC, multipath-dominated) &rarr; template blindness proof (PASS) &rarr; joint az/el multipath model (DIAGNOSTIC, injection recovery 1.00).  All kept: the chain bounds the non-exact sector below the site-multipath-modelled floor."),
    ]
    lines.append("| Superseded step | Superseded by | Note |")
    lines.append("|---|---|---|")
    for old, new, note in chains:
        lines.append(f"| `{old}` | {new} | {note} |")
    lines.append("")

    # Impact tiers
    lines.append("## Impact tiers")
    lines.append("")
    lines.append(
        "Steps ranked by how much weight they carry in the manuscript's "
        "argument.  Tier 1 steps are load-bearing: remove one and a gate "
        "or a section collapses.  Tier 2 steps provide independent "
        "confirmation or controls.  Tier 3 steps add robustness or "
        "sensitivity analysis.  Tier 4 steps are exploratory, diagnostic, "
        "or superseded."
    )
    lines.append("")
    tiers = [
        ("Tier 1 &mdash; Load-bearing flagships",
         "10, 14, 19, 23, 25, 26, 34, 38, 40, 48, 53, 55, 56, 57, 58, 59, "
         "60, 62, 64, 65_osc, 66, 68, 72_housing, 72_sidereal, 75, 77, 80_loc, "
         "83, 90_holdover, 91, 92, 96, 97, 98, 99, 101, 102, 103, 106",
         "Each of these steps feeds a PASS verdict on a gate in "
         "EVIDENCE_STATUS.md or carries a manuscript section.  The paper's "
         "central claims rest on: Step 53 (phase alignment, 27&sigma;), "
         "Step 64/96/102 (P1 gate), Step 55/56 (S6 outage gate), Step 62 "
         "(P3 PPP gate), Step 66 (scalar-sector bound), Step 77/80 "
         "(co-located anchor/localisation), Step 106 (degraded-sky "
         "position flagship)."),
        ("Tier 2 &mdash; Independent confirmation / controls",
         "03b, 03c, 05, 06, 11b, 12, 15, 16, 27, 32, 36, 43, 45, 45b, 50, "
         "65_drift, 69, 70_quartz, 71_field, 71_anchor, 73, 74, 74b, 75b, "
         "75c, 76, 80_wide, 81, 82, 84, 89, 94, 104, 105",
         "These do not carry a gate alone but provide the independent-product "
         "replication, null controls, or cross-checks that make the Tier 1 "
         "results credible.  Without them a reviewer could attribute a "
         "Tier 1 result to an estimator artefact or a single-product fluke."),
        ("Tier 3 &mdash; Robustness / sensitivity",
         "24, 41, 85, 86, 87, 88, 90_physics, 100",
         "Density sweeps, pre-window sweeps, band-decomposed kernels, "
         "angular-structure operational payoff, physics-kernel stacking, "
         "drone projection.  These show the Tier 1 result is not fragile "
         "to a hyperparameter choice, but are not themselves gates."),
        ("Tier 4 &mdash; Exploratory / diagnostic / superseded",
         "04, 07, 08, 20, 21, 29, 33, 33_meo, 34_meo_val, 35_pos, 35_tep, "
         "37, 37b, 37c, 39, 42, 44, 46, 47, 48_rej, 49, 51_dyn, 51_tax, "
         "52, 52_extrap, 54, 55_pos, 67, 70_sidereal, 72_uav, 73_uav, "
         "73_unwrap, 74_clocks, 79, 82_audit, 88_per, 93, 95",
         "Diagnostic-only arms, superseded implementations, exploratory "
         "sweeps, or data-acquisition utilities.  Many are already "
         "archived.  A few (Step 39, Step 44, Step 79, Step 93) "
         "are kept because they appear in the manuscript as part of a "
         "progressive-rigour narrative even though their verdict is "
         "DIAGNOSTIC."),
    ]
    lines.append("| Tier | Steps | Role |")
    lines.append("|---|---|---|")
    for name, steps, role in tiers:
        lines.append(f"| {name} | {steps} | {role} |")
    lines.append("")

    # Evidence strength summary
    lines.append("## Evidence strength summary")
    lines.append("")
    lines.append(
        "The pipeline's evidence is strong on the physics-identification "
        "axis and strong on the operational-clock axis; the residual open "
        "fronts are catalogued below."
    )
    lines.append("")
    lines.append("**Where the evidence is strong:**")
    lines.append("")
    lines.append(
        "- The phase-alignment scale (Step 53, 27&sigma;; Step 83, correct "
        "estimator) is the single strongest result in the pipeline and is "
        "independent of the operational gates.  It matches Papers 1&ndash;2 "
        "across two independent products (COD, GFZ)."
    )
    lines.append(
        "- The P1 gate (TEP-specific predictive superiority) passes on the "
        "frozen 2025 held-out year via Step 96 (leftover-clock supremacy) "
        "and Step 102 (causal deployable gate), after six earlier attempts. "
        " The convergence of multiple independent constructions on the same "
        "verdict is itself evidence, though the number of attempts also "
        "reflects how contested this gate is."
    )
    lines.append(
        "- The S6 outage gate (Step 55/56) passes with +5&ndash;19% clock "
        "RMS reduction vs persistence and temporal baselines, replicated "
        "across density (Step 85), pre-window (Step 87), and band-decomposed "
        "(Step 88) sweeps."
    )
    lines.append(
        "- The co-located anchor / clock-field localisation (Steps 77, 80) "
        "recovers the TEP length scale on two independent sites (Matera "
        "3,282 km, Tahiti 3,732 km, p=0.001 each) and localises Matera to "
        "339 km from a target-blind 43-mode inversion."
    )
    lines.append(
        "- The degraded-sky position flagship (Step 106) beats both "
        "free-clock SPP and a deployable persistence-trend prior by ~12%, "
        "day-clustered CI-backed, seven days."
    )
    lines.append("")
    lines.append("**Where the evidence is weak or open:**")
    lines.append("")
    lines.append(
        "- The non-exact (holonomy) sector is bounded below the site-"
        "multipath floor, not detected: Step 72 proved the sidereal template "
        "is blind to a static geometry-locked signal, while Step 79's joint "
        "az/el multipath estimator (injection recovery 1.00 on all 8 pairs, "
        "zero-area controls null) finds no area-scaling holonomy above site "
        "systematics."
    )
    # Pull the live clean-origin figure from the committed output rather
    # than hard-coding it: a literal goes stale the moment the sealed
    # replay is re-run on updated Grams.
    try:
        _s174 = json.loads(
            (OUTPUTS_DIR
             / "step_174_sealed_envelope_prediction.json").read_text())
        _cz = _s174["purged_epoch_disjoint_horizons"][
            "clean_origin_fit_half"]
        _fwd = (
            f"{_cz['one_step_sealed_stack_composite']['mean_pct']:+.2f}% "
            f"composite / "
            f"{_cz['one_step_sealed_pure_correction']['mean_pct']:+.2f}% "
            "pure at the deployed penalty")
        _wf = _s174["walk_forward_sealed_envelopes"][
            "full_causal_sealed"]["mean_pct"]
    except Exception:
        _fwd = "negative at the deployed penalty"
        _wf = None
    lines.append(
        "- The sealed-envelope forward component is bounded: the zero-"
        "overlap clean-origin arm (Step 174) transfers "
        f"{_fwd} at unseen epochs (rising to +2.75% at lambda=5e6, +2.57% "
        "under causal inner-CV holdout-Gram selection, and +3.39% untapered), "
        "so the "
        + (f"+{_wf:.2f}% " if _wf is not None else "")
        + "sealed headline verifies field-map fidelity (retrodiction of "
        "member-observed epochs) with a small, coherence-limited forward "
        "component.  The strictly post-seal live windows are accumulating "
        "separately."
    )
    lines.append(
        "- The S2 physics-separability gate (Step 45/45b) is PARTIAL: with "
        "the corrected pooled common-view slope on real-SP3 co-visibility, "
        "the broadcast arm meets the frozen criterion outright (PC=-0.164, "
        "10.8 sigma, OOS R^2 gain +0.022 > 0.02, lambda=4,349 km), while the "
        "precise arm shows the expected attenuation (PC=-0.040, 9.2 sigma) "
        "under common-mode absorption and atomic screening.  Physical "
        "identification of &lambda;<sub>T</sub> additionally rests on the "
        "phase observable (Step 53, 25-27 sigma)."
    )
    lines.append(
        "- The moving-platform boundary (Steps 104/105) is negative: the "
        "field does not transfer to a fast orbital receiver (Swarm is "
        "steered; GRACE-FO USO is free-running but the result is negative).  "
        "This is a boundary, not a failure, but it limits the deployable scope."
    )
    lines.append(
        "- The drone / UAV arm (Steps 52, 100) is a projection, not a live "
        "test.  Step 100 separates the learning-curve and density factors "
        "but the decisive demonstration requires a live dual-clock phase "
        "package with GNSS disabled after launch."
    )
    lines.append("")
    lines.append("**Redundancy assessment:**")
    lines.append("")
    lines.append(
        "The pipeline is over-built in three areas: (1) the P1 gate has "
        "been attempted eight times (Steps 25, 38, 49, 64, 92, 96, 101, 102), "
        "with Step 49 a pure re-aggregation and Steps 25/38 earlier "
        "nested-CV versions; (2) the anisotropy cluster has nine steps "
        "testing the same EW/NS signature with progressively corrected "
        "estimators; (3) the position-domain chain has six steps of "
        "progressive rigour (33&rarr;39&rarr;44&rarr;99&rarr;106).  In each "
        "case the redundancy reflects a genuine scientific progression "
        "(each step fixes a flaw in its predecessor), not wasted effort.  "
        "The earlier steps in each chain have now been retired: the "
        "candidate set identified in earlier revisions of this audit is "
        "archived under `scripts/steps/archive/`, and no active step is "
        "currently flagged as an archive candidate."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by `scripts/audit_pipeline.py`.  Re-run after adding or "
        "retiring steps to refresh the objective columns; update the "
        "`ANNOTATIONS` table in that script when evidence verdicts change."
    )
    lines.append("")

    AUDIT_PATH.write_text("\n".join(lines))
    print(f"Wrote {AUDIT_PATH.relative_to(PROJECT_ROOT)} "
          f"({n_active} active, {n_archived} archived, "
          f"{n_in_orch} in orchestrator, {n_archive_cand} archive candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
