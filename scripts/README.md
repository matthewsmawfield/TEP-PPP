# TEP-PPP Analysis Pipeline

The orchestrator (`scripts/run_pipeline.py`) and step suite (`scripts/steps/step_01` .. `step_180`) expose the full verification and reproduction workflow:

```bash
# Show the ordered orchestrated step list
python scripts/run_pipeline.py --list

# Full run, including data acquisition and processing
python scripts/run_pipeline.py

# Reuse data already present in data/
python scripts/run_pipeline.py --skip-data

# Fast code-path smoke run (skips network and heavy steps)
python scripts/run_pipeline.py --quick --skip-data

# Resume or isolate a specific step from --list
python scripts/run_pipeline.py --from-step step_25_kernel_selection --skip-data
python scripts/run_pipeline.py --only-step step_38_baseline_battery --skip-data
```

Do not delete committed outputs to perform verification. The non-destructive verification entry points compile the code, run the 157-test suite, regenerate the evidence and pipeline ledgers, and verify manuscript numbers against committed JSON artifacts:

```bash
# Fast manuscript + evidence ledger verification
python scripts/verify_manuscript.py

# Full non-destructive verification (compileall + pytest + evidence audit + manuscript pins)
python scripts/reproduce.py --verify-only

# Refresh the per-step pipeline audit table (PIPELINE_AUDIT.md)
python scripts/audit_pipeline.py

# Refresh the SHA-256 artifact checksum manifest
python scripts/utils/update_checksum_manifest.py
```

Key evidence and operational steps across the pipeline:

| Step(s) | Purpose | Primary Artifact(s) |
|---|---|---|
| `10`–`16`, `40` | Global T1–T5 correlation battery, draconitic falsification, and 5-filter robustness | `step_10_global_correlation.json`, `step_16_draconitic_falsification.json`, `step_40_bandpass_sensitivity.json` |
| `23`, `62` | Production PRIDE-PPP-AR state-coupling probe and $k=4$ GPS-thinned cold-start convergence (P3) | `step_23_pride_endtoend.json`, `step_62_ppp_degraded.json` |
| `25`, `34`, `38` | Nested-CV kernel selection, 318-day LODO, and training-selected conventional baseline battery | `step_25_kernel_selection.json`, `step_34_heldout_lodo.json`, `step_38_baseline_battery.json` |
| `36`, `98`, `178` | Orbiting MEO satellite-clock cross-spectral phase alignment and GPS×Galileo cross-constellation arm | `step_36_meo_csd.json`, `step_98_satellite_channel_flagship.json`, `step_178_satellite_scale_reconciliation.json` |
| `37b`, `37c` | Blind live NTRIP 10-station capture and obstruction replay (P4) | `step_37b_ntrip_process.json`, `step_37c_ntrip_report.json` |
| `45`, `45b`, `83`, `176` | Temporal T4 co-visibility separation (pooled-$\kappa$ on real SP3) and common-view-free T2 anisotropy | `step_45_temporal_t4.json`, `step_45b_t4_forensic.json`, `step_83_t4_phasecos.json`, `step_176_t2_alpha_anisotropy.json` |
| `48`, `106` | Blind nonlinear $k=3$ three-satellite solve and $k=4\text{–}5$ degraded-sky positioning flagship | `step_48_three_satellite.json`, `step_48_three_satellite_precise.json`, `step_106_degraded_sky_position_flagship.json` |
| `53`, `103`, `158` | Corrected Papers 1–2 phase-alignment estimator ($\cos(\arg \sum_f S_{ij}(f))$) on COD/GFZ and oscillator-class gradients | `step_53_coherent_lambda.json`, `step_103_atomic_quartz_gradients.json`, `step_158_screening_replication.json` |
| `55`, `56`, `60`, `85`–`88`, `90` | Causal dense-network outage bridging, prior-day phase Gram, robustness sweeps, and ITU-T holdover specification | `step_55_dense_outage_prediction.json`, `step_56_phase_predictive_bridge.json`, `step_88_per_station_breakdown.json`, `step_90_operational_holdover_spec.json` |
| `65`, `65c`, `71`, `77`, `80`, `170`, `177` | Oscillator screening, co-located differential anchors, blind coordinate inversion, and distance-blind MDS geography | `step_65_oscillator_screening.json`, `step_77_colocated_screening_differential.json`, `step_80_clock_field_localisation.json`, `step_170_clock_field_localisation_live.json`, `step_177_distance_blind_geography.json` |
| `68`, `69`, `72b`, `79` | Synchronization holonomy bound and joint site-frame az/el multipath separation | `step_68_synchronization_holonomy.json`, `step_72b_sidereal_template_blindness.json`, `step_79_azel_holonomy.json` |
| `96`, `104b`, `127`, `140`, `153` | Frozen 2024→2025 leftover-clock supremacy (P1), 30/30 regional-seasonal sweep (P5), solar-tidal notching, and 3-band COD+GFZ replication | `step_96_leftover_clock_supremacy.json`, `step_104b_seasonal_regional_sweep.json`, `step_140_multiband_controls.json`, `step_153_cross_product.json` |
| `137`, `179` | Sub-diurnal Earth-rotation aperture synthesis and day-clustered/split-half CMB dipole axis robustness | `step_137_aperture_synthesis.json`, `step_179_axis_robustness.json` |
| `162`, `164`, `167`, `169` | High-resolution $M_2$ vs $S_2$ tidal spectroscopy, co-located differential spectrometer, and station-level gravitational phase test | `step_162_m2_vs_s2_clocktype.json`, `step_164_fullyear_differential_spectrometer.json`, `step_169_corrected_gravitational_phase.json` |
| `168`, `174`, `174b`, `180` | Live 68-station walk-forward observatory, SHA-256 sealed-envelope ex-ante validation, clean-origin decomposition, and preregistration | `step_168_live_network_tep_evidence.json`, `step_174_sealed_envelope_prediction.json`, `step_174_clean_untapered_arm.json`, `step_180_prospective_preregistration.json` |

Generated JSON is stored in `results/outputs/`. The gate-aware summary is `results/outputs/evidence_audit.json`, the readable gate ledger is `EVIDENCE_STATUS.md`, and the per-script rubric audit is `PIPELINE_AUDIT.md`.

