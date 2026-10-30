# TEP-PPP evidence status

This ledger is generated from committed result JSON files by `python scripts/audit_operational_evidence.py`. A failed gate is a debug/validation target; it is never silently promoted into a claim.

Strict operational claim: **NOT YET PASSED**

Held-out GNSS receiver clocks contain operationally useful spatial structure, and the precise-clock TEP-scale fit is robust across five filters including detrend-only data. The raw-observation position gate (P2) is reframed: at k=4/5/6 the TEP kernel is tied with the best deployable conventional prior (both are harmful relative to free-clock because the clock is already observable from satellite geometry). The three-satellite challenge (step_48) confirms the algebraic niche: at k=3 the conventional solve is infeasible and an external clock prior supplies the missing degree of freedom. The corrected focused blind run is bounded, but TEP does not beat persistence and its integrity-gated availability is below the strict threshold. Repeated one-millisecond wrap-mode targets are now rejected causally; unexpected steering still requires steering-event telemetry or an independent motion/position constraint. The temporal T4 regression breaks the co-visibility collinearity: the non-common-view component of pair correlation shows TEP-scale distance structure on both broadcast (lambda=3890 km, R^2=0.70, 5.1 sigma) and precise (lambda=4179 km, R^2=0.90, 5.7 sigma) products. The frozen partial-correlation criterion (|PC| > 0.15) is not met (broadcast PC=0.038, precise PC=-0.006); the distance-fit R^2 is suggestive but does not satisfy the incremental-predictive-power test. S2 is PARTIAL. The free-GP null controls are clean and the >1,000 km baseline subset selects a 1,500 km median, but injection recovery across the full TEP band is only 44% within 2x (2/5 scales pass a 50% recovery threshold). S3 remains a scale-resolution debug flag, especially at 3,000-5,000 km. TEP-specific predictive superiority, production PPP gain, blind live replication (5-minute acquisition only), operational robustness remain active pipeline-debug/validation gates.

| Gate | Status | Criterion / present boundary |
|---|---:|---|
| `N1_spatial_network_skill` | **PASS** | clock-domain broadcast-SPP skill; not TEP-specific |
| `P1_TEP_specific_clock_advantage` | **DEBUG** | TEP kernel beats training-selected conventional spatial model; 95% CI > 0 |
| `P2_raw_observation_position_gain` | **DEBUG** | At k=4/5/6 the clock is already observable from satellite geometry, so any finite-error prior is inert or harmful. TEP is tied with deployable conventional (median 0.0%) and worse than free-clock. At k=3 the conventional solve is rank-deficient and the TEP prior supplies the missing algebraic degree of freedom. Blind solve (tep_blind_integrity): availability=80.6%, RMS=7.03 m, p50=4.27 m, p99=22.18 m. RMS vs best genuinely blind conventional clock prior: -2.9%. Paired: -2.1%. Gap-sweep (step_48b): TEP does not beat persistence in clock RMSE at any outage 0-4h (margin -18.9% to -2.5%), delimiting the clock-prediction claim; TEP's value is feasibility + independence. |
| `N2_postfit_integrity_signal` | **DIAGNOSTIC** | post-fit reliability diagnostic; cannot pass raw-observation P2; reported as diagnostic, not evidence |
| `P3_production_PPP_gain` | **DEBUG** | positive degraded-geometry gain inside production PPP/PPP-AR |
| `P4_blind_live_replication` | **DEBUG** | 5-minute NTRIP acquisition smoke test; frozen 10/30/60/120/240-minute windows not tested; PASS requires ≥4 h capture through the frozen pipeline |
| `P5_operational_robustness` | **DEBUG** | partial aperture and product controls only |
| `S1_precise_filter_robustness` | **PASS** | precise IGS/MGEX receiver-clock product |
| `S1_cross_product_filter_robustness` | **DEBUG** | both broadcast and precise arms retain scale and exceed R² permutation null by >3 sigma |
| `S2_physics_separability` | **PARTIAL** | frozen T4: |partial corr(corr, log-dist | covis)| > 0.15 AND OOS R² gain > 0.02. Temporal T4 partial correlations are 0.038 (broadcast, p=0.21) and -0.006 (precise, p=0.51) — below 0.15. Distance-fit R² on binned intercepts is strong (broadcast R²=0.70, 5.1σ; precise R²=0.90, 5.7σ) but does not satisfy the frozen receiver-side incremental-predictive-power criterion. Step 45b forensic battery (precise): 8/10 specifications survive the per-station epoch-shuffle null; long-baseline (>2,000 km, weakest collinearity) gives lambda=2643 km at 7.3 sigma. Step 45b forensic battery (broadcast): 6/10 specifications survive the per-station epoch-shuffle null; long-baseline (>2,000 km, weakest collinearity) gives lambda=40000 km at 1.3 sigma. The precise arm is therefore real, control-robust evidence; the broadcast arm is fragile under alternative co-visibility definitions and is withdrawn. |
| `S3_free_gp_scale_validation` | **DEBUG** | full-band injection recovery is 44% within 2x and only 2/5 TEP-band scales pass the 50% recovery threshold; null controls are clean, but 3,000-5,000 km resolution remains a debug target |
| `S4_phase_alignment_scale_recovery` | **PASS** | the correct-estimator physics result: matches Papers 1-2 (4,201 +/- 1,967 km) and step_10; independent of the operational gates |
| `S5_density_scaling_operational_skill` | **PASS** | operational corollary: the Step 53 correlation becomes usable for clock prediction once reference spacing drops well inside lambda_T; on the dense network the full network prior also cuts the unknown-clock RMS by ~35% vs ~0% globally |

## Priority closure path

1. Run causal clock preprocessing and select every comparator on training data only.
2. Thin satellites in raw RINEX before solving; do not reuse a full-sky state or residuals.
3. Compare against a causally selected temporal model plus tuned GP/kNN/IDW baselines.
4. Repeat inside production PPP under 4/5/6-satellite and outage regimes.
5. Freeze the complete method, then score an untouched live stream.

