# Temporal Equivalence Principle: Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23704945.svg)](https://doi.org/10.5281/zenodo.23704945)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Temporal Equivalence Principle: Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.2 (Pune)  
**First published:** 29 October 2026 · **Last updated:** 30 October 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.23704945](https://doi.org/10.5281/zenodo.23704945)  
**Website:** [https://mlsmawfield.com/tep/ppp/](https://mlsmawfield.com/tep/ppp/)  
**Paper Series:** TEP Series · Group B (II. Global Time Echoes: GNSS Atomic Clock Networks) · Paper 33 (`B33`)

## Abstract

<!-- Generated from site/components/00_abstract.html; update the manuscript source, then run tools/sync_current_corpus_readmes.py. -->

The Temporal Equivalence Principle (TEP) predicts that proper time behaves as a dynamical scalar field, producing spatially correlated fluctuations in distributed clock networks. Papers 1–3 established the field across 364–539 GNSS stations and up to 1.17 billion station-pair measurements, identifying a spatial correlation length $\lambda_T \approx 4200$ km in receiver clock residuals. Because every one of those measurements traversed the same ground-receiver processing chain, a correlated instrumental systematic remained a candidate explanation. This paper carries the detection onto a channel whose pairwise geometry is independent of the receiver-side measurements, separates the detection from its leading confounds, and converts it into deployment-ready predictive power.

Between orbiting satellite clocks — a channel whose pairwise statistic uses no ground receiver and no common-view geometry — the corrected phase-alignment index detects a clock correlation falling off with inter-satellite separation at $10.6\sigma$ and $12.6\sigma$ above phase-randomised and geometry-shuffled nulls, independently confirmed by a GPS×Galileo cross-constellation arm that shares no orbital plane or clock family ($26.4\sigma$ log-slope $-0.172$ over 180 days, near/far contrast $+0.30$ on 90-day windows, pooled exponential fit $\lambda = 8{,}140 \pm 795$ km with sub-window scale trading off against the long-baseline pedestal across $8{,}140$–$14{,}911$ km; the satellite clock estimates themselves share analysis-centre provenance with the receiver products, the residual systematic addressed by Step 172). On the ground, the non-common-view intercept separates from satellite co-visibility on the minimally processed broadcast product (partial correlation $-0.164$ at $10.8\sigma$, strengthening to $-0.267$ with the local-solar-time systematic controlled; out-of-sample $R²$ gain $+0.022$; raw intercept fit $\lambda = 4{,}349$ km, and $\lambda = 1{,}008$ km at $6.5\sigma$ after projecting out the collinear co-visibility and solar-time profiles, which absorbs the long-baseline tail), while the network-solved precise product shows the expected datum-absorbed attenuation ($\mathrm{PC} = -0.040$ at $9.2\sigma$). The predicted East–West anisotropy at $500$–$800$ km survives common-view removal essentially unchanged ($+0.130 \rightarrow +0.126$, $z = +1.98 \rightarrow +1.91$). Oscillator-class dependence supplies a physical discriminator: idiosyncratic clock noise is identical between quartz and atomic stations ($0.99\times$) while the network-coupled component is $3.01\times$ larger on quartz — a separation that persists on co-located receivers sharing site, sky and monument, and yields a clean double dissociation between the atmospheric solar $S_2$ tide (atomic clocks null at $p = 0.66$, quartz $r = +0.230$ to $+0.280$, station-level $p = 0.0099$ and $0.0010$, station-permutation $p = 10^{-4}$; pair-level $p = 1.8\times10^{-93}$) and the non-radiative lunar $M_2$ tide ($p = 8.3\times10^{-6}$ multi-product Fisher combined under station-level permutation, and $R² = 0.702, p = 0.0050$ on the co-located Matera quartz-minus-maser differential where site ocean-loading displacement cancels). Sub-diurnal Earth-rotation aperture synthesis identifies the field's preferred axis with the CMB dipole at $\Delta\mathrm{BIC} = -2{,}315.5$ over the next candidate, winning 300 of 300 day-clustered bootstrap draws and both disjoint half-year splits (the raw $\Delta\mathrm{BIC}$ scales with pair-window count, while the bootstrap and split-half ordering is cluster-invariant).

The detection is predictive ex-ante: 120 cryptographically sealed weight envelopes score their first genuinely post-seal windows at $+29.01\%$ mean (116 of 120 positive) under SHA-256 verification. The same protocol bounds its own reach: the sealed walk-forward figure decomposes into verified retrodiction of member-observed epochs plus a spatially specific forward component at never-observed epochs ($+2.57\%$, $p = 1.8\times10^{-5}$, under causal penalty selection; $+3.39\%$ untapered, $z = +5.8$ to $+6.6$ above station-label permutation, frozen under prospective preregistration), with a measured $\sim$90-minute frozen-weight coherence time. With station coordinates withheld, the 43-mode phase fingerprint localises co-located anchors globally to 339 km (Matera) and 744 km (Tahiti) and 43 live NTRIP stations to a 191.8 km median error on enclosed stations (214.7 km across all 43) from 30 minutes of clock data alone. The one remaining systematic — a correlated error inside the GNSS processing stack itself — is assigned to a pre-registered satellite-free hardware replication (Step 172).

Operationally, a phase-resolved TEP clock prior delivers three deployment-ready benefits on existing CORS/RTK infrastructure with no new hardware and sub-millisecond per-epoch cost. For timing holdover, during a sky-view outage the network prediction reduces each receiver's clock-residual error by $38.1\%$ at 1 h ($39.9 \rightarrow 24.7$ ns), $48.8\%$ at 2 h ($57.2 \rightarrow 29.3$ ns), and $55.7\%$ at 8 h ($148.0 \rightarrow 65.6$ ns) relative to the receiver's own best causal temporal extrapolator — a network-assisted residual bound, not an absolute UTC error measured against a traceable hardware reference; contextualised against manufacturer datasheet drift curves it outperforms a GNSS-disciplined OCXO ($\sim\$300$) beyond 30 min and a disciplined Rubidium ($\sim\$2{,}000$) beyond $\sim$2 h. For outage bridging, across 90 days on a dense EUREF network the fixed kernel reduces clock RMS by $5.0$–$18.9\%$ versus the strongest temporal predictor at 10 min–8 h, with positive day-clustered intervals at every horizon; frozen on 2024 and applied unchanged to held-out 2025, the construction beats every deployable baseline on three coherent regional networks. For degraded-geometry positioning, at three visible satellites, where free-clock SPP is rank-deficient, the prior supplies the missing degree of freedom (5.76 m RMS, 95.2% availability in single-epoch code SPP, at pseudorange-noise-limited parity with a conventional AR(1) clock prior at 5.53 m, and $+29.6\%$ 3-D / $+46.7\%$ vertical RMSE reduction over persistence across 38,900 1 Hz carrier-phase outage epochs); at $k = 4$–$5$ it improves 3-D RMS by $+14.5$–$48.0\%$ over a deployable persistence-trend prior; and on a 68-station live observatory the causal field prior reduces two-hour holdover RMSE by $+35.84\%$ across 185 walk-forward windows (275 of 276 recorded live archives positive over $95.6$ h, $+42.93\%$ mean). With a full sky the clock is already observable and the prior is correctly inert.

The methodological point is that cross-spectral phase alignment, rather than overall pair amplitude, isolates the field's spatial structure: the corrected estimator $\cos(\arg \sum_f S_{ij}(f))$ recovers $\lambda = 4{,}348$ km on COD and $4{,}076$ km on GFZ precise receiver clocks ($R² = 0.898$ and $0.955$, 25–27$\sigma$ above per-station phase-randomised nulls that preserve every frequency bin's amplitude spectrum identically), matching the Papers 1–2 measurement to within 3.5%. The result is therefore a detection claim, not only an engineering one: the predicted correlation is recovered on a channel whose pairwise statistic shares no receivers and no common-view geometry with the original measurements (the satellite clock estimates still traverse the shared tracking segment — the one residual systematic the satellite-free hardware replication is designed to close); it separates from the leading instrumental confound at a pre-registered threshold; and it predicts unscored data under cryptographic seal. Its bounds are stated equally plainly in the text.

In plain terms. Clocks separated by up to thousands of kilometres are found to drift together, and the strength of that togetherness fades with distance across the $1{,}000$–$5{,}000$ km screening band predicted by the theory ($\lambda = 4{,}076$–$4{,}348$ km on precise phase alignment, matching the $4{,}201$ km scale established in Papers 1–2, and a dependence-aware combined median of $2{,}822$ km, 95% CI $[1{,}949, 3{,}763]$ km, across amplitude and Gaussian-process estimators). The same distance-structured decay appears between clocks aboard the satellites themselves, in a pairwise test that involves no ground receiver and no shared viewing geometry — although the satellite clock estimates still pass through the same ground segment, which is why the no-GPS hardware test remains; on the ground it survives removal of the specified satellite co-visibility geometry on the minimally processed broadcast product (and remains directionally present, though attenuated by the network datum adjustment, on precise clocks); its spatially coherent component is three times stronger in quartz clocks than in atomic ones even when both sit on the same monument sharing the same sky, isolating the differential to oscillator physics or clock-product referencing rather than local site noise; and when the prediction machinery is locked away under cryptographic seal before the data exist, it beats persistence in 116 of 120 first post-seal evaluations (combining high-precision network retrodiction of member-observed epochs with a smaller, statistically significant $+2.57\%$ forward forecast at never-observed epochs inside the $\sim$90-minute coherence window). Every alternative instrumental or environmental explanation tested to date has failed to reproduce this combination of signatures; the one test that cannot be done with GNSS data alone — measuring the field with clocks that never touch GPS — is registered as the closing experiment.

**Keywords:** Temporal Equivalence Principle; GNSS atomic clocks; Precise Point Positioning (PPP); spatial clock priors; GNSS outage bridging; timing holdover; cross-spectral phase alignment; cryptographically sealed evaluation

Current manuscript source: [00_abstract.html](site/components/00_abstract.html).

## Overview

TEP-PPP tests whether the spatial structure reported in GNSS receiver clocks can
be turned into a useful receiver-clock prior for GPS/GNSS positioning. The
repository contains the acquisition and processing code for the public
IGS/CODE/GFZ GNSS products (raw data is fetched by the `data`-tagged pipeline
steps and is not committed due to size), spatial-kernel code, falsification
controls, held-out benchmarks, position replays, the committed machine-readable
results they produced, and a machine-readable evidence ledger.

## Key Findings

The current findings, numerical results and comparison assumptions are reported in [3. Results](site/components/03_results.html), [5. Conclusion](site/components/05_conclusion.html). The abstract above is synchronized from the current manuscript.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Temporal Evolution | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Validation of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Validation of the Temporal Equivalence Principle | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales | [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: A Runaway Black Hole as a Gravitational Soliton | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) |
| **Paper 14** | [TEP-GNSS-MGEX](https://github.com/matthewsmawfield/TEP-GNSS-MGEX) | MGEX Multi-GNSS Clock Replication | [10.5281/zenodo.20572726](https://doi.org/10.5281/zenodo.20572726) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454862](https://doi.org/10.5281/zenodo.19454862) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454619](https://doi.org/10.5281/zenodo.19454619) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446028](https://doi.org/10.5281/zenodo.19446028) |
| **Paper 18** | [TEP-HC](https://github.com/matthewsmawfield/TEP-HC) | EFT Mapping and Acoustic Peak Constraints via hi_class | [10.5281/zenodo.20572722](https://doi.org/10.5281/zenodo.20572722) |
| **Paper 19** | [TEP-LENS](https://github.com/matthewsmawfield/TEP-LENS) | Blind-Prediction Residual Test in Multiply-Imaged Supernovae | [10.5281/zenodo.20572724](https://doi.org/10.5281/zenodo.20572724) |
| **Paper 26** | [TEP-C0](https://github.com/matthewsmawfield/TEP-C0) | A Covariant Alternative to Cosmic Expansion | [10.5281/zenodo.20370143](https://doi.org/10.5281/zenodo.20370143) |
| **Paper 28** | [TEP-BH](https://github.com/matthewsmawfield/TEP-BH) | Black Holes and the Temporal Horizon | [10.5281/zenodo.21677826](https://doi.org/10.5281/zenodo.21677826) |
| **Paper 33** | **TEP-PPP** (This repo) | Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging | [10.5281/zenodo.23704945](https://doi.org/10.5281/zenodo.23704945) |

## Evidence Pipeline

| Stage | Purpose | Primary artifacts |
|---|---|---|
| T1–T5 & phase scale | global structure, phase-cos $\lambda_T$, and satellite CSD | `step_10`–`step_16`, `step_36_meo_csd.json`, `step_53_coherent_lambda.json` |
| Nested CV & LODO | held-out clock prediction and conventional battery | `step_25_kernel_selection.json`, `step_34_heldout_lodo.json`, `step_38_baseline_battery.json` |
| Causal outage & holdover | dense-network bridging, phase Gram, and TCXO/OCXO/Rb benchmark | `step_55_dense_outage_prediction.json`, `step_56_phase_predictive_bridge.json`, `step_60_operational_benefits.json` |
| Out-of-year & multi-region | frozen 2024→2025 LF+MF Gram, 3-band $\lambda$ on COD+GFZ, 30/30 seasonal sweep | `step_96_leftover_clock_supremacy.json`, `step_104b_seasonal_regional_sweep.json`, `step_140_multiband_controls.json`, `step_153_cross_product.json` |
| Degraded-sky positioning | $k=3$ blind solve, $k=4$ PRIDE-PPP-AR (P3), $k=4\text{–}5$ SPP flagship | `step_48_three_satellite_precise.json`, `step_62_ppp_degraded.json`, `step_106_degraded_sky_position_flagship.json` |
| Oscillator screening & tides | quartz/atomic decomposition, co-located anchors, $M_2/S_2$ phase, blind localisation | `step_65c_oscillator_noise_null.json`, `step_77_colocated_screening_differential.json`, `step_80_clock_field_localisation.json`, `step_164`–`step_171` |
| Live NTRIP & 68-station RRR | 10-station blind live replication (P4) and continuous walk-forward observatory | `step_37b_ntrip_process.json`, `live_rrr_attribution_and_position.json`, `step_168_live_network_tep_evidence.json`, `step_170_clock_field_localisation_live.json` |
| Gate ledger | claim/evidence consistency | `evidence_audit.json`, `EVIDENCE_STATUS.md` |

## Repository Structure

```text
TEP-PPP/
├── core/                    # Canonical TEP and PPP kernels
├── rpi/                     # Live GNSS Skill Observatory server, 1 Hz EKF, RRR, and PPP-AR engine
├── scripts/
│   ├── lib/                 # Shared data, kernel, NTRIP/OS Net clients, and estimator code
│   ├── live/                # Real-time 68-station NTRIP kriging server, product fetcher, and capture tools
│   └── steps/               # Numbered experiments and controls (step_01 .. step_180)
├── tests/                   # Validity and live-engine regression tests (157 tests)
├── data/                    # GNSS inputs and deterministic caches (not versioned)
├── results/
│   ├── outputs/             # Machine-readable experiment results
│   └── figures/             # Generated figures
├── logs/                    # Per-step execution logs
├── site/                    # Manuscript website source (components/ -> dist/)
├── README.md
├── CITATION.cff
├── CITATION.bib
├── VERSION.json
├── version.txt
├── zenodo.txt
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/matthewsmawfield/TEP-PPP.git
cd TEP-PPP
python -m pip install -r requirements-lock.txt
```

## Reproduction Pipeline

Python 3.10+ is required. Verify committed artifacts or rerun the pipeline:

```bash
# Fast code/result/manuscript consistency check
python scripts/reproduce.py --verify-only

# List the real pipeline step names
python scripts/run_pipeline.py --list

# Reuse local data and run the complete analysis
python scripts/run_pipeline.py --skip-data

# Fast subset (skips network and heavy steps)
python scripts/run_pipeline.py --quick --skip-data

# Resume or run one step
python scripts/run_pipeline.py --from-step step_25_kernel_selection --skip-data
python scripts/run_pipeline.py --only-step step_38_baseline_battery --skip-data

# Run tests and byte-compile check
python -m pytest -q
python -m compileall -q core scripts tests

# Build manuscript from HTML components (static site + markdown)
cd site && npm ci && npm run build
```

## Citation

```bibtex
@article{smawfield2026tepppp,
  title={Temporal Equivalence Principle: Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.23704945},
  url={https://doi.org/10.5281/zenodo.23704945},
  note={Preprint v0.2 (Pune)}
}
```

See [CITATION.cff](CITATION.cff) and [CITATION.bib](CITATION.bib).

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0); see [LICENSE](LICENSE).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction. Feedback and collaboration are warmly invited.

## Contact

**Email:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
