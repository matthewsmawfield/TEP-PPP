# Temporal Equivalence Principle: Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23704945.svg)](https://doi.org/10.5281/zenodo.23704945)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Temporal Equivalence Principle: Phase-Resolved Spatial Clock Priors for GNSS Outage Bridging](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.5 (Pune)  
**First published:** 29 October 2026 · **Last updated:** 2 November 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.23704945](https://doi.org/10.5281/zenodo.23704945)  
**Website:** [https://mlsmawfield.com/tep/ppp/](https://mlsmawfield.com/tep/ppp/)  
**Paper Series:** TEP Series · Group B (II. Global Time Echoes: GNSS Atomic Clock Networks) · Paper 33 (`B33`)

## Abstract

<!-- Generated from site/components/00_abstract.html; update the manuscript source, then run tools/sync_current_corpus_readmes.py. -->

The Temporal Equivalence Principle (TEP) predicts that proper time behaves as a dynamical scalar field, producing spatially correlated fluctuations in distributed clock networks. Papers 1–3 established persistent, distance-structured clock correlations across 364–539 GNSS stations and up to 1.17 billion station-pair measurements, identifying a spatial correlation length $\lambda_T \approx 4{,}200$ km in receiver clock residuals. This study extends the detection onto independent satellite-pair and optical-clock channels, separates the signal from its leading geodetic and instrumental confounds, and converts the spatial covariance into a deployment-ready navigation and timing prior.

In the cross-spectral phase-alignment observable $\cos(\arg \sum_f S_{ij}(f))$, precise receiver clocks recover $\lambda = 4{,}348$ km on COD and $4{,}076$ km on GFZ ($R^2 = 0.898$ and $0.955$, $25$–$27\sigma$ above phase-randomised nulls), while orbiting satellite clock pairs — sharing no ground receiver and no common-view geometry — exhibit distance-structured phase alignment at $10.6$–$15.9\sigma$ and $26.4\sigma$ on a GPS×Galileo cross-constellation arm ($\lambda = 8{,}140 \pm 795$ km). On the ground, two-stage panel regression separates the zero-shared-satellite intercept from co-visibility ($\mathrm{PC} = -0.164$ at $10.8\sigma$, strengthening to $-0.267$ with solar-time control), preserves the $500$–$800$ km East–West anisotropy ($+0.130 \rightarrow +0.126$), and discriminates the Cosmic Microwave Background dipole axis under sub-diurnal Earth-rotation aperture synthesis (winning $300/300$ day-clustered bootstrap draws). Oscillator-class stratification shows that idiosyncratic noise is identical between quartz and atomic stations ($0.99\times$) whereas network-coupled coherent power is $3.01\times$ larger on quartz ($7.37\times$ in station-median LF coherent RMS, $p = 0.0273$), isolating lunar $M_2$ tidal phase coupling to metadata-verified quartz clocks (dependence-preserving joint COD+GFZ station-permutation $z_{\mathrm{joint}} = +5.73\sigma, p_{\mathrm{joint}} = 0.0003$, conditional Fisher $p_{\mathrm{perm}} = 1.9\times 10^{-6}$, vs atomic $z_{\mathrm{joint}} = -0.87\sigma, p_{\mathrm{joint}} = 0.374$, Fisher $0.55$) and yielding a stability-stratified double dissociation between atmospheric solar $S_2$ and lunar $M_2$ tides.

Out-of-ecosystem reanalysis of the 65-day 2023 European optical-fibre clock campaign across NPL, LNE-OP/SYRTE, PTB, and INRIM constrains differential frequency-ratio offsets across the Doppler-cancelled NPL–PTB link to $-7.1 \pm 7.7\times 10^{-18}$ ($1\sigma$; conditional $95\%$ single-effect envelope $\le 2.2\times 10^{-17}$, with $1$ s 3-node star-tree closure at $3.61\times 10^{-20}$ on the $b_1 = 0$ tree), unmasks a $\tau_{\mathrm{DL}} = 8.68$–$27.92\times 10^{-18}$ sub-mHz remote floor versus $0.00\times 10^{-18}$ co-located at NPL ($7.86$ and $24.76\times 10^{-18}$ locally at PTB; 4-clock baseline excess $7.62$–$7.74\times 10^{-18}$ on $212{,}605$ simultaneous $1$ s common epochs and $7.30\times 10^{-18}$ full-campaign; 8-pair $\rho = +0.927, p = 9.4\times 10^{-4}$, 4-group $\rho_{\mathrm{group}} = +0.800$) alongside $705$ km lunar $M_2$ amplitude enhancement (Fisher $p = 0.0014$ raw, $0.0019$ demeaned; joint species-frame $\mathrm{Yb^+\,E3}-\mathrm{Sr}$ simultaneous phase clustering $R = 0.990, p_{\mathrm{joint,6h}} = 0.0230$ vs $R = 0.477$ (four geographic ratios on a separate window; $R = 0.358$ for the same three simultaneous ratios) in the geographic frame), detects zero-fibre $705$ km anti-phase coherence probing a non-universal $\Delta K_\alpha$ electromagnetic coupling extension between co-located optical pairs at NPL and PTB ($\mathrm{PA} = -0.335 \pm 0.091, z = -3.69, z_{\mathrm{null}} = -3.27, p_{\mathrm{perm}} = 0.0010$), and localises the zero-satellite optical clock pairs globally to $879$–$1{,}416$ km against the 366-station GNSS archive (conditional two-site Fisher $p = 0.0030$; sub-diurnal $f > 30\,\mu\text{Hz}$ error $704$–$1{,}285$ km).

Operationally, a phase-resolved TEP clock prior requires no new hardware and under 1 ms per epoch on existing CORS/RTK networks: (i) for timing holdover, it reduces sky-outage clock-residual error by $38.1\%$ at 1 h ($24.7$ ns) and $55.7\%$ at 8 h ($65.6$ ns), outperforming datasheet holdover of a $\sim\$300$ OCXO beyond 30 min and a $\sim\$2{,}000$ Rubidium standard beyond 2 h; (ii) for outage bridging, it reduces clock RMS by $5.0$–$18.9\%$ over 90 EUREF days and beats every deployable baseline on held-out 2025 across Europe, North America, and East Asia; and (iii) for degraded-geometry positioning, it restores $k = 3$ availability ($+29.6\%$ 3-D carrier-phase RMSE reduction), improves $k = 4$–$5$ 3-D RMS by $+14.5$–$48.0\%$, localises co-located GNSS anchors to $339$ km (Matera) and $744$ km (Tahiti) and 43 live NTRIP stations to $191.8$ km median error, and achieves $+35.84\%$ mean two-hour holdover gain across 185 walk-forward windows on a 68-station live observatory ($116$ of $120$ SHA-256-sealed ex-ante envelopes positive, $+29.01\%$ mean on overlapping sliding windows; zero-target-overlap clean-origin forward component $+2.57\%, p = 1.8\times 10^{-5}$).

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
  note={Preprint v0.5 (Pune)}
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
