# TEP-PPP analysis pipeline

The orchestrator exposes the commands documented by the repository:

```bash
# Show the exact ordered step list
python scripts/run_pipeline.py --list

# Full run, including acquisition/processing
python scripts/run_pipeline.py

# Reuse data already present in data/
python scripts/run_pipeline.py --skip-data

# Fast code-path smoke run
python scripts/run_pipeline.py --quick --skip-data

# Resume or isolate a real step name from --list
python scripts/run_pipeline.py --from-step step_25_kernel_selection --skip-data
python scripts/run_pipeline.py --only-step step_38_baseline_battery
```

Do not delete committed outputs to perform verification. The non-destructive
verification entry point compiles the code, runs tests, regenerates the evidence
ledger and checks manuscript claims against result JSON:

```bash
python scripts/reproduce.py --verify-only
```

Key evidence-hardening steps are:

| Step | Purpose |
|---|---|
| 25/34 | held-out station/day clock prediction |
| 38 | training-selected conventional baseline battery |
| 39 | causally calibrated post-fit position diagnostic |
| 40 | filter sensitivity with R² permutation nulls |
| 23 | production PRIDE-PPP-AR state-coupling probe |

Generated JSON is stored in `results/outputs/`. The gate-aware summary is
`results/outputs/evidence_audit.json` and the readable ledger is
`EVIDENCE_STATUS.md`.
