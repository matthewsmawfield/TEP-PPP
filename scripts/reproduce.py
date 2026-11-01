#!/usr/bin/env python3
"""Fast reproducibility and consistency verification entry point."""
from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="verify code/results/manuscript without rerunning analyses")
    args = parser.parse_args()
    if not args.verify_only:
        run([sys.executable, "scripts/run_pipeline.py", "--skip-data"])
    if not compileall.compile_dir(ROOT / "core", quiet=1):
        return 1
    if not compileall.compile_dir(ROOT / "scripts", quiet=1):
        return 1
    run([sys.executable, "-m", "pytest", "-q"])
    audit_flags = ["--verify-artifacts", "--check"] if args.verify_only else []
    run([sys.executable, "scripts/audit_operational_evidence.py", *audit_flags])
    run([sys.executable, "scripts/check_manuscript_numbers.py"])
    print("Reproducibility verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
