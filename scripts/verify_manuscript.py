#!/usr/bin/env python3
"""Run the maintained manuscript/evidence verification.

The previous verifier contained more than one hundred hard-coded historical
assertions and could report success while current outputs contradicted the
manuscript.  Verification now delegates to the gate-aware consistency check.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [sys.executable, str(ROOT / "scripts" / "audit_operational_evidence.py")],
        [sys.executable, str(ROOT / "scripts" / "check_manuscript_numbers.py")],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    print("verification complete: evidence ledger and manuscript are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
