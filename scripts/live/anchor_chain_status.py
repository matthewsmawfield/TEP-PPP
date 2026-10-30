#!/usr/bin/env python3
"""Verify the sealed-envelope anchor chain and pin its state.

Each record in data/ppp_live/sealed_envelopes/anchors.log carries

    anchor = SHA256(prev_anchor + weights_sha256 + utc)

with prev starting as "GENESIS", so the sequence cannot be reordered,
truncated, or back-filled without breaking the head digest.  This script
replays the full chain, re-hashes every *_weights.npz payload, and writes
the verified head to results/outputs/sealed_anchor_chain_status.json.
Because results/ is git-tracked, committing that file gives the chain an
external (repository-history) timestamp: any later tampering with earlier
links would have to reproduce a head committed in the past.

Writes: results/outputs/sealed_anchor_chain_status.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "data" / "ppp_live" / "sealed_envelopes"
LOG = ENV_DIR / "anchors.log"
OUT = ROOT / "results" / "outputs" / "sealed_anchor_chain_status.json"


def verify() -> dict:
    records = [json.loads(l) for l in
               LOG.read_text().strip().splitlines() if l.strip()]
    prev = "GENESIS"
    failures = []
    for i, r in enumerate(records):
        w = ENV_DIR / f"{r['envelope_id']}_weights.npz"
        if not w.exists():
            failures.append({"index": i, "envelope": r["envelope_id"],
                             "error": "missing weights"})
            continue
        sha = hashlib.sha256(w.read_bytes()).hexdigest()
        if sha != r["weights_sha256"]:
            failures.append({"index": i, "envelope": r["envelope_id"],
                             "error": "weights sha256 mismatch"})
            continue
        if r.get("prev_anchor_sha256") != prev:
            failures.append({"index": i, "envelope": r["envelope_id"],
                             "error": "prev anchor mismatch"})
            continue
        expect = hashlib.sha256(
            (prev + sha + r["utc"]).encode()).hexdigest()
        if expect != r["anchor_sha256"]:
            failures.append({"index": i, "envelope": r["envelope_id"],
                             "error": "chain link mismatch"})
            continue
        prev = expect
    return {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "chain": "SHA256(prev_anchor + weights_sha256 + utc), "
                 "prev starts 'GENESIS'; weights_sha256 is the plain "
                 "sha256 of the *_weights.npz file bytes",
        "n_links": len(records),
        "n_links_verified": len(records) - len(failures),
        "first_envelope": records[0]["envelope_id"],
        "first_seal_utc": records[0]["seal_utc"],
        "last_envelope": records[-1]["envelope_id"],
        "last_seal_utc": records[-1]["seal_utc"],
        "head_sha256": prev,
        "failures": failures,
    }


def main() -> int:
    out = verify()
    OUT.write_text(json.dumps(out, indent=2))
    status = "INTACT" if not out["failures"] else "BROKEN"
    print(f"anchor chain {status}: {out['n_links_verified']}/"
          f"{out['n_links']} links verified, "
          f"{out['first_envelope']} -> {out['last_envelope']}, "
          f"head {out['head_sha256'][:16]}...")
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
