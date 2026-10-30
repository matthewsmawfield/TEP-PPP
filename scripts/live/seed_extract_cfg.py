#!/usr/bin/env python3
"""Pre-seed extract_cache.pkl with the NEW content-based cfg hash.

Replicates PPPProcessor._extract_cfg_hash (rpi/gps_web.py) exactly:
  sha1( receiver_ecef.round(2).tobytes()
        + for each *.sp3/*.clk in data/ppp_live/precise (sorted):
            filename.encode() + file bytes )

receiver_ecef is _external_truth_ecef when set, i.e. truth_ref.json's
truth_ecef (external RTK-AR consensus anchor), else the RINEX header
position.  Run AFTER a product fetch, BEFORE the first extraction under
the content-hash code, so the rolling cache is a warm hit instead of a
guaranteed ~25-40 min cold start.

Usage: python3 scripts/live/seed_extract_cfg.py [--write]
Without --write it only prints the computed hash and the pkl's stored cfg.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PDIR = ROOT / "data" / "ppp_live" / "precise"
CACHE = ROOT / "data" / "ppp_live" / "extract_cache.pkl"
TRUTH = ROOT / "data" / "ppp_live" / "truth_ref.json"


def cfg_hash(receiver_ecef) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(receiver_ecef, float).round(2).tobytes())
    if PDIR.exists():
        for pf in sorted(PDIR.iterdir()):
            if pf.suffix.lower() in (".sp3", ".clk"):
                try:
                    h.update(pf.name.encode())
                    h.update(pf.read_bytes())
                except OSError:
                    pass
    return h.hexdigest()


def main() -> int:
    write = "--write" in sys.argv
    pd = json.loads(TRUTH.read_text())
    ecef = np.array(pd["truth_ecef"], float)
    cfg = cfg_hash(ecef)
    print("receiver_ecef:", ecef.tolist())
    print("computed cfg:", cfg)
    if not CACHE.exists():
        print("no extract_cache.pkl — nothing to seed")
        return 1
    blob = pickle.loads(CACHE.read_bytes())
    print("stored cfg:  ", blob.get("cfg"))
    print("epochs:      ", len(blob.get("epochs") or {}))
    if blob.get("cfg") == cfg:
        print("already matches — warm start guaranteed")
        return 0
    if not write:
        print("mismatch — rerun with --write to seed")
        return 2
    blob["cfg"] = cfg
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(blob))
    tmp.replace(CACHE)
    print("seeded cfg ->", cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
