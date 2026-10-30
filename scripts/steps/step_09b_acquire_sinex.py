#!/usr/bin/env python3
"""
STEP 09b -- Acquire official IGS station coordinates (daily SINEX).

The end-to-end positioning gates (P2/P3) must score against a surveyed-truth
coordinate, not the day's own SPP mean (which is circular and, empirically,
10-17 m from the official value for broadcast SPP).  This step downloads the
daily IGS combined SINEX solution ``IGS0OPSSNX_*_01D_01D_SOL.SNX.gz`` for every
day the positioning experiments use (~11 MB/day, ~1,490 stations/day).

Consumed by scripts/lib/tep_sinex.py -> station_xyz(code4, year, doy).
"""
from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib import tep_sinex as sx           # noqa: E402
from scripts.utils.logger import print_status     # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--doys", type=str, default="100-106",
                    help="e.g. 100-106 or 1-45,100-106")
    args = ap.parse_args()

    doys = []
    for tok in args.doys.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            doys.extend(range(int(a), int(b) + 1))
        elif tok:
            doys.append(int(tok))

    import requests
    from scripts.steps.step_01_acquire_data import _get_auth
    sess = requests.Session()
    sess.auth = _get_auth()

    print_status(f"Acquiring daily IGS SINEX for {args.year} DOY "
                 f"{doys[0]}-{doys[-1]} ({len(doys)} days)", "TITLE")
    have, got, miss = 0, 0, []
    for d in doys:
        p = sx._snx_path(args.year, d)
        if p.exists() and p.stat().st_size > 500_000:
            have += 1
        else:
            r = sx.download(args.year, d, session=sess)
            if r is not None:
                got += 1
            else:
                miss.append(d)
        # parse + cache the coord table now so downstream steps are instant
        n = len(sx._load_day(args.year, d))
        if (have + got) % 10 == 0:
            print_status(f"{have + got}/{len(doys)} days  "
                         f"({n} stations in DOY {d:03d})", "INFO")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step_09b_acquire_sinex.json").write_text(json.dumps({
        "year": args.year, "n_doys": len(doys),
        "on_disk": have, "downloaded": got, "missing": miss,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print_status(f"done -- {have + got}/{len(doys)} daily SINEX available, "
                 f"{len(miss)} missing", "SUCCESS")


if __name__ == "__main__":
    main()
