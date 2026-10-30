#!/usr/bin/env python3
"""
STEP 09 -- Acquire the full-year global analysis-centre receiver-clock product.

The physics battery (T1 correlation length, T2 anisotropy, T3 aperture sweep)
and the global held-out skill are currently limited to 7 days (DOY 100-106)
because that is all of the COD0MGXFIN RINEX CLK product on disk.  That product
already carries ~128 globally distributed receiver clocks per day, estimated by
CODE -- no raw RINEX or RTKLIB reprocessing is needed to extend it to the whole
year.  This step downloads every daily COD0MGXFIN_*_CLK.CLK.gz for a year
(~5 MB/day, ~1.8 GB for 366 days) into data/rinex/clk/, where tep_data's
"clk_cod" product picks it up automatically.

Optionally also fetches GFZ / GRG / JPL finals for the multi-analysis-centre
consistency check (T1 across centres).

Usage:
    python scripts/steps/step_09_acquire_global_clk.py --year 2024
    python scripts/steps/step_09_acquire_global_clk.py --year 2024 --centres cod,gfz,grg,jpl
    python scripts/steps/step_09_acquire_global_clk.py --year 2024 --doys 1-366
"""
from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime, date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.steps.step_01_acquire_data import _get_auth, _download_with_retry  # noqa: E402
from scripts.utils.logger import print_status  # noqa: E402

CLK_DIR = PROJECT_ROOT / "data" / "rinex" / "clk"
OUT = PROJECT_ROOT / "results" / "outputs"

# RINEX3 long-name prefixes for the daily 30 s clock finals
CENTRES = {
    "cod": "COD0MGXFIN",
    "gfz": "GFZ0OPSFIN",
    "grg": "GRG0MGXFIN",
    "jpl": "JPL0OPSFIN",
}


def _gps_week(year: int, doy: int) -> int:
    return (date(year, 1, 1).toordinal() + doy - 1 - date(1980, 1, 6).toordinal()) // 7


def _clk_url_candidates(prefix: str, year: int, doy: int):
    wk = _gps_week(year, doy)
    fname = f"{prefix}_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
    return [
        (f"https://cddis.nasa.gov/archive/gnss/products/{wk}/{fname}", fname),
        # some centres publish a 05M clock instead of 30S
        (f"https://cddis.nasa.gov/archive/gnss/products/{wk}/"
         f"{prefix}_{year}{doy:03d}0000_01D_05M_CLK.CLK.gz",
         f"{prefix}_{year}{doy:03d}0000_01D_05M_CLK.CLK.gz"),
    ]


def acquire(year: int, doys, centres, timeout_s: int = 120):
    auth = _get_auth()
    if not auth:
        print_status("Missing CDDIS auth (~/.netrc urs.earthdata.nasa.gov).", "ERROR")
        return None
    session = requests.Session()
    session.auth = auth
    CLK_DIR.mkdir(parents=True, exist_ok=True)

    summary = {c: {"have": 0, "downloaded": 0, "missing": []} for c in centres}
    for c in centres:
        prefix = CENTRES[c]
        for doy in doys:
            # already on disk?
            existing = list(CLK_DIR.glob(f"{prefix}_{year}{doy:03d}0000_01D_*_CLK.CLK.gz"))
            if existing and existing[0].stat().st_size > 100_000:
                summary[c]["have"] += 1
                continue
            got = False
            for url, fname in _clk_url_candidates(prefix, year, doy):
                content = _download_with_retry(session, url, timeout_s=timeout_s, label="CLK")
                if content and len(content) > 100_000:
                    (CLK_DIR / fname).write_bytes(content)
                    summary[c]["downloaded"] += 1
                    got = True
                    break
            if not got:
                summary[c]["missing"].append(doy)
            done = summary[c]["have"] + summary[c]["downloaded"]
            if done % 20 == 0:
                print_status(f"{c}: {done}/{len(doys)} days "
                             f"({summary[c]['downloaded']} new)", "INFO")
        s = summary[c]
        print_status(f"{c}: {s['have']} on disk + {s['downloaded']} downloaded, "
                     f"{len(s['missing'])} missing", "SUCCESS")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--doys", type=str, default="1-366")
    ap.add_argument("--centres", type=str, default="cod",
                    help="comma list of cod,gfz,grg,jpl")
    args = ap.parse_args()

    a, b = args.doys.split("-") if "-" in args.doys else (args.doys, args.doys)
    doys = list(range(int(a), int(b) + 1))
    centres = [c.strip().lower() for c in args.centres.split(",") if c.strip()]
    bad = [c for c in centres if c not in CENTRES]
    if bad:
        print_status(f"unknown centres {bad}; valid: {list(CENTRES)}", "ERROR")
        return 2

    print_status(f"Acquiring {args.year} DOY {doys[0]}-{doys[-1]} "
                 f"clock finals for {centres}", "TITLE")
    summary = acquire(args.year, doys, centres)
    if summary is None:
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step_09_acquire_global_clk.json").write_text(json.dumps({
        "year": args.year, "n_doys": len(doys), "centres": centres,
        "summary": {c: {"have": v["have"], "downloaded": v["downloaded"],
                        "n_missing": len(v["missing"])}
                    for c, v in summary.items()},
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    total = sum(v["have"] + v["downloaded"] for v in summary.values())
    print_status(f"done -- {total} centre-days of global receiver clocks available", "SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
