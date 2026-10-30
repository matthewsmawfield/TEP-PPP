#!/usr/bin/env python3
"""
STEP 78  --  Widen the clock network: acquire COD0OPSFIN.

Every constraint this programme has hit is the same one -- too few INDEPENDENT
station pairs -- and it turns out to be an acquisition gap, not a limit of the
data that exists.  The pipeline downloads COD0MGXFIN (the multi-GNSS
experimental product, 128 receiver stations).  CODE also publishes
COD0OPSFIN, the GPS operational final, on the same CDDIS path:

                        COD0MGXFIN     COD0OPSFIN
    receiver stations        128            318
    quartz / atomic        89 / 32       194 / 88
    unique station pairs   8,128         50,403
    atomic-atomic pairs      496          ~3,828
    coordinate-verified co-located
      quartz/atomic pairs      1              4   (incl. MATE/MAT1)

The co-located count is the important one.  A co-located quartz/atomic pair is
the only configuration that measures screening without a geography confound:
both clocks see the identical field, troposphere, satellites and network datum,
so their difference isolates the oscillator response alone.  The current product
contains exactly one such pair (THTG/THTI), which is why that test could not be
replicated.  COD0OPSFIN contains four within 100 m, including MATE/MAT1 -- and note that MATE and
MAT1 are each present in some product today but never in the SAME one, and
clocks from different analysis centres cannot be differenced because they are
estimated against different datums.

This step downloads COD0OPSFIN 30 s clocks for a full year (~5 MB/day, ~1.8 GB)
and reports the co-located and pair-count coverage gained.

Usage:
  python scripts/steps/step_78_acquire_wide_clock_network.py --year 2024
  python scripts/steps/step_78_acquire_wide_clock_network.py --year 2024 --doys 1-120

Output: data/rinex/clk/COD0OPSFIN_*.CLK.gz
        results/outputs/step_78_acquire_wide_clock_network.json
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import netrc
import sys
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLK_DIR = ROOT / "data" / "rinex" / "clk"
OUT = ROOT / "results" / "outputs" / "step_78_acquire_wide_clock_network.json"
PRODUCT = "COD0OPSFIN"


def auth():
    try:
        a = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        return (a[0], a[2]) if a else None
    except Exception:
        return None


def gps_week(year, doy):
    return (datetime(year, 1, 1).toordinal() + doy - 1
            - datetime(1980, 1, 6).toordinal()) // 7


def station_coords_in(path):
    """Return CLK header coordinates in metres by four-character station."""
    coords = {}
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            for ln in f:
                if "SOLN STA NAME / NUM" not in ln:
                    continue
                toks = ln.split("SOLN STA NAME / NUM")[0].split()
                try:
                    # RINEX CLK SOLN STA coordinates are millimetres.
                    coords[toks[0][:4].upper()] = 1e-3 * np.array(
                        [float(toks[-3]), float(toks[-2]), float(toks[-1])])
                except (ValueError, IndexError):
                    continue
    except Exception:
        return {}
    return coords


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--doys", default="1-366", help="e.g. 1-366 or 100-140")
    args = ap.parse_args()
    lo, _, hi = args.doys.partition("-")
    doys = range(int(lo), int(hi or lo) + 1)

    CLK_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    a = auth()
    if a:
        s.auth = a
    else:
        print("WARNING: no urs.earthdata.nasa.gov entry in ~/.netrc; CDDIS may refuse")

    got, skipped, failed = 0, 0, []
    for doy in doys:
        name = f"{PRODUCT}_{args.year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        dst = CLK_DIR / name
        if dst.exists() and dst.stat().st_size > 500_000:
            skipped += 1
            continue
        url = (f"https://cddis.nasa.gov/archive/gnss/products/"
               f"{gps_week(args.year, doy)}/{name}")
        try:
            r = s.get(url, timeout=300)
            if r.status_code == 200 and len(r.content) > 500_000:
                dst.write_bytes(r.content)
                got += 1
            else:
                failed.append((doy, r.status_code, len(r.content)))
        except Exception as e:
            failed.append((doy, repr(e)[:60], 0))
        if (got + skipped) % 30 == 0 and (got + skipped):
            print(f"  {got} downloaded, {skipped} already present, "
                  f"{len(failed)} failed", flush=True)

    have = sorted(CLK_DIR.glob(f"{PRODUCT}_*_30S_CLK.CLK.gz"))
    cov = {}
    if have:
        coords = station_coords_in(have[len(have) // 2])
        st = set(coords)
        cp = json.loads((ROOT / "data" / "meta" / "colocated_pairs.json").read_text())
        ct = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
        metadata_pairs = sorted({(c["atomic"], c["quartz"]) for c in cp
                                 if c["atomic"] in st and c["quartz"] in st})
        pair_distances = {p: float(np.linalg.norm(coords[p[0]] - coords[p[1]]))
                          for p in metadata_pairs}
        pairs = [p for p in metadata_pairs if pair_distances[p] <= 100.0]
        rejected = [{"atomic": p[0], "quartz": p[1],
                     "separation_m": pair_distances[p]}
                    for p in metadata_pairs if pair_distances[p] > 100.0]
        known = [n for n in st if n in ct]
        cov = {
            "n_stations": len(st),
            "n_quartz": sum(1 for n in known if ct[n][0] == "QUARTZ"),
            "n_atomic": sum(1 for n in known if ct[n][0] == "ATOMIC"),
            "n_unique_station_pairs": len(st) * (len(st) - 1) // 2,
            "colocated_quartz_atomic_pairs": [list(p) for p in pairs],
            "n_colocated": len(pairs),
            "coordinate_threshold_m": 100.0,
            "distance_rejected_metadata_pairs": rejected,
        }
        print(f"\ncoverage from {have[len(have)//2].name}:")
        print(f"  {cov['n_stations']} stations "
              f"({cov['n_quartz']} quartz, {cov['n_atomic']} atomic)")
        print(f"  {cov['n_unique_station_pairs']:,} unique station pairs")
        print(f"  {cov['n_colocated']} co-located quartz/atomic pairs: {pairs}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "step": 78, "product": PRODUCT, "year": args.year,
        "timestamp": datetime.now().astimezone().isoformat(),
        "downloaded": got, "already_present": skipped,
        "n_failed": len(failed), "failed_sample": failed[:10],
        "days_on_disk": len(have), "coverage": cov,
        "next": "register COD0OPSFIN in scripts/lib/tep_data.py, then re-run "
                "step_77 (co-located screening), step_74 (short-baseline anisotropy) "
                "and step_75c (screening-stratified lambda) on the wider network",
    }, indent=2))
    print(f"\n{got} downloaded, {skipped} present, {len(failed)} failed "
          f"-> {len(have)} days on disk")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
