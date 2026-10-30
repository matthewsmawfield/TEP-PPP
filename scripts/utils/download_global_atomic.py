#!/usr/bin/env python3
"""Download RINEX observation files for the best atomic clock stations
from the global IGS network via CDDIS.

Downloads RINEX 3 compressed files for:
1. All missing Rb stations (critical for same-type satellite comparison)
2. High-altitude atomic stations (for gravitational potential test)
3. Key metrology stations (NIST, OP71, etc.)
4. Additional Cs and H-Maser stations for statistics
5. Multiple days for temporal stability

Data source: CDDIS (https://cddis.nasa.gov/archive/gnss/data/daily/)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_BASE = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024"
CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"

# Days to download (for temporal stability analysis)
DOYS = ["097", "098", "099", "100", "101", "102", "103"]

# Priority stations to download (from igs_missing_atomic.json)
# Ordered by TEP analysis value: Rb first, then high-altitude, then metrology
PRIORITY_STATIONS = [
    # Rb stations (critical for same-type satellite Rb comparison)
    "BJFS00CHN", "BOGI00POL", "DRDN00IND", "DUBO00CAN", "GOP700CZE",
    "HLFX00CAN", "JPLM00USA", "MCIL00JPN", "SCOR00GRL", "STFU00USA",
    "TASH00UZB", "TNML00TWN", "WUHN00CHN",
    # High-altitude stations
    "MKEA00USA", "PIE100USA", "QUI300ECU", "QUI400ECU",
    # Metrology / unique types
    "NIST00USA", "OP7100FRA", "LCK400IND", "TWTF00TWN",
    # Additional Cs for statistics
    "KIRU00SWE", "USUD00JPN", "TSKB00JPN", "KOUG00GUF",
    "VILL00ESP", "DAEJ00KOR", "SUWN00KOR", "GMSD00JPN",
    # Additional H-Maser for statistics
    "KOKB00USA", "ONSA00SWE", "KOUR00GUF", "AMC400USA",
    "NRC100CAN", "DRAO00CAN", "WSRT00NLD", "NLIB00USA",
    "BREW00USA", "WES200USA", "TIDB00AUS", "ALBH00CAN",
    "MATE00ITA", "MEDI00ITA", "CEBR00ESP", "CHAN00CHN",
    "GOL200USA", "GODN00USA", "GODS00USA",
    # Polar / extreme latitude
    "OHI200ATA", "OHI300ATA", "NABG00NOR", "EUR200CAN",
    # Other Rb
    "BJFS00CHN",
    # Steered / special
    "AGGO00ARG", "REDU00BEL", "SFER00ESP",
    # Additional H-Maser
    "HRAG00ZAF", "HRAO00ZAF", "MSSA00JPN", "ISHI00JPN",
    "STK200JPN", "TSK200JPN", "MDVJ00RUS", "IRKJ00RUS",
    "NOVM00RUS", "ZECK00RUS", "BADG00RUS", "SVTL00RUS",
    "WARK00NZL", "HOB200AUS", "MRL100NZL", "MRL200NZL",
    "CRO100VIR", "RAEG00PRT", "SPT000SWE", "BOR100POL",
    "WDC500USA", "WDC600USA", "USN700USA", "USN900USA",
    "GODN00USA", "GODS00USA", "GOL200USA",
    # Additional Cs
    "BHPL00IND", "BHR300BHR", "BHR400BHR", "EIL300USA", "EIL400USA",
    "MAS100ESP", "MRL100NZL", "MRL200NZL", "OAK100GBR", "OAK200GBR",
    "OSN300KOR", "OSN400KOR", "PBR400IND", "STK200JPN",
    "BAKO00IDN",
]

# Remove duplicates while preserving order
seen = set()
PRIORITY_STATIONS = [s for s in PRIORITY_STATIONS if not (s in seen or seen.add(s))]


def download_file(station: str, doy: str, year: str = "2024") -> bool:
    """Download a RINEX 3 file from CDDIS."""
    fname = f"{station}_R_{year}{doy}0000_01D_30S_MO.crx.gz"
    out_dir = OBS_BASE / doy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    if out_path.exists() and out_path.stat().st_size > 1000:
        return True  # already downloaded

    url = f"{CDDIS_BASE}/{year}/{doy}/24o/{fname}"
    r = subprocess.run(
        ["curl", "-s", "-L", "-o", str(out_path),
         "--connect-timeout", "15", "--max-time", "120",
         "-w", "%{http_code}", url],
        capture_output=True, text=True, timeout=150)

    http_code = r.stdout.strip() if r.stdout else "000"

    if http_code == "200" and out_path.exists() and out_path.stat().st_size > 1000:
        return True
    else:
        if out_path.exists() and out_path.stat().st_size < 1000:
            out_path.unlink()
        return False


def main() -> int:
    print(f"=== Downloading expanded IGS atomic clock network ===")
    print(f"Stations: {len(PRIORITY_STATIONS)}")
    print(f"Days: {DOYS} ({len(DOYS)} days)")
    print(f"Total files: {len(PRIORITY_STATIONS) * len(DOYS)}")
    print()

    # Download in parallel
    tasks = []
    for doy in DOYS:
        for station in PRIORITY_STATIONS:
            tasks.append((station, doy))

    n_success = 0
    n_fail = 0
    n_skip = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, s, d): (s, d) for s, d in tasks}
        for future in as_completed(futures):
            station, doy = futures[future]
            try:
                result = future.result()
                if result:
                    n_success += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

    print(f"\nResults: {n_success} downloaded, {n_fail} failed")

    # Show what we got per day
    for doy in DOYS:
        d = OBS_BASE / doy
        if d.exists():
            n = len(list(d.glob("*.crx.gz")))
            print(f"  DOY {doy}: {n} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
