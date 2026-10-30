#!/usr/bin/env python3
"""Download a dense EUREF RINEX network from the EPN CB FTP server.

Downloads RINEX 3 observation files for a configurable set of European
countries and DOYs, placing them in the step_48 RINEX directory structure.
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_BASE = PROJECT_ROOT / "data" / "rinex" / "obs" / "2024"
EPN_BASE = "ftp://epncb.oma.be/pub/obs/2024"

# Central/western European countries for a dense network
EURO_COUNTRIES = {
    "DEU", "FRA", "ITA", "NLD", "BEL", "CHE", "AUT", "CZE", "POL",
    "DNK", "LUX", "SVK", "SVN", "HUN", "ESP", "PRT", "GBR", "IRL",
    "HRV", "ROU", "BUL", "GRC", "SWE", "NOR", "FIN", "LTU", "LVA",
    "EST", "BLR", "UKR", "TUR", "CYP", "SVK",
}

DOYS = ["099", "100", "101", "102", "103", "104", "105", "106"]
MAX_STATIONS = int(os.environ.get("EUREF_MAX_STATIONS", "80"))
PARALLEL = int(os.environ.get("EUREF_PARALLEL", "8"))


def list_files(doy):
    """Get the list of RINEX 3 MO files for a DOY from EPN."""
    url = f"{EPN_BASE}/{doy}/"
    r = subprocess.run(
        ["curl", "-s", "--connect-timeout", "15", "--max-time", "120", url],
        capture_output=True, text=True, timeout=150)
    files = []
    for line in r.stdout.strip().split("\n"):
        if "_MO.crx.gz" in line:
            parts = line.split()
            if parts:
                fname = parts[-1]
                files.append(fname)
    return files


def filter_european(files):
    """Filter to European country codes and R (receiver) marker."""
    result = []
    for f in files:
        # Filename: STATION00CCY_R_YYYYDOY0000_01D_30S_MO.crx.gz
        # Country code is chars 7-9 (0-indexed: 6-8)
        if len(f) < 10:
            continue
        country = f[6:9]
        # Only receiver-type markers (not _S_ stream or _U_ unknown)
        if "_R_" not in f:
            continue
        if country in EURO_COUNTRIES:
            result.append(f)
    return result


def download_file(doy, fname):
    """Download a single RINEX file."""
    dest_dir = OBS_BASE / doy
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    if dest.exists() and dest.stat().st_size > 1000:
        return f"SKIP {doy}/{fname} (exists)"
    url = f"{EPN_BASE}/{doy}/{fname}"
    r = subprocess.run(
        ["curl", "-s", "--connect-timeout", "15", "--max-time", "120",
         "-o", str(dest), url],
        capture_output=True, text=True, timeout=150)
    if dest.exists() and dest.stat().st_size > 1000:
        return f"OK   {doy}/{fname} ({dest.stat().st_size} bytes)"
    return f"FAIL {doy}/{fname}"


def main():
    # Get file list for DOY 100 to select stations
    print("Listing EPN files for DOY 100...")
    all_files = list_files("100")
    euro_files = filter_european(all_files)
    print(f"  Total: {len(all_files)}, European R-type: {len(euro_files)}")

    # Select a consistent set of stations across all DOYs
    # Sort by station ID for reproducibility
    euro_files.sort()
    if len(euro_files) > MAX_STATIONS:
        # Take the first MAX_STATIONS (sorted alphabetically = roughly by region)
        selected_stations = set(f[:4] for f in euro_files[:MAX_STATIONS])
    else:
        selected_stations = set(f[:4] for f in euro_files)
    print(f"  Selected {len(selected_stations)} stations")

    # Download for all DOYs
    all_downloads = []
    for doy in DOYS:
        files = list_files(doy)
        euro = [f for f in files if f[:4] in selected_stations and "_R_" in f]
        for f in euro:
            all_downloads.append((doy, f))

    print(f"Total downloads needed: {len(all_downloads)}")
    print(f"Downloading with {PARALLEL} parallel workers...")

    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(download_file, doy, fname): (doy, fname)
                   for doy, fname in all_downloads}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0 or i + 1 == len(all_downloads):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(all_downloads)}] ok={ok} skip={skip} "
                      f"fail={fail} ({elapsed:.0f}s)")

    print(f"\nDone: {ok} downloaded, {skip} skipped, {fail} failed")
    print(f"Time: {time.time()-t0:.0f}s")

    # Print selected stations
    print(f"\nSelected stations ({len(selected_stations)}):")
    print(" ".join(sorted(selected_stations)))


if __name__ == "__main__":
    main()
