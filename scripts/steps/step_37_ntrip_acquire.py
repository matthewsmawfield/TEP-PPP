#!/usr/bin/env python3
"""
STEP 37 -- Live NTRIP Stream Acquisition
=========================================

Connects to free public NTRIP casters (AGRS-NL in the Netherlands,
ReNEP in Portugal) and records raw RTCM3 streams from reference-station
mountpoints to timestamped binary files in ``data/ntrip_live/``.  The
recorded files are committed to the repository so the live experiment
is fully reproducible: anyone can replay the exact byte stream through
the deterministic pipeline.  When EUREF IP credentials are available
(NTRIP_USER environment variable), the EUREF IP caster can also be
used for access to the full European Permanent Network.

Design
------
1. Fetch the sourcetable from the free casters (and EUREF IP if
   credentials are available).
2. Select target + reference mountpoints from the Dutch and Portuguese
   national CORS networks (DELF, WSRT, GAIA and surrounding stations).
3. Record all mountpoints concurrently for a configurable duration
   (default 1 hour; longer for a full operational demonstration).
4. Write a manifest JSON recording mountpoint metadata, coordinates,
   timestamps, and byte counts.

The recorded RTCM3 files are converted to RINEX by step_37b and
processed through the same SPP + kriging obstruction protocol as
Step 33, producing a live end-to-end position-domain result that is
directly comparable to the archived Step 33 demonstration.

Usage
-----
    python scripts/steps/step_37_ntrip_acquire.py [--duration 3600]
        [--stations ZIMM BRUX WTZR GRAS GENO]
        [--caster euref]
        [--dry-run]
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # noqa: E402
from scripts.lib.ntrip_client import (  # noqa: E402
    NTRIP_CASTERS, fetch_sourcetable, find_stations,
    record_mountpoints, MountPoint, get_default_caster,
    FREE_CASTERS,
)

OUT = PROJECT_ROOT / "results" / "outputs"
DATA_LIVE = PROJECT_ROOT / "data" / "ntrip_live"

# Default station set: high-quality reference stations from the free
# Dutch (AGRS_NL) and Portuguese (ReNEP) NTRIP casters.  No registration
# required.  These are geodetic-grade receivers (TRIMBLE ALLOY, LEICA
# GR50, SEPT POLARX5E) carrying RTCM 3.3 MSM observables.
#
# If NTRIP_USER is set, the pipeline defaults to EUREF IP instead and
# uses the paper's original European network.
DEFAULT_STATIONS_FREE = [
    "DELF", "WSRT", "KOS1", "AMEL", "EIJS",  # Netherlands (AGRS_NL)
    "GAIA", "BRAG", "AVI1", "BAIO", "MIRA",  # Portugal (ReNEP)
]
DEFAULT_STATIONS_EUREF = [
    "ZIM2", "BRUX", "WTZR", "GRAS", "GENO",
    "POTS", "GOPE", "GRAZ", "MATE",
]


def main():
    default_caster = get_default_caster()
    if default_caster in FREE_CASTERS:
        default_stations = DEFAULT_STATIONS_FREE
    else:
        default_stations = DEFAULT_STATIONS_EUREF

    parser = argparse.ArgumentParser(description="NTRIP live stream recorder")
    parser.add_argument("--duration", type=float, default=3600.0,
                        help="Recording duration in seconds (default 3600)")
    parser.add_argument("--stations", nargs="+", default=default_stations,
                        help="Station 4-char IDs to record")
    parser.add_argument("--caster", default="",
                        choices=list(NTRIP_CASTERS.keys()) + [""],
                        help="NTRIP caster to use (empty = auto-select)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch sourcetable and list matches without recording")
    parser.add_argument("--output-dir", type=str, default=str(DATA_LIVE),
                        help="Directory for recorded RTCM3 files")
    parser.add_argument("--stall-timeout", type=float, default=60.0,
                        help="Reconnect if a stream delivers no data for "
                             "this many seconds (default 60; 0 disables)")
    args = parser.parse_args()

    # Resolve caster: explicit arg > auto-detect
    caster_key = args.caster or default_caster

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_37", log_file_path=log_dir / "step_37.log")
    set_step_logger(logger)

    print_status("STEP 37: Live NTRIP Stream Acquisition", "TITLE")
    print_status(f"Caster: {caster_key} ({NTRIP_CASTERS[caster_key]['host']})", "INFO")
    print_status(f"Target stations: {', '.join(args.stations)}", "INFO")
    print_status(f"Duration: {args.duration:.0f} s ({args.duration/3600:.1f} h)", "INFO")

    # Check credentials for registered casters
    cfg = NTRIP_CASTERS[caster_key]
    if not cfg.get("free", False) and cfg["user"] == "anon":
        print_status(
            "Note: NTRIP_USER/NTRIP_PASSWORD not set.  EUREF IP and IGS-RTS "
            "require free registration.  Register at "
            "https://registration.euref.eu/ and set NTRIP_USER and "
            "NTRIP_PASSWORD environment variables.  Without credentials, "
            "the sourcetable is fetched but mountpoint recording will fail.  "
            "Alternatively, use --caster agrs_nl or --caster renep for "
            "free casters that do not require registration.",
            "WARN")

    # --------------------------------------------------------- sourcetable
    print_status("\nFetching sourcetable...", "PROCESS")
    try:
        # Search the specified caster, then fall back to all free casters
        mps = find_stations(caster_key, station_names=args.stations)
        found_ids = {m.station_id for m in mps}
        missing = set(s.upper() for s in args.stations) - found_ids
        if missing:
            # Search all free casters for missing stations
            extra = find_stations(station_names=list(missing))
            for m in extra:
                if m.station_id not in found_ids:
                    mps.append(m)
                    found_ids.add(m.station_id)
    except Exception as e:
        print_status(f"Failed to fetch sourcetable: {e}", "ERROR")
        # If we cannot reach the caster (e.g. offline, firewall), we
        # still write a manifest so the pipeline can report the failure
        # deterministically rather than crashing.
        OUT.mkdir(parents=True, exist_ok=True)
        manifest = {
            "test": "ntrip_acquire",
            "status": "failed",
            "error": str(e),
            "caster": caster_key,
            "stations_requested": args.stations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / "step_37_ntrip_acquire.json").write_text(
            json.dumps(manifest, indent=2))
        print_status(f"Manifest written: {OUT / 'step_37_ntrip_acquire.json'}", "INFO")
        return

    print_status(f"Found {len(mps)} matching mountpoints:", "INFO")
    for mp in mps:
        print_status(
            f"  {mp.name:12s} {mp.format:16s} {mp.nav_system:20s} "
            f"({mp.lat_deg:.3f}, {mp.lon_deg:.3f}) {mp.country}",
            "INFO")

    if not mps:
        print_status("No matching mountpoints found.", "ERROR")
        OUT.mkdir(parents=True, exist_ok=True)
        manifest = {
            "test": "ntrip_acquire",
            "status": "no_mountpoints",
            "caster": caster_key,
            "stations_requested": args.stations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / "step_37_ntrip_acquire.json").write_text(
            json.dumps(manifest, indent=2))
        return

    if args.dry_run:
        print_status("\n--dry-run: not recording.", "INFO")
        return

    # --------------------------------------------------------- record
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_utc = datetime.now(timezone.utc)
    print_status(f"\nRecording {len(mps)} mountpoints for {args.duration:.0f} s...",
                 "PROCESS")

    recorders = record_mountpoints(mps, output_dir, duration_s=args.duration,
                                   stall_timeout_s=args.stall_timeout)

    # --------------------------------------------------------- manifest
    end_utc = datetime.now(timezone.utc)
    files = []
    total_bytes = 0
    for rec in recorders:
        info = {
            "mountpoint": rec.mountpoint.name,
            "station_id": rec.mountpoint.station_id,
            "format": rec.mountpoint.format,
            "nav_system": rec.mountpoint.nav_system,
            "lat_deg": rec.mountpoint.lat_deg,
            "lon_deg": rec.mountpoint.lon_deg,
            "country": rec.mountpoint.country,
            "file": str(rec.output_path.relative_to(PROJECT_ROOT)),
            "bytes": rec.bytes_written,
            "connected": rec.connected,
            "reconnects": rec.reconnects,
            "stalls": rec.stalls,
            "error": rec.error,
        }
        files.append(info)
        total_bytes += rec.bytes_written
        status = "OK" if rec.error is None else f"ERROR: {rec.error}"
        if rec.reconnects:
            status += f" ({rec.reconnects} reconnects, {rec.stalls} stalls)"
        print_status(
            f"  {rec.mountpoint.name:12s} {rec.bytes_written/1024:8.1f} KB  {status}",
            "INFO")

    n_ok = sum(1 for f in files if f["error"] is None and f["bytes"] > 0)
    print_status(f"\nRecorded {n_ok}/{len(mps)} mountpoints, "
                 f"{total_bytes/1024:.1f} KB total", "SUCCESS")

    manifest = {
        "test": "ntrip_acquire",
        "status": "ok" if n_ok > 0 else "failed",
        "caster": caster_key,
        "caster_host": NTRIP_CASTERS[caster_key]["host"],
        "stations_requested": args.stations,
        "stations_recorded": [f["station_id"] for f in files
                              if f["error"] is None and f["bytes"] > 0],
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "duration_s": args.duration,
        "total_bytes": total_bytes,
        "n_mountpoints": len(mps),
        "n_ok": n_ok,
        "files": files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step_37_ntrip_acquire.json").write_text(json.dumps(manifest, indent=2))
    # Per-run side copy: concurrent acquires on different casters overwrite
    # the shared manifest above, so keep a per-caster record for
    # scripts/p4_merge_manifest.py-style rebuilds.
    run_tag = f"{caster_key}_{start_utc.strftime('%Y%m%d_%H%M%S')}"
    (OUT / f"step_37_ntrip_acquire_{run_tag}.json").write_text(
        json.dumps(manifest, indent=2))
    print_status(f"Manifest saved: {OUT / 'step_37_ntrip_acquire.json'}", "SUCCESS")


if __name__ == "__main__":
    main()
