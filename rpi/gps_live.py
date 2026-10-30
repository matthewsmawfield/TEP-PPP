#!/usr/bin/env python3
"""
Live GPS satellite monitor for the Waveshare LC29H(BS) HAT.

Displays a real-time terminal dashboard with:
  - Fix status, coordinates, altitude, HDOP
  - Per-satellite table: PRN, constellation, elevation, azimuth, SNR, used
  - Sky plot (ASCII polar projection)
  - NMEA / MTK packet counters

Usage:
  python3 gps_live.py --port /dev/cu.usbserial-210 --baud 115200
"""
from __future__ import annotations

import math
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gps_reader import GPSReader, GPSFix


# Constellation prefix map
_CONST = {
    "GP": "GPS",
    "GL": "GLO",
    "GA": "GAL",
    "GB": "BDS",
    "QZ": "QZS",
    "GN": "MIX",
}


def _const_name(prn: int, talker_hint: str = "") -> str:
    if talker_hint in _CONST:
        return _CONST[talker_hint]
    if 1 <= prn <= 32:
        return "GPS"
    if 33 <= prn <= 64:
        return "GLO"
    if 65 <= prn <= 96:
        return "GAL"
    if 201 <= prn <= 237:
        return "BDS"
    if 193 <= prn <= 202:
        return "QZS"
    return "???"


def _sky_plot(sats: list, width: int = 40, height: int = 18) -> list[str]:
    """Render an ASCII sky plot. sats = [(prn, el, az, snr, used)]."""
    cx = width // 2
    cy = height // 2
    radius = min(width, height * 2) // 2 - 2

    grid = [[" "] * width for _ in range(height)]

    # Elevation rings
    for el in (0, 30, 60):
        r = int(radius * (1 - el / 90.0))
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            x = cx + int(r * math.sin(rad))
            y = cy - int(r * math.cos(rad) / 2)
            if 0 <= x < width and 0 <= y < height:
                if grid[y][x] == " ":
                    grid[y][x] = "."

    # Cross hairs
    for x in range(width):
        if grid[cy][x] == " ":
            grid[cy][x] = "-"
    for y in range(height):
        if grid[y][cx] == " ":
            grid[y][cx] = "|"
    grid[cy][cx] = "+"

    # Cardinal labels
    if cy > 0:
        grid[0][cx] = "N"
    if cy < height - 1:
        grid[height - 1][cx] = "S"
    if cx > 0:
        grid[cy][0] = "W"
    if cx < width - 1:
        grid[cy][width - 1] = "E"

    # Plot satellites
    for prn, el, az, snr, used in sats:
        if el < 0:
            continue
        r = radius * (1 - el / 90.0)
        rad = math.radians(az)
        x = cx + int(r * math.sin(rad))
        y = cy - int(r * math.cos(rad) / 2)
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "#" if used else "."

    lines = ["".join(row) for row in grid]
    return lines


def _bar(snr: float, width: int = 10) -> str:
    filled = int(snr / 50.0 * width)
    filled = max(0, min(width, filled))
    if snr >= 35:
        c = "="
    elif snr >= 20:
        c = "-"
    else:
        c = "."
    return "[" + c * filled + " " * (width - filled) + "]"


def render(fix: GPSFix, reader: GPSReader) -> str:
    lines = []

    # Header
    lines.append("=" * 72)
    lines.append(
        f"  LC29H(BS) LIVE  |  Fix: {fix.fix_type:6s}  |  "
        f"Sats used: {fix.num_satellites:2d}  |  HDOP: {fix.hdop:4.1f}  |  "
        f"VDOP: {fix.vdop:4.1f}"
    )
    lines.append("=" * 72)

    # Position
    if fix.datetime_utc:
        utc_str = fix.datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        utc_str = "--:--:--"

    lines.append(
        f"  Lat: {fix.latitude:10.6f}  Lon: {fix.longitude:11.6f}  "
        f"Alt: {fix.altitude:7.1f} m"
    )
    lines.append(f"  UTC: {utc_str}  Speed: {fix.speed_knots:.1f} kn  Course: {fix.course:5.1f}")
    lines.append("")

    # Satellite table
    all_sats = sorted(
        fix.satellites.values(),
        key=lambda s: (-s.snr if s.snr > 0 else 999, s.prn),
    )

    if all_sats:
        lines.append(f"  {'PRN':>4s}  {'Const':4s}  {'El':>5s}  {'Az':>5s}  "
                      f"{'SNR':>5s}  {'Bar':12s}  {'Used'}")
        lines.append(f"  {'----':>4s}  {'----':4s}  {'-----':>5s}  {'-----':>5s}  "
                      f"{'-----':>5s}  {'------------':12s}  {'----'}")

        for sat in all_sats:
            used = "YES" if sat.used_in_fix else "no"
            snr_str = f"{sat.snr:.0f}" if sat.snr > 0 else "--"
            lines.append(
                f"  {sat.prn:4d}  {_const_name(sat.prn):4s}  "
                f"{sat.elevation:5.1f}  {sat.azimuth:5.1f}  "
                f"{snr_str:>5s}  {_bar(sat.snr)}  {used}"
            )
    else:
        lines.append("  (no satellite data yet)")

    lines.append("")

    # Sky plot
    sats_list = [
        (s.prn, s.elevation, s.azimuth, s.snr, s.used_in_fix)
        for s in fix.satellites.values()
    ]
    sky = _sky_plot(sats_list)
    lines.append("  Sky Plot (# = used, . = visible):")
    for row in sky:
        lines.append(f"  {row}")

    lines.append("")

    # Counters
    lines.append(
        f"  NMEA: {reader.nmea_count:6d}  |  MTK: {reader.mtk_count:6d}  |  "
        f"UBX: {reader.ubx_count:4d}  |  Errors: {reader.error_count:4d}  |  "
        f"Uptime: {time.time() - reader.last_read_time + 0.001:.0f}s"
    )
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Live GPS satellite monitor")
    parser.add_argument("--port", default="/dev/cu.usbserial-210", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--raw", action="store_true", help="Enable raw output")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval (s)")
    args = parser.parse_args()

    reader = GPSReader(args.port, args.baud, enable_raw=args.raw)
    reader.start()

    def _sigint(_sig, _frame):
        reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    print(f"LC29H(BS) Live Monitor — {args.port} @ {args.baud} baud. Ctrl+C to stop.\n")
    time.sleep(2.0)

    try:
        while True:
            fix = reader.get_fix()
            os.system("clear" if os.name != "nt" else "cls")
            print(render(fix, reader))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        reader.stop()
        print("\nDone.")


if __name__ == "__main__":
    main()
