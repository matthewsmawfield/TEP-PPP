"""Generate L1/L5 wide-lane bias clock files for PPP-AR.

The CNES GRG clock files contain GPS L1/L2 wide-lane biases.  For a
Waveshare LC29H(BS)-class receiver tracking L1/L5 (not L1/L2), the L1/L5
wide-lane biases must be computed from the CNES Bias-SINEX observable-
specific signal bias (OSB) values.

The Melbourne-Wubbena wide-lane combination for L1/L5 is:

    MW = (L1 - L5) / lam_WL - (P1 - P5) / lam_WL

where lam_WL = c / (f1 - f5) is the wide-lane wavelength.

The wide-lane satellite bias (WSB) in cycles is:

    WSB = (b_L1 - b_L5) / lam_WL - (b_P1 - b_P5) / lam_WL

where b_* are the CNES OSB values in metres for each observable.

This module downloads the CNES Bias-SINEX file for a given date,
computes the L1/L5 WSB for each GPS satellite, and embeds the values
in a copy of the GRG clock file so RTKLIB can load them via its
RINEX COMMENT-based WL bias parser.
"""
from __future__ import annotations

import gzip
import io
import logging
import os
import re
import urllib.request
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Physical constants
CLIGHT = 299792458.0  # m/s
F_L1 = 1575.42e6  # Hz
F_L5 = 1176.45e6  # Hz
LAM_WL = CLIGHT / (F_L1 - F_L5)  # L1/L5 wide-lane wavelength (m)

# CNES Bias-SINEX observable codes for GPS L1/L5
# L1C = L1 C/A phase; L5I = L5 in-phase phase (mapped to L5Q by receiver)
# C1C = L1 C/A pseudorange; C5Q = L5 Q pseudorange
GPS_OBS_TYPES = ["L1C", "L5I", "C1C", "C5Q"]


def _doy_to_date(year: int, doy: int) -> date:
    return date(year, 1, 1) + __import__("datetime").timedelta(days=doy - 1)


def download_bias_sinex(year: int, doy: int, out_dir: Path) -> Path | None:
    """Download the CNES GRG Bias-SINEX file for a given year/DOY.

    Returns the path to the downloaded .bia file, or None if the
    download fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"GRG0OPSFIN_{year}{doy:03d}0000_01D_01D_OSB.BIA"
    out_path = out_dir / fname
    if out_path.exists() and out_path.stat().st_size > 1000:
        logger.info("Bias-SINEX already cached: %s", out_path)
        return out_path

    # Try CNES FTP server
    urls = [
        f"ftp://igs.ign.fr/pub/igs/products/obs_bias/{year}/{fname}.gz",
        f"ftp://igs.ign.fr/pub/igs/products/obs_bias/{year}/{fname}",
        f"https://igs.ign.fr/pub/igs/products/obs_bias/{year}/{fname}.gz",
    ]
    for url in urls:
        try:
            logger.info("Downloading Bias-SINEX: %s", url)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            if len(data) < 1000:
                continue
            out_path.write_bytes(data)
            logger.info("Bias-SINEX saved: %s (%d bytes)", out_path, len(data))
            return out_path
        except Exception as e:
            logger.debug("Download failed for %s: %s", url, e)
            continue
    logger.warning("Could not download Bias-SINEX for %d DOY %d", year, doy)
    return None


def parse_bias_sinex(bia_path: Path) -> dict[str, dict[str, float]]:
    """Parse a CNES Bias-SINEX file and return OSB values in metres.

    Returns a dict: {sat_id: {obs_type: bias_metres}}
    """
    opener = gzip.open if bia_path.suffix == ".gz" else open
    biases: dict[str, dict[str, float]] = {}
    in_bias = False
    try:
        with opener(bia_path, "rt", errors="ignore") as f:
            for line in f:
                if "+BIAS/SOLUTION" in line:
                    in_bias = True
                    continue
                if "-BIAS/SOLUTION" in line:
                    break
                if not in_bias:
                    continue
                # Format: D SB SAT PT year mo day hr min sec nval bias [std]
                # Example: D SB G01  L1C  2026  9  5  0  0  0.000  1  1.8943E-9
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[0] != "D" or parts[1] != "SB":
                    continue
                sat = parts[2]
                obs_type = parts[3]
                try:
                    # Bias is in seconds; convert to metres
                    bias_s = float(parts[9].replace("D", "E").replace("d", "e"))
                    bias_m = bias_s * CLIGHT
                except (ValueError, IndexError):
                    continue
                biases.setdefault(sat, {})[obs_type] = bias_m
    except Exception as e:
        logger.error("Failed to parse Bias-SINEX %s: %s", bia_path, e)
        return {}
    logger.info("Parsed Bias-SINEX: %d satellites", len(biases))
    return biases


def compute_l1_l5_wl_bias(osb: dict[str, float]) -> float | None:
    """Compute the L1/L5 wide-lane satellite bias in cycles.

    WSB = ((b_L1 - b_L5) - (b_P1 - b_P5)) / lam_WL

    where b_* are the CNES OSB values in metres.

    L5I is mapped to L5Q (the receiver tracks L5Q but CNES provides L5I).
    """
    b_l1 = osb.get("L1C")
    b_l5 = osb.get("L5I") or osb.get("L5Q")
    b_p1 = osb.get("C1C")
    b_p5 = osb.get("C5Q") or osb.get("C5I")
    if b_l1 is None or b_l5 is None or b_p1 is None or b_p5 is None:
        return None
    wsb = ((b_l1 - b_l5) - (b_p1 - b_p5)) / LAM_WL
    return wsb


def generate_wl_bias_clock(
    clk_in: Path,
    clk_out: Path,
    wl_biases: dict[str, float],
    obs_date: date,
) -> bool:
    """Embed L1/L5 WL biases into a copy of a GRG clock file.

    The GRG clock file contains GPS L1/L2 WL biases in COMMENT records.
    This function removes the original GPS WL biases and inserts the
    new L1/L5 WL biases in the same COMMENT format so RTKLIB can parse
    them via its RINEX COMMENT-based WL bias reader.

    The COMMENT keyword must start at column 60 (0-indexed) for RTKLIB's
    fixed-column parser to recognise the records.
    """
    try:
        opener = gzip.open if clk_in.suffix == ".gz" else open
        with opener(clk_in, "rb") as f:
            lines = f.read().decode("ascii", errors="ignore").splitlines()
    except Exception as e:
        logger.error("Failed to read clock file %s: %s", clk_in, e)
        return False

    out_lines: list[str] = []
    bias_inserted = False
    skip_original_gps_wl = False
    date_str = f"{obs_date.year:4d} {obs_date.month:2d} {obs_date.day:2d}"

    for line in lines:
        # Skip original GPS L1/L2 WL bias COMMENT records
        if line.startswith("WL G") and "0102" in line:
            skip_original_gps_wl = True
            continue
        if skip_original_gps_wl and line.startswith("WL G"):
            continue
        skip_original_gps_wl = False

        # Insert L1/L5 WL biases after the first COMMENT line
        # (after "TIME SYSTEM ID" or the blank COMMENT)
        if not bias_inserted and "TIME SYSTEM ID" in line:
            out_lines.append(line)
            # Insert the WL bias header and records
            header = "WIDELANE SATELLITE FRACTIONAL BIASES USED IN THIS SOLUTION"
            out_lines.append(f"{header:60s}COMMENT")
            for sat_id in sorted(wl_biases.keys()):
                wsb = wl_biases[sat_id]
                # Format: WL G01  2026  9  5 12  0  0.000000  1   +9.691064E-01  0102 COMMENT
                prn = int(sat_id[1:])
                bias_str = f"{wsb:+.6E}"
                rec = (f"WL G{prn:02d}  {date_str} 12  0  0.000000"
                       f"  1   {bias_str}  0102")
                out_lines.append(f"{rec:60s}COMMENT")
            bias_inserted = True
            continue

        out_lines.append(line)

    if not bias_inserted:
        logger.warning("Could not find insertion point for WL biases")
        return False

    try:
        clk_out.write_text("\n".join(out_lines) + "\n")
        logger.info("WL bias clock file written: %s (%d WL entries)",
                    clk_out, len(wl_biases))
        return True
    except Exception as e:
        logger.error("Failed to write WL bias clock file: %s", e)
        return False


def generate_wl_bias_for_date(
    year: int,
    doy: int,
    clk_in: Path,
    clk_out: Path,
    bia_dir: Path | None = None,
) -> bool:
    """End-to-end: download Bias-SINEX, compute L1/L5 WL biases, embed in clock file.

    Returns True if the WL bias clock file was successfully generated.
    """
    obs_date = _doy_to_date(year, doy)
    if bia_dir is None:
        bia_dir = Path("data/rinex/bia")
    bia_path = download_bias_sinex(year, doy, bia_dir)
    if bia_path is None:
        logger.warning("Using existing WL bias clock file (no Bias-SINEX available)")
        return False
    osb = parse_bias_sinex(bia_path)
    if not osb:
        return False
    wl_biases: dict[str, float] = {}
    for sat_id, sat_osb in osb.items():
        if not sat_id.startswith("G"):
            continue
        wsb = compute_l1_l5_wl_bias(sat_osb)
        if wsb is not None and abs(wsb) < 20.0:  # sanity check
            wl_biases[sat_id] = wsb
    if len(wl_biases) < 4:
        logger.warning("Too few GPS WL biases computed (%d)", len(wl_biases))
        return False
    return generate_wl_bias_clock(clk_in, clk_out, wl_biases, obs_date)
