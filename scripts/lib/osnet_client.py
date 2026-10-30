#!/usr/bin/env python3
"""OS Net API client for fetching RINEX data from UK Ordnance Survey stations.

The OS Net API provides hourly RINEX v3 files for ~100 GNSS reference
stations across Great Britain.  This module fetches the most recent
complete hourly RINEX file for a given station, processes it with
RTKLIB rnx2rtkp to extract per-epoch receiver clock biases, and
returns the data in the same format as the NTRIP-based StationStream.

API documentation: https://docs.os.uk/os-apis/accessing-os-apis/os-net-api
Base URL: https://api.os.uk/positioning/osnet/v1

Authentication: API key passed as query parameter (?key=...).
The key is obtained from the OS Data Hub (osdatahub.os.uk) by creating
an API project and adding the OS Net API.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTKLIB_DIR = PROJECT_ROOT / "RTKLIB_src" / "app" / "console" / "rnx2rtkp"

_OSNET_API_KEY = os.environ.get("OSNET_API_KEY", "")
_BASE_URL = "https://api.os.uk/positioning/osnet/v1"


def _header_lla(rinex_zip: bytes) -> tuple[float, float] | None:
    """Extract (lat, lon) from the APPROX POSITION XYZ in a zipped RINEX header.

    Returns None if the header has no position block or if the position
    is (0, 0, 0) -- a receiver that did not produce a fix, which would
    otherwise sit on Null Island and corrupt the kriging geometry.
    """
    import math
    try:
        with zipfile.ZipFile(io.BytesIO(rinex_zip)) as zf:
            names = [n for n in zf.namelist() if n.endswith((".rnx", ".obs", "o"))]
            if not names:
                names = zf.namelist()
            text = zf.read(names[0]).decode("ascii", errors="ignore")
    except Exception:
        return None
    x = y = z = None
    for line in text.splitlines():
        if "APPROX POSITION XYZ" in line:
            # RINEX 3 uses 3F14.4, but some receivers use slightly
            # different field widths.  Parse by splitting on whitespace
            # and taking the first three numeric tokens.
            parts = line.split()
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except (ValueError, IndexError):
                return None
            break
    if x is None:
        return None
    if abs(x) < 1.0 and abs(y) < 1.0 and abs(z) < 1.0:
        return None
    # ECEF to geodetic (WGS84)
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = 1.0 - (b / a) ** 2
    ep2 = (a / b) ** 2 - 1.0
    p = math.sqrt(x * x + y * y)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * math.sin(th) ** 3,
                    p - e2 * a * math.cos(th) ** 3)
    return (math.degrees(lat), math.degrees(lon))


def _rnx2rtkp_path() -> str:
    """Find the rnx2rtkp binary."""
    for name in ["rnx2rtkp_demo5", "rnx2rtkp"]:
        p = os.environ.get("RNX2RTKP", "")
        if p and Path(p).exists():
            return p
        import shutil
        sp = shutil.which(name)
        if sp:
            return sp
        local = str(RTKLIB_DIR / name)
        if Path(local).exists():
            return local
    raise RuntimeError("rnx2rtkp not found")


def _brdc_doy(path: str) -> int | None:
    """Extract the DOY from a BRDC nav filename (brdcDDD0.YYn)."""
    import re
    m = re.search(r"brdc(\d{3})0\.\d{2}n", Path(path).name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _best_nav() -> str | None:
    """Select the best available nav file for today's observations.

    GPS broadcast ephemeris is valid for only 2-4 hours.  A BRDC file from
    yesterday has completely expired ephemeris that produces wrong satellite
    positions, causing the direct clock computation to fail for every OS Net
    station (observed: only EASN succeeded while SKEE/LINO/SCAU/YEAL all
    returned "no clocks" because the stale brdc2510.26n was used instead of
    the shared nav from the live receiver, which has current ephemeris).

    Prefer the shared nav (current ephemeris from the live receiver's RTCM3
    stream) when the BRDC file is not from today's DOY.  Fall back to the
    stale BRDC only when no shared nav is available.
    """
    brdc = _find_brdc_nav()
    shared = _find_shared_nav()
    if brdc and shared:
        brdc_doy = _brdc_doy(brdc)
        today_doy = datetime.now(timezone.utc).timetuple().tm_yday
        if brdc_doy is not None and brdc_doy != today_doy:
            return shared
        return brdc
    return brdc or shared


def _find_brdc_nav() -> str | None:
    """Find a broadcast ephemeris file for the current DOY."""
    nav_dir = PROJECT_ROOT / "data" / "rinex" / "nav"
    if not nav_dir.exists():
        return None
    now = datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    year = now.year
    # Try current DOY first, then previous.  Reject empty files —
    # failed downloads leave 0-byte shells that rnx2rtkp silently fails on.
    for d in [doy, doy - 1, doy + 1]:
        for ext in [f"{str(year)[2:]}.n", f"{str(year)[2:]}.g", f"{str(year)[2:]}.p"]:
            pattern = f"brdc{d:03d}0.{ext}"
            for m in nav_dir.glob(pattern):
                if m.stat().st_size > 1000:
                    return str(m)
            pattern = f"BRDC{d:03d}0.{ext}"
            for m in nav_dir.glob(pattern):
                if m.stat().st_size > 1000:
                    return str(m)
        pattern = f"BRDC00IGS_R_{year}{d:03d}0000_01D_MN.rnx"
        for m in nav_dir.glob(pattern):
            if m.stat().st_size > 1000:
                return str(m)
    # Try any recent nav file FROM THE CURRENT YEAR.  Returning a file
    # from a previous year gives rnx2rtkp stale ephemeris that produces
    # no $CLK lines (the satellite positions don't match today's
    # constellation), which silently starves the OS Net clock pipeline.
    # Also reject empty files (failed downloads leave 0-byte shells).
    all_nav = sorted(nav_dir.glob("brdc*.??n"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in all_nav:
        if p.stat().st_size < 1000:
            continue
        name = p.name
        if f"_{year}" in name or name.endswith(f".{str(year)[2:]}n"):
            return str(p)
    # Also check RINEX 3 format (BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx)
    all_rnx = sorted(nav_dir.glob("BRDC00IGS_R_*.rnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in all_rnx:
        if p.stat().st_size < 1000:
            continue
        if f"_{year}" in p.name:
            return str(p)
    return None


def _find_shared_nav() -> str | None:
    """Find the most recent shared nav file from the live receiver."""
    shared_dir = PROJECT_ROOT / "data" / "ntrip_live" / "shared_nav"
    if not shared_dir.exists():
        return None
    navs = sorted(shared_dir.glob("*.nav"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if navs and navs[0].stat().st_size > 0:
        return str(navs[0])
    return None


def _find_precise_products() -> tuple[str | None, str | None]:
    """Find SP3 and CLK precise products for satellite clock correction."""
    precise_dir = PROJECT_ROOT / "data" / "ppp_live" / "precise"
    if not precise_dir.exists():
        return None, None
    sp3 = None
    clk = None
    # Find SP3 file (prefer orbit.sp3, then orbit_cod.sp3)
    for name in ["orbit.sp3", "orbit_cod.sp3"]:
        p = precise_dir / name
        if p.exists() and p.stat().st_size > 0:
            sp3 = str(p)
            break
    if not sp3:
        sp3_files = sorted(precise_dir.glob("*.sp3"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if sp3_files:
            sp3 = str(sp3_files[0])
    # Find CLK file (prefer clock_cod_wl.clk for PPP-AR consistency)
    for name in ["clock_cod_wl.clk", "clock.clk", "clock_cod_combined.clk"]:
        p = precise_dir / name
        if p.exists() and p.stat().st_size > 0:
            clk = str(p)
            break
    if not clk:
        clk_files = sorted(precise_dir.glob("*.clk"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if clk_files:
            clk = str(clk_files[0])
    return sp3, clk


def fetch_station_info(station_id: str, api_key: str | None = None) -> dict | None:
    """Fetch station metadata from the OS Net API."""
    key = api_key or _OSNET_API_KEY
    if not key:
        logger.warning("OS Net API key not set (OSNET_API_KEY env var)")
        return None
    url = f"{_BASE_URL}/stations/{station_id}?key={key}"
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except Exception as e:
        logger.warning("OS Net station info for %s failed: %s", station_id, e)
        return None


def fetch_rinex_file(station_id: str, year: int, doy: int, hour: int,
                     api_key: str | None = None) -> bytes | None:
    """Download a specific hourly RINEX file for a station.

    Returns the raw zip file bytes, or None if the file is not available.
    """
    key = api_key or _OSNET_API_KEY
    if not key:
        return None
    url = f"{_BASE_URL}/rinex/{year}/{doy}?key={key}"
    # Retry transient transport failures (SSL EOF, reset, timeout): the
    # OS Net endpoint drops connections sporadically, and without a retry
    # a single flake costs the hour's file until _fill_gaps' 2 h throttle
    # comes back around (observed 2026-10-27: UNEXPECTED_EOF on
    # SWAN/WEYB/PETE/GORS/ATTL list + download calls).  HTTP errors are
    # deterministic (404 = the day has no files) — no retry.
    files = None
    for _att in range(3):
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=30)
            files = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            logger.warning("OS Net RINEX list for %s %d/%d failed: %s",
                           station_id, year, doy, e)
            return None
        except Exception as e:
            if _att == 2:
                logger.warning("OS Net RINEX list for %s %d/%d failed: %s",
                               station_id, year, doy, e)
                return None
            time.sleep(1.5 * (_att + 1))
    # Find the file for this station and hour
    target_hour = f"{hour:02d}00"
    for f in files:
        if (f.get("station") == station_id.upper()
                and f.get("percentageComplete", 0) == 100
                and target_hour in f.get("fileName", "")):
            for _att in range(3):
                try:
                    req = urllib.request.Request(f["url"])
                    resp = urllib.request.urlopen(req, timeout=60)
                    return resp.read()
                except urllib.error.HTTPError as e:
                    logger.warning("OS Net RINEX download for %s failed: %s",
                                   f["fileName"], e)
                    return None
                except Exception as e:
                    if _att == 2:
                        logger.warning("OS Net RINEX download for %s "
                                       "failed: %s", f["fileName"], e)
                        return None
                    time.sleep(1.5 * (_att + 1))
    return None


def fetch_latest_rinex(station_id: str, api_key: str | None = None
                       ) -> tuple[bytes, str] | None:
    """Fetch the most recent complete hourly RINEX file for a station.

    Returns (zip_bytes, filename) or None.
    """
    key = api_key or _OSNET_API_KEY
    if not key:
        logger.warning("OS Net API key not set")
        return None
    now = datetime.now(timezone.utc)
    # Start from the PREVIOUS complete hour (delta_h=1).  The current hour
    # (delta_h=0) is still being recorded — its RINEX file may be listed as
    # 100% complete by the API but rnx2rtkp cannot solve SPP for it because
    # the broadcast ephemeris for the current hour may not yet be available
    # in the daily BRDC file.  This produces empty .stat files with zero
    # $CLK epochs, which the caller interprets as "station unavailable".
    for delta_h in range(1, 48):
        t = now - timedelta(hours=delta_h)
        year = t.year
        doy = t.timetuple().tm_yday
        hour = t.hour
        data = fetch_rinex_file(station_id, year, doy, hour, key)
        if data:
            fname = f"{station_id}_{year}_{doy}_{hour:02d}.rnx.zip"
            logger.info("OS Net: fetched %s (%d bytes)", fname, len(data))
            return data, fname
    logger.warning("OS Net: no complete RINEX found for %s in last 48h", station_id)
    return None


def _compute_direct_clock_from_rinex(
        rnx_path: Path, nav: str | None,
        station_id: str) -> tuple[dict[int, float],
                                  dict[int, dict[str, float]]] | None:
    """Compute per-epoch clock biases using the station's known position.

    Reference stations have surveyed coordinates in the RINEX header
    (APPROX POSITION XYZ).  Instead of running SPP (which estimates
    position AND clock, introducing ~5-20 m of position-error noise into
    the clock), compute the clock directly:

        clk = median_i(pseudorange_i - geometric_range_i)

    where the geometric range uses the KNOWN station position and
    precise satellite positions/clocks from SP3/CLK products.  This
    eliminates position error entirely, leaving only code noise
    (~0.3-3 m per epoch, averages out over epochs) and multipath
    (~2-5 m per satellite, partially cancels in median).

    The resulting clock bias has ~0.5-2 m noise — close to the TEP
    field signal level — versus ~5-20 m for SPP.  This is the key
    upgrade that makes the TEP clock prediction demonstrable on live
    hardware.
    """
    try:
        from scripts.lib.rinex_pseudorange import (
            parse_rinex_header_position, get_measurements)
        from scripts.lib.precise_products import load_cached_products
    except ImportError as e:
        logger.warning("OS Net: direct clock imports failed: %s", e)
        return None

    try:
        station_pos = parse_rinex_header_position(rnx_path)
    except Exception:
        station_pos = None
    if station_pos is None:
        logger.warning("OS Net %s: no APPROX POSITION XYZ in header", station_id)
        return None

    # Load precise products for satellite positions and clock corrections.
    # Parse the epoch date from the RINEX file to determine which day's
    # SP3/CLK products to load.  RINEX 3.0x epoch lines start with ">":
    #   > 2026 09 08 06 00 00.000  0
    # RINEX 2.11 epoch lines start with a 2-digit year:
    #   26  9  8  0  0  0.0000000  0
    # The old code matched the first line where parts[0] was a digit, but
    # RINEX 3.x header lines like " 24 R01  1 R02 -4 ..." (GLONASS slot/frq)
    # have parts[0]=24 (a digit) and >= 6 fields, causing int("R01") to
    # raise ValueError inside the try/except — silently leaving pp=None.
    # Without precise products the clock biases use broadcast ephemeris
    # (~90 m detrended RMS vs ~1.4 m with precise), which exceeds the
    # kriging matrix outlier threshold and prevents the TEP field from
    # being computed at all.
    precise_dir = PROJECT_ROOT / "data" / "ppp_live" / "precise"
    pp = None
    try:
        from datetime import datetime as _dt
        header_done = False
        with open(rnx_path, "r", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                # Track the END OF HEADER marker.  RINEX 2.11 epoch
                # lines only appear after this point; checking before it
                # causes false matches on "PRN / # OF OBS" header lines
                # whose observation counts look like valid dates (e.g.
                # "6 6 7 7 6 6 7 PRN / # OF OBS").
                if "END OF HEADER" in stripped:
                    header_done = True
                    continue
                # RINEX 3.0x epoch line: "> YYYY MM DD HH MM SS.sss ..."
                if stripped.startswith(">"):
                    parts = stripped[1:].split()
                    if len(parts) >= 3:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        doy = _dt(year, month, day).timetuple().tm_yday
                        pp = load_cached_products(year, doy, precise_dir)
                        if pp is not None:
                            pp.set_epoch_doy(doy)
                        break
                    continue
                # RINEX 2.11 epoch line: " YY MM DD HH MM SS.sss ..."
                # Only check after the header ends to avoid false matches
                # on header lines with all-digit fields.
                if not header_done:
                    continue
                parts = stripped.split()
                if (len(parts) >= 6
                        and parts[0].isdigit() and parts[1].isdigit()
                        and parts[2].isdigit()):
                    yy = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    if 0 <= yy <= 99 and 1 <= month <= 12 and 1 <= day <= 31:
                        year = 2000 + yy if yy < 80 else 1900 + yy
                        doy = _dt(year, month, day).timetuple().tm_yday
                        pp = load_cached_products(year, doy, precise_dir)
                        if pp is not None:
                            pp.set_epoch_doy(doy)
                        break
    except Exception:
        pass

    nav_paths = [nav] if nav else []
    # Also include ALL broadcast nav files from the current year, so
    # that RINEX files from a different DOY than the nav file still
    # have matching ephemeris.  The _find_brdc_nav() function returns
    # only the most recent file, which may be from a different day than
    # the OS Net RINEX being processed.
    nav_dir = PROJECT_ROOT / "data" / "rinex" / "nav"
    if nav_dir.exists():
        for sn in sorted(nav_dir.glob("brdc*.??n"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            if str(sn) not in [str(n) for n in nav_paths]:
                if sn.stat().st_size > 1000:
                    nav_paths.append(str(sn))
    # Also include shared GPS broadcast ephemeris from the local receiver.
    # The live receiver's convbin output goes to data/ntrip_live/shared_nav/
    # (written by gps_web's KrigingProxy nav-sharing).  The old path
    # data/ppp_live/nav/ was never created, so the shared ephemeris was
    # silently absent — leaving OS Net RINEX from the current day with no
    # matching broadcast ephemeris (the brdc files are from yesterday).
    for shared_nav_dir in (PROJECT_ROOT / "data" / "ntrip_live" / "shared_nav",
                           PROJECT_ROOT / "data" / "ppp_live" / "nav"):
        if not shared_nav_dir.exists():
            continue
        for sn in sorted(shared_nav_dir.glob("*.nav"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            if str(sn) not in [str(n) for n in nav_paths]:
                nav_paths.append(str(sn))

    try:
        raw = get_measurements(rnx_path, nav_paths, station_pos,
                               precise_products=pp,
                               allow_gps_broadcast_fallback=True)
    except Exception as e:
        logger.warning("OS Net %s: get_measurements failed: %s", station_id, e)
        return None
    if not raw:
        logger.warning("OS Net %s: no measurements extracted", station_id)
        return None

    clocks: dict[int, float] = {}
    persat: dict[int, dict[str, float]] = {}
    absmap: dict[int, float] = {}
    for rm in raw:
        sod = int(round(rm["sod"])) % 86400
        # Absolute unix axis (rm["sod"] is GPS-absolute seconds) — the
        # per-station sidereal-echo store needs day-resolution keys;
        # sod alone collides across days.
        absmap[sod] = float(rm["sod"]) + 315964800.0 - 18.0
        clk_vals = []
        raw_res: dict[str, float] = {}
        for sid in rm.get("sat_ids", []):
            sp = rm["sat_positions"].get(sid)
            pr = rm["pseudoranges"].get(sid)
            if sp is not None and pr is not None and np.isfinite(pr):
                geom = float(np.linalg.norm(sp - station_pos))
                raw_res[sid] = float(pr) - geom
                # Use only GPS and Galileo (best precise product
                # coverage) for the clock median.
                if sid[0] in ('G', 'E'):
                    clk_vals.append(pr - geom)
        if clk_vals:
            clk_m = float(np.median(clk_vals))
            clocks[sod] = clk_m
            # Per-satellite residual = deviation of each satellite's
            # (pseudorange - geometric range) from the station clock.
            # Includes GLONASS — its station-receiver code biases are
            # absorbed by the consumer's per-sat DCB gate.
            if len(raw_res) >= 4:
                persat[sod] = {s: v - clk_m for s, v in raw_res.items()}

    if clocks:
        logger.info("OS Net %s: direct clock extracted %d epochs "
                    "(range %.2f-%.2f m)", station_id, len(clocks),
                    min(clocks.values()), max(clocks.values()))
    return (clocks, persat, absmap) if clocks else None


def _filter_clock_outliers(clocks: dict[int, float],
                           station_id: str = "") -> dict[int, float]:
    """Reject outlier epochs from a per-epoch clock series.

    Two rules, applied in SOD order:

    * |clk| > 1000 m is junk — a healthy SPP clock bias is typically
      < 1000 m; larger values indicate SPP divergence from incomplete
      ephemeris (the shared nav file from the live receiver may not have
      all satellites visible at a distant OS Net station).
    * A > 50 m jump vs the previous accepted epoch is a cycle slip or
      ephemeris loss.

    Free-running clock accommodation — the same shape test the NTRIP
    drain path applies (the _bigclk whitelist): a station whose receiver
    clock sits at a large but STABLE DC offset (unsteered quartz/atomic
    at km scale — observed: KLYN at ~-85 km with ~0.4 m jitter,
    2026-10-27) carries a usable detrended field curve.  The matrix
    builder removes a per-station linear trend and the accumulated-offset
    predictor differences the DC away, so the offset is harmless
    downstream.  The big-DC exemption applies only when EVERY epoch is
    beyond the small-DC bound and a linear detrend leaves <=15 m
    residual — a mixed regime or a runaway ramp keeps the old
    reject-everything behaviour.
    """
    sorted_sods = sorted(clocks.keys())
    _clkv = np.asarray([clocks[s] for s in sorted_sods], float)
    _big_ok = False
    if len(_clkv) >= 12 and bool(np.all(np.abs(_clkv) > 1000.0)):
        _tt = np.arange(len(_clkv), dtype=float)
        _A = np.stack([np.ones_like(_tt), _tt], 1)
        _cf, *_ = np.linalg.lstsq(_A, _clkv, rcond=None)
        _res = float(np.sqrt(np.mean((_clkv - _A @ _cf) ** 2)))
        _big_ok = _res <= 15.0
        if _big_ok:
            logger.info("OS Net %s: stable free-running clock "
                        "(median %.0f m, detrended rms %.2f m) — "
                        "keeping %d epochs", station_id,
                        float(np.median(_clkv)), _res, len(_clkv))
    filtered: dict[int, float] = {}
    prev_clk = None
    for sod in sorted_sods:
        clk = clocks[sod]
        if abs(clk) > 1000.0 and not _big_ok:
            continue
        if prev_clk is not None and abs(clk - prev_clk) > 50.0:
            # Large jump — likely ephemeris loss, skip this epoch
            continue
        filtered[sod] = clk
        prev_clk = clk
    n_rejected = len(clocks) - len(filtered)
    if n_rejected > 0:
        logger.info("OS Net %s: filtered %d/%d outlier epochs",
                    station_id, n_rejected, len(clocks))
    return filtered


def process_rinex_to_clocks(rinex_zip: bytes, station_id: str,
                            brdc_nav: str | None = None
                            ) -> tuple[dict[int, float],
                                       dict[int, dict[str, float]]] | None:
    """Process a RINEX zip file to extract per-epoch clock biases.

    Uses the direct clock computation method (pseudorange minus geometric
    range at the station's surveyed position) as the primary path, which
    gives ~0.5-2 m clock noise versus ~5-20 m for SPP.  Falls back to
    RTKLIB rnx2rtkp .stat $CLK parsing if the direct method fails (e.g.
    no APPROX POSITION XYZ in the RINEX header).

    Returns ({epoch_sod: clock_bias_m},
             {epoch_sod: {sat_id: per_sat_residual_m}}) or None on
    failure.  The per-satellite residuals are (pseudorange - geometric
    range) minus the epoch clock median; they are empty when the SPP
    fallback produced the clocks.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(rinex_zip)) as zf:
            rnx_name = zf.namelist()[0]
            with zf.open(rnx_name) as rf:
                rnx_data = rf.read()
    except Exception as e:
        logger.warning("OS Net: unzip failed: %s", e)
        return None

    # Find broadcast ephemeris — prefer the broadcast nav file (which has
    # complete ephemeris for all satellites), then fall back to the shared
    # nav from the live receiver (which may be incomplete for distant
    # stations).  The shared nav only has ephemeris for satellites the
    # live receiver tracked, which may not include all satellites visible
    # at an OS Net station 50+ km away.
    nav = brdc_nav or _find_brdc_nav() or _find_shared_nav()
    if not nav:
        logger.warning("OS Net: no broadcast ephemeris found for SPP")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        rnx_path = Path(tmpdir) / rnx_name
        rnx_path.write_bytes(rnx_data)

        # --- Primary: direct clock computation ---
        persat: dict[int, dict[str, float]] = {}
        absmap: dict[int, float] = {}
        direct = _compute_direct_clock_from_rinex(rnx_path, nav, station_id)
        clocks: dict[int, float] = {}
        if direct:
            clocks, persat, absmap = direct

        # --- Fallback: RTKLIB rnx2rtkp .stat $CLK parsing ---
        if not clocks:
            sp3, clk = _find_precise_products()
            pos_path = Path(tmpdir) / "spp.pos"
            stat_path = Path(str(pos_path) + ".stat")
            try:
                rtk = _rnx2rtkp_path()
                if not rtk:
                    logger.warning("OS Net: rnx2rtkp not found, direct "
                                   "method also failed for %s", station_id)
                    return None
                # Use -x 3 (trace level 3 for .stat with $CLK lines), -y 1
                cmd = [rtk, "-x", "3", "-m", "10", "-p", "0",
                       "-y", "1", "-o", str(pos_path), str(rnx_path), nav]
                if sp3:
                    cmd.append(str(sp3))
                if clk:
                    cmd.append(str(clk))
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode != 0:
                    logger.warning("OS Net: rnx2rtkp failed: %s",
                                   result.stderr.decode()[:200])
                    return None
            except Exception as e:
                logger.warning("OS Net: rnx2rtkp error: %s", e)
                return None

            clocks = {}
            if not stat_path.exists():
                logger.warning("OS Net: no .stat file produced for %s", rnx_name)
                return None
            try:
                with open(stat_path, "r", errors="ignore") as f:
                    for line in f:
                        p = line.rstrip().split(",")
                        if len(p) < 6:
                            continue
                        if p[0] == "$CLK":
                            # p[3] is sol.stat — $CLK is emitted even on
                            # SOLQ_NONE epochs (dtr=0 or stale), which
                            # produced km-scale junk series in live
                            # station_clocks (2026-09-30).
                            if int(float(p[3])) == 0:
                                continue
                            sod = int(round(float(p[2]) % 86400.0))
                            clk_m = float(p[5])
                            # No magnitude cap here: the outlier filter
                            # below owns the policy, and its stable-big-DC
                            # accommodation needs to see free-running
                            # clocks (km-scale offsets) reach the shape
                            # test rather than die at parse.
                            clocks[sod] = clk_m
            except Exception as e:
                logger.warning("OS Net: .stat parse error: %s", e)
                return None

            if not clocks:
                logger.warning("OS Net: no $CLK epochs in .stat for %s", rnx_name)
                return None

    # Outlier filtering — see _filter_clock_outliers.
    if len(clocks) >= 3:
        clocks = _filter_clock_outliers(clocks, station_id)
        persat = {s: r for s, r in persat.items() if s in clocks}

    if not clocks:
        logger.warning("OS Net %s: all epochs filtered as outliers", station_id)
        return None

    logger.info("OS Net %s: extracted %d clock epochs (range %.2f-%.2f m)",
                station_id, len(clocks),
                min(clocks.values()), max(clocks.values()))
    clocks_abs = {absmap[s]: c for s, c in clocks.items() if s in absmap}
    return clocks, persat, clocks_abs


class OSNetStation:
    """A single OS Net station fetched via the REST API.

    Polls the OS Net API every hour for new RINEX data and processes
    it to extract clock biases.  The clock biases are stored in the
    same format as the NTRIP-based StationStream.
    """

    def __init__(self, station_id: str, api_key: str | None = None,
                 lat: float = 0.0, lon: float = 0.0):
        self.station_id = station_id.upper()
        self.api_key = api_key or _OSNET_API_KEY
        self.lat = lat
        self.lon = lon
        self.clocks: dict[int, float] = {}
        # Per-satellite residuals {sod: {sat_id: resid_m}} — the
        # satellite-specific deviation from the station clock, kept
        # alongside the clock history for the per-sat network
        # correction product.
        self.persat: dict[int, dict[str, float]] = {}
        # Absolute-unix-keyed clock history {unix: clk_m} — feeds the
        # per-station sidereal-echo store (needs >24 h, unlike the
        # SOD-keyed `clocks` buffer which collides across days).
        self.clocks_abs: dict[float, float] = {}
        # Guards clocks/persat: update()/backfill() run on the kriging
        # server's backfill thread while get_clocks()/get_persat() are
        # called from the epoch thread — an unguarded dict copy can hit
        # "dictionary changed size during iteration" mid-merge.
        self._lock = threading.Lock()
        self.nsat: dict[int, int] = {}
        self.last_fetch = 0.0
        self.fetch_interval = 3600.0  # 1 hour (RINEX is hourly)
        # How much history to retain.  The spectral phase Gram needs the
        # receiver and the network to share a >=12 h window -- its uplift for
        # a quartz target is +1.8 points at 4 h but +15.0 at 12 h, because the
        # coupling lives at 6-12 h periods.  Four hours cannot resolve it.
        self.history_hours = float(os.environ.get("OSNET_HISTORY_HOURS", "4"))
        # Wall-clock hours (year, doy, hour) already ingested / attempted.
        # update() only pulls the newest file, so without this a single
        # failed or late-arriving hour leaves a permanent hole in the
        # clock series (observed: EASN missing the 12:00 file left a
        # 3630 s gap that excluded the closest station from the
        # accumulated-offset predictor's spanning set even though the
        # file existed on OS Net).
        self._hours_seen: set[tuple[int, int, int]] = set()
        self._hours_tried: dict[tuple[int, int, int], float] = {}
        # Do NOT cache the nav file at init time — the shared nav file
        # is updated continuously by the live receiver, and a cached
        # reference can become stale if the server runs for hours.
        # Instead, re-discover the nav file on each update() call.

        # Fetch station info if lat/lon not provided
        if lat == 0.0 and lon == 0.0:
            info = fetch_station_info(self.station_id, self.api_key)
            if info:
                # Parse ETRS89 lat/lon from DMS format
                lat_str = info.get("ETRS89_Lat", "")
                lon_str = info.get("ETRS89_Long", "")
                if lat_str and lon_str:
                    self.lat = _parse_dms(lat_str)
                    self.lon = _parse_dms(lon_str)

    def update(self) -> bool:
        """Fetch new RINEX data if available.  Returns True if updated."""
        now = time.time()
        if now - self.last_fetch < self.fetch_interval:
            return False
        self.last_fetch = now
        self._fill_gaps()

        result = fetch_latest_rinex(self.station_id, self.api_key)
        if result is None:
            return False

        rinex_zip, fname = result
        try:
            y, d, h = fname.split(".")[0].split("_")[-3:]
            self._hours_seen.add((int(y), int(d), int(h)))
        except Exception:
            pass
        # Re-discover the nav file on each update to avoid stale references.
        # Use _best_nav() which prefers the shared nav (current ephemeris
        # from the live receiver) when the BRDC file is from a previous day
        # and has expired ephemeris.
        nav = _best_nav()
        res = process_rinex_to_clocks(
            rinex_zip, self.station_id, nav)
        if res is None:
            return False
        new_clocks, new_persat, new_abs = res

        # Merge new clocks (keep the most recent hour's data)
        with self._lock:
            self.clocks.update(new_clocks)
            self.persat.update(new_persat)
            self.clocks_abs.update(new_abs)
            self._prune()
        return True

    def _fill_gaps(self) -> int:
        """Re-fetch hourly RINEX files missing from the merged history.

        Scans the last ``history_hours`` wall-clock hours for ones that were
        never ingested and re-fetches them.  Throttled to one attempt per
        2 h per hour and at most two files per call so a permanently absent
        file cannot stall the kriging loop.  Runs inside update(), i.e. at
        most once per hour per station.
        """
        now = datetime.now(timezone.utc)
        nav = None
        got = 0
        # Skip the two most recent hours: their files may still be
        # incomplete on the server (fetch_latest_rinex also starts at
        # delta_h=1).
        for back in range(2, int(self.history_hours) + 2):
            if got >= 2:
                break
            t = now - timedelta(hours=back)
            key = (t.year, t.timetuple().tm_yday, t.hour)
            if key in self._hours_seen:
                continue
            if time.time() - self._hours_tried.get(key, 0.0) < 7200.0:
                continue
            self._hours_tried[key] = time.time()
            try:
                data = fetch_rinex_file(self.station_id, key[0], key[1],
                                        key[2], self.api_key)
            except Exception:
                data = None
            if not data:
                continue
            if nav is None:
                nav = _best_nav()
            try:
                res = process_rinex_to_clocks(data, self.station_id, nav)
            except Exception:
                res = None
            if res:
                clocks, persat, clocks_abs = res
                with self._lock:
                    self.clocks.update(clocks)
                    self.persat.update(persat)
                    self.clocks_abs.update(clocks_abs)
                self._hours_seen.add(key)
                got += 1
        if got:
            with self._lock:
                self._prune()
            logger.info("OS Net %s: refilled %d missing hourly files",
                        self.station_id, got)
        return got

    def _prune(self):
        """Trim to `history_hours`, correctly across the UTC midnight wrap.

        Keys are seconds-of-day, so a plain `max_sod - N*3600` cutoff silently
        discards everything before midnight the moment the window spans it --
        exactly when a >=12 h history is most likely to.  Compare circularly.

        The reference point must be the CURRENT wall-clock SOD, not
        ``max(self.clocks.keys())``: when the buffer spans midnight, the
        maximum SOD is yesterday 23:59 (86370), so today's SOD 0-06:00
        appears 18-24 h away in circular distance and is silently pruned.
        Using the real current time keeps the last ``history_hours`` of
        data regardless of where midnight falls.
        """
        if not self.clocks:
            return
        keep_s = self.history_hours * 3600.0
        now = datetime.now(timezone.utc)
        now_sod = now.hour * 3600 + now.minute * 60 + now.second
        self.clocks = {k: v for k, v in self.clocks.items()
                       if ((now_sod - k) % 86400.0) <= keep_s}
        self.persat = {k: v for k, v in self.persat.items()
                       if ((now_sod - k) % 86400.0) <= keep_s}
        # clocks_abs keeps 5 days (absolute axis — no midnight wrap);
        # the per-station sidereal echo needs >24 h of history, unlike
        # the SOD buffer above.
        _abs_keep = 5 * 86400.0
        _now_t = time.time()
        self.clocks_abs = {k: v for k, v in self.clocks_abs.items()
                           if _now_t - k <= _abs_keep}

    def backfill(self, hours: float | None = None) -> int:
        """Fetch the last `hours` of hourly RINEX, not just the newest file.

        `update()` pulls one hour per call, so a cold start would need `hours`
        wall-clock hours to build the window the Gram requires.  OS Net serves
        historical files, so the history can simply be fetched.
        """
        hours = int(hours or self.history_hours)
        now = datetime.now(timezone.utc)
        nav = _best_nav()
        got = 0
        for back in range(1, hours + 2):
            t = now - timedelta(hours=back)
            try:
                data = fetch_rinex_file(self.station_id, t.year,
                                        t.timetuple().tm_yday, t.hour,
                                        self.api_key)
            except Exception:
                continue
            if not data:
                continue
            try:
                res = process_rinex_to_clocks(data, self.station_id, nav)
            except Exception:
                res = None
            if res:
                clocks, persat, clocks_abs = res
                with self._lock:
                    self.clocks.update(clocks)
                    self.persat.update(persat)
                    self.clocks_abs.update(clocks_abs)
                self._hours_seen.add((t.year, t.timetuple().tm_yday,
                                      t.hour))
                got += 1
        n_clocks = 0
        if got:
            with self._lock:
                self._prune()
                n_clocks = len(self.clocks)
            logger.info("OS Net %s: backfilled %d hourly files (%d epochs)",
                        self.station_id, got, n_clocks)
        return got

    def get_clocks(self) -> dict[int, float]:
        """Return the current clock bias data."""
        with self._lock:
            return dict(self.clocks)

    def get_persat(self) -> dict[int, dict[str, float]]:
        """Return the current per-satellite residual history."""
        with self._lock:
            return {s: dict(r) for s, r in self.persat.items()}

    def get_clocks_abs(self) -> dict[float, float]:
        """Return the absolute-unix-keyed clock history (sidereal store)."""
        with self._lock:
            return dict(self.clocks_abs)


def _parse_dms(dms_str: str) -> float:
    """Parse a DMS coordinate string like 'N 53 38 56.366383' to degrees."""
    import re
    parts = re.split(r"[\s°'\"]+", dms_str.strip())
    if len(parts) < 4:
        return 0.0
    hemi = parts[0]
    d = float(parts[1])
    m = float(parts[2])
    s = float(parts[3])
    val = d + m / 60.0 + s / 3600.0
    if hemi in ("S", "W"):
        val = -val
    return val
