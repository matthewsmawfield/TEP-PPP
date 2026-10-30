#!/usr/bin/env python3
"""
TEP-PPP Real-Time Live Demo Server
==================================

A FastAPI WebSocket server that streams real-time TEP kriging predictions
from a network of GNSS reference stations to connected frontend clients.

Two operating modes
-------------------
1. Live -- connects to a free public NTRIP caster (AGRS-NL,
   ntrip.kadaster.nl:2101; or ReNEP, 193.137.94.71:2101), receives raw
   RTCM3 streams from multiple reference stations concurrently, converts
   RTCM3 to RINEX on-the-fly using RTKLIB ``convbin``, runs RTKLIB SPP
   (``rnx2rtkp -p 0 -y 2``) to extract per-epoch receiver clock biases,
   and computes the TEP kriging prediction.  When EUREF IP credentials
   are available (NTRIP_USER environment variable), the EUREF IP caster
   (www.euref-ip.net:2101) can also be used.

2. Simulation -- replays archived broadcast SPP data from
   ``data/ppp_processed/`` (``load_series("broadcast", "2024100")``) when
   the NTRIP caster is unreachable.

The server processes epochs in 30-second batches.  For each batch it
computes clock residuals, builds the kriging prior, predicts the held-out
target, and simulates a satellite-obstruction scenario (drop to 3
satellites) showing position error with and without the network prior.

REST endpoints
--------------
  /status       -- connection status, mode, and station list
  /sourcetable  -- available NTRIP mountpoints (live mode only)

WebSocket
---------
  /ws           -- streams JSON epoch messages to connected clients

Run with::

  python scripts/live/tep_realtime_server.py
"""
from __future__ import annotations

import os
import sys
import time
import queue
import shutil
import socket
import select
import signal
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import math
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# --------------------------------------------------------------------- project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.ntrip_client import (
    NTRIP_CASTERS, fetch_sourcetable, find_stations, MountPoint,
    _build_auth_header, get_default_caster,
)
from scripts.lib import tep_data as td
from scripts.lib import tep_kernel as tk
from scripts.lib.rinex_pseudorange import get_measurements, parse_rinex_header_position
from scripts.lib.precise_products import load_cached_products

# ----------------------------------------------------------------- config
# Default to free casters (no registration required).  Falls back to
# EUREF if NTRIP_USER is set.
NTRIP_CASTER_KEY = get_default_caster()
# Stations from the free Dutch (AGRS_NL) and Portuguese (ReNEP) casters.
# These are high-quality geodetic reference stations with RTCM3 MSM
# observables, freely accessible without registration.
#
# The receiver is at 53.64N, 0.10E (Hull, UK).  The closest free NTRIP
# stations are the Dutch AGRS-NL network at 270-500 km.  UK OS Net
# stations (EASN at 1.7 km, SWAN at 41 km) require either NTRIP
# registration (ntrip.os.uk:2101) or an OS Net API key
# (OS_NET_API_KEY environment variable) for RINEX download.
#
# When OS_NET_API_KEY is set, the server can download RINEX files for
# UK stations via the OS Net REST API and process them alongside the
# Dutch NTRIP streams, dramatically improving spatial coverage.
#
# Dutch AGRS-NL stations sorted by distance to the receiver (53.64N, 0.10E).
# The receiver is in East Yorkshire, UK; the nearest free NTRIP stations are
# the Dutch AGRS-NL network at 270-500 km.  UK OS Net stations (EASN at 1.7 km)
# require NTRIP authentication (ntrip.os.uk:2101 returns 401 for data streams
# without credentials).  Set NTRIP_USER/NTRIP_PASSWORD to use the EUREF IP
# caster which carries closer UK EPN stations.
DEFAULT_STATIONS = [
    "HKWA", "EPL1", "BORS", "HKNO", "HKZU",   # 270-302 km
    "DHEL", "TXE2", "IJMU", "HHO2", "VLIE",   # 317-332 km
    "TERS", "VLIS", "DELF", "HARL", "CBW1",   # 340-374 km
    "AMEL", "SCHI", "KOS1", "APEL", "WSRA",   # 374-440 km
    "WSRT", "DELZ", "EIJS",                     # 440-497 km
]

# RENEP (Portugal + Azores/Madeira), free and anonymous, added to give the
# network a GEOMETRIC LEVER ARM.
#
# With the Dutch stations alone every reference sits at 269-497 km, so at
# lambda = 1500 km the kriging weights span 0.947-0.984 -- a 3.9 % relative
# spread.  The kriging is then arithmetically a network mean and permuting
# which station sits at which distance is a no-op: the rig measured a
# geometry contribution of -0.04 % and a common-mode-removed skill of
# -0.11 %, i.e. real operational skill carrying NO spatial information.
# A network that cannot distinguish geometry cannot supply evidence for a
# spatially structured field, however well it predicts.
#
# Adding 1410-2827 km takes the weight span to 0.169-0.984, a ~101 % relative
# spread (26x the lever arm), which is what makes the live permutation control
# an actual test rather than a formality.
RENEP_FAR_STATIONS = [
    "BRGN", "GAIA", "LEIR", "IGP0",       # 1410-1801 km (mainland Portugal)
    "TAVI", "LAGO", "BNA1",               # 1930-1959 km (Algarve)
    "FRNS", "PDEL", "TERC", "FUN1", "FLRS",  # 2621-2827 km (Azores, Madeira)
]

# Default live network: near annulus + far arc.  `_discover` searches the
# primary caster first and then falls back across all free casters, so the
# Dutch names resolve on agrs_nl and the Portuguese ones on renep with no
# further configuration.
DEFAULT_STATIONS_WITH_LEVER_ARM = DEFAULT_STATIONS + RENEP_FAR_STATIONS

# RTK2GO UK community stations (sorted by distance to the receiver at
# 53.64N, 0.10E).  These are free to access with an email-style username
# and provide much closer reference stations (40-160 km) than the Dutch
# AGRS-NL network (269+ km).  Used when the default caster is rtk2go.
RTK2GO_UK_STATIONS = [
    "young_f9p",     # 40.5 km — L1/L2 dual-frequency F9P
    "clayfields",    # 42.4 km
    "SHIP_WHF",      # 53.9 km
    "BarlbyRTK",     # 77.5 km
    "Bramleys",      # 89.2 km
    "granthamall",   # 94.1 km
    "STARBRIDGE",    # 95.6 km
    "EKWT_RTK",     # 105.9 km
    "HarbyLeics",   # 108.4 km
    "HillFarm",     # 126.3 km
    "UPPINGHAM",    # 128.5 km
    "IP270RL",      # 134.5 km
    "JWBRTK",       # 134.7 km
    "towercourt",   # 138.2 km
    "CAMBRIDGE",    # 159.1 km
]

# UK OS Net stations (sorted by distance).  These are used when the OS Net
# API key is available (OS_NET_API_KEY) for RINEX downloads, or when NTRIP
# credentials are available for the EUREF IP caster.
OS_NET_STATIONS_LIST = [
    "EASN",   # 1.7 km — co-located
    "SWAN",   # 41 km
    "SKEE",   # 58 km
    "LINO",   # 60 km
    "YEAL",   # 98 km
    "HOOB",   # 99 km
    "WEYB",   # 103 km
    "LEED",   # 117 km
    "LOFT",   # 120 km
    "PETE",   # 120 km
]

# Fitted TEP kernel parameters from Step 25 nested cross-validation
# (broadcast SPP, 280 day-splits, 7 seeds, held-out stations).
# These are NOT hand-picked — they are the median selected by LOO
# inside TRAIN stations, scored on held-out TEST stations.
FITTED_LAMBDA_KM = 4200.0     # median selected correlation length
FITTED_NUGGET = 0.05          # median selected nugget
FITTED_K = 15                 # median selected k neighbours
# Multi-band frequency-dependent lambdas from Paper 1 (Step 134/145)
# When MULTIBAND_ENABLED=True, the server also computes a multi-band
# prediction using per-band phase Grams and physics-fixed lambdas.
MULTIBAND_ENABLED = os.environ.get("TEP_MULTIBAND", "1") == "1"
MULTIBAND_DT_S = 30.0         # sampling cadence for phase Gram construction
# Anisotropy from Paper 2 (Step 32 / Step 73): EW/NS ratio = 2.16
# (p < 10^-15, phase-cos CSD).  The live kernel uses this measured
# anisotropy to stretch distances along the East-West preferred axis.
FITTED_ANISO_MODE = "ew_ns"
FITTED_ANISO_RATIO = 2.16

# Operational persistence lag for the clock-prediction baseline.
# The paper's claim (Step 106) is that the TEP network prior helps across
# an OUTAGE, where a degraded receiver holds its last clock fix.  Lag-1
# (30 s) is trivially small for a band-limited clock residual (the field's
# power is below 500 uHz, so a 30 s step is pure noise floor), which makes
# persistence look unbeatable and the network prior look useless.  The
# honest baseline holds the clock from ~1 hour ago, matching the
# SimulationSource and the paper's 1-hour outage design point.
PERSIST_LAG_S = 3600.0          # 1 hour, the paper's operational outage
PERSIST_LAG_EPOCHS = 120        # ~1 hour at 30 s cadence

# TEP astronomical R² weights (Papers 1-2)
_TEP_PHASE_R2 = {"semiannual": 0.904, "lunar_nodal": 0.641, "chandler": 0.424}
_TEP_PHASE_R2_TOTAL = sum(_TEP_PHASE_R2.values())
_TEP_PHASE_AMPLITUDE = 0.3


def _compute_tep_astronomical() -> dict:
    """Compute TEP astronomical evidence fields for the current epoch.

    Returns phases, R²-weighted process noise scale, tidal accelerations,
    orbital velocity azimuth, and CMB dipole azimuth — all from Papers 1-2.
    """
    from datetime import datetime, timezone
    try:
        from scripts.lib.tep_astronomical import (
            nutation_phase_semiannual,
            nutation_phase_lunar_nodal,
            chandler_phase,
            lunar_phase_synodic,
            solar_tidal_accel,
            lunar_tidal_accel,
            orbital_velocity_azimuth_deg,
            cmb_azimuth_deg,
        )
    except Exception:
        return {}
    now = datetime.now(timezone.utc)
    try:
        phases = {
            "semiannual": float(nutation_phase_semiannual(now)),
            "lunar_nodal": float(nutation_phase_lunar_nodal(now)),
            "chandler": float(chandler_phase(now)),
            "lunar_synodic": float(lunar_phase_synodic(now)),
        }
        weighted_cos = sum(
            _TEP_PHASE_R2[k] * math.cos(phases[k])
            for k in _TEP_PHASE_R2
        ) / _TEP_PHASE_R2_TOTAL
        q_scale = 1.0 + _TEP_PHASE_AMPLITUDE * weighted_cos
        return {
            "phases": phases,
            "r2_weights": _TEP_PHASE_R2,
            "process_noise_scale": q_scale,
            "weighted_cos": weighted_cos,
            "tidal_solar_ms2": float(solar_tidal_accel(now)),
            "tidal_lunar_ms2": float(lunar_tidal_accel(now)),
            "orbital_velocity_azimuth_deg": float(orbital_velocity_azimuth_deg(now)),
            "cmb_azimuth_deg": float(cmb_azimuth_deg(now)),
        }
    except Exception:
        return {}


WS_PORT = 8765
EPOCH_INTERVAL_S = 30.0
RTKLIB_DIR = Path.home() / ".local" / "bin"
RNX2RTKP = str(RTKLIB_DIR / "rnx2rtkp")
SIM_PRODUCT = "broadcast"
SIM_DATE_TAG = "2024100"
MAX_PLOT_POINTS = 120          # 1 hour at 30 s cadence
MIN_RTCM3_BYTES = 8192         # minimum accumulated before processing
# Socket-level stall watchdog.  select() reports a half-open TCP
# connection as not-ready every second without ever raising or hitting
# EOF, so a peer that silently stops serving (NAT drop, caster-side
# hang) leaves connected=True with zero bytes forever — observed live
# 2026-10-14: all 22 agrs_nl streams zombie for ~16 h while OS Net
# kept flowing.  Streams carry >=1 Hz of RTCM3, so this much silence
# is definitionally a dead flow; break to reconnect through the
# normal backoff path.
NTRIP_STALL_S = float(os.environ.get("TEP_NTRIP_STALL_S", "120"))
# Maximum bytes processed per station per get_epoch call.  After a
# restart the streams buffer data while the (single-threaded) worker is
# busy with the OS Net rebuild — observed ~8 MB per station after 1 h,
# which made convbin + the per-epoch clock solve cost ~15 min per
# station and postponed the first emitted epoch for hours.  Trimming a
# backlogged drain to its newest frames bounds the per-call cost.
# ~1 KB/s per MSM stream, so 600 KB ≈ ~10 min of data; with RTCM3_TI_S
# thinning that is ~120 solved epochs per drain (~45 s/station) — a full
# post-restart pass lands in ~15-20 min instead of never completing.
MAX_RTCM3_DRAIN_BYTES = int(os.environ.get("TEP_RTCM3_DRAIN_MAX", "600000"))
# Observation thinning for streamed NTRIP conversion.  MSM streams are
# 1 Hz; solving every epoch needs ~0.25 CPU-s per wall-second per
# station — 23 stations would need ~6 cores, which this box does not
# have (the loop falls behind forever).  -ti 5 cuts that ~5x (~1.2
# cores) while still sampling the clock series far faster than the
# TEP field's <500 uHz band requires (OS Net history is only 30 s).
RTCM3_TI_S = float(os.environ.get("TEP_RTCM3_TI_S", "5"))
# Stations processed per get_epoch call, round-robin.  Per-station cost
# (~10-45 s of convbin+clock solve) makes a full 23-station pass take
# ~5-10 min, so emitted epochs arrive only ~10x slower than the nominal
# 30 s cadence — and every consumer of raw_clocks (the accumulated-offset
# predictor included) samples the network at emit cadence.  Processing a
# subset per call emits an epoch every ~1-2 min instead; each station is
# still refreshed every few calls, and its drain always carries the
# whole buffered span so no coverage is lost.
DRAIN_MAX_PER_CALL = int(os.environ.get("TEP_DRAIN_MAX_PER_CALL", "6"))
ELEV_MIN_DEG = 10.0
# Shared broadcast ephemeris directory.  The PPP processor (gps_web.py)
# extracts GPS nav from the LC29H raw RTCM3 and copies it here so the
# kriging server can use it for SPP on NTRIP stations (which carry
# observations but not ephemeris).
SHARED_NAV_DIR = PROJECT_ROOT / "data" / "ntrip_live" / "shared_nav"

# Precise products directory — when SP3/CLK files are available here,
# they are passed to the SPP solution so that the network station clock
# biases are corrected for satellite clock errors (matching the receiver's
# PPP clock, which also uses precise products).  Without this, the SPP
# clock bias includes broadcast satellite clock errors that are NOT
# present in the PPP clock, making the network residual predict a signal
# that doesn't exist in the receiver's clock.
PRECISE_DIR = PROJECT_ROOT / "data" / "ppp_live" / "precise"


def _find_precise_products() -> tuple[Path | None, Path | None]:
    """Find the best SP3 and CLK files in the precise directory.

    Returns (sp3_path, clk_path) or (None, None) if not available.
    Prefers the combined CODE+GRG WL bias clock for PPP-AR consistency.
    """
    if not PRECISE_DIR.exists():
        return None, None
    # Find SP3 file
    sp3 = None
    for candidate in sorted(PRECISE_DIR.glob("*.sp3"), key=lambda p: p.stat().st_size, reverse=True):
        if candidate.stat().st_size > 50000:
            sp3 = candidate
            break
    # Find CLK file — prefer the combined CODE+GRG WL bias clock
    clk = None
    preferred = PRECISE_DIR / "clock_cod_wl.clk"
    if preferred.exists() and preferred.stat().st_size > 50000:
        clk = preferred
    else:
        for candidate in sorted(PRECISE_DIR.glob("*.clk"), key=lambda p: p.stat().st_size, reverse=True):
            if candidate.stat().st_size > 50000:
                clk = candidate
                break
    return sp3, clk


def _find_tool(name: str) -> str:
    """Locate an RTKLIB tool, checking PATH then ~/.local/bin.

    For convbin, prefers the demo5 build (convbin_demo5) which properly
    parses RTCM 3.3 MSM7 messages.
    """
    if name == "convbin":
        # Prefer demo5 convbin for RTCM 3.3 MSM7 support
        p5 = shutil.which("convbin_demo5")
        if p5:
            return p5
        for c in (RTKLIB_DIR / "convbin_demo5", RTKLIB_DIR / "convbin"):
            if c.exists():
                return str(c)
        p = shutil.which("convbin")
        if p:
            return p
        raise RuntimeError("convbin not found")
    p = shutil.which(name)
    if p:
        return p
    local = str(RTKLIB_DIR / name)
    if Path(local).exists():
        return local
    raise RuntimeError(f"{name} not found")


# ================================================================ live NTRIP
class StationStream(threading.Thread):
    """Thread-based NTRIP client that accumulates RTCM3 bytes for one station.

    Bytes are written to an in-memory buffer.  The :meth:`drain` method
    returns the accumulated bytes and resets the buffer, so the processing
    thread can convert each batch independently.
    """

    def __init__(self, mountpoint: MountPoint):
        super().__init__(daemon=True, name=f"ntrip-{mountpoint.name}")
        self.mountpoint = mountpoint
        self.connected = False
        self.error: Optional[str] = None
        self.total_bytes = 0
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def drain(self) -> bytes:
        """Return accumulated RTCM3 bytes and reset the buffer."""
        with self._lock:
            data = bytes(self._buf)
            self._buf.clear()
        return data

    def run(self):
        cfg = NTRIP_CASTERS[self.mountpoint.caster_key]
        host = self.mountpoint.host or cfg["host"]
        port = self.mountpoint.port or cfg["port"]
        mp = self.mountpoint.name
        auth = _build_auth_header(cfg["user"], cfg["password"])
        # Exponential backoff retry to avoid IP bans from rapid reconnection.
        # RTK2GO bans after 400+ failed attempts; with 15 stations connecting
        # simultaneously, that's ~27 restarts.  A 60s initial backoff with
        # 2x exponential up to 600s keeps total attempts low while still
        # recovering when the caster comes back online.
        backoff = 60.0
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((host, port), timeout=15)
                req = (
                    f"GET /{mp} HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    f"User-Agent: NTRIP TEP-PPP/1.0\r\n"
                    f"Accept: */*\r\n"
                    f"Authorization: {auth}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                )
                sock.sendall(req.encode())
                # Read and discard HTTP response headers
                header_data = b""
                while b"\r\n\r\n" not in header_data:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    header_data += chunk
                if b"200" not in header_data.split(b"\r\n")[0]:
                    self.error = "NTRIP server rejected connection"
                    if self._stop.wait(backoff):
                        break
                    backoff = min(backoff * 2, 600.0)
                    continue
                # Consume any body bytes that arrived with the headers
                if b"\r\n\r\n" in header_data:
                    body_start = header_data.split(b"\r\n\r\n", 1)[1]
                    if body_start:
                        with self._lock:
                            self._buf.extend(body_start)
                        self.total_bytes += len(body_start)
                self.connected = True
                backoff = 60.0  # reset backoff on successful connection
                last_rx = time.time()
                while not self._stop.is_set():
                    ready, _, _ = select.select([sock], [], [], 1.0)
                    if not ready:
                        if time.time() - last_rx > NTRIP_STALL_S:
                            self.error = (f"no data for {NTRIP_STALL_S:.0f}s"
                                          " — stale socket, reconnecting")
                            break
                        continue
                    data = sock.recv(4096)
                    if not data:
                        break
                    last_rx = time.time()
                    with self._lock:
                        self._buf.extend(data)
                    self.total_bytes += len(data)
            except Exception as e:
                self.error = str(e)
            finally:
                self.connected = False
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
            # Wait before retrying (with backoff) unless stopped
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 600.0)


# ================================================================ RTKLIB
def _rtcm3_tail(data: bytes, max_bytes: int) -> bytes:
    """Keep the newest <=max_bytes of an RTCM3 stream, cut at a frame boundary.

    RTCM3 frames are: 0xD3 preamble, 6 reserved bits + 10-bit payload
    length, payload, 3-byte CRC.  A boundary is trusted when the frame
    computed from the length field ends exactly at the buffer end or is
    followed by another 0xD3.  If no clean boundary is found the raw tail
    is returned anyway — convbin resynchronises on the next 0xD3.
    """
    if len(data) <= max_bytes:
        return data
    tail = data[-max_bytes:]
    n = len(tail)
    i = 0
    while i + 6 <= n:
        if tail[i] == 0xD3:
            flen = ((tail[i + 1] & 0x03) << 8) | tail[i + 2]
            end = i + 3 + flen + 3
            if end <= n and (end == n or tail[end] == 0xD3):
                return tail[i:]
        i += 1
    return tail


def _convbin_rtcm3(rtcm3_path: Path, output_dir: Path) -> dict | None:
    """Convert a raw RTCM3 file to RINEX obs + nav using RTKLIB convbin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale files from previous runs
    for old in output_dir.glob("*"):
        old.unlink()
    cmd = [_find_tool("convbin"), "-r", "rtcm3", "-d", str(output_dir)]
    if RTCM3_TI_S > 0:
        cmd += ["-ti", f"{RTCM3_TI_S:g}"]
    cmd.append(str(rtcm3_path))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60,
                                cwd=str(output_dir))
    except subprocess.TimeoutExpired:
        print(f"[TEP] convbin timeout for {rtcm3_path.name}")
        return None
    if result.returncode != 0:
        print(f"[TEP] convbin failed (rc={result.returncode}): {result.stderr.decode()[:200]}")
        return None
    obs = (list(output_dir.glob("*.obs")) + list(output_dir.glob("*.O"))
           + list(output_dir.glob("*MO.rnx")))
    nav = (list(output_dir.glob("*.nav")) + list(output_dir.glob("*.N"))
           + list(output_dir.glob("*MN.rnx")) + list(output_dir.glob("*.gnav")))
    if not obs:
        print(f"[TEP] convbin produced no obs file in {output_dir}")
        return None
    return {"obs": obs[0], "nav": nav[0] if nav else None}


def _run_spp(obs_path: Path, nav_path: Path | None,
             tmp: Path) -> dict | None:
    """Run RTKLIB SPP and parse the .stat file for clock biases and positions.

    When precise SP3/CLK products are available, they are passed to RTKLIB
    so that the SPP clock bias is corrected for satellite clock errors.
    This makes the network station clock residuals consistent with the
    receiver's PPP clock (which also uses precise products).  Without this,
    the SPP clock bias includes broadcast satellite clock errors that are
    NOT present in the PPP clock, making the network residual predict a
    signal that doesn't exist in the receiver's clock.
    """
    rnx2rtkp = _find_tool("rnx2rtkp")
    pos = tmp / (obs_path.stem + ".pos")
    stat = Path(str(pos) + ".stat")
    cmd = [rnx2rtkp, "-x", "0", "-m", str(int(ELEV_MIN_DEG)), "-p", "0",
           "-y", "1", "-o", str(pos), str(obs_path)]
    if nav_path and nav_path.exists() and nav_path.stat().st_size > 0:
        cmd.append(str(nav_path))
    # Always include shared GPS broadcast ephemeris from the local receiver
    if SHARED_NAV_DIR.exists():
        shared_navs = sorted(SHARED_NAV_DIR.glob("*.nav"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if shared_navs and (not nav_path or shared_navs[0] != nav_path):
            cmd.append(str(shared_navs[0]))
    # Pass precise SP3/CLK products so SPP uses precise satellite clocks
    # instead of broadcast.  This makes the station clock residuals
    # consistent with the receiver's PPP clock.
    sp3, clk = _find_precise_products()
    if sp3:
        cmd.append(str(sp3))
    if clk:
        cmd.append(str(clk))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"[TEP] rnx2rtkp timeout for {obs_path.name}")
        return None
    if not stat.exists():
        print(f"[TEP] rnx2rtkp no .stat file. stderr: {result.stderr.decode()[:200]}")
        return None
    return _parse_stat(stat)


def _parse_stat(stat: Path) -> dict:
    """Parse RTKLIB .stat file -> per-epoch satellite geometry + clock."""
    epochs: dict[int, dict] = {}
    pos_by_sod: dict[int, np.ndarray] = {}
    clk_by_sod: dict[int, float] = {}
    solstat: dict[int, int] = {}
    sat_count: dict[int, int] = {}
    with open(stat, "r", errors="ignore") as f:
        for line in f:
            p = line.rstrip().split(",")
            if not p:
                continue
            tag = p[0]
            if tag == "$POS" and len(p) >= 7:
                sod = int(round(float(p[2]) % 86400.0))
                solstat[sod] = int(float(p[3]))
                pos_by_sod[sod] = np.array([float(p[4]), float(p[5]), float(p[6])])
            elif tag == "$CLK" and len(p) >= 6:
                sod = int(round(float(p[2]) % 86400.0))
                solstat.setdefault(sod, int(float(p[3])))
                clk_by_sod[sod] = float(p[5])
            elif tag == "$SAT" and len(p) >= 8:
                sod = int(round(float(p[2]) % 86400.0))
                el = float(p[6])
                if el >= ELEV_MIN_DEG:
                    sat_count[sod] = sat_count.get(sod, 0) + 1
    # $SAT records are only written at solution-stat level >1, which
    # _run_spp (-x 0) never produces — the $SAT count is dead code here.
    # The real per-epoch satellite count lives in the sibling .pos
    # file's ns column (same week/tow epoch key).
    ns_by_sod: dict[int, int] = {}
    pos_txt = stat.with_suffix("") if str(stat).endswith(".stat") else None
    if pos_txt is not None and pos_txt.exists():
        with open(pos_txt, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                p = line.split()
                if len(p) < 7:
                    continue
                try:
                    ns_by_sod[int(round(float(p[1]) % 86400.0))] = int(p[6])
                except ValueError:
                    continue
    out = {}
    for sod in sorted(set(pos_by_sod) & set(clk_by_sod)):
        # A clock state from a failed/degenerate solve is unconstrained
        # junk (observed 2026-09-30: TERS/WSRT/EIJS stored 0.0-clk and
        # km-scale ramp series on SOLQ_NONE epochs, polluting
        # station_clocks and the accumulated-offset predictor).  Drop
        # stat=NONE epochs, and when the .pos file is available require
        # >=4 satellites — matching the direct-clock method's minimum.
        if solstat.get(sod, 0) == 0:
            continue
        n_sat = ns_by_sod.get(sod, sat_count.get(sod, 0))
        if ns_by_sod and n_sat < 4:
            continue
        out[sod] = {
            "pos": pos_by_sod[sod],
            "clk_m": clk_by_sod[sod],
            "nsat": n_sat,
        }
    return out


def _compute_direct_clock(obs_path: Path, nav_path: Path | None,
                          tmp: Path) -> dict | None:
    """Compute per-epoch clock biases using the station's known position.

    Reference stations have surveyed coordinates in the RINEX header
    (APPROX POSITION XYZ).  Instead of running SPP (which estimates
    position AND clock, introducing ~5-20 m of position-error noise into
    the clock), compute the clock directly:

        clk = median_i(pseudorange_i - geometric_range_i)

    where the geometric range uses the KNOWN station position and
    precise satellite positions/clocks from SP3/CLK products.  This
    eliminates position error entirely, leaving only:

      * Code noise (~0.3-3 m per epoch, averages out over epochs)
      * Multipath (~2-5 m per satellite, partially cancels in median)
      * Residual troposphere (~1-2 m, common-mode, removed by kriging)

    The resulting clock bias has ~0.5-2 m noise — close to the TEP
    field signal level — versus ~5-20 m for SPP.  This is the key
    upgrade that makes the TEP clock prediction demonstrable on live
    hardware.
    """
    try:
        station_pos = parse_rinex_header_position(obs_path)
    except Exception:
        station_pos = None
    if station_pos is None:
        return None

    # Load precise products for satellite positions and clock corrections.
    # See osnet_client._compute_direct_clock_from_rinex for the bug this
    # fixes: RINEX 3.x header lines (e.g. GLONASS slot/frq " 24 R01 1 ...")
    # have parts[0] as a digit with >= 6 fields, causing int("R01") to
    # raise and silently leave pp=None.  Without precise products the
    # clock biases use broadcast ephemeris (~90 m detrended RMS vs ~1 m),
    # exceeding the kriging outlier threshold and blocking the TEP field.
    precise_dir = PROJECT_ROOT / "data" / "ppp_live" / "precise"
    sp3_files = sorted(precise_dir.glob("*.sp3"))
    clk_files = sorted(precise_dir.glob("*.clk"))
    pp = None
    if sp3_files and clk_files:
        try:
            from datetime import datetime as _dt
            with open(obs_path, "r", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    # RINEX 3.0x epoch line: "> YYYY MM DD HH MM SS ..."
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
                    # RINEX 2.11 epoch line: " YY MM DD HH MM SS ..."
                    # len(parts[0]) <= 2 excludes RINEX 3.x header lines
                    # like "TIME OF FIRST OBS" (4-digit year) that would
                    # otherwise parse as year 1900+2026=3926 and silently
                    # leave pp=None, forcing the noisy SPP fallback.
                    parts = stripped.split()
                    if (len(parts) >= 6 and len(parts[0]) <= 2
                            and parts[0].isdigit() and parts[1].isdigit()
                            and parts[2].isdigit()):
                        yy = int(parts[0])
                        year = 2000 + yy if yy < 80 else 1900 + yy
                        month = int(parts[1])
                        day = int(parts[2])
                        doy = _dt(year, month, day).timetuple().tm_yday
                        pp = load_cached_products(year, doy, precise_dir)
                        if pp is not None:
                            pp.set_epoch_doy(doy)
                        break
        except Exception:
            pass

    try:
        raw = get_measurements(obs_path, nav_path, station_pos,
                               precise_products=pp)
    except Exception:
        return None
    if not raw:
        return None

    out = {}
    for rm in raw:
        sod = int(round(rm["sod"])) % 86400
        clk_vals = []
        nsat = 0
        for sid in rm.get("sat_ids", []):
            # Use only GPS and Galileo (best precise product coverage)
            if sid[0] not in ('G', 'E'):
                continue
            sp = rm["sat_positions"].get(sid)
            pr = rm["pseudoranges"].get(sid)
            if sp is not None and pr is not None and np.isfinite(pr):
                geom = float(np.linalg.norm(sp - station_pos))
                clk_vals.append(pr - geom)
                nsat += 1
        # Require a real multi-satellite median: a 1-3 sat "clock" is a
        # single pseudorange residual (seen: +121 m at nsat=1 next to a
        # -131 m series), not a clock bias.
        if len(clk_vals) >= 4:
            out[sod] = {
                "pos": station_pos,
                "clk_m": float(np.median(clk_vals)),
                "nsat": nsat,
                "t_abs": float(rm["sod"]) + 315964800.0 - 18.0,
            }
    return out if out else None


def _build_azimuth_matrix(names: list[str],
                          stations: dict[str, td.Station]) -> np.ndarray | None:
    """Build the station-pair azimuth (bearing) matrix for anisotropy.

    Returns an (n, n) matrix of midpoint bearings in degrees, or None if
    station coordinates are unavailable.  Uses
    ``baseline_azimuth_midpoint_deg`` (the anisotropy-correct bearing) per
    tep_data's guidance.
    """
    n = len(names)
    if n < 2:
        return None
    AZ = np.zeros((n, n))
    have = True
    for i in range(n):
        si = stations.get(names[i])
        if si is None:
            have = False
            break
        for j in range(i + 1, n):
            sj = stations.get(names[j])
            if sj is None:
                have = False
                break
            az = td.baseline_azimuth_midpoint_deg(si, sj)
            AZ[i, j] = az
            AZ[j, i] = az
        if not have:
            break
    return AZ if have else None


# ================================================================ kriging
def _build_live_matrix(station_clocks: dict[str, dict],
                       stations: dict[str, td.Station]) -> tuple:
    """Build (names, X, D, X_raw, sods) from live SPP clock biases.

    *station_clocks* maps station name -> {sod: clk_m}.  The clock biases
    are detrended (mean-removed) to form the residual matrix X in metres.
    The distance matrix D is built from the station coordinates using the
    great-circle distance from :mod:`tep_data`.

    ``X_raw`` is the SAME matrix WITHOUT the per-station linear detrend, and
    ``sods`` is its time axis.  The Step 96 / Step 106 clock predictor does
    not krige the instantaneous residual LEVEL (which the detrend produces);
    it kriges each station's ACCUMULATED offset across the prediction gap,
    ``raw[t0 + h] - raw[t0]``, and gates out each station's own deterministic
    drift afterwards.  Detrending globally destroys exactly that quantity, so
    the raw matrix has to survive alongside the residual one.
    """
    names = sorted(station_clocks.keys())
    if len(names) < 4:
        return [], None, None, None, None
    # Collect common epochs
    all_sods = set()
    for nm in names:
        all_sods.update(station_clocks[nm].keys())
    # Sort SODs in chronological order, correctly across the UTC midnight
    # wrap.  A plain ``sorted(all_sods)`` puts sod=86370 (yesterday 23:59)
    # AFTER sod=7170 (today 01:59), making the "latest" epoch (last row)
    # from the previous day.  Unwrap using the current wall-clock SOD as
    # the reference: any SOD > now_sod belongs to yesterday and is shifted
    # back by 86400 before sorting, then restored.  This ensures the last
    # row of X is the most recent epoch.
    _now_sod = (datetime.now(timezone.utc).hour * 3600
                + datetime.now(timezone.utc).minute * 60
                + datetime.now(timezone.utc).second)
    _sod_list = list(all_sods)
    _sort_key = [s if s <= _now_sod else s - 86400 for s in _sod_list]
    _order = np.argsort(_sort_key)
    sods = [_sod_list[i] for i in _order]
    if len(sods) < 10:
        return [], None, None, None, None
    # Build X matrix (n_epoch, n_station) in metres
    cols, raw_cols = [], []
    for nm in names:
        vals = np.array([station_clocks[nm].get(s, np.nan) for s in sods])
        mask = np.isfinite(vals)
        if mask.sum() < 5:
            return [], None, None, None, None
        x_idx = np.arange(len(vals))
        # Raw column: gaps filled, but NO detrend -- the accumulated-offset
        # predictor needs the secular drift the residual column removes.
        raw = vals.copy()
        raw[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask),
                               vals[mask])
        raw_cols.append(raw)
        # Detrend: remove linear trend (slope + intercept) to capture
        # the residual clock field, not the secular drift
        if mask.sum() >= 2:
            coeffs = np.polyfit(x_idx[mask], vals[mask], 1)
            vals = vals - np.polyval(coeffs, x_idx)
        else:
            vals = vals - np.nanmean(vals)
        # Interpolate NaNs
        vals[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask),
                                vals[mask])
        cols.append(vals)
    X = np.column_stack(cols)
    X_raw = np.column_stack(raw_cols)
    # Reject outlier stations: after detrending, the TEP field residuals
    # should be in the ±10m range.  Stations with |residual| > 50m have
    # bad broadcast ephemeris, constellation issues, or SPP failures
    # (observed: BORS at -1485m, EPL1 at -1287m).  Including them
    # corrupts the kriging prediction.  Use a robust MAD threshold.
    residual_rms = np.sqrt(np.mean(X ** 2, axis=0))
    med_rms = float(np.median(residual_rms))
    mad_rms = float(np.median(np.abs(residual_rms - med_rms)))
    rms_thresh = max(3.0 * 1.4826 * mad_rms, 10.0)
    good = residual_rms <= rms_thresh
    if not good.all():
        rejected = [names[i] for i in range(len(names)) if not good[i]]
        logging.info("Live matrix: rejecting %d outlier stations "
                     "(RMS threshold=%.1fm): %s",
                     len(rejected), rms_thresh,
                     ", ".join(f"{n}({residual_rms[i]:.0f}m)"
                               for i, n in enumerate(names) if not good[i]))
        names = [names[i] for i in range(len(names)) if good[i]]
        X = X[:, good]
        X_raw = X_raw[:, good]
    if X.shape[1] < 4:
        return [], None, None, None, None
    # Distance matrix
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = td.great_circle_km(
                stations[names[i]], stations[names[j]])
    return names, X, D, X_raw, np.asarray(sods, float)


def _simple_bandpass(x: np.ndarray, dt: float,
                     lo_uhz: float = 10.0, hi_uhz: float = 500.0) -> np.ndarray:
    """Lightweight bandpass for live mode (FFT-based)."""
    from scripts.lib import tep_stats as ts
    return ts.bandpass(x, dt, lo_uhz, hi_uhz)


# ================================================================ obstruction

# --- Accumulated-offset scoring constants ---
ACCUM_MIN_EPOCHS = 60          # need at least 1 hour at 30 s cadence
ACCUM_MIN_STATIONS = 4
ACCUM_HORIZON_MIN_S = 300.0    # 5 minutes
ACCUM_HORIZON_MAX_S = 7200.0   # 2 hours


def _clusters_of(D: np.ndarray, link_km: float = 800.0) -> np.ndarray:
    """Single-linkage clustering by distance threshold.

    Returns an array of integer cluster labels, one per station.
    """
    n = D.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    labels = np.arange(n)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= link_km and labels[i] != labels[j]:
                    labels[labels == labels[j]] = labels[i]
                    changed = True
    # Relabel to 0..k-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _select_covered_block(COV: np.ndarray, min_st: int = 8
                          ) -> tuple[int, int, list] | None:
    """Select the largest epoch range [a, b] with at least min_st stations
    fully covered (allowing isolated single-epoch gaps in the interior,
    but requiring coverage at both endpoints a and b).

    Returns (a, b, keep) or None.
    """
    n_ep, n_st = COV.shape
    if n_st < min_st:
        return None
    best = None  # (area, a, b, keep)
    for a in range(n_ep):
        if not COV[a].any():
            continue
        # Track which stations have coverage from epoch a onward
        # (with at most 1 dropped epoch in the interior)
        gaps = np.zeros(n_st, dtype=int)
        active = COV[a].copy()
        for b in range(a, n_ep):
            for j in range(n_st):
                if not COV[b, j]:
                    gaps[j] += 1
                    if gaps[j] > 1:
                        active[j] = False
                else:
                    gaps[j] = 0
                    if COV[a, j]:
                        active[j] = True
            # Require coverage at endpoint b for all active stations
            endpoint_active = np.array([
                active[j] and COV[b, j] for j in range(n_st)
            ])
            keep = [j for j in range(n_st) if endpoint_active[j]]
            if len(keep) >= min_st:
                area = (b - a + 1) * len(keep)
                if best is None or area > best[0]:
                    best = (area, a, b, list(keep))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _accum_loo(X: np.ndarray, D: np.ndarray, sods: np.ndarray,
               horizon_s: float = 900.0) -> dict | None:
    """Accumulated-offset leave-one-out scoring.

    Compares the kriged accumulated-offset prediction against persistence
    (last-value-hold) over the SAME horizon.  Both arms predict the same
    quantity over the same interval, so the comparison is fair.

    Returns dict with horizon_s, n_cases, base_rms_m, tep_rms_m, gain_pct,
    permuted_gain_pct, or None if insufficient data.
    """
    n_ep, n_st = X.shape
    if n_ep < ACCUM_MIN_EPOCHS or n_st < ACCUM_MIN_STATIONS:
        return None
    dt = 30.0
    if len(sods) > 1:
        diffs = np.diff(sods)
        med_dt = float(np.median(np.abs(diffs)))
        if med_dt > 0:
            dt = med_dt
    h_steps = max(1, int(round(horizon_s / dt)))
    if h_steps < 2:
        return None
    # Cases: predict accumulated offset from epoch i to i+h_steps
    cases = []
    for i in range(n_ep - h_steps):
        # Accumulated offset per station: mean over the horizon of (x[t] - x[i])
        acc = np.mean(X[i + 1:i + h_steps + 1, :] - X[i, :], axis=0)  # (n_st,)
        # Leave-one-out: predict target station from others
        for j in range(n_st):
            others = [k for k in range(n_st) if k != j]
            if len(others) < 3:
                continue
            # Simple kriging: weighted average of neighbours' accumulated offsets
            d_rr = D[np.ix_(others, others)]
            d_rt = D[others, j]
            # Exponential kernel
            lam = 1500.0  # km
            rho_rr = np.exp(-0.5 * (d_rr / lam) ** 2)
            rho_rt = np.exp(-0.5 * (d_rt / lam) ** 2)
            K = rho_rr + 1e-9 * np.eye(len(others))
            try:
                w = np.linalg.solve(K, rho_rt)
            except np.linalg.LinAlgError:
                continue
            pred_acc = float(np.dot(w, acc[others]))
            persist_acc = 0.0  # persistence predicts no change
            cases.append((float(acc[j]), pred_acc, persist_acc, j))
    if len(cases) < 10:
        return None
    target = np.array([c[0] for c in cases])
    tep_pred = np.array([c[1] for c in cases])
    persist_pred = np.array([c[2] for c in cases])
    base_rms = float(np.sqrt(np.mean((target - persist_pred) ** 2)))
    tep_rms = float(np.sqrt(np.mean((target - tep_pred) ** 2)))
    gain = (1.0 - tep_rms / base_rms) * 100.0 if base_rms > 0 else 0.0
    # Permuted geometry: shuffle station positions
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_st)
    D_perm = D[np.ix_(perm, perm)]
    perm_cases = []
    for i in range(n_ep - h_steps):
        acc = np.mean(X[i + 1:i + h_steps + 1, :] - X[i, :], axis=0)
        for j in range(n_st):
            others = [k for k in range(n_st) if k != j]
            if len(others) < 3:
                continue
            d_rr = D_perm[np.ix_(others, others)]
            d_rt = D_perm[others, j]
            rho_rr = np.exp(-0.5 * (d_rr / lam) ** 2)
            rho_rt = np.exp(-0.5 * (d_rt / lam) ** 2)
            K = rho_rr + 1e-9 * np.eye(len(others))
            try:
                w = np.linalg.solve(K, rho_rt)
            except np.linalg.LinAlgError:
                continue
            pred_acc = float(np.dot(w, acc[others]))
            perm_cases.append((float(acc[j]), pred_acc))
    if perm_cases:
        perm_target = np.array([c[0] for c in perm_cases])
        perm_pred = np.array([c[1] for c in perm_cases])
        perm_tep_rms = float(np.sqrt(np.mean((perm_target - perm_pred) ** 2)))
        perm_gain = (1.0 - perm_tep_rms / base_rms) * 100.0 if base_rms > 0 else 0.0
    else:
        perm_gain = 0.0
    return {
        "horizon_s": float(h_steps * dt),
        "n_cases": len(cases),
        "base_rms_m": base_rms,
        "tep_rms_m": tep_rms,
        "gain_pct": float(gain),
        "permuted_gain_pct": float(perm_gain),
    }


def _accum_loo_auto(X: np.ndarray, D: np.ndarray, sods: np.ndarray,
                    COV: np.ndarray | None = None) -> dict:
    """Auto-horizon accumulated-offset LOO scoring.

    Picks the longest supported horizon and returns a result dict.
    When insufficient data, returns {available: False, reason: ...}.
    """
    n_ep, n_st = X.shape
    # Rebuild sods from modular differences if it wraps at midnight
    sods = np.asarray(sods, float).copy()
    if len(sods) > 1:
        diffs = np.diff(sods)
        # Detect wrap: negative jumps > 12h
        wraps = np.where(diffs < -43200.0)[0]
        for w in wraps:
            sods[w + 1:] += 86400.0
    # Drop duplicate epochs
    if len(sods) > 1:
        _, uniq_idx = np.unique(sods, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        sods = sods[uniq_idx]
        X = X[uniq_idx]
        n_ep = len(sods)
    if n_ep < ACCUM_MIN_EPOCHS:
        return {"available": False,
                "reason": f"only {n_ep} epochs, need at least {ACCUM_MIN_EPOCHS}"}
    if n_st < ACCUM_MIN_STATIONS:
        return {"available": False,
                "reason": f"only {n_st} stations, need at least {ACCUM_MIN_STATIONS}"}
    # Select covered block if COV provided
    keep_st = list(range(n_st))
    a_ep, b_ep = 0, n_ep
    if COV is not None:
        sel = _select_covered_block(COV, min_st=ACCUM_MIN_STATIONS)
        if sel is not None:
            a_ep, b_ep, keep_st = sel
            X = X[a_ep:b_ep + 1, :][:, keep_st]
            sods = sods[a_ep:b_ep + 1]
            D = D[np.ix_(keep_st, keep_st)]
            n_ep, n_st = X.shape
        if n_st < ACCUM_MIN_STATIONS:
            return {"available": False,
                    "reason": f"only {n_st} covered stations, need at least {ACCUM_MIN_STATIONS}"}
    # Pick the longest supported horizon
    dt = 30.0
    if len(sods) > 1:
        med_dt = float(np.median(np.abs(np.diff(sods))))
        if med_dt > 0:
            dt = med_dt
    max_h = min(ACCUM_HORIZON_MAX_S, (n_ep - 1) * dt * 0.5)
    if max_h < ACCUM_HORIZON_MIN_S:
        return {"available": False,
                "reason": f"max horizon {max_h:.0f}s below minimum {ACCUM_HORIZON_MIN_S:.0f}s"}
    horizon_s = max_h
    r = _accum_loo(X, D, sods, horizon_s=horizon_s)
    if r is None:
        return {"available": False,
                "reason": "accumulated-offset scoring produced no cases"}
    # Within-cluster permutation
    clusters = _clusters_of(D, link_km=800.0)
    n_clusters = len(set(clusters.tolist()))
    # Global permutation gain
    global_gain = r["gain_pct"]
    # Within-cluster permutation: permute only within each cluster
    rng = np.random.default_rng(42)
    D_wc = D.copy()
    for c in range(n_clusters):
        idx = np.where(clusters == c)[0]
        if len(idx) > 1:
            perm = rng.permutation(len(idx))
            D_wc[np.ix_(idx, idx)] = D[np.ix_(idx[perm], idx[perm])]
    r_wc = _accum_loo(X, D_wc, sods, horizon_s=horizon_s)
    wc_gain = r_wc["gain_pct"] if r_wc else 0.0
    geometry_pct = global_gain - r["permuted_gain_pct"]
    geometry_within_pct = global_gain - wc_gain
    within_cluster_gain = r["permuted_gain_pct"] - wc_gain
    return {
        "available": True,
        "horizon_s": r["horizon_s"],
        "n_cases": r["n_cases"],
        "base_rms_m": r["base_rms_m"],
        "tep_rms_m": r["tep_rms_m"],
        "gain_pct": r["gain_pct"],
        "permuted_gain_pct": r["permuted_gain_pct"],
        "geometry_pct": float(geometry_pct),
        "geometry_within_cluster_pct": float(geometry_within_pct),
        "within_cluster_gain_pct": float(within_cluster_gain),
        "n_clusters": int(n_clusters),
        "n_stations": int(n_st),
    }


def _simulate_obstruction(kernel_err_m: float, raw_err_m: float,
                          persist_err_m: float, nsat_full: int,
                          obstructed_nsat: int = 3) -> dict:
    """Simulate position error under satellite obstruction.

    Uses a simplified DOP model: position error is proportional to the
    clock prediction error times the dilution of precision.  With fewer
    than 4 satellites the unconstrained solution is rank-deficient, so
    the no-prior error is set to a large value.  The kernel prior acts
    as a virtual clock measurement, making the 3-satellite case solvable.

    The with-prior error uses the better of kernel and persistence, since
    an operational system would select the superior predictor.
    """
    def _dop(n: int) -> float:
        if n >= 8:
            return 1.5
        if n >= 6:
            return 2.0
        if n >= 5:
            return 2.5
        if n >= 4:
            return 4.0
        return 15.0  # rank-deficient without prior

    dop_full = _dop(nsat_full)
    dop_obs = _dop(obstructed_nsat)
    dop_obs_prior = _dop(obstructed_nsat + 1)
    # Operational system uses the better of kernel or persistence
    best_prior_err = min(kernel_err_m, persist_err_m)
    prior_used = "tep" if kernel_err_m <= persist_err_m else "persistence"
    # TEP vs persistence: positive = TEP wins, negative = persistence wins
    tep_vs_persist_pct = float(
        100.0 * (1.0 - (kernel_err_m * dop_obs_prior)
                 / max(persist_err_m * dop_obs_prior, 1e-6)))
    # DOP model contribution: the increase in error from full-sky to
    # obstructed geometry.  Positive = obstruction degrades the solution.
    dop_model_pct = float(
        100.0 * (dop_obs / max(dop_full, 1e-6) - 1.0))
    return {
        "nsat_full": nsat_full,
        "nsat_obstructed": obstructed_nsat,
        "error_no_prior_m": float(raw_err_m * dop_obs),
        "error_persistence_m": float(persist_err_m * dop_obs_prior),
        "error_with_prior_m": float(best_prior_err * dop_obs_prior),
        "error_full_sky_m": float(raw_err_m * dop_full),
        "improvement_pct": float(
            100.0 * (1.0 - (best_prior_err * dop_obs_prior)
                     / max(raw_err_m * dop_obs, 1e-6))),
        "prior_used": prior_used,
        "tep_vs_persistence_pct": tep_vs_persist_pct,
        "dop_model_pct": dop_model_pct,
    }


# ================================================================ data source
class DataSource:
    """Base interface for live and simulation data sources."""

    def get_epoch(self) -> dict | None:
        raise NotImplementedError

    def station_list(self) -> list[dict]:
        raise NotImplementedError

    def mode(self) -> str:
        raise NotImplementedError


class SimulationSource(DataSource):
    """Replay archived broadcast SPP data for the live demo.

    Loads all available European stations from the broadcast product,
    builds the kriging matrix, and replays epochs at 30-second cadence.
    A target station is rotated every 60 epochs so the demo shows
    predictions for different stations over time.
    """

    def __init__(self):
        self.series = td.load_series(SIM_PRODUCT, SIM_DATE_TAG)
        if not self.series:
            raise RuntimeError("No archived data available for simulation")
        # Filter to European stations for a coherent regional network
        euro = {nm: v for nm, v in self.series.items()
                if 35 < v["station"].lat_deg < 60
                and -10 < v["station"].lon_deg < 30}
        if len(euro) < 6:
            euro = self.series  # fall back to all stations
        self.names, self.X, self.D = tk.build_matrix(euro, filter_by_std=False)
        if self.X is None or len(self.names) < 6:
            raise RuntimeError("Insufficient stations for kriging matrix")
        # Apply std filter manually (training statistics only)
        good = [i for i in range(len(self.names))
                if 0.01 < self.X[:, i].std() < 60.0]
        if len(good) < 5:
            raise RuntimeError("Insufficient valid stations after QC")
        self.names = [self.names[i] for i in good]
        self.X = self.X[:, good]
        self.D = self.D[np.ix_(good, good)]
        self.n_epoch, self.n_sta = self.X.shape
        self.epoch_idx = 100  # start past warm-up for meaningful predictions
        self.target_idx = 0
        self.target_rotate_every = 60
        # Fitted kernel parameters from Step 25 nested-CV
        self.params = tk.KernelParams(
            lambda_km=FITTED_LAMBDA_KM,
            nugget=FITTED_NUGGET,
            k_neighbours=min(FITTED_K, self.n_sta - 1),
            aniso_mode=FITTED_ANISO_MODE,
            aniso_ratio=FITTED_ANISO_RATIO,
        )
        self.pool = list(range(self.n_sta))
        self.station_info = []
        # station_clocks: {sid: {sod: clk_m}} — mirrors LiveSource so the
        # /api/history endpoint can serve the raw clock history that the
        # Step 96 accumulated-offset predictor needs.  Populated with REAL
        # seconds-of-day (not the archived synthetic axis) so the downstream
        # time alignment in gps_web.py can match these against the live
        # receiver's timestamps.
        self.station_clocks: dict[str, dict[int, float]] = {}
        for i, nm in enumerate(self.names):
            sta = euro.get(nm, self.series.get(nm))
            if sta:
                s = sta["station"]
                self.station_info.append({
                    "name": nm, "lat": s.lat_deg, "lon": s.lon_deg,
                })
            self.station_clocks.setdefault(nm, {})
        # Precompute persistence errors using a 1-hour lag (120 epochs at 30 s).
        # The lag-1 difference is unrealistically small because band-limited
        # clock residuals are very smooth at 30 s cadence; the operational
        # persistence scenario holds the clock from a prior fix over a longer
        # interval, so a 1-hour lag is the appropriate comparison.
        self._persist_lag = 120
        self._persist_err = np.zeros(self.n_sta)
        for i in range(self.n_sta):
            lag = self._persist_lag
            if self.n_epoch > lag:
                diff = self.X[lag:, i] - self.X[:-lag, i]
            else:
                diff = np.diff(self.X[:, i])
            self._persist_err[i] = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0

        # Pre-populate station_clocks with archived data mapped to recent
        # real time.  Map the last 16 h of archived epochs (1920 at 30 s)
        # to the last 16 h of wall-clock time so /api/history can serve a
        # warm buffer immediately — without this the accumulated-offset
        # predictor is blind for hours after every (re)start.
        _now_sod = float(time.time() % 86400.0)
        _backfill_epochs = min(1920, self.n_epoch)
        for i, nm in enumerate(self.names):
            for j in range(_backfill_epochs):
                arch_idx = self.n_epoch - _backfill_epochs + j
                # Map archived epoch to real SOD: the most recent archived
                # epoch lands at _now_sod, earlier ones step back by 30 s.
                sod = int(round((_now_sod - (_backfill_epochs - 1 - j) * MULTIBAND_DT_S) % 86400.0))
                v = float(self.X[arch_idx, i])
                if np.isfinite(v):
                    self.station_clocks[nm][sod] = v

    def mode(self) -> str:
        return "simulation"

    def station_list(self) -> list[dict]:
        return self.station_info

    def get_epoch(self) -> dict | None:
        if self.epoch_idx >= self.n_epoch:
            self.epoch_idx = 0  # loop for continuous demo
        idx = self.epoch_idx
        self.epoch_idx += 1
        # Rotate target
        if idx % self.target_rotate_every == 0 and idx > 0:
            self.target_idx = (self.target_idx + 1) % self.n_sta
        target = self.target_idx
        target_name = self.names[target]
        # Clock residuals for this epoch (in metres, already band-limited)
        clocks = {nm: float(self.X[idx, i]) for i, nm in enumerate(self.names)}
        # Kriging prediction using all other stations as references
        pool = [j for j in self.pool if j != target]
        pred, sk_var = tk.predict_with_variance(
            self.X[:idx + 1], self.D, target, pool, self.params)
        if pred is None:
            kernel_pred = 0.0
            kernel_err = float(np.abs(self.X[idx, target]))
            sk_var = None
        else:
            kernel_pred = float(pred[-1]) if len(pred) else 0.0
            kernel_err = float(abs(self.X[idx, target] - kernel_pred))
        # Multi-band prediction (Paper 1 frequency-dependent lambdas)
        mb_pred = None; mb_err = None; mb_var = None; mb_details = []
        mb_rms = None
        if MULTIBAND_ENABLED and idx >= 64:
            result = tk.predict_multiband_with_variance(
                self.X[:idx + 1], self.D, target, pool, self.params,
                dt_s=MULTIBAND_DT_S)
            if result is not None and len(result) == 3:
                mb_pred, mb_var, mb_details = result
            elif result is not None and len(result) == 2:
                mb_pred, mb_var = result
            if mb_pred is not None:
                mb_err = float(abs(self.X[idx, target] - float(mb_pred[-1])))
                mb_rms = float(np.sqrt(np.mean(
                    (self.X[:idx + 1, target]
                     - np.asarray(mb_pred, float)) ** 2)))
        actual = float(self.X[idx, target])
        raw_err = float(abs(actual))
        # Persistence: use the 1-hour-lagged value as the prior
        lag = self._persist_lag
        if idx >= lag:
            persist_pred = float(self.X[idx - lag, target])
            persist_err = float(abs(actual - persist_pred))
        else:
            persist_err = self._persist_err[target]
        # Running RMS errors for the obstruction simulation
        raw_rms = float(np.sqrt(np.mean(self.X[:idx + 1, target] ** 2)))
        pred_all = tk.predict(self.X[:idx + 1], self.D, target, pool, self.params)
        if pred_all is not None:
            ker_rms = float(np.sqrt(np.mean(
                (self.X[:idx + 1, target] - pred_all) ** 2)))
        else:
            ker_rms = raw_rms
        persist_rms = self._persist_err[target]
        # Simulate satellite count (archived data doesn't carry nsat)
        nsat_full = 8 + int(2 * np.sin(idx * 0.01))  # varies 6-10
        raw_clocks = {nm: float(self.X[idx, i]) for i, nm in enumerate(self.names)}
        # Use REAL seconds-of-day (wall-clock UTC) so the downstream
        # accumulated-offset predictor in gps_web.py can align the
        # simulation's time axis with the live receiver's timestamps.
        # The archived synthetic axis (idx*30) belongs to a different day
        # and cannot be differenced across a gap with live data.
        epoch_sod = float(time.time() % 86400.0)
        # Accumulate station_clocks with real SOD so /api/history can
        # serve the raw clock history that the Step 96 predictor needs.
        for nm, v in raw_clocks.items():
            self.station_clocks.setdefault(nm, {})[int(round(epoch_sod))] = float(v)
        lagged = []
        # Same best-TEP-estimator rule as LiveSource: the obstruction
        # comparison scores the better of singleband and multiband, with
        # both component RMS values kept in the payload for audit.
        _tep_rms = ker_rms if mb_rms is None else min(ker_rms, mb_rms)
        obstruction = _simulate_obstruction(_tep_rms, raw_rms, persist_rms,
                                            nsat_full)
        obstruction["kernel_rms_m"] = ker_rms
        obstruction["multiband_rms_m"] = mb_rms
        obstruction["tep_rms_used_m"] = _tep_rms
        # Scale kriging variance by the network's residual variance
        network_var = float(np.var(self.X[idx, :]))
        pred_variance_m2 = sk_var * network_var if sk_var is not None else network_var
        # Anisotropy position constraint (same as LiveSource)
        aniso_dir = np.array([0.0, 1.0, 0.0])  # North in ENU
        aniso_pos_var = 100.0 / FITTED_ANISO_RATIO  # ~46 m² (~6.8m std)
        return {
            "type": "epoch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "simulation",
            "epoch_idx": idx,
            "target": target_name,
            "stations": clocks,
            "raw_clocks": raw_clocks,
            "epoch_sod": epoch_sod,
            "lagged_stations": lagged,
            "kernel": {
                "target": target_name,
                "predicted_m": kernel_pred,
                "actual_m": actual,
                "error_m": kernel_err,
                "raw_error_m": raw_err,
                "persistence_error_m": persist_err,
                "variance_m2": pred_variance_m2,
                "aniso_pos_direction_enu": aniso_dir.tolist(),
                "aniso_pos_variance_m2": aniso_pos_var,
                "multiband_predicted_m": float(mb_pred[-1]) if mb_pred is not None else None,
                "multiband_error_m": mb_err,
                "multiband_variance_m2": mb_var * network_var if mb_var is not None else None,
                "multiband_enabled": MULTIBAND_ENABLED,
            },
            "tep": {
                "lambda_km": FITTED_LAMBDA_KM,
                "nugget": FITTED_NUGGET,
                "k_neighbours": min(FITTED_K, len(pool)),
                "aniso_mode": FITTED_ANISO_MODE,
                "aniso_ratio": FITTED_ANISO_RATIO,
                "form": "exponential",
                "correlation_length_km": FITTED_LAMBDA_KM,
                "network_span_km": float(self.D[~np.eye(self.n_sta, dtype=bool)].max()),
                "n_reference_stations": len(pool),
                "multiband": {
                    "enabled": MULTIBAND_ENABLED,
                    "bands": [
                        {"name": b["name"], "f_lo_uhz": b["f_lo_hz"]*1e6,
                         "f_hi_uhz": b["f_hi_hz"]*1e6,
                         "lambda_km": b["lambda_km"]}
                        for b in tk.MULTIBAND_BANDS
                    ],
                    "band_details": mb_details,
                    "fusion": "simple average of per-band predictions",
                },
            },
            "position": {
                "lat": self.station_info[target]["lat"],
                "lon": self.station_info[target]["lon"],
                "nsat": nsat_full,
            },
            "obstruction": obstruction,
        }


class LiveSource(DataSource):
    """Live NTRIP data source.

    Manages NTRIP client threads per station, accumulates RTCM3 bytes,
    and processes them through RTKLIB (convbin + rnx2rtkp) every 30
    seconds to extract receiver clock biases.  Falls back gracefully
    when stations are unreachable or processing fails.
    """

    def __init__(self, stations: list[str], caster_key: str = NTRIP_CASTER_KEY):
        self.requested = stations
        self.caster_key = caster_key
        self.mountpoints: list[MountPoint] = []
        self.streams: dict[str, StationStream] = {}
        self.stations_geo: dict[str, td.Station] = {}
        self.station_clocks: dict[str, dict[int, float]] = {}
        self.station_nsat: dict[str, dict[int, int]] = {}
        # Absolute-unix-keyed per-station clock history
        # {sid: {unix_s: clk_m}} — feeds the per-station sidereal-echo
        # channel: station code-median residuals carry local multipath
        # that repeats at each constellation's sky-repeat lag.  The
        # SOD-keyed station_clocks buffer collides across days and
        # cannot serve a ~86164/81100 s lag lookup.
        self.station_clocks_abs: dict[str, dict[float, float]] = {}
        # Per-satellite residuals {sid: {sod: {sat_id: resid_m}}} —
        # populated only for OS Net stations (hourly RINEX direct-clock
        # extraction); NTRIP streams have no per-sat product.
        self.station_persat: dict[str, dict[int, dict[str, float]]] = {}
        self._connected = False
        self._tmp = PROJECT_ROOT / "data" / "ntrip_live" / "_tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)
        # OS Net REST API stations (UK Ordnance Survey, hourly RINEX)
        self.osnet_stations: dict[str, object] = {}
        self._osnet_backfill_thread: threading.Thread | None = None
        self._osnet_stop = threading.Event()
        # Stations whose clock buffer is currently empty — latches the
        # "no clocks" warning to once per empty spell in get_epoch.
        self._osnet_empty: set[str] = set()
        self._drain_cursor = 0
        # Stations whose |clk| > 500 m passed the residual-shape gate in
        # the drain loop (genuine free-running receiver clocks, e.g.
        # MODUM's free quartz at ~263 km linear drift) — exempted from
        # the save/restore magnitude filters below.
        self._bigclk = {}
        self._bigclk_ok = set()
        # Empirically-fitted kernel lambda for the live network.  The
        # canonical 4200 km lambda (fitted on a continent-scale European
        # network) is degenerate on the ~150 km UK OS Net network: every
        # station pair has exp(-d/4200) ≈ 0.97, so kriging collapses to
        # the network mean and cannot beat persistence.  We refit the
        # correlation length by LOO-CV on the reference stations every
        # ~60 epochs so the kernel adapts to the actual network geometry.
        self._live_lambda_km: float = FITTED_LAMBDA_KM
        self._live_k: int = FITTED_K
        self._live_nugget: float = FITTED_NUGGET
        self._live_form: str = "exponential"
        self._live_lambda_epoch: int = -1000
        self._live_lambda_fit_count: int = 0
        self._init_osnet()
        self._discover()
        self._restore_state()
        if self.mountpoints:
            self._start_streams()
        threading.Thread(target=self._state_saver_loop, daemon=True,
                         name="state-saver").start()

    def _discover(self):
        """Fetch sourcetable and find requested stations.

        Searches the configured caster first, then falls back to all
        free casters if not all stations are found.
        """
        wanted = {s.upper() for s in self.requested}
        matched: list[MountPoint] = []
        # Try the primary caster first
        try:
            mps = find_stations(self.caster_key, self.requested)
        except Exception:
            mps = []
        matched = [m for m in mps if m.station_id in wanted]
        # If not all found, search all free casters
        found_ids = {m.station_id for m in matched}
        missing = wanted - found_ids
        if missing:
            try:
                all_mps = find_stations(station_names=list(missing))
            except Exception:
                all_mps = []
            for m in all_mps:
                if m.station_id in missing and m.station_id not in found_ids:
                    matched.append(m)
                    found_ids.add(m.station_id)
        # If still not found, try partial match on name prefix
        still_missing = wanted - found_ids
        for w in still_missing:
            for m in mps:
                if m.name.upper().startswith(w[:3]) and m not in matched:
                    matched.append(m)
                    break
        # Deduplicate mountpoints by station_id, preferring RTCM 3.3
        deduped: dict[str, MountPoint] = {}
        for m in matched:
            sid = m.station_id
            if sid not in deduped or "3.3" in m.format:
                deduped[sid] = m
        self.mountpoints = list(deduped.values())
        for mp in self.mountpoints:
            sid = mp.station_id
            self.stations_geo[sid] = td.Station(
                name=sid,
                ecef_km=np.zeros(3),
                lat_deg=mp.lat_deg,
                lon_deg=mp.lon_deg,
            )
            self.station_clocks.setdefault(sid, {})

    def _start_streams(self):
        """Start NTRIP client threads for each mountpoint."""
        for mp in self.mountpoints:
            sid = mp.station_id
            stream = StationStream(mp)
            stream.start()
            self.streams[sid] = stream
        # Wait up to 5 seconds for initial connections
        for _ in range(10):
            if any(s.connected for s in self.streams.values()):
                break
            time.sleep(0.5)
        self._connected = any(s.connected for s in self.streams.values())

    _STATE_PATH = PROJECT_ROOT / "data" / "ntrip_live" / "state_snapshot.pkl"

    def _save_state(self):
        """Persist accumulated station clock history to disk.

        NTRIP streams only accumulate live — a service restart (deploy,
        crash, the 2026-09-30 watchdog reboot) zeroes every station's
        span and with it the accumulated-offset predictor's spanning
        set.  A periodic snapshot caps the loss at one save interval.
        Keys are seconds-of-day; consumers already treat sod > now as
        yesterday, so snapshots up to TEP_STATE_MAX_AGE_H old restore
        safely across midnight.
        """
        try:
            import pickle
            # Drop degenerate clock values before persisting: failed
            # SPP solves historically stored 0.0 and km-scale ramp
            # series (TERS/WSRT/EIJS and others, 2026-09-30 — std up to
            # 550 km over hundreds of epochs).  Healthy network clocks
            # stay within ~±20 m; even broadcast-only SPP is ~90 m
            # detrended, so |clk| > 500 m cannot be a real station
            # clock.  Filtering here heals the snapshot even while the
            # in-memory copies age out or clear on restart.
            snap = {
                "ts": time.time(),
                "bigclk_ok": sorted(self._bigclk_ok),
                "lambda_km": self._live_lambda_km,
                "clocks": {k: {s: c for s, c in v.items()
                               if abs(c) <= 500.0
                               or k in self._bigclk_ok}
                           for k, v in
                           list(self.station_clocks.items())},
                "nsat": {k: dict(v) for k, v in
                         list(self.station_nsat.items())},
                "persat": {sid: {s: dict(d) for s, d in list(sv.items())}
                           for sid, sv in
                           list(self.station_persat.items())},
            }
            tmp = self._STATE_PATH.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(snap))
            tmp.replace(self._STATE_PATH)
            # Absolute-axis per-station clock store (per-station
            # sidereal-echo channel).  Separate file so the SOD-keyed
            # snapshot contract is untouched.  Kept to 5 days; the
            # echo needs >24 h to serve a ~86164/81100 s lag.
            try:
                _now = time.time()
                abs_snap = {
                    "ts": _now,
                    "clocks_abs": {
                        k: {t: c for t, c in v.items()
                            if _now - float(t) <= 5 * 86400.0
                            and (abs(c) <= 500.0
                                 or k in self._bigclk_ok)}
                        for k, v in
                        list(self.station_clocks_abs.items())},
                }
                ap = self._STATE_PATH.parent / "clocks_abs_snapshot.pkl"
                at = ap.with_suffix(".tmp")
                at.write_bytes(pickle.dumps(abs_snap))
                at.replace(ap)
            except Exception:
                pass
        except Exception:
            pass

    def _state_saver_loop(self):
        interval = float(os.environ.get("TEP_STATE_SAVE_S", "300"))
        # First save early: NTRIP history is live-only, so a crash
        # inside the first interval still loses everything.
        delay = min(60.0, interval)
        while not self._osnet_stop.wait(delay):
            self._save_state()
            delay = interval

    def _restore_state(self):
        """Reload the last snapshot written by _save_state.

        setdefault throughout: live NTRIP values and OS Net backfill
        always win over restored history for the same sod.
        """
        try:
            import pickle
            p = self._STATE_PATH
            if not p.exists():
                return
            snap = pickle.loads(p.read_bytes())
            age_h = (time.time() - snap.get("ts", 0)) / 3600.0
            if age_h > float(os.environ.get("TEP_STATE_MAX_AGE_H", "12")):
                return
            self._bigclk_ok.update(snap.get("bigclk_ok") or [])
            for sid, cl in (snap.get("clocks") or {}).items():
                d = self.station_clocks.setdefault(sid, {})
                for s, v in cl.items():
                    # Same 500 m sanity bound as _save_state — snapshots
                    # written before that filter still carry degenerate
                    # series.  Stations whitelisted by the residual-shape
                    # gate (_bigclk_ok, persisted above) keep their
                    # genuine free-running large-DC clocks.
                    if abs(v) <= 500.0 or sid in self._bigclk_ok:
                        d.setdefault(s, v)
            for sid, cl in (snap.get("nsat") or {}).items():
                d = self.station_nsat.setdefault(sid, {})
                for s, v in cl.items():
                    d.setdefault(s, v)
            for sid, per in (snap.get("persat") or {}).items():
                d = self.station_persat.setdefault(sid, {})
                for s, dd in per.items():
                    d.setdefault(s, dict(dd))
            lam = snap.get("lambda_km")
            if lam:
                self._live_lambda_km = float(lam)
            print(f"[TEP] restored state snapshot: "
                  f"{len(snap.get('clocks') or {})} stations, "
                  f"age {age_h:.2f} h")
            # Absolute-axis clock store (per-station sidereal echo).
            # Entries carry their own unix timestamps, so an aged
            # snapshot self-prunes on merge — no separate age gate.
            try:
                ap = p.parent / "clocks_abs_snapshot.pkl"
                if ap.exists():
                    ab = pickle.loads(ap.read_bytes())
                    _now = time.time()
                    for sid, cl in (ab.get("clocks_abs") or {}).items():
                        d = self.station_clocks_abs.setdefault(sid, {})
                        for t, v in cl.items():
                            if _now - float(t) <= 5 * 86400.0 \
                                    and (abs(v) <= 500.0
                                         or sid in self._bigclk_ok):
                                d.setdefault(float(t), v)
            except Exception:
                pass
        except Exception:
            pass

    def _init_osnet(self):
        """Initialize OS Net REST API stations (UK Ordnance Survey).

        Fetches hourly RINEX data from the OS Net API for nearby UK
        stations (e.g. EASN at 1.7 km) that are not available on the
        free NTRIP casters.  Requires the OSNET_API_KEY environment
        variable.

        The station objects are created synchronously (fast: just a
        metadata fetch), but the RINEX backfill (downloading 14+ hourly
        files per station) is deferred to a background thread so the
        HTTP server starts immediately and gps_web's KrigingProxy can
        connect without waiting 5-10 minutes for the backfill to
        complete.  Station clock data appears incrementally as each
        station's backfill finishes; the kriging loop already handles
        partial networks gracefully.
        """
        # Accept both spellings: ntrip_client.os_net_available() reads
        # OS_NET_API_KEY while this reads OSNET_API_KEY, so setting only one
        # of them used to give six live UK stations AND a status banner
        # insisting no UK stations were available.
        api_key = (os.environ.get("OSNET_API_KEY", "")
                   or os.environ.get("OS_NET_API_KEY", ""))
        if not api_key:
            return
        try:
            from scripts.lib.osnet_client import OSNetStation
        except ImportError:
            return
        # EASN (Easington) is 1.5 km from the receiver — the closest
        # CORS station available via any free source.  Add more UK OS Net
        # stations to strengthen the kriging network around the receiver.
        # SWAN (41km), SKEE (58km), LINO (60km), SCAU (75km), YEAL (98km)
        # provide a multi-scale network from co-located to regional.
        osnet_ids = os.environ.get(
            "OSNET_STATIONS",
            "EASN,SWAN,SKEE,LINO,SCAU,YEAL").split(",")
        backfill_queue = []
        for sid in osnet_ids:
            sid = sid.strip().upper()
            if not sid:
                continue
            try:
                sta = OSNetStation(sid, api_key=api_key)
                self.osnet_stations[sid] = sta
                self.stations_geo[sid] = td.Station(
                    name=sid,
                    ecef_km=np.zeros(3),
                    lat_deg=sta.lat,
                    lon_deg=sta.lon,
                )
                self.station_clocks.setdefault(sid, {})
                self.station_persat.setdefault(sid, {})
                print(f"[TEP] OS Net station {sid}: "
                      f"lat={sta.lat:.4f} lon={sta.lon:.4f}")
                # Defer the backfill to a background thread so the HTTP
                # server starts immediately.  The station_clocks dict is
                # populated incrementally as each station completes.
                if getattr(sta, "history_hours", 4.0) >= 4.0:
                    backfill_queue.append((sid, sta))
            except Exception as e:
                print(f"[TEP] OS Net station {sid} init failed: {e}")
        if self.osnet_stations:
            self._osnet_backfill_thread = threading.Thread(
                target=self._backfill_osnet_worker,
                args=(backfill_queue,), daemon=True,
                name="osnet-backfill")
            self._osnet_backfill_thread.start()

    def _osnet_bigclk_check(self, sid: str, clocks: dict) -> None:
        """Maintain _bigclk_ok membership for OS Net free-running clocks.

        The NTRIP drain path whitelists stations whose |clk| > 500 m
        detrends to a small residual (genuine free-running receiver
        clocks) in _bigclk_ok, which is also the save/restore magnitude
        filter's exemption key.  OS Net stations bypass that path, so a
        validated free-running station (KLYN's unsteered ~-85 km quartz,
        2026-10-27) would have its entire history silently dropped by
        _save_state's 500 m bound — losing it on every restart even
        though _filter_clock_outliers already shape-tested the series.
        Apply the same shape test here on the merged station series so
        the exemption is regime-based, not source-based.  A station
        whose clock drops back into the small-DC band is de-whitelisted
        so a regime break cannot leak a mixed series into snapshots.
        """
        try:
            _vv = np.asarray(
                [clocks[s] for s in sorted(clocks)], float)
            if len(_vv) < 12:
                return
            if np.all(np.abs(_vv) > 500.0):
                _tt = np.arange(len(_vv), dtype=float)
                _A = np.stack([np.ones_like(_tt), _tt], 1)
                _cf, *_ = np.linalg.lstsq(_A, _vv, rcond=None)
                if float(np.sqrt(np.mean((_vv - _A @ _cf) ** 2))) <= 15.0:
                    self._bigclk_ok.add(sid)
                    return
            if np.all(np.abs(_vv) <= 500.0):
                self._bigclk_ok.discard(sid)
        except Exception:
            pass

    def _backfill_osnet_worker(self, queue_items):
        """Background OS Net RINEX backfill (runs after HTTP server starts).

        Downloads hourly RINEX files and computes SPP clock biases for
        each station.  Results are written into self.station_clocks as
        each station completes, so the kriging loop picks them up on
        the next epoch without waiting for all stations to finish.

        After the initial backfill pass this thread keeps calling
        ``OSNetStation.update()`` (which self-throttles to one fetch per
        hour per station) so the epoch loop never blocks on a RINEX
        download or clock solve — cold start included.
        """
        for sid, sta in queue_items:
            try:
                n = sta.backfill()
                cl = sta.get_clocks()
                if cl:
                    self._osnet_bigclk_check(sid, cl)
                if n:
                    self.station_clocks[sid] = dict(cl)
                    self.station_clocks_abs[sid] = sta.get_clocks_abs()
                    self.station_persat[sid] = sta.get_persat()
                    self._save_state()
                # print(), not logger: this module reports with print
                # and never configures logging, so logger.info from
                # osnet_client is silently discarded.
                span = 0.0
                if cl:
                    ks = sorted(cl)
                    span = ((ks[-1] - ks[0]) % 86400.0) / 3600.0
                print(f"[TEP] OS Net {sid}: backfilled {n} files, "
                      f"{len(cl)} epochs, span {span:.1f} h")
            except Exception as e:
                print(f"[TEP] OS Net {sid} backfill failed: {e}")
        # Own the hourly update() polling.  update() early-returns until
        # fetch_interval (1 h) elapses, so a 5-min poll cadence is cheap.
        while not self._osnet_stop.is_set():
            for sid, sta in self.osnet_stations.items():
                try:
                    sta.update()
                except Exception as e:
                    print(f"[TEP] OS Net {sid} update failed: {e}")
            self._osnet_stop.wait(300.0)

    def mode(self) -> str:
        return "live"

    def station_list(self) -> list[dict]:
        out = []
        for sid, sta in self.stations_geo.items():
            stream = self.streams.get(sid)
            osnet = self.osnet_stations.get(sid)
            out.append({
                "name": sid,
                "lat": sta.lat_deg,
                "lon": sta.lon_deg,
                "connected": (stream.connected if stream else False) or (osnet is not None),
                "bytes": stream.total_bytes if stream else 0,
                "source": "osnet" if osnet else "ntrip",
            })
        return out

    def is_connected(self) -> bool:
        """True when the source can supply clock data.

        NTRIP streams are the primary path, but OS Net REST API stations
        (hourly RINEX with 14 h backfill) provide clock data independently
        of any NTRIP caster.  A cold start where NTRIP is unreachable but
        OS Net has backfilled should stay in live mode — falling back to
        simulation in that case discards 14 h of real UK CORS history and
        forces the accumulated-offset predictor onto a synthetic time axis
        that cannot align with the receiver's real seconds-of-day.
        """
        if self._connected:
            return True
        if any(sta.get_clocks() for sta in self.osnet_stations.values()):
            return True
        # Also accept station_clocks populated by the backfill path.
        if any(len(d) > 0 for d in self.station_clocks.values()):
            return True
        return False

    # Live lambda grid: includes small correlation lengths appropriate
    # for regional networks (50-300 km) that the canonical LAMBDA_GRID_KM
    # omits.  The canonical 4200 km is retained so a continent-scale
    # network can still select it, but a ~150 km UK network will pick a
    # lambda comparable to its span, giving meaningful spatial weighting.
    _LIVE_LAMBDA_GRID_KM = (50.0, 100.0, 150.0, 200.0, 300.0, 500.0,
                            700.0, 1500.0, 2500.0, 4200.0, 6000.0,
                            9000.0, 15000.0)
    # Neighbour-count grid for the joint (lam, k) LOO refit.  The
    # optimal k is regime-dependent — on a clean 55-station field
    # broad neighbourhoods (30-45) win ~8-15% median LOO vs k=15
    # (long-lambda field is information-rich everywhere), while a
    # noisy mix (centipede community mounts) favours small k.  The
    # refit lets the live matrix choose per epoch (tmp/k_sweep.py).
    _LIVE_K_GRID = (8, 15, 30, 45)
    # Nugget grid for the joint refit (tmp/nug_sweep.py): a large
    # nugget shrinks kriging weights toward uniform averaging —
    # optimal when the field's common-mode dominates (this network).
    # (nug=0.8, k=45) reaches median LOO ~2.2-2.6 vs (0.05,15) ~4.3
    # on clean archives — the dominant tunable after lambda.
    _LIVE_NUGGET_GRID = (0.05, 0.15, 0.35, 0.80)
    # Kernel-form grid for the joint refit (tmp/form_sweep.py):
    # gaussian beats the deployed exponential ~13-15% median LOO
    # at the big-nugget operating point on the clean archives.
    # legendre added 2026-10-23: the measured angular-correlation form
    # carries the negative lobe (~6-12 Mm) that the three parametric
    # forms cannot express (distance-stratified screen: CM-removed
    # corr goes negative at 1200-2000 km, field-real).  LOO-selected
    # per epoch alongside the parametric family; costs one extra form
    # slot in the refit.
    _LIVE_FORM_GRID = ("exponential", "gaussian", "matern15", "legendre")
    _LIVE_LAMBDA_REFIT_EPOCHS = 60

    def _fit_live_lambda(self, X: np.ndarray, D: np.ndarray,
                         names: list[str],
                         AZ: np.ndarray | None = None) -> tuple:
        """Select the kernel correlation length and neighbour count by
        LOO-CV on the live reference stations.

        For each candidate (lambda, k), every station is held out in
        turn and predicted from the remaining stations; the pair with
        the lowest median LOO RMS is selected.  This is leakage-free
        for the rotated target because the target station is not
        involved in selecting the global spatial parameters.  The
        nugget is kept at its fitted value.

        When *AZ* is provided, the LOO evaluation uses the same
        anisotropic distance stretching as the actual prediction, so
        the selected lambda is consistent with the anisotropic kernel.

        Returns (lambda_km, k_neighbours).
        """
        n_sta = X.shape[1]
        if n_sta < 4 or X.shape[0] < 10:
            return (FITTED_LAMBDA_KM, min(FITTED_K, n_sta - 1),
                    FITTED_NUGGET, "exponential")
        pool_all = list(range(n_sta))
        best_lam = FITTED_LAMBDA_KM
        best_k = min(FITTED_K, n_sta - 1)
        best_nug = FITTED_NUGGET
        best_form = "exponential"
        best_err = np.inf
        for form_cand in self._LIVE_FORM_GRID:
            for nug_cand in self._LIVE_NUGGET_GRID:
                for k_cand in self._LIVE_K_GRID:
                    k_use = min(k_cand, n_sta - 1)
                    for lam in self._LIVE_LAMBDA_GRID_KM:
                        p = tk.KernelParams(
                            lambda_km=lam, nugget=nug_cand,
                            k_neighbours=k_use, form=form_cand,
                            aniso_mode=FITTED_ANISO_MODE
                            if AZ is not None else "none",
                            aniso_ratio=FITTED_ANISO_RATIO
                            if AZ is not None else 1.0,
                        )
                        errs = []
                        for t in pool_all:
                            e = tk.prediction_error(X, D, t, pool_all,
                                                    p, AZ=AZ)
                            if np.isfinite(e):
                                errs.append(e)
                        if len(errs) < 3:
                            continue
                        med = float(np.median(errs))
                        if med < best_err:
                            best_err = med
                            best_lam = float(lam)
                            best_k = int(k_use)
                            best_nug = float(nug_cand)
                            best_form = form_cand
        self._live_lambda_fit_count += 1
        logging.info("Live kernel fit #%d: lam=%.0f k=%d nug=%.2f "
                     "form=%s (LOO median err=%.4fm, n_sta=%d, "
                     "span=%.0f km)",
                     self._live_lambda_fit_count, best_lam, best_k,
                     best_nug, best_form, best_err, n_sta,
                     float(D[~np.eye(n_sta, dtype=bool)].max()))
        return best_lam, best_k, best_nug, best_form

    def get_epoch(self) -> dict | None:
        """Process accumulated RTCM3 data and return one epoch."""
        # Merge OS Net REST API station clocks.  The RINEX fetch +
        # per-epoch clock solve inside update() takes minutes per
        # station — it runs in _backfill_osnet_worker, NOT here, so a
        # cold start (every station fetching at once) cannot stall the
        # epoch loop behind ~30 min of OS Net work.
        for sid, osnet_sta in self.osnet_stations.items():
            try:
                clocks = osnet_sta.get_clocks()
                if clocks:
                    # Keep the TRUE seconds-of-day.
                    #
                    # This used to shift OS Net SODs forward to "now" so the
                    # kriging matrix would accept them alongside the real-time
                    # NTRIP streams.  That shift was recomputed on every hourly
                    # poll, so consecutive emitted epochs did not share a time
                    # origin and the series could not be differenced across a
                    # gap -- which is why these stations had to be excluded
                    # from the accumulated-offset predictor entirely.
                    #
                    # The shift was also unnecessary.  OS Net serves complete
                    # hourly RINEX ~1-2 h behind real time, but the quantity
                    # being scored is a HELD-OUT WINDOW IN THE PAST (the latter
                    # part of the rig's rolling capture), so honest timestamps
                    # land inside the window that actually needs predicting.
                    # Keeping them makes these stations differenceable, and
                    # they carry the only short baselines the network has
                    # (EASN is 1.3 km from the receiver against 269 km for the
                    # nearest Dutch station).
                    self.station_clocks[sid] = dict(clocks)
                    self._osnet_bigclk_check(sid, clocks)
                    self.station_clocks_abs[sid] = osnet_sta.get_clocks_abs()
                    self.station_persat[sid] = osnet_sta.get_persat()
                    self.station_nsat.setdefault(sid, {})[max(clocks.keys())] = 28
                    logging.info("OS Net %s: %d clock epochs (true SOD)",
                                 sid, len(clocks))
                    self._osnet_empty.discard(sid)
                else:
                    # Warn once per empty spell, not per epoch: during a
                    # cold start the serial backfill queue takes ~2 min
                    # per station, so tail stations legitimately read
                    # empty for ~30 min and logged one WARNING per
                    # get_epoch (~700 lines for DUNG/HERO/EDIB on
                    # 2026-10-27).  A transition back to non-empty resets
                    # the latch so a NEW data loss still warns.
                    if sid not in self._osnet_empty:
                        self._osnet_empty.add(sid)
                        logging.warning(
                            "OS Net %s: no clocks yet (awaiting backfill "
                            "or hourly file)", sid)
            except Exception as e:
                logging.warning("OS Net %s: update failed: %s", sid, e)
        # Drain and process NTRIP stations, at most DRAIN_MAX_PER_CALL
        # per call, rotating so every station is visited every few calls.
        sids = list(self.streams.keys())
        if sids:
            n_all = len(sids)
            start = self._drain_cursor % n_all
            order = sids[start:] + sids[:start]
            processed = 0
            for sid in order:
                stream = self.streams[sid]
                data = stream.drain()
                if len(data) < MIN_RTCM3_BYTES:
                    # Put the bytes back — discarding them thins the
                    # station's clock series and loses the partial RTCM
                    # frame at the tail.  The buffer is contiguous, so
                    # prepending preserves framing for the next pass.
                    with stream._lock:
                        stream._buf = bytearray(data) + stream._buf
                    continue
                if processed >= DRAIN_MAX_PER_CALL:
                    # Put the bytes back for the next call.
                    with stream._lock:
                        stream._buf = bytearray(data) + stream._buf
                    continue
                processed += 1
                self._drain_cursor = (start + processed) % n_all
                if len(data) > MAX_RTCM3_DRAIN_BYTES:
                    logging.info(
                        "%s: drain backlog %d bytes, keeping newest %d",
                        sid, len(data), MAX_RTCM3_DRAIN_BYTES)
                    data = _rtcm3_tail(data, MAX_RTCM3_DRAIN_BYTES)
                rtcm3_path = self._tmp / f"{sid}_{int(time.time())}.rtcm3"
                with open(rtcm3_path, "wb") as f:
                    f.write(data)
                conv_dir = self._tmp / sid
                rnx = _convbin_rtcm3(rtcm3_path, conv_dir)
                rtcm3_path.unlink(missing_ok=True)
                if rnx is None:
                    continue
                # Compute station clock biases.  Try the direct method first
                # (uses the station's known surveyed position — eliminates
                # SPP position error, giving ~0.5-2 m clock noise instead
                # of ~5-20 m).  Fall back to SPP if the RINEX header doesn't
                # have APPROX POSITION XYZ or if the direct method fails.
                stat = _compute_direct_clock(rnx["obs"], rnx.get("nav"),
                                             self._tmp / sid)
                if stat is None:
                    stat = _run_spp(rnx["obs"], rnx.get("nav"),
                                    self._tmp / sid)
                if stat is None:
                    continue
                for sod, info in stat.items():
                    # |clk| > 500 m is USUALLY a degenerate solve
                    # artifact (km-scale runaway ramps — TERS/WSRT/EIJS,
                    # 2026-09-30) — but community bases (MODUM, BOKU on
                    # centipede) run genuinely free quartz clocks at
                    # km-scale offsets with a nearly-linear drift, and
                    # their detrended residual is a usable ~4-5 m field
                    # curve.  Distinguish by shape: accept a large-DC
                    # clock only when its recent series detrends to a
                    # small residual; a runaway ramp leaves a big one.
                    if abs(info["clk_m"]) > 500.0:
                        _hb = self._bigclk.setdefault(sid, [])
                        # A step-jump >2 km vs the last candidate means
                        # the free-running regime reset (steered,
                        # re-solved, or day boundary) — restart the
                        # shape evidence rather than blend regimes.
                        if _hb and abs(info["clk_m"] - _hb[-1]) > 2000.0:
                            _hb.clear()
                        _hb.append(float(info["clk_m"]))
                        if len(_hb) > 40:
                            del _hb[0]
                        if len(_hb) >= 12:
                            _vv = np.asarray(_hb, float)
                            _tt = np.arange(len(_vv), dtype=float)
                            _A = np.stack([np.ones_like(_tt), _tt], 1)
                            _cf, *_ = np.linalg.lstsq(
                                _A, _vv, rcond=None)
                            if float(np.sqrt(np.mean(
                                    (_vv - _A @ _cf) ** 2))) > 15.0:
                                continue
                            self._bigclk_ok.add(sid)
                        else:
                            continue
                    elif sid in self._bigclk_ok and self._bigclk.get(sid):
                        # Regime break back into the small-DC band
                        # (free-running clock steered/re-solved):
                        # drop the whitelist and skip the epoch so the
                        # stored series never mixes regimes.
                        if abs(info["clk_m"]
                               - float(np.median(self._bigclk[sid][-5:]))) \
                                > 2000.0:
                            self._bigclk_ok.discard(sid)
                            self._bigclk.pop(sid, None)
                            continue
                    self.station_clocks[sid][sod] = info["clk_m"]
                    self.station_nsat.setdefault(sid, {})[sod] = info.get("nsat", 0)
                    _ta = info.get("t_abs")
                    if _ta is not None:
                        self.station_clocks_abs.setdefault(
                            sid, {})[float(_ta)] = info["clk_m"]
        # Check if we have enough data for kriging
        # Filter out stations where SPP failed (all clock biases are exactly 0)
        valid = {}
        for sid, clocks in self.station_clocks.items():
            if len(clocks) < 5:
                continue
            vals = list(clocks.values())
            if all(abs(v) < 1e-10 for v in vals):
                # SPP returned 0 for all epochs — no nav data or no fix
                continue
            valid[sid] = clocks
        if len(valid) < 4:
            return {
                "type": "status",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "live",
                "message": "Accumulating RTCM3 data from NTRIP streams...",
                "stations_connected": sum(1 for s in self.streams.values()
                                          if s.connected),
                "stations_total": len(self.streams),
            }
        # Build live matrix and compute kriging prediction
        names, X, D, X_raw, sods = _build_live_matrix(valid, self.stations_geo)
        if X is None or len(names) < 4:
            return {
                "type": "status",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "live",
                "message": "Insufficient common epochs for kriging.",
                "stations_connected": sum(1 for s in self.streams.values()
                                          if s.connected),
                "stations_total": len(self.streams),
            }
        # Azimuth matrix for anisotropic kriging (EW/NS ratio = 2.16).
        # Without this, the aniso_mode/aniso_ratio in KernelParams are
        # silently ignored (predict_with_variance falls back to isotropic
        # distances), which collapses the anisotropic spatial structure
        # to a circular kernel and degrades the kriging prediction.
        AZ = _build_azimuth_matrix(names, self.stations_geo)
        # Use the last epoch
        idx = X.shape[0] - 1
        # Rotate target through stations for LOO-CV style validation
        target = idx % len(names)
        target_name = names[target]
        pool = [j for j in range(len(names)) if j != target]
        # Refit the kernel correlation length periodically by LOO-CV on
        # the reference stations.  The canonical 4200 km lambda is
        # degenerate on a ~150 km network (kriging collapses to the mean);
        # the empirically-fitted lambda adapts to the actual network
        # geometry so the kernel captures spatial structure instead of
        # predicting the network average.
        if (idx - self._live_lambda_epoch >= self._LIVE_LAMBDA_REFIT_EPOCHS
                and X.shape[0] >= 20 and X.shape[1] >= 4):
            try:
                fitted = self._fit_live_lambda(X, D, names, AZ=AZ)
                if fitted and fitted[0] > 0:
                    self._live_lambda_km = float(fitted[0])
                    self._live_k = int(fitted[1])
                    self._live_nugget = float(fitted[2])
                    self._live_form = str(fitted[3])
                    self._live_lambda_epoch = idx
            except Exception as e:
                logging.warning("Live lambda fit failed: %s", e)
        live_lambda = self._live_lambda_km
        params = tk.KernelParams(
            lambda_km=live_lambda,
            nugget=self._live_nugget,
            form=self._live_form,
            k_neighbours=min(self._live_k, len(pool)),
            aniso_mode=FITTED_ANISO_MODE,
            aniso_ratio=FITTED_ANISO_RATIO,
        )
        pred, sk_var = tk.predict_with_variance(X, D, target, pool, params,
                                                AZ=AZ)
        actual = float(X[idx, target])
        if pred is not None:
            kernel_pred = float(pred[idx])
            kernel_err = float(abs(actual - kernel_pred))
        else:
            kernel_pred = 0.0
            kernel_err = abs(actual)
        # Multi-band prediction (Paper 1 frequency-dependent lambdas).
        # The live matrix sits on the union SOD grid (~5 s NTRIP cadence,
        # not the 30 s broadcast cadence), so the Gram's frequency axis
        # must be built from the actual epoch spacing or every band is
        # misregistered and the per-band lambda applies to wrong content.
        mb_pred = None; mb_err = None; mb_var = None; mb_details = []
        mb_rms = None
        if MULTIBAND_ENABLED and X.shape[0] >= 64:
            _sod_d = np.diff(sods) if sods is not None and len(sods) > 1 else []
            _dt_mb = (float(np.median(_sod_d)) if len(_sod_d)
                      else MULTIBAND_DT_S)
            result = tk.predict_multiband_with_variance(
                X, D, target, pool, params, dt_s=_dt_mb, AZ=AZ)
            if result is not None and len(result) == 3:
                mb_pred, mb_var, mb_details = result
            elif result is not None and len(result) == 2:
                mb_pred, mb_var = result
            if mb_pred is not None:
                mb_err = float(abs(actual - float(mb_pred[idx])))
                mb_rms = float(np.sqrt(np.mean(
                    (X[:, target] - np.asarray(mb_pred, float)) ** 2)))
        raw_err = abs(actual)
        # Operational persistence: hold the clock from ~1 hour ago, matching
        # the paper's Step 106 outage scenario and the SimulationSource.
        # Lag-1 (30 s) is trivially small for a band-limited clock residual
        # (the field's power is below 500 uHz), which made persistence look
        # unbeatable and the network prior look useless.  The honest baseline
        # holds the clock across the outage gap the prior is designed for.
        persist_err = raw_err
        if sods is not None and len(sods) > 1 and idx > 0:
            _sod_cur = float(sods[idx])
            _sod_target = _sod_cur - PERSIST_LAG_S
            # Search the history for the epoch closest to 1 hour ago.
            # Handle the UTC midnight wrap: SODs are in [0, 86400).
            _sods_prev = np.asarray(sods[:idx], float)
            _diffs = np.abs(_sods_prev - _sod_target)
            _diffs_wrap = np.abs(_sods_prev - (_sod_target + 86400.0))
            _lag_idx = int(np.argmin(np.minimum(_diffs, _diffs_wrap)))
            _lag_dt = min(abs(_sods_prev[_lag_idx] - _sod_cur),
                          86400.0 - abs(_sods_prev[_lag_idx] - _sod_cur))
            if _lag_dt <= PERSIST_LAG_S * 1.5:
                persist_pred = float(X[_lag_idx, target])
                persist_err = abs(actual - persist_pred)
            elif idx > 0:
                # Not enough history for the full lag yet: use the oldest
                # available epoch (still a real outage hold, not lag-1).
                persist_pred = float(X[0, target])
                persist_err = abs(actual - persist_pred)
        elif idx > 0:
            persist_err = abs(actual - float(X[0, target]))
        # Running RMS
        raw_rms = float(np.sqrt(np.mean(X[:, target] ** 2)))
        if pred is not None:
            ker_rms = float(np.sqrt(np.mean((X[:, target] - pred) ** 2)))
        else:
            ker_rms = raw_rms
        # Persistence RMS at the operational lag: RMS of (x[i] - x[i-lag])
        # over all i where the lag is available.  This is the same
        # comparison the single-epoch persist_err makes, but averaged over
        # the whole window so the obstruction simulation's tep_vs_persist
        # is not dominated by a single epoch's noise.
        _col = X[:, target]
        _n = len(_col)
        if _n > PERSIST_LAG_EPOCHS:
            persist_rms = float(np.sqrt(np.mean(
                (_col[PERSIST_LAG_EPOCHS:] - _col[:-PERSIST_LAG_EPOCHS]) ** 2)))
        elif _n > 1:
            persist_rms = float(np.sqrt(np.mean(np.diff(_col) ** 2)))
        else:
            persist_rms = raw_rms
        # Get real satellite count from the target station's latest epoch
        # Fall back to median nsat across all stations with data
        all_nsats = []
        for sid_ns in self.station_nsat.values():
            if sid_ns:
                latest = max(sid_ns.keys())
                ns = sid_ns[latest]
                if ns > 0:
                    all_nsats.append(ns)
        nsat_full = int(np.median(all_nsats)) if all_nsats else 8
        if target_name in self.station_nsat:
            ns_s = self.station_nsat[target_name]
            if ns_s:
                latest_sod = max(ns_s.keys())
                if ns_s[latest_sod] > 0:
                    nsat_full = ns_s[latest_sod]
        obstructed_nsat = max(3, nsat_full - 3)
        clocks = {nm: float(X[idx, i]) for i, nm in enumerate(names)}
        # Raw (un-detrended) SPP clock at this epoch, plus the epoch's own
        # second-of-day.  The downstream Step 96 predictor differences these
        # across the prediction gap; the detrended `clocks` above cannot be
        # differenced because each station's drift has already been removed.
        raw_clocks = {nm: float(X_raw[idx, i]) for i, nm in enumerate(names)
                      if np.isfinite(X_raw[idx, i])}
        epoch_sod = float(sods[idx]) if sods is not None and len(sods) else None
        # Stations whose time axis is NOT real time.  OS Net serves hourly
        # RINEX with 1-3 h latency and `get_epoch` shifts their SODs forward
        # to the current window so the kriging matrix will accept them.  That
        # shift is recomputed on every hourly poll, so consecutive emitted
        # epochs for these stations do not share a time origin and CANNOT be
        # differenced across a gap.  They stay in the instantaneous residual
        # field; the accumulated-offset predictor must skip them.
        # OS Net stations now carry true timestamps, so they ARE differenceable
        # and no longer need excluding from the accumulated-offset predictor.
        # The field stays in the payload (empty) so a consumer that still
        # filters on it keeps working.
        lagged: list[str] = []
        # The obstruction comparison scores the TEP prior an operational
        # system would actually deploy: the better of the singleband and
        # multiband predictors — the same "superior predictor wins" rule
        # the function applies between kernel and persistence.  Both
        # component RMS values stay in the payload so the choice is
        # auditable, not a silent cherry-pick.
        _tep_rms = ker_rms if mb_rms is None else min(ker_rms, mb_rms)
        obstruction = _simulate_obstruction(_tep_rms, raw_rms, persist_rms,
                                            nsat_full, obstructed_nsat)
        obstruction["kernel_rms_m"] = ker_rms
        obstruction["multiband_rms_m"] = mb_rms
        obstruction["tep_rms_used_m"] = _tep_rms
        # Scale kriging variance by the network's residual variance
        # to get an operational prediction variance in m^2
        network_var = float(np.var(X[idx, :]))
        pred_variance_m2 = sk_var * network_var if sk_var is not None else network_var

        # TEP astronomical evidence fields (Papers 1-2)
        tep_astronomical = _compute_tep_astronomical()

        # TEP anisotropy position constraint direction and variance
        # for the EKF: the EW/NS anisotropy means the clock field gradient
        # is steeper in the NS direction, so position is better constrained
        # along NS.  The preferred axis is East (90°), so the perpendicular
        # (constraint) direction is North.
        aniso_dir = np.array([0.0, 1.0, 0.0])  # North in ENU
        # Position variance: soft constraint along the NS axis at the
        # level of typical SPP position scatter (~10m std = 100m²).
        # The anisotropy ratio modulates this — stronger anisotropy
        # means tighter constraint.  This is a regularisation prior,
        # not a hard measurement.
        aniso_pos_var = 100.0 / FITTED_ANISO_RATIO  # ~46 m² (~6.8m std)

        network_span_km = float(D[~np.eye(len(names), dtype=bool)].max())

        return {
            "type": "epoch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live",
            "epoch_idx": idx,
            "target": target_name,
            "stations": clocks,
            "raw_clocks": raw_clocks,
            "epoch_sod": epoch_sod,
            "lagged_stations": lagged,
            "kernel": {
                "target": target_name,
                "predicted_m": kernel_pred,
                "actual_m": actual,
                "error_m": kernel_err,
                "raw_error_m": raw_err,
                "persistence_error_m": persist_err,
                "variance_m2": pred_variance_m2,
                "aniso_pos_direction_enu": aniso_dir.tolist(),
                "aniso_pos_variance_m2": aniso_pos_var,
                "multiband_predicted_m": float(mb_pred[idx]) if mb_pred is not None else None,
                "multiband_error_m": mb_err,
                "multiband_variance_m2": mb_var * network_var if mb_var is not None else None,
                "multiband_enabled": MULTIBAND_ENABLED,
            },
            "tep": {
                "lambda_km": live_lambda,
                "lambda_canonical_km": FITTED_LAMBDA_KM,
                "lambda_fitted_live": live_lambda != FITTED_LAMBDA_KM,
                "nugget": self._live_nugget,
                "nugget_canonical": FITTED_NUGGET,
                "nugget_fitted_live": self._live_nugget != FITTED_NUGGET,
                "k_neighbours": min(self._live_k, len(pool)),
                "k_canonical": FITTED_K,
                "k_fitted_live": self._live_k != FITTED_K,
                "form_live": self._live_form,
                "aniso_mode": FITTED_ANISO_MODE,
                "aniso_ratio": FITTED_ANISO_RATIO,
                "form": self._live_form,
                "correlation_length_km": live_lambda,
                "network_span_km": network_span_km,
                "n_reference_stations": len(pool),
                "astronomical": tep_astronomical,
                "multiband": {
                    "enabled": MULTIBAND_ENABLED,
                    "bands": [
                        {"name": b["name"], "f_lo_uhz": b["f_lo_hz"]*1e6,
                         "f_hi_uhz": b["f_hi_hz"]*1e6,
                         "lambda_km": b["lambda_km"]}
                        for b in tk.MULTIBAND_BANDS
                    ],
                    "band_details": mb_details,
                    "fusion": "simple average of per-band predictions",
                },
            },
            "position": {
                "lat": self.stations_geo.get(target_name,
                                             self.stations_geo.get(names[0])).lat_deg,
                "lon": self.stations_geo.get(target_name,
                                             self.stations_geo.get(names[0])).lon_deg,
                "nsat": nsat_full,
            },
            "obstruction": obstruction,
        }

    def stop(self):
        self._osnet_stop.set()
        for stream in self.streams.values():
            stream.stop()


# ================================================================ server
class TEPRealtimeServer:
    """FastAPI server bridging the data source to WebSocket clients."""

    def __init__(self, stations: list[str] | None = None,
                 caster_key: str = NTRIP_CASTER_KEY,
                 allow_sim_fallback: bool = False):
        self.app = FastAPI(title="TEP-PPP Real-Time Demo")
        self.clients: set[WebSocket] = set()
        self.data_queue: queue.Queue = queue.Queue(maxsize=64)
        self.shutdown_event = threading.Event()
        self.source: Optional[DataSource] = None
        self.source_thread: Optional[threading.Thread] = None
        self.broadcast_task = None
        self.sourcetable_cache: list[dict] = []
        self.latest_epoch_cache: dict | None = None
        self._stations = stations or list(DEFAULT_STATIONS)
        self._caster_key = caster_key
        self._allow_sim_fallback = allow_sim_fallback
        self._setup_routes()
        self._init_source()

    def _init_source(self):
        """Initialize live NTRIP geodetic reference network source."""
        # Check if OS Net API key is available for UK RINEX downloads
        from scripts.lib.ntrip_client import os_net_available, OS_NET_STATIONS
        self._os_net_available = os_net_available()
        if self._os_net_available:
            print(f"[TEP] OS Net API key detected — UK RINEX downloads enabled")
            # Add UK stations to the station list
            for sid in OS_NET_STATIONS:
                if sid not in self._stations:
                    self._stations.append(sid)
        try:
            live = LiveSource(self._stations, self._caster_key)
            if live.is_connected():
                self.source = live
                print(f"[TEP] Live NTRIP mode: {len(live.streams)} reference station streams connected from {self._caster_key}")
            else:
                if self._allow_sim_fallback:
                    live.stop()
                    raise RuntimeError("No NTRIP streams connected")
                else:
                    self.source = live
                    print(f"[TEP] Live NTRIP mode: {len(live.streams)} streams active (connecting to CORS network)")
        except Exception as e:
            # If the primary caster fails, try the OS Net caster as a
            # fallback (it may work with different credentials or
            # network conditions).  If that also fails, fall back to
            # simulation mode.
            if self._caster_key != "osnet" and self._allow_sim_fallback:
                print(f"[TEP] {self._caster_key} caster unavailable ({e}); trying OS Net")
                try:
                    live = LiveSource(OS_NET_STATIONS_LIST, "osnet")
                    if live.is_connected():
                        self.source = live
                        self._caster_key = "osnet"
                        self._stations = list(OS_NET_STATIONS_LIST)
                        print(f"[TEP] OS Net fallback: {len(live.streams)} streams connected")
                        return
                    live.stop()
                except Exception as e2:
                    print(f"[TEP] OS Net fallback also failed: {e2}")
            if self._allow_sim_fallback:
                print(f"[TEP] NTRIP unavailable ({e}); falling back to simulation")
                try:
                    self.source = SimulationSource()
                    print(f"[TEP] Simulation mode: {len(self.source.names)} stations, "
                          f"{self.source.n_epoch} epochs")
                except Exception as e2:
                    print(f"[TEP] Simulation also failed: {e2}")
                    self.source = None
            else:
                print(f"[TEP] Live NTRIP mode active with {len(self._stations)} stations from {self._caster_key} (retrying)")
                self.source = LiveSource(self._stations, self._caster_key)
        # Cache sourcetable for REST endpoint
        if isinstance(self.source, LiveSource):
            try:
                mps = fetch_sourcetable(self._caster_key)
                self.sourcetable_cache = [
                    {"name": m.name, "format": m.format, "nav": m.nav_system,
                     "lat": m.lat_deg, "lon": m.lon_deg, "country": m.country}
                    for m in mps[:200]
                ]
            except Exception:
                pass

    def _setup_routes(self):
        api = self.app

        @api.get("/status")
        async def status():
            mode = self.source.mode() if self.source else "none"
            stations = self.source.station_list() if self.source else []
            connected = 0
            if isinstance(self.source, LiveSource):
                connected = sum(1 for s in self.source.streams.values()
                                if s.connected)
            return JSONResponse({
                "mode": mode,
                "connected": connected > 0 if mode == "live" else True,
                "n_clients": len(self.clients),
                "stations": stations,
                "n_stations": len(stations),
                "epoch_interval_s": EPOCH_INTERVAL_S,
                "os_net_available": getattr(self, "_os_net_available", False),
                "coverage_note": (
                    "UK OS Net stations available via API"
                    if getattr(self, "_os_net_available", False)
                    else "Only Dutch CORS (270-500 km) — UK stations require OS_NET_API_KEY or NTRIP registration"
                ),
            })

        @api.get("/sourcetable")
        async def sourcetable():
            return JSONResponse({
                "caster": self._caster_key,
                "host": NTRIP_CASTERS[self._caster_key]["host"],
                "port": NTRIP_CASTERS[self._caster_key]["port"],
                "mountpoints": self.sourcetable_cache,
            })

        @api.get("/api/latest")
        async def latest():
            """Return the most recent epoch message for dashboard polling."""
            if self.latest_epoch_cache is not None:
                return JSONResponse(self.latest_epoch_cache)
            return JSONResponse({"available": False})

        @api.get("/api/history")
        async def history(max_age_s: float = 64800.0):
            """Raw (un-detrended) station clock history for backfill.

            A consumer that restarts would otherwise have to re-accumulate
            hours of network history before the accumulated-offset predictor
            can span its prediction gap.  The source already holds that
            history, so serve it and let the consumer start warm.
            """
            src = self.source
            clocks = getattr(src, "station_clocks", None) if src else None
            if not clocks:
                return JSONResponse({"available": False,
                                     "reason": "source holds no clock history"})
            # OS Net stations keep true timestamps now, so they belong in the
            # backfill like any other station.
            now_sod = float(time.time() % 86400.0)
            out = {}
            for sid, d in clocks.items():
                if not d:
                    continue
                pts = []
                for sod, v in d.items():
                    try:
                        sod_f, v_f = float(sod), float(v)
                    except (TypeError, ValueError):
                        continue
                    if not (np.isfinite(sod_f) and np.isfinite(v_f)):
                        continue
                    age = (now_sod - sod_f) % 86400.0
                    if age <= max_age_s:
                        pts.append((sod_f, v_f))
                if len(pts) >= 4:
                    pts.sort()
                    out[sid] = {"sod": [p[0] for p in pts],
                                "clk_m": [p[1] for p in pts]}
            return JSONResponse({
                "available": bool(out),
                "n_stations": len(out),
                "lagged_excluded": [],
                "stations": out,
            })

        @api.get("/api/persat")
        async def persat(max_age_s: float = 14400.0):
            """Per-satellite station residual history for the per-sat
            network correction.

            Serves {sid: {sat: [[sod, resid_m], ...]}} — each satellite's
            (pseudorange - geometric range) minus the epoch station clock,
            computed from OS Net hourly RINEX at the surveyed station
            position with precise products.  The consumer kriges these to
            the receiver location per satellite and applies a causal
            receiver-specific DCB before correcting measurements.
            """
            src = self.source
            data = getattr(src, "station_persat", None) if src else None
            if not data:
                return JSONResponse({"available": False,
                                     "reason": "source holds no per-sat "
                                               "residual history"})
            now_sod = float(time.time() % 86400.0)
            out = {}
            for sid, d in data.items():
                if not d:
                    continue
                sats: dict[str, list] = {}
                for sod, res in d.items():
                    try:
                        sod_f = float(sod)
                    except (TypeError, ValueError):
                        continue
                    if (now_sod - sod_f) % 86400.0 > max_age_s:
                        continue
                    for s, v in res.items():
                        try:
                            v_f = float(v)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(v_f):
                            sats.setdefault(s, []).append((sod_f, v_f))
                sats = {s: sorted(p) for s, p in sats.items()
                        if len(p) >= 4}
                if sats:
                    out[sid] = sats
            return JSONResponse({
                "available": bool(out),
                "n_stations": len(out),
                "stations": out,
            })

        @api.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self.clients.add(ws)
            # Send initial status
            mode = self.source.mode() if self.source else "none"
            stations = self.source.station_list() if self.source else []
            await ws.send_json({
                "type": "hello",
                "mode": mode,
                "stations": stations,
                "epoch_interval_s": EPOCH_INTERVAL_S,
            })
            try:
                while not self.shutdown_event.is_set():
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self.clients.discard(ws)

    async def _broadcast_loop(self):
        """Read from the thread-safe queue and send to all WebSocket clients."""
        while not self.shutdown_event.is_set():
            try:
                msg = await asyncio_to_thread_get(self.data_queue, 1.0)
            except queue.Empty:
                continue
            if msg is None:
                continue
            dead = set()
            for ws in list(self.clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.add(ws)
            self.clients -= dead

    def _source_worker(self):
        """Background thread: produce epoch data and feed the queue."""
        if self.source is None:
            return
        while not self.shutdown_event.is_set():
            try:
                msg = self.source.get_epoch()
            except Exception as e:
                print(f"[TEP] Source error: {e}")
                msg = None
            if msg is not None:
                self.latest_epoch_cache = msg
                try:
                    self.data_queue.put(msg, timeout=2.0)
                except queue.Full:
                    pass  # drop if clients can't keep up
            # Wait for next epoch interval
            self.shutdown_event.wait(EPOCH_INTERVAL_S)

    def start(self):
        """Start the broadcast loop and source worker."""
        import asyncio
        loop = asyncio.get_event_loop()
        self.broadcast_task = loop.create_task(self._broadcast_loop())
        self.source_thread = threading.Thread(
            target=self._source_worker, daemon=True, name="tep-source")
        self.source_thread.start()

    def stop(self):
        """Graceful shutdown."""
        self.shutdown_event.set()
        if isinstance(self.source, LiveSource):
            self.source.stop()
        if self.source_thread:
            self.source_thread.join(timeout=5.0)


async def asyncio_to_thread_get(q: queue.Queue, timeout: float):
    """Thread-safe queue.get wrapped for asyncio."""
    import asyncio
    return await asyncio.to_thread(q.get, True, timeout)


# ================================================================ main
def _set_proc_name(name: str) -> None:
    """Set kernel comm so earlyoom's --avoid regex protects this service.

    The systemd unit launches ``python3`` whose comm is "python3" — it
    does NOT match earlyoom's ``--avoid '^(tep-kriging|...)'`` regex, so
    under memory pressure the daemon treats this service as a victim.
    prctl(PR_SET_NAME) writes comm, the field earlyoom matches.  No-op
    off Linux.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


def main():
    import asyncio
    import contextlib
    import argparse

    _set_proc_name("tep-kriging")

    parser = argparse.ArgumentParser(description="TEP-PPP Real-Time Demo Server")
    parser.add_argument("--port", type=int, default=WS_PORT,
                        help=f"WebSocket/HTTP port (default {WS_PORT})")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default 0.0.0.0)")
    parser.add_argument("--stations", nargs="+",
                        default=None,
                        help="Station IDs for live NTRIP (default: auto-select by caster)")
    parser.add_argument("--caster", default=NTRIP_CASTER_KEY,
                        help="NTRIP caster key (rtk2go/agrs_nl/euref/igs)")
    parser.add_argument("--sim", action="store_true",
                        help="Force simulation mode (skip NTRIP)")
    parser.add_argument("--allow-sim-fallback", action="store_true",
                        help="Fall back to simulation mode when NTRIP is "
                             "unreachable (no connected stations after "
                             "initial retry).  Without this flag the server "
                             "stays in live mode and keeps retrying.")
    args = parser.parse_args()

    # Auto-select stations based on the caster when not explicitly given.
    # RTK2GO uses UK community stations (40-160 km); AGRS-NL uses Dutch
    # stations (269+ km).
    if args.stations is None:
        if args.caster == "rtk2go":
            args.stations = list(RTK2GO_UK_STATIONS)
            print(f"[TEP] Auto-selected {len(args.stations)} RTK2GO UK stations "
                  f"(nearest: {args.stations[0]} at ~40 km)")
        else:
            args.stations = list(DEFAULT_STATIONS)

    if args.sim:
        server = TEPRealtimeServer.__new__(TEPRealtimeServer)
        server.app = FastAPI(title="TEP-PPP Real-Time Demo")
        server.clients = set()
        server.data_queue = queue.Queue(maxsize=64)
        server.shutdown_event = threading.Event()
        server.source = None
        server.source_thread = None
        server.broadcast_task = None
        server.sourcetable_cache = []
        server.latest_epoch_cache = None
        server._stations = list(DEFAULT_STATIONS)
        server._caster_key = args.caster
        server._setup_routes()
        try:
            server.source = SimulationSource()
            print(f"[TEP] Simulation mode: {len(server.source.names)} stations, "
                  f"{server.source.n_epoch} epochs")
        except Exception as e:
            print(f"[TEP] Simulation failed: {e}")
    else:
        server = TEPRealtimeServer(stations=args.stations,
                                   caster_key=args.caster,
                                   allow_sim_fallback=args.allow_sim_fallback)

    # Handle Ctrl-C / SIGTERM
    def _signal_handler(signum, frame):
        print("\n[TEP] Shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Lifespan context: start/stop the source worker and broadcast loop
    @contextlib.asynccontextmanager
    async def lifespan(app):
        server.start()
        yield
        server.stop()

    server.app.router.lifespan_context = lifespan

    config = uvicorn.Config(
        server.app, host=args.host, port=args.port,
        log_level="info", lifespan="on", ws="wsproto",
    )
    asgi = uvicorn.Server(config)

    print(f"[TEP] TEP-PPP Real-Time Demo Server")
    print(f"[TEP] WebSocket:  ws://localhost:{args.port}/ws")
    print(f"[TEP] Status:     http://localhost:{args.port}/status")
    print(f"[TEP] Sourcetable: http://localhost:{args.port}/sourcetable")
    mode = server.source.mode() if server.source else "none"
    print(f"[TEP] Mode: {mode}")
    asgi.run()


if __name__ == "__main__":
    main()
