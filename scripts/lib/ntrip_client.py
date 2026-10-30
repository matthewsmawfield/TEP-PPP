#!/usr/bin/env python3
"""
NTRIP Client for Live RTCM3 Stream Acquisition
==============================================

Connects to NTRIP casters and records raw RTCM3 streams from
reference-station mountpoints to timestamped binary files.
The recorded files are committed to the repository so that the live
experiment is fully reproducible: anyone can replay the exact same
byte stream through the deterministic pipeline.

Two modes
---------
1. **Record** -- connect to a caster, pull one or more mountpoints
   concurrently, and write raw RTCM3 bytes to ``data/ntrip_live/``.
   Each mountpoint gets its own file named
   ``<mountpoint>_<YYYYMMDD>_<HHMMSS>.rtcm3``.

2. **Source table** -- fetch and parse the caster's sourcetable to
   discover available mountpoints, their coordinates, and format.

The client uses the standard NTRIP HTTP protocol (RTCM 2.0/3.x over
HTTP chunked transfer).  Free public casters (AGRS-NL in the
Netherlands, ReNEP in Portugal) require no registration; the EUREF IP
and IGS-RTS casters require credentials (set via the NTRIP_USER and
NTRIP_PASSWORD environment variables).
"""
from __future__ import annotations

import os
import re
import sys
import socket
import ssl
import time
import select
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------- casters
# Public NTRIP casters carrying raw station observations (RTCM3 MSM).
#
# Free casters (no registration required, anonymous credentials accepted):
#   agrs_nl:  Dutch national CORS (AGRS.NET / NETPOS), 35 RTCM3 mountpoints
#             across the Netherlands.  High-quality geodetic receivers
#             (TRIMBLE ALLOY, LEICA GR50, SEPT POLARX5E).  Multi-constellation
#             (GPS+GLO+GAL+BDS).  Hosted by Kadaster / NSGI.
#   renep:    Portuguese national GNSS network (ReNEP), 46 RTCM3 mountpoints
#             across mainland Portugal, Azores, and Madeira.  GPS+GLO.
#             Hosted by DGT / DGeod.
#
# Registered casters (free registration, credentials required for streams):
#   EUREF IP: European reference stations, raw observables, matches the
#             paper's existing European network (ZIMM, BRUX, WTZR, etc.).
#             Free registration required at https://registration.euref.eu/
#   euref_be: ROB (Belgium) EPN broadcaster, carries UK EPN stations
#             (DARE, HERS, LICC, SHOE).  Free registration at
#             https://register.rtcm-ntrip.org/cgi-bin/registration.cgi
#   IGS-RTS:  global stations; some mountpoints carry raw obs, others
#             carry SSR corrections only.  Free registration required at
#             https://www.igs.org/projects/real-time-service
#
# Credentials are read from environment variables:
#   NTRIP_USER      -- registered username
#   NTRIP_PASSWORD  -- registered password
# If not set, the anonymous credentials are used (which work for the
# free casters and for sourcetable fetches on registered casters, but
# not for individual mountpoints on registered casters).
#
# Default caster priority: try free casters first (osnet, agrs_nl, renep), then
# fall back to registered casters if NTRIP_USER is set.
FREE_CASTERS = ["osnet", "agrs_nl", "renep", "rtk2go", "centipede"]
REGISTERED_CASTERS = ["euref", "euref_be", "igs", "igs_euref"]

NTRIP_CASTERS = {
    # --- free casters (no registration) ---
    "agrs_nl": {
        "host": "ntrip.kadaster.nl",
        "port": 2101,
        "user": "anon",
        "password": "anon",
        "free": True,
    },
    "renep": {
        "host": "193.137.94.71",
        "port": 2101,
        "user": "anon",
        "password": "anon",
        "free": True,
    },
    "osnet": {
        "host": "ntrip.os.uk",
        "port": 2101,
        "user": "anon",
        "password": "anon",
        "free": True,
    },
    "rtk2go": {
        "host": "rtk2go.com",
        "port": 2101,
        # RTK2GO requires a valid-looking email as the username.
        # Data consumers do not need to pre-register.
        "user": os.environ.get("RTK2GO_USER", "tep.user@gmail.com"),
        "password": "",
        "free": True,
    },
    "centipede": {
        "host": "caster.centipede.fr",
        "port": 2101,
        # French community RTK network (~1300 mountpoints).  No
        # registration; streams are anonymous.  Carries the RENAG
        # national geodetic stations (TRIMBLE NETR9/ALLOY, surveyed
        # positions) plus community F9P/UM980 bases — fills the
        # 500-1400 km mid-band gap between AGRS-NL and ReNEP.
        "user": "anon",
        "password": "anon",
        "free": True,
    },
    # --- registered casters (free registration) ---
    "euref": {
        "host": "www.euref-ip.net",
        "port": 2101,
        "user": os.environ.get("NTRIP_USER", "anon"),
        "password": os.environ.get("NTRIP_PASSWORD", "anon"),
        "free": False,
    },
    "euref_be": {
        "host": "euref-ip.be",
        "port": 2101,
        "user": os.environ.get("NTRIP_USER", "anon"),
        "password": os.environ.get("NTRIP_PASSWORD", "anon"),
        "free": False,
    },
    "igs": {
        "host": "products.igs-ip.net",
        "port": 2101,
        "user": os.environ.get("NTRIP_USER", "anon"),
        "password": os.environ.get("NTRIP_PASSWORD", "anon"),
        "free": False,
    },
    "igs_euref": {
        "host": "euref-ip.net",
        "port": 2101,
        "user": os.environ.get("NTRIP_USER", "anon"),
        "password": os.environ.get("NTRIP_PASSWORD", "anon"),
        "free": False,
    },
}


def get_default_caster() -> str:
    """Return the best available caster key.

    Prefers registered casters (EUREF, EUREF-BE) if NTRIP_USER is set,
    since they carry UK EPN stations closer to the receiver.  Falls back
    to RTK2GO (free community caster with UK stations at 40-160 km) when
    no registration is available.  The Dutch AGRS-NL caster (269 km) is
    the last resort.  The UK OS Net caster (ntrip.os.uk) requires
    authentication for data streams (the sourcetable is publicly
    readable but mountpoints return 401), so it is only used when
    NTRIP_USER is set.
    """
    # Explicit override wins — lets .env.local pick a working caster
    # (e.g. NTRIP_CASTER=agrs_nl while rtk2go bans this IP).
    _ovr = os.environ.get("NTRIP_CASTER", "").strip().lower()
    if _ovr in NTRIP_CASTERS:
        return _ovr
    if os.environ.get("NTRIP_USER"):
        return "euref"
    # RTK2GO has UK community stations at 40-160 km, much closer than
    # the Dutch AGRS-NL network at 269 km.  It requires an email-style
    # username but no registration.
    return "rtk2go"


@dataclass
class MountPoint:
    """One entry from an NTRIP sourcetable."""
    name: str
    caster_key: str
    host: str
    port: int
    format: str          # e.g. "RTCM 3.2"
    format_details: str  # RTCM message list, e.g. "1006(10),1077(1),1087(1)"
    carrier: str         # e.g. "L1+L2"
    nav_system: str      # e.g. "GPS+GLO+GAL+BDS"
    network: str
    country: str
    lat_deg: float
    lon_deg: float
    nmea: bool
    solution: str
    generator: str
    compress: str
    auth: str
    fee: bool
    bitrate: int

    @property
    def is_rtcm3(self) -> bool:
        # Community casters (centipede) report bare "RTCM3" without the
        # space; EPN-style casters use "RTCM 3.2 MSM7".
        f = self.format.lower()
        return "rtcm 3" in f or "rtcm3" in f

    @property
    def has_raw_obs(self) -> bool:
        """True if the mountpoint carries raw observables (MSM), not just SSR."""
        # SSR correction streams typically have "SSR" in the format or
        # format details.  Raw observation streams carry MSM messages
        # (107x, 108x, 109x, 110x, 111x, 112x).  When the sourcetable
        # supplies the message list, test it directly: legacy "RTCM3"
        # mounts without MSM (e.g. 1004-only RTK streams) are excluded.
        f = self.format.lower()
        if "ssr" in f or "clk" in f or not self.is_rtcm3:
            return False
        if self.format_details:
            return bool(_MSM_MSG_RE.search(self.format_details))
        return True

    @property
    def needs_auth(self) -> bool:
        """True if the mountpoint requires Basic authentication."""
        return self.auth.upper() == "B"

    @property
    def station_id(self) -> str:
        """Extract the 4-character station ID from the mountpoint name."""
        # EUREF mountpoints are typically the 4-char station ID (e.g. "ZIM0")
        # IGS mountpoints may have longer names
        return self.name.upper()[:4].ljust(4, "0")


def _build_auth_header(user: str, password: str) -> str:
    """Build the NTRIP HTTP Authorization header (Basic auth)."""
    import base64
    cred = f"{user}:{password}"
    return "Basic " + base64.b64encode(cred.encode()).decode()


def fetch_sourcetable(caster_key: str | None = None,
                      timeout: float = 30.0) -> list[MountPoint]:
    """Fetch and parse the sourcetable from an NTRIP caster.

    NTRIP casters return a non-standard ``SOURCETABLE 200 OK``
    response (not ``HTTP/1.1 200 OK``), so urllib fails.  We use a
    raw socket and strip the sourcetable header lines before parsing.
    Each line is either a CAS, NET, or STR record.  STR records
    describe available mountpoints.
    """
    if caster_key is None:
        caster_key = get_default_caster()
    cfg = NTRIP_CASTERS[caster_key]
    host, port = cfg["host"], cfg["port"]
    return _fetch_sourcetable_socket(host, port, timeout, caster_key)


def _fetch_sourcetable_socket(host: str, port: int, timeout: float,
                              caster_key: str = "euref") -> list[MountPoint]:
    """Fetch sourcetable via raw socket (NTRIP casters use non-standard HTTP)."""
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        req = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"User-Agent: NTRIP TEP-PPP/1.0\r\n"
            f"Accept: */*\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        sock.sendall(req.encode())
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
        raw = b"".join(chunks).decode("ascii", errors="replace")
        # Strip the sourcetable header (SOURCETABLE 200 OK ... \n\n)
        # or standard HTTP header (HTTP/1.1 200 OK ... \r\n\r\n)
        for sep in ("\r\n\r\n", "\n\n"):
            if sep in raw:
                raw = raw.split(sep, 1)[1]
                break
        return _parse_sourcetable(raw, caster_key)
    finally:
        sock.close()


def _parse_sourcetable(raw: str, caster_key: str) -> list[MountPoint]:
    """Parse a sourcetable text blob into MountPoint objects.

    The EUREF/IGS STR record format is:
      STR;mountpoint;description;format;format_details;carrier;
          nav_system;network;country;lat;lon;nmea;solution;
          generator;compress;auth;fee;[bitrate]
    """
    cfg = NTRIP_CASTERS.get(caster_key, {})
    mps = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("CAS") or line.startswith("NET"):
            continue
        if not line.startswith("STR"):
            continue
        fields = line.split(";")
        if len(fields) < 17:
            continue
        try:
            mp = MountPoint(
                name=fields[1],
                caster_key=caster_key,
                host=cfg.get("host", ""),
                port=cfg.get("port", 2101),
                format=fields[3],
                format_details=fields[4],
                carrier=fields[5],
                nav_system=fields[6],
                network=fields[7],
                country=fields[8],
                lat_deg=float(fields[9]) if fields[9] else 0.0,
                lon_deg=float(fields[10]) if fields[10] else 0.0,
                nmea=fields[11] == "1",
                solution=fields[12],
                generator=fields[13],
                compress=fields[14],
                auth=fields[15],
                fee=fields[16] == "Y",
                bitrate=int(fields[17]) if len(fields) > 17 and fields[17] else 0,
            )
            mps.append(mp)
        except (ValueError, IndexError):
            continue
    return mps


# MSM message IDs: 107x GPS, 108x GLO, 109x GAL, 110x SBAS, 111x QZSS,
# 112x BDS, 113x NavIC — any MSM type (1-7) carries raw observables.
_MSM_MSG_RE = re.compile(r"(?:^|[^\d])1(?:0[7-9]|1[0-3])[1-7](?:[^\d]|$)")


# --------------------------------------------------------------------- recorder
@dataclass
class StreamRecorder:
    """Records raw RTCM3 bytes from a single NTRIP mountpoint to a file.

    The file is written incrementally as data arrives, so a partial
    recording is available even if the connection drops.  The file
    format is raw RTCM3 (no framing), suitable for RTKLIB ``convbin``.

    Public casters silently drop sessions (all mountpoints on a caster
    die together ~15-75 min in) and the resulting half-open TCP sockets
    never deliver another byte or a FIN, so a plain read loop stalls
    forever.  The recorder therefore reconnects: if no bytes arrive for
    ``stall_timeout_s`` the socket is closed and a fresh session is
    opened, appending to the same file until ``duration_s`` expires.
    """
    mountpoint: MountPoint
    output_path: Path
    duration_s: float
    auth_header: str = ""
    stall_timeout_s: float = 60.0
    max_reject_retries: int = 3
    bytes_written: int = 0
    connected: bool = False
    reconnects: int = 0
    stalls: int = 0
    error: Optional[str] = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def stop(self):
        self._stop.set()

    def run(self):
        """Connect and record until duration elapses or stop() is called."""
        cfg = NTRIP_CASTERS[self.mountpoint.caster_key]
        host = self.mountpoint.host or cfg["host"]
        port = self.mountpoint.port or cfg["port"]
        mp = self.mountpoint.name
        req = (
            f"GET /{mp} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"User-Agent: NTRIP TEP-PPP/1.0\r\n"
            f"Accept: */*\r\n"
            f"Authorization: {self.auth_header}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        rejects = 0
        attempts = 0
        try:
            with open(self.output_path, "wb") as f:
                while not self._stop.is_set():
                    if time.monotonic() - start >= self.duration_s:
                        break
                    sock = None
                    try:
                        sock = socket.create_connection((host, port), timeout=15)
                        sock.sendall(req.encode())

                        # Read and discard HTTP response headers
                        header_data = b""
                        while b"\r\n\r\n" not in header_data:
                            chunk = sock.recv(1024)
                            if not chunk:
                                break
                            header_data += chunk
                        # Check for 200 OK (handles both "HTTP/1.1 200 OK"
                        # and "ICY 200 OK" responses -- free casters use ICY)
                        first_line = header_data.split(b"\r\n")[0]
                        if b"200" not in first_line:
                            rejects += 1
                            self.error = (f"NTRIP server rejected: "
                                          f"{first_line.decode('ascii', errors='replace')}")
                            if rejects >= self.max_reject_retries:
                                return
                            time.sleep(min(5.0 * rejects, 30.0))
                            continue
                        rejects = 0
                        attempts += 1
                        if attempts > 1:
                            self.reconnects += 1
                            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] "
                                  f"{mp}: reconnect #{self.reconnects} "
                                  f"(stalls={self.stalls}, "
                                  f"{self.bytes_written/1024:.0f} KB so far)",
                                  file=sys.stderr, flush=True)
                        self.connected = True
                        self.error = None

                        # Record the body (raw RTCM3).  Any body bytes that
                        # arrived with the header are written first.
                        if b"\r\n\r\n" in header_data:
                            body_start = header_data.split(b"\r\n\r\n", 1)[1]
                            if body_start:
                                f.write(body_start)
                                self.bytes_written += len(body_start)

                        last_data = time.monotonic()
                        while not self._stop.is_set():
                            if time.monotonic() - start >= self.duration_s:
                                break
                            ready, _, _ = select.select([sock], [], [], 1.0)
                            if ready:
                                data = sock.recv(4096)
                                if not data:
                                    break
                                f.write(data)
                                self.bytes_written += len(data)
                                last_data = time.monotonic()
                            elif (self.stall_timeout_s > 0
                                  and time.monotonic() - last_data
                                  > self.stall_timeout_s):
                                self.stalls += 1
                                break
                    except Exception as e:
                        self.error = str(e)
                    finally:
                        self.connected = False
                        if sock:
                            try:
                                sock.close()
                            except Exception:
                                pass
                    # Brief backoff before the next connection attempt.
                    if (time.monotonic() - start < self.duration_s
                            and not self._stop.is_set()):
                        self._stop.wait(2.0)
        except Exception as e:
            self.error = str(e)
        finally:
            self.connected = False


def record_mountpoints(mountpoints: list[MountPoint],
                       output_dir: Path,
                       duration_s: float = 3600.0,
                       stall_timeout_s: float = 60.0,
                       ) -> list[StreamRecorder]:
    """Record multiple mountpoints concurrently.

    Each mountpoint is recorded in its own thread.  The function
    blocks until all recordings complete (or the duration elapses).
    Connections that stop delivering data for ``stall_timeout_s``
    seconds are re-established automatically (the recording keeps
    appending to the same file).

    Returns the list of StreamRecorder objects with their status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    recorders = []
    for mp in mountpoints:
        cfg = NTRIP_CASTERS[mp.caster_key]
        auth = _build_auth_header(cfg["user"], cfg["password"])
        fname = f"{mp.name}_{timestamp}.rtcm3"
        rec = StreamRecorder(
            mountpoint=mp,
            output_path=output_dir / fname,
            duration_s=duration_s,
            auth_header=auth,
            stall_timeout_s=stall_timeout_s,
        )
        recorders.append(rec)

    threads = []
    for rec in recorders:
        t = threading.Thread(target=rec.run, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=duration_s + 30)

    return recorders


# --------------------------------------------------------------------- discovery
def find_stations(caster_key: str = "",
                   station_names: Optional[list[str]] = None,
                   nav_filter: str = "GPS",
                   ) -> list[MountPoint]:
    """Find mountpoints for specific stations or all GPS-capable ones.

    If *caster_key* is empty, searches all free casters (AGRS-NL, ReNEP)
    and the EUREF caster (if credentials are available).  If
    *station_names* is given, match by 4-char ID (e.g. ["DELF", "WSRT"]).
    Otherwise return all RTCM3 mountpoints carrying the requested nav
    system.
    """
    if caster_key:
        caster_keys = [caster_key]
    else:
        caster_keys = list(FREE_CASTERS)
        if os.environ.get("NTRIP_USER"):
            caster_keys.extend(REGISTERED_CASTERS)

    results: list[MountPoint] = []
    seen: set[str] = set()
    for ck in caster_keys:
        if ck not in NTRIP_CASTERS:
            continue
        try:
            mps = fetch_sourcetable(ck)
        except Exception:
            continue
        if station_names:
            wanted = {s.upper() for s in station_names}
            for m in mps:
                if (m.is_rtcm3 and m.has_raw_obs
                        and (m.station_id in wanted or m.name.upper() in wanted)
                        and m.name not in seen):
                    results.append(m)
                    seen.add(m.name)
        else:
            for m in mps:
                if (m.is_rtcm3 and m.has_raw_obs
                        and nav_filter.upper() in m.nav_system.upper()
                        and m.name not in seen):
                    results.append(m)
                    seen.add(m.name)
    return results


# --------------------------------------------------------------- OS Net API
#
# The Ordnance Survey OS Net API provides RINEX v3 files for UK CORS
# stations via a RESTful HTTP interface.  This is the only free source
# of UK GNSS reference data that includes stations co-located with the
# receiver (e.g. EASN at 1.7 km).  The API requires a free API key
# from https://osdatahub.os.uk/.
#
# Environment variable: OS_NET_API_KEY
#
# The API is documented at:
#   https://docs.os.uk/os-apis/accessing-os-apis/os-net-api
#
# Key endpoints:
#   GET /positioning/osnet/v1/rinex/stations
#       List all available stations.
#   GET /positioning/osnet/v1/stations/{stationId}/rinex?start=...&end=...
#       List RINEX files for a station in a time window.
#   GET {file.url}
#       Download a specific RINEX file (URL from the list response).
#
# RINEX files are delivered as hourly 30-second-epoch files in RINEX 3
# format, compressed with gzip (.gz).  The API requires the key as
# either a query parameter (?key=...) or a header (key: ...).

OS_NET_API_BASE = "https://api.os.uk/positioning/osnet/v1"
OS_NET_STATIONS = {
    # 4-char station IDs, sorted by distance to the receiver (53.64N, 0.10E).
    # Coordinates from the OS Net sourcetable (ntrip.os.uk:2101).
    "EASN": (53.65, 0.12, 1.7),    # Easington — co-located
    "SWAN": (53.74, -0.50, 41.0),  # Swanland
    "SKEE": (53.14, 0.35, 58.0),   # Skegness
    "LINO": (53.25, -0.52, 59.7),  # Lincoln
    "YEAL": (54.16, -1.10, 97.6),  # Yealam
    "HOOB": (53.48, -1.38, 99.4),  # Hoober
    "WEYB": (52.95, 1.13, 102.8),  # Weybourne
    "LEED": (53.80, -1.66, 117.2), # Leeds
    "LOFT": (54.56, -0.86, 119.9), # Loftus
    "PETE": (52.58, -0.26, 120.3), # Peterborough
}


def os_net_api_key() -> str:
    """Return the OS Net API key from the environment, or empty string."""
    return (os.environ.get("OS_NET_API_KEY", "")
            or os.environ.get("OSNET_API_KEY", ""))


def os_net_available() -> bool:
    """True if the OS Net API key is configured."""
    return bool(os_net_api_key())


def os_net_list_rinex(station_id: str,
                      start_iso: str,
                      end_iso: str,
                      timeout: float = 30.0) -> list[dict]:
    """List available RINEX files for a station in a time window.

    Parameters
    ----------
    station_id : str
        4-character OS Net station ID (e.g. "EASN").
    start_iso : str
        Start time in ISO 8601 format (e.g. "2026-09-05T22:00:00Z").
    end_iso : str
        End time in ISO 8601 format.
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    list[dict]
        List of RINEX file metadata dicts, each with keys "fileName",
        "url", "station", etc.  Empty list on error.
    """
    key = os_net_api_key()
    if not key:
        return []
    url = (f"{OS_NET_API_BASE}/stations/{station_id.upper()}/rinex"
           f"?start={start_iso}&end={end_iso}&key={key}")
    try:
        req = urllib.request.Request(url, headers={"key": key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "files" in data:
                return data["files"]
            return []
    except Exception:
        return []


def os_net_download_rinex(file_url: str,
                          dest_path: Path,
                          timeout: float = 60.0) -> bool:
    """Download a RINEX file from the OS Net API.

    Parameters
    ----------
    file_url : str
        The file URL from os_net_list_rinex().
    dest_path : Path
        Destination file path (will be created/overwritten).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    bool
        True on success, False on error.
    """
    key = os_net_api_key()
    if not key:
        return False
    try:
        url = file_url
        if "?" not in url:
            url = f"{url}?key={key}"
        elif "key=" not in url:
            url = f"{url}&key={key}"
        req = urllib.request.Request(url, headers={"key": key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception:
        return False
