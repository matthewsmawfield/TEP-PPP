#!/usr/bin/env python3
"""Fetch fresh IGS ultra-rapid precise products for the live PPP rig.

The archive (data/rinex/sp3, data/rinex/clk) holds historical final/rapid
products, but for the CURRENT day only ultra-rapid (ULT) products are
available in real time.  This script downloads the latest CODE ultra-rapid
SP3 and GRG ultra-rapid CLK from CDDIS or the Scripps/UCSD open-access
mirror, decompresses them into data/ppp_live/precise/, and names them
orbit_cod.sp3 / clock.clk so both the custom EKF (via load_cached_products)
and RTKLIB (via its cached-file fallback) pick them up.

Run periodically (every 2-4 h) so the products always cover the live
observation window:

    python3 scripts/live/fetch_precise_products.py
    # or via cron / launchd every 3 hours

Tries CDDIS first (authenticates with Earthdata Login via ~/.netrc for
urs.earthdata.nasa.gov).  Without CDDIS credentials, falls back to the
Scripps/UCSD mirror (https://garner.ucsd.edu/products/) which is
open-access and requires no authentication.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import netrc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST = PROJECT_ROOT / "data" / "ppp_live" / "precise"
CDDIS_PRODUCTS = "https://cddis.nasa.gov/archive/gnss/products"
CDDIS_DAILY = "https://cddis.nasa.gov/archive/gnss/data/daily"
SCRIPPTS_PRODUCTS = "https://garner.ucsd.edu/products"


def _clock_delta_stats(prev_recs, new_path: Path):
    """Per-satellite clock delta prev->new at each sat's newest common
    epoch, in metres.

    The live pos_datum stack learns a product-datum jump only after the
    next few windows solve; this measures the jump AT merge time so the
    shift is a recorded observable, not a post-hoc inference.  It also
    separates real issue changes from byte-identical re-merges — a
    re-merge of the same products yields deltas ~0, so downstream
    datum bookkeeping can tell a true datum move from an mtime-only
    refresh.
    """
    if not prev_recs:
        return None
    try:
        _, new = _parse_clk_records(new_path, keep_gps=None)
    except Exception:
        return None
    if not new:
        return None
    per_sat: dict[str, float] = {}
    latest = None
    for sat in sorted(set(prev_recs) & set(new)):
        po, pn = prev_recs[sat], new[sat]
        if not po or not pn:
            continue
        t = min(po[-1][0], pn[-1][0])
        ro, rn = _clk_interp(po, t), _clk_interp(pn, t)
        if ro is None or rn is None:
            continue
        per_sat[sat] = round((rn[0] - ro[0]) * 299792458.0, 4)
        if latest is None or t > latest:
            latest = t
    if not per_sat:
        return None

    def _med(a):
        a = sorted(a)
        n = len(a)
        return a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])

    vals = list(per_sat.values())
    gps = [v for s, v in per_sat.items() if s.startswith("G")]
    non = [v for s, v in per_sat.items() if not s.startswith("G")]
    return {
        "per_sat_m": per_sat,
        "n": len(vals),
        "med_m": round(_med(vals), 4),
        "gps_med_m": round(_med(gps), 4) if gps else None,
        "nongps_med_m": round(_med(non), 4) if non else None,
        "absmax_m": round(max(abs(v) for v in vals), 3),
        "epoch": latest.isoformat() + "Z" if latest else None,
    }


def _mark_clock_update(issue: str = "", merged_with: str = "",
                       prev_recs=None) -> None:
    """Record a clock.clk replacement so the live clock-prior chain
    can detect product-datum boundaries (the merge-straddle blowup
    class: a prior anchored on pre-merge history can't transfer across
    the issue swap).  Written on every clock.clk promotion, including
    the igu merge.  When ``prev_recs`` (parsed records of the outgoing
    clock.clk) is supplied, the per-satellite clock delta to the new
    file is recorded as ``dclk`` — the product-datum jump measured at
    merge time."""
    try:
        import json as _json
        _rec = {
            "ts": time.time(),
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "issue": issue,
            "merged_with": merged_with,
        }
        if prev_recs:
            _d = _clock_delta_stats(prev_recs, DEST / "clock.clk")
            if _d is not None:
                _rec["dclk"] = _d
        (DEST / "last_clock_update.json").write_text(_json.dumps(_rec))
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch-precise")

# Thrash guard (added 2026-10-03 after the 2nd watchdog hard-reset): the
# clock-merge step (~100-200 MB burst) overlapping a gps_web cold-start
# extraction starved the systemd watchdog pet -> box reset mid-fetch.
# Defer heavy work while MemAvailable is low; cap the wait so a
# chronically-tight box still fetches rather than going stale.
_FETCH_MIN_MEM_MB = int(os.environ.get("TEP_FETCH_MIN_MEM_MB", "250"))
_FETCH_MAX_WAIT_S = int(os.environ.get("TEP_FETCH_MAX_WAIT_S", "1200"))


def _mem_available_mb() -> float:
    try:
        m = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
        return int(m.group(1)) / 1024.0 if m else 1e9
    except Exception:
        return 1e9


def _wait_for_memory() -> None:
    deadline = time.monotonic() + _FETCH_MAX_WAIT_S
    while True:
        avail = _mem_available_mb()
        if avail >= _FETCH_MIN_MEM_MB:
            return
        if time.monotonic() >= deadline:
            log.warning("MemAvailable %.0fMB still below %dMB after %ds wait — fetching anyway",
                        avail, _FETCH_MIN_MEM_MB, _FETCH_MAX_WAIT_S)
            return
        log.info("MemAvailable %.0fMB < %dMB — waiting 30s before heavy fetch work",
                 avail, _FETCH_MIN_MEM_MB)
        time.sleep(30)


def _gps_week(d: dt.date) -> int:
    """GPS week number for a given date."""
    epoch = dt.date(1980, 1, 6)  # GPS epoch: 1980-01-06 (Sunday)
    return (d - epoch).days // 7


def _doy(d: dt.date) -> int:
    return (d - dt.date(d.year, 1, 1)).days + 1


def _ult_issue_dt(name: str) -> dt.datetime | None:
    """Nominal issue epoch of an ultra-rapid SP3 filename.

    Two naming schemes exist for the same product:
    - Scripps legacy: ``igu<wk><d>_<hh>.sp3.Z`` — GPS week + day-of-week +
      issue hour.
    - CDDIS long name: ``<AC>0OPSULT_<YYYYDDDHHMM>_..._ORB.SP3[.gz]`` —
      the filename epoch is the start of the 2-day product.
    """
    m = re.match(r"igu(\d{4})(\d)_(\d{2})\.sp3\.Z$", name)
    if m:
        wk, dow, hh = int(m.group(1)), int(m.group(2)), int(m.group(3))
        base = dt.date(1980, 1, 6) + dt.timedelta(weeks=wk, days=dow)
        return dt.datetime(base.year, base.month, base.day, hh)
    m = re.match(r"[A-Z0-9]{3}0OPSULT_(\d{4})(\d{3})(\d{2})(\d{2})_", name)
    if m:
        y, doy, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        d = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
        return dt.datetime(d.year, d.month, d.day, hh, mm)
    return None


def _ult_sp3_candidates(files: list[str]) -> list[str]:
    """Ultra-rapid SP3 filenames sorted by issue epoch (newest first).

    Matches Scripps ``igu*.sp3.Z`` names and the CDDIS long-name
    ``*0OPSULT_*_ORB.SP3*`` products (IGS0OPSULT is the same product as
    igu; COD0OPSULT etc. are other analysis centres, often fresher).
    """
    out = []
    for f in files:
        if (f.startswith("igu") and f.endswith(".sp3.Z")) or \
                (re.match(r"[A-Z0-9]{3}0OPSULT_", f) and "ORB.SP3" in f):
            key = _ult_issue_dt(f)
            if key is not None:
                out.append((key, f))
    out.sort(key=lambda t: t[0], reverse=True)
    return [f for _, f in out]


def _auth() -> tuple[str, str] | None:
    try:
        creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if creds:
            return creds[0], creds[2]
    except Exception:
        pass
    return None


def _http_get_bytes(url: str, auth: tuple | None = None,
                    timeout: int = 60) -> bytes | None:
    """GET *url* and return the body, or None on failure.

    Uses requests first; on failure falls back to the system curl,
    which resolves through the OS trust store.  The Scripps mirror
    (garner.ucsd.edu) serves an incomplete TLS chain — leaf only,
    no InCommon intermediate — which certifi cannot build but the
    macOS/Linux system stores can.
    """
    try:
        if auth:
            s = requests.Session()
            s.auth = auth
            r = s.get(url, allow_redirects=True, timeout=timeout)
        else:
            r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        pass
    try:
        import subprocess
        cmd = ["curl", "-fsSL", "--max-time", str(timeout), url]
        if auth:
            cmd[1:1] = ["-u", f"{auth[0]}:{auth[1]}"]
        p = subprocess.run(cmd, capture_output=True,
                           timeout=timeout + 20)
        if p.returncode == 0 and p.stdout:
            return p.stdout
    except Exception:
        pass
    return None


def _list_scripps_products(gps_week: int) -> list[str]:
    """List filenames on the Scripps/UCSD mirror only.

    The IGS combined ultra-rapid (igu<wk><d>_<hh>.sp3.Z) is published
    here hours before the equivalent IGS0OPSULT long-name product
    reaches CDDIS, so the freshest GPS clocks often live in this
    listing alone.
    """
    url = f"{SCRIPPTS_PRODUCTS}/{gps_week}/"
    body = _http_get_bytes(url, timeout=60)
    if body is None:
        log.warning("Scripps listing %s failed", url)
        return []
    files = re.findall(r'href="([^"]+)"', body.decode("utf-8", "ignore"))
    files = [f for f in files if not f.startswith("/") and "." in f]
    if files:
        log.info("Scripps listing: %d files", len(files))
    return files


def _list_products(gps_week: int) -> list[str]:
    """List product filenames for a GPS week.

    Tries CDDIS (with auth) first, then falls back to the Scripps/UCSD
    open-access mirror which requires no authentication.
    """
    # Try CDDIS first — with retries on one shared session.  The
    # Earthdata oauth hop occasionally drops a connection mid-handshake
    # (SSL EOF); reusing the session keeps the proxyauth cookies so a
    # retry skips most of the handshake.  A listing failure costs us
    # every CDDIS-only product in the pool (GRG CLK, COD ULT) for a
    # whole fetch cycle.
    creds = _auth()
    if creds:
        url = f"{CDDIS_PRODUCTS}/{gps_week}/"
        s = requests.Session()
        s.auth = creds
        for attempt in range(3):
            try:
                r = s.get(url, allow_redirects=True, timeout=60)
                if r.status_code == 200:
                    files = [f for f in
                             re.findall(r'id="([^"]+)"', r.text)
                             if "." in f]
                    if files:
                        log.info("CDDIS listing: %d files", len(files))
                        return files
                    # 200 with no product rows is the Earthdata login
                    # page (transient auth/rate-limit), not an empty
                    # week directory — retry before giving up.
                    log.warning("CDDIS listing HTTP 200 but 0 product "
                                "files (login page?)%s",
                                " (retrying)" if attempt < 2 else "")
                else:
                    log.warning("CDDIS listing HTTP %d%s",
                                r.status_code,
                                " (retrying)" if attempt < 2 else "")
            except Exception as e:
                log.warning("CDDIS listing failed%s: %s",
                            " (retrying)" if attempt < 2 else "", e)
            if attempt < 2:
                time.sleep(20)

    # Fall back to Scripps/UCSD open-access mirror
    log.info("Trying Scripps/UCSD mirror (open access)...")
    files = _list_scripps_products(gps_week)
    if files:
        return files

    log.error("No product listing available from any mirror")
    return []


def _download(filename: str, dest_name: str,
               gps_week: int | None = None) -> Path | None:
    """Download a product file and decompress into DEST.

    Tries CDDIS (with auth) first, then falls back to Scripps/UCSD.
    If gps_week is None, uses the current GPS week.
    """
    if gps_week is None:
        gps_week = _gps_week(dt.date.today())
    DEST.mkdir(parents=True, exist_ok=True)

    urls_to_try = []
    # CDDIS (with auth if available)
    creds = _auth()
    if creds:
        urls_to_try.append((f"{CDDIS_PRODUCTS}/{gps_week}/{filename}", creds))
    # Scripps (open access, always try as fallback)
    urls_to_try.append((f"{SCRIPPTS_PRODUCTS}/{gps_week}/{filename}", None))

    # One retry pass over all mirrors — products are often listed in the
    # directory index before they are fully staged, so an immediate 404
    # can succeed a short while later.
    urls_to_try = urls_to_try + [(None, None)] + urls_to_try

    for url, auth in urls_to_try:
        if url is None:
            time.sleep(30)
            continue
        try:
            if auth:
                s = requests.Session()
                s.auth = auth
                r = s.get(url, allow_redirects=True, timeout=120)
                if r.status_code != 200:
                    log.debug("Download %s -> HTTP %d", url, r.status_code)
                    continue
                # Check we got actual data, not an HTML login page
                content_type = r.headers.get("content-type", "")
                if "html" in content_type.lower():
                    log.debug("Download %s returned HTML (auth redirect)",
                              url)
                    continue
                raw = r.content
            else:
                # _http_get_bytes falls back to system curl for mirrors
                # with incomplete TLS chains (garner.ucsd.edu).
                raw = _http_get_bytes(url, timeout=120)
                if raw is None or raw[:1] == b"<":
                    log.debug("Download %s failed or returned HTML", url)
                    continue
            # Decompress if gzipped (.gz) or Unix-compressed (.Z)
            if filename.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                except Exception:
                    log.error("Decompress failed for %s", filename)
                    continue
            elif filename.endswith(".Z"):
                try:
                    # .Z files use Unix compress (LZW).  Try gzip first
                    # (some mirrors serve gzip with .Z extension), then
                    # fall back to the system 'uncompress' command.
                    try:
                        raw = gzip.decompress(raw)
                    except Exception:
                        import subprocess
                        import tempfile
                        import os
                        # Write to a temp file and use uncompress
                        with tempfile.NamedTemporaryFile(
                                suffix='.Z', delete=False) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name
                        try:
                            result = subprocess.run(
                                ['uncompress', '-c', tmp_path],
                                capture_output=True, timeout=60)
                            if result.returncode == 0:
                                raw = result.stdout
                            else:
                                log.error(
                                    "uncompress failed for %s: %s",
                                    filename, result.stderr.decode(
                                        'ascii', errors='ignore')[:200])
                                os.unlink(tmp_path)
                                continue
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                except Exception:
                    log.error("Decompress failed for %s", filename)
                    continue
            out = DEST / dest_name
            out.write_bytes(raw)
            log.info("Downloaded %s -> %s (%d bytes) from %s",
                     filename, out, len(raw),
                     "CDDIS" if auth else "Scripps")
            return out
        except Exception as e:
            log.debug("Download %s failed: %s", url, e)
            continue

    log.warning("Download failed for %s from all mirrors", filename)
    return None


def _extract_clk_from_sp3(sp3_path: Path, clk_path: Path) -> bool:
    """Extract satellite clocks from an SP3 file into a RINEX CLK file.

    SP3 files contain clock biases in the 5th column of PG/PC lines
    (in microseconds).  This converts them to RINEX CLK format so the
    EKF and RTKLIB can use them when no dedicated CLK file is available.

    Used as a fallback when CODE/GRG ultra-rapid CLK products are not
    yet available for the current GPS week but the IGS ultra-rapid SP3
    (igu<wk><day>_00.sp3) is available on the Scripps mirror.
    """
    import datetime as _dt
    opener = gzip.open if sp3_path.suffix in (".gz", ".Z") else open
    try:
        with opener(sp3_path, "rt", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        log.warning("Failed to read SP3 for clock extraction: %s", e)
        return False

    # Parse SP3 epochs and satellite clocks
    clk_records: list[str] = []
    current_epoch: _dt.datetime | None = None
    for line in lines:
        if line.startswith("*  "):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    _year = int(parts[1])
                    _month = int(parts[2])
                    _day = int(parts[3])
                    _hour = int(parts[4])
                    _minute = int(parts[5])
                    _sec_float = float(parts[6])
                    _sec = int(_sec_float)
                    _usec = int((_sec_float - _sec) * 1e6)
                    current_epoch = _dt.datetime(
                        _year, _month, _day, _hour, _minute, _sec, _usec)
                except (ValueError, IndexError):
                    current_epoch = None
        elif line.startswith("P") and current_epoch is not None:
            # SP3 format: PG01 where P=record type, G=constellation, 01=PRN.
            # Any position record (PG/PR/PE/PC/...) counts — multi-GNSS
            # SP3 files (COD ULT) carry G+R+E clocks.
            # Satellite ID is characters 1-3 (e.g. "G01")
            sat_id = line[1:4].strip()
            cols = line.split()
            if len(cols) >= 5:
                try:
                    clk_us = float(cols[4])  # microseconds
                    clk_s = clk_us * 1e-6   # seconds
                    # RINEX CLK AS record, fixed-column format so both the
                    # token parser (gps_web, parts[9]) and the column parser
                    # (RTKLIB readrnxclk: epoch at cols 8-33, value at cols
                    # 40-58) read the same fields:
                    #   "AS C19  2026  9 14 12  0  0.000000  2   -9.3E-04 ..."
                    # A previous version wrote the seconds field as
                    # "ss.00000000" (8 decimals), which shifted the clock
                    # value to column 42; RTKLIB then read a truncated
                    # exponent ("1.7E-0" -> 1.7 s clock) and PPP failed.
                    sec = current_epoch.second + current_epoch.microsecond / 1e6
                    clk_records.append(
                        f"AS {sat_id:4s}{current_epoch.year:5d}"
                        f"{current_epoch.month:3d}{current_epoch.day:3d}"
                        f"{current_epoch.hour:3d}{current_epoch.minute:3d}"
                        f"{sec:10.6f}  2   {clk_s: .12E}"
                        f"  0.000000000000E+00\n")
                except (ValueError, IndexError):
                    continue

    if not clk_records:
        log.warning("No clock records extracted from SP3")
        return False

    # Write RINEX CLK file
    header = (
        "RINEX CLOCK TYPE\n"
        f"IGS ULTRA-RAPID CLOCK FROM SP3\n"
        f"PG CLOCK EXTRACTION\n"
        "  1.0           T\n"
        f"{'':60s}COMMENT\n"
        f"{len(clk_records):6d}{'':54s}# OF EPOCHS\n"
        "G   2 G   C\n"
        "  1   1   0   0   0   0   0   0   0   0   0\n"
        "  1.0\n"
        "END OF HEADER\n"
    )
    try:
        clk_path.write_text(header + "".join(clk_records))
        log.info("Extracted %d clock records from SP3 -> %s",
                 len(clk_records), clk_path)
        return True
    except Exception as e:
        log.warning("Failed to write CLK from SP3: %s", e)
        return False


def _parse_clk_records(path: Path, keep_gps: bool | None = None):
    """Parse a RINEX CLK file.

    Returns (header_lines, {sat_id: [(epoch_dt, clk_s, std_s), ...]}).
    ``keep_gps`` True keeps only GPS sats, False keeps only non-GPS,
    None keeps all.  Records are sorted per satellite.
    """
    import datetime as _dt
    header: list[str] = []
    recs: dict[str, list] = {}
    with open(path, "rt", errors="ignore") as f:
        for line in f:
            if not line.startswith("AS "):
                header.append(line)
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            sat = parts[1]
            is_gps = sat.startswith("G")
            if keep_gps is not None and is_gps != keep_gps:
                continue
            try:
                epoch = _dt.datetime(int(parts[2]), int(parts[3]),
                                     int(parts[4]), int(parts[5]),
                                     int(parts[6]), int(float(parts[7])))
                clk, std = float(parts[9]), float(parts[10]) if len(parts) > 10 else 0.0
            except (ValueError, IndexError):
                continue
            recs.setdefault(sat, []).append((epoch, clk, std))
    for sat in recs:
        recs[sat].sort()
    return header, recs


def _clk_max_epoch(path: Path):
    """Latest record epoch in a RINEX CLK file (or None if unparseable)."""
    try:
        _, recs = _parse_clk_records(path)
    except Exception:
        return None
    best = None
    for samples in recs.values():
        if samples and (best is None or samples[-1][0] > best):
            best = samples[-1][0]
    return best


def _sp3_obs_boundary(path: Path):
    """Observed/predicted boundary of an ultra-rapid SP3.

    CODE and others annotate the split explicitly
    (``/* Observed/predicted: 15/33 hours``) and it is NOT always a
    symmetric 24/24 — COD ULT is 15h observed + 33h predicted, so the
    coverage midpoint overstates the boundary by ~9h.  Parse the comment
    when present; fall back to the coverage midpoint otherwise."""
    obs_h = None
    first = last = None
    try:
        with open(path, "rt", errors="ignore") as f:
            for line in f:
                if line.startswith("/*") and obs_h is None:
                    m = re.search(
                        r"[Oo]bserved/[Pp]redicted:\s*(\d+)\s*/\s*\d+",
                        line)
                    if m:
                        obs_h = int(m.group(1))
                elif line.startswith("* "):
                    p = line.split()
                    if len(p) >= 7:
                        try:
                            e = dt.datetime(int(p[1]), int(p[2]),
                                            int(p[3]), int(p[4]),
                                            int(p[5]),
                                            int(float(p[6])))
                        except (ValueError, IndexError):
                            continue
                        if first is None:
                            first = e
                        last = e
    except Exception:
        return None
    if obs_h is not None and first is not None:
        return first + dt.timedelta(hours=obs_h)
    if first is not None and last is not None:
        return first + (last - first) / 2
    return None


def _clk_mid_epoch(path: Path):
    """Midpoint of the record-epoch coverage — the obs/pred boundary for
    the 24h-observed + 24h-predicted ultra-rapid products."""
    try:
        _, recs = _parse_clk_records(path)
    except Exception:
        return None
    lo = hi = None
    for samples in recs.values():
        if not samples:
            continue
        if lo is None or samples[0][0] < lo:
            lo = samples[0][0]
        if hi is None or samples[-1][0] > hi:
            hi = samples[-1][0]
    if lo is None or hi is None:
        return None
    return lo + (hi - lo) / 2


def _clk_interp(samples: list, t, max_gap_s: float = 960.0):
    """Linear-interpolate (clk, std) at epoch ``t`` from sorted samples.
    Returns None outside coverage or across a gap > max_gap_s."""
    import bisect
    epochs = [s[0] for s in samples]
    i = bisect.bisect_left(epochs, t)
    if i < len(epochs) and epochs[i] == t:
        return samples[i][1], samples[i][2]
    if i == 0 or i >= len(samples):
        return None
    lo, hi = samples[i - 1], samples[i]
    gap = (hi[0] - lo[0]).total_seconds()
    if gap > max_gap_s:
        return None
    f = (t - lo[0]).total_seconds() / gap
    return lo[1] + f * (hi[1] - lo[1]), lo[2] + f * (hi[2] - lo[2])


def _merge_clocks(grg_path: Path, fresh_path: Path, out_path: Path) -> bool:
    """Merge fresh ultra-rapid SP3 clocks with GRG clocks.

    When the GRG CLK is >24h old, the predicted clocks have degraded.
    A fresher ultra-rapid SP3 (igu / IGS0OPSULT / COD0OPSULT etc.) has
    observed clocks.  The merge replaces every satellite present in the
    fresh file — GPS-only for igu, G+R+E for COD ULT — and keeps GRG
    records for the rest (the igu has 15-min intervals vs GRG's 5-min,
    so the EKF interpolates between epochs for the replaced sats).

    The output is written on a sorted union time grid with per-epoch
    completeness: at every epoch each satellite emits a record (exact or
    linearly interpolated across <=16 min gaps).  RINEX CLK requires
    monotonically increasing epochs, and RTKLIB's readrnxclk groups
    records into per-epoch blocks while pephclk binary-searches them --
    the previous append-only layout produced a non-sorted file where
    each epoch contained only one system's records, so multi-system
    RTKLIB consumers failed clock lookup for every satellite.
    """
    try:
        _hdr, fresh = _parse_clk_records(fresh_path, keep_gps=None)
    except Exception:
        return False
    if not fresh:
        log.warning("No clock records in fresh SP3-extracted CLK file")
        return False
    try:
        grg_hdr, grg = _parse_clk_records(grg_path, keep_gps=None)
    except Exception:
        return False

    merged: dict[str, list] = dict(grg)
    n_replaced_grg = sum(len(grg.get(s, ())) for s in fresh)
    merged.update(fresh)

    epochs = sorted({e for ss in merged.values() for e, _, _ in ss})
    out_lines = [l for l in grg_hdr if not l.startswith(("AS ", "AR "))]
    n_recs = 0
    for t in epochs:
        for sat in sorted(merged):
            r = _clk_interp(merged[sat], t)
            if r is None:
                continue
            clk, std = r
            out_lines.append(
                "AS %-4s %4d%3d%3d%3d%3d%10.6f%3d   %19.12E%20.12E\n"
                % (sat, t.year, t.month, t.day, t.hour, t.minute,
                   t.second + t.microsecond / 1e6, 2, clk, std))
            n_recs += 1
    try:
        out_path.write_text("".join(out_lines))
        log.info("Merged clocks: replaced %d GRG records across %d sats, "
                 "added %d fresh SP3 records (%d records on %d sorted epochs)",
                 n_replaced_grg, len(fresh),
                 sum(len(v) for v in fresh.values()),
                 n_recs, len(epochs))
        return True
    except Exception:
        return False


def _verify_coverage(path: Path, target_date: dt.date) -> bool:
    """Check that an SP3 or CLK file contains epochs on the target date."""
    import re
    opener = gzip.open if path.suffix in (".gz", ".Z") else open
    try:
        with opener(path, "rt", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 50000:
                    break
                if line.startswith("*  "):
                    p = line.split()
                    if len(p) >= 4:
                        rec_date = dt.date(int(p[1]), int(p[2]), int(p[3]))
                        if rec_date == target_date:
                            return True
                m = re.match(r"^(?:AS|AR|CR)\s+\S+\s+(\d{4})\s+(\d+)\s+(\d+)", line)
                if m:
                    rec_date = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if rec_date == target_date:
                        return True
    except Exception:
        pass
    return False


def main() -> int:
    try:
        os.nice(19)
    except OSError:
        pass
    _wait_for_memory()
    today = dt.date.today()
    gps_week = _gps_week(today)
    doy = _doy(today)
    log.info("Fetching ultra-rapid products for %s (GPS week %d, DOY %d)",
             today, gps_week, doy)

    files = _list_products(gps_week)
    # Also check the PREVIOUS GPS week — ultra-rapid products issued
    # yesterday or 2 days ago may be in the previous week's directory
    # and still cover today (2-day products).  This is critical at the
    # start of a new GPS week when the current week's products aren't
    # available yet.
    prev_week_files = []
    # Check if current week has CODE/GRG products.  If not, check
    # previous week (the current week may only have IGS ultra-rapid
    # which is less accurate than GRG from the previous week).
    _has_cod_grg = (files and
                    any(f.startswith("COD0OPSULT") for f in files) and
                    any(f.startswith("GRG0OPSULT") for f in files))
    if not files or not _has_cod_grg:
        prev_gps_week = gps_week - 1
        log.info("No CODE/GRG products for current week %d — checking previous week %d",
                 gps_week, prev_gps_week)
        prev_week_files = _list_products(prev_gps_week)
        # Merge: prefer previous week's CODE/GRG over current week's IGS
        if prev_week_files:
            _prev_cod_grg = [f for f in prev_week_files
                             if f.startswith("COD0OPSULT") or f.startswith("GRG0OPSULT")]
            if _prev_cod_grg:
                files = prev_week_files
    if not files:
        log.error("No products found for GPS week %d or %d",
                 gps_week, gps_week - 1)
        return 1

    target_tag = f"{today.year}{doy:03d}"
    yesterday = today - dt.timedelta(days=1)
    ytag = f"{yesterday.year}{_doy(yesterday):03d}"
    two_days_ago = today - dt.timedelta(days=2)
    y2tag = f"{two_days_ago.year}{_doy(two_days_ago):03d}"

    # Best SP3: CODE ultra-rapid issued today (covers today + tomorrow),
    # then yesterday/2-day issues, then any ULT covering today, then
    # GRG ULT — chained into ONE ordered list (each group newest-first)
    # so a listed-but-bad candidate falls through to the next group
    # instead of leaving sp3_path unset.
    sp3_candidates = list(dict.fromkeys(
        sorted([f for f in files
                if f.startswith("COD0OPSULT") and target_tag in f
                and "ORB.SP3" in f], reverse=True)
        + sorted([f for f in files
                  if f.startswith("COD0OPSULT") and ytag in f
                  and "ORB.SP3" in f], reverse=True)
        + sorted([f for f in files
                  if f.startswith("COD0OPSULT") and y2tag in f
                  and "ORB.SP3" in f], reverse=True)
        + sorted([f for f in files
                  if "ULT" in f and "ORB.SP3" in f and target_tag in f],
                 reverse=True)
        + sorted([f for f in files
                  if f.startswith("GRG0OPSULT")
                  and (ytag in f or y2tag in f) and "ORB.SP3" in f],
                 reverse=True)))
    # Fallback: any ultra-rapid SP3 — Scripps igu<wk><day>_00.sp3.Z names
    # and CDDIS long-name <AC>0OPSULT_*_ORB.SP3.gz products.  Available
    # even when CODE/GRG products are not yet up for the current GPS
    # week; the SP3 contains both orbits and clocks (clocks are extracted
    # separately below).
    if not sp3_candidates:
        sp3_candidates = _ult_sp3_candidates(files)

    # Best CLK: GRG ultra-rapid (has 5-min clocks + WL biases for AR)
    clk_candidates = [
        f for f in files
        if f.startswith("GRG0OPSULT") and "CLK.CLK" in f
    ]
    # Prefer the latest issue that covers today
    clk_candidates.sort(reverse=True)

    # Determine which GPS week to download from
    download_week = gps_week
    if prev_week_files:
        download_week = gps_week - 1

    # Snapshot the outgoing clock.clk records BEFORE any promotion so
    # the per-satellite datum delta (old product -> new product) can be
    # recorded in last_clock_update.json at each replacement.
    _prev_clk_recs = None
    try:
        _prev_clk_path = DEST / "clock.clk"
        if _prev_clk_path.exists():
            _, _prev_clk_recs = _parse_clk_records(_prev_clk_path,
                                                 keep_gps=None)
    except Exception:
        _prev_clk_recs = None

    # Candidates download to staging names and are promoted only after
    # coverage verification — a listed-but-bad product must never
    # overwrite the currently-good orbit_cod.sp3 / clock.clk.
    sp3_path = None
    for sp3_name in sp3_candidates[:5]:
        _cand = _download(sp3_name, "orbit_cod_cand.sp3",
                          gps_week=download_week)
        if _cand and _verify_coverage(_cand, today):
            _cand.replace(DEST / "orbit_cod.sp3")
            sp3_path = DEST / "orbit_cod.sp3"
            # Also copy to orbit.sp3 (RTKLIB fallback name)
            (DEST / "orbit.sp3").write_bytes(sp3_path.read_bytes())
            break

    clk_path = None
    for clk_name in clk_candidates[:4]:
        _cand = _download(clk_name, "clock_cand.clk", gps_week=download_week)
        if _cand and _verify_coverage(_cand, today):
            _cand.replace(DEST / "clock.clk")
            clk_path = DEST / "clock.clk"
            _mark_clock_update(issue=clk_name, prev_recs=_prev_clk_recs)
            break

    # If every current-week GRG CLK failed, try the previous week's
    # listing too — the previous week's last GRG issues are 2-day
    # products that can still cover today at the start of a new GPS
    # week, and a file that is listed-but-undownloadable in the current
    # week may be fully staged in the previous one.
    if not clk_path:
        if not prev_week_files:
            prev_week_files = _list_products(gps_week - 1)
        for clk_name in sorted(
                (f for f in prev_week_files
                 if f.startswith("GRG0OPSULT") and "CLK.CLK" in f),
                reverse=True)[:2]:
            _cand = _download(clk_name, "clock_cand.clk",
                              gps_week=gps_week - 1)
            if _cand and _verify_coverage(_cand, today):
                _cand.replace(DEST / "clock.clk")
                clk_path = DEST / "clock.clk"
                break
    for _stale in ("orbit_cod_cand.sp3", "clock_cand.clk"):
        try:
            (DEST / _stale).unlink()
        except OSError:
            pass

    # Download the matching GRG ultra-rapid SP3 for the chosen GRG CLK
    # (same issue tag) into orbit_grg_ult.sp3.  The primary orbit_cod.sp3
    # stays the COD ULT for continuity, but a consistent single-AC
    # orbit+clock pair is now available for PPP-AR/diagnostics — mixing
    # COD orbits with GRG clocks leaves the inter-AC orbit difference in
    # the effective satellite clocks.
    try:
        for _gclk in clk_candidates[:4]:
            _gsp3 = _gclk.replace("CLK.CLK", "ORB.SP3")
            if _gsp3 == _gclk:
                continue
            _p = _download(_gsp3, "orbit_grg_ult.sp3", gps_week=download_week)
            if _p and _verify_coverage(_p, today):
                log.info("Matched GRG orbit saved: %s", _gsp3)
                break
    except Exception as e:
        log.info("GRG SP3 fetch skipped: %s", e)

    # Freshness check: if the GRG CLK is >24h old (predicted clocks have
    # degraded significantly), try extracting fresher clocks from the
    # latest IGS ultra-rapid SP3 from the current week.  The IGS ultra-rapid
    # is updated every 6h and contains clocks in the SP3 file.  When the
    # current data is in the observed part (not predicted), the clocks are
    # much more accurate than the 32h-old GRG predicted clocks.
    # The IGS clocks only cover GPS satellites (32 vs 81 for GRG), but the
    # EKF's per-satellite bias calibration removes the clock prediction
    # errors, so freshness matters more than satellite count.
    #
    # NOTE: This check runs for ALL GRG downloads (current AND previous
    # week).  The current week's GRG can also be stale — e.g., at the
    # start of a new GPS week, the only available GRG may be from 2 days
    # ago (36h+ old), and without this merge the EKF diverges because the
    # GPS predicted clocks are 14h+ into the prediction.
    if clk_path:
        _latest_grg_name = clk_candidates[0] if clk_candidates else ""
        try:
            # The GRG ULT filename epoch marks the START of the 2-day
            # product (24h observed + 24h predicted), so filename age
            # overstates staleness by up to a day.  The obs/pred boundary
            # is the coverage midpoint; prediction depth = now - midpoint.
            # Merge only when a candidate's own midpoint is later by a
            # real margin — otherwise the swap trades equal-depth 5-min
            # GRG clocks for coarser 15-min SP3-extract clocks, a
            # measured accuracy regression.
            _grg_mid = _clk_mid_epoch(clk_path)
            _depth_h = ((dt.datetime.now() - _grg_mid).total_seconds()
                        / 3600.0 if _grg_mid else float("inf"))
            if _depth_h > 0.0:
                log.info("GRG CLK is %.1fh into its predicted half — "
                         "trying fresher ultra-rapid clocks from current "
                         "week %d", _depth_h, gps_week)
                _cur_files = _list_products(gps_week)
                # CDDIS lists IGS ultra-rapid as IGS0OPSULT_*_ORB.SP3.gz
                # (and COD/EMR/ESA/GFZ/JGX/SIO equivalents — often newer
                # issues than the igu combination); the Scripps mirror
                # lists the legacy igu<wk><d>_<hh>.sp3.Z names.
                _igu_candidates = _ult_sp3_candidates(_cur_files)
                _igu_week = gps_week
                # The IGS combined igu on Scripps is often hours fresher
                # than any *0OPSULT long-name product on CDDIS — merge
                # both listings and re-sort the pool by issue epoch.
                _scr_files = _list_scripps_products(gps_week)
                if _scr_files:
                    _merged_pool = _ult_sp3_candidates(
                        list(dict.fromkeys(_cur_files + _scr_files)))
                    if _merged_pool:
                        if (_merged_pool[0] != (_igu_candidates[0]
                               if _igu_candidates else None)):
                            log.info("Scripps listing adds fresher ULT "
                                     "candidate: %s", _merged_pool[0])
                        _igu_candidates = _merged_pool
                # At the start of a new GPS week the current-week listing
                # may be empty (404 on all mirrors) or not have igu files
                # yet.  The previous week's LAST igu issue still covers
                # today (24h products) and its clocks are much fresher
                # than a >24h-old GRG prediction.
                if not _igu_candidates and prev_week_files:
                    _igu_candidates = _ult_sp3_candidates(prev_week_files)
                    _igu_week = gps_week - 1
                    if _igu_candidates:
                        log.info("No igu in week %d listing — falling back "
                                 "to previous week %d (%s)", gps_week,
                                 _igu_week, _igu_candidates[0])
                for _igu_name in _igu_candidates[:3]:
                    _igu_path = _download(_igu_name, "igu_fresh.sp3",
                                         gps_week=_igu_week)
                    if _igu_path and _verify_coverage(_igu_path, today):
                        _igs_clk_path = DEST / "clock_igs.clk"
                        if _extract_clk_from_sp3(_igu_path, _igs_clk_path):
                            # Require a real freshness gain: the
                            # candidate's obs/pred boundary must be
                            # >=3h later than the GRG's, else the merge
                            # trades equal-depth 5-min GRG clocks for
                            # coarser 15-min SP3-extract clocks.  Parse
                            # the boundary from the SP3 header comment
                            # (COD ULT is 15/33h, not 24/24); coverage
                            # midpoint is the fallback.
                            _cand_mid = (_sp3_obs_boundary(_igu_path)
                                         or _clk_mid_epoch(_igs_clk_path))
                            _gain_h = ((_cand_mid - _grg_mid)
                                       .total_seconds() / 3600.0
                                       if (_cand_mid and _grg_mid)
                                       else (0.0 if _cand_mid is None
                                             else float("inf")))
                            if _gain_h < 3.0:
                                log.info("ULT candidate %s obs boundary "
                                         "only %+.1fh vs GRG — keeping "
                                         "GRG clocks", _igu_name, _gain_h)
                                _igu_path = None
                                continue
                            # Merge: fresh SP3 clocks replace the stale GRG
                            # records for every sat the fresh file covers
                            _grg_clk_path = clk_path
                            _merged_path = DEST / "clock.clk"
                            _merge_clocks(_grg_clk_path, _igs_clk_path, _merged_path)
                            log.info("Using merged fresh+GRG clocks — "
                                     "fresh issue: %s (obs boundary "
                                     "%+.1fh vs GRG)", _igu_name, _gain_h)
                            _mark_clock_update(issue=_igu_name,
                                               merged_with=_latest_grg_name,
                                               prev_recs=_prev_clk_recs)
                            break
                    _igu_path = None
        except Exception as e:
            log.info("Freshness check failed: %s", e)

    # Fallback: extract clocks from the downloaded SP3 file if no CLK
    # file was available (e.g., IGS ultra-rapid SP3 only, no GRG CLK).
    # Do NOT blindly overwrite an existing clock.clk — the SP3 extract
    # may cover fewer constellations than the file already present
    # (COD ULT carries G+R+E but no BeiDou; GRG carries C+G+E).  Merge
    # the fresh extract into the existing file so uncovered sats keep
    # their precise clocks; overwrite only when nothing usable exists.
    if not clk_path and sp3_path:
        log.info("No dedicated CLK file available — extracting clocks from SP3")
        _igs = DEST / "clock_igs.clk"
        if _extract_clk_from_sp3(sp3_path, _igs):
            _old = DEST / "clock.clk"
            if _old.exists() and _verify_coverage(_old, today):
                _new_max = _clk_max_epoch(_igs)
                _old_max = _clk_max_epoch(_old)
                if (_new_max is not None and _old_max is not None
                        and (_old_max - _new_max).total_seconds() > 3600):
                    # Existing file is fresher than the SP3 fallback —
                    # keep it rather than regressing.
                    clk_path = _old
                    log.info("Existing clock.clk is fresher than the SP3 "
                             "extract — keeping it")
                else:
                    _tmp = DEST / "clock_merged.clk"
                    if _merge_clocks(_old, _igs, _tmp):
                        _tmp.replace(_old)
                        clk_path = _old
                        log.info("Merged fresh SP3 clocks into existing "
                                 "clock.clk")
                    else:
                        _igs.replace(_old)
                        clk_path = _old
            else:
                _igs.replace(_old)
                clk_path = _old
            if clk_path and not _verify_coverage(clk_path, today):
                clk_path = None

    if sp3_path and clk_path:
        log.info("Products updated successfully for %s", today)
        rc = 0
    else:
        log.warning("Partial update: SP3=%s CLK=%s",
                    "OK" if sp3_path else "MISSING",
                    "OK" if clk_path else "MISSING")
        rc = 1 if not sp3_path else 0

    # Daily CODE rapid IAR products (never fatal — best-effort add-on).
    try:
        _fetch_rap_iar(files, prev_week_files, gps_week)
    except Exception as e:
        log.info("IAR product fetch skipped: %s", e)

    # Daily BRDC merged broadcast nav (never fatal).  Without this the
    # pipeline silently falls back to month-old brdc files for every
    # satellite lacking a precise ephemeris.
    try:
        _fetch_brdc(today)
    except Exception as e:
        log.info("BRDC fetch skipped: %s", e)

    return rc


def _fetch_brdc(today: dt.date) -> None:
    """Fetch the IGS merged daily broadcast nav file(s) into data/rinex/nav.

    CDDIS publishes ``BRDC00IGS_R_<YYYY><DOY>0000_01D_MN.rnx.gz`` under
    ``/archive/gnss/data/daily/<YYYY>/<DOY>/<YY>n/``.  Today's file may
    not be complete yet — also ensure yesterday's file exists so windows
    crossing UTC midnight and same-day windows both have fresh nav.
    """
    nav_dir = PROJECT_ROOT / "data" / "rinex" / "nav"
    nav_dir.mkdir(parents=True, exist_ok=True)
    creds = _auth()
    s = requests.Session()
    if creds:
        s.auth = creds
    for day in (today, today - dt.timedelta(days=1)):
        doy = _doy(day)
        yy = day.year % 100
        short = f"brdc{doy:03d}0.{yy:02d}n"
        long = f"BRDC00IGS_R_{day.year}{doy:03d}0000_01D_MN.rnx"
        # Already have this day? (either naming convention)
        if (nav_dir / short).exists() or (nav_dir / long).exists():
            continue
        for name in (short + ".gz", long + ".gz"):
            url = (f"{CDDIS_DAILY}/{day.year}/{doy:03d}/{yy:02d}n/{name}")
            try:
                r = s.get(url, allow_redirects=True, timeout=120)
                if r.status_code != 200 or "html" in r.headers.get(
                        "content-type", "").lower():
                    continue
                raw = gzip.decompress(r.content)
                out = nav_dir / name[:-3]
                out.write_bytes(raw)
                log.info("BRDC nav fetched: %s (%d bytes)",
                         out.name, len(raw))
                break
            except Exception as e:
                log.debug("BRDC fetch %s failed: %s", url, e)


def _maybe_iar_replay(doy_tag: str) -> None:
    """Run the gated PPP-AR replay once per fresh RAP data day.

    The replay picks the newest obs_archive window inside the RAP day
    and scores free+pinned rnx2rtkp_pppar solves against the truth
    anchor, writing pppar_iar_latest.json.  It was a manual step for
    its first two weeks (last run 2026-10-14): the ~16 h RAP
    publication lag outlasted obs_archive's ~8 h retention, so an
    in-day window almost never survived — retention is now ~20 h.
    Skips when the latest result already covers this day; the every-3 h
    fetch cadence then retries automatically on the next tick if
    archive coverage had not yet caught up.
    """
    res_json = PROJECT_ROOT / "data" / "ppp_live" / "pppar_iar_latest.json"
    try:
        cur = json.loads(res_json.read_text())
        if str(cur.get("rap_day")) == str(doy_tag):
            return
    except Exception:
        pass
    script = PROJECT_ROOT / "scripts" / "live" / "pppar_iar_replay.py"
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--day", str(doy_tag)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=1200)
        for ln in (r.stdout or "").strip().splitlines()[-4:]:
            log.info("IAR replay: %s", ln)
        if r.returncode:
            for ln in (r.stderr or "").strip().splitlines()[-3:]:
                log.info("IAR replay rc=%d: %s", r.returncode, ln)
    except Exception as e:
        log.debug("IAR replay failed: %s", e)


def _fetch_rap_iar(files: list[str], prev_week_files: list[str],
                   gps_week: int) -> None:
    """Fetch the latest daily CODE rapid IAR product set.

    COD0OPSRAP_<year><doy>00000_01D_{01D_OSB.BIA,05M_ORB.SP3,30S_CLK.CLK}.gz
    publishes ~daily with ~16h lag and is an ambiguity-recoverable (IAR)
    product line: the clocks carry satellite phase biases (E clock
    reference = C1C/C5Q, matching the LC29H rover codes exactly) and the
    OSB.BIA carries per-satellite code+phase OSBs for G/R/E.  Landing the
    latest set in precise/cod_rap_* gives the offline gated PPP-AR path
    (PPPAR_AR_DELAY_S/PPPAR_AR_POS_STD in the custom rnx2rtkp_pppar) real
    products to fix against for replay arcs, and lets the live PPP-AR arm
    attempt true integer fixing on day-old windows.
    """
    src_marker = DEST / "cod_rap_src.txt"

    # Find the newest DOY with a complete ORB.SP3 + CLK.CLK + OSB.BIA set.
    # A RAP file lives in the GPS-week directory of its DATA day, so
    # recent days may be in the previous week's listing.
    by_doy: dict[str, dict[str, tuple[str, int]]] = {}
    for lst, wk in ((files, gps_week), (prev_week_files, gps_week - 1)):
        for f in lst or []:
            m = re.match(r"COD0OPSRAP_(\d{7})\d{4}_01D_01D_OSB\.BIA\.gz", f)
            if m:
                by_doy.setdefault(m.group(1), {})["bia"] = (f, wk)
                continue
            m = re.match(r"COD0OPSRAP_(\d{7})\d{4}_01D_05M_ORB\.SP3\.gz", f)
            if m:
                by_doy.setdefault(m.group(1), {})["sp3"] = (f, wk)
                continue
            m = re.match(r"COD0OPSRAP_(\d{7})\d{4}_01D_30S_CLK\.CLK\.gz", f)
            if m:
                by_doy.setdefault(m.group(1), {})["clk"] = (f, wk)
                continue
    complete = sorted(d for d, parts in by_doy.items()
                      if {"bia", "sp3", "clk"} <= set(parts))
    if not complete:
        log.info("IAR: no complete COD0OPSRAP set found in listings")
        return
    doy_tag = complete[-1]

    if src_marker.exists() and src_marker.read_text().strip() == doy_tag \
            and all((DEST / n).exists() for n in
                    ("cod_rap_orbit.sp3", "cod_rap_clock.clk",
                     "cod_rap_osb.bia")):
        log.info("IAR: day %s set already present", doy_tag)
        _maybe_iar_replay(doy_tag)
        return

    parts = by_doy[doy_tag]
    ok = True
    for key, dest_name in (("sp3", "cod_rap_orbit.sp3"),
                           ("clk", "cod_rap_clock.clk"),
                           ("bia", "cod_rap_osb.bia")):
        fname, wk = parts[key]
        if _download(fname, dest_name, gps_week=wk) is None:
            ok = False
    if ok:
        src_marker.write_text(doy_tag)
        log.info("IAR: fetched COD0OPSRAP day %s -> cod_rap_* "
                 "(ambiguity-aligned rapid products)", doy_tag)
        _maybe_iar_replay(doy_tag)


if __name__ == "__main__":
    sys.exit(main())
