#!/usr/bin/env python3
"""Re-download corrupt RINEX obs files from CDDIS and atomically replace them.

Context: a batch of files under data/rinex/obs/<year>/<doy>/ was saved as
~11 KB HTML error pages (Earthdata login redirect bodies) or as truncated
gzip archives.  Loaders skip them gracefully, but coverage on the affected
days (2024:097-101) is reduced.

For every *.crx.gz under the obs tree that fails a full gzip decompress,
this script fetches the canonical CDDIS daily URL

    {CDDIS_BASE}/{year}/{doy}/{yy}d/{filename}

verifies the downloaded bytes as gzip, and atomically replaces the corrupt
file.  Earthdata credentials come from ~/.netrc (urs.earthdata.nasa.gov) or
the CDDIS_USER / CDDIS_PASS environment variables; requests applies netrc
auth per-host across the OAuth redirect chain.

Usage:
    .venv/bin/python scripts/utils/redownload_corrupt_rinex.py [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import sys
import tempfile
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from pathlib import Path

import requests
import threading

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "data" / "rinex" / "obs"
CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"
WORKERS = 4
MAX_RETRIES = 6


def is_corrupt(p: Path) -> bool:
    try:
        with gzip.open(p, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return False
    except Exception:
        return True


def download(session: requests.Session, url: str) -> bytes | str | None:
    """Return content bytes, "ABSENT" on a definitive 4xx, None on
    exhausted transient retries."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=180, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            if 400 <= r.status_code < 500:
                return "ABSENT"
        except requests.RequestException:
            pass
        time.sleep(delay + random.random() * 2)
        delay = min(delay * 2, 60)
    return None


_tls = threading.local()


def repair_one(path: Path) -> str:
    year = path.parts[-3]
    doy = path.parts[-2]
    yy = year[-2:]
    url = f"{CDDIS_BASE}/{year}/{doy}/{yy}d/{path.name}"
    # Per-worker persistent session: after the first successful Earthdata
    # OAuth exchange CDDIS sets a session cookie, so later downloads skip
    # the (TLS-flaky) redirect chain entirely.
    session = getattr(_tls, "session", None)
    if session is None:
        session = _tls.session = requests.Session()
    data = download(session, url)
    if data == "ABSENT":
        return f"ABSENT {path.name}"
    if data is None:
        return f"FAIL {path.name}"
    try:
        with gzip.GzipFile(fileobj=__import__("io").BytesIO(data)) as fh:
            while fh.read(1 << 20):
                pass
    except Exception:
        return f"BADGZIP {path.name}"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return f"OK {path.name} ({len(data)} bytes)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list corrupt files without downloading")
    args = ap.parse_args()

    print(f"Scanning {OBS} for corrupt gzip members...", flush=True)
    files = sorted(OBS.rglob("*.crx.gz"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        flags = list(ex.map(is_corrupt, files, chunksize=32))
    corrupt = [p for p, bad in zip(files, flags) if bad]
    print(f"{len(corrupt)} corrupt files", flush=True)
    if args.dry_run or not corrupt:
        for p in corrupt:
            print(" ", p.relative_to(OBS))
        return 0

    done = ok = absent = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(repair_one, p): p for p in corrupt}
        for fut in as_completed(futs):
            msg = fut.result()
            done += 1
            ok += msg.startswith("OK")
            absent += msg.startswith("ABSENT")
            failed += msg.startswith(("FAIL", "BADGZIP"))
            print(f"[{done}/{len(corrupt)}] {msg}", flush=True)
    dt = time.time() - t0
    print(f"Done in {dt:.0f}s: {ok} repaired, {absent} confirmed absent "
          f"from CDDIS, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
