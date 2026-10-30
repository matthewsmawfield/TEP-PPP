#!/usr/bin/env python3
"""
Official IGS station coordinates from the daily SINEX combination solution.

The daily IGS product ``IGS0OPSSNX_{yyyy}{ddd}0000_01D_01D_SOL.SNX.gz`` carries
the officially-estimated ECEF position of ~1,490 IGS stations FOR THAT DAY
(reference epoch = that day, noon).  This is the surveyed-truth coordinate the
end-to-end positioning gates (P2/P3) score against -- not the day's own SPP
mean, which is circular.

Public API:
    station_xyz(code4, year, doy)  -> np.array([x, y, z]) metres, or None
    station_sigma(code4, year, doy) -> np.array([sx, sy, sz]) metres, or None
    available(year, doy)           -> bool  (is the daily SINEX on disk / fetchable)
Coordinates are cached in memory and on disk (data/rinex/snx/).
"""
from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
SNX_DIR = _ROOT / "data" / "rinex" / "snx"
_PARSED_DIR = SNX_DIR / "parsed"
_MEM: dict = {}


def _gps_week(year: int, doy: int) -> int:
    return (date(year, 1, 1).toordinal() + doy - 1 - date(1980, 1, 6).toordinal()) // 7


def _snx_name(year: int, doy: int) -> str:
    return f"IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_SOL.SNX.gz"


def _snx_path(year: int, doy: int) -> Path:
    return SNX_DIR / _snx_name(year, doy)


def download(year: int, doy: int, session=None) -> Path | None:
    """Fetch the daily IGS SINEX for (year, doy) into data/rinex/snx/."""
    out = _snx_path(year, doy)
    if out.exists() and out.stat().st_size > 500_000:
        return out
    SNX_DIR.mkdir(parents=True, exist_ok=True)
    if session is None:
        import requests
        from scripts.steps.step_01_acquire_data import _get_auth
        session = requests.Session()
        session.auth = _get_auth()
    from scripts.steps.step_01_acquire_data import _download_with_retry
    wk = _gps_week(year, doy)
    for url in (
        f"https://cddis.nasa.gov/archive/gnss/products/{wk}/{_snx_name(year, doy)}",
        # weekly fallback (7-day combined), same block layout
        f"https://cddis.nasa.gov/archive/gnss/products/{wk}/"
        f"IGS0OPSSNX_{year}{doy:03d}0000_07D_07D_SOL.SNX.gz",
    ):
        content = _download_with_retry(session, url, timeout_s=120, label="SNX")
        if content and len(content) > 500_000:
            out.write_bytes(content)
            return out
    return None


def _parse(path: Path) -> dict:
    """Parse SOLUTION/ESTIMATE -> {code4: {"xyz": [x,y,z], "sig": [sx,sy,sz],
    "epoch": "yy:doy:sec"}}.  Keeps the first (or only) solution per station."""
    op = gzip.open if path.suffix == ".gz" else open
    coords: dict = {}
    in_blk = False
    with op(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("+SOLUTION/ESTIMATE"):
                in_blk = True
                continue
            if line.startswith("-SOLUTION/ESTIMATE"):
                break
            if not in_blk or line.startswith("*") or len(line) < 60:
                continue
            p = line.split()
            # INDEX TYPE CODE PT SOLN REF_EPOCH UNIT S VALUE STD
            if len(p) < 10:
                continue
            typ, code = p[1], p[2].upper()
            try:
                val, std = float(p[8]), float(p[9])
            except ValueError:
                continue
            rec = coords.setdefault(code, {"xyz": [np.nan] * 3,
                                          "sig": [np.nan] * 3, "epoch": p[5]})
            if typ == "STAX":
                rec["xyz"][0], rec["sig"][0] = val, std
            elif typ == "STAY":
                rec["xyz"][1], rec["sig"][1] = val, std
            elif typ == "STAZ":
                rec["xyz"][2], rec["sig"][2] = val, std
    return {c: v for c, v in coords.items() if np.all(np.isfinite(v["xyz"]))}


def _load_day(year: int, doy: int) -> dict:
    key = (year, doy)
    if key in _MEM:
        return _MEM[key]
    _PARSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = _PARSED_DIR / f"{year}{doy:03d}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        _MEM[key] = d
        return d
    path = _snx_path(year, doy)
    if not path.exists():
        path = download(year, doy)
    if path is None or not path.exists():
        _MEM[key] = {}
        return {}
    d = _parse(path)
    try:
        cache.write_text(json.dumps(d))
    except Exception:
        pass
    _MEM[key] = d
    return d


def station_xyz(code4: str, year: int, doy: int):
    """Official ECEF position (m) of an IGS station for that day, or None.

    Falls back to the nearest available daily SINEX within +/- 7 days (the
    solution moves < 1 mm over a week for a stable station)."""
    code4 = code4.upper()[:4]
    d = _load_day(year, doy)
    if code4 in d:
        return np.asarray(d[code4]["xyz"], float)
    for off in (1, -1, 2, -2, 3, -3, 7, -7):
        dd = doy + off
        if 1 <= dd <= 366:
            d2 = _load_day(year, dd)
            if code4 in d2:
                return np.asarray(d2[code4]["xyz"], float)
    return None


def station_sigma(code4: str, year: int, doy: int):
    d = _load_day(year, doy)
    r = d.get(code4.upper()[:4])
    return np.asarray(r["sig"], float) if r else None


def available(year: int, doy: int) -> bool:
    return bool(_load_day(year, doy))
