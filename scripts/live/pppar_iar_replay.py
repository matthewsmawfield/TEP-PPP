#!/usr/bin/env python3
"""Lagged gated PPP-AR replay on archived obs windows covered by COD0OPSRAP.

The daily CODE rapid product set (fetched by fetch_precise_products.py into
data/ppp_live/precise/cod_rap_{orbit.sp3,clock.clk,osb.bia}) is
ambiguity-recoverable: its clocks fold satellite phase biases in (GAL clock
reference = C1C/C5Q, matching the LC29H rover codes exactly), so Galileo
E1/E5a narrow-lane ambiguities resolve to integers.  It publishes ~16 h
after the data day, so this replay runs on lagged archive windows rather
than the live stream.

For the newest obs_archive window that falls inside the RAP data day this
script:
  1. derives an L1/L5 Melbourne-Wuebbena WL-bias clock from the obs data
     itself (estimate_wl_bias) and merges it into the RAP clock file,
  2. runs the custom RTKLIB pppar build twice on it:
     (a) FREE  -p7 PPP-static with the NL convergence gate
               (PPPAR_AR_DELAY_S / PPPAR_AR_POS_STD) -> honest fix rate
               plus truth-referenced ENU error,
     (b) PINNED -p8 PPP-fixed with the rover pinned at the measured
               antenna (PPPAR_FIX_ECEF = truth_ref.json external anchor)
               -> pure integer-recovery validation: fixes only survive
               here if the ambiguities are truly integer at the right
               position,
  3. writes data/ppp_live/pppar_iar_latest.json and keeps the .pos
     files under data/ppp_live/pppar_iar/ (8 kept).

Honesty: a Q=1 count is reported alongside its truth-referenced error so
"integer-fixed" never silently implies "accurate" — the free datum is
receiver-domain limited (~8 m on this LC29H) while the pinned run
validates the integer recovery itself.

Usage:
  .venv/bin/python3 scripts/live/pppar_iar_replay.py [--day YYYYDDD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PRECISE = ROOT / "data" / "ppp_live" / "precise"
OBS_ARCHIVE = ROOT / "data" / "ppp_live" / "obs_archive"
RAW_DIRS = (ROOT / "data" / "ppp_live" / "raw_archive",
            ROOT / "data" / "ppp_live")
OUT_DIR = ROOT / "data" / "ppp_live" / "pppar_iar"
RESULT_JSON = ROOT / "data" / "ppp_live" / "pppar_iar_latest.json"
ATX = ROOT / "RTKLIB_src" / "data" / "igs14.atx"
PPPAR = ROOT / "RTKLIB_src" / "build" / "rnx2rtkp_pppar"
# The repo build is a Linux aarch64 binary (built on the Pi).  On other
# hosts use a locally installed build (~/.local/bin on the Mac) — but only
# if it carries the custom -A/PPPAR_* hooks (the plain demo5 builds don't).
import platform as _platform
if _platform.system() != "Linux" or _platform.machine() not in ("aarch64", "arm64"):
    _alt_pppar = Path.home() / ".local" / "bin" / "rnx2rtkp_pppar"
    if _alt_pppar.exists() and b"PPPAR_NAVSYS" in _alt_pppar.read_bytes():
        PPPAR = _alt_pppar

WINDOW_S = 14400.0          # obs_archive windows are ~4 h captures
KEEP_POS = 8


def _truth_ecef() -> tuple[float, float, float] | None:
    try:
        d = json.loads((ROOT / "data" / "ppp_live" / "truth_ref.json")
                       .read_text())
        e = d.get("truth_ecef")
        if e and len(e) == 3 and all(map(math.isfinite, e)):
            return float(e[0]), float(e[1]), float(e[2])
    except Exception:
        pass
    return None


def _ecef_to_enu(dx: float, dy: float, dz: float,
                 ref: tuple[float, float, float]) -> tuple[float, float, float]:
    a = 6378137.0
    e2 = 6.69437999014e-3
    x, y, z = ref
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    lon = math.atan2(y, x)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def _llh_to_ecef(lat_deg: float, lon_deg: float, h: float):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s = math.sin(lat)
    n = a / math.sqrt(1 - e2 * s * s)
    x = (n + h) * math.cos(lat) * math.cos(lon)
    y = (n + h) * math.cos(lat) * math.sin(lon)
    z = (n * (1 - e2) + h) * s
    return x, y, z


def _score_pos(pos_path: Path, truth: tuple[float, float, float]):
    """Return {n, n_fixed, fix_rate, med_e/n/u, err3d_m, float_n}."""
    rows = []
    for ln in pos_path.read_text().splitlines():
        if ln.startswith("%"):
            continue
        p = ln.split()
        if len(p) < 6:
            continue
        try:
            rows.append((p[5], float(p[2]), float(p[3]), float(p[4])))
        except ValueError:
            continue
    out = {"n": len(rows)}
    fixed = [(q, la, lo, h) for q, la, lo, h in rows if q == "1"]
    float_ = [(q, la, lo, h) for q, la, lo, h in rows if q != "1"]
    out["n_fixed"] = len(fixed)
    out["fix_rate"] = round(len(fixed) / len(rows), 4) if rows else 0.0
    out["float_n"] = len(float_)

    def _med_enu(sel):
        if not sel:
            return None
        enus = []
        for _, la, lo, h in sel:
            x, y, z = _llh_to_ecef(la, lo, h)
            enus.append(_ecef_to_enu(x - truth[0], y - truth[1],
                                     z - truth[2], truth))
        enus.sort(key=lambda v: math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
        me, mn, mu = enus[len(enus) // 2]
        return {"med_e": round(me, 3), "med_n": round(mn, 3),
                "med_u": round(mu, 3),
                "err3d_m": round(math.sqrt(me * me + mn * mn + mu * mu), 3)}

    out["fixed_enu"] = _med_enu(fixed)
    out["float_enu"] = _med_enu(float_)
    return out


def _obs_epoch_s(line: str) -> float | None:
    p = line.split()
    try:
        y, mo, d = int(p[1]), int(p[2]), int(p[3])
        hh, mm = int(p[4]), int(p[5])
        ss = float(p[6])
    except (ValueError, IndexError):
        return None
    t = dt.datetime(y, mo, d, hh, mm, int(ss),
                    int((ss - int(ss)) * 1e6), tzinfo=dt.timezone.utc)
    return t.timestamp() + 18.0


def _obs_first_epoch_s(obs_path: Path) -> float | None:
    """First epoch time (unix s) of a RINEX obs file."""
    try:
        with open(obs_path) as f:
            for line in f:
                if line.startswith("> "):
                    return _obs_epoch_s(line)
    except Exception:
        pass
    return None


def _obs_last_epoch_s(obs_path: Path) -> float | None:
    """Last epoch time (unix s) of a RINEX obs file (tail scan)."""
    try:
        import collections as _c
        with open(obs_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200000))
            tail = f.read().decode("utf-8", errors="replace")
        last = None
        for line in tail.splitlines():
            if line.startswith("> "):
                last = _obs_epoch_s(line) or last
        return last
    except Exception:
        return None


def _convbin_tool() -> str | None:
    for name in ("convbin_demo5", "convbin"):
        p = Path.home() / ".local" / "bin" / name
        if p.exists():
            return str(p)
    return shutil.which("convbin_demo5") or shutil.which("convbin")


def _synth_window_from_raw(d0_ts: float, d1_ts: float,
                           workdir: Path) -> Path | None:
    """Fallback: synthesise a ~4 h RINEX window ending inside the RAP day
    from contiguous raw RTCM3 chunks.

    obs_archive keeps only the trailing ~24 h, so when it was cleared or
    started after the RAP day's last windows solved there is nothing in
    it to score (both day-299 attempts missed 2026-10-27).  The raw chunk
    archive spans ~10 d, so a qualifying window can always be rebuilt:
    merge the chunks covering [end-WINDOW_S, end) for the newest chunk
    end <= d1+40 s (matching the obs_archive end-inside-day rule),
    convbin to RINEX, and keep the pair only if >=3 h of epochs survive.
    """
    convbin = _convbin_tool()
    if convbin is None:
        return None
    chunks = {}
    for d in RAW_DIRS:
        if not d.exists():
            continue
        for p in d.glob("chunk_*.bin"):
            try:
                ts = float(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            chunks.setdefault(ts, p)
    if not chunks:
        return None
    end = max((ts + 300.0 for ts in chunks
               if ts + 300.0 <= d1_ts + 40.0), default=None)
    if end is None or end < d0_ts + WINDOW_S:
        return None
    sel = [p for ts, p in chunks.items()
           if end - WINDOW_S <= ts < end]
    if len(sel) < int(WINDOW_S / 300.0 * 0.75):
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    merged = workdir / "synth_merged.bin"
    with open(merged, "wb") as out:
        for p in sorted(sel, key=lambda q: q.name):
            out.write(p.read_bytes())
    conv_dir = workdir / "synth_conv"
    if conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir()
    r = subprocess.run(
        [convbin, "-r", "rtcm3", "-f", "3", "-d", str(conv_dir),
         str(merged)],
        capture_output=True, timeout=300)
    obs = next(conv_dir.glob("*.obs"), None)
    nav = next(conv_dir.glob("*.nav"), None)
    if obs is None or r.returncode != 0:
        return None
    t0, t1 = _obs_first_epoch_s(obs), _obs_last_epoch_s(obs)
    if (t0 is None or t1 is None or t1 - t0 < 3.0 * 3600.0
            or t1 < end - 900.0):
        return None
    obs_dst = workdir / f"window_{int(end)}.obs"
    nav_dst = obs_dst.with_suffix(".nav")
    shutil.move(str(obs), obs_dst)
    if nav is not None:
        shutil.move(str(nav), nav_dst)
    print(f"synthesised {obs_dst.name} from {len(sel)} raw chunks "
          f"(obs span {(t1 - t0) / 3600.0:.2f} h)")
    return obs_dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="override RAP day tag YYYYDDD")
    ap.add_argument("--obs", help="explicit obs file (skip archive search)")
    args = ap.parse_args()

    if not PPPAR.exists():
        print(f"no pppar binary at {PPPAR}", file=sys.stderr)
        return 1
    src = args.day or PRECISE.joinpath("cod_rap_src.txt").read_text().strip()
    year, doy = int(src[:4]), int(src[4:7])
    d0 = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=doy - 1)
    d1 = d0 + dt.timedelta(days=1)
    print(f"RAP day {src}: {d0} .. {d1}")

    # Convert day bounds to the obs-archive window naming axis (unix s,
    # GPST ~+18s).  Use the obs file's own first-epoch for the real check.
    candidates = []
    if args.obs:
        candidates = [Path(args.obs)]
    else:
        for obs in OBS_ARCHIVE.glob("window_*.obs"):
            try:
                end_ep = float(obs.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if not (d0.timestamp() + WINDOW_S <= end_ep
                    <= d1.timestamp() + 40.0):
                continue
            candidates.append(obs)
        candidates.sort(key=lambda p: float(p.stem.split("_")[1]))
    if not candidates:
        # obs_archive only retains ~24 h — when it was cleared or started
        # after the RAP day's last windows solved, rebuild the day's last
        # full window from the ~10 d raw RTCM3 chunk archive instead of
        # skipping the whole day (day-299 misses, 2026-10-27).
        synth = _synth_window_from_raw(d0.timestamp(), d1.timestamp(),
                                       OUT_DIR / "synth")
        if synth is None:
            print("no obs_archive window inside the RAP day and "
                  "raw-archive synthesis unavailable", file=sys.stderr)
            return 1
        candidates = [synth]
    obs_path = candidates[-1]
    nav_path = obs_path.with_suffix(".nav")
    print(f"window: {obs_path.name}")

    truth = _truth_ecef()

    # 1. WL-bias clock derived from THIS window's data.
    from scripts.lib.estimate_wl_bias import generate_wl_bias_clock_from_data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wl_clk = OUT_DIR / f"wl_{obs_path.stem}.clk"
    if not generate_wl_bias_clock_from_data(
            obs_path, PRECISE / "cod_rap_clock.clk", wl_clk):
        print("WL bias generation failed", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["PPPAR_NAVSYS"] = env.get("PPPAR_NAVSYS", "E")
    env["PPPAR_AR_DELAY_S"] = env.get("PPPAR_AR_DELAY_S", "1800")
    env["PPPAR_AR_POS_STD"] = env.get("PPPAR_AR_POS_STD", "0.5")
    if truth:
        env["PPPAR_FIX_ECEF"] = f"{truth[0]},{truth[1]},{truth[2]}"

    files = [str(obs_path)]
    if nav_path.exists():
        files.append(str(nav_path))
    files += [str(wl_clk), str(PRECISE / "cod_rap_orbit.sp3"), str(ATX)]

    result = {
        "available": True,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rap_day": src,
        "window": obs_path.name,
        "truth_ecef": list(truth) if truth else None,
    }

    for label, mode, extra in (("free", "7", {}), ("pinned", "8", {})):
        pos_out = OUT_DIR / f"{obs_path.stem}_{label}.pos"
        cmd = [str(PPPAR), "-m", "10", "-p", mode, "-f", "3", "-A",
               "-o", str(pos_out)] + files
        print("running:", " ".join(os.path.basename(c) for c in cmd[:9]),
              "...")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=3600)
        if r.returncode != 0 or not pos_out.exists():
            result[label] = {"error": (r.stderr or "")[-400:]}
            continue
        result[label] = _score_pos(pos_out, truth) if truth else {}
        if label == "pinned" and truth:
            result[label]["note"] = (
                "rover pinned at truth_ecef — position error is 0 by "
                "construction; only n_fixed/fix_rate are meaningful")

    RESULT_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

    kept = sorted(OUT_DIR.glob("*.pos"),
                  key=lambda p: p.stat().st_mtime)[-KEEP_POS:]
    for p in OUT_DIR.glob("*.pos"):
        if p not in kept:
            p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
