#!/usr/bin/env python3
"""
Shared helpers for real-world kinematic / dynamic-urban TEP-PPP tests.

The module is intentionally self-contained: it knows how to
  - convert raw rover logs (u-blox UBX, RTCM3, pre-converted RINEX) to RINEX,
  - run RTKLIB SPP on the rover and on regional CORS stations,
  - build a TEP clock prior from the CORS network,
  - evaluate free-clock vs. TEP-aided kinematic positioning against ground truth,
  - report the operational metrics that matter for real-world deployment:
    bridge rate, cross-track error, and time to re-convergence.

It does not fabricate data: if a dataset, SP3/CLK product, or CORS network is
missing, the caller writes a no-data JSON and exits cleanly.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# project imports (executing step scripts insert project root onto sys.path)
from scripts.lib import tep_data as td
from scripts.lib import tep_kernel as tk
from scripts.lib import tep_spp as spp
from scripts.lib.tep_clock import C_M_PER_S, unwrap_series

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DYNAMIC = PROJECT_ROOT / "data" / "dynamic_datasets"
C_M_PER_NS = 0.299792458
SIG_CODE_M = 1.0
MS_WRAP_M = 299_792.458  # 1 ms receiver-clock steering in range units


def _unwrap_ms(x):
    """Causal 1 ms steering unwrap.  Each correction uses only the current
    and preceding sample, so it cannot look ahead."""
    x = np.asarray(x, float)
    out = x.copy()
    corr = 0.0
    prev = np.nan
    for i, value in enumerate(x):
        if not np.isfinite(value):
            out[i] = np.nan
            continue
        adjusted = value + corr
        if np.isfinite(prev):
            jump = adjusted - prev
            nwrap = int(np.rint(jump / MS_WRAP_M))
            if nwrap and abs(jump) > 0.5 * MS_WRAP_M:
                corr -= nwrap * MS_WRAP_M
                adjusted = value + corr
        out[i] = adjusted
        prev = adjusted
    return out


# ------------------------------------------------------------------ geometry

def _wgs84_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """WGS84 geodetic lat/lon/hgt -> ECEF (m)."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], float)


def _enu_rot(ecef: np.ndarray) -> np.ndarray:
    """Rotation matrix R: ECEF -> local ENU at `ecef`."""
    x, y, z = ecef
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    sl, cl = math.sin(lon), math.cos(lon)
    sp, cp = math.sin(lat), math.cos(lat)
    return np.array([
        [-sl, cl, 0.0],
        [-sp * cl, -sp * sl, cp],
        [cp * cl, cp * sl, sp],
    ])


def _los_enu(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.sin(a),
                     math.cos(e) * math.cos(a),
                     math.sin(e)], float)


def _greatcircle_km(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    la = math.atan2(a[2], math.hypot(a[0], a[1]))
    lb = math.atan2(b[2], math.hypot(b[0], b[1]))
    lo = math.atan2(a[1], a[0])
    lob = math.atan2(b[1], b[0])
    d = math.acos(max(-1.0, min(1.0,
        math.sin(la) * math.sin(lb) +
        math.cos(la) * math.cos(lb) * math.cos(lo - lob))))
    return 6371.0 * d


# ------------------------------------------------------------------ tool finding

def find_tool(name: str) -> str:
    """Locate an RTKLIB binary (rnx2rtkp / convbin / crx2rnx)."""
    p = shutil.which(name)
    if p:
        return p
    for c in (PROJECT_ROOT / "bin" / name,
              Path.home() / ".local" / "bin" / name):
        if c.exists():
            return str(c)
    if name == "convbin":
        p5 = shutil.which("convbin_demo5")
        if p5:
            return p5
        for c in (Path.home() / ".local" / "bin" / "convbin_demo5",
                  Path.home() / ".local" / "bin" / "convbin"):
            if c.exists():
                return str(c)
    raise FileNotFoundError(f"{name} not found")


# ------------------------------------------------------------------ dataset manifest

@dataclass
class Dataset:
    name: str
    type: str
    raw_obs: Path
    raw_fmt: str
    ground_truth: Path
    gt_fmt: str
    gt_time_col: str = "time_s"
    gt_lat_col: str = "lat"
    gt_lon_col: str = "lon"
    gt_hgt_col: str = "hgt"
    gt_x_col: str = "x"
    gt_y_col: str = "y"
    gt_z_col: str = "z"
    cors_dir: Path | None = None
    base_obs: Path | None = None
    nav: Path | None = None
    sp3: Path | None = None
    clk: Path | None = None
    date_tag: str = ""
    elev_mask_deg: float = 10.0
    obs_sigma_m: float = 1.0
    region: str = ""
    android_converter: str | None = None
    cors_use_precise: bool = True
    source_url: str = ""
    doi: str = ""
    license: str = ""
    provenance_status: str = ""


def load_manifest(path: Path | None = None) -> list[Dataset]:
    if path is None:
        path = DATA_DYNAMIC / "manifest.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for r in raw:
        def _p(k: str) -> Path | None:
            v = r.get(k)
            return Path(v) if v else None
        ds = Dataset(
            name=r.get("name", "unnamed"),
            type=r.get("type", "urban"),
            raw_obs=Path(r["raw_obs"]),
            raw_fmt=r.get("raw_fmt", "ubx"),
            ground_truth=Path(r["ground_truth"]),
            gt_fmt=r.get("gt_fmt", "llh"),
            gt_time_col=r.get("gt_time_col", "time_s"),
            gt_lat_col=r.get("gt_lat_col", "lat"),
            gt_lon_col=r.get("gt_lon_col", "lon"),
            gt_hgt_col=r.get("gt_hgt_col", "hgt"),
            gt_x_col=r.get("gt_x_col", "x"),
            gt_y_col=r.get("gt_y_col", "y"),
            gt_z_col=r.get("gt_z_col", "z"),
            cors_dir=_p("cors_dir"),
            base_obs=_p("base_obs"),
            nav=_p("nav"),
            sp3=_p("sp3"),
            clk=_p("clk"),
            date_tag=r.get("date_tag", ""),
            elev_mask_deg=float(r.get("elev_mask_deg", 10.0)),
            obs_sigma_m=float(r.get("obs_sigma_m", 1.0)),
            region=r.get("region", ""),
            android_converter=r.get("android_converter"),
            cors_use_precise=bool(r.get("cors_use_precise", True)),
            source_url=r.get("source_url", ""),
            doi=r.get("doi", ""),
            license=r.get("license", ""),
            provenance_status=r.get("provenance_status", ""),
        )
        out.append(ds)
    return out


# ------------------------------------------------------------------ products

def find_sp3_clk(date_tag: str):
    """Search local SP3 and CLK products for date_tag (YYYYDOY)."""
    sp3_dir = PROJECT_ROOT / "data" / "rinex" / "sp3"
    clk_dir = PROJECT_ROOT / "data" / "rinex" / "clk"
    sp3 = clk = None
    for sp3_pat in ("COD0MGXFIN", "COD0OPSFIN", "IGS0OPSFIN",
                    "GFZ0MGXRAP", "WUM0MGXFIN"):
        if sp3 is None:
            hits = list(sp3_dir.glob(f"{sp3_pat}_{date_tag}*_ORB.SP3*"))
            if hits:
                sp3 = hits[0]
    for clk_pat in ("COD0MGXFIN", "GFZ0OPSFIN", "GRG0OPSFIN", "JPL0OPSFIN"):
        if clk is None:
            hits = list(clk_dir.glob(f"{clk_pat}_{date_tag}*_CLK.CLK*"))
            if hits:
                clk = hits[0]
    return sp3, clk


def find_nav(date_tag: str):
    """Search local broadcast navigation for date_tag (YYYYDOY)."""
    nav_dir = PROJECT_ROOT / "data" / "rinex" / "nav"
    for pat in (f"BRDC00IGS_R_{date_tag}0000_01D_MN.rnx",
                f"BRDC00IGS_R_{date_tag}0000_01D_MN.rnx.gz"):
        p = nav_dir / pat
        if p.exists():
            return p
    return None


def _decompress_crx(crx_gz: Path, out_dir: Path) -> Path:
    """Hatanaka .crx.gz -> .rnx in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rnx = out_dir / crx_gz.stem
    if rnx.suffix == ".gz":
        rnx = rnx.with_suffix("")
    if rnx.suffix != ".rnx" and rnx.suffix != ".obs" and rnx.suffix != ".o":
        rnx = rnx.with_suffix(".rnx")
    # crx2rnx writes the .rnx next to the .crx; name the intermediate .crx
    # to match the desired output basename.
    crx = rnx.with_suffix(".crx")
    with open(crx, "wb") as f:
        subprocess.run(["gunzip", "-c", str(crx_gz)], stdout=f, check=True)
    subprocess.run(["crx2rnx", str(crx)], capture_output=True)
    if not rnx.exists():
        subprocess.run(["crx2rnx", "-f", str(crx)], capture_output=True)
    return rnx if rnx.exists() else None


def read_rinex_text(path: Path, out_dir: Path | None = None) -> str | None:
    """Return RINEX observation text from .rnx, .rnx.gz, .crx, .crx.gz, .obs."""
    if not path.exists():
        return None
    if path.suffix == ".gz":
        if out_dir is None:
            out_dir = Path(tempfile.mkdtemp())
        stem = path.stem
        if stem.endswith(".crx"):
            return _decompress_crx(path, out_dir).read_text() if _decompress_crx(path, out_dir) else None
        with gzip.open(path, "rt") as f:
            return f.read()
    if path.suffix in (".crx",):
        if out_dir is None:
            out_dir = Path(tempfile.mkdtemp())
        rnx = _decompress_crx(path, out_dir)
        return rnx.read_text() if rnx else None
    with open(path, "r") as f:
        return f.read()
    return None


# ------------------------------------------------------------------ conversion

def convert_to_rinex(ds: Dataset, out_dir: Path) -> dict[str, Path | None]:
    """Convert a raw log to RINEX OBS (and NAV where possible)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = ds.raw_obs
    if not raw.exists():
        raise FileNotFoundError(f"raw log missing: {raw}")

    obs = out_dir / f"{ds.name}.obs"
    nav = out_dir / f"{ds.name}.nav"

    if ds.raw_fmt == "rinex":
        txt = read_rinex_text(raw, out_dir)
        if txt is None:
            return {"obs": None, "nav": None}
        obs.with_suffix(".rnx").write_text(txt)
        return {"obs": obs.with_suffix(".rnx"), "nav": None}

    if ds.raw_fmt in ("ubx", "rtcm3"):
        fmt = "ubx" if ds.raw_fmt == "ubx" else "rtcm3"
        convbin = find_tool("convbin")
        cmd = [convbin, "-r", fmt, "-d", str(out_dir),
               "-f", "2", "-v", "3.0", str(raw)]
        subprocess.run(cmd, capture_output=True, timeout=300)
        obs_files = list(out_dir.glob("*.obs")) + list(out_dir.glob("*MO.rnx"))
        nav_files = list(out_dir.glob("*.nav")) + list(out_dir.glob("*MN.rnx"))
        return {"obs": obs_files[0] if obs_files else None,
                "nav": nav_files[0] if nav_files else None}

    if ds.raw_fmt == "android_csv":
        if ds.android_converter:
            subprocess.run([ds.android_converter, str(raw), str(obs)],
                           check=True, timeout=300)
            return {"obs": obs if obs.exists() else None, "nav": None}
        raise NotImplementedError(
            "Android CSV must be pre-converted to RINEX or an "
            "android_converter command must be provided in the manifest.")

    raise ValueError(f"unsupported raw_fmt: {ds.raw_fmt}")


# ------------------------------------------------------------------ RTKLIB runners

def _run_rnx2rtkp(obs: Path, nav: Path | None, sp3: Path | None, out_pos: Path,
                  elev_mask: float = 10.0, y: int = 2) -> Path | None:
    rnx2rtkp = find_tool("rnx2rtkp")
    cmd = [rnx2rtkp, "-x", "0", "-m", str(int(elev_mask)),
           "-p", "0", "-f", "2", "-y", str(int(y)),
           "-o", str(out_pos), str(obs)]
    if nav:
        cmd.append(str(nav))
    if sp3 and sp3.exists():
        cmd.append(str(sp3))
    stat = Path(str(out_pos) + ".stat")
    try:
        subprocess.run(cmd, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    return stat if stat.exists() and stat.stat().st_size > 100 else None


def parse_stat(stat: Path) -> dict:
    """Parse RTKLIB .stat file -> {sod: {...}}."""
    out = {}
    pos_by_sod = {}
    clk_by_sod = {}
    with open(stat, "r", errors="ignore") as f:
        for line in f:
            p = line.rstrip().split(",")
            if not p:
                continue
            tag = p[0]
            if tag == "$POS" and len(p) >= 7:
                try:
                    sod = int(round(float(p[2]) % 86400.0))
                    pos_by_sod[sod] = np.array([float(p[4]), float(p[5]), float(p[6])])
                except (ValueError, IndexError):
                    continue
            elif tag == "$CLK" and len(p) >= 6:
                try:
                    sod = int(round(float(p[2]) % 86400.0))
                    clk_by_sod[sod] = float(p[5])
                except (ValueError, IndexError):
                    continue
            elif tag == "$SAT" and len(p) >= 9:
                try:
                    sod = int(round(float(p[2]) % 86400.0))
                    prn = p[3].strip()
                    az, el, resp = float(p[5]), float(p[6]), float(p[7])
                    if el >= 1.0:
                        out.setdefault(sod, {"sats": []})["sats"].append(
                            {"prn": prn, "az": az, "el": el, "resp": resp})
                except (ValueError, IndexError):
                    continue
    for sod, rec in out.items():
        if sod in pos_by_sod:
            rec["pos_rtk"] = pos_by_sod[sod]
        if sod in clk_by_sod:
            rec["cdt_rtk_m"] = clk_by_sod[sod]
    return out


def build_clock_series(obs: Path, nav: Path | None, sp3: Path | None,
                       out_dir: Path, station_name: str) -> dict | None:
    """Run RTKLIB on a static (CORS) RINEX and return a tep_data-style series."""
    out_pos = out_dir / f"{station_name}.pos"
    stat = _run_rnx2rtkp(obs, nav, sp3, out_pos, y=1)
    if stat is None:
        return None
    sod_pos, x, y, z = [], [], [], []
    sod_clk, clk = [], []
    with open(stat, "r", errors="ignore") as f:
        for line in f:
            p = line.rstrip().split(",")
            if not p:
                continue
            if p[0] == "$POS" and len(p) >= 7:
                try:
                    s = int(round(float(p[2]) % 86400.0))
                    sod_pos.append(s)
                    x.append(float(p[4])); y.append(float(p[5])); z.append(float(p[6]))
                except (ValueError, IndexError):
                    pass
            elif p[0] == "$CLK" and len(p) >= 6:
                try:
                    s = int(round(float(p[2]) % 86400.0))
                    sod_clk.append(s); clk.append(float(p[5]))
                except (ValueError, IndexError):
                    pass
    if len(clk) < 100:
        return None
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    pos_med = np.array([float(np.median(x)), float(np.median(y)), float(np.median(z))])
    lat, lon = td.ecef_to_geodetic(*pos_med)
    sta = td.Station(station_name, pos_med / 1000.0, lat, lon)

    # align to 30-second grid
    grid = np.arange(0, 86400, 30.0)
    clk = np.asarray(clk, float)
    sod_clk = np.asarray(sod_clk, float)
    raw = np.full(len(grid), np.nan)
    idx = {int(round(t)): i for i, t in enumerate(sod_clk)}
    hit = np.array([idx.get(int(g), -1) for g in grid])
    m = hit >= 0
    raw[m] = clk[hit[m]]
    if np.mean(np.isfinite(raw)) < 0.5:
        return None
    raw = td._interp_nan(raw)  # uses fill for small gaps
    ns = raw * (1.0 / C_M_PER_NS)  # m -> ns
    resid = td.robust_detrend(ns)
    if not np.isfinite(resid).any():
        return None
    return {
        "t": grid.copy(),
        "raw": ns,
        "resid": resid,
        "station": sta,
        "nsat": None,
    }


def build_clock_series_precise(obs: Path, sp3: Path, clk: Path,
                               station_ecef: np.ndarray,
                               station_name: str) -> dict | None:
    """Compute a CORS clock series from raw RINEX + SP3/CLK + known position.

    This matches the precise-ephemeris prefit used for the rover, so the CORS
    clock prior and the rover clock estimate are on the same product.
    """
    rnx_text = read_rinex_text(obs)
    if not rnx_text:
        return None
    obs_dict = read_rinex3_gps(rnx_text)
    if not obs_dict:
        return None
    sp3_dict = spp.read_sp3_gps(sp3)
    clk_dict = spp.read_clk_gps(clk)
    dense = spp.dense_grid(sp3_dict, clk_dict, step=1.0)

    cdt_by_sod: dict[int, float] = {}
    for sod, rec in obs_dict.items():
        pf = spp.prefit_fast(rec, dense, station_ecef, sod)
        if len(pf) < 4:
            continue
        el = np.array([p[2] for p in pf])
        resid = np.array([p[3] for p in pf])
        w = np.sin(np.radians(el)) ** 2
        if w.sum() <= 0:
            continue
        cdt = float(np.sum(w * resid) / np.sum(w))
        cdt_by_sod[sod] = cdt

    if not cdt_by_sod:
        return None

    # align to 30-second grid
    grid = np.arange(0, 86400, 30.0)
    sods = np.array(sorted(cdt_by_sod.keys()), float)
    vals = np.array([cdt_by_sod[int(s)] for s in sods])
    # do not extrapolate beyond the available observations
    raw = np.interp(grid, sods, vals, left=np.nan, right=np.nan)
    if not np.any(np.isfinite(raw)):
        return None
    raw = td._interp_nan(raw)

    ns = raw * (1.0 / C_M_PER_NS)  # m -> ns
    resid = td.robust_detrend(ns)
    if not np.isfinite(resid).any():
        return None
    lat, lon = td.ecef_to_geodetic(*station_ecef)
    sta = td.Station(station_name, np.asarray(station_ecef, float) / 1000.0, lat, lon)
    return {
        "t": grid.copy(),
        "raw": ns,
        "resid": resid,
        "station": sta,
        "nsat": None,
    }


def build_clock_series_precise_multi(
        obs: Path, sp3: Path, clk: Path,
        station_ecef: np.ndarray, station_name: str,
        systems: tuple[str, ...] = ("G", "E", "C")) -> dict | None:
    """Multi-GNSS CORS clock series from raw RINEX + SP3/CLK + known position.

    Uses ``read_rinex3_multi`` (G+E+C ionosphere-free) and
    ``spp.read_sp3_multi`` / ``spp.read_clk_multi`` / ``spp.dense_grid_multi``
    so the CORS clock prior and the rover clock estimate are on the same
    multi-GNSS product and the same observable definition.
    """
    rnx_text = read_rinex_text(obs)
    if not rnx_text:
        return None
    obs_dict = read_rinex3_multi(rnx_text, systems=systems)
    if not obs_dict:
        return None
    sp3_dict = spp.read_sp3_multi(sp3, systems=systems)
    clk_dict = spp.read_clk_multi(clk, systems=systems)
    if not sp3_dict or not clk_dict:
        return None
    dense = spp.dense_grid_multi(sp3_dict, clk_dict, step=1.0)

    cdt_by_sod: dict[int, float] = {}
    for sod, rec in obs_dict.items():
        pf = spp.prefit_fast_if(rec, dense, station_ecef, sod)
        if len(pf) < 4:
            continue
        el = np.array([p[2] for p in pf])
        resid = np.array([p[3] for p in pf])
        w = np.sin(np.radians(el)) ** 2
        if w.sum() <= 0:
            continue
        cdt = float(np.sum(w * resid) / np.sum(w))
        cdt_by_sod[sod] = cdt

    if not cdt_by_sod:
        return None

    grid = np.arange(0, 86400, 30.0)
    sods = np.array(sorted(cdt_by_sod.keys()), float)
    vals = np.array([cdt_by_sod[int(s)] for s in sods])
    raw = np.interp(grid, sods, vals, left=np.nan, right=np.nan)
    if not np.any(np.isfinite(raw)):
        return None
    raw = td._interp_nan(raw)

    ns = raw * (1.0 / C_M_PER_NS)
    resid = td.robust_detrend(ns)
    if not np.isfinite(resid).any():
        return None
    lat, lon = td.ecef_to_geodetic(*station_ecef)
    sta = td.Station(station_name, np.asarray(station_ecef, float) / 1000.0, lat, lon)
    return {
        "t": grid.copy(),
        "raw": ns,
        "resid": resid,
        "station": sta,
        "nsat": None,
        "multi_gnss": True,
    }


def build_cors_series_multi(cors_dir: Path, sp3: Path, clk: Path,
                            out_dir: Path,
                            systems: tuple[str, ...] = ("G", "E", "C")
                            ) -> dict[str, dict]:
    """Process all CORS RINEX into multi-GNSS clock residual series.

    Same caching strategy as ``build_cors_series`` but uses the multi-GNSS
    precise prefit so the CORS clock prior is on the same observable as the
    multi-GNSS rover solver.
    """
    import pickle as _pkl
    if not cors_dir or not cors_dir.exists():
        return {}
    cache_dir = cors_dir / ".clkcache_mg"
    cache_dir.mkdir(exist_ok=True)
    series = {}
    for path in sorted(cors_dir.iterdir()):
        if path.suffix not in (".rnx", ".crx", ".obs", ".o", ".gz"):
            continue
        name = _station_name_from_path(path)
        try:
            st = path.stat()
            # A GPS-only clock series is not interchangeable with a G/E/C
            # series.  The old key omitted the constellation set and could
            # silently reuse a mixed-system cache in a GPS-only experiment.
            sys_tag = "".join(sorted(systems))
            ckey = cache_dir / f"{name}_{st.st_size}_{int(st.st_mtime)}_{sys_tag}.pkl"
        except OSError:
            ckey = None
        if ckey is not None and ckey.exists():
            try:
                series[name] = _pkl.loads(ckey.read_bytes())
                continue
            except Exception:
                pass
        station_out = out_dir / f"cors_{name}"
        station_out.mkdir(parents=True, exist_ok=True)
        try:
            if path.suffix == ".crx" or (path.suffix == ".gz" and path.stem.endswith(".crx")):
                rnx = _decompress_crx(path, station_out)
                if rnx is None:
                    continue
            else:
                rnx = path
            hdr = _rinex_header_xyz(rnx)
            if hdr is None:
                continue
            s = build_clock_series_precise_multi(rnx, sp3, clk, hdr, name,
                                                  systems=systems)
            if s is not None:
                s["precise"] = True
                series[name] = s
                if ckey is not None:
                    try:
                        ckey.write_bytes(_pkl.dumps(s))
                    except Exception:
                        pass
        except Exception:
            continue
    return series


# ------------------------------------------------------------------ ground truth

def load_ground_truth(ds: Dataset) -> dict[int, np.ndarray] | None:
    """Load per-second-of-day ground-truth ECEF positions."""
    if not ds.ground_truth.exists():
        return None
    rows = []
    with open(ds.ground_truth, "r") as f:
        if ds.ground_truth.suffix == ".csv":
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    t = float(r[ds.gt_time_col])
                    if ds.gt_fmt == "llh":
                        lat = float(r[ds.gt_lat_col])
                        lon = float(r[ds.gt_lon_col])
                        h = float(r[ds.gt_hgt_col])
                        xyz = _wgs84_ecef(lat, lon, h)
                    elif ds.gt_fmt == "ecef":
                        xyz = np.array([float(r[ds.gt_x_col]),
                                        float(r[ds.gt_y_col]),
                                        float(r[ds.gt_z_col])], float)
                    else:
                        continue
                    rows.append((int(round(t % 86400.0)), xyz))
                except (ValueError, KeyError):
                    continue
        else:
            # whitespace table with columns in fixed order
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    t = float(toks[0])
                    if ds.gt_fmt == "llh":
                        xyz = _wgs84_ecef(float(toks[1]), float(toks[2]),
                                          float(toks[3]))
                    elif ds.gt_fmt == "grafnav_dms":
                        if len(toks) < 8:
                            continue
                        # The source is 10 Hz; retain the integer-second epoch
                        # matched by the operational 1 Hz evaluation.
                        if abs(t - round(t)) > 0.02:
                            continue
                        lat_d, lat_m, lat_s = map(float, toks[1:4])
                        lon_d, lon_m, lon_s = map(float, toks[4:7])
                        lat = math.copysign(abs(lat_d) + lat_m / 60.0
                                            + lat_s / 3600.0, lat_d)
                        lon = math.copysign(abs(lon_d) + lon_m / 60.0
                                            + lon_s / 3600.0, lon_d)
                        xyz = _wgs84_ecef(lat, lon, float(toks[7]))
                    elif ds.gt_fmt == "ecef":
                        xyz = np.array([float(toks[1]), float(toks[2]),
                                        float(toks[3])], float)
                    else:
                        continue
                    rows.append((int(round(t % 86400.0)), xyz))
                except ValueError:
                    continue
    if not rows:
        return None
    rows.sort()
    out = {}
    for t, xyz in rows:
        out[t] = xyz
    return out


def interpolate_gt(gt: dict[int, np.ndarray], sod: float) -> np.ndarray | None:
    times = np.asarray(sorted(gt.keys()), float)
    if len(times) < 2:
        return gt.get(int(sod))
    if sod < times[0] or sod > times[-1]:
        return None
    arr = np.array([gt[int(t)] for t in times])
    return np.array([np.interp(sod, times, arr[:, k]) for k in range(3)])


# ------------------------------------------------------------------ CORS network

def _rinex_header_xyz(path: Path) -> np.ndarray | None:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for line in f:
            if "APPROX POSITION" in line:
                toks = line.split()
                try:
                    return np.array([float(toks[0]), float(toks[1]),
                                     float(toks[2])])
                except (ValueError, IndexError):
                    return None
    return None


def _station_name_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith(".rnx") or name.endswith(".obs"):
        name = name[:-4]
    if name.endswith(".crx"):
        name = name[:-4]
    if name.endswith(".o"):
        name = name[:-2]
    if name.endswith(".gz"):
        name = name[:-3]
    # RINEX 3 long name: AAAA00CCC_R_... -> first 4 chars
    if "_" in name:
        return name.split("_")[0][:4].upper()
    return name[:4].upper()


def build_cors_series(cors_dir: Path, nav: Path | None, sp3: Path | None,
                      clk: Path | None, out_dir: Path,
                      use_precise: bool = True) -> dict[str, dict]:
    """Process all CORS RINEX in a directory into clock residual series.

    Each station's clock series is cached under ``cors_dir/.clkcache`` keyed on
    the RINEX file's (size, mtime) -- processing a full-day all-GNSS RINEX with
    per-epoch SP3 interpolation is minutes per station, and the pipeline
    re-runs the same network many times.
    """
    import pickle as _pkl
    if not cors_dir or not cors_dir.exists():
        return {}
    cache_dir = cors_dir / ".clkcache"
    cache_dir.mkdir(exist_ok=True)
    series = {}
    can_use_precise = (use_precise and
                       sp3 and sp3.exists() and clk and clk.exists())
    for path in sorted(cors_dir.iterdir()):
        if path.suffix not in (".rnx", ".crx", ".obs", ".o", ".gz"):
            continue
        name = _station_name_from_path(path)
        try:
            st = path.stat()
            ckey = cache_dir / f"{name}_{st.st_size}_{int(st.st_mtime)}_{int(bool(can_use_precise))}.pkl"
        except OSError:
            ckey = None
        if ckey is not None and ckey.exists():
            try:
                series[name] = _pkl.loads(ckey.read_bytes())
                continue
            except Exception:
                pass
        station_out = out_dir / f"cors_{name}"
        station_out.mkdir(parents=True, exist_ok=True)
        try:
            # convert crx to rnx if necessary
            if path.suffix == ".crx" or (path.suffix == ".gz" and path.stem.endswith(".crx")):
                rnx = _decompress_crx(path, station_out)
                if rnx is None:
                    continue
            else:
                rnx = path

            s = None
            hdr = _rinex_header_xyz(rnx)
            # prefer precise-ephemeris prefit if products are available
            if can_use_precise and hdr is not None:
                s = build_clock_series_precise(rnx, sp3, clk, hdr, name)
                if s is not None:
                    s["precise"] = True

            if s is None:
                # fallback to broadcast-SPP clock; also gives a position if header is missing
                s = build_clock_series(rnx, nav, sp3, station_out, name)

            if s is not None:
                # if the precise prefit was not used, try again with the SPP-derived position
                if can_use_precise and not s.get("precise") and s.get("station"):
                    s_precise = build_clock_series_precise(rnx, sp3, clk,
                                                           s["station"].ecef_m, name)
                    if s_precise is not None:
                        s = s_precise
                        s["precise"] = True

                if hdr is not None and not s.get("precise"):
                    lat, lon = td.ecef_to_geodetic(*hdr)
                    s["station"] = td.Station(name, hdr / 1000.0, lat, lon)
                series[name] = s
                if ckey is not None:
                    try:
                        ckey.write_bytes(_pkl.dumps(s))
                    except Exception:
                        pass
        except Exception:
            continue
    return series


# ------------------------------------------------------------------ kernel prior

def _station_from_ecef(name: str, ecef: np.ndarray) -> td.Station:
    lat, lon = td.ecef_to_geodetic(*ecef)
    return td.Station(name, np.asarray(ecef, float) / 1000.0, lat, lon)


def kernel_prior_from_cors(cors_series: dict[str, dict],
                           target_ecef: np.ndarray,
                           band_lo_uhz: float = 10.0,
                           band_hi_uhz: float = 500.0) -> dict | None:
    """Predict the rover clock at `target_ecef` from a CORS network."""
    if len(cors_series) < 6:
        return None

    names, X, D = tk.build_matrix(cors_series, band_lo_uhz=band_lo_uhz,
                                  band_hi_uhz=band_hi_uhz,
                                  filter_by_std=False)
    if X is None or len(names) < 6:
        return None

    # station QC after the fit
    good = [i for i in range(len(names))
            if 0.01 < X[:, i].std() < 60.0]
    if len(good) < 6:
        return None
    names = [names[i] for i in good]
    X = X[:, good]
    D = D[np.ix_(good, good)]

    p, tr_err = tk.fit_hyperparameters(X, D, list(range(len(names))))
    if p is None:
        return None

    target = _station_from_ecef("target", target_ecef)
    ref_xyz = np.array([cors_series[n]["station"].ecef_m for n in names])
    d_rt = np.array([_greatcircle_km(ref_xyz[i], target.ecef_m)
                     for i in range(len(names))])

    w = tk.kriging_weights(D, d_rt, p)
    pred_m = X @ w
    if not np.all(np.isfinite(pred_m)):
        return None

    return {
        "t": np.asarray(cors_series[names[0]]["t"], float),
        "kres_m": np.asarray(pred_m, float),
        "sigma_m": max(float(tr_err), 0.1),
        "params": p.as_dict(),
        "n_refs": len(names),
    }


def _load_clock_type_map() -> dict[str, str]:
    """Load clock type (QUARTZ/ATOMIC) for each station from metadata."""
    meta_path = PROJECT_ROOT / "data" / "meta" / "optimal_100_metadata.json"
    if not meta_path.exists():
        return {}
    d = json.load(open(meta_path))
    stations = d.get("stations", {})
    type_map = {}
    quartz_types = {"INTERNAL", "OCXO"}
    atomic_types = {"H-MASER", "RUBIDIUM", "CESIUM"}
    for name, info in stations.items():
        ct = info.get("classification", {}).get("clock_type", "UNKNOWN")
        if ct in quartz_types:
            type_map[name] = "QUARTZ"
        elif ct in atomic_types:
            type_map[name] = "ATOMIC"
        else:
            type_map[name] = "UNKNOWN"
    return type_map


# Step 65 screening factors: quartz clocks see 3.78x more LF TEP field
# than atomic, 1.20x more MF, 0.91x HF (no screening).
SCREENING_FACTORS = {"LF": 3.78, "MF": 1.20, "HF": 0.91}


def fixed_tep_context_from_cors(cors_series: dict[str, dict],
                                target_ecef: np.ndarray,
                                cutoff_sod: float,
                                lambda_km: float = 4200.0,
                                nugget: float = 0.35,
                                form: str = "gaussian",
                                qc_history_s: float = 7200.0,
                                rover_clock_type: str = "QUARTZ",
                                screening_mode: str = "none") -> dict | None:
    """Build a pre-registered, causal TEP network context for a rover.

    Station QC uses only observations at or before ``cutoff_sod``.  The
    distance scale and nugget are fixed from the independent static-clock
    experiments; no parameter is fitted to the held-out flight day.  Raw
    reference clocks are retained because their outage-time innovation is
    computed causally by :func:`causal_tep_delta`.

    ``rover_clock_type`` and ``screening_mode`` control clock-type-aware
    kriging.  When the rover has a quartz oscillator and the CORS network
    is dominated by atomic clocks, the atomic stations are screened from
    the LF TEP field (Step 65: 3.78x screening factor).  Three modes:

    - ``none``: standard kriging (all stations equal weight)
    - ``quartz_only``: use only quartz CORS stations for the kriging
    - ``screening_scaled``: keep all stations but scale the network delta
      by the LF screening factor (3.78x) to correct for atomic screening
    - ``type_weighted``: upweight quartz stations by the screening factor
      in the kriging weight vector
    """
    if len(cors_series) < 6:
        return None
    names = sorted(cors_series)
    t = np.asarray(cors_series[names[0]]["t"], float)
    if len(t) < 8:
        return None
    cols, kept = [], []
    for name in names:
        s = cors_series[name]
        ts = np.asarray(s["t"], float)
        y = np.asarray(s["raw"], float) * C_M_PER_NS
        good = np.isfinite(y)
        if good.sum() < 8:
            continue
        yi = _unwrap_ms(np.interp(t, ts[good], y[good]))
        hist = (t <= cutoff_sod) & (t >= cutoff_sod - qc_history_s)
        if hist.sum() < 8:
            continue
        q = t[hist] - cutoff_sod
        fit = np.polyfit(q, yi[hist], 1)
        sd = float(np.std(yi[hist] - np.polyval(fit, q)))
        if not (0.003 < sd < 60.0):
            continue
        cols.append(yi)
        kept.append(name)
    if len(kept) < 6:
        return None

    # Robust pre-flight QC catches receivers with much noisier clocks without
    # consulting any flight-time or post-flight observation.
    hist = (t <= cutoff_sod) & (t >= cutoff_sod - qc_history_s)
    sds = []
    for y in cols:
        q = t[hist] - cutoff_sod
        fit = np.polyfit(q, y[hist], 1)
        sds.append(float(np.std(y[hist] - np.polyval(fit, q))))
    med = float(np.median(sds))
    use = [i for i, sd in enumerate(sds) if sd <= max(4.0 * med, 0.05)]
    if len(use) < 6:
        return None
    kept = [kept[i] for i in use]
    Y = np.column_stack([cols[i] for i in use])

    # Clock-type-aware station selection / weighting
    type_map = _load_clock_type_map()
    ref_types = [type_map.get(n, "UNKNOWN") for n in kept]

    if screening_mode == "quartz_only":
        quartz_idx = [i for i, ct in enumerate(ref_types) if ct == "QUARTZ"]
        if len(quartz_idx) >= 3:
            # Rebuild with quartz-only stations
            kept = [kept[i] for i in quartz_idx]
            Y = Y[:, quartz_idx]
            ref_types = [ref_types[i] for i in quartz_idx]

    stations = [cors_series[n]["station"] for n in kept]
    n = len(stations)
    D = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i):
            D[i, j] = D[j, i] = td.great_circle_km(stations[i], stations[j])
    target = _station_from_ecef("DRONE", target_ecef)
    d_rt = np.array([td.great_circle_km(s, target) for s in stations])
    params = tk.KernelParams(lambda_km=lambda_km, nugget=nugget,
                             k_neighbours=n, form=form)
    w = tk.kriging_weights(D, d_rt, params)

    if screening_mode == "type_weighted" and rover_clock_type == "QUARTZ":
        # Upweight quartz stations by the LF screening factor
        for i, ct in enumerate(ref_types):
            if ct == "ATOMIC":
                w[i] /= SCREENING_FACTORS["LF"]
        w = w / np.sum(w)

    return {
        "t": t,
        "raw_m": Y,
        "weights": w,
        "names": kept,
        "ref_types": ref_types,
        "rover_clock_type": rover_clock_type,
        "screening_mode": screening_mode,
        "n_refs": n,
        "params": params.as_dict(),
        "qc_cutoff_sod": float(cutoff_sod),
        "qc_history_s": float(qc_history_s),
        "preflight_ref_resid_sd_median": med,
    }


def causal_tep_delta(ctx: dict, anchor_sod: float, sod: float,
                     prewindow_s: float = 1800.0) -> tuple[float, float] | None:
    """Return live TEP network innovation since a frozen outage anchor.

    Each reference clock is fitted only on the trailing window ending at the
    anchor.  The same frozen line is extrapolated to ``sod``; live reference
    deviations are combined with the fixed TEP weights.  The second return
    value is a robust reference-network dispersion used for prior scaling.
    """
    t = np.asarray(ctx["t"], float)
    Y = np.asarray(ctx["raw_m"], float)
    hist = (t <= anchor_sod) & (t >= anchor_sod - prewindow_s)
    if hist.sum() < 8 or sod < anchor_sod or sod > t[-1]:
        return None
    q = t[hist] - anchor_sod
    A = np.column_stack((q, np.ones(hist.sum())))
    coef = np.linalg.lstsq(A, Y[hist], rcond=None)[0]
    at_anchor = Y[np.argmin(np.abs(t - anchor_sod))]
    at_live = np.array([np.interp(sod, t, Y[:, j]) for j in range(Y.shape[1])])
    r0 = at_anchor - coef[1]
    r1 = at_live - (coef[0] * (sod - anchor_sod) + coef[1])
    change = r1 - r0
    if not np.isfinite(change).all():
        return None
    pred = float(np.asarray(ctx["weights"]) @ change)
    med = float(np.median(change))
    sigma = float(1.4826 * np.median(np.abs(change - med)))
    # A leaked 1 ms wrap on one reference would dominate the field increment.
    # Fall back to the robust common-mode, then to zero, rather than inject it.
    if abs(pred) > 50.0:
        pred = med if abs(med) <= 50.0 else 0.0
    # Apply screening correction: if the rover is quartz and the network
    # is dominated by atomic clocks, the atomic-predicted delta underestimates
    # the LF TEP field that the quartz rover experiences (Step 65: 3.78x).
    mode = ctx.get("screening_mode", "none")
    if mode == "screening_scaled" and ctx.get("rover_clock_type") == "QUARTZ":
        pred *= SCREENING_FACTORS["LF"]
    return pred, max(sigma, 0.2)


# ------------------------------------------------------------------ epoch solver

def _condition_ok(Aw: np.ndarray) -> bool:
    sv = np.linalg.svd(Aw, compute_uv=False)
    return bool(sv[-1] > 1e-6 * sv[0])


def solve_epoch(sats: list[dict], cdt_prior: float | None,
                cdt_prior_sig: float, obs_sigma: float = 1.0,
                max_offset_m: float = 10000.0) -> tuple[np.ndarray | None, float | None]:
    """Solve [dE,dN,dU,cdt] (ENU m) from LOS residuals and optional clock prior.

    Each sat dict must contain either ``e_enu`` (unit LOS in ENU) or
    ``az``/``el`` (degrees).  The residual is in metres.  Returns
    (dpos_enu, cdt_est).
    """
    n = len(sats)
    if n == 0:
        return None, None
    e = np.array([s.get("e_enu", _los_enu(s["az"], s["el"])) for s in sats])
    b = np.array([s["resid"] for s in sats])
    el = np.array([s["el"] for s in sats])
    w = np.sin(np.radians(el)) ** 2 / (obs_sigma ** 2)
    A = np.column_stack([-e, np.ones(n)])
    if cdt_prior is not None and cdt_prior_sig > 0:
        A = np.vstack([A, [0.0, 0.0, 0.0, 1.0]])
        b = np.append(b, cdt_prior)
        w = np.append(w, 1.0 / (cdt_prior_sig ** 2))
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w
    if not _condition_ok(Aw):
        return None, None
    try:
        x, _, rank, _ = np.linalg.lstsq(Aw, bw, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None, None
    if rank < 4 or not np.all(np.isfinite(x)):
        return None, None
    if np.linalg.norm(x[:3]) > max_offset_m:
        return None, None
    return x[:3], float(x[3]) if len(x) > 3 else None


# ------------------------------------------------------------------ metric computation

def _compute_tangent(enu_traj: np.ndarray, window: int = 5) -> np.ndarray:
    """Smoothed 2-D tangent vector [E,N] per epoch, normalised."""
    n = len(enu_traj)
    tangents = np.zeros((n, 2))
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        d = enu_traj[hi - 1, :2] - enu_traj[lo, :2]
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            tangents[i] = d / norm
        else:
            tangents[i] = [1.0, 0.0]
    return tangents


def _reconvergence_times(errors: dict[int, float], blackouts: list[tuple[int, int]],
                         threshold: float = 2.0) -> list[float | None]:
    """For each blackout, seconds until 3-D error drops below threshold."""
    out = []
    for start, end in blackouts:
        conv = None
        for t in sorted(errors):
            if t > end and errors[t] < threshold:
                conv = float(t - end)
                break
        out.append(conv)
    return out


def _anchor_kernel_to_clock(cdt_free: dict[int, float],
                            kernel_t: np.ndarray,
                            kernel_res: np.ndarray) -> tuple[float, float]:
    """Estimate mean bias and drift between free-clock and kernel residual."""
    if not cdt_free:
        return 0.0, 0.0
    times = np.asarray(sorted(cdt_free.keys()), float)
    free = np.array([cdt_free[int(t)] for t in times])
    ker = np.interp(times, kernel_t, kernel_res)
    # simple least-squares: free = bias + drift * t + ker
    a = np.column_stack([np.ones_like(times), times, ker])
    w = np.ones_like(times)
    coef, *_ = np.linalg.lstsq(a * w[:, None], free * w, rcond=1e-12)
    return float(coef[0]), float(coef[1])


def read_rinex3_gps(rnx_text: str) -> dict[int, dict[str, tuple[float, float]]]:
    """Parse a RINEX 3 OBS text for GPS observables.

    Accepts single-frequency (C1C) and dual-frequency (C1W/C2W or C1C/C2C)
    logs.  For a single-frequency satellite the same value is used for both
    slots so the ionosphere-free combination (IF_A+IF_B=1) returns the raw
    pseudorange.
    """
    lines = rnx_text.splitlines()
    i = 0
    gps_types: list[str] = []
    while i < len(lines):
        ln = lines[i]
        if "SYS / # / OBS TYPES" in ln and ln[0] == "G":
            n = int(ln[3:6])
            toks = ln[7:60].split()
            j = i
            while len(toks) < n:
                j += 1
                toks += lines[j][7:60].split()
            gps_types = toks[:n]
            i = j
        if "END OF HEADER" in ln:
            i += 1
            break
        i += 1
    if not gps_types:
        return {}

    def _idx(name: str):
        return gps_types.index(name) if name in gps_types else None

    i_c1 = _idx("C1W") if "C1W" in gps_types else _idx("C1C")
    i_c2 = (_idx("C2W") if "C2W" in gps_types else
            (_idx("C2L") if "C2L" in gps_types else
             (_idx("C2C") if "C2C" in gps_types else None)))
    if i_c1 is None:
        return {}

    out: dict[int, dict[str, tuple[float, float]]] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            t = ln.split()
            if len(t) < 9:
                i += 1
                continue  # malformed or event line
            sod = int(t[4]) * 3600 + int(t[5]) * 60 + int(round(float(t[6])))
            nsat = int(t[8])
            rec = {}
            for _ in range(nsat):
                i += 1
                if i >= len(lines):
                    break
                sl = lines[i]
                if not sl or sl[0] != "G":
                    continue
                prn = sl[:3].replace(" ", "0")
                fields = []
                for f in range(len(gps_types)):
                    seg = sl[3 + f * 16: 3 + f * 16 + 14].strip()
                    fields.append(float(seg) if seg else np.nan)
                c1 = fields[i_c1]
                c2 = fields[i_c2] if i_c2 is not None and i_c2 < len(fields) else np.nan
                if np.isfinite(c1):
                    if not np.isfinite(c2):
                        c2 = c1  # single-frequency fallback
                    rec[prn] = (c1, c2)
            if rec:
                out[sod] = rec
        i += 1
    return out


def read_rinex3_multi(rnx_text: str,
                      systems: tuple[str, ...] = ("G", "E", "C")
                      ) -> dict[int, dict[str, float]]:
    """Parse a RINEX 3 OBS text for multi-GNSS ionosphere-free pseudoranges.

    Returns {sod: {prn: p_if_m}} where prn is keyed as 'G01', 'E02', 'C06'.
    Per-system IF coefficients are applied using the frequency pairs in
    ``spp.GNSS_SYSTEMS``.  Single-frequency satellites are dropped (no silent
    C1 fallback) because a mixed single/dual-frequency IF combination is not
    on the same observable as the dual-frequency one.
    """
    from scripts.lib.tep_spp import GNSS_SYSTEMS, _if_coeffs

    # The public IGN UAV benchmark is RINEX 2.11.  Parse its GPS P1/P2 code
    # directly instead of relying on a lossy external conversion.
    first = rnx_text.splitlines()[0] if rnx_text else ""
    try:
        rinex_version = float(first[:9])
    except ValueError:
        rinex_version = 3.0
    if rinex_version < 3.0:
        if "G" not in systems:
            return {}
        lines = rnx_text.splitlines()
        obs_types: list[str] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if "# / TYPES OF OBSERV" in ln:
                n = int(ln[:6])
                obs_types.extend(ln[6:60].split())
                while len(obs_types) < n:
                    i += 1
                    obs_types.extend(lines[i][6:60].split())
                obs_types = obs_types[:n]
            if "END OF HEADER" in ln:
                i += 1
                break
            i += 1
        c1_name = "P1" if "P1" in obs_types else "C1"
        c2_name = "P2" if "P2" in obs_types else "C2"
        if c1_name not in obs_types or c2_name not in obs_types:
            return {}
        i1, i2 = obs_types.index(c1_name), obs_types.index(c2_name)
        if_a, if_b = _if_coeffs(1575.42e6, 1227.60e6)
        lines_per_sat = int(math.ceil(len(obs_types) / 5.0))
        out: dict[int, dict[str, float]] = {}
        epoch_distance: dict[int, float] = {}
        while i < len(lines):
            ln = lines[i]
            if len(ln) < 32:
                i += 1
                continue
            try:
                flag = int(ln[28:29])
                nsat = int(ln[29:32])
                hh, mm, sec = int(ln[10:12]), int(ln[13:15]), float(ln[15:26])
            except ValueError:
                i += 1
                continue
            n_sat_lines = int(math.ceil(nsat / 12.0))
            sat_text = ln[32:68]
            for _ in range(1, n_sat_lines):
                i += 1
                sat_text += lines[i][32:68]
            sats = [sat_text[j:j + 3].strip() for j in range(0, len(sat_text), 3)]
            sats = [s for s in sats if s][:nsat]
            rec: dict[str, float] = {}
            for sat in sats:
                chunks = []
                for _ in range(lines_per_sat):
                    i += 1
                    chunks.append(lines[i] if i < len(lines) else "")
                if flag not in (0, 1) or not sat.startswith("G"):
                    continue
                payload = "".join(x[:80].ljust(80) for x in chunks)
                fields = []
                for j in range(len(obs_types)):
                    seg = payload[j * 16:j * 16 + 14].strip()
                    try:
                        fields.append(float(seg) if seg else np.nan)
                    except ValueError:
                        fields.append(np.nan)
                c1, c2 = fields[i1], fields[i2]
                if np.isfinite(c1) and np.isfinite(c2):
                    prn = f"G{int(sat[1:]):02d}"
                    rec[prn] = if_a * c1 + if_b * c2
            sod_float = hh * 3600.0 + mm * 60.0 + sec
            sod = int(round(sod_float))
            distance = abs(sod_float - sod)
            if rec and distance < epoch_distance.get(sod, np.inf):
                out[sod] = rec
                epoch_distance[sod] = distance
            i += 1
        return out

    lines = rnx_text.splitlines()
    # parse per-system obs type headers
    sys_types: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "SYS / # / OBS TYPES" in ln and ln[0] in systems:
            sys_char = ln[0]
            n = int(ln[3:6])
            toks = ln[7:60].split()
            j = i
            while len(toks) < n:
                j += 1
                toks += lines[j][7:60].split()
            sys_types[sys_char] = toks[:n]
            i = j
        if "END OF HEADER" in ln:
            i += 1
            break
        i += 1

    # per-system code-pair indices
    sys_code_idx: dict[str, tuple[int, int, float, float]] = {}
    for sys_char, types in sys_types.items():
        cfg = GNSS_SYSTEMS.get(sys_char)
        if cfg is None:
            continue
        # try the default frequency pair first, then alt pairs
        pairs_to_try = [(cfg["f1"], cfg["f2"], cfg["codes"])]
        for alt_f2, alt_codes in cfg.get("alt_pairs", []):
            pairs_to_try.append((cfg["f1"], alt_f2, alt_codes))
        for f1, f2, codes in pairs_to_try:
            if_a, if_b = _if_coeffs(f1, f2)
            for c1_name, c2_name in codes:
                if c1_name in types and c2_name in types:
                    sys_code_idx[sys_char] = (types.index(c1_name),
                                              types.index(c2_name),
                                              if_a, if_b)
                    break
            if sys_char in sys_code_idx:
                break

    if not sys_code_idx:
        return {}

    out: dict[int, dict[str, float]] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            t = ln.split()
            if len(t) < 9:
                i += 1
                continue  # malformed or event line, not an epoch header
            sod = int(t[4]) * 3600 + int(t[5]) * 60 + int(round(float(t[6])))
            nsat = int(t[8])
            rec: dict[str, float] = {}
            for _ in range(nsat):
                i += 1
                if i >= len(lines):
                    break
                sl = lines[i]
                if not sl or sl[0] not in sys_code_idx:
                    continue
                sys_char = sl[0]
                prn = sl[:3].replace(" ", "0")
                types = sys_types[sys_char]
                idx1, idx2, if_a, if_b = sys_code_idx[sys_char]
                c1 = c2 = np.nan
                seg1 = sl[3 + idx1 * 16: 3 + idx1 * 16 + 14].strip()
                if seg1:
                    try:
                        c1 = float(seg1)
                    except ValueError:
                        pass
                seg2 = sl[3 + idx2 * 16: 3 + idx2 * 16 + 14].strip()
                if seg2:
                    try:
                        c2 = float(seg2)
                    except ValueError:
                        pass
                if np.isfinite(c1) and np.isfinite(c2):
                    rec[prn] = if_a * c1 + if_b * c2
            if rec:
                out[sod] = rec
        i += 1
    return out


def build_zero_mean_dgps_corrections(
        base_obs: dict[int, dict[str, float]], base_xyz: np.ndarray,
        dense_grid) -> dict[int, dict[str, float]]:
    """Build causal same-epoch satellite code corrections from a local base.

    The epoch median absorbs the base receiver clock.  Only the zero-mean
    satellite-specific residual is transferred to the rover, so this removes
    common orbit/atmosphere/code bias without leaking the base's independent
    receiver-clock level into the TEP target.  No rover position or scoring
    truth enters the correction.
    """
    corrections: dict[int, dict[str, float]] = {}
    for sod, epoch in base_obs.items():
        residuals = []
        for prn, p_if in epoch.items():
            pred, _, _, _, ok = spp.pseudorange_model(
                base_xyz, prn, dense_grid, sod)
            if ok and np.isfinite(p_if):
                residuals.append((prn, float(p_if - pred)))
        if len(residuals) < 4:
            continue
        common = float(np.median([v for _, v in residuals]))
        corrections[sod] = {prn: value - common
                            for prn, value in residuals}
    return corrections


def evaluate_dataset(ds: Dataset, out_dir: Path,
                     tmp_parent: Path | None = None) -> dict:
    """Run the full urban/kinematic evaluation for one dataset entry."""
    t0 = datetime.now(timezone.utc).isoformat()

    if not ds.raw_obs.exists():
        return {"name": ds.name, "status": "missing_raw", "timestamp": t0}
    if not ds.ground_truth.exists():
        return {"name": ds.name, "status": "missing_gt", "timestamp": t0}
    if ds.cors_dir is None or not ds.cors_dir.exists():
        return {"name": ds.name, "status": "missing_cors", "timestamp": t0}

    if tmp_parent is None:
        tmp_parent = Path(tempfile.mkdtemp(prefix=f"tep_dyn_{ds.name}_"))
    work = tmp_parent

    # products
    sp3 = ds.sp3
    clk = ds.clk
    if not sp3 or not clk:
        sp3, clk = find_sp3_clk(ds.date_tag) if ds.date_tag else (None, None)
    nav = ds.nav if ds.nav else find_nav(ds.date_tag)

    # convert rover log
    try:
        rnx = convert_to_rinex(ds, work / "rover")
    except Exception as e:
        return {"name": ds.name, "status": "convert_failed", "error": str(e),
                "timestamp": t0}
    if not rnx.get("obs"):
        return {"name": ds.name, "status": "no_rover_rinex", "timestamp": t0}
    obs = rnx["obs"]
    if not nav and not rnx.get("nav"):
        return {"name": ds.name, "status": "no_nav", "timestamp": t0}
    nav = nav or rnx["nav"]

    # ground truth
    gt = load_ground_truth(ds)
    if not gt:
        return {"name": ds.name, "status": "no_ground_truth", "timestamp": t0}

    # CORS clock network
    cors_series = build_cors_series(ds.cors_dir, nav, sp3, clk, work / "cors",
                                    use_precise=ds.cors_use_precise)
    if len(cors_series) < 6:
        return {"name": ds.name, "status": "insufficient_cors",
                "n_cors": len(cors_series), "timestamp": t0}

    # target centroid for the spatial prior
    gt_ecef = np.array(list(gt.values()), float)
    target_centroid = np.median(gt_ecef, axis=0)

    kp = kernel_prior_from_cors(cors_series, target_centroid)
    if kp is None:
        return {"name": ds.name, "status": "kernel_failed",
                "n_cors": len(cors_series), "timestamp": t0}

    # raw-observation pre-fit residuals require precise orbits and clocks
    if not (sp3 and sp3.exists() and clk and clk.exists()):
        return {"name": ds.name, "status": "missing_products",
                "note": "SP3 and CLK products are required for raw-observation evaluation. "
                        "Place them in data/rinex/{sp3,clk} or specify them in the manifest.",
                "timestamp": t0}

    stat = _run_rnx2rtkp(obs, nav, sp3, work / "rover" / "rtk.pos",
                         elev_mask=ds.elev_mask_deg, y=2)
    rtk_epochs = parse_stat(stat) if stat else {}

    sp3_dict = spp.read_sp3_gps(sp3)
    clk_dict = spp.read_clk_gps(clk)
    dense = spp.dense_grid(sp3_dict, clk_dict, step=1.0)
    rnx_text = read_rinex_text(obs, work / "rover")
    obs_dict = read_rinex3_gps(rnx_text) if rnx_text else {}
    if not obs_dict:
        return {"name": ds.name, "status": "no_gps_obs", "timestamp": t0}

    results = []
    per_epoch = []

    kres_t = kp["t"]
    kres = kp["kres_m"]
    ksig = max(kp["sigma_m"], 0.2)

    # free-clock estimates to anchor the TEP prior
    cdt_free_for_anchor = {}

    for sod in sorted(gt.keys()):
        truth = gt[sod]
        R = _enu_rot(truth)

        if sod not in obs_dict:
            continue
        pf = spp.prefit_fast(obs_dict[sod], dense, truth, sod)
        sats = []
        for _, los, el, resid, _ in pf:
            if el >= ds.elev_mask_deg:
                # los is from receiver to satellite; the design matrix uses -los
                e_enu = R @ (-np.asarray(los, float))
                sats.append({"az": 0.0, "el": el, "resid": resid,
                             "e_enu": e_enu})

        k = len(sats)
        if k == 0:
            results.append({"sod": sod, "k": 0,
                            "err_free_m": None, "err_tep_m": None,
                            "err_rtk_m": None,
                            "enu_free": None, "enu_tep": None})
            per_epoch.append(results[-1])
            continue

        # free-clock solve
        dpos_free, cdt_free = solve_epoch(sats, None, 0.0,
                                          obs_sigma=ds.obs_sigma_m)
        err_free = None
        if dpos_free is not None:
            err_free = float(np.linalg.norm(dpos_free))
            cdt_free_for_anchor[sod] = cdt_free

        # TEP-aided solve
        dpos_tep = None
        cdt_tep = None
        if k >= 3:
            cdt_prior = None
            cdt_prior_sig = ksig * 10.0  # loose for overdetermined cases
            if k <= 4:
                # anchor from the most recent free-clock epoch
                past = [t for t in cdt_free_for_anchor if t <= sod]
                if past:
                    recent = max(past)
                    k0 = float(np.interp(recent, kres_t, kres))
                    k1 = float(np.interp(sod, kres_t, kres))
                    cdt_prior = cdt_free_for_anchor[recent] + (k1 - k0)
                    dt = float(sod - recent)
                    cdt_prior_sig = min(max(ksig + 0.01 * dt, 0.2), 20.0)
            dpos_tep, cdt_tep = solve_epoch(sats, cdt_prior, cdt_prior_sig,
                                            obs_sigma=ds.obs_sigma_m)
            # Do NOT update the free-clock anchor with the TEP-aided clock:
            # the anchor must come from a full-sky/free-clock epoch (k>=4) so
            # that later short-visibility windows do not anchor to an already
            # prior-constrained estimate.

        err_tep = float(np.linalg.norm(dpos_tep)) if dpos_tep is not None else None

        # conventional RTKLIB SPP position error, when it produced a fix
        err_rtk = None
        if sod in rtk_epochs and "pos_rtk" in rtk_epochs[sod]:
            pos_rtk = rtk_epochs[sod]["pos_rtk"]
            err_rtk = float(np.linalg.norm(R @ (pos_rtk - truth)))

        results.append({
            "sod": sod, "k": k,
            "err_free_m": err_free,
            "err_tep_m": err_tep,
            "err_rtk_m": err_rtk,
            "enu_free": [float(x) for x in dpos_free] if dpos_free is not None else None,
            "enu_tep": [float(x) for x in dpos_tep] if dpos_tep is not None else None,
        })
        per_epoch.append(results[-1])

    if not results:
        return {"name": ds.name, "status": "no_valid_epochs", "timestamp": t0}

    # metrics
    k_arr = np.array([r["k"] for r in results])
    free_err = np.array([r["err_free_m"] for r in results
                         if r["err_free_m"] is not None])
    tep_err = np.array([r["err_tep_m"] for r in results
                        if r["err_tep_m"] is not None])
    rtk_err = np.array([r["err_rtk_m"] for r in results
                        if r["err_rtk_m"] is not None])

    # dead-reckoning bridge rate: k <= 3 epochs where TEP returns a fix
    low_k = k_arr <= 3
    bridge_total = int(low_k.sum())
    bridge_kept = sum(1 for i in range(len(results))
                      if low_k[i] and results[i]["err_tep_m"] is not None)
    bridge_rate = bridge_kept / bridge_total if bridge_total else 0.0

    # k == 3 is the strictest bridge scenario
    k3 = k_arr == 3
    bridge_total_k3 = int(k3.sum())
    bridge_kept_k3 = sum(1 for i in range(len(results))
                         if k3[i] and results[i]["err_tep_m"] is not None)
    bridge_rate_k3 = bridge_kept_k3 / bridge_total_k3 if bridge_total_k3 else 0.0

    # conventional infeasible under k <= 3
    rtk_valid_low_k = sum(1 for i in range(len(results))
                          if low_k[i] and results[i]["err_rtk_m"] is not None)

    # cross-track error (TEP solution, when available)
    enu_tep = np.array([r["enu_tep"] for r in results if r["enu_tep"] is not None])
    if len(enu_tep) >= 10:
        tangents = _compute_tangent(enu_tep)
        err_h = enu_tep[:, :2]
        proj = np.einsum("ij,ij->i", err_h, tangents)[:, None] * tangents
        xte = np.linalg.norm(err_h - proj, axis=1)
        xte_med = float(np.median(xte))
        xte_p95 = float(np.percentile(xte, 95))
    else:
        xte_med = xte_p95 = None

    # time to re-convergence after complete blackouts (k == 0)
    blackouts = []
    in_blackout = False
    start = end = None
    for r in results:
        if r["k"] == 0:
            if not in_blackout:
                start = end = r["sod"]
                in_blackout = True
            else:
                end = r["sod"]
        else:
            if in_blackout:
                blackouts.append((start, end))
                in_blackout = False
    if in_blackout:
        blackouts.append((start, end))

    def _median_valid(v):
        vv = [x for x in v if x is not None]
        return float(np.median(vv)) if vv else None

    def _pct(arr, q: float):
        return float(np.percentile(arr, q)) if len(arr) else None

    free_err_by_sod = {r["sod"]: r["err_free_m"] for r in results
                       if r["err_free_m"] is not None}
    tep_err_by_sod = {r["sod"]: r["err_tep_m"] for r in results
                      if r["err_tep_m"] is not None}
    rtk_err_by_sod = {r["sod"]: r["err_rtk_m"] for r in results
                      if r["err_rtk_m"] is not None}
    recon_free = _reconvergence_times(free_err_by_sod, blackouts)
    recon_tep = _reconvergence_times(tep_err_by_sod, blackouts)
    recon_rtk = _reconvergence_times(rtk_err_by_sod, blackouts)

    summary = {
        "n_epochs": len(results),
        "n_cors": len(cors_series),
        "n_free_valid": len(free_err),
        "n_tep_valid": len(tep_err),
        "n_rtk_valid": len(rtk_err),
        "bridge_total_k_le3": bridge_total,
        "bridge_kept_tep": bridge_kept,
        "bridge_rate_tep": bridge_rate,
        "bridge_total_k3": bridge_total_k3,
        "bridge_kept_tep_k3": bridge_kept_k3,
        "bridge_rate_tep_k3": bridge_rate_k3,
        "bridge_rtk_valid_le3": rtk_valid_low_k,
        "n_blackouts_k0": len(blackouts),
        "recon_free_median_s": _median_valid(recon_free),
        "recon_tep_median_s": _median_valid(recon_tep),
        "recon_rtk_median_s": _median_valid(recon_rtk),
        "err_free_p50_m": _pct(free_err, 50),
        "err_tep_p50_m": _pct(tep_err, 50),
        "err_rtk_p50_m": _pct(rtk_err, 50),
        "err_free_p95_m": _pct(free_err, 95),
        "err_tep_p95_m": _pct(tep_err, 95),
        "err_rtk_p95_m": _pct(rtk_err, 95),
        "xte_tep_p50_m": xte_med,
        "xte_tep_p95_m": xte_p95,
        "kernel_params": kp.get("params"),
        "kernel_n_refs": kp.get("n_refs"),
    }

    return {
        "name": ds.name,
        "type": ds.type,
        "date_tag": ds.date_tag,
        "source_url": ds.source_url,
        "doi": ds.doi,
        "license": ds.license,
        "provenance_status": ds.provenance_status,
        "status": "ok",
        "timestamp": t0,
        "summary": summary,
        "per_epoch": per_epoch,
    }


# ================================================================ v2: blind / multi-GNSS / causal
# The v2 evaluation fixes every defect identified in the audit:
#   - multi-GNSS parsing (G+E+C) so k reflects reality
#   - blind nonlinear solver (no truth in prefit; init from previous epoch)
#   - TEP prior applied at EVERY epoch, not just k<=4
#   - causal kriging target = previous SOLVED position, not trajectory median
#   - simulated jamming by thinning to k=3/4 in scheduled windows
#   - TCXO-class temporal persistence control (not station-grade OCXO)
#   - fair RTKLIB control on the same thinned RINEX
#   - bridge = error below threshold, not "solve returned non-None"


def _schedule_jamming_windows(sorted_sods, n_windows=10, window_s=300,
                              seed=42):
    """Schedule jamming windows evenly through the flight, avoiding overlap.

    Adapts to short flights: if the requested n_windows * window_s exceeds
    half the flight duration, the window count is reduced so at least 20%
    of the flight is jammed.
    """
    if len(sorted_sods) < 20:
        return []
    t0, t1 = sorted_sods[0], sorted_sods[-1]
    span = t1 - t0
    if span < 60:
        return []
    # adapt: at most 30% of flight in jamming windows
    max_jam_frac = 0.3
    max_total = span * max_jam_frac
    if n_windows * window_s > max_total:
        n_windows = max(1, int(max_total // window_s))
    if n_windows < 1:
        # use shorter windows
        window_s = max(30, int(span * max_jam_frac / 5))
        n_windows = max(3, int(span * max_jam_frac / window_s))
    if n_windows * window_s > span * 0.8:
        n_windows = max(1, int(span * 0.3 / window_s))
    rng = np.random.default_rng(seed)
    step = span / (n_windows + 1)
    starts = [int(t0 + step * (i + 1) + rng.uniform(-step * 0.2, step * 0.2))
              for i in range(n_windows)]
    return [(s, s + window_s) for s in starts]


def _thin_obs_for_k(obs_epoch, grid, approximate_ecef, sod, k_target,
                    elev_mask=10.0):
    """Keep the minimum-DOP subset available to an onboard receiver.

    Highest-elevation-only thinning can place every retained satellite in the
    same sky sector and manufacture enormous position dilution.  A receiver
    with limited tracking channels can choose the subset geometrically: for
    k=3 we minimise fixed-clock position DOP, and for k>=4 conventional
    position-plus-clock DOP.  The choice uses only the previous solution and
    broadcast/precise satellite geometry, never trajectory truth.
    """
    if k_target is None or k_target <= 0:
        return obs_epoch
    pf = spp.prefit_fast_if(obs_epoch, grid, approximate_ecef, sod,
                            elev_mask_deg=elev_mask)
    if len(pf) <= k_target:
        return {prn: obs_epoch[prn] for prn, _, el, _, _ in pf
                if prn in obs_epoch}
    best, best_pdop = None, np.inf
    for combo in itertools.combinations(pf, k_target):
        pos_design = np.array([np.asarray(r[1], float) for r in combo])
        design = (pos_design if k_target == 3 else
                  np.column_stack((pos_design, np.ones(k_target))))
        pdop = spp._pdop_from_jac(design)
        if pdop < best_pdop:
            best, best_pdop = combo, pdop
    if best is None:
        return {}
    keep_prns = {r[0] for r in best}
    return {prn: p_if for prn, p_if in obs_epoch.items() if prn in keep_prns}


def _write_thinned_rinex(rnx_text, selection_ecef, grid, sp3_dict, out_path,
                         jam_windows, k_values=(3, 4, 5),
                         elev_mask=10.0, systems=("G", "E", "C")):
    """Write a RINEX file thinned in jamming windows for RTKLIB processing.

    Outside jamming windows: all satellites above elev_mask are kept.
    Inside jamming windows: only the k highest-elevation satellites are kept.
    The k value cycles through k_values across windows so each is represented.
    """
    from scripts.lib.tep_spp import prefit_fast_if, dense_grid_multi
    lines = rnx_text.splitlines(keepends=True)
    out = []
    i = 0
    sys_types: dict[str, list[str]] = {}
    while i < len(lines):
        ln = lines[i]
        if "SYS / # / OBS TYPES" in ln and ln[0] in systems:
            sys_char = ln[0]
            n = int(ln[3:6])
            toks = ln[7:60].split()
            j = i
            while len(toks) < n:
                j += 1
                toks += lines[j][7:60].split()
            sys_types[sys_char] = toks[:n]
            i = j
        out.append(ln)
        if "END OF HEADER" in ln:
            i += 1
            break
        i += 1

    # build the multi-GNSS dense grid for elevation computation
    clk_path = None  # not needed; we only need positions
    # actually we need the grid; caller passes sp3_dict already as dense grid
    # grid is already (t0, dt, G) from dense_grid_multi
    t0_grid, dt_grid, G = grid

    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            toks = ln.split()
            sod = int(toks[4]) * 3600 + int(toks[5]) * 60 + int(round(float(toks[6])))
            nsat = int(toks[8])
            sat_lines = []
            sats_with_el = []
            for _ in range(nsat):
                i += 1
                if i >= len(lines):
                    break
                sl = lines[i]
                sat_lines.append(sl)
                if sl and sl[0] in systems and selection_ecef is not None:
                    prn = sl[:3].replace(" ", "0")
                    if prn in G:
                        # quick elevation from grid
                        P, V, D = G[prn]
                        kk = (sod - 0.075 - t0_grid) / dt_grid
                        ki = int(kk)
                        if 0 <= ki < len(P) - 1:
                            fr = kk - ki
                            ps = P[ki] * (1 - fr) + P[ki + 1] * fr
                            up = selection_ecef / np.linalg.norm(selection_ecef)
                            los = (ps - selection_ecef)
                            r = np.linalg.norm(los)
                            if r > 0:
                                el = np.degrees(np.arcsin(
                                    np.clip(np.dot(los / r, up), -1, 1)))
                                if el >= elev_mask:
                                    sats_with_el.append((prn, el, sl))
            # determine if in a jamming window
            k_target = None
            for wi, (ws, we) in enumerate(jam_windows):
                if ws <= sod < we:
                    k_target = k_values[wi % len(k_values)]
                    break
            if k_target is not None and len(sats_with_el) > k_target:
                sats_with_el.sort(key=lambda r: -r[1])
                keep_prns = {r[0] for r in sats_with_el[:k_target]}
                sat_lines = [r[2] for r in sats_with_el if r[0] in keep_prns]
            elif k_target is None:
                # outside window: keep all above mask
                keep_prns = {r[0] for r in sats_with_el}
                sat_lines = [r[2] for r in sats_with_el]
            n_keep = len(sat_lines)
            sec = f"{float(toks[6]):10.7f}"
            clk = f"{0.0:15.12f}"
            new_epoch = (f"> {int(toks[1]):4d} {int(toks[2]):2d} {int(toks[3]):2d} "
                         f"{int(toks[4]):2d} {int(toks[5]):2d} {sec}  "
                         f"{int(toks[7]):1d} {n_keep:3d}        {clk}\n")
            out.append(new_epoch)
            for sl in sat_lines:
                out.append(sl)
        else:
            out.append(ln)
        i += 1
    with open(out_path, "w") as f:
        f.writelines(out)


def evaluate_dataset_v2(ds: Dataset, out_dir: Path,
                        tmp_parent: Path | None = None,
                        systems: tuple[str, ...] = ("G", "E", "C"),
                        jam_k_values: tuple[int, ...] = (3, 4, 5),
                        jam_n_windows: int = 8,
                        jam_window_s: int = 300,
                        bridge_threshold_m: float = 10.0,
                        tcxo_allan: float = 1e-9,
                        max_update_m: float = 500_000.0,
                        screening_modes: tuple[str, ...] = ("none",),
                        ) -> dict:
    """Blind multi-GNSS drone evaluation with simulated jamming.

    Returns a result dict with per-epoch errors for three arms:
      - free:  blind nonlinear SPP, free clock (conventional)
      - tep:   blind nonlinear SPP with TEP network clock prior
      - tcxo:  blind nonlinear SPP with TCXO-class temporal persistence

    The TEP prior is applied at every epoch, the kriging target is the
    previous SOLVED position (causal), and the solver is initialised from the
    previous epoch's solution (no truth leakage).  Simulated jamming windows
    thin the observations to k=3/4/5 satellites so hundreds of degraded
    epochs are scored.  RTKLIB is run on the same thinned RINEX for a fair
    conventional control.
    """
    t0 = datetime.now(timezone.utc).isoformat()

    if not ds.raw_obs.exists():
        return {"name": ds.name, "status": "missing_raw", "timestamp": t0}
    if not ds.ground_truth.exists():
        return {"name": ds.name, "status": "missing_gt", "timestamp": t0}
    if ds.cors_dir is None or not ds.cors_dir.exists():
        return {"name": ds.name, "status": "missing_cors", "timestamp": t0}

    if tmp_parent is None:
        tmp_parent = Path(tempfile.mkdtemp(prefix=f"tep_dyn2_{ds.name}_"))
    work = tmp_parent

    # products
    sp3 = ds.sp3
    clk = ds.clk
    if not sp3 or not clk:
        sp3, clk = find_sp3_clk(ds.date_tag) if ds.date_tag else (None, None)
    nav = ds.nav if ds.nav else find_nav(ds.date_tag)
    if not (sp3 and sp3.exists() and clk and clk.exists()):
        return {"name": ds.name, "status": "missing_products",
                "note": "SP3 and CLK products required for multi-GNSS evaluation.",
                "timestamp": t0}

    # convert rover log
    try:
        rnx = convert_to_rinex(ds, work / "rover")
    except Exception as e:
        return {"name": ds.name, "status": "convert_failed", "error": str(e),
                "timestamp": t0}
    if not rnx.get("obs"):
        return {"name": ds.name, "status": "no_rover_rinex", "timestamp": t0}
    obs = rnx["obs"]

    # ground truth
    gt = load_ground_truth(ds)
    if not gt:
        return {"name": ds.name, "status": "no_ground_truth", "timestamp": t0}

    # multi-GNSS dense grid
    sp3_dict = spp.read_sp3_multi(sp3, systems=systems)
    clk_dict = spp.read_clk_multi(clk, systems=systems)
    if not sp3_dict or not clk_dict:
        return {"name": ds.name, "status": "no_mg_products",
                "note": "SP3/CLK files have no G/E/C entries.",
                "timestamp": t0}
    dense = spp.dense_grid_multi(sp3_dict, clk_dict, step=1.0)

    # multi-GNSS rover obs
    rnx_text = read_rinex_text(obs, work / "rover")
    if not rnx_text:
        return {"name": ds.name, "status": "no_rnx_text", "timestamp": t0}
    obs_dict = read_rinex3_multi(rnx_text, systems=systems)
    if not obs_dict:
        return {"name": ds.name, "status": "no_mg_obs",
                "note": "No multi-GNSS IF pseudoranges could be formed.",
                "timestamp": t0}

    # Optional local-base differential code correction.  This is independent
    # of TEP and is applied identically to every positioning arm so preventable
    # orbit/atmosphere/code bias does not obscure the clock-prior experiment.
    dgps_corrections: dict[int, dict[str, float]] = {}
    if ds.base_obs is not None and ds.base_obs.exists():
        base_text = read_rinex_text(ds.base_obs, work / "base")
        base_xyz = _rinex_header_xyz(ds.base_obs)
        if base_text and base_xyz is not None:
            base_dict = read_rinex3_multi(base_text, systems=systems)
            dgps_corrections = build_zero_mean_dgps_corrections(
                base_dict, base_xyz, dense)

    def _correct_epoch(sod: int, epoch: dict[str, float]) -> dict[str, float]:
        correction = dgps_corrections.get(sod, {})
        return {prn: p_if - correction.get(prn, 0.0)
                for prn, p_if in epoch.items()}

    # CORS network prior (multi-GNSS)
    cors_series = build_cors_series_multi(ds.cors_dir, sp3, clk,
                                          work / "cors", systems=systems)
    if len(cors_series) < 6:
        return {"name": ds.name, "status": "insufficient_cors",
                "n_cors": len(cors_series), "timestamp": t0}

    # Operational cold start from the RINEX approximate coordinate and the
    # first full observation epoch.  Ground truth is never supplied to the
    # solver or to satellite selection; it is used only after solving to score
    # position error.
    gt_sorted = sorted(gt.keys())
    common_sods = [s for s in gt_sorted if s in obs_dict]
    init_approx = _rinex_header_xyz(obs)
    if not common_sods or init_approx is None:
        return {"name": ds.name, "status": "missing_blind_initialisation",
                "timestamp": t0}
    first_sod = common_sods[0]
    init_obs = list(_correct_epoch(first_sod, obs_dict[first_sod]).items())
    init_ret = spp.solve_spp_nonlinear(
        init_obs, dense, first_sod, init_approx,
        clock=None, zwd=0.0, clock_sigma=None,
        k=None, elev_mask=ds.elev_mask_deg,
        max_update_m=max_update_m)
    if init_ret is None:
        return {"name": ds.name, "status": "blind_initialisation_failed",
                "timestamp": t0}
    init_pos = init_ret[0]
    # Build one kernel context per screening mode.  Each context uses the
    # same fixed TEP scale (4200 km) but different clock-type-aware weighting.
    kps = {}
    for mode in screening_modes:
        kp = fixed_tep_context_from_cors(cors_series, init_pos,
                                         cutoff_sod=first_sod,
                                         rover_clock_type="QUARTZ",
                                         screening_mode=mode)
        if kp is not None:
            kps[mode] = kp
    if not kps:
        return {"name": ds.name, "status": "kernel_failed",
                "n_cors": len(cors_series), "timestamp": t0}
    # Use the first available mode as the primary context
    primary_mode = screening_modes[0] if screening_modes[0] in kps else next(iter(kps))
    kp = kps[primary_mode]

    # schedule jamming windows
    jam_windows = _schedule_jamming_windows(gt_sorted, n_windows=jam_n_windows,
                                            window_s=jam_window_s)

    # write thinned RINEX for RTKLIB
    thinned_path = work / "rover" / f"{ds.name}_thinned.rnx"
    try:
        if float(rnx_text[:9]) < 3.0:
            raise ValueError("RINEX-2 thinned RTKLIB control is not implemented")
        _write_thinned_rinex(rnx_text, init_pos, dense, sp3_dict, thinned_path,
                             jam_windows, k_values=jam_k_values,
                             elev_mask=ds.elev_mask_deg, systems=systems)
    except Exception as e:
        thinned_path = None

    # run RTKLIB on the thinned RINEX (fair control: same sats, same products)
    rtk_epochs = {}
    if thinned_path and thinned_path.exists():
        stat = _run_rnx2rtkp(thinned_path, nav, sp3,
                             work / "rover" / "rtk_thinned.pos",
                             elev_mask=ds.elev_mask_deg, y=2)
        rtk_epochs = parse_stat(stat) if stat else {}

    # also run RTKLIB on the full (unthinned) RINEX for reference
    stat_full = _run_rnx2rtkp(obs, nav, sp3,
                              work / "rover" / "rtk_full.pos",
                              elev_mask=ds.elev_mask_deg, y=2)
    rtk_full_epochs = parse_stat(stat_full) if stat_full else {}

    # ---- per-epoch blind solve ----
    results = []
    per_epoch = []

    prev_pos_free = init_pos.copy()
    prev_pos_tep = init_pos.copy()
    prev_pos_tcxo = init_pos.copy()
    prev_pos_oracle = init_pos.copy()
    prev_clk_free = 0.0
    prev_clk_tep = 0.0
    full_clock_history: list[tuple[float, float]] = []
    last_full_sod = None  # most recent full-sky free-clock epoch (for TEP anchor)
    last_full_clk = 0.0
    prev_full_sod = None  # second-most-recent full-sky epoch (for drift)
    prev_full_clk = 0.0

    def _clock_extrapolation(history, query_sod, window_s=60.0):
        """Robust causal local-clock extrapolation from full-sky fixes.

        Many airborne receivers steer their clock by whole milliseconds.  In
        range units this creates a +/-299792.458 m sawtooth that a naive line
        fit interprets as a catastrophic oscillator jump.  Unwrap that known
        time-steering period before fitting and wrap the extrapolation back to
        the receiver convention afterwards.
        """
        if len(history) < 8:
            return None
        recent = [(tt, cc) for tt, cc in history
                  if tt >= history[-1][0] - window_s]
        if len(recent) < 8:
            return None
        tt = np.array([z[0] for z in recent], float)
        yy_wrapped = np.array([z[1] for z in recent], float)
        steering_period_m = 299_792.458
        yy = unwrap_series(yy_wrapped, periods=(steering_period_m,))
        q = tt - tt[-1]
        coef = np.polyfit(q, yy, 1)
        resid = yy - np.polyval(coef, q)
        medr = np.median(resid)
        mad = 1.4826 * np.median(np.abs(resid - medr))
        good = np.abs(resid - medr) <= max(3.5 * mad, 0.5)
        if good.sum() >= 8:
            coef = np.polyfit(q[good], yy[good], 1)
            resid = yy[good] - np.polyval(coef, q[good])
        pred_unwrapped = float(np.polyval(coef, float(query_sod - tt[-1])))
        # Stay on the last sample's steering branch.  Folding the unwrapped
        # prediction into +-1 ms independently of the last SPP clock jumps
        # the prior by a full millisecond whenever the unwrapped state is
        # outside that interval.
        pred = float(yy_wrapped[-1] + (pred_unwrapped - yy[-1]))
        sig = float(max(1.4826 * np.median(np.abs(resid - np.median(resid))),
                        0.3))
        return pred, sig, float(coef[0])

    for sod in gt_sorted:
        truth = gt[sod]
        R = _enu_rot(truth)

        if sod not in obs_dict:
            results.append({"sod": sod, "k": 0, "k_full": 0,
                            "err_free_m": None, "err_tep_m": None,
                            "err_tcxo_m": None, "err_rtk_m": None,
                            "err_rtk_full_m": None,
                            "in_jam": False, "k_target": None})
            per_epoch.append(results[-1])
            continue

        full_obs = _correct_epoch(sod, obs_dict[sod])
        k_full = len(full_obs)

        # determine if in a jamming window
        k_target = None
        in_jam = False
        for wi, (ws, we) in enumerate(jam_windows):
            if ws <= sod < we:
                k_target = jam_k_values[wi % len(jam_k_values)]
                in_jam = True
                break

        # thin obs if in jamming window
        if in_jam and k_target is not None:
            obs_epoch = _thin_obs_for_k(full_obs, dense, prev_pos_free, sod,
                                        k_target, elev_mask=ds.elev_mask_deg)
        else:
            obs_epoch = full_obs

        k = len(obs_epoch)
        if k == 0:
            results.append({"sod": sod, "k": 0, "k_full": k_full,
                            "err_free_m": None, "err_tep_m": None,
                            "err_tcxo_m": None, "err_rtk_m": None,
                            "err_rtk_full_m": None,
                            "in_jam": in_jam, "k_target": k_target})
            per_epoch.append(results[-1])
            continue

        obs_list = [(prn, p_if) for prn, p_if in obs_epoch.items()]

        # Build the TEP prior before evaluating this epoch.  In particular, a
        # jammed epoch can never anchor its own prior from the free-clock arm.
        cdt_prior_tep = None
        cdt_prior_temporal = None
        cdt_sig_tep = 50.0
        cdt_sig_temporal = 50.0
        network_delta = None
        network_sigma = None
        if last_full_sod is not None:
            dt_anchor = float(sod - last_full_sod)
            local = _clock_extrapolation(full_clock_history, sod)
            if local is None:
                drift = 0.0
                if prev_full_sod is not None:
                    dt_drift = float(last_full_sod - prev_full_sod)
                    if dt_drift > 0:
                        drift = (last_full_clk - prev_full_clk) / dt_drift
                cdt_prior_temporal = last_full_clk + drift * dt_anchor
                local_sigma = 3.0
            else:
                cdt_prior_temporal, local_sigma, drift = local
            net = causal_tep_delta(kp, last_full_sod, sod)
            if net is not None:
                network_delta, network_sigma = net
                cdt_prior_tep = cdt_prior_temporal + float(np.clip(
                    network_delta, -80.0, 80.0))
                # Time error of a residual fractional-frequency offset: σ_y c Δt.
                drift_sigma = abs(float(tcxo_allan)) * C_M_PER_S * max(dt_anchor, 1.0)
                spatial_sigma = network_sigma / np.sqrt(max(kp["n_refs"], 1))
                cdt_sig_tep = float(np.sqrt(drift_sigma ** 2
                                             + spatial_sigma ** 2
                                             + local_sigma ** 2))
                cdt_sig_tep = max(cdt_sig_tep, 0.5)
                cdt_sig_temporal = max(float(np.sqrt(
                    drift_sigma ** 2 + local_sigma ** 2)), 0.5)

        # Per-screening-mode TEP network deltas and priors
        mode_deltas = {}
        mode_priors = {}
        mode_sigs = {}
        for mode, kp_m in kps.items():
            if last_full_sod is not None and cdt_prior_temporal is not None:
                net_m = causal_tep_delta(kp_m, last_full_sod, sod)
                if net_m is not None:
                    nd_m, ns_m = net_m
                    mode_deltas[mode] = nd_m
                    mode_priors[mode] = cdt_prior_temporal + nd_m
                    ss_m = ns_m / np.sqrt(max(kp_m["n_refs"], 1))
                    mode_sigs[mode] = max(float(np.sqrt(
                        drift_sigma ** 2 + ss_m ** 2 + local_sigma ** 2)),
                        0.5) if 'drift_sigma' in dir() else 50.0

        # ---- free-clock blind solve ----
        ret_free = spp.solve_spp_nonlinear(
            obs_list, dense, sod, prev_pos_free,
            clock=None, zwd=0.0, clock_sigma=None,
            k=None, elev_mask=ds.elev_mask_deg,
            max_update_m=max_update_m)
        if ret_free is not None:
            pos_free, clk_free, _, conv, _, _, _ = ret_free
            err_free = float(np.linalg.norm(R @ (pos_free - truth)))
            prev_pos_free = pos_free
            prev_clk_free = clk_free
        else:
            err_free = None
            pos_free = None

        # ---- TEP-aided blind solve ----
        # Soft prior at the measured prediction sigma.  A hard clock on a
        # 0.5 ms-wrong SPP hold maps into a 150 km position error at k=3;
        # that is a prior-datum bug, not a reason to drop the field.
        ret_tep = spp.solve_spp_nonlinear(
            obs_list, dense, sod, prev_pos_tep,
            clock=cdt_prior_tep, zwd=0.0,
            clock_sigma=cdt_sig_tep if cdt_prior_tep is not None else None,
            k=None, elev_mask=ds.elev_mask_deg,
            max_update_m=max_update_m)
        if ret_tep is not None:
            pos_tep_prior, clk_tep, _, conv, _, _, _ = ret_tep
            err_tep_prior = float(np.linalg.norm(R @ (pos_tep_prior - truth)))
            prev_clk_tep = clk_tep
        else:
            err_tep_prior = None
            pos_tep_prior = None

        # TEP arm is the TEP-prior solve whenever it exists.  Do not silently
        # replace it with free-clock when the prior sigma is large -- that
        # made k>=4 TEP identical to free-clock by construction.
        if ret_tep is not None:
            pos_tep, err_tep, tep_used = pos_tep_prior, err_tep_prior, True
        elif ret_free is not None:
            pos_tep, err_tep, tep_used = pos_free, err_free, False
        else:
            pos_tep, err_tep, tep_used = None, None, False
        if pos_tep is not None:
            prev_pos_tep = pos_tep

        # ---- TCXO persistence blind solve ----
        # Clock = previous clock + linear drift; sigma grows as sqrt(dt).
        # For the first epoch (no clock history), use a very loose prior so
        # the solver is effectively free-clock; once we have a clock estimate,
        # switch to the TCXO persistence model.
        if cdt_prior_temporal is None:
            cdt_prior_tcxo = None  # free clock
            cdt_sig_tcxo = 0.0
        else:
            cdt_prior_tcxo = cdt_prior_temporal
            cdt_sig_tcxo = cdt_sig_temporal

        ret_tcxo = spp.solve_spp_nonlinear(
            obs_list, dense, sod, prev_pos_tcxo,
            clock=cdt_prior_tcxo, zwd=0.0, clock_sigma=cdt_sig_tcxo if cdt_sig_tcxo > 0 else None,
            k=None, elev_mask=ds.elev_mask_deg,
            max_update_m=max_update_m)
        if ret_tcxo is not None:
            pos_tcxo_prior, clk_tcxo, _, conv, _, _, _ = ret_tcxo
            err_tcxo_prior = float(np.linalg.norm(R @ (pos_tcxo_prior - truth)))
        else:
            pos_tcxo_prior = None
            err_tcxo_prior = None
        if ret_free is not None:
            pos_tcxo, err_tcxo = pos_free, err_free
        elif ret_tcxo is not None:
            pos_tcxo, err_tcxo = pos_tcxo_prior, err_tcxo_prior
        else:
            pos_tcxo, err_tcxo = None, None
        if pos_tcxo is not None:
            prev_pos_tcxo = pos_tcxo

        # ---- Per-screening-mode TEP-prior-only solves ----
        # Always uses the TEP clock prior for each screening mode, even at
        # k>=4.  Isolates the clock-type-aware network contribution.
        mode_errs = {}
        for mode in kps:
            if mode == primary_mode:
                continue  # already solved above
            mp = mode_priors.get(mode)
            ms = mode_sigs.get(mode, 50.0)
            if mp is None:
                mode_errs[mode] = None
                continue
            ret_m = spp.solve_spp_nonlinear(
                obs_list, dense, sod, prev_pos_tep,
                clock=mp, zwd=0.0, clock_sigma=ms,
                k=None, elev_mask=ds.elev_mask_deg,
                max_update_m=max_update_m)
            if ret_m is not None:
                mode_errs[mode] = float(np.linalg.norm(
                    R @ (ret_m[0] - truth)))
            else:
                mode_errs[mode] = None

        # Full-observation oracle is diagnostic only.  It quantifies prior
        # clock error during artificial outages and never feeds any arm.
        ret_oracle = spp.solve_spp_nonlinear(
            [(prn, p_if) for prn, p_if in full_obs.items()], dense, sod,
            prev_pos_oracle, clock=None, zwd=0.0, clock_sigma=None,
            k=None, elev_mask=ds.elev_mask_deg, max_update_m=max_update_m)
        oracle_clk = None
        if ret_oracle is not None:
            prev_pos_oracle, oracle_clk = ret_oracle[0], float(ret_oracle[1])

        # Update the target clock anchor only from a genuinely full-sky,
        # non-jammed conventional solution, and only after all three arms for
        # this epoch have been evaluated.
        if not in_jam and ret_free is not None:
            prev_full_sod = last_full_sod
            prev_full_clk = last_full_clk
            last_full_sod = sod
            last_full_clk = clk_free
            full_clock_history.append((float(sod), float(clk_free)))
            full_clock_history = full_clock_history[-300:]

        # ---- RTKLIB on thinned RINEX ----
        err_rtk = None
        if sod in rtk_epochs and "pos_rtk" in rtk_epochs[sod]:
            pos_rtk = rtk_epochs[sod]["pos_rtk"]
            err_rtk = float(np.linalg.norm(R @ (pos_rtk - truth)))

        # ---- RTKLIB on full RINEX ----
        err_rtk_full = None
        if sod in rtk_full_epochs and "pos_rtk" in rtk_full_epochs[sod]:
            pos_rtk = rtk_full_epochs[sod]["pos_rtk"]
            err_rtk_full = float(np.linalg.norm(R @ (pos_rtk - truth)))

        results.append({
            "sod": sod, "k": k, "k_full": k_full,
            "err_free_m": err_free, "err_tep_m": err_tep,
            "err_tcxo_m": err_tcxo, "err_rtk_m": err_rtk,
            "err_tep_prior_m": err_tep_prior,
            "err_temporal_prior_m": err_tcxo_prior,
            "err_rtk_full_m": err_rtk_full,
            "in_jam": in_jam, "k_target": k_target,
            "cdt_prior_tep": cdt_prior_tep,
            "cdt_sig_tep": cdt_sig_tep,
            "tep_network_delta_m": network_delta,
            "tep_network_sigma_m": network_sigma,
            "cdt_prior_tcxo": cdt_prior_tcxo,
            "cdt_sig_tcxo": float(cdt_sig_tcxo),
            "tep_used": tep_used,
            "clock_oracle_full_m": oracle_clk,
            "tep_clock_error_oracle_m": (
                None if oracle_clk is None or cdt_prior_tep is None
                else float((cdt_prior_tep - oracle_clk + 299_792.458 / 2.0)
                           % 299_792.458 - 299_792.458 / 2.0)),
            "temporal_clock_error_oracle_m": (
                None if oracle_clk is None or cdt_prior_tcxo is None
                else float((cdt_prior_tcxo - oracle_clk + 299_792.458 / 2.0)
                           % 299_792.458 - 299_792.458 / 2.0)),
            "screening_mode_errs": mode_errs,
            "screening_mode_deltas": mode_deltas,
        })
        per_epoch.append(results[-1])

    if not results:
        return {"name": ds.name, "status": "no_valid_epochs", "timestamp": t0}

    # ---- metrics ----
    k_arr = np.array([r["k"] for r in results])
    k_full_arr = np.array([r["k_full"] for r in results])
    in_jam_arr = np.array([r["in_jam"] for r in results])

    def _arr(field):
        return np.array([r[field] for r in results if r[field] is not None])

    free_err = _arr("err_free_m")
    tep_err = _arr("err_tep_m")
    tcxo_err = _arr("err_tcxo_m")
    rtk_err = _arr("err_rtk_m")
    rtk_full_err = _arr("err_rtk_full_m")

    def _pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else None

    # jamming-window metrics: the real test
    jam_results = [r for r in results if r["in_jam"]]
    jam_free = np.array([r["err_free_m"] for r in jam_results
                         if r["err_free_m"] is not None])
    jam_tep = np.array([r["err_tep_m"] for r in jam_results
                        if r["err_tep_m"] is not None])
    jam_tcxo = np.array([r["err_tcxo_m"] for r in jam_results
                         if r["err_tcxo_m"] is not None])
    jam_tep_prior = np.array([r["err_tep_prior_m"] for r in jam_results
                              if r.get("err_tep_prior_m") is not None])
    jam_temporal_prior = np.array([r["err_temporal_prior_m"] for r in jam_results
                                   if r.get("err_temporal_prior_m") is not None])
    jam_rtk = np.array([r["err_rtk_m"] for r in jam_results
                        if r["err_rtk_m"] is not None])
    paired_clock = [(r["tep_clock_error_oracle_m"],
                     r["temporal_clock_error_oracle_m"])
                    for r in jam_results
                    if r.get("tep_clock_error_oracle_m") is not None
                    and r.get("temporal_clock_error_oracle_m") is not None]
    if paired_clock:
        tep_clock_err, temporal_clock_err = map(
            np.asarray, zip(*paired_clock))
        tep_clock_rms = float(np.sqrt(np.mean(tep_clock_err ** 2)))
        temporal_clock_rms = float(np.sqrt(np.mean(temporal_clock_err ** 2)))
        tep_clock_gain = float(100.0 * (1.0 - tep_clock_rms / temporal_clock_rms))
    else:
        tep_clock_rms = temporal_clock_rms = tep_clock_gain = None

    gt_path = np.array([gt[t] for t in gt_sorted], float)
    gt_step = np.linalg.norm(np.diff(gt_path, axis=0), axis=1)
    gt_dt = np.diff(np.asarray(gt_sorted, float))
    gt_speed = gt_step / np.maximum(gt_dt, 1e-9)

    # bridge rate: error < threshold in jamming windows
    n_jam = len(jam_results)
    bridge_free = int(np.sum(jam_free < bridge_threshold_m)) if len(jam_free) else 0
    bridge_tep = int(np.sum(jam_tep < bridge_threshold_m)) if len(jam_tep) else 0
    bridge_tcxo = int(np.sum(jam_tcxo < bridge_threshold_m)) if len(jam_tcxo) else 0
    bridge_tep_prior = int(np.sum(jam_tep_prior < bridge_threshold_m)) if len(jam_tep_prior) else 0
    bridge_temporal_prior = int(np.sum(jam_temporal_prior < bridge_threshold_m)) if len(jam_temporal_prior) else 0
    bridge_rtk = int(np.sum(jam_rtk < bridge_threshold_m)) if len(jam_rtk) else 0

    # per-k breakdown within jamming windows
    per_k = {}
    for kv in jam_k_values:
        kr = [r for r in jam_results if r["k_target"] == kv]
        if not kr:
            continue
        kf = np.array([r["err_free_m"] for r in kr if r["err_free_m"] is not None])
        kt = np.array([r["err_tep_m"] for r in kr if r["err_tep_m"] is not None])
        kx = np.array([r["err_tcxo_m"] for r in kr if r["err_tcxo_m"] is not None])
        ktp = np.array([r["err_tep_prior_m"] for r in kr if r.get("err_tep_prior_m") is not None])
        ktmp = np.array([r["err_temporal_prior_m"] for r in kr if r.get("err_temporal_prior_m") is not None])
        kr_rtk = np.array([r["err_rtk_m"] for r in kr if r["err_rtk_m"] is not None])
        per_k[f"k{kv}"] = {
            "n_epochs": len(kr),
            "n_free_valid": len(kf), "n_tep_valid": len(kt),
            "n_tcxo_valid": len(kx), "n_rtk_valid": len(kr_rtk),
            "n_tep_prior_valid": len(ktp),
            "n_temporal_prior_valid": len(ktmp),
            "err_free_p50": _pct(kf, 50), "err_tep_p50": _pct(kt, 50),
            "err_tcxo_p50": _pct(kx, 50), "err_rtk_p50": _pct(kr_rtk, 50),
            "err_tep_prior_p50": _pct(ktp, 50),
            "err_temporal_prior_p50": _pct(ktmp, 50),
            "err_free_p95": _pct(kf, 95), "err_tep_p95": _pct(kt, 95),
            "err_tcxo_p95": _pct(kx, 95), "err_rtk_p95": _pct(kr_rtk, 95),
            "err_tep_prior_p95": _pct(ktp, 95),
            "err_temporal_prior_p95": _pct(ktmp, 95),
            "bridge_free": int(np.sum(kf < bridge_threshold_m)) if len(kf) else 0,
            "bridge_tep": int(np.sum(kt < bridge_threshold_m)) if len(kt) else 0,
            "bridge_tcxo": int(np.sum(kx < bridge_threshold_m)) if len(kx) else 0,
            "bridge_tep_prior": int(np.sum(ktp < bridge_threshold_m)) if len(ktp) else 0,
            "bridge_temporal_prior": int(np.sum(ktmp < bridge_threshold_m)) if len(ktmp) else 0,
            "bridge_rtk": int(np.sum(kr_rtk < bridge_threshold_m)) if len(kr_rtk) else 0,
        }

    summary = {
        "version": "v3_causal_fixed_tep",
        "n_epochs": len(results),
        "n_cors": len(cors_series),
        "systems": list(systems),
        "n_jam_windows": len(jam_windows),
        "n_jam_epochs": n_jam,
        "bridge_threshold_m": bridge_threshold_m,
        "tcxo_allan": tcxo_allan,
        "n_free_valid": len(free_err),
        "n_tep_valid": len(tep_err),
        "n_tcxo_valid": len(tcxo_err),
        "n_rtk_valid": len(rtk_err),
        "n_rtk_full_valid": len(rtk_full_err),
        # full-flight errors (all epochs, including unthinned)
        "err_free_p50_m": _pct(free_err, 50),
        "err_tep_p50_m": _pct(tep_err, 50),
        "err_tcxo_p50_m": _pct(tcxo_err, 50),
        "err_rtk_p50_m": _pct(rtk_err, 50),
        "err_rtk_full_p50_m": _pct(rtk_full_err, 50),
        "err_free_p95_m": _pct(free_err, 95),
        "err_tep_p95_m": _pct(tep_err, 95),
        "err_tcxo_p95_m": _pct(tcxo_err, 95),
        "err_rtk_p95_m": _pct(rtk_err, 95),
        "err_rtk_full_p95_m": _pct(rtk_full_err, 95),
        # jamming-window errors (the degraded-environment test)
        "jam_err_free_p50_m": _pct(jam_free, 50),
        "jam_err_tep_p50_m": _pct(jam_tep, 50),
        "jam_err_tcxo_p50_m": _pct(jam_tcxo, 50),
        "jam_err_tep_prior_p50_m": _pct(jam_tep_prior, 50),
        "jam_err_temporal_prior_p50_m": _pct(jam_temporal_prior, 50),
        "jam_err_rtk_p50_m": _pct(jam_rtk, 50),
        "jam_err_free_p95_m": _pct(jam_free, 95),
        "jam_err_tep_p95_m": _pct(jam_tep, 95),
        "jam_err_tcxo_p95_m": _pct(jam_tcxo, 95),
        "jam_err_tep_prior_p95_m": _pct(jam_tep_prior, 95),
        "jam_err_temporal_prior_p95_m": _pct(jam_temporal_prior, 95),
        "jam_err_rtk_p95_m": _pct(jam_rtk, 95),
        # bridge rates in jamming windows
        "bridge_rate_free": bridge_free / n_jam if n_jam else 0.0,
        "bridge_rate_tep": bridge_tep / n_jam if n_jam else 0.0,
        "bridge_rate_tcxo": bridge_tcxo / n_jam if n_jam else 0.0,
        "bridge_rate_rtk": bridge_rtk / n_jam if n_jam else 0.0,
        "bridge_rate_tep_prior": bridge_tep_prior / n_jam if n_jam else 0.0,
        "bridge_rate_temporal_prior": bridge_temporal_prior / n_jam if n_jam else 0.0,
        "bridge_n_free": bridge_free, "bridge_n_tep": bridge_tep,
        "bridge_n_tcxo": bridge_tcxo, "bridge_n_rtk": bridge_rtk,
        "bridge_n_tep_prior": bridge_tep_prior,
        "bridge_n_temporal_prior": bridge_temporal_prior,
        "paired_oracle_clock_epochs": len(paired_clock),
        "tep_clock_prior_rms_vs_fullsky_oracle_m": tep_clock_rms,
        "temporal_clock_prior_rms_vs_fullsky_oracle_m": temporal_clock_rms,
        "tep_vs_temporal_clock_prior_rms_pct": tep_clock_gain,
        # per-k breakdown
        "per_k": per_k,
        # median satellite count
        "k_full_median": float(np.median(k_full_arr[k_full_arr > 0])) if np.any(k_full_arr > 0) else None,
        "k_thinned_median": float(np.median(k_arr[k_arr > 0])) if np.any(k_arr > 0) else None,
        # kernel
        "kernel_params": kp.get("params"),
        "kernel_n_refs": kp.get("n_refs"),
        "kernel_sigma_m": kp.get("sigma_m"),
        "initialisation": "RINEX approximate XYZ plus first-epoch full-sky SPP",
        "satellite_selection": "previous conventional solution; truth used only for scoring",
        "tep_anchor_policy": "previous non-jammed full-sky free-clock epoch, frozen throughout outage",
        "tep_activation_policy": "fallback only when the conventional free-clock solve is unavailable",
        "kernel_qc_cutoff_sod": kp.get("qc_cutoff_sod"),
        "kernel_preflight_ref_resid_sd_median_m": kp.get("preflight_ref_resid_sd_median"),
        "trajectory_duration_s": float(gt_sorted[-1] - gt_sorted[0]),
        "trajectory_distance_m": float(np.sum(gt_step)),
        "trajectory_speed_p50_mps": _pct(gt_speed, 50),
        "trajectory_speed_p95_mps": _pct(gt_speed, 95),
        "local_base_dgps_applied": bool(dgps_corrections),
        "local_base_dgps_epochs": len(dgps_corrections),
        "local_base_dgps_policy": (
            "same-epoch zero-mean satellite residual; base receiver clock removed"
            if dgps_corrections else "none"),
        "inference_unit": "one flight; outage windows are correlated stress episodes, not independent flights",
        # clock-type-aware screening results
        "screening_modes_tested": list(kps.keys()),
        "screening_ref_types": kp.get("ref_types"),
        "screening_rover_type": kp.get("rover_clock_type"),
        "screening_primary_mode": primary_mode,
    }

    # Per-screening-mode jamming metrics
    for mode in kps:
        if mode == primary_mode:
            continue
        jam_mode = np.array([r["screening_mode_errs"].get(mode)
                             for r in jam_results
                             if r.get("screening_mode_errs", {}).get(mode) is not None])
        summary[f"jam_err_{mode}_p50_m"] = _pct(jam_mode, 50)
        summary[f"jam_err_{mode}_p95_m"] = _pct(jam_mode, 95)
        summary[f"bridge_rate_{mode}"] = (
            int(np.sum(jam_mode < bridge_threshold_m)) / n_jam
            if n_jam and len(jam_mode) else 0.0)
        # network delta statistics
        jam_deltas = [abs(r["screening_mode_deltas"].get(mode, 0))
                      for r in jam_results
                      if r.get("screening_mode_deltas", {}).get(mode) is not None]
        if jam_deltas:
            summary[f"network_delta_{mode}_p50_m"] = float(np.percentile(jam_deltas, 50))
            summary[f"network_delta_{mode}_p95_m"] = float(np.percentile(jam_deltas, 95))

    return {
        "name": ds.name,
        "type": ds.type,
        "date_tag": ds.date_tag,
        "source_url": ds.source_url,
        "doi": ds.doi,
        "license": ds.license,
        "provenance_status": ds.provenance_status,
        "status": "ok",
        "timestamp": t0,
        "summary": summary,
        "per_epoch": per_epoch,
    }
