#!/usr/bin/env python3
"""
Unified GNSS clock-series loader, station registry, and QC.
=========================================================

Phase 0 of the evidence plan: one place that knows how to read every clock
product family in ``data/``, applies consistent quality control, and exposes
station geometry helpers.  All Phase 1 steps import from here so they run on
the same footing.

Product families
----------------
  ppp_processed     broadcast-ephemeris SPP, 30 s, ~76 global stations, DOY 100-106 2024
  ppp_dense         precise/MGEX/ionofree, 5 min, ~30 European stations, full-year 2024
  ppp_longitudinal  mixed: 2022 files carry precise_clock_bias_ns (5 min);
                    2024 files carry broadcast clock_bias_ns (30 s).
                    5 European stations, 2022-2024.

Schemas differ (``clock_bias_ns``+``sod`` vs ``*_clock_bias_ns``+``timestamps``);
:func:`load_series` normalises both.
"""
from __future__ import annotations

import math
import gzip
import glob
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"

# stations that are actually test fixtures, not real sites
_NOT_STATIONS = {"kill", "signal", "multi", "agency", "std", "test", "kill_signal",
                 "multi_agency", "test_step2"}

_PRODUCT_DIRS = {
    "broadcast": [DATA / "ppp_processed", DATA / "ppp_longitudinal", DATA / "ppp_dense"],
    "precise":   [DATA / "ppp_dense", DATA / "ppp_longitudinal"],
    "mgex":      [DATA / "ppp_dense"],
    "ionofree":  [DATA / "ppp_dense"],
}

# Precise analysis-centre receiver-clock products (RINEX CLK, "AR" records).
# Global, multi-GNSS, satellite-clock error removed -> the arm that can separate
# a distance-structured field from the shared-constellation footprint.
_CLK_DIR = DATA / "rinex" / "clk"
_CLK_PREFIX = {
    "clk_cod": "COD0MGXFIN", "clk_cod_ops": "COD0OPSFIN",
    "clk_gfz": "GFZ0OPSFIN",
    "clk_grg": "GRG0OPSFIN", "clk_jpl": "JPL0OPSFIN",
}
_clk_cache: dict = {}

_CLOCK_KEYS = {
    "broadcast": ["clock_bias_ns"],
    "precise":   ["precise_clock_bias_ns"],
    "mgex":      ["mgex_clock_bias_ns"],
    "ionofree":  ["ionofree_clock_bias_ns"],
}

C_M_PER_NS = 0.299792458
EARTH_RADIUS_KM = 6371.0


@dataclass
class Station:
    name: str
    ecef_km: np.ndarray          # (3,) mean ECEF in km
    lat_deg: float
    lon_deg: float

    @property
    def ecef_m(self) -> np.ndarray:
        return self.ecef_km * 1000.0


# --------------------------------------------------------------------------- geometry
# WGS84 ellipsoid parameters
_WGS84_A = 6378.137  # semi-major axis, km
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)  # semi-minor axis, km
_WGS84_E2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2  # first eccentricity squared
_WGS84_EP2 = (_WGS84_A ** 2 - _WGS84_B ** 2) / _WGS84_B ** 2  # second eccentricity squared


def ecef_to_geodetic(x, y, z):
    """WGS84 lat/lon in degrees from ECEF (any consistent unit).

    Uses the Bowring / Heikkinen closed-form conversion for accuracy
    at all latitudes.  The previous spherical approximation introduced
    up to ~0.2 deg latitude error and ~10 km great-circle distance
    error at long baselines.
    """
    # Inputs may be in m or km; work in km for consistency with EARTH_RADIUS_KM
    unit = 1.0
    if max(abs(x), abs(y), abs(z)) > 1e6:  # likely metres
        unit = 1e-3
    xk, yk, zk = x * unit, y * unit, z * unit
    lon = math.degrees(math.atan2(yk, xk))
    p = math.hypot(xk, yk)
    if p < 1e-9:
        return math.degrees(math.asin(1.0 if zk >= 0 else -1.0)), lon
    # Bowring's formula: numerator uses second eccentricity (e'²),
    # denominator uses first eccentricity (e²).
    theta = math.atan2(zk * _WGS84_A, p * _WGS84_B)
    st, ct = math.sin(theta), math.cos(theta)
    num = zk + _WGS84_EP2 * _WGS84_B * st ** 3
    den = p - _WGS84_E2 * _WGS84_A * ct ** 3
    lat = math.degrees(math.atan2(num, den))
    return lat, lon


def great_circle_km(a: Station, b: Station) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(d)))


def chord_km(a: Station, b: Station) -> float:
    return float(np.linalg.norm(a.ecef_km - b.ecef_km))


def baseline_azimuth_midpoint_deg(a: Station, b: Station) -> float:
    """Baseline orientation at the great-circle MIDPOINT, degrees from north
    (0-180 folded: orientation only).

    Use THIS for EW/NS sector classification, not baseline_azimuth_deg.  A great
    circle between two mid-latitude stations bends poleward, so the bearing where
    it leaves station A is not the orientation of the baseline as a whole: a true
    east-west 6,000 km pair can depart A on a bearing tens of degrees off 90.
    Measured on the 128-station CODE network, initial and midpoint bearings differ
    by 20.5 deg on average and put 35.8% of pairs in a different sector, with the
    discrepancy growing from 2 deg below 1,000 km to 26 deg beyond 10,000 km.  The
    initial-bearing convention therefore strips long east-west pairs out of the EW
    sector at exactly the distances where the decay tail fixes lambda, biasing
    lambda_EW downward and suppressing any measured EW/NS anisotropy.
    """
    la1, la2 = math.radians(a.lat_deg), math.radians(b.lat_deg)
    lo1, lo2 = math.radians(a.lon_deg), math.radians(b.lon_deg)
    Bx = math.cos(la2) * math.cos(lo2 - lo1)
    By = math.cos(la2) * math.sin(lo2 - lo1)
    lam = math.atan2(math.sin(la1) + math.sin(la2),
                     math.sqrt((math.cos(la1) + Bx) ** 2 + By ** 2))
    lom = lo1 + math.atan2(By, math.cos(la1) + Bx)
    dlo = lo2 - lom
    x = math.sin(dlo) * math.cos(la2)
    y = math.cos(lam) * math.sin(la2) - math.sin(lam) * math.cos(la2) * math.cos(dlo)
    return math.degrees(math.atan2(x, y)) % 180.0


def baseline_azimuth_deg(a: Station, b: Station) -> float:
    """Initial bearing a->b, degrees from north (0-180 folded: orientation only).

    BIASED for EW/NS sector classification on long baselines -- see
    baseline_azimuth_midpoint_deg, which should be preferred for anisotropy work.
    """
    la1, la2 = math.radians(a.lat_deg), math.radians(b.lat_deg)
    dlo = math.radians(b.lon_deg - a.lon_deg)
    x = math.sin(dlo) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    az = math.degrees(math.atan2(x, y)) % 180.0
    return az


def is_ew(az_deg: float, tol: float = 45.0) -> bool:
    return abs(az_deg - 90.0) <= tol / 2


def is_ns(az_deg: float, tol: float = 45.0) -> bool:
    return (az_deg <= tol / 2) or (az_deg >= 180.0 - tol / 2)


# --------------------------------------------------------------------------- QC
def robust_detrend(v: np.ndarray, clip_sigma: float = 8.0) -> np.ndarray:
    """Linear detrend + MAD outlier clip.  Returns mean-zero residual."""
    t = np.arange(len(v))
    good = np.isfinite(v)
    if good.sum() < 10:
        return np.full_like(v, np.nan, dtype=float)
    coef = np.polyfit(t[good], v[good], 1)
    r = v - np.polyval(coef, t)
    med = np.nanmedian(r)
    mad = 1.4826 * np.nanmedian(np.abs(r - med)) + 1e-12
    r = np.clip(r, med - clip_sigma * mad, med + clip_sigma * mad)
    r = r - np.nanmean(r)
    return r


def detect_clock_jumps(v: np.ndarray, n_sigma: float = 12.0) -> int:
    """Count first-difference outliers (clock resets / ms jumps)."""
    d = np.diff(v[np.isfinite(v)])
    if len(d) < 10:
        return 0
    s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
    return int(np.sum(np.abs(d - np.median(d)) > n_sigma * s))


# --------------------------------------------------------------------------- loading
def _read_npz(path: Path):
    try:
        return np.load(path)
    except Exception:
        return None


def _station_from_npz(name: str, z) -> Station | None:
    for kx, ky, kz in (("x_mean", "y_mean", "z_mean"),
                       ("approx_x", "approx_y", "approx_z")):
        if kx in z:
            x, y, zc = float(z[kx]), float(z[ky]), float(z[kz])
            if abs(x) > 1e6:
                lat, lon = ecef_to_geodetic(x, y, zc)
                return Station(name, np.array([x, y, zc]) / 1000.0, lat, lon)
    return None


def _parse_clk_file(path: Path) -> dict:
    """Parse a RINEX CLK .gz -> {sta4: {"t": (n,), "clk_ns": (n,), "station": Station}}."""
    if path in _clk_cache:
        return _clk_cache[path]
    coords: dict[str, np.ndarray] = {}
    recs: dict[str, list] = {}
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        in_hdr = True
        for line in f:
            if in_hdr:
                if "SOLN STA NAME / NUM" in line:
                    toks = line.split("SOLN STA NAME / NUM")[0].split()
                    nm4 = toks[0][:4].upper() if toks else ""
                    try:
                        x, y, z = (float(toks[-3]) * 1e-3, float(toks[-2]) * 1e-3,
                                   float(toks[-1]) * 1e-3)
                        if abs(x) + abs(y) + abs(z) > 1e5:
                            coords[nm4] = np.array([x, y, z])
                    except (ValueError, IndexError):
                        pass
                if "END OF HEADER" in line:
                    in_hdr = False
                continue
            if not line.startswith("AR "):
                continue
            p = line.split()
            # AR name yyyy mm dd hh mm ss nval bias [sigma]
            if len(p) < 10:
                continue
            nm4 = p[1][:4].upper()
            try:
                hh, mm = int(p[5]), int(p[6])
                ss = float(p[7]); bias = float(p[9])
            except (ValueError, IndexError):
                continue
            sod = hh * 3600 + mm * 60 + ss
            recs.setdefault(nm4, []).append((sod, bias * 1e9))
    out = {}
    for nm4, rows in recs.items():
        if nm4 not in coords or len(rows) < 100:
            continue
        rows.sort()
        t = np.array([r[0] for r in rows], float)
        clk = np.array([r[1] for r in rows], float)
        x, y, z = coords[nm4]
        lat, lon = ecef_to_geodetic(x, y, z)
        out[nm4] = {"t": t, "clk_ns": clk, "station": Station(nm4, np.array([x, y, z]) / 1000.0, lat, lon)}
    _clk_cache[path] = out
    return out


def _clk_path(product: str, date_tag: str) -> Path | None:
    pfx = _CLK_PREFIX[product]
    # Prefer 30S, fall back to 05M if that's what was downloaded
    for pat in (f"{pfx}_{date_tag}0000_01D_30S_CLK.CLK.gz",
                f"{pfx}_{date_tag}0000_01D_05M_CLK.CLK.gz"):
        p = _CLK_DIR / pat
        if p.exists():
            return p
    return None


def available_dates(product: str, year: int | None = None) -> list[str]:
    if product in _CLK_PREFIX:
        pfx = _CLK_PREFIX[product]
        out = []
        for f in glob.glob(str(_CLK_DIR / f"{pfx}_*_CLK.CLK.gz")):
            m = re.search(r"_(\d{7})0000_", Path(f).name)
            if m and (year is None or m.group(1).startswith(str(year))):
                out.append(m.group(1))
        return sorted(out)
    dates: set[str] = set()
    for d in _PRODUCT_DIRS[product]:
        for f in glob.glob(str(d / "*_*.npz")):
            tag = Path(f).stem.split("_")[-1]
            if len(tag) == 7 and tag.isdigit():
                if year is None or tag.startswith(str(year)):
                    dates.add(tag)
    return sorted(dates)


def load_series(product: str, date_tag: str, stations: list[str] | None = None,
                min_valid_frac: float = 0.85, max_jumps: int = 6):
    """
    Load one day of clock residuals for a product family.

    Returns dict: name -> {"resid": (n,) detrended ns, "raw": (n,) ns,
                            "t": (n,) seconds, "station": Station, "nsat": (n,) or None}
    Only stations passing QC are returned; series are aligned to a common grid.
    """
    if product in _CLK_PREFIX:
        return _load_clk_series(product, date_tag, stations, min_valid_frac, max_jumps)

    dirs = _PRODUCT_DIRS[product]
    clock_keys = _CLOCK_KEYS[product]
    want = None if stations is None else set(stations)
    out: dict[str, dict] = {}

    files: dict[str, Path] = {}
    for d in dirs:
        for f in glob.glob(str(d / f"*_{date_tag}.npz")):
            nm = Path(f).stem.split("_")[0]
            if nm in _NOT_STATIONS or (want is not None and nm not in want):
                continue
            files.setdefault(nm, Path(f))          # first dir wins

    for nm, path in files.items():
        z = _read_npz(path)
        if z is None:
            continue
        clk = next((np.asarray(z[k], float) for k in clock_keys if k in z), None)
        if clk is None:
            continue
        if "timestamps" in z:
            t = np.asarray(z["timestamps"], float)
        elif "sod" in z:
            t = np.asarray(z["sod"], float)
        else:
            t = np.arange(len(clk)) * 30.0
        n = min(len(clk), len(t))
        clk, t = clk[:n], t[:n]
        if np.mean(np.isfinite(clk)) < min_valid_frac:
            continue
        if detect_clock_jumps(clk, 12.0) > max_jumps:
            continue
        sta = _station_from_npz(nm, z)
        if sta is None:
            continue
        resid = robust_detrend(clk)
        if not np.isfinite(resid).any() or np.nanstd(resid) < 1e-6:
            continue
        nsat = np.asarray(z["nsat"], float)[:n] if "nsat" in z else None
        out[nm] = {"resid": resid, "raw": clk, "t": t, "station": sta, "nsat": nsat}

    if not out:
        return out
    # Align every station onto the common integer-second grid rather than blind
    # truncation, so one short series does not clip the whole network.
    lens = sorted(len(v["resid"]) for v in out.values())
    ref = next(v for v in out.values() if len(v["resid"]) == lens[-1])
    grid = np.round(ref["t"]).astype(np.int64)
    keep = {}
    for nm, v in out.items():
        tt = np.round(v["t"]).astype(np.int64)
        idx = {int(x): i for i, x in enumerate(tt)}
        sel = np.array([idx.get(int(g), -1) for g in grid])
        hit = sel >= 0
        if hit.mean() < 0.80:                       # too little overlap with the grid
            continue
        for k in ("resid", "raw"):
            a = np.full(len(grid), np.nan)
            a[hit] = v[k][sel[hit]]
            v[k] = a
        # The time axis is the common grid; it must not contain NaN.
        v["t"] = grid.astype(float)
        if v["nsat"] is not None:
            a = np.full(len(grid), np.nan)
            a[hit] = v["nsat"][sel[hit]]
            v["nsat"] = a
        # fill small gaps for correlation stability
        v["resid"] = _interp_nan(v["resid"])
        keep[nm] = v
    return keep


def _interp_nan(a: np.ndarray) -> np.ndarray:
    m = np.isfinite(a)
    if m.all() or m.sum() < 10:
        return a
    x = np.arange(len(a))
    a = a.copy()
    a[~m] = np.interp(x[~m], x[m], a[m])
    return a


def _load_clk_series(product, date_tag, stations, min_valid_frac, max_jumps):
    path = _clk_path(product, date_tag)
    if path is None:
        return {}
    parsed = _parse_clk_file(path)
    want = None if stations is None else set(stations)
    # Determine the common grid cadence.  CLK files may contain a mix of
    # 30 s and 300 s stations; the grid must match the MAJORITY cadence
    # so most stations have native hits.  The filename advertises the
    # finest cadence, but most stations are typically at 300 s.
    # Use 300 s as the common grid; 30 s stations are decimated but still
    # valid (their hit fraction will be 1.0 on the 300 s grid).
    grid = np.arange(0, 86400, 300.0)
    out = {}
    for nm, v in parsed.items():
        if want is not None and nm not in want:
            continue
        idx = {int(round(x)): i for i, x in enumerate(v["t"])}
        sel = np.array([idx.get(int(g), -1) for g in grid])
        hit = sel >= 0
        if hit.mean() < min_valid_frac:
            continue
        raw = np.full(len(grid), np.nan)
        raw[hit] = v["clk_ns"][sel[hit]]
        if detect_clock_jumps(raw, 12.0) > max_jumps:
            continue
        # For precise CLK products, use mean-removal only (not robust_detrend).
        # robust_detrend fits a linear trend + MAD clip; on 30 s data with
        # a strong clock drift this removes almost all variance, leaving
        # only numerical noise.  The bandpass in build_matrix handles the
        # detrending properly.
        interp = _interp_nan(raw)
        resid = interp - np.nanmean(interp)
        if not np.isfinite(resid).any() or np.nanstd(resid) < 1e-6:
            continue
        out[nm] = {"resid": resid, "raw": raw, "t": grid.copy(),
                   "station": v["station"], "nsat": None}
    return out


def load_multi_day(product: str, date_tags: list[str], stations: list[str] | None = None,
                   **qc):
    """Yield (date_tag, series_dict) for each day that has >=2 usable stations."""
    for dt in date_tags:
        s = load_series(product, dt, stations, **qc)
        if len(s) >= 2:
            yield dt, s


def registry(product: str = "broadcast", year: int = 2024, min_days: int = 5,
             stations: list[str] | None = None) -> dict[str, Station]:
    """Stations with >= min_days usable days for a product/year, with geometry."""
    dates = available_dates(product, year)
    seen: dict[str, Station] = {}
    count: dict[str, int] = {}
    for _, s in load_multi_day(product, dates, stations):
        for nm, v in s.items():
            seen.setdefault(nm, v["station"])
            count[nm] = count.get(nm, 0) + 1
    return {nm: st for nm, st in seen.items() if count.get(nm, 0) >= min_days}


def load_meo_series(year: int, doy: int, prns: list[str]) -> dict:
    """
    Parse a daily SP3 orbit file for MEO kinematics.
    Returns: dict[prn] -> {"t": array, "station": object with trajectory_km and velocity_kms, "resid": array}

    SP3 files mark unavailable satellites with position [0,0,0] and clock
    999999.999999.  These entries are filtered out so that velocities are
    computed only from genuine orbital positions.
    """
    sp3_path = DATA / "rinex" / "sp3" / f"COD0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"
    if not sp3_path.exists():
        return {}

    op = gzip.open if sp3_path.suffix == ".gz" else open

    # Collect (epoch_sod, [x,y,z], clock) per PRN, skipping unavailable entries.
    epoch_sods = []
    raw_positions: dict[str, list[tuple[int, list[float]]]] = {p: [] for p in prns}
    raw_clocks: dict[str, list[tuple[int, float]]] = {p: [] for p in prns}

    current_sod = -1
    with op(sp3_path, "rt") as f:
        for line in f:
            if line.startswith("*  "):
                parts = line.split()
                h, m, s = int(parts[4]), int(parts[5]), float(parts[6])
                current_sod = h * 3600 + m * 60 + s
                epoch_sods.append(current_sod)
            elif line.startswith("P"):
                sat_id = line[1:4].strip()
                if sat_id in prns and current_sod != -1:
                    parts = line.split()
                    # SP3 coordinate format: PRN x y z clock
                    # We want only the coordinates (km) and clock (us)
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    if x == 0.0 and y == 0.0 and z == 0.0:
                        continue
                        
                    raw_positions[sat_id].append((current_sod, [x, y, z]))
                    
                    if len(parts) > 4:
                        clk = float(parts[4])
                        if clk != 999999.999999:
                            raw_clocks[sat_id].append((current_sod, clk * 1000.0)) # convert us to ns

    epoch_sods = sorted(list(set(epoch_sods)))
    if not epoch_sods:
        return {}

    class KinematicStation:
        pass

    out = {}
    for prn in prns:
        entries = raw_positions[prn]
        clk_entries = raw_clocks[prn]
        if len(entries) < 4:
            continue

        # Match clocks and positions by epoch_sod.
        # SP3 files can contain a duplicate SOD=0.0 at the day boundary
        # (the last epoch of the previous day AND the first epoch of the
        # current day both have SOD=0).  Using dict() would keep the LAST
        # entry, which may be from the wrong day.  Instead, keep the FIRST
        # occurrence of each SOD to preserve the correct day's data.
        sod_to_pos = {}
        for sod, pos in entries:
            if sod not in sod_to_pos:
                sod_to_pos[sod] = pos
        sod_to_clk = {}
        for sod, clk in clk_entries:
            if sod not in sod_to_clk:
                sod_to_clk[sod] = clk

        t_arr = []
        pos_arr = []
        clk_arr = []

        for sod in epoch_sods:
            if sod in sod_to_pos and sod in sod_to_clk:
                t_arr.append(sod)
                pos_arr.append(sod_to_pos[sod])
                clk_arr.append(sod_to_clk[sod])
                
        if len(t_arr) < 4:
            continue
            
        t_arr = np.array(t_arr, dtype=float)
        pos_arr = np.array(pos_arr, dtype=float)
        clk_arr = np.array(clk_arr, dtype=float)
        
        dt = float(np.median(np.diff(t_arr)))
        if not np.isfinite(dt) or dt <= 0:
            continue
        # Velocity via central difference (km/s)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:-1] = (pos_arr[2:] - pos_arr[:-2]) / (2.0 * dt)
        vel_arr[0] = (pos_arr[1] - pos_arr[0]) / dt
        vel_arr[-1] = (pos_arr[-1] - pos_arr[-2]) / dt

        sta = KinematicStation()
        sta.name = prn
        sta.trajectory_km = pos_arr
        sta.velocity_kms = vel_arr
        out[prn] = {"t": t_arr, "station": sta, "resid": clk_arr}

    return out

if __name__ == "__main__":
    for prod in ("broadcast", "precise"):
        reg = registry(prod, 2024, min_days=3)
        lats = [s.lat_deg for s in reg.values()]
        print(f"{prod:10s}: {len(reg):3d} stations   lat range [{min(lats):+.0f}, {max(lats):+.0f}]")
        if reg:
            import itertools
            mx = max(great_circle_km(a, b) for a, b in itertools.combinations(reg.values(), 2))
            print(f"{'':10s}  max baseline {mx:.0f} km")
