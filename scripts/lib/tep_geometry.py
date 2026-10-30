"""Shared SP3 orbit and geometry utilities.

Extracted from the original step_20_operational_ekf module so that active
pipeline steps (21, 24, 26, 35) can import them without depending on an
archived exploratory script.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SP3_DIR = PROJECT_ROOT / "data" / "rinex" / "sp3"

ELEV_MASK = np.radians(10.0)


def load_sp3(date_tag, systems=("G", "E", "R")):
    """Load satellite positions from a CODE SP3 file.

    Parameters
    ----------
    date_tag : str
        e.g. "2024100" for DOY 100 of 2024.
    systems : tuple of str
        GNSS system letters to include (G=GPS, E=Galileo, R=GLONASS).

    Returns
    -------
    dict[str, np.ndarray] or None
        {prn: (n, 4) array of (t_seconds, x, y, z) in ECEF metres}, or
        None if the file does not exist.
    """
    f = SP3_DIR / f"COD0MGXFIN_{date_tag}0000_01D_05M_ORB.SP3.gz"
    if not f.exists():
        return None
    raw: dict[str, list] = {}
    te = None
    for line in gzip.open(f, "rt"):
        if line.startswith("*  "):
            p = line.split()
            te = int(p[4]) * 3600 + int(p[5]) * 60 + float(p[6])
        elif line.startswith("P") and te is not None and line[1] in systems:
            try:
                x, y, z = float(line[4:18]), float(line[18:32]), float(line[32:46])
            except ValueError:
                continue
            if abs(x) > 1e-6:
                raw.setdefault(line[1:4], []).append((te, x * 1e3, y * 1e3, z * 1e3))
    return {s: np.array(sorted(v)) for s, v in raw.items() if len(v) >= 12}


def sat_grid(sp3, grid):
    """Interpolate every satellite to the filter time grid.

    Uses cubic spline interpolation (scipy.interpolate.CubicSpline) instead of
    high-degree polynomial fitting, which is numerically unstable for orbit
    data and can produce ringing with only ~10 sample points.

    Parameters
    ----------
    sp3 : dict
        Output of :func:`load_sp3`.
    grid : np.ndarray
        1-D array of time offsets in seconds.

    Returns
    -------
    dict[str, np.ndarray]
        {prn: (n, 3) ECEF metres}, with NaN where the grid point falls
        outside the SP3 time span.
    """
    from scipy.interpolate import CubicSpline
    out = {}
    for s, arr in sp3.items():
        ts, xyz = arr[:, 0], arr[:, 1:4]
        _, unique_idx = np.unique(ts, return_index=True)
        ts_u, xyz_u = ts[unique_idx], xyz[unique_idx]
        pos = np.full((len(grid), 3), np.nan)
        if len(ts_u) >= 4:
            splines = [CubicSpline(ts_u, xyz_u[:, k]) for k in range(3)]
            for gi, t in enumerate(grid):
                if t < ts_u[0] or t > ts_u[-1]:
                    continue
                pos[gi] = [splines[k](t) for k in range(3)]
        out[s] = pos
    return out


def elev_los(sat, sta):
    """Elevation, line-of-sight unit vector, and geometric range.

    Parameters
    ----------
    sat : np.ndarray
        Satellite ECEF position (3,).
    sta : np.ndarray
        Station ECEF position (3,).

    Returns
    -------
    (float, np.ndarray, float)
        Elevation angle (rad), LOS unit vector (3,), slant range (m).
    """
    up = sta / np.linalg.norm(sta)
    d = sat - sta
    r = np.linalg.norm(d)
    return np.arcsin(np.clip(d @ up / r, -1, 1)), d / r, r
