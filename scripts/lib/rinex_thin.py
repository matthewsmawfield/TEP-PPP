#!/usr/bin/env python3
"""
RINEX 3 observation thinning for raw-observation obstruction tests.

The TEP-PPP Phase-2 gate requires satellites to be removed *before* the
position solver sees them.  This module creates a new RINEX 3 observation
file (or rewrites an existing one) that keeps only a selected subset of
satellites at each epoch.  Selection is based on satellite elevation
computed from the station's approximate or surveyed ECEF position and a
precise orbit product (SP3).  The full-sky RTKLIB state is never used.

Usage:
    python -m scripts.lib.rinex_thin \
        data/rinex/obs/2024/100/BRST00FRA_R_20241000000_01D_30S_MO.crx.gz \
        data/rinex/sp3/COD0MGXFIN_20241000000_01D_05M_ORB.SP3.gz \
        thinned.rnx --k 5 --strategy highest --elev-min 10
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.interpolate import CubicSpline

C_M_PER_S = 299_792_458.0
ELEV_MASK_DEG = 10.0


def load_sp3(date_tag: str, sp3_dir: Path, systems: tuple[str, ...] = ("G", "E", "R", "C")):
    """Load a daily SP3 precise orbit file."""
    for pat in (f"COD0MGXFIN_{date_tag}0000_01D_05M_ORB.SP3.gz",
                f"COD0OPSFIN_{date_tag}0000_01D_05M_ORB.SP3.gz",
                f"IGS0OPSFIN_{date_tag}0000_01D_05M_ORB.SP3.gz"):
        f = sp3_dir / pat
        if f.exists():
            break
    else:
        return None
    raw: dict[str, list] = {}
    te = None
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith("*  "):
                p = line.split()
                te = int(p[4]) * 3600 + int(p[5]) * 60 + float(p[6])
            elif line.startswith("P") and te is not None and line[1] in systems:
                try:
                    x = float(line[4:18])
                    y = float(line[18:32])
                    z = float(line[32:46])
                except ValueError:
                    continue
                if abs(x) > 1e-6:
                    raw.setdefault(line[1:4], []).append((te, x * 1e3, y * 1e3, z * 1e3))
    out = {s: np.array(sorted(v)) for s, v in raw.items() if len(v) >= 12}
    return out if out else None


def sat_grid(sp3: dict, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate every satellite to the observation time grid -> {sat: (n,3) ecef m}."""
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


def elev_los(sat: np.ndarray, sta: np.ndarray):
    """Elevation (rad), LOS unit vector (sat->sta), and range."""
    up = sta / np.linalg.norm(sta)
    d = sat - sta
    r = np.linalg.norm(d)
    if r <= 0:
        return -np.inf, np.zeros(3), 0.0
    u = d / r
    e = np.arcsin(np.clip(u @ up, -1.0, 1.0))
    return e, u, r


def normalise_sv(sv: str) -> str:
    """RINEX body SV 'G 6' or 'G06' -> canonical SP3 form 'G06'."""
    if len(sv) < 2:
        return sv
    sys = sv[0]
    prn = sv[1:].strip()
    try:
        return f"{sys}{int(prn):02d}"
    except ValueError:
        return sv


def rinex_body_sv(sv: str) -> str:
    """Canonical 'G06' -> RINEX body form 'G 6'."""
    if len(sv) < 2:
        return sv
    sys = sv[0]
    try:
        prn = int(sv[1:])
        return f"{sys}{prn:2d}"
    except ValueError:
        return sv


def parse_header_position(header_lines: list[str]) -> np.ndarray | None:
    """Extract APPROX POSITION XYZ from a RINEX header."""
    for line in header_lines:
        if "APPROX POSITION" in line:
            toks = line.split()
            try:
                # RINEX 3 puts the numeric values first and the label at the end.
                return np.array([float(toks[0]), float(toks[1]), float(toks[2])])
            except (ValueError, IndexError):
                return None
    return None


def select_sats(elevations: dict[str, float], k: int, strategy: str,
                elev_min_deg: float = ELEV_MASK_DEG, rng=None) -> list[str]:
    """Choose up to k visible satellites from an elevation dictionary."""
    visible = sorted([s for s, e in elevations.items() if e >= np.radians(elev_min_deg)],
                     key=lambda s: -elevations[s])
    if strategy == "highest":
        return visible[:k]
    if strategy == "lowest":
        return list(reversed(visible))[:k]
    if strategy == "random":
        if rng is None:
            rng = np.random.default_rng()
        if len(visible) <= k:
            return visible
        idx = rng.choice(len(visible), k, replace=False)
        return [visible[i] for i in sorted(idx)]
    return visible[:k]


def thin_rinex_stream(reader, writer, sp3_grid: dict, k: int, strategy: str = "highest",
                      elev_min_deg: float = ELEV_MASK_DEG, rng=None,
                      station_ecef: np.ndarray | None = None,
                      system: str | None = None):
    """Copy a RINEX 3 observation stream, keeping only k satellites per epoch.

    reader: file-like object yielding RINEX 3.0x lines (already de-Hatanaka'd).
    writer: file-like object to write the thinned RINEX.
    sp3_grid: {sv: (n_epochs, 3)} ECEF satellite positions on the obs time grid.
    station_ecef: optional station ECEF position.  If None, extracted from header.
    """
    grid = None
    for s, pos in sp3_grid.items():
        grid = np.arange(0, len(pos) * 30.0, 30.0)[:len(pos)]
        if len(grid) > 0:
            break
    if grid is None or len(grid) == 0:
        raise ValueError("sp3_grid is empty")

    header_lines = []
    in_header = True
    for line in reader:
        if in_header:
            header_lines.append(line)
            if "END OF HEADER" in line:
                in_header = False
                if station_ecef is None:
                    station_ecef = parse_header_position(header_lines)
                for h in header_lines:
                    writer.write(h)
            continue

        if not line.startswith(">"):
            continue

        toks = line.split()
        year, month, day = int(toks[1]), int(toks[2]), int(toks[3])
        hour, minute = int(toks[4]), int(toks[5])
        seconds = float(toks[6])
        flag = int(toks[7])
        n_sats = int(toks[8])

        sod = hour * 3600 + minute * 60 + seconds
        gi = int(np.clip(round(sod / 30.0), 0, len(grid) - 1))

        sv_lines: dict[str, str] = {}
        sv_order: list[tuple[str, str]] = []
        elevations: dict[str, float] = {}
        for _ in range(n_sats):
            sv_line = next(reader)
            if not sv_line:
                break
            sv = sv_line[:3]
            sv_canon = normalise_sv(sv)
            if station_ecef is not None and sv_canon in sp3_grid:
                pos = sp3_grid[sv_canon][gi]
                if np.all(np.isfinite(pos)):
                    e, _, _ = elev_los(pos, station_ecef)
                    elevations[sv_canon] = e
            if sv_canon not in sv_lines:
                sv_lines[sv_canon] = sv_line
                sv_order.append((sv_canon, sv_line))

        # If a system is requested (or auto-detected), restrict the selection
        # to satellites from a single GNSS.  SPP on a thinned multi-system mix
        # of one or two satellites per system is rank-deficient; keeping one
        # dominant system makes the independent RTKLIB solution well-defined.
        if system is None:
            sys_counts = {}
            for sv, e in elevations.items():
                if e >= np.radians(elev_min_deg):
                    sys_counts[sv[0]] = sys_counts.get(sv[0], 0) + 1
            target = max(sys_counts, key=sys_counts.get) if sys_counts else None
            if target is not None and sys_counts[target] >= 4:
                elevations = {sv: e for sv, e in elevations.items() if sv.startswith(target)}
        elif system:
            elevations = {sv: e for sv, e in elevations.items() if sv.startswith(system)}

        keep_canon = select_sats(elevations, k, strategy, elev_min_deg=elev_min_deg, rng=rng)
        # Fall back to all visible if the chosen count is below 4.
        if len(keep_canon) < 4:
            keep_canon = [s for s, e in elevations.items() if e >= np.radians(elev_min_deg)]

        # Preserve original order of the satellite records.
        keep_set = set(keep_canon)
        ordered = [ln for sv, ln in sv_order if sv in keep_set]

        n_keep = len(ordered)
        # RINEX 3 epoch records are fixed-column.  Keep the receiver clock
        # placeholder at columns 41-55 to remain byte-compatible with readers
        # that parse fixed columns (georinex, RTKLIB).
        sec = f"{seconds:10.7f}"
        clk = f"{0.0:15.12f}"
        new_epoch = (f"> {year:4d} {month:2d} {day:2d} {hour:2d} {minute:2d} "
                     f"{sec}  {flag:1d} {n_keep:2d}        {clk}\n")
        writer.write(new_epoch)
        for ln in ordered:
            writer.write(ln)


def crx_to_text(crx_path: Path) -> str:
    """Decompress a Hatanaka-compressed .crx.gz and return uncompressed RINEX text."""
    if crx_path.suffix == ".gz":
        gz = subprocess.Popen(["gunzip", "-c", str(crx_path)], stdout=subprocess.PIPE)
        rnx = subprocess.run(["crx2rnx", "-"], stdin=gz.stdout, capture_output=True, text=True)
        gz.wait()
        if rnx.returncode != 0:
            raise RuntimeError(f"crx2rnx failed for {crx_path}: {rnx.stderr}")
        return rnx.stdout
    with open(crx_path, "r") as f:
        return f.read()


def thin_rinex_file(crx_path: Path, sp3_path: Path, out_path: Path, k: int,
                    strategy: str = "highest", elev_min_deg: float = ELEV_MASK_DEG,
                    station_ecef: np.ndarray | None = None, seed: int | None = None,
                    system: str | None = None):
    """High-level wrapper: thin one RINEX file and write a new .rnx."""
    sp3_dir = sp3_path.parent
    # RINEX filename contains YYYYDDDHHMM in field 2 (e.g. 20241000000).
    # SP3 orbit files use the 7-character YYYYDDD tag.
    dt_field = crx_path.stem.split("_")[2]
    date_tag = dt_field[:4] + dt_field[4:7]
    sp3 = load_sp3(date_tag, sp3_dir)
    if sp3 is None:
        raise FileNotFoundError(f"no SP3 orbit for {date_tag} in {sp3_dir}")

    # Build a 30-second grid aligned with typical daily RINEX OBS files.
    grid = np.arange(0.0, 86400.0, 30.0)
    sp3_grid = sat_grid(sp3, grid)

    rng = np.random.default_rng(seed) if strategy == "random" and seed is not None else None
    text = crx_to_text(crx_path)
    with open(out_path, "w") as fout:
        thin_rinex_stream(iter(text.splitlines(keepends=True)), fout, sp3_grid,
                          k=k, strategy=strategy, elev_min_deg=elev_min_deg,
                          rng=rng, station_ecef=station_ecef, system=system)


def main():
    ap = argparse.ArgumentParser(description="Thin a RINEX 3 observation file")
    ap.add_argument("crx", type=Path)
    ap.add_argument("sp3", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="highest", choices=("highest", "lowest", "random"))
    ap.add_argument("--elev-min", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system", default=None,
                    help="Restrict to one GNSS (G/R/E/C/S).  Default: auto-pick the dominant.")
    ap.add_argument("--station-xyz", default=None,
                    help="Optional station ECEF x,y,z in metres (overrides header)")
    a = ap.parse_args()
    station = None
    if a.station_xyz:
        station = np.array([float(x) for x in a.station_xyz.split(",")])
    thin_rinex_file(a.crx, a.sp3, a.out, a.k, a.strategy, a.elev_min, station, a.seed, a.system)
    print(f"thinned -> {a.out}")


if __name__ == "__main__":
    main()
