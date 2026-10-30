"""Precise satellite orbit (SP3) and clock (CLK) product loading and interpolation.

Broadcast ephemeris has ~1-2 m orbit error and ~5-10 ns clock error
(~1.5-3 m range error).  IGS MGEX precise products (CODE, GFZ, etc.)
reduce this to ~2-5 cm orbit and ~0.1-0.3 ns clock, which is essential
for cm-level PPP.  This module loads SP3 and CLK files, interpolates
satellite positions and clock corrections to arbitrary times, and
provides a unified interface that ``rinex_pseudorange.get_measurements``
uses when precise products are available.

SP3 files give satellite positions at 5-minute or 15-minute intervals;
CLK files give clock corrections at 30-second or 5-minute intervals.
Both are interpolated (cubic spline for positions, linear for clocks).
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

C = 299_792_458.0


def _gps_week_seconds(year: int, month: int, day: int,
                      hour: int = 0, minute: int = 0, sec: float = 0.0
                      ) -> tuple[int, float]:
    """Convert calendar date/time to GPS week number and seconds of week.

    Mirrors ``rinex_pseudorange._gps_week_seconds`` so this module can
    compute SOW for multi-day cached products without importing from
    ``rinex_pseudorange`` (which would create a circular dependency).
    """
    import datetime as dt
    gps_epoch = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime(year, month, day, hour, minute,
                        int(sec), int((sec % 1) * 1e6),
                        tzinfo=dt.timezone.utc)
    except ValueError:
        t = dt.datetime(year, month, day, hour, minute, 0,
                        tzinfo=dt.timezone.utc)
    delta = t - gps_epoch
    total_sec = delta.total_seconds()
    week = int(total_sec // 604800)
    sow = total_sec % 604800
    return week, sow


def _glob_product(directory: Path, prefix: str, date_tag: str,
                  suffix: str) -> Path | None:
    """Find a precise-product file, tolerating variable issue-time and
    duration fields in the filename.

    Final (FIN) products use ``_01D_`` and a fixed ``0000`` issue time.
    Rapid (RAP) products use ``_01D_`` and ``0000``.  Ultra-rapid (ULT)
    products use ``_02D_`` and a 4-digit issue time that varies every 6 h
    (0000, 0600, 1200, 1800), so we glob for any match and pick the
    latest issue time (best predicted quality for the end of the day).
    """
    # Exact FIN/RAP patterns (fixed 0000 issue time, _01D_ duration)
    for dur in ("01D", "02D"):
        cand = directory / f"{prefix}_{date_tag}0000_{dur}_{suffix}"
        if cand.exists():
            return cand
    # ULT patterns: glob for any issue time and duration, pick the latest.
    # Duration field is 3 chars (e.g. "01D", "02D"), issue time is 4 chars.
    matches = sorted(directory.glob(f"{prefix}_{date_tag}????_???_{suffix}"))
    if matches:
        return matches[-1]
    return None


def load_sp3(date_tag: str, sp3_dir: Path,
             systems: tuple[str, ...] = ("G", "E", "R", "C"),
             target_year: int | None = None,
             target_doy: int | None = None):
    """Load a daily SP3 precise orbit file.

    Returns dict: {sat_id: np.array(N, 4)} where columns are
    [sod, x_km*1000, y_km*1000, z_km*1000] in metres, or None if no
    file is found.  Reuses the parser from rinex_thin but converts
    km to metres.

    Product preference: final (FIN) > rapid (RAP) > ultra-rapid (ULT).
    Ultra-rapid orbits are ~5-20 cm (vs ~2-5 cm final), still 5-10x
    better than broadcast (~1-2 m), and are the only real-time option.

    When ``target_year``/``target_doy`` are given, only epochs whose
    calendar date matches that day-of-year are kept.  This is essential
    for ultra-rapid (_02D_) files, which span two or three days: without
    it the seconds-of-day index collides between days and the
    interpolation returns positions from the wrong day, producing
    satellite positions off by thousands of kilometres.
    """
    import datetime as _dt
    # Resolve the target day for filtering.
    if target_year is None or target_doy is None:
        target_year = int(date_tag[:4])
        target_doy = int(date_tag[4:7])
    target_date = _dt.date(target_year, 1, 1) + _dt.timedelta(
        days=target_doy - 1)

    f = None
    for ac, typ in [("COD0MGX", "FIN"), ("COD0OPS", "FIN"),
                    ("IGS0OPS", "FIN"), ("GFZ0OPS", "FIN"),
                    ("WUM0MGX", "FIN"),
                    ("COD0OPS", "RAP"), ("IGS0OPS", "RAP"),
                    ("GFZ0OPS", "RAP"),
                    ("COD0OPS", "ULT"), ("IGS0OPS", "ULT"),
                    ("GRG0OPS", "ULT"), ("GFZ0OPS", "ULT")]:
        f = _glob_product(sp3_dir, f"{ac}{typ}", date_tag,
                          "05M_ORB.SP3.gz")
        if f:
            break
    if f is None:
        return None

    raw: dict[str, list] = {}
    te = None
    te_valid = False
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith("*  "):
                p = line.split()
                te = int(p[4]) * 3600 + int(p[5]) * 60 + float(p[6])
                # Filter to the target calendar date for multi-day files.
                rec_date = _dt.date(int(p[1]), int(p[2]), int(p[3]))
                te_valid = (rec_date == target_date)
            elif line.startswith("P") and te is not None and te_valid and line[1] in systems:
                try:
                    x = float(line[4:18])
                    y = float(line[18:32])
                    z = float(line[32:46])
                except ValueError:
                    continue
                if abs(x) > 1e-6:
                    raw.setdefault(line[1:4], []).append(
                        (te, x * 1e3, y * 1e3, z * 1e3))
    out = {}
    for s, v in raw.items():
        if len(v) < 4:
            continue
        arr = np.array(sorted(v))
        # Remove duplicate timestamps (SP3 files sometimes repeat the
        # last epoch of one day as the first of the next)
        t = arr[:, 0]
        unique_mask = np.concatenate(([True], t[1:] != t[:-1]))
        arr = arr[unique_mask]
        if len(arr) >= 4:
            out[s] = arr
    return out if out else None


def load_clk(date_tag: str, clk_dir: Path,
             systems: tuple[str, ...] = ("G", "E", "C"),
             target_year: int | None = None,
             target_doy: int | None = None):
    """Load a daily CLK precise satellite clock file.

    Returns dict: {sat_id: np.array(N, 2)} where columns are
    [sod, clock_correction_seconds], or None if no file is found.

    CLK files (RINEX 3.x clock format) contain AS (satellite) records:
    'AS G01 2024 04 09 00 00 30.000  ...  clock_bias_seconds'

    Product preference: final (FIN, 30 s) > rapid (RAP, 30 s) >
    ultra-rapid (ULT, 5 min).  Ultra-rapid clocks are ~0.3-0.5 ns
    (vs ~0.02-0.05 ns final), still far better than broadcast (~5-10 ns).

    When ``target_year``/``target_doy`` are given, only records whose
    calendar date matches that day-of-year are kept.  This is essential
    for ultra-rapid (_02D_) files, which span two days: without it the
    seconds-of-day index collides between days and the wrong epoch is
    returned.  The file is located by ``date_tag`` (which may be the
    previous day for an ULT fallback), but records are filtered to the
    target day.
    """
    import datetime as _dt
    # Resolve the target day for filtering.  Default to the date_tag's
    # own day (backward compatible for single-day FIN/RAP files).
    if target_year is None or target_doy is None:
        target_year = int(date_tag[:4])
        target_doy = int(date_tag[4:7])
    target_date = _dt.date(target_year, 1, 1) + _dt.timedelta(
        days=target_doy - 1)

    candidates = [
            ("COD0MGX", "FIN", "30S"), ("COD0OPS", "FIN", "30S"),
            ("IGS0OPS", "FIN", "30S"), ("GFZ0OPS", "FIN", "30S"),
            ("WUM0MGX", "FIN", "30S"),
            ("COD0MGX", "FIN", "05M"), ("COD0OPS", "FIN", "05M"),
            ("COD0OPS", "RAP", "30S"), ("IGS0OPS", "RAP", "05M"),
            ("GFZ0OPS", "RAP", "30S"),
            ("COD0OPS", "ULT", "05M"), ("GRG0OPS", "ULT", "05M"),
            ("GFZ0OPS", "ULT", "05M"), ("IGS0OPS", "ULT", "05M"),
            ("SHA0MGX", "ULT", "05M"), ("WUM0MGX", "ULT", "05M")]
    for ac, typ, interval in candidates:
        f = _glob_product(clk_dir, f"{ac}{typ}", date_tag,
                          f"{interval}_CLK.CLK.gz")
        if f is None:
            continue

        raw: dict[str, list] = {}
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if not line.startswith("AS "):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                sat_id = parts[1]
                if sat_id[0] not in systems:
                    continue
                try:
                    year = int(parts[2])
                    month = int(parts[3])
                    day = int(parts[4])
                    hour = int(parts[5])
                    minute = int(parts[6])
                    sec = float(parts[7])
                    # Filter to the target calendar date for multi-day files.
                    rec_date = _dt.date(year, month, day)
                    if rec_date != target_date:
                        continue
                    # Format: AS SAT year mo day hr min sec  nval  bias  [rate]
                    # nval (parts[8]) is 1 or 2; clock bias is parts[9].
                    # Using parts[-1] would grab the rate, not the bias.
                    clk_bias = float(parts[9].replace("D", "E").replace("d", "e"))
                except (ValueError, IndexError):
                    continue
                sod = hour * 3600 + minute * 60 + sec
                raw.setdefault(sat_id, []).append((sod, clk_bias))

        out = {}
        for s, v in raw.items():
            if len(v) < 2:
                continue
            arr = np.array(sorted(v))
            t = arr[:, 0]
            unique_mask = np.concatenate(([True], t[1:] != t[:-1]))
            arr = arr[unique_mask]
            if len(arr) >= 2:
                out[s] = arr
        if out:
            return out
        # This file had no records for the target date (e.g. a RAP file
        # for the previous day when we need today's ULT coverage).
        # Continue to the next candidate.

    return None


class PreciseProducts:
    """Interpolated precise satellite positions and clock corrections.

    Uses cubic spline interpolation for satellite positions (smooth
    orbital motion) and linear interpolation for clock corrections
    (piecewise-linear broadcast/precise clock model).

    Satellite clock corrections from CLK files are referenced to the
    ionosphere-free combination, so they do NOT include TGD.  This
    matches the IF combination used in dual-frequency PPP.
    """

    def __init__(self, sp3: dict, clk: dict | None = None):
        self._sp3 = sp3
        self._clk = clk
        # Day this product set describes, and the day the caller is currently
        # asking about.  Both None keeps the legacy behaviour.
        self._doy = getattr(self, "_doy", None)
        self._epoch_doy = None
        # When True, the time index is absolute GPS seconds
        # (week*604800 + SOW) instead of seconds-of-day.  The absolute
        # axis is unambiguous across UTC midnight AND the GPS week
        # boundary, so the day check in satellite_clock is unnecessary
        # and multi-day cached products (ultra-rapid) can be loaded in a
        # single PreciseProducts.
        self._use_sow = getattr(self, "_use_sow", False)
        self._pos_splines: dict[str, CubicSpline] = {}
        self._vel_splines: dict[str, CubicSpline] = {}
        self._pos_bounds: dict[str, tuple[float, float]] = {}
        self._clk_splines: dict[str, CubicSpline] = {}
        self._clk_vel_splines: dict[str, CubicSpline] = {}
        self._clk_bounds: dict[str, tuple[float, float]] = {}
        self._build_splines()

    def _build_splines(self):
        for sat_id, arr in self._sp3.items():
            if len(arr) < 4:
                continue
            t = arr[:, 0]
            xyz = arr[:, 1:4]
            try:
                spline = CubicSpline(t, xyz)
                self._pos_splines[sat_id] = spline
                # Analytic derivative of the orbit spline: satellite velocity
                # in ECEF, needed for the periodic relativistic clock term.
                self._vel_splines[sat_id] = spline.derivative()
                self._pos_bounds[sat_id] = (t[0], t[-1])
            except Exception:
                pass
            # Build SP3 clock spline for fallback when CLK file doesn't
            # cover the observation time.  The SP3 clock is in seconds
            # (column 5, converted from microseconds).  SP3 clocks are
            # less accurate than CLK (~0.1-1 ns vs ~0.01 ns) but far
            # better than broadcast (~10-100 ns) and cover 48h vs 12h.
            if arr.shape[1] >= 5:
                try:
                    clk_vals = arr[:, 4]
                    # Filter out zero/invalid clock values
                    valid = np.abs(clk_vals) > 1e-12
                    if np.sum(valid) >= 4:
                        clk_spline = CubicSpline(t[valid], clk_vals[valid])
                        self._clk_splines[sat_id] = clk_spline
                        self._clk_bounds[sat_id] = (
                            float(t[valid][0]), float(t[valid][-1]))
                        # Store the derivative for linear extrapolation
                        self._clk_vel_splines[sat_id] = clk_spline.derivative()
                except Exception:
                    pass

    def has_satellite(self, sat_id: str) -> bool:
        return sat_id in self._pos_splines

    def is_within_bounds(self, sat_id: str, sod: float) -> bool:
        """Check if sod is within the SP3 interpolation bounds (not extrapolated)."""
        if sat_id not in self._pos_bounds:
            return False
        t0, t1 = self._pos_bounds[sat_id]
        return t0 <= sod <= t1

    def is_clk_within_bounds(self, sat_id: str, sod: float) -> bool:
        """Check if sod is within the CLK interpolation bounds (not extrapolated)."""
        # Check the CLK file first
        if self._clk is not None and sat_id in self._clk:
            arr = self._clk[sat_id]
            t = arr[:, 0]
            return t[0] <= sod <= t[-1]
        # Check SP3 clock splines
        clk_splines = getattr(self, "_clk_splines", {})
        if sat_id in clk_splines:
            t0, t1 = self._clk_bounds[sat_id]
            return t0 <= sod <= t1
        return False

    def satellite_position(self, sat_id: str, sod: float) -> np.ndarray | None:
        """Interpolated ECEF position (metres) at time sod (seconds of day).

        Returns None if the satellite is not in the SP3 file or the time
        is outside the interpolation bounds.
        """
        if sat_id not in self._pos_splines:
            return None
        t0, t1 = self._pos_bounds[sat_id]
        # Allow extrapolation up to 8 hours beyond the SP3 bounds using
        # LINEAR extrapolation (pos + vel * dt).  Cubic spline
        # extrapolation diverges wildly beyond the bounds (tested: 87M m
        # error at 5h beyond).  Linear extrapolation from the boundary
        # velocity is accurate to ~1-3 km over 5h — far worse than
        # interpolation but far better than spline divergence, and
        # sufficient for the EKF's median/MAD statistics to reject
        # outlier satellites.  RTKLIB uses a similar linear extrapolation
        # in peph2pos() for positions beyond the SP3 bounds.
        MAX_POS_EXTRAP_S = 8 * 3600  # 8 h max extrapolation
        if sod < t0 - 30 or sod > t1 + MAX_POS_EXTRAP_S:
            return None
        if t0 <= sod <= t1:
            return self._pos_splines[sat_id](sod)
        # Linear extrapolation using the spline derivative at the boundary
        if sat_id in self._vel_splines:
            if sod > t1:
                pos = self._pos_splines[sat_id](t1)
                vel = self._vel_splines[sat_id](t1)
                return pos + vel * (sod - t1)
            else:  # sod < t0
                pos = self._pos_splines[sat_id](t0)
                vel = self._vel_splines[sat_id](t0)
                return pos + vel * (sod - t0)
        return None

    def satellite_clock(self, sat_id: str, sod: float) -> float | None:
        """Interpolated satellite clock correction (seconds) at time sod.

        Returns None if no CLK data is available for this satellite.
        The correction is referenced to the IF combination (no TGD).
        """
        # Try CLK file first, then fall back to SP3 clock.
        if self._clk is not None and sat_id in self._clk:
            result = self._sat_clock_from_clk(sat_id, sod)
            if result is not None:
                return result
        # SP3 clock fallback: the SP3 file has wider time coverage (48h)
        # than the CLK file (12-16h), so when the CLK ends before the
        # observation the SP3 clock is the best available source.
        # RTKLIB does the same in peph2pos() -> interpc().
        clk_splines = getattr(self, "_clk_splines", None)
        if clk_splines and sat_id in clk_splines:
            t0, t1 = self._clk_bounds[sat_id]
            MAX_SP3_CLK_EXTRAP_S = 8 * 3600  # 8 h, same as position
            if sod < t0 - 30 or sod > t1 + MAX_SP3_CLK_EXTRAP_S:
                return None
            try:
                if t0 <= sod <= t1:
                    dts = float(clk_splines[sat_id](sod))
                elif sod > t1:
                    # Linear extrapolation using the spline derivative at t1
                    c_end = float(clk_splines[sat_id](t1))
                    clk_vel = getattr(self, "_clk_vel_splines", {})
                    if sat_id in clk_vel:
                        slope = float(clk_vel[sat_id](t1))
                    else:
                        slope = 0.0
                    dts = c_end + slope * (sod - t1)
                else:  # sod < t0
                    c_start = float(clk_splines[sat_id](t0))
                    clk_vel = getattr(self, "_clk_vel_splines", {})
                    if sat_id in clk_vel:
                        slope = float(clk_vel[sat_id](t0))
                    else:
                        slope = 0.0
                    dts = c_start + slope * (sod - t0)
                if not np.isfinite(dts):
                    return None
                return dts + self._relativistic_correction(sat_id, sod)
            except Exception:
                return None
        return None

    def _sat_clock_from_clk(self, sat_id: str, sod: float) -> float | None:
        """Satellite clock from the CLK file with linear extrapolation."""
        if self._clk is None or sat_id not in self._clk:
            return None
        # Day check FIRST.  The range test below is on seconds-of-day, which
        # wraps at UTC midnight -- so an epoch at 01:00 on day N+1 has sod
        # 3600, lands comfortably inside a day-N table, and silently receives a
        # satellite clock 24 h stale.  Satellite clocks drift far enough in a
        # day that this shows up as hundreds of kilometres of range error.
        #
        # Observed: a rolling capture crossing UTC midnight drove the derived
        # receiver clock from 5.8 m to 262 km across successive windows, while
        # full-sky position stayed at centimetres, because only this path was
        # using the stale corrections.
        #
        # When _use_sow is True the time index is absolute GPS seconds,
        # which is unique across midnight and the week boundary, so the
        # day check is unnecessary and multi-day cached products
        # (ultra-rapid) can be loaded in one table.
        if (not getattr(self, "_use_sow", False)
                and self._doy is not None
                and self._epoch_doy is not None):
            if self._epoch_doy != self._doy:
                return None
        arr = self._clk[sat_id]
        t = arr[:, 0]
        c = arr[:, 1]
        if sod < t[0] - 30 or sod > t[-1] + 30:
            # Linear extrapolation beyond the file bounds.
            #
            # Satellite clock corrections from precise products change
            # slowly (~cm/minute for the linear drift; the relativistic
            # term is handled separately by _relativistic_correction from
            # the orbit spline, which usually has wider coverage).  Linear
            # extrapolation from the last ~1 hour of records is therefore
            # accurate to ~0.1-0.3 m over several hours -- far better than
            # the 100-200 m broadcast-clock fallback the EKF would otherwise
            # use.  RTKLIB does exactly this in peph2pos()/interpc().
            #
            # Without this, the custom EKF diverges whenever the ultra-rapid
            # CLK file ends before the live observation (common on the rig:
            # the GRG ULT is downloaded every 2-4 h but the capture runs 4 h).
            # The EKF converges to ~1.5 m while the CLK covers, then diverges
            # to 70+ m once it falls back to broadcast, while RTKLIB (which
            # extrapolates) stays at cm-level on the same data.
            MAX_EXTRAP_S = 12 * 3600  # 12 h max extrapolation
            if sod < t[0] - MAX_EXTRAP_S or sod > t[-1] + MAX_EXTRAP_S:
                return None
            if len(t) < 2:
                return None
            # Robust slope from ~1 hour of 5-min ULT records (12 points).
            # A 2-point slope from adjacent 5-min records is noisy
            # (~0.03 m / 300 s = 1e-4 m/s -> ~2 m over 5.6 h); a 1-hour
            # linear fit reduces this by sqrt(12) ~ 3.5x.
            if sod > t[-1]:
                n_fit = min(12, len(t) // 2)
                if n_fit >= 2:
                    slope = float(np.polyfit(t[-n_fit:], c[-n_fit:], 1)[0])
                else:
                    slope = float((c[-1] - c[-2]) / (t[-1] - t[-2]))
                dts = float(c[-1] + slope * (sod - t[-1]))
            else:  # sod < t[0]
                n_fit = min(12, len(t) // 2)
                if n_fit >= 2:
                    slope = float(np.polyfit(t[:n_fit], c[:n_fit], 1)[0])
                else:
                    slope = float((c[1] - c[0]) / (t[1] - t[0]))
                dts = float(c[0] + slope * (sod - t[0]))
            if not np.isfinite(dts):
                return None
            return dts + self._relativistic_correction(sat_id, sod)
        dts = float(np.interp(sod, t, c))
        return dts + self._relativistic_correction(sat_id, sod)

    def _relativistic_correction(self, sat_id: str, sod: float) -> float:
        """Periodic relativistic clock correction, -2 * (r . v) / c^2 seconds.

        IGS CLK products do NOT contain this term -- by convention the user
        applies it, exactly as RTKLIB does in ``peph2pos()``.  It is the
        eccentricity-driven part of the satellite clock's relativistic
        offset and reaches +/-13 m (45 ns) of range over an orbit, varying
        sinusoidally with a ~12 h period.  Omitting it leaves a slowly
        swinging per-satellite range bias that the estimator cannot absorb
        into a constant ambiguity, so it leaks into the receiver clock and
        the vertical.

        Returns 0.0 when no orbit spline is available for the satellite, so
        a missing orbit degrades to the previous behaviour rather than
        failing the lookup.
        """
        # getattr rather than direct access: callers (and tests) may build a
        # PreciseProducts via __new__ to exercise the clock table alone, with
        # no orbit splines present.  No orbit means no velocity, hence no
        # correction -- not an error.
        splines = getattr(self, "_pos_splines", None) or {}
        vels = getattr(self, "_vel_splines", None) or {}
        bounds = getattr(self, "_pos_bounds", None) or {}
        spline = splines.get(sat_id)
        vel = vels.get(sat_id)
        if spline is None or vel is None or sat_id not in bounds:
            return 0.0
        t0, t1 = bounds[sat_id]
        if sod < t0 - 30 or sod > t1 + 30:
            return 0.0
        try:
            r = np.asarray(spline(sod), dtype=float)
            v = np.asarray(vel(sod), dtype=float)
        except Exception:
            return 0.0
        if r.shape != (3,) or v.shape != (3,):
            return 0.0
        return -2.0 * float(np.dot(r, v)) / (C * C)

    def set_epoch_doy(self, doy: int | None) -> None:
        """Declare which day-of-year the epochs being looked up belong to.

        A window spanning UTC midnight contains two days but carries one
        seconds-of-day axis, so the caller has to say which day each lookup is
        for.  When it does not, behaviour is unchanged.
        """
        self._epoch_doy = doy

    @property
    def available_sats(self) -> set[str]:
        return set(self._pos_splines.keys())


def find_products_for_date(year: int, doy: int,
                           sp3_dir: Path, clk_dir: Path
                           ) -> PreciseProducts | None:
    """Load precise products for a given year and day-of-year.

    Returns a PreciseProducts instance, or None if no SP3 file is found
    (CLK is optional — without it, broadcast clocks are used with
    precise orbits, which still improves positioning significantly).

    Ultra-rapid (ULT) products span 2-3 days, so a CLK or SP3 file dated
    one or two days before the target may be the only one covering
    late-UTC observations on the target day (e.g. an 18:00 ULT issued on
    DOY N covers DOY N 18:00 through DOY N+2 18:00, so a DOY-246 file
    covers all of DOY 248 up to 18:00).  If no product is found for the
    target date, the previous two days are tried as fallbacks, filtered
    to the target year/doy so only matching epochs are kept.
    """
    import datetime as _dt

    def _prev_tag(delta: int) -> str:
        d = _dt.date(year, 1, 1) + _dt.timedelta(days=doy - 1 - delta)
        return f"{d.year}{d.timetuple().tm_yday:03d}"

    date_tag = f"{year}{doy:03d}"
    sp3 = load_sp3(date_tag, sp3_dir, target_year=year, target_doy=doy)
    if sp3 is None:
        # Ultra-rapid SP3 from up to 2 days earlier may cover the target.
        for delta in (1, 2):
            sp3 = load_sp3(_prev_tag(delta), sp3_dir,
                           target_year=year, target_doy=doy)
            if sp3 is not None:
                break
    if sp3 is None:
        return None
    clk = load_clk(date_tag, clk_dir)
    if clk is None:
        # Ultra-rapid CLK from up to 2 days earlier may cover the target
        # (e.g. a DOY-246 18:00 ULT covers DOY 248 up to 18:00).
        for delta in (1, 2):
            clk = load_clk(_prev_tag(delta), clk_dir,
                           target_year=year, target_doy=doy)
            if clk is not None:
                break
    pp = PreciseProducts(sp3, clk)
    # Record which day this set describes so satellite_clock() can refuse a
    # lookup for a different day (the seconds-of-day range test cannot see a
    # UTC midnight crossing on its own).
    pp._doy = int(doy)
    return pp


def load_cached_clk(clk_path: Path,
                    target_year: int,
                    target_doy: int,
                    systems: tuple[str, ...] = ("G", "E", "C")
                    ) -> dict | None:
    """Load a cached RINEX CLK file by explicit path (not naming pattern).

    Used for the ``data/ppp_live/precise/clock_cod_wl.clk`` file that the
    RTKLIB PPP-AR path already uses successfully.  This file contains
    CODE combined rapid+ultra satellite clocks (with WL bias comments at
    the top) covering the live observation period, but its filename does
    not match the ``load_clk`` glob pattern so ``find_products_for_date``
    never finds it.

    Multi-day cached CLK files are indexed by absolute GPS seconds
    (week*604800 + SOW) so that a single PreciseProducts can serve epochs
    on both sides of UTC midnight and the GPS week boundary without the
    seconds-of-day/week collision that otherwise silently returns the
    wrong day's satellite clocks.

    Returns dict {sat_id: np.array(N, 2)} with columns
    ``[t_abs, clk_s]`` or ``None``.
    """
    import datetime as _dt
    if not clk_path.exists():
        return None
    target_date = _dt.date(target_year, 1, 1) + _dt.timedelta(
        days=target_doy - 1)
    raw: dict[str, list] = {}
    # The file may be gzipped or plain text.
    opener = gzip.open if clk_path.suffix in (".gz", ".Z") else open
    try:
        with opener(clk_path, "rt") as fh:
            for line in fh:
                if not line.startswith("AS "):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                sat_id = parts[1]
                if sat_id[0] not in systems:
                    continue
                try:
                    year = int(parts[2]); month = int(parts[3])
                    day = int(parts[4]); hour = int(parts[5])
                    minute = int(parts[6]); sec = float(parts[7])
                    rec_date = _dt.date(year, month, day)
                    # Keep records within +/- 1 day of the target date so
                    # a multi-day file covers a rolling capture crossing
                    # UTC midnight.
                    if abs((rec_date - target_date).days) > 1:
                        continue
                    clk_bias = float(parts[9].replace("D", "E").replace("d", "e"))
                except (ValueError, IndexError):
                    continue
                _wk, sow = _gps_week_seconds(
                    year, month, day, hour, minute, sec)
                # Index by ABSOLUTE GPS seconds (week*604800 + SOW) so a
                # capture spanning the GPS week boundary is served on
                # both sides — plain SOW collides across the boundary
                # (prev week's ~604800 s tail vs the new week's ~0 s
                # start) and silently drops one side of the window.
                raw.setdefault(sat_id, []).append(
                    (_wk * 604800.0 + sow, clk_bias))
    except Exception:
        return None
    out = {}
    for s, v in raw.items():
        if len(v) < 2:
            continue
        arr = np.array(sorted(v))
        t = arr[:, 0]
        unique_mask = np.concatenate(([True], t[1:] != t[:-1]))
        arr = arr[unique_mask]
        if len(arr) >= 2:
            out[s] = arr
    return out if out else None


def load_cached_sp3(sp3_path: Path,
                    target_year: int,
                    target_doy: int,
                    systems: tuple[str, ...] = ("G", "E", "R", "C")
                    ) -> dict | None:
    """Load a cached SP3 precise-orbit file by explicit path (not naming pattern).

    Mirrors :func:`load_sp3` but reads a single file given by ``sp3_path``
    instead of globbing a directory by date tag.  Used for the live rig's
    cached products in ``data/ppp_live/precise/`` (e.g. a freshly downloaded
    IGS ultra-rapid SP3 whose filename does not match the archive naming
    convention).

    Multi-day (ultra-rapid, ``_02D_``) files are indexed by absolute GPS
    seconds (week*604800 + SOW) so that a single PreciseProducts can
    serve epochs on both sides of UTC midnight AND the GPS week boundary
    without the seconds-of-day/week collision that otherwise silently
    returns wrong satellite positions.  The absolute index is unambiguous
    across midnight, so the day check in
    :meth:`PreciseProducts.satellite_clock` is bypassed via ``_use_sow``.

    Returns ``{sat_id: np.array(N, 4)}`` with columns
    ``[t_abs, x_m, y_m, z_m, clk_s]`` (absolute GPS seconds), or ``None``.
    """
    import datetime as _dt
    if not sp3_path.exists():
        return None
    target_date = _dt.date(target_year, 1, 1) + _dt.timedelta(
        days=target_doy - 1)
    # Collect epochs across ALL days present in the file so a multi-day
    # ultra-rapid product can serve a rolling capture that crosses UTC
    # midnight or the GPS week boundary.  The time index is absolute GPS
    # seconds, which is unique across both and avoids the
    # seconds-of-day/week collision.
    raw: dict[str, list] = {}
    te = None
    te_sow = None
    opener = gzip.open if sp3_path.suffix in (".gz", ".Z") else open
    try:
        with opener(sp3_path, "rt") as fh:
            for line in fh:
                if line.startswith("*  "):
                    p = line.split()
                    yr, mo, dy = int(p[1]), int(p[2]), int(p[3])
                    hr, mi = int(p[4]), int(p[5])
                    sc = float(p[6])
                    te = int(hr) * 3600 + int(mi) * 60 + sc
                    _wk, te_sow = _gps_week_seconds(yr, mo, dy, hr, mi, sc)
                    # Index by ABSOLUTE GPS seconds (week*604800 + SOW),
                    # not plain SOW: SOW wraps at the week boundary and a
                    # capture spanning Sunday 00:00 UTC needs records
                    # from both weeks — with plain SOW the previous
                    # week's tail collides with the new week's start and
                    # one side of the window loses product coverage.
                    te_sow = _wk * 604800.0 + te_sow
                    # Keep epochs within +/- 1 day of the target date so
                    # the table covers the target day and its neighbours
                    # (a rolling capture may start on day N-1 and end on
                    # day N, or start on day N and end on day N+1).
                    rec_date = _dt.date(yr, mo, dy)
                    te_valid = abs((rec_date - target_date).days) <= 1
                elif (line.startswith("P") and te is not None
                      and te_valid and line[1] in systems):
                    try:
                        x = float(line[4:18])
                        y = float(line[18:32])
                        z = float(line[32:46])
                    except ValueError:
                        continue
                    if abs(x) > 1e-6:
                        # Also extract the SP3 clock (column 47-60, in
                        # microseconds).  Used as a fallback when the CLK
                        # file ends before the observation (common on the
                        # live rig: the GRG ULT CLK ends 12-16h before the
                        # capture, but the SP3 covers 48h).
                        try:
                            clk_us = float(line[46:60])
                        except ValueError:
                            clk_us = 0.0
                        raw.setdefault(line[1:4], []).append(
                            (te_sow, x * 1e3, y * 1e3, z * 1e3, clk_us * 1e-6))
    except Exception:
        return None
    out = {}
    for s, v in raw.items():
        if len(v) < 4:
            continue
        arr = np.array(sorted(v))
        t = arr[:, 0]
        unique_mask = np.concatenate(([True], t[1:] != t[:-1]))
        arr = arr[unique_mask]
        if len(arr) >= 4:
            out[s] = arr
    return out if out else None


def load_cached_products(target_year: int,
                         target_doy: int,
                         cached_dir: Path,
                         sp3_names: tuple[str, ...] = (
                             "orbit_cod.sp3", "orbit.sp3",
                             "orbit_cod_ult.sp3", "orbit_grg_ult.sp3"),
                         clk_names: tuple[str, ...] = (
                             "clock.clk", "clock_grg_ult.clk",
                             "clock_cod_combined.clk", "clock_cod_wl.clk")
                         ) -> PreciseProducts | None:
    """Build a :class:`PreciseProducts` from cached SP3/CLK files in a directory.

    This is the live-rig fallback when :func:`find_products_for_date` finds
    nothing in the archive (common for the current day before rapid products
    are downloaded).  It loads the cached SP3 and CLK files that the RTKLIB
    PPP-AR path already uses, filtered to the target observation day, and
    returns a ``PreciseProducts`` so the custom EKF can use the same precise
    satellite orbits/clocks as RTKLIB instead of falling back to broadcast.

    Only files whose epochs actually cover the target day are used; a stale
    cached file ending before the target day produces an empty filtered
    table and is skipped.  Returns ``None`` if no usable SP3 is found.
    """
    sp3 = None
    for name in sp3_names:
        p = cached_dir / name
        if not p.exists() or p.stat().st_size < 50000:
            continue
        sp3 = load_cached_sp3(p, target_year, target_doy)
        if sp3:
            break
    if not sp3:
        return None
    clk = None
    for name in clk_names:
        p = cached_dir / name
        if not p.exists() or p.stat().st_size < 100000:
            continue
        clk = load_cached_clk(p, target_year, target_doy)
        if clk:
            break
    pp = PreciseProducts(sp3, clk)
    pp._doy = int(target_doy)
    # Cached SP3/CLK files use absolute GPS seconds as the time index
    # (see load_cached_sp3 / load_cached_clk), so the day check in
    # satellite_clock is unnecessary and multi-day ultra-rapid products
    # can serve a rolling capture that crosses UTC midnight or the GPS
    # week boundary.
    pp._use_sow = True
    return pp
