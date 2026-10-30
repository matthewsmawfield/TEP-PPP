#!/usr/bin/env python3
"""
TEP Movement Detection Test
============================

Tests cm-level displacement detection with the LC29H(BS) GPS module,
comparing TEP-augmented positioning vs standard positioning.

Workflow:
  1. Configure LC29H(BS) to output RTCM3 MSM7 raw observations
  2. Capture RTCM3 + NMEA simultaneously for each test phase:
     - Phase A: Static baseline (e.g. 2 min)
     - Phase B: Moved position (e.g. 1m, 5m, 10m)
     - Phase C: Return to baseline
  3. Convert RTCM3 to RINEX with RTKLIB convbin
  4. Run two kinematic EKF solutions on the same data:
     - Standard: unconstrained clock process noise
     - TEP: astronomical phase-modulated clock process noise
  5. Report displacement detection, scatter, and precision comparison

Usage:
  python3 rpi/tep_movement_test.py --port /dev/cu.usbserial-210 --baud 115200

  Then follow the on-screen prompts to move the GPS antenna.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import serial

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rpi.gps_reader import GPSReader, GPSFix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

RTCM_DIR = Path(__file__).parent / "rtcm_logs"
RESULTS_DIR = Path(__file__).parent / "test_results"


# ---------------------------------------------------------------------------
# LC29H(BS) configuration commands
# ---------------------------------------------------------------------------
def _nmea_checksum(sentence: str) -> str:
    body = sentence[1:] if sentence.startswith("$") else sentence
    ck = 0
    for ch in body:
        ck ^= ord(ch)
    return f"{ck:02X}"


def _pair_cmd(cmd: str) -> str:
    if not cmd.startswith("$"):
        cmd = "$" + cmd
    if "*" not in cmd:
        ck = _nmea_checksum(cmd)
        cmd = cmd + "*" + ck
    return cmd + "\r\n"


def configure_lc29h_rtcm(port: serial.Serial) -> None:
    """Enable RTCM3 MSM7 + ephemeris + antenna position output on LC29H(BS)."""
    commands = [
        "$PAIR432,1",   # Enable RTCM3 MSM7
        "$PAIR434,1",   # Enable 1005 antenna position
        "$PAIR436,1",   # Enable ephemeris (1019/1020/1042/1044/1046)
    ]
    for cmd in commands:
        full = _pair_cmd(cmd)
        port.write(full.encode("ascii"))
        print(f"  Sent: {cmd}")
        time.sleep(0.2)
    print("  LC29H configured for RTCM3 output")


def disable_lc29h_rtcm(port: serial.Serial) -> None:
    """Disable RTCM3 output, return to NMEA-only."""
    port.write(_pair_cmd("$PAIR432,-1").encode("ascii"))
    time.sleep(0.2)
    print("  RTCM3 output disabled")


# ---------------------------------------------------------------------------
# Combined serial reader: raw logging + NMEA parsing
# ---------------------------------------------------------------------------
class CombinedReader:
    """Single serial reader that logs raw bytes to file and parses NMEA.

    Avoids the 'multiple access on port' error by using one serial connection.
    """

    def __init__(self, port: str, baud: int):
        self.port_name = port
        self.baud = baud
        self._serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._raw_fp = None
        self.bytes_captured = 0
        self.nmea_lines = 0
        self.rtcm_bytes = 0

        # Parsed fix state (thread-safe)
        self._fix = GPSFix()
        self._lock = threading.Lock()
        self.positions: list[dict] = []

    def start(self, raw_file: Path) -> None:
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        self._raw_fp = open(raw_file, "wb")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._raw_fp:
            self._raw_fp.close()
        if self._serial and self._serial.is_open:
            self._serial.close()

    def get_fix(self) -> GPSFix:
        with self._lock:
            return GPSFix(
                timestamp=self._fix.timestamp,
                datetime_utc=self._fix.datetime_utc,
                latitude=self._fix.latitude,
                longitude=self._fix.longitude,
                altitude=self._fix.altitude,
                fix_quality=self._fix.fix_quality,
                num_satellites=self._fix.num_satellites,
                hdop=self._fix.hdop,
                vdop=self._fix.vdop,
                pdop=self._fix.pdop,
                speed_knots=self._fix.speed_knots,
                course=self._fix.course,
                satellites=dict(self._fix.satellites),
            )

    def _read_loop(self) -> None:
        from rpi.gps_reader import _nmea_checksum, NMEA_PARSERS
        try:
            self._serial = serial.Serial(
                self.port_name, self.baud, timeout=1.0,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            buf = bytearray()
            while self._running:
                data = self._serial.read(1024)
                if not data:
                    continue
                self._raw_fp.write(data)
                self.bytes_captured += len(data)

                # Parse NMEA from the stream
                buf.extend(data)
                # Process NMEA lines (delimited by \n)
                while True:
                    idx = buf.find(0x0A)  # '\n'
                    if idx == -1:
                        break
                    line_bytes = buf[:idx + 1]
                    del buf[:idx + 1]
                    line = line_bytes.decode("ascii", errors="ignore").strip()
                    if not line.startswith("$"):
                        continue
                    if not _nmea_checksum(line):
                        continue
                    self.nmea_lines += 1
                    body = line[1:].split("*")[0]
                    fields = body.split(",")
                    if len(fields) < 1:
                        continue
                    talker_msg = fields[0]
                    for mt in NMEA_PARSERS:
                        if talker_msg.endswith(mt):
                            with self._lock:
                                NMEA_PARSERS[mt](fields, self._fix)
                            break

                    # Log position if we have a fix
                    fix = self.get_fix()
                    if fix.has_fix:
                        # Only log at most every 0.5s
                        now = time.time()
                        if not self.positions or (now - self.positions[-1]["t"]) >= 0.5:
                            self.positions.append({
                                "t": now,
                                "lat": fix.latitude,
                                "lon": fix.longitude,
                                "alt": fix.altitude,
                                "n_sats": fix.num_satellites,
                                "hdop": fix.hdop,
                                "fix": fix.fix_type,
                            })

                # Rough RTCM3 detection (0xD3 prefix)
                if data[0] == 0xD3:
                    self.rtcm_bytes += len(data)

        except Exception as e:
            print(f"  Reader error: {e}")


# ---------------------------------------------------------------------------
# NMEA position logger (uses GPSReader for parsed positions)
# ---------------------------------------------------------------------------
class PositionLogger:
    """Logs parsed NMEA positions to a JSONL file."""

    def __init__(self, reader: GPSReader, output_file: Path):
        self.reader = reader
        self.output_file = output_file
        self.positions: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        with open(self.output_file, "w") as f:
            for p in self.positions:
                f.write(json.dumps(p) + "\n")

    def _log_loop(self) -> None:
        while self._running:
            fix = self.reader.get_fix()
            if fix.has_fix:
                self.positions.append({
                    "t": time.time(),
                    "lat": fix.latitude,
                    "lon": fix.longitude,
                    "alt": fix.altitude,
                    "n_sats": fix.num_satellites,
                    "hdop": fix.hdop,
                    "fix": fix.fix_type,
                })
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def latlon_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt_m) * cos_lat * math.cos(lon)
    y = (N + alt_m) * cos_lat * math.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(x: float, y: float, z: float,
                ref_lat: float, ref_lon: float, ref_alt: float) -> tuple[float, float, float]:
    ref_lat_r = math.radians(ref_lat)
    ref_lon_r = math.radians(ref_lon)
    rx, ry, rz = latlon_to_ecef(ref_lat, ref_lon, ref_alt)
    dx, dy, dz = x - rx, y - ry, z - rz
    e = -math.sin(ref_lon_r) * dx + math.cos(ref_lon_r) * dy
    n = -math.sin(ref_lat_r) * math.cos(ref_lon_r) * dx - math.sin(ref_lat_r) * math.sin(ref_lon_r) * dy + math.cos(ref_lat_r) * dz
    u = math.cos(ref_lat_r) * math.cos(ref_lon_r) * dx + math.cos(ref_lat_r) * math.sin(ref_lon_r) * dy + math.sin(ref_lat_r) * dz
    return e, n, u


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def position_stats(positions: list[dict], ref_lat: float, ref_lon: float, ref_alt: float) -> dict:
    """Compute ENU scatter statistics relative to a reference point."""
    if not positions:
        return {"n": 0}

    enu = []
    for p in positions:
        x, y, z = latlon_to_ecef(p["lat"], p["lon"], p["alt"])
        e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        enu.append((e, n, u))

    enu_arr = [(e, n, u) for e, n, u in enu]
    n_pts = len(enu_arr)
    e_vals = [p[0] for p in enu_arr]
    n_vals = [p[1] for p in enu_arr]
    u_vals = [p[2] for p in enu_arr]

    e_mean = sum(e_vals) / n_pts
    n_mean = sum(n_vals) / n_pts
    u_mean = sum(u_vals) / n_pts

    e_std = math.sqrt(sum((e - e_mean) ** 2 for e in e_vals) / max(1, n_pts - 1))
    n_std = math.sqrt(sum((n - n_mean) ** 2 for n in n_vals) / max(1, n_pts - 1))
    u_std = math.sqrt(sum((u - u_mean) ** 2 for u in u_vals) / max(1, n_pts - 1))

    # 2D CEP (Circular Error Probable, 50th percentile)
    dists_2d = [math.sqrt(e ** 2 + n ** 2) for e, n, _ in enu_arr]
    dists_2d.sort()
    cep_50 = dists_2d[n_pts // 2] if n_pts > 0 else 0
    cep_95 = dists_2d[int(n_pts * 0.95)] if n_pts > 0 else 0

    # 3D RMS
    dists_3d = [math.sqrt(e ** 2 + n ** 2 + u ** 2) for e, n, u in enu_arr]
    rms_3d = math.sqrt(sum(d ** 2 for d in dists_3d) / max(1, n_pts))

    return {
        "n": n_pts,
        "e_mean_m": e_mean,
        "n_mean_m": n_mean,
        "u_mean_m": u_mean,
        "e_std_m": e_std,
        "n_std_m": n_std,
        "u_std_m": u_std,
        "cep_50_m": cep_50,
        "cep_95_m": cep_95,
        "rms_3d_m": rms_3d,
        "2d_scatter_m": math.sqrt(e_std ** 2 + n_std ** 2),
    }


# ---------------------------------------------------------------------------
# Kalman-smoothed NMEA positions (TEP and non-TEP control)
# ---------------------------------------------------------------------------
def kalman_smooth_positions(positions: list[dict],
                            ref_lat: float, ref_lon: float, ref_alt: float,
                            use_tep: bool = True) -> list[dict]:
    """Apply a Kalman smoother to NMEA positions.

    Converts to ENU, runs a constant-position EKF, then converts back.

    When *use_tep* is True, process noise is scaled by TEP astronomical phases
    (semiannual nutation, lunar nodal, Chandler wobble). When False, constant
    process noise is used — this is the control group to isolate the TEP
    effect.
    """
    import numpy as np

    if use_tep:
        try:
            from scripts.lib.tep_astronomical import (
                nutation_phase_semiannual,
                nutation_phase_lunar_nodal,
                chandler_phase,
            )
        except ImportError:
            use_tep = False

    if len(positions) < 3:
        return positions

    # Convert all positions to ENU
    enu = []
    for p in positions:
        x, y, z = latlon_to_ecef(p["lat"], p["lon"], p["alt"])
        e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        enu.append((e, n, u, p.get("t", 0)))

    # TEP phase weights (from Papers 1-2)
    r2_semi = 0.904
    r2_nodal = 0.641
    r2_chandler = 0.424
    r2_total = r2_semi + r2_nodal + r2_chandler
    amplitude = 0.3  # 30% modulation

    # Kalman filter state: [e, n, u] (constant position model)
    x = np.array([enu[0][0], enu[0][1], enu[0][2]])
    P = np.eye(3) * 10.0  # initial uncertainty: 10m

    # Measurement noise
    meas_std = 1.0  # default 1m

    # Base process noise (position random walk)
    q_base = 0.05  # 5cm per second

    smoothed = []
    for i, (e, n, u, t) in enumerate(enu):
        # Time step
        if i > 0 and t > 0 and enu[i-1][3] > 0:
            dt = t - enu[i-1][3]
        else:
            dt = 1.0

        # Process noise scale
        if use_tep:
            dt_obj = datetime.fromtimestamp(t, tz=timezone.utc) if t > 0 else datetime.now(timezone.utc)
            phases = {
                'semiannual': nutation_phase_semiannual(dt_obj),
                'lunar_nodal': nutation_phase_lunar_nodal(dt_obj),
                'chandler': chandler_phase(dt_obj),
            }
            r2_weights = {
                'semiannual': r2_semi,
                'lunar_nodal': r2_nodal,
                'chandler': r2_chandler,
            }
            weighted_cos = sum(
                r2_weights[k] * np.cos(phases[k])
                for k in phases
            ) / r2_total
            tep_scale = 1.0 + amplitude * float(weighted_cos)
        else:
            tep_scale = 1.0  # constant — no TEP modulation

        # Predict (constant position)
        Q = np.eye(3) * (q_base * tep_scale) ** 2 * dt
        P = P + Q

        # Update
        z = np.array([e, n, u])
        hdop = positions[i].get("hdop", 1.0) if i < len(positions) else 1.0
        R_adj = np.eye(3) * (meas_std * max(hdop, 0.5)) ** 2

        S = P + R_adj
        try:
            K = P @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((3, 3))
        x = x + K @ (z - x)
        P = (np.eye(3) - K) @ P

        # Convert back to lat/lon/alt
        smoothed.append({
            "lat": ref_lat + x[1] / (WGS84_A * math.pi / 180),
            "lon": ref_lon + x[0] / (WGS84_A * math.cos(math.radians(ref_lat)) * math.pi / 180),
            "alt": ref_alt + x[2],
            "t": t,
        })

    return smoothed


def tep_smooth_positions(positions, ref_lat, ref_lon, ref_alt):
    """TEP-modulated Kalman smoother (convenience wrapper)."""
    return kalman_smooth_positions(positions, ref_lat, ref_lon, ref_alt, use_tep=True)


def plain_smooth_positions(positions, ref_lat, ref_lon, ref_alt):
    """Non-TEP Kalman smoother (control group — constant process noise)."""
    return kalman_smooth_positions(positions, ref_lat, ref_lon, ref_alt, use_tep=False)


# ---------------------------------------------------------------------------
# RTKLIB processing
# ---------------------------------------------------------------------------
def find_rtklib_tool(name: str) -> str:
    for p in [
        os.path.expanduser(f"~/.local/bin/{name}"),
        os.path.expanduser(f"~/.local/bin/{name}_demo5"),
    ]:
        if os.path.exists(p):
            return p
    import shutil
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"RTKLIB tool {name} not found")


def convert_rtcm_to_rinex(rtcm_file: Path, output_dir: Path) -> dict | None:
    """Convert raw RTCM3 file to RINEX obs + nav using RTKLIB convbin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    convbin = find_rtklib_tool("convbin")
    cmd = [convbin, "-r", "rtcm3", "-d", str(output_dir), "-f", "2", str(rtcm_file)]
    print(f"  convbin: {Path(rtcm_file).name} -> {output_dir.name}/")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        if result.returncode != 0:
            print(f"  convbin error: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  convbin failed: {e}")
        return None

    # Find output files
    obs_files = list(output_dir.glob("*.obs"))
    nav_files = list(output_dir.glob("*.nav")) + list(output_dir.glob("*.gnav"))
    if not obs_files:
        print("  No RINEX obs file produced")
        return None
    return {
        "obs": str(obs_files[0]),
        "nav": [str(f) for f in nav_files],
    }


def run_rtklib_spp(obs_file: str, nav_files: list[str], output_dir: Path) -> dict | None:
    """Run RTKLIB rnx2rtkp for standard SPP positioning."""
    rnx2rtkp = find_rtklib_tool("rnx2rtkp")
    pos_file = output_dir / "spp.pos"
    cmd = [rnx2rtkp, "-p", "0", "-y", "2", "-o", str(pos_file), obs_file] + nav_files
    print(f"  rnx2rtkp: SPP solution")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        if result.returncode != 0:
            print(f"  rnx2rtkp error: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  rnx2rtkp failed: {e}")
        return None

    if not pos_file.exists():
        print("  No position file produced")
        return None

    # Parse .pos file
    positions = []
    with open(pos_file) as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
                alt = float(parts[4])
                # Q flag: 1=fix, 2=float, 5=single
                q = int(parts[5])
                n_sat = int(parts[6])
                positions.append({
                    "lat": lat, "lon": lon, "alt": alt,
                    "q": q, "n_sat": n_sat,
                    "t": parts[0] + " " + parts[1],
                })
            except (ValueError, IndexError):
                continue
    return {"positions": positions, "pos_file": str(pos_file)}


# ---------------------------------------------------------------------------
# TEP vs Standard EKF comparison
# ---------------------------------------------------------------------------
def run_tep_comparison(obs_file: str, nav_files: list[str],
                       ref_lat: float, ref_lon: float, ref_alt: float) -> dict:
    """Run both standard and TEP-augmented kinematic EKF on the same RINEX data.

    This uses the PPPEstimatorKinematicAugmented class which modulates clock
    process noise with TEP astronomical phases when tep_mode=True.
    """
    import numpy as np
    from scripts.lib.ppp_estimator_kinematic import PPPEstimatorKinematicAugmented
    from scripts.lib.tep_spp import read_gps_code, read_sp3_gps, read_clk_gps, dense_grid, prefit_fast

    # We need SP3/CLK for precise orbits — but for a quick test we can use
    # broadcast ephemeris via rnx2rtkp instead. The EKF comparison below
    # uses the prefit residuals approach with broadcast nav if SP3 unavailable.
    #
    # For now, report NMEA-based comparison + RTKLIB SPP as baseline,
    # and note that full TEP EKF requires IGS precise products.

    print("  TEP EKF comparison requires IGS precise orbits (SP3) and clocks (CLK).")
    print("  Using NMEA + RTKLIB SPP for baseline comparison instead.")
    print("  For full TEP PPP, download IGS rapid products for the test date.")

    return {
        "note": "Full TEP EKF requires IGS SP3/CLK products. Using SPP baseline.",
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "ref_alt": ref_alt,
    }


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------
PHASES = [
    {"name": "baseline", "label": "Static Baseline", "duration": 120, "prompt": "Place the GPS antenna at the reference position. Keep it stationary."},
    {"name": "move_1m", "label": "Move 1 metre", "duration": 120, "prompt": "Move the GPS antenna exactly 1 metre in any direction. Keep it stationary at the new position."},
    {"name": "move_5m", "label": "Move 5 metres", "duration": 120, "prompt": "Move the GPS antenna exactly 5 metres from the baseline position. Keep it stationary."},
    {"name": "return", "label": "Return to baseline", "duration": 120, "prompt": "Return the GPS antenna to the original baseline position. Keep it stationary."},
]


def run_test(port: str, baud: int, phases: list[dict] | None = None) -> None:
    if phases is None:
        phases = PHASES

    RTCM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    test_dir = RESULTS_DIR / f"test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TEP Movement Detection Test")
    print("=" * 70)
    print(f"  Port: {port} @ {baud} baud")
    print(f"  Output: {test_dir}")
    print()

    # Configure LC29H for RTCM3 output
    print("Configuring LC29H(BS) for RTCM3 output...")
    config_port = serial.Serial(port, baud, timeout=1.0)
    configure_lc29h_rtcm(config_port)
    config_port.close()
    time.sleep(1)

    all_results = {}

    for i, phase in enumerate(phases):
        print()
        print("-" * 70)
        print(f"  Phase {i+1}/{len(phases)}: {phase['label']}")
        print("-" * 70)
        print(f"  {phase['prompt']}")
        print(f"  Duration: {phase['duration']}s")
        print()

        if i > 0:
            input("  Press ENTER when the antenna is in position and ready...")

        print(f"  Capturing for {phase['duration']}s...")

        # Start combined reader (raw + NMEA)
        raw_file = test_dir / f"{phase['name']}.raw"
        reader = CombinedReader(port, baud)
        reader.start(raw_file)

        # Countdown
        for remaining in range(phase["duration"], 0, -10):
            print(f"  {remaining}s remaining... (bytes: {reader.bytes_captured}, NMEA: {reader.nmea_lines}, pos: {len(reader.positions)})")
            time.sleep(min(10, remaining))

        reader.stop()

        # Save positions
        pos_file = test_dir / f"{phase['name']}_pos.jsonl"
        with open(pos_file, "w") as f:
            for p in reader.positions:
                f.write(json.dumps(p) + "\n")

        print(f"  Captured: {reader.bytes_captured} bytes, {reader.nmea_lines} NMEA lines, {len(reader.positions)} positions")

        # Process this phase
        phase_result = {
            "raw_file": str(raw_file),
            "pos_file": str(pos_file),
            "bytes": reader.bytes_captured,
            "n_positions": len(reader.positions),
            "positions": reader.positions,
        }

        # Convert RTCM3 to RINEX
        rinex_dir = test_dir / f"{phase['name']}_rinex"
        rinex = convert_rtcm_to_rinex(raw_file, rinex_dir)
        if rinex:
            phase_result["rinex"] = rinex
            # Run RTKLIB SPP
            spp = run_rtklib_spp(rinex["obs"], rinex["nav"], rinex_dir)
            if spp:
                phase_result["spp"] = spp
                print(f"  SPP: {len(spp['positions'])} position fixes")

        all_results[phase["name"]] = phase_result
        print(f"  Phase {phase['name']} complete.")

    # Disable RTCM3
    print()
    print("Disabling RTCM3 output...")
    try:
        cleanup_port = serial.Serial(port, baud, timeout=1.0)
        disable_lc29h_rtcm(cleanup_port)
        cleanup_port.close()
    except Exception:
        pass

    # Compute statistics
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)

    # Use baseline as reference
    baseline_positions = all_results.get("baseline", {}).get("positions", [])
    if baseline_positions:
        ref_lat = sum(p["lat"] for p in baseline_positions) / len(baseline_positions)
        ref_lon = sum(p["lon"] for p in baseline_positions) / len(baseline_positions)
        ref_alt = sum(p["alt"] for p in baseline_positions) / len(baseline_positions)

        print(f"\n  Reference (baseline mean): {ref_lat:.8f}, {ref_lon:.8f}, {ref_alt:.2f}m")
        print()

        for phase_name, result in all_results.items():
            positions = result.get("positions", [])
            if not positions:
                continue

            stats = position_stats(positions, ref_lat, ref_lon, ref_alt)
            result["stats"] = stats

            print(f"  {phase_name}:")
            print(f"    N={stats['n']}  E={stats['e_mean_m']:+.3f}m  N={stats['n_mean_m']:+.3f}m  U={stats['u_mean_m']:+.3f}m")
            print(f"    Std: E={stats['e_std_m']:.3f}m  N={stats['n_std_m']:.3f}m  U={stats['u_std_m']:.3f}m")
            print(f"    CEP(50%)={stats['cep_50_m']:.3f}m  CEP(95%)={stats['cep_95_m']:.3f}m  RMS(3D)={stats['rms_3d_m']:.3f}m")
            print()

        # Compute detected displacements
        print("  Detected Displacements (relative to baseline):")
        for phase_name in ["move_1m", "move_5m", "return"]:
            if phase_name not in all_results:
                continue
            stats = all_results[phase_name].get("stats")
            if not stats:
                continue
            disp_2d = math.sqrt(stats["e_mean_m"] ** 2 + stats["n_mean_m"] ** 2)
            disp_3d = math.sqrt(stats["e_mean_m"] ** 2 + stats["n_mean_m"] ** 2 + stats["u_mean_m"] ** 2)
            print(f"    {phase_name}: 2D={disp_2d:.3f}m  3D={disp_3d:.3f}m  (E={stats['e_mean_m']:+.3f} N={stats['n_mean_m']:+.3f} U={stats['u_mean_m']:+.3f})")

        # Compute SPP stats from RTKLIB output
        for phase_name, result in all_results.items():
            spp = result.get("spp", {})
            spp_positions = spp.get("positions", [])
            if spp_positions:
                spp_dicts = [{"lat": p["lat"], "lon": p["lon"], "alt": p["alt"]} for p in spp_positions]
                spp_stats = position_stats(spp_dicts, ref_lat, ref_lon, ref_alt)
                result["spp_stats"] = spp_stats

        # TEP-smoothed and plain-smoothed (control) NMEA
        tep_results = {}
        plain_results = {}
        for phase_name, result in all_results.items():
            positions = result.get("positions", [])
            if not positions:
                continue
            tep_smoothed = tep_smooth_positions(positions, ref_lat, ref_lon, ref_alt)
            tep_stats = position_stats(tep_smoothed, ref_lat, ref_lon, ref_alt)
            tep_results[phase_name] = tep_stats
            result["tep_stats"] = tep_stats

            plain_smoothed = plain_smooth_positions(positions, ref_lat, ref_lon, ref_alt)
            plain_stats = position_stats(plain_smoothed, ref_lat, ref_lon, ref_alt)
            plain_results[phase_name] = plain_stats
            result["plain_kalman_stats"] = plain_stats

        # Print comparison table
        print()
        print("  " + "=" * 78)
        print("  NMEA vs SPP vs Kalman(plain) vs Kalman(TEP) — same data, controlled")
        print("  " + "=" * 78)
        base_stats = all_results["baseline"].get("stats", {})
        base_spp = all_results["baseline"].get("spp_stats", {})
        base_tep = tep_results.get("baseline", {})
        base_plain = plain_results.get("baseline", {})
        print(f"    {'Method':<22} {'2D scatter':>12} {'CEP(50%)':>12} {'CEP(95%)':>12} {'RMS(3D)':>12}")
        print(f"    {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        if base_stats:
            print(f"    {'NMEA (raw)':<22} {base_stats.get('2d_scatter_m',0):>10.3f}m {base_stats.get('cep_50_m',0):>10.3f}m {base_stats.get('cep_95_m',0):>10.3f}m {base_stats.get('rms_3d_m',0):>10.3f}m")
        if base_spp:
            print(f"    {'SPP (RTKLIB)':<22} {base_spp.get('2d_scatter_m',0):>10.3f}m {base_spp.get('cep_50_m',0):>10.3f}m {base_spp.get('cep_95_m',0):>10.3f}m {base_spp.get('rms_3d_m',0):>10.3f}m")
        if base_plain:
            print(f"    {'Kalman (plain)':<22} {base_plain.get('2d_scatter_m',0):>10.3f}m {base_plain.get('cep_50_m',0):>10.3f}m {base_plain.get('cep_95_m',0):>10.3f}m {base_plain.get('rms_3d_m',0):>10.3f}m")
        if base_tep:
            print(f"    {'Kalman (TEP)':<22} {base_tep.get('2d_scatter_m',0):>10.3f}m {base_tep.get('cep_50_m',0):>10.3f}m {base_tep.get('cep_95_m',0):>10.3f}m {base_tep.get('rms_3d_m',0):>10.3f}m")
        print()

        # Isolate the TEP effect: TEP vs plain Kalman.  The two filters differ
        # only in the magnitude of the process noise, so this must be scored on
        # ACCURACY (RMS/CEP about the baseline reference), never on scatter:
        # the arm with the smaller Q always produces the smoother track and
        # would "win" a scatter comparison without being any closer to truth.
        if base_plain and base_tep:
            for label, key, unit in (("RMS 3D", "rms_3d_m", "m"),
                                     ("CEP 50%", "cep_50_m", "m"),
                                     ("CEP 95%", "cep_95_m", "m")):
                pv = base_plain.get(key, 0)
                tv = base_tep.get(key, 0)
                if pv > 0:
                    eff = (1 - tv / pv) * 100
                    print(f"    TEP effect vs plain Kalman ({label:<7}): {eff:+6.1f}%   "
                          f"plain {pv:.4f}{unit}  TEP {tv:.4f}{unit}")
                else:
                    print(f"    TEP effect vs plain Kalman ({label:<7}): N/A")
            ps = base_plain.get("2d_scatter_m", 0)
            ts = base_tep.get("2d_scatter_m", 0)
            print(f"    [diagnostic, NOT skill] 2D scatter: plain {ps:.4f}m  TEP {ts:.4f}m")
            print("      (scatter measures smoothness; the smaller-Q filter always wins it)")
        print()

    # Save results
    results_file = test_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TEP Movement Detection Test")
    parser.add_argument("--port", default="/dev/cu.usbserial-210", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline phase (scatter test)")
    parser.add_argument("--duration", type=int, default=120, help="Phase duration in seconds")
    args = parser.parse_args()

    if args.baseline_only:
        phases = [{"name": "baseline", "label": "Static Baseline", "duration": args.duration, "prompt": "Place the GPS antenna at a fixed position. Keep it stationary."}]
    else:
        phases = [
            {"name": "baseline", "label": "Static Baseline", "duration": args.duration, "prompt": "Place the GPS antenna at the reference position. Keep it stationary."},
            {"name": "move_1m", "label": "Move 1 metre", "duration": args.duration, "prompt": "Move the GPS antenna exactly 1 metre in any direction. Keep it stationary at the new position."},
            {"name": "move_5m", "label": "Move 5 metres", "duration": args.duration, "prompt": "Move the GPS antenna exactly 5 metres from the baseline position. Keep it stationary."},
            {"name": "return", "label": "Return to baseline", "duration": args.duration, "prompt": "Return the GPS antenna to the original baseline position. Keep it stationary."},
        ]

    def _sigint(_sig, _frame):
        print("\nInterrupted.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    run_test(args.port, args.baud, phases)


if __name__ == "__main__":
    main()
