#!/usr/bin/env python3
"""
TEP-PPP Raspberry Pi main service.

Orchestrates:
  1. GPS serial reader (LEA-6H / LC29H)
  2. Optional NTRIP connection for reference-station clock data
  3. TEP kriging prediction (when network data available)
  4. TFT display output (Adafruit 2.8" PiTFT)

Runs as a systemd service.  Configurable via environment variables:
  GPS_SERIAL_PORT   - serial device (default: /dev/serial0)
  GPS_BAUDRATE      - baud rate (default: 9600)
  NTRIP_ENABLED     - "1" to enable NTRIP (default: "0")
  NTRIP_USER        - EUREF IP credentials (optional)
  TEP_ENABLED       - "1" to enable TEP processing (default: "1")
  DISPLAY_HEADLESS  - "1" to skip framebuffer (for testing, default: "0")
  LOG_LEVEL         - logging level (default: INFO)
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gps_reader import GPSReader, GPSFix, SatelliteInfo

try:
    from tep_display import TEPDisplay, DisplayState, read_cpu_temp, read_uptime
except ImportError:
    # Allow the service to run headless on macOS or other machines without pygame
    from dataclasses import dataclass as _dataclass, field

    @_dataclass
    class DisplayState:
        fix_type: str = "NO FIX"
        latitude: float = 0.0
        longitude: float = 0.0
        altitude: float = 0.0
        num_sats: int = 0
        hdop: float = 99.0
        satellites: list = field(default_factory=list)
        tep_enabled: bool = False
        clock_bias_ns: float = 0.0
        tep_prediction_ns: float = 0.0
        tep_error_ns: float = 0.0
        tep_skill_pct: float = 0.0
        tep_status: str = "IDLE"
        cpu_temp: float = 0.0
        uptime_str: str = "0m"
        ntrip_status: str = "OFF"
        nmea_count: int = 0
        ubx_count: int = 0
        gps_connected: bool = False
        utc_time: str = "--:--:--"

    class TEPDisplay:
        def __init__(self, headless: bool = True):
            self.headless = headless

        def quit(self):
            pass

        def update(self, state):
            pass

    def read_cpu_temp() -> float:
        return 0.0

    def read_uptime() -> str:
        return "0m"

logger = logging.getLogger("tep-pi")


class TEPPiService:
    """Main orchestrator for the Pi-based TEP-PPP demonstration."""

    def __init__(self):
        self.gps_port = os.environ.get("GPS_SERIAL_PORT", "/dev/serial0")
        self.gps_baud = int(os.environ.get("GPS_BAUDRATE", "9600"))
        self.ntrip_enabled = os.environ.get("NTRIP_ENABLED", "0") == "1"
        self.tep_enabled = os.environ.get("TEP_ENABLED", "1") == "1"
        self.headless = os.environ.get("DISPLAY_HEADLESS", "0") == "1"
        self.update_interval = float(os.environ.get("UPDATE_INTERVAL", "1.0"))

        self.gps_reader: GPSReader | None = None
        self.display: TEPDisplay | None = None
        self._running = False

        # TEP state
        self._tep_clock_history: list[float] = []
        self._tep_pred_history: list[float] = []
        self._ntrip_connected = False

    def start(self) -> None:
        logger.info("Starting TEP-PPP Pi service")
        logger.info("  GPS port: %s @ %d baud", self.gps_port, self.gps_baud)
        logger.info("  NTRIP:    %s", "enabled" if self.ntrip_enabled else "disabled")
        logger.info("  TEP:      %s", "enabled" if self.tep_enabled else "disabled")
        logger.info("  Display:  %s", "headless" if self.headless else "framebuffer")

        # Start GPS reader
        self.gps_reader = GPSReader(
            port=self.gps_port,
            baudrate=self.gps_baud,
            enable_raw=self.tep_enabled,
        )
        self.gps_reader.start()

        # Give the serial port a moment to open
        time.sleep(2.0)

        # Optionally enable UBX raw output on u-blox receivers
        if self.tep_enabled and self.gps_reader._serial and self.gps_reader._serial.is_open:
            try:
                from gps_reader import enable_ubx_raw
                enable_ubx_raw(self.gps_reader._serial)
                # Bump baud to 38400 for raw data throughput
                # (LEA-6H default is 9600; raw obs needs higher rate)
                logger.info("UBX raw output enabled")
            except Exception as e:
                logger.warning("Could not enable UBX raw: %s", e)

        # Start display
        self.display = TEPDisplay(headless=self.headless)

        self._running = True
        logger.info("Service started — entering main loop")

    def stop(self) -> None:
        logger.info("Stopping TEP-PPP Pi service")
        self._running = False
        if self.gps_reader:
            self.gps_reader.stop()
        if self.display:
            self.display.quit()
        logger.info("Service stopped")

    def _build_display_state(self, fix: GPSFix) -> DisplayState:
        state = DisplayState()

        # GPS info
        state.fix_type = fix.fix_type
        state.latitude = fix.latitude
        state.longitude = fix.longitude
        state.altitude = fix.altitude
        state.num_sats = fix.num_satellites
        state.hdop = fix.hdop

        # Satellites for sky plot
        state.satellites = [
            (sat.prn, sat.elevation, sat.azimuth, sat.snr, sat.used_in_fix)
            for sat in fix.satellites.values()
        ]

        # Time
        if fix.datetime_utc:
            state.utc_time = fix.datetime_utc.strftime("%H:%M:%S")
        else:
            state.utc_time = time.strftime("%H:%M:%S", time.gmtime())

        # GPS connection status
        if self.gps_reader:
            state.gps_connected = (
                self.gps_reader._serial is not None
                and self.gps_reader._serial.is_open
                and (time.time() - self.gps_reader.last_read_time) < 10.0
            )
            state.nmea_count = self.gps_reader.nmea_count
            state.ubx_count = self.gps_reader.ubx_count

        # NTRIP status
        state.ntrip_status = "CONN" if self._ntrip_connected else "OFF"

        # TEP processing
        state.tep_enabled = self.tep_enabled
        if self.tep_enabled and self._tep_clock_history:
            state.clock_bias_ns = self._tep_clock_history[-1]
            if self._tep_pred_history:
                state.tep_prediction_ns = self._tep_pred_history[-1]
                state.tep_error_ns = (
                    self._tep_clock_history[-1] - self._tep_pred_history[-1]
                )
                # Running skill metric (fraction of epochs where |error| < |clock|)
                if len(self._tep_clock_history) > 10:
                    recent_clk = self._tep_clock_history[-20:]
                    recent_pred = self._tep_pred_history[-20:]
                    errors = [abs(c - p) for c, p in zip(recent_clk, recent_pred)]
                    clocks = [abs(c) for c in recent_clk]
                    mean_err = sum(errors) / len(errors)
                    mean_clk = sum(clocks) / len(clocks)
                    if mean_clk > 0:
                        state.tep_skill_pct = max(0.0, 100.0 * (1 - mean_err / mean_clk))
                state.tep_status = "KRIGING ACTIVE" if self._ntrip_connected else "STANDBY"
            else:
                state.tep_status = "COLLECTING"
        elif self.tep_enabled:
            state.tep_status = "WAITING FOR GPS"
        else:
            state.tep_status = "DISABLED"

        # System
        state.cpu_temp = read_cpu_temp()
        state.uptime_str = read_uptime()

        return state

    def _process_tep(self, fix: GPSFix) -> None:
        """
        Process TEP clock prediction.

        With the LEA-6H (NMEA only), we extract the receiver clock bias from
        the difference between GPS time and receiver time.  When NTRIP is
        connected, the network kriging prediction is applied.

        This is a simplified real-time version of the full TEP pipeline.  The
        complete analysis runs in the offline pipeline (scripts/steps/).
        """
        if not fix.has_fix or not fix.datetime_utc:
            return

        # Extract a coarse clock bias estimate from the GPS fix
        # The NMEA time is GPS time; the receiver clock offset is embedded
        # in the pseudorange solution.  For a standalone receiver without
        # raw obs, we use the fix quality and timing as a proxy.
        #
        # When UBX-RAWX is available (enable_raw=True), we get the actual
        # receiver clock bias in metres from the raw measurements.
        raw_obs = self.gps_reader.get_raw_obs() if self.gps_reader else []
        if raw_obs:
            # Use the most recent UBX-RAWX clock bias (convert metres to ns)
            latest = raw_obs[-1]
            clk_bias_ns = latest.recv_clk_bias * 1e9 / 299_792_458.0
            self._tep_clock_history.append(clk_bias_ns)
        else:
            # NMEA-only mode: estimate clock bias from fix timing
            # This is a placeholder — the real clock bias requires raw obs
            # or a differential comparison.  We record a nominal value.
            gps_t = fix.datetime_utc.timestamp()
            sys_t = time.time()
            clk_bias_ns = (gps_t - sys_t) * 1e9
            self._tep_clock_history.append(clk_bias_ns)

        # Keep a rolling window
        if len(self._tep_clock_history) > 300:
            self._tep_clock_history.pop(0)

        # TEP kriging prediction (when NTRIP network data is available)
        if self._ntrip_connected and len(self._tep_clock_history) >= 10:
            # Placeholder for the real-time kriging prediction.
            # The full implementation connects to the NTRIP caster,
            # collects reference station clock biases, builds the TEP
            # spatial kernel, and predicts the receiver clock.
            #
            # For now, use a simple exponential smoothing as a stand-in
            # until the NTRIP integration is wired up.
            alpha = 0.3
            if not self._tep_pred_history:
                pred = self._tep_clock_history[-1]
            else:
                pred = alpha * self._tep_clock_history[-1] + (1 - alpha) * self._tep_pred_history[-1]
            self._tep_pred_history.append(pred)
            if len(self._tep_pred_history) > 300:
                self._tep_pred_history.pop(0)
        elif self._tep_clock_history and not self._tep_pred_history:
            # No network: prediction tracks the clock (no skill)
            self._tep_pred_history.append(self._tep_clock_history[-1])

    def run(self) -> None:
        self.start()

        try:
            while self._running:
                fix = self.gps_reader.get_fix() if self.gps_reader else GPSFix()

                # TEP processing
                if self.tep_enabled:
                    try:
                        self._process_tep(fix)
                    except Exception as e:
                        logger.debug("TEP processing error: %s", e)

                # Update display
                state = self._build_display_state(fix)
                if self.display:
                    self.display.update(state)

                # Console log for headless / macOS testing
                if self.headless:
                    logger.info(
                        "fix=%s lat=%.6f lon=%.6f alt=%.1f sats=%d HDOP=%.2f "
                        "nmea=%d ubx=%d | TEP status=%s clk=%.1f pred=%.1f err=%.1f skill=%.1f%%",
                        state.fix_type, state.latitude, state.longitude, state.altitude,
                        state.num_sats, state.hdop, state.nmea_count, state.ubx_count,
                        state.tep_status, state.clock_bias_ns, state.tep_prediction_ns,
                        state.tep_error_ns, state.tep_skill_pct,
                    )

                # Sleep until next update
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    service = TEPPiService()

    def _signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    service.run()


if __name__ == "__main__":
    main()
