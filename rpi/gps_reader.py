#!/usr/bin/env python3
"""
GPS serial reader for TEP-PPP Raspberry Pi.

Supports:
  - u-blox LEA-6H (NMEA + UBX binary protocol)
  - Waveshare LC29H(BS) HAT (Quectel dual-band, NMEA + raw)

Reads from /dev/serial0 (configurable), parses NMEA sentences in a background
thread, and exposes the latest fix via a thread-safe snapshot.

UBX binary parsing is included for future raw-observation extraction (needed
for PPP).  NMEA provides position, satellites, DOP, and clock timing for the
initial SPP demonstration.
"""
from __future__ import annotations

import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import serial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SatelliteInfo:
    prn: int
    elevation: float = 0.0       # degrees
    azimuth: float = 0.0         # degrees
    snr: float = 0.0             # dB-Hz
    used_in_fix: bool = False


@dataclass
class GPSFix:
    timestamp: float = 0.0           # Unix time
    datetime_utc: Optional[datetime] = None
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0            # metres
    fix_quality: int = 0             # 0=none, 1=GPS, 2=DGPS, 6=dead reckoning
    num_satellites: int = 0          # satellites used in fix
    hdop: float = 99.0              # horizontal dilution of precision
    vdop: float = 99.0
    pdop: float = 99.0
    speed_knots: float = 0.0
    course: float = 0.0             # heading degrees
    satellites: dict[int, SatelliteInfo] = field(default_factory=dict)

    @property
    def has_fix(self) -> bool:
        return self.fix_quality >= 1

    @property
    def fix_type(self) -> str:
        return {0: "NO FIX", 1: "GPS", 2: "DGPS", 6: "DR"}.get(
            self.fix_quality, f"Q{self.fix_quality}"
        )


@dataclass
class UBXRawObs:
    """Parsed UBX-RXM-RAWX message (raw measurement data)."""
    rcv_tow: float = 0.0
    week: int = 0
    leap_s: int = 0
    recv_clk_bias: float = 0.0     # metres (c*dt_rx)
    measurements: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# NMEA checksum
# ---------------------------------------------------------------------------

def _nmea_checksum(sentence: str) -> bool:
    if "*" not in sentence:
        return False
    body, _, cksum = sentence.partition("*")
    if not body.startswith("$"):
        return False
    calculated = 0
    for ch in body[1:]:
        calculated ^= ord(ch)
    try:
        return calculated == int(cksum.strip()[:2], 16)
    except ValueError:
        return False


def _parse_coord(value: str, direction: str) -> float:
    if not value:
        return 0.0
    try:
        dot = value.index(".")
        deg = int(value[: dot - 2])
        minutes = float(value[dot - 2 :])
        result = deg + minutes / 60.0
        if direction in ("S", "W"):
            result = -result
        return result
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# NMEA sentence parsers
# ---------------------------------------------------------------------------

def _parse_gga(fields: list[str], fix: GPSFix) -> None:
    """$GPGGA,hhmmss.ss,llll.ll,N,yyyyy.yy,W,q,nn,h.h,alt,a,h.h,h.h,ttt*cc"""
    if len(fields) < 15:
        return
    try:
        time_str = fields[1]
        if time_str:
            h = int(time_str[0:2])
            m = int(time_str[2:4])
            s = float(time_str[4:])
            today = datetime.now(timezone.utc)
            fix.datetime_utc = today.replace(hour=h, minute=m,
                                             second=int(s),
                                             microsecond=int((s % 1) * 1e6),
                                             tzinfo=timezone.utc)
            fix.timestamp = fix.datetime_utc.timestamp()
    except (ValueError, IndexError):
        pass

    fix.latitude = _parse_coord(fields[2], fields[3])
    fix.longitude = _parse_coord(fields[4], fields[5])
    try:
        fix.fix_quality = int(fields[6])
        fix.num_satellites = int(fields[7])
        fix.hdop = float(fields[8]) if fields[8] else 99.0
        fix.altitude = float(fields[9]) if fields[9] else 0.0
    except ValueError:
        pass


def _parse_rmc(fields: list[str], fix: GPSFix) -> None:
    """$GPRMC,hhmmss.ss,A,llll.ll,N,yyyyy.yy,W,spd,course,ddmmyy,d.d,W*cc"""
    if len(fields) < 12:
        return
    try:
        status = fields[2]
        if status == "A":
            if fix.fix_quality == 0:
                fix.fix_quality = 1
        elif status == "V":
            fix.fix_quality = 0

        fix.latitude = _parse_coord(fields[3], fields[4])
        fix.longitude = _parse_coord(fields[5], fields[6])
        fix.speed_knots = float(fields[7]) if fields[7] else 0.0
        fix.course = float(fields[8]) if fields[8] else 0.0

        date_str = fields[9]
        time_str = fields[1]
        if date_str and time_str:
            day = int(date_str[0:2])
            mon = int(date_str[2:4])
            yr = 2000 + int(date_str[4:6])
            h = int(time_str[0:2])
            m = int(time_str[2:4])
            s = float(time_str[4:])
            fix.datetime_utc = datetime(yr, mon, day, h, m,
                                        int(s), int((s % 1) * 1e6),
                                        tzinfo=timezone.utc)
            fix.timestamp = fix.datetime_utc.timestamp()
    except (ValueError, IndexError):
        pass


def _parse_gsa(fields: list[str], fix: GPSFix) -> None:
    """$GPGSA,mode,fix,prn1,prn2,...,pdop,hdop,vdop*cc"""
    if len(fields) < 18:
        return
    try:
        for i in range(3, 15):
            prn_str = fields[i]
            if prn_str:
                prn = int(prn_str)
                sat = fix.satellites.setdefault(prn, SatelliteInfo(prn=prn))
                sat.used_in_fix = True
        fix.pdop = float(fields[15]) if fields[15] else 99.0
        fix.hdop = float(fields[16]) if fields[16] else 99.0
        fix.vdop = float(fields[17]) if fields[17] else 99.0
    except (ValueError, IndexError):
        pass


def _parse_gsv(fields: list[str], fix: GPSFix) -> None:
    """$GPGSV,total_msgs,msg_num,total_sats,prn,el,az,snr,...*cc"""
    if len(fields) < 8:
        return
    try:
        idx = 4
        while idx + 3 < len(fields):
            prn = int(fields[idx]) if fields[idx] else 0
            if prn == 0:
                idx += 4
                continue
            el = float(fields[idx + 1]) if fields[idx + 1] else 0.0
            az = float(fields[idx + 2]) if fields[idx + 2] else 0.0
            snr = float(fields[idx + 3]) if fields[idx + 3] else 0.0
            sat = fix.satellites.setdefault(prn, SatelliteInfo(prn=prn))
            sat.elevation = el
            sat.azimuth = az
            sat.snr = snr
            idx += 4
    except (ValueError, IndexError):
        pass


NMEA_PARSERS = {
    "GGA": _parse_gga,
    "RMC": _parse_rmc,
    "GSA": _parse_gsa,
    "GSV": _parse_gsv,
}


# ---------------------------------------------------------------------------
# UBX binary protocol (for raw observations)
# ---------------------------------------------------------------------------

UBX_SYNC1 = 0xB5
UBX_SYNC2 = 0x62

UBX_CLASS_RXM = 0x02
UBX_MSG_RAWX = 0x15


def _parse_ubx_rawx(payload: bytes) -> Optional[UBXRawObs]:
    """Parse UBX-RXM-RAWX payload."""
    if len(payload) < 24:
        return None
    try:
        rcv_tow, week, leap_s = struct.unpack_from("<dHh", payload, 0)
        recv_clk_bias = struct.unpack_from("<f", payload, 14)[0]
        num_meas = payload[18]
        obs = UBXRawObs(
            rcv_tow=rcv_tow,
            week=week,
            leap_s=leap_s,
            recv_clk_bias=recv_clk_bias,
        )
        offset = 24
        for _ in range(num_meas):
            if offset + 32 > len(payload):
                break
            (pr_mes, cp_mes, do_mes, gnss_id, sv_id, _, _, locktime, cno,
             _, _, pr_std, cp_std, do_std) = struct.unpack_from(
                "<dddBBBBHBBBfff", payload, offset
            )
            obs.measurements.append({
                "pr_mes": pr_mes,
                "cp_mes": cp_mes,
                "do_mes": do_mes,
                "gnss_id": gnss_id,
                "sv_id": sv_id,
                "locktime": locktime,
                "cno": cno,
                "pr_std": pr_std,
                "cp_std": cp_std,
                "do_std": do_std,
            })
            offset += 32
        return obs
    except (struct.error, IndexError):
        return None


# ---------------------------------------------------------------------------
# GPS Reader
# ---------------------------------------------------------------------------

class GPSReader:
    """
    Threaded GPS serial reader.

    Reads NMEA and UBX messages from a serial port, maintains a thread-safe
    GPSFix snapshot, and optionally queues raw UBX observations for PPP.
    """

    def __init__(
        self,
        port: str = "/dev/serial0",
        baudrate: int = 9600,
        enable_raw: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.enable_raw = enable_raw

        self._fix = GPSFix()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._serial: Optional[serial.Serial] = None

        self._raw_obs_queue: list[UBXRawObs] = []
        self._raw_lock = threading.Lock()

        self.nmea_count = 0
        self.ubx_count = 0
        self.mtk_count = 0
        self.error_count = 0
        self.last_read_time = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("GPS reader started on %s @ %d baud", self.port, self.baudrate)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._serial and self._serial.is_open:
            self._serial.close()
        logger.info("GPS reader stopped")

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

    def get_raw_obs(self) -> list[UBXRawObs]:
        with self._raw_lock:
            obs = self._raw_obs_queue[:]
            self._raw_obs_queue.clear()
            return obs

    def _read_loop(self) -> None:
        retry_delay = 1.0
        while self._running:
            try:
                self._serial = serial.Serial(
                    self.port,
                    baudrate=self.baudrate,
                    timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                logger.info("Serial port %s opened", self.port)
                retry_delay = 1.0

                buf = bytearray()
                while self._running:
                    data = self._serial.read(256)
                    if not data:
                        # Guard against a broken serial adapter (e.g.
                        # CP2102N driver stuck after a kill) where read
                        # returns immediately instead of blocking for the
                        # timeout.  Without this sleep the loop spins at
                        # 100% CPU.
                        time.sleep(0.5)
                        continue
                    self.last_read_time = time.time()
                    buf.extend(data)

                    # Process buffer: extract NMEA and UBX messages
                    buf = self._process_buffer(buf)

            except serial.SerialException as e:
                logger.error("Serial error: %s", e)
                self.error_count += 1
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                self.error_count += 1

            if self._serial and self._serial.is_open:
                self._serial.close()

            if self._running:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10.0)

    def _process_buffer(self, buf: bytearray) -> bytearray:
        """Extract and process complete NMEA/UBX/MTK messages from buffer."""
        pos = 0
        while pos < len(buf):
            if buf[pos] == 0x24:  # '$' — NMEA
                end = buf.find(0x0A, pos)
                if end == -1:
                    break
                # Guard against a 0x24 inside a Quectel MTK binary packet:
                # valid NMEA always begins immediately after \n (or start of buffer).
                # If the previous byte is a printable/binary byte, this is not the
                # start of a sentence; skip to the terminator and keep scanning.
                if pos > 0 and buf[pos - 1] not in (0x0A, 0x0D):
                    pos = end + 1
                    continue
                line = buf[pos : end + 1].decode("ascii", errors="ignore").strip()
                self._handle_nmea(line)
                pos = end + 1
            elif pos + 1 < len(buf) and buf[pos] == 0x04 and buf[pos + 1] == 0x24:
                # Quectel / MTK binary raw-measurement packet (Preamble 0x04 0x24 ... 0x0D 0x0A)
                consumed = self._skip_mtk_packet(buf, pos)
                if consumed > 0:
                    pos += consumed
                else:
                    break  # incomplete, wait for more data
            elif buf[pos] == UBX_SYNC1:
                if pos + 1 >= len(buf):
                    break
                if buf[pos + 1] == UBX_SYNC2:
                    consumed = self._try_parse_ubx(buf, pos)
                    if consumed > 0:
                        pos += consumed
                    else:
                        break  # incomplete, wait for more data
                else:
                    pos += 1
            else:
                pos += 1

        return buf[pos:]

    def _skip_mtk_packet(self, buf: bytearray, pos: int) -> int:
        """Skip a Quectel/MTK binary packet delimited by 0x04 0x24 ... 0x0D 0x0A."""
        end = buf.find(b"\x0d\x0a", pos)
        if end == -1:
            return 0  # incomplete
        # Sanity-check minimum length for a real packet (preamble + cmd + len + cksum)
        if end - pos < 8:
            return 2  # too short, skip preamble and continue scanning
        self.mtk_count += 1
        logger.debug("Skipped MTK packet of %d bytes", end - pos + 2)
        return end - pos + 2

    def _handle_nmea(self, line: str) -> None:
        if not line.startswith("$") or not _nmea_checksum(line):
            return
        self.nmea_count += 1
        body = line[1:].split("*")[0]
        fields = body.split(",")
        if len(fields) < 1:
            return
        talker_msg = fields[0]

        msg_type = None
        for mt in NMEA_PARSERS:
            if talker_msg.endswith(mt):
                msg_type = mt
                break

        if msg_type is None:
            return

        with self._lock:
            NMEA_PARSERS[msg_type](fields, self._fix)

    def _try_parse_ubx(self, buf: bytearray, pos: int) -> int:
        """Try to parse a UBX message at pos. Returns bytes consumed, 0 if incomplete."""
        if pos + 6 > len(buf):
            return 0
        msg_class = buf[pos + 2]
        msg_id = buf[pos + 3]
        length = struct.unpack_from("<H", buf, pos + 4)[0]
        total_len = 8 + length  # sync(2) + class(1) + id(1) + len(2) + payload + cksum(2)

        if pos + total_len > len(buf):
            return 0  # incomplete

        payload = bytes(buf[pos + 6 : pos + 6 + length])
        ck_a = buf[pos + 6 + length]
        ck_b = buf[pos + 7 + length]

        # Verify checksum
        ca, cb = 0, 0
        for b in buf[pos + 2 : pos + 6 + length]:
            ca = (ca + b) & 0xFF
            cb = (cb + ca) & 0xFF
        if ca != ck_a or cb != ck_b:
            return total_len  # skip bad message

        self.ubx_count += 1

        if msg_class == UBX_CLASS_RXM and msg_id == UBX_MSG_RAWX:
            obs = _parse_ubx_rawx(payload)
            if obs:
                with self._raw_lock:
                    self._raw_obs_queue.append(obs)
                    if len(self._raw_obs_queue) > 60:
                        self._raw_obs_queue.pop(0)
                logger.debug("UBX-RAWX: week=%d tow=%.1f sats=%d clk_bias=%.1f m",
                             obs.week, obs.rcv_tow, len(obs.measurements),
                             obs.recv_clk_bias)

        return total_len


# ---------------------------------------------------------------------------
# Utility: configure LEA-6H for raw output (future use)
# ---------------------------------------------------------------------------

def enable_ubx_raw(serial_port: serial.Serial) -> None:
    """
    Send UBX configuration messages to enable RXM-RAWX output on u-blox
    receivers (LEA-6H, NEO-M8N).  Requires the port to be open.
    """
    def _ubx_msg(msg_class: int, msg_id: int, payload: bytes) -> bytes:
        ca, cb = 0, 0
        for b in bytes([msg_class, msg_id]) + struct.pack("<H", len(payload)) + payload:
            ca = (ca + b) & 0xFF
            cb = (cb + ca) & 0xFF
        return bytes([UBX_SYNC1, UBX_SYNC2, msg_class, msg_id]) + \
               struct.pack("<H", len(payload)) + payload + bytes([ca, cb])

    # CFG-MSG (0x06 0x01): enable RXM-RAWX (0x02 0x15) on UART
    # rate=1 means every epoch
    payload = bytes([UBX_CLASS_RXM, UBX_MSG_RAWX, 0x01])  # class, id, rate on port 1 (UART)
    serial_port.write(_ubx_msg(0x06, 0x01, payload))
    logger.info("Sent UBX CFG-MSG to enable RXM-RAWX on UART")


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import signal
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GPS reader test")
    parser.add_argument("--port", default="/dev/serial0", help="Serial port")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--raw", action="store_true", help="Enable UBX raw output")
    args = parser.parse_args()

    reader = GPSReader(args.port, args.baud, enable_raw=args.raw)
    reader.start()

    def _sigint(_sig, _frame):
        reader.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    print(f"Reading GPS from {args.port} @ {args.baud} baud. Ctrl+C to stop.\n")

    try:
        while True:
            fix = reader.get_fix()
            print(
                f"\r{fix.fix_type:6s} sats={fix.num_satellites:2d} "
                f"HDOP={fix.hdop:4.1f} "
                f"lat={fix.latitude:10.6f} lon={fix.longitude:11.6f} "
                f"alt={fix.altitude:7.1f}m "
                f"NMEA={reader.nmea_count} UBX={reader.ubx_count} "
                f"err={reader.error_count}",
                end="", flush=True,
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        reader.stop()
        print("\nDone.")
