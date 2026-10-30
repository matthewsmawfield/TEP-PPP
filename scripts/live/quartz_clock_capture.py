#!/usr/bin/env python3
"""Satellite-free oscillator phase capture — Step 172 DUT channel(s).

Logs free-running oscillator phase vs a reference timescale to CSV,
in the (t, dut_phase_s) form step_172_satellite_free_holdover.py
scores against the network dep(t) prediction.

Channels (any subset; each column NaN when absent):

  mac        this host's CLOCK_MONOTONIC_RAW — the raw crystal
             counter, UNSTEERED by NTP (macOS only; Linux uses
             CLOCK_MONOTONIC_RAW).  phase = raw - realtime.
  avr        a microcontroller's free-running tick counter over
             serial (e.g. the APM 2.5 / any Arduino running the
             companion sketch in quartz_clock_capture.ino).  No
             satellites, no firmware clock model — the raw crystal.
  sntp:HOST  SNTP offset polls: server_time - local_realtime.  The
             remote host's clock vs the local crystal — a free
             global-oscillator channel (RTT asymmetry adds ~ms
             noise, harmless over the 10-35 uHz band after binning).

Reference: the host's NTP-steered CLOCK_REALTIME unless --ref-sntp
is given (then the first SNTP server doubles as the reference —
fully GNSS-free *local* measurement; remote servers may themselves
be GNSS-disciplined, flagged in the CSV header).

Usage:
  python3 scripts/live/quartz_clock_capture.py \\
      --out data/satfree/quartz_run1.csv \\
      --channels mac avr=/dev/cu.usbmodem* sntp:pool.ntp.org \\
      --period 1.0 --sntp-period 30

The AVR sketch (quartz_clock_capture.ino) just prints micros() each
second; counter values carry the oscillator phase, transport jitter
is irrelevant.
"""
import argparse
import ctypes
import csv
import glob
import socket
import struct
import sys
import threading
import time
from pathlib import Path

_MAC = sys.platform == "darwin"
if _MAC:
    _libc = ctypes.CDLL("/usr/lib/libc.dylib")
    _libc.clock_gettime_nsec_np.restype = ctypes.c_uint64
    _CID_RAW = 4                     # CLOCK_MONOTONIC_RAW
    _CID_REAL = 0                    # CLOCK_REALTIME


def _now_pair():
    if _MAC:
        return (float(_libc.clock_gettime_nsec_np(_CID_RAW)) * 1e-9,
                float(_libc.clock_gettime_nsec_np(_CID_REAL)) * 1e-9)
    # portable fallback: CLOCK_MONOTONIC (steering-free on Linux)
    return time.monotonic(), time.time()


def _sntp_poll(host, timeout=4.0):
    """Return (offset_s, rtt_s) or (None, None)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        pkt = b"\x1b" + 47 * b"\0"
        t0 = time.time()
        s.sendto(pkt, (host, 123))
        data, _ = s.recvfrom(48)
        t1 = time.time()
        s.close()
        ints = struct.unpack("!12I", data)
        srv = ints[10] - 2208988800 + ints[11] / 2 ** 32
        return srv - (t0 + t1) / 2.0, t1 - t0
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", nargs="+", default=["mac"],
                    help="mac | avr=/dev/cu.X | sntp:host")
    ap.add_argument("--period", type=float, default=1.0,
                    help="mac/avr sample period s (default 1.0)")
    ap.add_argument("--sntp-period", type=float, default=30.0,
                    help="SNTP poll period s (default 30)")
    ap.add_argument("--ref-sntp", action="store_true",
                    help="use first sntp: host as the reference "
                         "timescale instead of local REALTIME")
    ap.add_argument("--mark-simulation", action="store_true")
    ap.add_argument("--to-ticks", metavar="CSV",
                    help="convert a capture CSV to step_172 ticks "
                         "format instead of capturing; use with "
                         "--tick-channel and --out")
    ap.add_argument("--tick-channel", default="mac_phase_s")
    args = ap.parse_args()

    if args.to_ticks:
        _to_ticks(args.to_ticks, args.out, args.tick_channel)
        return

    ch_mac = "mac" in args.channels
    avr_paths = [c.split("=", 1)[1] for c in args.channels
                 if c.startswith("avr=")]
    sntp_hosts = [c.split(":", 1)[1] for c in args.channels
                  if c.startswith("sntp:")]
    ref_host = sntp_hosts[0] if (args.ref_sntp and sntp_hosts) else None

    avr_ser = None
    if avr_paths:
        try:
            import serial
        except ImportError:
            sys.exit("avr channel needs pyserial: pip install pyserial")
        cands = []
        for pat in avr_paths:
            cands.extend(sorted(glob.glob(pat)))
        if not cands:
            sys.exit(f"no serial device matching {avr_paths}")
        avr_ser = serial.Serial(cands[0], 115200, timeout=0.05)
        print(f"avr on {cands[0]}")
        time.sleep(2.0)          # AVR reset-on-open
        avr_ser.reset_input_buffer()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = (["epoch_s"]
            + (["mac_phase_s"] if ch_mac else [])
            + (["avr_phase_s"] if avr_ser else [])
            + [f"sntp_{h}_off_s" for h in sntp_hosts]
            + [f"sntp_{h}_rtt_s" for h in sntp_hosts])
    f = out.open("w")
    if args.mark_simulation:
        f.write("# SIMULATION\n")
    f.write(f"# quartz_clock_capture ref="
            f"{'sntp:' + ref_host if ref_host else 'local_REALTIME'}"
            f" host_crystal_class=quartz\n")
    f.write(",".join(cols) + "\n")

    raw0, real0 = _now_pair()
    avr0 = None
    avr_prev = None                # last raw tick for unwrap
    avr_wraps = 0                  # 2^32 us wrap counter (~71.6 min)
    ref0 = None                    # ref_t at capture start
    sntp_last = {h: (float("nan"), float("nan")) for h in sntp_hosts}
    _sntp_lock = threading.Lock()
    _sntp_stop = threading.Event()

    def _sntp_worker():
        while not _sntp_stop.is_set():
            res = {h: _sntp_poll(h) for h in sntp_hosts}
            with _sntp_lock:
                # Failed polls come back (None, None) — store NaN so
                # the row writer emits the documented absent marker
                # and _ref_time's NaN check falls back to REALTIME.
                for h, (off, rtt) in res.items():
                    sntp_last[h] = (
                        off if off is not None else float("nan"),
                        rtt if rtt is not None else float("nan"))
            _sntp_stop.wait(args.sntp_period)

    if sntp_hosts:
        threading.Thread(target=_sntp_worker, daemon=True).start()

    def _ref_time(real):
        """Reference clock reading: local steered REALTIME, or the
        ref-host server time estimated as real + last_offset."""
        if ref_host:
            with _sntp_lock:
                off = sntp_last[ref_host][0]
            if off == off:                      # NaN-safe
                return real + off
        return real

    print(f"writing {out} — ctrl-C to stop")
    try:
        while True:
            raw, real = _now_pair()
            epoch = real
            ref_t = _ref_time(real)
            if ref0 is None:
                ref0 = ref_t
            row = [f"{epoch:.6f}"]
            if ch_mac:
                mac_phase = (raw - raw0) - (ref_t - ref0)
                row.append(f"{mac_phase:.9f}")
            if avr_ser is not None:
                try:
                    line = (avr_ser.readline().decode().strip()
                            if avr_ser.in_waiting else "")
                    if line.isdigit():
                        ticks = int(line)
                        # micros() wraps every 2^32 us (~71.6 min):
                        # unwrap so the phase series stays monotone
                        if (avr_prev is not None and
                                ticks < avr_prev - 2 ** 31):
                            avr_wraps += 1
                        avr_prev = ticks
                        avr_t = (ticks + avr_wraps * 2 ** 32) * 1e-6
                        if avr0 is None:
                            avr0, avr_ref0 = avr_t, ref_t
                        row.append(
                            f"{(avr_t - avr0) - (ref_t - avr_ref0):.9f}")
                    else:
                        row.append("nan")
                except Exception:
                    row.append("nan")
            with _sntp_lock:
                snap = dict(sntp_last)
            for h in sntp_hosts:
                row.append(f"{snap[h][0]:.6f}")
            for h in sntp_hosts:
                row.append(f"{snap[h][1]:.6f}")
            f.write(",".join(row) + "\n")
            f.flush()
            time.sleep(args.period)
    except KeyboardInterrupt:
        pass
    finally:
        _sntp_stop.set()
        f.close()
        if avr_ser is not None:
            avr_ser.close()


def _to_ticks(src, dst, channel):
    """Convert a capture CSV (epoch_s,<channel>,...) into the flat
    step_172 ticks schema (edge_idx,ref_epoch_s,dut_phase_s)."""
    with open(src, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_idx", "ref_epoch_s", "dut_phase_s"])
        for i, r in enumerate(rows):
            try:
                ph = float(r[channel])
                ep = float(r["epoch_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if ph != ph:
                continue
            w.writerow([i, f"{ep:.6f}", f"{ph:.9f}"])
            n += 1
    print(f"{n} ticks -> {out} (channel {channel})")


if __name__ == "__main__":
    main()
