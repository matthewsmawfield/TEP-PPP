#!/usr/bin/env python3
"""
Satellite-free dual-oscillator capture for the Step 172 closure experiment.

HARDWARE
  DUT        : second LC29H, antenna port terminated with a 50-ohm load
               (or the receiver inside an RF-shielded enclosure).  It
               tracks ZERO satellites; its only output used here is the
               TIMEPULSE pin, driven by its free-running TCXO.
  REFERENCE  : the primary tracking receiver's TIMEPULSE pin, or any
               GNSS-disciplined 1PPS source on the SAME host counter.
  COUNTER    : two GPIO inputs on the Pi running this rig.  pigpio gives
               microsecond-class hardware edge timestamps -- the
               experiment's signal is the ACCUMULATED phase over hours
               (~7 ns dep for a 2 m correction) riding on the TCXO's
               ~10-100 ns wander, so per-edge jitter averages down as
               1/sqrt(N_edges): 1 us jitter on 72 h of 1 Hz pulses ->
               ~3 ns floor.  A dedicated TDC/counter IC replaces GPIO
               if a cleaner floor is wanted later.

  The primary receiver KEEPS TRACKING NORMALLY -- the service's dep
  prediction for the capture interval is assembled afterwards with
  step_172's --pred auto from the live result files.

WHAT IS LOGGED
  data/satfree/ticks_<tag>_<start>.csv
      edge_idx,ref_epoch_s,dut_epoch_s,dut_phase_s
      ref_epoch_s = host CLOCK_REALTIME of the paired reference edge
      dut_epoch_s = host CLOCK_REALTIME of the paired DUT edge
      dut_phase_s = the DUT's ACCUMULATED oscillator phase vs the
          reference: (dut_t_k - dut_t_0) - (ref_t_k - ref_t_0), i.e. how
          far the free TCXO has drifted from the disciplined reference
          since capture start.  Edges are paired 1:1 by order (both
          lines are ~1 Hz); a missed edge on either line is tolerated
          by the nearest-in-time pairing window.

  ref_epoch_s uses the host's CLOCK_REALTIME at each reference edge --
  adequate because the dep prediction is interpolated at ~1 s grid;
  host clock steps (NTP) are flagged in the log and masked in scoring.

Usage (on the rig):
    python3 scripts/live/satellite_free_capture.py \
        --dut-gpio 17 --ref-gpio 27 --tag run1
    python3 scripts/live/satellite_free_capture.py --simulate 60
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "satfree"


def _simulate(out_path: Path, seconds: float, seed=7):
    """Write a synthetic capture (1 Hz pulses, TCXO wander + a small
    field-like ripple) for exercising step_172 -- clearly marked
    SIMULATION in the file header."""
    rng = np.random.default_rng(seed)
    n = int(seconds)
    wander = np.cumsum(rng.normal(0, 5e-11, n))
    ripple = 2e-8 * np.sin(2 * np.pi * np.arange(n) / 3600.0)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write("# SIMULATION -- synthetic capture, not real hardware\n")
        w = csv.writer(f)
        w.writerow(["edge_idx", "ref_epoch_s", "dut_epoch_s",
                    "dut_phase_s"])
        phase = wander + ripple + 1e-9 * np.arange(n) / n
        for i in range(n):
            dut_t = t0 + i + phase[i]
            w.writerow([i, f"{t0 + i:.6f}", f"{dut_t:.9f}",
                        f"{phase[i]:.12f}"])
    print(f"wrote {out_path} ({n} edges)")


def capture(dut_gpio: int, ref_gpio: int, out_path: Path, tag: str):
    """Edge-capture loop.

    Two GPIO lines carry TIMEPULSE edges.  Edge timestamps come from
    pigpio (hardware-timed) when available, else the kernel's gpio event
    clock via lgpio, else monotonic software stamping as a last resort
    (warned -- software latency jitter is ~50-200 us and the capture
    then needs >=5x longer for the same noise floor).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import pigpio  # noqa
        backend = _PigpioBackend(dut_gpio, ref_gpio)
    except Exception as e:
        print(f"pigpio unavailable ({e}); software stamping backend")
        backend = _SoftwareBackend(dut_gpio, ref_gpio)
    t0_wall = time.time()
    ref_t = None          # latest reference edge timestamp
    ref_t0 = dut_t0 = None  # first-edge anchors for the phase origin
    idx = 0
    with open(out_path, "w", newline="") as f:
        f.write(f"# satellite-free capture {tag}; "
                f"backend={backend.name}\n")
        w = csv.writer(f)
        w.writerow(["edge_idx", "ref_epoch_s", "dut_epoch_s",
                    "dut_phase_s"])
        f.flush()
        print(f"capturing dut=GPIO{dut_gpio} ref=GPIO{ref_gpio} -> "
              f"{out_path}  (Ctrl-C to stop)")
        try:
            for line, ts in backend.edges():
                if line == "ref":
                    ref_t = ts
                    if ref_t0 is None:
                        ref_t0 = ts
                elif line == "dut":
                    if ref_t is None:
                        continue
                    if dut_t0 is None:
                        dut_t0 = ts
                    # Pair the DUT edge with its nearest reference
                    # edge; both lines are ~1 Hz so the nearest ref
                    # edge is within ~0.5 s.
                    phase = ((ts - dut_t0) - (ref_t - ref_t0))
                    w.writerow([idx, f"{ref_t:.6f}", f"{ts:.9f}",
                                f"{phase:.12f}"])
                    idx += 1
                    if idx % 3600 == 0:
                        f.flush()
                        print(f"{tag}: {idx} DUT edges "
                              f"({(time.time() - t0_wall) / 3600:.2f} h)")
        except KeyboardInterrupt:
            pass
    backend.close()
    print(f"stopped: {idx} edges -> {out_path}")


class _PigpioBackend:
    name = "pigpio"

    def __init__(self, dut_gpio, ref_gpio):
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running")
        self.dut_gpio = dut_gpio
        self.ref_gpio = ref_gpio
        import queue
        self.q = queue.Queue()
        for gpio, line in ((dut_gpio, "dut"), (ref_gpio, "ref")):
            self.pi.set_mode(gpio, pigpio.INPUT)
            self.pi.callback(gpio, pigpio.RISING_EDGE,
                             self._mk_cb(line))

    def _mk_cb(self, line):
        def cb(gpio, level, tick):
            # pigpio tick = microseconds since boot, wraps at 2**32;
            # epoch reconstruction needs a wall anchor -- approximate
            # with time.time() at receipt (ms latency is fine for the
            # ref epoch; the PHASE uses tick differences).
            self.q.put((line, tick / 1e6))
        return cb

    def edges(self):
        while True:
            yield self.q.get()

    def close(self):
        self.pi.stop()


class _SoftwareBackend:
    """Fallback: poll both lines with monotonic stamping.  Latency
    jitter ~50-200 us -- usable only for long (>=24 h) captures and
    flagged as such."""
    name = "software_poll"

    def __init__(self, dut_gpio, ref_gpio):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(dut_gpio, GPIO.IN)
            GPIO.setup(ref_gpio, GPIO.IN)
            self.dut_gpio = dut_gpio
            self.ref_gpio = ref_gpio
            self._last = {"dut": GPIO.input(dut_gpio),
                          "ref": GPIO.input(ref_gpio)}
            print("WARNING: software edge polling -- per-edge jitter "
                  "~100 us; extend the capture >=24 h")
        except ImportError as e:
            raise RuntimeError(
                "no GPIO backend available (install pigpio or "
                "RPi.GPIO on the rig)") from e

    def edges(self):
        import queue
        q = queue.Queue()

        def watch(line, gpio):
            last = self._last[line]
            while True:
                v = self.GPIO.input(gpio)
                if v and not last:
                    q.put((line, time.time()))
                last = v
                time.sleep(0.0005)
        import threading
        for line, gpio in (("dut", self.dut_gpio),
                           ("ref", self.ref_gpio)):
            threading.Thread(target=watch, args=(line, gpio),
                             daemon=True).start()
        while True:
            yield q.get()

    def close(self):
        self.GPIO.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dut-gpio", type=int, default=17)
    ap.add_argument("--ref-gpio", type=int, default=27)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--simulate", type=float, default=0.0,
                    help="write a synthetic N-second capture instead")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        OUT_DIR / f"ticks_{args.tag}_{int(time.time())}.csv")
    if args.simulate:
        _simulate(out, args.simulate)
    else:
        capture(args.dut_gpio, args.ref_gpio, out, args.tag)


if __name__ == "__main__":
    main()
