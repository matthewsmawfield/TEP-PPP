# TEP-PPP Raspberry Pi Real-World Test Rig

Hardware:
- **Raspberry Pi 3 Model B** (1.2 GHz quad-core, 1 GB RAM)
- **GPS:** u-blox LEA-6H (initial) → Waveshare LC29H(BS) HAT (dual-band, later)
- **Display:** Adafruit 2.8" PiTFT (ILI9341, 240×320, SPI resistive touch)

## Quick Start

1. Flash Raspberry Pi OS Lite (Bookworm, 64-bit) to a microSD card.
2. Enable SSH and copy this repo to the Pi.
3. Run the setup script:

```bash
sudo bash rpi/setup_pi.sh
sudo reboot
```

After reboot, the TFT should display the TEP-PPP status screen and the
`tep-pi` systemd service will be running.

## Wiring

### GPS (LEA-6H) to Pi GPIO

| GPS Pin | Pi GPIO | Pin # |
|---------|---------|-------|
| TX      | RX      | GPIO 15 (pin 10) |
| RX      | TX      | GPIO 14 (pin 8)  |
| VCC     | 3.3V    | pin 1 or 17      |
| GND     | GND     | pin 6 or 9       |

The Adafruit PiTFT seats directly on the 40-pin header. GPS wires connect
to the PiTFT's pass-through pins.

### LC29H(BS) HAT (later)

The Waveshare LC29H HAT plugs directly onto the GPIO header. No wiring
needed — just seat it on top of the PiTFT (or instead of, if using
stacking headers).

## Files

| File | Purpose |
|------|---------|
| `setup_pi.sh` | One-shot OS config: serial, TFT, RTKLIB, Python, systemd |
| `gps_reader.py` | Threaded GPS serial reader (NMEA + UBX binary) |
| `tep_display.py` | Pygame framebuffer renderer for the PiTFT |
| `tep_pi_service.py` | Main daemon: GPS → TEP → display |
| `gps_web.py` | Web dashboard + the live TEP experiment (see below) |
| `gps_live.py` | Terminal live view |
| `tep_movement_test.py` | Scripted static/movement test protocol |
| `requirements_pi.txt` | Pi-specific Python packages |

## The live TEP experiment (`gps_web.py --ppp`)

Three clock-prior arms are scored on identical epochs and identical
satellite subsets, mirroring Step 106:

| Arm | Clock prior |
|-----|-------------|
| `free` | none |
| `persist` | the receiver's own causally-fitted clock trend, extrapolated — **the baseline the headline claim is made against** |
| `tep` | that own-trend **plus** the network's kriged TEP field residual, plus astronomical phase-modulated clock process noise, plus the EW/NS anisotropy constraint |

Scored on **accuracy** — held-out clock prediction error, and 3-D position
RMSE about a truth reference — never on trajectory scatter. Scatter measures
smoothness: whichever arm has the smaller gain wins it regardless of whether
it is closer to the truth, so it is reported only as a diagnostic.

### ⚠ Capture length is not a free parameter

**The TEP field's exploitable power lies below 500 µHz** — periods of 33
minutes to 5 days (Steps 96/106). A capture shorter than one band period
contains **no TEP signal at all**: its entire bandwidth sits above the
field, so the comparison measures receiver steering and code noise.

A 120 s capture resolves nothing below 8.3 mHz — **16× above the band**.
It will show TEP ≈ 0% no matter how well everything else works. This was
the original cause of the rig reporting a null result.

| `--ppp-capture` | Verdict |
|-----------------|---------|
| < 2000 s | `out_of_band` — cannot show TEP, flagged red on the dashboard |
| ≥ 2000 s | `marginal` — only the top of the band |
| ≥ 3600 s | `good` — matches Step 106's 1 h outage |
| **≥ 14400 s** | **`optimal` — matches Step 106's 4 h outage (its strongest result, +48%). This is the default.** |

### Rolling window, not a block capture

`--ppp-capture` is a **rolling analysis window**, not a blocking capture.
The receiver is read continuously in `--ppp-chunk` slices (default 900 s);
after every chunk the whole window is re-scored. So the first numbers
appear one chunk in and improve as the window fills toward 4 h — you are
not blind for four hours.

NMEA is demultiplexed out of the same byte stream and fed to the GPS
reader, so **the live map and satellite view keep working throughout** a
multi-hour capture.

```bash
# Default: 4 h rolling window, re-scored every 15 min. Just run it.
python3 rpi/gps_web.py --ref-ecef 3789577.96,6807.84,5113119.85

# Faster feedback while setting up (still reaches the 4 h window)
python3 rpi/gps_web.py --ppp-chunk 300 --ref-ecef X,Y,Z
```

Without `--ref-ecef` the position truth is the session running mean, which
is only valid for a **static** receiver and converges slowly.

### Ionosphere-free PPP on single-frequency data (GRAPHIC)

The LC29H is a dual-band L1+L5 receiver, but the standard RTKLIB `convbin`
outputs RINEX 2.11, which has no L5 field — so the second frequency is
dropped and only C1/L1 survive in the RINEX file. Without an
ionosphere-free combination the residual ionosphere after Klobuchar
(~30-50% of the total, 1-5 m) dominates the error budget and caps accuracy
at metre level.

The rig therefore uses the **GRAPHIC** (Group and Phase Ionospheric
Correction) observable when only L1 is available:

    P_GRAPHIC = (C1 + L1·λ) / 2

The code pseudorange is delayed by the ionosphere (+I) and the carrier
phase is advanced by it (−I), so averaging the two cancels the
first-order ionosphere entirely. The remaining constant (the effective
ambiguity N·λ/2) is absorbed by the EKF's float ambiguity state. The
GRAPHIC noise is code-level (~0.3-3 m / √2), not millimetre-level like
dual-frequency IF phase, so the EKF weights it accordingly — but the
ionosphere, which Klobuchar only removes partially, is gone completely.

When dual-frequency observations are available (RINEX 3.x with L5
preserved), the standard L1/L5 ionosphere-free combination is used
instead, with millimetre-level carrier-phase noise and faster convergence.

The dashboard reports which mode is active via the `freq_mode` field:
`dual_l1l5`, `dual_l1l2`, or `single_l1_graphic`.

The dashboard also reports the **separation between this receiver and the
network's target station** against λ_T ≈ 4200 km. That is the TEP claim
being exercised: a remote station's clock residual can only help here if
the proper-time field is genuinely coherent across that distance.

### Provenance

The kriging server can replay archived data (`"mode": "simulation"`). The
dashboard states which mode produced every number — a result computed
against replayed data is **not** a live-network result, even though the
GNSS observations are live hardware either way.

## Manual Operation

```bash
# Stop the service
sudo systemctl stop tep-pi

# Run manually with logging
cd /path/to/TEP-PPP
~/.venv/tep-pi/bin/python rpi/tep_pi_service.py

# Test GPS reader only
~/.venv/tep-pi/bin/python rpi/gps_reader.py --port /dev/serial0 --baud 9600

# Test display only (mock data)
~/.venv/tep-pi/bin/python rpi/tep_display.py

# Test display without TFT (headless)
~/.venv/tep-pi/bin/python rpi/tep_display.py --headless
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPS_SERIAL_PORT` | auto-detect | GPS serial device (probes `/dev/serial0`, then `/dev/cu.usbserial*` etc.) |
| `GPS_BAUDRATE` | `9600` | Serial baud rate |
| `NTRIP_ENABLED` | `0` | Enable NTRIP reference-station connection |
| `TEP_ENABLED` | `1` | Enable TEP clock processing |
| `DISPLAY_HEADLESS` | `0` | Skip framebuffer (for testing) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `UPDATE_INTERVAL` | `1.0` | Display refresh interval (seconds) |

`gps_web.py` additionally honours:

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_HOST` | `127.0.0.1` | Bind address. The dashboard is **unauthenticated** and exposes live position — bind `0.0.0.0` only on a trusted network |
| `WEB_PORT` | `8000` | Dashboard port |
| `TEP_KRIGING_URL` | `http://localhost:8765` | Kriging server (`scripts/live/tep_realtime_server.py`, default port 8765) |
| `TEP_KRIGING_ENABLED` | `1` | Set `0` to disable the network term |
| `TEP_PPP_ENABLED` | `1` | Set `0` to disable the TEP experiment |
| `TEP_PPP_CAPTURE_S` | `14400` | Rolling analysis window (4 h = optimal) |
| `TEP_PPP_CHUNK_S` | `900` | Capture chunk / re-score cadence |
| `TEP_REF_ECEF` | — | Surveyed truth position `X,Y,Z` in ECEF metres |

## Display Layout

```
+-------------------------------------------+
| TEP-PPP  GPS:GPS  NTRIP:OFF  TEP:ON  time |  Header
+-------------------------------------------+
| Lat 52.012345   Lon  4.356789             |
| Alt  45.2m      S08  HDOP 1.2             |  GPS Info
+-------------------+------------------------+
|                   | TEP CLOCK              |
|    Sky Plot       | Clk ns:    1234.5      |
|    (satellites)   | Pred:      1230.1      |
|                   | Err:        +4.4       |
|                   | Skill:      53.1%      |
|                   | [████████░░]           |  Sky + TEP
+-------------------+------------------------+
| CPU 52C  Up 2h15m  GPS  NMEA:12345 UBX:42 |  Footer
+-------------------------------------------+
```

## Upgrading to LC29H(BS)

When switching from LEA-6H to the Waveshare LC29H(BS) HAT:

1. The LC29H connects via UART (same `/dev/serial0`) but may need a
   different baud rate (115200 default for Quectel).
2. Enable raw observation output via Quectel QZSS protocol commands.
3. Update the service environment:

```bash
sudo systemctl edit tep-pi
# Add:
# [Service]
# Environment=GPS_BAUDRATE=115200
```

4. The `gps_reader.py` UBX parser will need a Quectel raw-message parser
   added (different binary protocol than u-blox). This is a planned
   enhancement.

## Troubleshooting

**No GPS fix:** Check wiring, antenna placement (outdoor with clear sky
view), and that `/dev/serial0` exists. The LEA-6H can take 2-5 minutes
for a cold start fix.

**TFT blank:** Verify `dtoverlay=pitft28-resistive` in `/boot/config.txt`
and that `/dev/fb1` exists after reboot. Check `dmesg | grep fbtft`.

**Service crashes:** Check `journalctl -u tep-pi -f` for errors. Common
issues: serial port permissions (user must be in `dialout` group),
missing `/dev/fb1` (TFT overlay not loaded).

**High CPU temp:** The Pi 3 can hit 70C+ in enclosed setups. Add a small
heatsink or fan. The display shows CPU temp in red above 75C.
