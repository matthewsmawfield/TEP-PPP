#!/usr/bin/env python3
"""
TEP-PPP TFT display for Raspberry Pi (Adafruit 2.8" PiTFT, ILI9341, 320x240).

Renders directly to the SPI framebuffer (/dev/fb1) using pygame.  Shows:
  - GPS fix status, coordinates, satellites, DOP
  - Satellite sky plot (elevation / azimuth)
  - TEP clock-bias prediction and skill metrics
  - System status (CPU temp, uptime, NTRIP connection)

Designed for Pi 3 Model B (1 GB RAM): minimal memory footprint, no X11.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pygame

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

WIDTH = 320
HEIGHT = 240

# Colour palette (dark theme for readability on TFT)
BG = (10, 12, 18)
FG = (200, 210, 220)
DIM = (90, 100, 110)
ACCENT = (80, 180, 255)
GREEN = (80, 220, 120)
YELLOW = (240, 200, 80)
RED = (240, 80, 80)
ORANGE = (255, 160, 60)
GREY = (40, 45, 55)

# Layout regions
HEADER_H = 22
FOOTER_H = 20
GPS_INFO_H = 36
SKY_W = 156
SKY_H = HEIGHT - HEADER_H - GPS_INFO_H - FOOTER_H
SKY_X = 0
SKY_Y = HEADER_H + GPS_INFO_H
TEP_X = SKY_W
TEP_Y = SKY_Y
TEP_W = WIDTH - SKY_W
TEP_H = SKY_H


# ---------------------------------------------------------------------------
# Display state
# ---------------------------------------------------------------------------

@dataclass
class DisplayState:
    # GPS
    fix_type: str = "NO FIX"
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    num_sats: int = 0
    hdop: float = 99.0
    satellites: list = field(default_factory=list)  # [(prn, el, az, snr, used)]

    # TEP
    tep_enabled: bool = False
    clock_bias_ns: float = 0.0
    tep_prediction_ns: float = 0.0
    tep_error_ns: float = 0.0
    tep_skill_pct: float = 0.0
    tep_status: str = "IDLE"

    # System
    cpu_temp: float = 0.0
    uptime_str: str = "0m"
    ntrip_status: str = "OFF"
    nmea_count: int = 0
    ubx_count: int = 0
    gps_connected: bool = False

    # Time
    utc_time: str = "--:--:--"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

class FontManager:
    def __init__(self):
        self._fonts: dict[str, pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = f"{size}_{bold}"
        if key not in self._fonts:
            try:
                font = pygame.font.Font(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size
                )
            except (FileNotFoundError, pygame.error):
                font = pygame.font.SysFont("monospace", size)
            font.set_bold(bold)
            self._fonts[key] = font
        return self._fonts[key]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_text(surf, font, text, pos, color=FG, bg=None):
    img = font.render(text, True, color, bg)
    surf.blit(img, pos)


def _draw_bar(surf, x, y, w, h, fill_pct, color, bg_color=GREY):
    pygame.draw.rect(surf, bg_color, (x, y, w, h))
    fill_w = int(w * max(0.0, min(1.0, fill_pct)))
    if fill_w > 0:
        pygame.draw.rect(surf, color, (x, y, fill_w, h))


def _fix_color(fix_type: str) -> tuple:
    if fix_type in ("GPS", "DGPS"):
        return GREEN
    elif fix_type == "NO FIX":
        return RED
    else:
        return YELLOW


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_header(surf, state: DisplayState, fonts: FontManager):
    pygame.draw.rect(surf, BG, (0, 0, WIDTH, HEADER_H))
    pygame.draw.line(surf, DIM, (0, HEADER_H), (WIDTH, HEADER_H))

    f = fonts.get(11, bold=True)
    _draw_text(surf, f, "TEP-PPP", (4, 4), ACCENT)

    fix_c = _fix_color(state.fix_type)
    f_small = fonts.get(10)
    _draw_text(surf, f_small, f"GPS:{state.fix_type}", (70, 5), fix_c)

    ntrip_c = GREEN if state.ntrip_status == "CONN" else DIM
    _draw_text(surf, f_small, f"NTRIP:{state.ntrip_status}", (130, 5), ntrip_c)

    tep_c = ACCENT if state.tep_enabled else DIM
    _draw_text(surf, f_small, f"TEP:ON" if state.tep_enabled else "TEP:OFF",
               (210, 5), tep_c)

    _draw_text(surf, f_small, state.utc_time, (WIDTH - 60, 5), FG)


def render_gps_info(surf, state: DisplayState, fonts: FontManager):
    y0 = HEADER_H
    pygame.draw.rect(surf, BG, (0, y0, WIDTH, GPS_INFO_H))
    pygame.draw.line(surf, DIM, (0, y0 + GPS_INFO_H), (WIDTH, y0 + GPS_INFO_H))

    f = fonts.get(11)
    f_bold = fonts.get(11, bold=True)

    _draw_text(surf, f, f"Lat {state.latitude:10.6f}", (4, y0 + 3), FG)
    _draw_text(surf, f, f"Lon {state.longitude:11.6f}", (4, y0 + 18), FG)

    _draw_text(surf, f_bold, f"{state.altitude:6.1f}m", (170, y0 + 3), FG)
    _draw_text(surf, f, f"S{state.num_sats:2d}", (230, y0 + 3), FG)
    _draw_text(surf, f, f"HDOP {state.hdop:4.1f}", (230, y0 + 18), FG)


def render_sky_plot(surf, state: DisplayState, fonts: FontManager):
    x0, y0 = SKY_X, SKY_Y
    w, h = SKY_W, SKY_H
    pygame.draw.rect(surf, BG, (x0, y0, w, h))

    cx = x0 + w // 2
    cy = y0 + h // 2
    radius = min(w, h) // 2 - 8

    # Concentric circles for elevation rings
    for el, label in [(0, "0"), (30, "30"), (60, "60")]:
        r = radius * (1 - el / 90.0)
        pygame.draw.circle(surf, DIM, (cx, cy), int(r), 1)

    # Cross hairs (N-S, E-W)
    pygame.draw.line(surf, DIM, (cx, cy - radius), (cx, cy + radius))
    pygame.draw.line(surf, DIM, (cx - radius, cy), (cx + radius, cy))

    # Cardinal labels
    f_tiny = fonts.get(8)
    _draw_text(surf, f_tiny, "N", (cx - 3, y0 + 2), DIM)
    _draw_text(surf, f_tiny, "S", (cx - 3, y0 + h - 10), DIM)
    _draw_text(surf, f_tiny, "E", (x0 + w - 8, cy - 4), DIM)
    _draw_text(surf, f_tiny, "W", (x0 + 2, cy - 4), DIM)

    # Plot satellites
    for prn, el, az, snr, used in state.satellites:
        if el < 0:
            continue
        r = radius * (1 - el / 90.0)
        angle = math.radians(az)
        sx = cx + r * math.sin(angle)
        sy = cy - r * math.cos(angle)

        sat_color = GREEN if used else (60, 70, 80)
        sat_size = 3 if used else 2
        pygame.draw.circle(surf, sat_color, (int(sx), int(sy)), sat_size)

        if used and snr > 0:
            label_color = GREEN if snr > 35 else (YELLOW if snr > 20 else ORANGE)
            _draw_text(surf, f_tiny, str(prn), (int(sx) + 4, int(sy) - 4), label_color)


def render_tep_panel(surf, state: DisplayState, fonts: FontManager):
    x0, y0 = TEP_X, TEP_Y
    w, h = TEP_W, TEP_H
    pygame.draw.rect(surf, BG, (x0, y0, w, h))
    pygame.draw.line(surf, DIM, (x0, y0), (x0, y0 + h))

    f_title = fonts.get(10, bold=True)
    f = fonts.get(10)
    f_small = fonts.get(9)

    _draw_text(surf, f_title, "TEP CLOCK", (x0 + 4, y0 + 4), ACCENT)

    if not state.tep_enabled:
        _draw_text(surf, f, "Disabled", (x0 + 4, y0 + 22), DIM)
        _draw_text(surf, f_small, state.tep_status, (x0 + 4, y0 + 38), DIM)
        return

    row = y0 + 22
    _draw_text(surf, f, "Clk ns:", (x0 + 4, row), DIM)
    _draw_text(surf, f, f"{state.clock_bias_ns:8.1f}", (x0 + 56, row), FG)

    row += 14
    _draw_text(surf, f, "Pred:", (x0 + 4, row), DIM)
    _draw_text(surf, f, f"{state.tep_prediction_ns:8.1f}", (x0 + 56, row), ACCENT)

    row += 14
    _draw_text(surf, f, "Err:", (x0 + 4, row), DIM)
    err_color = GREEN if abs(state.tep_error_ns) < 10 else YELLOW
    _draw_text(surf, f, f"{state.tep_error_ns:+8.1f}", (x0 + 56, row), err_color)

    row += 14
    _draw_text(surf, f, "Skill:", (x0 + 4, row), DIM)
    skill_color = GREEN if state.tep_skill_pct > 50 else YELLOW
    _draw_text(surf, f, f"{state.tep_skill_pct:5.1f}%", (x0 + 56, row), skill_color)

    # Skill bar
    row += 14
    _draw_bar(surf, x0 + 4, row, w - 8, 6, state.tep_skill_pct / 100.0, skill_color)

    row += 12
    _draw_text(surf, f_small, state.tep_status, (x0 + 4, row), DIM)


def render_footer(surf, state: DisplayState, fonts: FontManager):
    y0 = HEIGHT - FOOTER_H
    pygame.draw.rect(surf, BG, (0, y0, WIDTH, FOOTER_H))
    pygame.draw.line(surf, DIM, (0, y0), (WIDTH, y0))

    f = fonts.get(9)

    temp_color = RED if state.cpu_temp > 75 else (YELLOW if state.cpu_temp > 65 else FG)
    _draw_text(surf, f, f"CPU {state.cpu_temp:.0f}C", (4, y0 + 4), temp_color)
    _draw_text(surf, f, f"Up {state.uptime_str}", (70, y0 + 4), DIM)

    gps_c = GREEN if state.gps_connected else RED
    _draw_text(surf, f, "GPS", (130, y0 + 4), gps_c)

    _draw_text(surf, f, f"NMEA:{state.nmea_count}", (160, y0 + 4), DIM)
    _draw_text(surf, f, f"UBX:{state.ubx_count}", (230, y0 + 4), DIM)


# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------

def read_cpu_temp() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return 0.0


def read_uptime() -> str:
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
        if secs < 60:
            return f"{int(secs)}s"
        elif secs < 3600:
            return f"{int(secs // 60)}m"
        else:
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            return f"{h}h{m}m"
    except (FileNotFoundError, ValueError, IndexError):
        return "0m"


# ---------------------------------------------------------------------------
# Main display class
# ---------------------------------------------------------------------------

class TEPDisplay:
    """
    Manages the pygame framebuffer surface and renders the TEP-PPP display.

    Call update(state) each epoch to refresh the screen.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.fonts = FontManager()
        self.screen: Optional[pygame.Surface] = None
        self._init_pygame()

    def _init_pygame(self):
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        else:
            os.environ.setdefault("SDL_FBDEV", "/dev/fb1")
            os.environ.setdefault("SDL_NOMOUSE", "1")

        pygame.init()
        try:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        except pygame.error:
            # Fallback to dummy driver if framebuffer not available
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.mouse.set_visible(False)
        self.screen.fill(BG)
        pygame.display.flip()

    def update(self, state: DisplayState):
        if self.screen is None:
            return

        self.screen.fill(BG)
        render_header(self.screen, state, self.fonts)
        render_gps_info(self.screen, state, self.fonts)
        render_sky_plot(self.screen, state, self.fonts)
        render_tep_panel(self.screen, state, self.fonts)
        render_footer(self.screen, state, self.fonts)
        pygame.display.flip()

    def quit(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# CLI test mode (renders a mock display)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import signal

    disp = TEPDisplay(headless="--headless" in sys.argv)

    def _sigint(_sig, _frame):
        disp.quit()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    print("TFT display test. Ctrl+C to stop.")

    mock_sats = []
    for prn in range(1, 13):
        mock_sats.append((
            prn,
            random.uniform(5, 85),
            random.uniform(0, 360),
            random.uniform(15, 45),
            random.random() > 0.3,
        ))

    state = DisplayState(
        fix_type="GPS",
        latitude=52.012345,
        longitude=4.356789,
        altitude=45.2,
        num_sats=8,
        hdop=1.2,
        satellites=mock_sats,
        tep_enabled=True,
        clock_bias_ns=1234.5,
        tep_prediction_ns=1230.1,
        tep_error_ns=4.4,
        tep_skill_pct=53.1,
        tep_status="KRIGING ACTIVE",
        ntrip_status="CONN",
        utc_time="14:23:01",
        gps_connected=True,
    )

    try:
        while True:
            state.cpu_temp = read_cpu_temp()
            state.uptime_str = read_uptime()
            state.nmea_count += random.randint(1, 5)
            state.ubx_count += random.randint(0, 1)
            now = time.strftime("%H:%M:%S", time.gmtime())
            state.utc_time = now

            # Jitter the TEP values slightly
            state.clock_bias_ns += random.uniform(-2, 2)
            state.tep_prediction_ns = state.clock_bias_ns + random.uniform(-5, 5)
            state.tep_error_ns = state.clock_bias_ns - state.tep_prediction_ns

            disp.update(state)
            time.sleep(1.0)
    except KeyboardInterrupt:
        disp.quit()
        print("\nDone.")
