#!/usr/bin/env python3
"""
Live telescope image viewer for the Adafruit 2.8" PiTFT.

Fetches FRESH telescope observations from multiple sources:
  - ESA/Hubble RSS feed (latest Hubble releases)
  - NOIRLab RSS feed (Rubin, Gemini, Blanco — ground-based surveys)
  - ESO Picture of the Week (VLT, ALMA)
  - NASA image archive (Hubble, JWST, Spitzer, Chandra — infinite backbone)

Only real telescope observations of deep-sky objects (galaxies, nebulae,
star clusters, supernova remnants).  No Sun, no planets, no hardware
photos, no illustrations, no people.

A background thread prefetches the next image so transitions are
instant.  The viewer runs INDEFINITELY — it never stops when a source
fails, a queue empties, or a worker throws an exception.
"""
import os
import re
import time
import json
import random
import logging
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from collections import deque

import numpy as np
import pygame

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("telescope-tft")

WIDTH, HEIGHT = 320, 240
FB = "/dev/fb1"
DISPLAY_SECONDS = 20
CACHE_DIR = Path("/tmp/telescope_cache")
CACHE_DIR.mkdir(exist_ok=True)

# GPS web dashboard API (localhost on the Pi)
GPS_WEB_URL = "http://localhost:8000"
_gps_cache: dict = {}
_gps_last_fetch: float = 0.0
_GPS_FETCH_INTERVAL = 5.0  # refresh every 5s


def _fetch_gps_status():
    """Fetch headline GPS numbers from the web dashboard.

    Returns a dict with the fields needed for the top overlay.  Cached
    for _GPS_FETCH_INTERVAL seconds so the display loop (which runs every
    20s per image) doesn't hammer the API.  Fails silently — if gps_web
    isn't running, the overlay just shows nothing.
    """
    global _gps_cache, _gps_last_fetch
    now = time.time()
    if now - _gps_last_fetch < _GPS_FETCH_INTERVAL:
        return _gps_cache
    _gps_last_fetch = now
    try:
        fix = http_get_json(f"{GPS_WEB_URL}/api/fix", timeout=3)
        n_sats = fix.get("num_sats", 0)
        fix_type = fix.get("fix_type", "NO FIX")
        hdop = fix.get("hdop", 99.0)
        pdop = fix.get("pdop", 99.0)
        satellites = fix.get("satellites", []) or []
    except Exception:
        _gps_cache = {}
        return _gps_cache
    ppp = {}
    try:
        ppp = http_get_json(f"{GPS_WEB_URL}/api/ppp", timeout=3)
    except Exception:
        pass
    fs = ppp.get("position_fullsky", {}) or {}
    clk = ppp.get("clock_prediction", {}) or {}
    deg = ppp.get("position_degraded", {}) or {}
    ar_block = ppp.get("ppp_ar", {}) or {}
    ar_acc = fs.get("ppp_ar_accuracy_m")
    if ar_acc is None and ar_block.get("available"):
        ar_acc = ar_block.get("rmse_3d_m")
    ar_fix_rate = fs.get("ppp_ar_fix_rate")
    if ar_fix_rate is None and ar_block.get("available"):
        ar_fix_rate = ar_block.get("fix_rate")
    converged = fs.get("converged_rmse_3d_m")
    rmse_2d = fs.get("rmse_2d_m")
    fullsky_src = fs.get("source", "")
    n_sat_mean = fs.get("n_sat_mean", 0)
    tep_vs_persist = clk.get("tep_vs_persist_pct")
    deg_tep_vs_persist = deg.get("tep_vs_persist_pct")
    deg_tep_vs_free = deg.get("tep_vs_free_pct")
    available = ppp.get("available", False)
    # SPP-level position accuracy from the fix API (always available)
    tep_block = (fix.get("tep") or {})
    spp_acc = tep_block.get("accuracy", {}) or {}
    spp_rmse_2d = spp_acc.get("raw_rmse_2d")
    spp_tep_rmse_2d = spp_acc.get("tep_rmse_2d")
    spp_advantage = tep_block.get("tep_advantage_pct")
    _gps_cache = {
        "n_sats": n_sats,
        "n_sat_mean": n_sat_mean,
        "fix_type": fix_type,
        "hdop": hdop,
        "pdop": pdop,
        "satellites": satellites,
        "ar_accuracy_m": ar_acc,
        "ar_fix_rate": ar_fix_rate,
        "converged_m": converged,
        "rmse_2d_m": rmse_2d,
        "fullsky_source": fullsky_src,
        "tep_vs_persist_pct": tep_vs_persist,
        "deg_tep_vs_persist_pct": deg_tep_vs_persist,
        "deg_tep_vs_free_pct": deg_tep_vs_free,
        "ppp_available": available,
        "spp_rmse_2d": spp_rmse_2d,
        "spp_tep_rmse_2d": spp_tep_rmse_2d,
        "spp_advantage_pct": spp_advantage,
    }
    return _gps_cache


def draw_gps_overlay(surf, font):
    """Draw a subtle single-line top bar with the most goal-relevant metrics.

      SATS <n>  HDOP <x.x>  POS <cm/m>  TEP <+/->%  DEG <+/->%
    """
    gps = _fetch_gps_status()
    if not gps:
        return
    bar_h = 18
    bar = pygame.Surface((WIDTH, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    surf.blit(bar, (0, 0))
    light_blue = (120, 200, 255)
    good = (100, 255, 100)
    bad = (255, 170, 100)

    n_sats = gps.get("n_sats", 0)
    hdop = gps.get("hdop", 99.0)
    ar = gps.get("ar_accuracy_m")
    conv = gps.get("converged_m")
    rmse_2d = gps.get("rmse_2d_m")
    spp_rmse = gps.get("spp_rmse_2d")
    tep_pct = gps.get("tep_vs_persist_pct")
    deg_pct = gps.get("deg_tep_vs_free_pct")
    spp_adv = gps.get("spp_advantage_pct")

    segments = []
    colors = []
    if n_sats:
        segments.append(f"SATS {n_sats}")
        colors.append(light_blue)
    if hdop < 99:
        segments.append(f"HDOP {hdop:.1f}")
        colors.append(light_blue)
    # Position accuracy: PPP-AR (best) > PPP converged > SPP RMSE
    if ar is not None and ar > 0:
        segments.append(f"POS {ar*100:.1f}cm")
        colors.append(good if ar < 0.1 else light_blue)
    elif conv is not None and conv > 0:
        segments.append(f"POS {conv:.1f}m")
        colors.append(light_blue)
    elif spp_rmse is not None and spp_rmse > 0:
        segments.append(f"POS {spp_rmse:.1f}m")
        colors.append(light_blue)
    if tep_pct is not None:
        sign = "+" if tep_pct >= 0 else ""
        segments.append(f"TEP {sign}{tep_pct:.1f}%")
        colors.append(good if tep_pct >= 0 else bad)
    if deg_pct is not None:
        sign = "+" if deg_pct >= 0 else ""
        segments.append(f"DEG {sign}{deg_pct:.1f}%")
        colors.append(good if deg_pct >= 0 else bad)
    if not segments:
        return
    widths = [font.size(s)[0] for s in segments]
    n_gap = len(segments) - 1
    gap = max(4, (WIDTH - 8 - sum(widths)) // n_gap) if n_gap > 0 else 0
    x = 4
    for s, w, c in zip(segments, widths, colors):
        draw_text(surf, s, (x, 2), font, color=c)
        x += w + gap


def draw_skyplot(surf, font_tiny=None):
    """Draw a polar satellite skymap in the top-left, matching the web-gis style.

    Placed just below the top metrics bar.  Centre = zenith, outer ring = horizon.
    Concentric elevation rings with labels, cardinal axes, used sats
    in green with white outline, unused in grey (or red if no SNR).
    """
    import math
    gps = _fetch_gps_status()
    sats = gps.get("satellites", [])
    if not sats:
        return
    bar_h = 18
    r_max = 38                          # outer radius (horizon)
    cx = r_max + 4                      # centre x (left margin)
    cy = bar_h + r_max + 4              # centre y (just below the bar)

    # Semi-transparent background circle
    bg_size = r_max * 2 + 12
    bg = pygame.Surface((bg_size, bg_size), pygame.SRCALPHA)
    pygame.draw.circle(bg, (0, 0, 0, 140), (bg_size // 2, bg_size // 2), r_max + 4)
    surf.blit(bg, (cx - bg_size // 2, cy - bg_size // 2))

    ring_color = (120, 180, 220, 90)
    axis_color = (120, 180, 220, 50)
    label_color = (148, 163, 184)

    # Concentric elevation rings: 0°, 30°, 60°
    for el_ring in [0, 30, 60]:
        rr = int(r_max * (1 - el_ring / 90.0))
        ring = pygame.Surface((rr * 2 + 4, rr * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ring, ring_color, (rr + 2, rr + 2), rr, 1)
        surf.blit(ring, (cx - rr - 2, cy - rr - 2))

    # Cardinal axes (N-S, E-W)
    axis = pygame.Surface((r_max * 2 + 4, r_max * 2 + 4), pygame.SRCALPHA)
    pygame.draw.line(axis, axis_color, (r_max + 2, 2), (r_max + 2, r_max * 2 + 2), 1)
    pygame.draw.line(axis, axis_color, (2, r_max + 2), (r_max * 2 + 2, r_max + 2), 1)
    surf.blit(axis, (cx - r_max - 2, cy - r_max - 2))

    # Plot each satellite
    for s in sats:
        el = s.get("elevation", 0)
        az = s.get("azimuth", 0)
        snr = s.get("snr", 0)
        used = s.get("used", False)
        if el < 0:
            continue
        r = r_max * (1 - el / 90.0)
        az_rad = math.radians(az)
        sx = cx + int(r * math.sin(az_rad))
        sy = cy - int(r * math.cos(az_rad))
        if used:
            color = (100, 255, 100)
            dot_r = 2
            pygame.draw.circle(surf, color, (sx, sy), dot_r)
            pygame.draw.circle(surf, (255, 255, 255), (sx, sy), dot_r, 1)
        else:
            if snr > 0:
                color = (120, 130, 148)
            else:
                color = (180, 60, 60)
            dot_r = 1
            pygame.draw.circle(surf, color, (sx, sy), dot_r)

NASA_IMAGE_API = "https://images-api.nasa.gov/search"
ESA_HUBBLE_RSS = "https://esahubble.org/rss/feed.xml"
ESA_CDN = "https://cdn.esahubble.org/archives/images/screen/"
NOIRLAB_RSS = "https://noirlab.edu/public/news/feed/"
NOIRLAB_IMG_CDN_NEWS = "https://storage.noirlab.edu/media/archives/images/news/"
NOIRLAB_IMG_CDN_FEATURE = "https://storage.noirlab.edu/media/archives/images/newsfeature/"
ESO_POTW_FEED = "https://www.eso.org/public/images/feed/"

# Telescope+object queries for the NASA archive (infinite backbone)
TELESCOPE_QUERIES = [
    "Hubble galaxy", "Hubble nebula", "Hubble star cluster",
    "Hubble supernova", "Hubble deep field", "Hubble spiral galaxy",
    "Hubble globular cluster", "Hubble planetary nebula",
    "Hubble andromeda", "Hubble whirlpool", "Hubble sombrero",
    "Hubble crab nebula", "Hubble eagle nebula", "Hubble ring nebula",
    "Hubble orion nebula", "Hubble pillars of creation",
    "Hubble black hole", "Hubble quasar", "Hubble gravitational lens",
    "Hubble tarantula", "Hubble carina", "Hubble lagoon",
    "Hubble triangulum", "Hubble centaurus", "Hubble virgo",
    "Hubble coma cluster", "Hubble hercules", "Hubble antennae",
    "Hubble cigar galaxy", "Hubble tadpole galaxy",
    "Webb galaxy", "Webb nebula", "Webb deep field",
    "Webb spiral galaxy", "Webb cosmic", "Webb supernova",
    "Webb planetary nebula", "Webb pillar", "Webb cosmic cliffs",
    "Webb cartwheel", "Webb stephan quintet", "Webb smacs",
    "Webb tarantula", "Webb ring nebula", "Webb orion",
    "Spitzer galaxy", "Spitzer nebula", "Spitzer star cluster",
    "Spitzer supernova", "Spitzer planetary nebula",
    "Chandra galaxy", "Chandra nebula", "Chandra supernova",
    "Chandra black hole", "Chandra pulsar", "Chandra star cluster",
    "Chandra cassiopeia", "Chandra crab", "Chandra centaurus",
]

# Words that indicate a non-observation image.
# These are checked against the TITLE only (not the description,
# which is too verbose and causes false positives like "team of
# astronomers" or "Earth's atmosphere" in valid observation records).
# Matching uses word boundaries to avoid substring false positives
# (e.g. "sun" matching "sunlight", "team" matching "steam").
SKIP_WORDS_TITLE = [
    "illustration", "diagram", "concept", "artwork",
    "infographic", "graphic", "logo", "portrait",
    "mirror", "sunshield", "deployment", "launch",
    "sitemap", "badge", "patch",
    "history of", "instrument photo",
    "spacecraft", "antenna", "dish",
    "facility", "building", "construction",
    "astronaut", "mission patch", "program manager",
    "ceremony", "staff", "crew photo",
    "photo of", "picture of",
    "philharmonic", "symphony", "concert", "perform",
    "festival", "exhibit", "exhibition", "museum",
    "school", "student", "teacher", "classroom",
    "lecture", "speech", "presentation",
    "magazine", "interview", "documentary",
    "anniversary", "tribute", "memorial", "award",
    "iss0", "iss1", "iss03", "iss04", "iss05", "iss06",
    "iss07", "iss08", "iss09", "expedition",
    "jaxa", "noaa", "usgs", "landsat", "modis",
    "hurricane", "storm", "aerial", "astronaut photo",
    "cassini", "voyager", "new horizons",
    "rover", "perseverance", "curiosity",
    "sdo", "ksc-", "n/a", "shuttle", "space shuttle",
    "prelaunch", "pre-launch", "pre launch",
    "deployed", "deploy", "antennae", "antenna",
    "golden spider", "high gain",
    "legacy", "future of", "history",
    "sts-", "rocket", "ariane",
    "first full-color", "first full color",
    "electronic still came",
]

# Planet/body names — only checked in title, with word boundaries.
# These are solar system objects, not deep-sky observations.
SKIP_BODIES_TITLE = [
    "mars", "saturn", "jupiter", "pluto", "venus",
    "mercury", "comet", "asteroid", "moon", "lunar",
    "sun", "solar", "corona", "chromosphere",
]

# Hard-rejection words checked against BOTH title and description
# (with word boundaries).  Very small set — only things that are
# never part of a valid observation record.
SKIP_WORDS_HARD = [
    "philharmonic", "symphony", "concert",
    "exhibition", "museum", "classroom",
    "interview", "documentary",
]

# Positive keywords — NASA archive items MUST contain at least one
# of these in the title to be accepted.  This filters out hardware
# photos, mission patches, crew photos, etc. that have telescope
# names but aren't actual observations.
OBSERVATION_KEYWORDS = [
    "galaxy", "galaxies", "nebula", "nebulae",
    "cluster", "supernova", "remnant", "quasar",
    "pulsar", "black hole", "star", "stellar",
    "spiral", "elliptical", "dwarf",
    "field", "deep field", "cosmic",
    "andromeda", "orion", "carina", "tarantula",
    "crab", "eagle", "ring", "lagoon",
    "whirlpool", "sombrero", "pinwheel",
    "triangulum", "centaurus", "virgo",
    "coma", "hercules", "antennae",
    "magellanic", "milky way",
    "pillar", "cliffs", "mountain",
    "bubble", "jet", "outflow",
    "disk", "dust", "gas",
    "horizon", "lens", "lensing",
    "nova", "binary", "exoplanet",
    "sagittarius", "cassiopeia",
    "cygnus", "lyra", "veil",
    "planetary nebula",
    "globular", "open cluster",
    "barred spiral", "irregular",
    "group", "pair", "interacting",
    "merger", "merging", "collision",
    "nucleus", "core", "halo",
]

import re as _re
_SKIP_TITLE_RE = _re.compile(
    r"(?:" + "|".join(_re.escape(w) for w in SKIP_WORDS_TITLE) + r")",
    _re.IGNORECASE)
_SKIP_BODIES_RE = _re.compile(
    r"\b(?:" + "|".join(_re.escape(w) for w in SKIP_BODIES_TITLE) + r")\b",
    _re.IGNORECASE)
_SKIP_HARD_RE = _re.compile(
    r"\b(?:" + "|".join(_re.escape(w) for w in SKIP_WORDS_HARD) + r")\b",
    _re.IGNORECASE)
_OBS_KEYWORD_RE = _re.compile(
    r"\b(?:" + "|".join(_re.escape(w) for w in OBSERVATION_KEYWORDS) + r")\b",
    _re.IGNORECASE)

# Rolling history of recently-shown URLs — items cycle back after
# RECENT_HISTORY images, so the display never gets stuck but also
# avoids immediate repeats.  With 300, an item won't reappear for
# ~100 minutes (300 × 20s), by which point the archive has supplied
# hundreds of unique images.
RECENT_HISTORY = 300
_recently_shown: deque = deque(maxlen=RECENT_HISTORY)

# When every source fails (network outage), stop hammering dead
# endpoints — fail fast until this timestamp, then try again.
OFFLINE_BACKOFF_S = 120.0
_offline_until: float = 0.0


def _was_recently_shown(url: str) -> bool:
    return url in set(_recently_shown)


def _mark_shown(url: str):
    _recently_shown.append(url)


def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={
        "User-Agent": "tep-pi-telescope-viewer/1.0"
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_json(url, timeout=15):
    return json.loads(http_get(url, timeout))


def _is_observation(title, desc, require_keyword=False):
    """Check if a title+description looks like a real telescope observation.

    Uses word-boundary matching to avoid false positives:
      - Title is checked against SKIP_WORDS_TITLE and SKIP_BODIES_TITLE
      - Both title and description are checked against SKIP_WORDS_HARD
      - If require_keyword=True (for NASA archive), the title must
        contain at least one positive astronomical keyword.
    """
    if _SKIP_TITLE_RE.search(title):
        return False
    if _SKIP_BODIES_RE.search(title):
        return False
    text = title + " " + desc
    if _SKIP_HARD_RE.search(text):
        return False
    if require_keyword:
        if not _OBS_KEYWORD_RE.search(title):
            return False
    return True


# ── Source 1: NOIRLab RSS — fresh ground-based telescope images ──────────
def fetch_noirlab():
    """Fetch latest images from NOIRLab (Rubin, Gemini, Blanco, etc.)."""
    images = []
    try:
        rss = http_get(NOIRLAB_RSS, timeout=15).decode("utf-8", "ignore")
        items = re.findall(r"<item>(.*?)</item>", rss, re.DOTALL)
        for item in items[:20]:
            title_m = re.search(r"<title>(.*?)</title>", item)
            link_m = re.search(r"<link>(.*?)</link>", item)
            if not title_m or not link_m:
                continue
            title = title_m.group(1).strip()
            link = link_m.group(1).strip()
            title = re.sub(r"^(Science|Photo|Organization) Release:\s*",
                          "", title)
            if not _is_observation(title, ""):
                continue
            news_id = link.rstrip("/").split("/")[-1]
            # Try news/ first, then newsfeature/
            for cdn in (NOIRLAB_IMG_CDN_NEWS, NOIRLAB_IMG_CDN_FEATURE):
                img_url = f"{cdn}{news_id}a.jpg"
                if _was_recently_shown(img_url):
                    continue
                images.append({
                    "url": img_url,
                    "title": title,
                    "date": "recent",
                    "source": "NOIRLab",
                    "telescope": "Ground-based Survey",
                    "freshness": "LATEST",
                })
                log.info("NOIRLab: %s", title[:50])
                break
            if len(images) >= 8:
                break
    except Exception as e:
        log.warning("NOIRLab RSS failed: %s", e)
    return images


# ── Source 2: ESO Picture of the Week — fresh VLT/ALMA images ────────────
def fetch_eso():
    """Fetch latest images from ESO (VLT, ALMA, La Silla)."""
    images = []
    try:
        feed = http_get(ESO_POTW_FEED, timeout=15).decode("utf-8", "ignore")
        items = re.findall(r"<item>(.*?)</item>", feed, re.DOTALL)
        for item in items[:15]:
            title_m = re.search(r"<title>(.*?)</title>", item)
            link_m = re.search(r"<link>(.*?)</link>", item)
            if not title_m or not link_m:
                continue
            title = title_m.group(1).strip()
            link = link_m.group(1).strip()
            if not _is_observation(title, ""):
                continue
            try:
                page = http_get(link, timeout=10).decode("utf-8", "ignore")
                img_m = re.search(
                    r'src="(https://[^"]*esocdn[^"]*\.(?:jpg|png))"', page)
                if not img_m:
                    img_m = re.search(
                        r'href="(https://www\.eso\.org[^"]*\.(?:jpg|png))"',
                        page)
                if not img_m:
                    continue
                img_url = img_m.group(1)
                img_url = img_url.replace("original", "screen")
                if _was_recently_shown(img_url):
                    continue
                images.append({
                    "url": img_url,
                    "title": title,
                    "date": "recent",
                    "source": "ESO",
                    "telescope": "VLT / ALMA",
                    "freshness": "LATEST",
                })
                log.info("ESO: %s", title[:50])
                if len(images) >= 4:
                    break
            except Exception:
                continue
    except Exception as e:
        log.warning("ESO feed failed: %s", e)
    return images


# ── Source 3: ESA/Hubble RSS — latest Hubble releases ────────────────────
def fetch_esa_hubble():
    """Fetch the latest images from ESA/Hubble's RSS feed."""
    images = []
    try:
        rss = http_get(ESA_HUBBLE_RSS, timeout=15).decode("utf-8", "ignore")
        news_links = re.findall(
            r"<link>(https://esahubble\.org/news/[^/]+/)</link>", rss)
        titles = re.findall(r"<title>([^<]+)</title>", rss)
        if len(titles) > 1:
            titles = titles[1:]
        else:
            titles = []

        for i, link in enumerate(news_links[:20]):
            title = titles[i] if i < len(titles) else ""
            title = title.replace(" | ESA/Hubble", "").strip()
            title = re.sub(r"^(Science Release|Photo Release):\s*", "", title)
            if not _is_observation(title, ""):
                continue
            news_id = link.rstrip("/").split("/")[-1]
            # Try 'a' suffix first, then 'b', 'c'
            for suffix in ("a", "b", "c"):
                img_url = f"{ESA_CDN}{news_id}{suffix}.jpg"
                if _was_recently_shown(img_url):
                    continue
                images.append({
                    "url": img_url,
                    "title": title,
                    "date": "recent",
                    "source": "Hubble",
                    "telescope": "Hubble Space Telescope",
                    "freshness": "LATEST",
                })
                log.info("ESA/Hubble: %s", title[:50])
                break
            if len(images) >= 8:
                break
    except Exception as e:
        log.warning("ESA/Hubble RSS failed: %s", e)
    return images


# ── Source 4: NASA archive — infinite backbone of telescope observations ─
def fetch_nasa_archive():
    """Fetch a batch of telescope observations from NASA's image archive.

    This is the INFINITE backbone — the archive has thousands of
    Hubble/JWST/Spitzer/Chandra observations.  Random queries ensure
    we always get new material.
    """
    images = []
    term = random.choice(TELESCOPE_QUERIES)
    url = (f"{NASA_IMAGE_API}?q={quote(term)}&media_type=image"
           f"&page=1&page_size=50")
    try:
        data = http_get_json(url, timeout=15)
        items = data.get("collection", {}).get("items", [])
        if not items:
            return images
        random.shuffle(items)
        for item in items:
            d = item.get("data", [{}])[0]
            title = d.get("title", "")
            desc = d.get("description", "")
            if not _is_observation(title, desc, require_keyword=True):
                continue

            links = item.get("links", [])
            img_url = None
            for link in links:
                if link.get("render") == "image":
                    href = link.get("href", "")
                    if "~medium" in href:
                        img_url = href
                        break
            if not img_url:
                for link in links:
                    if link.get("render") == "image":
                        img_url = link.get("href")
                        break
            if not img_url:
                continue
            if _was_recently_shown(img_url):
                continue
            if "~orig" in img_url:
                img_url = img_url.replace("~orig", "~medium")

            telescope = "Telescope"
            source_name = "NASA"
            text = (title + " " + desc).lower()
            if "hubble" in text:
                telescope = "Hubble Space Telescope"
                source_name = "Hubble"
            elif "webb" in text or "jwst" in text:
                telescope = "James Webb Space Telescope"
                source_name = "JWST"
            elif "spitzer" in text:
                telescope = "Spitzer Space Telescope"
                source_name = "Spitzer"
            elif "chandra" in text:
                telescope = "Chandra X-ray Observatory"
                source_name = "Chandra"

            images.append({
                "url": img_url,
                "title": title,
                "date": d.get("date_created", "")[:10],
                "source": source_name,
                "telescope": telescope,
                "freshness": "ARCHIVE",
            })
            if len(images) >= 10:
                break
        log.info("NASA archive '%s': %d items", term, len(images))
    except Exception as e:
        log.warning("NASA archive failed for '%s': %s", term, e)
    return images


# ── Fetch one random fresh telescope image ───────────────────────────────
def fetch_random_telescope_image():
    """Pull a single fresh telescope observation.

    Strategy:
      1. Try fresh sources (NOIRLab, ESO, Hubble) — these have the
         newest releases.
      2. If fresh pool is empty (all recently shown), fall back to the
         NASA archive which has thousands of items — this is the
         infinite backbone that keeps the display running forever.
      3. Never give up — always return something or retry.
    """
    global _offline_until
    if time.time() < _offline_until:
        return None

    # Build a pool from fresh sources AND the NASA archive.
    # Fresh sources (NOIRLab, ESO, Hubble) have ~15 latest items.
    # NASA archive has thousands — it provides infinite variety so the
    # display never cycles through the same small set repeatedly.
    pool = []
    n_err = 0

    for name, fn in (("NOIRLab", fetch_noirlab),
                     ("ESO", fetch_eso),
                     ("Hubble", fetch_esa_hubble)):
        try:
            pool.extend(fn())
        except Exception as e:
            n_err += 1
            log.warning("%s fetch failed: %s", name, e)

    # Pull from MULTIPLE NASA archive queries to build a large, diverse
    # pool.  Each query returns ~10 items, so 4 queries give ~40 archive
    # items — far more variety than the ~15 fresh items alone.
    for _ in range(4):
        try:
            pool.extend(fetch_nasa_archive())
        except Exception as e:
            n_err += 1
            log.warning("NASA archive fetch failed: %s", e)

    # Filter out recently-shown URLs
    pool = [item for item in pool if not _was_recently_shown(item["url"])]

    fresh_count = sum(1 for i in pool if i.get("freshness") == "LATEST")
    archive_count = len(pool) - fresh_count
    log.info("Pool: %d fresh, %d archive", fresh_count, archive_count)

    if pool:
        item = random.choice(pool)
        _mark_shown(item["url"])
        return item

    if not pool and n_err:
        # Every source errored and nothing came back — almost certainly
        # a network outage.  Back off instead of hot-looping retries.
        _offline_until = time.time() + OFFLINE_BACKOFF_S
        log.warning("All sources failing (%d errors) — offline for %ds",
                    n_err, OFFLINE_BACKOFF_S)
        return None

    # All sources returned nothing or all recently shown — retry archive
    for attempt in range(3):
        try:
            nasa = fetch_nasa_archive()
            if nasa:
                item = random.choice(nasa)
                _mark_shown(item["url"])
                return item
        except Exception as e:
            log.warning("NASA archive retry failed: %s", e)
        time.sleep(1)

    # Last resort: clear history and try once more
    log.warning("All sources exhausted — clearing recent history")
    _recently_shown.clear()
    try:
        nasa = fetch_nasa_archive()
        if nasa:
            item = random.choice(nasa)
            _mark_shown(item["url"])
            return item
    except Exception as e:
        log.warning("NASA archive final retry failed: %s", e)

    return None


# ── Display ──────────────────────────────────────────────────────────────
def download_image(url, cache_path):
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    try:
        data = http_get(url, timeout=30)
        # Sanity check: reject HTML error pages (must look like an image)
        if len(data) < 500:
            return None
        if data[:4] not in (b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1",
                            b"\x89PNG\r\n", b"\xff\xd8\xff\xdb"):
            log.warning("Not an image (got %d bytes, starts %r)",
                       len(data), data[:8])
            return None
        cache_path.write_bytes(data)
        return cache_path
    except Exception as e:
        log.warning("Download failed: %s", e)
        return None


def fit_image(surf, target_w, target_h):
    sw, sh = surf.get_size()
    if sw <= 0 or sh <= 0:
        return pygame.Surface((target_w, target_h))
    scale = max(target_w / sw, target_h / sh)
    new_w = max(int(sw * scale), target_w)
    new_h = max(int(sh * scale), target_h)
    scaled = pygame.transform.smoothscale(surf, (new_w, new_h))
    x = max(0, (new_w - target_w) // 2)
    y = max(0, (new_h - target_h) // 2)
    x = min(x, max(0, new_w - target_w))
    y = min(y, max(0, new_h - target_h))
    return scaled.subsurface((x, y, target_w, target_h)).copy()


def draw_text(surf, text, pos, font, color=(255, 255, 255), shadow=True):
    if shadow:
        s = font.render(text, True, (0, 0, 0))
        surf.blit(s, (pos[0] + 1, pos[1] + 1))
    surf.blit(font.render(text, True, color), pos)


def wrap_text(text, font, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def blit_to_fb(surface, fb_path=FB):
    arr = pygame.surfarray.array3d(surface)
    arr = arr.transpose(1, 0, 2)
    r = arr[:, :, 0].astype(np.uint16)
    g = arr[:, :, 1].astype(np.uint16)
    b = arr[:, :, 2].astype(np.uint16)
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    buf = rgb565.astype("<u2").tobytes()
    with open(fb_path, "wb") as fb:
        fb.write(buf)


def prepare_image(item, fonts):
    """Download and render an image to a surface (no framebuffer blit)."""
    font_title, font_small = fonts
    ext = ".png" if ".png" in item["url"] else ".jpg"
    cache_name = f"{abs(hash(item['url'] + item.get('date', ''))) % 999999}{ext}"
    cache_path = CACHE_DIR / cache_name
    local = download_image(item["url"], cache_path)
    if local is None:
        return None
    try:
        img = pygame.image.load(str(local)).convert()
        fitted = fit_image(img, WIDTH, HEIGHT)
    except Exception as e:
        log.warning("Image processing failed: %s", e)
        cache_path.unlink(missing_ok=True)
        return None

    surf = pygame.Surface((WIDTH, HEIGHT))
    surf.blit(fitted, (0, 0))

    # Bottom info bar
    bar_h = 40
    bar = pygame.Surface((WIDTH, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 175))
    surf.blit(bar, (0, HEIGHT - bar_h))

    y = HEIGHT - bar_h + 3
    source = item.get("source", "")
    freshness = item.get("freshness", "")

    fresh_colors = {
        "TODAY": (100, 255, 100),
        "YESTERDAY": (200, 255, 100),
        "LATEST": (100, 255, 200),
        "ARCHIVE": (180, 200, 255),
        "FRESH": (180, 200, 255),
    }
    fc = fresh_colors.get(freshness, (150, 150, 150))

    if source and freshness:
        draw_text(surf, f"{source} {freshness}", (4, y), font_small, color=fc)
    y += 11

    title = item.get("title", "")
    if title:
        for line in wrap_text(title, font_title, WIDTH - 8)[:2]:
            draw_text(surf, line, (4, y), font_title)
            y += 14

    date_text = item.get("date", "")
    if date_text:
        draw_text(surf, date_text[:20], (WIDTH - 90, 4), font_small,
                 color=(200, 200, 200))

    return surf


def _set_proc_name(name: str) -> None:
    """Set kernel comm so earlyoom's --avoid regex protects this service.

    The systemd unit launches ``python`` whose comm is "python" — it
    does NOT match earlyoom's ``--avoid '^(...|telescope-tft|...)'``
    regex.  prctl(PR_SET_NAME) writes comm, the field earlyoom matches.
    No-op off Linux.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


def main():
    import threading

    _set_proc_name("telescope-tft")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))
    screen = pygame.Surface((WIDTH, HEIGHT))

    try:
        font_title = pygame.font.SysFont("dejavusans", 12, bold=True)
        font_small = pygame.font.SysFont("dejavusans", 9)
        font_gps = pygame.font.SysFont("dejavusans", 14, bold=True)
        font_tiny = pygame.font.SysFont("dejavusans", 8)
    except Exception:
        font_title = pygame.font.Font(None, 14)
        font_small = pygame.font.Font(None, 11)
        font_gps = pygame.font.Font(None, 17)
        font_tiny = pygame.font.Font(None, 10)
    fonts = (font_title, font_small)

    log.info("Telescope viewer started — prefetching for seamless transitions")

    next_surface = [None]
    next_item = [None]
    next_lock = threading.Lock()

    def prefetch():
        """Background thread: continuously fetch and prepare the next image.

        This thread NEVER exits.  If a fetch fails, it retries.  If all
        sources are exhausted, it clears history and retries.  The main
        loop can always get a new image.
        """
        while True:
            if next_surface[0] is not None:
                time.sleep(0.5)
                continue
            item = None
            for attempt in range(8):
                try:
                    item = fetch_random_telescope_image()
                except Exception as e:
                    log.warning("Fetch error (attempt %d): %s", attempt + 1, e)
                if item is not None:
                    break
                time.sleep(2)
            if item is None:
                wait = max(5.0, _offline_until - time.time())
                log.warning("No image after 8 tries, waiting %.0fs and retrying", wait)
                time.sleep(wait)
                continue
            try:
                surf = prepare_image(item, fonts)
            except Exception as e:
                log.warning("Prepare error: %s", e)
                surf = None
            if surf is None:
                # Mark this URL as shown so we don't retry it immediately,
                # then loop to fetch a different one
                _mark_shown(item["url"])
                time.sleep(1)
                continue
            with next_lock:
                next_surface[0] = surf
                next_item[0] = item

    t = threading.Thread(target=prefetch, daemon=True)
    t.start()

    screen.fill((0, 0, 0))
    msg = "Loading telescope images..."
    tw = font_small.size(msg)[0]
    draw_text(screen, msg, ((WIDTH - tw) // 2, HEIGHT // 2 - 6),
              font_small, color=(100, 180, 255))
    draw_gps_overlay(screen, font_gps)
    draw_skyplot(screen, font_tiny)
    blit_to_fb(screen)

    # Wait for first image (with timeout — show status if slow)
    wait_start = time.time()
    while next_surface[0] is None:
        time.sleep(0.5)
        if time.time() - wait_start > 30:
            screen.fill((0, 0, 0))
            msg2 = "Still loading... fetching from telescopes"
            tw2 = font_small.size(msg2)[0]
            draw_text(screen, msg2, ((WIDTH - tw2) // 2, HEIGHT // 2 - 6),
                      font_small, color=(255, 200, 100))
            draw_gps_overlay(screen, font_gps)
            draw_skyplot(screen, font_tiny)
            blit_to_fb(screen)
            wait_start = time.time()

    while True:
        with next_lock:
            surf = next_surface[0]
            item = next_item[0]
            next_surface[0] = None
            next_item[0] = None

        if surf is None or item is None:
            time.sleep(1)
            continue

        try:
            draw_gps_overlay(surf, font_gps)
            draw_skyplot(surf, font_tiny)
            blit_to_fb(surf)
            log.info("Showing: [%s/%s] %s",
                     item.get("source", ""),
                     item.get("freshness", ""),
                     item.get("title", "")[:60])
        except Exception as e:
            log.warning("Framebuffer write failed: %s", e)

        time.sleep(DISPLAY_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Exiting...")
    except Exception as e:
        log.exception("Fatal error: %s", e)
