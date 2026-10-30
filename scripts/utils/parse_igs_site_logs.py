#!/usr/bin/env python3
"""Download and parse IGS site logs to build a correct clock-type table.

The IGS site log format documents the frequency standard history for each
station, including effective dates.  This script:

1. Downloads site logs from garner.ucsd.edu for all stations in our metadata.
2. Parses the frequency standard section (section 6) to find the clock type
   active on a given date.
3. Outputs a dated reference-standard table.

Important taxonomy rule: the IGS ``Standard Type: INTERNAL`` value says which
frequency input the receiver used; it does *not* identify the physical
oscillator technology.  It must therefore never be silently relabelled
``QUARTZ``.  Only an explicit QUARTZ/OCXO/TCXO entry is technology-confirmed
quartz.  This distinction matters directly to the screening hypothesis.

The site log format for section 6:

6.1  Standard Type            : EXTERNAL H-MASER
       Input Frequency        : 10.0 MHz
       Effective Dates        : 2021-01-29/2021-10-06
       Notes                  : ...

6.2  Standard Type            : INTERNAL
       Input Frequency        :
       Effective Dates        : 2021-10-06/CCYY-MM-DD
       Notes                  :

The "Effective Dates" field is START/END where CCYY-MM-DD means "present".
"""
import json
import re
import sys
import time
from pathlib import Path
from datetime import date
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_FILE = PROJECT_ROOT / "data" / "meta" / "optimal_100_metadata.json"
OUT_FILE = PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"

SITE_LOG_URL = "http://garner.ucsd.edu/pub/docs/site_logs/{name}.log"


def download_site_log(name: str) -> str | None:
    """Download a site log from UCSD."""
    url = SITE_LOG_URL.format(name=name.lower())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def parse_frequency_standards(log_text: str) -> list[dict]:
    """Parse section 6 (Frequency Standard) from a site log.

    Returns a list of entries with 'type', 'start', 'end', 'notes'.
    """
    entries = []
    lines = log_text.split("\n")
    in_section = False
    current = {}

    for i, line in enumerate(lines):
        if re.match(r"^6\.\s+Frequency Standard", line):
            in_section = True
            continue
        if in_section and re.match(r"^7\.\s+", line):
            break
        if not in_section:
            continue

        # Match "6.x Standard Type : VALUE"
        m = re.match(r"^6\.\d+\s+Standard Type\s*:\s*(.+)", line)
        if m:
            if current:
                entries.append(current)
            current = {"type": m.group(1).strip()}
            continue

        # Match "Effective Dates : START/END"
        m = re.match(r"^\s+Effective Dates\s*:\s*(.+)", line)
        if m and current:
            dates = m.group(1).strip()
            # Skip template placeholders
            if dates.startswith("("):
                continue
            parts = dates.split("/")
            current["start"] = parts[0].strip() if len(parts) > 0 else ""
            current["end"] = parts[1].strip() if len(parts) > 1 else ""
            continue

        # Match "Notes : ..."
        m = re.match(r"^\s+Notes\s*:\s*(.*)", line)
        if m and current:
            note = m.group(1).strip()
            if note and note != "(multiple lines)":
                current["notes"] = current.get("notes", "") + note + " "
            continue

    if current:
        entries.append(current)

    # Filter out template entries
    entries = [e for e in entries
               if e.get("type") and
               not e["type"].startswith("(INTERNAL") and
               e.get("start") and
               not e["start"].startswith("(CCYY")]

    return entries


def classify_clock_type(std_type: str) -> str:
    """Classify the documented reference standard without inventing material.

    ``INTERNAL_UNSPECIFIED`` is deliberate: an internal receiver reference is
    commonly quartz-based, but the site log does not establish that fact, nor
    whether receiver clock steering was enabled.
    """
    t = std_type.upper()
    if "QUARTZ" in t or "OCXO" in t or "TCXO" in t:
        return "QUARTZ"
    if "INTERNAL" in t:
        return "INTERNAL_UNSPECIFIED"
    if "GPS" in t or "GNSS" in t or "DORIS" in t:
        if "MASER" in t or "CESIUM" in t or "RUBIDIUM" in t or "RB" in t:
            return "GNSS_STEERED_ATOMIC"
        return "EXTERNAL_NAV_STEERED"
    if "MASER" in t or "CESIUM" in t or "RUBIDIUM" in t or "RB" in t:
        return "ATOMIC"
    if "UTC" in t or "MASTER CLOCK" in t:
        return "EXTERNAL_TIMESCALE_UNSPECIFIED"
    return "UNKNOWN"


def get_clock_type_on_date(entries: list[dict], target: str) -> dict | None:
    """Find the clock type active on a given date (YYYY-MM-DD)."""
    target_date = date.fromisoformat(target)
    matches = []
    for e in entries:
        start = e.get("start", "")
        end = e.get("end", "")
        if not start:
            continue
        try:
            start_d = date.fromisoformat(start)
        except ValueError:
            continue
        if start_d > target_date:
            continue
        if end and not end.startswith("CCYY"):
            try:
                end_d = date.fromisoformat(end)
                # Site-log intervals are switch epochs: the old source ceases
                # to be active on the end date.
                if target_date >= end_d:
                    continue
            except ValueError:
                pass
        matches.append((start_d, e))
    if matches:
        # Logs occasionally contain overlapping historical entries (THTI is a
        # real example).  The most recently started matching record is the
        # operative one, not the first record in file order.
        _, e = max(matches, key=lambda item: item[0])
        return {
            "standard_type": e["type"],
            "classification": classify_clock_type(e["type"]),
            "start": e.get("start", ""),
            "end": e.get("end", ""),
            "notes": e.get("notes", "").strip(),
        }
    return None


def main() -> int:
    meta = json.load(open(META_FILE))
    stations = meta.get("stations", {})
    all_names = sorted(stations.keys())

    # Target date: 2024 day 100 = April 9, 2024
    target_date = "2024-04-09"

    print(f"Downloading IGS site logs for {len(all_names)} stations...")
    print(f"Target date: {target_date}")

    results = {}
    n_downloaded = 0
    n_failed = 0

    for name in all_names:
        log_text = download_site_log(name)
        if log_text is None or len(log_text) < 100:
            n_failed += 1
            # Fall back to metadata classification
            old = stations[name].get("classification", {}).get("clock_type", "UNKNOWN")
            results[name] = {
                "site_log_available": False,
                "old_metadata_type": old,
                "corrected_type": "UNKNOWN",
                "standard_type": None,
                "source": "metadata_fallback",
            }
            continue

        n_downloaded += 1
        entries = parse_frequency_standards(log_text)
        active = get_clock_type_on_date(entries, target_date)

        old = stations[name].get("classification", {}).get("clock_type", "UNKNOWN")
        old_class = "QUARTZ" if old in ("INTERNAL", "OCXO") else \
                     "ATOMIC" if old in ("H-MASER", "RUBIDIUM", "CESIUM") else "UNKNOWN"

        if active:
            results[name] = {
                "site_log_available": True,
                "old_metadata_type": old,
                "old_classification": old_class,
                "corrected_type": active["classification"],
                "standard_type": active["standard_type"],
                "effective_start": active["start"],
                "effective_end": active["end"],
                "notes": active["notes"],
                "source": "igs_site_log",
                "all_history": [{"type": e["type"],
                                 "start": e.get("start", ""),
                                 "end": e.get("end", "")}
                                for e in entries],
            }
        else:
            results[name] = {
                "site_log_available": True,
                "old_metadata_type": old,
                "old_classification": old_class,
                "corrected_type": "UNKNOWN",
                "standard_type": None,
                "source": "site_log_no_match",
            }

        # Print mismatches
        if active and old_class != active["classification"]:
            print(f"  MISMATCH {name}: metadata={old}({old_class}) -> "
                  f"site_log={active['standard_type']}({active['classification']})")

        time.sleep(0.3)  # be polite

    # Summary
    n_quartz = sum(1 for r in results.values() if r["corrected_type"] == "QUARTZ")
    n_internal_unspecified = sum(
        1 for r in results.values() if r["corrected_type"] == "INTERNAL_UNSPECIFIED")
    n_atomic = sum(1 for r in results.values() if r["corrected_type"] == "ATOMIC")
    n_unknown = sum(1 for r in results.values() if r["corrected_type"] == "UNKNOWN")
    n_mismatch = sum(1 for r in results.values()
                     if r.get("old_classification") and r.get("corrected_type")
                     and r["old_classification"] != r["corrected_type"]
                     and r["corrected_type"] != "UNKNOWN")

    print(f"\nSummary:")
    print(f"  Downloaded: {n_downloaded}, Failed: {n_failed}")
    print(f"  QUARTZ (explicit): {n_quartz}, INTERNAL_UNSPECIFIED: "
          f"{n_internal_unspecified}, ATOMIC: {n_atomic}, UNKNOWN: {n_unknown}")
    print(f"  Mismatches: {n_mismatch}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_date": target_date,
        "source": "IGS site logs from garner.ucsd.edu",
        "n_stations": len(results),
        "n_quartz": n_quartz,
        "n_internal_unspecified": n_internal_unspecified,
        "n_atomic": n_atomic,
        "n_unknown": n_unknown,
        "n_mismatches": n_mismatch,
        "stations": results,
    }, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
