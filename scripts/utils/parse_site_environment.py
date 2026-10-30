#!/usr/bin/env python3
"""Download and parse IGS site logs to extract comprehensive metadata:
- Section 1: Monument Description, Height, Foundation, Foundation Depth, Geologic Characteristic, Bedrock Type
- Section 2: Elevation, City, Country
- Section 3: Temperature Stabilization
- Section 6: Frequency Standard (already parsed separately)
- Section 8: Collocation Information

This builds on parse_igs_site_logs.py but extracts much more metadata.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOCK_FILE = PROJECT_ROOT / "data" / "meta" / "clock_types_corrected.json"
ALT_FILE = PROJECT_ROOT / "data" / "meta" / "station_altitudes_rinex.json"
OUT_FILE = PROJECT_ROOT / "data" / "meta" / "station_environment.json"

SITE_LOG_URL = "http://garner.ucsd.edu/pub/docs/site_logs/{name}.log"
SITE_LOG_URL2 = "https://files.igs.org/pub/station/log/{name}.log"


def download_site_log(name: str) -> str | None:
    """Download a site log from UCSD or IGS."""
    for url_template in [SITE_LOG_URL, SITE_LOG_URL2]:
        url = url_template.format(name=name.lower())
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    return None


def parse_section(text, start_pattern, end_pattern):
    """Extract a section from the site log between start and end patterns."""
    lines = text.split("\n")
    in_section = False
    section_lines = []
    for line in lines:
        if re.match(start_pattern, line):
            in_section = True
            continue
        if in_section and re.match(end_pattern, line):
            break
        if in_section:
            section_lines.append(line)
    return "\n".join(section_lines)


def parse_field(section_text, field_name):
    """Extract a field value from a section."""
    m = re.search(rf"{field_name}\s*:\s*(.+)", section_text)
    if m:
        val = m.group(1).strip()
        if val.startswith("(") or val == "":
            return None
        return val
    return None


def parse_multiline_field(section_text, field_name):
    """Extract a field that may span multiple lines (continuation with leading spaces)."""
    lines = section_text.split("\n")
    result = []
    capturing = False
    for line in lines:
        m = re.match(rf"\s*{field_name}\s*:\s*(.*)", line)
        if m:
            val = m.group(1).strip()
            if val and not val.startswith("("):
                result.append(val)
            capturing = True
            continue
        if capturing:
            # Continuation lines start with spaces and a colon
            m2 = re.match(r"^\s+:\s*(.+)", line)
            if m2:
                val = m2.group(1).strip()
                if val and not val.startswith("("):
                    result.append(val)
            else:
                capturing = False
    return " ".join(result) if result else None


def parse_site_log(log_text: str) -> dict:
    """Parse a site log for all relevant metadata."""
    info = {}

    # Section 1: Site Identification of the GNSS Monument
    sec1 = parse_section(log_text, r"^1\.\s+Site Identification", r"^2\.\s+")
    info["monument_description"] = parse_field(sec1, "Monument Description")
    info["monument_height_m"] = parse_field(sec1, "Height of the Monument")
    info["monument_foundation"] = parse_field(sec1, "Monument Foundation")
    info["foundation_depth_m"] = parse_field(sec1, "Foundation Depth")
    info["geologic_characteristic"] = parse_field(sec1, "Geologic Characteristic")
    info["bedrock_type"] = parse_field(sec1, "Bedrock Type")
    info["bedrock_condition"] = parse_field(sec1, "Bedrock Condition")
    info["monument_additional_info"] = parse_multiline_field(sec1, "Additional Information")

    # Section 2: Site Location Information
    sec2 = parse_section(log_text, r"^2\.\s+Site Location", r"^3\.\s+")
    info["city"] = parse_field(sec2, "City or Town")
    info["country"] = parse_field(sec2, "Country or Region")
    info["tectonic_plate"] = parse_field(sec2, "Tectonic Plate")
    info["elevation_m"] = parse_field(sec2, "Elevation")
    info["latitude"] = parse_field(sec2, "Latitude")
    info["longitude"] = parse_field(sec2, "Longitude")

    # Section 3: GNSS Receiver Information (temperature stabilization)
    sec3 = parse_section(log_text, r"^3\.\s+GNSS Receiver", r"^4\.\s+")
    info["temp_stabilization"] = parse_field(sec3, "Temperature Stabiliz")
    info["receiver_additional_info"] = parse_multiline_field(sec3, "Additional Information")

    # Section 8: Collocation Information
    sec8 = parse_section(log_text, r"^8\.\s+", r"^9\.\s+")
    info["collocation_info"] = parse_multiline_field(sec8, "Additional Information")

    # Section 9: Local Conditions
    sec9 = parse_section(log_text, r"^9\.\s+", r"^10\.\s+")
    info["signal_obstructions"] = parse_multiline_field(sec9, "Signal Obstructions")
    info["radio_proximity"] = parse_multiline_field(sec9, "Radio")
    info["multipath_sources"] = parse_multiline_field(sec9, "Multipath Sources")

    # Section 13: More Information
    sec13 = parse_section(log_text, r"^13\.\s+", r"^$")
    info["url_more_info"] = parse_field(sec13, "URL for More Information")
    info["additional_info"] = parse_multiline_field(sec13, "Additional Information")

    # Clean up None values
    return {k: v for k, v in info.items() if v is not None}


def classify_environment(info: dict) -> dict:
    """Classify the station environment from parsed metadata."""
    env = {
        "on_bedrock": False,
        "on_building": False,
        "underground": False,
        "temperature_controlled": False,
        "temp_tolerance_c": None,
        "collocated_with_vlbi": False,
        "collocated_with_slr": False,
        "collocated_with_atomic_clocks": False,
        "environment_notes": [],
    }

    # Bedrock
    geo = (info.get("geologic_characteristic") or "").upper()
    bedrock = (info.get("bedrock_type") or "").upper()
    foundation = (info.get("monument_foundation") or "").upper()
    monument = (info.get("monument_description") or "").upper()

    if "BEDROCK" in geo:
        env["on_bedrock"] = True
    if "ROOF" in foundation or "BUILDING" in foundation or "WALL" in monument:
        env["on_building"] = True

    # Underground
    all_text = " ".join(str(v) for v in info.values()).upper()
    if "UNDERGROUND" in all_text or "UNDER GROUND" in all_text or "TUNNEL" in all_text or "MINE" in all_text or "CAVERN" in all_text:
        env["underground"] = True
        env["environment_notes"].append("Possible underground/tunnel location")

    # Temperature
    temp = info.get("temp_stabilization") or ""
    if temp and temp.lower() not in ("none", "no", "n/a"):
        env["temperature_controlled"] = True
        # Try to extract numeric tolerance
        m = re.search(r"(\d+\.?\d*)\s*(?:deg|°|C|Celsius)", temp, re.I)
        if m:
            env["temp_tolerance_c"] = float(m.group(1))
        else:
            env["temp_tolerance_c"] = temp  # keep as string if not parseable

    # Collocation
    coll = (info.get("collocation_info") or "").upper()
    if "VLBI" in coll:
        env["collocated_with_vlbi"] = True
    if "SLR" in coll or "LASER" in coll:
        env["collocated_with_slr"] = True
    if "ATOMIC" in coll or "MASER" in coll or "CESIUM" in coll or "FOUNTAIN" in coll:
        env["collocated_with_atomic_clocks"] = True

    # Check for metrology institutes
    city = (info.get("city") or "").upper()
    additional = " ".join(str(v) for v in info.values()).upper()
    if any(kw in additional for kw in ["PTB", "NIST", "USNO", "NMi", "METROLOGY", "OBSERVATORY",
                                        "TIME SCALE", "UTC(", "PRIMARY STANDARD",
                                        "FREQUENCY STANDARD LABORATORY"]):
        env["collocated_with_atomic_clocks"] = True
        env["environment_notes"].append("Co-located with national metrology institute")

    return env


def main() -> int:
    ct = json.load(open(CLOCK_FILE))
    altitudes = json.load(open(ALT_FILE)) if ALT_FILE.exists() else {}

    all_names = sorted(ct["stations"].keys())
    print(f"Downloading IGS site logs for {len(all_names)} stations...")

    results = {}
    n_processed = 0

    for name in all_names:
        log_text = download_site_log(name)
        if log_text is None or len(log_text) < 100:
            results[name] = {"site_log_available": False}
            continue

        n_processed += 1
        parsed = parse_site_log(log_text)
        env = classify_environment(parsed)

        # Add altitude from RINEX if not in site log
        if "elevation_m" not in parsed and name in altitudes:
            parsed["elevation_m"] = f"{altitudes[name]:.1f}"

        results[name] = {
            "site_log_available": True,
            "parsed": parsed,
            "environment": env,
            "clock_type": ct["stations"][name].get("corrected_type", "?"),
            "standard_type": ct["stations"][name].get("standard_type", "?"),
        }

        time.sleep(0.2)

    # Summary
    n_bedrock = sum(1 for r in results.values() if r.get("environment", {}).get("on_bedrock"))
    n_building = sum(1 for r in results.values() if r.get("environment", {}).get("on_building"))
    n_underground = sum(1 for r in results.values() if r.get("environment", {}).get("underground"))
    n_temp_ctrl = sum(1 for r in results.values() if r.get("environment", {}).get("temperature_controlled"))
    n_atomic_colloc = sum(1 for r in results.values() if r.get("environment", {}).get("collocated_with_atomic_clocks"))
    n_vlbi = sum(1 for r in results.values() if r.get("environment", {}).get("collocated_with_vlbi"))

    print(f"\nProcessed: {n_processed}/{len(all_names)}")
    print(f"On bedrock: {n_bedrock}")
    print(f"On building: {n_building}")
    print(f"Underground/tunnel: {n_underground}")
    print(f"Temperature controlled: {n_temp_ctrl}")
    print(f"Co-located with atomic clocks: {n_atomic_colloc}")
    print(f"Co-located with VLBI: {n_vlbi}")

    # Show underground/shielded stations
    print("\n=== Underground/tunnel stations ===")
    for name, r in sorted(results.items()):
        if r.get("environment", {}).get("underground"):
            print(f"  {name}: {r['environment']['environment_notes']}")

    # Show metrology stations
    print("\n=== Co-located with atomic clocks/metrology ===")
    for name, r in sorted(results.items()):
        if r.get("environment", {}).get("collocated_with_atomic_clocks"):
            notes = r["environment"].get("environment_notes", [])
            print(f"  {name}: clock={r['clock_type']}, standard={r['standard_type']}, notes={notes}")

    # Show bedrock vs building
    print("\n=== Bedrock stations (atomic clocks) ===")
    for name, r in sorted(results.items()):
        if r.get("environment", {}).get("on_bedrock") and r.get("clock_type") == "ATOMIC":
            geo = r.get("parsed", {}).get("geologic_characteristic", "?")
            bedrock = r.get("parsed", {}).get("bedrock_type", "?")
            print(f"  {name}: geo={geo}, bedrock={bedrock}, standard={r['standard_type']}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT_FILE, "w"), indent=2)
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
