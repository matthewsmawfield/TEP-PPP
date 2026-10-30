#!/usr/bin/env python3
"""Fetch + parse BIPM TWSTFT lab-clock link files.

Two-way satellite time transfer (geostationary transponder) between
national timing labs — a clock-comparison channel fully orthogonal
to GNSS: one-way GNSS systematics (orbit/clock product errors,
ionosphere) cannot enter a two-way measurement.  Per-link series
are ~2-hourly, ~0.1-0.3 ns noise, over years — covering the
10-35 uHz TEP band at atomic-lab precision.

Files: https://webtai.bipm.org/ftp/pub/tai/data/<YEAR>/time_transfer/
twstft/<lab>/tw<lab>60.<mjd-60000>
Labs seen: nist usno ptb op sp it ch vsl roa npl ipq aos (also
twstft_sdr for the software-defined radio variant).

Output CSV: mjd, sttime_s, loc, rem, tw_s, drms_ns, atl,
refdelay_s, rsig_ns, ci, s, calr, esdvar_ns, esig_ns, tmp_c,
hum_pct, pres_mbar

The TW column is the lab-pair clock difference in seconds —
the direct inter-lab clock-correlation channel for the
distance-structure screen (lab coordinates live in each file's
ES/REF-FRAME header lines).

Copyright note (BIPM): data remain the property of the contributing
laboratories; standard scientific-analysis use is the norm in the
timing literature, cite BIPM + labs in any publication.

Usage:
  python3 scripts/live/fetch_twstft.py --labs nist ptb op sp it ch \\
      --year 2025 --mjd-lo 60600 --mjd-hi 60767 --out data/twstft/
"""
import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

BASE = ("https://webtai.bipm.org/ftp/pub/tai/data/{year}"
        "/time_transfer/twstft/{lab}/tw{lab}{cent}.{suf:03d}")
LABS = ["nist", "usno", "ptb", "op", "sp", "it", "ch", "vsl",
        "roa", "npl", "ipq", "aos"]


def parse_sats(text):
    """Extract `* LINK <n> SAT: <name>` header lines ->
    {link_number: satellite_name}."""
    import re
    out = {}
    for ln in text.splitlines():
        m = re.search(r"LINK\s+(\d+)\s+SAT:\s*(.+?)\s+NLO:", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_coords(text):
    """Extract `* ES <ID> LA: <N|S> d m s  LO: <E|W> d m s  HT: <h> m`
    header lines -> {es_id: (lat_deg, lon_deg, h_m)}."""
    import re
    out = {}
    rx = re.compile(
        r"ES\s+(\S+)\s+LA:\s*([NS])\s*(\d+)\s+(\d+)\s+([\d.]+)\s+"
        r"LO:\s*([EW])\s*(\d+)\s+(\d+)\s+([\d.]+)\s+HT:\s*\+?\s*"
        r"([-\d.]+)\s*m?")
    for ln in text.splitlines():
        m = rx.search(ln)
        if m:
            es, ns, d1, m1, s1, ew, d2, m2, s2, h = m.groups()
            lat = float(d1) + float(m1) / 60 + float(s1) / 3600
            lon = float(d2) + float(m2) / 60 + float(s2) / 3600
            if ns == "S":
                lat = -lat
            if ew == "W":
                lon = -lon
            out[es] = (lat, lon, float(h))
    return out


def parse_file(text, sats=None):
    """Parse one tw<lab>60.* file -> list of record dicts.
    sats: optional {link_number: satellite_name} -> `sat` column."""
    sats = sats or {}
    recs = []
    for ln in text.splitlines():
        if ln.startswith("*") or ln.startswith("#"):
            continue
        parts = ln.split()
        # data lines: LOC REM LINK MJD STTIME NTL TW DRMS SMP ATL
        #             REFDELAY RSIG CI S CALR ESDVAR ESIG TMP HUM PRES
        # verified against twnist60.767 (real file, 2026-10-26)
        if len(parts) < 19:
            continue
        try:
            mjd = int(parts[3])
            if not (40000 <= mjd <= 70000):
                continue
            float(parts[6])          # TW
        except (ValueError, IndexError):
            continue
        try:
            recs.append({
                "loc": parts[0], "rem": parts[1],
                "link": parts[2],
                "sat": sats.get(parts[2], ""),
                "mjd": mjd, "sttime_s": int(parts[4]),
                "ntl": int(parts[5]),
                "tw_s": float(parts[6]),
                "drms_ns": float(parts[7]),
                "smp": int(parts[8]),
                "atl": int(parts[9]),
                "refdelay_s": float(parts[10]),
                "rsig": parts[11],
                # verified vs header CAL ids: parts[12] is the
                # calibration reference (553/570/572/573/593...).
                # aux1-3 are per-link delay-variation/error fields
                # whose exact header names differ by revision.
                "cal_id": parts[12],
                "s_flag": parts[13],
                "aux1": parts[14],
                "aux2": parts[15],
                "aux3": parts[16],
                "tmp_c": float(parts[17]),
                "hum_pct": float(parts[18]),
                "pres_mbar": float(parts[19]) if len(parts) > 19
                else "",
            })
        except (ValueError, IndexError):
            continue
    return recs


def fetch(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if k == tries - 1:
                return None
            time.sleep(2.0 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", nargs="+", default=LABS)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--mjd-lo", type=int, required=True)
    ap.add_argument("--mjd-hi", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="output dir; one CSV per lab")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    fields = ["loc", "rem", "link", "sat", "mjd", "sttime_s", "ntl",
              "tw_s", "drms_ns", "smp", "atl", "refdelay_s",
              "rsig", "cal_id", "s_flag", "aux1", "aux2", "aux3",
              "tmp_c", "hum_pct", "pres_mbar"]
    coords = {}
    for lab in args.labs:
        out = outdir / f"twstft_{lab}_{args.year}_" \
            f"{args.mjd_lo}_{args.mjd_hi}.csv"
        n_recs = n_missing = 0
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for mjd in range(args.mjd_lo, args.mjd_hi + 1):
                url = BASE.format(year=args.year, lab=lab,
                                  cent=mjd // 1000,
                                  suf=mjd % 1000)
                text = fetch(url)
                if text is None:
                    n_missing += 1
                    continue
                coords.update(parse_coords(text))
                recs = parse_file(text, parse_sats(text))
                for r in recs:
                    w.writerow(r)
                n_recs += len(recs)
                if mjd % 10 == 0:
                    print(f"  {lab} mjd {mjd}: {n_recs} recs so far")
        print(f"{lab}: {n_recs} records, {n_missing} days missing "
              f"-> {out}")

    if coords:
        import json
        cpath = outdir / "twstft_labs.json"
        merged = {}
        if cpath.exists():
            try:
                merged = json.loads(cpath.read_text())
            except Exception:
                pass
        merged.update(coords)
        cpath.write_text(json.dumps(merged, indent=1))
        print(f"{len(merged)} station coords -> {cpath}")


if __name__ == "__main__":
    sys.exit(main())
