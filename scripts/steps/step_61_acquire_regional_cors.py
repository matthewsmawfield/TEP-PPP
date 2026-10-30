#!/usr/bin/env python3
"""
STEP 61  --  Acquire a dense REGIONAL CORS network (independent of ppp_dense).

Steps 54-59 all run on the single ``ppp_dense`` product family (one processing
pipeline).  A convincing operational result needs an *independent* dense
network.  This step pulls a dense regional subset of the EUREF Permanent
Network (EPN) straight from the BKG GNSS Data Center -- fully open, no
registration -- and lays it out where ``tep_dynamic`` and a Step 54/55
replacement can consume it.

It also downloads the matching broadcast NAV and (via Step 01's helpers, if
importable) the precise SP3/CLK for the same day.

Regions (bounding boxes, lon/lat):
  germany_benelux   [ 2.0, 15.5, 47.0, 55.5 ]   ~50-100 km spacing
  alpine            [ 5.0, 17.0, 43.0, 49.0 ]
  iberia            [-10.0,  4.0, 36.0, 44.0 ]
  britain           [-11.0,  2.0, 49.0, 61.0 ]
  scandinavia       [ 4.0, 32.0, 55.0, 71.0 ]

Usage:
  python scripts/steps/step_61_acquire_regional_cors.py \
      --region germany_benelux --year 2024 --doys 100-106 [--max-stations 40]

Output:
  data/cors_<region>_<YYYYDDD>/<SITE>....crx.gz          (per day)
  data/cors_<region>_<YYYYDDD>/nav.rnx.gz
  results/outputs/step_61_acquire_regional_cors.json
"""
from __future__ import annotations

import sys
import json
import re
import argparse
import gzip
import io
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "results" / "outputs"

BKG_OBS = "https://igs.bkg.bund.de/root_ftp/EUREF/obs/{year}/{doy:03d}/"
BKG_NAV = "https://igs.bkg.bund.de/root_ftp/EUREF/nav/{year}/{doy:03d}/"
EPN_COORDS = "https://epncb.oma.be/_networkdata/stationlist.php?network=EPN"

REGIONS = {
    "germany_benelux": (2.0, 15.5, 47.0, 55.5),
    "alpine": (5.0, 17.0, 43.0, 49.0),
    "iberia": (-10.0, 4.0, 36.0, 44.0),
    "britain": (-11.0, 2.0, 49.0, 61.0),
    "scandinavia": (4.0, 32.0, 55.0, 71.0),
}

# ISO-3 country codes in RINEX-3 site IDs (chars 7-9), by region -- primary
# selector; the bounding box then refines using the file header.
REGION_CC = {
    "germany_benelux": {"DEU", "BEL", "NLD", "LUX", "FRA", "CHE", "AUT", "CZE", "POL", "DNK"},
    "alpine": {"CHE", "AUT", "ITA", "FRA", "DEU", "SVN"},
    "iberia": {"ESP", "PRT"},
    "britain": {"GBR", "IRL"},
    "scandinavia": {"SWE", "NOR", "FIN", "DNK", "EST"},
}


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (TEP-PPP data acquisition)"})
    s.verify = False
    try:
        import urllib3
        urllib3.disable_warnings()
    except Exception:
        pass
    return s


def _get(s, url, timeout=60):
    r = s.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def epn_coordinates(s):
    """{SITE4: (lon_deg, lat_deg)} from the EPN station list (best effort)."""
    try:
        html = _get(s, EPN_COORDS).text
    except Exception:
        return {}
    out = {}
    # rows carry a 9-char id and decimal lon/lat somewhere in the cells
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) < 4:
            continue
        site = None
        for c in cells[:2]:
            m = re.match(r"^([A-Z0-9]{4})\d{2}[A-Z]{3}$", c) or re.match(r"^([A-Z0-9]{4})$", c)
            if m:
                site = m.group(1)
                break
        if not site:
            continue
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", " ".join(cells))]
        # EPN station list gives LATITUDE then LONGITUDE (decimal degrees)
        lon = lat = None
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i + 1]
            if -90 <= a <= 90 and -180 <= b <= 180 and abs(a) > 1:
                lat, lon = a, b
                break
        if lon is not None:
            out[site] = (lon, lat)
    return out


def list_day_files(s, year, doy):
    html = _get(s, BKG_OBS.format(year=year, doy=doy)).text
    return sorted(set(re.findall(r'href="([A-Z0-9]{9}_[RS]_%d%03d0000_01D_30S_M[OM]\.crx\.gz)"' % (year, doy), html)))


def station_latlon_from_header(raw_bytes):
    """APPROX POSITION XYZ from a (hatanaka-)compressed RINEX header -> lon,lat."""
    try:
        txt = gzip.decompress(raw_bytes).decode("latin1", "replace")[:4000]
    except Exception:
        return None
    m = re.search(r"([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+APPROX POSITION XYZ", txt)
    if not m:
        return None
    import math
    x, y, z = (float(m.group(i)) for i in (1, 2, 3))
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.hypot(x, y)))
    return lon, lat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--doys", required=True, help="DOY or range, e.g. 100 or 100-106")
    ap.add_argument("--max-stations", type=int, default=40)
    ap.add_argument("--fetch-products", action="store_true",
                    help="also download SP3/CLK via step_01 helpers")
    args = ap.parse_args()

    lo, hi = (args.doys.split("-") + [args.doys])[:2]
    doys = list(range(int(lo), int(hi) + 1))
    W, E, S_, N = REGIONS[args.region]
    OUT.mkdir(parents=True, exist_ok=True)
    sess = _session()

    coords = epn_coordinates(sess)
    report = {"step": 61, "region": args.region, "bbox_WESN": [W, E, S_, N],
              "year": args.year, "doys": doys,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "days": {}}

    for doy in doys:
        tag = f"{args.year}{doy:03d}"
        dst = DATA / f"cors_{args.region}_{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        try:
            files = list_day_files(sess, args.year, doy)
        except Exception as e:
            report["days"][tag] = {"status": "listing_failed", "error": str(e)}
            print(f"[{tag}] listing failed: {e}")
            continue

        ccs = REGION_CC[args.region]
        cand = sorted({fn for fn in files if fn[6:9] in ccs})
        # candidates carry a coordinate hint from the EPN list where available
        cand = [(fn, fn[:4], coords.get(fn[:4])) for fn in cand]
        # spread: sort by known longitude (fallback: site id), spaced subset
        cand.sort(key=lambda q: (q[2][0] if q[2] else 999, q[1]))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(item):
            fn, site, latlon = item
            target = dst / fn
            if target.exists() and target.stat().st_size > 1000:
                return site, True
            try:
                raw = _get(_session(), BKG_OBS.format(year=args.year, doy=doy) + fn, timeout=90).content
            except Exception:
                return site, False
            ll = latlon or station_latlon_from_header(raw)
            if ll is None or not (W <= ll[0] <= E and S_ <= ll[1] <= N):
                return site, False
            target.write_bytes(raw)
            return site, True

        # take a generous over-selection; parallel fetch; keep the first max_stations that land
        pool_items = cand[: max(args.max_stations * 2, 40)]
        kept = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_fetch_one, it): it for it in pool_items}
            for f in as_completed(futs):
                site, ok = f.result()
                if ok and site not in kept:
                    kept.append(site)
        kept = kept[: args.max_stations]

        # broadcast nav for the day
        nav_ok = False
        try:
            nhtml = _get(sess, BKG_NAV.format(year=args.year, doy=doy)).text
            nm = re.search(r'href="(BRDC[^"]*%s%03d[^"]*MN\.rnx\.gz)"' % (args.year, doy), nhtml)
            if nm:
                (dst / "nav.rnx.gz").write_bytes(
                    _get(sess, BKG_NAV.format(year=args.year, doy=doy) + nm.group(1)).content)
                nav_ok = True
        except Exception:
            pass

        report["days"][tag] = {"status": "ok", "n_stations": len(kept),
                               "stations": sorted(set(kept)), "nav": nav_ok,
                               "dir": str(dst.relative_to(PROJECT_ROOT))}
        print(f"[{tag}] {len(kept)} CORS stations -> {dst.relative_to(PROJECT_ROOT)}  (nav: {nav_ok})")

        if args.fetch_products:
            try:
                from scripts.steps import step_01_acquire_data as s01  # noqa
                sp = _session()
                s01.download_sp3(sp, args.year, doy)      # names differ per repo version
                s01.download_clk(sp, args.year, doy)
                report["days"][tag]["products"] = "requested via step_01"
            except Exception as e:
                report["days"][tag]["products"] = f"step_01 helpers unavailable: {e}"

    (OUT / "step_61_acquire_regional_cors.json").write_text(json.dumps(report, indent=2))
    total = sum(d.get("n_stations", 0) for d in report["days"].values())
    print(f"\nsaved {OUT / 'step_61_acquire_regional_cors.json'}  ({total} station-days)")
    if not args.fetch_products:
        print("Next: fetch SP3/CLK/NAV for these dates:\n"
              f"  python scripts/steps/step_01_acquire_data.py --year {args.year} "
              f"--doys {args.doys} --download-nav --download-sp3 --download-clk")


if __name__ == "__main__":
    main()
