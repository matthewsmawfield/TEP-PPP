#!/usr/bin/env python3
"""
STEP 63  --  Acquire urban kinematic datasets + their reference network.

Pulls the UrbanNav (Hong Kong) benchmark collections -- a moving receiver in a
real urban canyon with a NovAtel SPAN reference trajectory -- converts the
ground truth to the pipeline CSV format, downloads a regional CORS network and
the precise SP3/CLK/NAV for the date, and writes the manifest entry that
the kinematic analysis steps consume.

Collections (UrbanNav, https://github.com/IPNL-POLYU/UrbanNavDataset):
  mongkok   Harsh Urban  -- deepest canyon, most k<=3 epochs   (2021-05-18, DOY 138)
  tst       Medium Urban -- typical canyon                     (2021-05-17, DOY 137)
  whampoa   Deep Urban   -- deep canyon                        (2021-xx)

Downloads use ``curl -C -`` (resume) because the Dropbox folder zips are large
and the plain HTTP client truncates them.

Usage:
  python scripts/steps/step_63_acquire_kinematic.py --collection mongkok
  python scripts/steps/step_63_acquire_kinematic.py --collection all

Output:
  data/dynamic_datasets/urbannav/<collection>/{*.obs, ground_truth.csv, cors/}
  data/rinex/{sp3,clk,nav}/...   (products for the date)
  manifest entry appended to data/dynamic_datasets/manifest.json
  results/outputs/step_63_acquire_kinematic.json
"""
from __future__ import annotations

import sys
import os
import re
import json
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "results" / "outputs"

# collection -> (gnss folder zip url, ground-truth txt url, date_tag YYYYDOY, gps_week)
COLLECTIONS = {
    "mongkok": (
        "https://www.dropbox.com/sh/xej1cxv3a0erffc/AADfhRODJk__Z49WC1R0Nw5ea?dl=1",
        "https://www.dropbox.com/s/zqmxgf59bp7en6c/UrbanNav_mongkok_GT_part_raw.txt?dl=1",
        "2021138", 2158,
        "UrbanNav-HK-Harsh-Urban-1.ublox.f9p.obs"),
    "tst": (
        "https://www.dropbox.com/sh/2haoy68xekg95zl/AAAkcN4FwhFxkPY1lXsxbJrxa?dl=1",
        "https://www.dropbox.com/s/twsvwftucoytfpc/UrbanNav_TST_GT_raw.txt?dl=1",
        "2021137", 2158,
        None),
    "whampoa": (
        "https://www.dropbox.com/sh/7ox7718bzcjqtlf/AABH_Kjm65gHQ09K3antBRdua?dl=1",
        "https://www.dropbox.com/s/ej2mkue2w3r36s2/UrbanNav_whampoa_raw.txt?dl=1",
        "2021135", 2157,
        None),
}

CDDIS = "https://cddis.nasa.gov/archive/gnss"
REGIONAL_CC = ("HKG", "CHN", "TWN", "JPN", "KOR", "PHL", "THA", "AUS", "IND", "SGP")


def _curl(url, out_path, min_bytes=1000):
    for _ in range(6):
        subprocess.run(["curl", "-sL", "-C", "-", "--retry", "5", "--retry-delay", "3",
                        "-A", "Mozilla/5.0", "-o", str(out_path), url], check=False)
        if out_path.exists() and out_path.stat().st_size >= min_bytes:
            return True
    return out_path.exists() and out_path.stat().st_size >= min_bytes


def convert_gt(raw_txt: Path, csv_out: Path):
    """UrbanNav GT (UTCtime week gpssec  latD latM latS  lonD lonM lonS  H-Ell ...)
       -> sod,lat,lon,hgt"""
    rows = []
    for ln in raw_txt.read_text(errors="replace").splitlines():
        p = ln.split()
        if len(p) < 10 or not re.match(r"^\d+\.\d+$", p[0]):
            continue
        try:
            gpssec = float(p[2])
            latd, latm, lats = float(p[3]), float(p[4]), float(p[5])
            lond, lonm, lons = float(p[6]), float(p[7]), float(p[8])
            h = float(p[9])
        except (ValueError, IndexError):
            continue
        lat = (abs(latd) + latm / 60 + lats / 3600) * (-1 if latd < 0 else 1)
        lon = (abs(lond) + lonm / 60 + lons / 3600) * (-1 if lond < 0 else 1)
        rows.append((gpssec % 86400.0, lat, lon, h))
    with open(csv_out, "w") as f:
        f.write("sod,lat,lon,hgt\n")
        for sod, lat, lon, h in rows:
            f.write(f"{sod:.1f},{lat:.9f},{lon:.9f},{h:.3f}\n")
    return len(rows)


def fetch_cors(date_tag: str, dst: Path, max_stn=14):
    import requests
    y, d = int(date_tag[:4]), int(date_tag[4:])
    s = requests.Session()
    base = f"{CDDIS}/data/daily/{y}/{d:03d}/{str(y)[2:]}d/"
    r = s.get(base, timeout=90)
    r.raise_for_status()
    files = re.findall(r'href="([A-Z0-9]{9}_R_%d%03d0000_01D_30S_MO\.crx\.gz)\s*"' % (y, d), r.text)
    hk = [f for f in files if f[6:9] == "HKG"]
    reg = [f for f in files if f[6:9] in REGIONAL_CC and f not in hk]
    got = []
    for f in (hk + reg)[:max_stn]:
        t = dst / f.strip()
        if t.exists() and t.stat().st_size > 5000:
            got.append(f[:4])
            continue
        try:
            c = s.get(base + f.strip(), timeout=120).content
            if len(c) > 5000:
                t.write_bytes(c)
                got.append(f[:4])
        except Exception:
            pass
    return sorted(got)


def fetch_products(date_tag: str, gps_week: int):
    import requests
    y, d = int(date_tag[:4]), int(date_tag[4:])
    s = requests.Session()
    out = {}
    prod = f"{CDDIS}/products/{gps_week}/"
    for sub, pat in (("sp3", r"COD0MGXFIN_%d%03d0000_01D_05M_ORB\.SP3\.gz" % (y, d)),
                     ("clk", r"COD0MGXFIN_%d%03d0000_01D_30S_CLK\.CLK\.gz" % (y, d))):
        dst = DATA / "rinex" / sub
        dst.mkdir(parents=True, exist_ok=True)
        try:
            m = re.search(pat, s.get(prod, timeout=60).text)
            if m and not (dst / m.group(0)).exists():
                (dst / m.group(0)).write_bytes(s.get(prod + m.group(0), timeout=180).content)
            out[sub] = bool(m)
        except Exception as e:
            out[sub] = f"fail: {e}"
    navdir = f"{CDDIS}/data/daily/{y}/{d:03d}/{str(y)[2:]}p/"
    dst = DATA / "rinex" / "nav"
    dst.mkdir(parents=True, exist_ok=True)
    try:
        m = re.search(r"BRDC00IGS_R_%d%03d0000_01D_MN\.rnx\.gz" % (y, d), s.get(navdir, timeout=60).text)
        if m and not (dst / m.group(0)).exists():
            (dst / m.group(0)).write_bytes(s.get(navdir + m.group(0), timeout=180).content)
        out["nav"] = bool(m)
    except Exception as e:
        out["nav"] = f"fail: {e}"
    return out


def do_collection(name: str) -> dict:
    zip_url, gt_url, date_tag, gps_week, rover_hint = COLLECTIONS[name]
    root = DATA / "dynamic_datasets" / "urbannav" / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "cors").mkdir(exist_ok=True)
    rec = {"collection": name, "date_tag": date_tag}

    zpath = root / "gnss.zip"
    if not any(root.glob("*.obs")):
        if not _curl(zip_url, zpath, min_bytes=1_000_000):
            return {**rec, "status": "gnss_download_failed"}
        try:
            with zipfile.ZipFile(zpath) as z:
                for n in z.namelist():
                    if n.lower().endswith((".obs", ".rnx", ".o")):
                        z.extract(n, root)
                        # flatten
                        src = root / n
                        if src.parent != root:
                            src.rename(root / Path(n).name)
        except zipfile.BadZipFile:
            return {**rec, "status": "bad_zip"}
    obs = [p for p in root.glob("*.obs") if "f9p" in p.name.lower()] or list(root.glob("*.obs"))
    rec["rover_obs"] = obs[0].name if obs else None

    gtx = root / "gt_raw.txt"
    if not (root / "ground_truth.csv").exists():
        if not _curl(gt_url, gtx, min_bytes=1000):
            return {**rec, "status": "gt_download_failed"}
        rec["gt_epochs"] = convert_gt(gtx, root / "ground_truth.csv")

    rec["cors"] = fetch_cors(date_tag, root / "cors")
    rec["products"] = fetch_products(date_tag, gps_week)

    # manifest entry
    if obs:
        mf = DATA / "dynamic_datasets" / "manifest.json"
        data = json.load(open(mf)) if mf.exists() else []
        entry = {
            "name": f"urbannav_{name}_hk", "type": "urban",
            "raw_obs": str((root / obs[0].name).relative_to(PROJECT_ROOT)),
            "raw_fmt": "rinex",
            "ground_truth": str((root / "ground_truth.csv").relative_to(PROJECT_ROOT)),
            "gt_fmt": "llh", "gt_time_col": "sod", "gt_lat_col": "lat",
            "gt_lon_col": "lon", "gt_hgt_col": "hgt",
            "cors_dir": str((root / "cors").relative_to(PROJECT_ROOT)),
            "date_tag": date_tag, "elev_mask_deg": 10.0, "obs_sigma_m": 2.0,
            "region": "hongkong", "cors_use_precise": True,
        }
        data = [d for d in data if d.get("name") != entry["name"]] + [entry]
        json.dump(data, open(mf, "w"), indent=2)
        rec["manifest_entry"] = entry["name"]
    rec["status"] = "ok" if (obs and rec.get("cors")) else "incomplete"
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="mongkok",
                    choices=list(COLLECTIONS) + ["all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(COLLECTIONS) if args.collection == "all" else [args.collection]
    report = {"step": 63, "timestamp": datetime.now(timezone.utc).isoformat(),
              "collections": {}}
    for n in names:
        print(f"--- {n} ---")
        r = do_collection(n)
        report["collections"][n] = r
        print(json.dumps(r, indent=2))
    (OUT / "step_63_acquire_kinematic.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {OUT / 'step_63_acquire_kinematic.json'}")
    print("Next: python scripts/steps/step_52_drone_kinematic.py")


if __name__ == "__main__":
    main()
