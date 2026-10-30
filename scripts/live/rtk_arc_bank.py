#!/usr/bin/env python3
"""RTK-AR arc banker: accumulate good rover-vs-OS-Net arcs toward a
tighter antenna anchor.

Chunks only persist ~4 h, so this runs opportunistically (hourly via
launchd or by hand).  Each invocation:
  1. picks the newest ~2 h span of contiguous chunks,
  2. runs scripts/rtk_easn_replay.py per --bases entry,
  3. quality-gates the arc (Q1 count, fix-vs-fix cross-base agreement,
     internal scatter), and
  4. banks per-base Q1 medians into data/ppp_live/rtk_arc_bank.json.

The bank is DATA, not a promotion: re-anchoring truth_ref.json stays a
human decision (the Oct-14 null-conv_rmse incident showed why).

Usage:
  . .env.local && .venv/bin/python3 scripts/rtk_arc_bank.py \
      [--span-s 7200] [--bases EASN,SWAN] [--min-q1 8]
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "data" / "ppp_live" / "rtk_arc_bank.json"


def _llh_rows(pos_path):
    rows = []
    for line in open(pos_path):
        if line.startswith("%") or not line.strip():
            continue
        p = line.split()
        if len(p) < 7 or not p[0][0].isdigit():
            continue
        try:
            rows.append((float(p[2]), float(p[3]), float(p[4]),
                         int(p[5])))
        except ValueError:
            continue
    return rows


def _enu(rows, anchor_ecef):
    import gps_web as gw  # noqa: E402 — sys.path set by caller
    lat = math.radians(gw.ecef_to_latlon(*anchor_ecef)[0])
    lon = math.radians(gw.ecef_to_latlon(*anchor_ecef)[1])
    E = np.array([-math.sin(lon), math.cos(lon), 0.0])
    N = np.array([-math.sin(lat) * math.cos(lon),
                  -math.sin(lat) * math.sin(lon), math.cos(lat)])
    U = np.array([math.cos(lat) * math.cos(lon),
                  math.cos(lat) * math.sin(lon), math.sin(lat)])
    out = []
    for lt, ln, h, q in rows:
        d = np.array(gw.latlon_to_ecef(lt, ln, h)) - anchor_ecef
        out.append((q, d @ E, d @ N, d @ U))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-s", type=float, default=7200)
    ap.add_argument("--bases", default="EASN,SWAN")
    ap.add_argument("--min-q1", type=int, default=8)
    ap.add_argument("--max-scatter", type=float, default=0.35,
                    help="max Q1 3D std within the arc (m)")
    ap.add_argument("--max-off-anchor", type=float, default=6.0,
                    help="reject arcs whose Q1 median sits this far from "
                         "the current anchor (wrong-integer guard)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "rpi"))
    import gps_web  # noqa: E402

    truth = json.loads((ROOT / "data/ppp_live/truth_ref.json").read_text())
    # RTK fixes inherit the OS Net base frame (ETRS89) — score against the
    # raw survey anchor, not the ITRF working truth.
    anchor = np.array(
        truth.get("truth_ecef_etrs89") or truth["truth_ecef"], float)

    # newest contiguous span of chunks
    chunks = sorted((ROOT / "data/ppp_live").glob("chunk_*.bin"),
                    key=lambda p: float(p.stem.split("_")[1]))
    if len(chunks) < 2:
        sys.exit("no chunks")
    t1 = float(chunks[-1].stem.split("_")[1]) + 300.0
    t0 = t1 - args.span_s
    per_base = {}
    for base in args.bases.split(","):
        out = ROOT / "logs" / "replay" / f"rtk_bank_{base}"
        cmd = [str(ROOT / ".venv/bin/python3"),
               str(ROOT / "scripts/live/rtk_easn_replay.py"),
               "--since", str(t0), "--until", str(t1),
               "--station", base, "--out", str(out)]
        if args.dry:
            print("dry:", " ".join(cmd)); continue
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        pos = out / "rtk.pos"
        if not pos.exists():
            print(f"{base}: no pos (rc={r.returncode})")
            continue
        enu = _enu(_llh_rows(pos), anchor)
        q1 = np.array([r[1:] for r in enu if r[0] == 1])
        if len(q1) < args.min_q1:
            print(f"{base}: only {len(q1)} Q1 — not banked")
            continue
        med = np.median(q1, 0)
        sc = float(np.sqrt(np.sum(np.std(q1, 0) ** 2)))
        off = float(np.linalg.norm(med))
        ok = sc <= args.max_scatter and off <= args.max_off_anchor
        print(f"{base}: Q1 n={len(q1)} med ENU=({med[0]:+.3f},"
              f"{med[1]:+.3f},{med[2]:+.3f}) scatter={sc:.3f} "
              f"off={off:.2f} -> {'BANKED' if ok else 'rejected'}")
        if ok:
            per_base[base] = {"enu": med.tolist(), "n": int(len(q1)),
                              "scatter": sc}

    if args.dry or not per_base:
        return
    bank = json.loads(BANK.read_text()) if BANK.exists() else {"arcs": []}
    bank["arcs"].append({"t0": t0, "t1": t1, "bases": per_base,
                         "anchor_at_time": anchor.tolist()})
    BANK.write_text(json.dumps(bank, indent=1))

    # consensus across ALL banked arcs x bases (median, robust)
    pts = [np.array(b["enu"]) + np.array(a["anchor_at_time"]) - anchor
           for a in bank["arcs"] for b in a["bases"].values()]
    pts = np.array(pts)
    cons = np.median(pts, 0)
    spr = np.std(pts, 0)
    print(f"\nbank: {len(pts)} arc-bases | consensus off current anchor "
          f"=({cons[0]:+.3f},{cons[1]:+.3f},{cons[2]:+.3f}) "
          f"std=({spr[0]:.3f},{spr[1]:.3f},{spr[2]:.3f})")
    if float(np.linalg.norm(cons)) > 0.5:
        print("NOTE: consensus differs from anchor by "
              f"{np.linalg.norm(cons):.2f} m — candidate re-anchor is "
              "a human decision (see truth_ref.json provenance rules).")


if __name__ == "__main__":
    main()
