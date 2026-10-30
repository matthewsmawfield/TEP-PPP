#!/usr/bin/env python3
"""RTK replay: rover contiguous chunk stream vs EASN (OS Net) 30 s base.

Now that capture is contiguous (2026-09-25 change), the chunk_*.bin files
in data/ppp_live/ form a gap-free raw RTCM3 record.  This script merges a
span of them, converts to RINEX (1 Hz rover + 30 s decimated), downloads
the EASN hourly RINEX files covering the arc, and runs rnx2rtkp.

EASN is 1.7 km away — a genuine short-baseline RTK case.  The known
blockers from 2026-09-25 morning work: ~20 m systematic L5-code residuals
(LC29H C5Q pilot vs base C5X — GAL E1/E5a avoids it), permanent LLI flags
on E10/E18/E30 (excluded via pos1-exclsats), and an unexplained ~8-12 m
vertical float bias.  Best float config: GAL-only l1+l2+l5.

Usage:
  .venv/bin/python3 scripts/rtk_easn_replay.py \
      [--since EPOCH] [--until EPOCH] [--conf CONF] [--out DIR]

Requires OSNET_API_KEY / OS_NET_API_KEY in the environment (source
.env.local).  rnx2rtkp_demo5 and convbin must be on PATH or in
~/.local/bin / /usr/local/bin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.ntrip_client import (  # noqa: E402
    os_net_list_rinex, os_net_download_rinex, os_net_api_key)

# RTK-AR multi-baseline consensus (110 Q=1 fixes, 4+ OS Net bases,
# ~0.9 m cross-baseline spread) — the externally-measured antenna
# position adopted as truth_ref.json's external anchor 2026-09-27,
# superseded 2026-10-14 by the post-move RTK-AR re-survey in
# truth_ref.json (the antenna physically moved ~Oct-1..3 — see that
# file's provenance).  Truth is now READ FROM truth_ref.json at call
# time so scoring tracks re-anchors; the constant is the last-resort
# fallback only.
TRUTH_ECEF = (3789584.8757, 6815.2491, 5113126.5157)  # fallback ETRS89 anchor


def _truth_ecef():
    """Current external antenna anchor, ETRS89 frame, from truth_ref.json.

    RTK rover fixes inherit the OS Net base frame (ETRS89), so scoring uses
    truth_ecef_etrs89 — NOT truth_ecef, which since 2026-10-19 is the anchor
    expressed in the PPP solver's ITRF frame.
    """
    try:
        import json as _j
        d = _j.loads((Path(__file__).resolve().parents[2]
                      / "data" / "ppp_live" / "truth_ref.json").read_text())
        e = d.get("truth_ecef_etrs89") or d.get("truth_ecef")
        if e and len(e) == 3:
            return tuple(float(v) for v in e)
    except Exception:
        pass
    return TRUTH_ECEF
CHUNK_DIR = Path(__file__).resolve().parents[2] / "data" / "ppp_live"
DEFAULT_CONF = (Path(__file__).resolve().parents[2]
                / "data" / "final_replay" / "rtk_gal_easn.conf")


def _find_tool(name: str) -> str | None:
    for p in [f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}",
              f"/usr/bin/{name}", str(Path.home() / f".local/bin/{name}")]:
        if Path(p).exists():
            return p
    import shutil
    return shutil.which(name)


def chunks_in(since: float, until: float) -> list[Path]:
    """Chunk files whose capture start falls inside [since, until)."""
    out = []
    for d in (CHUNK_DIR, CHUNK_DIR / "raw_archive"):
        if not d.exists():
            continue
        for p in d.glob("chunk_*.bin"):
            try:
                ts = float(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if since <= ts < until:
                out.append((ts, p))
    return [p for _, p in sorted(out)]


def _rinex_text(path: Path):
    """Yield the RINEX text of an OS Net download (.zip, .gz, or plain)."""
    import zipfile
    magic = path.read_bytes()[:4]
    if magic[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.upper().endswith((".RNX", ".OBS")):
                    import io
                    return io.StringIO(
                        z.read(name).decode("utf-8", errors="replace"))
        raise ValueError(f"no RINEX member in {path.name}")
    if magic[:2] == b"\x1f\x8b":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def merge_base_rinex(gz_paths: list[Path], dest: Path) -> None:
    """Concatenate hourly RINEX obs files, keeping only the first header."""
    wrote_header = False
    with open(dest, "w") as out:
        for gz in gz_paths:
            in_header = True
            with _rinex_text(gz) as f:
                for line in f:
                    if in_header:
                        if wrote_header:
                            if "END OF HEADER" in line:
                                in_header = False
                            continue
                        out.write(line)
                        if "END OF HEADER" in line:
                            in_header = False
                            wrote_header = True
                    else:
                        out.write(line)


def _llh2ecef(lat_deg: float, lon_deg: float, h: float):
    import numpy as np
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])


def per_epoch_ecef(pos_file: Path) -> dict:
    """epoch_str -> (Q, ecef np.array).  Dedups like score() (keeps the
    first row per epoch — soltype=combined writes dup rows)."""
    import numpy as np
    out = {}
    for line in open(pos_file):
        if line.startswith("%") or not line.strip():
            continue
        p = line.split()
        if len(p) < 10:
            continue
        epoch = f"{p[0]} {p[1]}"
        if epoch in out:
            continue
        try:
            q = int(p[5])
            llh = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        out[epoch] = (q, _llh2ecef(*llh))
    return out


def consensus_veto(fix_pos: Path, ref_pos: Path, thresh_m: float,
                   truth_ecef=None) -> dict:
    """Veto Q=1 epochs in fix_pos that disagree with an independent
    reference solution (ref_pos, any Q) by > thresh_m in ECEF.

    Truth-free integrity channel: wrong-integer fixes sit 1-12 m off a
    bounded float reference while true fixes agree to <0.6 m, a split
    no internal statistic (ratio, formal sd) exposes.  If truth_ecef is
    given, vetoed/kept epochs are additionally audited against it —
    evaluation only, the veto itself never uses truth."""
    import numpy as np
    fx, rf = per_epoch_ecef(fix_pos), per_epoch_ecef(ref_pos)
    truth = np.array(truth_ecef) if truth_ecef is not None else None
    n_cmp = n_veto = n_no_ref = 0
    veto_tp = veto_fp = kept_verified = 0
    kept_errs, vetoed_errs = [], []
    # First pass: per-epoch fix-ref offset vectors for all arbitrable
    # epochs (ref not itself fixed — a wrong-integer ref would inherit
    # the failure mode it is meant to flag).
    cmp_eps, offs, errs = [], [], []
    for ep, (q, e) in fx.items():
        if q != 1:
            continue
        ref = rf.get(ep)
        if ref is None or ref[0] == 1:
            n_no_ref += 1
            continue
        cmp_eps.append(ep)
        offs.append(np.asarray(e, float) - np.asarray(ref[1], float))
        errs.append(float(np.linalg.norm(e - truth))
                    if truth is not None else None)
    n_cmp = len(cmp_eps)
    # Common-mode centring (2026-09-30, live arc rtk_1790738424): the
    # float reference carries a stable arc-level offset vs the fix
    # (observed 0.04-0.91 m, consistent direction per arc — a float-side
    # bias, e.g. residual iono/datum).  Once it exceeds thresh the raw
    # |fix-ref| veto fires on EVERY true fix (fp=44/102 live).  True
    # fixes sit <=0.46 m off the arc's median offset while wrong
    # integers sit >=0.55-0.99 m away, so veto on the DEVIATION from
    # the common mode instead.  The component median is robust to a
    # ~50% false-fix fraction only while the false offsets are diverse;
    # a single-mode false-fix majority inverts it (arc rtk_1790809642)
    # — caught by the inversion guard below.  Centering needs
    # >=TEP_RTK_EASN_VETO_MINCMP(10) compared epochs to be meaningful —
    # below that, fall back to the raw distance.
    centered = (n_cmp >= int(os.environ.get("TEP_RTK_EASN_VETO_MINCMP",
                                           "10"))
                and int(os.environ.get("TEP_RTK_EASN_VETO_CENTER", "1")))
    med_off = (np.median(np.asarray(offs), axis=0)
               if centered else np.zeros(3))
    # Centre-inversion guard (2026-10-01, live arc rtk_1790809642): when
    # >half the compared fixes share ONE consistent wrong-integer offset,
    # the component median locks onto the wrong track and the centred
    # veto keeps it — kept 37 all-false, vetoed the 10 true (fp=10).
    # Signature: the kept set's RAW |fix-ref| median exceeds thresh
    # while a >=TEP_RTK_EASN_VETO_INV_MIN(3) minority still sits within
    # thresh of the reference — the reference (bounded float) is the
    # arbiter of which mode is real, so flip to the raw-distance veto.
    # A genuine arc-level float bias (arc rtk_1790738424) leaves ZERO
    # near-float epochs, so it cannot trigger the flip.  Verified on all
    # 4 archived pairs: 9642 fp 10->0 / tp 21->47; 782406, 792168,
    # 800831 unchanged.  Cross-base fix-vs-fix does NOT help (measured
    # on SWAN 41 km: wrong integers are rover-side, both bases' fixes
    # agree to 0.04 m median).  Disable with TEP_RTK_EASN_VETO_INV=0.
    d_raw = np.linalg.norm(np.asarray(offs), axis=1) if offs else None
    inv_flip = False
    inv_unreliable = False
    if (centered and n_cmp
            and int(os.environ.get("TEP_RTK_EASN_VETO_INV", "1"))):
        d_c = np.linalg.norm(np.asarray(offs) - med_off, axis=1)
        kept_raw = d_raw[d_c <= thresh_m]
        n_near = int((d_raw <= thresh_m).sum())
        if len(kept_raw) and float(np.median(kept_raw)) > thresh_m:
            if n_near >= int(os.environ.get(
                    "TEP_RTK_EASN_VETO_INV_MIN", "3")):
                inv_flip = True
            else:
                inv_unreliable = True
    for ep, off, err in zip(cmp_eps, offs, errs):
        d = (float(np.linalg.norm(off))
             if inv_flip else float(np.linalg.norm(off - med_off)))
        if d > thresh_m:
            n_veto += 1
            if err is not None:
                vetoed_errs.append(err)
                if err > 0.5:
                    veto_tp += 1
                else:
                    veto_fp += 1
        else:
            if err is not None:
                kept_errs.append(err)
                kept_verified += int(err <= 0.5)
    n_kept = n_cmp - n_veto
    res = {"thresh_m": thresh_m, "n_q1_compared": n_cmp,
           "n_q1_no_ref": n_no_ref, "n_vetoed": n_veto,
           "n_q1_consensus": n_kept,
           "ref_common_offset_m": round(float(np.linalg.norm(med_off)), 3)
           if centered else 0.0,
           "centered": bool(centered),
           "inversion_raw_fallback": bool(inv_flip),
           "consensus_unreliable": bool(inv_unreliable)}
    if truth is not None:
        res.update({
            "n_consensus_verified_0p5m": kept_verified,
            "purity_pct": (round(100.0 * kept_verified / n_kept, 1)
                           if n_kept else None),
            "veto_tp": veto_tp, "veto_fp": veto_fp,
            "kept_err_max_m": (round(max(kept_errs), 3)
                               if kept_errs else None),
            "vetoed_err_min_m": (round(min(vetoed_errs), 3)
                                 if vetoed_errs else None)})
    return res


def second_base_check(fix_pos: Path, ref_pos: Path,
                      agree_m: float) -> dict:
    """Cross-check Q=1 fix sets between two independent bases.

    Centre-inverted veto failure (live arc rtk_1790809642): when >half
    an arc's fixes share ONE consistent wrong-integer offset, the
    median-centred df-float veto subtracts the wrong track and keeps
    them.  No float reference can disambiguate that (a rover-side float
    bias appears on every baseline's float), but an independent base's
    INTEGER draw can: agreeing fix sets are the same integer family
    (measured 0.01-0.11 m on 2026-10-01 test arc, even when both sit at
    a shared ~0.5-0.8 m RTK datum offset), while a wrong-integer fix
    sits ~1 m+ off the other base's fix (measured 1.96 m).

    Audit-only return — decision wiring lives in the caller."""
    import numpy as np
    fx, rf = per_epoch_ecef(fix_pos), per_epoch_ecef(ref_pos)
    ds, agree, disagree = [], 0, 0
    n_fx_q1 = n_rf_q1 = 0
    for ep, (q, e) in fx.items():
        if q != 1:
            continue
        n_fx_q1 += 1
        s = rf.get(ep)
        if s is None or s[0] != 1:
            continue
        ds.append(float(np.linalg.norm(e - s[1])))
    n_rf_q1 = sum(1 for q, _ in rf.values() if q == 1)
    for d in ds:
        if d <= agree_m:
            agree += 1
        else:
            disagree += 1
    med = float(np.median(ds)) if ds else None
    return {"n_ref_q1": n_rf_q1, "n_q1_matched": len(ds),
            "agree_m": agree_m, "n_agree": agree, "n_disagree": disagree,
            "median_dist_m": round(med, 3) if med is not None else None,
            "max_dist_m": round(max(ds), 3) if ds else None,
            # True only when enough matched epochs exist AND the fix
            # sets agree — a corrupt-integer arc reads False, no-data
            # reads None (caller decides how to weigh it).
            "arc_consistent": (med <= agree_m) if len(ds) >= 5 else None}


def score(pos_file: Path) -> dict:
    """Q=1/Q=2 quality counts and 3D error vs surveyed truth."""
    import numpy as np
    truth = np.array(_truth_ecef())
    q1 = q2 = q5 = 0
    errs1, errs2 = [], []
    seen = set()
    for line in open(pos_file):
        if line.startswith("%") or not line.strip():
            continue
        p = line.split()
        if len(p) < 10:
            continue
        epoch = f"{p[0]} {p[1]}"
        if epoch in seen:            # soltype=combined writes dup rows
            continue
        seen.add(epoch)
        try:
            q = int(p[5])
            llh = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        err = float(np.linalg.norm(_llh2ecef(*llh) - truth))
        if q == 1:
            q1 += 1
            errs1.append(err)
        elif q == 2:
            q2 += 1
            errs2.append(err)
        else:
            q5 += 1
    def _stats(a):
        if not a:
            return "n/a"
        return (f"n={len(a)} median={float(np.median(a)):.2f}m "
                f"p90={float(np.percentile(a, 90)):.2f}m "
                f"min={min(a):.2f}m max={max(a):.2f}m")
    return {"q1": q1, "q2": q2, "q5": q5,
            "q1_within_05": int(sum(1 for e in errs1 if e <= 0.5)),
            "q1_err": _stats(errs1), "q2_err": _stats(errs2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0.0,
                    help="chunk start >= this unix epoch")
    ap.add_argument("--until", type=float, default=9e18)
    ap.add_argument("--conf", default=str(DEFAULT_CONF))
    ap.add_argument("--out", default=None, help="output dir (default: mktemp)")
    ap.add_argument("--station", default="EASN")
    ap.add_argument("--veto-ref", default=None,
                    help="second .pos file (e.g. dual-freq float arm) used "
                         "as truth-free consensus reference: Q=1 epochs "
                         ">--veto-m from it are wrong-integer fixes")
    ap.add_argument("--veto-m", type=float, default=0.75)
    args = ap.parse_args()

    convbin = _find_tool("convbin_demo5") or _find_tool("convbin")
    rnx2rtkp = _find_tool("rnx2rtkp_demo5") or _find_tool("rnx2rtkp")
    if not convbin or not rnx2rtkp:
        sys.exit("convbin/rnx2rtkp not found")
    if not os_net_api_key():
        sys.exit("OS_NET_API_KEY/OSNET_API_KEY not set (source .env.local)")

    out_dir = Path(args.out) if args.out else Path(
        tempfile.mkdtemp(prefix="rtk_easn_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunks_in(args.since, args.until)
    if len(chunks) < 2:
        sys.exit(f"only {len(chunks)} chunks in range")
    t0 = float(chunks[0].stem.split("_")[1])
    t1 = float(chunks[-1].stem.split("_")[1]) + 300.0
    print(f"[rtk] {len(chunks)} chunks "
          f"{dt.datetime.fromtimestamp(t0, dt.timezone.utc):%H:%M}-"
          f"{dt.datetime.fromtimestamp(t1, dt.timezone.utc):%H:%M} UTC")

    raw = out_dir / "rover.bin"
    with open(raw, "wb") as f:
        for c in chunks:
            f.write(c.read_bytes())

    # Rover obs at 1 Hz and decimated to the base's 30 s epochs.
    for tag, extra in (("rover", []), ("rover30", ["-ti", "30"])):
        d = out_dir / f"conv_{tag}"
        d.mkdir(exist_ok=True)
        subprocess.run([convbin, "-r", "rtcm3", "-f", "3", "-d", str(d),
                        *extra, str(raw)],
                       capture_output=True, timeout=120)
    # Solve the 1 Hz stream — a 2026-10-19 A/B showed -ti 30 costs
    # ~2.5x fix accuracy and ~20x fewer fixed epochs (misc-timeinterp=on
    # covers the base cadence).  conv_rover30 kept for comparisons.
    rover_obs = next((out_dir / "conv_rover").glob("*.obs"))
    rover_nav = next((out_dir / "conv_rover").glob("*.nav"))

    # EASN hourly RINEX covering the arc.
    s_iso = dt.datetime.fromtimestamp(t0 - 3600, dt.timezone.utc) \
        .strftime("%Y-%m-%dT%H:00:00Z")
    e_iso = dt.datetime.fromtimestamp(t1 + 3600, dt.timezone.utc) \
        .strftime("%Y-%m-%dT%H:00:00Z")
    files = os_net_list_rinex(args.station, s_iso, e_iso)
    print(f"[rtk] {len(files)} {args.station} RINEX files listed")
    base_parts = []
    for i, f in enumerate(files):
        dest = out_dir / f"base_{i:02d}.rnx.gz"
        if os_net_download_rinex(f.get("url", ""), dest):
            base_parts.append(dest)
            print(f"[rtk]   got {f.get('fileName', dest.name)}")
    if not base_parts:
        sys.exit("no base RINEX downloaded (hour may not be published yet)")
    base_rnx = out_dir / "base.rnx"
    merge_base_rinex(base_parts, base_rnx)

    pos = out_dir / "rtk.pos"
    cmd = [rnx2rtkp, "-k", args.conf, "-o", str(pos),
           str(rover_obs), str(base_rnx), str(rover_nav)]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    print(f"[rtk] rnx2rtkp rc={r.returncode}")
    if r.returncode != 0 or not pos.exists():
        print(r.stderr.decode(errors="ignore")[-500:])
        sys.exit(2)
    st = score(pos)
    print(f"[rtk] Q=1 fixed: {st['q1']}   Q=2 float: {st['q2']}   "
          f"Q=5/0: {st['q5']}")
    print(f"[rtk] Q=1 3D err vs truth: {st['q1_err']}")
    print(f"[rtk] Q=2 3D err vs truth: {st['q2_err']}")
    if args.veto_ref:
        cv = consensus_veto(pos, Path(args.veto_ref), args.veto_m,
                            truth_ecef=_truth_ecef())
        print(f"[rtk] consensus veto @{args.veto_m}m vs {args.veto_ref}: "
              f"{cv['n_vetoed']}/{cv['n_q1_compared']} vetoed, kept "
              f"{cv['n_q1_consensus']} (verified "
              f"{cv.get('n_consensus_verified_0p5m')}, purity "
              f"{cv.get('purity_pct')}%, tp={cv.get('veto_tp')} "
              f"fp={cv.get('veto_fp')})")
    print(f"[rtk] outputs in {out_dir}")
    print("[rtk] HONESTY: Q=1 epochs landing >0.5 m off truth are FALSE "
          "fixes — do not report n_fixed from them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
