#!/usr/bin/env python3
"""Incrementally extract per-sat sidereal residuals for a UTC day into
data/ppp_live/sidereal_resid/<day>.pkl — run hourly (launchd) so the
daily map build sees a FULL day's residuals instead of whatever
obs_archive still held at 02:15.

Why this exists: the obs_archive only retains the trailing ~16-40 h of
window files, and the daily build_sidereal_map.py run scans it once for
yesterday's residuals.  Obs from yesterday's early UTC hours are
already evicted by then, so the residual day table has a multi-hour
hole and the sidereal template serves nothing to early-UTC windows
(measured 2026-10-14: Oct-13 table covered only 22:33-00:31 before
incremental capture, vs 11:29-24:00 after one catch-up pass).

Each run appends only epochs not already in the day's table (the
residuals are epoch-median-removed, so overlapping re-extraction would
just duplicate work).  The day's truth stamp is refreshed only when the
table is empty; a >2 m re-anchor means the whole table is rebuilt
(same guard as the builder).

Usage:
  update_sidereal_resid.py [--day YYYY-MM-DD] [--stride 5]
"""
import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "live"))
sys.path.insert(0, str(ROOT / "rpi"))

from build_sidereal_map import (ARCH, RESID_DIR, TRUTH_REF,  # noqa: E402
                                extract_residuals, load_products,
                                obs_day, obs_epoch_sods, gps_to_unix)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None,
                    help="UTC date YYYY-MM-DD (default: today)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--days-back", type=int, default=1,
                    help="also update this many days before --day "
                    "(backfills midnight-straddling windows' early "
                    "epochs that arrived after yesterday's last run)")
    args = ap.parse_args()

    day = (dt.date.fromisoformat(args.day) if args.day
           else dt.datetime.now(dt.timezone.utc).date())

    tr = json.loads(TRUTH_REF.read_text())
    truth = np.array(tr.get("truth_ecef") or tr.get("ecef"), float)

    # 2026-10-19: run the pass for today AND the trailing day(s).  The
    # old today-only behaviour truncated each day's table ~1-2 h before
    # UTC midnight — windows starting e.g. 22:45 yesterday cover the
    # missing span but were never revisited once "today" rolled over.
    rc = 0
    for back in range(args.days_back + 1):
        rc |= (_update_day(day - dt.timedelta(days=back), truth,
                           args) or 0)
    return rc


def _update_day(day, truth, args) -> int:
    RESID_DIR.mkdir(parents=True, exist_ok=True)
    cp = RESID_DIR / f"{day.isoformat()}.pkl"
    res = {}
    if cp.exists():
        try:
            blob = pickle.loads(cp.read_bytes())
            r = blob.get("res", {}) if isinstance(blob, dict) else blob
            t0 = blob.get("truth") if isinstance(blob, dict) else None
            if r and (t0 is None or
                      float(np.linalg.norm(np.asarray(t0) - truth))
                      < 2.0):
                res = dict(r)
            else:
                print(f"truth shift detected — rebuilding {day} "
                      f"table from scratch")
        except Exception:
            pass

    # Parallel PHASE-dev table (sidereal_resid_ph/<day>.pkl): feeds the
    # sidereal phase map — multipath + product-error AC repeat at
    # sidereal lag in carrier phase (measured rho 0.36-0.62 on the
    # 2026-10-13/14 pair).  The ambiguity constant is left IN each dev;
    # arc-splitting at build time uses the stored slip flags + gaps.
    php = RESID_DIR.parent / "sidereal_resid_ph"
    php.mkdir(parents=True, exist_ok=True)
    pph = php / f"{day.isoformat()}.pkl"
    ph_res, ph_sl = {}, {}
    if pph.exists():
        try:
            blob = pickle.loads(pph.read_bytes())
            r = blob.get("res", {}) if isinstance(blob, dict) else blob
            t0 = blob.get("truth") if isinstance(blob, dict) else None
            if r and (t0 is None or
                      float(np.linalg.norm(np.asarray(t0) - truth))
                      < 2.0):
                ph_res = dict(r)
                ph_sl = dict(blob.get("slips", {})
                             if isinstance(blob, dict) else {})
        except Exception:
            pass

    pp = load_products(day)
    if pp is None:
        print(f"no products covering {day}")
        return 2

    day0 = dt.datetime.combine(day, dt.time(),
                               tzinfo=dt.timezone.utc).timestamp()
    day1 = day0 + 86400.0
    # Windows whose obs may intersect `day`: first-epoch day within
    # [day-1, day].  The window's own span check inside
    # extract_residuals filters to the day's epochs.
    added = 0
    for ob in sorted(ARCH.glob("*.obs")):
        d = obs_day(ob)
        if d is None or not (d <= day <= d + dt.timedelta(days=1)):
            continue
        # Only extract epochs that are (a) inside `day` and (b) not
        # already in the table — incremental runs re-parse the window's
        # epoch list (cheap) but skip the per-epoch measurement solve
        # for anything already captured.
        sods = obs_epoch_sods(ob)
        # Phase missing sods are wanted only inside the existing phase
        # table's span: the table is seeded from measurement caches, so
        # epochs before its first entry are a deliberate no-backfill
        # zone (re-solving the whole day for phase costs ~90 min).
        _ph_floor = min(ph_res) - 1e-6 if ph_res else -np.inf
        want = {s for s in sods
                if (s not in res or (s not in ph_res and s > _ph_floor))
                and day0 <= gps_to_unix(s) < day1}
        if not want:
            continue
        nav = ob.with_suffix(".nav")
        navs = [nav] if nav.exists() else []
        tab, ptab, psl = extract_residuals(
            ob, navs, truth, pp, stride=args.stride, only_date=day,
            want_sods=want, with_phase=True)
        for g, dd in tab.items():
            res.setdefault(g, dd)
        for g, dd in ptab.items():
            ph_res.setdefault(g, dd)
        for g, ss in psl.items():
            ph_sl.setdefault(g, set()).update(ss)
        added += len(tab)
        if tab:
            print(f"  {ob.name}: +{len(tab)} epochs (+{len(ptab)} ph)")
    if added:
        tmp = cp.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(
            {"res": res, "truth": np.asarray(truth, float).tolist()}))
        tmp.rename(cp)
        tmp = pph.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(
            {"res": ph_res, "slips": ph_sl,
             "truth": np.asarray(truth, float).tolist()}))
        tmp.rename(pph)
    print(f"{day}: {len(res)} epochs total (+{added} new), "
          f"{len(ph_res)} phase epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
