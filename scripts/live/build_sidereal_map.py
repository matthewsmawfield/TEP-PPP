#!/usr/bin/env python3
"""Build the sidereal residual map consumed by
PPPProcessor._apply_sidereal_correction (TEP_SIDEREAL=1).

Scans data/ppp_live/obs_archive for windows covering a UTC day
(default: yesterday), extracts per-sat code residuals against
truth_ref.json + precise products for that DOY, and writes
data/ppp_live/sidereal_map.pkl:

    {"sats": {sat: {"u": unix_seconds, "r": residuals_m}},
     "lag": {sat: lag_seconds},          # ~86164, per-sat learned
     "corr": {sat: last-pair corr},      # validation, gating apply
     "meta": {...}}

Per-sat lags/corrs are learned from the previous day-pair when that
day's residuals can also be built from the archive, else carried from
the existing map, else nominal 86164.  Apply is gated per-sat on the
map's stored corr (see _apply_sidereal_correction).

Usage:
  build_sidereal_map.py [--day YYYY-MM-DD] [--stride N] [--out PATH]
"""
import argparse, datetime as dt, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "rpi"))

ARCH = ROOT / "data" / "ppp_live" / "obs_archive"
# Own product dir: staging cod_rap_* into the LIVE precise dir would
# thrash the extract-cache content hash on the running service.
PRECISE = ROOT / "data" / "ppp_live" / "precise_sidereal"
PRECISE_LIVE = ROOT / "data" / "ppp_live" / "precise"
TRUTH_REF = ROOT / "data" / "ppp_live" / "truth_ref.json"
OUT = ROOT / "data" / "ppp_live" / "sidereal_map.pkl"
# Per-day residual tables persist beyond obs_archive retention (~40 h) —
# validation and the multi-day stack need 3+ days of depth, and reusing
# yesterday's table skips its ~10 min re-extraction.
RESID_DIR = ROOT / "data" / "ppp_live" / "sidereal_resid"
RESID_KEEP = 8
SIDEREAL_LAG = 86164.0
# GLONASS 2-orbit quasi-repeat (2026-10-18, tmp/esat_lags.py +
# rsat_verify.py on the resid tables): R-sat residuals echo at
# ~2x the GLONASS orbital period (~22.4-22.7 h, pooled corr
# +0.12..+0.29, per-sat to +0.70 — replicated on all three day
# pairs 18/17, 17/16, 16/15), NOT at the GPS 86164 s lag where the
# previous single-centre search left them with ~0-corr garbage
# lags.  Scan R sats around this second centre and keep whichever
# centre's fit correlates better.
GLO_LAG = 81100.0
GLO_LAG_SPAN = 1800.0


def obs_day(path):
    """UTC date of the FIRST epoch in a RINEX obs file."""
    for ln in open(path):
        if ln.startswith(">"):
            p = ln.split()
            return dt.date(int(p[1]), int(p[2]), int(p[3]))
        if "END OF HEADER" in ln:
            continue


def obs_epoch_sods(path):
    """Sorted GPS-absolute epoch keys of an obs file (the keys
    get_measurements uses), via the shared RINEX 3 parser."""
    from lib.rinex_pseudorange import _parse_rinex3_obs
    return sorted(_parse_rinex3_obs(Path(path)).keys())


# Unix epoch of GPS time zero (1980-01-06) plus accumulated leap seconds:
# the parser's absolute-GPS axis converted to ~UTC unix for day filtering.
GPS_UNIX_OFFSET = 315964800 - 18


def gps_to_unix(g):
    return g + GPS_UNIX_OFFSET


def load_products(day):
    """Precise products covering `day`: prefer COD RAP in
    PRECISE/cod_rap_<doy>_*, else the live merged products if their
    epoch coverage reaches `day`."""
    from lib.precise_products import (PreciseProducts, load_cached_sp3,
                                      load_cached_clk, find_products_for_date,
                                      load_cached_products)
    doy = (day - dt.date(day.year, 1, 1)).days + 1
    year = day.year
    sp3 = clk = None
    for nm in (f"cod_rap_{doy}_orb.sp3", f"cod_rap_{doy}.sp3"):
        p = PRECISE / nm
        if p.exists():
            sp3 = load_cached_sp3(p, year, doy)
            break
    for nm in (f"cod_rap_{doy}_clk.clk", f"cod_rap_{doy}.clk",
               "clock_cod_combined.clk", "clock.clk"):
        p = PRECISE / nm
        if p.exists():
            clk = load_cached_clk(p, year, doy)
            if clk is not None:
                break
    if sp3 is None:
        pp = find_products_for_date(
            year, doy, ROOT / "data" / "rinex" / "sp3",
            ROOT / "data" / "rinex" / "clk")
        if pp is not None:
            return pp
        pp = load_cached_products(year, doy, PRECISE)
        if pp is not None:
            return pp
        # Final fallback: the live merged ultra-rapid stack in
        # data/ppp_live/precise/ (orbit_cod.sp3 + clock.clk cover
        # ~2 days back).  Read-only lookup — nothing is staged into
        # that dir (staging would thrash the live extract-cache hash).
        return load_cached_products(year, doy, PRECISE_LIVE)
    pp = PreciseProducts(sp3, clk)
    pp._use_sow = True
    pp._doy = doy
    return pp


def extract_residuals(obs_path, nav_paths, truth, pp, stride=1,
                      only_date=None, want_sods=None, with_phase=False):
    """{gps_abs: {sat: resid}} for epochs in `only_date` (else all).

    `want_sods` (set of absolute-GPS-second keys) additionally restricts
    the extraction to epochs still needed — used by
    update_sidereal_resid.py to skip epochs already captured.

    `with_phase=True` returns (code_res, phase_dev, slip_flags): the
    phase-dev table feeds the sidereal PHASE map (multipath + product
    error repeat at sidereal lag in carrier phase too — measured
    rho 0.36-0.62 on the 2026-10-13/14 pair, G25 0.62, G17 0.49) and
    the slip flags let the builder split ambiguity arcs."""
    from gps_web import parse_rinex_header_position
    from lib.rinex_pseudorange import get_measurements
    rx = parse_rinex_header_position(obs_path)
    if rx is None:
        rx = truth
    sods = obs_epoch_sods(obs_path)
    if not sods:
        return ({}, {}, {}) if with_phase else {}
    keep = set(sods[::max(1, stride)])
    if want_sods is not None:
        keep &= set(want_sods)
        if not keep:
            return ({}, {}, {}) if with_phase else {}
    rms = get_measurements(str(obs_path), [str(n) for n in nav_paths],
                           rx, precise_products=pp, only_sods=keep)
    from lib.sidereal_map import (residuals_from_measurements,
                                  phase_dev_from_measurements)
    res = residuals_from_measurements(rms, truth)
    ph = sl = {}
    if with_phase:
        ph, sl = phase_dev_from_measurements(rms, truth)
    if only_date:
        day0 = dt.datetime.combine(only_date, dt.time(),
                                   tzinfo=dt.timezone.utc).timestamp()
        res = {g: d for g, d in res.items()
               if day0 <= gps_to_unix(g) < day0 + 86400}
        if with_phase:
            ph = {g: d for g, d in ph.items()
                  if day0 <= gps_to_unix(g) < day0 + 86400}
            sl = {g: s for g, s in sl.items()
                  if day0 <= gps_to_unix(g) < day0 + 86400}
    return (res, ph, sl) if with_phase else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None,
                    help="UTC date YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--stride", type=int, default=5,
                    help="epoch stride (s) — 5 keeps extraction ~10 min")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--lag-span", type=float, default=900,
                    help="per-sat best-lag half-span around 86164 s")
    ap.add_argument("--stack-days", type=int, default=1,
                    help="total days of residual realisations to stack; "
                         ">1 shifts each older day's table forward by "
                         "k*lag_s onto yesterday's axis so the apply "
                         "interpolation averages realisations (lower "
                         "template noise, wider sat coverage)")
    args = ap.parse_args()

    import json
    day = (dt.date.fromisoformat(args.day) if args.day
           else dt.datetime.now(dt.timezone.utc).date()
           - dt.timedelta(days=1))
    print(f"building sidereal map for {day} (stride={args.stride}s)")

    tr = json.loads(TRUTH_REF.read_text())
    truth = np.array(tr.get("truth_ecef") or tr.get("ecef"), float)
    pp = load_products(day)
    if pp is None:
        print(f"no products covering {day} in {PRECISE}")
        sys.exit(2)

    obs_files = sorted(ARCH.glob("*.obs"))
    RESID_DIR.mkdir(parents=True, exist_ok=True)
    res_day = {}

    def day_residuals(dk, tag):
        """Residuals for UTC day dk — cached pkl first, else extract."""
        import pickle as _pk
        cp = RESID_DIR / f"{dk.isoformat()}.pkl"
        if cp.exists():
            try:
                blob = _pk.loads(cp.read_bytes())
                r = blob.get("res", {}) if isinstance(blob, dict) else blob
                t0 = blob.get("truth") if isinstance(blob, dict) else None
                # Re-anchor guard: residuals are epoch-median-removed so a
                # small truth shift is near-common-mode, but a >2 m re-
                # anchor is not — rebuild rather than ship a stale datum.
                ok = (r and (t0 is None or
                             float(np.linalg.norm(np.asarray(t0) - truth))
                             < 2.0))
                if ok:
                    print(f"  {tag} {dk}: {len(r)} epochs (cache)")
                    return r
            except Exception:
                pass
        pp_k = load_products(dk)
        if pp_k is None:
            print(f"  {tag} {dk}: no products")
            return {}
        r = {}
        for ob in obs_files:
            d = obs_day(ob)
            if d is None or not (d <= dk <= d + dt.timedelta(days=1)):
                continue
            nav = ob.with_suffix(".nav")
            navs = [nav] if nav.exists() else []
            print(f"  {tag} {ob.name} ...", flush=True)
            tab = extract_residuals(ob, navs, truth, pp_k,
                                    stride=args.stride, only_date=dk)
            for u, dd in tab.items():
                r.setdefault(u, dd)
        if r:
            try:
                cp.write_bytes(_pk.dumps(
                    {"res": r, "truth": np.asarray(truth, float).tolist()}))
            except Exception:
                pass
        print(f"  {tag} {dk}: {len(r)} epochs")
        return r

    res_day = day_residuals(day, "day")
    print(f"day residual epochs: {len(res_day)}")
    # Prune residual tables older than RESID_KEEP days.
    try:
        for cp in RESID_DIR.glob("*.pkl"):
            try:
                if (dt.date.fromisoformat(cp.stem)
                        < day - dt.timedelta(days=RESID_KEEP)):
                    cp.unlink()
            except ValueError:
                cp.unlink()
    except Exception:
        pass

    # Lag validation: build yesterday-minus-one residuals if the archive
    # covers it and the same product class exists; else carry the
    # existing map's lags/corr.
    lags, corrs = {}, {}
    old = {}
    out_p = Path(args.out)
    if out_p.exists():
        try:
            old = pickle.load(open(out_p, "rb"))
        except Exception:
            old = {}
    lags = dict(old.get("lag") or {})
    corrs = dict(old.get("corr") or {})

    prev = day - dt.timedelta(days=1)
    res_prev = day_residuals(prev, "pair")
    _rp2 = _rp3 = _rp4 = _rp5 = _rp6 = None
    if res_prev and res_day:
        from lib.sidereal_map import best_lags
        bl = best_lags(res_prev, res_day, center=SIDEREAL_LAG,
                       half_span=args.lag_span)
        # GLONASS second centre (~22.5 h 2-orbit quasi-repeat): the
        # R echo replicates across day pairs; keep the better-
        # correlating centre per R sat.
        bl_r = best_lags(res_prev, res_day, center=GLO_LAG,
                         half_span=GLO_LAG_SPAN)
        for s, (lg, c) in bl_r.items():
            if s.startswith("R") and c > bl.get(s, (0.0, -1.0))[1]:
                bl[s] = (lg, c)
        for s, (lg, c) in bl.items():
            lags[s], corrs[s] = float(lg), float(c)
        # Deep-lag centres (2026-10-18, tmp/multi_orbit.py +
        # g2day.py): GPS residual echoes at 2 sidereal days
        # (~172300 s, corr 0.44-0.79 on the 18<-16 pair — half the
        # 1-day strength) and GLONASS at 4 orbits (~162000 s,
        # 0.26-0.73, replicated on two pairs).  Validated against
        # the day-2 table; a sat keeps whichever of its candidate
        # lags correlates best, so deep lags only win where the
        # day-1 echo is weak or the sat was thin on day-1.
        _rp2 = day_residuals(day - dt.timedelta(days=2), "deep k=2")
        if _rp2:
            for s, (lg, c) in best_lags(
                    _rp2, res_day, center=GLO_LAG * 2,
                    half_span=GLO_LAG_SPAN).items():
                if s.startswith("R") and c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
            for s, (lg, c) in best_lags(
                    _rp2, res_day, center=SIDEREAL_LAG * 2,
                    half_span=args.lag_span).items():
                # G and E both echo at ~2 sidereal days (E verified
                # 2026-10-18 on the 18<-16 pair: E31 +0.69, E09 +0.62,
                # E26 +0.59 — often STRONGER than the E day-1 echo;
                # tmp/e_2day.py).  The same-day live gate rejects
                # false picks, so an optimistic scan here is safe.
                if s[:1] in ("G", "E") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
        # Deeper centres (2026-10-18, tmp/deep3_screen.py): the G/E
        # code-residual echo is still strong at 3 (pooled mean +0.33,
        # 12 sats >0.6 on the 18<-15 pair) and 4 (+0.37) sidereal days,
        # and R adds a third centre at 6 GLONASS orbits (~243300 s:
        # R08 +0.60, R21 +0.55, R02 +0.51).  _rp3/_rp4 are the tables a
        # ~3/4-day lag reaches back to for validation; apply-side
        # coverage comes from the unshifted merge below (a 4-day lag on
        # day X+1 queries res_{X-3}, so _rp3 joins stack_tables).
        _rp3 = day_residuals(day - dt.timedelta(days=3), "deep k=3")
        if _rp3:
            for s, (lg, c) in best_lags(
                    _rp3, res_day, center=GLO_LAG * 3,
                    half_span=GLO_LAG_SPAN * 1.5).items():
                if s.startswith("R") and c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
            for s, (lg, c) in best_lags(
                    _rp3, res_day, center=SIDEREAL_LAG * 3,
                    half_span=args.lag_span).items():
                if s[:1] in ("G", "E") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
        _rp4 = day_residuals(day - dt.timedelta(days=4), "deep k=4")
        if _rp4:
            for s, (lg, c) in best_lags(
                    _rp4, res_day, center=SIDEREAL_LAG * 4,
                    half_span=args.lag_span).items():
                if s[:1] in ("G", "E") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
            # R 8-orbit centre (~324400 s): u = t_X - 324400 lands on
            # day X-4 — validated here (tmp/deep5_screen.py: +0.21
            # pooled, R08/R19 +0.59 on the 18<-14 pair).
            for s, (lg, c) in best_lags(
                    _rp4, res_day, center=GLO_LAG * 4,
                    half_span=GLO_LAG_SPAN * 1.5).items():
                if s.startswith("R") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
        # 5/6-sidereal-day + R 9-orbit centres (2026-10-18,
        # tmp/deep5_screen.py): the echo ridge barely decays —
        # 5d mean +0.39 (G12 +0.90), 6d +0.33 on a thin table, R
        # 9-orb ~364950 s +0.30 (R05 +0.73, R24 +0.63, R23 +0.54).
        _rp5 = day_residuals(day - dt.timedelta(days=5), "deep k=5")
        if _rp5:
            for s, (lg, c) in best_lags(
                    _rp5, res_day, center=SIDEREAL_LAG * 5,
                    half_span=args.lag_span).items():
                if s[:1] in ("G", "E") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
            for s, (lg, c) in best_lags(
                    _rp5, res_day, center=GLO_LAG * 4.5,
                    half_span=GLO_LAG_SPAN * 1.5).items():
                if s.startswith("R") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
            # R 10-orbit (~405500 s = 5x GLO_LAG, tmp/r10_g6_screen.py
            # 2026-10-19): +0.24 mean on the 18<-14 pair, R06 +0.45,
            # R15 +0.40 — validation u=sod-405500 lands inside the
            # day-5 table; apply lands inside the merged day-4/3
            # coverage.
            for s, (lg, c) in best_lags(
                    _rp5, res_day, center=GLO_LAG * 5.0,
                    half_span=GLO_LAG_SPAN * 1.5).items():
                if s.startswith("R") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
        _rp6 = day_residuals(day - dt.timedelta(days=6), "deep k=6")
        if _rp6:
            for s, (lg, c) in best_lags(
                    _rp6, res_day, center=SIDEREAL_LAG * 6,
                    half_span=args.lag_span).items():
                if s[:1] in ("G", "E") and \
                        c > bl.get(s, (0.0, -1.0))[1]:
                    bl[s] = (lg, c)
                    lags[s], corrs[s] = float(lg), float(c)
        print(f"lag-validated sats: {len(bl)} "
              f"(corr>0.35: {sum(1 for v in bl.values() if v[1]>=0.35)})")
        for s in sorted(bl):
            print(f"    {s}: lag={bl[s][0]:.0f} corr={bl[s][1]:+.2f}")
    else:
        print("no previous-day pair — carrying existing lags/corr")

    # ---- multi-day stack (--stack-days > 1) --------------------------
    # Each older day's table is shifted forward by k*lag_s onto `day`'s
    # axis and merged into the map, so the apply-side interpolation at
    # u-lag averages n realisations of the same repeatable sky-anchored
    # error — template noise falls ~sqrt(n) and sats thin on the single
    # most-recent day still yield a template.  Lags for shifting come
    # from the freshest available validation pair; sats absent from the
    # (prev, day) pair try the (prev-1, prev) pair next, and so on, with
    # a per-hop correlation discount so the apply gate stays honest.
    stack_tables = [res_day]
    # Deep-lag coverage layer (2026-10-18, tmp/g2day.py +
    # multi_orbit.py): merge the validation day and day-2 tables
    # UNSHIFTED.  Their epochs sit on their own absolute axes, so a
    # per-sat lag of ~162200 s (GLONASS 4-orbit) or ~172300 s (GPS
    # 2 sidereal days) is served by the same u=sod-lag interp —
    # the query lands inside that day's own coverage.  Day-1
    # queries (lag ~86152/81000) never reach these ranges, so the
    # fresh template is unaffected.  This is coverage for deep
    # lags only, not the averaged-stack the --stack-days path
    # builds (which dilutes the fresh realisation — falsified
    # 2026-10-18 on tmp/persat_stack_ab.py).
    for _rk in (res_prev, _rp2, _rp3, _rp4, _rp5):
        if _rk:
            stack_tables.append(_rk)
    if args.stack_days > 1:
        from lib.sidereal_map import best_lags
        res_prevs = {1: res_prev} if res_prev else {}
        for k in range(2, args.stack_days):
            rk = day_residuals(day - dt.timedelta(days=k), f"stack k={k}")
            if rk:
                res_prevs[k] = rk
        # Validation fallback: sats missing from the freshest pair try
        # the next-older adjacent pair, corr discounted one hop.
        for k in sorted(res_prevs):
            if k + 1 not in res_prevs:
                continue
            sats_k = {s for dd in res_prevs[k].values() for s in dd}
            sats_k1 = {s for dd in res_prevs[k + 1].values() for s in dd}
            miss = [s for s in sats_k | sats_k1
                    if s not in lags or corrs.get(s, 0.0) < 0.35]
            if not miss:
                continue
            bl2 = best_lags(res_prevs[k + 1], res_prevs[k],
                            center=SIDEREAL_LAG,
                            half_span=args.lag_span)
            for s, (lg, c) in bl2.items():
                if s in miss:
                    lags[s] = float(lg)
                    corrs[s] = max(corrs.get(s, 0.0),
                                   float(c) * 0.7)
        # Shift each older table onto `day`'s axis (k hops) and merge.
        from lib.sidereal_map import build_map as _bm
        for k in sorted(res_prevs):
            rk = res_prevs[k]
            shifted = {}
            for u, dd in rk.items():
                for s, v in dd.items():
                    ls = lags.get(s, SIDEREAL_LAG)
                    shifted.setdefault(u + k * ls, {})[s] = v
            stack_tables.append(shifted)
            print(f"  stacked day-{k}: {len(shifted)} shifted epochs")

    from lib.sidereal_map import build_map
    tmp = out_p.with_suffix(".tmp")
    build_map(stack_tables, per_sat_lag=lags, corr=corrs, out_path=tmp,
              meta={"day": str(day), "built": dt.datetime.now(
                  dt.timezone.utc).isoformat(),
                  "n_epochs": len(res_day)})
    # Carry the same-day validation gate across the rebuild: live,
    # live_meta and live_hist are per-UTC-day MEASURED transfers, not
    # build artifacts.  Wiping them dropped the cross-day bridge and the
    # yesterday-gate for every window whose obs predate the next hourly
    # refresh — observed 2026-10-19 02:5x-03:3x apply logs: midnight-
    # straddlers served the build-corr fallback on their Oct-18 epochs
    # (fallback set G14/G22/G29/G31/R02/E33 + live G08/G23/R03 in one
    # window) because hist no longer held the Oct-18 gate.
    try:
        # Re-read the on-disk map, not the build-time snapshot: an
        # hourly refresh can land mid-build and its gate is fresher.
        _cur = old
        try:
            _cur = pickle.loads(out_p.read_bytes())
        except Exception:
            pass
        _doc = pickle.loads(tmp.read_bytes())
        for _gk in ("live", "live_meta", "live_hist"):
            _gv = (_cur or {}).get(_gk) or old.get(_gk)
            if _gv:
                _doc[_gk] = _gv
        tmp.write_bytes(pickle.dumps(_doc))
    except Exception:
        pass
    tmp.replace(out_p)
    print(f"wrote {out_p} sats={len((pickle.load(open(tmp,'rb')) if tmp.exists() else {}).get('sats',{})) if False else 'see map'}")

    # ---- sidereal PHASE map (sidereal_map_ph.pkl) --------------------
    # Per-sat carrier-phase deviations repeat at the same sky lags as
    # code residuals (measured rho 0.24-0.85 on the 2026-10-13/14
    # seeded pair, median +0.25; R16 0.85, G29 0.51).  The devs carry
    # the per-arc ambiguity constant, so the template is per-arc
    # demeaned (slip flags + 60 s gaps) — the AC component is what
    # subtracts; today's ambiguity absorbs the DC.
    try:
        import pickle as _pk
        from lib.sidereal_map import (phase_arcs_demean,
                                      series_transfer_score)
        ph_dir = RESID_DIR.parent / "sidereal_resid_ph"

        def _ph_day(dk):
            p = ph_dir / f"{dk.isoformat()}.pkl"
            if not p.exists():
                return {}, {}
            b = _pk.loads(p.read_bytes())
            t0 = b.get("truth") if isinstance(b, dict) else None
            if t0 is not None and float(np.linalg.norm(
                    np.asarray(t0) - truth)) >= 2.0:
                return {}, {}
            return (b.get("res", {}) if isinstance(b, dict) else b,
                    b.get("slips", {}) if isinstance(b, dict) else {})

        rd, sd = _ph_day(day)
        rp_, sp_ = _ph_day(prev)
        rp2_, sp2_ = _ph_day(day - dt.timedelta(days=2))
        rp3_, sp3_ = _ph_day(day - dt.timedelta(days=3))
        rp4_, sp4_ = _ph_day(day - dt.timedelta(days=4))
        rp5_, sp5_ = _ph_day(day - dt.timedelta(days=5))
        rp6_, sp6_ = _ph_day(day - dt.timedelta(days=6))
        if rd:
            td = phase_arcs_demean(rd, sd)
            tp_ = phase_arcs_demean(rp_, sp_) if rp_ else {}
            tp2 = phase_arcs_demean(rp2_, sp2_) if rp2_ else {}
            tp3 = phase_arcs_demean(rp3_, sp3_) if rp3_ else {}
            tp4 = phase_arcs_demean(rp4_, sp4_) if rp4_ else {}
            tp5 = phase_arcs_demean(rp5_, sp5_) if rp5_ else {}
            tp6 = phase_arcs_demean(rp6_, sp6_) if rp6_ else {}
            # Deep-lag coverage layer (mirrors the code-side unshifted
            # merge, 2026-10-18): the prev + day-2 demeaned arc tables
            # sit on their own absolute axes, so a per-sat phase lag of
            # ~162200 s (GLONASS 4-orbit) or ~172300 s (GPS 2 sidereal
            # days) is served by the same u=sod-lag interp.  Phase echo
            # verified at deep lags on the 18<-16 pair (tmp/ph_deep.py:
            # R19 +0.67, R24 +0.63, R09 +0.55, R18 +0.50 @162k; G26
            # +0.43, G12 +0.30 @172k).  Without the merge every deep-
            # lag phase sat was dead — corr=nan AND zero coverage.
            # tp3/tp4 join the merge (2026-10-18, tmp/ph_deep3.py):
            # the phase echo extends to 3 sidereal days (mean +0.27,
            # R21 +0.76, R18 +0.70) and 4 (+0.19), plus a strong
            # R-band at ~242-243k s (6 GLONASS orbits).  Validation of
            # a 3/4-day centre needs template coverage on day X-3/X-4;
            # apply of a 4-day phase lag queries day X-3 (tp3).
            # tp5/tp6 extend the same scheme to the 5/6-day + R 8/9-orbit
            # centres (code-side deep5_screen: 5d +0.39, R9orb +0.30).
            hist = {}
            for tab in (tp6, tp5, tp4, tp3, tp_, tp2):
                for s, (hu, hr) in tab.items():
                    hist.setdefault(s, []).append(
                        (hu, hr))
            hist = {s: (np.concatenate([a[0] for a in prs]),
                        np.concatenate([a[1] for a in prs]))
                    for s, prs in hist.items()}
            for s in hist:
                o = np.argsort(hist[s][0])
                hist[s] = (hist[s][0][o], hist[s][1][o])
            ph_lags, ph_corr = {}, {}
            # Per-family lag centres for phase validation (mirrors the
            # code-side multi-centre scan): the phase echo can peak at a
            # different family centre than the code-chosen lag — e.g.
            # R19's code lag is 80260 but its phase echo measured
            # +0.67 @162k vs +0.41 @81k on the 18<-16 pair
            # (tmp/ph_deep.py).  The code lag remains the fallback when
            # no centre scores (keeps the live-gate rescan anchored).
            _ph_centres = {"G": [(SIDEREAL_LAG, args.lag_span),
                                 (2 * SIDEREAL_LAG, args.lag_span),
                                 (3 * SIDEREAL_LAG, args.lag_span),
                                 (4 * SIDEREAL_LAG, args.lag_span),
                                 (5 * SIDEREAL_LAG, args.lag_span),
                                 (6 * SIDEREAL_LAG, args.lag_span)],
                           "E": [(SIDEREAL_LAG, args.lag_span),
                                 (2 * SIDEREAL_LAG, args.lag_span),
                                 (3 * SIDEREAL_LAG, args.lag_span),
                                 (4 * SIDEREAL_LAG, args.lag_span),
                                 (5 * SIDEREAL_LAG, args.lag_span),
                                 (6 * SIDEREAL_LAG, args.lag_span)],
                           "R": [(SIDEREAL_LAG, args.lag_span),
                                 (GLO_LAG, GLO_LAG_SPAN),
                                 (2 * GLO_LAG, GLO_LAG_SPAN),
                                 (3 * GLO_LAG, 1.5 * GLO_LAG_SPAN),
                                 (4 * GLO_LAG, 1.5 * GLO_LAG_SPAN),
                                 (4.5 * GLO_LAG, 1.5 * GLO_LAG_SPAN),
                                 (5 * GLO_LAG, 1.5 * GLO_LAG_SPAN)]}
            for s, (ut, rr) in td.items():
                ph_lags[s] = float(lags.get(s, SIDEREAL_LAG))
                hpair = hist.get(s)
                if hpair is None:
                    continue
                uy, vy = hpair
                for cen, hsp in _ph_centres.get(
                        s[:1], [(SIDEREAL_LAG, args.lag_span)]):
                    sc_ = series_transfer_score(
                        uy, vy, ut, rr, cen, rescan=hsp)
                    if sc_ is not None and \
                            sc_[0] > ph_corr.get(s, -1.0):
                        ph_corr[s], _, _, ph_lags[s] = sc_
            merged = dict(hist)
            for s, (ut, rr) in td.items():
                if s in merged:
                    hu, hr = merged[s]
                    merged[s] = (np.concatenate([hu, ut]),
                                 np.concatenate([hr, rr]))
                    o = np.argsort(merged[s][0])
                    merged[s] = (merged[s][0][o], merged[s][1][o])
                else:
                    merged[s] = (ut, rr)
            doc = {"sats": {s: {"u": u, "r": r} for s, (u, r)
                            in merged.items()},
                   "lag": ph_lags, "corr": ph_corr,
                   "meta": {"day": str(day), "kind": "phase",
                            "built": dt.datetime.now(
                                dt.timezone.utc).isoformat()}}
            outp = out_p.parent / "sidereal_map_ph.pkl"
            # Same gate carry-over as the code map: the phase channel is
            # live-only (no build-corr fallback), so a wiped live_hist
            # silences phase corrections until the next hourly refresh.
            try:
                _old_ph = _pk.loads(outp.read_bytes()) \
                    if outp.exists() else {}
            except Exception:
                _old_ph = {}
            for _gk in ("live", "live_meta", "live_hist"):
                if _old_ph.get(_gk):
                    doc[_gk] = _old_ph[_gk]
            tmp = outp.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(doc))
            tmp.replace(outp)
            print(f"wrote {outp} sats={len(doc['sats'])} "
                  f"(pair-validated {len(ph_corr)})")
    except Exception as e:
        print(f"phase map build skipped: {e}")


if __name__ == "__main__":
    main()
