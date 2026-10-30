#!/usr/bin/env python3
"""Apply the sidereal-residual map to a RINEX 3 obs file.

Subtracts yesterday's per-satellite IF-code residual template at each
satellite's sidereal lag from the C1C and C5Q code fields of every
epoch record.  Equal subtraction on both members shifts the
ionosphere-free code by exactly the correction.

Replicates the gate semantics of gps_web._apply_sidereal_correction:
  - live (same-day-validated) entry: corr >= TEP_SIDEREAL_LIVE_MIN_CORR
    (0.25), gain = clip(live gain, 0, 1.2), lag from live entry
  - else build-gate: corr >= TEP_SIDEREAL_MIN_CORR (0.55), gain 1.0
  - interpolation only inside <=120 s segments of the map axis
  - |corr| > 300 m rejected as map pollution

Only GPS and Galileo records are touched (the AR constellation set);
GLONASS/BeiDou records pass through unchanged.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

GPS_EPOCH = datetime(1980, 1, 6)

# obs-code field indices within the 16-char RINEX fields, by obs order
# in our convbin output:  G/E = [C1C, L1C, C5Q, L5Q], R = [C1C, L1C]
_CODE_FIELDS = {"G": (0, 2), "E": (0, 2), "R": (0,)}
# fields to correct: C1C (idx0) and C5Q (idx2); all other codes stay.


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _gps_abs(year: int, mon: int, day: int,
             hh: int, mm: int, ss: float) -> float:
    dt = datetime(year, mon, day, hh, mm) - GPS_EPOCH
    # +18: the map/resid axis is GPS timescale seconds (week*604800+sow),
    # 18 s ahead of the UTC-anchored offset (2026-10-19 — the legacy
    # convention here read the template 18 s early).
    return dt.total_seconds() + ss + 18.0


def _live_deltas(obs_in: Path) -> dict:
    """{sod: {sid: applied_delta}} computed by the DEPLOYED apply path
    (gps_web._apply_sidereal_correction): ridge jt basis, winsorised+
    smoothed template, per-epoch day keying, xday bridge, corr_cap —
    no parallel semantics to drift out of sync."""
    import sys as _sys
    _root = Path(__file__).resolve().parents[2]
    for _p in (str(_root / "rpi"), str(_root / "scripts")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    from gps_web import PPPProcessor  # noqa
    import logging
    # When this runs inside the live gps_web process the temporary
    # processor shares the "tep-web" logger — silence it only for the
    # duration of this call, then restore.  A bare setLevel(WARNING)
    # here permanently muted every tep-web INFO line for the rest of
    # the process lifetime (observed live 2026-10-19: ~3 h of missing
    # EKF/PPP-AR telemetry after the first obs rewrite).
    _lg = logging.getLogger("tep-web")
    _prev_level = _lg.level
    _lg.setLevel(logging.WARNING)
    # Bare instance, not __init__: _apply_sidereal_correction reads only
    # ref_ecef (None == the headless default) plus lazily-created _sid*
    # caches.  A full construction runs the data/ppp_live workdir flock —
    # which fails against the live process and logged a spurious
    # "another instance owns" WARNING into gps_web.log every window —
    # plus a fresh last_ppp_result.json load and tool discovery each call.
    proc = PPPProcessor.__new__(PPPProcessor)
    proc.ref_ecef = None
    epochs = []
    cur_t = None
    cur_sids: list = []
    with open(obs_in, "r", errors="ignore") as fi:
        in_header = True
        for line in fi:
            if in_header:
                if "END OF HEADER" in line:
                    in_header = False
                continue
            if line.startswith(">"):
                if cur_t is not None:
                    epochs.append((cur_t, cur_sids))
                p = line[1:].split()
                try:
                    cur_t = _gps_abs(int(p[0]), int(p[1]), int(p[2]),
                                     int(p[3]), int(p[4]), float(p[5]))
                except (ValueError, IndexError):
                    cur_t = None
                cur_sids = []
                continue
            if cur_t is not None and line[:3].strip() \
                    and line[0] != ' ':
                cur_sids.append(line[:3].strip())
        if cur_t is not None:
            epochs.append((cur_t, cur_sids))
    rms = [{"sod": t, "pseudoranges": {s: 0.0 for s in sl}}
           for t, sl in epochs]
    try:
        proc._apply_sidereal_correction(rms, Path(obs_in))
    finally:
        _lg.setLevel(_prev_level)
    return {rm["sod"]: dict(rm.get("_sid_corr") or {}) for rm in rms}


def _gated_sats(mp: dict, obs_day: str | None):
    """Return {sat: (gain, lag_s)} for sats that pass the apply gates."""
    sats = (mp or {}).get("sats") or {}
    lags = mp.get("lag") or {}
    corrs = mp.get("corr") or {}
    hist = mp.get("live_hist") or {}
    if obs_day and obs_day in hist:
        live = hist[obs_day]
    elif obs_day and (mp.get("live_meta") or {}).get("day") == obs_day:
        live = mp.get("live") or {}
    else:
        live = None
    out = {}
    live_min = _env_float("TEP_SIDEREAL_LIVE_MIN_CORR", 0.25)
    bld_min = _env_float("TEP_SIDEREAL_MIN_CORR", 0.55)
    for sid, ent in sats.items():
        if sid[0] not in _CODE_FIELDS:
            continue
        gain = 1.0
        lag_s = float(lags.get(sid, 86164.0))
        le = live.get(sid) if live is not None else None
        if le is not None:
            if float(le.get("corr", 0.0)) < live_min:
                continue
            gain = float(np.clip(le.get("gain", 1.0), 0.0, 1.2))
            lag_s = float(le.get("lag", lag_s))
        elif float(corrs.get(sid, 0.0)) < bld_min:
            continue
        out[sid] = (gain, lag_s, ent)
    return out


def apply_sidereal_code_correction(obs_in: Path, obs_out: Path,
                                   mp_path: Path,
                                   obs_day: str | None = None,
                                   demean: bool = True,
                                   live: bool = True) -> dict:
    """Write obs_out = obs_in with sidereal code corrections applied.

    live=True (default, 2026-10-19) computes the per-epoch deltas via
    the deployed apply path — ridge jt basis, winsorised/smoothed
    template, per-epoch day keying, xday G bridge, corr_cap — so the
    PPP-AR-observed corrections track the live EKF exactly.  Falls
    back to the legacy single-lag path only when the live path is
    unavailable.

    demean=True removes each satellite's own applied-delta mean over the
    file before writing — the constant per-sat part (receiver DCB) is
    already handled by the PPP-AR ISB table, so only the time-varying
    multipath component should be subtracted here; otherwise the ISB
    and the map would double-count the same bias.

    Returns stats {n_corr, sats_hit, applied_delta_rms_m}.
    """
    live_delta = None
    if live:
        try:
            live_delta = _live_deltas(obs_in)
        except Exception:
            live_delta = None
    try:
        mp = pickle.load(open(mp_path, "rb"))
    except Exception:
        mp = {}
    corrs = (mp or {}).get("corr") or {}
    lags_w = (mp or {}).get("lag") or {}
    sats0 = (mp or {}).get("sats") or {}
    if live_delta is not None:
        # Wide tier (2026-10-19): sats that fail the strict live gate
        # but pass a looser build-corr floor still get a Wiener-scaled
        # single-lag correction.  The PPP-AR float is geometry-limited
        # — breadth dominates per-sat fidelity (w1792367828 + w1792369
        # 072 A/B: the strict 12-sat set moved nothing while the
        # ~32-sat corr>=0.35 set cut median err ~45-55%).
        #
        # Hardening (2026-10-19 later): two protections the wide tier
        # previously skipped.
        #  a) A sat VETOED by today's live gate (entry present but
        #     corr < live-min — forward-check anti-transfer or rho
        #     floor) must NOT be served the stale build template: the
        #     gate measured it harmful TODAY.  Only sats absent from
        #     today's gate entirely (never measured) remain eligible.
        #  b) |delta| capped at TEP_SID_OBSW_MAX_M (default 50 m — the
        #     fallback arm's template-quality bound): the wide tier
        #     interpolates the raw residual axis, which carries
        #     eclipse/maneuver-edge pollution up to the +-300 m clip
        #     (G13 p95|r|=225 m documented); a 300 m clip is not a
        #     quality gate.
        _wmin = _env_float("TEP_SID_OBSW_MIN_CORR", 0.35)
        _wgain = _env_float("TEP_SID_OBSW_GAIN", 1.0)
        _wmax = _env_float("TEP_SID_OBSW_MAX_M", 50.0)
        # TEP_SID_OBSW_GAIN_MODE=corr: per-sat Wiener-style gain
        # (gain = build corr) instead of flat — shrinks correction on
        # weakly-transferring sats rather than trusting the template
        # at unit amplitude.
        _wgmode = os.environ.get("TEP_SID_OBSW_GAIN_MODE", "flat")
        _hist = mp.get("live_hist") or {}
        _lm = mp.get("live_meta") or {}
        _lv = (_hist.get(_lm.get("day"))
               or (mp.get("live") or {}))
        _live_min = _env_float("TEP_SIDEREAL_LIVE_MIN_CORR", 0.25)
        _vetoed = {s for s, e in _lv.items()
                   if float(e.get("corr", 0.0)) < _live_min}
        wide = {s: (float(np.clip(
                        _wgain if _wgmode != "corr"
                        else float(corrs.get(s, 0.0)), 0.0, 1.2)),
                    float(lags_w.get(s, 86164.0)), ent)
                for s, ent in sats0.items()
                if s[:1] in _CODE_FIELDS
                and s not in _vetoed
                and float(corrs.get(s, 0.0)) >= _wmin}

        def _delta(sid, cur_t):
            e = live_delta.get(cur_t)
            if e is not None and sid in e:
                return e[sid]
            w = wide.get(sid)
            if w is None:
                return None
            gain, lag_s, ent = w
            u = cur_t - lag_s
            ut, rr = ent["u"], ent["r"]
            if u < ut[0] or u > ut[-1]:
                return None
            j = int(np.searchsorted(ut, u))
            if j == 0 or float(ut[j] - ut[j - 1]) > 120.0:
                return None
            corr = float(np.interp(u, ut, rr))
            if abs(corr) > _wmax:
                return None
            return corr * gain
    else:
        gated = _gated_sats(mp, obs_day)
        if not gated:
            return {"n_corr": 0, "error": "no gated sats"}

        def _delta(sid, cur_t):
            gain, lag_s, ent = gated[sid]
            u = cur_t - lag_s
            ut, rr = ent["u"], ent["r"]
            if u < ut[0] or u > ut[-1]:
                return None
            j = int(np.searchsorted(ut, u))
            if j == 0 or float(ut[j] - ut[j - 1]) > 120.0:
                return None
            corr = float(np.interp(u, ut, rr))
            if abs(corr) > 300.0:
                return None
            return corr * gain

    # pass 1: collect per-sat applied deltas for the demean
    sat_mean = {}
    if demean:
        acc = {}
        cur_t = None
        with open(obs_in, "r", errors="ignore") as fi:
            in_header = True
            for line in fi:
                if in_header:
                    if "END OF HEADER" in line:
                        in_header = False
                    continue
                if line.startswith(">"):
                    p = line[1:].split()
                    try:
                        cur_t = _gps_abs(int(p[0]), int(p[1]), int(p[2]),
                                         int(p[3]), int(p[4]),
                                         float(p[5]))
                    except (ValueError, IndexError):
                        cur_t = None
                    continue
                if len(line) < 4 or cur_t is None or line[0] == ' ':
                    continue
                sid = line[:3].strip()
                if (live_delta is not None) or (sid in gated):
                    d = _delta(sid, cur_t)
                    if d is not None:
                        acc.setdefault(sid, []).append(d)
        sat_mean = {s: float(np.mean(v)) for s, v in acc.items()}

    n_corr = 0
    hits = set()
    deltas = []
    cur_t = None
    with open(obs_in, "r", errors="ignore") as fi, \
            open(obs_out, "w") as fo:
        in_header = True
        for line in fi:
            if in_header:
                fo.write(line)
                if "END OF HEADER" in line:
                    in_header = False
                continue
            if line.startswith(">"):
                p = line[1:].split()
                try:
                    cur_t = _gps_abs(int(p[0]), int(p[1]), int(p[2]),
                                     int(p[3]), int(p[4]), float(p[5]))
                except (ValueError, IndexError):
                    cur_t = None
                fo.write(line)
                continue
            if len(line) < 4 or cur_t is None or line[0] == ' ':
                fo.write(line)
                continue
            sid = line[:3].strip()
            if (live_delta is None) and (sid not in gated):
                fo.write(line)
                continue
            d0 = _delta(sid, cur_t)
            if d0 is None:
                fo.write(line)
                continue
            delta = d0 - sat_mean.get(sid, 0.0)
            # rewrite the code fields: 16-char fields starting col 3
            fields = [line[3 + 16 * k: 3 + 16 * (k + 1)]
                      for k in range((len(line.rstrip("\n")) - 3) // 16 + 1)]
            changed = False
            for fi_idx in _CODE_FIELDS.get(sid[0], ()):
                if fi_idx >= len(fields):
                    continue
                fld = fields[fi_idx]
                if len(fld) < 14:
                    fld = fld.ljust(16)
                try:
                    v = float(fld[:14])
                except ValueError:
                    continue
                nv = v - delta
                flags = fld[14:16] if len(fld) > 14 else "  "
                fields[fi_idx] = f"{nv:14.3f}" + flags
                changed = True
            if changed:
                body = "".join(f.ljust(16)[:16] for f in fields)
                fo.write(line[:3] + body[:len(line.rstrip("\n")) - 3]
                         .ljust(len(line.rstrip("\n")) - 3) + "\n")
                n_corr += 1
                hits.add(sid)
                deltas.append(delta)
            else:
                fo.write(line)
    rms = float(np.sqrt(np.mean(np.square(deltas)))) if deltas else 0.0
    return {"n_corr": n_corr, "sats_hit": len(hits),
            "applied_delta_rms_m": rms,
            "sats": sorted(hits)}
