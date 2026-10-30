#!/usr/bin/env python3
"""Sidereal per-satellite residual map.

The repeatable component of per-satellite code error (multipath +
product residuals relative to the epoch sky-median) recurs after ~1
sidereal day at per-satellite lags clustered around 86164 s.  A map of
yesterday's residuals can therefore be subtracted from today's
pseudoranges — a truth-free, fully causal correction that needs only
the receiver's own archived observations and precise products.

Validated 2026-10-13 on COD RAP products (unconfounded by ULT
prediction error):

  Oct-2 -> Oct-3 (n=7496): pooled demeaned corr +0.071 at 86164 s, but
      per-sat peaks: G15 +0.95@86284, G12 +0.83@86164, G28 +0.66@87364,
      G22 +0.56@85324, G23 +0.50@86284, G05/G08/G20/G25 ~+0.4-0.5.
  Oct-3 -> Oct-4 (n=545, 2.4h): pooled demeaned corr +0.558, nearly all
      tracked GPS sats +0.4-0.8 (G03 +0.82, G28 +0.77, G02 +0.72,
      G08 +0.67, G19 +0.65, G01 +0.64).
  Replay A/B (Oct-2 map on Oct-3 windows): free 105.7 -> 85.0 m
      (w1791046234), 0.304 -> 0.231 m (w1791068936) — +20-24% with zero
      harm on both.

Residual definition (per epoch, per sat):
    r_i = pr_IF_i - |satpos_i - truth| - median_j(pr_IF_j - |satpos_j - truth|)
i.e. the per-sat deviation from the epoch sky-median.  The median
removes receiver clock + shared common-mode, leaving ISB + multipath +
per-sat product error — the component the k=4 position has to fit.
"""
import numpy as np
from pathlib import Path


def residuals_from_measurements(raw_measurements, truth_ecef):
    """{gps_abs_s: {sat: residual}} per-epoch-median-subtracted code
    residuals.  Keys are `rm["sod"]` — absolute GPS seconds
    (week*604800 + SOW), monotonic across midnight and week boundaries;
    the map and the apply path share this axis directly."""
    truth = np.asarray(truth_ecef, float)
    out = {}
    for rm in raw_measurements:
        t = rm.get("sod")
        prs, sps = rm.get("pseudoranges"), rm.get("sat_positions")
        if t is None or not prs or not sps:
            continue
        dev = {}
        for sid, pr in prs.items():
            sp = sps.get(sid)
            if sp is None:
                continue
            dev[sid] = float(pr) - float(np.linalg.norm(
                np.asarray(sp, float) - truth))
        if len(dev) >= 4:
            med = float(np.median(list(dev.values())))
            out[float(t)] = {s: v - med for s, v in dev.items()}
    return out


def phase_dev_from_measurements(raw_measurements, truth_ecef):
    """({gps_abs: {sat: dev_m}}, {gps_abs: {slipped_sats}}) — the PHASE
    analogue of residuals_from_measurements.  dev = ph_m -
    |satpos - truth| - epoch_median, so it carries the per-sat ambiguity
    constant + ISB + phase multipath AC + product-error AC.  The
    ambiguity constant is removed downstream by per-arc demeaning (arc
    boundaries = gaps in the sampled table + these slip flags).
    """
    truth = np.asarray(truth_ecef, float)
    dev_tab, slip_tab = {}, {}
    for rm in raw_measurements:
        t = rm.get("sod")
        ph, sps = rm.get("phases"), rm.get("sat_positions")
        if t is None or not ph or not sps:
            continue
        dev = {}
        for sid, p in ph.items():
            sp = sps.get(sid)
            if sp is None or not np.isfinite(p):
                continue
            dev[sid] = float(p) - float(np.linalg.norm(
                np.asarray(sp, float) - truth))
        if len(dev) >= 4:
            med = float(np.median(list(dev.values())))
            dev_tab[float(t)] = {s: v - med for s, v in dev.items()}
            sl = rm.get("slips")
            if sl:
                slip_tab[float(t)] = set(sl)
    return dev_tab, slip_tab


def build_map(resid_tables, per_sat_lag=None, corr=None, out_path=None,
              meta=None):
    """Merge residual tables into {sats:{sat:{u,r}}, lag, corr, meta}."""
    acc = {}
    for tab in resid_tables:
        for t, d in tab.items():
            for s, v in d.items():
                acc.setdefault(s, []).append((t, v))
    m = {}
    for s, pairs in acc.items():
        pairs.sort()
        # Product-edge failures produce residuals of ~1e5-1e6 m (sat
        # clock/orbit lookups failing at spline boundaries).  Real
        # per-sat residuals are <=~50 m; drop anything beyond 300 m so
        # a polluted extraction can never ship a kilometre-scale
        # "correction".
        pairs = [(t, r) for t, r in pairs if abs(r) < 300.0]
        if len(pairs) < 100:
            continue
        m[s] = {"u": np.array([p[0] for p in pairs]),
                "r": np.array([p[1] for p in pairs])}
    doc = {"sats": m,
           "lag": per_sat_lag or {},
           "corr": corr or {},
           "meta": meta or {}}
    if out_path:
        import pickle
        Path(out_path).write_bytes(pickle.dumps(doc))
    return doc


def phase_arcs_demean(res, slips, gap_s=60.0, min_arc=30):
    """{sat: (u, r)} — split each sat's phase-dev series into arcs at
    gaps > gap_s or a stored slip flag, then remove each arc's mean
    (the ambiguity + ISB constant), leaving the AC component that a
    phase template predicts.  Returns only sats with >= min_arc points
    after demeaning."""
    import collections
    per_sat = collections.defaultdict(list)
    for g, dd in res.items():
        for s, v in dd.items():
            per_sat[s].append((g, v))
    slip_t = sorted(slips)
    slip_ep = {s: set() for s in per_sat}
    for g in slip_t:
        for s in slips[g]:
            if s in slip_ep:
                slip_ep[s].add(g)
    out = {}
    for s, pairs in per_sat.items():
        pairs.sort()
        seps = slip_ep[s]
        arcs, cur = [], [pairs[0]]
        for a, b in zip(pairs, pairs[1:]):
            if b[0] - a[0] > gap_s or b[0] in seps:
                arcs.append(cur)
                cur = []
            cur.append(b)
        arcs.append(cur)
        rr = []
        for arc in arcs:
            if len(arc) < min_arc:
                continue
            vals = np.array([v for _, v in arc], float)
            vals -= vals.mean()
            rr += [(t, float(v)) for (t, _), v in zip(arc, vals)]
        if len(rr) >= min_arc:
            u = np.array([t for t, _ in rr])
            out[s] = (u, np.array([v for _, v in rr]))
    return out


def series_transfer_score(ut, rr, gt, rv, lag, rescan=0.0, step=60.0,
                          min_pairs=100, max_gap_s=120.0):
    """Score how well template series (ut, rr) predicts today's residual
    rv at gt: pairs obs at u = t - lag inside contiguous template spans
    (<= max_gap_s), returns (rho, wiener_gain, n_pairs, best_lag) or
    None.  With rescan>0 the lag is refined in +-rescan s steps of
    `step`, switching only on a >0.05 corr improvement."""
    def _score(lag):
        u = gt - lag
        m = (u >= ut[0]) & (u <= ut[-1])
        j = np.searchsorted(ut, u)
        ok = m.copy()
        ok[m] = (j[m] > 0) & (
            (ut[np.minimum(j[m], len(ut) - 1)]
             - ut[np.maximum(j[m] - 1, 0)]) <= max_gap_s)
        if int(ok.sum()) < min_pairs:
            return None
        pred = np.interp(u[ok], ut, rr)
        obs = rv[ok]
        pv = pred - pred.mean(); ov = obs - obs.mean()
        if pv.std() < 1e-9 or ov.std() < 1e-9:
            return None
        rho = float(np.corrcoef(pv, ov)[0, 1])
        return rho, rho * float(ov.std() / pv.std()), int(ok.sum())
    best = _score(lag)
    best_lag = lag
    if rescan > 0:
        for dl in np.arange(-rescan, rescan + 1, step):
            if dl == 0.0:
                continue
            r = _score(lag + float(dl))
            if r is not None and (best is None or
                                  r[0] > best[0] + 0.05):
                best, best_lag = r, lag + float(dl)
    if best is None:
        return None
    return best[0], best[1], best[2], best_lag


def correlate_lag(tab_a, tab_b, lag, min_pairs=30):
    """Per-sat demeaned correlation between two residual tables at `lag`
    seconds (b_time = a_time + lag).  Returns {sat: (corr, n)}."""
    tb = np.array(sorted(tab_b))
    out = {}
    # collect per-sat matched arrays
    import collections
    pairs = collections.defaultdict(list)
    for ta, da in tab_a.items():
        tgt = ta + lag
        i = int(np.searchsorted(tb, tgt))
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(tb) and \
                    (best is None or abs(tb[j] - tgt) < abs(best - tgt)):
                best = tb[j]
        if best is None or abs(best - tgt) > 8:
            continue
        db = tab_b[best]
        for sid, va in da.items():
            vb = db.get(sid)
            if vb is not None:
                pairs[sid].append((va, vb))
    for sid, ps in pairs.items():
        if len(ps) < min_pairs:
            continue
        xa = np.array([p[0] for p in ps]); xb = np.array([p[1] for p in ps])
        xa = xa - xa.mean(); xb = xb - xb.mean()
        out[sid] = (float(np.corrcoef(xa, xb)[0, 1]), len(ps))
    return out


def best_lags(tab_a, tab_b, center=86164, half_span=900, step=120,
              min_corr=None):
    """Per-sat best-lag correlation scan around a sidereal centre.
    Returns {sat: (lag, corr)}."""
    lags = range(int(center - half_span), int(center + half_span) + 1,
                 step)
    best = {}
    for lag in lags:
        for sid, (c, n) in correlate_lag(tab_a, tab_b, lag).items():
            if sid not in best or c > best[sid][1]:
                best[sid] = (lag, c)
    if min_corr is not None:
        best = {s: v for s, v in best.items() if v[1] >= min_corr}
    return best
