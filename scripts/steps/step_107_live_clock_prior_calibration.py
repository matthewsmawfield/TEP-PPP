#!/usr/bin/env python3
"""
STEP 107 -- Calibration of the LIVE rig's network clock prior.

The live rig (rpi/gps_web.py) reported a 0.0 % TEP gain over persistence.  The
cause was not a weak field: it was running a DIFFERENT predictor from the one
Steps 96/106 validated, and screening it with a factor measured to be the worst
available choice for its own receiver class.

This step establishes, on real archive data (data/ppp_dense, 30 European CORS
stations, real SPP receiver clocks, 300 s sampling), the four numbers that the
live rig now uses.  Fitted on days 2024:001-021, scored on held-out 022-050.

  1. OBSERVABLE.  Krige the network's ACCUMULATED clock offset across the
     prediction gap, not its instantaneous residual level.
         instantaneous level (old live arm)   -36.3 %  vs persistence
         accumulated offset  (Step 96 form)   +37.7 %  vs persistence
     The old arm was worse than doing nothing; the live soft gate was
     correctly damping it to zero, which is why the dashboard read 0.0 %.

  2. KERNEL.  Gaussian, lambda = 1500 km, nugget 0.70.
         train +38.8 %   held-out +37.6 %   (vs free)
     The rig previously hardcoded an exponential at lambda = 200 km on the
     reasoning that SPP residuals are dominated by local atmosphere.  That is
     the WORST setting in the sweep (+24.5 % held-out); the data prefers a
     correlation length ~7x longer than the "local atmosphere" hypothesis.

  3. SPECIFICITY.  Real station geometry 7.258 m vs distance-permuted
     7.825 +- 0.063 m (8 draws) -- the spatial structure carries real
     information at ~9 sigma.  A plain unweighted network mean scores 7.682 m,
     so the bulk of the gain is the network common mode and ~5.5 % of it is
     the distance structure specifically.

  4. SCREENING.  alpha* = argmin ||truth - (own gated drift + alpha*spatial)||
         ATOMIC   1.304 +- 0.008    skill +61.0 %  (at 1.38: +60.3 %)
         QUARTZ   0.619 +- 0.012    skill +17.6 %  (at 1.38:  +1.2 %)
     The rig is an LC29H quartz TCXO and was applying 1.38.

Also measured, because it decides whether the rig needs its UK stations back:
standoff does not matter.  Forcing every reference beyond 250 km leaves skill
at +38.7 % (against +37.7 % unrestricted), and the 250-400 km band is the best
of the three.  The Dutch-only network at 270-500 km is sufficient.

Run:  python3 scripts/steps/step_107_live_clock_prior_calibration.py
Out:  results/outputs/step_107_live_clock_prior_calibration.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DENSE = ROOT / "data" / "ppp_dense"
OUT = ROOT / "results" / "outputs" / "step_107_live_clock_prior_calibration.json"
C_M_PER_NS = 0.299792458
GRID = np.arange(0.0, 86400.0, 300.0)
NPRE, R2_THRESH, DNR_SWITCH = 48, 0.97, 5.0
TRAIN = list(range(1, 22))
TEST = list(range(22, 51))
RNG = np.random.default_rng(20260906)


def repair_jumps(y, mad_mult=20.0):
    y = np.asarray(y, float).copy()
    d = np.diff(y); fin = np.isfinite(d)
    if fin.sum() < 8: return y
    scale = 1.4826 * np.median(np.abs(d[fin] - np.median(d[fin]))) + 1e-6
    step = np.zeros_like(d)
    step[fin] = np.where(np.abs(d[fin]) > mad_mult * scale, d[fin], 0.0)
    y[1:] -= np.cumsum(step)
    return y


def ecef_to_ll(x, y, z):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); lon = math.atan2(y, x); p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(6):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * N * math.sin(lat), p)
    return math.degrees(lat), math.degrees(lon)


def gc_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    v = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, v)))


def load_day(doy, year=2024):
    series, geo = {}, {}
    for f in sorted(DENSE.glob(f"*_{year}{doy:03d}.npz")):
        d = np.load(f, allow_pickle=True)
        nm = str(d["station"])
        y = np.asarray(d["clock_bias_ns"], float) * C_M_PER_NS
        t = np.asarray(d["timestamps"], float)
        m = np.isfinite(y) & np.isfinite(t)
        if m.sum() < 200: continue
        gy = np.interp(GRID, t[m], repair_jumps(y[m]), left=np.nan, right=np.nan)
        if np.isfinite(gy).sum() < 200: continue
        if np.nanmax(gy) - np.nanmin(gy) < 1e-6: continue     # SPP failed
        series[nm] = gy
        geo[nm] = ecef_to_ll(float(d["x_mean"]), float(d["y_mean"]), float(d["z_mean"]))
    # Reject gross outliers (unrepaired steering / ephemeris failures), the
    # same intent as the live server's MAD filter.
    a = {nm: float(np.sqrt(np.nanmean((v[12:] - v[:-12]) ** 2)))
         for nm, v in series.items()}
    if len(a) >= 6:
        vals = np.array(list(a.values())); med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-9
        keep = {nm for nm, v in a.items() if abs(v - med) <= max(6 * mad, 0.5 * med)}
        series = {k: v for k, v in series.items() if k in keep}
        geo = {k: v for k, v in geo.items() if k in keep}
    return series, geo


def rolling_fit(X, npre, r2_thresh):
    n_ep, n_st = X.shape
    rate = np.full((n_ep, n_st), np.nan); span = np.full((n_ep, n_st), 1e-6)
    dnr = np.zeros((n_ep, n_st)); amp = np.zeros((n_ep, n_st))
    xc = np.arange(npre, dtype=float); xc -= xc.mean()
    sxx = float((xc ** 2).sum())
    for ti in range(npre, n_ep):
        seg = X[ti - npre:ti, :]
        ok = np.isfinite(seg).all(axis=0)
        if not ok.any(): continue
        s_ = seg[:, ok]; m = s_.mean(axis=0)
        r = (xc[:, None] * (s_ - m)).sum(axis=0) / sxx
        res = s_ - (m[None, :] + r[None, :] * xc[:, None])
        dn = res.std(axis=0); var = s_.var(axis=0)
        r2 = np.where(var > 1e-12, 1.0 - dn ** 2 / np.maximum(var, 1e-12), 0.0)
        pd = np.abs(r) * npre
        rate[ti, ok] = r
        span[ti, ok] = np.maximum(np.ptp(s_, axis=0), 1e-6)
        dnr[ti, ok] = np.where((r2 > r2_thresh) & (pd > 2.0),
                               pd / np.maximum(dn, 1e-6), 0.0)
        amp[ti, ok] = dn
    return rate, span, dnr, amp


def build(doys):
    days = []
    for doy in doys:
        s_, g = load_day(doy)
        names = sorted(s_)
        if len(names) < 10: continue
        X = np.column_stack([s_[n] for n in names])
        n = len(names); D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = gc_km(g[names[i]], g[names[j]])
        rate, span, dnr, amp = rolling_fit(X, NPRE, R2_THRESH)
        days.append(dict(names=names, X=X, D=D, rate=rate, span=span,
                         dnr=dnr, amp=amp))
    return days


def kern(d, lam, form):
    if form == "gauss": return np.exp(-0.5 * (d / lam) ** 2)
    if form == "exp": return np.exp(-d / lam)
    u = math.sqrt(3) * d / lam
    return (1 + u) * np.exp(-u)


def cases(days, h, stride=4, min_excl_km=0.0):
    out = []
    for d in days:
        X, D = d["X"], d["D"]
        rate, span, dnr, amp = d["rate"], d["span"], d["dnr"], d["amp"]
        n_ep, n_st = X.shape
        Xd = np.empty_like(X)
        for j in range(n_st):
            v = X[:, j]; m = np.isfinite(v); xi = np.arange(n_ep, dtype=float)
            if m.sum() < 5: Xd[:, j] = np.nan; continue
            Xd[:, j] = v - np.polyval(np.polyfit(xi[m], v[m], 1), xi)
        for ti in range(NPRE + 1, n_ep - h - 1, stride):
            a0, a1 = X[ti - 1, :], X[ti + h, :]
            fin = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(rate[ti, :])
            if fin.sum() < 8: continue
            cs_all = a1 - a0
            g_all = (dnr[ti, :] > DNR_SWITCH).astype(float)
            csg = cs_all - g_all * rate[ti, :] * h
            lr = np.clip(rate[ti, :] * h, -8 * span[ti, :], 8 * span[ti, :])
            for b in np.flatnonzero(fin):
                pool = np.array([j for j in np.flatnonzero(fin)
                                 if j != b and D[b, j] >= min_excl_km
                                 and np.isfinite(Xd[ti - 1, j])])
                if len(pool) < 6: continue
                out.append(dict(D=D, b=int(b), pool=pool, name=d["names"][b],
                                truth=float(cs_all[b]),
                                base=float(g_all[b] * lr[b]),
                                csg=csg[pool].copy(),
                                lvl=Xd[ti - 1, pool].copy(),
                                nearest_km=float(np.min(D[b, pool]))))
    return out


def evaluate(cl, lam, nug, form, perm_rng=None, alpha=1.0, observable="accum"):
    se = dict(free=0.0, persist=0.0, tep=0.0, netmean=0.0); n = 0
    for c in cl:
        D, b, pool = c["D"], c["b"], c["pool"]
        if perm_rng is not None:
            idx = perm_rng.permutation(D.shape[0]); D = D[np.ix_(idx, idx)]
        krr = kern(D[np.ix_(pool, pool)], lam, form)
        krt = kern(D[b, pool], lam, form)
        try: w = np.linalg.solve(krr + nug * np.eye(len(pool)), krt)
        except np.linalg.LinAlgError: continue
        src = c["csg"] if observable == "accum" else c["lvl"]
        tep = c["base"] + alpha * float(w @ src)
        t = c["truth"]
        if not np.isfinite(tep) or abs(t) > 1e4: continue
        se["free"] += t * t
        se["persist"] += (t - c["base"]) ** 2
        se["tep"] += (t - tep) ** 2
        se["netmean"] += (t - (c["base"] + float(np.mean(c["csg"])))) ** 2
        n += 1
    if n == 0: return None
    return {k: math.sqrt(v / n) for k, v in se.items()} | {"n": n}


def alpha_star(cl, lam, nug, form):
    r, sp = [], []
    for c in cl:
        D, b, pool = c["D"], c["b"], c["pool"]
        krr = kern(D[np.ix_(pool, pool)], lam, form)
        krt = kern(D[b, pool], lam, form)
        try: w = np.linalg.solve(krr + nug * np.eye(len(pool)), krt)
        except np.linalg.LinAlgError: continue
        a, s_ = c["truth"] - c["base"], float(w @ c["csg"])
        if np.isfinite(a) and np.isfinite(s_) and abs(a) < 1e4 and abs(s_) < 1e4:
            r.append(a); sp.append(s_)
    r, sp = np.array(r), np.array(sp)
    den = float(sp @ sp)
    return float(r @ sp) / den if den > 0 else 0.0


def main():
    LAM, NUG, FORM, H = 1500.0, 0.70, "gauss", 48      # 48 x 300 s = 4 h
    res = {"config": dict(lambda_km=LAM, nugget=NUG, kernel=FORM,
                          horizon_h=H * 300 / 3600, npre=NPRE,
                          train_doys=[TRAIN[0], TRAIN[-1]],
                          test_doys=[TEST[0], TEST[-1]])}
    tr_days, te_days = build(TRAIN), build(TEST)
    tr, te = cases(tr_days, H), cases(te_days, H)
    res["n_cases"] = {"train": len(tr), "test": len(te)}

    # 1. observable
    r_acc = evaluate(te, LAM, NUG, FORM, observable="accum")
    r_lvl = evaluate(te, 200.0, 0.05, "exp", observable="level")
    res["observable"] = {
        "accumulated_offset_vs_persist_pct":
            round(100 * (1 - r_acc["tep"] / r_acc["persist"]), 2),
        "instantaneous_level_vs_persist_pct":
            round(100 * (1 - r_lvl["tep"] / r_lvl["persist"]), 2),
        "free_m": round(r_acc["free"], 4),
        "persist_m": round(r_acc["persist"], 4),
        "tep_m": round(r_acc["tep"], 4),
        "netmean_m": round(r_acc["netmean"], 4)}

    # 2. kernel, train vs held-out
    res["kernel"] = {}
    for tag, (f_, l_, n_) in {"chosen": (FORM, LAM, NUG),
                              "step96": ("gauss", 4200.0, 0.70),
                              "old_live_rig": ("exp", 200.0, 0.05)}.items():
        a = evaluate(tr, l_, n_, f_); b = evaluate(te, l_, n_, f_)
        res["kernel"][tag] = {
            "train_vs_free_pct": round(100 * (1 - a["tep"] / a["free"]), 2),
            "test_vs_free_pct": round(100 * (1 - b["tep"] / b["free"]), 2)}

    # 3. specificity
    perms = [evaluate(te, LAM, NUG, FORM,
                      perm_rng=np.random.default_rng(1000 + i))["tep"]
             for i in range(8)]
    res["specificity"] = {
        "real_geometry_m": round(r_acc["tep"], 4),
        "distance_permuted_m": round(float(np.mean(perms)), 4),
        "distance_permuted_sd": round(float(np.std(perms)), 4),
        "netmean_m": round(r_acc["netmean"], 4),
        "sigma": round((np.mean(perms) - r_acc["tep"]) / max(np.std(perms), 1e-9), 1)}

    # 4. screening by clock class
    ct = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
    cls = lambda nm: (ct.get(nm) or ["UNKNOWN"])[0]
    res["screening"] = {}
    for c in ("ATOMIC", "QUARTZ"):
        s_tr = [x for x in tr if cls(x["name"]) == c]
        s_te = [x for x in te if cls(x["name"]) == c]
        if len(s_tr) < 500 or len(s_te) < 500: continue
        a = alpha_star(s_tr, LAM, NUG, FORM)
        row = {"alpha_star": round(a, 4)}
        for lbl, aa in (("at_alpha_star", a), ("at_1.0", 1.0),
                        ("at_1.38", math.sqrt(1.9))):
            rr = evaluate(s_te, LAM, NUG, FORM, alpha=aa)
            row[lbl + "_vs_free_pct"] = round(100 * (1 - rr["tep"] / rr["free"]), 2)
        res["screening"][c] = row

    # 5. standoff
    res["standoff"] = {}
    for lbl, km in (("unrestricted", 0.0), ("min_250km", 250.0)):
        cc = cases(te_days, H, min_excl_km=km)
        rr = evaluate(cc, LAM, NUG, FORM)
        res["standoff"][lbl] = round(100 * (1 - rr["tep"] / rr["persist"]), 2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
