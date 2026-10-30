#!/usr/bin/env python3
"""
STEP 105  --  Moving-platform clock bridging on a FREE-RUNNING space clock
              (GRACE-FO USO).

Step 104 (Swarm-A) returned a negative, but the Swarm receiver clock is
STEERED (raw-SPP reconstruction: 0.7 m bounded, range ~15 m/day, corr 0.87
with the product) -- steering removes the physical proper-time signal, the
same way a GPS-disciplined ground rubidium does.

GRACE-FO carries an Ultra-Stable Oscillator that is NOT steered: the CLK1B
`eps_time` offset is ~-23 ms and drifts at ~-1.5e-8 s/s (399 km of offset
change per day).  After the deterministic offset+drift the residual clock is
random-walk-like: 0.04 m at 60 s, 0.4 m at 10 min, 1.9 m at 1 h -- exactly the
band where a ~1-3 m proper-time field would be visible.  This is the correct
testbed for the operational question:

  during a simulated GNSS gap, can the terrestrial clock network predict the
  GRACE-FO USO clock innovation better than a persistence-trend?

Arms: persistence-trend (baseline), TEP distance kriging to the sub-satellite
point, network mean, distance-scrambled.  Plus the GRACE C - GRACE D clock
differential (common-mode-cancelled).  A ground-to-ground control on the same
machinery must reproduce the Step 55/56 positive.

Data (public, no special access):
  data/leo/gracefo/CLK1B_YYYY-MM-DD_{C,D}_04.txt   (receiver clock, 10 s)
  data/leo/gracefo/GNV1B_YYYY-MM-DD_{C,D}_04.txt   (ITRF position, 1 s)
    from PODAAC GRACEFO_L1B_ASCII_GRAV_JPL_RL04 (Earthdata login)
  clk_cod_ops precise clocks (already in the pipeline)

Output: results/outputs/step_105_gracefo_moving_platform.json
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_105_gracefo_moving_platform.json"
GF_DIR = ROOT / "data" / "leo" / "gracefo"
C_LIGHT = 299_792_458.0
LAMBDA_KM = 4200.0
PRE_S = 3600.0
GAPS = [600.0, 1800.0, 3600.0, 7200.0]
MAX_GC_KM = 8000.0
RNG = np.random.default_rng(20260904)


def _yaml_end(path: Path):
    for i, ln in enumerate(path.read_text().splitlines()):
        if ln.strip() == "# End of YAML header":
            return i
    return 0


def parse_clk1b(path: Path):
    """Return (t_gps, eps_time_s, eps_drift) with duplicate epochs removed."""
    rows = []
    start = _yaml_end(path)
    for ln in path.read_text().splitlines()[start + 1:]:
        f = ln.split()
        if len(f) >= 6:
            try:
                rows.append((float(f[0]), float(f[3]), float(f[5])))
            except ValueError:
                continue
    a = np.asarray(rows)
    _, idx = np.unique(a[:, 0], return_index=True)
    return a[idx]


def parse_gnv1b(path: Path):
    """Return (t_gps, ecef_m (n,3)) at 1 s."""
    rows = []
    start = _yaml_end(path)
    for ln in path.read_text().splitlines()[start + 1:]:
        f = ln.split()
        if len(f) >= 6:
            try:
                rows.append((float(f[0]), float(f[3]), float(f[4]), float(f[5])))
            except ValueError:
                continue
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4]


_GF_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sod(t_gps):
    # GRACE-FO t = seconds past 2000-01-01 12:00:00; shift to 00:00 boundary
    return np.mod(t_gps - 43200.0, 86400.0)


def _day_start_gps(day: str) -> float:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (d - _GF_EPOCH).total_seconds()


def _trim_day(t, *arrs, day: str):
    """Keep only samples inside the exact UTC day so seconds-of-day is
    monotonic (CLK1B/GNV1B files overrun the day by ~5 min each side)."""
    ds = _day_start_gps(day)
    m = (t >= ds) & (t < ds + 86400.0)
    out = [t[m]]
    for a in arrs:
        out.append(a[m])
    return tuple(out)


def load_ground(tag: str):
    s = td.load_series("clk_cod_ops", tag, min_valid_frac=0.0, max_jumps=10000) or {}
    out = {}
    for n, d in s.items():
        tt = np.asarray(d["t"], float)
        yy = np.asarray(d["raw"], float)
        st = d["station"]
        g = np.isfinite(tt) & np.isfinite(yy)
        if g.sum() < 250:
            continue
        out[n] = dict(t=tt[g], y=yy[g], lat=st.lat_deg, lon=st.lon_deg,
                      ecef=st.ecef_m)
    return out


def resid_innov(t, y, t0, t1, pre):
    m = (t >= t0 - pre) & (t <= t0)
    if m.sum() < 4:
        return None
    rate = np.polyfit(t[m], y[m], 1)[0]
    y0 = np.interp(t0, t, y, left=np.nan, right=np.nan)
    y1 = np.interp(t1, t, y, left=np.nan, right=np.nan)
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return None
    return float((y1 - y0) - rate * (t1 - t0))


def _krige(target_pt, target_innov, ground, tgt_name, t0, t1):
    r, w = [], []
    for n, d in ground.items():
        if n == tgt_name:
            continue
        ri = resid_innov(d["t"], d["y"], t0, t1, PRE_S)
        if ri is None or abs(ri) > 50.0:
            continue
        gc = td.great_circle_km(
            target_pt, td.Station(n, d["ecef"] / 1e3, d["lat"], d["lon"]))
        if gc > MAX_GC_KM:
            continue
        r.append(ri)
        w.append(np.exp(-0.5 * (gc / LAMBDA_KM) ** 2))
    if len(r) < 8:
        return None
    r = np.asarray(r); w = np.asarray(w)
    tep = float(w @ r / w.sum())
    nm = float(np.mean(r))
    dsc = float(RNG.permutation(w) @ r / w.sum())
    return (target_innov ** 2, (target_innov - tep) ** 2,
            (target_innov - nm) ** 2, (target_innov - dsc) ** 2)


def _summ(per_day):
    days = sorted(per_day)
    arms = ("persist", "tep", "netmean", "distscr")
    pooled = {a: np.concatenate([np.asarray(per_day[d][a]) for d in days]) for a in arms}
    n = len(pooled["persist"])
    if n < 20:
        return {"n": n}
    rms = {a: float(np.sqrt(np.mean(pooled[a]))) for a in arms}

    def day_rms(sel, a):
        return np.sqrt(np.mean(np.concatenate([per_day[d][a] for d in sel])))
    bs = []
    for _ in range(3000):
        sel = list(RNG.choice(days, len(days), replace=True))
        bs.append(100.0 * (1 - day_rms(sel, "tep") / day_rms(sel, "persist")))
    return {
        "n": n, "n_days": len(days), "rms_m": rms,
        "tep_vs_persist_pct": 100.0 * (1 - rms["tep"] / rms["persist"]),
        "tep_vs_persist_ci95": [float(np.percentile(bs, 2.5)),
                                float(np.percentile(bs, 97.5))],
        "tep_vs_netmean_pct": 100.0 * (1 - rms["tep"] / rms["netmean"]),
        "tep_vs_distscr_pct": 100.0 * (1 - rms["tep"] / rms["distscr"]),
    }


def main() -> int:
    clk_c = sorted(glob.glob(str(GF_DIR / "CLK1B_*_C_*.txt")))
    if not clk_c:
        OUT.write_text(json.dumps({
            "step": 105, "status": "no_data",
            "note": ("Acquire GRACE-FO CLK1B + GNV1B (C and D) into "
                     "data/leo/gracefo/ from PODAAC "
                     "GRACEFO_L1B_ASCII_GRAV_JPL_RL04 (Earthdata login).")},
            indent=2))
        print("no GRACE-FO CLK1B -- see note")
        return 0

    leo_days: dict = {}
    diff_days: dict = {}
    ctl_days: dict = {}
    for cp in clk_c:
        day = Path(cp).name.split("_")[1]                 # 2024-04-09
        dt = datetime.strptime(day, "%Y-%m-%d")
        tag = f"{dt.year}{dt.timetuple().tm_yday:03d}"
        dp = cp.replace("_C_", "_D_")
        gnc = cp.replace("CLK1B", "GNV1B")
        gnd = dp.replace("CLK1B", "GNV1B")
        if not all(Path(x).exists() for x in (dp, gnc, gnd)):
            print(f"  {day}: missing C/D CLK or GNV"); continue
        try:
            cc = parse_clk1b(Path(cp)); cd = parse_clk1b(Path(dp))
            tgc, xgc = parse_gnv1b(Path(gnc)); tgd, xgd = parse_gnv1b(Path(gnd))
            # trim to the exact UTC day (files overrun ~5 min each side)
            tcc, e_c, dr_c = _trim_day(cc[:, 0], cc[:, 1], cc[:, 2], day=day)
            tcd, e_d, dr_d = _trim_day(cd[:, 0], cd[:, 1], cd[:, 2], day=day)
            cc = np.column_stack((tcc, e_c, dr_c))
            cd = np.column_stack((tcd, e_d, dr_d))
            tgc, xgc = _trim_day(tgc, xgc, day=day)
            tgd, xgd = _trim_day(tgd, xgd, day=day)
            ground = load_ground(tag)
        except Exception as exc:
            print(f"  {day}: skip ({exc})"); continue
        if len(ground) < 50 or len(cc) < 4000:
            print(f"  {day}: skip (ground {len(ground)}, clk {len(cc)})"); continue

        # clock residual (metres) after removing the CLK1B offset+drift model:
        # eps_time already carries eps_drift; use a fresh per-window linear fit
        # so the residual is the part persistence-trend cannot predict.
        sc, yc = _sod(cc[:, 0]), cc[:, 1] * C_LIGHT
        sd, yd = _sod(cd[:, 0]), cd[:, 1] * C_LIGHT
        # C - D differential on the common 10 s grid
        both = np.intersect1d(cc[:, 0], cd[:, 0])
        yc_b = np.interp(both, cc[:, 0], cc[:, 1]) * C_LIGHT
        yd_b = np.interp(both, cd[:, 0], cd[:, 1]) * C_LIGHT
        s_b = _sod(both)
        y_diff = yc_b - yd_b

        gcx = lambda s: np.array([np.interp(s, _sod(tgc), xgc[:, k]) for k in range(3)])

        print(f"{tag} ({day}): clk_C std {np.std(yc - np.polyval(np.polyfit(sc, yc, 2), sc)):.2f} m, "
              f"{len(ground)} ground stations", flush=True)

        gnames = list(ground)
        ctl_targets = [n for n in gnames
                       if 35 < ground[n]["lat"] < 70 and -12 < ground[n]["lon"] < 40][:30]

        for gap in GAPS:
            L = leo_days.setdefault(tag, {}).setdefault(
                gap, {"persist": [], "tep": [], "netmean": [], "distscr": []})
            DF = diff_days.setdefault(tag, {}).setdefault(
                gap, {"persist": [], "tep": [], "netmean": [], "distscr": []})
            Cd = ctl_days.setdefault(tag, {}).setdefault(
                gap, {"persist": [], "tep": [], "netmean": [], "distscr": []})
            for t0 in np.arange(PRE_S + 100.0, 86400.0 - gap - 100.0, 400.0):
                t1 = t0 + gap
                # --- GRACE-C absolute clock ---
                li = resid_innov(sc, yc, t0, t1, PRE_S)
                if li is not None and abs(li) < 50.0:
                    p = gcx(t1)
                    la, lo = td.ecef_to_geodetic(*p)
                    e = _krige(td.Station("GFC", p / 1e3, la, lo), li,
                               ground, None, t0, t1)
                    if e is not None:
                        for a, v in zip(("persist", "tep", "netmean", "distscr"), e):
                            L[a].append(v)
                # --- C - D differential ---
                di = resid_innov(s_b, y_diff, t0, t1, PRE_S)
                if di is not None and abs(di) < 50.0:
                    p = gcx(t1)
                    la, lo = td.ecef_to_geodetic(*p)
                    e = _krige(td.Station("GFCD", p / 1e3, la, lo), di,
                               ground, None, t0, t1)
                    if e is not None:
                        for a, v in zip(("persist", "tep", "netmean", "distscr"), e):
                            DF[a].append(v)
                # --- ground-to-ground control ---
                for tgt in ctl_targets:
                    d = ground[tgt]
                    ci_ = resid_innov(d["t"], d["y"], t0, t1, PRE_S)
                    if ci_ is None or abs(ci_) > 50.0:
                        continue
                    e = _krige(td.Station(tgt, d["ecef"] / 1e3, d["lat"], d["lon"]),
                               ci_, ground, tgt, t0, t1)
                    if e is not None:
                        for a, v in zip(("persist", "tep", "netmean", "distscr"), e):
                            Cd[a].append(v)

    result = {
        "step": 105,
        "title": "Moving-platform clock bridging (GRACE-FO free-running USO vs terrestrial TEP field)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "satellite": "GRACE-FO C/D (CLK1B USO clock 10 s, GNV1B position 1 s)",
        "ground_network": "clk_cod_ops (300+ stations, 300 s)",
        "lambda_km": LAMBDA_KM, "pre_window_s": PRE_S,
        "note_swarm": ("Step 104 (Swarm-A) was negative but the Swarm clock is "
                       "steered; GRACE-FO's USO free-runs (offset ~-23 ms, drift "
                       "~-1.5e-8 s/s) so its clock carries the physical signal."),
        "grace_c_absolute": {}, "grace_c_minus_d": {}, "ground_control": {},
    }
    for gap in GAPS:
        gs = str(int(gap))
        result["grace_c_absolute"][gs] = _summ(
            {d: leo_days[d][gap] for d in leo_days if gap in leo_days.get(d, {})})
        result["grace_c_minus_d"][gs] = _summ(
            {d: diff_days[d][gap] for d in diff_days if gap in diff_days.get(d, {})})
        result["ground_control"][gs] = _summ(
            {d: ctl_days[d][gap] for d in ctl_days if gap in ctl_days.get(d, {})})

    def _all_pos(block):
        vs = [block[str(int(g))].get("tep_vs_persist_ci95", [0, 0])[0]
              for g in GAPS if "tep_vs_persist_ci95" in block[str(int(g))]]
        return bool(vs and all(v > 0 for v in vs))

    result["verdict"] = {
        "grace_c_bridgeable_from_ground": _all_pos(result["grace_c_absolute"]),
        "grace_cd_diff_bridgeable_from_ground": _all_pos(result["grace_c_minus_d"]),
        "ground_control_reproduces_step56": _all_pos(result["ground_control"]),
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nsaved {OUT}")
    for g in GAPS:
        gs = str(int(g))
        a = result["grace_c_absolute"][gs]
        c = result["ground_control"][gs]
        if "tep_vs_persist_pct" in a:
            print(f"  gap {g:6.0f}s  GRACE-C abs {a['tep_vs_persist_pct']:+.1f}% "
                  f"CI{a['tep_vs_persist_ci95']}  n={a['n']}  |  control "
                  f"{c.get('tep_vs_persist_pct', float('nan')):+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
