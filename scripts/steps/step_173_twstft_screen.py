#!/usr/bin/env python3
"""Step 173 — TWSTFT independent-clock distance-correlation screen.

Frozen pre-registered design (v1, 2026-10-27): every quantity below
is fixed BEFORE scoring; the analysis may be re-run on additional
years without changing any parameter.

Data: BIPM two-way satellite time transfer (TWSTFT) between national
timing laboratories, fetched by scripts/live/fetch_twstft.py into
data/twstft/twstft_<lab>_<year>_<lo>_<hi>.csv plus twstft_labs.json
(station coordinates parsed from the `* ES` header lines).

Rationale: TWSTFT links a geostationary transponder to two atomic-
clock labs and cancels path/systematics by construction — no
one-way GNSS product (orbit/clock model, ionosphere map) enters the
measurement.  If the distance-dependent correlation structure found
in the GNSS station-clock corpus is a processing artifact of that
corpus, it MUST NOT appear here; if it is a physical field, it may.

Pipeline (as implemented; see AUDIT note below — the result is
NOT usable as clock evidence):
  1. epoch_s = (mjd - 40587) * 86400 + HHMMSS-decoded sttime.
  2. Per link (loc,rem,cal_id) segment: QUADRATIC detrend of
     tw_s vs time (offset + frequency drift); optional removal
     of the pooled-median common satellite-path signature with
     a per-link projection gain.
  3. Bin residuals into BIN_S = 7200 s bins; per (link, bin)
     take the MEDIAN; link quality gate (residual std < 1 ms).
  4. Per-bin inverse-variance WLSQ inversion c_i with
     r_link = c_loc - c_rem, gauge c -> c - mean(c).
  5. Pairwise Pearson r_ij (joint coverage >= MIN_COV), distance
     from twstft_labs.json (haversine, R=6371 km).
  6. Bin Fisher-z by DIST_EDGES_KM; S = z(TEP band) - z(rest).
  7. Nulls: (a) uniform circular shifts of each lab series;
     (b) geography permutation of pair distances (preserves the
     correlation matrix — the meaningful null for the distance
     claim).

The GNSS-corpus result this mirrors: inter-station clock
correlation shows a negative lobe at ~1200-2500 km and a ~2-3 Mm
characteristic decay (see manuscript Results).  DIST_EDGES_KM is
chosen so that band is one isolated bin.

AUDIT (2026-10-27, post-v1): three defects found and fixed —
  (a) STTIME is HHMMSS and was parsed as raw seconds: 91.7% of
      rows mis-binned (up to ~2.5 days displaced);
  (b) equal-weight per-bin LSQ let ~us-variance transatlantic
      links dominate the inverted clocks (now inverse-variance
      WLSQ + a documented 1 ms quality gate);
  (c) decisive: the reported TW observable in these raw files is
      PATH-dominated — +-135 us diurnal structure (peaks at
      23.8/24.3/12.0 h) plus ~3.5 us/link residual wander,
      consistent with TELSTAR 11N geostationary libration and
      impossible for atomic-clock differences (ns-class).  The
      BIPM applies ephemeris/ESDVAR/CALR corrections downstream
      that are absent from the raw per-MJD files.  CONSEQUENCE:
      any correlation statistic computed on these residuals is a
      measurement of satellite-path systematics, not clock
      physics; the earlier "+0.46 disjoint / TEP-sign fails"
      reading is withdrawn.  ns-class TWSTFT analysis requires
      TLE-based path correction or corrected lab products.

Usage:
  python3 scripts/steps/step_173_twstft_screen.py \
      --data-dir data/twstft --year 2025 \
      --n-shifts 500 --out results/outputs/step_173_twstft_screen.json
  python3 scripts/steps/step_173_twstft_screen.py --selftest
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BIN_S = 7200.0                    # frozen: 2h bins (link cadence)
DIST_EDGES_KM = [0, 500, 1200, 2500, 4000, 6000, 20000]
TEP_BAND = (1200, 2500)           # frozen: the negative-lobe bin
MIN_COV = 0.60                    # frozen: joint coverage for pairs
R_EARTH_KM = 6371.0


def _hav_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(d))


def _station_of(es_id):
    """NIST01 -> strip trailing digits -> lab key used in coords."""
    return es_id


def load_links(data_dir, year=None):
    """Return (records_df, coords)."""
    frames = []
    for fp in sorted(glob.glob(str(Path(data_dir) / "twstft_*.csv"))):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit(f"no twstft CSVs in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["loc", "rem", "mjd", "sttime_s"])
    if "sat" not in df.columns:
        df["sat"] = ""               # CSVs predate the sat column
    cpath = Path(data_dir) / "twstft_labs.json"
    coords = json.loads(cpath.read_text()) if cpath.exists() else {}
    # STTIME is HHMMSS (e.g. "202800" = 20:28:00), NOT seconds
    st = df["sttime_s"].astype(int)
    sttime_s = (st // 10000) * 3600 + ((st // 100) % 100) * 60 + st % 100
    df["epoch_s"] = (df["mjd"] - 40587) * 86400.0 + sttime_s
    return df, coords


def common_mode_curve(r, grid_s=900.0, win=9):
    """Estimate the shared satellite-path signature: every link in
    this corpus rides TELSTAR 11N, whose geostationary libration
    (~+-50 km/day) injects ~+-100-300 us of common path wander —
    3 orders of magnitude above the ns-class clock signal.  The
    pooled residual cloud densely samples that signature at each
    link's own session times; a rolling median over 15-min cells
    (~2h window) traces it.  Returns (grid_t, cm_s)."""
    g = (r.assign(cell=(r["epoch_s"] // grid_s).astype(np.int64))
          .groupby("cell")["resid_s"].median())
    t = g.index.values * grid_s + grid_s / 2
    y = g.values.astype(float)
    k = np.ones(win) / win
    ys = np.convolve(y, k, mode="same")
    # keep edges honest: recompute boundary cells by truncation
    for i in range(win // 2):
        ys[i] = y[:i + win // 2 + 1].mean()
        ys[-i - 1] = y[-(i + win // 2 + 1):].mean()
    return t, ys


def link_residuals(df, remove_cm=True):
    """Step 2-3: per-link cal_id-segment quadratic detrend, common
    satellite-path subtraction, then 2h median.

    Returns DataFrame [loc, rem, bin_idx, resid_s, drms_ns, sat,
    w]."""
    df = df.copy()
    df["bin_idx"] = (df["epoch_s"] // BIN_S).astype(np.int64)
    out = []
    for (loc, rem, cal), g in df.groupby(["loc", "rem", "cal_id"]):
        g = g.sort_values("epoch_s")
        if len(g) < 8:
            continue
        t = g["epoch_s"].values - g["epoch_s"].values[0]
        y = g["tw_s"].values.astype(float)
        # quadratic detrend per cal segment: atomic pairs drift in
        # FREQUENCY (quadratic phase) as well as linear offset
        tt = t / 86400.0                       # days, well-conditioned
        A = np.column_stack([np.ones_like(tt), tt, tt ** 2])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        g = g.assign(resid_s=y - A @ coef)
        out.append(g[["loc", "rem", "bin_idx", "epoch_s",
                     "resid_s", "drms_ns"]])
    if not out:
        return pd.DataFrame(columns=["loc", "rem", "bin_idx",
                                     "epoch_s", "resid_s",
                                     "drms_ns"])
    r = pd.concat(out, ignore_index=True)
    if remove_cm:
        t_cm, cm = common_mode_curve(r)
        print(f"common satellite-path signature: "
              f"std={np.std(cm)*1e6:.1f} us, "
              f"range=[{cm.min()*1e6:.1f},{cm.max()*1e6:.1f}] us")
        # per-link projection: each station pair sees the same
        # satellite libration through a different elevation/
        # geometry gain -> fit alpha_i per link, subtract alpha*cm
        r["cmv"] = np.interp(r["epoch_s"].values, t_cm, cm)
        alpha = (r.groupby(["loc", "rem"], group_keys=False)
                  .apply(lambda g: float(
                      (g["resid_s"] * g["cmv"]).sum()
                      / max((g["cmv"] ** 2).sum(), 1e-30))))
        r["alpha"] = r.set_index(["loc", "rem"]).index.map(alpha)
        r["resid_s"] = r["resid_s"] - r["alpha"] * r["cmv"]
        r = r.drop(columns=["cmv", "alpha"])
    if not out:
        return pd.DataFrame(columns=["loc", "rem", "bin_idx",
                                     "resid_s", "drms_ns"])
    g = (r.groupby(["loc", "rem", "bin_idx"])
          .agg(resid_s=("resid_s", "median"),
               drms_ns=("drms_ns", "median"))
          .reset_index())
    sat_map = (df.groupby(["loc", "rem"])["sat"].first()
                 .rename("sat").reset_index())
    g = g.merge(sat_map, on=["loc", "rem"], how="left")

    # data-quality gate: drop degenerate links only (PL01-class
    # junk showed ~12 ms residual std; healthy links post-CM
    # removal land ~ns-us).  Report, do not tune.
    std = g.groupby(["loc", "rem"])["resid_s"].transform("std")
    n_before = g[["loc", "rem"]].drop_duplicates().shape[0]
    g = g[std < 1e-3].copy()
    n_after = g[["loc", "rem"]].drop_duplicates().shape[0]
    print(f"link quality gate: kept {n_after}/{n_before} links "
          f"(resid std < 1 ms)")
    # per-link inverse-variance weight for the WLSQ inversion
    var = g.groupby(["loc", "rem"])["resid_s"].transform("var")
    g["w"] = 1.0 / np.maximum(var, 1e-24)
    g["w"] /= g["w"].median()
    return g


def invert_panel(link_res):
    """Step 4: per-bin LSQ c_i with mean gauge -> {lab: {bin: c}}."""
    labs = sorted(set(link_res["loc"]) | set(link_res["rem"]))
    lidx = {l: i for i, l in enumerate(labs)}
    panel = {l: {} for l in labs}
    for b, g in link_res.groupby("bin_idx"):
        rows, rhs, wts = [], [], []
        for _, r in g.iterrows():
            v = np.zeros(len(labs))
            v[lidx[r["loc"]]] = 1.0
            v[lidx[r["rem"]]] = -1.0
            rows.append(v)
            rhs.append(r["resid_s"])
            wts.append(r.get("w", 1.0))
        if len(rows) < len(labs):          # underdetermined bin
            continue
        A = np.vstack(rows)
        y = np.asarray(rhs)
        w = np.sqrt(np.asarray(wts))[:, None]
        c, *_ = np.linalg.lstsq(A * w, y * w[:, 0], rcond=None)
        c = c - c.mean()                   # gauge: remove common mode
        for l, i in lidx.items():
            panel[l][int(b)] = c[i]
    return panel, labs


def pair_stats(panel, labs, coords, bins):
    """Steps 6-8: pairwise r vs distance; TEP sign statistic."""
    all_bins = sorted({b for l in labs for b in panel[l]})
    if len(all_bins) < 24:
        return None
    pairs = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            if a not in coords or b not in coords:
                continue
            d = _hav_km(*coords[a][:2], *coords[b][:2])
            common = sorted(set(panel[a]) & set(panel[b]))
            if len(common) < MIN_COV * len(all_bins):
                continue
            x = np.array([panel[a][k] for k in common])
            y = np.array([panel[b][k] for k in common])
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            pairs.append({"a": a, "b": b, "d_km": d, "r": r,
                          "n": len(common)})
    if not pairs:
        return None
    bin_z = {i: [] for i in range(len(bins) - 1)}
    for p in pairs:
        for i in range(len(bins) - 1):
            if bins[i] <= p["d_km"] < bins[i + 1]:
                p["bin"] = i
                z = np.arctanh(np.clip(p["r"], -0.999, 0.999))
                bin_z[i].append(z)
                break
    mean_z = {i: (float(np.mean(v)) if v else np.nan)
              for i, v in bin_z.items()}
    tep_i = next(i for i in range(len(bins) - 1)
                 if bins[i] == TEP_BAND[0] and
                 bins[i + 1] == TEP_BAND[1])
    rest = [z for i, v in bin_z.items() if i != tep_i for z in v]
    if not bin_z[tep_i] or not rest:
        return None
    S = float(np.mean(bin_z[tep_i]) - np.mean(rest))
    return {"pairs": pairs, "mean_z": mean_z, "S": S,
            "tep_bin": tep_i, "n_pairs": len(pairs),
            "n_bins_total": len(all_bins)}


def permute_S_nulls(pairs, bins, n_perm, rng):
    """Geography null: permute pair distances among the observed
    pairs and recompute S.  Preserves the actual correlation matrix
    (common mode included); tests whether correlation is organized
    by DISTANCE specifically."""
    ds = np.array([p["d_km"] for p in pairs])
    rs = np.array([p["r"] for p in pairs])
    z = np.arctanh(np.clip(rs, -0.999, 0.999))

    def _S(d_assign):
        tep_m = (d_assign >= TEP_BAND[0]) & (d_assign < TEP_BAND[1])
        rest_m = (d_assign >= bins[0]) & (d_assign < bins[-1]) & ~tep_m
        if tep_m.sum() == 0 or rest_m.sum() == 0:
            return np.nan
        return float(z[tep_m].mean() - z[rest_m].mean())

    out = np.empty(n_perm)
    for k in range(n_perm):
        out[k] = _S(rng.permutation(ds))
    return out[np.isfinite(out)]


def link_diagnostics(link_res):
    """Quantify link-level common-mode contamination BEFORE the lab
    inversion: correlate binned residual series between pairs of
    links, grouped by their relationship —
      share_loc:  links (L,X) vs (L,Y) — share the local lab's
                  clock AND its satellite hop;
      share_rem:  (X,L) vs (Y,L);
      disjoint_samesat / disjoint_diffsat: no shared lab, split by
                  whether both links ride the same satellite.
    A satellite/transponder common-mode shows as
    disjoint_samesat >> disjoint_diffsat; a lab-clock or UTC-
    steering common-mode lifts share_* above the disjoint
    baseline."""
    piv = link_res.pivot_table(index="bin_idx",
                               columns=["loc", "rem"],
                               values="resid_s")
    sat_of = {(l, r): s for (l, r), s in
              link_res.groupby(["loc", "rem"])["sat"]
              .first().items()}
    cols = list(piv.columns)
    groups = {"share_loc": [], "share_rem": [],
              "disjoint_samesat": [], "disjoint_diffsat": []}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            (l1, r1), (l2, r2) = cols[i], cols[j]
            both = piv[[(l1, r1), (l2, r2)]].dropna()
            if len(both) < 100:
                continue
            rr = float(np.corrcoef(both.iloc[:, 0],
                                   both.iloc[:, 1])[0, 1])
            if l1 == l2:
                groups["share_loc"].append(rr)
            elif r1 == r2:
                groups["share_rem"].append(rr)
            elif len({l1, r1} & {l2, r2}) == 0:
                s1, s2 = (sat_of.get((l1, r1), ""),
                          sat_of.get((l2, r2), ""))
                if s1 and s2 and s1 == s2:
                    groups["disjoint_samesat"].append(rr)
                else:
                    groups["disjoint_diffsat"].append(rr)
    out = {}
    for k, v in groups.items():
        v = np.asarray(v)
        out[k] = {"n": int(len(v)),
                  "mean_r": float(np.mean(v)) if len(v) else None,
                  "std_r": float(np.std(v)) if len(v) else None}
    return out


def circular_shift_nulls(panel, labs, coords, bins, n_shift, rng):
    """Step 9: each lab column is circularly shifted by an
    independent uniform-random number of bins (>=12 bins = half a
    day, avoiding near-zero shifts).  Preserves each series'
    autocorrelation while destroying inter-lab alignment — the
    weakest null: it also destroys any genuine common-mode
    structure, so it only tests alignment, not distance
    organisation (that is permute_S_nulls' job)."""
    S_null = np.empty(n_shift)
    for k in range(n_shift):
        shifted = {}
        for l in labs:
            ks = sorted(panel[l])
            n = len(ks)
            if n < 26:
                shifted[l] = dict(panel[l])
                continue
            s = int(rng.integers(12, n - 12)) * \
                int(rng.choice([-1, 1]))
            k0 = ks[0]
            vals = [panel[l][kk] for kk in ks]
            newk = [(kk - k0 + s) % n + k0 for kk in ks]
            shifted[l] = dict(zip(newk, vals))
        st = pair_stats(shifted, labs, coords, bins)
        S_null[k] = st["S"] if st else np.nan
    return S_null[np.isfinite(S_null)]


def selftest():
    """Inject a standing-wave spatial field (pair correlation =
    cos(2*pi*d/Lambda), negative at ~Lambda/2 separation inside the
    TEP band) and verify the screen recovers the TEP-sign
    statistic."""
    rng = np.random.default_rng(7)
    n_bins = 24 * 14                      # 14 days of 2h bins
    labs = ["L%02d" % i for i in range(8)]
    lat = rng.uniform(30, 55, 8)
    lon = rng.uniform(-110, 20, 8)
    coords = {l: (lat[i], lon[i], 0.0) for i, l in enumerate(labs)}
    # plane-wave field: corr(a,b) ~ cos(2 pi d_proj / Lambda),
    # Lambda chosen so anti-correlation lands in 1200-2500 km
    LAM = 4400.0
    u = np.array([0.8, 0.6])              # propagation direction
    proj = np.array([u @ np.array(
        [coords[l][1] * 111.0 * np.cos(np.radians(coords[l][0])),
         coords[l][0] * 111.0]) for l in labs])
    F1 = rng.normal(0, 1, n_bins)
    F2 = rng.normal(0, 1, n_bins)
    lab_series = {}
    for i, l in enumerate(labs):
        ph = 2 * np.pi * proj[i] / LAM
        lab_series[l] = (np.cos(ph) * F1 + np.sin(ph) * F2
                         + rng.normal(0, 0.3, n_bins))
    panel = {l: {b: lab_series[l][b] for b in range(n_bins)}
             for l in labs}
    st = pair_stats(panel, labs, coords, DIST_EDGES_KM)
    if st is None:
        print("SELFTEST FAIL: no stats")
        return 1
    print("selftest S =", st["S"], "(expect negative)")
    nulls = circular_shift_nulls(panel, labs, coords,
                                 DIST_EDGES_KM, 200, rng)
    p = float(np.mean(np.abs(nulls) >= abs(st["S"])))
    p_tep = float(np.mean(nulls <= st["S"]))
    print(f"selftest: p_family={p:.4f} p_tep={p_tep:.4f} "
          f"(n_null={len(nulls)})")
    ok = st["S"] < 0 and p < 0.05
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/twstft"))
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--n-shifts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=173)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    df, coords = load_links(args.data_dir, args.year)
    print(f"{len(df)} raw records, {len(coords)} stations")

    result = {
        "step": "step_173_twstft_screen",
        "frozen_spec": {"BIN_S": BIN_S, "DIST_EDGES_KM": DIST_EDGES_KM,
                        "TEP_BAND": TEP_BAND, "MIN_COV": MIN_COV},
        "n_records": int(len(df)), "n_stations": len(coords),
        "modes": {},
    }
    rng = np.random.default_rng(args.seed)
    for label, lr in (("raw", link_residuals(df, remove_cm=False)),
                      ("clean", link_residuals(df, remove_cm=True))):
        if not len(lr):
            print(f"[{label}] no residuals")
            continue
        print(f"\n[{label}] {len(lr)} binned residuals, "
              f"{lr['bin_idx'].nunique()} bins")
        diag = link_diagnostics(lr)
        print(f"\n[{label}] link-residual correlation diagnostics:")
        for k, v in diag.items():
            if v["n"]:
                print(f"  {k:10s}: mean r={v['mean_r']:+.3f} "
                      f"± {v['std_r']:.3f} (n={v['n']})")
        panel, labs = invert_panel(lr)
        n_cov = {l: len(panel[l]) for l in labs}
        st = pair_stats(panel, labs, coords, DIST_EDGES_KM)
        if st is None:
            print(f"[{label}] insufficient data")
            continue
        print(f"\n[{label}] per-bin mean Fisher-z correlation:")
        for i in range(len(DIST_EDGES_KM) - 1):
            z = st["mean_z"][i]
            n = sum(1 for p in st["pairs"] if p.get("bin") == i)
            tag = " <== TEP band" if i == st["tep_bin"] else ""
            print(f"  {DIST_EDGES_KM[i]:>5}-"
                  f"{DIST_EDGES_KM[i + 1]:<5} km: z={z:+.4f} "
                  f"(n={n}){tag}")
        print(f"[{label}] S = {st['S']:+.4f} on {st['n_pairs']} "
              f"pairs / {st['n_bins_total']} bins")
        nulls = circular_shift_nulls(panel, labs, coords,
                                     DIST_EDGES_KM, args.n_shifts,
                                     rng)
        p_fam = float(np.mean(np.abs(nulls) >= abs(st["S"])))
        p_tep = float(np.mean(nulls <= st["S"]))
        # geography permutation null — preserves the correlation
        # matrix incl. common mode; the primary test for the
        # DISTANCE claim
        perm = permute_S_nulls(st["pairs"], DIST_EDGES_KM,
                               args.n_shifts, rng)
        p_perm = float(np.mean(np.abs(perm) >= abs(st["S"])))
        p_perm_tep = float(np.mean(perm <= st["S"]))
        print(f"[{label}] nulls n={len(nulls)}: "
              f"p_family={p_fam:.4f} p_tep={p_tep:.4f} | "
              f"perm nulls n={len(perm)}: p={p_perm:.4f} "
              f"p_tep={p_perm_tep:.4f}")
        result["modes"][label] = {
            "link_diagnostics": diag,
            "lab_coverage_bins": {k: int(v) for k, v
                                  in n_cov.items()},
            "pairs": st["pairs"],
            "mean_z_per_bin": {
                f"{DIST_EDGES_KM[i]}-{DIST_EDGES_KM[i+1]}":
                (float(st["mean_z"][i])
                 if np.isfinite(st["mean_z"][i]) else None)
                for i in range(len(DIST_EDGES_KM) - 1)},
            "S_stat": st["S"], "n_null": int(len(nulls)),
            "p_family": p_fam, "p_tep_negative": p_tep,
            "null_S_mean": float(np.mean(nulls)),
            "null_S_std": float(np.std(nulls)),
            "n_perm": int(len(perm)),
            "p_perm": p_perm, "p_perm_tep_negative": p_perm_tep,
            "perm_S_mean": float(np.mean(perm)),
            "perm_S_std": float(np.std(perm)),
        }

    out = args.out or str(ROOT / "results/outputs/step_173_twstft_screen.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
