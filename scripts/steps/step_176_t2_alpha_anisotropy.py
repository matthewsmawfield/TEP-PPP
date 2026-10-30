#!/usr/bin/env python3
"""
STEP 176 -- Does the short-baseline EW/NS anisotropy survive common-view
             removal?  (T2 on the non-common-view intercept)

Step 82 measured a significant EW excess over NS in the phase-alignment
index cos(arg sum_f S_ij(f)) on the broadcast product at 500-800 km
(+0.130, z = +1.94) and 300-800 km -- the only azimuthally significant
bands on any operational product.  The T2 paragraph already identifies
co-visibility anisotropy as the dominant confound on products carrying
satellite common-mode: EW baselines systematically share more satellites
than NS baselines at the same distance.

The corrected Step-45/83 machinery supplies the control directly: each
pair-day's phase-cos is regressed on that pair-day's real-SP3 mean
co-visibility Jbar with a single pooled slope kappa, and the intercept
alpha = mean_d(pc - kappa*Jbar_d) is the non-common-view component.  If
the raw EW excess were the proper-time field it must appear in alpha
(the field at zero common view); if it were co-visibility-structured it
must not.

The band/sector/null design is Step 82's own: the same phase-cos
observable, the same midpoint-bearing convention, the same bands, the
same 22.5-degree sector half-angle, and the same sector-label
permutation null, over the same 2024 corpus (identical pair set --
1,016 broadcast pairs / 116,997 pair-days).  The only change is what is
tested: raw phase-cos versus its common-view intercept alpha.  A wider
45-degree sector cone (the full cardinal quadrant, no excluded diagonal
band) is reported alongside as a power variant; it is not the Step-82
cone.

Output: results/outputs/step_176_t2_alpha_anisotropy.json
"""
from __future__ import annotations

import sys
import json
import itertools
import importlib.util
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_s53 = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s53)
_s53.loader.exec_module(s53)
_s45 = importlib.util.spec_from_file_location(
    "s45", ROOT / "scripts" / "steps" / "step_45_temporal_t4.py")
s45 = importlib.util.module_from_spec(_s45)
_s45.loader.exec_module(s45)
td = s53.td

OUT = ROOT / "results" / "outputs" / "step_176_t2_alpha_anisotropy.json"
RNG = np.random.default_rng(176)
BANDS = [(300, 500), (500, 800), (300, 800), (300, 1500), (500, 2000)]
HALF = 22.5            # sector half-angle, verbatim step_82
HALF_WIDE = 45.0       # full-quadrant cone, reported as a power variant
N_PERM = 5000          # verbatim step_82 permutation count
COVIS_EPOCHS = 8


def band_test(up, A_, V, half):
    out = {}
    for lo, hi in BANDS:
        inb = (np.array(up) >= lo) & (np.array(up) < hi)
        ew = inb & (np.abs(A_ - 90) <= half)
        ns = inb & ((A_ <= half) | (A_ >= 180 - half))
        if ew.sum() < 6 or ns.sum() < 6:
            continue
        obs = float(V[ew].mean() - V[ns].mean())
        idx = np.where(inb)[0]
        nulls = []
        for _ in range(N_PERM):
            sh = RNG.permutation(A_[idx])
            e = np.abs(sh - 90) <= half
            n = (sh <= half) | (sh >= 180 - half)
            if e.sum() >= 6 and n.sum() >= 6:
                nulls.append(V[idx][e].mean() - V[idx][n].mean())
        nl = np.array(nulls)
        z = float((obs - nl.mean()) / (nl.std() + 1e-30)) if len(nl) else np.nan
        out[f"{lo}-{hi}"] = {
            "n_ew_pairs": int(ew.sum()), "n_ns_pairs": int(ns.sum()),
            "diff_ew_minus_ns": obs,
            "perm_p": float((np.abs(nl) >= abs(obs)).mean()) if len(nl) else np.nan,
            "z": z, "significant": bool(abs(z) > 2.0),
        }
    return out


def run_arm(product, max_days=366):
    rows = []          # (dd, az, phasecos, jbar, pair_name)
    sta = {}
    nd = 0
    for tag in td.available_dates(product, 2024)[:max_days]:
        try:
            ser = td.load_series(product, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        if len(RF) < 6:
            continue
        nd += 1
        nm = list(RF)
        for n in nm:
            sta[n] = STA[n]
        sts = [STA[n] for n in nm]
        # Real-SP3 co-visibility for this day (GPS), mean over a few epochs;
        # nominal model as documented fallback.
        ep_sod = np.linspace(0, 86400, COVIS_EPOCHS, endpoint=False)
        rv = s45.load_real_visibility(tag, sts, ep_sod, systems=("G",))
        if rv is not None:
            J, pidx = s45._per_epoch_covisibility_from_vis(rv[0])
        else:
            ep = (int(tag[4:]) - 1) * 86400.0 + ep_sod
            J, pidx = s45._per_epoch_covisibility(sts, ep)
        Jbar_day = J.mean(axis=1)
        for pi, (i, j) in enumerate(pidx):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            dd = td.great_circle_km(A, B)
            if not (300. <= dd < 2000.):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((dd, td.baseline_azimuth_midpoint_deg(A, B),
                         float(np.cos(np.angle(S))), float(Jbar_day[pi]),
                         f"{min(A.name, B.name)}-{max(A.name, B.name)}"))
        if nd % 40 == 0:
            print(f"    {nd} days, {len(rows):,} pair-days", flush=True)
    if len(rows) < 2000:
        return None

    # pooled kappa over pair-days: kappa = sum cov(pc, J) / sum var(J)
    pc_ = np.array([r[2] for r in rows]); jv = np.array([r[3] for r in rows])
    pid = np.array([r[4] for r in rows])
    up = np.unique(pid)
    # within-pair pooled slope (pair FE), as step_83
    knum = kden = 0.0
    for u in up:
        m = pid == u
        jc = jv[m] - jv[m].mean()
        knum += float(np.dot(jc, pc_[m] - pc_[m].mean()))
        kden += float(np.dot(jc, jc))
    kappa = knum / kden if kden > 1e-12 else 0.0

    P = np.array([np.array([r[0] for r in rows])[pid == u].mean() for u in up])
    A_ = np.array([np.array([r[1] for r in rows])[pid == u][0] for u in up])
    Vr = np.array([pc_[pid == u].mean() for u in up])
    Va = np.array([(pc_[pid == u] - kappa * jv[pid == u]).mean() for u in up])
    Jb = np.array([jv[pid == u].mean() for u in up])

    return {
        "n_days": nd, "n_pair_days": len(rows), "n_unique_pairs": len(up),
        "pooled_kappa": kappa,
        "sector_half_angle_deg_verbatim": HALF,
        "sector_half_angle_deg_wide": HALF_WIDE,
        "raw_bands": band_test(P, A_, Vr, HALF),
        "alpha_bands": band_test(P, A_, Va, HALF),
        # control: EW-NS of Jbar itself (the confound amplitude)
        "jbar_bands": band_test(P, A_, Jb, HALF),
        "raw_bands_wide": band_test(P, A_, Vr, HALF_WIDE),
        "alpha_bands_wide": band_test(P, A_, Va, HALF_WIDE),
        "jbar_bands_wide": band_test(P, A_, Jb, HALF_WIDE),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in ("broadcast", "clk_cod"):
        print(f"=== {product} ===", flush=True)
        a = run_arm(product)
        if a:
            arms[product] = a
    OUT.write_text(json.dumps({
        "step": 176,
        "title": "T2 anisotropy on the non-common-view intercept",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": ("Step-82 design verbatim: phase-cos observable, midpoint "
                   "bearing, 22.5-degree sector cones, bands, and sector-label "
                   "permutation null over the same 2024 corpus (identical "
                   "1,016-pair / 116,997 pair-day broadcast set) -- tested "
                   "on the raw phase-cos and on its common-view intercept "
                   "alpha (pooled kappa on real-SP3 Jbar), plus the EW-NS "
                   "contrast of Jbar itself as the confound amplitude.  "
                   "The *_wide blocks repeat the battery at a 45-degree "
                   "sector half-angle (the full cardinal quadrant) as a "
                   "higher-power variant."),
        "arms": arms,
        "verdict": {
            p: {
                "bands_significant_in_raw": [b for b, v in a["raw_bands"].items()
                                             if v["significant"]],
                "bands_significant_in_alpha": [b for b, v in a["alpha_bands"].items()
                                               if v["significant"]],
            } for p, a in arms.items()
        }}, indent=2))
    for p, a in arms.items():
        print(f"\n=== {p}: kappa={a['pooled_kappa']:+.4f} ===")
        for b in a["raw_bands"]:
            r, al = a["raw_bands"][b], a["alpha_bands"][b]
            print(f"  {b}: raw {r['diff_ew_minus_ns']:+.4f} (z {r['z']:+.2f})   "
                  f"alpha {al['diff_ew_minus_ns']:+.4f} (z {al['z']:+.2f})   "
                  f"jbar {a['jbar_bands'][b]['diff_ew_minus_ns']:+.4f}")
        for b in a.get("raw_bands_wide", {}):
            r, al = a["raw_bands_wide"][b], a["alpha_bands_wide"][b]
            print(f"  {b} wide: raw {r['diff_ew_minus_ns']:+.4f} "
                  f"(z {r['z']:+.2f})   "
                  f"alpha {al['diff_ew_minus_ns']:+.4f} (z {al['z']:+.2f})   "
                  f"jbar {a['jbar_bands_wide'][b]['diff_ew_minus_ns']:+.4f}")


if __name__ == "__main__":
    main()
