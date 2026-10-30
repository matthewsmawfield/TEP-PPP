#!/usr/bin/env python3
"""
STEP 76  --  Is the quartz-only anisotropy real screening, or station geography?

Step 75 found the EW/NS phase-alignment anisotropy in the 500-800 km band is
carried almost entirely by quartz-quartz pairs, on three independent products:

    product      QQ        QA        AA
    clk_cod    +0.311    -0.244    +0.046
    broadcast  +0.356    -0.043    -0.006
    precise    +0.443    -0.078    -0.038

against a pooled (mixed-population) value of only +0.06 to +0.08 in Step 74.
If real, this says the corpus's directional signature was diluted 5-7x by
including screened atomic stations, and that Papers 1-3 measured every
directional result on a mixed population.

The confound that would fake it exactly: quartz and atomic stations are not
randomly placed.  IGS core sites tend to run atomic standards while regional
sites run internal quartz, so QQ pairs may sample different geography -- and
therefore different baseline orientations -- than AA pairs.  A geographic
difference would produce a QQ/AA anisotropy contrast with no screening at all.

The decisive control is a CLOCK-LABEL PERMUTATION: reassign the QUARTZ/ATOMIC
labels at random among the stations actually present, preserving the network
geometry, the baseline orientation distribution, the number of each clock type,
and every pair's measured phase alignment.  Only the association between clock
type and station is destroyed.  If the observed quartz anisotropy sits inside
that null, it is geography; if it sits outside, it is screening.

Also reported: a pair-clustered bootstrap CI (the same pairs recur daily, so
day-clustering overstates significance -- see Step 74b), and the EW/NS pair
counts, since a large contrast built on a handful of pairs is not a measurement.

Output: results/outputs/step_76_screening_anisotropy_null.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
import itertools
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_76_screening_anisotropy_null.json"
CLOCKS = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
RNG = np.random.default_rng(20260904)

PRODUCTS = ["clk_cod", "broadcast", "precise"]
N_DAYS = 366
LO, HI = 500.0, 800.0
# sweep: the 500-800 km window was inherited from Step 74.  If screening is real
# the effect should not be confined to one band, and a wider band is also the only
# way to raise the unique-pair count, which is what actually limits the inference.
BANDS = [(300.0, 800.0), (500.0, 800.0), (300.0, 1500.0), (500.0, 2000.0),
         (800.0, 3000.0), (300.0, 6000.0)]
HALF = 22.5
N_PERM = 500
N_BOOT = 2000


def collect(product, dates):
    rows = []
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(product, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        if len(RF) < 6:
            continue
        nm = list(RF)
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            d = td.great_circle_km(A, B)
            if not (min(b[0] for b in BANDS) <= d < max(b[1] for b in BANDS)):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((A.name, B.name,
                         td.baseline_azimuth_midpoint_deg(A, B),
                         float(np.cos(np.angle(S))), d))
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{len(dates)}  {len(rows):,} pair-days", flush=True)
    return rows


def aniso(mask, ew, ns, val):
    e, n = mask & ew, mask & ns
    if e.sum() < 15 or n.sum() < 15:
        return None
    return float(val[e].mean() - val[n].mean())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for product in PRODUCTS:
        print(f"\n=== {product} ===")
        rows = collect(product, td.available_dates(product, 2024)[:N_DAYS])
        if len(rows) < 500:
            print("  insufficient"); continue
        sa = np.array([r[0] for r in rows]); sb = np.array([r[1] for r in rows])
        az = np.array([r[2] for r in rows], float); val = np.array([r[3] for r in rows], float)
        ew = np.abs(az - 90.0) <= HALF
        ns = (az <= HALF) | (az >= 180.0 - HALF)

        stations = sorted(set(sa) | set(sb))
        truth = {s: (CLOCKS.get(s) or ["UNKNOWN"])[0] for s in stations}
        known = [s for s in stations if truth[s] in ("QUARTZ", "ATOMIC")]
        n_q = sum(1 for s in known if truth[s] == "QUARTZ")
        print(f"  {len(rows):,} pair-days in {LO:.0f}-{HI:.0f} km | "
              f"{len(known)} stations ({n_q} quartz, {len(known)-n_q} atomic)")

        def qq_mask(lbl):
            return np.array([lbl.get(a) == "QUARTZ" and lbl.get(b) == "QUARTZ"
                             for a, b in zip(sa, sb)])

        m_qq = qq_mask(truth)
        pid = np.array([f"{a}-{b}" for a, b in zip(sa, sb)])
        dist = np.array([r[4] for r in rows], float)

        band_res = {}
        print("  %-14s %9s %7s %7s %8s %7s %9s" % ("band(km)", "QQ_aniso", "nEW", "nNS", "null_mu", "z", "p"))
        for blo, bhi in BANDS:
            inb = (dist >= blo) & (dist < bhi)
            o = aniso(m_qq & inb, ew, ns, val)
            if o is None:
                continue
            ne = len(np.unique(pid[m_qq & inb & ew])); nn = len(np.unique(pid[m_qq & inb & ns]))
            nl = []
            for _ in range(N_PERM):
                perm = list(known); RNG.shuffle(perm)
                lb = {st: ("QUARTZ" if i < n_q else "ATOMIC") for i, st in enumerate(perm)}
                v = aniso(qq_mask(lb) & inb, ew, ns, val)
                if v is not None:
                    nl.append(v)
            nl = np.array(nl)
            zz = float((o - nl.mean()) / (nl.std() + 1e-30)) if len(nl) else None
            pp = float(np.mean(nl >= o)) if len(nl) else None
            band_res[f"{blo:.0f}-{bhi:.0f}"] = {
                "qq_anisotropy": o, "n_ew_pairs": int(ne), "n_ns_pairs": int(nn),
                "null_mean": float(nl.mean()) if len(nl) else None,
                "z": zz, "p": pp}
            print("  %-14s %+9.4f %7d %7d %+8.4f %+7.2f %9.4f" % (
                f"{blo:.0f}-{bhi:.0f}", o, ne, nn, nl.mean(), zz, pp))

        inb0 = (dist >= LO) & (dist < HI)
        m_qq = m_qq & inb0
        obs = aniso(m_qq, ew, ns, val)
        nq_ew = len(np.unique(pid[m_qq & ew])); nq_ns = len(np.unique(pid[m_qq & ns]))

        # --- clock-label permutation null: geometry preserved, association destroyed
        nulls = []
        for _ in range(N_PERM):
            perm = list(known); RNG.shuffle(perm)
            lbl = {s: ("QUARTZ" if i < n_q else "ATOMIC") for i, s in enumerate(perm)}
            v = aniso(qq_mask(lbl), ew, ns, val)
            if v is not None:
                nulls.append(v)
        nulls = np.array(nulls)
        z = float((obs - nulls.mean()) / (nulls.std() + 1e-30)) if len(nulls) else None
        p = float(np.mean(nulls >= obs)) if len(nulls) else None
        print(f"  clock-label permutation null: mean {nulls.mean():+.4f} "
              f"sd {nulls.std():.4f}  ->  z = {z:+.2f}, p = {p:.4f}")

        # --- pair-clustered bootstrap on the QQ contrast
        by = defaultdict(list)
        for i in np.where(m_qq)[0]:
            by[pid[i]].append(i)
        up = np.array(list(by)); bs = []
        for _ in range(N_BOOT):
            pk = RNG.choice(up, len(up), replace=True)
            ii = np.concatenate([by[q] for q in pk])
            e = ew[ii]; n = ns[ii]
            if e.sum() >= 15 and n.sum() >= 15:
                bs.append(val[ii][e].mean() - val[ii][n].mean())
        bs = np.array(bs)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 50 else None
        print(f"  pair-clustered CI95 {ci}  P(>0) = {np.mean(bs>0):.3f}")

        arms[product] = {
            "band_sweep": band_res,
            "n_pair_days": len(rows), "n_stations": len(known),
            "n_quartz": n_q, "n_atomic": len(known) - n_q,
            "qq_anisotropy": obs,
            "n_unique_qq_ew_pairs": int(nq_ew), "n_unique_qq_ns_pairs": int(nq_ns),
            "clock_label_perm_null_mean": float(nulls.mean()) if len(nulls) else None,
            "clock_label_perm_null_sd": float(nulls.std()) if len(nulls) else None,
            "clock_label_perm_z": z, "clock_label_perm_p": p,
            "pair_clustered_ci95": ci,
            "pair_clustered_p_gt_0": float(np.mean(bs > 0)) if len(bs) else None,
            "survives_geography_control": bool(p is not None and p < 0.05),
            "significant_pair_clustered": bool(ci and ci[0] > 0),
        }

    ok = [k for k, v in arms.items() if v["survives_geography_control"]]
    sig = [k for k, v in arms.items() if v["significant_pair_clustered"]]
    verdict = {
        "question": "is the quartz-only 500-800 km anisotropy screening, or station geography?",
        "arms_surviving_clock_label_permutation": ok,
        "arms_significant_pair_clustered": sig,
        "n_arms": len(arms),
        "conclusion": ("screening: survives on " + ", ".join(ok)) if ok else
                      "not separable from station geography on any arm",
    }
    OUT.write_text(json.dumps({
        "step": 76, "title": "Clock-label permutation control on the quartz-only anisotropy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "band_km": [LO, HI], "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
