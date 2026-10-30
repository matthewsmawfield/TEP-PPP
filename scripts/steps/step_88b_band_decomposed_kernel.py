#!/usr/bin/env python3
"""
STEP 88  --  Band-decomposed kernels: one spatial structure per frequency regime.

Steps 86-87 established the route -- measured structure enters the kernel through
the form the physics dictates, and each addition buys held-out skill.  Two
measurements are still averaged away by using ONE kernel over the whole
10-500 uHz band.

  FREQUENCY-DEPENDENT SCREENING (Section 3.5f).  The quartz/atomic contrast is
  3.8x in the low-frequency band, 1.2x in the mid band and 0.9x at high frequency
  (where the residual is true oscillator noise, not field).  A single
  sqrt(f_i f_j) factor for the whole band therefore applies a low-frequency
  screening contrast to high-frequency noise that has none.

  TWO LENGTH SCALES (Section 3.5d).  Phase-lock extends to ~2,800 km while
  coherent POWER decays at ~1,200 km.  One correlation length cannot describe
  both, and a single kernel fitted across the band splits the difference.

Both point the same way: fit a separate spatial kernel, and separate screening
factors, in each frequency sub-band; predict each band with its own kernel; sum.
Everything is measured on the training half -- the per-band Legendre profiles,
the per-band correlation lengths and the per-band screening factors -- and
applied unchanged to held-out days.

Arms: single-band (the Step 87 winner) against the band-decomposed predictor,
plus a diagnostic table of what each band's spatial structure turns out to be.

Output: results/outputs/step_88b_band_decomposed_kernel.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
import itertools
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_s = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_88b_band_decomposed_kernel.json"
CLOCKS = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
RNG = np.random.default_rng(20260910)

PRODUCT = "clk_cod_ops"
N_TRAIN, N_TEST = 150, 120
N_BINS, LMAX = 40, 6
NUGGET = 0.30
MAX_STN = 90
R_E = 6371.0

# sub-bands inside the Papers 1-2 window, split at the Section 3.5f boundaries
BANDS = [("LF", 10e-6, 60e-6), ("MF", 60e-6, 200e-6), ("HF", 200e-6, 500e-6)]
# The honest baseline is a kernel fitted on the WHOLE Papers 1-2 window.  An
# earlier version used the LF profile as the stand-in for "one global kernel",
# which is a straw man: it measured band-specific-vs-LF, not band-specific-vs-
# global, and made the comparison look better than it is.
FULL = ("FULL", 10e-6, 500e-6)


def legP(l, x):
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x
    Pm1, Pc = np.ones_like(x), x
    for k in range(2, l + 1):
        Pm1, Pc = Pc, ((2 * k - 1) * x * Pc - (k - 1) * Pm1) / k
    return Pc


def band_matrix(ser, names, flo, fhi):
    """Standardised residual matrix restricted to one frequency sub-band."""
    grid, rows, keep = None, [], []
    for n in names:
        if n not in ser:
            continue
        t = np.asarray(ser[n]["t"], float); x = np.asarray(ser[n]["resid"], float)
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 200:
            continue
        if grid is None:
            grid = np.arange(t[ok].min(), t[ok].max(), 30.0)
            if len(grid) < 200:
                return None, [], None
        v = np.interp(grid, t[ok], x[ok])
        tt = np.arange(len(v), dtype=float)
        v = v - np.polyval(np.polyfit(tt, v, 2), tt)
        F = np.fft.rfft(v); fr = np.fft.rfftfreq(len(v), 30.0)
        F[(fr < flo) | (fr > fhi)] = 0.0
        v = np.fft.irfft(F, len(v))
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826
        if not np.isfinite(mad) or mad < 1e-12:
            continue
        if np.max(np.abs(v - med)) > 50.0 * mad:
            continue
        v = np.clip(v, med - 6 * mad, med + 6 * mad) - med
        rows.append(v); keep.append(n)
    if len(keep) < 20:
        return None, [], None
    X = np.array(rows)
    sds = X.std(axis=1); msd = float(np.median(sds))
    ok = sds < 5.0 * msd
    X, keep = X[ok], [n for n, f in zip(keep, ok) if f]
    if len(keep) < 20:
        return None, [], None
    amp = X.std(axis=1).copy()
    # Normalise by a NETWORK-level scale, not per station: dividing each station
    # by its own amplitude destroys the relative weighting the bands must keep
    # when they are summed back together, and reintroducing it afterwards would
    # use the held-out station's own amplitude -- a leak.
    # Standardise per station: kriging with a CORRELATION matrix assumes
    # homoscedastic data, and within a band station amplitudes span two orders of
    # magnitude (H-masers ~5 cm against internal quartz ~4 m).  Predicting a quiet
    # station from noisy neighbours under a correlation kernel otherwise diverges.
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X, keep, amp


def fit_profile(dd, vv):
    edges = np.linspace(0, 20100, N_BINS + 1)
    ctr, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() >= 40:
            ctr.append(0.5 * (lo + hi)); val.append(float(vv[m].mean()))
    if len(ctr) < 8:
        return None, None, None
    ctr = np.array(ctr); val = np.array(val)
    A = np.column_stack([legP(l, np.cos(ctr / R_E)) for l in range(LMAX + 1)])
    a, *_ = np.linalg.lstsq(A, val, rcond=None)
    # a crude single-scale summary for the diagnostic table
    pos = val > 0.5 * max(val.max(), 1e-9)
    lam = float(ctr[pos].max()) if pos.any() else float("nan")
    return np.clip(a, 0.0, None), lam, (ctr, val)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)
    train, test = dates[:N_TRAIN], dates[N_TRAIN:N_TRAIN + N_TEST]
    print(f"STEP 88  |  train {len(train)}, test {len(test)} days, "
          f"{len(BANDS)} sub-bands")

    # ---- training: per-band spatial profile and per-band screening ----------
    ALL = list(BANDS) + [FULL]
    acc = {b[0]: {"d": [], "v": []} for b in ALL}
    amps = {b[0]: {"QUARTZ": [], "ATOMIC": []} for b in ALL}
    for k, tag in enumerate(train):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        names = sorted(ser)
        for bname, flo, fhi in ALL:
            X, keep, amp = band_matrix(ser, names, flo, fhi)
            if X is None:
                continue
            S = [ser[n]["station"] for n in keep]
            # FIELD SHARE, not amplitude.  A station with large local oscillator
            # noise has MORE total variance but a LOWER field fraction, so an
            # amplitude ratio is the wrong proxy and inverts the weighting (the
            # first version of this step derived f = 0.99 / 0.01 from a 71x
            # amplitude ratio and destroyed the kernel conditioning).  The field
            # share is the squared correlation of a station with the network,
            # which is what kriging actually needs.
            netm = X.mean(axis=0)
            nn_ = np.linalg.norm(netm)
            for r_i, n_ in zip(X, keep):
                ct = (CLOCKS.get(n_) or ["?"])[0]
                if ct not in amps[bname]:
                    continue
                d_ = np.linalg.norm(r_i) * nn_
                if d_ > 1e-12:
                    amps[bname][ct].append(float((np.dot(r_i, netm) / d_) ** 2))
            nsub = min(len(keep), 60)
            for i, j in itertools.combinations(range(nsub), 2):
                ni = np.linalg.norm(X[i]); nj = np.linalg.norm(X[j])
                if ni > 1e-12 and nj > 1e-12:
                    acc[bname]["d"].append(td.great_circle_km(S[i], S[j]))
                    acc[bname]["v"].append(float(np.dot(X[i], X[j]) / (ni * nj)))
        if (k + 1) % 40 == 0:
            print(f"    train {k+1}/{len(train)}", flush=True)

    prof, fld = {}, {}
    print("\n  %-5s %10s %12s %11s %11s %8s" % (
        "band", "n_pairs", "lambda_half", "quartz_f", "atomic_f", "ratio"))
    for bname, flo, fhi in ALL:
        dd = np.array(acc[bname]["d"]); vv = np.array(acc[bname]["v"])
        if len(dd) < 500:
            continue
        a, lam, _ = fit_profile(dd, vv)
        if a is None:
            continue
        prof[bname] = a
        q = float(np.median(amps[bname]["QUARTZ"])) if amps[bname]["QUARTZ"] else np.nan
        at = float(np.median(amps[bname]["ATOMIC"])) if amps[bname]["ATOMIC"] else np.nan
        ratio = q / at if at and np.isfinite(at) and at > 0 else np.nan
        # f is already a variance FRACTION in [0,1]; clip away from 0 so the
        # kernel cannot become singular for one clock class
        fld[bname] = {"QUARTZ": float(np.clip(q, 0.05, 0.95)) if np.isfinite(q) else 0.5,
                      "ATOMIC": float(np.clip(at, 0.05, 0.95)) if np.isfinite(at) else 0.5}
        print("  %-5s %10d %12.0f %11.4f %11.4f %8.2f" % (
            bname, len(dd), lam, q, at, ratio))

    if len(prof) < 3 or "FULL" not in prof:
        OUT.write_text(json.dumps({"step": 88, "status": "insufficient_bands"}, indent=2))
        print("insufficient bands"); return

    def K_band(bname, D, fvec):
        K = sum(prof[bname][l] * legP(l, np.cos(D / R_E)) for l in range(LMAX + 1))
        k0 = float(sum(prof[bname][l] * legP(l, np.ones(1)) for l in range(LMAX + 1))[0])
        if abs(k0) > 1e-12:
            K = K / k0
        K = K * np.sqrt(np.outer(fvec, fvec)) / max(np.mean(fvec), 1e-9)
        np.fill_diagonal(K, 1.0)
        K = np.clip(0.5 * (K + K.T), -1.0, 1.0)
        w = np.linalg.eigvalsh(K)
        if w.min() < 1e-8:
            # Add a ridge, then renormalise to unit diagonal the CORRECT way:
            # dividing by sqrt(diag) outer-product preserves positive
            # definiteness.  Calling fill_diagonal(K, 1) here instead simply
            # deletes the ridge that was just added and leaves K non-PSD, which
            # is what made the band-decomposed arm diverge.
            K = K + (abs(w.min()) + 1e-3) * np.eye(len(K))
            dg = np.sqrt(np.clip(np.diag(K), 1e-12, None))
            K = K / np.outer(dg, dg)
        return K

    # ---- test: single-band vs band-decomposed ------------------------------
    ARMS = ["zero", "single_band", "band_decomposed"]
    sse = {m: 0.0 for m in ARMS}; n_pred = 0; per_day = []
    for k, tag in enumerate(test):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        names = sorted(ser)[:MAX_STN]
        bands_X = {}
        for bname, flo, fhi in BANDS:
            if bname not in prof:
                continue
            X, keep, amp = band_matrix(ser, names, flo, fhi)
            if X is not None:
                bands_X[bname] = (X, keep, amp)
        if len(bands_X) < 2:
            continue
        common = set.intersection(*[set(v[1]) for v in bands_X.values()])
        common = sorted(common)
        if len(common) < 20:
            continue
        prev = {m: sse[m] for m in ARMS}
        S = [ser[n]["station"] for n in common]
        n = len(common)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = td.great_circle_km(S[i], S[j])
        Xb, Ab = {}, {}
        for bname, (X, keep, amp) in bands_X.items():
            idx = [keep.index(c) for c in common]
            Xb[bname] = X[idx]; Ab[bname] = amp[idx]
        Ks = {b: K_band(b, D, np.array([fld[b].get((CLOCKS.get(c) or ["?"])[0], 0.5)
                                        for c in common])) for b in Xb}
        # Evaluate EACH BAND separately.  The question is whether a band-specific
        # spatial kernel beats one global kernel within that band -- summing
        # standardised bands would not reconstruct a physical residual.
        # baseline: the kernel fitted on the full 10-500 uHz window
        Kglob = K_band("FULL", D, np.ones(n)) if "FULL" in prof \
            else K_band(list(Xb)[0], D, np.ones(n))
        for t_ in range(n):
            o = [q for q in range(n) if q != t_]
            for b in Xb:
                y = Xb[b][t_]
                sse["zero"] += float(y @ y)
                for tag_, Km in (("single_band", Kglob), ("band_decomposed", Ks[b])):
                    Koo = Km[np.ix_(o, o)].copy()
                    Koo[np.diag_indices_from(Koo)] += NUGGET
                    try:
                        w_ = np.linalg.solve(Koo, Km[o, t_])
                    except np.linalg.LinAlgError:
                        w_ = np.zeros(len(o))
                    pr = w_ @ Xb[b][o]
                    if not np.all(np.isfinite(pr)):
                        pr = np.zeros_like(y)
                    r_ = y - pr
                    sse[tag_] += float(r_ @ r_)
            n_pred += 1
        per_day.append({"n": int(n), "sse": {m: float(sse[m] - prev[m]) for m in ARMS}})
        if (k + 1) % 30 == 0:
            print(f"    test {k+1}/{len(test)}, {n_pred} predictions", flush=True)

    rms = {m: float(np.sqrt(sse[m] / max(n_pred, 1))) for m in ARMS}
    print(f"\n  {n_pred} predictions over {len(per_day)} held-out days")
    for m in ARMS:
        print("  %-18s RMS %10.5f   vs zero %+7.2f%%" % (
            m, rms[m], 100 * (1 - rms[m] / rms["zero"])))
    gain = float(100 * (1 - rms["band_decomposed"] / rms["single_band"]))
    boots = []
    idx = list(range(len(per_day)))
    for _ in range(4000):
        pick = RNG.choice(idx, len(idx), replace=True)
        a1 = sum(per_day[q]["sse"]["single_band"] for q in pick)
        a2 = sum(per_day[q]["sse"]["band_decomposed"] for q in pick)
        if a1 > 0:
            boots.append(100 * (1 - np.sqrt(a2) / np.sqrt(a1)))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    print(f"\n  band-decomposed vs single-band: {gain:+.2f}%  "
          f"CI95 [{ci[0]:+.2f}, {ci[1]:+.2f}]  P(>0) = {np.mean(np.array(boots)>0):.3f}")

    verdict = {
        "n_predictions": n_pred, "n_test_days": len(per_day),
        "bands": {b[0]: [b[1], b[2]] for b in BANDS},
        "per_band_field_share": fld,
        "rms": rms,
        "band_decomposed_vs_single_band_pct": gain,
        "bootstrap_ci95": ci,
        "bootstrap_p_gt_0": float(np.mean(np.array(boots) > 0)),
        "helps": bool(ci[0] > 0),
        "causality": "per-band profiles and screening factors fitted on the training half only",
    }
    OUT.write_text(json.dumps({
        "step": 88, "title": "Band-decomposed spatial kernels",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
