#!/usr/bin/env python3
"""
STEP 86  --  Does the measured angular structure improve OPERATIONAL prediction?

Step 81 measured the correlation function over the full sphere and found two
features that every kernel in this pipeline is structurally unable to represent:

  * C(theta) crosses zero near 47 deg and is NEGATIVE from ~5,250 to ~15,750 km
    -- 22 of 40 distance bins, 55% of the sampled range;
  * it rebounds to POSITIVE toward the antipode (+0.057 at 18,250 km).

A monotone decaying kernel (Gaussian or exponential) assigns a small positive
weight everywhere.  At 6,250 km the fitted Gaussian gives +0.021 where the truth
is -0.051: not a small inefficiency but the wrong SIGN, and the rebound is
missed entirely.  Kriging with a mis-signed covariance actively mis-weights
distant stations.

This step asks whether fixing that buys operational accuracy.  Held-out clock
prediction, leave-one-station-out, comparing:

    zero         predict nothing (the baseline any prior must beat)
    mean         unweighted network mean
    gaussian     isotropic Gaussian at the Step 81 lambda -- the current kernel
    exponential  isotropic exponential
    empirical    the measured C(d) profile, negative lobe and rebound included

CAUSALITY.  The empirical profile is built ONLY from the training half of the
year and applied unchanged to the held-out half, so it cannot borrow information
from the epochs it is scored on.  The Gaussian's lambda is fitted on the same
training half for the same reason.

Output: results/outputs/step_86_angular_kernel_operational.json
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

OUT = ROOT / "results" / "outputs" / "step_86_angular_kernel_operational.json"
RNG = np.random.default_rng(20260908)
PRODUCT = "clk_cod_ops"
N_TRAIN, N_TEST = 150, 120
N_BINS = 40
NUGGET = 0.30            # correlation-matrix units; regularises the solve
MAX_STN = 90          # kriging cost is O(n^3) per target


def day_matrix(ser, names):
    """Common-grid, detrended, band-limited residual matrix (n_stn, n_epoch)."""
    grid = None
    rows, keep = [], []
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
                return None, []
        v = np.interp(grid, t[ok], x[ok])
        tt = np.arange(len(v), dtype=float)
        v = v - np.polyval(np.polyfit(tt, v, 2), tt)
        F = np.fft.rfft(v); fr = np.fft.rfftfreq(len(v), 30.0)
        F[(fr < s53.F_LO_HZ) | (fr > s53.F_HI_HZ)] = 0.0
        v = np.fft.irfft(F, len(v))
        # ROBUST QC.  Clock products carry millisecond jumps (~300 km expressed in
        # metres); a quadratic detrend does not touch them, and one such station
        # destroys every kriging system it enters.  Reject on a robust scale and
        # clip the survivors.
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        if not np.isfinite(mad) or mad < 1e-12:
            continue
        if np.max(np.abs(v - med)) > 50.0 * mad:
            continue                                   # jump / reset: drop station
        v = np.clip(v, med - 6 * mad, med + 6 * mad) - med
        sd = np.std(v)
        if not np.isfinite(sd) or sd < 1e-12 or sd > 1e4:
            continue
        rows.append(v); keep.append(n)
    if not rows:
        return None, []
    X = np.array(rows)
    # Network-relative variance cut: a station whose band-limited residual is far
    # above the network median that day is carrying a product artefact, not the
    # field, and one such station dominates every kriging system it enters.
    sds = X.std(axis=1)
    med_sd = float(np.median(sds))
    ok = sds < 5.0 * med_sd
    X, keep = X[ok], [n for n, f in zip(keep, ok) if f]
    if len(keep) < 20:
        return None, []
    # Standardise: kriging assumes a stationary field, so predict the shape of
    # each station's variation rather than its amplitude, and pair that with a
    # CORRELATION matrix (unit diagonal) below.
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X, keep


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)
    train, test = dates[:N_TRAIN], dates[N_TRAIN:N_TRAIN + N_TEST]
    print(f"STEP 86  |  train {len(train)} days, test {len(test)} days")

    # ---- TRAINING: empirical C(d) and the fitted isotropic kernels -----------
    pairs = []
    for k, tag in enumerate(train):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        nm = list(RF)
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
            S = np.sum(np.conj(Fi) * Fj)
            if np.isfinite(S) and S != 0:
                pairs.append((d, float(np.cos(np.angle(S)))))
        if (k + 1) % 50 == 0:
            print(f"    train {k+1}/{len(train)}", flush=True)
    dd = np.array([p[0] for p in pairs]); vv = np.array([p[1] for p in pairs])
    edges = np.linspace(0, 20100, N_BINS + 1)
    ctr, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() >= 50:
            ctr.append(0.5 * (lo + hi)); val.append(float(vv[m].mean()))
    ctr = np.array(ctr); val = np.array(val)
    print(f"  empirical profile: {len(ctr)} bins, "
          f"{int((val < 0).sum())} negative, min {val.min():+.4f}, "
          f"antipodal {val[-1]:+.4f}")

    # isotropic fits on the SAME training data
    from scipy.optimize import curve_fit
    def gauss(d, A, lam, C0): return A * np.exp(-(d / lam) ** 2) + C0
    def expo(d, A, lam, C0): return A * np.exp(-d / lam) + C0
    fits = {}
    for nm_, fn in (("gaussian", gauss), ("exponential", expo)):
        try:
            po, _ = curve_fit(fn, ctr, val, p0=[val.max(), 4000.0, 0.0],
                              bounds=([0, 200, -1], [5, 40000, 1]), maxfev=20000)
            fits[nm_] = po
            print(f"  {nm_}: A={po[0]:.3f} lambda={po[1]:.0f} km C0={po[2]:+.4f}")
        except Exception as e:
            print(f"  {nm_} fit failed: {e}")

    # ---- a VALID covariance carrying the measured angular structure ---------
    # Schoenberg: C(theta) = sum_l a_l P_l(cos theta) on the sphere is positive
    # semi-definite IF AND ONLY IF every a_l >= 0.  The raw empirical profile is
    # not -- Step 81 returns a0 = -0.011 -- so feeding it straight into kriging
    # gives an indefinite system and explosive weights (that is exactly what the
    # first version of this step produced).  Fitting the profile in the Legendre
    # basis and clipping negative coefficients to zero yields the closest valid
    # covariance that still carries the quadrupole, which is the operationally
    # usable form of the Step 81 result.
    R_E = 6371.0
    LMAX = 6

    def legP(l, x):
        if l == 0: return np.ones_like(x)
        if l == 1: return x
        Pm1, Pc = np.ones_like(x), x
        for k_ in range(2, l + 1):
            Pm1, Pc = Pc, ((2 * k_ - 1) * x * Pc - (k_ - 1) * Pm1) / k_
        return Pc

    xt = np.cos(ctr / R_E)
    Aleg = np.column_stack([legP(l, xt) for l in range(LMAX + 1)])
    a_raw, *_ = np.linalg.lstsq(Aleg, val, rcond=None)
    a_psd = np.clip(a_raw, 0.0, None)
    print(f"  Legendre coefficients (raw):  {np.round(a_raw, 4)}")
    print(f"  clipped to PSD (Schoenberg):  {np.round(a_psd, 4)}"
          f"   [{int((a_raw < 0).sum())} negative coefficient(s) removed]")

    def K_of(name, D):
        if name == "gaussian":
            A, lam, C0 = fits["gaussian"]; return A * np.exp(-(D / lam) ** 2) + C0
        if name == "exponential":
            A, lam, C0 = fits["exponential"]; return A * np.exp(-D / lam) + C0
        if name == "legendre_psd":
            x = np.cos(D / R_E)
            return sum(a_psd[l] * legP(l, x) for l in range(LMAX + 1))
        raise ValueError(name)

    # ---- TEST: leave-one-station-out kriging on held-out days ---------------
    METHODS = ["zero", "mean", "gaussian", "exponential", "legendre_psd"]
    sse = {m: 0.0 for m in METHODS}; tot = 0.0; n_pred = 0
    per_day = []          # (day, {method: sse}, n) for a day-clustered bootstrap
    for k, tag in enumerate(test):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        names = sorted(ser)[:MAX_STN]
        X, keep = day_matrix(ser, names)
        if X is None or len(keep) < 20:
            continue
        P = np.array([ser[n]["station"].ecef_km for n in keep])
        S = [ser[n]["station"] for n in keep]
        n = len(keep)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = td.great_circle_km(S[i], S[j])
        # Normalise each kernel to a CORRELATION matrix: unit diagonal, values in
        # [-1, 1].  Without this a fitted constant offset C0 makes the covariance
        # nearly rank-one and the solve returns enormous weights whatever the
        # nugget -- which is what produced the earlier 1e5 RMS on the Gaussian.
        prev = {m: sse[m] for m in METHODS}
        Kc = {}
        for m in ("gaussian", "exponential", "legendre_psd"):
            Km = K_of(m, D)
            k0 = float(K_of(m, np.zeros(1))[0])
            if abs(k0) > 1e-12:
                Km = Km / k0
            np.fill_diagonal(Km, 1.0)
            Kc[m] = np.clip(Km, -1.0, 1.0)
        for t_ in range(n):
            o = [q for q in range(n) if q != t_]
            y = X[t_]
            tot += float(y @ y); n_pred += 1
            sse["zero"] += float(y @ y)
            mpred = X[o].mean(axis=0)
            sse["mean"] += float((y - mpred) @ (y - mpred))
            for m in ("gaussian", "exponential", "legendre_psd"):
                Koo = Kc[m][np.ix_(o, o)].copy()
                Koo[np.diag_indices_from(Koo)] += NUGGET
                kot = Kc[m][o, t_]
                try:
                    w = np.linalg.solve(Koo, kot)
                except np.linalg.LinAlgError:
                    w = np.zeros(len(o))
                r = y - w @ X[o]
                sse[m] += float(r @ r)
        per_day.append({"tag": tag, "n": int(n),
                        "sse": {m: float(sse[m] - prev[m]) for m in METHODS}})
        if (k + 1) % 30 == 0:
            print(f"    test {k+1}/{len(test)}, {n_pred} predictions", flush=True)

    rms = {m: float(np.sqrt(sse[m] / max(n_pred, 1))) for m in METHODS}
    base = rms["zero"]
    skill = {m: float(100.0 * (1 - rms[m] / base)) for m in METHODS}
    print(f"\n  {n_pred} leave-one-station-out predictions on held-out days")
    print("  %-13s %12s %12s" % ("kernel", "RMS", "skill vs zero"))
    for m in METHODS:
        print("  %-13s %12.5f %11.2f%%" % (m, rms[m], skill[m]))
    gain = float(100.0 * (1 - rms["legendre_psd"] / rms["gaussian"]))

    # Day-clustered bootstrap: predictions within a day share the same network
    # state and the same kernel solve, so they are not independent samples.
    boots = {"legendre_psd_vs_gaussian": [], "legendre_psd_vs_zero": [],
             "gaussian_vs_zero": []}
    days = list(range(len(per_day)))
    for _ in range(4000):
        pick = RNG.choice(days, len(days), replace=True)
        acc = {m: 0.0 for m in METHODS}; nn = 0
        for q in pick:
            for m in METHODS:
                acc[m] += per_day[q]["sse"][m]
            nn += per_day[q]["n"]
        if nn == 0:
            continue
        r_ = {m: np.sqrt(acc[m] / nn) for m in METHODS}
        boots["legendre_psd_vs_gaussian"].append(100 * (1 - r_["legendre_psd"] / r_["gaussian"]))
        boots["legendre_psd_vs_zero"].append(100 * (1 - r_["legendre_psd"] / r_["zero"]))
        boots["gaussian_vs_zero"].append(100 * (1 - r_["gaussian"] / r_["zero"]))
    ci = {k_: [float(np.percentile(v_, 2.5)), float(np.percentile(v_, 97.5))]
          for k_, v_ in boots.items() if len(v_) > 50}
    pos = {k_: float(np.mean(np.array(v_) > 0)) for k_, v_ in boots.items() if len(v_) > 50}
    print("\n  day-clustered bootstrap (%d test days):" % len(per_day))
    for k_ in ci:
        print("    %-30s CI95 [%+.2f, %+.2f]%%   P(>0) = %.3f"
              % (k_, ci[k_][0], ci[k_][1], pos[k_]))
    print(f"\n  legendre_psd vs gaussian: {gain:+.2f}%")

    verdict = {
        "n_train_days": len(train), "n_test_days": len(test),
        "n_predictions": n_pred, "max_stations": MAX_STN, "nugget": NUGGET,
        "empirical_profile_bins": int(len(ctr)),
        "empirical_negative_bins": int((val < 0).sum()),
        "fitted_lambda_km": {k_: float(v_[1]) for k_, v_ in fits.items()},
        "rms": rms, "skill_vs_zero_pct": skill,
        "legendre_psd_vs_gaussian_pct": gain,
        "bootstrap_ci95_day_clustered": ci,
        "bootstrap_p_gt_0": pos,
        "n_test_days_used": len(per_day),
        "legendre_psd_beats_gaussian": bool(gain > 0),
        "legendre_coefficients_raw": [float(x) for x in a_raw],
        "legendre_coefficients_psd": [float(x) for x in a_psd],
        "causality": "empirical profile and isotropic lambdas fitted on the training "
                     "half only, applied unchanged to held-out days",
    }
    OUT.write_text(json.dumps({
        "step": 86,
        "title": "Operational value of the measured angular correlation structure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "empirical_profile": [{"d_km": float(a), "C": float(b)} for a, b in zip(ctr, val)],
        "verdict": verdict}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
