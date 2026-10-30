#!/usr/bin/env python3
"""
STEP 90  --  Stacking the measured TEP physics into the operational kernel.

Step 86 showed the route: a structural physics result (the shape of C(theta) on
the sphere) identified a sign error in the operational kernel, and correcting it
turned a predictor that was doing no useful work (Gaussian, -0.78% against
predicting nothing) into one that does (+2.73%, and +3.48% over the Gaussian).

Two further measured results are still outside the kernel, and both have a
principled covariance form:

  OSCILLATOR SCREENING (Section 3.5f).  If resid_i = s_i * field + noise_i and
  f_i is the fraction of station i's variance carried by the field, then for
  standardised series the correlation is

        rho_ij = sqrt(f_i f_j) * C(d_ij)

  Step 65 measures f directly as the network-predictable fraction: quartz 0.6152,
  atomic 0.5352.  The screening result therefore enters as a per-station factor,
  not a fudge -- a quartz pair should correlate ~1.15x more strongly than an
  atomic pair at the same separation.

  ANISOTROPY (Section 3.7c).  East-west pairs correlate more strongly than
  north-south at matched distance.  Rather than impose a stretch ratio, the
  training half is used to fit SEPARATE Legendre profiles for EW-oriented and
  NS-oriented baselines, and the kernel interpolates between them by orientation.

Both are fitted on the training half only and applied unchanged to held-out days.
Positive semi-definiteness is enforced explicitly: an orientation-dependent
kernel is not guaranteed PSD, so the smallest eigenvalue is checked and a ridge
added when required (the Legendre coefficients themselves are clipped to be
non-negative per Schoenberg, as in Step 86).

Arms: base (Step 86 winner) / +screening / +anisotropy / +both.

Output: results/outputs/step_90b_physics_kernel_stack.json
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
_s86 = importlib.util.spec_from_file_location(
    "s86", ROOT / "scripts" / "steps" / "step_86_angular_kernel_operational.py")
s86 = importlib.util.module_from_spec(_s86)
_s86.loader.exec_module(s86)

OUT = ROOT / "results" / "outputs" / "step_90b_physics_kernel_stack.json"
CLOCKS = json.loads((ROOT / "data" / "meta" / "igs_clock_type_full.json").read_text())
RNG = np.random.default_rng(20260909)

PRODUCT = "clk_cod_ops"
N_TRAIN, N_TEST = 150, 120
N_BINS, LMAX = 40, 6
NUGGET = 0.30
MAX_STN = 90
R_E = 6371.0
HALF = 45.0                       # sector half-width for the EW/NS profiles

# measured field share of each clock type (Step 65 network-predictable fraction)
F_FIELD = {"QUARTZ": 0.6152, "ATOMIC": 0.5352}
F_DEFAULT = 0.575


def legP(l, x):
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x
    Pm1, Pc = np.ones_like(x), x
    for k in range(2, l + 1):
        Pm1, Pc = Pc, ((2 * k - 1) * x * Pc - (k - 1) * Pm1) / k
    return Pc


def fit_profile(dd, vv):
    """Binned C(d) -> non-negative Legendre coefficients (Schoenberg PSD)."""
    edges = np.linspace(0, 20100, N_BINS + 1)
    ctr, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() >= 40:
            ctr.append(0.5 * (lo + hi)); val.append(float(vv[m].mean()))
    if len(ctr) < 8:
        return None
    ctr = np.array(ctr); val = np.array(val)
    A = np.column_stack([legP(l, np.cos(ctr / R_E)) for l in range(LMAX + 1)])
    a, *_ = np.linalg.lstsq(A, val, rcond=None)
    return np.clip(a, 0.0, None)


def K_from(a, D):
    x = np.cos(D / R_E)
    return sum(a[l] * legP(l, x) for l in range(LMAX + 1))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)
    train, test = dates[:N_TRAIN], dates[N_TRAIN:N_TRAIN + N_TEST]
    print(f"STEP 90  |  train {len(train)}, test {len(test)} days")

    dd, vv, az_ = [], [], []
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
            A_, B_ = STA[nm[i]], STA[nm[j]]
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            dd.append(td.great_circle_km(A_, B_))
            vv.append(float(np.cos(np.angle(S))))
            az_.append(td.baseline_azimuth_midpoint_deg(A_, B_))
        if (k + 1) % 50 == 0:
            print(f"    train {k+1}/{len(train)}", flush=True)
    dd = np.array(dd); vv = np.array(vv); az_ = np.array(az_)

    a_iso = fit_profile(dd, vv)
    ew = np.abs(az_ - 90.0) <= HALF
    ns = ~ew
    a_ew = fit_profile(dd[ew], vv[ew])
    a_ns = fit_profile(dd[ns], vv[ns])
    print(f"  isotropic Legendre coefficients: {np.round(a_iso, 4)}")
    print(f"  EW profile: {np.round(a_ew, 4)}")
    print(f"  NS profile: {np.round(a_ns, 4)}")
    print(f"  screening factors sqrt(f_i f_j): QQ {F_FIELD['QUARTZ']:.4f} "
          f"AA {F_FIELD['ATOMIC']:.4f} ratio {F_FIELD['QUARTZ']/F_FIELD['ATOMIC']:.3f}")

    def build(D, AZ, fvec, use_aniso, use_screen):
        if use_aniso and a_ew is not None and a_ns is not None:
            wgt = np.cos(np.radians(2.0 * (AZ - 90.0))) * 0.5 + 0.5   # 1 at EW, 0 at NS
            K = wgt * K_from(a_ew, D) + (1 - wgt) * K_from(a_ns, D)
            k0 = 0.5 * (K_from(a_ew, np.zeros(1))[0] + K_from(a_ns, np.zeros(1))[0])
        else:
            K = K_from(a_iso, D)
            k0 = float(K_from(a_iso, np.zeros(1))[0])
        if abs(k0) > 1e-12:
            K = K / k0
        if use_screen:
            K = K * np.sqrt(np.outer(fvec, fvec)) / np.mean(fvec)
        np.fill_diagonal(K, 1.0)
        K = np.clip(K, -1.0, 1.0)
        K = 0.5 * (K + K.T)
        w = np.linalg.eigvalsh(K)
        if w.min() < 1e-8:                 # orientation blending can break PSD
            K = K + (abs(w.min()) + 1e-6) * np.eye(len(K))
            K = K / K[0, 0]
            np.fill_diagonal(K, 1.0)
        return K

    ARMS = ["zero", "mean", "base", "screening", "anisotropy", "both"]
    CFG = {"base": (False, False), "screening": (False, True),
           "anisotropy": (True, False), "both": (True, True)}
    sse = {m: 0.0 for m in ARMS}; n_pred = 0; per_day = []
    for k, tag in enumerate(test):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        names = sorted(ser)[:MAX_STN]
        X, keep = s86.day_matrix(ser, names)
        if X is None or len(keep) < 20:
            continue
        S = [ser[n]["station"] for n in keep]
        n = len(keep)
        D = np.zeros((n, n)); AZ = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = td.great_circle_km(S[i], S[j])
                a_ = td.baseline_azimuth_midpoint_deg(S[i], S[j])
                AZ[i, j] = AZ[j, i] = a_
        fvec = np.array([F_FIELD.get((CLOCKS.get(nn) or ["?"])[0], F_DEFAULT)
                         for nn in keep])
        Ks = {m: build(D, AZ, fvec, *CFG[m]) for m in CFG}
        prev = {m: sse[m] for m in ARMS}
        for t_ in range(n):
            o = [q for q in range(n) if q != t_]
            y = X[t_]; n_pred += 1
            sse["zero"] += float(y @ y)
            mp = X[o].mean(axis=0)
            sse["mean"] += float((y - mp) @ (y - mp))
            for m in CFG:
                Koo = Ks[m][np.ix_(o, o)].copy()
                Koo[np.diag_indices_from(Koo)] += NUGGET
                try:
                    w_ = np.linalg.solve(Koo, Ks[m][o, t_])
                except np.linalg.LinAlgError:
                    w_ = np.zeros(len(o))
                r_ = y - w_ @ X[o]
                sse[m] += float(r_ @ r_)
        per_day.append({"n": int(n), "sse": {m: float(sse[m] - prev[m]) for m in ARMS}})
        if (k + 1) % 30 == 0:
            print(f"    test {k+1}/{len(test)}, {n_pred} predictions", flush=True)

    rms = {m: float(np.sqrt(sse[m] / max(n_pred, 1))) for m in ARMS}
    skill = {m: float(100 * (1 - rms[m] / rms["zero"])) for m in ARMS}
    print(f"\n  {n_pred} predictions over {len(per_day)} held-out days")
    print("  %-12s %12s %14s %14s" % ("kernel", "RMS", "vs zero", "vs base"))
    for m in ARMS:
        vb = 100 * (1 - rms[m] / rms["base"]) if m not in ("zero", "mean") else float("nan")
        print("  %-12s %12.5f %13.2f%% %13s" % (
            m, rms[m], skill[m], f"{vb:+.2f}%" if np.isfinite(vb) else "-"))

    boots = {m: [] for m in ("screening", "anisotropy", "both")}
    idx = list(range(len(per_day)))
    for _ in range(4000):
        pick = RNG.choice(idx, len(idx), replace=True)
        acc = {m: 0.0 for m in ARMS}; nn = 0
        for q in pick:
            for m in ARMS:
                acc[m] += per_day[q]["sse"][m]
            nn += per_day[q]["n"]
        if nn == 0:
            continue
        rr = {m: np.sqrt(acc[m] / nn) for m in ARMS}
        for m in boots:
            boots[m].append(100 * (1 - rr[m] / rr["base"]))
    ci = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for m, v in boots.items() if len(v) > 50}
    pos = {m: float(np.mean(np.array(v) > 0)) for m, v in boots.items() if len(v) > 50}
    print("\n  day-clustered bootstrap, gain over base:")
    for m in ci:
        print("    %-12s CI95 [%+.2f, %+.2f]%%   P(>0) = %.3f" % (m, ci[m][0], ci[m][1], pos[m]))

    verdict = {
        "n_predictions": n_pred, "n_test_days": len(per_day),
        "rms": rms, "skill_vs_zero_pct": skill,
        "gain_over_base_pct": {m: float(100 * (1 - rms[m] / rms["base"])) for m in CFG},
        "bootstrap_ci95_vs_base": ci, "bootstrap_p_gt_0": pos,
        "screening_factors": F_FIELD,
        "legendre_isotropic": [float(x) for x in a_iso],
        "legendre_ew": [float(x) for x in a_ew] if a_ew is not None else None,
        "legendre_ns": [float(x) for x in a_ns] if a_ns is not None else None,
        "causality": "all profiles and factors fitted on the training half only",
    }
    OUT.write_text(json.dumps({
        "step": 90, "title": "Stacking screening and anisotropy into the operational kernel",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
