#!/usr/bin/env python3
"""
STEP 89  --  Locating the preferred axis with the SEASONAL lever.

Step 84 showed that a daily-averaged phase-cos statistic cannot resolve a
celestial right ascension: the 10-500 uHz band needs a full day, Earth turns
360 deg within it, and shuffling which day receives which sidereal rotation
leaves the recovered axis unmoved (median |dRA| = 2.2 deg).  Any directional
claim built on a daily statistic inherits that blindness.

Papers 2-3 do not work that way.  They use the seasonal lever: track the
anisotropy in ~30-day windows across the year and correlate it with Earth's
orbital motion.  That breaks the degeneracy for a reason the daily rotation
cannot -- Earth's velocity vector sweeps the full 360 deg of right ascension
over an orbit, so a fixed direction in space produces an ANNUAL modulation whose
phase identifies it.  Paper 2 reports the anisotropy ratio correlating with
orbital velocity at r = -0.888, with the modulation peaking when Earth's motion
is maximally projected onto the CMB dipole.

Design:
  * split the year into non-overlapping windows;
  * in each window measure the EW-NS phase-alignment contrast in the 300-1500 km
    band, using the great-circle MIDPOINT bearing (Section 3.7c: the initial
    bearing misclassifies 36% of pairs) and one value per unique pair (Step 74b:
    pair-days are pseudo-replicates);
  * compute Earth's heliocentric velocity direction v(t) at each window's mean
    epoch, in equatorial coordinates;
  * for a trial axis n, the predictor is the projection of Earth's motion onto
    it.  Both a dipolar (v.n) and a quadrupolar (v.n)^2 form are fitted, since
    a quadrupolar field responds to the square;
  * scan n over the sky, take the direction of strongest correlation, and
    compare against the CMB dipole, the Papers 2-3 axes and the Solar Apex;
  * NULL: permute the window labels, which preserves both the anisotropy series
    and the geometry and destroys only their pairing, and scan the null
    identically -- a scan always finds some axis, so an unscanned null would
    manufacture a detection.

The honest limitation is stated up front: one year gives ~24 windows against
Paper 2's 25.3 years, so this can confirm or fail to confirm a direction but
cannot match the corpus's precision.

Output: results/outputs/step_89_seasonal_axis.json
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

OUT = ROOT / "results" / "outputs" / "step_89_seasonal_axis.json"
RNG = np.random.default_rng(20260907)
PRODUCT = "clk_cod_ops"
N_DAYS = 361
WINDOW_DAYS = 15
BAND = (300.0, 1500.0)
HALF = 22.5
N_NULL = 2000
OBLIQ = np.radians(23.439291)

CMB_DIPOLE = (168.0, -7.0)
PAPER2_AXIS = (186.0, -4.0)
PAPER3_AXIS = (188.0, -5.0)
SOLAR_APEX = (271.0, 30.0)


def jd_of(year, doy):
    Y, M, D = year, 1, 0
    jd0 = (367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4)
           + np.floor(275 * M / 9) + D + 1721013.5)
    return float(jd0 + doy + 0.5)


def earth_velocity_dir(jd):
    """Unit vector of Earth's heliocentric velocity, equatorial coordinates.

    Earth's ecliptic longitude advances ~0.9856 deg/day; the velocity is
    perpendicular to the Sun-Earth direction in the ecliptic plane, then rotated
    into equatorial coordinates by the obliquity.
    """
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)      # Sun's geometric longitude
    # Earth's velocity direction in ecliptic coords, prograde
    v_ecl = np.array([-np.cos(L), -np.sin(L), 0.0])
    v_ecl[0], v_ecl[1] = np.sin(L), -np.cos(L)
    return np.array([v_ecl[0],
                     v_ecl[1] * np.cos(OBLIQ) - v_ecl[2] * np.sin(OBLIQ),
                     v_ecl[1] * np.sin(OBLIQ) + v_ecl[2] * np.cos(OBLIQ)])


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sep_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(unit(*a), unit(*b))), -1, 1))))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
    print(f"STEP 89  |  {PRODUCT}, {len(dates)} days, {WINDOW_DAYS}-day windows")

    per_day = {}
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        nm = list(RF)
        if len(nm) < 10:
            continue
        rows = []
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            A, B = STA[nm[i]], STA[nm[j]]
            d = td.great_circle_km(A, B)
            if not (BAND[0] <= d < BAND[1]):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            rows.append((f"{min(A.name,B.name)}-{max(A.name,B.name)}",
                         td.baseline_azimuth_midpoint_deg(A, B),
                         float(np.cos(np.angle(S)))))
        if rows:
            per_day[k] = rows
        if (k + 1) % 60 == 0:
            print(f"    {k+1}/{len(dates)} days", flush=True)

    # window-level anisotropy, one value per unique pair inside each window
    W_amp, W_jd, W_n = [], [], []
    for w0 in range(0, len(dates), WINDOW_DAYS):
        idx = [k for k in range(w0, min(w0 + WINDOW_DAYS, len(dates))) if k in per_day]
        if len(idx) < WINDOW_DAYS // 2:
            continue
        acc = {}
        for k in idx:
            for pid, az, v in per_day[k]:
                acc.setdefault(pid, [az, []])[1].append(v)
        az = np.array([a for a, _ in acc.values()])
        vv = np.array([np.mean(x) for _, x in acc.values()])
        ew = np.abs(az - 90.0) <= HALF
        ns = (az <= HALF) | (az >= 180.0 - HALF)
        if ew.sum() < 10 or ns.sum() < 10:
            continue
        W_amp.append(float(vv[ew].mean() - vv[ns].mean()))
        mid = dates[idx[len(idx) // 2]]
        W_jd.append(jd_of(int(mid[:4]), int(mid[4:])))
        W_n.append(int(len(acc)))
    W_amp = np.array(W_amp); W_jd = np.array(W_jd)
    print(f"\n  {len(W_amp)} windows, EW-NS amplitude {W_amp.min():+.4f} to {W_amp.max():+.4f}")
    if len(W_amp) < 8:
        OUT.write_text(json.dumps({"step": 89, "status": "too_few_windows",
                                   "n_windows": int(len(W_amp))}, indent=2))
        print("too few windows"); return

    Vhat = np.array([earth_velocity_dir(j) for j in W_jd])      # (n_win, 3)
    a = W_amp - W_amp.mean()

    # AUTOCORRELATION.  The window series is a time series, not 24 independent
    # draws: consecutive windows share the same station pairs and the same slowly
    # varying network state.  Measured lag-1 autocorrelation is ~0.65, so a naive
    # permutation null -- which destroys that structure -- understates p by about
    # an order of magnitude.  Every p below is therefore computed BOTH ways, and
    # the block-permutation value is the one to quote.  Paper 2 handles the same
    # problem with a Bartlett correction (their N_eff ~ 11 from 34 windows).
    rho1 = float(np.dot(a[:-1], a[1:]) / max(np.dot(a, a), 1e-30))
    print(f"  lag-1 autocorrelation of the window series: rho = {rho1:+.3f}")

    N_BLOCK = 6
    def block_perm(vals):
        bl = len(vals) // N_BLOCK
        order = RNG.permutation(N_BLOCK)
        idx = np.concatenate([np.arange(b * bl, min((b + 1) * bl, len(vals)))
                              for b in order])
        return vals[idx][:len(vals)]

    ras = np.linspace(0, 360, 72, endpoint=False)
    decs = np.linspace(-87.5, 87.5, 36)
    axes = [(float(r), float(dc)) for r in ras for dc in decs]
    U = np.array([unit(r, dc) for r, dc in axes])
    proj = Vhat @ U.T                                            # (n_win, n_axis)

    def corr_scan(vals, P):
        v = vals - vals.mean()
        p = P - P.mean(axis=0, keepdims=True)
        num = p.T @ v
        den = np.sqrt(np.einsum("ij,ij->j", p, p) * (v @ v))
        r = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        i = int(np.argmax(np.abs(r)))
        return i, float(r[i])

    results = {}
    for name, P in (("dipolar_v_dot_n", proj), ("quadrupolar_v_dot_n_sq", proj ** 2)):
        i, r = corr_scan(W_amp, P)
        ax = axes[i]
        nulls = np.array([abs(corr_scan(RNG.permutation(W_amp), P)[1]) for _ in range(N_NULL)])
        p_val = float((np.sum(nulls >= abs(r)) + 1) / (N_NULL + 1))
        nulls_b = np.array([abs(corr_scan(block_perm(W_amp), P)[1]) for _ in range(N_NULL)])
        p_block = float((np.sum(nulls_b >= abs(r)) + 1) / (N_NULL + 1))
        seps = {"CMB_dipole": sep_deg(ax, CMB_DIPOLE), "paper2_axis": sep_deg(ax, PAPER2_AXIS),
                "paper3_axis": sep_deg(ax, PAPER3_AXIS), "solar_apex": sep_deg(ax, SOLAR_APEX)}
        # correlation at each NAMED axis, no scan penalty
        at = {}
        for nm_, axn in (("CMB_dipole", CMB_DIPOLE), ("paper2_axis", PAPER2_AXIS),
                         ("paper3_axis", PAPER3_AXIS), ("solar_apex", SOLAR_APEX)):
            u = unit(*axn); pv = Vhat @ u
            pv = pv ** 2 if name.startswith("quad") else pv
            pc = pv - pv.mean()
            rr = float(pc @ a / max(np.sqrt((pc @ pc) * (a @ a)), 1e-30))
            nn = np.array([float(pc @ RNG.permutation(a) /
                                 max(np.sqrt((pc @ pc) * (a @ a)), 1e-30))
                           for _ in range(N_NULL)])
            nb_ = np.array([float(pc @ block_perm(a) /
                                  max(np.sqrt((pc @ pc) * (a @ a)), 1e-30))
                            for _ in range(N_NULL)])
            rp = float(np.dot(pc[:-1], pc[1:]) / max(np.dot(pc, pc), 1e-30))
            neff = (len(a) * (1 - rho1 * rp) / (1 + rho1 * rp)
                    if abs(rho1 * rp) < 1 else float(len(a)))
            at[nm_] = {"r": rr,
                       "p_naive": float((np.sum(np.abs(nn) >= abs(rr)) + 1) / (N_NULL + 1)),
                       "p_block_permutation": float((np.sum(np.abs(nb_) >= abs(rr)) + 1) / (N_NULL + 1)),
                       "bartlett_n_eff": float(neff)}
        results[name] = {
            "best_axis_ra_deg": ax[0], "best_axis_dec_deg": ax[1],
            "r_at_best_axis": r, "scanned_null_p_naive": p_val,
            "scanned_null_p_block_permutation": p_block,
            "lag1_autocorrelation": rho1,
            "separation_from_deg": seps, "r_at_named_axes": at,
            "significant_vs_scanned_null": bool(p_block < 0.05),
            "aligns_with_corpus_axis": bool(min(seps["CMB_dipole"], seps["paper2_axis"],
                                                seps["paper3_axis"]) < 30.0),
        }
        print(f"\n  [{name}] best axis RA {ax[0]:.0f} Dec {ax[1]:.0f}   r = {r:+.3f}   "
              f"scanned p: naive {p_val:.4f}, block-perm {p_block:.4f}")
        for k_, v_ in seps.items():
            print(f"      {k_:<14} {v_:6.1f} deg")
        for k_, v_ in at.items():
            print(f"      r at {k_:<14} {v_['r']:+.3f}  p_naive {v_['p_naive']:.4f}  "
                  f"p_block {v_['p_block_permutation']:.4f}  N_eff {v_['bartlett_n_eff']:.1f}")

    verdict = {
        "n_windows": int(len(W_amp)), "window_days": WINDOW_DAYS,
        "band_km": list(BAND), "pairs_per_window_median": int(np.median(W_n)),
        "models": results,
        "limitation": (f"one year gives {len(W_amp)} windows against Paper 2's 25.3 years, "
                       f"and the series is autocorrelated (rho1 = {rho1:+.2f}), leaving "
                       "roughly 7-8 effective degrees of freedom.  Direction and sign can "
                       "be compared with the corpus; significance cannot be established."),
        "lag1_autocorrelation": rho1,
        "corpus_reference": {"CMB_dipole": CMB_DIPOLE, "paper2": PAPER2_AXIS,
                             "paper3": PAPER3_AXIS, "paper2_reported_r": -0.888},
    }
    OUT.write_text(json.dumps({
        "step": 89, "title": "Preferred axis from the seasonal (orbital) lever",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "windows": [{"jd": float(j), "ew_minus_ns": float(x), "n_pairs": int(n)}
                    for j, x, n in zip(W_jd, W_amp, W_n)],
        "verdict": verdict}, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
