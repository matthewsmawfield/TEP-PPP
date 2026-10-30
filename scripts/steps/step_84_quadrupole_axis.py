#!/usr/bin/env python3
"""
STEP 84  --  Where does the quadrupole point?  Scanning the preferred axis and
             testing it against the CMB dipole.

Step 81 established that the angular correlation function is quadrupole-
dominated: expanding C(theta) in Legendre polynomials, adding the P2 term lifts
R2 from 0.240 to 0.894 (dAIC = -76.7, a2 = +0.117), and the measured shape --
positive at 0 deg, minimum at 90 deg, positive at 180 deg -- is exactly
P2(cos theta).  A field with a preferred axis produces that; an isotropic one
does not.

But a quadrupole in SEPARATION angle does not by itself locate an axis.  This
step asks the question that does: taking each baseline's orientation in the
INERTIAL frame, is there a direction n on the sky such that the phase-alignment
index varies as P2(cos psi), psi being the angle between the baseline and n?

That is a direct test of the corpus's headline directional claim.  Papers 2 and 3
both report a preferred axis near the CMB dipole:

    CMB dipole (369 km/s)      RA 168 deg, Dec  -7 deg
    Paper 2 best fit           RA 186 deg, Dec  -4 deg   (18.2 deg from CMB)
    Paper 3 best fit           RA 188 deg, Dec  -5 deg   (20.0 deg from CMB)
    Solar Apex                 strongly disfavoured (86.5 deg in Paper 2)

Method:
  * phase-alignment index cos(arg(sum_f S_ij(f))) per pair-day, imported from
    Step 53 so the statistic cannot drift;
  * each pair's ECEF baseline rotated into the inertial frame by the Greenwich
    mean sidereal time of that day, so a fixed celestial axis stays fixed;
  * P2 is even under both n -> -n and baseline reversal, which is exactly right
    for an unordered station pair with no intrinsic direction;
  * a full-sky grid scan over (RA, Dec), fitting C = a + b*P2(cos psi) at each
    trial axis and taking the direction that maximises |b|;
  * null: the same scan with the pair-day phase-cos values shuffled, which
    preserves the sky coverage and the baseline geometry and destroys only the
    association between orientation and correlation.  Because a grid scan
    maximises over many directions it will always find SOME axis, so the null
    must be scanned identically -- comparing a scanned maximum against an
    unscanned null would manufacture a detection.

Output: results/outputs/step_84_quadrupole_axis.json
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

OUT = ROOT / "results" / "outputs" / "step_84_quadrupole_axis.json"
RNG = np.random.default_rng(20260906)
PRODUCT = "clk_cod_ops"
N_DAYS = 60
N_RA, N_DEC = 36, 18          # 10 deg sky grid
N_NULL = 200

CMB_DIPOLE = (168.0, -7.0)
PAPER2_AXIS = (186.0, -4.0)
PAPER3_AXIS = (188.0, -5.0)
SOLAR_APEX = (271.0, 30.0)


def gmst_deg(year, doy):
    """Greenwich mean sidereal time at 12h UT of the given day, degrees.

    Uses the standard Julian-day expression evaluated at Jan 0 of the year.  An
    earlier version of this function used a mis-stated variant that placed
    2024 DOY 100 at JD 2460416.0 instead of the correct 2460410.0 -- six days
    out, which rotates every recovered right ascension by about 6 degrees.
    """
    Y, M, D = year, 1, 0                       # Jan 0 = 31 Dec of Y-1
    jd0 = (367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4)
           + np.floor(275 * M / 9) + D + 1721013.5)
    jd = jd0 + doy + 0.5                       # 12h UT on day `doy`
    return float((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sep_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(unit(*a), unit(*b))), -1, 1))))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
    B, V, D, Becef, DAYIDX = [], [], [], [], []
    for k, tag in enumerate(dates):
        try:
            ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
        except Exception:
            continue
        RF, STA = s53._spectra(ser)
        nm = list(RF)
        if len(nm) < 10:
            continue
        g = np.radians(gmst_deg(int(tag[:4]), int(tag[4:])))
        # ECEF -> inertial: rotate about Z by GMST
        Rz = np.array([[np.cos(g), -np.sin(g), 0.0],
                       [np.sin(g), np.cos(g), 0.0], [0.0, 0.0, 1.0]])
        for i, j in itertools.combinations(range(len(nm)), 2):
            Fi, Fj = RF[nm[i]], RF[nm[j]]
            if Fi.shape != Fj.shape:
                continue
            d = td.great_circle_km(STA[nm[i]], STA[nm[j]])
            if not (s53.DMIN_KM <= d <= s53.DMAX_KM):
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S) or S == 0:
                continue
            bvec = STA[nm[j]].ecef_km - STA[nm[i]].ecef_km
            n_ = np.linalg.norm(bvec)
            if n_ < 1e-6:
                continue
            Becef.append(bvec / n_)
            DAYIDX.append(k)
            B.append(Rz @ (bvec / n_))
            V.append(float(np.cos(np.angle(S))))
            D.append(d)
        if (k + 1) % 15 == 0:
            print(f"  {k+1}/{len(dates)} days, {len(V):,} pair-days", flush=True)
    B = np.array(B); V = np.array(V); D = np.array(D); Becef = np.array(Becef)

    # DISTANCE CONTROL.  V depends strongly on separation -- that is the lambda
    # result -- and in a geographically clustered network baseline ORIENTATION is
    # correlated with separation, so an apparent preferred axis can be distance
    # structure leaking through that correlation.  Subtract the binned distance
    # mean first; the axis is then sought only in what distance cannot explain.
    edges = np.geomspace(max(D.min(), 1.0), D.max() * 1.001, 41)
    ib = np.clip(np.digitize(D, edges) - 1, 0, len(edges) - 2)
    Vd = V.copy()
    for b_ in range(len(edges) - 1):
        m_ = ib == b_
        if m_.sum() >= 20:
            Vd[m_] -= V[m_].mean()
    print(f"  distance control: removed binned mean over {len(edges)-1} log bins")
    print(f"\n{len(V):,} pair-days with inertial baseline orientation")

    # CLOSED FORM, no grid scan.  P2(cos psi) = (3(b.u)^2 - 1)/2, so the
    # regression numerator sum_i (V_i - Vbar) P2_i is exactly (3/2) u^T M u with
    #     M = sum_i (V_i - Vbar) b_i b_i^T          (a single 3x3 matrix)
    # because the -1/2 term drops against sum_i (V_i - Vbar) = 0.  The axis that
    # maximises the quadrupole response is therefore the principal eigenvector of
    # M, obtained exactly rather than searched, and each null costs one pass
    # instead of one pass per trial axis.  (A literal (n_pair x n_axis) scan here
    # would allocate 2.4e6 x 648 floats.)
    Vc = Vd - Vd.mean()      # distance-controlled
    M = (B * Vc[:, None]).T @ B
    M1 = B.T @ B
    w, Q = np.linalg.eigh(M)
    k_ = int(np.argmax(np.abs(w)))
    u_best = Q[:, k_]
    lam_best = float(w[k_])

    def axis_to_radec(u):
        u = u / np.linalg.norm(u)
        if u[2] < 0:
            u = -u                      # an axis has no sign; fix a hemisphere
        ra = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
        dec = float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
        return ra, dec

    best_axis = axis_to_radec(u_best)
    denom = float(1.5 * (u_best @ M1 @ u_best) - 0.5 * len(V))
    bbest = float(1.5 * lam_best / max(abs(denom), 1e-30))
    print(f"  principal axis: RA {best_axis[0]:.1f} deg, Dec {best_axis[1]:.1f} deg")
    print(f"  quadrupole eigenvalue {lam_best:+.4g}   normalised b = {bbest:+.5f}")
    print(f"  eigenvalues (sorted): {np.round(np.sort(w), 4)}")

    # Null: permute the phase-cos values, recompute M, take its top |eigenvalue|.
    # This is the identical statistic (a maximisation over all axes), so the
    # comparison carries no scan penalty.
    nulls = []
    for _ in range(N_NULL):
        vc = RNG.permutation(Vc)
        Mn = (B * vc[:, None]).T @ B
        wn = np.linalg.eigvalsh(Mn)
        nulls.append(float(np.max(np.abs(wn))))
    nulls = np.array(nulls)
    p_val = float((np.sum(nulls >= abs(lam_best)) + 1) / (N_NULL + 1))
    z = float((abs(lam_best) - nulls.mean()) / (nulls.std() + 1e-30))
    print(f"  scanned null |lambda_max|: mean {nulls.mean():.4g}  ->  z = {z:+.2f}, p = {p_val:.4f}")

    # ---- BLINDNESS CONTROL -------------------------------------------------
    # Can a DAILY statistic resolve a celestial axis at all?  phase-cos needs a
    # full day (10-500 uHz), and Earth turns 360 deg within it, so every baseline
    # sweeps a complete circle in inertial space during the measurement.  If the
    # per-day sidereal rotation carries real information, shuffling which day
    # receives which GMST must destroy the recovered axis.
    day_of = np.array(DAYIDX)
    gtrue = np.array([gmst_deg(int(t[:4]), int(t[4:])) for t in dates])

    def axis_under(gmap):
        Bi = np.empty_like(Becef)
        for kk in np.unique(day_of):
            gg = np.radians(gmap[kk]); mm = day_of == kk
            Rz_ = np.array([[np.cos(gg), -np.sin(gg), 0.0],
                            [np.sin(gg), np.cos(gg), 0.0], [0.0, 0.0, 1.0]])
            Bi[mm] = Becef[mm] @ Rz_.T
        Mx = (Bi * Vc[:, None]).T @ Bi
        wx, Qx = np.linalg.eigh(Mx)
        kx = int(np.argmax(np.abs(wx)))
        return axis_to_radec(Qx[:, kx]), float(wx[kx])

    (ra_t, dec_t), lam_t = axis_under(gtrue)
    shuffled = [axis_under(RNG.permutation(gtrue)) for _ in range(5)]
    ra_shuf = [x[0][0] for x in shuffled]
    dra = [min(abs(r - ra_t), 360 - abs(r - ra_t)) for r in ra_shuf]
    blind = bool(np.median(dra) < 25.0)
    print(f"\n  BLINDNESS CONTROL: true axis RA {ra_t:.1f}; shuffled-GMST axes "
          f"RA {[round(r,1) for r in ra_shuf]}")
    print(f"    median |dRA| under shuffle = {np.median(dra):.1f} deg  ->  "
          f"daily statistic is {'BLIND to celestial RA' if blind else 'sensitive'}")

    seps = {"CMB_dipole": sep_deg(best_axis, CMB_DIPOLE),
            "paper2_axis": sep_deg(best_axis, PAPER2_AXIS),
            "paper3_axis": sep_deg(best_axis, PAPER3_AXIS),
            "solar_apex": sep_deg(best_axis, SOLAR_APEX)}
    for k2, v2 in seps.items():
        print(f"    {k2:<14} {v2:6.1f} deg from principal axis")

    # amplitude at each NAMED axis, with no maximisation, so these p-values are
    # the clean ones to quote for a pre-specified direction
    at = {}
    for name, ax in (("CMB_dipole", CMB_DIPOLE), ("paper2_axis", PAPER2_AXIS),
                     ("paper3_axis", PAPER3_AXIS), ("solar_apex", SOLAR_APEX)):
        u = unit(*ax)
        obs = float(u @ M @ u)
        nn = np.array([float(u @ ((B * RNG.permutation(Vc)[:, None]).T @ B) @ u)
                       for _ in range(N_NULL)])
        at[name] = {"u_M_u": obs,
                    "z": float((abs(obs) - np.abs(nn).mean()) / (np.abs(nn).std() + 1e-30)),
                    "p": float((np.sum(np.abs(nn) >= abs(obs)) + 1) / (N_NULL + 1))}
        print(f"    at {name:<14} u^T M u = {obs:+.4g}  z {at[name]['z']:+.2f}  p {at[name]['p']:.4f}")

    verdict = {
        "n_pair_days": int(len(V)), "n_days": len(dates), "product": PRODUCT,
        "distance_controlled": True,
        "best_axis_ra_deg": best_axis[0], "best_axis_dec_deg": best_axis[1],
        "b_at_best_axis": bbest,
        "quadrupole_eigenvalue": lam_best,
        "eigenvalues": [float(x) for x in np.sort(w)],
        "scanned_null_p": p_val, "scanned_null_z": z,
        "separation_from_deg": seps,
        "b_at_named_axes": at,
        "corpus_reference": {"CMB_dipole": CMB_DIPOLE, "paper2": PAPER2_AXIS,
                             "paper3": PAPER3_AXIS, "solar_apex": SOLAR_APEX},
        "axis_significant_vs_scanned_null": bool(p_val < 0.05),
        "blindness_control": {
            "true_axis_ra_deg": ra_t, "true_axis_dec_deg": dec_t,
            "shuffled_gmst_axes_ra_deg": ra_shuf,
            "median_delta_ra_under_shuffle_deg": float(np.median(dra)),
            "daily_statistic_blind_to_RA": blind},
        "axis_claim_permitted": bool(not blind),
        "aligns_with_corpus_axis": (None if blind else
            bool(min(seps["CMB_dipole"], seps["paper2_axis"],
                     seps["paper3_axis"]) < 30.0)),
        "why_no_axis_claim": (
            "phase-cos requires a full day (10-500 uHz) and Earth rotates 360 deg "
            "within it, so each baseline sweeps a circle in inertial space during "
            "the measurement.  Shuffling which day receives which sidereal rotation "
            "leaves the recovered axis unchanged, so the daily statistic carries no "
            "celestial right-ascension information and NO alignment claim -- for or "
            "against the CMB dipole -- can be made from it.  The declination "
            "reflects Earth-fixed network geometry.  Locating the axis needs the "
            "seasonal lever Papers 2-3 use: track the anisotropy over 30-day "
            "windows across the year and correlate with orbital velocity."
            if blind else "daily rotation carries information; axis claim allowed"),
        "note": ("the null is scanned identically to the observation: a grid scan "
                 "always finds some axis, so an unscanned null would manufacture "
                 "a detection"),
    }
    OUT.write_text(json.dumps({
        "step": 84, "title": "Preferred-axis scan of the quadrupole vs the CMB dipole",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict}, indent=2))
    print(f"\nVERDICT: significant vs scanned null: {verdict['axis_significant_vs_scanned_null']}"
          f" | aligns with a corpus axis (<30 deg): {verdict['aligns_with_corpus_axis']}")


if __name__ == "__main__":
    main()
