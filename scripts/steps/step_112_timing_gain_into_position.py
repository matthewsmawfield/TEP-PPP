#!/usr/bin/env python3
"""
STEP 112 -- Propagating the TIMING gain into POSITION precision.

Steps 110/111 measured a 3x better clock under holdover (11.4 ns against
34.6 ns at 4 h, atomic).  The live EKF gave the TEP and persistence arms the
SAME prior variance, so only the better prior MEAN reached the position solve
and the better ACCURACY -- most of the value -- was discarded.

Formal position dilution as a function of the clock prior the solver is given,
median over 4000 random real-geometry skies:

    k sats   standard 34.6 ns    TEP 11.4 ns      gain
    -----    ----------------    -----------    -------
      4      PDOP 8.499          4.829          +43.2 %
      5           4.811          3.852          +19.9 %
      6           3.522          3.144          +10.7 %
      8           2.519          2.348           +6.8 %
     12           1.729          1.682           +2.8 %   <- full sky

VDOP gains more (6.163 -> 3.397 at k=4, +45 %), which matters because height is
GNSS's weakest axis and the receiver clock is precisely what it trades against.

Two corrections to earlier reasoning this forces:

  * "a clock prior adds nothing at full sky" is wrong.  It is small (+2.8 % at
    k=12) but positive, and it is the VERTICAL that benefits.
  * "the k=4 gain is a degrees-of-freedom effect, not the prior's accuracy" is
    only half right.  Removing the unknown is worth the free-running -> standard
    step (10.121 -> 8.499).  The prior's ACCURACY is worth MORE than that:
    8.499 -> 4.829.

The sigma used is each arm's MEASURED held-out clock RMSE, so per-epoch
excursions of the field residual are already inside it -- this is not an
optimistic formal variance.

Run:  python3 scripts/steps/step_112_timing_gain_into_position.py
"""
import sys, math
import numpy as np

# Clock prior sigmas from step_110 (4 h outage, atomic), converted m <- ns
C = 0.299792458
SIG = {
    "free-running (no prior)": None,
    "standard holdover 34.6 ns": 34.6 * C,
    "TEP holdover 11.4 ns": 11.4 * C,
    "TEP 8h 14.0 ns": 14.0 * C,
}

rng = np.random.default_rng(7)


def dop(los, clock_sigma_m=None, sigma_obs=1.0):
    """PDOP / VDOP with an optional clock prior.

    Unknowns are (x, y, z, clk).  A clock prior adds one pseudo-observation
    row weighting the clock state by 1/sigma_c.
    """
    n = len(los)
    H = np.hstack([-los, np.ones((n, 1))])
    W = np.eye(n) / sigma_obs ** 2
    A = H.T @ W @ H
    if clock_sigma_m is not None:
        A = A + np.diag([0, 0, 0, 1.0 / clock_sigma_m ** 2])
    try:
        Q = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None, None
    if not np.all(np.isfinite(Q)) or np.any(np.diag(Q)[:3] < 0):
        return None, None
    return math.sqrt(np.sum(np.diag(Q)[:3])), math.sqrt(Q[2, 2])


def sky(k, rng, min_el=10.0):
    """k satellites drawn from a realistic sky, ENU line-of-sight."""
    out = []
    while len(out) < k:
        az = rng.uniform(0, 2 * math.pi)
        el = math.radians(rng.uniform(min_el, 85.0))
        out.append([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az),
                    math.sin(el)])
    return np.array(out)


print("POSITION DILUTION vs the CLOCK PRIOR the solver is given")
print("(median over 4000 random skies; lower is better)\n")
print(f"{'k sats':>6s}  {'prior':28s} {'PDOP':>8s} {'VDOP':>8s}  {'vs standard':>12s}")
print("-" * 74)
for k in (4, 5, 6, 8, 12):
    res = {}
    for label, sig in SIG.items():
        p, v = [], []
        for _ in range(4000):
            los = sky(k, rng)
            a, b = dop(los, sig)
            if a is not None and a < 1e4:
                p.append(a); v.append(b)
        res[label] = (np.median(p) if p else None, np.median(v) if v else None,
                      len(p))
    base = res["standard holdover 34.6 ns"][0]
    for label in SIG:
        pd_, vd_, n = res[label]
        if pd_ is None:
            print(f"{k:6d}  {label:28s} {'singular':>8s} {'-':>8s}")
            continue
        gain = f"{100*(1-pd_/base):+11.1f}%" if base else "-"
        print(f"{k:6d}  {label:28s} {pd_:8.3f} {vd_:8.3f}  {gain:>12s}")
    print()
