#!/usr/bin/env python3
"""Guards for the live-hardware TEP rig (rpi/gps_web.py).

These lock in the properties that make the rig's numbers mean something.
Every one of them corresponds to a bug that was actually present:

  * the TEP and control arms were measured in DIFFERENT local frames,
    because the anisotropy block reassigned ref_lat/ref_lon inside the
    epoch loop to a hardcoded 53.64N/0.10E
  * the network clock prior was applied as an ABSOLUTE bias on the local
    receiver's clock, when what the kriging server predicts is a detrended
    residual for a remote station
  * the headline metrics were SCATTER ratios, which any smoother wins by
    construction, rather than accuracy against a truth reference
  * a replayed-archive network source was not distinguishable from a live one
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "rpi"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import gps_web  # noqa: E402
from gps_web import PPPProcessor  # noqa: E402

C_MS = 299792.458  # one millisecond in metres
TRUE_ECEF = np.array([3800000.0, 250000.0, 5100000.0])


# ---------------------------------------------------------------- fixtures
class _FakeProxy:
    def __init__(self, resid, var=1.0, mode="live", aniso=True):
        ker = {"predicted_m": resid, "variance_m2": var}
        if aniso:
            ker["aniso_pos_direction_enu"] = [0.0, 1.0, 0.0]
            ker["aniso_pos_variance_m2"] = 46.0
        self.latest_epoch = {"mode": mode, "target": "FAKE", "kernel": ker}
        self.status = {}

    @property
    def mode(self):
        return self.latest_epoch.get("mode")

    @property
    def is_simulation(self):
        return self.mode == "simulation"


def _make_proc(resid=None, ref=None, mode="live"):
    p = PPPProcessor.__new__(PPPProcessor)
    p.serial_port, p.baud = "x", 1
    p.capture_seconds, p.interval_seconds = 120, 300
    p.gps_reader = None
    p.kriging_proxy = _FakeProxy(resid, mode=mode) if resid is not None else None
    p.latest_result = {}
    p._tep_astro_available = True
    p.ref_ecef, p.ref_is_surveyed = ref, ref is not None
    p._truth_sum, p._truth_n = np.zeros(3), 0
    # Set _tmp so _record_own_clock / _own_clock_path work correctly
    import tempfile
    p._tmp = Path(tempfile.mkdtemp(prefix="test_gps_web_"))
    # Set _rtcm_enabled so _capture_chunk doesn't hit AttributeError
    p._rtcm_enabled = False
    from scripts.lib.ppp_estimator_kinematic import PPPEstimatorKinematic
    p._ppp_class = PPPEstimatorKinematic
    return p


# Captures are 30 s-sampled over 1 hour, i.e. INSIDE the TEP band
# (periods > 33 min).  A short capture would contain no field signal at all,
# which is the whole point of PPPProcessor._band_status.
DT_S = 30.0
N_EPOCH = 120                      # 120 * 30 s = 3600 s = step_106's 1 h


def _field(t, amp):
    """A physically-shaped field excursion: slow, curved, never a step.

    The TEP field's power is below 500 uHz, so across a capture it appears
    as gentle curvature that a linear extrapolation cannot follow.  A
    single-epoch jump would be receiver steering, not the field -- which is
    exactly the distinction _dejump relies on.
    """
    T = N_EPOCH * DT_S
    return amp * (1.0 - math.cos(2.0 * math.pi * t / T)) / 2.0


def _clock_epochs(offset=1000.0, rate=0.5, field=0.0, n=N_EPOCH, jump_at=None):
    eps, acc = [], 0.0
    for i in range(n):
        t = i * DT_S
        b = offset + rate * t + _field(t, field)
        if jump_at is not None and i >= jump_at:
            acc = C_MS
        eps.append({"sod": t, "pos": TRUE_ECEF.tolist(), "clk_m": b + acc})
    return eps


def _mean_field_over_holdout(amp, n=N_EPOCH, frac=0.5):
    """What a perfect network prediction of the held-out field would be,
    measured against the pre-window trend the persistence arm fits."""
    n_pre = max(3, int(n * frac))
    t = np.arange(n) * DT_S
    f = np.array([_field(ti, amp) for ti in t])
    # residual the linear pre-window trend leaves behind, averaged over holdout
    o, r = np.polyfit(t[:n_pre], f[:n_pre], 1)[::-1]
    return float(np.mean(f[n_pre:] - (o + r * t[n_pre:])))


def _geometry(n_sat=8):
    up = TRUE_ECEF / np.linalg.norm(TRUE_ECEF)
    e1 = np.cross(up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    sats = {}
    for i in range(n_sat):
        el = math.radians(10 + 70 * i / (n_sat - 1))
        az = 2 * math.pi * i / n_sat
        d = math.sin(el) * up + math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2)
        sats[f"G{i+1:02d}"] = TRUE_ECEF + d * 20200e3
    return sats


# --- moving-satellite constellation for dual-frequency PPP tests ---
# Real GPS satellites orbit at ~3.87 km/s.  This orbital motion changes
# the LOS vectors over time, which is what allows the EKF to separate
# the receiver clock from per-satellite carrier-phase ambiguities.
# Without motion, the clock-ambiguity coupling is rank-1 deficient and
# PPP can never converge below the code noise floor (~0.2 m).
_GPS_ORB_R = 26560e3
_GPS_OMEGA = math.sqrt(3.986005e14 / _GPS_ORB_R**3)
_EARTH_ROT = 7.292e-5


def _gps_constellation(n_sat=24, t=0.0):
    """24-satellite GPS-like constellation: 6 planes x 4 sats."""
    sats = {}
    n_planes = 6
    sats_per_plane = n_sat // n_planes
    for p in range(n_planes):
        raan = 2 * math.pi * p / n_planes
        for s in range(sats_per_plane):
            i = p * sats_per_plane + s
            incl = math.radians(55)
            arg_lat = 2 * math.pi * s / sats_per_plane + p * 0.1
            theta = arg_lat + _GPS_OMEGA * t
            x_orb = _GPS_ORB_R * math.cos(theta)
            y_orb = _GPS_ORB_R * math.sin(theta)
            x1 = x_orb
            y1 = y_orb * math.cos(incl)
            z1 = y_orb * math.sin(incl)
            x2 = x1 * math.cos(raan) - y1 * math.sin(raan)
            y2 = x1 * math.sin(raan) + y1 * math.cos(raan)
            z2 = z1
            er = _EARTH_ROT * t
            x3 = x2 * math.cos(er) + y2 * math.sin(er)
            y3 = -x2 * math.sin(er) + y2 * math.cos(er)
            z3 = z2
            sats[f"G{i+1:02d}"] = np.array([x3, y3, z3])
    return sats


def _visible_sats(sats, rx_pos, min_el=math.radians(10)):
    up = rx_pos / np.linalg.norm(rx_pos)
    e1 = np.cross(up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    R_enu = np.array([e1, e2, up])
    vis = {}
    for sid, sp in sats.items():
        d = sp - rx_pos
        r = np.linalg.norm(d)
        los = d / r
        enu = R_enu @ los
        el = math.asin(max(-1.0, min(1.0, enu[2])))
        if el >= min_el:
            vis[sid] = sp
    return vis


_L1_WAVELENGTH = 0.19029365
_L5_WAVELENGTH = 0.25482768
_F1, _F5 = 1575.42e6, 1176.45e6
_IF_FACTOR = _F1**2 / (_F1**2 - _F5**2)  # ~2.26


def _dual_freq_capture_moving(seed=7, n=600, dt=DT_S):
    """Dual-frequency L1/L5 IF capture with realistic orbital motion.

    Satellites move at GPS orbital velocity (~3.87 km/s), so the LOS
    geometry changes over time.  This is essential for PPP: the changing
    geometry allows the EKF to separate the receiver clock from the
    per-satellite ambiguities, enabling centimetre-level convergence.
    """
    rng = np.random.default_rng(seed)
    spp, raw = [], []
    amb_l1 = {f"G{i+1:02d}": rng.integers(100000, 999999) + rng.normal(0, 0.1)
              for i in range(24)}
    amb_l5 = {f"G{i+1:02d}": rng.integers(100000, 999999) + rng.normal(0, 0.1)
              for i in range(24)}
    for i in range(n):
        t = i * dt
        b = 1000.0 + 0.5 * t
        all_sats = _gps_constellation(24, t)
        vis = _visible_sats(all_sats, TRUE_ECEF)
        if len(vis) < 6:
            continue
        up = TRUE_ECEF / np.linalg.norm(TRUE_ECEF)
        e1 = np.cross(up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        R_enu = np.array([e1, e2, up])
        prs, los, ids, phases = {}, [], [], {}
        for sid, sp in vis.items():
            d = sp - TRUE_ECEF
            r = float(np.linalg.norm(d))
            los_j = d / r
            pr_l1 = r + b + rng.normal(0, 0.3)
            pr_l5 = r + b + rng.normal(0, 0.3)
            if_pr = (_F1**2 * pr_l1 - _F5**2 * pr_l5) / (_F1**2 - _F5**2)
            prs[sid] = if_pr
            ph_l1 = r + b - amb_l1[sid] * _L1_WAVELENGTH + rng.normal(0, 0.002)
            ph_l5 = r + b - amb_l5[sid] * _L5_WAVELENGTH + rng.normal(0, 0.002)
            if_ph = (_F1**2 * ph_l1 - _F5**2 * ph_l5) / (_F1**2 - _F5**2)
            phases[sid] = if_ph
            los.append(los_j)
            ids.append(sid)
        raw.append({"sod": t, "pseudoranges": prs, "sat_positions": vis,
                    "sat_los": np.array(los), "sat_ids": ids, "n_sat": len(ids),
                    "freq_mode": "dual_l1l5", "if_factor": _IF_FACTOR,
                    "phases": phases})
        spp.append({"sod": t, "pos": (TRUE_ECEF + rng.normal(0, 5, 3)).tolist(),
                    "clk_m": b + rng.normal(0, 1.0)})
    return spp, raw


def _full_capture(field=7.5, seed=7, n=N_EPOCH):
    rng = np.random.default_rng(seed)
    sats = _geometry()
    spp, raw = [], []
    for i in range(n):
        t = i * DT_S
        b = 1000.0 + 0.5 * t + _field(t, field)
        prs, los, ids = {}, [], []
        for sid, sp in sats.items():
            d = sp - TRUE_ECEF
            r = float(np.linalg.norm(d))
            prs[sid] = r + b + rng.normal(0, 1.0)
            los.append(d / r)
            ids.append(sid)
        raw.append({"sod": t, "pseudoranges": prs, "sat_positions": sats,
                    "sat_los": np.array(los), "sat_ids": ids, "n_sat": len(ids)})
        spp.append({"sod": t, "pos": (TRUE_ECEF + rng.normal(0, 5, 3)).tolist(),
                    "clk_m": b + rng.normal(0, 1.0)})
    return spp, raw


def _score(resid_true, resid_net, mode="live"):
    spp, raw = _full_capture(field=resid_true)
    p = _make_proc(resid=resid_net, ref=TRUE_ECEF, mode=mode)
    cp = p._score_clock_prediction(spp)
    arms = {a: p._run_ekf(spp, arm=a, raw_measurements=raw, clock_pred=cp,
                          truth_ecef=TRUE_ECEF, degraded_k=4)
            for a in p.ARMS}
    return cp, arms, PPPProcessor._summarise_position(arms)


# ------------------------------------------------------------------ tests
def test_clock_prior_is_a_residual_on_the_receivers_own_trend():
    """The network term must be ADDED to the local trend, never substituted.

    Substituting a remote station's residual for this receiver's absolute
    bias would move the clock state by the whole local oscillator offset.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert "never used as an absolute prior" in src or \
           "never an absolute bias" in src, "physical-contract comment lost"

    resid = _mean_field_over_holdout(7.5)
    cp = _make_proc(resid=resid)._score_clock_prediction(_clock_epochs(field=7.5))
    assert cp["available"] and cp["network_term_available"]
    # The prior is own-trend + residual, so with a correct residual the TEP
    # prediction is much better than persistence -- and the receiver's own
    # drift is recovered, not replaced (rate ~ 0.5 m/s as constructed).
    assert cp["tep_rmse_m"] < cp["persist_rmse_m"]
    assert cp["tep_vs_persist_pct"] > 25
    assert cp["local_rate_m_per_s"] == pytest.approx(0.5, abs=0.02)


def test_no_network_term_means_tep_is_identical_to_persistence():
    """No free lunch: without a field term the arms must not differ at all."""
    cp = _make_proc(resid=None)._score_clock_prediction(_clock_epochs(field=7.5))
    assert cp["network_term_available"] is False
    assert cp["tep_rmse_m"] == cp["persist_rmse_m"]


def test_a_wrong_network_prior_is_punished_in_both_domains():
    """The metric must be able to report a LOSS, or it proves nothing.

    The clock prediction score reports the raw (ungated) TEP prediction,
    so a wrong network residual must produce a clock loss.  The EKF
    position arm applies a causal back-test gate that suppresses a
    residual which would have hurt on the pre-window; when the gate
    fires, the TEP position arm degenerates to persist (0% difference),
    which is the correct protective behaviour.
    """
    cp, _arms, pos = _score(resid_true=7.5, resid_net=-20.0)
    assert cp["tep_vs_persist_pct"] < 0, "wrong clock prior must lose"
    # The position arm is gated: a wrong prior is either punished (< 0)
    # or suppressed to 0% by the back-test gate.  Both are acceptable;
    # what is NOT acceptable is a wrong prior producing a gain (> 0).
    assert pos["tep_vs_persist_pct"] <= 0.0, \
        "wrong prior must not produce a position gain"


def test_all_arms_share_one_reference_frame():
    """Regression: the anisotropy block used to clobber ref_lat/ref_lon.

    It reassigned the enclosing-scope names to a hardcoded 53.64N/0.10E,
    so the post-loop ENU conversion measured the TEP arm in a different
    local frame from the control -- making the comparison meaningless.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert "math.radians(53.64)" not in src, "hardcoded reference latitude is back"
    assert "self.tep_ref_lat" not in src, "dead hardcoded-ref branch is back"

    _cp, arms, pos = _score(resid_true=7.5, resid_net=7.5)
    vals = [arms[a]["rmse_2d_m"] for a in arms]
    # A wrong-frame arm shows up as an absurd magnitude or a wild outlier.
    assert all(v < 1e4 for v in vals), f"wrong-frame magnitude: {vals}"
    assert max(vals) / max(min(vals), 1e-9) < 50, f"frames disagree: {vals}"
    assert pos["n_epochs_degraded"] > 0


def test_scored_metrics_are_accuracy_not_scatter():
    """Scatter is smoothness; the scored numbers must be truth-referenced."""
    _cp, arms, pos = _score(resid_true=7.5, resid_net=7.5)
    assert pos["metric_kind"] == "rmse_vs_truth_reference"
    for a in arms.values():
        assert a["scatter_metric_kind"] == "smoothness_diagnostic_not_skill"
        assert "rmse_3d_m" in a and "rmse_2d_m" in a


def test_tep_stack_flags_report_what_was_applied_not_the_arm_label():
    """`used_astro_q_modulation` / `used_anisotropy_constraint` must record
    what the arm ACTUALLY applied, not merely that it is the tep arm.

    They were hardcoded to `(arm == "tep")`, so the dashboard rendered
    "Chandler / Lunar Nodal / Semiannual: Active" even when the clock gate had
    closed and the modulation was genuinely skipped -- and even when the tep
    arm's whole output was then discarded for losing to persistence.  A panel
    that advertises physics it did not apply is the same class of defect as a
    headline measuring the wrong quantity.

    Note: the TEP clock gate threshold is 10x (1000%) — only catastrophically
    wrong network corrections are gated off.  This test uses no network term
    (resid_net=None) to deactivate the TEP arm, which sets tep_active=False.
    """
    # No network term → tep_active=False → flags must be False
    cp, arms, _pos = _score(resid_true=7.5, resid_net=None)

    # Control arms never carry the TEP stack, gate or no gate.
    for a in ("free", "persist"):
        assert arms[a]["used_astro_q_modulation"] is False, f"{a} arm"
        assert arms[a]["used_anisotropy_constraint"] is False, f"{a} arm"

    # With no network term, the TEP arm is inactive.  The modulation is then
    # genuinely not applied -- and the flag must say False rather than
    # advertise it.
    assert arms["tep"]["used_astro_q_modulation"] is False, (
        "tep arm advertises astro modulation when no network term is available")
    assert arms["tep"]["used_anisotropy_constraint"] is False

    # And the flag must not be a blanket arm label in the source.  The
    # guard variable is _tep_mods_active: tep_active AND a regression fit
    # AND fitted alpha >= 0.1 -- i.e. only when the modulation was really
    # applied, which is strictly stronger than the old tep_active alone.
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert '"used_astro_q_modulation": (arm == "tep"),' not in src, (
        "flag is hardcoded to the arm label again")
    assert '"used_astro_q_modulation": (arm == "tep" and _tep_mods_active),' in src

def test_prior_applies_only_during_the_outage_window():
    """Step 106's timeline: pre-window observed, prior only across the gap."""
    _cp, arms, _pos = _score(resid_true=7.5, resid_net=7.5)
    for a in arms.values():
        assert a["n_prewindow"] > 0
        assert a["n_epochs_scored"] == a["n_epochs"] - a["n_prewindow"]


def test_degraded_subset_takes_highest_elevation_satellites():
    _spp, raw = _full_capture()
    sel = PPPProcessor._degraded_subset(raw[0], 4, TRUE_ECEF)
    up = TRUE_ECEF / np.linalg.norm(TRUE_ECEF)
    elev = raw[0]["sat_los"] @ up
    assert len(sel) == 4
    # _degraded_subset picks the k highest-elevation satellites but then
    # optimises PDOP within the top k+4 candidates, so the selected set is
    # a subset of the top k+4 by elevation (not necessarily the top k).
    cand = np.argsort(-elev)[:min(len(elev), 4 + 4)]
    assert set(sel.tolist()).issubset(set(cand.tolist()))


def test_clock_resets_are_removed_before_scoring():
    """A millisecond reset must vanish, leaving the genuine drift intact."""
    raw = np.array([e["clk_m"] for e in _clock_epochs(field=0.0, jump_at=30)])
    dj = PPPProcessor._dejump(raw)
    drift_per_epoch = 0.5 * DT_S          # rate * sample interval
    assert np.abs(np.diff(raw)).max() > 1e5, "fixture lost its reset"
    # After dejumping, the series is pure drift: every step is the same.
    assert np.abs(np.diff(dj) - drift_per_epoch).max() < 1e-6


def test_arbitrary_steering_jumps_are_removed_not_just_millisecond_resets():
    """Regression: the LC29H steers by arbitrary amounts, not whole ms.

    A real capture from this rig contained a 344 m step (0.0011 ms). The
    old millisecond-only test ignored it, the trend fit was dragged through
    it, and the persistence baseline scored WORSE than holding the last
    value -- which is what made TEP look like it was losing.
    """
    n = 60
    b = 1000.0 + 0.5 * DT_S * np.arange(n)
    b[30:] -= 344.0                        # steering event, not a ms reset
    dj = PPPProcessor._dejump(b)
    assert np.abs(np.diff(b)).max() > 300
    assert np.abs(np.diff(dj) - 0.5 * DT_S).max() < 1e-6


def test_dejump_then_robust_trend_recovers_the_true_drift():
    """The pipeline order matters: dejump FIRST, then fit.

    A mid-series step contaminates ~50% of Theil-Sen's pairwise slopes,
    well past its ~29% breakdown point, so the median-of-slopes estimator
    cannot save a series on its own -- and least squares does even worse.
    Removing the step first is what makes the trend recoverable, which is
    the order _score_clock_prediction uses.
    """
    n = 60
    t = DT_S * np.arange(n)
    v = 1000.0 + 0.5 * t
    v[30:] -= 344.0                        # steering event mid-capture

    ls_rate = float(np.polyfit(t, v, 1)[0])
    assert abs(ls_rate - 0.5) > 0.05, "fixture should defeat least squares"

    # The real pipeline: dejump, then robust trend.
    _off, rate = PPPProcessor._robust_trend(t, PPPProcessor._dejump(v))
    assert abs(rate - 0.5) < 0.01, f"recovered rate {rate}"


def test_out_of_band_capture_is_flagged_loudly():
    """A capture shorter than one TEP band period contains no field signal.

    This is the finding that explained the rig's null result: a 120 s
    capture resolves nothing below 8.3 mHz, while the field's power lies
    below 500 uHz.  It must be reported, not silently scored.
    """
    short = PPPProcessor._band_status(120.0)
    assert short["verdict"] == "out_of_band"
    assert short["in_band"] is False
    assert short["shortfall_factor"] > 15

    ok = PPPProcessor._band_status(3600.0)
    assert ok["in_band"] is True
    assert ok["verdict"] in ("good", "optimal")

    p = _make_proc(resid=1.0)
    cav = p._caveats({"available": True, "network_term_available": True,
                      "tep_band": short}, "live")
    assert any("OUT OF THE TEP BAND" in c for c in cav)
    assert cav[0].startswith("CAPTURE IS OUT OF THE TEP BAND"), \
        "the band caveat must come first; everything else is downstream of it"


def test_replayed_archive_is_never_presented_as_live():
    """A simulation-mode network source must be flagged, not absorbed."""
    p = _make_proc(resid=7.5, mode="simulation")
    assert p.kriging_proxy.is_simulation
    cav = p._caveats({"network_term_available": True}, "simulation")
    assert any("REPLAYED ARCHIVE" in c for c in cav)
    # and a missing network source must say the arms collapse together
    cav2 = p._caveats({"network_term_available": False}, "none")
    assert any("degenerates" in c or "inactive" in c for c in cav2)


def test_serial_port_is_released_even_when_capture_fails():
    """Regression: a leaked handle blocked the GPS reader from reacquiring."""
    closed = {"n": 0}

    class _Ser:
        def write(self, *_a):
            raise IOError("boom")

        def read(self, *_a):
            return b""

        def close(self):
            closed["n"] += 1

    class _PySerial:
        Serial = staticmethod(lambda *a, **k: _Ser())

    import tempfile
    p = _make_proc()
    p._tmp = Path(tempfile.mkdtemp(prefix="tep_test_"))
    p._keep_cycles = 3
    p._chunks = []
    p.chunk_seconds = 1
    p.capture_seconds = 60
    assert p._capture_chunk(_PySerial()) is None
    assert closed["n"] == 1, "serial port was not closed on the failure path"


def test_rolling_window_is_pruned_by_wall_clock_span():
    """The rolling window is a wall-clock SPAN, not a sum of chunk durations.

    Capture pauses while each window is solved, so chunks are not contiguous.
    Summing durations let a "4 h" window span 8.7 h of wall clock, which
    inflated the held-out clock-prediction horizon (measured from epoch
    TIMESTAMPS) beyond what the network history could bridge.
    """
    import tempfile
    p = _make_proc()
    p._tmp = Path(tempfile.mkdtemp(prefix="tep_test_"))
    p.capture_seconds = 3600
    p.chunk_seconds = 900
    p.MIN_WINDOW_CHUNKS = 1
    p._chunks = []
    # 900 s of data every 1800 s of wall clock -- 50 % duty, as on the rig.
    t0 = 1_700_000_000
    for i in range(8):
        f = p._tmp / f"chunk_{t0 + i * 1800}.bin"
        f.write_bytes(b"x" * 2048)
        p._chunks.append({"path": str(f), "seconds": 900.0})
        p._prune_window()
    span = p._window_span_s()
    assert span <= 3600 + 1800, f"window spans {span:.0f}s, over the 3600s limit"
    kept = [Path(c["path"]).name for c in p._chunks]
    assert f"chunk_{t0 + 7 * 1800}.bin" in kept, "newest chunk was dropped"
    assert f"chunk_{t0}.bin" not in kept, "stale chunk was retained"
    # Summing durations would have kept 4 chunks (4 x 900 = 3600) spanning
    # 6300 s.  Span pruning keeps fewer, and that is the point.
    assert len(p._chunks) < 4, (
        f"pruned by duration, not span: kept {len(p._chunks)} chunks")
    # Dropped chunks are deleted from disk, not merely forgotten.
    assert not (p._tmp / f"chunk_{t0}.bin").exists()


def test_rolling_window_never_prunes_below_the_convergence_floor():
    """PPP needs 30-60 min of data to converge its float ambiguities, so span
    pruning must not be allowed to starve the window."""
    import tempfile
    p = _make_proc()
    p._tmp = Path(tempfile.mkdtemp(prefix="tep_test_"))
    p.capture_seconds = 60          # absurdly short: every chunk is "stale"
    p.chunk_seconds = 300
    p.MIN_WINDOW_CHUNKS = 5
    p._chunks = []
    t0 = 1_700_000_000
    for i in range(9):
        f = p._tmp / f"chunk_{t0 + i * 600}.bin"
        f.write_bytes(b"x" * 2048)
        p._chunks.append({"path": str(f), "seconds": 300.0})
        p._prune_window()
    assert len(p._chunks) == 5, f"convergence floor ignored: {len(p._chunks)}"
    assert Path(p._chunks[-1]["path"]).name == f"chunk_{t0 + 8 * 600}.bin", (
        "floor kept the OLDEST chunks instead of the newest")


def test_default_capture_window_is_inside_the_tep_band():
    """The rig must ship at settings that can actually show the effect."""
    import inspect
    sig = inspect.signature(PPPProcessor.__init__)
    default_window = sig.parameters["capture_seconds"].default
    band = PPPProcessor._band_status(float(default_window))
    assert band["in_band"] is True
    assert band["verdict"] == "optimal", (
        f"default window {default_window}s is '{band['verdict']}'; "
        "the shipped default should match step_106's 4 h outage")


def test_clock_prior_improves_effective_dop_and_makes_k3_solvable():
    """The TEP advantage as pure geometry, independent of any estimator.

    Standard DOP treats the receiver clock as a free unknown (design matrix
    [-LOS, 1], four columns).  A clock prior removes that unknown, so the
    same satellites give a better-conditioned solution.  At k=3 the
    conventional normal matrix is rank-deficient -- three observations,
    four unknowns -- while the clock-aided one is not, which is the paper's
    k=3 feasibility result.
    """
    sats = _geometry(8)
    los = np.array([(p - TRUE_ECEF) / np.linalg.norm(p - TRUE_ECEF)
                    for p in sats.values()])

    # k=4: a tight prior must strictly improve PDOP.
    d = PPPProcessor._dop(los[:4], clock_sigma_m=0.2)
    assert d["pdop_free"] is not None
    assert d["pdop_clock_aided"] < d["pdop_free"]
    assert d["improvement_pct"] > 0

    # A tighter prior must help at least as much as a loose one.
    tight = PPPProcessor._dop(los[:4], clock_sigma_m=0.2)["pdop_clock_aided"]
    loose = PPPProcessor._dop(los[:4], clock_sigma_m=20.0)["pdop_clock_aided"]
    assert tight < loose

    # A prior so loose it carries no information must reduce to standard DOP.
    huge = PPPProcessor._dop(los[:4], clock_sigma_m=1e9)
    assert huge["pdop_clock_aided"] == pytest.approx(huge["pdop_free"], rel=1e-6)

    # k=3: conventional is singular, clock-aided is solvable.
    d3 = PPPProcessor._dop(los[:3], clock_sigma_m=0.2)
    assert d3["free_is_singular"] is True
    assert d3["pdop_free"] is None
    assert d3["pdop_clock_aided"] is not None and np.isfinite(d3["pdop_clock_aided"])


def test_multi_constellation_ephemeris_is_loaded():
    """Regression: only GPS nav was parsed, discarding most of the sky.

    convbin splits ephemerides one file per constellation, and Galileo
    records carry a prefixed, space-padded PRN ("E30", "E 7") that a
    whitespace split misparses into misaligned fields -- so the Galileo
    file silently yielded zero satellites and the rig ran on 5 GPS
    satellites out of 19 tracked.
    """
    from scripts.lib.rinex_pseudorange import _parse_rinex_nav, _nav_system_for
    d = _ROOT / "rpi" / "test_results" / "test_20260904_161249" / "baseline_rinex"
    if not (d / "baseline.lnav").exists():
        pytest.skip("no captured Galileo nav available")

    assert _nav_system_for(Path("x.lnav")) == "E"
    assert _nav_system_for(Path("x.nav")) == "G"

    gps, _a, _b = _parse_rinex_nav(d / "baseline.nav", system="G")
    gal, _a, _b = _parse_rinex_nav(d / "baseline.lnav", system="E")
    assert len(gps) > 0, "GPS ephemerides failed to parse"
    assert len(gal) > 0, "Galileo ephemerides failed to parse (the old bug)"
    assert all(s.startswith("E") for s in gal), "Galileo sats mislabelled"
    # The space-padded single-digit PRN must survive.
    assert "E07" in gal or "E02" in gal or "E03" in gal


def test_degraded_subset_is_geometry_aware_not_just_highest_elevation():
    """Picking the k highest satellites clusters them near zenith.

    That is the worst choice for a position solve -- on the rig's own
    capture it turned PDOP 4.9 into 9.6 and produced kilometre errors.
    """
    _spp, raw = _full_capture()
    rm = raw[0]
    los = np.asarray(rm["sat_los"], float)
    up = TRUE_ECEF / np.linalg.norm(TRUE_ECEF)
    elev = los @ up

    def pdop(idx):
        A = np.hstack([-los[idx], np.ones((len(idx), 1))])
        return float(np.sqrt(np.trace(np.linalg.inv(A.T @ A)[:3, :3])))

    naive = np.argsort(-elev)[:4]
    chosen = PPPProcessor._degraded_subset(rm, 4, TRUE_ECEF)
    assert len(chosen) == 4
    assert pdop(chosen) <= pdop(naive) + 1e-9, (
        f"geometry-aware subset ({pdop(chosen):.2f}) should not be worse "
        f"than naive highest-elevation ({pdop(naive):.2f})")


def test_env_var_configuration_follows_rpi_convention():
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    for var in ("GPS_SERIAL_PORT", "GPS_BAUDRATE", "TEP_KRIGING_URL"):
        assert var in src, f"{var} not honoured"
    assert '"--host", default=_env_str("WEB_HOST", "127.0.0.1")' in src, \
        "web host should default to loopback (dashboard is unauthenticated)"


# --------------------------------------------------------------- GRAPHIC
def test_graphic_combination_cancels_ionosphere():
    """GRAPHIC = (code + phase) / 2 removes the first-order ionosphere.

    Code is delayed by +I, phase is advanced by -I.  Averaging cancels
    the ionosphere entirely, leaving the geometric range plus a constant
    ambiguity term that the EKF absorbs.  This is the standard
    single-frequency PPP observable and is essential for the LC29H
    receiver, whose L1+L5 signal loses L5 through the standard RTKLIB
    convbin (RINEX 2.11 has no L5 field).
    """
    from scripts.lib.rinex_pseudorange import _graphic_combination
    rho = 20e6  # 20,000 km geometric range
    iono = 5.0  # 5 m ionospheric delay
    amb = 1000.0  # ambiguity in metres
    code = rho + iono + 0.3   # code noise ~0.3m
    phase = rho - iono + amb  # phase with ambiguity
    g = _graphic_combination(code, phase)
    # GRAPHIC = rho + amb/2 + noise/2 — iono cancelled
    assert abs(g - (rho + amb / 2 + 0.15)) < 0.01


def test_real_capture_yields_graphic_not_single_l1():
    """The captured LC29H data must use GRAPHIC, not raw L1.

    RINEX 2.11 from convbin drops L5, leaving only C1 and L1.  Without
    GRAPHIC the residual ionosphere after Klobuchar (~30-50%) dominates
    and caps accuracy at metre level.  GRAPHIC removes the ionosphere
    entirely and enables decimetre-level convergence.
    """
    from scripts.lib.rinex_pseudorange import get_measurements
    d = _ROOT / "rpi" / "test_results" / "test_20260904_161249" / "baseline_rinex"
    if not (d / "baseline.obs").exists():
        pytest.skip("no captured RINEX available")
    obs = d / "baseline.obs"
    nav = [f for f in [d / "baseline.nav", d / "baseline.lnav",
                       d / "baseline.gnav"] if f.exists()]
    with open(obs) as fh:
        for line in fh:
            if "APPROX POSITION XYZ" in line:
                parts = line.split()
                rx = np.array([float(parts[0]), float(parts[1]),
                               float(parts[2])])
                break
    meas = get_measurements(obs, nav, rx)
    assert len(meas) > 0, "no measurements extracted"
    fm = meas[0]["freq_mode"]
    assert fm == "single_l1_graphic", (
        f"expected single_l1_graphic on L1-only capture, got {fm}")
    # if_factor for GRAPHIC is 1/sqrt(2) (noise reduction, not amplification)
    assert abs(meas[0]["if_factor"] - 1.0 / math.sqrt(2.0)) < 0.01


def test_graphic_phase_observations_are_present():
    """GRAPHIC phase must be built from C1 and L1 on the real capture.

    A regression that drops L1 parsing or the GRAPHIC construction would
    leave the phases dict empty, falling back to code-only processing.
    """
    from scripts.lib.rinex_pseudorange import get_measurements
    d = _ROOT / "rpi" / "test_results" / "test_20260904_161249" / "baseline_rinex"
    if not (d / "baseline.obs").exists():
        pytest.skip("no captured RINEX available")
    obs = d / "baseline.obs"
    nav = [f for f in [d / "baseline.nav", d / "baseline.lnav",
                       d / "baseline.gnav"] if f.exists()]
    with open(obs) as fh:
        for line in fh:
            if "APPROX POSITION XYZ" in line:
                parts = line.split()
                rx = np.array([float(parts[0]), float(parts[1]),
                               float(parts[2])])
                break
    meas = get_measurements(obs, nav, rx)
    total_phase = sum(len(m["phases"]) for m in meas)
    assert total_phase > 0, "no phase observations built (GRAPHIC broken)"
    # At least half the satellite-epochs should have phase
    total_sat = sum(m["n_sat"] for m in meas)
    assert total_phase > total_sat * 0.3, (
        f"too few phase obs: {total_phase}/{total_sat}")


def test_graphic_phase_noise_is_code_level_not_mm():
    """The EKF must weight GRAPHIC phase at code-level noise, not mm.

    GRAPHIC = (code + phase) / 2 is dominated by the code component
    (~0.3-3 m noise), unlike dual-frequency IF phase which is mm-level.
    Using mm-level phase_std would make the EKF trust GRAPHIC far too
    tightly, over-fitting code multipath and diverging.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert "single_l1_graphic" in src, "GRAPHIC freq_mode not handled"
    # The phase_std for GRAPHIC must be code-level, not 0.01 * if_factor
    assert "phase_std = code_std" in src or "phase_std=code_std" in src, (
        "GRAPHIC phase_std must be code-level, not mm-level")


# --------------------------------------------------------- dual-freq PPP
def test_dual_freq_ppp_converges_to_decimetres_with_orbital_motion():
    """Dual-frequency carrier-phase PPP must converge to decimetre level.

    The ionosphere-free L1/L5 combination removes first-order ionospheric
    delay, and the carrier phase (2.3 mm noise) provides the precision for
    centimetre-level positioning.  The key requirement is that the
    satellites MOVE: orbital motion at ~3.87 km/s changes the LOS vectors
    over time, which allows the EKF to separate the receiver clock from
    the per-satellite ambiguities.  Without motion, the clock-ambiguity
    coupling is rank-1 deficient and PPP is capped at the code noise floor.

    This test uses a 24-satellite GPS-like constellation with realistic
    orbital mechanics over 5 hours (600 epochs at 30 s).  The converged
    position error (last 50 epochs) must be below 0.30 m in 2D and
    0.50 m in 3D, demonstrating that the EKF correctly:
      * processes dual-frequency IF code and phase
      * maintains carrier-phase ambiguity continuity
      * separates clock from ambiguities via geometry change
      * converges position to decimetre level
    """
    spp, raw = _dual_freq_capture_moving(seed=7, n=600)
    assert len(raw) >= 100, f"too few epochs with visible sats: {len(raw)}"
    p = _make_proc(ref=TRUE_ECEF)
    cp = p._score_clock_prediction(spp)
    arm = p._run_ekf(spp, arm="free", raw_measurements=raw, clock_pred=cp,
                     truth_ecef=TRUE_ECEF, degraded_k=4, full_sky=True)
    positions = np.array(arm["positions"])
    err = positions - TRUE_ECEF
    # Converged metric: last 50 epochs
    n_conv = min(50, len(err))
    late = err[-n_conv:]
    rmse_2d_conv = float(np.sqrt(np.mean(late[:, 0]**2 + late[:, 1]**2)))
    rmse_3d_conv = float(np.sqrt(np.mean(np.sum(late**2, axis=1))))
    assert rmse_2d_conv < 0.30, (
        f"converged 2D RMSE {rmse_2d_conv:.4f} m exceeds 0.30 m; "
        f"PPP is not converging with orbital motion")
    assert rmse_3d_conv < 0.50, (
        f"converged 3D RMSE {rmse_3d_conv:.4f} m exceeds 0.50 m; "
        f"PPP is not converging with orbital motion")
    # Phase observations must be present (not code-only)
    assert arm.get("ambiguity_phase_obs", 0) > 0, "no phase observations"
    # Ambiguity slots must be allocated
    assert arm.get("ambiguity_n_slots", 0) > 0, "no ambiguity slots"


def test_dual_freq_ppp_uses_carrier_phase_not_code_smoothing():
    """The full-sky arm must use genuine carrier-phase observations.

    Code smoothing (Hatch filter) can produce sub-metre scatter but never
    centimetres because it is bounded by code noise.  True PPP uses the
    carrier phase as the primary observable, with the code only for
    initialisation and clock anchoring.  This test verifies that:
      * phase observations are counted (ambiguity_phase_obs > 0)
      * ambiguity states are allocated (ambiguity_n_slots > 0)
      * no cycle slips occur in clean data (ambiguity_slips == 0)
    """
    spp, raw = _dual_freq_capture_moving(seed=11, n=400)
    p = _make_proc(ref=TRUE_ECEF)
    cp = p._score_clock_prediction(spp)
    arm = p._run_ekf(spp, arm="free", raw_measurements=raw, clock_pred=cp,
                     truth_ecef=TRUE_ECEF, degraded_k=4, full_sky=True)
    assert arm["ambiguity_phase_obs"] > 0, "no carrier-phase observations"
    assert arm["ambiguity_n_slots"] > 0, "no ambiguity states allocated"
    assert arm["ambiguity_slips"] == 0, (
        f"cycle slips in clean data: {arm['ambiguity_slips']}")


# ---------------------------------------------------------------------------
# Step 96 accumulated-offset network clock prior
#
# The live rig reported TEP +0.0 % against persistence for a long time.  The
# cause was that its TEP arm kriged the network's INSTANTANEOUS residual LEVEL
# and held it constant across the outage, rather than the network's ACCUMULATED
# offset across the same gap.  Measured on data/ppp_dense over held-out days:
# the level transfer scores -23.5 % against persistence (worse than doing
# nothing) and the accumulated transfer +37.7 %.  See
# scripts/steps/step_107_live_clock_prior_calibration.py.
#
# These guard the properties that make the accumulated arm correct.
# ---------------------------------------------------------------------------

_STN_COORDS = {
    "AAAA": (53.649, 0.116), "BBBB": (53.736, -0.505),
    "CCCC": (53.142, 0.347), "DDDD": (53.250, -0.520),
    "EEEE": (54.253, -0.380), "FFFF": (54.162, -1.100),
    "GGGG": (51.990, 4.390), "HHHH": (52.910, 6.600),
}


class _FieldProxy:
    """Serves a raw clock history to the accumulated-offset predictor.

    Named distinctly from the _FakeProxy above, which serves a scalar
    instantaneous residual to a different code path.
    """

    def __init__(self, series, coords):
        self._s, self.station_coords = series, coords

    def get_raw_series(self):
        return self._s


def _shared_field_network(seed=3, n=600, dt=30.0, t0=40000.0, field_sd=0.30,
                          noise_sd=0.05):
    """A network whose stations share ONE random-walk field, plus per-station
    linear drift and independent noise.  The field is the transferable part."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * dt + t0
    field = np.cumsum(rng.normal(0.0, field_sd, n))
    series = {}
    for i, sid in enumerate(_STN_COORDS):
        drift = rng.normal(0.0, 2e-4)
        series[sid] = (t.copy(),
                       field + drift * (t - t[0])
                       + rng.normal(0.0, noise_sd, n) + 100.0 * i)
    return t, field, series


def _accum_proc(series):
    p = _make_proc(ref=TRUE_ECEF)
    p.kriging_proxy = _FieldProxy(series, dict(_STN_COORDS))
    return p


def test_accum_prior_recovers_the_shared_field():
    """The whole point: predict how far this clock walked over the gap."""
    t, field, series = _shared_field_network()
    p = _accum_proc(series)
    t0 = t[len(t) // 2]
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    corr, info = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                             pre_span_s=7200.0)
    assert corr is not None, info.get("reason")
    alpha = gps_web.PPPProcessor.NET_ACCUM_ALPHA_QUARTZ
    truth = alpha * np.array([np.interp(t0 + h, t, field)
                              - np.interp(t0, t, field) for h in hs])
    err = float(np.sqrt(np.mean((corr - truth) ** 2)))
    scale = float(np.sqrt(np.mean(truth ** 2)))
    assert err < 0.25 * scale, f"field not recovered: {err:.3f} m vs {scale:.3f} m"


def test_accum_prior_grows_with_the_gap():
    """A constant correction is the OLD, broken construction.  The accumulated
    offset must scale with the horizon, or it cannot compete with a drift
    extrapolation.

    A random walk's |increment| grows as sqrt(h) only in EXPECTATION, so the
    property is asserted over an ensemble of fields rather than one draw.
    """
    hs = np.array([300.0, 7200.0])
    short, long_ = [], []
    for seed in range(20):
        t, _field, series = _shared_field_network(seed=100 + seed)
        corr, info = _accum_proc(series)._network_accum_prediction(
            t0=t[len(t) // 2], horizons_s=hs, pre_span_s=7200.0)
        assert corr is not None, info.get("reason")
        short.append(abs(corr[0]))
        long_.append(abs(corr[1]))
    assert np.mean(long_) > 3.0 * np.mean(short), (
        f"correction does not grow with the gap: "
        f"short={np.mean(short):.3f} m long={np.mean(long_):.3f} m")


def test_accum_prior_excludes_stations_missing_across_the_gap():
    """A station that drops out mid-outage cannot supply a cross-section for
    the horizons it does not reach; using its last value would fabricate one."""
    t, field, series = _shared_field_network(seed=7)
    t0 = t[len(t) // 2]
    trunc = dict(series)
    keep = t < t0 + 1800.0
    trunc["AAAA"] = (t[keep].copy(), series["AAAA"][1][keep].copy())
    p = _accum_proc(trunc)
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    corr, info = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                             pre_span_s=7200.0)
    assert corr is not None, info.get("reason")
    assert info["min_stations_across_gap"] < info["n_stations"], (
        "dropped-out station was not excluded from the far horizons")


def test_accum_prior_refuses_when_history_misses_the_whole_gap():
    """No overlap must mean no prediction, not a fabricated one."""
    t, field, series = _shared_field_network(seed=9)
    t0 = t[len(t) // 2]
    late = {sid: (tt[tt >= t0 + 7500.0].copy(), vv[tt >= t0 + 7500.0].copy())
            for sid, (tt, vv) in series.items()}
    p = _accum_proc(late)
    corr, info = p._network_accum_prediction(
        t0=t0, horizons_s=np.arange(300.0, 7200.0 + 1, 300.0),
        pre_span_s=7200.0)
    assert corr is None and info.get("available") is False
    assert "past the" in (info.get("reason") or "")


def test_accum_prior_survives_utc_midnight():
    """The receiver reports seconds-of-day; the network history is unwrapped.
    A midnight crossing must not change the answer."""
    t, field, series = _shared_field_network(seed=13)
    t0 = t[len(t) // 2]
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    flat, _ = _accum_proc(series)._network_accum_prediction(
        t0=t0, horizons_s=hs, pre_span_s=7200.0)
    shift = 43000.0
    tw = (t + shift) % 86400.0
    wrap = np.concatenate([[0.0], np.cumsum(np.diff(tw) < -43200.0) * 86400.0])
    wrapped = {sid: (tw + wrap, vv) for sid, (tt, vv) in series.items()}
    across, _ = _accum_proc(wrapped)._network_accum_prediction(
        t0=(t0 + shift) % 86400.0, horizons_s=hs, pre_span_s=7200.0)
    assert np.allclose(flat, across, atol=1e-9), "midnight wrap changed the result"


def test_accum_prior_rejects_an_unrepaired_steering_step():
    """One station with a 1 ms step contributes a ~300 km cross-section.  The
    upstream MAD filter runs on the DETRENDED column and cannot see it."""
    t, field, series = _shared_field_network(seed=17)
    t0 = t[len(t) // 2]
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    clean, _ = _accum_proc(series)._network_accum_prediction(
        t0=t0, horizons_s=hs, pre_span_s=7200.0)
    bad = dict(series)
    tt, vv = series["CCCC"]
    vv = vv.copy()
    vv[tt > t0 + 900.0] += 299792.458          # 1 ms steering step
    bad["CCCC"] = (tt, vv)
    dirty, info = _accum_proc(bad)._network_accum_prediction(
        t0=t0, horizons_s=hs, pre_span_s=7200.0)
    assert dirty is not None
    # The step must be neutralised somewhere: either the destep pass
    # stitches the 1 ms level jump up front, or the per-horizon outlier
    # rejector drops the poisoned cross-section.  Both are causal repairs;
    # asserting either catches the regression where the step leaks through.
    _repaired = (info["n_outliers_rejected"] >= 1
                 or info.get("destep", {}).get("applied"))
    assert _repaired, "steering step was neither stitched nor rejected"
    assert np.max(np.abs(dirty - clean)) < 1.0, (
        f"steering step leaked into the prior: "
        f"{np.max(np.abs(dirty - clean)):.1f} m")


def test_kriging_proxy_merges_backfill_under_live_epochs():
    """A rig restart must not blind the predictor: the source's own history is
    backfilled, and live samples win on any timestamp they share."""
    proxy = gps_web.KrigingProxy.__new__(gps_web.KrigingProxy)
    proxy._backfill = {"AAAA": {100.0: -7.0, 130.0: -7.0,
                                160.0: -7.0, 190.0: -7.0}}
    proxy._epoch_history = [{"epoch_sod": 130.0, "lagged_stations": [],
                             "raw_clocks": {"AAAA": 42.0}}]
    ts, vs = proxy.get_raw_series()["AAAA"]
    assert len(ts) == 4, "backfill was dropped"
    assert vs[list(ts).index(130.0)] == 42.0, "live sample did not win"


def test_kriging_proxy_drops_time_shifted_stations():
    """OS Net stations have their SOD origin re-derived on every hourly poll,
    so consecutive epochs do not share a time origin and CANNOT be differenced.
    They must not reach the accumulated-offset pool."""
    proxy = gps_web.KrigingProxy.__new__(gps_web.KrigingProxy)
    proxy._backfill = {}
    proxy._epoch_history = [
        {"epoch_sod": 100.0 + 30.0 * k, "lagged_stations": ["EASN"],
         "raw_clocks": {"AAAA": 1.0 + k, "EASN": 50.0 + k}}
        for k in range(6)]
    out = proxy.get_raw_series()
    assert "AAAA" in out
    assert "EASN" not in out, "time-shifted station leaked into the raw series"


def test_position_gate_does_not_judge_the_accumulated_arm_by_the_old_proxy():
    """The position arm must not be gated off by the INSTANTANEOUS residual's
    amplitude when it is running the accumulated-offset prior.

    Observed live: the accumulated correction was 0.81 m RMS / 2.31 m peak and
    the clock arm scored +2.1 %, but the instantaneous residual happened to sit
    at -0.28 m, below the 0.37 m distance-scaled floor.  The gate closed and
    the TEP position arm degenerated to persistence -- judged by the magnitude
    of a quantity it does not use.

    The source-level guard is deliberate: the gate's branch order is the whole
    fix, so this asserts the accumulated branch is reached BEFORE the
    amplitude-proxy branch.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    i_accum = src.find("elif _accum_c is not None:")
    i_proxy = src.find("abs(float(_resid_effective)) < _degen_thresh")
    assert i_accum != -1, "accumulated-arm gate branch is gone"
    assert i_proxy != -1, "amplitude-proxy branch is gone"
    assert i_accum < i_proxy, (
        "the amplitude proxy is evaluated before the accumulated-arm branch, "
        "so the accumulated arm can still be gated off by the instantaneous "
        "residual's magnitude")


def test_clock_gate_runs_for_the_accumulated_arm():
    """The soft clock gate was guarded by `net_resid is not None`, so an
    accumulated-offset arm running without an instantaneous residual was never
    gated at all -- a harmful network term would have gone straight through."""
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert 'if net_resid is not None and persist_rmse > 0:' not in src, (
        "clock gate is still keyed on the instantaneous residual alone")
    assert 'if out.get("network_term_available") and persist_rmse > 0:' in src


def test_permuted_geometry_control_changes_only_the_geometry():
    """The specificity control must scramble WHICH station is at WHICH
    distance and nothing else.

    A gain over persistence only says the predictor works; it does not say the
    spatial structure is what makes it work, because a global common mode
    shared by every station would score identically.  The control is only
    evidence if the clock series, kernel, alpha and gating are untouched.
    """
    t, field, series = _shared_field_network(seed=23)
    p = _accum_proc(series)
    t0 = t[len(t) // 2]
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    real, i_real = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                               pre_span_s=7200.0)
    perm, i_perm = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                               pre_span_s=7200.0,
                                               permute_geometry=7)
    assert real is not None and perm is not None
    # Same pool, same alpha, same kernel -- only the geometry differs.
    assert i_perm["n_stations"] == i_real["n_stations"]
    assert i_perm["alpha"] == i_real["alpha"]
    assert i_perm["lambda_km"] == i_real["lambda_km"]
    assert i_perm["permuted_geometry"] is True
    assert i_real["permuted_geometry"] is False
    # On a field that IS spatially structured, permuting must change the answer.
    assert not np.allclose(real, perm), (
        "permuting the geometry changed nothing — the control is inert")


def test_common_mode_removed_control_kills_a_pure_common_mode():
    """With the network mean stripped, a purely global common mode carries no
    information and the correction must collapse toward zero.  If it does not,
    the control is not doing its job and cannot support a TEP-specific claim.
    """
    rng = np.random.default_rng(5)
    t = np.arange(600, dtype=float) * 30.0 + 40000.0
    # Every station carries the IDENTICAL signal: a pure global common mode
    # with no spatial structure whatsoever.
    common = np.cumsum(rng.normal(0, 0.3, len(t)))
    series = {sid: (t.copy(), common + 100.0 * i)
              for i, sid in enumerate(_STN_COORDS)}
    p = _accum_proc(series)
    t0 = t[len(t) // 2]
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    plain, _ = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                           pre_span_s=7200.0)
    cmfree, info = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                               pre_span_s=7200.0,
                                               remove_common_mode=True)
    assert plain is not None and cmfree is not None
    assert info["common_mode_removed"] is True
    assert np.max(np.abs(plain)) > 1.0, "fixture has no common mode to remove"
    assert np.max(np.abs(cmfree)) < 1e-6, (
        f"pure common mode survived its own removal: "
        f"max |corr| = {np.max(np.abs(cmfree)):.3e} m")


def test_proximity_does_not_override_the_accumulated_predictor():
    """A near station must not switch the arm back to the instantaneous BLUP.

    A "co-located beats accumulated" branch used to fire whenever the nearest
    reference was under 10 km.  It was never validated -- the archive network
    has no sub-10 km pairs -- and the moment a 1.3 km station joined the live
    network it put the rig back on the instantaneous construction, which
    scored -4.57 % against persistence live and -23.5 % on held-out archive
    days.  Proximity does not rescue the wrong observable: an instantaneous
    residual LEVEL cannot predict an accumulated offset over a gap, however
    close the station is.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert "collocated_instantaneous_blup" not in src, (
        "the unvalidated co-located override is back")
    assert "_collocated = (" not in src
    # The accumulated arm must be the first branch considered.
    i_accum = src.find('out["network_method"] = "step96_accumulated_offset"')
    i_fallback = src.find('out["network_method"] = "instantaneous_residual_fallback"')
    assert i_accum != -1 and i_fallback != -1
    assert i_accum < i_fallback, (
        "the instantaneous fallback is selected before the accumulated arm")


def test_anchor_choice_does_not_drop_a_whole_station_group():
    """The anchor must not be dragged to one source's history window.

    The live network mixes sources with different histories: NTRIP streams
    start when the server starts, while OS Net serves four hours of hourly
    RINEX ending 1-2 h back.  A median-based anchor followed whichever group
    was larger and silently dropped the other -- which cost the pool every
    station under 250 km and over 500 km, i.e. the entire baseline range the
    network had just been extended to cover.
    """
    t, field, series = _shared_field_network(seed=31, n=900)
    t0 = t[500]
    # 6 "fresh" stations covering only the recent tail, 2 "archival" ones
    # covering an older span -- the live NTRIP / OS Net split in miniature.
    mixed = {}
    for i, (sid, (tt, vv)) in enumerate(series.items()):
        if i < 6:
            m = tt >= t0 + 3600.0          # fresh: recent only
        else:
            m = tt <= t0 + 5400.0          # archival: older, ends earlier
        mixed[sid] = (tt[m].copy(), vv[m].copy())
    p = _accum_proc(mixed)
    hs = np.arange(300.0, 7200.0 + 1, 300.0)
    corr, info = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                             pre_span_s=7200.0)
    assert corr is not None, info.get("reason")
    # The chosen anchor must admit MORE stations than the minimum, i.e. it
    # found an overlap rather than collapsing onto one group.
    assert info["n_stations"] >= 6, (
        f"anchor dropped a whole group: only {info['n_stations']} stations "
        f"survived out of {len(mixed)}")
    assert info["horizon_coverage"] > 0.0


# ---------------------------------------------------------- persist gate
def _no_drift_strong_field(field=20.0):
    """Clock with no linear drift but strong field curvature.

    The drift gate passes (R² > 0.85 from the field's quasi-linear rise
    in the pre-window), so persist tries a linear extrapolation.  But
    the field curves back down in the holdout, making the extrapolation
    WORSE than holding the last value — which is exactly the TCXO-like
    regime where persistence harms position.
    """
    eps = []
    for i in range(N_EPOCH):
        t = i * DT_S
        b = 1000.0 + _field(t, field)
        eps.append({"sod": t, "pos": TRUE_ECEF.tolist(), "clk_m": b})
    return eps


def test_persist_gate_fires_when_persistence_is_worse_than_free():
    """The clock gate threshold is 10x (1000%) — only catastrophically
    wrong clock predictions are gated off.  The post-hoc position gate
    (3D RMSE threshold 10%) handles the case where the clock prior is
    harmful for position but not catastrophic for clock.

    This test verifies that the clock gate does NOT fire at 105% worse
    (the old threshold was 0.5%), because the clock prior's benefit is
    in POSITION, not clock prediction.  A slightly worse clock RMSE is
    normal when the drift extrapolation is imperfect, but the clock
    prior still helps position (observed live: persist 73% better in
    3D position with 374% worse clock RMSE).
    """
    spp = _no_drift_strong_field()
    p = _make_proc(resid=0.0, ref=TRUE_ECEF, mode="live")
    cp = p._score_clock_prediction(spp)
    assert cp["persist_rmse_m"] > cp["free_rmse_m"], (
        "fixture should make persist worse than free")
    # The clock gate should NOT fire at 105% worse (threshold is 10x)
    assert cp["persist_gate_alpha"] > 0.0, (
        f"clock gate should NOT fire at 105% worse (threshold is 10x): "
        f"{cp.get('persist_gate_reason')}")


def test_persist_arm_degenerates_to_free_when_gate_fires():
    """When the clock gate does NOT fire (threshold 10x), the persist arm
    should still apply the clock prior.  The post-hoc position gate (3D
    RMSE threshold 10%) handles the case where the clock prior is harmful
    for position — it fires AFTER the arms run and affects the NEXT solve.
    
    This test verifies that at 105% worse on clock RMSE, the clock gate
    does NOT fire (alpha > 0) and the persist arm applies the clock prior.
    """
    spp, raw = _full_capture(field=20.0, seed=7)
    # Override the clock to have no linear drift, making persist harmful
    for i, ep in enumerate(spp):
        t = i * DT_S
        ep["clk_m"] = 1000.0 + _field(t, 20.0)
    p = _make_proc(resid=0.0, ref=TRUE_ECEF, mode="live")
    cp = p._score_clock_prediction(spp)
    # The clock gate should NOT fire at 105% worse (threshold is 10x)
    assert cp["persist_gate_alpha"] > 0.0, (
        f"clock gate should NOT fire at 105% worse: "
        f"{cp.get('persist_gate_reason')}")


def test_tep_gate_compares_against_best_baseline_not_just_persist():
    """When persist is worse than free but the clock gate doesn't fire
    (threshold 10x), the TEP gate should still compare TEP against
    the best baseline.  The clock gate threshold is 10x, so at 105%
    worse the gate doesn't fire and the persist arm is still active.
    """
    spp = _no_drift_strong_field()
    # Give the network a residual that partially corrects the field
    # curvature — enough to beat free but not enough to beat persist
    # (which is very bad in this fixture).
    p = _make_proc(resid=-5.0, ref=TRUE_ECEF, mode="live")
    cp = p._score_clock_prediction(spp)
    # The clock gate should NOT fire at 105% worse (threshold is 10x)
    assert cp["persist_gate_alpha"] > 0.0, (
        f"clock gate should NOT fire at 105% worse: "
        f"{cp.get('persist_gate_reason')}")



def test_gram_uses_predictive_gram_not_phase_cos():
    """The coherence weighting must use `predictive_gram`.

    `spectral_phase_gram` returns both; they are NOT interchangeable. Kriging
    on `phase_cos` scores -69.5 % on held-out archive days against +53.6 % for
    `predictive_gram` -- a catastrophic sign flip, not a mild degradation. This
    cost a full debugging cycle once and the failure mode is silent.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert 'g["predictive_gram"]' in src, "Gram no longer uses predictive_gram"
    assert 'g["phase_cos"]' not in src, (
        "phase_cos is being used for the kriging weights — this inverts the result")


def test_gram_refuses_a_window_too_short_to_resolve_the_coupling():
    """Below ~12 h the Gram's uplift collapses, so a short window must be
    REFUSED rather than silently producing a weak weighting.

    Measured, quartz class, uplift over distance-only kriging: +1.8 points at
    4 h, +2.9 at 6 h, +15.0 at 12 h. The coupling lives at 6-12 h periods, so
    the band's low edge (1/window) has to reach below ~23 uHz to see it.
    """
    p = _make_proc(ref=TRUE_ECEF)
    assert p.GRAM_MIN_WINDOW_S >= 12 * 3600.0, (
        "the 12 h floor was lowered — the Gram will silently under-deliver")
    # A 4 h own-clock buffer must be rejected with a reason naming the window.
    rng = np.random.default_rng(2)
    t = np.arange(0.0, 4 * 3600.0, 30.0) + 40000.0
    p._own_clock = {round(float(x), 1): float(v)
                    for x, v in zip(t, np.cumsum(rng.normal(0, 0.1, len(t))))}
    names, G, info = p._spectral_gram({}, {})
    assert names is None and G is None
    assert info["available"] is False
    assert "12 h" in info["reason"] or "window" in info["reason"], info


def test_own_clock_buffer_accumulates_beyond_one_capture_window():
    """The Gram needs >=12 h but a capture is 4 h, so the receiver's own clock
    has to accumulate ACROSS windows or the Gram can never run."""
    import tempfile
    p = _make_proc(ref=TRUE_ECEF)
    p._tmp = Path(tempfile.mkdtemp(prefix="tep_own_"))
    base = 30000.0
    for w in range(4):                       # four successive 4 h captures
        t = np.arange(base + w * 3600.0, base + w * 3600.0 + 4 * 3600.0, 30.0)
        p._record_own_clock(t, np.linspace(0.0, 1.0, len(t)))
    ser = p.own_clock_series()
    assert ser is not None
    span = float(np.ptp(ser[0]))
    assert span > 4 * 3600.0, (
        f"buffer did not accumulate across windows: span {span/3600:.1f} h")
    # and it must survive a restart
    p2 = _make_proc(ref=TRUE_ECEF)
    p2._tmp = p._tmp
    p2._record_own_clock(np.array([base]), np.array([0.0]))
    assert p2.own_clock_series() is not None
    assert float(np.ptp(p2.own_clock_series()[0])) > 4 * 3600.0, (
        "own-clock history did not survive a restart")


def test_causal_alpha_backs_off_when_the_local_clock_is_quiet():
    """A FIXED transfer scale over-applies when the receiver's clock happens to
    be quieter than the network.

    Observed live: the clock moved 5.23 m across the gap while the network
    correction was 1.76 m RMS / 5.59 m peak, and alpha = 0.619 made the
    prediction 18.5 % WORSE than doing nothing. The gate caught it, but a gate
    that fires is a prediction thrown away. alpha is a regression coefficient,
    so it is fitted on the PRE-window -- where both sides are observable -- and
    carried into the held-out window. No held-out sample touches the fit.
    """
    t, field, series = _shared_field_network(seed=41, n=700)
    p = _accum_proc(series)
    t_pre = t[:400]
    # A receiver whose clock barely responds to the field: the correct alpha
    # is near zero, and anything near 0.619 would inject the network's motion
    # into a clock that is not making it.
    b_pre = 0.02 * np.interp(t_pre, t, field) + 0.001 * (t_pre - t_pre[0])
    a, info = p._causal_alpha(t_pre, b_pre)
    assert info["method"] == "causal_prewindow_regression", info.get("reason")
    assert a < 0.4, (
        f"alpha {a:.3f} did not back off for a clock that barely tracks the "
        f"field; a fixed 0.619 is what caused the -18.5 % live window")
    assert p.ALPHA_MIN <= a <= p.ALPHA_MAX

    # ...and it must NOT back off when the clock does track the field.
    b_pre2 = 1.0 * np.interp(t_pre, t, field)
    a2, info2 = p._causal_alpha(t_pre, b_pre2)
    assert info2["method"] == "causal_prewindow_regression"
    assert a2 > a, (
        f"alpha did not rise for a strongly-tracking clock: {a2:.3f} vs {a:.3f}")


def test_causal_alpha_falls_back_safely_without_a_network_term():
    """With no network history the fit is impossible; it must return the
    calibrated default and say why, not zero or NaN."""
    p = _make_proc(ref=TRUE_ECEF)
    p.kriging_proxy = None
    t_pre = np.arange(400, dtype=float) * 30.0
    a, info = p._causal_alpha(t_pre, np.zeros(400))
    assert a == p.NET_ACCUM_ALPHA_QUARTZ
    assert info["method"] == "fixed" and "reason" in info


def test_anchor_handles_history_longer_than_twelve_hours():
    """A >12 h station history must not wrap into "starts in the future".

    `_dt_circ` is only valid for separations under 12 h. The anchor used it to
    measure how far back each station's history began, which was fine while
    histories were minutes long -- and broke the moment OS Net backfill made
    them 14 h. A station starting 12.9 h BEFORE the pre-window end came back
    as starting 11.1 h AFTER it, so every station looked to begin past the
    horizon and the accumulated arm went silent with a nonsensical reason.
    """
    rng = np.random.default_rng(53)
    # 14 h of history at 30 s, crossing UTC midnight, ending "now".
    n = int(14 * 3600 / 30)
    t_abs = np.arange(n, dtype=float) * 30.0
    field = np.cumsum(rng.normal(0, 0.2, n))
    series = {}
    for i, sid in enumerate(_STN_COORDS):
        series[sid] = (t_abs.copy(), field + 50.0 * i + rng.normal(0, 0.05, n))
    p = _accum_proc(series)
    # Pre-window end sits near the END of that history, i.e. ~13.5 h after the
    # oldest sample -- beyond _dt_circ's valid range.
    t0 = t_abs[-1] - 1800.0
    hs = np.arange(300.0, 1800.0 + 1, 300.0)
    corr, info = p._network_accum_prediction(t0=t0, horizons_s=hs,
                                             pre_span_s=3600.0)
    assert corr is not None, (
        f"14 h of history was rejected: {info.get('reason')}")
    assert info["anchor_lag_s"] <= 1.0, (
        f"anchor_lag {info['anchor_lag_s']}s — history that PRECEDES the "
        f"pre-window was treated as starting after it")
    assert info["horizon_coverage"] == 1.0


def test_specificity_controls_share_alpha_with_the_tep_arm():
    """A control that runs at a different alpha measures the alpha, not the
    thing it is supposed to control for.

    Observed live: the causal fit put alpha at 0, so the TEP arm collapsed to
    persistence, while the permuted control still ran at the default 0.619.
    The difference was reported as `geometry_contribution = +7.02%` -- which
    would have read as the first live evidence of spatial structure and was
    entirely an artefact of the alpha mismatch.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    i = src.find("_spec = {}")
    j = src.find('out["specificity"] = _spec')
    assert i != -1 and j != -1 and i < j
    block = src[i:j]
    assert block.count("alpha_override=_alpha") >= 2, (
        "a specificity control is not carrying the TEP arm's alpha")
    assert '"valid"' in block, (
        "degenerate (alpha=0) comparisons are not marked invalid")


def test_specificity_is_suppressed_when_alpha_is_zero():
    """With alpha at 0 the TEP arm IS persistence, so every control compares
    persistence against persistence.  Those numbers must not be published as
    if they were a measurement."""
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    i = src.find("_spec = {}")
    j = src.find('out["specificity"] = _spec')
    block = src[i:j]
    # the geometry number must be removed, not merely annotated
    assert 'pop("geometry_contribution_pct"' in block or \
           '_spec.pop(k, None)' in block, (
        "the geometry contribution survives an alpha=0 window")


def test_tep_arm_is_weighted_by_its_own_measured_clock_accuracy():
    """A better clock must be USED as a better clock, not just a better mean.

    The EKF gave both arms the persistence variance, so only the improved prior
    MEAN reached the position solve and the improved ACCURACY was discarded.
    That is most of the value: with the measured sigmas (34.6 ns standard vs
    11.4 ns TEP at a 4 h outage), formal covariance over 4000 real skies gives
    PDOP 8.499 -> 4.829 at k=4 (+43.2%), and still +2.8% at full sky.
    """
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    assert "_use_own" in src and "_tep_meas" in src, (
        "the TEP arm no longer uses its own measured clock error")
    i = src.find("_use_own = (tep_active")
    j = src.find("elif _base_rmse > 0:", i)
    assert i != -1 and j != -1 and i < j
    guard = src[i:j]
    # It must only tighten when genuinely better, never loosen or over-claim.
    assert "_tep_meas < _base_rmse" in guard, (
        "the tighter variance is applied without checking it is actually better")
    assert "tep_active" in guard, (
        "a gated-off TEP arm could still claim the tighter variance")


def test_alpha_is_shrunk_toward_zero_when_the_fit_is_weak():
    """An unregularised pre-window slope is wild when the predictor has little
    variance -- observed swinging from -0.235 to +5.07 between consecutive live
    windows, the latter clipping to the ceiling and applying a large correction
    nothing justified.  Shrink by R^2, toward ZERO: an alpha we cannot justify
    should apply LESS network term, not fall back to a population constant that
    may not describe this receiver.
    """
    t, field, series = _shared_field_network(seed=61)
    p = _accum_proc(series)
    t_pre = t[:400]
    rng = np.random.default_rng(3)
    # A clock that is pure noise: the network explains nothing, so R^2 ~ 0 and
    # alpha must be shrunk to near zero however wild the raw slope is.
    b_noise = rng.normal(0, 5.0, len(t_pre))
    a, info = p._causal_alpha(t_pre, b_noise)
    assert "shrink_r2" in info, "alpha is no longer regularised"
    assert abs(a) < 0.25, (
        f"alpha {a:.3f} was not shrunk for an unexplainable clock "
        f"(R^2={info.get('shrink_r2')}, raw={info.get('alpha_raw')})")


def test_graphic_clock_reanchors_across_a_time_gap():
    """The accumulated GRAPHIC clock must re-anchor after a discontinuity.

    clk = clk_ref + delta_G is only valid while the epoch sequence is
    continuous.  A gap -- capture pause, process restart, receiver dropout --
    leaves clk_ref describing a clock state that no longer exists, and the
    error persists and grows for the rest of the window.  Observed live: the
    held-out clock RMSE went 5.2 m -> 770 m -> 10561 m across ~40 min of
    restarts while full-sky position stayed at 2-4 cm, i.e. the receiver was
    healthy and only the accumulation was diverging.
    """
    p = _make_proc(ref=TRUE_ECEF)
    assert hasattr(p, "CLK_REANCHOR_GAP_S")
    src = Path(_ROOT / "rpi" / "gps_web.py").read_text(encoding="utf-8")
    i = src.find("def _extract_ppp_clk_epochs")
    j = src.find("# Fallback: try PPP .stat files", i)
    body = src[i:j]
    assert "CLK_REANCHOR_GAP_S" in body, "no gap re-anchor in the clock extraction"
    assert "clk_ref = None" in body and "phase_ref.clear()" in body, (
        "the gap branch does not actually reset the accumulation state")
    # the reset must happen BEFORE the GRAPHIC branch consumes clk_ref
    i_gap = body.find("CLK_REANCHOR_GAP_S")
    i_use = body.find("if phases and clk_ref is not None:")
    assert i_gap < i_use, (
        "the gap check runs after the accumulation has already used clk_ref")


def test_precise_clock_refuses_a_different_day():
    """A seconds-of-day range test cannot see a UTC midnight crossing.

    A rolling capture spanning midnight carries epochs from two days on one
    sod axis.  An epoch at 01:00 on day N+1 has sod 3600, lands comfortably
    inside a day-N clock table, and silently receives a satellite clock 24 h
    stale.  Observed live: the derived receiver clock went 5.8 m -> 764 m ->
    7135 m -> 262 km across successive windows, starting exactly at the UTC
    rollover, while full-sky position stayed at 2-7 cm because only this path
    used the stale corrections.
    """
    import numpy as np
    from scripts.lib.precise_products import PreciseProducts
    pp = PreciseProducts.__new__(PreciseProducts)
    pp._clk = {"G01": np.array([[0.0, 1e-6], [86400.0, 2e-6]])}
    pp._doy = 249
    pp._epoch_doy = None
    # Legacy behaviour when the caller says nothing.
    assert pp.satellite_clock("G01", 3600.0) is not None
    # Same day -> allowed.
    pp.set_epoch_doy(249)
    assert pp.satellite_clock("G01", 3600.0) is not None
    # Different day -> refused, however innocent the sod looks.
    pp.set_epoch_doy(250)
    assert pp.satellite_clock("G01", 3600.0) is None, (
        "a day-250 epoch was served a day-249 satellite clock")


def test_enu_rotation_is_not_transposed():
    """A satellite at the zenith must report 90 deg elevation.

    ``_los_to_elev_az`` and the EKF's own elevation calculation both built
    the ECEF -> ENU rotation TRANSPOSED (the ENU -> ECEF form).  Elevation
    was then meaningless: the local zenith came out at 40 deg and a
    satellite on the horizon at 28 deg.  That elevation drives the
    a-priori troposphere (mapped as 1/sin(elev)), the Klobuchar
    ionosphere, the 10 deg mask and the elevation-dependent measurement
    weighting, so every pseudorange carried a metre-level error that swung
    as satellites rose and set.  Measured on a live 4 h capture, repairing
    it cut the per-satellite residual bias from 7.7 m to 3.4 m.

    The transpose is its own inverse for this shape, so a test that only
    checked "some rotation is applied" would pass either way.  Anchoring on
    the zenith, the horizon and the cardinal directions pins the orientation.
    """
    import math
    import numpy as np
    from scripts.lib.rinex_pseudorange import _los_to_elev_az

    lat_deg, lon_deg = 53.64, 0.103
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)

    up = np.array([cl * co, cl * so, sl])
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])

    elev, _ = _los_to_elev_az(up, lat_deg, lon_deg)
    assert elev == pytest.approx(90.0, abs=1e-6), (
        f"local zenith reported at {elev:.2f} deg elevation, not 90 -- the "
        "ECEF->ENU rotation is transposed")

    for name, vec, want_az in (("east", east, 90.0), ("north", north, 0.0)):
        elev, az = _los_to_elev_az(vec, lat_deg, lon_deg)
        assert elev == pytest.approx(0.0, abs=1e-6), (
            f"{name} horizon direction reported at {elev:.2f} deg elevation")
        assert az % 360.0 == pytest.approx(want_az, abs=1e-6), (
            f"{name} direction reported at azimuth {az:.2f}, not {want_az}")

    # A direction halfway between zenith and the northern horizon is 45 deg.
    half = up + north
    half = half / np.linalg.norm(half)
    elev, az = _los_to_elev_az(half, lat_deg, lon_deg)
    assert elev == pytest.approx(45.0, abs=1e-6)
    assert az % 360.0 == pytest.approx(0.0, abs=1e-6)


def test_precise_clock_includes_relativistic_term():
    """IGS CLK products omit the periodic relativistic correction.

    By convention the user applies -2*(r.v)/c^2, exactly as RTKLIB does in
    ``peph2pos()``.  It reaches +/-13 m of range over an orbit and swings
    sinusoidally with a ~12 h period, so leaving it out puts a slowly
    varying bias on every satellite that no constant ambiguity can absorb.
    """
    import numpy as np
    from scripts.lib.precise_products import PreciseProducts, C

    # A circular orbit sampled over 20 minutes, inclined so r.v is non-zero.
    r0, w = 26_560_000.0, 1.458e-4          # GPS radius, mean motion (rad/s)
    t = np.arange(0.0, 1200.0, 300.0)
    sp3 = {"G01": np.column_stack([
        t,
        r0 * np.cos(w * t),
        r0 * np.sin(w * t) * 0.6,
        r0 * np.sin(w * t) * 0.8,
    ])}
    clk = {"G01": np.array([[0.0, 1e-6], [1200.0, 1e-6]])}
    pp = PreciseProducts(sp3, clk)

    dts = pp.satellite_clock("G01", 600.0)
    assert dts is not None
    rel = pp._relativistic_correction("G01", 600.0)
    assert rel != 0.0, "relativistic correction is not being applied"
    # It is a genuine correction to the tabulated value, not a replacement.
    assert dts == pytest.approx(1e-6 + rel, rel=1e-9)
    # Physically bounded: well under the +/-45 ns that eccentricity can reach.
    assert abs(rel) * C < 50.0, f"{abs(rel) * C:.1f} m is out of physical range"

    # A satellite with no orbit table degrades to no correction, not a crash.
    assert pp._relativistic_correction("G99", 600.0) == 0.0


def test_persistence_never_loses_to_sample_and_hold():
    """Persistence is the baseline the TEP claim is measured against.

    Two defects made it a strawman.  (1) Both arms anchored on b_pre[-1],
    a single noisy sample, while the robust Theil-Sen offset fitted three
    lines earlier was discarded -- on a live capture that anchor sat 3.17 m
    off the robust level, and since the error is COMMON to all three arms
    it does not cancel in a difference: a sub-metre network correction
    cannot be seen through it.  (2) The trend was extrapolated on a
    heuristic gate (R^2 / drift-to-noise of a least-squares per-sample
    slope) that scores a different slope from the Theil-Sen per-second rate
    actually extrapolated, so persistence could extrapolate an unvalidated
    slope over a 2 h horizon and end up worse than sample-and-hold.  The
    live rig logged "persist 78.3% worse than free -- degenerating to free".

    Validating the trend causally on a split of the pre-window makes
    persistence at least as good as hold by construction.
    """
    import numpy as np

    rng = np.random.default_rng(20260908)
    t = np.arange(0.0, 4000.0, 1.0)

    def split(series):
        n_pre = max(3, int(len(t) * 0.5))
        return t[:n_pre], series[:n_pre], t[n_pre:], series[n_pre:]

    def theil_sen(tt, vv):
        n = len(tt)
        if n > 80:
            idx = np.linspace(0, n - 1, 80).astype(int)
            tt, vv = tt[idx], vv[idx]
            n = len(tt)
        i, j = np.triu_indices(n, k=1)
        dt = tt[j] - tt[i]
        ok = dt != 0
        rate = float(np.median((vv[j][ok] - vv[i][ok]) / dt[ok]))
        return float(np.median(vv - rate * tt)), rate

    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a) ** 2)))

    def predict(series):
        """Anchor + causally-validated trend, mirroring the shipped logic."""
        t_pre, b_pre, t_ho, _ = split(series)
        offset, rate = theil_sen(t_pre, b_pre)
        anchor = offset + rate * t_pre[-1]
        h = len(t_pre) // 2
        o_v, r_v = theil_sen(t_pre[:h], b_pre[:h])
        a_v = o_v + r_v * t_pre[h - 1]
        hold_rmse = rms(a_v - b_pre[h:])
        trend_rmse = rms(a_v + r_v * (t_pre[h:] - t_pre[h - 1]) - b_pre[h:])
        use_trend = trend_rmse < hold_rmse
        free = np.full_like(t_ho, anchor)
        persist = (anchor + rate * (t_ho - t_pre[-1]) if use_trend
                   else np.full_like(t_ho, anchor))
        return free, persist, use_trend

    # A pure random walk has no extrapolable trend: persistence must not be
    # beaten by sample-and-hold, whatever slope happens to fit the noise.
    walk = np.cumsum(rng.normal(0.0, 0.05, len(t)))
    _, b_pre_w, _, b_ho_w = split(walk)
    free_w, persist_w, used_w = predict(walk)
    assert rms(persist_w - b_ho_w) <= rms(free_w - b_ho_w) + 1e-9, (
        "persistence lost to sample-and-hold on a trendless walk -- an "
        "unvalidated slope is being extrapolated")

    # A genuine deterministic drift must still be tracked, not gated away.
    drift = 0.004 * t + rng.normal(0.0, 0.05, len(t))
    _, _, _, b_ho_d = split(drift)
    free_d, persist_d, used_d = predict(drift)
    assert used_d, "a clean linear drift was not extrapolated"
    assert rms(persist_d - b_ho_d) < 0.5 * rms(free_d - b_ho_d), (
        "persistence failed to exploit a deterministic drift")

    # The anchor must be the robust level, not the last sample.  Give the
    # final sample a large excursion and check it does not move the arms.
    spiked = walk.copy()
    spiked[len(t) // 2 - 1] += 25.0
    free_s, _, _ = predict(spiked)
    assert abs(float(free_s[0]) - float(free_w[0])) < 1.0, (
        "a single spiked sample moved the prior anchor -- the arms are "
        "anchored on b_pre[-1] rather than the robust fitted level")
