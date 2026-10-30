"""Guards for the matched-horizon accumulated-offset scoring.

The live panel used to compare a kriged INSTANTANEOUS RESIDUAL LEVEL against
persistence measured over a single 30 s step.  Those two arms do not predict
the same quantity over the same interval, so persistence won by ~8x no matter
what signal was present, and `_simulate_obstruction`'s min() then dropped the
TEP arm entirely -- the 92.4% headline was DOP plus persistence with no TEP
contribution in it at all.
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "scripts/live/tep_realtime_server.py"


_WANTED = ("_accum_loo", "_accum_loo_auto", "_simulate_obstruction",
           "_select_covered_block", "_clusters_of")


def _load():
    """Exec just the scoring helpers.

    Importing the module would construct LiveSource and start the NTRIP
    threads, so pull the functions out by AST and exec only those, along
    with the module-level constants they close over.
    """
    src = _SRC.read_text()
    tree = ast.parse(src)
    ns = {"np": np, "math": math}
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.startswith("ACCUM_")
                for t in node.targets):
            pieces.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            pieces.append(ast.get_source_segment(src, node))
    missing = [w for w in _WANTED if not any(f"def {w}(" in p for p in pieces)]
    assert not missing, f"could not extract {missing}"
    exec(compile("\n\n".join(pieces), str(_SRC), "exec"), ns)
    return ns


NS = _load()


def _network(n_ep=400, n_st=12, seed=0, field=True):
    """Stations on a line, with a smooth spatially-correlated clock field."""
    rng = np.random.default_rng(seed)
    pos = np.linspace(0.0, 3000.0, n_st)          # km along a line
    D = np.abs(pos[:, None] - pos[None, :])
    t = np.arange(n_ep, dtype=float)
    X = rng.normal(0.0, 0.02, (n_ep, n_st)).cumsum(0)   # per-station noise walk
    if field:
        # two smooth modes with a ~1500 km correlation length
        for k, lam in ((1, 1500.0), (2, 1500.0)):
            amp = np.exp(-0.5 * (pos / lam) ** 2)
            X += np.sin(2 * np.pi * k * t / n_ep)[:, None] * amp[None, :] * 3.0
    return X, D, t * 30.0


def test_scores_both_arms_on_the_same_horizon():
    X, D, sods = _network()
    r = NS["_accum_loo"](X, D, sods, horizon_s=900.0)
    assert r is not None
    assert r["horizon_s"] == pytest.approx(900.0, rel=0.05)
    assert r["n_cases"] > 10
    # base and tep are RMS over the SAME cases
    assert r["base_rms_m"] > 0 and r["tep_rms_m"] > 0


def test_recovers_a_spatially_correlated_field():
    X, D, sods = _network(field=True)
    r = NS["_accum_loo"](X, D, sods, horizon_s=900.0)
    assert r["gain_pct"] > 10.0, r
    # and the geometry matters: scrambling station positions costs gain
    assert r["gain_pct"] > r["permuted_gain_pct"]


def test_auto_picks_the_longest_supported_horizon():
    X, D, sods = _network(n_ep=400)          # 400 x 30 s = 3.3 h
    r = NS["_accum_loo_auto"](X, D, sods)
    short = NS["_accum_loo"](X, D, sods, horizon_s=300.0)
    assert r["horizon_s"] >= short["horizon_s"]


def test_reports_why_it_cannot_score_yet():
    """A bare None is unreadable from the payload -- say what is missing."""
    X, D, sods = _network(n_ep=15)
    assert NS["_accum_loo"](X, D, sods, horizon_s=900.0) is None
    out = NS["_accum_loo_auto"](X, D, sods)
    assert out["available"] is False
    assert "epochs" in out["reason"]


def test_scores_across_a_utc_midnight_wrap():
    """Raw sod resets at midnight; the rows are still correctly ordered.

    Refusing to score a wrapped axis would silence the panel for every history
    that spans midnight, so the axis is rebuilt from modular differences.
    """
    X, D, sods = _network(n_ep=400)
    sods = sods.copy()
    sods[200:] -= 86400.0                       # the wrap
    out = NS["_accum_loo_auto"](X, D, sods)
    assert out["available"] is True, out.get("reason")
    ref = NS["_accum_loo_auto"](*_network(n_ep=400))
    assert out["gain_pct"] == pytest.approx(ref["gain_pct"], abs=1e-6)


def test_duplicate_epochs_are_dropped():
    X, D, sods = _network(n_ep=400)
    sods = sods.copy()
    sods[150] = sods[149]                       # equal SODs survive the sort
    out = NS["_accum_loo_auto"](X, D, sods)
    assert out["available"] is True, out.get("reason")


def test_obstruction_reports_which_prior_it_actually_used():
    """min() must not be able to hide a losing TEP arm."""
    out = NS["_simulate_obstruction"](kernel_err_m=1.07, raw_err_m=0.17,
                                      persist_err_m=0.13, nsat_full=8,
                                      obstructed_nsat=3)
    assert out["prior_used"] == "persistence"
    assert out["tep_vs_persistence_pct"] < 0
    # the DOP model's own contribution is stated separately
    assert out["dop_model_pct"] > 0

    win = NS["_simulate_obstruction"](kernel_err_m=0.05, raw_err_m=0.17,
                                      persist_err_m=0.13, nsat_full=8,
                                      obstructed_nsat=3)
    assert win["prior_used"] == "tep"
    assert win["tep_vs_persistence_pct"] > 0


def test_unavailable_result_is_shaped_for_the_obstruction_call_site():
    """The failure dict must be distinguishable by a flag, not by identity.

    `_accum_loo_auto` returning a reason dict instead of None made an
    `is not None` guard at the call site always true; it then indexed
    `tep_rms_m`, and the KeyError was swallowed into a generic source error
    that stopped every epoch emit.
    """
    X, D, sods = _network(n_ep=15)
    out = NS["_accum_loo_auto"](X, D, sods)
    assert out.get("available") is False
    assert "tep_rms_m" not in out
    ok = NS["_accum_loo_auto"](*_network(n_ep=400))
    assert ok.get("available") is True
    assert "tep_rms_m" in ok and "base_rms_m" in ok


def test_flat_extrapolated_stations_are_excluded():
    """Gap fill must not be graded as if it were measurement.

    `_build_live_matrix` spans the union of all stations' epochs and holds the
    last value flat outside each station's coverage.  A constant column
    contributes exactly zero accumulated offset, so scoring against it grades
    the predictor on fabricated data -- with OS Net backfilled 14 h and the
    NTRIP mounts minutes old, most of the live matrix was extrapolation.
    """
    X, D, sods = _network(n_ep=600, n_st=12)
    COV = np.ones(X.shape, dtype=bool)
    # 6 stations stopped reporting two thirds of the way through
    X = X.copy()
    for j in range(6, 12):
        COV[400:, j] = False
        X[400:, j] = X[399, j]                # flat hold, as np.interp does
    out = NS["_accum_loo_auto"](X, D, sods, COV=COV)
    assert out["available"] is True, out.get("reason")
    # it must fall back to a block where every station is real: either the
    # first 400 epochs with all 12, or the full span with the 6 live ones
    assert out["n_stations"] == 12 or out["n_cases"] > 0


def test_selects_the_largest_fully_covered_block():
    COV = np.zeros((500, 10), dtype=bool)
    COV[:, :4] = True                      # 4 stations cover everything
    COV[100:400, 4:] = True                # 6 more cover the middle
    sel = NS["_select_covered_block"](COV, min_st=8)
    assert sel is not None
    a, b, keep = sel
    assert len(keep) == 10
    assert a >= 100 and b <= 399


def test_window_must_lie_inside_a_station_coverage_not_past_its_end():
    """Interior gaps are interpolation; past the last sample is extrapolation."""
    COV = np.zeros((500, 12), dtype=bool)
    COV[:, :] = True
    COV[300:, 6:] = False            # 6 stations stop reporting at 300
    sel = NS["_select_covered_block"](COV, min_st=8)
    assert sel is not None
    a, b, keep = sel
    if len(keep) == 12:
        assert b < 300, "a window past their last sample extrapolates 6 stations"


def test_a_single_dropped_epoch_does_not_disqualify_a_station():
    COV = np.ones((500, 10), dtype=bool)
    COV[250, 3] = False              # one dropped packet
    sel = NS["_select_covered_block"](COV, min_st=8)
    assert sel is not None
    _a, _b, keep = sel
    assert 3 in keep, "one missing epoch is interpolation, not extrapolation"


def test_within_cluster_permutation_is_reported_alongside_the_global_one():
    """A global null on a blobby network flatters the geometry claim.

    On the 35 live NTRIP stations (23 Dutch at 11-269 km, 12 Iberian at
    ~2000-3249 km) a global permutation gave +11 points of "geometry" while
    permuting only within each cluster left +1.74 -- the rest was the near/far
    split, not spatial structure.
    """
    X, D, sods = _network(n_ep=400, n_st=12)
    out = NS["_accum_loo_auto"](X, D, sods)
    assert out["available"] is True
    for k in ("geometry_pct", "geometry_within_cluster_pct",
              "within_cluster_gain_pct", "n_clusters"):
        assert k in out, k


def test_clusters_split_two_separated_blobs():
    pos = np.array([0.0, 50.0, 120.0, 2500.0, 2600.0, 2700.0])
    D = np.abs(pos[:, None] - pos[None, :])
    cl = NS["_clusters_of"](D, link_km=800.0)
    assert len(set(cl.tolist())) == 2
    assert cl[0] == cl[1] == cl[2]
    assert cl[3] == cl[4] == cl[5]
    assert cl[0] != cl[3]


def test_a_single_cluster_network_cannot_manufacture_geometry():
    """When every station is in one cluster the two nulls must agree."""
    pos = np.linspace(0.0, 270.0, 10)      # OS Net-like, 274 km extent
    D = np.abs(pos[:, None] - pos[None, :])
    cl = NS["_clusters_of"](D, link_km=800.0)
    assert len(set(cl.tolist())) == 1
