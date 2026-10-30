"""Guards for the Step 172 satellite-free holdover scorer.

The --selftest entries (synthetic_coupled / synthetic_uncoupled) must be
quarantined from the real ensemble: they are pipeline proofs, not captures.
Before the quarantine flag was applied on the selftest path, a run that
combined --selftest with real --ticks files let synthetic_coupled's ~128
high-alignment terms enter the pooled run-level statistic, fabricating a
run-level PASS (pooled PA ~0.95, z ~4.5) on a real ensemble that was in
fact null.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "steps"))

import step_172_satellite_free_holdover as s172  # noqa: E402


def _write_capture(tmp_path: Path, n=19000, dt=1.15):
    """A null 'real' capture: quadratic drift + random-walk wander, no
    coupling to dep.  ~6 h span resolves >=4 in-band bins at nseg=2."""
    rng = np.random.default_rng(7)
    t0 = 1_793_200_000.0
    t = t0 + np.arange(n) * dt
    ph = (1e-4 * np.arange(n) / n + np.cumsum(rng.normal(0, 5e-11, n)))
    ticks = tmp_path / "ticks_null.csv"
    with ticks.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_idx", "ref_epoch_s", "dut_phase_s"])
        for i in range(n):
            w.writerow([i, f"{t[i]:.6f}", f"{ph[i]:.9f}"])
    # dep prediction with in-band structure on the same span
    dep = np.zeros(n)
    for per in (2400, 3600, 5400, 7200):
        dep += np.sin(2 * np.pi * np.arange(n) * dt / per + 0.3)
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(
        {"t_epoch": t.tolist(), "dep_m": dep.tolist()}))
    return str(ticks), str(pred)


def _run(tmp_path, argv, monkeypatch, n_null=50):
    monkeypatch.setattr(s172, "N_NULL", n_null)
    monkeypatch.setattr(s172, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(sys, "argv", ["step_172", *argv])
    s172.main()
    return json.loads((tmp_path / "out.json").read_text())


def test_selftest_quarantined_from_pool(tmp_path, monkeypatch):
    ticks, pred = _write_capture(tmp_path)
    out = _run(tmp_path,
               ["--selftest", "--ticks", ticks, "--pred", pred],
               monkeypatch)
    synth = [c for c in out["captures"]
             if c["tag"].startswith("synthetic_")]
    assert synth and all(c.get("simulated") for c in synth)
    # The pooled ensemble must contain only the real capture's terms:
    # ~2 segs x ~5 bins, never the selftest's 128 coupled terms.
    assert out["pooled"]["n_terms"] < 40
    assert out["verdict"] == "FAIL"


def test_selftest_only_verdict(tmp_path, monkeypatch):
    out = _run(tmp_path, ["--selftest"], monkeypatch)
    assert out["verdict"] == "SELFTEST_PASS"
    assert out["pooled"] is None


def test_simulation_tagged_ticks_quarantined(tmp_path, monkeypatch):
    ticks, pred = _write_capture(tmp_path)
    src = Path(ticks).read_text()
    sim_ticks = tmp_path / "ticks_sim.csv"
    sim_ticks.write_text("# SIMULATION\n" + src)
    out = _run(tmp_path,
               ["--ticks", str(sim_ticks), "--pred", pred],
               monkeypatch)
    # a SIMULATION-tagged capture is scored but quarantined: no real
    # ensemble remains, and no verdict may be read as evidence
    assert out["pooled"] is None or out["pooled"]["n_terms"] == 0
    assert out["verdict"] in ("NO_DATA", "FAIL")
