"""Guards for receiver-clock reset rejection and clock-source ranking.

The rig ranked candidate .stat clock solutions by MAD, but published an RMS.
MAD cannot see a reset: on a real window the bare SPP solution scored MAD
5.68 m against PPP-static's 5.58 m -- effectively a tie -- while its RMS was
4237 m against 5.51 m, because 6% of its epochs were resets spanning 20 km
peak to peak.  MAD ranking therefore chose the source whose reported RMS was
770x worse, the rig published a 1077 m clock RMSE, and the 0.42 m TEP
correction against it displayed as +0.0%.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpi.gps_web import PPPProcessor  # noqa: E402

_M_PER_NS = 0.299792458


def _write_stat(path: Path, clocks_m):
    """Write a .stat whose $CLK column 5 is in NANOSECONDS, as RTKLIB does.

    RTKLIB emits $CLK,week,tow,stat,1,clk1,clk2,clk3,clk4 where stat is the
    solution status (5=fixed, 4=float, ...).  stat=0 is SOLQ_NONE and the
    parser correctly drops those epochs, so the fixture must write a real
    stat value, not 0.
    """
    with open(path, "w") as f:
        for i, c in enumerate(clocks_m):
            f.write(f"$CLK,2026,{1000.0 + i},5,1,{c / _M_PER_NS:.6f},0,0\n")


@pytest.fixture
def proc(tmp_path):
    p = PPPProcessor.__new__(PPPProcessor)
    p._tmp = tmp_path
    return p


def _series(n=400, sigma=5.0, seed=0):
    return list(np.random.default_rng(seed).normal(0.0, sigma, n))


def test_resets_are_dropped_not_merely_logged(proc, tmp_path):
    clean = _series()
    with_resets = list(clean)
    for i in (10, 50, 123, 300):
        with_resets[i] += 10000.0            # receiver clock reset
    _write_stat(tmp_path / "w.pos.stat", with_resets)
    ep = proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    assert ep is not None
    vals = np.array([e["clk_m"] for e in ep])
    assert len(vals) == len(clean) - 4, "reset epochs must leave the series"
    assert np.abs(vals).max() < 1000.0


def test_clean_source_beats_one_with_resets(proc, tmp_path):
    """A tie on MAD must not let a reset-ridden source win."""
    base = _series(sigma=5.0, seed=1)
    spp = list(base)
    for i in range(0, len(spp), 17):          # ~6% of epochs, as observed
        spp[i] += 8000.0
    _write_stat(tmp_path / "w.pos.stat", spp)                 # bare SPP
    _write_stat(tmp_path / "w_ppp_sta.pos.stat", base)        # PPP-static
    ep = proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    vals = np.array([e["clk_m"] for e in ep])
    # the static solution is intact, so it should be the one returned
    assert len(vals) == len(base)
    assert float(np.sqrt(np.mean((vals - np.median(vals)) ** 2))) < 20.0


def test_nanosecond_conversion_is_applied(proc, tmp_path):
    """$CLK is nanoseconds; the pipeline is metres."""
    _write_stat(tmp_path / "w.pos.stat", [3.0] * 40)
    ep = proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    assert np.allclose([e["clk_m"] for e in ep], 3.0, atol=1e-6)


def test_a_quiet_clock_is_not_over_screened(proc, tmp_path):
    """The absolute floor protects a clock whose MAD is very small."""
    quiet = list(np.random.default_rng(3).normal(0.0, 0.02, 300))
    _write_stat(tmp_path / "w.pos.stat", quiet)
    ep = proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    assert len(ep) == len(quiet)


def test_unparseable_stat_is_reported_not_silently_skipped(proc, tmp_path, caplog):
    """A corrupt .stat must not look identical to one that was never written.

    Both used to vanish into `except Exception: continue`, so the selection
    fell through to a worse clock source with nothing to say why.
    """
    import logging
    (tmp_path / "w.pos.stat").write_bytes(b"\x00\xff not a stat file\n")
    _write_stat(tmp_path / "w_ppp_sta.pos.stat", [1.0] * 40)
    with caplog.at_level(logging.INFO):
        ep = proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    # the good candidate still wins
    assert ep is not None and len(ep) == 40
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "w.pos.stat" in msgs, "the unusable candidate must be named"


def test_thin_stat_is_reported(proc, tmp_path, caplog):
    import logging
    _write_stat(tmp_path / "w.pos.stat", [1.0] * 3)          # below the 8 floor
    _write_stat(tmp_path / "w_ppp_sta.pos.stat", [1.0] * 40)
    with caplog.at_level(logging.INFO):
        proc._extract_ppp_clk_epochs(tmp_path / "w.obs")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "only 3 $CLK epochs" in msgs
