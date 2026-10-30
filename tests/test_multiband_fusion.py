"""Guards for the multi-band kriging fusion.

The fusion is designed as inverse-variance weighting across frequency bands
(Paper 1's lambda(f)).  On the live network it silently degenerated into a
plain unweighted average, and the arm came out 525.9% WORSE than the single
band it is built from (6.222 m vs 0.994 m).

The mechanism is a saturating variance.  Simple kriging gives
sigma^2 = 1 - w.rho, and at lambda = 1,459-4,677 km across a ~3,000 km
aperture the correlations are ~1, so w.rho >= 1 and `max(..., 0.0)` returns
exactly 0.0 for every band while the full band -- computed by a different
routine -- returns 0.0206.  Three identical zeros clamp to a common floor,
the weights come out 1/3 each, and the weighting carries no information at
all.  The clamp hid this completely.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_saturated_band_variance_is_recorded_not_hidden():
    src = (Path(__file__).resolve().parents[1]
           / "scripts/lib/tep_kernel.py").read_text()
    # c_rt = rho_rt * g_rt is the Schur-consistent cross-covariance the
    # kriging system actually solves; the recorded raw variance must use it.
    assert "_raw_var = 1.0 - float(np.dot(w, c_rt))" in src, (
        "the pre-clamp variance must be kept")
    assert '"variance_saturated": bool(_saturated)' in src
    assert '"variance_raw": float(_raw_var)' in src
    # the clamp itself is still required -- a negative variance is not usable
    assert "sk_var = float(max(_raw_var, 0.0))" in src


def test_uninformative_weights_are_flagged():
    """1/n weights produced by saturation are not an inverse-variance fusion."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts/lib/tep_kernel.py").read_text()
    assert "weights_uninformative" in src
    assert "_all_saturated" in src
    # full_band must be excluded from the saturation test: it uses the other
    # routine and legitimately reports a non-zero variance
    assert 'bd.get("name") != "full_band"' in src


def test_near_unit_correlations_collapse_the_variance_weighting():
    """The mechanism: tiny variances clamp to a common floor -> equal weights.

    Live data drives the simple-kriging variance to exactly 0.0 for every
    band.  A synthetic with near-unit correlations lands close (~1.7e-3), and
    either way the clamp in the fusion (`np.clip(var, floor, 1e6)` with
    floor >= 0.01) maps them all to the SAME value, so 1/var gives 1/n
    weights that encode no evidence at all.
    """
    rho_rt = np.array([0.999, 0.998, 0.9985])
    K = np.array([[1.0, 0.998, 0.999],
                  [0.998, 1.0, 0.9985],
                  [0.999, 0.9985, 1.0]]) + 1e-9 * np.eye(3)
    w = np.linalg.solve(K, rho_rt)
    raw = 1.0 - float(np.dot(w, rho_rt))
    assert raw < 0.01, (
        f"near-unit correlations must drive the variance far below the "
        f"fusion floor (got {raw:.6f})")

    # three such variances, clamped exactly as the fusion clamps them
    floor = 0.01
    band_vars = np.clip(np.array([raw, raw * 0.5, 0.0]), floor, 1e6)
    weights = 1.0 / band_vars
    weights = weights / weights.sum()
    assert np.allclose(weights, 1.0 / 3.0), (
        "clamped variances must be shown to yield uninformative 1/n weights")


def test_multiband_is_not_silently_preferred_when_it_loses():
    """The arm ran 525.9% worse than single-band and was still displayed.

    It is diagnostic-only, so this asserts the provenance a reader needs to
    see that the weighting was uninformative rather than measured.
    """
    src = (Path(__file__).resolve().parents[1]
           / "scripts/lib/tep_kernel.py").read_text()
    assert "weights_uninformative" in src
    assert "variance_saturated" in src
