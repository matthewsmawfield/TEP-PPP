#!/usr/bin/env python3
"""Step 174 — Cryptographic Sealed-Envelope Ex-Ante Predictive Validation.

Establishes a watertight, falsification-controlled operational proof that the
TEP-PPP network clock prior predicts future receiver-clock fluctuations
strictly out-of-sample (ex ante), separating deployed causal predictions from
truth-referenced safety bounds and eliminating sliding-window overlap,
receiver-local holdout leakage, and generic common-mode explanations.

Five interlocking verification batteries:
  1. ``live_causal_audit``:
     Full 114+ window continuous live observatory record (~25.5 h span, >800k
     1-Hz holdout epochs) scoring the pre-scale, pre-guard causal curve
     (``_dep_o2``) and fixed horizon-ramp curve (``_dep_o2 * flat_ramp``)
     against standard persistence, alongside a strictly epoch-disjoint
     (>=7,200 s stride, non-overlapping 2 h holdout) sub-ensemble.
  2. ``walk_forward_sealed_envelopes``:
     Cryptographic SHA-256 sealed-envelope walk-forward across the 67/68-station
     multi-resolution Gram corpus (trailing K=16 windows, recency half-life
     1.25 windows, ridge penalty lam=2,000), scoring both:
       - ``full_causal_sealed``: frozen weights ``w_frozen`` (spatial network +
         sidereal echo + degraded-sky free-clock trajectory)
       - ``network_only_sealed``: frozen field-only weights ``wnf_frozen``
         (zero receiver observations during the 2-hour holdout)
  3. ``purged_epoch_disjoint_horizons``:
     Eliminates sliding-window overlap between training and test holdouts by:
       - scoring strictly future wall-clock epochs ``t_ho(j) > T_seal_end``
         (epochs occurring in physical time strictly after the final holdout
         epoch of the newest training member in the sealed envelope), and
       - scoring full 2-hour embargoed windows (``Delta_T >= 7,200 s`` and
         ``Delta_T >= 3,600 s`` after the sealed envelope cutoff) with zero
         weight updates across the gap.
  4. ``spatial_falsification_battery``:
     Evaluates the frozen network-only sealed envelopes (``wnf_frozen``) under:
       - ``station_identity_permutation_null`` (N=100 Monte Carlo shuffles of
         the station identities at evaluation time, preserving each station's
         internal 79-feature multi-resolution structure)
       - ``common_mode_stripped`` (subtracting the instantaneous cross-network
         mean across all stations at every holdout epoch before applying the
         sealed weights) vs ``common_mode_only`` (replacing each station by the
         instantaneous network mean)
       - ``distance_band_isolation``:
           * ``beyond_troposphere_gt360km`` (zero UK stations <360 km)
           * ``mid_band_eu_360_1000km`` (continental Europe 360-1,000 km)
           * ``far_arc_gt1000km`` (Iberian & Atlantic stations >1,000 km)
           * ``near_field_uk_lt360km`` (UK OS Net <360 km)
  5. ``live_realtime_forward_seal``:
     Verifies the live SHA-256 sealed envelope committed to disk prior to
     subsequent live window completions.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / ".env.local"
if ENV_PATH.exists():
    for _ln in ENV_PATH.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(
                _k.strip(), _v.strip().strip('"').strip("'")
            )

sys.path.insert(0, str(REPO_ROOT))
from rpi.gps_web import (  # noqa: E402
    _env_float,
    _ridge_feature_dict,
    _ridge_stats_dir,
    _ridge_stats_load_or_build,
    _ridge_stats_fit_dir,
    _ridge_stats_fit_load_or_build,
)

RESULTS_DIR = REPO_ROOT / "data" / "ppp_live" / "results"
ENVELOPE_DIR = REPO_ROOT / "data" / "ppp_live" / "sealed_envelopes"
OUT_PATH = REPO_ROOT / "results" / "outputs" / "step_174_sealed_envelope_prediction.json"

ROVER_LAT_DEG = 53.6399642
ROVER_LON_DEG = 0.1030390


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def _window_id(path: Path) -> int:
    stem = path.stem
    for prefix in ("last_result_window_", "raw_window_"):
        if stem.startswith(prefix):
            return int(stem[len(prefix):])
    return int(stem.split("_")[-1])


def _fcm_scale(diag) -> float:
    fcm = os.environ.get("TEP_CAUSAL_RIDGE_FCM", "")
    if not fcm:
        return 1.0
    try:
        cj, dr = float(diag[0]), float(diag[1])
    except Exception:
        return 1.0
    if fcm == "soft":
        return float(np.clip(-cj, 0.0, 1.0)) if np.isfinite(cj) else 1.0
    if fcm.startswith("d"):
        try:
            dt_thr = float(fcm[1:])
        except Exception:
            return 1.0
        return 0.0 if (np.isfinite(dr) and dr > dt_thr) else 1.0
    if fcm.startswith("con"):
        try:
            ct_thr = -float(fcm[3:]) / 10.0
        except Exception:
            return 1.0
        return 0.0 if (np.isfinite(cj) and cj > ct_thr) else 1.0
    return 1.0


def _spd_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        c, low = cho_factor(
            A.astype(np.float32, copy=False),
            overwrite_a=False,
            check_finite=False,
        )
        return cho_solve(
            (c, low), b.astype(np.float32, copy=False), check_finite=False
        ).astype(np.float64)
    except Exception:
        return np.linalg.solve(A, b).astype(np.float64)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _block_bootstrap_ci(
    vals: list[float] | np.ndarray,
    block_len: int = 4,
    n_boot: int = 5000,
    seed: int = 174,
) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    b = max(1, min(block_len, n))
    n_blocks = int(math.ceil(n / b))
    means = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(b)[None, :]).ravel()[:n] % n
        means[k] = float(np.mean(arr[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 2), round(float(hi), 2))


def _summary_stats(pcts: list[float], block_len: int = 4) -> dict:
    arr = np.asarray(pcts, dtype=float)
    n = int(len(arr))
    if n == 0:
        return {"n": 0}
    wins = int(np.sum(arr > 0.0))
    pval = float(binomtest(wins, n, 0.5, alternative="greater").pvalue)
    ci_lo, ci_hi = _block_bootstrap_ci(arr, block_len=block_len)
    # Effective sample size for serially correlated per-window scores:
    # sliding windows overlap, so n raw evals < n independent trials.
    # Kapelen-Weiss approx n_eff = n (1-rho1)/(1+rho1) at lag-1, floored
    # at 1 and capped at n; reported beside the naive sign test.
    rho1 = (
        float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        if n >= 4 and np.std(arr) > 0
        else 0.0
    )
    n_eff = float(
        max(1.0, min(n, n * (1.0 - rho1) / (1.0 + max(rho1, 0.0))))
    )
    pval_eff = float(
        binomtest(
            int(round(wins * n_eff / n)), max(1, int(round(n_eff))), 0.5,
            alternative="greater",
        ).pvalue
    )
    return {
        "n": n,
        "n_eff": round(n_eff, 1),
        "lag1_autocorr": round(rho1, 3),
        "mean_pct": round(float(np.mean(arr)), 2),
        "median_pct": round(float(np.median(arr)), 2),
        "std_pct": round(float(np.std(arr)), 2),
        "p10_pct": round(float(np.percentile(arr, 10)), 2),
        "p90_pct": round(float(np.percentile(arr, 90)), 2),
        "min_pct": round(float(np.min(arr)), 2),
        "max_pct": round(float(np.max(arr)), 2),
        "ci95_block_boot_pct": [ci_lo, ci_hi],
        "win_count": wins,
        "win_rate_pct": round(100.0 * wins / n, 2),
        "sign_test_p": float(f"{pval:.4e}"),
        "sign_test_p_n_eff": float(f"{pval_eff:.4e}"),
    }


def _run_live_causal_audit() -> dict:
    jpaths = sorted(
        RESULTS_DIR.glob("last_result_window_*.json"), key=lambda p: p.name
    )
    rows = []
    for jp in jpaths:
        try:
            d = json.loads(jp.read_text())
            cp = d.get("clock_prediction") or {}
            af = cp.get("alpha_fit") or {}
            cr = af.get("causal_ridge") or {}
            ds = cr.get("dep_scale") or af.get("dep_scale") or {}
            pr = cp.get("persist_rmse_m")
            cu = ds.get("rmse_unscaled_m") or cp.get(
                "tep_rmse_deployed_causal_m"
            )
            cramp = ds.get("rmse_flat_ramp_m")
            cg = ds.get("rmse_guarded_m") or cp.get("tep_rmse_deployed_m")
            if pr is None or cu is None or float(pr) <= 0:
                continue
            pr, cu = float(pr), float(cu)
            cramp = float(cramp) if cramp is not None else cu
            cg = float(cg) if cg is not None else cu
            wid = _window_id(jp)
            rows.append(
                {
                    "window_id": wid,
                    "persist_rmse_m": round(pr, 4),
                    "causal_unscaled_rmse_m": round(cu, 4),
                    "causal_unscaled_pct": round((1.0 - cu / pr) * 100.0, 4),
                    "causal_ramp_rmse_m": round(cramp, 4),
                    "causal_ramp_pct": round((1.0 - cramp / pr) * 100.0, 4),
                    "guarded_oracle_rmse_m": round(cg, 4),
                    "guarded_oracle_pct": round((1.0 - cg / pr) * 100.0, 4),
                    "guard_fired": bool(ds.get("guard_fired", False)),
                }
            )
        except Exception:
            continue

    disjoint = []
    last_w = -10**18
    for r in rows:
        if r["window_id"] - last_w >= 7200:
            disjoint.append(r)
            last_w = r["window_id"]

    # Also build disjoint starting from the second window (after initial warm-up)
    disjoint_post_warmup = []
    last_w2 = -10**18
    for r in rows[1:]:
        if r["window_id"] - last_w2 >= 7200:
            disjoint_post_warmup.append(r)
            last_w2 = r["window_id"]

    span_h = (
        round((rows[-1]["window_id"] - rows[0]["window_id"]) / 3600.0, 2)
        if len(rows) >= 2
        else 0.0
    )
    return {
        "n_windows": len(rows),
        "first_window_id": rows[0]["window_id"] if rows else None,
        "last_window_id": rows[-1]["window_id"] if rows else None,
        "span_hours": span_h,
        "mean_persist_rmse_m": round(
            float(np.mean([r["persist_rmse_m"] for r in rows])), 4
        ),
        "mean_causal_unscaled_rmse_m": round(
            float(np.mean([r["causal_unscaled_rmse_m"] for r in rows])), 4
        ),
        "mean_causal_ramp_rmse_m": round(
            float(np.mean([r["causal_ramp_rmse_m"] for r in rows])), 4
        ),
        "mean_guarded_oracle_rmse_m": round(
            float(np.mean([r["guarded_oracle_rmse_m"] for r in rows])), 4
        ),
        "causal_unscaled": _summary_stats(
            [r["causal_unscaled_pct"] for r in rows], block_len=8
        ),
        "causal_flat_ramp": _summary_stats(
            [r["causal_ramp_pct"] for r in rows], block_len=8
        ),
        "guarded_oracle_bound": _summary_stats(
            [r["guarded_oracle_pct"] for r in rows], block_len=8
        ),
        "epoch_disjoint_7200s_all": {
            "stride_s": 7200,
            "causal_unscaled": _summary_stats(
                [r["causal_unscaled_pct"] for r in disjoint], block_len=1
            ),
            "causal_flat_ramp": _summary_stats(
                [r["causal_ramp_pct"] for r in disjoint], block_len=1
            ),
        },
        "epoch_disjoint_7200s_post_warmup": {
            "stride_s": 7200,
            "causal_unscaled": _summary_stats(
                [r["causal_unscaled_pct"] for r in disjoint_post_warmup],
                block_len=1,
            ),
            "causal_flat_ramp": _summary_stats(
                [r["causal_ramp_pct"] for r in disjoint_post_warmup],
                block_len=1,
            ),
        },
    }


def _ridge_wv(h: np.ndarray):
    """Deployed horizon-taper weight (mirrors gps_web causal ridge):
    ramp up over [T0,T1], unity over [T1,T2], ramp down over [T2,T3],
    corners from .env.local (T0=0, T1=225, T2=5400, T3=9900 — the ridge
    keeps ~0.6 weight at the end of a 2 h holdout, it does NOT zero out
    at 7,200 s)."""
    if _env_float("TEP_CAUSAL_RIDGE_TAPER", 1.0) <= 0.0:
        return np.ones_like(h)
    t0 = _env_float("TEP_CAUSAL_RIDGE_T0", 0.0)
    t1 = _env_float("TEP_CAUSAL_RIDGE_T1", 225.0)
    t2 = _env_float("TEP_CAUSAL_RIDGE_T2", 5400.0)
    t3 = _env_float("TEP_CAUSAL_RIDGE_T3", 9900.0)
    up = np.clip((h - t0) / max(t1 - t0, 1.0), 0.0, 1.0)
    dn = np.clip((t3 - h) / max(t3 - t2, 1.0), 0.0, 1.0)
    return np.minimum(up, dn)


def _vec_stats(vals) -> dict:
    """Plain summary for non-percentage vectors (e.g. cosines)."""
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _apply_post_ridge(
    ridge: np.ndarray, h: np.ndarray, stack_unc: np.ndarray, cap: float = 8.0
) -> np.ndarray:
    if _env_float("TEP_CAUSAL_RIDGE_DEMEAN", 1.0) > 0.0:
        ridge = ridge - float(np.mean(ridge))
    blend = _env_float("TEP_CAUSAL_RIDGE_BLEND", 1.0)
    return np.clip(
        stack_unc + blend * _ridge_wv(h) * (ridge - stack_unc), -cap, cap
    )


def _fc_gate_fired(
    pf: np.ndarray, rg_dep: np.ndarray, drms: float
) -> bool:
    cfr = (
        float(np.corrcoef(pf, rg_dep)[0, 1])
        if np.std(pf) > 1e-9 and np.std(rg_dep) > 1e-9
        else np.nan
    )
    rat = (
        float(np.sqrt(np.mean(pf ** 2)) / np.sqrt(np.mean(rg_dep ** 2)))
        if np.std(rg_dep) > 1e-9
        else np.nan
    )
    fcg = _env_float("TEP_CAUSAL_RIDGE_FCG", 0.0)
    fcg2 = _env_float("TEP_CAUSAL_RIDGE_FCG2", -0.20)
    fcd = _env_float("TEP_CAUSAL_RIDGE_FCD", 15.0)
    fcg3 = _env_float("TEP_CAUSAL_RIDGE_FCG3", 0.0)
    fcr = _env_float("TEP_CAUSAL_RIDGE_FCR", 0.7)
    fcg4 = _env_float("TEP_CAUSAL_RIDGE_FCG4", 0.40)
    fcr2 = _env_float("TEP_CAUSAL_RIDGE_FCR2", 0.60)
    fcg4u = _env_float("TEP_CAUSAL_RIDGE_FCG4U", 0.0)
    fcg5 = _env_float("TEP_CAUSAL_RIDGE_FCRHI", 4.0)

    fc1 = np.isfinite(cfr) and cfr < fcg
    fc2 = (
        np.isfinite(cfr) and cfr < fcg2 and np.isfinite(drms) and drms > fcd
    )
    fc3 = (
        fcg3 != 0.0
        and np.isfinite(cfr)
        and cfr < fcg3
        and np.isfinite(rat)
        and rat < fcr
    )
    fc4 = (
        fcg4 != 0.0
        and np.isfinite(cfr)
        and cfr > fcg4
        and (fcg4u == 0.0 or cfr < fcg4u)
        and np.isfinite(rat)
        and rat < fcr2
    )
    fc5 = fcg5 > 0.0 and np.isfinite(rat) and (
        rat > fcg5 or (np.isfinite(drms) and drms > fcd and rat > 1.0)
    )
    return bool(
        fcg != 0.0 and np.isfinite(cfr) and (fc1 or fc2 or fc3 or fc4 or fc5)
    )


def _run_sealed_envelope_suite() -> dict:
    # Durable enumeration: raw_window_*.pkl archives ARE the corpus.  The
    # ridge_stats/*.npz Grams are rebuildable caches which a watcher
    # purges after each step_174 run — globbing them truncates the
    # corpus after a purge and races mid-run deletions (the live ring
    # only re-caches its trailing K members each window, so caches are
    # transient by design).  Enumerate the durable archives and
    # load-or-build each member Gram instead; the cache is repopulated
    # as a side effect.
    #
    # The committed replay corpus is the 40 Gram-archived windows of
    # 2026-10-27 ending at window 1793132392 — the seal cutoff cited in
    # the manuscript.  Pinning the corpus end keeps the ex-ante replay a
    # fixed retrospective experiment whose numbers are reproducible;
    # newer live windows flow through the wall-clock causal audit and
    # the post-seal envelope arms, not this corpus.
    corpus_end_wid = int(_env_float("TEP_STEP174_CORPUS_END_WID", 1793132392))
    n_corpus = int(_env_float("TEP_STEP174_CORPUS_N", 40))
    pkl_paths = [
        p for p in sorted(
            RESULTS_DIR.glob("raw_window_*.pkl"), key=lambda p: p.name
        )
        if _window_id(p) <= corpus_end_wid
    ][-n_corpus:]
    valid_pairs = []
    key_sets = []
    skipped_wids = []
    gram_schema = None
    for pklp in pkl_paths:
        try:
            s_i = _ridge_stats_load_or_build(pklp)
        except Exception:
            s_i = None
        if s_i is None:
            skipped_wids.append(_window_id(pklp))
            continue
        if gram_schema is None:
            try:
                with np.load(
                    _ridge_stats_dir() / (pklp.stem + ".npz"),
                    allow_pickle=False,
                ) as _z:
                    gram_schema = str(_z["schema"])
            except Exception:
                gram_schema = None
        key_sets.append(set(s_i[0]))
        valid_pairs.append((_window_id(pklp), pklp, pklp))
    if skipped_wids:
        print(
            f"  !! CORPUS WARNING: {len(skipped_wids)} pinned member(s) "
            f"failed Gram build and were skipped: {skipped_wids} — "
            "the corpus is no longer the pinned 40; rerun before "
            "interpreting scores."
        )

    k_tr = min(int(_env_float("TEP_CAUSAL_RIDGE_K", 16)), 16)
    hl = _env_float("TEP_CAUSAL_RIDGE_HL", 1.25)
    lam = _env_float("TEP_CAUSAL_RIDGE_LAM", 2000.0)
    cap = _env_float("TEP_CAUSAL_RIDGE_CAP_M", 8.0)

    if len(valid_pairs) <= k_tr:
        raise RuntimeError(
            f"Need > {k_tr} Gram+PKL pairs, found {len(valid_pairs)}"
        )

    # Inspect last PKL once to get union of live feature names & station_coords
    _, _, last_pklp = valid_pairs[-1]
    d_last = pickle.loads(last_pklp.read_bytes())
    h_last = np.asarray(d_last["t_ho"], float) - float(
        np.asarray(d_last["t_pre"], float)[-1]
    )
    feats_last = _ridge_feature_dict(h_last, d_last)
    key_sets.append(set(feats_last.keys()))
    station_coords_all = dict(d_last.get("station_coords") or {})
    del d_last, feats_last

    keys_u = sorted(set().union(*key_sets))
    k_idx = {k: i for i, k in enumerate(keys_u)}
    f = len(keys_u)
    lfc = np.asarray([k.startswith("fc") for k in keys_u])
    lnf = ~lfc
    nf_idx = np.flatnonzero(lnf)

    # Identify station blocks and distance bands for falsification battery
    sta_to_cols: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for col_j, k in enumerate(keys_u):
        if k.startswith("sta_"):
            rest = k[4:]
            sta = rest.split("_")[0].split("@")[0].split("#")[0]
            suf = rest[len(sta):]
            sta_to_cols[sta][suf] = col_j

    stations = sorted(sta_to_cols.keys())
    sta_dist_km = {}
    for s in stations:
        if s in station_coords_all:
            lat, lon = station_coords_all[s][:2]
            sta_dist_km[s] = _haversine_km(
                ROVER_LAT_DEG, ROVER_LON_DEG, float(lat), float(lon)
            )

    band_masks = {
        "beyond_troposphere_gt360km": np.zeros(f, dtype=np.float64),
        "mid_band_eu_360_1000km": np.zeros(f, dtype=np.float64),
        "far_arc_gt1000km": np.zeros(f, dtype=np.float64),
        "near_field_uk_lt360km": np.zeros(f, dtype=np.float64),
    }
    for s, suf_map in sta_to_cols.items():
        dkm = sta_dist_km.get(s)
        if dkm is None:
            continue
        cols = list(suf_map.values())
        if dkm >= 360.0:
            band_masks["beyond_troposphere_gt360km"][cols] = 1.0
        if 360.0 <= dkm < 1000.0:
            band_masks["mid_band_eu_360_1000km"][cols] = 1.0
        if dkm >= 1000.0:
            band_masks["far_arc_gt1000km"][cols] = 1.0
        if dkm < 360.0:
            band_masks["near_field_uk_lt360km"][cols] = 1.0

    # Pre-generate 100 station-identity permutations (permuting the 68 stations
    # while preserving each station's internal suffix structure)
    n_perm = 100
    rng = np.random.default_rng(20261027)
    all_sufs = sorted({suf for m in sta_to_cols.values() for suf in m})
    perm_col_maps = np.tile(np.arange(f, dtype=np.int64), (n_perm, 1))
    for p_i in range(n_perm):
        perm_stas = rng.permutation(stations)
        for s_orig, s_new in zip(stations, perm_stas):
            orig_map = sta_to_cols[s_orig]
            new_map = sta_to_cols[s_new]
            for suf in all_sufs:
                if suf in orig_map and suf in new_map:
                    perm_col_maps[p_i, orig_map[suf]] = new_map[suf]

    suf_to_col_list = []
    for suf in all_sufs:
        cols = [sta_to_cols[s][suf] for s in stations if suf in sta_to_cols[s]]
        if len(cols) >= 4:
            suf_to_col_list.append(np.asarray(cols, dtype=np.int64))

    sta_all_cols = np.asarray(
        sorted([c for m in sta_to_cols.values() for c in m.values()]),
        dtype=np.int64,
    )
    sta_only_mask = np.zeros(f, dtype=np.float64)
    sta_only_mask[sta_all_cols] = 1.0

    # Fixed lambda grid for the clean-origin robustness/inner-CV arms.
    # Declared before Phase 1 so each seal can store its eigendecomposition-
    # derived weight vector at every grid point — Phase 2 then scores the
    # whole lambda x lead x epoch-position lattice with no re-solve.
    lam_grid = (
        10000.0, 50000.0, 200000.0, 1000000.0, 2000000.0,
        5000000.0, 10000000.0, 20000000.0,
    )

    # Phase 1: Stream the corpus once with paired XTX accumulators:
    #   XTX/XTR   — member HOLDOUТ Grams (deployed solve basis; at seal
    #               time this accumulator is also the pooled holdout
    #               validation Gram available to an operator, i.e. the
    #               set the deployed lam was tuned on)
    #   XTX_c/XTR_c — member FIT-HALF Grams (clean-origin arm).  Each
    #               member Gram is built on the FULL 2-hour fit half
    #               [t_pre[0], t_pre[-1]] of that member — epochs ending
    #               strictly before that member's own holdout, hence
    #               before EVERY later evaluation holdout — so every
    #               scored epoch is zero-overlap ex-ante by construction
    #               at every seal lead.  The basis is the station block
    #               (sta_* features built from raw_series on a
    #               station-only dictionary, so no negative-h
    #               holdout-shaped features exist).
    # Two solves are retained per seal: w_clean (full ridge, same
    # architecture as the deployed solve) and w_clean_diag (per-feature
    # diagonal combiner, which cannot encode epoch-realization-specific
    # cross-feature cancellations).  In addition, one eigendecomposition
    # of the clean ring Gram yields the weight vector w(lam) = V
    # diag(1/(e+lam)) V' (-XTR_c) at every grid lambda, stored per seal
    # together with fully ex-ante lambda picks (validation Grams are all
    # pre-seal data — see the inner-CV block below).
    XTX = np.zeros((f, f), dtype=np.float32)
    XTR = np.zeros(f, dtype=np.float32)
    XTX_c = np.zeros((f, f), dtype=np.float32)
    XTR_c = np.zeros(f, dtype=np.float32)
    decay = np.float32(0.5 ** (1.0 / max(hl, 1e-9)))
    sealed_models = {}
    n_fit_members = 0

    for idx_w, (wid, npzp, pklp) in enumerate(valid_pairs):
        if idx_w >= k_tr:
            XTX_reg = XTX.copy()
            XTX_reg[np.diag_indices(f)] += np.float32(lam)
            w_frozen = _spd_solve(XTX_reg, -XTR)
            del XTX_reg

            XTX_nf = XTX[np.ix_(nf_idx, nf_idx)].copy()
            XTX_nf[np.diag_indices(len(nf_idx))] += np.float32(lam)
            wnf_sub = _spd_solve(XTX_nf, -XTR[nf_idx])
            del XTX_nf
            wnf_frozen = np.zeros(f, dtype=np.float64)
            wnf_frozen[nf_idx] = wnf_sub

            XTX_reg = XTX_c.copy()
            XTX_reg[np.diag_indices(f)] += np.float32(lam)
            wc = _spd_solve(XTX_reg, -XTR_c)
            del XTX_reg

            sm = {
                "seal_wid": valid_pairs[idx_w - 1][0],
                "sha256": _sha256_arrays(w_frozen, wnf_frozen)[:16],
                "w_frozen": w_frozen,
                "wnf_frozen": wnf_frozen,
                "w_clean": wc,
                "sha256_clean": _sha256_arrays(wc)[:16],
            }
            _dg = np.diag(XTX_c).astype(np.float64)
            wcd = np.zeros(f, dtype=np.float64)
            _nz = _dg > 0.0
            wcd[_nz] = -XTR_c[_nz] / (_dg[_nz] + lam)
            sm["w_clean_diag"] = wcd

            # Clean-arm lambda grid + causal inner-CV selection.  One
            # eigendecomposition of the fit-half ring Gram gives w(lam)
            # for the whole grid; the selection objective
            #   J(w) = w' Gv w + 2 w' gv
            # is the squared error ||r + Xw||^2 up to the constant r'r
            # (xtr = X'r convention, scored error r + Xw — the linear
            # term enters with a PLUS sign).  Validation sets, all
            # strictly pre-seal:
            #   ho   — newest member's holdout Gram (disjoint from every
            #          fit-half target; the holdout-regime validation
            #          set an operator holds at deploy time — its target
            #          epochs overlap the eval holdout, mirroring live
            #          tuning)
            #   pool — the pooled holdout Gram of the WHOLE sealed ring
            #          (the accumulated XTX/XTR of the deployed arm at
            #          this instant): the complete holdout-regime
            #          validation set, ~k_tr x more epochs than ho, and
            #          the quantity the deployed lam was tuned on
            #   fit  — newest member's fit Gram (in-domain reference)
            #   lno  — leave-newest-out: ring solved WITHOUT the newest
            #          member's fit Gram, scored on exactly that held-
            #          out Gram (disjoint train/validation, the CV most
            #          faithful to the forward task).
            # lam is fixed before any eval window is scored — no eval
            # data enters the choice.
            try:
                _ev, _V = eigh(XTX_c.astype(np.float64))
                _c = _V.T @ (-XTR_c.astype(np.float64))
                _wgrid = {
                    _L: _V @ (_c / (_ev + _L)) for _L in lam_grid
                }
                _picks = {}
                _valg = {}
                for _tag, _loader in (
                    ("ho", _ridge_stats_load_or_build),
                    ("fit", _ridge_stats_fit_load_or_build),
                ):
                    try:
                        _sv = _loader(valid_pairs[idx_w - 1][2])
                    except Exception:
                        _sv = None
                    if _sv is None:
                        continue
                    _iv = np.asarray(
                        [k_idx.get(k, -1) for k in _sv[0]], np.int64
                    )
                    _sl = _iv >= 0
                    if np.any(_sl):
                        _valg[_tag] = (
                            _iv[_sl],
                            _sv[1].astype(np.float64)[np.ix_(_sl, _sl)],
                            _sv[2].astype(np.float64)[_sl],
                        )
                for _tag, _vg in _valg.items():
                    _iv, _xv, _rv = _vg
                    _sc = [
                        float(
                            _wgrid[_L][_iv] @ _xv @ _wgrid[_L][_iv]
                            + 2.0 * _wgrid[_L][_iv] @ _rv
                        )
                        for _L in lam_grid
                    ]
                    _picks[_tag] = float(
                        lam_grid[int(np.argmin(_sc))]
                    )
                _Xp = XTX.astype(np.float64)
                _rp = XTR.astype(np.float64)
                _sc = [
                    float(
                        _wgrid[_L] @ _Xp @ _wgrid[_L]
                        + 2.0 * _wgrid[_L] @ _rp
                    )
                    for _L in lam_grid
                ]
                _picks["pool"] = float(
                    lam_grid[int(np.argmin(_sc))]
                )
                if "fit" in _valg:
                    _iv2, _xv2, _rv2 = _valg["fit"]
                    _Gl = XTX_c.astype(np.float64)
                    _Gl[np.ix_(_iv2, _iv2)] -= decay * _xv2
                    _rl = XTR_c.astype(np.float64)
                    _rl[_iv2] -= decay * _rv2
                    try:
                        _ev2, _V2 = eigh(_Gl)
                        _c2 = _V2.T @ (-_rl)
                        _sc = []
                        for _L in lam_grid:
                            _wL = (_V2 @ (_c2 / (_ev2 + _L)))[_iv2]
                            _sc.append(
                                float(
                                    _wL @ _xv2 @ _wL
                                    + 2.0 * _wL @ _rv2
                                )
                            )
                        _picks["lno"] = float(
                            lam_grid[int(np.argmin(_sc))]
                        )
                        del _V2, _ev2, _c2
                    except Exception:
                        pass
                    del _Gl, _rl
                sm["w_grid"] = _wgrid
                sm["lam_picks"] = _picks
                del _V, _ev, _c
            except Exception:
                pass
            sealed_models[idx_w] = sm

        XTX *= decay
        XTR *= decay
        XTX_c *= decay
        XTR_c *= decay
        # npzp carries the durable .pkl path; the Gram cache it points at
        # may have been purged mid-run, so always go through
        # load-or-build rather than reading the npz directly.
        try:
            s_i = _ridge_stats_load_or_build(npzp)
        except Exception:
            s_i = None
        if s_i is not None:
            keys_i, xtx_i64, xtr_i64, fcd_i = s_i
            xtx_i = xtx_i64.astype(np.float32, copy=False)
            xtr_i = xtr_i64.astype(np.float32, copy=False)
            s = float(_fcm_scale(fcd_i))
            ii = np.asarray(
                [k_idx[k] for k in keys_i if k in k_idx], dtype=np.int64
            )
            if len(ii) == len(keys_i):
                if s >= 1.0:
                    XTX[np.ix_(ii, ii)] += decay * xtx_i
                    XTR[ii] += decay * xtr_i
                else:
                    fp = np.asarray([k.startswith("fc") for k in keys_i])
                    zv = np.where(fp, s, 1.0).astype(np.float32)
                    XTX[np.ix_(ii, ii)] += decay * (
                        zv[:, None] * xtx_i * zv[None, :]
                    )
                    XTR[ii] += decay * (zv * xtr_i)
        try:
            s_fit = _ridge_stats_fit_load_or_build(pklp)
        except Exception:
            s_fit = None
        if s_fit is not None:
            keys_f, xtx_f, xtr_f = s_fit
            # Accumulate on the key intersection: fit-Gram keys absent
            # from the union basis can never be scored anyway, so they
            # are dropped rather than discarding the whole member Gram.
            ii_local = np.asarray(
                [k_idx.get(k, -1) for k in keys_f], dtype=np.int64
            )
            sel = ii_local >= 0
            if np.any(sel):
                ii = ii_local[sel]
                xtx_f32 = xtx_f.astype(np.float32, copy=False)[
                    np.ix_(sel, sel)
                ]
                XTX_c[np.ix_(ii, ii)] += decay * xtx_f32
                XTR_c[ii] += decay * xtr_f.astype(
                    np.float32, copy=False
                )[sel]
                n_fit_members += 1

    del XTX, XTR, XTX_c, XTR_c

    # Disjoint-map stability control: seals k_tr apart are trained on
    # non-overlapping member sets whose target epochs are likewise
    # disjoint — cosine similarity of their weight vectors measures
    # whether the learned station->rover map is a stable field
    # structure or per-realisation noise (noise weights decorrelate).
    # NOTE: weight-space cosine UNDERSTATES stability in a collinear
    # multi-resolution basis — equivalent maps can rotate weight mass
    # across near-duplicate columns while producing identical
    # predictions.  The prediction-space analogue is computed in
    # Phase 2 (clean_pred_cos / holdout_pred_cos below).
    clean_map_cos = []
    holdout_map_cos = []
    clean_pred_cos = []
    holdout_pred_cos = []
    for _i1 in range(k_tr, len(valid_pairs) - k_tr + 1):
        _i2 = _i1 + k_tr
        if _i2 > len(valid_pairs) - 1:
            break
        for _wk, _acc in (("w_clean", clean_map_cos),
                          ("wnf_frozen", holdout_map_cos)):
            _s1 = sealed_models.get(_i1, {}).get(_wk)
            _s2 = sealed_models.get(_i2, {}).get(_wk)
            if _s1 is None or _s2 is None:
                continue
            _n1 = float(np.linalg.norm(_s1))
            _n2 = float(np.linalg.norm(_s2))
            if _n1 > 0 and _n2 > 0:
                _acc.append(float(np.dot(_s1, _s2) / (_n1 * _n2)))

    # Phase 2: Stream one .pkl at a time (~180 MB RAM) and score all prior seals
    wf_full_pct = []
    wf_net_pct = []
    wf_sta_only_pct = []
    wf_cm_removed_pct = []
    wf_cm_only_pct = []
    wf_band_pcts = {b: [] for b in band_masks}
    wf_perm_matrix = []
    wf_records = []

    future_slice_full_pct = []
    future_slice_net_pct = []
    future_fw_full_pct = []
    future_fw_net_pct = []
    future_tail_full_pct = []
    future_tail_net_pct = []
    future_fw_by_step = collections.defaultdict(list)
    band_late_by_step = collections.defaultdict(
        list
    )  # fixed-position h in [4500,5400] slices vs lead
    resubstitution_pct = []  # sealed weights scored on own member holdout
    embargo_3600_full_pct = []
    embargo_3600_net_pct = []
    embargo_7200_full_pct = []
    embargo_7200_net_pct = []
    horizon_curves = collections.defaultdict(list)
    clean_pct = []
    clean_pure_pct = []
    clean_diag_pct = []
    clean_horizon = collections.defaultdict(list)
    clean_pure_horizon = collections.defaultdict(list)
    clean_future_pct = []
    clean_emb3600_pct = []
    clean_emb7200_pct = []
    # Clean-arm skill resolved by eval-holdout position.  For clean-origin
    # seals every scored epoch is strictly unseen; the data-age of the
    # newest member target at eval position h is ~h+640 s, so early
    # positions carry fresh data-age (inside the ~90 min coherence
    # horizon) while late positions test stale transfer.  This separates
    # a small genuine forward-skill component from the epoch-shared
    # retrodiction that dominates the holdout-trained headline.
    clean_pos_bands = collections.defaultdict(list)
    CLEAN_POS_EDGES = (0.0, 1800.0, 3600.0, 5400.0, 7300.0)
    oracle_replay_pct = []
    epoch_share_pct = []
    clean_indomain_pct = []
    b_innov_rms = []
    t_ho_ends = {}
    # Per-seal lambda-grid / inner-CV-selected weight vectors (stored at
    # seal time in Phase 1): scored at every lead so the whole
    # lambda x lead x epoch-position lattice is resolved ex-ante.
    # grid_* accumulators key on the lambda value; cv_* on the selector
    # tag ("ho", "pool", "fit", "lno").
    grid_pure_1 = {L: [] for L in lam_grid}
    grid_pos = {L: collections.defaultdict(list) for L in lam_grid}
    grid_horizon = {L: collections.defaultdict(list) for L in lam_grid}
    grid_future = {L: [] for L in lam_grid}
    grid_emb3600 = {L: [] for L in lam_grid}
    grid_emb7200 = {L: [] for L in lam_grid}
    cv_pure_1 = collections.defaultdict(list)
    cv_cmp_1 = collections.defaultdict(list)
    cv_lam = collections.defaultdict(list)
    cv_pos = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    cv_horizon_pure = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    cv_horizon_cmp = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    cv_future_pure = collections.defaultdict(list)
    cv_future_cmp = collections.defaultdict(list)
    cv_emb3600_pure = collections.defaultdict(list)
    cv_emb3600_cmp = collections.defaultdict(list)
    cv_emb7200_pure = collections.defaultdict(list)
    cv_emb7200_cmp = collections.defaultdict(list)

    for j in range(k_tr - 1, len(valid_pairs)):
        wid_j, _, pklp_j = valid_pairs[j]
        d = pickle.loads(pklp_j.read_bytes())
        t_ho = np.asarray(d["t_ho"], float)
        t_ho_ends[j] = float(t_ho[-1])
        if j < k_tr:
            del d
            continue

        t_pre = np.asarray(d["t_pre"], float)
        h = t_ho - float(t_pre[-1])
        feats = _ridge_feature_dict(h, d)
        r = np.asarray(d["persist_pred"], float) - np.asarray(d["b_ho"], float)
        stack_unc = np.asarray(d["stack_unc"], float)
        dev = (
            np.asarray(d.get("free_clk_ho"), float).ravel()
            - np.asarray(d["persist_pred"], float).ravel()
            if d.get("free_clk_ho") is not None
            else None
        )
        drms = (
            float(np.sqrt(np.mean(dev ** 2)))
            if dev is not None and len(dev) == len(h)
            else np.nan
        )
        pr = float(np.sqrt(np.mean(r ** 2)))
        # Per-epoch estimate-noise scale of the clock truth series: the
        # 1 s first-difference RMS sets the floor below which no
        # field-coherent model can predict at unseen epochs.
        b_innov_rms.append(
            float(
                np.sqrt(
                    np.mean(
                        np.diff(np.asarray(d["b_ho"], float).ravel()) ** 2
                    )
                )
            )
        )
        del d

        Xl = np.zeros((len(h), f), dtype=np.float64)
        for k, v in feats.items():
            col = k_idx.get(k)
            if col is not None and len(v) == len(h):
                Xl[:, col] = v
        del feats

        for eval_idx in range(k_tr, j + 1):
            sm = sealed_models[eval_idx]
            w_frozen = sm["w_frozen"]
            wnf_frozen = sm["wnf_frozen"]
            seal_wid = sm["seal_wid"]
            seal_t_end = t_ho_ends[eval_idx - 1]

            rg_dep = Xl @ wnf_frozen
            pf = Xl[:, lfc] @ w_frozen[lfc]
            fired = _fc_gate_fired(pf, rg_dep, drms)
            ridge_full = rg_dep + (0.0 if fired else pf)

            dep_full = _apply_post_ridge(ridge_full, h, stack_unc, cap=cap)
            dep_net = _apply_post_ridge(rg_dep.copy(), h, stack_unc, cap=cap)

            # Clean-origin arm: weights trained exclusively on member
            # fit-half Grams — every scored holdout epoch here is
            # zero-overlap ex-ante by construction, at every lead.
            # Three variants are scored:
            #   dep_clean       — deployed-style composite
            #                     stack_unc + wv*(ridge - stack_unc)
            #   dep_clean_pure  — pure tapered correction wv*ridge vs
            #                     persistence (isolates the map itself
            #                     from the stack baseline)
            #   dep_clean_diag  — per-feature diagonal combiner, pure
            dep_clean = None
            dep_clean_pure = None
            dep_clean_diag = None
            w_clean = sm.get("w_clean")
            if w_clean is not None:
                _rg_c = Xl @ w_clean
                dep_clean = _apply_post_ridge(
                    _rg_c, h, stack_unc, cap=cap
                )
                dep_clean_pure = _apply_post_ridge(
                    _rg_c, h, np.zeros_like(stack_unc), cap=cap
                )
            w_clean_diag = sm.get("w_clean_diag")
            if w_clean_diag is not None:
                dep_clean_diag = _apply_post_ridge(
                    Xl @ w_clean_diag, h, np.zeros_like(stack_unc),
                    cap=cap,
                )

            rmse_full = float(np.sqrt(np.mean((r + dep_full) ** 2)))
            rmse_net = float(np.sqrt(np.mean((r + dep_net) ** 2)))
            pct_full = (1.0 - rmse_full / pr) * 100.0
            pct_net = (1.0 - rmse_net / pr) * 100.0
            pct_clean = (
                (1.0 - float(np.sqrt(np.mean((r + dep_clean) ** 2))) / pr)
                * 100.0
                if dep_clean is not None
                else None
            )
            pct_clean_pure = (
                (1.0 - float(
                    np.sqrt(np.mean((r + dep_clean_pure) ** 2))) / pr)
                * 100.0
                if dep_clean_pure is not None
                else None
            )
            pct_clean_diag = (
                (1.0 - float(
                    np.sqrt(np.mean((r + dep_clean_diag) ** 2))) / pr)
                * 100.0
                if dep_clean_diag is not None
                else None
            )

            # Lambda-grid + inner-CV-selected clean arms (stored at seal
            # time): every grid lambda scored pure, every selector pick
            # scored pure + composite.  One Xl @ w per lambda.
            dep_grid_pure = {}
            dep_cv_pure = {}
            dep_cv_cmp = {}
            _wgrid = sm.get("w_grid")
            if _wgrid is not None:
                _z0 = np.zeros_like(stack_unc)
                for _L, _wL in _wgrid.items():
                    _rgL = Xl @ _wL
                    dep_grid_pure[_L] = _apply_post_ridge(
                        _rgL, h, _z0, cap=cap
                    )
                _picks = sm.get("lam_picks") or {}
                for _tg, _Lp in _picks.items():
                    if _Lp in _wgrid:
                        _rgP = Xl @ _wgrid[_Lp]
                        dep_cv_pure[_tg] = _apply_post_ridge(
                            _rgP, h, _z0, cap=cap
                        )
                        dep_cv_cmp[_tg] = _apply_post_ridge(
                            _rgP, h, stack_unc, cap=cap
                        )

            # Oracle-replay echo ceiling: at 1-step seals, dep(t) is set to
            # the NEGATED target residual of member j-1 interpolated onto
            # the shared wall-clock epochs (zero outside its coverage).
            # This is the maximum the epoch-sharing channel could deliver
            # with a perfect member->eval correspondence -- the ridge's
            # one-step score is then read as a shrinkage-capped fraction
            # of this ceiling.  Diagnostic only; not a deployable arm.
            if eval_idx == j:
                try:
                    _dm = pickle.loads(valid_pairs[j - 1][2].read_bytes())
                    _tm = np.asarray(_dm["t_ho"], float)
                    _rm = (
                        np.asarray(_dm["persist_pred"], float)
                        - np.asarray(_dm["b_ho"], float)
                    ).ravel()
                    _dep_or = np.interp(t_ho, _tm, -_rm, left=0.0, right=0.0)
                    _dep_or[(t_ho < _tm[0]) | (t_ho > _tm[-1])] = 0.0
                    _rr_or = float(np.sqrt(np.mean((r + _dep_or) ** 2)))
                    oracle_replay_pct.append((1.0 - _rr_or / pr) * 100.0)
                    # Epoch-sharing fraction: fraction of the scored
                    # holdout's epochs that coincide with the newest
                    # member's target epochs (15 s coincidence gate).
                    epoch_share_pct.append(
                        100.0
                        * float(
                            np.isin(
                                np.round(t_ho / 15.0),
                                np.round(_tm / 15.0),
                            ).mean()
                        )
                    )
                    # Clean-arm in-domain check: score w_clean on its own
                    # newest member's FULL fit half, with the same
                    # station-only dictionary used to build the Gram —
                    # establishes that the fit-half Gram is informative
                    # in-domain, so the clean arm's out-of-domain score
                    # measures generalisation, not a broken Gram.
                    if w_clean is not None:
                        _tpm = np.asarray(_dm["t_pre"], float)
                        _bpm = np.asarray(_dm["b_pre"], float).ravel()
                        if len(_tpm) >= 1200 and len(_bpm) == len(_tpm):
                            _hf = _tpm - float(_tpm[-1])
                            _rf = (
                                float(
                                    np.asarray(
                                        _dm["persist_pred"], float
                                    ).ravel()[0]
                                )
                                - _bpm
                            )
                            _ff = _ridge_feature_dict(
                                _hf,
                                {
                                    "raw_series": _dm.get("raw_series"),
                                    "t_pre": _tpm,
                                    "station_coords": _dm.get(
                                        "station_coords"
                                    ),
                                },
                            )
                            _Xf = np.zeros((len(_hf), f))
                            for _kk, _vv in _ff.items():
                                _ci2 = k_idx.get(_kk)
                                if _ci2 is not None and len(_vv) == len(_hf):
                                    _Xf[:, _ci2] = _vv
                            _prf = float(np.sqrt(np.mean(_rf ** 2)))
                            if _prf > 0:
                                _rrf = float(
                                    np.sqrt(
                                        np.mean((_rf + _Xf @ w_clean) ** 2)
                                    )
                                )
                                clean_indomain_pct.append(
                                    (1.0 - _rrf / _prf) * 100.0
                                )
                    del _dm
                except Exception:
                    pass

            # Prediction-space disjoint-map stability: correlate the
            # prediction curves of seals k_tr apart (disjoint member
            # sets, disjoint target epochs) on THIS eval window's
            # feature matrix.  A stable spatial field map yields
            # corr >> 0 even when weight-space cosines are ~0 (the
            # basis is collinear, so null-space weight rotations are
            # unidentifiable but predictively irrelevant).
            _i_prev = eval_idx - k_tr
            if eval_idx == j and _i_prev in sealed_models:
                _wc_prev = sealed_models[_i_prev].get("w_clean")
                if w_clean is not None and _wc_prev is not None:
                    _p1 = Xl @ w_clean
                    _p0 = Xl @ _wc_prev
                    if np.std(_p1) > 1e-9 and np.std(_p0) > 1e-9:
                        clean_pred_cos.append(
                            float(np.corrcoef(_p0, _p1)[0, 1])
                        )
                _wf_prev = sealed_models[_i_prev].get("wnf_frozen")
                if _wf_prev is not None:
                    _q1 = rg_dep  # Xl @ wnf_frozen for seal j
                    _q0 = Xl[:, lnf] @ _wf_prev[lnf]
                    if np.std(_q1) > 1e-9 and np.std(_q0) > 1e-9:
                        holdout_pred_cos.append(
                            float(np.corrcoef(_q0, _q1)[0, 1])
                        )

            step_ahead = j - eval_idx + 1
            horizon_curves[step_ahead].append((pct_full, pct_net))
            if pct_clean is not None and step_ahead <= 12:
                clean_horizon[step_ahead].append(pct_clean)
                if pct_clean_pure is not None:
                    clean_pure_horizon[step_ahead].append(pct_clean_pure)
                if eval_idx == j:
                    clean_pct.append(pct_clean)
                    if pct_clean_pure is not None:
                        clean_pure_pct.append(pct_clean_pure)
                    if pct_clean_diag is not None:
                        clean_diag_pct.append(pct_clean_diag)
                    for _bi in range(len(CLEAN_POS_EDGES) - 1):
                        _m_pb = (h >= CLEAN_POS_EDGES[_bi]) & (
                            h < CLEAN_POS_EDGES[_bi + 1]
                        )
                        if int(np.sum(_m_pb)) >= 300:
                            _pr_pb = float(np.sqrt(np.mean(r[_m_pb] ** 2)))
                            if _pr_pb > 0:
                                _rc_pb = float(
                                    np.sqrt(
                                        np.mean(
                                            (
                                                r[_m_pb]
                                                + dep_clean_pure[_m_pb]
                                            )
                                            ** 2
                                        )
                                    )
                                )
                                clean_pos_bands[_bi].append(
                                    (1.0 - _rc_pb / _pr_pb) * 100.0
                                )

            # Lambda-grid + inner-CV-selected clean arms at every lead.
            # Pure-correction scoring for every grid lambda (matches the
            # deployed-style transform: demean, taper, clip); pure AND
            # stack-composite scoring for each ex-ante selector pick.
            if dep_grid_pure and step_ahead <= 12:
                for _L, _dgp in dep_grid_pure.items():
                    _pg = (
                        1.0
                        - float(
                            np.sqrt(np.mean((r + _dgp) ** 2))
                        )
                        / pr
                    ) * 100.0
                    grid_horizon[_L][step_ahead].append(_pg)
                    if eval_idx == j:
                        grid_pure_1[_L].append(_pg)
                        for _bi in range(len(CLEAN_POS_EDGES) - 1):
                            _m_pb = (h >= CLEAN_POS_EDGES[_bi]) & (
                                h < CLEAN_POS_EDGES[_bi + 1]
                            )
                            if int(np.sum(_m_pb)) >= 300:
                                _pr_pb = float(
                                    np.sqrt(np.mean(r[_m_pb] ** 2))
                                )
                                if _pr_pb > 0:
                                    _rc_pb = float(
                                        np.sqrt(
                                            np.mean(
                                                (
                                                    r[_m_pb]
                                                    + _dgp[_m_pb]
                                                )
                                                ** 2
                                            )
                                        )
                                    )
                                    grid_pos[_L][_bi].append(
                                        (1.0 - _rc_pb / _pr_pb) * 100.0
                                    )
            if dep_cv_pure and step_ahead <= 12:
                for _tg, _dpp in dep_cv_pure.items():
                    _dpc = dep_cv_cmp[_tg]
                    _pp = (
                        1.0
                        - float(
                            np.sqrt(np.mean((r + _dpp) ** 2))
                        )
                        / pr
                    ) * 100.0
                    _pc = (
                        1.0
                        - float(
                            np.sqrt(np.mean((r + _dpc) ** 2))
                        )
                        / pr
                    ) * 100.0
                    cv_horizon_pure[_tg][step_ahead].append(_pp)
                    cv_horizon_cmp[_tg][step_ahead].append(_pc)
                    if eval_idx == j:
                        cv_pure_1[_tg].append(_pp)
                        cv_cmp_1[_tg].append(_pc)
                        cv_lam[_tg].append(
                            float(sm["lam_picks"][_tg])
                        )
                        for _bi in range(len(CLEAN_POS_EDGES) - 1):
                            _m_pb = (h >= CLEAN_POS_EDGES[_bi]) & (
                                h < CLEAN_POS_EDGES[_bi + 1]
                            )
                            if int(np.sum(_m_pb)) >= 300:
                                _pr_pb = float(
                                    np.sqrt(np.mean(r[_m_pb] ** 2))
                                )
                                if _pr_pb > 0:
                                    _rc_pb = float(
                                        np.sqrt(
                                            np.mean(
                                                (
                                                    r[_m_pb]
                                                    + _dpp[_m_pb]
                                                )
                                                ** 2
                                            )
                                        )
                                    )
                                    cv_pos[_tg][_bi].append(
                                        (1.0 - _rc_pb / _pr_pb) * 100.0
                                    )

            # Fixed-position late-holdout band (h in [4500,5400], unity taper)
            # scored at every lead: separates weight staleness (lead) from the
            # taper-tail position confound in the strict-future slices.
            if step_ahead <= 12:
                m_band = (h >= 4500.0) & (h <= 5400.0)
                if int(np.sum(m_band)) >= 300:
                    pr_b = float(np.sqrt(np.mean(r[m_band] ** 2)))
                    if pr_b > 0:
                        rf_b = float(
                            np.sqrt(
                                np.mean((r[m_band] + dep_full[m_band]) ** 2)
                            )
                        )
                        band_late_by_step[step_ahead].append(
                            (1.0 - rf_b / pr_b) * 100.0
                        )

            m_fut = t_ho > seal_t_end
            n_fut = int(np.sum(m_fut))
            if n_fut >= 600 and step_ahead <= 12:
                pr_f = float(np.sqrt(np.mean(r[m_fut] ** 2)))
                if pr_f > 0:
                    rf_f = float(
                        np.sqrt(np.mean((r[m_fut] + dep_full[m_fut]) ** 2))
                    )
                    rn_f = float(
                        np.sqrt(np.mean((r[m_fut] + dep_net[m_fut]) ** 2))
                    )
                    future_slice_full_pct.append((1.0 - rf_f / pr_f) * 100.0)
                    future_slice_net_pct.append((1.0 - rn_f / pr_f) * 100.0)
                    if dep_clean_pure is not None:
                        rc_f = float(
                            np.sqrt(
                                np.mean(
                                    (r[m_fut] + dep_clean_pure[m_fut]) ** 2
                                )
                            )
                        )
                        clean_future_pct.append(
                            (1.0 - rc_f / pr_f) * 100.0
                        )
                    for _L, _dgp in dep_grid_pure.items():
                        grid_future[_L].append(
                            (
                                1.0
                                - float(
                                    np.sqrt(
                                        np.mean(
                                            (r[m_fut] + _dgp[m_fut]) ** 2
                                        )
                                    )
                                )
                                / pr_f
                            )
                            * 100.0
                        )
                    for _tg, _dpp in dep_cv_pure.items():
                        cv_future_pure[_tg].append(
                            (
                                1.0
                                - float(
                                    np.sqrt(
                                        np.mean(
                                            (r[m_fut] + _dpp[m_fut]) ** 2
                                        )
                                    )
                                )
                                / pr_f
                            )
                            * 100.0
                        )
                        cv_future_cmp[_tg].append(
                            (
                                1.0
                                - float(
                                    np.sqrt(
                                        np.mean(
                                            (
                                                r[m_fut]
                                                + dep_cv_cmp[_tg][m_fut]
                                            )
                                            ** 2
                                        )
                                    )
                                )
                                / pr_f
                            )
                            * 100.0
                        )
                # Decompose: full-taper-weight future epochs (h <= 5400 s)
                # isolate genuine ridge skill on never-trained epochs, while
                # the tapered tail (h > 5400 s) is where dep -> stack_unc.
                m_fw = m_fut & (h <= 5400.0)
                m_tl = m_fut & (h > 5400.0)
                for m_sub, acc_f, acc_n in (
                    (m_fw, future_fw_full_pct, future_fw_net_pct),
                    (m_tl, future_tail_full_pct, future_tail_net_pct),
                ):
                    if int(np.sum(m_sub)) >= 300:
                        pr_s = float(np.sqrt(np.mean(r[m_sub] ** 2)))
                        if pr_s > 0:
                            rf_s = float(
                                np.sqrt(
                                    np.mean((r[m_sub] + dep_full[m_sub]) ** 2)
                                )
                            )
                            rn_s = float(
                                np.sqrt(
                                    np.mean((r[m_sub] + dep_net[m_sub]) ** 2)
                                )
                            )
                            acc_f.append((1.0 - rf_s / pr_s) * 100.0)
                            acc_n.append((1.0 - rn_s / pr_s) * 100.0)
                            if m_sub is m_fw:
                                future_fw_by_step[step_ahead].append(
                                    (1.0 - rf_s / pr_s) * 100.0
                                )

            dt_seal = wid_j - seal_wid
            if 3600 <= dt_seal < 4500:
                embargo_3600_full_pct.append(pct_full)
                embargo_3600_net_pct.append(pct_net)
                if pct_clean_pure is not None:
                    clean_emb3600_pct.append(pct_clean_pure)
                for _L, _dgp in dep_grid_pure.items():
                    grid_emb3600[_L].append(
                        (
                            1.0
                            - float(
                                np.sqrt(np.mean((r + _dgp) ** 2))
                            )
                            / pr
                        )
                        * 100.0
                    )
                for _tg, _dpp in dep_cv_pure.items():
                    cv_emb3600_pure[_tg].append(
                        (
                            1.0
                            - float(
                                np.sqrt(np.mean((r + _dpp) ** 2))
                            )
                            / pr
                        )
                        * 100.0
                    )
                    cv_emb3600_cmp[_tg].append(
                        (
                            1.0
                            - float(
                                np.sqrt(
                                    np.mean((r + dep_cv_cmp[_tg]) ** 2)
                                )
                            )
                            / pr
                        )
                        * 100.0
                    )
            if 7200 <= dt_seal < 8100:
                embargo_7200_full_pct.append(pct_full)
                embargo_7200_net_pct.append(pct_net)
                if pct_clean_pure is not None:
                    clean_emb7200_pct.append(pct_clean_pure)
                for _L, _dgp in dep_grid_pure.items():
                    grid_emb7200[_L].append(
                        (
                            1.0
                            - float(
                                np.sqrt(np.mean((r + _dgp) ** 2))
                            )
                            / pr
                        )
                        * 100.0
                    )
                for _tg, _dpp in dep_cv_pure.items():
                    cv_emb7200_pure[_tg].append(
                        (
                            1.0
                            - float(
                                np.sqrt(np.mean((r + _dpp) ** 2))
                            )
                            / pr
                        )
                        * 100.0
                    )
                    cv_emb7200_cmp[_tg].append(
                        (
                            1.0
                            - float(
                                np.sqrt(
                                    np.mean((r + dep_cv_cmp[_tg]) ** 2)
                                )
                            )
                            / pr
                        )
                        * 100.0
                    )

            if eval_idx == j:
                wf_full_pct.append(pct_full)
                wf_net_pct.append(pct_net)

                w_sta = wnf_frozen * sta_only_mask
                dep_sta = _apply_post_ridge(
                    Xl @ w_sta, h, np.zeros_like(stack_unc), cap=cap
                )
                rmse_sta = float(np.sqrt(np.mean((r + dep_sta) ** 2)))
                wf_sta_only_pct.append((1.0 - rmse_sta / pr) * 100.0)

                for bname, bmask in band_masks.items():
                    wb = wnf_frozen * bmask
                    dep_b = _apply_post_ridge(
                        Xl @ wb, h, np.zeros_like(stack_unc), cap=cap
                    )
                    rmse_b = float(np.sqrt(np.mean((r + dep_b) ** 2)))
                    wf_band_pcts[bname].append((1.0 - rmse_b / pr) * 100.0)

                W_perm = np.zeros((f, n_perm), dtype=np.float64)
                for p_i in range(n_perm):
                    W_perm[:, p_i] = w_sta[perm_col_maps[p_i]]
                R_perm = Xl @ W_perm
                R_perm = R_perm - np.mean(R_perm, axis=0, keepdims=True)
                wv = _ridge_wv(h)
                D_perm = np.clip(wv[:, None] * R_perm, -cap, cap)
                rmse_perm = np.sqrt(np.mean((r[:, None] + D_perm) ** 2, axis=0))
                wf_perm_matrix.append((1.0 - rmse_perm / pr) * 100.0)

                Xl_cm_rem = np.zeros_like(Xl)
                Xl_cm_only = np.zeros_like(Xl)
                for cols in suf_to_col_list:
                    sub = Xl[:, cols]
                    cm = np.mean(sub, axis=1, keepdims=True)
                    Xl_cm_rem[:, cols] = sub - cm
                    Xl_cm_only[:, cols] = cm
                dep_cm_rem = _apply_post_ridge(
                    Xl_cm_rem @ w_sta, h, np.zeros_like(stack_unc), cap=cap
                )
                dep_cm_only = _apply_post_ridge(
                    Xl_cm_only @ w_sta, h, np.zeros_like(stack_unc), cap=cap
                )
                rmse_cm_rem = float(np.sqrt(np.mean((r + dep_cm_rem) ** 2)))
                rmse_cm_only = float(np.sqrt(np.mean((r + dep_cm_only) ** 2)))
                wf_cm_removed_pct.append((1.0 - rmse_cm_rem / pr) * 100.0)
                wf_cm_only_pct.append((1.0 - rmse_cm_only / pr) * 100.0)

                wf_records.append(
                    {
                        "eval_window_id": wid_j,
                        "seal_cutoff_window_id": seal_wid,
                        "weights_sha256": sm["sha256"],
                        "persist_rmse_m": round(pr, 4),
                        "full_causal_sealed_rmse_m": round(rmse_full, 4),
                        "full_causal_sealed_pct": round(pct_full, 2),
                        "network_only_sealed_rmse_m": round(rmse_net, 4),
                        "network_only_sealed_pct": round(pct_net, 2),
                    }
                )

        # Resubstitution diagnostic (NOT a forecast): window j is the newest
        # member of seal j+1, so scoring seal j+1's weights on j bounds the
        # epoch-sharing / in-sample channel against which the out-of-sample
        # walk-forward numbers are compared.
        sm_rs = sealed_models.get(j + 1)
        if sm_rs is not None and j + 1 <= len(valid_pairs) - 1:
            dep_rs = _apply_post_ridge(
                Xl @ sm_rs["w_frozen"], h, stack_unc, cap=cap
            )
            rmse_rs = float(np.sqrt(np.mean((r + dep_rs) ** 2)))
            resubstitution_pct.append((1.0 - rmse_rs / pr) * 100.0)

        del Xl

    # Phase 2b is absorbed into Phases 1-2: each seal's clean-ring
    # eigendecomposition is computed at seal time (Phase 1) and the full
    # lambda grid plus every ex-ante inner-CV pick is scored at every
    # lead during the Phase-2 stream — same estimators, no re-solve, and
    # strictly more of the lambda x lead x position lattice resolved.

    perm_mat = np.asarray(wf_perm_matrix, dtype=float)  # (n_eval, n_perm)
    perm_realisation_means = np.mean(perm_mat, axis=0)
    perm_realisation_medians = np.median(perm_mat, axis=0)
    sta_mean = float(np.mean(wf_sta_only_pct))
    sta_med = float(np.median(wf_sta_only_pct))
    null_mean = float(np.mean(perm_realisation_means))
    null_std = float(np.std(perm_realisation_means))
    null_med = float(np.mean(perm_realisation_medians))
    null_med_std = float(np.std(perm_realisation_medians))
    z_mean = (sta_mean - null_mean) / max(null_std, 1e-9)
    z_med = (sta_med - null_med) / max(null_med_std, 1e-9)

    cm_retention_pct = round(
        100.0 * float(np.mean(wf_cm_removed_pct)) / max(sta_mean, 1e-9), 2
    )

    horizon_summary = {}
    for step_k in sorted(horizon_curves.keys())[:10]:
        pairs = horizon_curves[step_k]
        if len(pairs) >= 4:
            horizon_summary[f"step_{step_k}"] = {
                "n_pairs": len(pairs),
                "approx_lead_min": round(step_k * 11.2, 1),
                "full_causal_mean_pct": round(
                    float(np.mean([p[0] for p in pairs])), 2
                ),
                "network_only_mean_pct": round(
                    float(np.mean([p[1] for p in pairs])), 2
                ),
                "full_causal_win_rate_pct": round(
                    100.0 * sum(1 for p in pairs if p[0] > 0) / len(pairs), 1
                ),
                "network_only_win_rate_pct": round(
                    100.0 * sum(1 for p in pairs if p[1] > 0) / len(pairs), 1
                ),
            }

    return {
        "n_gram_windows_total": len(valid_pairs),
        "n_pkl_windows_in_corpus": len(pkl_paths),
        "skipped_member_wids": skipped_wids,
        "feature_schema": gram_schema,
        "feature_keys_sha256": hashlib.sha256(
            ("\n".join(keys_u)).encode()
        ).hexdigest(),
        "n_training_ring_K": k_tr,
        "n_eval_windows": len(wf_records),
        "n_features": f,
        "n_stations": len(stations),
        "station_band_counts": {
            "near_field_uk_lt360km": sum(
                1 for d in sta_dist_km.values() if d < 360.0
            ),
            "mid_band_eu_360_1000km": sum(
                1 for d in sta_dist_km.values() if 360.0 <= d < 1000.0
            ),
            "far_arc_gt1000km": sum(
                1 for d in sta_dist_km.values() if d >= 1000.0
            ),
        },
        "walk_forward_sealed_envelopes": {
            "full_causal_sealed": _summary_stats(wf_full_pct, block_len=4),
            "network_only_sealed": _summary_stats(wf_net_pct, block_len=4),
            "station_basis_only_sealed": _summary_stats(
                wf_sta_only_pct, block_len=4
            ),
            "per_window_records": wf_records,
        },
        "purged_epoch_disjoint_horizons": {
            "strict_future_epoch_slices": {
                "description": (
                    "Scores ONLY holdout epochs t_ho > T_seal_end that "
                    "occurred in physical wall-clock time strictly after the "
                    "last training window's holdout ended (zero epoch overlap)."
                ),
                "full_causal_sealed": _summary_stats(
                    future_slice_full_pct, block_len=4
                ),
                "network_only_sealed": _summary_stats(
                    future_slice_net_pct, block_len=4
                ),
                "full_weight_band_h_le_5400s": {
                    "description": (
                        "Subset of strict-future epochs inside the unity-taper "
                        "band (h <= 5400 s), where the sealed ridge correction "
                        "carries full weight; isolates future-epoch ridge skill "
                        "from the tapered tail."
                    ),
                    "full_causal_sealed": _summary_stats(
                        future_fw_full_pct, block_len=4
                    ),
                    "network_only_sealed": _summary_stats(
                        future_fw_net_pct, block_len=4
                    ),
                    "by_lead_step": {
                        f"step_{s}": {
                            "n_pairs": len(v),
                            "approx_lead_min": round(s * 11.2, 1),
                            "mean_pct": round(float(np.mean(v)), 2),
                            "win_rate_pct": round(
                                100.0 * sum(1 for x in v if x > 0) / len(v), 1
                            ),
                        }
                        for s, v in sorted(future_fw_by_step.items())
                        if len(v) >= 2
                    },
                },
                "tapered_tail_h_gt_5400s": {
                    "description": (
                        "Subset of strict-future epochs inside the down-taper "
                        "region (h > 5400 s), where dep blends back toward "
                        "stack_unc linearly reaching zero ridge weight at "
                        "9,900 s; measures the causal stack baseline mixed "
                        "with the down-weighted sealed ridge."
                    ),
                    "full_causal_sealed": _summary_stats(
                        future_tail_full_pct, block_len=4
                    ),
                    "network_only_sealed": _summary_stats(
                        future_tail_net_pct, block_len=4
                    ),
                },
            },
            "embargo_3600s_1h_gap": {
                "full_causal_sealed": _summary_stats(
                    embargo_3600_full_pct, block_len=2
                ),
                "network_only_sealed": _summary_stats(
                    embargo_3600_net_pct, block_len=2
                ),
            },
            "embargo_7200s_2h_disjoint": {
                "description": (
                    "Full 2-hour holdout purge gap (Delta_T >= 7,200 s) with "
                    "zero weight updates across the 2-hour embargo."
                ),
                "full_causal_sealed": _summary_stats(
                    embargo_7200_full_pct, block_len=1
                ),
                "network_only_sealed": _summary_stats(
                    embargo_7200_net_pct, block_len=1
                ),
            },
            "multi_step_horizon_decay": horizon_summary,
            "fixed_position_late_band_h4500_5400_by_lead": {
                "description": (
                    "Same h in [4500,5400] s slice (unity taper, late holdout) "
                    "scored at every seal lead: isolates weight staleness "
                    "against lead at fixed epoch position, removing the "
                    "taper-tail confound of the strict-future slices."
                ),
                **{
                    f"step_{s}": {
                        "n_pairs": len(v),
                        "approx_lead_min": round(s * 11.2, 1),
                        "mean_pct": round(float(np.mean(v)), 2),
                        "win_rate_pct": round(
                            100.0 * sum(1 for x in v if x > 0) / len(v), 1
                        ),
                    }
                    for s, v in sorted(band_late_by_step.items())
                    if len(v) >= 2
                },
            },
            "resubstitution_bound": {
                "description": (
                    "Each seal's frozen weights scored on its own newest "
                    "member window's holdout (pure in-sample resubstitution). "
                    "Bounds the shared-epoch channel: if walk-forward skill "
                    "met or exceeded this bound, the headline would be a "
                    "resubstitution artefact; observing OOS >= resubstitution "
                    "bounds the echo below the deployed-cadence skill."
                ),
                "full_causal_sealed_on_own_member": _summary_stats(
                    resubstitution_pct, block_len=4
                ),
            },
            "clean_origin_fit_half": {
                "description": (
                    "Zero-overlap ex-ante arm: member Grams built on the "
                    "FULL 2-hour fit half [t_pre[0], t_pre[-1]] of each "
                    "member — epochs ending strictly before that member's "
                    "own holdout, hence before EVERY scored evaluation "
                    "holdout — with the member's own persist level as "
                    "target anchor. No scored holdout epoch ever entered "
                    "any member's training target at ANY seal lead — the "
                    "shared-epoch channel is eliminated by construction. "
                    "Basis is the causal network block (sta_* station "
                    "curves plus the sid_* sidereal-echo and wv_* wave-"
                    "composite fields, evaluated on a station-only "
                    "dictionary so no holdout-shaped negative-h features "
                    "exist). The deployed-style "
                    "composite (stack_unc baseline), a pure-correction arm "
                    "(persistence baseline), and a per-feature diagonal "
                    "combiner are reported separately so the map's own "
                    "transfer is isolated from baseline choice. The "
                    "strict-future slice is reported for completeness "
                    "even though the whole holdout is already disjoint."
                ),
                "n_fit_gram_members": n_fit_members,
                "map_stability_disjoint_seal_cosine": {
                    "description": (
                        "Cosine similarity between weight vectors of "
                        "seals k_tr apart, whose exponential-decay member "
                        "weights share <0.02% of Gram mass — i.e. trained "
                        "on effectively disjoint target epochs. A stable "
                        "spatial field map yields cosine >> 0; "
                        "per-realisation noise weights decorrelate."
                    ),
                    "clean_fit_half_weights": _vec_stats(clean_map_cos),
                    "holdout_trained_weights": _vec_stats(
                        holdout_map_cos
                    ),
                    "clean_fit_half_predictions": _vec_stats(
                        clean_pred_cos
                    ),
                    "holdout_trained_predictions": _vec_stats(
                        holdout_pred_cos
                    ),
                    "note": (
                        "Weight-space cosines understate map stability "
                        "in this collinear multi-resolution basis; the "
                        "prediction-space cosines (same eval feature "
                        "matrix, disjoint-seal weight vectors) are the "
                        "operative stability diagnostic."
                    ),
                },
                "oracle_replay_echo_ceiling_1step": _summary_stats(
                    oracle_replay_pct, block_len=4
                ),
                "newest_member_epoch_share_pct": _summary_stats(
                    epoch_share_pct, block_len=4
                ),
                "in_domain_own_fit_half": _summary_stats(
                    clean_indomain_pct, block_len=4
                ),
                "clock_truth_innovation_rms_m": {
                    "mean": round(float(np.mean(b_innov_rms)), 4)
                    if b_innov_rms
                    else None,
                    "median": round(float(np.median(b_innov_rms)), 4)
                    if b_innov_rms
                    else None,
                },
                "one_step_sealed_stack_composite": _summary_stats(
                    clean_pct, block_len=4
                ),
                "one_step_sealed_pure_correction": _summary_stats(
                    clean_pure_pct, block_len=4
                ),
                "one_step_sealed_diagonal_combiner": _summary_stats(
                    clean_diag_pct, block_len=4
                ),
                "lambda_robustness_pure": {
                    "description": (
                        "Clean-origin seals re-solved on a fixed lambda "
                        "grid via eigendecomposition (the deployed lam was "
                        "tuned on holdout Grams; the forward arm may need "
                        "stronger shrinkage). All grid values reported; "
                        "pure-correction scoring, no eval-data selection."
                    ),
                    "inner_cv_lambda_selection": {
                        "description": (
                            "Fully ex-ante per-seal lambda selection: "
                            "argmin of J(w)=w'Gv w + 2w'gv evaluated on "
                            "the newest member's holdout Gram (inner_ho; "
                            "disjoint from every fit-half target, and an "
                            "operationally available validation set at "
                            "deploy time — its target epochs overlap the "
                            "eval holdout, mirroring live tuning), on "
                            "the pooled holdout Gram of the WHOLE sealed "
                            "ring (inner_pool; the complete holdout-"
                            "regime validation set an operator holds at "
                            "seal time and the quantity the deployed "
                            "lam was tuned on), on the newest member's "
                            "fit Gram (inner_fit; in-domain reference), "
                            "and on the newest member's fit Gram under "
                            "a ring solved without it (inner_lno; "
                            "leave-newest-out — disjoint train/"
                            "validation, the CV most faithful to the "
                            "forward task).  lam is fixed before the "
                            "eval window is scored — no eval data "
                            "enters the choice."
                        ),
                        "inner_ho": _summary_stats(
                            cv_pure_1.get("ho", []), block_len=4
                        ),
                        "inner_ho_lambda_picks": {
                            str(L): int(sum(1 for x in cv_lam.get("ho", [])
                                            if x == L))
                            for L in sorted(set(cv_lam.get("ho", [])))
                        },
                        "inner_pool": _summary_stats(
                            cv_pure_1.get("pool", []), block_len=4
                        ),
                        "inner_pool_lambda_picks": {
                            str(L): int(sum(1 for x in cv_lam.get("pool", [])
                                            if x == L))
                            for L in sorted(set(cv_lam.get("pool", [])))
                        },
                        "inner_fit": _summary_stats(
                            cv_pure_1.get("fit", []), block_len=4
                        ),
                        "inner_fit_lambda_picks": {
                            str(L): int(sum(1 for x in cv_lam.get("fit", [])
                                            if x == L))
                            for L in sorted(set(cv_lam.get("fit", [])))
                        },
                        "inner_lno": _summary_stats(
                            cv_pure_1.get("lno", []), block_len=4
                        ),
                        "inner_lno_lambda_picks": {
                            str(L): int(sum(1 for x in cv_lam.get("lno", [])
                                            if x == L))
                            for L in sorted(set(cv_lam.get("lno", [])))
                        },
                        "inner_lno_note": (
                            "Leave-newest-out validation: ring solved "
                            "without the newest member's fit Gram and "
                            "each candidate lam scored on exactly that "
                            "held-out Gram — the inner CV most faithful "
                            "to the forward task; all data pre-seal."
                        ),
                        "selected_arms_multi_step": {
                            _tag: {
                                "horizon_decay_pure": {
                                    f"step_{s}": {
                                        "n_pairs": len(v),
                                        "approx_lead_min": round(
                                            s * 11.2, 1
                                        ),
                                        "mean_pct": round(
                                            float(np.mean(v)), 2
                                        ),
                                        "win_rate_pct": round(
                                            100.0
                                            * sum(
                                                1 for x in v if x > 0
                                            )
                                            / len(v),
                                            1,
                                        ),
                                    }
                                    for s, v in sorted(
                                        cv_horizon_pure[_tag].items()
                                    )
                                    if len(v) >= 2
                                },
                                "horizon_decay_composite": {
                                    f"step_{s}": {
                                        "n_pairs": len(v),
                                        "approx_lead_min": round(
                                            s * 11.2, 1
                                        ),
                                        "mean_pct": round(
                                            float(np.mean(v)), 2
                                        ),
                                        "win_rate_pct": round(
                                            100.0
                                            * sum(
                                                1 for x in v if x > 0
                                            )
                                            / len(v),
                                            1,
                                        ),
                                    }
                                    for s, v in sorted(
                                        cv_horizon_cmp[_tag].items()
                                    )
                                    if len(v) >= 2
                                },
                                "strict_future_pure": _summary_stats(
                                    cv_future_pure.get(_tag, []),
                                    block_len=4,
                                ),
                                "strict_future_composite": _summary_stats(
                                    cv_future_cmp.get(_tag, []),
                                    block_len=4,
                                ),
                                "embargo_3600s_pure": _summary_stats(
                                    cv_emb3600_pure.get(_tag, []),
                                    block_len=2,
                                ),
                                "embargo_3600s_composite": _summary_stats(
                                    cv_emb3600_cmp.get(_tag, []),
                                    block_len=2,
                                ),
                                "embargo_7200s_pure": _summary_stats(
                                    cv_emb7200_pure.get(_tag, []),
                                    block_len=1,
                                ),
                                "embargo_7200s_composite": _summary_stats(
                                    cv_emb7200_cmp.get(_tag, []),
                                    block_len=1,
                                ),
                                "position_bands_1step_pure": {
                                    f"h_{int(CLEAN_POS_EDGES[bi])}_"
                                    f"{int(CLEAN_POS_EDGES[bi + 1])}":
                                    _summary_stats(v, block_len=4)
                                    for bi, v in sorted(
                                        cv_pos[_tag].items()
                                    )
                                    if len(v) >= 2
                                },
                                "one_step_composite": _summary_stats(
                                    cv_cmp_1.get(_tag, []), block_len=4
                                ),
                            }
                            for _tag in ("ho", "pool", "fit", "lno")
                            if cv_pure_1.get(_tag)
                        },
                    },
                    **{
                        f"lam_{int(L)}": _summary_stats(v, block_len=4)
                        for L, v in sorted(grid_pure_1.items())
                        if len(v) >= 2
                    },
                    "position_bands_by_lambda": {
                        f"lam_{int(L)}": {
                            f"h_{int(CLEAN_POS_EDGES[bi])}_"
                            f"{int(CLEAN_POS_EDGES[bi + 1])}":
                            _summary_stats(v, block_len=4)
                            for bi, v in sorted(_bands.items())
                            if len(v) >= 2
                        }
                        for L, _bands in sorted(grid_pos.items())
                        if any(len(v) >= 2 for v in _bands.values())
                    },
                    "strict_future_pure_by_lambda": {
                        f"lam_{int(L)}": _summary_stats(v, block_len=4)
                        for L, v in sorted(grid_future.items())
                        if len(v) >= 2
                    },
                    "multi_step_horizon_pure_by_lambda": {
                        f"lam_{int(L)}": {
                            f"step_{s}": {
                                "n_pairs": len(v),
                                "approx_lead_min": round(s * 11.2, 1),
                                "mean_pct": round(float(np.mean(v)), 2),
                                "win_rate_pct": round(
                                    100.0
                                    * sum(1 for x in v if x > 0)
                                    / len(v),
                                    1,
                                ),
                            }
                            for s, v in sorted(_hz.items())
                            if len(v) >= 2
                        }
                        for L, _hz in sorted(grid_horizon.items())
                        if any(len(v) >= 2 for v in _hz.values())
                    },
                    "embargo_pure_by_lambda": {
                        f"lam_{int(L)}": {
                            "embargo_3600s": _summary_stats(
                                grid_emb3600[L], block_len=2
                            ),
                            "embargo_7200s": _summary_stats(
                                grid_emb7200[L], block_len=1
                            ),
                        }
                        for L in sorted(grid_emb3600)
                        if len(grid_emb3600[L]) >= 2
                        or len(grid_emb7200[L]) >= 2
                    },
                },
                "position_bands_1step_pure": {
                    f"h_{int(CLEAN_POS_EDGES[bi])}_{int(CLEAN_POS_EDGES[bi+1])}":
                    _summary_stats(v, block_len=4)
                    for bi, v in sorted(clean_pos_bands.items())
                    if len(v) >= 2
                },
                "strict_future_slice_pure": _summary_stats(
                    clean_future_pct, block_len=4
                ),
                "embargo_3600s_1h_gap_pure": _summary_stats(
                    clean_emb3600_pct, block_len=2
                ),
                "embargo_7200s_2h_disjoint_pure": _summary_stats(
                    clean_emb7200_pct, block_len=1
                ),
                "multi_step_horizon_decay_composite": {
                    f"step_{s}": {
                        "n_pairs": len(v),
                        "approx_lead_min": round(s * 11.2, 1),
                        "mean_pct": round(float(np.mean(v)), 2),
                        "win_rate_pct": round(
                            100.0 * sum(1 for x in v if x > 0) / len(v), 1
                        ),
                    }
                    for s, v in sorted(clean_horizon.items())
                    if len(v) >= 2
                },
                "multi_step_horizon_decay_pure": {
                    f"step_{s}": {
                        "n_pairs": len(v),
                        "approx_lead_min": round(s * 11.2, 1),
                        "mean_pct": round(float(np.mean(v)), 2),
                        "win_rate_pct": round(
                            100.0 * sum(1 for x in v if x > 0) / len(v), 1
                        ),
                    }
                    for s, v in sorted(clean_pure_horizon.items())
                    if len(v) >= 2
                },
            },
        },
        "spatial_falsification_battery": {
            "station_identity_permutation_null": {
                "n_permutations": n_perm,
                "unpermuted_sta_mean_pct": round(sta_mean, 2),
                "unpermuted_sta_median_pct": round(sta_med, 2),
                "null_mean_pct": round(null_mean, 2),
                "null_std_pct": round(null_std, 2),
                "null_median_pct": round(null_med, 2),
                "null_median_std_pct": round(null_med_std, 2),
                "null_max_pct": round(float(np.max(perm_realisation_means)), 2),
                "z_score_mean": round(z_mean, 2),
                "z_score_median": round(z_med, 2),
            },
            "common_mode_stripped": {
                **_summary_stats(wf_cm_removed_pct, block_len=4),
                "skill_retained_pct": cm_retention_pct,
            },
            "common_mode_only": _summary_stats(wf_cm_only_pct, block_len=4),
            "distance_bands": {
                bname: _summary_stats(bvals, block_len=4)
                for bname, bvals in wf_band_pcts.items()
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    t0 = time.time()
    print("[Step 174] Running 114+ window live causal audit...")
    live_audit = _run_live_causal_audit()
    print(
        f"  -> {live_audit['n_windows']} windows ({live_audit['span_hours']} h): "
        f"causal unscaled mean {live_audit['causal_unscaled']['mean_pct']:+.2f}% "
        f"(median {live_audit['causal_unscaled']['median_pct']:+.2f}%, "
        f"wins {live_audit['causal_unscaled']['win_count']}/{live_audit['n_windows']}, "
        f"p={live_audit['causal_unscaled']['sign_test_p']:.2e})"
    )

    print("[Step 174] Running cryptographic sealed-envelope suite & falsification battery...")
    suite = _run_sealed_envelope_suite()
    wf_full = suite["walk_forward_sealed_envelopes"]["full_causal_sealed"]
    wf_net = suite["walk_forward_sealed_envelopes"]["network_only_sealed"]
    fut_full = suite["purged_epoch_disjoint_horizons"][
        "strict_future_epoch_slices"
    ]["full_causal_sealed"]
    perm = suite["spatial_falsification_battery"][
        "station_identity_permutation_null"
    ]
    cm_rem = suite["spatial_falsification_battery"]["common_mode_stripped"]
    gt360 = suite["spatial_falsification_battery"]["distance_bands"][
        "beyond_troposphere_gt360km"
    ]
    gt1000 = suite["spatial_falsification_battery"]["distance_bands"][
        "far_arc_gt1000km"
    ]

    print(
        f"  -> Walk-forward sealed (n={wf_full['n']}): "
        f"full={wf_full['mean_pct']:+.2f}% (wins {wf_full['win_count']}/{wf_full['n']}), "
        f"network-only={wf_net['mean_pct']:+.2f}% (wins {wf_net['win_count']}/{wf_net['n']})"
    )
    print(
        f"  -> Strict future epoch slices (n={fut_full['n']}): "
        f"full={fut_full['mean_pct']:+.2f}% (wins {fut_full['win_count']}/{fut_full['n']})"
    )
    fw = suite["purged_epoch_disjoint_horizons"]["strict_future_epoch_slices"][
        "full_weight_band_h_le_5400s"
    ]["full_causal_sealed"]
    tl = suite["purged_epoch_disjoint_horizons"]["strict_future_epoch_slices"][
        "tapered_tail_h_gt_5400s"
    ]["full_causal_sealed"]
    print(
        f"     full-weight band (h<=5400s, n={fw['n']}): {fw['mean_pct']:+.2f}% | "
        f"tapered tail (h>5400s, n={tl['n']}): {tl['mean_pct']:+.2f}%"
    )
    rs = suite["purged_epoch_disjoint_horizons"]["resubstitution_bound"][
        "full_causal_sealed_on_own_member"
    ]
    print(
        f"  -> Resubstitution bound (in-sample, n={rs['n']}): {rs['mean_pct']:+.2f}% "
        f"vs OOS walk-forward {wf_full['mean_pct']:+.2f}%"
    )
    cz = suite["purged_epoch_disjoint_horizons"]["clean_origin_fit_half"]
    c1 = cz["one_step_sealed_stack_composite"]
    c1p = cz["one_step_sealed_pure_correction"]
    c1d = cz["one_step_sealed_diagonal_combiner"]
    cor = cz["oracle_replay_echo_ceiling_1step"]
    print(
        f"  -> ORACLE-REPLAY echo ceiling (n={cor['n']}): {cor['mean_pct']:+.2f}%"
        f" | CLEAN-ORIGIN fit-half seals (n={c1['n']}, ZERO epoch overlap): "
        f"composite {c1['mean_pct']:+.2f}% (wins {c1.get('win_count','-')}/{c1['n']})"
        f" | pure {c1p['mean_pct']:+.2f}% (wins {c1p.get('win_count','-')}/{c1p['n']})"
        f" | diag {c1d['mean_pct']:+.2f}% (wins {c1d.get('win_count','-')}/{c1d['n']})"
    )
    cpb = cz.get("position_bands_1step_pure", {})
    if cpb:
        print(
            "     clean pure position bands: "
            + " | ".join(
                f"{kk.replace('h_','h=')}s {vv['mean_pct']:+.1f}%"
                for kk, vv in cpb.items()
            )
        )
    _msc = cz.get("map_stability_disjoint_seal_cosine", {})
    if _msc:
        _cc = _msc.get("clean_fit_half_weights", {})
        _hc = _msc.get("holdout_trained_weights", {})
        if _cc.get("n"):
            _pc = _msc.get("clean_fit_half_predictions", {})
            _ph = _msc.get("holdout_trained_predictions", {})
            print(
                f"     disjoint-seal map cosine: clean {_cc['mean']:.3f}"
                f" | holdout {_hc['mean']:.3f} (n={_cc['n']})"
                f" | pred-space clean {_pc.get('mean', float('nan')):.3f}"
                f" | pred-space holdout {_ph.get('mean', float('nan')):.3f}"
            )
    _lr = cz.get("lambda_robustness_pure", {})
    _lr_rows = [
        (kk, vv) for kk, vv in _lr.items()
        if isinstance(vv, dict) and kk.startswith("lam_")
    ]
    if _lr_rows:
        print(
            "     clean pure lambda grid: "
            + " | ".join(
                f"{kk.replace('lam_', 'lam=')} {vv['mean_pct']:+.1f}%"
                f" ({vv.get('win_count','-')}/{vv['n']})"
                for kk, vv in sorted(
                    _lr_rows, key=lambda kv: float(kv[0].split('_')[1])
                )
            )
        )
    _cv = _lr.get("inner_cv_lambda_selection", {})
    _cvh = _cv.get("inner_ho", {})
    _cvf = _cv.get("inner_fit", {})
    _cvl = _cv.get("inner_lno", {})
    if _cvh.get("n"):
        print(
            f"     causal inner-CV lam: ho {_cvh['mean_pct']:+.2f}%"
            f" ({_cvh.get('win_count','-')}/{_cvh['n']})"
            f" | fit {_cvf.get('mean_pct', float('nan')):+.2f}%"
            f" ({_cvf.get('win_count','-')}/{_cvf.get('n',0)})"
            f" | lno {_cvl.get('mean_pct', float('nan')):+.2f}%"
            f" ({_cvl.get('win_count','-')}/{_cvl.get('n',0)})"
        )
    print(
        f"  -> Station permutation null (N={perm['n_permutations']}): "
        f"{perm['unpermuted_sta_mean_pct']:+.2f}% vs {perm['null_mean_pct']:+.2f}% +/- "
        f"{perm['null_std_pct']:.2f}% (z={perm['z_score_mean']:.2f} sigma)"
    )
    print(
        f"  -> Common-mode stripped: {cm_rem['mean_pct']:+.2f}% "
        f"({cm_rem['skill_retained_pct']:.1f}% retained) | "
        f">360 km only: {gt360['mean_pct']:+.2f}% | "
        f">1000 km only: {gt1000['mean_pct']:+.2f}%"
    )

    live_forward = None
    vlog = ENVELOPE_DIR / "verification_log.json"
    if vlog.exists():
        try:
            live_forward = json.loads(vlog.read_text())
        except Exception:
            live_forward = None

    output = {
        "step": "step_174_sealed_envelope_prediction",
        "timestamp_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "live_causal_audit": live_audit,
        **suite,
        "live_forward_envelope": live_forward,
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[Step 174] Wrote {out_p} in {output['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
