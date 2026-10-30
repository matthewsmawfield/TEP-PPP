#!/usr/bin/env python3
"""Step 174b — Clean-origin sealed arm WITHOUT the deployed horizon taper.

Companion diagnostic to ``step_174_sealed_envelope_prediction``.  The clean
(fit-half) seals are rebuilt identically — same pinned 40-window corpus,
same K=16 / half-life 1.25 recency weights, same ridge grid — but the
pure-correction scoring reports the untapered prediction
``clip(rg - mean(rg), +-cap)`` alongside the deployed-tapered one.

Motivation: the deployed trapezoid taper (T2=5400, T3=9900 s) is tuned for
the live holdout-trained stack, where it mutes stale-weight tail positions.
On the zero-overlap clean arm it damps exactly the band where the forward
map is strongest (h in [5400,7300] s — the quasi-static mode), so the
tapered headline understates genuine forward transfer.  Both transforms
are reported for every grid lambda, no eval-data selection.

Output: results/outputs/step_174_clean_untapered_arm.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "steps"))

import step_174_sealed_envelope_prediction as s174  # noqa: E402
from rpi.gps_web import (  # noqa: E402
    _env_float,
    _ridge_feature_dict,
    _ridge_stats_load_or_build,
    _ridge_stats_fit_load_or_build,
)

RESULTS_DIR = s174.RESULTS_DIR
OUT_PATH = (
    REPO_ROOT / "results" / "outputs" / "step_174_clean_untapered_arm.json"
)
LAM_GRID = (2000.0, 2e5, 1e6, 5e6, 1e7, 2e7)
POS_EDGES = (0.0, 1800.0, 3600.0, 5400.0, 7300.0)


def main() -> int:
    t0 = time.time()
    corpus_end_wid = int(
        _env_float("TEP_STEP174_CORPUS_END_WID", 1793132392))
    n_corpus = int(_env_float("TEP_STEP174_CORPUS_N", 40))
    pkl_paths = [
        p for p in sorted(
            RESULTS_DIR.glob("raw_window_*.pkl"), key=lambda p: p.name)
        if s174._window_id(p) <= corpus_end_wid
    ][-n_corpus:]
    valid_pairs, key_sets = [], []
    for pklp in pkl_paths:
        s_i = _ridge_stats_load_or_build(pklp)
        key_sets.append(set(s_i[0]))
        valid_pairs.append((s174._window_id(pklp), pklp, pklp))
    d_last = pickle.loads(valid_pairs[-1][2].read_bytes())
    h_last = np.asarray(d_last["t_ho"], float) - float(
        np.asarray(d_last["t_pre"], float)[-1])
    feats_last = _ridge_feature_dict(h_last, d_last)
    key_sets.append(set(feats_last.keys()))
    del d_last, feats_last
    keys_u = sorted(set().union(*key_sets))
    k_idx = {k: i for i, k in enumerate(keys_u)}
    f = len(keys_u)
    print(f"[174b] f={f}, corpus={len(valid_pairs)} windows", flush=True)

    k_tr = min(int(_env_float("TEP_CAUSAL_RIDGE_K", 16)), 16)
    hl = _env_float("TEP_CAUSAL_RIDGE_HL", 1.25)
    cap = _env_float("TEP_CAUSAL_RIDGE_CAP_M", 8.0)
    decay = np.float32(0.5 ** (1.0 / max(hl, 1e-9)))

    XTX_c = np.zeros((f, f), dtype=np.float32)
    XTR_c = np.zeros(f, dtype=np.float32)
    seals = {}
    for idx_w, (wid, npzp, pklp) in enumerate(valid_pairs):
        if idx_w >= k_tr:
            sm = {}
            for L in LAM_GRID:
                A = XTX_c.copy()
                A[np.diag_indices(f)] += np.float32(L)
                sm[L] = s174._spd_solve(A, -XTR_c)
            seals[idx_w] = sm
            print(f"  seal {idx_w} ({time.time() - t0:.0f}s)", flush=True)
        XTX_c *= decay
        XTR_c *= decay
        s_fit = _ridge_stats_fit_load_or_build(pklp)
        if s_fit is not None:
            keys_f, xtx_f, xtr_f = s_fit
            ii_local = np.asarray(
                [k_idx.get(k, -1) for k in keys_f], np.int64)
            sel = ii_local >= 0
            if np.any(sel):
                ii = ii_local[sel]
                XTX_c[np.ix_(ii, ii)] += decay * xtx_f.astype(
                    np.float32)[np.ix_(sel, sel)]
                XTR_c[ii] += decay * xtr_f.astype(np.float32)[sel]
    del XTX_c, XTR_c

    # Score: for every eval window, every 1-step seal, every grid lambda,
    # both the deployed-tapered and the untapered pure correction.
    tap_pct = {L: [] for L in LAM_GRID}
    raw_pct = {L: [] for L in LAM_GRID}
    tap_band = {L: {b: [] for b in range(4)} for L in LAM_GRID}
    raw_band = {L: {b: [] for b in range(4)} for L in LAM_GRID}

    for j in range(k_tr, len(valid_pairs)):
        wid_j, _, pklp_j = valid_pairs[j]
        d = pickle.loads(pklp_j.read_bytes())
        t_ho = np.asarray(d["t_ho"], float)
        t_pre = np.asarray(d["t_pre"], float)
        h = t_ho - float(t_pre[-1])
        feats = _ridge_feature_dict(h, d)
        r = np.asarray(d["persist_pred"], float) - np.asarray(
            d["b_ho"], float)
        del d
        Xl = np.zeros((len(h), f), dtype=np.float64)
        for k, v in feats.items():
            col = k_idx.get(k)
            if col is not None and len(v) == len(h):
                Xl[:, col] = v
        del feats
        pr = float(np.sqrt(np.mean(r ** 2)))
        wv = s174._ridge_wv(h)
        for L, w in seals[j].items():
            rg = Xl @ w - (Xl @ w).mean()
            dep_t = np.clip(wv * rg, -cap, cap)
            dep_u = np.clip(rg, -cap, cap)
            rt = float(np.sqrt(np.mean((r + dep_t) ** 2)))
            ru = float(np.sqrt(np.mean((r + dep_u) ** 2)))
            tap_pct[L].append((1.0 - rt / pr) * 100.0)
            raw_pct[L].append((1.0 - ru / pr) * 100.0)
            for bi in range(4):
                m = (h >= POS_EDGES[bi]) & (h < POS_EDGES[bi + 1])
                if int(m.sum()) >= 300:
                    prb = float(np.sqrt(np.mean(r[m] ** 2)))
                    tap_band[L][bi].append(
                        (1.0 - float(np.sqrt(
                            np.mean((r[m] + dep_t[m]) ** 2))) / prb)
                        * 100.0)
                    raw_band[L][bi].append(
                        (1.0 - float(np.sqrt(
                            np.mean((r[m] + dep_u[m]) ** 2))) / prb)
                        * 100.0)
        del Xl
        print(f"  eval {j} ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "step": "step_174b_clean_untapered_arm",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "description": (
            "Clean-origin (fit-half Gram) sealed arm of step_174 scored "
            "with and without the deployed horizon taper.  Corpus, seals, "
            "recency weights and lambda grid are identical to the pinned "
            "step_174 replay; weights are frozen at each seal before the "
            "eval window is scored — fully ex-ante.  demean retained "
            "(near no-op: features are standardised zero-mean); clip at "
            "+/-cap retained.  'tapered' reproduces the committed "
            "step_174 pure-correction arm; 'untapered' removes only the "
            "trapezoid horizon weight."
        ),
        "corpus_end_wid": corpus_end_wid,
        "n_eval_windows": len(tap_pct[LAM_GRID[0]]),
        "one_step_pure_by_lambda": {
            f"lam_{L:g}": {
                "tapered": s174._summary_stats(tap_pct[L], block_len=4),
                "untapered": s174._summary_stats(raw_pct[L], block_len=4),
            }
            for L in LAM_GRID
        },
        "position_bands_by_lambda": {
            f"lam_{L:g}": {
                f"h_{int(POS_EDGES[bi])}_{int(POS_EDGES[bi + 1])}": {
                    "tapered": s174._summary_stats(
                        tap_band[L][bi], block_len=4),
                    "untapered": s174._summary_stats(
                        raw_band[L][bi], block_len=4),
                }
                for bi in range(4)
                if len(raw_band[L][bi]) >= 2
            }
            for L in LAM_GRID
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    for L in LAM_GRID:
        t = out["one_step_pure_by_lambda"][f"lam_{L:g}"]
        print(
            f"lam={L:g}: tapered {t['tapered']['mean_pct']:+.2f}% "
            f"({t['tapered']['win_count']}/{t['tapered']['n']}) | "
            f"untapered {t['untapered']['mean_pct']:+.2f}% "
            f"({t['untapered']['win_count']}/{t['untapered']['n']})"
        )
    print(f"saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
