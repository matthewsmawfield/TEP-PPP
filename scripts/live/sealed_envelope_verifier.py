#!/usr/bin/env python3
"""Cryptographic Sealed-Envelope Verifier for Live TEP-PPP Clock Predictions.

Proves operational ex-ante (before-the-fact) predictive skill without touching
hardware or disturbing the running 72 h satellite-free quartz capture:

1. ``--seal``:
   Snapshots the causal ridge predictor state at wall-clock time ``T_seal`` using
   ONLY archived Gram matrices existing on disk at ``T_seal``. Solves and freezes
   the joint weight vector ``w_frozen`` and field-only weight vector
   ``wnf_frozen``, writes them to ``data/ppp_live/sealed_envelopes/``, and
   records their SHA-256 cryptographic hash in a signed JSON manifest alongside
   ``last_window_before_seal``.

2. ``--verify`` / ``--watch``:
   Verifies the SHA-256 hash of the sealed weight vectors, then discovers all
   live PPP windows solved strictly AFTER ``T_seal`` (``window_id >
   last_window_before_seal``). On each post-seal window, scores:
     - ``persist_rmse_m``: standard PPP clock persistence baseline
     - ``frozen_seal_rmse_m``: prediction using the exact weights frozen and
       hashed at ``T_seal`` (zero post-seal training or truth access)
     - ``live_causal_rmse_m``: the live rig's pre-scale, pre-guard causal
       prediction (``_dep_o2``, before any vertex guard)
     - ``live_ramp_rmse_m``: the live causal prediction with the fixed horizon
       ramp (still zero holdout-truth access)
     - ``guarded_oracle_rmse_m``: the 8-segment NNLS vertex-guarded curve
       (truth-referenced safety bound, reported separately)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

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
)

RESULTS_DIR = REPO_ROOT / "data" / "ppp_live" / "results"
ENVELOPE_DIR = REPO_ROOT / "data" / "ppp_live" / "sealed_envelopes"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast SPD solve in float32 with float64 fallback."""
    try:
        L = np.linalg.cholesky(A.astype(np.float32, copy=False))
        y = np.linalg.solve(L, b.astype(np.float32, copy=False))
        return np.linalg.solve(L.T, y).astype(np.float64)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, b).astype(np.float64)


def _scoring_params_from_env() -> dict:
    """The complete scoring transform applied to frozen weights.

    Every knob that maps a frozen ridge prediction onto the deployed
    correction lives here — horizon-taper flag and corners, blend,
    demean, cap, and the fc-gate thresholds.  These are sealed into the
    envelope manifest at seal time and replayed verbatim at verify
    time, so a post-seal edit of .env.local cannot silently re-tune a
    "frozen" evaluation.
    """
    return {
        "fcg": float(_env_float("TEP_CAUSAL_RIDGE_FCG", 0.0)),
        "fcg2": float(_env_float("TEP_CAUSAL_RIDGE_FCG2", -0.20)),
        "fcd": float(_env_float("TEP_CAUSAL_RIDGE_FCD", 15.0)),
        "fcg3": float(_env_float("TEP_CAUSAL_RIDGE_FCG3", 0.0)),
        "fcr": float(_env_float("TEP_CAUSAL_RIDGE_FCR", 0.7)),
        "fcg4": float(_env_float("TEP_CAUSAL_RIDGE_FCG4", 0.40)),
        "fcr2": float(_env_float("TEP_CAUSAL_RIDGE_FCR2", 0.60)),
        "fcg4u": float(_env_float("TEP_CAUSAL_RIDGE_FCG4U", 0.0)),
        "fcg5": float(_env_float("TEP_CAUSAL_RIDGE_FCRHI", 4.0)),
        "demean": float(_env_float("TEP_CAUSAL_RIDGE_DEMEAN", 1.0)),
        "taper": float(_env_float("TEP_CAUSAL_RIDGE_TAPER", 1.0)),
        "t0": float(_env_float("TEP_CAUSAL_RIDGE_T0", 0.0)),
        "t1": float(_env_float("TEP_CAUSAL_RIDGE_T1", 225.0)),
        "t2": float(_env_float("TEP_CAUSAL_RIDGE_T2", 5400.0)),
        "t3": float(_env_float("TEP_CAUSAL_RIDGE_T3", 9900.0)),
        "blend": float(_env_float("TEP_CAUSAL_RIDGE_BLEND", 1.0)),
        "cap": float(_env_float("TEP_CAUSAL_RIDGE_CAP_M", 8.0)),
    }


def _effective_scoring_params(sealed: dict | None) -> dict:
    """Sealed params win for every key they define; env fills the rest
    (legacy envelopes sealed before the transform was committed)."""
    eff = _scoring_params_from_env()
    if sealed:
        for k, v in sealed.items():
            if k in eff:
                try:
                    eff[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return eff


def _window_id(path: Path) -> int:
    stem = path.stem
    for prefix in ("last_result_window_", "raw_window_"):
        if stem.startswith(prefix):
            return int(stem[len(prefix):])
    return int(stem.split("_")[-1])


def _summarize_window_json(jpath: Path) -> dict | None:
    try:
        d = json.loads(jpath.read_text())
        cp = d.get("clock_prediction") or {}
        af = cp.get("alpha_fit") or {}
        cr = af.get("causal_ridge") or {}
        ds = cr.get("dep_scale") or af.get("dep_scale") or {}
        pr = float(cp.get("persist_rmse_m") or 0.0)
        cu = ds.get("rmse_unscaled_m") or cp.get("tep_rmse_deployed_causal_m")
        cramp = ds.get("rmse_flat_ramp_m")
        cg = ds.get("rmse_guarded_m") or cp.get("tep_rmse_deployed_m")
        if pr <= 0 or cu is None:
            return None
        cu = float(cu)
        cramp = float(cramp) if cramp is not None else cu
        cg = float(cg) if cg is not None else cu
        return {
            "window_id": _window_id(jpath),
            "mtime": round(jpath.stat().st_mtime, 3),
            "persist_rmse_m": round(pr, 4),
            "live_causal_rmse_m": round(cu, 4),
            "live_causal_vs_persist_pct": round((1.0 - cu / pr) * 100.0, 2),
            "live_ramp_rmse_m": round(cramp, 4),
            "live_ramp_vs_persist_pct": round((1.0 - cramp / pr) * 100.0, 2),
            "guarded_oracle_rmse_m": round(cg, 4),
            "guarded_oracle_vs_persist_pct": round((1.0 - cg / pr) * 100.0, 2),
            "guard_fired": bool(ds.get("guard_fired", False)),
        }
    except Exception:
        return None


def seal_envelope(
    tag: str | None = None,
    backtest_n: int = 0,
    update_latest: bool = True,
) -> Path:
    ENVELOPE_DIR.mkdir(parents=True, exist_ok=True)
    seal_epoch = time.time()
    seal_utc = dt.datetime.fromtimestamp(seal_epoch, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    stamp = tag or dt.datetime.fromtimestamp(
        seal_epoch, tz=dt.timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    pkls = sorted(RESULTS_DIR.glob("raw_window_*.pkl"), key=lambda p: p.name)
    if backtest_n > 0:
        if len(pkls) <= backtest_n + 4:
            raise RuntimeError(f"Not enough windows for backtest_n={backtest_n}")
        pkls = pkls[:-backtest_n]
    if not pkls:
        raise RuntimeError(f"No raw_window_*.pkl found in {RESULTS_DIR}")

    last_wid = _window_id(pkls[-1])
    k_tr = int(_env_float("TEP_CAUSAL_RIDGE_K", 16))
    hl = _env_float("TEP_CAUSAL_RIDGE_HL", 1.25)
    lam = _env_float("TEP_CAUSAL_RIDGE_LAM", 2000.0)

    mems = pkls[-k_tr:]
    # Gram caches are transient (the live ring purges them); always go
    # through load-or-build on the durable .pkl archives so a mid-seal
    # purge cannot truncate the member set.
    stats_mem = []
    for mp in mems:
        try:
            s_i = _ridge_stats_load_or_build(mp)
        except Exception:
            s_i = None
        if s_i is not None:
            stats_mem.append((mp, s_i))
    if len(stats_mem) < 4:
        raise RuntimeError(
            f"Insufficient Gram archives ({len(stats_mem)})"
        )

    key_sets = [set(s_i[0]) for _, s_i in stats_mem]

    keys_u = sorted(set().union(*key_sets))
    idx = {k: i for i, k in enumerate(keys_u)}
    f = len(keys_u)
    XTX = np.zeros((f, f), np.float32)
    XTR = np.zeros(f, np.float32)
    ns = len(stats_mem)
    hw = (0.5 ** (np.arange(ns, 0, -1) / max(hl, 1e-9))).astype(np.float32)

    for wi, (mp, s_i) in enumerate(stats_mem):
        keys_i, xtx_i64, xtr_i64, fcd_i = s_i
        xtx_i = xtx_i64.astype(np.float32, copy=False)
        xtr_i = xtr_i64.astype(np.float32, copy=False)
        s = float(_fcm_scale(fcd_i))
        ii = np.asarray([idx[k] for k in keys_i])
        w_i = float(hw[wi])
        if s >= 1.0:
            XTX[np.ix_(ii, ii)] += w_i * xtx_i
            XTR[ii] += w_i * xtr_i
        else:
            fp = np.asarray([k.startswith("fc") for k in keys_i])
            zv = np.where(fp, s, 1.0).astype(np.float32)
            XTX[np.ix_(ii, ii)] += w_i * (
                zv[:, None] * xtx_i * zv[None, :]
            )
            XTR[ii] += w_i * (zv * xtr_i)

    XTX_reg = XTX.copy()
    XTX_reg[np.diag_indices(f)] += np.float32(lam)
    w_frozen = _solve_spd(XTX_reg, -XTR)

    lfc = np.asarray([k.startswith("fc") for k in keys_u])
    lnf = ~lfc
    nf_idx = np.flatnonzero(lnf)
    XTX_nf = XTX[np.ix_(nf_idx, nf_idx)].copy()
    XTX_nf[np.diag_indices(len(nf_idx))] += np.float32(lam)
    wnf_sub = _solve_spd(XTX_nf, -XTR[nf_idx])
    wnf_frozen = np.zeros(f, np.float64)
    wnf_frozen[nf_idx] = wnf_sub

    npz_out = ENVELOPE_DIR / f"envelope_{stamp}_weights.npz"
    np.savez_compressed(
        npz_out,
        keys=np.asarray(keys_u),
        w_frozen=w_frozen,
        wnf_frozen=wnf_frozen,
    )
    weights_sha256 = _sha256_file(npz_out)

    jpaths = [
        jp
        for jp in sorted(
            RESULTS_DIR.glob("last_result_window_*.json"), key=lambda p: p.name
        )
        if _window_id(jp) <= last_wid
    ]
    pre_history = []
    for jp in jpaths[-10:]:
        s = _summarize_window_json(jp)
        if s is not None:
            pre_history.append(s)

    manifest = {
        "envelope_id": f"envelope_{stamp}",
        "seal_utc": seal_utc,
        "seal_epoch_s": round(seal_epoch, 3),
        "last_window_before_seal": last_wid,
        "weights_file": npz_out.name,
        "weights_sha256": weights_sha256,
        "hyperparameters": {
            "K": k_tr,
            "hl": hl,
            "lam": lam,
            "fcm": os.environ.get("TEP_CAUSAL_RIDGE_FCM", ""),
            "demean": float(_env_float("TEP_CAUSAL_RIDGE_DEMEAN", 1.0)),
            "blend": float(_env_float("TEP_CAUSAL_RIDGE_BLEND", 1.0)),
            "taper_corners_s": [
                float(_env_float("TEP_CAUSAL_RIDGE_T0", 0.0)),
                float(_env_float("TEP_CAUSAL_RIDGE_T1", 225.0)),
                float(_env_float("TEP_CAUSAL_RIDGE_T2", 5400.0)),
                float(_env_float("TEP_CAUSAL_RIDGE_T3", 9900.0)),
            ],
            "cap_m": float(_env_float("TEP_CAUSAL_RIDGE_CAP_M", 8.0)),
        },
        # The full scoring transform is sealed alongside the weights so a
        # post-seal .env.local edit cannot silently re-tune the frozen
        # evaluation; verify_envelope replays these verbatim.
        "scoring_params": _scoring_params_from_env(),
        "n_member_windows": ns,
        "member_windows": [_window_id(mp) for mp, _ in stats_mem],
        "n_features": f,
        "pre_seal_trailing_10_windows": pre_history,
    }
    manifest_path = ENVELOPE_DIR / f"envelope_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if update_latest:
        latest_link = ENVELOPE_DIR / "latest_envelope.json"
        latest_link.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def window_scoring_arrays(d: dict) -> dict:
    """The per-window quantities shared by every envelope scored on the
    same archive (features, persistence residual, guard diagnostics).
    Factored out so a lattice audit can load each window once and apply
    many seals without re-deriving these arrays."""
    t_ho = np.asarray(d["t_ho"], float)
    t_pre = np.asarray(d["t_pre"], float)
    h = t_ho - float(t_pre[-1])
    feats = _ridge_feature_dict(h, d)
    r = np.asarray(d["persist_pred"], float) - np.asarray(d["b_ho"], float)
    dev = (
        np.asarray(d.get("free_clk_ho"), float).ravel()
        - np.asarray(d["persist_pred"], float).ravel()
        if d.get("free_clk_ho") is not None
        else None
    )
    return {
        "t_ho": t_ho,
        "h": h,
        "feats": feats,
        "r": r,
        "stack_unc": np.asarray(d["stack_unc"], float),
        "dev": dev,
    }


def _score_frozen_core(
    wa: dict,
    keys_u: list[str],
    w_frozen: np.ndarray,
    wnf_frozen: np.ndarray,
    seal_t_end: float | None = None,
    params: dict | None = None,
    Xl: np.ndarray | None = None,
) -> dict | None:
    """Score a frozen seal on precomputed window scoring arrays.  This is
    the identical scoring path used by _score_frozen_on_pkl (and hence by
    --watch/--verify/--verify-all); the lattice audit calls it directly,
    optionally supplying a precomputed feature matrix (Xl) when several
    envelopes share an identical keys_u basis."""
    try:
        sp = _effective_scoring_params(params)
        t_ho = wa["t_ho"]
        h = wa["h"]
        feats = wa["feats"]
        r = wa["r"]

        if Xl is None:
            idx = {k: i for i, k in enumerate(keys_u)}
            Xl = np.zeros((len(h), len(keys_u)), np.float64)
            for k, v in feats.items():
                j = idx.get(k)
                if j is not None and len(v) == len(h):
                    Xl[:, j] = v

        lfc = np.asarray([k.startswith("fc") for k in keys_u])
        rg_dep = Xl @ wnf_frozen
        pf = Xl[:, lfc] @ w_frozen[lfc]

        cfr = (
            float(np.corrcoef(pf, rg_dep)[0, 1])
            if np.std(pf) > 1e-9 and np.std(rg_dep) > 1e-9
            else np.nan
        )
        dev = wa["dev"]
        drms = (
            float(np.sqrt(np.mean(dev ** 2)))
            if dev is not None and len(dev) == len(h)
            else np.nan
        )
        rat = (
            float(np.sqrt(np.mean(pf ** 2)) / np.sqrt(np.mean(rg_dep ** 2)))
            if np.std(rg_dep) > 1e-9
            else np.nan
        )

        fcg = sp["fcg"]
        fcg2 = sp["fcg2"]
        fcd = sp["fcd"]
        fcg3 = sp["fcg3"]
        fcr = sp["fcr"]
        fcg4 = sp["fcg4"]
        fcr2 = sp["fcr2"]
        fcg4u = sp["fcg4u"]
        fcg5 = sp["fcg5"]

        fc1 = np.isfinite(cfr) and cfr < fcg
        fc2 = (
            np.isfinite(cfr)
            and cfr < fcg2
            and np.isfinite(drms)
            and drms > fcd
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
        fired = bool(
            fcg != 0.0 and np.isfinite(cfr) and (fc1 or fc2 or fc3 or fc4 or fc5)
        )
        ridge = rg_dep + (0.0 if fired else pf)
        if sp["demean"] > 0.0:
            ridge = ridge - float(np.mean(ridge))

        # Horizon taper (matches gps_web live solve): ramp up over
        # [t0,t1], unity [t1,t2], ramp down [t2,t3] — corners replayed
        # from the sealed manifest when available (the deployed
        # .env.local values 0/225/5400/9900, NOT the 7200s-zeroing
        # legacy default).
        if sp["taper"] > 0.0:
            t0, t1, t2, t3 = sp["t0"], sp["t1"], sp["t2"], sp["t3"]
            up = np.clip((h - t0) / max(t1 - t0, 1.0), 0.0, 1.0)
            dn = np.clip((t3 - h) / max(t3 - t2, 1.0), 0.0, 1.0)
            wv = np.minimum(up, dn)
        else:
            wv = 1.0
        blend = sp["blend"]
        stack_unc = wa["stack_unc"]
        cap = sp["cap"]
        dep_frozen = np.clip(
            stack_unc + blend * wv * (ridge - stack_unc), -cap, cap
        )

        pr = float(np.sqrt(np.mean(r ** 2)))
        fr_rmse = float(np.sqrt(np.mean((r + dep_frozen) ** 2)))
        out = {
            "frozen_seal_rmse_m": round(fr_rmse, 4),
            "frozen_seal_vs_persist_pct": round(
                (1.0 - fr_rmse / pr) * 100.0, 2
            ),
        }
        # Strict-future slice: only epochs t > seal_t_end whose wall-clock
        # truth could not have entered ANY training member of the sealed fit.
        if seal_t_end is not None:
            m_fut = t_ho > float(seal_t_end)
            out["n_future_epochs"] = int(np.sum(m_fut))
            if int(np.sum(m_fut)) >= 120:
                pr_f = float(np.sqrt(np.mean(r[m_fut] ** 2)))
                rf_f = float(
                    np.sqrt(np.mean((r[m_fut] + dep_frozen[m_fut]) ** 2))
                )
                out["frozen_seal_future_rmse_m"] = round(rf_f, 4)
                out["frozen_seal_future_vs_persist_pct"] = round(
                    (1.0 - rf_f / pr_f) * 100.0, 2
                )
        return out
    except Exception:
        return None


def _score_frozen_on_pkl(
    pkl_path: Path,
    keys_u: list[str],
    w_frozen: np.ndarray,
    wnf_frozen: np.ndarray,
    seal_t_end: float | None = None,
    params: dict | None = None,
) -> dict | None:
    try:
        d = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return None
    return _score_frozen_core(
        window_scoring_arrays(d),
        keys_u, w_frozen, wnf_frozen,
        seal_t_end=seal_t_end, params=params,
    )


def verify_envelope(manifest_path: Path | None = None) -> dict:
    if manifest_path is None:
        manifest_path = ENVELOPE_DIR / "latest_envelope.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Envelope manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    npz_path = ENVELOPE_DIR / manifest["weights_file"]
    actual_sha = _sha256_file(npz_path)
    hash_ok = actual_sha == manifest["weights_sha256"]

    with np.load(npz_path, allow_pickle=False) as z:
        keys_u = [str(k) for k in z["keys"]]
        w_frozen = z["w_frozen"].astype(np.float64)
        wnf_frozen = z["wnf_frozen"].astype(np.float64)

    last_pre_wid = int(manifest["last_window_before_seal"])
    # The scoring transform is committed with the weights: envelopes sealed
    # after 2026-10-28 carry a full scoring_params block which is replayed
    # verbatim.  Legacy envelopes fall back to the deployed env (flagged).
    sealed_params = manifest.get("scoring_params")
    params_eff = _effective_scoring_params(sealed_params)
    env_now = _scoring_params_from_env()
    env_mismatches = (
        {
            k: {"sealed": params_eff[k], "deployed_env": env_now[k]}
            for k in params_eff
            if params_eff[k] != env_now[k]
        }
        if sealed_params is not None
        else {}
    )
    # Wall-clock end of the last pre-seal training window's holdout: epochs
    # strictly after this timestamp never appeared in any sealed training fit.
    seal_t_end = None
    cut_pkl = RESULTS_DIR / f"raw_window_{last_pre_wid}.pkl"
    if cut_pkl.exists():
        try:
            _d = pickle.loads(cut_pkl.read_bytes())
            seal_t_end = float(np.asarray(_d["t_ho"], float)[-1])
            del _d
        except Exception:
            seal_t_end = None
    jpaths = sorted(
        RESULTS_DIR.glob("last_result_window_*.json"), key=lambda p: p.name
    )
    post_windows = []
    for jp in jpaths:
        wid = _window_id(jp)
        if wid <= last_pre_wid:
            continue
        row = _summarize_window_json(jp)
        if row is None:
            continue
        row["lead_s_after_seal_cutoff"] = wid - last_pre_wid
        pkl_p = RESULTS_DIR / f"raw_window_{wid}.pkl"
        if pkl_p.exists():
            fr = _score_frozen_on_pkl(
                pkl_p, keys_u, w_frozen, wnf_frozen, seal_t_end=seal_t_end,
                params=sealed_params,
            )
            if fr is not None:
                row.update(fr)
        post_windows.append(row)

    report = {
        "verified_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "envelope_id": manifest["envelope_id"],
        "seal_utc": manifest["seal_utc"],
        "last_window_before_seal": last_pre_wid,
        "weights_sha256_expected": manifest["weights_sha256"],
        "weights_sha256_actual": actual_sha,
        "cryptographic_integrity_ok": hash_ok,
        "scoring_transform_source": (
            "sealed_manifest" if sealed_params is not None
            else "deployed_env_legacy_manifest"
        ),
        "scoring_params_used": params_eff,
        "legacy_manifest_taper_corners_s": (
            None if sealed_params is not None
            else (manifest.get("hyperparameters") or {}).get(
                "taper_corners_s"
            )
        ),
        "deployed_env_differs_from_sealed": env_mismatches,
        "n_post_seal_windows": len(post_windows),
        "post_seal_windows": post_windows,
    }
    if post_windows:
        report["post_seal_summary"] = {
            "mean_persist_rmse_m": round(
                float(np.mean([w["persist_rmse_m"] for w in post_windows])), 4
            ),
            "mean_live_causal_rmse_m": round(
                float(np.mean([w["live_causal_rmse_m"] for w in post_windows])),
                4,
            ),
            "mean_live_causal_vs_persist_pct": round(
                float(
                    np.mean(
                        [w["live_causal_vs_persist_pct"] for w in post_windows]
                    )
                ),
                2,
            ),
            "mean_live_ramp_vs_persist_pct": round(
                float(
                    np.mean(
                        [w["live_ramp_vs_persist_pct"] for w in post_windows]
                    )
                ),
                2,
            ),
            "mean_frozen_seal_vs_persist_pct": round(
                float(
                    np.mean(
                        [
                            w["frozen_seal_vs_persist_pct"]
                            for w in post_windows
                            if "frozen_seal_vs_persist_pct" in w
                        ]
                    )
                ),
                2,
            )
            if any("frozen_seal_vs_persist_pct" in w for w in post_windows)
            else None,
            "mean_frozen_seal_future_vs_persist_pct": round(
                float(
                    np.mean(
                        [
                            w["frozen_seal_future_vs_persist_pct"]
                            for w in post_windows
                            if "frozen_seal_future_vs_persist_pct" in w
                        ]
                    )
                ),
                2,
            )
            if any(
                "frozen_seal_future_vs_persist_pct" in w for w in post_windows
            )
            else None,
            "frozen_seal_win_count": sum(
                1
                for w in post_windows
                if w.get("frozen_seal_vs_persist_pct", -999.0) > 0.0
            ),
            "causal_win_count": sum(
                1
                for w in post_windows
                if w["live_causal_vs_persist_pct"] > 0.0
            ),
        }

    out_log = ENVELOPE_DIR / "verification_log.json"
    out_log.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_report(report: dict, manifest: dict | None = None) -> None:
    print(
        f"=== SEALED ENVELOPE: {report['envelope_id']} "
        f"(sealed {report['seal_utc']}) ==="
    )
    print(
        f"SHA-256 integrity: "
        f"{'PASS' if report['cryptographic_integrity_ok'] else 'FAIL'} "
        f"({report['weights_sha256_actual'][:16]}...)"
    )
    print(
        f"Cutoff window before seal: {report['last_window_before_seal']} | "
        f"Post-seal blind windows scored: {report['n_post_seal_windows']}"
    )
    if manifest and manifest.get("pre_seal_trailing_10_windows"):
        pre = manifest["pre_seal_trailing_10_windows"]
        print(
            f"\n--- Pre-seal trailing {len(pre)} windows (reference baseline) ---"
        )
        print(
            f"{'window':>12s} {'persist_m':>10s} {'causal_m':>10s} "
            f"{'causal_%':>9s} {'ramp_%':>8s} {'guarded_%':>10s}"
        )
        for w in pre:
            print(
                f"{w['window_id']:12d} {w['persist_rmse_m']:10.4f} "
                f"{w['live_causal_rmse_m']:10.4f} "
                f"{w['live_causal_vs_persist_pct']:+8.2f}% "
                f"{w['live_ramp_vs_persist_pct']:+7.2f}% "
                f"{w['guarded_oracle_vs_persist_pct']:+9.2f}%"
            )

    post = report.get("post_seal_windows") or []
    if post:
        print(
            f"\n--- Post-seal strictly ex-ante windows (n={len(post)}) ---"
        )
        print(
            f"{'window':>12s} {'lead_min':>9s} {'persist_m':>10s} "
            f"{'frozen_m':>10s} {'frozen_%':>9s} {'fut_%':>8s} "
            f"{'causal_m':>10s} {'causal_%':>9s} {'ramp_%':>8s}"
        )
        for w in post:
            fm = w.get("frozen_seal_rmse_m")
            fp = w.get("frozen_seal_vs_persist_pct")
            fut = w.get("frozen_seal_future_vs_persist_pct")
            lead = w.get("lead_s_after_seal_cutoff")
            fm_s = f"{fm:10.4f}" if fm is not None else f"{'n/a':>10s}"
            fp_s = f"{fp:+8.2f}%" if fp is not None else f"{'n/a':>9s}"
            fut_s = f"{fut:+7.2f}%" if fut is not None else f"{'n/a':>8s}"
            ld_s = (
                f"{lead / 60.0:9.1f}" if lead is not None else f"{'n/a':>9s}"
            )
            print(
                f"{w['window_id']:12d} {ld_s} {w['persist_rmse_m']:10.4f} "
                f"{fm_s} {fp_s} {fut_s} {w['live_causal_rmse_m']:10.4f} "
                f"{w['live_causal_vs_persist_pct']:+8.2f}% "
                f"{w['live_ramp_vs_persist_pct']:+7.2f}%"
            )
        s = report["post_seal_summary"]
        print(
            f"Summary: causal wins {s['causal_win_count']}/{len(post)} | "
            f"frozen wins {s.get('frozen_seal_win_count')}/{len(post)} | "
            f"mean causal {s['mean_live_causal_vs_persist_pct']:+.2f}% | "
            f"mean ramp {s['mean_live_ramp_vs_persist_pct']:+.2f}% | "
            f"mean frozen-at-seal {s.get('mean_frozen_seal_vs_persist_pct')} | "
            f"mean frozen strict-future "
            f"{s.get('mean_frozen_seal_future_vs_persist_pct')}"
        )


def _anchor_record(manifest: dict) -> dict:
    """Append a hash-chained provenance record to anchors.log.

    Each record carries the envelope digest, the UTC anchor time, and
    anchor_sha256 = SHA256(prev_anchor_sha256 + weights_sha256 + utc) —
    an append-only chain: editing or deleting any earlier record breaks
    every later link. The log is the externally publishable artefact
    (e.g. commit to a public repo / timestamping service at seal time).
    """
    log_path = ENVELOPE_DIR / "anchors.log"
    prev = "GENESIS"
    if log_path.exists():
        try:
            last = json.loads(
                log_path.read_text().strip().splitlines()[-1]
            )
            prev = last.get("anchor_sha256", "GENESIS")
        except Exception:
            pass
    utc = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    wsha = manifest.get("weights_sha256", "")
    rec = {
        "utc": utc,
        "envelope_id": manifest.get("envelope_id"),
        "seal_utc": manifest.get("seal_utc"),
        "last_window_before_seal": manifest.get(
            "last_window_before_seal"
        ),
        "weights_sha256": wsha,
        "prev_anchor_sha256": prev,
    }
    rec["anchor_sha256"] = hashlib.sha256(
        (prev + wsha + utc).encode()
    ).hexdigest()
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def verify_all_envelopes() -> list[dict]:
    """Verify EVERY sealed envelope against its post-seal windows —
    yields the empirical frozen-weight lead-decay ladder."""
    reports = []
    for mp in sorted(ENVELOPE_DIR.glob("envelope_*.json"),
                     key=lambda p: p.name):
        if mp.name == "latest_envelope.json":
            continue
        try:
            manifest = json.loads(mp.read_text())
            rep = verify_envelope(manifest_path=mp)
            rep["_manifest"] = manifest
            reports.append(rep)
        except Exception:
            continue
    return reports


def _print_ladder(reports: list[dict]) -> None:
    print("\n=== SEALED-ENVELOPE LEAD LADDER (all envelopes) ===")
    print(f"{'envelope':24s} {'seal_utc':20s} {'n_post':>6s} "
          f"{'lead_min':>9s} {'lead_max':>9s} {'frozen_%':>9s} {'wins':>6s}")
    for rep in reports:
        m = rep.get("_manifest", {})
        post = rep.get("post_seal_windows", [])
        if not post:
            continue
        leads = [p.get("lead_min", (p.get("lead_s_after_seal_cutoff", 0)
                                    or 0) / 60.0) for p in post]
        pcts = [p.get("frozen_seal_vs_persist_pct") for p in post
                if p.get("frozen_seal_vs_persist_pct") is not None]
        if not pcts:
            continue
        wins = sum(1 for x in pcts if x > 0)
        print(f"{str(m.get('envelope_id'))[:24]:24s} "
              f"{str(m.get('seal_utc'))[:20]:20s} "
              f"{len(post):6d} {min(leads):9.1f} {max(leads):9.1f} "
              f"{float(np.mean(pcts)):9.2f} {wins}/{len(pcts):4d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seal",
        action="store_true",
        help="Create a new SHA-256-sealed causal weight envelope now",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Verify the latest sealed envelope against post-seal windows",
    )
    ap.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Poll interval in seconds to continuously score new post-seal windows",
    )
    ap.add_argument(
        "--backtest",
        type=int,
        default=0,
        help="Seal an envelope N windows ago and verify across the N subsequent windows",
    )
    ap.add_argument(
        "--seal-every",
        action="store_true",
        help="In watch mode: seal a fresh envelope each time a new window lands",
    )
    ap.add_argument(
        "--verify-all",
        action="store_true",
        help="Verify every sealed envelope -> empirical lead-decay ladder",
    )
    ap.add_argument(
        "--anchor",
        action="store_true",
        help="Append the seal's digest to the hash-chained anchors.log",
    )
    ap.add_argument("--tag", type=str, default=None, help="Optional seal tag")
    args = ap.parse_args()

    if args.backtest > 0:
        mpath = seal_envelope(
            tag=args.tag or f"backtest_{args.backtest}w",
            backtest_n=args.backtest,
            update_latest=False,
        )
        manifest = json.loads(mpath.read_text())
        rep = verify_envelope(manifest_path=mpath)
        _print_report(rep, manifest=manifest)
        return 0

    if args.verify_all:
        _print_ladder(verify_all_envelopes())
        return 0

    manifest = None
    if args.seal or not (ENVELOPE_DIR / "latest_envelope.json").exists():
        mpath = seal_envelope(tag=args.tag)
        manifest = json.loads(mpath.read_text())
        if args.anchor:
            rec = _anchor_record(manifest)
            print(f"  anchored: {rec['anchor_sha256'][:20]}... -> anchors.log")

    if args.verify or args.seal:
        if manifest is None:
            manifest = json.loads(
                (ENVELOPE_DIR / "latest_envelope.json").read_text()
            )
        rep = verify_envelope()
        _print_report(rep, manifest=manifest)

    if args.watch > 0:
        seen = -1
        while True:
            rep = verify_envelope()
            n_post = rep["n_post_seal_windows"]
            if n_post != seen:
                _print_report(rep, manifest=None)
                seen = n_post
                sys.stdout.flush()
                if args.seal_every and n_post > 0:
                    mpath = seal_envelope()
                    manifest = json.loads(mpath.read_text())
                    if args.anchor:
                        _anchor_record(manifest)
            time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
