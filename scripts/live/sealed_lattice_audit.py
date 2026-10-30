#!/usr/bin/env python3
"""Sealed-envelope lattice audit.

Scores EVERY sealed envelope against EVERY raw window archived strictly
after that envelope's seal cutoff — a triangular envelope x window
lattice — in one pass that loads each raw_window_*.pkl exactly once
(window -> envelopes order).  This is the full-sample version of the
per-envelope ``--verify-all`` ladder (which re-loads every archive per
envelope) and of the first-window-only live seal ledger.

Every score uses the identical frozen-weight scoring path as the live
verifier (``_score_frozen_core`` on ``window_scoring_arrays`` from
scripts/live/sealed_envelope_verifier.py), the weights' SHA-256 digest
is re-verified against the signed manifest before any use, and each
envelope's sealed ``scoring_params`` block is replayed verbatim.  Every
scored pair is ex-ante: the weight vector was committed before the
window's data existed.  The ``*_backtest_*`` envelope is excluded — its
weights are causally constructed but the seal itself was not committed
before the scored data.

Outputs (results/outputs/sealed_lattice_audit.json):
  * per_envelope: n post-seal windows, mean/wins, lead range
  * lead_ladder: pooled skill by forecast-lead bin (the frozen-weight
    coherence curve measured on ~N seals x M windows rather than the
    24-seal corpus ladder)
  * strict_future: the same ladder on the epoch slice strictly
    posterior to each seal's training data (t_ho > seal_t_end)
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.live.sealed_envelope_verifier import (  # noqa: E402
    ENVELOPE_DIR,
    RESULTS_DIR,
    _score_frozen_core,
    window_scoring_arrays,
)

OUT = ROOT / "results" / "outputs" / "sealed_lattice_audit.json"
LEAD_BINS_MIN = [0, 20, 40, 60, 90, 120, 180, 300, 1e9]
LEAD_LABELS = ["<20", "20-40", "40-60", "60-90", "90-120",
               "120-180", "180-300", ">300"]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_envelopes() -> list[dict]:
    envs = []
    for mp in sorted(ENVELOPE_DIR.glob("envelope_*.json")):
        if mp.name == "latest_envelope.json" or "backtest" in mp.name:
            continue
        try:
            man = json.loads(mp.read_text())
            npz = ENVELOPE_DIR / man["weights_file"]
            sha = _sha256_file(npz)
            with np.load(npz, allow_pickle=False) as z:
                envs.append({
                    "envelope_id": man["envelope_id"],
                    "seal_utc": man["seal_utc"],
                    "cutoff": int(man["last_window_before_seal"]),
                    "sha_ok": sha == man["weights_sha256"],
                    "keys_u": [str(k) for k in z["keys"]],
                    "w": z["w_frozen"].astype(np.float64),
                    "wnf": z["wnf_frozen"].astype(np.float64),
                    "params": man.get("scoring_params"),
                })
        except Exception:
            continue
    return envs


def _bin_stats(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean_pct": float(np.mean(vals)),
        "median_pct": float(np.median(vals)),
        "win_count": int(np.sum(vals > 0)),
        "sign_test_p": float(stats.binomtest(
            int(np.sum(vals > 0)), int(len(vals)), 0.5,
            alternative="greater").pvalue),
    }


def main() -> int:
    envs = _load_envelopes()
    n_sha_bad = sum(1 for e in envs if not e["sha_ok"])
    print(f"{len(envs)} envelopes loaded ({n_sha_bad} sha failures)")

    # Group envelopes by identical feature basis so one feature matrix
    # serves every seal in the group (~12 bases vs ~100 seals).
    sig_of = {i: hashlib.sha256(repr(tuple(e["keys_u"])).encode())
              .hexdigest() for i, e in enumerate(envs)}

    by_env = {e["envelope_id"]: [] for e in envs}
    rows = []            # (env_idx, lead_min, pct)
    fut_rows = []        # (env_idx, lead_min, future_pct)
    t_ho_end: dict[int, float] = {}
    n_windows = 0

    for pkl in sorted(RESULTS_DIR.glob("raw_window_*.pkl"),
                      key=lambda p: p.name):
        wid = int(pkl.stem.rsplit("_", 1)[-1])
        elig = [i for i, e in enumerate(envs) if e["cutoff"] < wid]
        try:
            d = pickle.loads(pkl.read_bytes())
        except Exception:
            continue
        if not elig:
            continue
        try:
            wa = window_scoring_arrays(d)
        except Exception:
            continue
        n_windows += 1
        if n_windows % 25 == 0:
            print(f"  {n_windows} windows, {len(rows)} pairs scored",
                  flush=True)
        t_ho_end[wid] = float(wa["t_ho"][-1])
        groups: dict[str, list[int]] = {}
        for i in elig:
            groups.setdefault(sig_of[i], []).append(i)
        for sig, idxs in groups.items():
            e0 = envs[idxs[0]]
            idx = {k: j for j, k in enumerate(e0["keys_u"])}
            Xl = np.zeros((len(wa["h"]), len(e0["keys_u"])), np.float64)
            for k, v in wa["feats"].items():
                j = idx.get(k)
                if j is not None and len(v) == len(wa["h"]):
                    Xl[:, j] = v
            for i in idxs:
                e = envs[i]
                if not e["sha_ok"]:
                    continue
                sc = _score_frozen_core(
                    wa, e["keys_u"], e["w"], e["wnf"],
                    seal_t_end=t_ho_end.get(e["cutoff"]),
                    params=e["params"], Xl=Xl,
                )
                if sc is None:
                    continue
                lead_min = (wid - e["cutoff"]) / 60.0
                pct = sc["frozen_seal_vs_persist_pct"]
                rows.append((i, lead_min, pct))
                by_env[e["envelope_id"]].append(
                    {"window_id": wid, "lead_min": round(lead_min, 1),
                     "pct": pct})
                fp = sc.get("frozen_seal_future_vs_persist_pct")
                if fp is not None:
                    fut_rows.append((i, lead_min, fp))
        del d, wa

    pcts = np.array([r[2] for r in rows])
    leads = np.array([r[1] for r in rows])
    fut_pcts = np.array([r[2] for r in fut_rows])
    fut_leads = np.array([r[1] for r in fut_rows])

    per_envelope = {}
    for e in envs:
        rr = by_env[e["envelope_id"]]
        if not rr:
            continue
        v = np.array([x["pct"] for x in rr])
        l = np.array([x["lead_min"] for x in rr])
        short = v[l < 90.0]
        per_envelope[e["envelope_id"]] = {
            "cutoff_window": e["cutoff"],
            "n_post_windows": int(len(v)),
            "mean_pct": float(np.mean(v)),
            "win_count": int(np.sum(v > 0)),
            "lead_min": [float(l.min()), float(np.median(l)),
                         float(l.max())],
            "n_le90min": int(len(short)),
            "mean_le90min_pct": (float(np.mean(short))
                                 if len(short) else None),
            "windows": rr,
        }

    def ladder(lead_arr, pct_arr):
        out = {}
        for lo, hi, lab in zip(LEAD_BINS_MIN[:-1], LEAD_BINS_MIN[1:],
                               LEAD_LABELS):
            m = (lead_arr >= lo) & (lead_arr < hi)
            out[lab] = _bin_stats(pct_arr[m])
        return out

    # Envelope-level (cluster-honest) summaries: one number per seal.
    # Pair-level sign tests overstate independence (each window is scored
    # by many envelopes, and envelopes share windows); the per-envelope
    # means are the defensible cluster statistic.
    env_means = np.array([v["mean_pct"] for v in per_envelope.values()])
    env_means_all = np.array([x["pct"] for rr in by_env.values()
                              for x in rr]) if rows else np.array([])
    env_le90 = np.array([v["mean_le90min_pct"]
                         for v in per_envelope.values()
                         if v["mean_le90min_pct"] is not None])

    out = {
        "generated": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "definition": "every sealed envelope scored on every raw window "
                      "archived strictly after its seal cutoff; weights "
                      "SHA-256-verified and sealed scoring_params "
                      "replayed verbatim; *_backtest_* envelope "
                      "excluded",
        "n_envelopes": len(envs),
        "n_sha_failures": int(n_sha_bad),
        "n_windows_scored": int(n_windows),
        "n_envelope_window_pairs": int(len(rows)),
        "pooled_pairs": _bin_stats(pcts),
        "lead_ladder_pairs": ladder(leads, pcts),
        "strict_future": {
            "n_pairs": int(len(fut_rows)),
            "pooled_pairs": _bin_stats(fut_pcts),
            "lead_ladder_pairs": ladder(fut_leads, fut_pcts),
        },
        "per_envelope_summary": {
            "n_envelopes_with_post_windows": int(len(per_envelope)),
            "mean_of_env_means_pct": float(env_means.mean())
                if len(env_means) else None,
            "n_env_mean_positive": int(np.sum(env_means > 0)),
            "env_mean_min_pct": float(env_means.min())
                if len(env_means) else None,
            "env_mean_max_pct": float(env_means.max())
                if len(env_means) else None,
            "n_env_le90min": int(len(env_le90)),
            "n_env_le90min_positive": int(np.sum(env_le90 > 0)),
            "mean_env_le90min_pct": float(env_le90.mean())
                if len(env_le90) else None,
            "env_le90min_sign_test_p": float(stats.binomtest(
                int(np.sum(env_le90 > 0)), int(len(env_le90)), 0.5,
                alternative="greater").pvalue) if len(env_le90) else None,
            "pooled_pair_median_pct": float(np.median(env_means_all))
                if len(env_means_all) else None,
        },
        "per_envelope": per_envelope,
    }
    OUT.write_text(json.dumps(out, indent=2))
    pe = out["per_envelope_summary"]
    print(f"Wrote {OUT.name}: {len(rows)} envelope x window pairs across "
          f"{n_windows} windows and {len(per_envelope)} envelopes; pooled "
          f"mean {out['pooled_pairs']['mean_pct']:+.2f}% "
          f"({out['pooled_pairs']['win_count']}/{len(rows)} wins); "
          f"{pe['n_env_mean_positive']}/{len(per_envelope)} envelopes "
          f"positive on all-post-seal mean")
    for lab in LEAD_LABELS:
        s = out["lead_ladder_pairs"][lab]
        if s["n"]:
            print(f"  lead {lab:>7s} min: n={s['n']:5d} "
                  f"mean {s['mean_pct']:+.2f}% wins {s['win_count']}/{s['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
