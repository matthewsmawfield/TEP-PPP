#!/usr/bin/env python3
"""Epoch-sharing audit for the pinned Step 174 corpus.

Measures, for each 1-step sealed eval window, the fraction of its
holdout epochs that coincide (15 s coincidence gate) with (a) the
newest ring member's target epochs and (b) the pooled K=16 ring
member target epochs.  This quantifies the retrodiction channel that
inflates the sealed walk-forward headline relative to the
zero-overlap clean-origin arm.

Writes results/outputs/step_174_epoch_share.json.
"""

import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "ppp_live" / "results"
OUT = REPO / "results" / "outputs" / "step_174_epoch_share.json"

CORPUS_END_WID = 1793132392
N_CORPUS = 40
K_RING = 16
GATE_S = 15.0


def _wid(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def main() -> None:
    pkls = sorted(
        [p for p in RESULTS.glob("raw_window_*.pkl")
         if _wid(p) <= CORPUS_END_WID],
        key=lambda p: p.name,
    )[-N_CORPUS:]
    tho = {}
    for p in pkls:
        d = pickle.loads(p.read_bytes())
        tho[_wid(p)] = np.asarray(d["t_ho"], float)
        del d
    wids = [_wid(p) for p in pkls]
    newest, pooled = [], []
    for j in range(K_RING, len(wids)):
        e = np.round(tho[wids[j]] / GATE_S)
        newest.append(
            float(np.isin(e, np.round(tho[wids[j - 1]] / GATE_S)).mean())
            * 100.0
        )
        ring = np.concatenate(
            [tho[wids[i]] for i in range(j - K_RING, j)]
        )
        pooled.append(
            float(np.isin(e, np.round(ring / GATE_S)).mean()) * 100.0
        )
    out = {
        "description": (
            "Epoch-sharing fraction for the pinned step_174 corpus: "
            "fraction of each 1-step eval window's holdout epochs "
            "coinciding (15 s gate) with the newest ring member's "
            "target epochs, and with the pooled ring (K=16) member "
            "targets. Quantifies the retrodiction channel in the "
            "sealed headline."
        ),
        "corpus_end_wid": CORPUS_END_WID,
        "n_members": len(pkls),
        "n_eval_seals": len(newest),
        "newest_member_share_pct": {
            "mean": round(float(np.mean(newest)), 2),
            "min": round(min(newest), 2),
            "max": round(max(newest), 2),
        },
        "pooled_ring_share_pct": {
            "mean": round(float(np.mean(pooled)), 2),
            "min": round(min(pooled), 2),
            "max": round(max(pooled), 2),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pooled_ring_share_pct"], indent=2))


if __name__ == "__main__":
    main()
