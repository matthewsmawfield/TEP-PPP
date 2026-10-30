#!/usr/bin/env python3
"""Live Skill Observatory audit snapshot.

Recomputes the accumulating walk-forward audit quoted in the manuscript
("the continuous N-station live observatory reduces two-hour clock
holdover RMSE by X% across M scored windows") directly from the recorded
per-window result files:

    data/ppp_live/results/last_result_window_*.json
      -> clock_prediction.specificity.tep_vs_persist_pct
         (deployed causal predictor vs persistence, 2 h holdover)

Four populations are reported:

  * all_windows    -- every recorded window with a specificity score.
  * valid_only     -- the subset where the per-window alpha fit was
    non-degenerate (specificity.valid == True; when alpha collapses to 0
    the TEP arm IS persistence and the controls are uninformative, per
    the recorded note).
  * epoch_disjoint -- a greedy independence subset: windows accepted only
    if their start is >= 14,400 s (the 4 h capture span) after the last
    accepted window, so no two scored windows share an epoch.  This is
    the overlap-free replication count a sign test can honestly use.
  * post_warmup    -- windows starting >= 24 h after the first scored
    window (deployment transient excluded).

The span is the wall-clock distance between the first and last scored
window timestamps.  The sign test is the one-sided binomial test of
positive windows.

Writes: results/outputs/live_observatory_audit.json
Re-run any time: the observatory accumulates, so re-running refreshes
the pinned snapshot.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "ppp_live" / "results"
OUT = ROOT / "results" / "outputs" / "live_observatory_audit.json"


def collect() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("last_result_window_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        sp = (d.get("clock_prediction") or {}).get("specificity") or {}
        v = sp.get("tep_vs_persist_pct")
        if v is None:
            continue
        rows.append({
            "window": f.stem,
            "timestamp": d.get("timestamp"),
            "pct": float(v),
            "valid": bool(sp.get("valid")),
        })
    rows.sort(key=lambda r: r["timestamp"] or "")
    return rows


def summarise(rows: list[dict]) -> dict:
    vals = np.array([r["pct"] for r in rows])
    n = len(vals)
    k = int(np.sum(vals > 0))
    ts = [r["timestamp"] for r in rows if r["timestamp"]]
    span_h = None
    if len(ts) >= 2:
        t0 = datetime.fromisoformat(ts[0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(ts[-1].replace("Z", "+00:00"))
        span_h = (t1 - t0).total_seconds() / 3600.0
    return {
        "n_windows": n,
        "n_positive": k,
        "frac_positive": k / n if n else None,
        "mean_pct": float(vals.mean()),
        "median_pct": float(np.median(vals)),
        "min_pct": float(vals.min()),
        "max_pct": float(vals.max()),
        "sign_test_p_one_sided": float(
            stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        ),
        "wall_clock_span_h": span_h,
        "first_window": ts[0] if ts else None,
        "last_window": ts[-1] if ts else None,
    }


def _ts_epoch(r: dict) -> float:
    return datetime.fromisoformat(
        r["timestamp"].replace("Z", "+00:00")).timestamp()


def epoch_disjoint(rows: list[dict], span_s: float = 14400.0) -> list[dict]:
    """Greedy overlap-free subset: keep a window only if it starts at
    least span_s after the previously kept window's start."""
    out = []
    last = None
    for r in rows:
        t = _ts_epoch(r)
        if last is None or t - last >= span_s:
            out.append(r)
            last = t
    return out


def post_warmup(rows: list[dict], warmup_s: float = 86400.0) -> list[dict]:
    t0 = _ts_epoch(rows[0])
    return [r for r in rows if _ts_epoch(r) - t0 >= warmup_s]


def main() -> int:
    rows = collect()
    if not rows:
        print("no scored windows found", file=sys.stderr)
        return 1
    audit = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/ppp_live/results/last_result_window_*.json "
                  ":: clock_prediction.specificity.tep_vs_persist_pct",
        "metric": "deployed causal predictor vs persistence, "
                  "2 h clock holdover RMSE improvement (%)",
        "all_windows": summarise(rows),
        "valid_only": summarise([r for r in rows if r["valid"]]),
        "epoch_disjoint": summarise(epoch_disjoint(rows)),
        "post_warmup": summarise(post_warmup(rows)),
        "epoch_disjoint_rule": "greedy keep if start >= last kept start "
                               "+ 14,400 s (the 4 h capture span)",
        "n_alpha_degenerate": sum(1 for r in rows if not r["valid"]),
    }
    OUT.write_text(json.dumps(audit, indent=2))
    a = audit["all_windows"]
    print(f"Wrote {OUT.name}: {a['n_windows']} windows, "
          f"mean {a['mean_pct']:+.2f}%, median {a['median_pct']:+.2f}%, "
          f"{a['n_positive']}/{a['n_windows']} positive, "
          f"p={a['sign_test_p_one_sided']:.2e}, "
          f"span {a['wall_clock_span_h']:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
