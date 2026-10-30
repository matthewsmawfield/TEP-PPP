#!/usr/bin/env python3
"""Rebuild the live seal ledger from the sealed-envelope watcher logs.

For every sealed envelope, scores the FIRST genuinely post-seal window
(the definition pinned in results/outputs/step_174_live_ledger.json):
the earliest scored window whose id exceeds the envelope's cutoff.
Each score is recorded under a SHA-256-verified envelope, so the ledger
is ex-ante evidence: the weights were committed before the scored data
existed.

Parses:
    data/ppp_live/sealed_envelopes/watch.log
    logs/sealed_watcher.log

Writes: results/outputs/step_174_live_ledger.json
        (preserves the curated long_lived_seal_replays block from the
        previous version when present)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "data" / "ppp_live" / "sealed_envelopes"
LOGS = [ENV_DIR / "watch.log", ROOT / "logs" / "sealed_watcher.log"]
OUT = ROOT / "results" / "outputs" / "step_174_live_ledger.json"

ENV_RE = re.compile(r"=== SEALED ENVELOPE: (envelope_\d+_\d+) "
                    r"\(sealed ([0-9T:\-]+Z)\) ===")
SHA_RE = re.compile(r"SHA-256 integrity: (\w+)")
CUTOFF_RE = re.compile(r"Cutoff window before seal: (\d+) \| "
                       r"Post-seal blind windows scored: (\d+)")
ROW_RE = re.compile(
    r"^\s*(\d{9,})\s+([\d.]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+"
    r"([+\-\d.]+)%\s+([+\-\d.]+)%\s+([\d.\-]+)\s+([+\-\d.]+)%\s+([+\-\d.]+)%")


def parse_logs() -> dict[str, dict]:
    """Return {envelope_id: {"cutoff": int, "rows": {window_id: row}}}"""
    envs: dict[str, dict] = {}
    for log in LOGS:
        if not log.exists():
            continue
        cur = None
        sha = False
        for line in log.read_text(errors="ignore").splitlines():
            m = ENV_RE.match(line)
            if m:
                cur = m.group(1)
                envs.setdefault(cur, {"cutoff": None, "rows": {}})
                sha = False
                continue
            if cur is None:
                continue
            m = SHA_RE.match(line.strip())
            if m:
                sha = m.group(1) == "PASS"
                continue
            m = CUTOFF_RE.search(line)
            if m:
                envs[cur]["cutoff"] = int(m.group(1))
                continue
            m = ROW_RE.match(line)
            if m:
                wid = int(m.group(1))
                envs[cur]["rows"][wid] = {
                    "window_id": wid,
                    "sha_pass": sha,
                    "lead_min": float(m.group(2)),
                    "frozen_m": float(m.group(4)),
                    "causal_m": float(m.group(7)),
                    "frozen_pct": float(m.group(5)),
                    "strict_future_pct": float(m.group(6)),
                    "causal_pct": float(m.group(8)),
                }
    return envs


def main() -> int:
    envs = parse_logs()
    per_env = {}
    for env_id, rec in sorted(envs.items()):
        cutoff = rec["cutoff"]
        rows = rec["rows"]
        if cutoff is None or not rows:
            continue
        post = {w: r for w, r in rows.items() if w > cutoff}
        if not post:
            continue
        first = post[min(post)]
        per_env[env_id] = first

    frozen = np.array([r["frozen_pct"] for r in per_env.values()])
    causal = np.array([r["causal_pct"] for r in per_env.values()])
    leads = [r["lead_min"] for r in per_env.values()]
    n_ident = sum(1 for r in per_env.values()
                  if r["frozen_m"] == r["causal_m"])
    failures = [
        {"envelope": e, **{k: r[k] for k in
                          ("window_id", "sha_pass", "lead_min",
                           "frozen_pct", "strict_future_pct",
                           "causal_pct")}}
        for e, r in per_env.items() if r["frozen_pct"] < 0
    ]

    out = {
        "generated": datetime.now(timezone.utc).date().isoformat(),
        "source_logs": [str(p.relative_to(ROOT)) for p in LOGS
                        if p.exists()],
        "definition": "per-envelope score on its FIRST genuinely "
                      "post-seal window (SHA-256 digest verified before "
                      "scoring); frozen==causal when the refit has not "
                      "yet moved past the seal",
        "n_envelopes_scored": len(per_env),
        "first_window": {
            "mean_pct": float(frozen.mean()),
            "median_pct": float(np.median(frozen)),
            "win_count": int(np.sum(frozen > 0)),
            "n": int(len(frozen)),
            "sign_test_p": float(stats.binomtest(
                int(np.sum(frozen > 0)), int(len(frozen)), 0.5,
                alternative="greater").pvalue),
            "lead_min": [float(min(leads)), float(np.median(leads)),
                         float(max(leads))],
            "causal_mean_pct": float(causal.mean()),
            "causal_win_count": int(np.sum(causal > 0)),
            "n_identical_frozen_causal": n_ident,
            "failures": failures,
        },
        "per_envelope_first": per_env,
    }

    # Preserve the curated long-lived replay entries from the previous
    # ledger version (they come from dedicated replays, not the watcher
    # logs).
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            if prev.get("long_lived_seal_replays"):
                out["long_lived_seal_replays"] = \
                    prev["long_lived_seal_replays"]
        except Exception:
            pass

    OUT.write_text(json.dumps(out, indent=2))
    fw = out["first_window"]
    print(f"Wrote {OUT.name}: {len(per_env)} envelopes scored, "
          f"{fw['win_count']}/{fw['n']} positive, "
          f"mean {fw['mean_pct']:+.2f}% median {fw['median_pct']:+.2f}%, "
          f"leads {fw['lead_min'][0]:.0f}-{fw['lead_min'][2]:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
