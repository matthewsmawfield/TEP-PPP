#!/usr/bin/env python3
"""STEP 180 -- Prospective forward-skill preregistration (frozen protocol).

The external review's P1 asks that the clean-origin forecast method be
frozen and prospectively scored on genuinely independent future windows.
This script commits that protocol, hashed, before the scored windows
exist.  The frozen method is exactly the clean-origin arm of Step
174/174b that recovered the positive forward component:

  * Training Grams: rebuilt on the full two-hour FIT half of each member
    (target epochs end strictly before that member's holdout and never
    coincide with any scored epoch).
  * Penalty: causal holdout-Gram lambda selection at seal time (the
    validation set actually available at seal time, in the scored
    regime) -- the rule that produced +2.57% (22/24, p=1.8e-5).
  * Arm: untapered pure TEP correction (the taper was a stale-tail
    transform; Step 174b shows it masks the forward map).
  * Scoring: every live raw_window_*.pkl archived after the commitment
    timestamp, scored at its OWN post-seal leads only -- no
    member-observed epochs, which is the definition of forward skill.
  * Metrics: pure and composite percent improvement vs persistence,
    win fraction, per-lead ladder (<20, 20-40, 40-60, 60-90, 90-120,
    >120 min), plus the strict-future slice.
  * Falsification: all values are reported whatever their sign; a
    mean <= 0 on N >= 30 epoch-disjoint windows refutes the forward
    component claim at the measured cadence.
  * Integrity: envelopes remain SHA-256-sealed and anchor-chained; the
    anchor head is committed to the public repository history, giving
    an externally verifiable timestamp (the reviewer's recommended
    strengthening over locally timestamped hashes alone).

The commitment hash is the SHA-256 of the canonical JSON of `protocol`
(excluding the hash field itself).  Output:
results/outputs/step_180_prospective_preregistration.json
"""

import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "outputs",
                   "step_180_prospective_preregistration.json")


def main():
    protocol = {
        "title": "Prospective epoch-disjoint forward-skill evaluation "
                 "of the clean-origin TEP network prior",
        "version": "v1.0",
        "committed_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "training_grams": (
            "clean-origin: member Grams rebuilt on the full 2-h fit half "
            "of each member window; member target epochs end strictly "
            "before that member's holdout; zero overlap with any scored "
            "epoch at any seal lead"),
        "penalty_selection": (
            "causal holdout-Gram lambda selection at seal time (validation "
            "set available at seal time, in the scored positive-horizon "
            "regime); identical to the rule committed in "
            "step_174_sealed_envelope_prediction.json that produced "
            "+2.57% (22/24, CI [+1.42,+3.85], p=1.8e-5)"),
        "scored_arm": "untapered pure TEP correction vs persistence "
                      "(step_174b arm; taper removed because it damps the "
                      "regime where the forward map is strongest)",
        "scored_data": (
            "every data/ppp_live/results/raw_window_*.pkl archived after "
            "the commitment UTC; each window scored only at leads strictly "
            "posterior to its seal's last training window"),
        "metrics": [
            "pure_pct_improvement_vs_persistence (mean, median, wins)",
            "composite_pct_improvement_vs_persistence",
            "per-lead ladder: <20, 20-40, 40-60, 60-90, 90-120, >120 min",
            "strict-future slice (epochs with zero training coverage)",
            "sign test and day-clustered 95% CI",
        ],
        "falsification_rule": (
            "report all values regardless of sign; mean pure improvement "
            "<= 0 on >= 30 epoch-disjoint windows refutes the forward "
            "component at the deployed cadence"),
        "integrity": (
            "envelopes SHA-256-sealed, anchor-chained "
            "SHA256(prev + weights_sha256 + utc); chain head committed to "
            "public repository history = externally verifiable timestamp"),
        "evidence_base": {
            "clean_origin_retrodiction_pct": -5.63,
            "forward_at_causal_penalty_pct": 2.57,
            "forward_p": 1.8e-5,
            "untapered_forward_pct_at_5e6": 3.39,
            "quasi_static_band_pct": 6.98,
            "coherence_minutes": "~90",
            "sources": [
                "step_174_sealed_envelope_prediction.json",
                "step_174_clean_untapered_arm.json",
                "step_174b_station_permutation_null.json",
                "sealed_lattice_audit.json",
            ],
        },
    }
    canon = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canon.encode()).hexdigest()
    out = {"step": 180, "kind": "preregistration",
           "commitment_sha256": sha,
           "protocol": protocol,
           "note": "spec body hashed verbatim (canonical JSON); scoring "
                   "against this commitment is prospective from "
                   "committed_utc"}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"commitment {sha[:16]}...  saved {OUT}")


if __name__ == "__main__":
    main()
