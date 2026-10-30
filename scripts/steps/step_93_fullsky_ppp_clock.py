#!/usr/bin/env python3
"""Held-out production-PPP leftover clocks (full sky).

IGS/CODE receiver clocks are the estimated leftover time after a good
multi-GNSS PPP solve.  Geometry already ate what it can; whatever spatial
structure remains is the clock field.  This step never scores 3-D position.

Predictor: physics-fixed TEP Gaussian, lambda=4200 km, nugget=0.35, on the
RAW leftover clock (no full-day detrend, no network-mean removal).
Controls: persistence (last value) and AR1 on the target only.

Output: results/outputs/step_93_fullsky_ppp_clock.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td  # noqa: E402
from scripts.lib import tep_kernel as tk  # noqa: E402

OUT = ROOT / "results" / "outputs" / "step_93_fullsky_ppp_clock.json"
C_M_PER_NS = 0.299792458
LAMBDA_KM = 4200.0
NUGGET = 0.35
DT_S = 300.0
N_DAYS = int(os.environ.get("TEP_S93_DAYS", "8"))
STRIDE = int(os.environ.get("TEP_S93_STRIDE", "4"))  # 20 min on 300 s grid


def _ar1(pre, n_steps):
    if len(pre) < 5:
        return float(pre[-1])
    x = pre - np.mean(pre)
    den = np.sum(x[:-1] ** 2) + 1e-12
    phi = float(np.clip(np.sum(x[1:] * x[:-1]) / den, -0.999, 0.999))
    return float(np.mean(pre) + (pre[-1] - np.mean(pre)) * phi ** max(n_steps, 1))


def main() -> int:
    dates = td.available_dates("precise", 2024)
    days = [dates[i] for i in np.linspace(0, len(dates) - 1, N_DAYS).astype(int)]
    params = tk.KernelParams(lambda_km=LAMBDA_KM, nugget=NUGGET,
                             k_neighbours=25, form="gaussian")
    pers, ar1s, teps = [], [], []
    n_epochs = 0
    for tag in days:
        s = td.load_series("precise", tag, min_valid_frac=0.0, max_jumps=10_000)
        if not s or len(s) < 12:
            continue
        names = list(s)
        # RAW leftover PPP clocks in metres.  Do not detrend the day and do
        # not subtract the network mean — that is the field.
        raw = np.column_stack([s[n]["raw"] for n in names]).astype(float) * C_M_PER_NS
        t = s[names[0]]["t"].astype(float)
        D = np.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                D[i, j] = D[j, i] = td.great_circle_km(
                    s[names[i]]["station"], s[names[j]]["station"])
        for k in range(STRIDE, len(t), STRIDE):
            y = raw[k]
            y_prev = raw[k - 1]
            good = np.isfinite(y) & np.isfinite(y_prev)
            if good.sum() < 10:
                continue
            idx = np.where(good)[0]
            # Absolute leftover clocks are not in a common frame (each
            # receiver has its own km-scale bias).  Krige the causal
            # increment — the field change — then add it to persistence.
            dy = y - y_prev
            for ti in idx:
                refs = [j for j in idx if j != ti and abs(float(dy[j])) < 15.0]
                if len(refs) < 8:
                    continue
                truth = float(y[ti])
                pers_hat = float(y_prev[ti])
                hist = raw[:k, ti]
                hist = hist[np.isfinite(hist)]
                ar1_hat = _ar1(hist[-24:], 1) if len(hist) >= 8 else pers_hat
                d_rr = D[np.ix_(refs, refs)]
                d_t = D[ti, refs]
                w = tk.kriging_weights(d_rr, d_t, params)
                dhat = float(dy[refs] @ w)
                if not np.isfinite(dhat) or abs(dhat) > 15.0:
                    continue
                tep_hat = pers_hat + dhat
                pers.append(pers_hat - truth)
                ar1s.append(ar1_hat - truth)
                teps.append(tep_hat - truth)
                n_epochs += 1
    pers, ar1s, teps = map(np.asarray, (pers, ar1s, teps))

    def rms(e):
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else None

    def p50(e):
        return float(np.median(np.abs(e))) if len(e) else None

    rp, ra, rt = rms(pers), rms(ar1s), rms(teps)
    cap = 50.0
    def rms_clip(e):
        e = e[np.abs(e) < cap]
        return rms(e)
    out = {
        "step": 93,
        "title": "Held-out leftover PPP clocks at full sky (no 3-D position)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": "precise IGS/MGEX receiver clocks = leftover time after production PPP",
        "kernel": {"form": "gaussian", "lambda_km": LAMBDA_KM, "nugget": NUGGET},
        "preprocessing": ("raw leftover PPP clocks; per-station bias is not "
                          "kriged. TEP predicts the causal increment from "
                          "other stations and adds it to the target hold. "
                          "No full-day detrend, no network-mean removal."),
        "n_days": N_DAYS,
        "n_predictions": int(n_epochs),
        "rms_m": {"persistence": rp, "ar1": ra, "tep": rt},
        "rms_clip50_m": {"persistence": rms_clip(pers), "ar1": rms_clip(ar1s),
                         "tep": rms_clip(teps)},
        "p50_abs_m": {"persistence": p50(pers), "ar1": p50(ar1s), "tep": p50(teps)},
        "tep_vs_persistence_rms_pct": None if not rp else 100.0 * (1.0 - rt / rp),
        "tep_vs_ar1_rms_pct": None if not ra else 100.0 * (1.0 - rt / ra),
        "tep_vs_persistence_p50_pct": None if not p50(pers) else 100.0 * (
            1.0 - p50(teps) / p50(pers)),
        "note": ("If TEP beats persistence/AR1 here, 8+ satellites did not absorb "
                 "the field: the leftover PPP clock still has TEP spatial structure. "
                 "Position RMS at k>=4 is not this test."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "n_predictions", "rms_m", "p50_abs_m",
        "tep_vs_persistence_rms_pct", "tep_vs_ar1_rms_pct",
        "tep_vs_persistence_p50_pct")}, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
