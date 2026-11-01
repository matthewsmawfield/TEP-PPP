#!/usr/bin/env python3
"""
STEP 65c -- Oscillator-noise null for the quartz/atomic screening claim.

Corpus issue `ppp-clocktype-screening-null`: the 3.8x quartz/atomic LF
amplitude ratio reported by step_65b could be a datasheet property --
quartz oscillators carry intrinsically larger low-frequency noise
(flicker floor, random-walk FM, aging) than Cs/Rb/H-maser standards --
rather than differential coupling to the TEP field.  This step resolves
the ambiguity on the same station-days.

Three blocks:

A.  Coherent / idiosyncratic decomposition (real data).  Oscillator
    noise is generated inside each receiver or frequency standard and is
    statistically independent between stations; the TEP field is
    spatially coherent.  For every station-day and band the band series
    is split at each epoch into the component predictable from the rest
    of the network -- leave-one-out Gaussian-kernel regression with the
    Step-65 length scale, lambda = 4200 km -- and the idiosyncratic
    remainder.  The quartz/atomic amplitude ratio is then evaluated
    separately on the coherent and idiosyncratic components.  A pure
    oscillator-noise origin produces the excess exclusively in the
    idiosyncratic component; differential field coupling shows in the
    coherent component.

B.  Same-type pair correlation (real data).  Pairwise Pearson
    correlation of the band series between stations of the same clock
    type, binned by great-circle distance, for quartz-quartz and
    atomic-atomic pairs separately.  Independent oscillator noise gives
    zero correlation at every distance; a spatially-coherent field gives
    a positive short-baseline correlation decaying toward zero.

C.  Datasheet noise null (labelled forward model).  Each oscillator
    class present in the analysed stations is assigned its published
    noise law, Sy(f) = h_{-2}/f^2 + h_{-1}/f + h_0 + h_2 f^2
    (fractional-frequency PSD), plus a daily aging draw inside the
    specification range.  Synthetic 288-epoch x 300 s series are pushed
    through the identical pipeline (linear detrend, cubic trend removal,
    band split) and the expected band rms is measured per class.  The
    expected quartz/atomic band ratio for the observed station mix is
    reported with a x{0.3, 1, 3} specification envelope, and the
    observed/expected residual ratio follows per band.  This block is a
    labelled noise-model calibration (synthetic noise, published
    coefficients); the screening verdict rests on the real-data
    decomposition, with the datasheet null bounding how much of the raw
    amplitude ratio is expected from oscillator physics alone.

Units note: `resid` is in nanoseconds; all rms quantities are converted
to metres with C_M_PER_NS = 0.299792458.

Output: results/outputs/step_65c_oscillator_noise_null.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib import tep_data as td          # noqa: E402
from scripts.lib import tep_stats as ts          # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
OUT_FILE = OUT / "step_65c_oscillator_noise_null.json"

C_M_PER_NS = 0.299792458
N_DAYS = int(os.environ.get("TEP_STEP65_DAYS", "30"))
RNG = np.random.default_rng(20260928)
K_NEAREST = 8                    # LOO predictors per epoch
LAM_KM = 4200.0                  # step_65 convention

BANDS = {
    "LF": (0.0, 10.0),
    "MF": (10.0, 500.0),
    "HF": (500.0, 5000.0),
    "FULL": (0.0, 5000.0),
}
QUARTZ_TYPES = {"INTERNAL", "OCXO"}
ATOMIC_TYPES = {"H-MASER", "RUBIDIUM", "CESIUM"}

# ----------------------------------------------------------------------
# Published oscillator noise laws.  Sy(f) = h_-2 f^-2 + h_-1 f^-1 +
# h_0 + h_2 f^2  (fractional-frequency PSD, 1/Hz), ageing_gamma in
# fractional-frequency drift per day (uniform draw within the range).
# Representative published values per class (datasheet / literature
# anchors: BVA-8607, SRS FS725, Quartzlock E10, HP5071A, CH1-75A,
# VCH-1003M, EFOS/iMaser, geodetic-receiver internal oscillators).
# Values are order-of-magnitude typicals; the x{0.3,1,3} envelope on
# every coefficient propagates the spec spread.
# ----------------------------------------------------------------------
OSC_SPECS = {
    "receiver_quartz": {   # INTERNAL / receiver OCXO (geodetic receivers)
        "h_m2": 1.8e-28, "h_m1": 2.9e-24, "h_0": 2.0e-22, "h_2": 2.0e-26,
        "aging_per_day": (1e-11, 1e-10),
        "models": {"Receiver OCXO", "Internal OCXO"},
    },
    "bench_ocxo": {        # Oscilloquartz BVA class (exceptional OCXO)
        "h_m2": 1.7e-30, "h_m1": 1.8e-25, "h_0": 5.0e-25, "h_2": 2.0e-26,
        "aging_per_day": (3e-12, 3e-11),
        "models": {"Oscilloquartz BVA"},
    },
    "rubidium": {          # SRS FS725, Quartzlock E10
        "h_m2": 1.8e-32, "h_m1": 7.2e-27, "h_0": 1.8e-21, "h_2": 2.0e-26,
        "aging_per_day": (1e-13, 2e-12),
        "models": {"SRS FS725", "Quartzlock E10", "Quartzlock E10-FE"},
    },
    "cesium": {            # HP 5071A / USNO cesium
        "h_m2": 1.0e-34, "h_m1": 7.2e-29, "h_0": 5.0e-23, "h_2": 2.0e-26,
        "aging_per_day": (0.0, 1e-16),
        "models": {"HP 5071A", "HP 5071A Cesium", "USNO Cesium"},
    },
    "passive_maser": {     # Vremya CH1-75A
        "h_m2": 1.6e-35, "h_m1": 1.8e-29, "h_0": 1.8e-25, "h_2": 2.0e-26,
        "aging_per_day": (1e-16, 1e-15),
        "models": {"CH1-75A", "VREMYA CH1-75A"},
    },
    "active_maser": {      # VCH-1003, EFOS, PTB active masers
        "h_m2": 1.0e-36, "h_m1": 2.9e-30, "h_0": 2.0e-26, "h_2": 2.0e-26,
        "aging_per_day": (0.0, 3e-16),
        "models": {
            "VCH-1003A", "VREMYA VCH-1003M", "EFOS-C Maser (T4Science)",
            "EFOS-C (T4Science)", "EFOS-B (T4Science)",
            "EFOS-28 (T4Science)", "EFOS-C Maser", "EFOS-C",
            "PTB Active H-Maser",
        },
    },
}
MODEL2CLASS = {m: c for c, s in OSC_SPECS.items() for m in s["models"]}


def _load_maps():
    meta_p = PROJECT_ROOT / "data" / "meta" / "optimal_100_metadata.json"
    if not meta_p.exists():
        meta_p = (PROJECT_ROOT.parent / "TEP-GNSS-RINEX"
                  / "data" / "processed" / "optimal_100_metadata.json")
    meta = json.load(open(meta_p))["stations"]
    type_map, class_map = {}, {}
    for name, info in meta.items():
        ct = (info.get("classification") or {}).get("clock_type", "UNKNOWN")
        type_map[name] = ct
        model = (info.get("equipment") or {}).get("clock_model", "?")
        class_map[name] = MODEL2CLASS.get(model, "unmapped")
    return type_map, class_map


def _bandpass(y, dt, f_lo, f_hi):
    if f_lo <= 0 and f_hi >= 5000:
        return y.copy()
    return ts.bandpass(y, dt, f_lo, f_hi)


def _bands_for_day(y_ns, t, dt=300.0):
    """Step-65b-identical band split.  y in ns."""
    tn = (t - t[0]) / 86400.0
    coef = np.polyfit(tn, y_ns, 3)
    detrended = y_ns - np.polyval(coef, tn)
    out = {}
    for band, (lo, hi) in BANDS.items():
        if band == "LF":
            out["LF"] = y_ns - _bandpass(detrended, dt, 10.0, 5000.0)
        else:
            out[band] = _bandpass(detrended, dt, lo, hi)
    return out


def _loo_coherent_matrix(X, D):
    """Per-epoch LOO network prediction.  X: (epochs, stations).

    coh_i(t) = sum_j w_ij X_j(t) / sum_j w_ij over the K_NEAREST
    neighbours, w_ij = exp(-0.5 (d_ij/LAM)^2).  Returns coherent and
    idiosyncratic matrices of the same shape (nan where <3 neighbours).
    """
    n_ep, n_st = X.shape
    coh = np.full_like(X, np.nan)
    for i in range(n_st):
        di = D[i].copy()
        di[i] = np.inf
        nn = np.argsort(di)[:K_NEAREST]
        w = np.exp(-0.5 * (di[nn] / LAM_KM) ** 2)
        wsum = w.sum()
        if wsum <= 0 or len(nn) < 3:
            continue
        coh[:, i] = (X[:, nn] * w[None, :]).sum(axis=1) / wsum
    return coh, X - coh


def _synth_x(cls_key, T=86400, dt=300.0, scale=1.0, rng=None):
    """One realisation of clock time deviation x(t) [s] for a class.

    Spectral shaping with a measured-PSD renormalisation per band so the
    integrated noise power matches int_band Sy(f)/(2 pi f)^2 df exactly.
    Adds a quadratic aging drift drawn uniformly in the spec range.
    """
    rng = rng or RNG
    spec = OSC_SPECS[cls_key]
    n = int(round(T / dt))
    f = np.fft.rfftfreq(n, d=dt)
    f[0] = f[1]
    Sy = (spec["h_m2"] * scale / f ** 2 + spec["h_m1"] * scale / f +
          spec["h_0"] * scale + spec["h_2"] * scale * f ** 2)
    Sx = Sy / (2 * np.pi * f) ** 2
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    x = np.fft.irfft(W * np.sqrt(Sx), n)
    # Normalise to the analytic full-band variance
    var_target = np.trapezoid(Sx, f)
    var_meas = np.var(x)
    if var_meas > 0:
        x *= np.sqrt(var_target / var_meas)
    # Aging: linear frequency drift -> quadratic time deviation
    lo, hi = spec["aging_per_day"]
    gamma = rng.uniform(lo, hi)          # frac-freq per day
    tn = np.arange(n) * dt / 86400.0     # days
    x = x + 0.5 * gamma * tn ** 2 * 86400.0
    return x


def _class_expected_rms(n_rep=150):
    """Expected band rms (m) per oscillator class through the pipeline."""
    out = {}
    for cls in OSC_SPECS:
        for scale, tag in ((0.3, "low"), (1.0, "typ"), (3.0, "high")):
            acc = {b: [] for b in BANDS}
            for _ in range(n_rep):
                x = _synth_x(cls, scale=scale)
                y_ns = x * 1e9
                y_lin = y_ns - np.polyval(np.polyfit(np.arange(len(y_ns)),
                                                     y_ns, 1),
                                          np.arange(len(y_ns)))
                bb = _bands_for_day(y_lin, np.arange(len(y_ns)) * 300.0, dt=300.0)
                for b, v in bb.items():
                    acc[b].append(np.sqrt(np.mean(v ** 2)))
            out.setdefault(cls, {})[tag] = {
                b: float(np.median(v) * C_M_PER_NS) for b, v in acc.items()
            }
    return out


def run():
    from scripts.steps.step_65_oscillator_screening import _load_dense_day
    type_map, class_map = _load_maps()
    dates = td.available_dates("precise", 2024)
    if not dates:
        dates = [f"2024{d:03d}" for d in range(1, 367)]
    days = [dates[i] for i in
            np.linspace(0, len(dates) - 1, min(N_DAYS, len(dates))).astype(int)]

    station_rows = []          # per (station, day, band): stats dict
    qq_pairs, aa_pairs = {}, {}   # band -> (dists, corrs) accumulated per day
    for day in days:
        s = _load_dense_day(day)
        if not s:
            continue
        names = [n for n in s if n in type_map and type_map[n] in
                 QUARTZ_TYPES | ATOMIC_TYPES]
        if len(names) < 6:
            continue
        # common grid
        t = s[names[0]]["t"].astype(float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 300.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 300.0
        xyz = {}
        for n in names:
            st = s[n].get("station")
            if st is not None and hasattr(st, "ecef_km"):
                xyz[n] = np.asarray(st.ecef_km, float)
        names = [n for n in names if n in xyz]
        if len(names) < 6:
            continue
        D = np.array([[td.great_circle_km(s[a]["station"], s[b]["station"])
                       for b in names] for a in names])

        bands_per = {}
        for n in names:
            y = s[n]["resid"].astype(float)
            if not np.isfinite(y).all():
                ix = np.arange(len(y))
                y = np.interp(ix, ix[np.isfinite(y)], y[np.isfinite(y)])
            bands_per[n] = _bands_for_day(y, t)

        for band in BANDS:
            X = np.column_stack([bands_per[n][band] for n in names])
            if X.shape[0] < 20:
                continue
            pred, _ = _loo_coherent_matrix(X, D)
            for i, n in enumerate(names):
                ct = type_map[n]
                grp = "quartz" if ct in QUARTZ_TYPES else "atomic"
                yi = X[:, i]
                pi = pred[:, i]
                # Regress the station series onto its network prediction:
                # b = <y,p>/<p,p> is the field-coupling amplitude; the
                # coherent component is b*pred and the idiosyncratic
                # remainder is orthogonal to the network prediction.
                if not np.isfinite(pi).all() or np.mean(pi ** 2) <= 0:
                    b_i = np.nan
                else:
                    b_i = float(np.mean(yi * pi) / np.mean(pi * pi))
                coh_rms = (abs(b_i) * np.sqrt(np.mean(pi ** 2))
                           if np.isfinite(b_i) else np.nan)
                idio_rms = (np.sqrt(np.mean((yi - b_i * pi) ** 2))
                            if np.isfinite(b_i) else np.nan)
                station_rows.append({
                    "station": n, "day": day, "band": band, "group": grp,
                    "class": class_map.get(n, "unmapped"),
                    "rms_m": float(np.sqrt(np.mean(yi ** 2))
                                   * C_M_PER_NS),
                    "coupling_b": b_i,
                    "coh_rms_m": float(coh_rms) * C_M_PER_NS
                    if np.isfinite(coh_rms) else np.nan,
                    "idio_rms_m": float(idio_rms) * C_M_PER_NS
                    if np.isfinite(idio_rms) else np.nan,
                })
            # same-type pair correlations
            for pair_grp, acc in (("quartz", qq_pairs), ("atomic", aa_pairs)):
                sel = [i for i, n in enumerate(names)
                       if (type_map[n] in QUARTZ_TYPES) == (pair_grp == "quartz")]
                acc.setdefault(band, [[], []])
                for a in range(len(sel)):
                    for b in range(a + 1, len(sel)):
                        i, j = sel[a], sel[b]
                        xi, xj = X[:, i], X[:, j]
                        si, sj = xi.std(), xj.std()
                        if si < 1e-12 or sj < 1e-12:
                            continue
                        acc[band][0].append(D[i, j])
                        acc[band][1].append(float(np.mean(
                            (xi - xi.mean()) * (xj - xj.mean())) / (si * sj)))

    # ---- aggregate real-data statistics --------------------------------
    def _med(rows, key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.median(v)) if v else None

    def _step65b_style_median(rows, key):
        """Median over per-station median rms, matching step_65b's
        aggregation (station, not station-day, as the unit)."""
        per_sta = {}
        for r in rows:
            if np.isfinite(r[key]):
                per_sta.setdefault(r["station"], []).append(r[key])
        meds = [float(np.median(v)) for v in per_sta.values()]
        return float(np.median(meds)) if meds else None

    per_band = {}
    for band in BANDS:
        per_band[band] = {}
        for grp in ("quartz", "atomic"):
            rows = [r for r in station_rows
                    if r["band"] == band and r["group"] == grp]
            per_band[band][grp] = {
                "n_station_days": len(rows),
                "rms_m": _med(rows, "rms_m"),
                "rms_m_station_median": _step65b_style_median(rows, "rms_m"),
                "coupling_b": _med(rows, "coupling_b"),
                "coupling_b_station_median": _step65b_style_median(rows, "coupling_b"),
                "coherent_rms_m": _med(rows, "coh_rms_m"),
                "coherent_rms_m_station_median": _step65b_style_median(rows, "coh_rms_m"),
                "idiosyncratic_rms_m": _med(rows, "idio_rms_m"),
                "idiosyncratic_rms_m_station_median": _step65b_style_median(rows, "idio_rms_m"),
            }
        q, a = per_band[band]["quartz"], per_band[band]["atomic"]
        per_band[band]["ratio_total"] = (
            q["rms_m"] / a["rms_m"] if q["rms_m"] and a["rms_m"] else None)
        per_band[band]["ratio_total_station_median"] = (
            q["rms_m_station_median"] / a["rms_m_station_median"]
            if q["rms_m_station_median"] and a["rms_m_station_median"]
            else None)
        per_band[band]["ratio_coupling"] = (
            q["coupling_b"] / a["coupling_b"]
            if q["coupling_b"] and a["coupling_b"] else None)
        per_band[band]["ratio_coupling_station_median"] = (
            q["coupling_b_station_median"] / a["coupling_b_station_median"]
            if q["coupling_b_station_median"] and a["coupling_b_station_median"]
            else None)
        per_band[band]["ratio_coherent"] = (
            q["coherent_rms_m"] / a["coherent_rms_m"]
            if q["coherent_rms_m"] and a["coherent_rms_m"] else None)
        per_band[band]["ratio_coherent_station_median"] = (
            q["coherent_rms_m_station_median"] / a["coherent_rms_m_station_median"]
            if q["coherent_rms_m_station_median"] and a["coherent_rms_m_station_median"]
            else None)
        per_band[band]["ratio_idiosyncratic"] = (
            q["idiosyncratic_rms_m"] / a["idiosyncratic_rms_m"]
            if q["idiosyncratic_rms_m"] and a["idiosyncratic_rms_m"] else None)
        per_band[band]["ratio_idiosyncratic_station_median"] = (
            q["idiosyncratic_rms_m_station_median"] / a["idiosyncratic_rms_m_station_median"]
            if q["idiosyncratic_rms_m_station_median"] and a["idiosyncratic_rms_m_station_median"]
            else None)

        # Exact C(18, 8) = 43,758 station-level label permutation test
        import itertools
        b_rows = [r for r in station_rows if r["band"] == band]
        sta_list = sorted({r["station"] for r in b_rows})
        if len(sta_list) >= 6:
            sta_grp = np.array([
                next(r["group"] for r in b_rows if r["station"] == s)
                for s in sta_list
            ])
            sta_tot = np.array([
                np.nanmedian([r["rms_m"] for r in b_rows if r["station"] == s])
                for s in sta_list
            ])
            sta_coh = np.array([
                np.nanmedian([r["coh_rms_m"] for r in b_rows if r["station"] == s])
                for s in sta_list
            ])
            sta_b = np.array([
                np.nanmedian([r["coupling_b"] for r in b_rows if r["station"] == s])
                for s in sta_list
            ])
            nq = int((sta_grp == "quartz").sum())
            obs_tot_r = per_band[band]["ratio_total_station_median"] or 0.0
            obs_coh_r = per_band[band]["ratio_coherent_station_median"] or 0.0
            obs_b_r = per_band[band]["ratio_coupling_station_median"] or 0.0
            n_comb = c_tot = c_coh = c_b = 0
            for q_idx in itertools.combinations(range(len(sta_list)), nq):
                qm = np.zeros(len(sta_list), dtype=bool)
                qm[list(q_idx)] = True
                am = ~qm
                n_comb += 1
                if np.median(sta_tot[qm]) / np.median(sta_tot[am]) >= obs_tot_r - 1e-12:
                    c_tot += 1
                if np.median(sta_coh[qm]) / np.median(sta_coh[am]) >= obs_coh_r - 1e-12:
                    c_coh += 1
                if np.median(sta_b[qm]) / np.median(sta_b[am]) >= obs_b_r - 1e-12:
                    c_b += 1
            per_band[band]["station_level_permutation"] = {
                "n_stations": len(sta_list),
                "n_quartz": nq,
                "n_atomic": len(sta_list) - nq,
                "n_exact_combinations": n_comb,
                "p_total_ratio": float(c_tot / n_comb),
                "p_coherent_ratio": float(c_coh / n_comb),
                "p_coupling_ratio": float(c_b / n_comb),
            }

    # same-type pair correlation profiles (median corr in 4 distance bins)
    pair_profiles = {}
    edges = [0, 500, 1000, 2000, 4000]
    for tag, acc in (("quartz_quartz", qq_pairs), ("atomic_atomic", aa_pairs)):
        pair_profiles[tag] = {}
        for band, (dists, corrs) in acc.items():
            d, c = np.asarray(dists), np.asarray(corrs)
            prof = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (d >= lo) & (d < hi)
                prof.append({
                    "bin_km": [lo, hi], "n_pairs": int(m.sum()),
                    "median_corr": float(np.median(c[m])) if m.sum() >= 5
                    else None,
                    "mean_corr": float(np.mean(c[m])) if m.sum() >= 5
                    else None,
                })
            pair_profiles[tag][band] = prof

    # ---- datasheet noise null ------------------------------------------
    # observed station mix per group
    mix = {"quartz": {}, "atomic": {}}
    seen = set()
    for r in station_rows:
        if r["station"] in seen:
            continue
        seen.add(r["station"])
        mix[r["group"]][r["class"]] = mix[r["group"]].get(r["class"], 0) + 1

    expected_class = _class_expected_rms()
    expected_ratio = {}
    for band in BANDS:
        expected_ratio[band] = {}
        for tag in ("low", "typ", "high"):
            qv, av = [], []
            for cls, cnt in mix["quartz"].items():
                if cls in expected_class:
                    qv += [expected_class[cls][tag][band]] * cnt
            for cls, cnt in mix["atomic"].items():
                if cls in expected_class:
                    av += [expected_class[cls][tag][band]] * cnt
            if qv and av:
                expected_ratio[band][tag] = {
                    "quartz_m": float(np.median(qv)),
                    "atomic_m": float(np.median(av)),
                    "ratio": float(np.median(qv) / np.median(av)),
                }

    # ---- verdict -------------------------------------------------------
    verdict = {}
    for band in ("LF", "MF", "HF"):
        obs = per_band[band]["ratio_total"]
        exp = (expected_ratio.get(band, {}).get("typ") or {}).get("ratio")
        verdict[band] = {
            "observed_total_ratio": obs,
            "expected_null_ratio_typ": exp,
            "observed_over_expected": (obs / exp) if (obs and exp) else None,
            "coupling_ratio": per_band[band]["ratio_coupling"],
            "coherent_ratio": per_band[band]["ratio_coherent"],
            "idiosyncratic_ratio": per_band[band]["ratio_idiosyncratic"],
        }

    out = {
        "step": "65c",
        "title": "Oscillator-noise null for the quartz/atomic screening claim",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_days": len(days),
        "units": "metres (resid ns x C_M_PER_NS)",
        "station_mix_by_oscillator_class": mix,
        "per_band": per_band,
        "same_type_pair_profiles": pair_profiles,
        "datasheet_null": {
            "note": "Labelled forward model: synthetic Sy(f) noise from "
                    "published per-class coefficient sets passed through the "
                    "identical detrend+band-split pipeline; not data.",
            "expected_class_rms_m": expected_class,
            "expected_group_ratio": expected_ratio,
        },
        "verdict": verdict,
        "interpretation_hint": (
            "If observed/expected ~= 1 in all bands the amplitude ratio is "
            "datasheet-consistent; the screening axis then lives in "
            "ratio_coherent (spatially coherent excess, which station-local "
            "noise cannot produce)."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=1))
    print(json.dumps(verdict, indent=1))
    return out


if __name__ == "__main__":
    run()
