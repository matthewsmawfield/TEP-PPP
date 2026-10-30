#!/usr/bin/env python3
"""
STEP 77  --  Co-located internal/external-atomic reference differential
             against the global field, using phase-cos.

Comparing lambda between QQ / QA / AA pair classes (Steps 75, 75b, 75c) is
confounded beyond repair: the three classes sample different stations, hence
different geography and different distance distributions, and lambda from a few
hundred pairs is unconstrained.  Step 75c's own controls showed the "monotone"
ordering appears in 35% of RANDOM clock-label assignments.

A co-located pair removes every one of those confounds at once.  Two receivers
at the same DOMES site see the identical proper-time field, the identical
troposphere, the identical satellites and the identical network datum.  Form

        Delta(t) = resid_internal(t) - resid_external_atomic(t)

and the field CANCELS EXACTLY if both oscillators couple to it identically.
It also cancels the troposphere, the satellite clock/orbit error, and the
rank-1 network datum -- all of which are common to the site.  What survives is
(1 - s) * field + differential local noise under the screening hypothesis.
However, the IGS label INTERNAL does not document oscillator material, so this
is rigorously a reference-source contrast, not by itself a quartz-material test.

So the question becomes a sharper one: does Delta retain reproducible,
distance-structured coupling to DISTANT stations?  Local oscillator noise is
not expected to produce that coupling, but receiver/antenna/analysis-centre
systematics remain explicit alternatives and therefore require nulls and
replication across sites.

THE DECISIVE DISCRIMINATOR.  A mundane instrumental or processing difference
between two co-located receivers would couple to distant stations with NO
distance structure -- a flat offset.  Screening of a spatially structured field
must couple with the field's own correlation length.  So the test is not merely
"is the coupling non-zero" but "does it decay with distance at lambda_TEP".

Estimator is the Step 53 phase-alignment index cos(arg(sum_f S_ij(f))),
10-500 uHz, linear detrend -- imported, not reimplemented.  Fit is
A*exp(-d/lambda) + C0 to distance-bin means, count-weighted.

The operational CODE product contains three coordinate-verified
internal/external-free-atomic pairs within 100 m and one 100--200 m sensitivity
pair.  Metadata matches farther apart than 200 m are rejected;
sharing a DOMES stem or facility name is not sufficient.  WUHN is also flagged
as GPS-steered rather than treated as a high-immunity free atomic anchor.
Reported alongside: the same coupling for each clock individually, a
phase-randomised diagnostic, and the primary equal-power null: station-level
distance permutation after collapsing repeated days within each distant
reference station.

Output: results/outputs/step_77_colocated_screening_differential.json
"""
from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

_spec = importlib.util.spec_from_file_location(
    "s53", ROOT / "scripts" / "steps" / "step_53_coherent_lambda.py")
s53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s53)

OUT = ROOT / "results" / "outputs" / "step_77_colocated_screening_differential.json"
AUDIT_PATH = ROOT / "data" / "meta" / "clock_anchor_audit_2024.json"
AUDIT = json.loads(AUDIT_PATH.read_text())
RNG = np.random.default_rng(20260904)

PRODUCT = "clk_cod_ops"
N_DAYS = 366
N_NULL = 1000
N_NULL_REAL = 0       # retired: 40-day phase null had unequal power vs 280--349-day result
_FREQ = np.fft.rfftfreq(2880, d=30.0)
MODE_FREQUENCY_UHZ = (1e6 * _FREQ[(_FREQ >= s53.F_LO_HZ) & (_FREQ <= s53.F_HI_HZ)]).tolist()
PRIMARY_MAX_M = 100.0
SECONDARY_MAX_M = 200.0

ANCHOR_METADATA = {
    (p["external_station"], p["internal_station"]): p
    for p in AUDIT["pairs"].values()
}


def day_rows(ser, atomic, internal, phase_rng=None):
    """Phase-cos of (internal - external atomic), and of each clock alone."""
    if atomic not in ser or internal not in ser:
        return []
    ta = np.asarray(ser[atomic]["t"], float); xa = np.asarray(ser[atomic]["resid"], float)
    tq = np.asarray(ser[internal]["t"], float); xq = np.asarray(ser[internal]["resid"], float)
    oa = np.isfinite(ta) & np.isfinite(xa); oq = np.isfinite(tq) & np.isfinite(xq)
    if oa.sum() < 200 or oq.sum() < 200:
        return []
    t0 = max(ta[oa].min(), tq[oq].min()); t1 = min(ta[oa].max(), tq[oq].max())
    if t1 - t0 < 40000:
        return []
    grid = np.arange(t0, t1, 30.0)
    A = np.interp(grid, ta[oa], xa[oa]); Q = np.interp(grid, tq[oq], xq[oq])
    site = ser[atomic]["station"]

    # build a series dict the Step 53 estimator can consume unchanged
    sub = {n: v for n, v in ser.items() if n not in (atomic, internal)}
    sub["__DELTA__"] = {"t": grid, "resid": Q - A, "station": site}
    sub["__ATOMIC__"] = {"t": grid, "resid": A, "station": site}
    sub["__INTERNAL__"] = {"t": grid, "resid": Q, "station": site}
    RF, STA = s53._spectra(sub, phase_rng=None)
    for probe in ("__DELTA__", "__ATOMIC__", "__INTERNAL__"):
        if probe not in RF:
            return []
    if phase_rng is not None:
        F = RF["__DELTA__"]
        RF["__DELTA__"] = np.abs(F) * np.exp(1j * phase_rng.uniform(-np.pi, np.pi, F.shape))

    out = []
    for n, F in RF.items():
        if n.startswith("__"):
            continue
        d = td.great_circle_km(site, STA[n])
        for probe, tag in (("__DELTA__", "delta"), ("__ATOMIC__", "external_atomic"),
                           ("__INTERNAL__", "internal")):
            Fp = RF[probe]
            if Fp.shape != F.shape:
                continue
            cross = np.conj(Fp) * F
            S = np.sum(cross)
            if np.isfinite(S) and S != 0:
                modes = (np.cos(np.angle(cross)).tolist() if tag == "delta" else None)
                out.append((tag, n, d, float(np.cos(np.angle(S))), modes))
    return out


def fit_curve(pairs):
    pairs = [p for p in pairs if s53.DMIN_KM <= p[0] <= s53.DMAX_KM]
    if len(pairs) < 20:
        return None
    # Daily replication has deliberately been collapsed to one datum per
    # distant reference station.  The original 40 fixed bins require 20
    # *independent station pairs* per bin and are therefore too sparse here.
    # Equal-count distance bins retain 12--20 stations per bin without putting
    # the repeated days back into the nominal sample size.
    arr = np.asarray(pairs, float)
    arr = arr[np.argsort(arr[:, 0])]
    n_bins_target = min(15, max(6, len(arr) // 15))
    chunks = [c for c in np.array_split(arr, n_bins_target) if len(c) >= 3]
    ctr = np.asarray([np.exp(np.mean(np.log(c[:, 0]))) for c in chunks])
    val = np.asarray([np.mean(c[:, 1]) for c in chunks])
    cnt = np.asarray([len(c) for c in chunks])
    if len(ctr) < 6:
        return None
    f = s53._fit(ctr, val, cnt)
    if not f:
        return None
    e = f.get("exponential") or {}
    return {"lambda_km": e.get("lambda_km"), "R2_exponential": e.get("R2"),
            "best_model": f.get("best_model"), "lambda_best_km": f.get("lambda_km"),
            "R2_best": f.get("R2"), "n_bins": len(ctr),
            "selection_criterion": f.get("selection_criterion"),
            "delta_AICc_exponential_vs_best": f.get("delta_aicc_exponential_vs_best"),
            "mean_index": float(np.mean([p[1] for p in pairs]))}


def collapse_by_reference(rows):
    """One independent value per distant station, not one pseudo-replicate per day."""
    by_ref = {}
    for ref, d, value in rows:
        by_ref.setdefault(ref, {"distance_km": d, "values": []})["values"].append(value)
    collapsed = []
    fingerprints = []
    for ref, rec in sorted(by_ref.items()):
        values = np.asarray(rec["values"], float)
        if len(values) < 10:
            continue
        mean_value = float(np.mean(values))
        collapsed.append((float(rec["distance_km"]), mean_value))
        fingerprints.append({"reference": ref, "distance_km": float(rec["distance_km"]),
                             "mean_phase_cos": mean_value, "n_days": int(len(values))})
    return collapsed, fingerprints


def add_modes(store, ref, distance_km, modes):
    if modes is None:
        return
    a = np.asarray(modes, float)
    rec = store.setdefault(ref, {"distance_km": distance_km,
                                 "sum": np.zeros_like(a), "n": 0})
    if rec["sum"].shape == a.shape and np.all(np.isfinite(a)):
        rec["sum"] += a
        rec["n"] += 1


def collapse_modes(store):
    out = []
    for ref, rec in sorted(store.items()):
        if rec["n"] >= 10:
            out.append({"reference": ref, "distance_km": float(rec["distance_km"]),
                        "mean_phase_cos_by_frequency": (rec["sum"] / rec["n"]).tolist(),
                        "n_days": int(rec["n"])})
    return out


def distance_permutation_null(collapsed, observed_r2, seed):
    """Preserve station fingerprints and sample size; destroy only their geometry."""
    if not collapsed or observed_r2 is None:
        return None
    rng = np.random.default_rng(seed)
    distances = np.asarray([p[0] for p in collapsed], float)
    values = np.asarray([p[1] for p in collapsed], float)
    r2, lam = [], []
    for _ in range(N_NULL):
        f = fit_curve(list(zip(rng.permutation(distances), values)))
        if f and f["R2_exponential"] is not None and f["lambda_km"] is not None:
            r2.append(float(f["R2_exponential"]))
            lam.append(float(f["lambda_km"]))
    r2 = np.asarray(r2); lam = np.asarray(lam)
    if not len(r2):
        return None
    return {
        "unit": "distant reference station (daily repeats collapsed before permutation)",
        "n_realisations": int(len(r2)),
        "n_reference_stations": int(len(collapsed)),
        "R2_median": float(np.median(r2)), "R2_max": float(np.max(r2)),
        "lambda_median_km": float(np.median(lam)),
        "p_R2_ge_observed": float((1 + np.sum(r2 >= observed_r2)) / (1 + len(r2))),
        "p_TEP_band_and_R2_ge_observed": float((1 + np.sum(
            (r2 >= observed_r2) & (lam >= 1000) & (lam <= 10000))) / (1 + len(r2))),
    }


def main():
    if AUDIT.get("status") != "PASS" or AUDIT.get("analysis_year") != 2024:
        raise RuntimeError(f"Step 82 metadata audit is not valid for 2024: {AUDIT_PATH}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = td.available_dates(PRODUCT, 2024)[:N_DAYS]
    sers = {}
    cands = []
    probe = td.load_series(PRODUCT, dates[min(99, len(dates) - 1)],
                           min_valid_frac=0.0, max_jumps=10_000)
    rejected_distance = []
    seen = set()
    for _, c in AUDIT["pairs"].items():
        pair = (c["external_station"], c["internal_station"])
        if pair in seen or pair[0] not in probe or pair[1] not in probe:
            continue
        seen.add(pair)
        separation_m = 1000.0 * td.chord_km(
            probe[pair[0]]["station"], probe[pair[1]]["station"])
        if separation_m <= SECONDARY_MAX_M:
            cands.append(pair)
        else:
            rejected_distance.append({"external_atomic": pair[0], "internal": pair[1],
                                      "separation_m": separation_m})
    if not cands:
        OUT.write_text(json.dumps({"step": 77, "status": "no_colocated_pairs"}, indent=2))
        print("no co-located pairs in product"); return
    print(f"STEP 77  |  {PRODUCT}, {len(dates)} days, co-located pairs: {cands}")

    arms = {}
    for atomic, internal in cands:
        acc = {"delta": [], "external_atomic": [], "internal": []}
        mode_acc = {}
        split_acc = {"first_half": [], "second_half": []}
        split_mode_acc = {"first_half": {}, "second_half": {}}
        split_days = {"first_half": 0, "second_half": 0}
        nd = 0
        for k, tag in enumerate(dates):
            try:
                ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
            except Exception:
                continue
            r = day_rows(ser, atomic, internal)
            if r:
                nd += 1
                half = "first_half" if k < len(dates) // 2 else "second_half"
                split_days[half] += 1
                for t, ref, d, v, modes in r:
                    acc[t].append((ref, d, v))
                    if t == "delta" and modes is not None:
                        add_modes(mode_acc, ref, d, modes)
                        split_acc[half].append((ref, d, v))
                        add_modes(split_mode_acc[half], ref, d, modes)
            if (k + 1) % 90 == 0:
                print(f"    {k+1}/{len(dates)}  {nd} usable days, "
                      f"{len(acc['delta']):,} delta measurements", flush=True)
        if nd < 30:
            print(f"  {internal}-{atomic}: only {nd} usable days, skipping"); continue

        collapsed = {}
        fingerprints = {}
        for t in ("delta", "external_atomic", "internal"):
            collapsed[t], fingerprints[t] = collapse_by_reference(acc[t])
        mode_fingerprints = collapse_modes(mode_acc)
        temporal_halves = {}
        for half in ("first_half", "second_half"):
            cp, fp = collapse_by_reference(split_acc[half])
            hf = fit_curve(cp)
            hn = distance_permutation_null(
                cp, hf["R2_exponential"] if hf else None,
                seed=77500 + sum(ord(c) for c in atomic + internal + half))
            temporal_halves[half] = {
                "n_days": split_days[half], "fit": hf, "geometry_null": hn,
                "reference_fingerprints": fp,
                "frequency_resolved_delta_fingerprints": collapse_modes(split_mode_acc[half]),
            }
        fits = {t: fit_curve(collapsed[t]) for t in
                ("delta", "external_atomic", "internal")}
        print(f"\n  === {internal}(internal) - {atomic}(external atomic), {nd} days ===")
        for t in ("internal", "external_atomic", "delta"):
            f = fits[t]
            if f:
                print(f"    {t:<7} mean index {f['mean_index']:+.4f}  "
                      f"lambda_exp {f['lambda_km']:.0f} km  R2 {f['R2_exponential']:.3f}  "
                      f"best {f['best_model']}")

        # phase-randomised null on the differential: does its distance structure survive?
        # NULL DISTRIBUTION, not a single pooled null.  A single phase-randomised
        # fit is one draw and can be an outlier: the first version of this step
        # drew R2 = 0.490 and understated the result badly.  Each realisation here
        # phase-randomises the differential over a full block of days and is fitted
        # independently, so it is directly comparable to the observed fit.
        DAYS_PER_REAL = 40
        null_lam, null_r2 = [], []
        for i in range(N_NULL_REAL):
            rng = np.random.default_rng(4000 + i)
            acc_n = []
            for tag in dates[i::9][:DAYS_PER_REAL]:
                try:
                    ser = td.load_series(PRODUCT, tag, min_valid_frac=0.0, max_jumps=10_000)
                except Exception:
                    continue
                r = day_rows(ser, atomic, internal, phase_rng=rng)
                acc_n.extend([(ref, d, v) for t, ref, d, v, _ in r if t == "delta"])
            collapsed_n, _ = collapse_by_reference(acc_n)
            fn = fit_curve(collapsed_n)
            if fn and fn["lambda_km"]:
                null_lam.append(fn["lambda_km"]); null_r2.append(fn["R2_exponential"])
        null_lam = np.array(null_lam); null_r2 = np.array(null_r2)
        obs = fits["delta"]
        geometry_null = distance_permutation_null(
            collapsed["delta"], obs["R2_exponential"] if obs else None,
            seed=77000 + sum(ord(c) for c in atomic + internal))
        anchor_meta = ANCHOR_METADATA.get((atomic, internal), {"primary_anchor": False,
                                                              "exclusion_reason": "unverified"})
        p_primary = geometry_null["p_R2_ge_observed"] if geometry_null else None
        verdict_pair = {
            "n_days": nd,
            "separation_m": float(1000.0 * td.chord_km(
                probe[atomic]["station"], probe[internal]["station"])),
            "anchor_metadata": anchor_meta,
            "fits": fits,
            "reference_fingerprints": fingerprints,
            "mode_frequency_uhz": MODE_FREQUENCY_UHZ,
            "frequency_resolved_delta_fingerprints": mode_fingerprints,
            "temporal_halves": temporal_halves,
            "primary_geometry_null": geometry_null,
            "secondary_phase_null_unequal_duration": {
                "n_realisations": int(len(null_r2)),
                "days_per_realisation": DAYS_PER_REAL,
                "lambda_median_km": float(np.median(null_lam)) if len(null_lam) else None,
                "lambda_min_km": float(null_lam.min()) if len(null_lam) else None,
                "lambda_max_km": float(null_lam.max()) if len(null_lam) else None,
                "R2_median": float(np.median(null_r2)) if len(null_r2) else None,
                "R2_max": float(null_r2.max()) if len(null_r2) else None,
                "p_R2_ge_observed": float(np.mean(null_r2 >= obs["R2_exponential"]))
                                    if (len(null_r2) and obs) else None,
                "p_TEP_band_and_R2_ge_observed": float(np.mean(
                    (null_r2 >= obs["R2_exponential"]) & (null_lam >= 1000) & (null_lam <= 10000)))
                    if (len(null_r2) and obs) else None,
            },
            "delta_couples_to_distant_field": bool(
                obs and p_primary is not None and obs["R2_exponential"] is not None
                and p_primary < 0.05),
            "delta_lambda_in_TEP_band": bool(
                obs and obs["lambda_km"] and 1000 <= obs["lambda_km"] <= 10000),
        }
        if len(null_r2):
            print(f"    null distribution (n={len(null_r2)}): lambda median "
                  f"{np.median(null_lam):.0f} km  R2 median {np.median(null_r2):.3f} "
                  f"max {null_r2.max():.3f}")
            print(f"    p(null R2 >= observed) = {np.mean(null_r2 >= obs['R2_exponential']):.3f}")
        if geometry_null:
            print(f"    station-distance permutation (n={geometry_null['n_realisations']}): "
                  f"R2 median {geometry_null['R2_median']:.3f}, "
                  f"p={geometry_null['p_R2_ge_observed']:.4f}")
        print(f"    -> couples to distant field: {verdict_pair['delta_couples_to_distant_field']}"
              f" | lambda in TEP band: {verdict_pair['delta_lambda_in_TEP_band']}")
        arms[f"{internal}-{atomic}"] = verdict_pair

    showing_contrast = [k for k, v in arms.items()
                        if v["delta_couples_to_distant_field"]
                        and v["delta_lambda_in_TEP_band"]]
    primary_showing_contrast = [k for k in showing_contrast
                                if arms[k]["anchor_metadata"].get("primary_anchor")]
    temporally_replicated = []
    for name in primary_showing_contrast:
        halves = arms[name].get("temporal_halves", {})
        passed = True
        for half in ("first_half", "second_half"):
            fit = (halves.get(half) or {}).get("fit") or {}
            null = (halves.get(half) or {}).get("geometry_null") or {}
            passed &= bool(1000 <= (fit.get("lambda_km") or -1) <= 10000
                           and (fit.get("R2_exponential") or -1) >= 0.5
                           and (null.get("p_R2_ge_observed") or 1) <= 0.01)
        if passed:
            temporally_replicated.append(name)
    verdict = {
        "metadata_audit": str(AUDIT_PATH.relative_to(ROOT)),
        "metadata_audit_status": AUDIT["status"],
        "taxonomy": "internal receiver standard versus external free atomic standard",
        "internal_oscillator_material_confirmed": False,
        "design": "co-located differencing strongly suppresses common site, satellite and "
                  "network terms; residual distance structure is consistent with a screening prediction, "
                  "with receiver/antenna/processing systematics retained as alternatives",
        "discriminator": "TEP screening predicts a replicated distance-structured residual "
                         "at the established scale; generic local noise does not",
        "estimator": "Step 53 phase-alignment index cos(arg(sum_f S_ij(f)))",
        "primary_coordinate_threshold_m": PRIMARY_MAX_M,
        "secondary_sensitivity_threshold_m": SECONDARY_MAX_M,
        "distance_rejected_metadata_pairs": rejected_distance,
        "n_colocated_sites": len(arms),
        "n_primary_free_atomic_anchors": sum(
            bool(v["anchor_metadata"].get("primary_anchor")) for v in arms.values()),
        "n_secondary_near_colocated_anchors": sum(
            bool(v["anchor_metadata"].get("secondary_near_colocated")) for v in arms.values()),
        "sites_showing_reference_class_contrast": showing_contrast,
        "primary_sites_showing_reference_class_contrast": primary_showing_contrast,
        "primary_sites_replicating_in_both_temporal_halves": temporally_replicated,
        "strongest_controlled_anchor": "MAT1-MATE (same Leica GR30 receiver model and firmware epochs)",
        "limitation": "site is the independent replication unit; days and distant "
                      "stations increase precision within a site but do not increase the number "
                      "of independent anchors. INTERNAL is not a material identification, so a "
                      "direct oscillator-output experiment is still required for quartz-specific screening.",
    }
    OUT.write_text(json.dumps({
        "step": 77, "title": "Co-located internal/external-atomic reference differential vs the global field",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "verdict": verdict}, indent=2))
    print(f"\nVERDICT: {json.dumps(verdict, indent=1)}")


if __name__ == "__main__":
    main()
