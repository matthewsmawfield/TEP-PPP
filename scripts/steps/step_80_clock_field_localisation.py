#!/usr/bin/env python3
"""STEP 80 -- Invert co-located clock-differential fingerprints into position.

This is a clock-field observability replay, not yet a live zero-satellite trial.
The target observable is the internal-minus-external-atomic phase-cos fingerprint against
the global reference-clock network produced by Step 77.  Target coordinates are
hidden from the estimator.  The only spatial law supplied to it is the fixed
4,201 km TEP length measured in the earlier papers; amplitude and offset are
fit nuisance parameters at every candidate point.

The decisive controls preserve every clock value and reference station but
permute which station owns which value.  This destroys geometry only.  Because
the source receiver clocks are CODE GNSS clock products, a successful inversion
demonstrates information/algorithmic feasibility, not autonomous hardware.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td

SOURCE = ROOT / "results" / "outputs" / "step_77_colocated_screening_differential.json"
AUDIT_SOURCE = ROOT / "results" / "outputs" / "step_82_clock_metadata_audit.json"
OUT = ROOT / "results" / "outputs" / "step_80_clock_field_localisation.json"
LAMBDA_KM = 4201.0
N_NULL = 500


def distances_km(candidate_lat, candidate_lon, ref_lat, ref_lon):
    la1 = np.radians(np.asarray(candidate_lat, float))[:, None]
    lo1 = np.radians(np.asarray(candidate_lon, float))[:, None]
    la2 = np.radians(np.asarray(ref_lat, float))[None, :]
    lo2 = np.radians(np.asarray(ref_lon, float))[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * td.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def kernel_matrix(lat, lon, ref_lat, ref_lon, family):
    d = distances_km(lat, lon, ref_lat, ref_lon)
    if family == "exponential":
        k = np.exp(-d / LAMBDA_KM)
    elif family == "gaussian":
        k = np.exp(-0.5 * (d / LAMBDA_KM) ** 2)
    else:
        raise ValueError(family)
    k -= np.mean(k, axis=1, keepdims=True)
    norm = np.linalg.norm(k, axis=1, keepdims=True)
    return np.divide(k, norm, out=np.zeros_like(k), where=norm > 0)


def candidate_grid(step_deg=2.0):
    lat_axis = np.arange(-88.0, 88.0 + 1e-9, step_deg)
    lon_axis = np.arange(-180.0, 180.0, step_deg)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    return lat.ravel(), lon.ravel()


def refine_grid(lat0, lon0, radius=3.0, step=0.1):
    lat_axis = np.arange(max(-89.9, lat0 - radius), min(89.9, lat0 + radius) + 1e-9, step)
    lon_axis = np.arange(lon0 - radius, lon0 + radius + 1e-9, step)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat.ravel(), lon.ravel()


def gc_error(lat, lon, true_lat, true_lon):
    a = td.Station("a", np.zeros(3), float(lat), float(lon))
    b = td.Station("b", np.zeros(3), float(true_lat), float(true_lon))
    return td.great_circle_km(a, b)


def locate(y, ref_lat, ref_lon, true_lat, true_lon, family, rng):
    y0 = np.asarray(y, float) - np.mean(y)
    yn = np.linalg.norm(y0)
    if yn == 0:
        return None
    y0 /= yn
    lat, lon = candidate_grid()
    K = kernel_matrix(lat, lon, ref_lat, ref_lon, family)
    score = np.einsum("ij,j->i", K, y0, optimize=True)
    best = int(np.argmax(score))
    rlat, rlon = refine_grid(lat[best], lon[best])
    Kr = kernel_matrix(rlat, rlon, ref_lat, ref_lon, family)
    rscore = np.einsum("ij,j->i", Kr, y0, optimize=True)
    rb = int(np.argmax(rscore))

    # Equal-size, station-clustered geometry null.  Each column is a complete
    # permutation of the observed station fingerprint.
    Y = np.column_stack([rng.permutation(y0) for _ in range(N_NULL)])
    null_surface = np.einsum("ij,jk->ik", K, Y, optimize=True)
    null_score = np.max(null_surface, axis=0)
    null_idx = np.argmax(null_surface, axis=0)
    null_error = np.asarray([gc_error(lat[i], lon[i], true_lat, true_lon) for i in null_idx])
    error = gc_error(rlat[rb], rlon[rb], true_lat, true_lon)
    coarse_error = gc_error(lat[best], lon[best], true_lat, true_lon)
    truth_score = float(np.dot(kernel_matrix(np.array([true_lat]), np.array([true_lon]),
                                             ref_lat, ref_lon, family)[0], y0))
    return {
        "estimated_lat_deg": float(rlat[rb]), "estimated_lon_deg": float(rlon[rb]),
        "true_lat_deg": float(true_lat), "true_lon_deg": float(true_lon),
        "surface_error_km": float(error), "best_signed_correlation": float(rscore[rb]),
        "truth_signed_correlation": truth_score,
        "truth_score_percentile_among_global_grid": float(np.mean(score <= truth_score)),
        "geometry_null": {
            "n_permutations": N_NULL,
            "candidate_grid_step_deg": 2.0,
            "observed_comparison_error_km": float(coarse_error),
            "p_best_score_ge_observed": float((1 + np.sum(null_score >= score[best])) / (N_NULL + 1)),
            "p_localisation_error_le_observed": float((1 + np.sum(null_error <= coarse_error)) / (N_NULL + 1)),
            "median_localisation_error_km": float(np.median(null_error)),
        },
    }


def mode_signature(matrix, selector):
    """Normalize every Fourier mode across stations before combining modes."""
    z = np.asarray(matrix, float)[:, selector].copy()
    z -= np.mean(z, axis=0, keepdims=True)
    norm = np.linalg.norm(z, axis=0)
    keep = np.isfinite(norm) & (norm > 1e-12)
    if not np.any(keep):
        return None
    z = z[:, keep] / norm[keep]
    return np.mean(z, axis=1)


def estimate_separation(a, b):
    return gc_error(a["estimated_lat_deg"], a["estimated_lon_deg"],
                    b["estimated_lat_deg"], b["estimated_lon_deg"])


def main():
    src = json.loads(SOURCE.read_text())
    audit = json.loads(AUDIT_SOURCE.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Step 82 clock-metadata audit must pass before localisation")
    dates = td.available_dates("clk_cod_ops", 2024)
    probe = td.load_series("clk_cod_ops", dates[len(dates) // 2],
                           min_valid_frac=0.0, max_jumps=10_000)
    rng = np.random.default_rng(79001)
    sites = {}
    for site_name, arm in src["arms"].items():
        atomic, quartz = site_name.split("-")[1], site_name.split("-")[0]
        if atomic not in probe or quartz not in probe:
            continue
        records = arm.get("reference_fingerprints", {}).get("delta", [])
        records = [r for r in records if r["reference"] in probe and r["n_days"] >= 10]
        if len(records) < 30:
            continue
        target = probe[atomic]["station"]
        ref_lat = np.asarray([probe[r["reference"]]["station"].lat_deg for r in records])
        ref_lon = np.asarray([probe[r["reference"]]["station"].lon_deg for r in records])
        y = np.asarray([r["mean_phase_cos"] for r in records])
        fits = {family: locate(y, ref_lat, ref_lon, target.lat_deg, target.lon_deg,
                               family, rng) for family in ("exponential", "gaussian")}

        mode_records = {r["reference"]: r for r in
                        arm.get("frequency_resolved_delta_fingerprints", [])}
        common = [r["reference"] for r in records if r["reference"] in mode_records]
        mode_matrix = np.asarray([mode_records[n]["mean_phase_cos_by_frequency"] for n in common])
        mlat = np.asarray([probe[n]["station"].lat_deg for n in common])
        mlon = np.asarray([probe[n]["station"].lon_deg for n in common])
        nmodes = mode_matrix.shape[1]
        selectors = {"all_modes": np.ones(nmodes, bool),
                     "even_frequency_bins": np.arange(nmodes) % 2 == 0,
                     "odd_frequency_bins": np.arange(nmodes) % 2 == 1}
        multimode = {}
        for family in ("exponential", "gaussian"):
            multimode[family] = {}
            for label, selector in selectors.items():
                signature = mode_signature(mode_matrix, selector)
                multimode[family][label] = locate(
                    signature, mlat, mlon, target.lat_deg, target.lon_deg, family, rng)
            multimode[family]["odd_even_estimate_separation_km"] = estimate_separation(
                multimode[family]["odd_frequency_bins"],
                multimode[family]["even_frequency_bins"])
        temporal_half_localisation = {}
        for half, half_data in arm.get("temporal_halves", {}).items():
            half_modes = {r["reference"]: r for r in
                          half_data.get("frequency_resolved_delta_fingerprints", [])}
            half_common = [n for n in common if n in half_modes]
            if len(half_common) < 30:
                temporal_half_localisation[half] = None
                continue
            half_matrix = np.asarray([
                half_modes[n]["mean_phase_cos_by_frequency"] for n in half_common])
            half_sig = mode_signature(
                half_matrix, np.ones(half_matrix.shape[1], dtype=bool))
            hlat = np.asarray([probe[n]["station"].lat_deg for n in half_common])
            hlon = np.asarray([probe[n]["station"].lon_deg for n in half_common])
            temporal_half_localisation[half] = locate(
                half_sig, hlat, hlon, target.lat_deg, target.lon_deg,
                "exponential", rng)
            temporal_half_localisation[half]["n_reference_stations"] = len(half_common)
            temporal_half_localisation[half]["independent_time_partition"] = True
        pair_audit = audit["pairs"].get(site_name, {})
        sites[site_name] = {
            "n_reference_stations": len(records),
            "primary_free_atomic_anchor": bool(arm.get("anchor_metadata", {}).get("primary_anchor")),
            "receiver_model_matched_through_2024": pair_audit.get(
                "receiver_model_matched_through_2024"),
            "internal_oscillator_material_confirmed": pair_audit.get(
                "internal_material_confirmed"),
            "fits": fits,
            "multimode_fits": multimode,
            "temporal_half_multimode_exponential": temporal_half_localisation,
        }
        print(site_name, json.dumps({"scalar": fits, "multimode": multimode}, indent=1))

    primary = [v for v in sites.values() if v["primary_free_atomic_anchor"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "step": 80,
        "title": "Clock-field fingerprint localisation replay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixed_lambda_km": LAMBDA_KM,
        "target_coordinates_used_by_estimator": False,
        "target_coordinates_used_for_scoring_and_null_only": True,
        "clock_metadata_audit": str(AUDIT_SOURCE.relative_to(ROOT)),
        "source_observable": "internal-minus-external-atomic phase-cos fingerprint vs global reference clocks",
        "sites": sites,
        "summary": {
            "n_sites": len(sites), "n_primary_free_atomic_anchors": len(primary),
            "primary_exponential_error_km": [v["fits"]["exponential"]["surface_error_km"]
                                                for v in primary if v["fits"]["exponential"]],
            "primary_gaussian_error_km": [v["fits"]["gaussian"]["surface_error_km"]
                                             for v in primary if v["fits"]["gaussian"]],
            "primary_multimode_exponential_error_km": [
                v["multimode_fits"]["exponential"]["all_modes"]["surface_error_km"]
                for v in primary],
            "primary_temporal_half_multimode_exponential_error_km": {
                half: [v["temporal_half_multimode_exponential"][half]["surface_error_km"]
                       for v in primary
                       if v["temporal_half_multimode_exponential"].get(half)]
                for half in ("first_half", "second_half")
            },
        },
        "interpretation_boundary": "This is a blinded coordinate inversion of GNSS-derived precise-clock "
                                   "products. It tests clock-field spatial information and algorithmic "
                                   "observability; it is not a live satellite-free navigation demonstration.",
        "hardware_requirements_for_live_zero_satellite_test": [
            "co-located free-running atomic and explicitly characterised quartz outputs on the mobile platform",
            "direct local phase comparison independent of GNSS receiver clock estimation",
            "three or more independent field modes/gradients, or sequential IMU/barometer constraints",
            "a separately trained live anchor network and held-out route truth",
        ],
    }, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
