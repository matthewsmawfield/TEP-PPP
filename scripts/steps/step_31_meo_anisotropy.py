#!/usr/bin/env python3
"""
Step 31: MEO Spatio-Kinematic Anisotropy (T2)

IMPORTANT CAVEAT (corrected 2026-09-01):
1. The velocity-coupling metric computed here is a KERNEL-INTERNAL geometric
   quantity: mean exp(alpha_v * v_hat_i . v_hat_j) over satellite pairs.  It
   does NOT use any clock-residual data.  It is a property of the GPS orbital
   geometry and the kernel parameter alpha_v, and would produce the same
   result regardless of whether a TEP field exists.  It is therefore NOT
   independent empirical evidence for anisotropy; it is a consistency check
   on the kernel's directional response.
2. Velocities are now computed in the INERTIAL frame
   (v_inertial = v_ecef + omega_earth x r_ecef), which is the correct
   quantity for a Sagnac/rotational anisotropy claim.  The previous ECEF
   approximation was documented but incorrect.
3. The target satellite is now iterated over ALL satellites (leave-one-out),
   rather than fixed as sats[0], making the analysis symmetric.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib import tep_data
from scripts.lib import tep_kernel
from scripts.utils.logger import print_status

# Earth angular velocity (rad/s), IERS Conventions 2010
OMEGA_EARTH = 7.292115e-5


def ecef_to_inertial_velocity(pos_km, vel_kms):
    """Convert ECEF velocity to inertial velocity: v_inertial = v_ecef + omega x r."""
    r_m = pos_km * 1000.0
    # omega = [0, 0, OMEGA_EARTH], so omega x r = [-Omega*y, +Omega*x, 0].
    omega_cross_r = np.array([-OMEGA_EARTH * r_m[1], OMEGA_EARTH * r_m[0], 0.0])
    return vel_kms + omega_cross_r / 1000.0


def ecef_to_enu_vectors(pos):
    """Returns local East and North unit vectors for an ECEF position."""
    x, y, z = pos
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p)
    
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    return east, north


def run_meo_anisotropy(year: int, doys: list[int]):
    print_status("--- TEP-PPP Phase 3: MEO Kinematic Anisotropy (T2) ---", "TITLE")
    
    prns = ["G01", "G02", "G03", "G04", "G05"]
    kernel_params = tep_kernel.KernelParams(alpha_v=0.35) 
    
    ew_weights = []
    ns_weights = []
    
    total_epochs = 0
    
    for doy in doys:
        print_status(f"Processing Year {year}, DOY {doy:03d}...", "INFO")
        series = tep_data.load_meo_series(year, doy, prns)
        
        if not series:
            continue
            
        sats = list(series.keys())
        
        # Build a common time grid: intersection of all satellite epoch sets.
        common_t = set(series[sats[0]]["t"].tolist())
        for s in sats[1:]:
            common_t &= set(series[s]["t"].tolist())
        common_t = sorted(common_t)
        
        if len(common_t) < 4:
            continue
        
        # Build per-satellite index maps: epoch_sod -> array_index
        sat_t_idx = {}
        for s in sats:
            t_map = {t: i for i, t in enumerate(series[s]["t"])}
            sat_t_idx[s] = t_map
        
        n_epoch = len(common_t)
        
        for t_val in common_t:
            if len(sats) < 2:
                break

            # Collect all satellite positions and INERTIAL velocities at this epoch
            epoch_data = {}
            for s in sats:
                ti = sat_t_idx[s][t_val]
                pos = series[s]["station"].trajectory_km[ti]
                vel_ecef = series[s]["station"].velocity_kms[ti]
                if np.any(np.isnan(pos)) or np.any(np.isnan(vel_ecef)):
                    continue
                vel_inertial = ecef_to_inertial_velocity(pos, vel_ecef)
                epoch_data[s] = {"pos": pos, "vel": vel_inertial, "vel_ecef": vel_ecef}

            if len(epoch_data) < 2:
                continue

            avail = list(epoch_data.keys())

            # Symmetric leave-one-out: iterate over every satellite as target
            for target_name in avail:
                ref_names = [s for s in avail if s != target_name]
                if len(ref_names) < 1:
                    continue

                target_pos = epoch_data[target_name]["pos"]
                target_vel = epoch_data[target_name]["vel"]

                ref_pos = np.array([epoch_data[n]["pos"] for n in ref_names])
                ref_vel = np.array([epoch_data[n]["vel"] for n in ref_names])
                
                n_refs = len(ref_names)
                D_rr = np.zeros((n_refs, n_refs))
                for i in range(n_refs):
                    for j in range(n_refs):
                        D_rr[i, j] = np.linalg.norm(ref_pos[i] - ref_pos[j])
                        
                D_rt = np.array([np.linalg.norm(target_pos - rp) for rp in ref_pos])
                
                # Velocity-direction cosine similarity (inertial frame)
                V_rr = np.zeros((n_refs, n_refs))
                for i in range(n_refs):
                    vi = ref_vel[i]
                    ni = np.linalg.norm(vi)
                    vi_hat = vi / ni if ni > 1e-9 else vi
                    for j in range(n_refs):
                        vj = ref_vel[j]
                        nj = np.linalg.norm(vj)
                        vj_hat = vj / nj if nj > 1e-9 else vj
                        V_rr[i, j] = np.dot(vi_hat, vj_hat)

                nt = np.linalg.norm(target_vel)
                target_hat = target_vel / nt if nt > 1e-9 else target_vel
                V_rt = np.array([np.dot(target_hat, rv / np.linalg.norm(rv) if np.linalg.norm(rv) > 1e-9 else rv)
                                 for rv in ref_vel])
                
                # Velocity-coupling factor: the multiplier that the velocity
                # coupling applies to the target-reference covariance.
                coupling_factor = float(np.mean(np.exp(kernel_params.alpha_v * (np.clip(V_rt, -1.0, 1.0) - 1.0))))

                # Decompose ECEF velocity into EW and NS components.
                # ENU basis is Earth-fixed, so we project the ECEF velocity
                # (not the inertial velocity) to avoid a frame mismatch.
                east_vec, north_vec = ecef_to_enu_vectors(target_pos)
                v_ew = np.abs(np.dot(epoch_data[target_name]["vel_ecef"], east_vec))
                v_ns = np.abs(np.dot(epoch_data[target_name]["vel_ecef"], north_vec))

                if v_ew > v_ns * 1.5:  # Strongly East-West
                    ew_weights.append(coupling_factor)
                elif v_ns > v_ew * 1.5: # Strongly North-South
                    ns_weights.append(coupling_factor)
                    
                total_epochs += 1

    if len(ew_weights) == 0 or len(ns_weights) == 0:
        print_status("Insufficient anisotropic epochs extracted.", "ERROR")
        return

    ew_mean = float(np.nanmean(ew_weights))
    ns_mean = float(np.nanmean(ns_weights))
    if ns_mean == 0.0 or not np.isfinite(ns_mean):
        print_status("NS mean coupling is zero or NaN; cannot compute anisotropy ratio.", "ERROR")
        return
    anisotropy_ratio = ew_mean / ns_mean

    # Bootstrap CI for the ratio
    rng = np.random.default_rng(20260901)
    n_boot = 1000
    boot_ratios = []
    ew_arr = np.array(ew_weights)
    ns_arr = np.array(ns_weights)
    for _ in range(n_boot):
        ew_s = rng.choice(ew_arr, size=len(ew_arr), replace=True)
        ns_s = rng.choice(ns_arr, size=len(ns_arr), replace=True)
        ns_m = float(np.nanmean(ns_s))
        r = float(np.nanmean(ew_s) / ns_m) if ns_m > 0 else np.nan
        if np.isfinite(r):
            boot_ratios.append(r)
    ci_lo = float(np.percentile(boot_ratios, 2.5)) if boot_ratios else np.nan
    ci_hi = float(np.percentile(boot_ratios, 97.5)) if boot_ratios else np.nan
    p_le_1 = float(np.mean(np.array(boot_ratios) <= 1.0)) if boot_ratios else np.nan

    print_status(f"Analyzed {total_epochs} kinematic epochs (symmetric LOO, inertial frame).", "INFO")
    print_status(f"EW-Dominant Flight Mean Coupling: {ew_mean:.6f}", "INFO")
    print_status(f"NS-Dominant Flight Mean Coupling: {ns_mean:.6f}", "INFO")
    print_status(f"Kinematic Anisotropy Ratio (EW/NS): {anisotropy_ratio:.3f}", "INFO")
    print_status(f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  p(ratio<=1)={p_le_1:.3f}", "INFO")

    t2_pass = bool(anisotropy_ratio > 1.0 and p_le_1 < 0.05)

    if t2_pass:
        print_status("VERDICT: T2 Anisotropy PASSED via MEO Kinematics.", "SUCCESS")
    else:
        print_status("VERDICT: T2 Anisotropy NOT PASSED.", "WARNING")

    results_summary = {
        "description": "MEO Spatio-Kinematic Anisotropy (T2) — velocity-coupling factor metric (CORRECTED: inertial frame, symmetric LOO)",
        "metric": "mean exp(alpha_v * v_hat_target . v_hat_ref) over reference satellites (leave-one-out, inertial velocity)",
        "caveat": ("This metric is a kernel-internal geometric quantity computed from satellite "
                   "velocities only; it does NOT use clock-residual data and is NOT independent "
                   "empirical evidence for anisotropy. It characterises the kernel's directional "
                   "response to the GPS orbital geometry."),
        "velocity_frame": "inertial (v_ecef + omega_earth x r_ecef)",
        "target_assignment": "symmetric leave-one-out over all satellites",
        "total_epochs": total_epochs,
        "ew_epochs": len(ew_weights),
        "ns_epochs": len(ns_weights),
        "ew_mean_coupling": ew_mean,
        "ns_mean_coupling": ns_mean,
        "anisotropy_ratio": anisotropy_ratio,
        "ci95": [ci_lo, ci_hi],
        "p_ratio_le_1": p_le_1,
        "t2_pass": t2_pass
    }
    
    out_dir = Path(__file__).resolve().parents[2] / "results" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_31_meo_anisotropy.json"
    
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print_status(f"Results saved to {out_path}", "SUCCESS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doys", type=str, default="100-106")
    args = parser.parse_args()
    
    doy_list = []
    for part in args.doys.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            doy_list.extend(range(start, end + 1))
        else:
            doy_list.append(int(part))
            
    run_meo_anisotropy(args.year, doy_list)
