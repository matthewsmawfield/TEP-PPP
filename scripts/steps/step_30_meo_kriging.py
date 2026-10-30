#!/usr/bin/env python3
"""
Step 30: Dedicated MEO Spatio-Kinematic Kriging (Phase 3 Expansion)

This script proves the mathematical capability to perform epoch-by-epoch
Spatio-Kinematic prediction using real High-Velocity orbital kinematics.
Unlike static ground stations where the distance matrix `D` is computed once,
MEO GPS satellites travel at ~3 km/s. Their distance and relative velocity
vectors change every second.

Here, we compute the instantaneous distance matrix `D(t)` and velocity dot-product
matrix `V(t)` at each epoch and pass them into the upgraded `tep_kernel.kriging_weights`.

The predictand is the per-satellite detrended SP3 clock residual: at each epoch
we hold out one satellite and predict its clock residual from the others via
velocity-coupled kriging, then report the prediction RMSE.

NOTE: The raw SP3 clock biases are absolute values of order ~10^5 ns (milliseconds).
We detrend each satellite's clock series (linear + mean removal) before kriging
so the prediction target is the fluctuation around the satellite's own clock
trend, not the absolute bias. Without this detrending the RMSE is physically
meaningless (~100 km in range units).
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

def run_meo_spatiokinematic_prediction(year: int, doys: list[int]):
    print_status("--- TEP-PPP Phase 3: MEO Spatio-Kinematic Engine ---", "TITLE")
    
    # We will track 5 GPS satellites (MEO, ~3 km/s)
    prns = ["G01", "G02", "G03", "G04", "G05"]
    
    results_summary = {
        "description": "MEO Spatio-Kinematic Validation",
        "satellites": prns,
        "days": []
    }
    
    kernel_params = tep_kernel.KernelParams(alpha_v=0.35) # Velocity coupling activated
    
    total_successful_epochs = 0
    total_velocities = []
    all_pred_errors = []
    
    for doy in doys:
        print_status(f"Processing Year {year}, DOY {doy:03d}...", "INFO")
        series = tep_data.load_meo_series(year, doy, prns)
        
        if not series:
            print_status(f"  No MEO data available for {year}-{doy:03d}.", "WARNING")
            continue
            
        sats = list(series.keys())

        # Detrend each satellite's clock series: the raw SP3 clock is an
        # absolute bias of order ~10^5 ns. We remove the linear trend and
        # mean so the kriging target is the per-satellite clock fluctuation.
        for s in sats:
            clk = np.asarray(series[s]["resid"], float)
            t = np.asarray(series[s]["t"], float)
            ok = np.isfinite(clk) & np.isfinite(t)
            if ok.sum() >= 2:
                coeffs = np.polyfit(t[ok], clk[ok], 1)
                series[s]["resid"] = clk - np.polyval(coeffs, t)
            else:
                series[s]["resid"] = clk - np.nanmean(clk)
        
        # Build a common time grid: intersection of all satellite epoch sets.
        # load_meo_series filters unavailable epochs per satellite, so the
        # arrays can differ in length and alignment.  We must find the epochs
        # where ALL satellites have valid data.
        common_t = set(series[sats[0]]["t"].tolist())
        for s in sats[1:]:
            common_t &= set(series[s]["t"].tolist())
        common_t = sorted(common_t)
        
        if len(common_t) < 4:
            print_status(f"  Too few common epochs ({len(common_t)}) for {year}-{doy:03d}.", "WARNING")
            continue
        
        # Build per-satellite index maps: epoch_sod -> array_index
        sat_t_idx = {}
        for s in sats:
            t_map = {t: i for i, t in enumerate(series[s]["t"])}
            sat_t_idx[s] = t_map
        
        n_epoch = len(common_t)
        print_status(f"  Loaded {len(sats)} dynamic MEO satellites. {n_epoch} common epochs...", "INFO")
        
        successful_epochs = 0
        first_epoch_printed = False
        day_pred_errors = []
        
        for t_val in common_t:
            if len(sats) < 2:
                break
            
            # Rotate target satellite each epoch for cross-validation
            target_idx = successful_epochs % len(sats)
            target_name = sats[target_idx]
            ref_names = [s for s in sats if s != target_name]
            
            ti = sat_t_idx[target_name][t_val]
            target_pos = series[target_name]["station"].trajectory_km[ti]
            target_vel = series[target_name]["station"].velocity_kms[ti]
            target_clk = series[target_name]["resid"][ti]
            
            # Check for NaNs
            if np.any(np.isnan(target_pos)) or np.any(np.isnan(target_vel)) or not np.isfinite(target_clk):
                continue
            
            ref_pos = []
            ref_vel = []
            ref_clk = []
            valid_refs = []
            for n in ref_names:
                ri = sat_t_idx[n][t_val]
                rp = series[n]["station"].trajectory_km[ri]
                rv = series[n]["station"].velocity_kms[ri]
                rc = series[n]["resid"][ri]
                if not (np.any(np.isnan(rp)) or np.any(np.isnan(rv)) or not np.isfinite(rc)):
                    ref_pos.append(rp)
                    ref_vel.append(rv)
                    ref_clk.append(rc)
                    valid_refs.append(n)
                    
            if len(valid_refs) < 2:
                continue
                
            ref_pos = np.array(ref_pos)
            ref_vel = np.array(ref_vel)
            ref_clk = np.array(ref_clk)
            
            # D(t): Instantaneous Distance Matrix
            n_refs = len(valid_refs)
            D_rr = np.zeros((n_refs, n_refs))
            for i in range(n_refs):
                for j in range(n_refs):
                    D_rr[i, j] = np.linalg.norm(ref_pos[i] - ref_pos[j])
                    
            # target-ref distances
            D_rt = np.array([np.linalg.norm(target_pos - rp) for rp in ref_pos])
            
            # V(t): Instantaneous velocity-direction cosine similarity matrix.
            # The kernel expects v_hat_i . v_hat_j (range [-1, +1]), NOT the
            # relativistic v_i.v_j/c^2 (~1e-10, negligible).
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
            
            # Feed the dynamic matrices into the velocity-coupled kernel
            w = tep_kernel.kriging_weights(D_rr, D_rt, kernel_params, V_refs_refs=V_rr, V_refs_target=V_rt)
            
            # Prediction: kriging estimate of target clock residual
            pred_clk = float(np.dot(w, ref_clk))
            pred_err = pred_clk - target_clk
            day_pred_errors.append(pred_err)
            
            vel_mag = np.linalg.norm(target_vel)
            total_velocities.append(vel_mag)
            successful_epochs += 1
            
            if not first_epoch_printed:
                print_status(f"  Epoch t={t_val:.0f}s: Target {target_name} velocity = {vel_mag:.2f} km/s", "INFO")
                print_status(f"           Weights: {np.round(w, 5)}", "INFO")
                print_status(f"           Clock pred={pred_clk:.3f} ns, actual={target_clk:.3f} ns, err={pred_err:.3f} ns", "INFO")
                first_epoch_printed = True

        print_status(f"  Successfully executed prediction for {successful_epochs} epochs.", "SUCCESS")
        total_successful_epochs += successful_epochs
        
        if day_pred_errors:
            dpe = np.array(day_pred_errors)
            dpe = dpe[np.isfinite(dpe)]
            rmse_ns = float(np.sqrt(np.mean(dpe ** 2))) if len(dpe) else 0.0
            all_pred_errors.extend(dpe.tolist())
        else:
            rmse_ns = 0.0
        
        vel_slice = np.array(total_velocities[-successful_epochs:]) if successful_epochs > 0 else np.array([])
        vel_slice = vel_slice[np.isfinite(vel_slice)]
        results_summary["days"].append({
            "doy": doy,
            "successful_epochs": successful_epochs,
            "mean_velocity_kms": float(np.mean(vel_slice)) if len(vel_slice) else 0.0,
            "clock_pred_rmse_ns": rmse_ns
        })

    results_summary["total_epochs"] = total_successful_epochs
    tv = np.array(total_velocities) if total_velocities else np.array([])
    tv = tv[np.isfinite(tv)]
    results_summary["overall_mean_velocity_kms"] = float(np.mean(tv)) if len(tv) else 0.0
    ape = np.array(all_pred_errors) if all_pred_errors else np.array([])
    ape = ape[np.isfinite(ape)]
    results_summary["overall_clock_pred_rmse_ns"] = float(np.sqrt(np.mean(ape ** 2))) if len(ape) else 0.0
    
    # Save to JSON
    out_dir = Path(__file__).resolve().parents[2] / "results" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_30_meo_kriging.json"
    
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print_status(f"Results saved to {out_path}", "SUCCESS")
    print_status(f"Overall mean velocity: {results_summary['overall_mean_velocity_kms']:.2f} km/s", "INFO")
    print_status(f"Overall clock prediction RMSE: {results_summary['overall_clock_pred_rmse_ns']:.3f} ns", "INFO")
    print_status("------------------------------------------------------", "TITLE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEO Spatio-Kinematic Kriging")
    parser.add_argument("--year", type=int, default=2024, help="Year to process")
    parser.add_argument("--doys", type=str, default="100-106", help="Comma-separated or range of DOYs (e.g., 100-106)")
    args = parser.parse_args()
    
    doy_list = []
    for part in args.doys.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            doy_list.extend(range(start, end + 1))
        else:
            doy_list.append(int(part))
            
    run_meo_spatiokinematic_prediction(args.year, doy_list)
