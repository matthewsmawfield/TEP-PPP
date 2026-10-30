"""
TEP-PPP Step 08: Signal Integrity & Spoofing Detection
======================================================

(Recovered from orphaned bytecode as ``step_08_integrity_monitoring``; its
log/figure artefacts keep the historical ``step_12_*`` stems — declared in
``scripts/audit_pipeline.py`` ARTEFACT_ALIASES.)

DISCLAIMER: SIMULATED ATTACK ON REAL DATA
-----------------------------------------
This script loads ACTUAL clock bias data from the IGS network (ZIMM, GRAS, etc.).
It then mathematically INJECTS a "Spoofing Ramp" (linear drift) into the ZIMM signal
to simulate a "Clock Drag-Off" attack.

Goal: Prove that the TEP Integrity Monitor can detect this synthetic anomaly because
it violates the physical correlation structure of the real network.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from core.ppp_kernel import DynamicTEPKernel
from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.plotting import setup_publication_style, save_publication_figure, get_color_palette, get_figure_size


def load_data(data_dir, date_str='2024100'):
    stations = ['ZIMM', 'GRAS', 'GENO', 'BRUX', 'WTZR']
    clocks = {}
    coords = {}
    for sta in stations:
        f = data_dir / f"{sta}_{date_str}.npz"
        if not f.exists():
            continue
        d = np.load(f)

        if 'precise_clock_bias_ns' in d:
            clk = d['precise_clock_bias_ns']
        elif 'clock_bias_ns' in d:
            clk = d['clock_bias_ns']
        else:
            continue

        x = d['x_mean'] if 'x_mean' in d else np.nanmean(d['ecef_x'])
        y = d['y_mean'] if 'y_mean' in d else np.nanmean(d['ecef_y'])
        z = d['z_mean'] if 'z_mean' in d else np.nanmean(d['ecef_z'])

        clocks[sta] = clk
        coords[sta] = np.array([x, y, z]) / 1000.0

    return clocks, coords


def main():
    log_dir = Path(__file__).resolve().parents[2] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger('step_12_integrity', log_file_path=log_dir / 'step_12_integrity_test.log')
    set_step_logger(logger)

    setup_publication_style()
    colors = get_color_palette()

    print_status('TEP-PPP Signal Integrity Proof', 'TITLE')
    print('WARNING: SIMULATED ATTACK ON REAL DATA')
    print('Goal: Prove detection of mathematically injected drift.')

    data_dir = Path(__file__).resolve().parents[2] / 'data' / 'ppp_processed'
    clocks, coords = load_data(data_dir)

    if 'ZIMM' not in clocks:
        print_status('Missing ZIMM data.', 'ERROR')
        return

    target = 'ZIMM'
    refs = [s for s in clocks if s != target]
    n_epochs = min(len(c) for c in clocks.values())

    # Observation cadence from the target file (seconds of day).
    sample_data = np.load(data_dir / f'{target}_2024100.npz')
    if 'sod' in sample_data and len(sample_data['sod']) > 1:
        avg_interval = float(np.median(np.diff(sample_data['sod'][:n_epochs])))
    else:
        avg_interval = 30.0
    t = np.arange(n_epochs) * avg_interval / 60.0

    attack_start = 100
    drift_rate = 0.5   # ns per epoch

    # Kernel noise model: the network-predictable spatial fraction is 0.44.
    ref_std = float(np.mean([
        np.std(clocks[r][:n_epochs] - np.mean(clocks[r][:n_epochs]))
        for r in refs
    ]))
    SPATIAL_FRACTION = 0.44
    spatial_std = ref_std * np.sqrt(SPATIAL_FRACTION)
    local_noise_var = ref_std ** 2 * (1.0 - SPATIAL_FRACTION)

    kernel = DynamicTEPKernel(
        base_lambda_km=4200.0,
        base_sigma_ns=spatial_std,
        enable_rotation_effects=False,
        enable_velocity_coupling=True,
        enable_local_time=True,
        velocity_coupling_factor=0.35,
        cmb_coupling_factor=0.0,
        diurnal_coupling_factor=0.1,
        seasonal_coupling_factor=-0.2,
        enable_chandler_wobble=True,
        chandler_coupling_factor=0.15,
        enable_semiannual_nutation=True,
        semiannual_nutation_coupling_factor=0.12,
        semiannual_nutation_phase_offset_days=60.0,
        enable_multiscale=False,
        lambda_short_km=4200.0,
        lambda_long_km=4200.0,
        multiscale_mixing_weight=0.4,
    )

    print_status('Establishing Baseline TEP Field...', 'PROCESS')

    normal_residuals = []

    # De-meaned clock series, referenced to the pre-attack mean.
    ref_vals = []
    for r in refs:
        ref_vals.append(clocks[r][:n_epochs] - np.mean(clocks[r][:attack_start]))
    ref_vals = np.array(ref_vals)
    ref_pos = np.array([coords[r] for r in refs])
    target_pos = coords[target].reshape(1, 3)

    target_mean = np.mean(clocks[target][:attack_start])
    target_true = clocks[target][:n_epochs] - target_mean

    preds = []

    from datetime import datetime, timedelta
    start_time = datetime(2024, 4, 9, 0, 0, 0)

    for k in range(n_epochs):
        epoch_time = start_time + timedelta(seconds=k * avg_interval)
        p, _, _ = kernel.predict(
            known_coords=ref_pos,
            known_values=ref_vals[:, k],
            target_coords=target_pos,
            noise_variance=local_noise_var,
            epoch_utc=epoch_time,
            reference_epoch=start_time,
        )
        preds.append(p[0])
        normal_residuals.append(target_true[k] - p[0])

    # Leave-one-out residual scale over the reference network: the honest
    # calibration of what a "normal" prediction residual looks like.
    ref_loo_residuals = []
    for i in range(len(refs)):
        others = [j for j in range(len(refs)) if j != i]
        if len(others) < 2:
            continue
        for k in range(n_epochs):
            ov = np.array([ref_vals[j, k] for j in range(len(refs)) if j != i])
            op = np.array([ref_pos[j] for j in range(len(refs)) if j != i])
            p, _, _ = kernel.predict(
                known_coords=op,
                known_values=ov,
                target_coords=ref_pos[i:i + 1],
                noise_variance=local_noise_var,
                epoch_utc=start_time + timedelta(seconds=k * avg_interval),
                reference_epoch=start_time,
            )
            ref_loo_residuals.append(ref_vals[i, k] - p[0])
    ref_loo_residuals = np.array(ref_loo_residuals)

    ref_med = np.median(ref_loo_residuals)
    ref_mad = 1.4826 * np.median(np.abs(ref_loo_residuals - ref_med)) + 1e-12
    normal_std = ref_mad
    print_status(f'Reference LOO residual MAD scale: {normal_std:.2f} ns', 'INFO')

    # False-alarm check on the pre-attack baseline.
    normal_residuals_arr = np.array(normal_residuals)
    pre_attack = normal_residuals_arr[:attack_start]
    n_false_alarms = int(np.sum(np.abs(pre_attack) > 3.0 * normal_std))
    if n_false_alarms > 0:
        print_status(
            f'WARNING: {n_false_alarms}/{len(pre_attack)} pre-attack residuals exceed 3-sigma threshold',
            'WARN')
    else:
        print_status(
            f'Pre-attack false-alarm check: 0/{len(pre_attack)} exceed threshold (clean)',
            'INFO')

    print_status('Injecting Clock Spoofing Attack...', 'PROCESS')

    spoofed_signal = target_true.copy()
    attack_ramp = np.zeros(n_epochs)
    attack_ramp[attack_start:] = np.arange(n_epochs - attack_start) * drift_rate
    spoofed_signal += attack_ramp

    tep_residuals = []
    detection_epoch = -1
    threshold = 3.0 * normal_std

    for k in range(n_epochs):
        res = abs(spoofed_signal[k] - preds[k])
        tep_residuals.append(res)
        if k > attack_start and res > threshold and detection_epoch < 0:
            detection_epoch = k

    print_status(f'Attack Start: Epoch {attack_start}', 'INFO')
    if detection_epoch > 0:
        delay = detection_epoch - attack_start
        drift_at_detection = float(attack_ramp[detection_epoch])
        residual_at_detection = float(tep_residuals[detection_epoch])
        print_status(f'TEP Detection: Epoch {detection_epoch} (Delay: {delay} epochs)', 'SUCCESS')
        print_status(
            f'Drift at detection: {drift_at_detection:.2f} ns (~{drift_at_detection * 0.299792458:.2f} m)',
            'SUCCESS')
        print_status(f'Residual at detection: {residual_at_detection:.2f} ns', 'INFO')
    else:
        print_status('TEP Failed to detect attack.', 'ERROR')
        drift_at_detection = None
        residual_at_detection = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=get_figure_size(height=8), constrained_layout=True)

    ax1.plot(t, target_true, color='black', label='True Signal', alpha=0.3, linewidth=1)
    ax1.plot(t, spoofed_signal, color=colors[4], linestyle='--', label='Spoofed Signal (Attack)')
    ax1.plot(t, preds, color=colors[0], label='TEP Prediction (Secure)', linewidth=2)
    ax1.axvline(t[attack_start], color='gray', linestyle=':', label='Attack Start')
    ax1.set_ylabel('Clock Bias (ns)')
    ax1.set_title('TEP Signal Integrity: Detecting a Clock Spoofing Attack')
    ax1.legend()

    ax2.plot(t, tep_residuals, color=colors[0], label='TEP Residual |Obs - Pred|')
    ax2.axhline(threshold, color=colors[4], linestyle='--', label='Detection Threshold (3$\\sigma$)')
    if detection_epoch > 0:
        ax2.scatter([t[detection_epoch]], [tep_residuals[detection_epoch]],
                    color=colors[4], s=100, zorder=10, label='Detection')
    ax2.set_ylabel('Residual Anomaly (ns)')
    ax2.set_xlabel('Time (minutes)')
    ax2.legend()

    out_dir = Path(__file__).resolve().parents[2] / 'results' / 'figures'
    save_publication_figure(fig, 'step_12_integrity_proof.png', out_dir)
    print_status(f'Saved integrity plot to {out_dir}/step_12_integrity_proof.png', 'SUCCESS')

    import json
    C_M_PER_NS = 0.299792458
    detection_delay_s = (detection_epoch - attack_start) * avg_interval if detection_epoch > 0 else None
    summary = {
        'test': 'integrity_monitoring_clock_dragoff',
        'description': 'Simulated slow clock drag-off attack on ZIMM; TEP network prediction residual crosses 3-sigma threshold',
        'target_station': target,
        'reference_stations': refs,
        'attack_start_epoch': attack_start,
        'attack_start_s': attack_start * avg_interval,
        'drift_rate_ns_per_epoch': drift_rate,
        'drift_rate_ns_per_s': drift_rate / avg_interval,
        'cadence_s': avg_interval,
        'normal_residual_mad_ns': float(normal_std),
        'normal_residual_std_ns': float(np.std(ref_loo_residuals)),
        'pre_attack_false_alarms': n_false_alarms,
        'pre_attack_n_epochs': len(pre_attack),
        'detection_threshold_ns': float(threshold),
        'detection_epoch': int(detection_epoch) if detection_epoch > 0 else None,
        'detection_delay_epochs': int(detection_epoch - attack_start) if detection_epoch > 0 else None,
        'detection_delay_s': detection_delay_s,
        'drift_at_detection_ns': drift_at_detection,
        'residual_at_detection_ns': residual_at_detection,
        'drift_at_detection_m': drift_at_detection * C_M_PER_NS if drift_at_detection is not None else None,
        'standard_raim_simulated': False,
        'standard_raim_note': 'Standard RAIM checks inter-satellite consistency; a clock spoof affects all satellites identically and is absorbed by the clock state, so RAIM cannot detect it. This is a theoretical argument, not a simulation.',
        'timestamp': datetime.now().isoformat(),
    }
    json_path = Path(__file__).resolve().parents[2] / 'results' / 'outputs' / 'step_08_integrity.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print_status(f'Saved integrity summary to {json_path}', 'SUCCESS')


if __name__ == '__main__':
    main()
