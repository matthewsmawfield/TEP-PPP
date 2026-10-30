"""Estimate L1/L5 wide-lane biases directly from observation data.

When external L1/L5 WL bias products (CNES Bias-SINEX OSB) are unavailable,
the satellite WL biases can be estimated from the Melbourne-Wübbena (MW)
combination averaged over a long observation arc.  The MW combination is
geometry-free and ionosphere-free, so its average converges to the integer
wide-lane ambiguity plus the satellite+receiver WL biases.  The fractional
part of this average is the WL bias.

This is a standard technique used in PPP-AR when external bias products are
not available for the observed frequency combination (e.g. L1/L5 instead of
the standard L1/L2).

The estimated biases are embedded in a copy of a RINEX CLK file as COMMENT
records in the CNES WL-bias format so RTKLIB's RINEX header parser can load
them.
"""
from __future__ import annotations

import gzip
import logging
import math
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLIGHT = 299792458.0  # m/s
F_L1 = 1575.42e6  # Hz
F_L5 = 1176.45e6  # Hz
LAM_WL = CLIGHT / (F_L1 - F_L5)  # L1/L5 wide-lane wavelength ≈ 0.751 m


def _parse_rinex_obs(obs_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Parse a RINEX 3.x observation file and extract L1/L5 phase and code
    for GPS satellites.

    Returns {sat_id: [(epoch_sod, L1_cycles, L5_cycles, P1_m, P5_m), ...]}
    """
    from scripts.lib.rinex_pseudorange import _parse_rinex3_obs, _detect_rinex_version

    version = _detect_rinex_version(obs_path)
    if version == 3:
        raw = _parse_rinex3_obs(obs_path)
    else:
        logger.warning("RINEX version %d not supported for WL bias estimation", version)
        return {}

    # raw is {sod: {sat_id: {obs_code: value}}}
    # We need L1C (phase cycles), L5Q (phase cycles), C1C (code m), C5Q (code m)
    epochs: dict[str, list[tuple[float, float, float, float]]] = {}
    for sod, sats in raw.items():
        for sid, obs in sats.items():
            # GPS (G) and Galileo (E) both track L1+E5a (same frequencies
            # as GPS L1+L5), so the MW combination works identically.
            if not (sid.startswith("G") or sid.startswith("E")):
                continue
            l1 = obs.get("L1C")
            l5 = obs.get("L5Q") or obs.get("L5I") or obs.get("L5X")
            c1 = obs.get("C1C")
            c5 = obs.get("C5Q") or obs.get("C5I") or obs.get("C5X")
            if l1 and l5 and c1 and c5 and all(abs(v) > 0 for v in (l1, l5, c1, c5)):
                epochs.setdefault(sid, []).append((float(sod), float(l1), float(l5), float(c1), float(c5)))

    return epochs


def estimate_l1_l5_wl_fracs_raw(
    obs_path: Path,
) -> dict[str, tuple[float, int]]:
    """Per-satellite raw MW fractional biases (N_wl + b_sat + b_rx mod 1).

    Returns {sat_id: (frac_cycles, n_epochs)} for GPS+Galileo satellites.
    The raw fracs retain the receiver WL bias (common-mode) — they are the
    quantities suitable for cross-window stacking, since both the
    satellite FCBs and the receiver bias are instrument constants.
    """
    epochs = _parse_rinex_obs(obs_path)
    if not epochs:
        logger.warning("No GPS L1/L5 epochs parsed from %s", obs_path)
        return {}

    out: dict[str, tuple[float, int]] = {}
    for sid, obs_list in epochs.items():
        if len(obs_list) < 50:
            continue

        # Compute MW combination for each epoch.
        # MW = lam_WL * (L1 - L5) - (f1*P1 + f5*P5)/(f1+f5)
        # This is geometry-free and ionosphere-free; its average converges
        # to lam_WL * (N1 - N5 + wlbias) where N1, N5 are integer ambiguities.
        mw_values: list[float] = []
        prev_mw = None
        for sod, l1, l5, c1, c5 in obs_list:
            mw = LAM_WL * (l1 - l5) - (F_L1 * c1 + F_L5 * c5) / (F_L1 + F_L5)
            if not math.isfinite(mw):
                continue
            # Genuine cycle slip: MW jumps by many WL cycles (>50).
            # Code noise/multipath on the LC29H can cause ~30m MW jumps
            # per epoch, so only reset on very large jumps.
            if prev_mw is not None and abs(mw - prev_mw) > 50.0 * LAM_WL:
                mw_values = []
            mw_values.append(mw)
            prev_mw = mw

        if len(mw_values) < 50:
            continue

        # Robust average: iteratively reject outliers > 4 sigma from the mean.
        # This handles code multipath without needing a hard per-epoch threshold.
        arr = np.array(mw_values)
        for _ in range(3):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) + 1e-9
            sigma = 1.4826 * mad
            mask = np.abs(arr - med) < 4.0 * sigma
            if mask.sum() < 10:
                break
            arr = arr[mask]

        mean_mw = float(np.mean(arr))
        wl_n = mean_mw / LAM_WL
        frac = wl_n - math.floor(wl_n + 0.5)
        out[sid] = (frac, len(arr))

        logger.info("WL bias %s: mean_mw=%.4fm, N_wl=%.4f, frac=%.6f cycles "
                    "(n=%d/%d after outlier rejection)",
                    sid, mean_mw, wl_n, frac, len(arr), len(mw_values))

    logger.info("Estimated L1/L5 WL biases for %d GPS+Galileo satellites", len(out))
    return out


def estimate_l1_l5_wl_biases(obs_path: Path) -> dict[str, float]:
    """Estimate L1/L5 WL biases from observation data.

    Returns {sat_id: wl_bias_cycles} for GPS satellites.
    """
    raw = estimate_l1_l5_wl_fracs_raw(obs_path)
    if not raw:
        return {}

    wl_biases = {sid: fr for sid, (fr, _n) in raw.items()}

    # Remove the receiver WL bias (common-mode).  The MW combination gives
    # N_wl + b_sat + b_rx for each satellite.  The fractional part is
    # frac(b_sat + b_rx).  The satellite WL biases are zero-mean by
    # convention (IGS convention), so the mean of the estimated fractional
    # biases is approximately b_rx.  Subtracting the mean removes the
    # receiver bias, leaving only the satellite-specific biases.
    #
    # Without this correction, b_rx (which varies with receiver temperature
    # and clock state) propagates as a common-mode shift in all satellite
    # WL biases, causing a systematic position bias of 5-15 m in the
    # PPP-AR fixed solution.
    if len(wl_biases) >= 4:
        rx_bias = float(np.mean(list(wl_biases.values())))
        # Subtract receiver bias and re-normalise to [-0.5, 0.5]
        corrected = {}
        for sid, b in wl_biases.items():
            b_corr = b - rx_bias
            # Wrap to [-0.5, 0.5]
            while b_corr > 0.5:
                b_corr -= 1.0
            while b_corr < -0.5:
                b_corr += 1.0
            corrected[sid] = b_corr
        logger.info("Removed receiver WL bias: rx_bias=%.6f cycles "
                    "(mean of %d satellite biases)", rx_bias, len(wl_biases))
        wl_biases = corrected

    return wl_biases


def generate_wl_bias_clock_from_data(
    obs_path: Path,
    clk_in: Path,
    clk_out: Path,
    wl_biases: dict[str, float] | None = None,
) -> bool:
    """Estimate L1/L5 WL biases from observation data and embed in a clock file.

    When *wl_biases* is supplied (e.g. cross-window stacked values), those
    are written instead of the per-window estimates.

    Returns True if the WL bias clock file was successfully generated.
    """
    if wl_biases is None:
        wl_biases = estimate_l1_l5_wl_biases(obs_path)
    if len(wl_biases) < 4:
        logger.warning("Too few GPS+Galileo WL biases estimated (%d)", len(wl_biases))
        return False

    # Read the input clock file
    try:
        opener = gzip.open if clk_in.suffix in (".gz", ".Z") else open
        with opener(clk_in, "rb") as f:
            lines = f.read().decode("ascii", errors="ignore").splitlines()
    except Exception as e:
        logger.error("Failed to read clock file %s: %s", clk_in, e)
        return False

    # Get the observation date from the first WL bias record or the file
    obs_date_str = "2026  9  7"  # fallback
    # Try to extract from the clock file
    for line in lines:
        if line.startswith("AS ") and len(line.split()) >= 8:
            parts = line.split()
            try:
                obs_date_str = f"{int(parts[2]):4d} {int(parts[3]):2d} {int(parts[4]):2d}"
                break
            except (ValueError, IndexError):
                pass

    out_lines: list[str] = []
    bias_inserted = False
    skip_original_wl = False

    for line in lines:
        # Skip original WL bias COMMENT records (any satellite system)
        if line.startswith("WL ") and ("COMMENT" in line):
            skip_original_wl = True
            continue
        if skip_original_wl and line.startswith("WL "):
            continue
        skip_original_wl = False

        # Insert L1/L5 WL biases after "TIME SYSTEM ID"
        if not bias_inserted and "TIME SYSTEM ID" in line:
            out_lines.append(line)
            header = "WIDELANE SATELLITE FRACTIONAL BIASES USED IN THIS SOLUTION"
            out_lines.append(f"{header:60s}COMMENT")
            for sat_id in sorted(wl_biases.keys()):
                wsb = wl_biases[sat_id]
                prn = int(sat_id[1:])
                sys_char = sat_id[0]  # 'G' for GPS, 'E' for Galileo
                bias_str = f"{wsb:+.6E}"
                rec = (f"WL {sys_char}{prn:02d}  {obs_date_str} 12  0  0.000000"
                       f"  1   {bias_str}  0105")
                out_lines.append(f"{rec:60s}COMMENT")
            bias_inserted = True
            continue

        out_lines.append(line)

    if not bias_inserted:
        logger.warning("Could not find insertion point for WL biases")
        return False

    try:
        clk_out.write_text("\n".join(out_lines) + "\n")
        logger.info("Data-driven WL bias clock file written: %s (%d WL entries)",
                    clk_out, len(wl_biases))
        return True
    except Exception as e:
        logger.error("Failed to write WL bias clock: %s", e)
        return False
