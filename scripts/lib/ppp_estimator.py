#!/usr/bin/env python3
"""
PPPEstimator: Static Extended Kalman Filter for Precise Point Positioning.

State vector: [x, y, z, clock_bias, amb_1, ..., amb_n]
  - x, y, z: ECEF position (meters)
  - clock_bias: receiver clock bias (meters)
  - amb_i: float ambiguity for satellite i (meters)

Process model: random-walk position + random walk clock + constant ambiguities
  (position process noise is small, modelling a static receiver with
  residual motion / multipath scattering; set pos_proc_noise=0 for a
  truly constant position)
Measurement model:
  - code: -los · pos + clock + noise
  - phase: -los · pos + clock + amb + noise
"""

import numpy as np

# TEP astronomical phase R² weights (Papers 1-2)
# Semiannual nutation: R² = 0.904 (Paper 2)
# Lunar nodal nutation: R² = 0.641 (Paper 2)
# Chandler wobble: R² = 0.424 (Paper 1, midpoint of 0.377-0.471)
_TEP_PHASE_R2 = {
    'semiannual': 0.904,
    'lunar_nodal': 0.641,
    'chandler': 0.424,
}
_TEP_PHASE_R2_TOTAL = sum(_TEP_PHASE_R2.values())
_TEP_PHASE_AMPLITUDE = 0.3  # 30% modulation of clock process noise


def tep_process_noise_scale(epoch_datetime,
                             enable_semiannual: bool = True,
                             enable_lunar_nodal: bool = True,
                             enable_chandler: bool = True,
                             amplitude: float = _TEP_PHASE_AMPLITUDE,
                             enable_lunar_synodic: bool = False,
                             enable_lunar_anomalistic: bool = False,
                             enable_lunar_draconic: bool = False,
                             enable_diurnal_sidereal: bool = False,
                             enable_diurnal_solar: bool = False,
                             enable_tidal: bool = False,
                             enable_planetary_gravity: bool = False,
                             enable_eclipse: bool = False) -> float:
    """Compute a multiplicative scale for clock process noise from TEP astronomical phases.

    The TEP astronomical phases (semiannual nutation, lunar nodal, Chandler wobble)
    are predetermined and were shown in Papers 1-2 to correlate with clock
    variability.  When the phases align constructively (cos > 0), clock
    variability is predicted to be higher and the process noise is increased;
    when destructive (cos < 0), it is decreased.

    The scale is centred at 1.0 with a +-30% modulation range, weighted by
    each phase's R² contribution.

    Extended phases (synodic, anomalistic, draconic, sidereal, solar) are
    included with a lower default R² weight (0.1) since they were not part of
    the original Papers 1-2 analysis but may carry additional information.
    Tidal and planetary gravity modulations use normalised acceleration ratios.
    """
    from datetime import datetime, timezone
    from scripts.lib.tep_astronomical import (
        nutation_phase_semiannual,
        nutation_phase_lunar_nodal,
        chandler_phase,
        lunar_phase_synodic,
        lunar_phase_anomalistic,
        lunar_phase_draconic,
        diurnal_phase_sidereal,
        diurnal_phase_solar,
        total_tidal_accel,
        stacked_planetary_gravity,
        eclipse_proximity,
    )
    if epoch_datetime is None:
        return 1.0
    if isinstance(epoch_datetime, (int, float)):
        epoch_datetime = datetime.fromtimestamp(epoch_datetime, tz=timezone.utc)
    # Core phases with R² weights from Papers 1-2
    r2_weights = dict(_TEP_PHASE_R2)
    phases = {}
    if enable_semiannual:
        phases['semiannual'] = nutation_phase_semiannual(epoch_datetime)
    if enable_lunar_nodal:
        phases['lunar_nodal'] = nutation_phase_lunar_nodal(epoch_datetime)
    if enable_chandler:
        phases['chandler'] = chandler_phase(epoch_datetime)
    # Extended lunar and diurnal phases (lower R² weight, exploratory)
    _EXT_R2 = 0.1
    if enable_lunar_synodic:
        phases['lunar_synodic'] = lunar_phase_synodic(epoch_datetime)
        r2_weights['lunar_synodic'] = _EXT_R2
    if enable_lunar_anomalistic:
        phases['lunar_anomalistic'] = lunar_phase_anomalistic(epoch_datetime)
        r2_weights['lunar_anomalistic'] = _EXT_R2
    if enable_lunar_draconic:
        phases['lunar_draconic'] = lunar_phase_draconic(epoch_datetime)
        r2_weights['lunar_draconic'] = _EXT_R2
    if enable_diurnal_sidereal:
        phases['diurnal_sidereal'] = diurnal_phase_sidereal(epoch_datetime)
        r2_weights['diurnal_sidereal'] = _EXT_R2
    if enable_diurnal_solar:
        phases['diurnal_solar'] = diurnal_phase_solar(epoch_datetime)
        r2_weights['diurnal_solar'] = _EXT_R2
    # Tidal acceleration modulation (normalised to typical ~1e-6 m/s²)
    if enable_tidal:
        try:
            tidal_a = total_tidal_accel(epoch_datetime)
            # Normalise: typical lunar+solar tidal ~1e-6 m/s², modulate ±10%
            tidal_scale = float(tidal_a) / 1e-6
            phases['tidal'] = 0.0  # placeholder, handled separately
            r2_weights['tidal'] = 0.05
        except Exception:
            pass
    # Planetary gravity modulation
    if enable_planetary_gravity:
        try:
            pg = stacked_planetary_gravity(epoch_datetime)
            pg_a = float(pg.get('accel_m_s2', 0.0))
            # Normalise: typical planetary ~1e-7 m/s²
            phases['planetary'] = 0.0
            r2_weights['planetary'] = 0.05
        except Exception:
            pass
    # Eclipse proximity modulation
    if enable_eclipse:
        try:
            ep = eclipse_proximity(epoch_datetime)
            near = float(ep.get('near_eclipse', False))
            phases['eclipse'] = 0.0
            r2_weights['eclipse'] = 0.05
        except Exception:
            pass
    if not phases:
        return 1.0
    # Normalise by the total R² of the ENABLED components only
    r2_total = sum(r2_weights.get(k, 0.0) for k in phases)
    if r2_total <= 0:
        return 1.0
    weighted_cos = sum(
        r2_weights.get(k, 0.0) * np.cos(phases[k]) for k in phases
        if k not in ('tidal', 'planetary', 'eclipse')
    )
    # Add non-cosine modulations (tidal, planetary, eclipse)
    extra = 0.0
    if 'tidal' in phases and enable_tidal:
        extra += r2_weights['tidal'] * (tidal_scale - 1.0)
    if 'eclipse' in phases and enable_eclipse:
        extra += r2_weights['eclipse'] * (1.0 if near else -1.0)
    weighted_cos = (weighted_cos + extra) / r2_total
    return 1.0 + amplitude * float(weighted_cos)


class PPPEstimator:
    """Static PPP Extended Kalman Filter."""

    def __init__(self, n_sats: int = 8, tep_mode: bool = False,
                 pos_proc_noise: float = 0.001,
                 clock_proc_noise: float = 10.0):
        self.n_sats = n_sats
        self.tep_mode = tep_mode
        self.n_state = 4 + n_sats  # pos(3) + clock + amb(n)

        # State vector
        self.x = np.zeros(self.n_state)
        self.x[0:3] = 0.0       # position
        self.x[3] = 0.0         # clock
        self.x[4:] = 0.0        # ambiguities

        # Covariance — moderate initial uncertainty
        self.P = np.eye(self.n_state)
        self.P[0:3, 0:3] *= 1e4    # position: ~100m
        self.P[3, 3] = 1e6          # clock: ~1km (large but not infinite)
        self.P[4:, 4:] *= 1e4       # ambiguities: ~100m

        # Process noise
        self.q_pos = pos_proc_noise ** 2
        self.q_clk = clock_proc_noise ** 2
        self.q_amb = 0.0  # ambiguities are constant

        self.wavelength = 0.1903  # GPS L1 wavelength (m)

    def predict(self, dt: float, tep_q_scale: float | None = None):
        """Time update (random-walk position and clock, constant ambiguities).

        When *tep_q_scale* is provided, the clock process noise is multiplied
        by this factor.  This allows the TEP astronomical phases to modulate
        the EKF's clock dynamics: at epochs where the phases predict higher
        clock variability, the scale > 1 (more process noise, looser clock);
        where lower, scale < 1 (tighter clock, faster convergence).
        """
        F = np.eye(self.n_state)
        # Static receiver: no state transition (F = I).
        # Position and clock receive process noise (random walk);
        # ambiguities are constant (no process noise).

        q_clk_eff = self.q_clk * tep_q_scale if tep_q_scale is not None else self.q_clk

        Q = np.zeros((self.n_state, self.n_state))
        Q[0:3, 0:3] = self.q_pos * dt
        Q[3, 3] = q_clk_eff * dt
        # Ambiguities: no process noise

        self.x = F @ self.x
        with np.errstate(all="ignore"):
            self.P = F @ self.P @ F.T + Q
        # Numerical guard: if P has become non-finite or indefinite, reset
        # the diagonal to a safe large value to prevent cascade failure.
        if not np.all(np.isfinite(self.P)):
            self.P = np.eye(self.n_state) * 1e4
            self.P[3, 3] = 1e6
        # Symmetrize and clamp tiny negative eigenvalues to zero
        self.P = 0.5 * (self.P + self.P.T)
        # Clamp extreme values to prevent overflow cascade
        np.clip(self.P, -1e12, 1e12, out=self.P)
        eigvals = np.linalg.eigvalsh(self.P)
        if eigvals[0] < -1e-9:
            self.P = self.P + (abs(eigvals[0]) + 1e-9) * np.eye(self.n_state)

    def update_gnss(self, code_obs, phase_obs, sat_los,
                    code_std: float = 3.0, phase_std: float = 0.05):
        """Measurement update with code and phase observations.

        Args:
            code_obs: (n_sats,) pseudorange residuals
            phase_obs: (n_sats,) carrier phase residuals
            sat_los: (n_sats, 3) satellite line-of-sight unit vectors
            code_std: code measurement noise std (m)
            phase_std: phase measurement noise std (m)
        """
        n = len(code_obs)
        if n == 0:
            return

        # Measurement matrix H: [−los, 1, I_amb]
        # For code: H_code = [−los, 1, 0]
        # For phase: H_phase = [−los, 1, I]
        H_code = np.zeros((n, self.n_state))
        H_code[:, 0:3] = -sat_los
        H_code[:, 3] = 1.0

        H_phase = np.zeros((n, self.n_state))
        H_phase[:, 0:3] = -sat_los
        H_phase[:, 3] = 1.0
        for i in range(n):
            H_phase[i, 4 + i] = 1.0

        # Predicted measurements
        code_pred = H_code @ self.x

        # Innovation
        code_innov = code_obs - code_pred

        # Measurement noise
        R_code = np.eye(n) * code_std ** 2
        R_phase = np.eye(n) * phase_std ** 2

        # Kalman gain for code: K = P @ H.T @ inv(S) = (inv(S) @ H @ P).T
        S_code = H_code @ self.P @ H_code.T + R_code
        with np.errstate(all="ignore"):
            try:
                K_code = np.linalg.solve(S_code, H_code @ self.P).T
            except np.linalg.LinAlgError:
                K_code = np.zeros((self.n_state, n))
        if not np.all(np.isfinite(K_code)):
            K_code = np.zeros((self.n_state, n))
        self.x = self.x + K_code @ code_innov

        # Joseph form update for code P
        I_KH = np.eye(self.n_state) - K_code @ H_code
        self.P = I_KH @ self.P @ I_KH.T + K_code @ R_code @ K_code.T

        # Kalman gain for phase: K = P @ H.T @ inv(S) = (inv(S) @ H @ P).T
        S_phase = H_phase @ self.P @ H_phase.T + R_phase
        with np.errstate(all="ignore"):
            try:
                K_phase = np.linalg.solve(S_phase, H_phase @ self.P).T
            except np.linalg.LinAlgError:
                K_phase = np.zeros((self.n_state, n))
        if not np.all(np.isfinite(K_phase)):
            K_phase = np.zeros((self.n_state, n))
        phase_innov = phase_obs - (H_phase @ self.x)
        self.x = self.x + K_phase @ phase_innov

        # Joseph form update for phase P
        I_KH_phase = np.eye(self.n_state) - K_phase @ H_phase
        self.P = I_KH_phase @ self.P @ I_KH_phase.T + K_phase @ R_phase @ K_phase.T

        # Symmetrize P
        self.P = 0.5 * (self.P + self.P.T)
        np.clip(self.P, -1e12, 1e12, out=self.P)
        # Clamp non-finite values
        if not np.all(np.isfinite(self.P)):
            self.P = np.eye(self.n_state) * 1e4
            self.P[3, 3] = 1e6

    def update_tep_mode(self, predicted_bias_m: float, prior_variance: float,
                         tep_pos_direction: np.ndarray | None = None,
                         tep_pos_variance: float | None = None):
        """Apply TEP prior on clock state and optionally on position.

        Args:
            predicted_bias_m: kriging-predicted clock bias in meters
            prior_variance: variance of the prediction in m^2
            tep_pos_direction: (3,) unit vector in ECEF for the TEP anisotropy
                preferred axis.  When provided with *tep_pos_variance*, a soft
                position constraint is applied along the perpendicular direction
                (where the clock field gradient is steepest, so the clock prior
                constrains position most strongly).
            tep_pos_variance: variance of the position constraint along the
                TEP-perpendicular direction (m^2).  Should be proportional to
                the clock prior variance — tighter clock prior implies tighter
                position constraint.
        """
        if prior_variance <= 0:
            return

        # --- Clock prior (scalar measurement on clock state) ---
        H_tep = np.zeros((1, self.n_state))
        H_tep[0, 3] = 1.0  # only clock state

        innov = predicted_bias_m - self.x[3]
        R_tep = np.array([[prior_variance]])
        S = H_tep @ self.P @ H_tep.T + R_tep
        with np.errstate(all="ignore"):
            try:
                K = np.linalg.solve(S, H_tep @ self.P).T
            except np.linalg.LinAlgError:
                return
        if not np.all(np.isfinite(K)):
            return

        # Decouple clock prior from ALL non-clock states: zero every
        # row of the Kalman gain except the clock row (index 3).
        _clk_gain = float(K[3, 0])
        K[:] = 0.0
        K[3, 0] = _clk_gain

        self.x = self.x + (K @ np.array([innov])).flatten()

        I_KH = np.eye(self.n_state) - K @ H_tep
        self.P = I_KH @ self.P @ I_KH.T + K @ R_tep @ K.T

        # Zero the clock cross-covariance with all non-clock states to
        # prevent a wrong prior from corrupting position/ambiguity through
        # the next measurement update's cross-covariance propagation.
        for j in range(self.n_state):
            if j != 3:
                self.P[3, j] = 0.0
                self.P[j, 3] = 0.0

        # --- TEP anisotropy position constraint ---
        # The TEP field is anisotropic: longer correlation along the preferred
        # axis, shorter perpendicular.  The clock prior constrains position
        # most strongly in the perpendicular direction (steepest gradient).
        # Only apply after the filter has partially converged (position
        # uncertainty < 10 m), to avoid interfering with early convergence
        # where the position estimate needs to move freely.
        if tep_pos_direction is not None and tep_pos_variance is not None and tep_pos_variance > 0:
            pos_uncertainty = float(np.sqrt(np.trace(self.P[0:3, 0:3])))
            if pos_uncertainty < 10.0:  # only after partial convergence
                d = np.asarray(tep_pos_direction, float)
                d = d / (np.linalg.norm(d) + 1e-12)
                # Perpendicular direction in the horizontal plane
                perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
                pn = np.linalg.norm(perp)
                if pn > 1e-6:
                    perp = perp / pn
                    H_pos = np.zeros((1, self.n_state))
                    H_pos[0, 0:3] = perp
                    pos_innov = 0.0  # zero-innovation regularization
                    R_pos = np.array([[tep_pos_variance]])
                    S_pos = H_pos @ self.P @ H_pos.T + R_pos
                    with np.errstate(all="ignore"):
                        try:
                            K_pos = np.linalg.solve(S_pos, H_pos @ self.P).T
                        except np.linalg.LinAlgError:
                            K_pos = None
                    if K_pos is not None and np.all(np.isfinite(K_pos)):
                        I_KH_pos = np.eye(self.n_state) - K_pos @ H_pos
                        self.P = I_KH_pos @ self.P @ I_KH_pos.T + K_pos @ R_pos @ K_pos.T

        self.P = 0.5 * (self.P + self.P.T)
        if not np.all(np.isfinite(self.P)):
            self.P = np.eye(self.n_state) * 1e4
            self.P[3, 3] = 1e6
