#!/usr/bin/env python3
"""
PPPEstimatorKinematic: Kinematic PPP Extended Kalman Filter.

State vector: [x, y, z, vx, vy, vz, clock_bias, clock_rate, ztd, amb_1, ..., amb_n]
  - x, y, z: ECEF position (meters)
  - vx, vy, vz: velocity (m/s)
  - clock_bias: receiver clock bias (meters)
  - clock_rate: receiver clock drift rate (m/s, random walk).  The two-state
    clock model (bias + rate) is the standard GPS receiver-clock model: the
    rate state tracks the linear oscillator drift so the bias does not lag,
    which is what a single random-walk clock state cannot do.  Without it a
    receiver drifting at even 0.5 m/s (1.7 ppb, a mild quartz) accumulates a
    bias of ~15 m per 30 s epoch that the filter cannot follow, triggering
    repeated divergence resets and preventing carrier-phase convergence.
  - ztd: zenith tropospheric delay (metres, random walk)
  - amb_i: float ambiguity for satellite i (meters)

The ZTD state is essential for cm-level accuracy: the troposphere
contributes 2.3 m at zenith and up to 10 m at low elevation, and the
simplified Saastamoinen model in rinex_pseudorange.py only removes the
static part.  Estimating the residual ZTD as a random-walk state lets
the filter absorb the weather-dependent remainder, which is the
difference between metre-level and decimetre-level positioning.

Process model: constant velocity + acceleration noise + random walk clock
               + random walk ZTD
Measurement model:
  - code:  -los . pos + clock + ztd * mf(elev) + noise
  - phase: -los . pos + clock + ztd * mf(elev) + amb + noise

where mf(elev) = 1/sin(elev) is the troposphere mapping function.
"""

import math
import os
import numpy as np

# State indices (documented once so both this module and gps_web.py
# agree on where each state lives).
IDX_POS = 0        # 0:3
IDX_VEL = 3        # 3:6
IDX_CLK = 6        # 6
IDX_CLK_RATE = 7   # 7  (clock drift rate, m/s)
IDX_ZTD = 8        # 8
IDX_AMB = 9        # 9:9+n_sats


def tropo_mapping(elev_deg: float) -> float:
    """Troposphere mapping function (Black & Eisner / simplified Niell).

    Maps the zenith delay to a slant delay at the given elevation:

        mf(e) = 1.001 / sqrt(0.002001 + sin^2(e))

    accurate to better than 1% above about 5 degrees.  Capped at 5 degrees
    to avoid divergence at the horizon.

    This is the partial derivative the ZTD state is estimated through, so it
    MUST match the mapping used to remove the a-priori delay in
    ``rinex_pseudorange._tropo_correction``.  Both previously used the
    flat-Earth 1/sin(e) approximation, which is 8% high at 10 degrees and
    36% high at 5 degrees; the residual the ZTD state saw was then dominated
    by mapping-function error rather than by the wet delay it models.
    """
    e = max(math.radians(elev_deg), math.radians(5.0))
    return 1.001 / math.sqrt(0.002001 + math.sin(e) ** 2)


class PPPEstimatorKinematic:
    """Kinematic PPP Extended Kalman Filter."""

    def __init__(self, n_sats: int = 8, tep_mode: bool = False,
                 clock_proc_noise: float = 1.0,
                 accel_noise: float = 2.0,
                 ztd_proc_noise: float = 0.0001,
                 vel_damping: float = 0.0,
                 clock_rate_proc_noise: float = 1e-2,
                 amb_proc_noise: float = 0.0):
        self.n_sats = n_sats
        self.tep_mode = tep_mode
        self.n_state = 9 + n_sats  # pos(3)+vel(3)+clock+clock_rate+ztd+amb(n)

        # State vector
        self.x = np.zeros(self.n_state)

        # Covariance
        self.P = np.eye(self.n_state) * 100.0
        self.P[0:3, 0:3] *= 1e4    # position
        self.P[3:6, 3:6] *= 1e2    # velocity
        self.P[IDX_CLK, IDX_CLK] = 1e6        # clock
        # Clock rate (drift): a quartz oscillator drifts at up to ~100 ppb
        # (~0.03 m/s); allow an initial uncertainty of 0.3 m/s so the
        # filter can estimate the rate from the first few minutes of code
        # and SPP-clock observations before the carrier phase takes over.
        self.P[IDX_CLK_RATE, IDX_CLK_RATE] = 0.09
        # ZTD: the a-priori Saastamoinen model in rinex_pseudorange.py
        # already removes ~2.3 m at zenith, so the ZTD state models only
        # the residual (weather-dependent, typically < 0.05 m).  Start
        # very tight so the ZTD doesn't absorb clock or position error,
        # which are common-mode and poorly separable from ZTD with a
        # small single-frequency constellation.
        self.P[IDX_ZTD, IDX_ZTD] = 0.001      # ZTD residual: 0.03 m std
        self.P[IDX_AMB:, IDX_AMB:] = 1e6       # ambiguities

        # Process noise parameters
        self.accel_noise = accel_noise
        self.clock_proc_noise = clock_proc_noise
        self.q_clk = clock_proc_noise ** 2
        self.clock_rate_proc_noise = clock_rate_proc_noise
        self.q_clk_rate = clock_rate_proc_noise ** 2
        self.q_ztd = ztd_proc_noise ** 2   # ZTD random walk (m/sqrt(s))
        self.q_amb = amb_proc_noise ** 2   # Ambiguity random walk (m/sqrt(s))
        # Velocity damping factor (0 = no damping, 1 = fully damped per
        # second).  For a static receiver, damping=0.1-0.3 prevents the
        # velocity state from accumulating random walk noise that would
        # otherwise cause position to drift thousands of metres over a
        # 300 s window.  The constant-velocity model without damping is
        # only appropriate for vehicle dynamics.
        self.vel_damping = vel_damping

    def predict(self, dt: float, tep_q_scale: float | None = None):
        """Time update (constant velocity model with optional damping).

        When *tep_q_scale* is provided, the clock process noise is multiplied
        by this factor, allowing TEP astronomical phases to modulate clock
        dynamics.
        """
        F = np.eye(self.n_state)
        # Position += velocity * dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Clock bias += clock_rate * dt.  This is the two-state clock model:
        # the rate state carries the linear oscillator drift so the bias
        # does not lag, which is what a single random-walk clock cannot do.
        # Series-pin rate freeze (TEP_SERIES_RATE_FREEZE, default 1):
        # while the oracle clock-series pin is engaged the truth-level
        # series already carries the real drift, so propagating the
        # polyfit rate (fit to the raw-median wander, ~0.7 m/epoch
        # measured inter-pull drift on w1792286006/w1792290543) only
        # fights the pin — pull to series, rate-drag 0.7 m, pull again.
        # Freezing the propagation leaves the pull as the sole mover.
        _rate_freeze = (
            getattr(self, "_ser_active", False)
            and os.environ.get("TEP_SERIES_RATE_FREEZE", "1") not in
                ("0", "0.0", "false", "False"))
        F[IDX_CLK, IDX_CLK_RATE] = 0.0 if _rate_freeze else dt
        # Velocity damping: v *= (1 - damping*dt).  For a static receiver
        # this prevents the velocity state from drifting, which would
        # otherwise cause position to accumulate error linearly over time.
        if self.vel_damping > 0:
            decay = max(0.0, 1.0 - self.vel_damping * dt)
            F[3, 3] = decay
            F[4, 4] = decay
            F[5, 5] = decay

        q_clk_eff = self.q_clk * tep_q_scale if tep_q_scale is not None else self.q_clk

        # Process noise
        q_a = self.accel_noise ** 2
        Q = np.zeros((self.n_state, self.n_state))
        # Position/velocity process noise from acceleration
        for i in range(3):
            Q[i, i] = 0.25 * q_a * dt ** 4
            Q[i, 3 + i] = 0.5 * q_a * dt ** 3
            Q[3 + i, i] = 0.5 * q_a * dt ** 3
            Q[3 + i, 3 + i] = q_a * dt ** 2
        # Clock (bias) and clock rate: both random walks.  The bias process
        # noise is driven by the rate uncertainty (q_clk_rate * dt) plus the
        # direct bias noise (q_clk_eff * dt); the rate is a slow random walk.
        Q[IDX_CLK, IDX_CLK] = q_clk_eff * dt
        Q[IDX_CLK_RATE, IDX_CLK_RATE] = self.q_clk_rate * dt
        # ZTD: random walk (slowly varying weather-driven residual)
        Q[IDX_ZTD, IDX_ZTD] = self.q_ztd * dt
        # Ambiguities: small random walk to track time-varying satellite-
        # specific biases (stale CLK prediction errors drift at ~0.01-0.1
        # m/s).  Without this, the ambiguity covariance collapses to zero
        # and the ambiguities can't track the drifting biases.  The residual
        # biases then project into clock and position, causing the EKF to
        # diverge (observed: 200m divergence over 1800 epochs while RTKLIB
        # converges to 0.05m on the same data).  A small q_amb of 0.001
        # m²/s (0.03 m/sqrt(s)) allows slow tracking without preventing
        # ambiguity convergence.
        if self.n_sats > 0 and self.q_amb > 0:
            amb_q = self.q_amb  # m²/s — tracks time-varying sat-specific biases
            for i in range(IDX_AMB, self.n_state):
                Q[i, i] = amb_q * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Hard clip the absolute clock rate to a range that accommodates
        # both the physical quartz drift (~0.03 m/s) and the apparent drift
        # from stale ultra-rapid satellite clock predictions (up to ~0.5 m/s
        # over a 4-hour capture).  The clock rate is set by the caller's
        # polyfit EMA on the robust median clock, not by the Kalman update
        # (which zeros the clock-rate row).  This safety net catches any
        # residual transient that leaves the rate above the physical +
        # apparent drift range.
        self.x[IDX_CLK_RATE] = np.clip(self.x[IDX_CLK_RATE], -0.5, 0.5)

        # Numerical stability guards (mirrored from the analysis pipeline's
        # EKF).  Without these a single bad update can make P explode, which
        # makes the next Kalman gain enormous, which makes the next update
        # even worse — the classic positive-feedback divergence that turns
        # a 6 m solution into a 1.5 billion metre one over a few thousand
        # epochs.
        if not np.all(np.isfinite(self.P)) or not np.all(np.isfinite(self.x)):
            # Hard reset to a sane covariance — the state will be
            # re-initialised by the caller's clock reconciliation on the
            # next epoch that has raw measurements.
            self.P = np.eye(self.n_state) * 1e4
            self.P[IDX_CLK, IDX_CLK] = 1e6
            self.P[IDX_CLK_RATE, IDX_CLK_RATE] = 0.09
            self.P[IDX_ZTD, IDX_ZTD] = 0.001
            self.P[IDX_AMB:, IDX_AMB:] = np.eye(self.n_sats) * 1e4
            # Zero the velocity to stop drift propagation
            self.x[3:6] = 0.0
        # Clip covariance to prevent runaway growth
        np.clip(self.P, -1e10, 1e10, out=self.P)
        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        # Ensure positive definiteness.  The full eigvalsh check is O(n^3)
        # and was the dominant cost on the Pi (49x49 matrix, 3597 epochs x
        # 3+ arms = ~10000 eigvalsh calls per solve, ~40 min stall).  Use
        # a cheap O(n) diagonal check: if all diagonal elements are
        # positive and the matrix is symmetric, it is *likely* positive-
        # definite (diagonal positivity is necessary but not sufficient).
        # Only fall back to the expensive Cholesky/eigvalsh repair when
        # the diagonal check fails (rare — only on numerical divergence).
        diag_min = float(self.P.diagonal().min())
        if diag_min < 1e-12:
            try:
                eigvals = np.linalg.eigvalsh(self.P)
                if eigvals[0] < 1e-12:
                    self.P = self.P + (abs(eigvals[0]) + 1e-9) * np.eye(self.n_state)
            except np.linalg.LinAlgError:
                self.P = np.eye(self.n_state) * 1e4
                self.P[IDX_CLK, IDX_CLK] = 1e6
                self.P[IDX_CLK_RATE, IDX_CLK_RATE] = 0.09

    def update_gnss(self, code_obs, phase_obs, sat_los,
                    sat_indices=None,
                    code_std: float = 3.0, phase_std: float = 0.05,
                    phase_mask=None,
                    tropo_mf=None,
                    elevations=None,
                    if_factor: float = 1.0):
        """Measurement update with code and phase observations.

        Args:
            code_obs: (n_active,) pseudorange residuals
            phase_obs: (n_active,) carrier phase residuals
            sat_los: (n_active, 3) satellite line-of-sight unit vectors
            sat_indices: indices of active satellites into ambiguity state
            code_std: code measurement noise std (m) at zenith
            phase_std: phase measurement noise std (m) at zenith
            phase_mask: (n_active,) bool array — only sats with True get
                        a phase update.  If None, all sats get phase.
            tropo_mf: (n_active,) troposphere mapping factors (1/sin(elev)).
                      If None, ZTD state is not applied to the measurements.
            elevations: (n_active,) satellite elevations in degrees.  When
                        provided, measurement noise is scaled by
                        1/sin(elev) (low-elevation satellites have larger
                        atmospheric and multipath noise).  This is the
                        standard RTKLIB/RTKLIB-demo5 weighting model.
            if_factor: ionosphere-free noise amplification factor (~2.26
                       for L1/L5, ~2.55 for L1/L2).  Applied to code_std
                       when dual-frequency IF combination is used.
        """
        n = len(code_obs)
        if n == 0:
            return

        if sat_indices is None:
            sat_indices = np.arange(n)

        if tropo_mf is None:
            tropo_mf = np.zeros(n)

        # Elevation-dependent measurement weighting.
        # sigma(e) = sigma_0 * a / (sin(e) + b), where a and b are chosen
        # so that sigma(90°) = sigma_0 and sigma(15°) ≈ 2*sigma_0.
        # This is the RTKLIB model (a=1, b=0.003 for code; a=1, b=0.003
        # for phase).  Without this, a 10° satellite gets the same weight
        # as a zenith satellite, which corrupts the solution with
        # multipath and atmospheric noise.
        if elevations is not None:
            sin_e = np.sin(np.radians(np.clip(elevations, 5, 90)))
            weight_factor = 1.0 / (sin_e + 0.003)
            weight_factor = np.clip(weight_factor, 0.5, 10.0)
        else:
            weight_factor = np.ones(n)

        # Apply IF noise amplification to code noise
        code_std_eff = code_std * if_factor
        code_sigma = code_std_eff * weight_factor
        phase_sigma = phase_std * weight_factor

        # --- Code update for all satellites ---
        H_code = np.zeros((n, self.n_state))
        H_code[:, 0:3] = -sat_los
        H_code[:, IDX_CLK] = 1.0
        H_code[:, IDX_ZTD] = tropo_mf

        code_pred = H_code @ self.x
        code_innov = code_obs - code_pred
        R_code = np.diag(code_sigma ** 2)

        S_code = H_code @ self.P @ H_code.T + R_code
        K_code = np.linalg.solve(S_code, H_code @ self.P).T
        self.x = self.x + K_code @ code_innov

        I_KH = np.eye(self.n_state) - K_code @ H_code
        self.P = I_KH @ self.P @ I_KH.T + K_code @ R_code @ K_code.T

        # --- Phase update for satellites with valid phase only ---
        if phase_mask is None:
            phase_mask = np.ones(n, dtype=bool)
        n_phase = int(np.sum(phase_mask))
        if phase_std < 1e5 and n_phase > 0:
            los_ph = sat_los[phase_mask]
            si_ph = sat_indices[phase_mask]
            po_ph = phase_obs[phase_mask]
            mf_ph = tropo_mf[phase_mask]
            ph_sigma = phase_sigma[phase_mask]

            H_phase = np.zeros((n_phase, self.n_state))
            H_phase[:, 0:3] = -los_ph
            H_phase[:, IDX_CLK] = 1.0
            H_phase[:, IDX_ZTD] = mf_ph
            for i, si in enumerate(si_ph):
                H_phase[i, IDX_AMB + si] = 1.0

            phase_innov = po_ph - (H_phase @ self.x)
            R_phase = np.diag(ph_sigma ** 2)

            S_phase = H_phase @ self.P @ H_phase.T + R_phase
            K_phase = np.linalg.solve(S_phase, H_phase @ self.P).T
            self.x = self.x + K_phase @ phase_innov

            I_KH_phase = np.eye(self.n_state) - K_phase @ H_phase
            self.P = I_KH_phase @ self.P @ I_KH_phase.T + K_phase @ R_phase @ K_phase.T

        # Symmetrize P
        self.P = 0.5 * (self.P + self.P.T)

    def update_tep_mode(self, predicted_bias_m: float, prior_variance: float,
                         tep_pos_direction: np.ndarray | None = None,
                         tep_pos_variance: float | None = None):
        """Apply TEP prior on clock state and optionally on position.

        When *tep_pos_direction* and *tep_pos_variance* are provided, a soft
        position constraint is applied along the TEP-perpendicular direction
        (where the clock field gradient is steepest).

        The clock prior updates ONLY the clock state (and its covariance
        cross-terms), not position or velocity directly.  This decouples
        the clock prior from position: a wrong clock prior cannot corrupt
        position through the clock-position cross-covariance P[0:3, CLK].
        Position improvement comes from the measurement updates using the
        better-constrained clock, not from the prior directly moving
        position.  This is the key fix that prevents the TEP arm from
        diverging when the network residual is a mean over the holdout
        (correct on average but wrong at individual epochs).
        """
        if prior_variance <= 0:
            return

        H_tep = np.zeros((1, self.n_state))
        H_tep[0, IDX_CLK] = 1.0  # clock state

        innov = predicted_bias_m - self.x[IDX_CLK]
        R_tep = np.array([[prior_variance]])

        S = H_tep @ self.P @ H_tep.T + R_tep
        K = np.linalg.solve(S, H_tep @ self.P).T

        # Decouple clock prior from ALL non-clock states: zero every
        # row of the Kalman gain except the clock row.
        # The clock prior should ONLY constrain the clock state — it
        # must not shift position, velocity, ZTD, or ambiguities.
        #
        # Previously only K[0:3] and K[3:6] were zeroed, but the
        # ambiguity rows (K[AMB:]) were left active.  A wrong clock
        # prior shifted the ambiguities, which then corrupted position
        # through P[POS,AMB] in the next measurement update — the
        # remaining 15 m corruption after zeroing P[POS,CLK].
        _clk_gain = float(K[IDX_CLK, 0])
        K[:] = 0.0
        K[IDX_CLK, 0] = _clk_gain  # keep only the clock row

        self.x = self.x + (K @ np.array([innov])).flatten()

        I_KH = np.eye(self.n_state) - K @ H_tep
        self.P = I_KH @ self.P @ I_KH.T + K @ R_tep @ K.T

        # Zero the clock cross-covariance with ALL non-clock states after
        # the prior update.  A wrong prior shifts the clock state; the
        # Joseph form shrinks P[i,CLK] for every state i, which then
        # propagates the wrong shift to those states in the next
        # measurement update — corrupting position (through P[POS,CLK])
        # and ambiguities (through P[AMB,CLK]) by up to 25 m on a 15 m
        # wrong prior.
        #
        # Zeroing all P[i,CLK] (i != CLK) breaks this corruption path:
        # the wrong prior only affects the clock state.  The next
        # measurement update regenerates P[i,CLK] from the measurement
        # geometry and uses the reduced P[CLK,CLK] to better separate
        # clock from other states — so a CORRECT prior still helps
        # indirectly, just not through the prior's direct cross-covariance.
        for j in range(self.n_state):
            if j != IDX_CLK:
                self.P[IDX_CLK, j] = 0.0
                self.P[j, IDX_CLK] = 0.0

        # TEP anisotropy position constraint
        if tep_pos_direction is not None and tep_pos_variance is not None and tep_pos_variance > 0:
            d = np.asarray(tep_pos_direction, float)
            d = d / (np.linalg.norm(d) + 1e-12)
            perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
            pn = np.linalg.norm(perp)
            if pn > 1e-6:
                perp = perp / pn
                H_pos = np.zeros((1, self.n_state))
                H_pos[0, 0:3] = perp
                pos_innov = 0.0  # zero-innovation regularization
                R_pos = np.array([[tep_pos_variance]])
                S_pos = H_pos @ self.P @ H_pos.T + R_pos
                K_pos = np.linalg.solve(S_pos, H_pos @ self.P).T
                # With zero innovation, state unchanged but P shrinks
                I_KH_pos = np.eye(self.n_state) - K_pos @ H_pos
                self.P = I_KH_pos @ self.P @ I_KH_pos.T + K_pos @ R_pos @ K_pos.T

        self.P = 0.5 * (self.P + self.P.T)


class PPPEstimatorKinematicAugmented(PPPEstimatorKinematic):
    """Kinematic PPP EKF with TEP-constrained, physics-modulated clock process noise."""

    def __init__(self, n_sats: int = 8, tep_mode: bool = False,
                 clock_proc_noise: float = 1.0,
                 accel_noise: float = 2.0,
                 clock_proc_noise_unconstrained: float = 1e4):
        if tep_mode:
            super().__init__(n_sats=n_sats, tep_mode=tep_mode,
                             clock_proc_noise=clock_proc_noise,
                             accel_noise=accel_noise)
        else:
            super().__init__(n_sats=n_sats, tep_mode=tep_mode,
                             clock_proc_noise=clock_proc_noise_unconstrained,
                             accel_noise=accel_noise)

    def predict(self, dt: float, epoch_datetime=None):
        """Time update with TEP physics-modulated clock process noise."""
        from scripts.lib.ppp_estimator import tep_process_noise_scale
        tep_q_scale = None
        if self.tep_mode and epoch_datetime is not None:
            tep_q_scale = tep_process_noise_scale(epoch_datetime)
        super().predict(dt, tep_q_scale=tep_q_scale)
