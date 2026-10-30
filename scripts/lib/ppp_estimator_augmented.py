#!/usr/bin/env python3
"""
PPPEstimatorAugmented: Static PPP EKF with TEP-augmented clock process noise.

When tep_mode=True, the clock process noise is reduced (reflecting the TEP
spatial constraint) and modulated per-epoch by the TEP astronomical phases
(Chandler wobble, semiannual nutation, lunar nodal nutation).  The phase
modulation feeds TEP temporal physics directly into the EKF dynamics:
at epochs where the phases predict higher clock variability, the process
noise is increased; where lower, it is decreased.
"""

import numpy as np
from scripts.lib.ppp_estimator import PPPEstimator, tep_process_noise_scale


class PPPEstimatorAugmented(PPPEstimator):
    """PPP EKF with TEP-constrained, physics-modulated clock process noise."""

    def __init__(self, n_sats: int = 8, tep_mode: bool = False,
                 pos_proc_noise: float = 0.01,
                 clock_proc_noise_std: float = 1.0,
                 clock_proc_noise_unconstrained: float = 1e4):
        super().__init__(n_sats=n_sats, tep_mode=tep_mode,
                         pos_proc_noise=pos_proc_noise,
                         clock_proc_noise=clock_proc_noise_unconstrained)
        if tep_mode:
            # Constrained clock: much smaller process noise
            self.q_clk = clock_proc_noise_std ** 2
        # else: keep the large unconstrained value from parent

    def predict(self, dt: float, epoch_datetime=None):
        """Time update with TEP physics-modulated clock process noise.

        When tep_mode=True and *epoch_datetime* is provided, the clock process
        noise is scaled by the TEP astronomical phase factor.  This feeds the
        temporal structure of TEP (Chandler, semiannual, lunar nodal) directly
        into the EKF dynamics model.
        """
        tep_q_scale = None
        if self.tep_mode and epoch_datetime is not None:
            tep_q_scale = tep_process_noise_scale(epoch_datetime)
        super().predict(dt, tep_q_scale=tep_q_scale)
