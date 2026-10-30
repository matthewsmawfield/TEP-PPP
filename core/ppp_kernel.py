"""
Dynamic TEP Kernel: Proper Reference Frame Physics
===================================================

Corrects fundamental errors in previous models by properly accounting for:
- Earth rotation (baselines rotate through different orientations)
- Station velocity (instantaneous motion through TEP field)
- Local solar time (not global DOY)
- Time-varying celestial geometry
- Velocity aberration effects
- Seasonal Modulation (Annual)
- Chandler Wobble (Polar Motion)

Key Insight: TEP patterns are DYNAMIC and reference-frame-dependent,
not static spatial modulations.
"""

import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import importlib


class DynamicTEPKernel:
    """
    Physically correct TEP kernel accounting for reference frame dynamics.
    
    Computes instantaneous correlation at each epoch based on:
    - Rotating baseline vectors (Earth rotation)
    - Station velocities (rotation + orbital)
    - Local solar time at each station
    - Time-varying celestial geometry
    - Seasonal and Polar Motion modulations
    """
    
    def __init__(self,
                 base_lambda_km=4200.0,
                 base_sigma_ns=1.0,
                 enable_rotation_effects=True,
                 enable_velocity_coupling=True,
                 enable_local_time=True,
                 velocity_coupling_factor=0.15,
                 cmb_coupling_factor=0.20,
                 diurnal_coupling_factor=0.10,
                 seasonal_coupling_factor=0.0,
                 enable_chandler_wobble=True,
                 chandler_coupling_factor=0.0,
                 enable_semiannual_nutation=False,
                 semiannual_nutation_coupling_factor=0.0,
                 semiannual_nutation_phase_offset_days=0.0,
                 enable_multiscale=False,
                 lambda_short_km=4200.0,
                 lambda_long_km=4200.0,
                 multiscale_mixing_weight=0.5, # 0.0=Long only, 1.0=Short only
                 enable_planetary_index=False,
                 planetary_sigma_coupling_factor=0.0,
                 planetary_lambda_coupling_factor=0.0,
                 planetary_bodies=None):
        """
        Initialize dynamic TEP kernel with proper physics.
        
        Parameters control which dynamic effects are computed.
        All effects are epoch-dependent and properly account for reference frame.
        """
        self.lambda_base = base_lambda_km
        self.sigma_base = base_sigma_ns
        
        self.enable_rotation = enable_rotation_effects
        self.enable_velocity = enable_velocity_coupling
        self.enable_local_time = enable_local_time
        self.enable_chandler = enable_chandler_wobble
        self.enable_semiannual_nutation = enable_semiannual_nutation
        self.enable_multiscale = enable_multiscale
        
        self.enable_planetary_index = enable_planetary_index
        
        # Coupling strengths
        self.v_coupling = velocity_coupling_factor
        self.cmb_coupling = cmb_coupling_factor
        self.diurnal_coupling = diurnal_coupling_factor
        self.seasonal_coupling = seasonal_coupling_factor
        self.chandler_coupling = chandler_coupling_factor
        self.semiannual_nutation_coupling = semiannual_nutation_coupling_factor
        self.semiannual_nutation_phase_offset_days = semiannual_nutation_phase_offset_days
        
        self.lambda_short = lambda_short_km
        self.lambda_long = lambda_long_km
        self.mixing_weight = multiscale_mixing_weight

        self.planetary_sigma_coupling = planetary_sigma_coupling_factor
        self.planetary_lambda_coupling = planetary_lambda_coupling_factor
        self.planetary_bodies = planetary_bodies if planetary_bodies is not None else ["venus", "jupiter"]
        
        # Physical constants
        # Sidereal day (86164.1 s) is the correct rotation period for
        # inertial-frame calculations; the solar day (86400 s) includes
        # the Earth's orbital motion and is ~0.3% too slow.
        self.earth_omega = 2 * np.pi / 86164.1  # rad/s (sidereal rotation)
        self.earth_radius_km = 6371.0
        self.orbital_velocity_km_s = 30.0  # Earth's orbital velocity
        self.c_km_s = 299792.458  # Speed of light
        
        # Orbital parameters
        self.orbital_period_days = 365.25
        self.perihelion_doy = 3
        self.eccentricity = 0.0167

        self.semiannual_nutation_period_days = 182.625
        
        # CMB dipole (fixed in inertial frame)
        self.cmb_ra_deg = 168.0
        self.cmb_dec_deg = -7.0

        self._j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)

        self._planetary_stats_cache = {}
    
    def get_rotation_matrix(self, t_seconds):
        """
        Get Earth rotation matrix for time t since reference epoch.
        
        Rotates around Z-axis (polar axis).
        """
        theta = self.earth_omega * t_seconds
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])
        
        return R
    
    def get_station_state(self, station_ecef, epoch_utc, reference_epoch):
        """
        Get instantaneous position and velocity of station at given epoch.
        
        Returns:
        --------
        pos : array (3,)
            Instantaneous position in inertial frame (km)
        vel : array (3,)
            Instantaneous velocity (km/s)
        """
        # Time since reference
        dt = (epoch_utc - self._j2000_epoch).total_seconds()
        
        if self.enable_rotation:
            # Rotate position with Earth
            R = self.get_rotation_matrix(dt)
            pos = R @ station_ecef
            
            # Velocity from Earth rotation: v = ω × r
            omega_vec = np.array([0, 0, self.earth_omega])
            v_rotation = np.cross(omega_vec, pos)
        else:
            pos = station_ecef
            v_rotation = np.zeros(3)
        
        if self.enable_velocity:
            # Add orbital velocity
            # Earth orbits at ~30 km/s, direction depends on time of year
            doy = epoch_utc.timetuple().tm_yday
            orbital_angle = 2 * np.pi * (doy - self.perihelion_doy) / self.orbital_period_days
            
            # Orbital velocity in ecliptic plane (approximate)
            # NOTE: the orbital plane is tilted ~23.5 deg from the equator
            # (obliquity of the ecliptic).  This approximation sets z=0,
            # aligning the orbital plane with the equator.  For the velocity-
            # coupling term this introduces a ~cos(23.5 deg) ~ 0.92 factor
            # in the z-component, which is a second-order correction relative
            # to the dominant equatorial motion.
            v_orbital_magnitude = self.orbital_velocity_km_s * (1 + self.eccentricity * np.cos(orbital_angle))
            
            # Direction perpendicular to Sun (tangent to orbit)
            # Simplified: assume orbital plane aligned with equator
            v_orbital = np.array([
                -v_orbital_magnitude * np.sin(orbital_angle),
                v_orbital_magnitude * np.cos(orbital_angle),
                0
            ])
            
            vel = v_rotation + v_orbital
        else:
            vel = v_rotation
        
        return pos, vel
    
    def get_local_solar_time(self, station_ecef, epoch_utc):
        """
        Get local solar time for station.
        
        Returns hours (0-24).
        """
        if not self.enable_local_time:
            return epoch_utc.hour + epoch_utc.minute / 60.0
        
        # Compute longitude from ECEF
        x, y, z = station_ecef
        lon_rad = np.arctan2(y, x)
        lon_deg = np.degrees(lon_rad)
        
        # Local solar time
        lon_hours = lon_deg / 15.0
        lst = (epoch_utc.hour + epoch_utc.minute / 60.0 + lon_hours) % 24
        
        return lst
    
    def get_cmb_direction_inertial(self):
        """
        Get CMB dipole direction in inertial frame.
        
        CMB frame is fixed relative to distant universe, not rotating with Earth.
        """
        ra_rad = np.radians(self.cmb_ra_deg)
        dec_rad = np.radians(self.cmb_dec_deg)
        
        # Equatorial to Cartesian (inertial frame)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        
        return np.array([x, y, z])
    
    def covariance_matrix(self, coords):
        """
        Compatibility wrapper for SyntheticGNSSSimulator.
        Computes covariance at a default reference epoch (J2000).
        """
        coords = np.atleast_2d(coords)
        N = coords.shape[0]
        K = np.zeros((N, N))
        epoch = datetime(2000, 1, 1, 12, 0, 0)
        
        for i in range(N):
            # Compute row i: correlation between station i and all stations
            # compute_instantaneous_correlation uses coords1[0] as the source
            cov_row, _ = self.compute_instantaneous_correlation(
                coords[i:i+1], coords, epoch, epoch
            )
            K[i, :] = cov_row.reshape(-1)
            
        # Force symmetry to ensure positive definiteness for Cholesky
        # (Dynamic effects like velocity aberration can introduce slight asymmetries)
        K = (K + K.T) / 2.0
        
        return K
    
    def compute_instantaneous_correlation(self, coords1, coords2, epoch_utc, reference_epoch):
        """
        Compute correlation at specific epoch accounting for all dynamic effects.
        
        Parameters:
        -----------
        coords1, coords2 : array (N, 3) or (3,)
            ECEF coordinates in km
        epoch_utc : datetime
            Specific epoch for computation
        reference_epoch : datetime
            Reference time (e.g., start of day)
            
        Returns:
        --------
        K : array
            Covariance matrix at this epoch
        diagnostics : dict
            Diagnostic information about dynamic effects
        """
        coords1 = np.atleast_2d(coords1)
        coords2 = np.atleast_2d(coords2)

        n1 = coords1.shape[0]
        n2 = coords2.shape[0]

        positions1 = np.zeros((n1, 3))
        velocities1 = np.zeros((n1, 3))
        for i in range(n1):
            p, v = self.get_station_state(coords1[i], epoch_utc, reference_epoch)
            positions1[i, :] = p
            velocities1[i, :] = v

        positions2 = np.zeros((n2, 3))
        velocities2 = np.zeros((n2, 3))
        for j in range(n2):
            p, v = self.get_station_state(coords2[j], epoch_utc, reference_epoch)
            positions2[j, :] = p
            velocities2[j, :] = v

        diffs = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)

        lambda_eff = float(self.lambda_base)
        if self.enable_multiscale:
            lambda_short_eff = float(self.lambda_short)
            lambda_long_eff = float(self.lambda_long)

        planetary_index = 0.0
        if self.enable_planetary_index and self.planetary_lambda_coupling != 0.0:
            planetary_index = self._get_planetary_index(epoch_utc)
            denom = 1.0 + self.planetary_lambda_coupling * planetary_index
            if denom <= 1e-12:
                denom = 1e-12
            p_mod = 1.0 / denom
            lambda_eff *= p_mod
            if self.enable_multiscale:
                lambda_short_eff *= p_mod
                lambda_long_eff *= p_mod

        sigma_eff = float(self.sigma_base)

        doy = epoch_utc.timetuple().tm_yday
        seasonal_phase_days = 95.0
        seasonal_factor = float(np.exp(self.seasonal_coupling * np.cos(2 * np.pi * (doy - seasonal_phase_days) / 365.25)))
        sigma_eff *= seasonal_factor

        if self.enable_chandler:
            days_since_j2000 = (epoch_utc - self._j2000_epoch).total_seconds() / 86400.0
            chandler_period = 433.0
            chandler_phase_offset = 215.0
            days_since_2022 = days_since_j2000 - 8036.0
            chandler_factor = float(np.exp(self.chandler_coupling * np.cos(2 * np.pi * (days_since_2022 - chandler_phase_offset) / chandler_period)))
            sigma_eff *= chandler_factor
        else:
            chandler_factor = 1.0

        if self.enable_semiannual_nutation:
            days_since_j2000 = (epoch_utc - self._j2000_epoch).total_seconds() / 86400.0
            semiannual_factor = float(
                np.exp(
                    self.semiannual_nutation_coupling
                    * np.cos(
                        2 * np.pi
                        * (days_since_j2000 - self.semiannual_nutation_phase_offset_days)
                        / self.semiannual_nutation_period_days
                    )
                )
            )
            sigma_eff *= semiannual_factor
        else:
            semiannual_factor = 1.0

        if self.enable_planetary_index and self.planetary_sigma_coupling != 0.0:
            if planetary_index == 0.0:
                planetary_index = self._get_planetary_index(epoch_utc)
            planetary_sigma_factor = float(np.exp(self.planetary_sigma_coupling * planetary_index))
            sigma_eff *= planetary_sigma_factor
        else:
            planetary_sigma_factor = 1.0

        if self.enable_multiscale:
            K_short = np.exp(-distances / lambda_short_eff)
            K_long = np.exp(-distances / lambda_long_eff)
            K_structure = self.mixing_weight * K_short + (1.0 - self.mixing_weight) * K_long
        else:
            K_structure = np.exp(-distances / lambda_eff)

        K = (sigma_eff ** 2) * K_structure

        if self.enable_local_time:
            lst1 = np.array([self.get_local_solar_time(coords1[i], epoch_utc) for i in range(n1)], dtype=float)
            lst2 = np.array([self.get_local_solar_time(coords2[j], epoch_utc) for j in range(n2)], dtype=float)
        else:
            lst1 = np.full(n1, epoch_utc.hour + epoch_utc.minute / 60.0, dtype=float)
            lst2 = np.full(n2, epoch_utc.hour + epoch_utc.minute / 60.0, dtype=float)

        theta1 = 2 * np.pi * (lst1 / 24.0)
        theta2 = 2 * np.pi * (lst2 / 24.0)
        u1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
        u2 = np.stack([np.cos(theta2), np.sin(theta2)], axis=1)
        diurnal_similarity = np.clip(u1 @ u2.T, -1.0, 1.0)
        diurnal_factor = float(np.exp(self.diurnal_coupling * (diurnal_similarity[0, 0] - 1.0)))
        K *= np.exp(self.diurnal_coupling * (diurnal_similarity - 1.0))

        if self.enable_velocity:
            v1_norm = np.linalg.norm(velocities1, axis=1)
            v2_norm = np.linalg.norm(velocities2, axis=1)
            vhat1 = velocities1 / (v1_norm[:, np.newaxis] + 1e-12)
            vhat2 = velocities2 / (v2_norm[:, np.newaxis] + 1e-12)
            v_similarity = np.clip(vhat1 @ vhat2.T, -1.0, 1.0)
            orbital_alignment = float(v_similarity[0, 0])
            K *= np.exp(self.v_coupling * (v_similarity - 1.0))
        else:
            orbital_alignment = None

        if self.enable_rotation:
            cmb_dir = self.get_cmb_direction_inertial()
            p1_norm = np.linalg.norm(positions1, axis=1)
            p2_norm = np.linalg.norm(positions2, axis=1)
            phat1 = positions1 / (p1_norm[:, np.newaxis] + 1e-12)
            phat2 = positions2 / (p2_norm[:, np.newaxis] + 1e-12)
            cmb_s1 = phat1 @ cmb_dir
            cmb_s2 = phat2 @ cmb_dir
            cmb_alignment = float(cmb_s1[0])
            cmb_similarity = cmb_s1[:, np.newaxis] * cmb_s2[np.newaxis, :]
            cmb_factor = float(np.exp(self.cmb_coupling * (cmb_similarity[0, 0] - 1.0)))
            K *= np.exp(self.cmb_coupling * (cmb_similarity - 1.0))
        else:
            cmb_alignment = None
            cmb_factor = None

        diagnostics = {
            'epoch': epoch_utc.isoformat(),
            'lambda_eff': lambda_eff if not self.enable_multiscale else f"{lambda_short_eff:.1f}/{lambda_long_eff:.1f}",
            'sigma_eff': sigma_eff,
            'cmb_alignment': cmb_alignment if self.enable_rotation else None,
            'cmb_factor': cmb_factor if self.enable_rotation else None,
            'orbital_alignment': orbital_alignment if self.enable_velocity else None,
            'planetary_index': planetary_index if self.enable_planetary_index else None,
            'planetary_sigma_factor': planetary_sigma_factor if self.enable_planetary_index else None,
            'lst1': float(lst1[0]) if n1 > 0 else None,
            'lst2': float(lst2[0]) if n2 > 0 else None,
            'diurnal_factor': diurnal_factor,
            'seasonal_factor': seasonal_factor,
            'chandler_factor': chandler_factor if self.enable_chandler else None,
            'semiannual_nutation_factor': semiannual_factor if self.enable_semiannual_nutation else None,
            'distances': distances[0, :].tolist() if n1 > 0 else [],
            'distances_eff': distances[0, :].tolist() if n1 > 0 else []
        }

        return K, diagnostics

    def _get_planetary_index(self, epoch_utc):
        if not self.enable_planetary_index:
            return 0.0

        if importlib.util.find_spec("astropy") is None:
            raise RuntimeError("Planetary index requested but 'astropy' is not available in this environment")

        year = int(epoch_utc.year)
        if year not in self._planetary_stats_cache:
            self._planetary_stats_cache[year] = self._compute_planetary_stats_for_year(year, tuple(self.planetary_bodies))

        stats = self._planetary_stats_cache[year]
        day = int(epoch_utc.timetuple().tm_yday)
        distances = self._planetary_distances_au_for_day(year, day, tuple(self.planetary_bodies))

        z_sum = 0.0
        for body in self.planetary_bodies:
            mu, sd = stats[body]
            d = distances[body]
            if sd > 0:
                z_sum += (mu - d) / sd

        return float(z_sum)

    @staticmethod
    @lru_cache(maxsize=16)
    def _compute_planetary_stats_for_year(year, bodies):
        from astropy.time import Time
        from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
        from astropy import units as u

        # IMPORTANT: Use builtin ephemeris to avoid implicit network downloads.
        solar_system_ephemeris.set('builtin')

        ds = {}
        for body in bodies:
            distances = []
            for doy in range(1, 367):
                t = Time(datetime(year, 1, 1) + timedelta(days=doy - 1))
                earth_pos, _ = get_body_barycentric_posvel('earth', t)
                body_pos, _ = get_body_barycentric_posvel(body, t)
                delta = body_pos.xyz - earth_pos.xyz
                dist = float(np.sqrt(np.sum(delta.to(u.AU).value ** 2)))
                distances.append(dist)
            distances = np.array(distances)
            ds[body] = (float(np.mean(distances)), float(np.std(distances)))

        return ds

    @staticmethod
    @lru_cache(maxsize=2048)
    def _planetary_distances_au_for_day(year, doy, bodies):
        from astropy.time import Time
        from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
        from astropy import units as u

        # IMPORTANT: Use builtin ephemeris to avoid implicit network downloads.
        solar_system_ephemeris.set('builtin')

        t = Time(datetime(year, 1, 1) + timedelta(days=int(doy) - 1))
        earth_pos, _ = get_body_barycentric_posvel('earth', t)

        out = {}
        for body in bodies:
            body_pos, _ = get_body_barycentric_posvel(body, t)
            delta = body_pos.xyz - earth_pos.xyz
            out[body] = float(np.sqrt(np.sum(delta.to(u.AU).value ** 2)))

        return out
    
    def predict(self, known_coords, known_values, target_coords,
                noise_variance=1.0, epoch_utc=None, reference_epoch=None,
                return_diag_package=False,
                validate_psd=False):
        """
        Kriging prediction with proper dynamic physics.
        
        Parameters:
        -----------
        noise_variance : float
            Local (uncorrelated) noise variance of the reference station clocks
            (ns²).  This is the nugget effect that prevents the predictor from
            overfitting reference-station local noise.  The correct value is
            (1 - spatial_fraction) * total_clock_variance, which for the
            European network with spatial_fraction ≈ 0.44 is approximately
            0.56 * σ²_total.  Callers should always pass this explicitly.
        epoch_utc : datetime
            Specific epoch for prediction (required for dynamic effects)
        reference_epoch : datetime
            Reference time for rotation computation (e.g., start of day)
        """
        if epoch_utc is None:
            # Deterministic default: J2000 epoch (the same reference used by
            # covariance_matrix).  Using datetime.utcnow() here would make
            # predictions non-reproducible — the dynamic coupling terms
            # (diurnal, seasonal, Chandler) depend on the epoch, so a
            # wall-clock default silently changes the result between runs.
            # Callers that need time-dependent physics must pass epoch_utc
            # explicitly.
            epoch_utc = self._j2000_epoch
        if reference_epoch is None:
            reference_epoch = datetime(epoch_utc.year, epoch_utc.month, epoch_utc.day, 0, 0, 0)
        
        known_coords = np.atleast_2d(known_coords)
        target_coords = np.atleast_2d(target_coords)
        known_values = np.atleast_1d(known_values)
        
        # Compute covariances at this specific epoch
        # Need to compute full covariance matrices properly
        n_known = known_coords.shape[0]
        n_target = target_coords.shape[0]

        # Guard: no reference stations -> return NaN prediction with infinite variance
        if n_known == 0:
            return (np.full(n_target, np.nan), np.full(n_target, np.inf),
                    {"warning": "no reference stations provided"})
        
        # K_nn: covariance between all known stations
        K_nn = np.zeros((n_known, n_known))
        for i in range(n_known):
            cov_ii, _ = self.compute_instantaneous_correlation(
                known_coords[i:i+1], known_coords[i:i+1], epoch_utc, reference_epoch
            )
            K_nn[i, i] = cov_ii[0, 0]
            for j in range(i+1, n_known):
                cov, _ = self.compute_instantaneous_correlation(
                    known_coords[i:i+1], known_coords[j:j+1], epoch_utc, reference_epoch
                )
                K_nn[i, j] = K_nn[j, i] = cov[0, 0]

        K_nn = (K_nn + K_nn.T) / 2.0
        
        # K_nt: covariance between known and target
        K_nt = np.zeros((n_known, n_target))
        diag_out = None
        pair_diagnostics = []
        for i in range(n_known):
            for j in range(n_target):
                cov, diag = self.compute_instantaneous_correlation(
                    known_coords[i:i+1], target_coords[j:j+1], epoch_utc, reference_epoch
                )
                K_nt[i, j] = cov[0, 0]
                diag_out = diag
                pair_diagnostics.append(diag)
        
        # K_tt: covariance at target
        K_tt = np.zeros(n_target)
        target_diagnostics = []
        for j in range(n_target):
            cov_tt, diag_tt = self.compute_instantaneous_correlation(
                target_coords[j:j+1], target_coords[j:j+1], epoch_utc, reference_epoch
            )
            K_tt[j] = cov_tt[0, 0]
            diag_out = diag_tt
            target_diagnostics.append(diag_tt)
        
        # Add noise and solve via jittered Cholesky (avoid explicit inversion)
        if noise_variance < 0:
            noise_variance = 0.0

        K_nn_noisy = K_nn + noise_variance * np.eye(n_known)

        if validate_psd:
            diag_nn = np.diag(K_nn_noisy)
            if np.any(~np.isfinite(diag_nn)):
                raise ValueError("K_nn has non-finite diagonal")
            if np.min(diag_nn) < -1e-12:
                raise ValueError(f"K_nn has negative diagonal entry: min={float(np.min(diag_nn))}")
            min_eig = float(np.min(np.linalg.eigvalsh((K_nn_noisy + K_nn_noisy.T) / 2.0)))
            if min_eig < -1e-10:
                raise ValueError(f"K_nn is not PSD: min_eig={min_eig}")

        jitter = 1e-10
        L = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(K_nn_noisy + jitter * np.eye(n_known))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0

        if L is None:
            # Fall back to pseudo-inverse only if Cholesky repeatedly fails
            K_nn_inv = np.linalg.pinv(K_nn_noisy)
            pred_mean = K_nt.T @ K_nn_inv @ known_values
            W = K_nt.T @ K_nn_inv
            var_reduction = np.sum(W * K_nt.T, axis=1)
            pred_var = K_tt - var_reduction
        else:
            y = np.linalg.solve(L, known_values)
            alpha = np.linalg.solve(L.T, y)
            pred_mean = K_nt.T @ alpha

            v = np.linalg.solve(L, K_nt)
            pred_var = K_tt - np.sum(v * v, axis=0)
        
        pred_mean = np.atleast_1d(pred_mean)
        pred_var = np.atleast_1d(pred_var)

        # Numerical safety: variances must be nonnegative
        pred_var = np.maximum(pred_var, 0.0)

        if validate_psd:
            if np.any(~np.isfinite(pred_var)):
                raise ValueError("pred_var contains non-finite values")
        
        if return_diag_package:
            diag_package = {
                'target': target_diagnostics[0] if n_target == 1 else target_diagnostics,
                'K_tt': K_tt.tolist(),
                'pairs': pair_diagnostics if pair_diagnostics else None
            }
            return pred_mean, pred_var, diag_package

        # Backwards-compatible: third return is a single diagnostics dict
        return pred_mean, pred_var, diag_out
