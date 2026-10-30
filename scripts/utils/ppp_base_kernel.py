import numpy as np
from scipy.spatial.distance import cdist

class TEPKernel:
    """
    The TEP Spatial Correlation Kernel.
    
    Models the spatial covariance of clock/atmospheric errors as an 
    exponential decay function of distance.
    
    K(r) = sigma^2 * exp(-r / lambda)
    """
    
    def __init__(self, correlation_length_km=4200.0, sigma_ns=1.0):
        """
        Args:
            correlation_length_km (float): The decay length lambda (default: 4200 km, canonical TEP scale)
            sigma_ns (float): The magnitude of the noise in nanoseconds (default: 1 ns ~ 30cm)
        """
        self.lambda_km = correlation_length_km
        self.sigma = sigma_ns
        
    def covariance_matrix(self, coords1, coords2=None):
        """
        Compute covariance matrix between two sets of coordinates.
        
        Args:
            coords1 (np.ndarray): (N, 3) ECEF coordinates in km
            coords2 (np.ndarray): (M, 3) ECEF coordinates in km. If None, computes auto-covariance for coords1.
            
        Returns:
            np.ndarray: (N, M) Covariance matrix
        """
        if coords2 is None:
            coords2 = coords1
            
        # Compute distances (Euclidean/Chord distance is fine for this scale, 
        # or we could do Great Circle if we had Lat/Lon, but ECEF chord is conservative)
        dists_km = cdist(coords1, coords2, metric='euclidean')
        
        # Exponential Kernel
        K = (self.sigma ** 2) * np.exp(-dists_km / self.lambda_km)
        
        return K

    def predict(self, known_coords, known_values, target_coords, noise_variance=1.0):
        """
        Perform Simple Kriging (Gaussian Process Regression) to predict values at target locations.
        
        Args:
            known_coords (np.ndarray): (N, 3) locations of reference stations
            known_values (np.ndarray): (N,) residual values at reference stations
            target_coords (np.ndarray): (M, 3) locations of rover(s)
            noise_variance (float): Variance of the measurement noise at reference stations (nugget)
            
        Returns:
            tuple: (predictions, variances)
                predictions (np.ndarray): (M,) Predicted values
                variances (np.ndarray): (M,) Variance of the prediction (uncertainty)
        """
        # K_obs_obs (N x N)
        K_oo = self.covariance_matrix(known_coords, known_coords) + np.eye(len(known_coords)) * noise_variance
        
        # K_target_obs (M x N)
        K_to = self.covariance_matrix(target_coords, known_coords)
        
        # K_target_target (M x M) - diagonal only needed for variance
        K_tt_diag = np.full(len(target_coords), self.sigma**2)
        
        # Solve weights: w = K_oo^-1 * y
        # Using Cholesky for stability: K_oo = L * L.T
        try:
            L = np.linalg.cholesky(K_oo)
            # Forward substitute: L * z = y
            z = np.linalg.solve(L, known_values)
            # Backward substitute: L.T * w = z
            weights = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            # Fallback to standard solve if not pos-def (numerical issues)
            weights = np.linalg.solve(K_oo, known_values)
            
        # Predictions: f* = K_to * w
        predictions = K_to @ weights
        
        # Variance: Var = K_tt - K_to * K_oo^-1 * K_to.T
        # We only need diagonals for point variance
        # v = K_oo^-1 * K_to.T  -> solve K_oo * v = K_to.T
        if 'L' in locals():
            # v = L^-T * L^-1 * K_to.T
            # Let U = L^-1 * K_to.T -> solve L * U = K_to.T
            U = np.linalg.solve(L, K_to.T)
            # Variance reduction = sum(U^2) axis=0
            var_reduction = np.sum(U**2, axis=0)
        else:
            v = np.linalg.solve(K_oo, K_to.T)
            var_reduction = np.sum(K_to.T * v, axis=0) # element-wise mult then sum
            
        variances = K_tt_diag - var_reduction
        
        # Ensure non-negative variance
        variances = np.maximum(variances, 0.0)
        
        return predictions, variances
