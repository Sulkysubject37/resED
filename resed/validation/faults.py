"""
Failure Injection Module.

Injects deterministic failures into latent representations to test RLCS observability.
"""

import numpy as np

def inject_distribution_shift(z: np.ndarray, shift_magnitude: float = 5.0) -> np.ndarray:
    """
    Apply a sudden distribution shift (OOD).
    
    Args:
        z: Latent batch (batch_size, d_z).
        shift_magnitude: Magnitude of the mean shift.
        
    Returns:
        Shifted latent vectors.
    """
    return z + shift_magnitude

def inject_gradual_drift(z: np.ndarray, drift_rate: float = 0.5) -> np.ndarray:
    """
    Apply a gradual temporal drift across the batch.
    
    Args:
        z: Latent batch (batch_size, d_z).
        drift_rate: Accumulation rate of drift per index.
        
    Returns:
        Drifted latent vectors.
    """
    batch_size = z.shape[0]
    # Create drift: 0, drift_rate, 2*drift_rate, ...
    drift_factors = np.arange(batch_size)[:, np.newaxis] * drift_rate
    return z + drift_factors

def inject_view_disagreement(z: np.ndarray, noise_magnitude: float = 2.0) -> np.ndarray:
    """
    Generate an alternate view with high disagreement.
    
    Args:
        z: Primary latent batch.
        noise_magnitude: Magnitude of divergence for the alternate view.
        
    Returns:
        Perturbed latent batch (alternate view).
    """
    # Deterministic perturbation using fixed RNG
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(z.shape) * noise_magnitude
    return z + noise

def inject_single_point_shock(z: np.ndarray, index: int, magnitude: float = 10.0) -> np.ndarray:
    """
    Apply a large shock to a single sample in the batch.
    
    Args:
        z: Latent batch.
        index: Index of the sample to shock.
        magnitude: Shock magnitude.
        
    Returns:
        Latent batch with one shocked vector.
    """
    z_shocked = z.copy()
    if 0 <= index < z_shocked.shape[0]:
        z_shocked[index] += magnitude
    return z_shocked
