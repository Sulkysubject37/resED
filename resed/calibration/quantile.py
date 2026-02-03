"""
Quantile Estimation Utilities.

Provides non-parametric quantile estimation for calibration.
"""

import numpy as np

def estimate_quantiles(data: np.ndarray, num_quantiles: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical quantiles from reference data.
    
    Args:
        data: Reference data array (n_samples,).
        num_quantiles: Number of quantile bins.
        
    Returns:
        quantiles: Quantile probability levels (0 to 1).
        values: Data values at those quantiles.
    """
    if len(data) == 0:
        raise ValueError("Cannot estimate quantiles from empty data.")
        
    # Use linspace for quantiles
    q = np.linspace(0, 1, num_quantiles)
    values = np.quantile(data, q)
    
    return q, values

def map_to_quantile(value: float, quantiles: np.ndarray, values: np.ndarray) -> float:
    """
    Map a raw value to its quantile rank (CDF value).
    
    Args:
        value: Raw input value.
        quantiles: Quantile probability levels.
        values: Data values at those quantiles.
        
    Returns:
        Quantile rank (0.0 to 1.0).
    """
    # Use interpolation to find rank
    return np.interp(value, values, quantiles, left=0.0, right=1.0)
