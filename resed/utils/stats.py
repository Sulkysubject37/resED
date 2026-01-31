"""
Statistical Utilities.

Population and temporal statistics utilities for RLCS sensing.
Derived from resLik architecture to support ResLik, TCS, and Agreement sensors.
"""

import numpy as np

def population_mean(x: np.ndarray) -> float:
    """
    Compute the mean of a population batch.
    
    Args:
        x: Input population batch.
        
    Returns:
        Mean value.
    """
    return float(np.mean(x))

def population_std(x: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute the standard deviation of a population batch.
    
    Args:
        x: Input population batch.
        eps: Small constant for numerical stability (unused in std calc itself but consistent with API).
        
    Returns:
        Standard deviation.
    """
    return float(np.std(x))

def z_score(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Compute the Z-score for inputs given population stats.
    
    Args:
        x: Input data.
        mean: Population mean.
        std: Population standard deviation.
        
    Returns:
        Z-score (x - mean) / std.
    """
    # Avoid division by zero if std is effectively 0
    if abs(std) < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std

def ema(prev: np.ndarray, curr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute Exponential Moving Average.
    
    Args:
        prev: Previous EMA value.
        curr: Current value.
        alpha: Smoothing factor (0 < alpha <= 1).
        
    Returns:
        Updated EMA.
    """
    return alpha * curr + (1.0 - alpha) * prev

def rolling_difference(x_prev: np.ndarray, x_curr: np.ndarray) -> np.ndarray:
    """
    Compute the simple difference between current and previous states.
    
    Args:
        x_prev: Previous state.
        x_curr: Current state.
        
    Returns:
        Difference (x_curr - x_prev).
    """
    return x_curr - x_prev