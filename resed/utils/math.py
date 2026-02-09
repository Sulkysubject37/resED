"""
Math Utilities.

Pure, deterministic mathematical helpers .
Provides foundational operations for the RLCS layer.
"""

import numpy as np

def l2_norm(x: np.ndarray) -> float:
    """
    Compute the L2 norm of a vector or matrix.
    
    Args:
        x: Input array.
        
    Returns:
        L2 norm as a float.
    """
    return float(np.linalg.norm(x))

def cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        x: First input vector.
        y: Second input vector.
        eps: Small constant for numerical stability.
        
    Returns:
        Cosine similarity (-1.0 to 1.0).
    """
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    denominator = max(nx * ny, eps)
    return float(np.dot(x.flatten(), y.flatten()) / denominator)

def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Perform element-wise division with numerical stability.
    
    Args:
        a: Numerator.
        b: Denominator.
        eps: Small constant added to denominator.
        
    Returns:
        Result of a / (b + eps).
    """
    return a / (b + eps)

def clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Clip values to a specified range.
    
    Args:
        x: Input array.
        low: Lower bound.
        high: Upper bound.
        
    Returns:
        Clipped array.
    """
    return np.clip(x, low, high)