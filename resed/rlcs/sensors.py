"""
RLCS Sensors.

Implements the deterministic sensing logic for the RLCS layer.
These functions calculate reliability metrics based on latent representations.
"""

import numpy as np

def population_consistency(z: np.ndarray, mu: np.ndarray | float, sigma: float, epsilon: float = 1e-8) -> np.ndarray:
    """
    Compute Population Consistency (ResLik-style).
    
    D_i = ||z_i - mu||_2 / (sigma + epsilon)
    
    Args:
        z: Latent vectors (batch_size, d_z).
        mu: Reference mean (vector or scalar).
        sigma: Reference standard deviation (scalar).
        epsilon: Stability constant.
        
    Returns:
        D: Consistency scores (batch_size,).
    """
    # Calculate Euclidean distance for each sample
    # If mu is vector: broadcast subtract
    diff = z - mu
    dist = np.linalg.norm(diff, axis=1)
    
    return dist / (sigma + epsilon)

def temporal_consistency(z: np.ndarray) -> np.ndarray:
    """
    Compute Temporal Consistency.
    
    T_i = exp(-||z_i - z_{i-1}||_2)
    
    Defined only for sequential inputs.
    First element defaults to 1.
    
    Args:
        z: Latent vectors (batch_size, d_z).
        
    Returns:
        T: Temporal consistency scores (batch_size,).
    """
    batch_size = z.shape[0]
    t_scores = np.ones(batch_size, dtype=float)
    
    if batch_size > 1:
        # Calculate differences between adjacent steps
        z_curr = z[1:]
        z_prev = z[:-1]
        diff = z_curr - z_prev
        dists = np.linalg.norm(diff, axis=1)
        
        # Calculate exponential decay
        t_scores[1:] = np.exp(-dists)
        
    return t_scores

def agreement_consistency(z: np.ndarray, z_prime: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Compute Agreement Consistency.
    
    A_i = (z_i . z'_i) / (||z_i|| ||z'_i|| + epsilon)
    
    Args:
        z: Primary latent vectors (batch_size, d_z).
        z_prime: Alternate view latent vectors (batch_size, d_z).
        epsilon: Stability constant.
        
    Returns:
        A: Agreement scores (batch_size,).
    """
    # Compute dot products
    dot_products = np.sum(z * z_prime, axis=1)
    
    # Compute norms
    norm_z = np.linalg.norm(z, axis=1)
    norm_z_prime = np.linalg.norm(z_prime, axis=1)
    
    return dot_products / (norm_z * norm_z_prime + epsilon)
