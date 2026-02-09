"""
Math Invariants.

Defines reusable checks for mathematical invariants in the resED system.
Enforces shape preservation, finiteness, and norm inflation bounds.
"""

import numpy as np
from resed.utils.math import l2_norm

def check_shape_invariant(z_in: np.ndarray, z_out: np.ndarray):
    """
    Verify that input and output shapes are identical.
    
    Args:
        z_in: Input tensor.
        z_out: Output tensor.
        
    Raises:
        RuntimeError: If shapes do not match.
    """
    if z_in.shape != z_out.shape:
        raise RuntimeError(
            f"Shape invariant violated: Input {z_in.shape} vs Output {z_out.shape}"
        )

def check_finite_invariant(z: np.ndarray):
    """
    Verify that all values in the tensor are finite.
    
    Args:
        z: Tensor to check.
        
    Raises:
        RuntimeError: If any value is NaN or Inf.
    """
    if not np.all(np.isfinite(z)):
        raise RuntimeError("Finite invariant violated: Tensor contains NaN or Inf")

def check_norm_inflation_invariant(z_in: np.ndarray, z_out: np.ndarray, epsilon: float = 0.05):
    """
    Verify that output norm does not exceed input norm by more than a margin.
    ||Z_out|| <= (1 + epsilon) ||Z_in||
    
    Args:
        z_in: Input tensor.
        z_out: Output tensor.
        epsilon: Allowed inflation margin.
        
    Raises:
        RuntimeError: If norm inflation bound is violated.
    """
    norm_in = l2_norm(z_in)
    norm_out = l2_norm(z_out)
    
    bound = (1.0 + epsilon) * norm_in
    
    if norm_out > bound:
        raise RuntimeError(
            f"Norm inflation invariant violated: "
            f"Output norm {norm_out:.4f} > Bound {bound:.4f} "
            f"(Input norm {norm_in:.4f}, epsilon {epsilon})"
        )