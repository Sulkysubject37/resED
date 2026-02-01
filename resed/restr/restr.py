"""
Residual Transformer (resTR).

An optional, controlled refinement layer governed by RLCS.
Implements the residual-only logic:
Z_out = Z_in + alpha*Attn(Z_in) + beta*FFN(...)
"""

import numpy as np
from resed.restr.attention import MinimalMHSA
from resed.restr.ffn import FFN
from resed.math.invariants import (
    check_shape_invariant,
    check_finite_invariant,
    check_norm_inflation_invariant
)

class ResTR:
    """
    Residual Transformer Module.
    
    Attributes:
        attention: MinimalMHSA instance.
        ffn: FFN instance.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.attention = MinimalMHSA(d_model, n_heads)
        self.ffn = FFN(d_model)
        
    def forward(self, z: np.ndarray, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
        """
        Apply controlled residual refinement.
        
        Z_1 = Z + alpha * MHSA(Z)
        Z_2 = Z_1 + beta * FFN(Z_1)
        
        Args:
            z: Input latent tensor (batch, seq_len, d_model).
            alpha: Attention scaling factor [0, 1].
            beta: FFN scaling factor [0, 1].
            
        Returns:
            Refined latent tensor.
            
        Raises:
            RuntimeError: If invariants are violated.
        """
        # Pre-execution checks
        check_finite_invariant(z)
        z_in = z.copy() # Keep for invariant checks
        
        # 1. Self-Attention Refinement
        z_1 = z
        if alpha > 0.0:
            attn_out = self.attention.forward(z)
            z_1 = z + alpha * attn_out
            
        # 2. FFN Refinement
        z_2 = z_1
        if beta > 0.0:
            ffn_out = self.ffn.forward(z_1)
            z_2 = z_1 + beta * ffn_out
            
        z_out = z_2
        
        # Post-execution Invariant Checks
        check_finite_invariant(z_out)
        check_shape_invariant(z_in, z_out)
        check_norm_inflation_invariant(z_in, z_out)
        
        return z_out
