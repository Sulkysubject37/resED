"""
resENC: Deterministic Encoder with Statistical Channel.

Implements the Reference Encoder (resENC) for the resED system.
Enforces a deterministic projection X -> Z and calculates a side-channel 
statistical summary S for the RLCS layer.
"""

import numpy as np
from resed.encoders.base import BaseEncoder
from resed.utils.math import l2_norm

class ResENC(BaseEncoder):
    """
    Reference Encoder (resENC).
    
    Performs a deterministic projection Z = phi(XW + b) and computes
    statistical summaries of the latent representation.
    
    Attributes:
        W (np.ndarray): Weight matrix of shape (d_in, d_z).
        b (np.ndarray): Bias vector of shape (d_z,).
        phi (callable): Element-wise nonlinearity (default: np.tanh).
    """
    
    def __init__(self, d_in: int, d_z: int, phi=np.tanh):
        """
        Initialize the deterministic encoder.
        
        Args:
            d_in: Input dimensionality.
            d_z: Latent dimensionality.
            phi: Activation function (default: np.tanh).
        """
        super().__init__()
        self.W = np.zeros((d_in, d_z))
        self.b = np.zeros(d_z)
        self.phi = phi
        self._d_in = d_in
        self._d_z = d_z

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        """
        Set the parameters of the encoder.
        
        Args:
            W: Weight matrix (d_in, d_z).
            b: Bias vector (d_z,).
            
        Raises:
            ValueError: If shapes do not match dimensions provided at init.
        """
        if W.shape != (self._d_in, self._d_z):
            raise ValueError(f"W shape mismatch: expected {(self._d_in, self._d_z)}, got {W.shape}")
        if b.shape != (self._d_z,):
            raise ValueError(f"b shape mismatch: expected {(self._d_z,)}, got {b.shape}")
        
        self.W = W
        self.b = b

    def _compute_statistics(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the statistical channel S for a batch of latent vectors.
        
        S_i = [norm(z_i), var(z_i), entropy_proxy(z_i), sparsity(z_i)]
        
        Args:
            z: Latent vectors (batch_size, d_z).
            
        Returns:
            S: Statistical summary (batch_size, 4).
        """
        batch_size = z.shape[0]
        stats = np.zeros((batch_size, 4))
        
        for i in range(batch_size):
            zi = z[i]
            
            norm_val = l2_norm(zi)
            var_val = float(np.var(zi))
            
            # Shannon entropy of softmax
            exps = np.exp(zi - np.max(zi))
            probs = exps / np.sum(exps)
            log_probs = np.log(probs + 1e-12) 
            entropy_val = -np.sum(probs * log_probs)
            
            # Sparsity proxy (L1 norm / sqrt(d))
            sparsity_val = float(np.sum(np.abs(zi))) / np.sqrt(self._d_z)

            stats[i] = [norm_val, var_val, entropy_val, sparsity_val]
            
        return stats

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project inputs to latent space and return statistics.
        
        Args:
            x: Input data (batch_size, d_in).
            
        Returns:
            Z: Latent representation (batch_size, d_z).
            S: Statistical summary (batch_size, 4).
            
        Raises:
            ValueError: If x has incorrect shape.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, d_in), got {x.ndim}D")
        if x.shape[1] != self._d_in:
            raise ValueError(f"Input dimension mismatch: expected {self._d_in}, got {x.shape[1]}")

        linear = np.dot(x, self.W) + self.b
        z = self.phi(linear)
        s = self._compute_statistics(z)
        
        return z, s