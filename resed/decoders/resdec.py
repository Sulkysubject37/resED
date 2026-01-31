"""
resDEC: Deterministic Decoder with Control Gating.

This module implements the core Reference Decoder (resDEC) for the resED system.
It maps latent representations Z back to outputs, subject to strict
external control signals (PROCEED, DOWNWEIGHT, DEFER, ABSTAIN).
"""

import numpy as np
from resed.decoders.base import BaseDecoder

# Control Signals
PROCEED = "PROCEED"
DOWNWEIGHT = "DOWNWEIGHT"
DEFER = "DEFER"
ABSTAIN = "ABSTAIN"

VALID_CONTROLS = {PROCEED, DOWNWEIGHT, DEFER, ABSTAIN}

class ResDEC(BaseDecoder):
    """
    Reference Decoder (resDEC).
    
    Performs a deterministic mapping Y = psi(ZU + c) subject to
    an explicit control signal.
    
    Attributes:
        U (np.ndarray): Weight matrix (d_z, d_out).
        c (np.ndarray): Bias vector (d_out,).
        psi (callable): Output activation (default: identity).
        alpha (float): Scaling factor for DOWNWEIGHT signal (default: 0.5).
    """
    
    def __init__(self, d_z: int, d_out: int, psi=lambda x: x, alpha: float = 0.5):
        """
        Initialize the deterministic decoder.
        
        Args:
            d_z: Latent dimensionality.
            d_out: Output dimensionality.
            psi: Output activation function.
            alpha: Downweight scaling factor (0 < alpha < 1).
        """
        super().__init__()
        self.U = np.zeros((d_z, d_out))
        self.c = np.zeros(d_out)
        self.psi = psi
        self.alpha = alpha
        self._d_z = d_z
        self._d_out = d_out

    def set_weights(self, U: np.ndarray, c: np.ndarray):
        """
        Set the parameters of the decoder.
        
        Args:
            U: Weight matrix (d_z, d_out).
            c: Bias vector (d_out,).
            
        Raises:
            ValueError: If shapes do not match dimensions provided at init.
        """
        if U.shape != (self._d_z, self._d_out):
            raise ValueError(f"U shape mismatch: expected {(self._d_z, self._d_out)}, got {U.shape}")
        if c.shape != (self._d_out,):
            raise ValueError(f"c shape mismatch: expected {(self._d_out,)}, got {c.shape}")
        
        self.U = U
        self.c = c

    def decode(self, z: np.ndarray, control_signal: str) -> np.ndarray | None:
        """
        Reconstruct output from latent representation subject to control.
        
        Args:
            z: Latent representation (batch_size, d_z).
            control_signal: One of {PROCEED, DOWNWEIGHT, DEFER, ABSTAIN}.
            
        Returns:
            Y_hat: Reconstructed output (batch_size, d_out), or None if ABSTAIN/DEFER.
            
        Raises:
            ValueError: If inputs are invalid or control signal is unknown.
        """
        if control_signal not in VALID_CONTROLS:
            raise ValueError(f"Invalid control signal: {control_signal}")
            
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, d_z), got {z.ndim}D")
        if z.shape[1] != self._d_z:
            raise ValueError(f"Latent dimension mismatch: expected {self._d_z}, got {z.shape[1]}")

        # Handle suppressive signals immediately
        if control_signal in {ABSTAIN, DEFER}:
            return None

        # Nominal Decode
        # Y = psi(ZU + c)
        linear = np.dot(z, self.U) + self.c
        y_hat = self.psi(linear)
        
        # Apply Control Logic
        if control_signal == DOWNWEIGHT:
            y_hat = y_hat * self.alpha
            
        return y_hat
