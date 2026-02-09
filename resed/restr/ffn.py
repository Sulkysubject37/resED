"""
Feedforward Network for resTR.

A deterministic 2-layer FFN with ReLU activation.
"""

import numpy as np

class FFN:
    """
    Two-layer Feedforward Network.
    
    Attributes:
        W1: First layer weights (d_model, d_ff).
        b1: First layer bias (d_ff).
        W2: Second layer weights (d_ff, d_model).
        b2: Second layer bias (d_model).
    """
    
    def __init__(self, d_model: int, d_ff: int = None):
        if d_ff is None:
            d_ff = 4 * d_model
            
        rng = np.random.default_rng(42)
        scale1 = 1.0 / np.sqrt(d_model)
        scale2 = 1.0 / np.sqrt(d_ff)
        
        self.W1 = rng.uniform(-scale1, scale1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        
        self.W2 = rng.uniform(-scale2, scale2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute FFN(z) = ReLU(zW1 + b1)W2 + b2
        
        Args:
            z: Input tensor (batch, seq_len, d_model).
            
        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        h = np.dot(z, self.W1) + self.b1
        h = np.maximum(h, 0)
        out = np.dot(h, self.W2) + self.b2
        
        return out