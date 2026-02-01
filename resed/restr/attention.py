"""
Minimal Multi-Head Self-Attention.

A deterministic, single-layer MHSA implementation for resTR.
"""

import numpy as np

class MinimalMHSA:
    """
    Minimal Multi-Head Self-Attention.
    
    Attributes:
        d_model: Latent dimension.
        n_heads: Number of attention heads.
        d_head: Dimension per head.
        W_q, W_k, W_v, W_o: Projection matrices.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Initialize deterministic weights
        rng = np.random.default_rng(42)
        scale = 1.0 / np.sqrt(d_model)
        
        self.W_q = rng.uniform(-scale, scale, (d_model, d_model))
        self.W_k = rng.uniform(-scale, scale, (d_model, d_model))
        self.W_v = rng.uniform(-scale, scale, (d_model, d_model))
        self.W_o = rng.uniform(-scale, scale, (d_model, d_model))
        
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute Self-Attention: Softmax(QK^T / sqrt(d_k))V
        
        Args:
            z: Input tensor (batch, seq_len, d_model) or (batch, d_model).
               
        Returns:
            Output tensor matching input shape.
        """
        input_ndim = z.ndim
        
        if input_ndim == 2:
            # Assume (batch, d_model) -> (batch, 1, d_model)
            # We treat samples as independent sequences of length 1.
            z_in = z[:, np.newaxis, :]
        else:
            z_in = z
            
        batch, seq_len, d = z_in.shape
        
        # Linear Projections
        Q = np.dot(z_in, self.W_q)
        K = np.dot(z_in, self.W_k)
        V = np.dot(z_in, self.W_v)
        
        # Split Heads
        # (batch, seq_len, n_heads, d_head)
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        # (batch, n_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Aggregate
        attn_out = np.matmul(attn_weights, V)
        
        # Merge Heads
        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head) -> (batch, seq_len, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        
        # Output Projection
        output = np.dot(attn_out, self.W_o)
        
        # Restore dimensionality
        if input_ndim == 2:
            # (batch, 1, d_model) -> (batch, d_model)
            output = output.reshape(batch, self.d_model)
            
        return output