"""
Universality Models.

Wrappers for heterogeneous encoder architectures to test RLCS universality.
Includes:
1. resENC (Baseline, MLP-based)
2. VAE (Probabilistic, Convolutional/Linear)
3. Transformer (Attention-based)
"""

import numpy as np
import torch
import torch.nn as nn
from resed.encoders.resenc import ResENC

class BaseWrapper:
    def encode(self, x):
        raise NotImplementedError

class WrapperResENC(BaseWrapper):
    """
    Wrapper for the existing resED encoder (MLP/Tanh).
    """
    def __init__(self, d_in, d_z):
        self.model = ResENC(d_in, d_z)
        # Random init
        rng = np.random.default_rng(42)
        W = rng.uniform(-0.1, 0.1, (d_in, d_z))
        b = np.zeros(d_z)
        self.model.set_weights(W, b)
        
    def encode(self, x):
        # resENC returns (z, s)
        z, _ = self.model.encode(x)
        return z

class WrapperVAE(BaseWrapper):
    """
    Standard VAE Encoder (PyTorch).
    Returns the mean parameter (mu) as the latent representation.
    """
    def __init__(self, d_in, d_z):
        self.model = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_z * 2) # mu, logvar
        )
        self._init_weights()
        self.model.eval()
        
    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        # x is numpy (batch, d_in)
        x_t = torch.from_numpy(x).float()
        with torch.no_grad():
            out = self.model(x_t)
            mu, _ = out.chunk(2, dim=-1)
        return mu.numpy()

class WrapperTransformer(BaseWrapper):
    """
    Transformer Encoder (PyTorch).
    Uses Multi-Head Attention and Mean Pooling.
    """
    def __init__(self, d_in, d_z):
        # Transformer requires sequence. We treat input as sequence of chunks?
        # Or project input to d_model and use simple tokenization?
        # Architecture: Linear(d_in -> d_model) -> TransformerEnc -> Linear(d_model -> d_z)
        # For simplicity, we treat d_in as d_model and input as (Batch, 1, d_in) sequence?
        # Or split input features into tokens.
        # Let's split d_in (128) into 4 tokens of 32 dims.
        self.n_tokens = 4
        self.d_token = d_in // 4
        self.d_model = self.d_token
        
        self.embedding = nn.Linear(self.d_token, self.d_model)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(self.d_model, d_z)
        
        self._init_weights()
        self.model = nn.ModuleList([self.embedding, self.transformer, self.head])
        self.model.eval()

    def _init_weights(self):
        for p in self.head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        # x: (batch, 128) -> reshape to (batch, 4, 32)
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.n_tokens, self.d_token)
        x_t = torch.from_numpy(x_reshaped).float()
        
        with torch.no_grad():
            # Embed
            h = self.embedding(x_t)
            # Attention
            h = self.transformer(h)
            # Mean Pooling
            h_pool = h.mean(dim=1)
            # Projection
            z = self.head(h_pool)
            
        return z.numpy()
