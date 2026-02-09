"""
RLCS Control Surface.

Implements the central control logic that maps diagnostics to control signals.
"""

import numpy as np
from resed.rlcs.types import RlcsSignal
from resed.rlcs.thresholds import TAU_D, TAU_T, TAU_A
from resed.rlcs.sensors import population_consistency, temporal_consistency, agreement_consistency

def rlcs_control(z: np.ndarray, s: np.ndarray, diagnostics: dict = None, calibrator=None, **kwargs) -> list[RlcsSignal]:
    """
    Compute control signals for a batch of latent representations.
    
    Logic (Conservative OR):
    1. ABSTAIN if Population Consistency > TAU_D
    2. DEFER if Temporal Consistency < TAU_T
    3. DOWNWEIGHT if Agreement Consistency < TAU_A
    4. PROCEED otherwise
    
    Args:
        z: Latent representations (batch_size, d_z).
        s: Statistical summary from encoder (batch_size, k).
        diagnostics: Dictionary to populate with computed metrics.
        calibrator: Optional RlcsCalibrator instance to normalize scores.
        **kwargs: Optional inputs (mu, sigma, z_prime).
        
    Returns:
        List of RlcsSignal, one per sample.
    """
    batch_size = z.shape[0]
    signals = []
    
    # Defaults for reference stats
    mu = kwargs.get('mu', 0.0)
    sigma = kwargs.get('sigma', 1.0)
    z_prime = kwargs.get('z_prime', None)
    
    # 1. Compute Diagnostics
    d_scores = population_consistency(z, mu, sigma)
    t_scores = temporal_consistency(z)
    
    a_scores = None
    if z_prime is not None:
        if z_prime.shape != z.shape:
            raise ValueError(f"z_prime shape {z_prime.shape} must match z {z.shape}")
        a_scores = agreement_consistency(z, z_prime)
        
    if diagnostics is not None:
        diagnostics['population_consistency'] = d_scores
        diagnostics['temporal_consistency'] = t_scores
        if a_scores is not None:
            diagnostics['agreement_consistency'] = a_scores
            
    # 2. Calibration
    d_decision = d_scores
    t_decision = t_scores
    a_decision = a_scores
    
    if calibrator is not None and calibrator.is_calibrated:
        # Calibrate Population Consistency (unbounded distance) to Z-score
        d_decision = calibrator.calibrate_batch('population_consistency', d_scores)
        # Note: Temporal and Agreement consistency are naturally bounded [0, 1]
        # and are typically left uncalibrated to preserve absolute threshold semantics.
            
    # 3. Evaluate Control Logic
    for i in range(batch_size):
        if d_decision[i] > TAU_D:
            signals.append(RlcsSignal.ABSTAIN)
            continue
            
        if t_decision[i] < TAU_T:
            signals.append(RlcsSignal.DEFER)
            continue
            
        if a_decision is not None and a_decision[i] < TAU_A:
            signals.append(RlcsSignal.DOWNWEIGHT)
            continue
            
        signals.append(RlcsSignal.PROCEED)
        
    return signals