"""
RLCS Governance Layer.

Implements adapter and routing logic for the RLCS system.
Acts as the bridge between raw RLCS diagnostics and component execution.
"""

import numpy as np
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal

class RlcsGovernance:
    """
    Governance system for resED.
    
    1. Adapts system state to RLCS inputs.
    2. Computes control signals via RLCS.
    3. Routes execution parameters based on signals.
    """
    
    def __init__(self, attenuation_factor: float = 0.5):
        """
        Initialize governance.
        
        Args:
            attenuation_factor: Scaling factor for alpha/beta under DEFER/DOWNWEIGHT.
        """
        self.attenuation_factor = attenuation_factor

    def diagnose(self, z: np.ndarray, s: np.ndarray, calibrator=None, **kwargs) -> tuple[list[RlcsSignal], dict]:
        """
        Run RLCS diagnostics and compute signals.
        
        Args:
            z: Latent batch.
            s: Statistics batch.
            calibrator: Optional RlcsCalibrator for score normalization.
            **kwargs: Context (mu, sigma, z_prime).
            
        Returns:
            signals: List of control signals per sample.
            diagnostics: Dictionary of scores.
        """
        diagnostics = {}
        signals = rlcs_control(z, s, diagnostics=diagnostics, calibrator=calibrator, **kwargs)
        return signals, diagnostics

    def route(self, signal: RlcsSignal, nominal_alpha: float, nominal_beta: float) -> tuple[float, float]:
        """
        Determine execution parameters for ResTR based on control signal.
        
        Routing Logic:
        - PROCEED: Nominal alpha, beta.
        - DOWNWEIGHT: Nominal alpha, beta (Decoder handles attenuation).
        - DEFER: Attenuated alpha, beta (factor * nominal).
        - ABSTAIN: 0.0, 0.0 (Bypass).
        
        Args:
            signal: Control signal.
            nominal_alpha: Desired alpha.
            nominal_beta: Desired beta.
            
        Returns:
            (effective_alpha, effective_beta)
        """
        if signal == RlcsSignal.PROCEED:
            return nominal_alpha, nominal_beta
            
        if signal == RlcsSignal.DOWNWEIGHT:
            return nominal_alpha, nominal_beta
            
        if signal == RlcsSignal.DEFER:
            return nominal_alpha * self.attenuation_factor, nominal_beta * self.attenuation_factor
            
        if signal == RlcsSignal.ABSTAIN:
            return 0.0, 0.0
            
        return 0.0, 0.0