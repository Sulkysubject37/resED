"""
RLCS Governance Layer.

This module implements the adapter and routing logic for the RLCS system.
It acts as the bridge between raw RLCS diagnostics and the component execution.
"""

import numpy as np
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal

class RlcsGovernance:
    """
    Governance system for resED.
    
    Responsible for:
    1. Adapting system state to RLCS inputs.
    2. Computing control signals via RLCS.
    3. Routing execution parameters based on signals.
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
        - DOWNWEIGHT: Nominal alpha, beta (Dec handles attenuation).
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
            # Phase 4 prompt says "DEFER -> Attenuated or softened resTR".
            # It implies DOWNWEIGHT (Agreement issue) might also want attenuation or just let Dec handle it.
            # RLCS logic: ABSTAIN > DEFER > DOWNWEIGHT > PROCEED.
            # Phase 1 Dec: DOWNWEIGHT -> output scaled by alpha.
            # Phase 4 Table:
            # PROCEED -> Normal resTR -> resDEC
            # DEFER -> Attenuated resTR
            # ABSTAIN -> Skip resTR
            # DOWNWEIGHT is missing from Phase 4 Table but exists in RLCS.
            # I will treat DOWNWEIGHT as PROCEED for resTR (letting Decoder handle scaling), 
            # or maybe attenuated?
            # Given "Agreement Consistency" (DOWNWEIGHT), if views disagree, maybe we shouldn't refine strongly?
            # I'll stick to: DOWNWEIGHT -> Nominal resTR, Decoder scales output.
            return nominal_alpha, nominal_beta
            
        if signal == RlcsSignal.DEFER:
            # Attenuated resTR
            return nominal_alpha * self.attenuation_factor, nominal_beta * self.attenuation_factor
            
        if signal == RlcsSignal.ABSTAIN:
            # Skip resTR
            return 0.0, 0.0
            
        return 0.0, 0.0
