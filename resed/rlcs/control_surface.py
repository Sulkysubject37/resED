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
    
    # Defaults for reference stats (Standard Normal Assumption for Phase 2)
    mu = kwargs.get('mu', 0.0)
    sigma = kwargs.get('sigma', 1.0)
    z_prime = kwargs.get('z_prime', None)
    
    # 1. Compute Diagnostics
    # Population Consistency
    d_scores = population_consistency(z, mu, sigma)
    
    # Temporal Consistency (only for sequence of >1, but computed for all)
    t_scores = temporal_consistency(z)
    
    # Agreement Consistency (if Z' is available)
    a_scores = None
    if z_prime is not None:
        if z_prime.shape != z.shape:
            raise ValueError(f"z_prime shape {z_prime.shape} must match z {z.shape}")
        a_scores = agreement_consistency(z, z_prime)
        
    # Populate diagnostics dict with RAW scores (for analysis/logging)
    if diagnostics is not None:
        diagnostics['population_consistency'] = d_scores
        diagnostics['temporal_consistency'] = t_scores
        if a_scores is not None:
            diagnostics['agreement_consistency'] = a_scores
            
    # 2. Calibration (if enabled)
    # Replaces raw scores for decision making, but keeps raw in diagnostics
    d_decision = d_scores
    t_decision = t_scores
    a_decision = a_scores
    
    if calibrator is not None and calibrator.is_calibrated:
        d_decision = calibrator.calibrate_batch('population_consistency', d_scores)
        # We do NOT calibrate T and A because their thresholds (0.5, 0.8) are absolute
        # bounded values (0-1), whereas D is unbounded and scale-sensitive.
        # Calibrating T/A to Z-scores would break the logic (comparing Z to 0.5).
        # t_decision = calibrator.calibrate_batch('temporal_consistency', t_scores)
        # if a_scores is not None:
        #     a_decision = calibrator.calibrate_batch('agreement_consistency', a_scores)
            
    # 3. Evaluate Control Logic
    for i in range(batch_size):
        # Priority 1: Population (ABSTAIN)
        if d_decision[i] > TAU_D:
            signals.append(RlcsSignal.ABSTAIN)
            continue
            
        # Priority 2: Temporal (DEFER)
        # Note: T is similarity (1.0 is good). 
        # If calibrated to Z-score, it maps to "rarity". 
        # Low T (bad) -> High rarity? Or Low quantile?
        # Map_to_quantile: Low val -> Low rank. 
        # Raw T: 1.0 (Good) -> High rank. 0.0 (Bad) -> Low rank.
        # Z-score: High rank -> High Z. Low rank -> Low Z (negative).
        # Threshold TAU_T = 0.5.
        # If Raw T < 0.5 (Bad). 
        # If Calibrated:
        # We need "Badness" score? Or stick to Z-score of value.
        # If Z-score is -3.0 (Low T), is that < 0.5? Yes.
        # So Low T -> Low Z. Check T < Threshold works if Threshold is Z-score?
        # TAU_T = 0.5 (Raw).
        # If we calibrate, we compare Z-score to 0.5?
        # 0.5 Z-score is mean+0.5std.
        # Ideally, we want "Deviation from expected behavior".
        # For Distance (D): High is bad. High Z is bad. > 3.0 (Rarely high).
        # For Consistency (T): Low is bad. Low Z is bad. < -3.0 (Rarely low).
        # But TAU_T is 0.5. 
        # If we use Z-scores, we should probably standardize the check direction?
        # Or does calibrator map "Bad" to "High Z"?
        # Current calibrator maps "Value" to "Quantile".
        # For D: Value increases -> Quantile increases -> Z increases. (High D -> High Z).
        # For T: Value increases (Good) -> Quantile increases -> Z increases. (High T -> High Z).
        # Bad T is Low Z.
        # If we use TAU_T=0.5 with Z-scores... 0.5 Z is within 1 sigma.
        # If Z < 0.5, we DEFER? 
        # That means if T is below median+0.5std, we DEFER. That's very aggressive (50-70% defer).
        # Maybe for T, we want Z < -3.0 (Rarely low)?
        # But Phase 9 says "Preserves... thresholds".
        # If TAU_T stays 0.5, and Z-scores range [-inf, inf], checking Z < 0.5 is... plausible but implies aggressive gating.
        # Phase 8-B used raw T. Clean data T ~ 1.0?
        # Clean biological data (independent) T fluctuates.
        # If we calibrate T, the clean distribution becomes N(0, 1).
        # Z < 0.5 covers 69% of data.
        # So we would DEFER 69% of clean data.
        # That contradicts "Clean -> majority PROCEED".
        # So for T, the threshold 0.5 likely meant "Absolute score".
        # If we calibrate, we change the domain.
        # Maybe we only calibrate D (Population)?
        # Prompt: "Calibration must support: ResLik, TCS, Agreement".
        # If I calibrate T, I need a threshold compatible with Z-scores.
        # BUT "No threshold tuning".
        # Conflict:
        # 1. Calibrate T to Z-score.
        # 2. Use TAU_T = 0.5.
        # 3. Z < 0.5 implies ~70% rejection.
        # 4. Success criteria: "Clean -> majority PROCEED".
        
        # Resolution: 
        # Perhaps T should NOT be calibrated to Z-score, or T-check should be `Z < -3.0`?
        # Or maybe T calibration is "Low T -> High Risk Score"?
        # If I invert T before quantizing?
        # Or maybe I shouldn't calibrate T to Z-score if TAU is 0.5.
        # But `calibrator` is generic.
        # If I leave T uncalibrated in `rlcs_control` call?
        # "Calibration must support ResLik...".
        # But `rlcs_control` applies the logic.
        # If I calibrate D, D_Z > 3.0 (0.13% prob) -> ABSTAIN.
        # If I don't calibrate T, T_raw < 0.5.
        # Clean T_raw distribution? For independent samples, T depends on random distance.
        # In Phase 8-B, "Temporal Consistency ... fluctuates but generally stays stable".
        # If I calibrate T, I force it to N(0,1).
        # Using TAU=0.5 on Z-score is wrong for "Anomaly Detection" (usually want tails).
        # Using TAU=0.5 on Probability (0-1) is also aggressive (median).
        
        # Hypothesis: TAU_T=0.5 was designed for exp(-dist).
        # Calibrating T breaks this threshold's semantic.
        # HOWEVER, Phase 9 says "Preserves thresholds".
        # This implies the *output of calibration* should be compatible with 0.5?
        # If I calibrate D to Z-score (0..3..), it works with TAU_D=3.0.
        # If I calibrate T to ??? to work with TAU_T=0.5?
        # Maybe T doesn't need calibration if it's already bounded [0,1]?
        # But Phase 8-B showed T dropped for drift.
        # If I calibrate D (which was the main failure mode, "Collapse to ABSTAIN"), that fixes the main issue.
        # I will apply calibration *only* to D (ResLik) in `rlcs_control` call inside `Governance`, 
        # OR I will assume `rlcs_control` logic should use calibrated values if provided.
        
        # Let's look at `RlcsGovernance.diagnose`.
        # I can choose *which* sensors to calibrate.
        # I will calibrate 'population_consistency' (D) because that's the one with the Scale Mismatch.
        # I will NOT calibrate T and A if their raw values are fine (bounded 0-1) and thresholds are absolute (0.5, 0.8).
        # Actually, Z-score calibration for D makes sense (unbounded distance -> normalized dev).
        # T and A are bounded [0, 1]. Calibrating them to Z-score (unbounded) and comparing to 0.5/0.8 (bounded range) is type error.
        # So I should ONLY calibrate D.
        # The prompt says "Calibration must support ... TCS ... Agreement".
        # It doesn't say "Must USE it for control".
        # It says "Calibration output replaces raw diagnostics only at decision time".
        # If I replace T with Z(T), I break the check.
        # So I will only replace D.
        
        # Wait, if I calibrate D to Z-score, `d_decision` is Z-score.
        # `d_decision > TAU_D` (3.0).
        # This works.
        # I will modify `rlcs_control` to apply calibration *if available* for the specific sensor.
        # But `calibrator` has `calibrate_batch(sensor_name, ...)`
        # I will use it for `population_consistency`.
        # I will leave T and A raw in `rlcs_control`?
        # Or I handle T calibration differently (e.g. Probability)?
        # No, "Calibration must be quantile-based... Output: normalized risk score".
        # "Output: normalized risk score \in [0, 1]" (Scope 1).
        # WAIT. Scope 1 says "normalized risk score \in [0, 1]".
        # My `calibrator` implementation outputs Z-score (-inf, inf).
        # I deviated from Scope 1 "Output: normalized risk score".
        # But if output is [0, 1], then TAU_D=3.0 is unreachable.
        # CONTRADICTION in Prompt?
        # 1. "Output: normalized risk score [0, 1]"
        # 2. "Preserves ... thresholds" (TAU=3.0)
        # 3. "Clean ... no longer collapse".
        
        # If Risk is [0, 1] and Threshold is 3.0, condition `Risk > 3.0` is False.
        # So Clean proceeds.
        # But *Perturbed* (Risk=1.0) < 3.0. Also proceeds.
        # This fails "Stress conditions still escalate".
        
        # Interpretation:
        # The threshold `TAU_D` must be interpreted in the *Calibrated* domain?
        # "Preserves thresholds" means don't change the constant `3.0` in the code.
        # If the code checks `val > 3.0`, and val is `[0, 1]`, it's broken.
        # So `val` MUST be able to exceed 3.0.
        # Z-score can exceed 3.0.
        # So my Z-score deviation was *necessary* to satisfy constraints 2 & 3 & 4.
        # The "Output [0, 1]" requirement in Scope 1 might be "Quantile" (conceptual) vs "Operational" value.
        # Or maybe "Risk Score" means something else?
        # I will stick to Z-score because it preserves the *semantics* of "3-sigma" which `TAU=3.0` implies.
        # I will update the docstring in `calibrator.py` to explain this deviation from [0, 1] is for compatibility.
        
        if t_decision[i] < TAU_T:
            signals.append(RlcsSignal.DEFER)
            continue
            
        # Priority 3: Agreement (DOWNWEIGHT)
        if a_decision is not None and a_decision[i] < TAU_A:
            signals.append(RlcsSignal.DOWNWEIGHT)
            continue
            
        # Default: PROCEED
        signals.append(RlcsSignal.PROCEED)
        
    return signals
