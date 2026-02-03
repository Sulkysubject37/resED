"""
RLCS Calibrator.

Manages the calibration of raw sensor diagnostics into normalized risk scores.
"""

import numpy as np
from scipy.special import ndtri
from resed.calibration.quantile import estimate_quantiles, map_to_quantile

class RlcsCalibrator:
    """
    Calibrates RLCS sensor outputs using reference data.
    Maps raw scores to Z-scores (Standard Normal Quantiles) to align with
    fixed RLCS thresholds (e.g., TAU=3.0 implies 3-sigma rarity).
    """
    
    def __init__(self):
        self.reference_distributions = {}
        self.is_calibrated = False
        self.epsilon = 1e-6 # Bound for numerical stability (approx 4.75 sigma)

    def fit(self, diagnostics: dict):
        """
        Fit calibration curves from reference diagnostics.
        
        Args:
            diagnostics: Dictionary of {sensor_name: raw_scores_array}.
        """
        self.reference_distributions = {}
        for sensor, scores in diagnostics.items():
            q, vals = estimate_quantiles(scores)
            self.reference_distributions[sensor] = (q, vals)
        self.is_calibrated = True

    def _to_z_score(self, quantile: float) -> float:
        """Convert quantile (0, 1) to Z-score (-inf, inf)."""
        # Clamp to avoid inf
        q_clamped = np.clip(quantile, self.epsilon, 1.0 - self.epsilon)
        return float(ndtri(q_clamped))

    def _to_z_score_batch(self, quantiles: np.ndarray) -> np.ndarray:
        """Vectorized Z-score conversion."""
        q_clamped = np.clip(quantiles, self.epsilon, 1.0 - self.epsilon)
        return ndtri(q_clamped)

    def calibrate(self, sensor_name: str, raw_value: float) -> float:
        """
        Convert raw sensor value to calibrated Z-score.
        
        Args:
            sensor_name: Name of the sensor (e.g., 'population_consistency').
            raw_value: Raw diagnostic score.
            
        Returns:
            Z-score relative to reference distribution.
        """
        if not self.is_calibrated or sensor_name not in self.reference_distributions:
            return raw_value
            
        q, vals = self.reference_distributions[sensor_name]
        rank = map_to_quantile(raw_value, q, vals)
        return self._to_z_score(rank)

    def calibrate_batch(self, sensor_name: str, raw_values: np.ndarray) -> np.ndarray:
        """
        Vectorized calibration.
        """
        if not self.is_calibrated or sensor_name not in self.reference_distributions:
            return raw_values
            
        q, vals = self.reference_distributions[sensor_name]
        ranks = np.interp(raw_values, vals, q, left=0.0, right=1.0)
        return self._to_z_score_batch(ranks)
