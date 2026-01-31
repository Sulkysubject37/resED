"""
Tests for RLCS Control Surface.

Verifies the signal escalation logic and integration of sensors.
"""

import unittest
import numpy as np
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal
from resed.rlcs.thresholds import TAU_D, TAU_T, TAU_A

class TestRlcsControl(unittest.TestCase):
    
    def test_proceed_nominal(self):
        """Test nominal case (PROCEED)."""
        # z close to 0 (mean), T=1 (default for batch[0]), no Z'
        z = np.zeros((1, 5)) 
        s = np.zeros((1, 4)) # Dummy stats
        
        signals = rlcs_control(z, s)
        self.assertEqual(signals[0], RlcsSignal.PROCEED)
        
    def test_abstain_population(self):
        """Test ABSTAIN on population anomaly."""
        # z far from 0. Norm = 10. TAU_D = 3.0. D = 10/1 = 10 > 3.
        z = np.zeros((1, 5))
        z[0, 0] = 10.0
        s = np.zeros((1, 4))
        
        signals = rlcs_control(z, s)
        self.assertEqual(signals[0], RlcsSignal.ABSTAIN)
        
    def test_defer_temporal(self):
        """Test DEFER on temporal drift."""
        # Need batch > 1.
        # z0 = [0...], z1 = [100...]
        # z1 -> D > TAU_D (ABSTAIN).
        # We want D < TAU_D but T < TAU_T.
        # TAU_D = 3.0. Max norm ~3.
        # TAU_T = 0.5. exp(-dist) < 0.5 => -dist < ln(0.5) ~ -0.69 => dist > 0.69.
        # So we need norm < 3 but diff > 0.69.
        
        z = np.zeros((2, 5))
        z[0] = [0, 0, 0, 0, 0]
        z[1] = [1.0, 0, 0, 0, 0] # Dist = 1.0. T = exp(-1) = 0.36 < 0.5. D = 1.0 < 3.0.
        
        s = np.zeros((2, 4))
        
        signals = rlcs_control(z, s)
        
        self.assertEqual(signals[0], RlcsSignal.PROCEED) # T0 = 1
        self.assertEqual(signals[1], RlcsSignal.DEFER)
        
    def test_downweight_agreement(self):
        """Test DOWNWEIGHT on disagreement."""
        # Need D < 3, T > 0.5, A < 0.8.
        z = np.array([[1.0, 0.0]]) # Norm 1. D=1.
        # z' orthogonal
        z_prime = np.array([[0.0, 1.0]])
        s = np.zeros((1, 4))
        
        signals = rlcs_control(z, s, z_prime=z_prime)
        
        self.assertEqual(signals[0], RlcsSignal.DOWNWEIGHT)
        
    def test_priority_abstain_over_others(self):
        """Test that ABSTAIN overrides DEFER/DOWNWEIGHT."""
        # Z triggers D > 3 (ABSTAIN) AND A < 0.8 (DOWNWEIGHT)
        z = np.array([[10.0, 0.0]]) # D=10
        z_prime = np.array([[0.0, 1.0]]) # A=0
        s = np.zeros((1, 4))
        
        signals = rlcs_control(z, s, z_prime=z_prime)
        
        self.assertEqual(signals[0], RlcsSignal.ABSTAIN)

if __name__ == '__main__':
    unittest.main()