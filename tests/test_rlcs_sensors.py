"""
Tests for RLCS Sensors.

Verifies the mathematical correctness and determinism of the sensor functions.
"""

import unittest
import numpy as np
from resed.rlcs.sensors import population_consistency, temporal_consistency, agreement_consistency

class TestRlcsSensors(unittest.TestCase):
    
    def test_population_consistency(self):
        """Test ResLik-style population consistency calculation."""
        z = np.array([[3.0, 4.0], [0.0, 0.0]])
        mu = np.array([0.0, 0.0])
        sigma = 1.0
        
        # Norm of [3, 4] is 5. D = 5 / 1 = 5.
        # Norm of [0, 0] is 0. D = 0 / 1 = 0.
        scores = population_consistency(z, mu, sigma)
        
        self.assertAlmostEqual(scores[0], 5.0)
        self.assertAlmostEqual(scores[1], 0.0)
        
    def test_temporal_consistency(self):
        """Test temporal consistency calculation."""
        # z0 = [0, 0], z1 = [3, 4] (dist 5), z2 = [3, 4] (dist 0)
        z = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
        
        scores = temporal_consistency(z)
        
        # T0 defaults to 1
        self.assertEqual(scores[0], 1.0)
        
        # T1 = exp(-5)
        self.assertAlmostEqual(scores[1], np.exp(-5.0))
        
        # T2 = exp(-0) = 1
        self.assertAlmostEqual(scores[2], 1.0)
        
    def test_agreement_consistency(self):
        """Test agreement consistency (cosine similarity)."""
        z = np.array([[1.0, 0.0], [1.0, 0.0]])
        z_prime = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        scores = agreement_consistency(z, z_prime)
        
        # z[0] same as z_prime[0] -> cos sim 1.0
        self.assertAlmostEqual(scores[0], 1.0)
        
        # z[1] orthogonal to z_prime[1] -> cos sim 0.0
        self.assertAlmostEqual(scores[1], 0.0)

if __name__ == '__main__':
    unittest.main()
