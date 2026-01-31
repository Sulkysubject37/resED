"""
Tests for resENC (Reference Encoder).

Verifies determinism, shape correctness, and statistical channel output.
"""

import unittest
import numpy as np
from resed.encoders.resenc import ResENC

class TestResENC(unittest.TestCase):
    def setUp(self):
        self.d_in = 10
        self.d_z = 5
        self.encoder = ResENC(self.d_in, self.d_z)
        
        # Set deterministic weights
        self.W = np.eye(self.d_in, self.d_z) # Simple projection
        self.b = np.zeros(self.d_z)
        self.encoder.set_weights(self.W, self.b)

    def test_shape_preservation(self):
        """Test that output shapes match expectations."""
        batch_size = 3
        x = np.random.randn(batch_size, self.d_in)
        z, s = self.encoder.encode(x)
        
        self.assertEqual(z.shape, (batch_size, self.d_z))
        self.assertEqual(s.shape, (batch_size, 4)) # 4 stats

    def test_determinism(self):
        """Test that same input yields exact same output."""
        x = np.random.randn(2, self.d_in)
        z1, s1 = self.encoder.encode(x)
        z2, s2 = self.encoder.encode(x)
        
        np.testing.assert_array_equal(z1, z2)
        np.testing.assert_array_equal(s1, s2)

    def test_statistics_calculation(self):
        """Test correctness of statistical channel."""
        # Create a specific input to verify stats
        # x = [1, 0, ...] -> z = tanh([1, 0, ...])
        x = np.zeros((1, self.d_in))
        x[0, 0] = 10.0 # Large value to push tanh to 1.0
        
        # W maps x[0] to z[0] = 10.0 -> tanh(10.0) ~= 1.0
        # z will be approx [1.0, 0, 0, 0, 0] (since W is eye-like mapping)
        
        z, s = self.encoder.encode(x)
        zi = z[0]
        
        # Check Norm
        expected_norm = np.linalg.norm(zi)
        self.assertAlmostEqual(s[0, 0], expected_norm)
        
        # Check Variance
        expected_var = np.var(zi)
        self.assertAlmostEqual(s[0, 1], expected_var)
        
        # Check Entropy (softmax of zi)
        exps = np.exp(zi - np.max(zi))
        probs = exps / np.sum(exps)
        expected_entropy = -np.sum(probs * np.log(probs + 1e-12))
        self.assertAlmostEqual(s[0, 2], expected_entropy)

    def test_input_validation(self):
        """Test error handling for bad inputs."""
        with self.assertRaises(ValueError):
            self.encoder.encode(np.zeros((5, self.d_in + 1)))
            
        with self.assertRaises(ValueError):
            self.encoder.encode(np.zeros((self.d_in,))) # 1D instead of 2D

if __name__ == '__main__':
    unittest.main()
