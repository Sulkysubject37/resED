"""
Tests for resDEC (Reference Decoder).

Verifies deterministic decoding and correct handling of control signals.
"""

import unittest
import numpy as np
from resed.decoders.resdec import ResDEC, PROCEED, DOWNWEIGHT, DEFER, ABSTAIN

class TestResDEC(unittest.TestCase):
    def setUp(self):
        self.d_z = 5
        self.d_out = 3
        self.alpha = 0.5
        self.decoder = ResDEC(self.d_z, self.d_out, alpha=self.alpha)
        
        # Set deterministic weights
        self.U = np.zeros((self.d_z, self.d_out))
        # Make U map first latent dim to first output dim directly
        self.U[0, 0] = 1.0 
        self.c = np.zeros(self.d_out)
        self.decoder.set_weights(self.U, self.c)

    def test_proceed_signal(self):
        """Test normal decoding under PROCEED."""
        z = np.zeros((1, self.d_z))
        z[0, 0] = 10.0
        
        y = self.decoder.decode(z, PROCEED)
        
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (1, self.d_out))
        # With U[0,0]=1 and z[0,0]=10, y[0,0] should be 10 (identity activation)
        self.assertEqual(y[0, 0], 10.0)

    def test_downweight_signal(self):
        """Test scaled decoding under DOWNWEIGHT."""
        z = np.zeros((1, self.d_z))
        z[0, 0] = 10.0
        
        y = self.decoder.decode(z, DOWNWEIGHT)
        
        self.assertIsNotNone(y)
        # Should be scaled by alpha
        self.assertEqual(y[0, 0], 10.0 * self.alpha)

    def test_defer_abstain_signals(self):
        """Test null output under DEFER and ABSTAIN."""
        z = np.random.randn(2, self.d_z)
        
        y_defer = self.decoder.decode(z, DEFER)
        self.assertIsNone(y_defer)
        
        y_abstain = self.decoder.decode(z, ABSTAIN)
        self.assertIsNone(y_abstain)

    def test_invalid_control(self):
        """Test rejection of unknown control signals."""
        z = np.random.randn(1, self.d_z)
        with self.assertRaises(ValueError):
            self.decoder.decode(z, "INVALID_SIGNAL")

    def test_determinism(self):
        """Test that same input and control yields same output."""
        z = np.random.randn(2, self.d_z)
        y1 = self.decoder.decode(z, PROCEED)
        y2 = self.decoder.decode(z, PROCEED)
        
        np.testing.assert_array_equal(y1, y2)

if __name__ == '__main__':
    unittest.main()
