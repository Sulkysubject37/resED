"""
RLCS Thresholds.

Defines the constant thresholds for the RLCS control logic.
These values govern the sensitivity of the sensors.
"""

# Population Consistency Threshold (Z-score like)
TAU_D = 3.0

# Temporal Consistency Threshold (Decay factor)
TAU_T = 0.5

# Agreement Consistency Threshold (Cosine similarity)
TAU_A = 0.8
