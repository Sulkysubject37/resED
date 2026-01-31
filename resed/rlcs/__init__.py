"""
RLCS (Representation-Level Control Surfaces) Module.

Implements the control surface logic, sensors, and aggregation mechanisms
inherited from the resLik architecture.
"""

from .types import RlcsSignal
from .sensors import population_consistency, temporal_consistency, agreement_consistency
from .thresholds import TAU_D, TAU_T, TAU_A