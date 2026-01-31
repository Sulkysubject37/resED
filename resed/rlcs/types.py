"""
RLCS Types.

Defines the control signals and types for the Representation-Level Control Surface.
"""

from enum import Enum

class RlcsSignal(str, Enum):
    """
    Control signals emitted by the RLCS layer.
    """
    PROCEED = "PROCEED"
    DOWNWEIGHT = "DOWNWEIGHT"
    DEFER = "DEFER"
    ABSTAIN = "ABSTAIN"
