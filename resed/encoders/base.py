"""
Base Encoder Interface.

Defines the contract for all encoders in the resED system.

"""

class BaseEncoder:
    """
    Abstract base class for all resED encoders.
    """
    def __init__(self):
        pass

    def encode(self, x):
        """
        Transform input x into a latent representation.
        """
        raise NotImplementedError
