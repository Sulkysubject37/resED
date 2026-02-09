"""
Base Decoder Interface.

Defines the contract for all decoders in the resED system.

"""

class BaseDecoder:
    """
    Abstract base class for all resED decoders.
    """
    def __init__(self):
        pass

    def decode(self, z):
        """
        Reconstruct output from latent representation z.
        """
        raise NotImplementedError
