"""
ResED Block.

Fundamental building block of the resED system.
Integrates Encoder, Transformer, and Decoder under RLCS Governance.
"""

import numpy as np
from resed.encoders.resenc import ResENC
from resed.decoders.resdec import ResDEC
from resed.restr.restr import ResTR
from resed.system.governance import RlcsGovernance
from resed.rlcs.types import RlcsSignal

class ResEdBlock:
    """
    RLCS-Governed Encoder-Transformer-Decoder Block.
    """
    
    def __init__(self, d_in: int, d_z: int, d_out: int, 
                 n_heads: int = 4,
                 enc_phi=np.tanh, dec_psi=lambda x: x,
                 attenuation_factor: float = 0.5):
        """
        Initialize the system block.
        
        Args:
            d_in: Input dimension.
            d_z: Latent dimension.
            d_out: Output dimension.
            n_heads: Number of attention heads for resTR.
            enc_phi: Encoder activation.
            dec_psi: Decoder activation.
            attenuation_factor: Attenuation for DEFER signal.
        """
        self.encoder = ResENC(d_in, d_z, phi=enc_phi)
        self.restr = ResTR(d_z, n_heads)
        self.decoder = ResDEC(d_z, d_out, psi=dec_psi)
        self.governance = RlcsGovernance(attenuation_factor=attenuation_factor)
        
    def forward(self, x: np.ndarray, 
                nominal_alpha: float = 0.0, nominal_beta: float = 0.0,
                **rlcs_kwargs) -> tuple[list, dict]:
        """
        Execute the pipeline: Enc -> RLCS -> resTR -> Dec.
        
        Args:
            x: Input batch (batch_size, d_in).
            nominal_alpha: Desired attention refinement scale.
            nominal_beta: Desired FFN refinement scale.
            **rlcs_kwargs: Context for RLCS (mu, sigma, z_prime).
            
        Returns:
            outputs: List of outputs (np.ndarray or None).
            diagnostics: RLCS diagnostics dictionary.
        """
        # 1. Encode
        z_enc, s_enc = self.encoder.encode(x)
        
        # 2. RLCS Governance (Diagnose)
        signals, diagnostics = self.governance.diagnose(z_enc, s_enc, **rlcs_kwargs)
        
        # 3. Execution (Route & Execute)
        batch_size = x.shape[0]
        outputs = [None] * batch_size
        
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, sig in enumerate(signals):
            groups[sig].append(idx)
            
        for sig, indices in groups.items():
            eff_alpha, eff_beta = self.governance.route(sig, nominal_alpha, nominal_beta)
            
            indices_arr = np.array(indices)
            z_subset = z_enc[indices_arr]
            
            # Apply resTR (Refine)
            z_ref = self.restr.forward(z_subset, alpha=eff_alpha, beta=eff_beta)
            
            # Apply resDEC (Decode)
            y_subset = self.decoder.decode(z_ref, sig)
            
            if y_subset is not None:
                for i, original_idx in enumerate(indices):
                    outputs[original_idx] = y_subset[i]
            else:
                for original_idx in indices:
                    outputs[original_idx] = None
                    
        return outputs, diagnostics
