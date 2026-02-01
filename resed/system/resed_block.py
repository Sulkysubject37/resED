"""
ResED Block.

The fundamental building block of the resED system.
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
        # We process by grouping samples with same signal to batched execution
        # where possible, or just iterate. Given requirements for "System",
        # explicit routing is key.
        
        batch_size = x.shape[0]
        outputs = [None] * batch_size
        
        # Group indices by signal
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, sig in enumerate(signals):
            groups[sig].append(idx)
            
        # Execute per group
        for sig, indices in groups.items():
            # Get effective params
            eff_alpha, eff_beta = self.governance.route(sig, nominal_alpha, nominal_beta)
            
            # Extract batch subset
            indices_arr = np.array(indices)
            z_subset = z_enc[indices_arr]
            
            # Apply resTR (Refine)
            # Note: resTR is stateless, so passing subset is fine.
            # However, for Temporal Consistency, resTR doesn't care (it's forward only).
            # But RLCS Temporal Sensor relied on sequence order. 
            # RLCS Check happened *before* this split on the full batch.
            # So we are safe.
            z_ref = self.restr.forward(z_subset, alpha=eff_alpha, beta=eff_beta)
            
            # Apply resDEC (Decode)
            # Decoder takes signal and handles logic (e.g. None for ABSTAIN)
            # But ResDEC.decode takes ONE signal.
            # We call it with the signal for this group.
            # ResDEC returns None if signal is DEFER/ABSTAIN.
            y_subset = self.decoder.decode(z_ref, sig)
            
            # Place results back
            if y_subset is not None:
                for i, original_idx in enumerate(indices):
                    outputs[original_idx] = y_subset[i]
            else:
                # y_subset is None, meaning all outputs in this group are None
                for original_idx in indices:
                    outputs[original_idx] = None
                    
        return outputs, diagnostics