"""
Figure 2: resED OFF vs resED ON.

Demonstrates system behavior difference with and without RLCS governance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.system.resed_block import ResEdBlock
from resed.validation.faults import inject_distribution_shift
from resed.rlcs.types import RlcsSignal

def generate_figure2():
    # Setup
    d_in, d_z, d_out = 10, 16, 5
    block = ResEdBlock(d_in, d_z, d_out)
    
    # Sequence with sudden shift at index 25
    batch_size = 50
    x = np.zeros((batch_size, d_in))
    z_enc, s_enc = block.encoder.encode(x)
    
    # Inject sudden shift at halfway
    z_fail = z_enc.copy()
    z_fail[25:] = inject_distribution_shift(z_fail[25:], shift_magnitude=10.0)
    
    # 1. resED OFF (Bypass RLCS - everything is PROCEED)
    outputs_off = []
    for i in range(batch_size):
        # Decode ignoring any issues
        y = block.decoder.decode(z_fail[i:i+1], RlcsSignal.PROCEED)
        outputs_off.append(np.linalg.norm(y))
        
    # 2. resED ON (Governance active)
    # We need to simulate the forward pass with the failed latent
    # To do this cleanly, we manually trigger the governance and decode
    signals, _ = block.governance.diagnose(z_fail, s_enc)
    outputs_on = []
    control_codes = []
    
    signal_map = {RlcsSignal.PROCEED: 0, RlcsSignal.DOWNWEIGHT: 1, RlcsSignal.DEFER: 2, RlcsSignal.ABSTAIN: 3}
    
    for i in range(batch_size):
        sig = signals[i]
        control_codes.append(signal_map[sig])
        y = block.decoder.decode(z_fail[i:i+1], sig)
        outputs_on.append(np.linalg.norm(y) if y is not None else 0.0)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(outputs_off, label='resED OFF (Ungoverned)', color='gray', alpha=0.5, lw=2)
    ax1.plot(outputs_on, label='resED ON (RLCS Governed)', color='forestgreen', lw=2)
    ax1.set_ylabel('Output Norm (||Y||)')
    ax1.set_title('System Behavior: resED OFF vs ON (Sudden Distribution Shift)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.step(range(batch_size), control_codes, where='post', color='darkorange', lw=2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['PROCEED', 'DOWNWEIGHT', 'DEFER', 'ABSTAIN'])
    ax2.set_ylabel('RLCS Control Signal')
    ax2.set_xlabel('Batch Index')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'docs/figures/figure2_resed_on_off.png'
    plt.savefig(output_path)
    print(f"Figure 2 saved to {output_path}")

if __name__ == "__main__":
    generate_figure2()
