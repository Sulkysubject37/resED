"""
Figure 2: resED OFF vs resED ON.

Demonstrates system behavior difference with and without RLCS governance.
Updated for enhanced readability, fixing Subplot 1 text overlaps.
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
    
    # Initialize weights to simulate a trained model
    # Use fixed seed for determinism
    rng = np.random.default_rng(42)
    
    # Encoder: Random projection
    W_enc = rng.uniform(-0.5, 0.5, (d_in, d_z))
    b_enc = np.zeros(d_z)
    block.encoder.set_weights(W_enc, b_enc)
    
    # Decoder: Random projection
    U_dec = rng.uniform(-0.5, 0.5, (d_z, d_out))
    c_dec = np.zeros(d_out)
    block.decoder.set_weights(U_dec, c_dec)
    
    # Sequence with sudden shift at index 25
    batch_size = 50
    shift_magnitude = 10.0
    shift_index = 25
    
    x = np.zeros((batch_size, d_in))
    z_enc, s_enc = block.encoder.encode(x)
    
    # Inject sudden shift at halfway
    z_fail = z_enc.copy()
    z_fail[shift_index:] = inject_distribution_shift(z_fail[shift_index:], shift_magnitude=shift_magnitude)
    
    # 1. resED OFF (Bypass RLCS - everything is PROCEED)
    outputs_off = []
    for i in range(batch_size):
        # Decode ignoring any issues
        y = block.decoder.decode(z_fail[i:i+1], RlcsSignal.PROCEED)
        outputs_off.append(np.linalg.norm(y))
        
    # 2. resED ON (Governance active)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    
    # --- Subplot 1: System Output ---
    max_val = max(outputs_off)
    
    # Plot OFF first (background)
    ax1.plot(outputs_off, label='resED OFF (Ungoverned)', color='gray', linestyle='--', alpha=0.5, lw=2)
    # Plot ON second (foreground)
    ax1.plot(outputs_on, label='resED ON (Governed)', color='forestgreen', lw=3)
    
    # Event Line
    ax1.axvline(x=shift_index, color='red', linestyle=':', lw=2)
    
    # Text: Event Description (Bottom Center, away from data)
    ax1.text(shift_index, -max_val * 0.05, f'Event: Distribution Shift\n(Mag={shift_magnitude})', 
             color='red', fontweight='bold', ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'))
    
    # Annotate: Hallucination (Gray)
    # Find peak in fault region for arrow target
    fault_region = outputs_off[shift_index:]
    peak_rel_idx = np.argmax(fault_region)
    peak_idx = shift_index + peak_rel_idx
    peak_val = fault_region[peak_rel_idx]
    
    # Place text significantly above the peak
    ax1.annotate('Ungoverned Hallucination\n(High Variance)', 
                 xy=(peak_idx, peak_val), 
                 xytext=(peak_idx, max_val * 1.3),
                 arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5),
                 ha='center', va='bottom', fontsize=10, color='dimgray',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
    
    # Annotate: Suppression (Green)
    # Point to the zero line in the fault region
    suppress_idx = shift_index + 10
    ax1.annotate('RLCS Suppresses Output\n(Norm \u2192 0)', 
                 xy=(suppress_idx, 0), 
                 xytext=(suppress_idx + 10, max_val * 0.4),
                 arrowprops=dict(facecolor='forestgreen', shrink=0.05, width=1.5),
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='forestgreen', alpha=0.9))

    ax1.set_ylabel('Decoder Output Norm ||Y||', fontsize=12)
    ax1.set_title('System Response: Governance vs. Bypass', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0.95) 
    ax1.grid(True, alpha=0.3)
    # Expand Y limit to accommodate the text at the top and bottom
    ax1.set_ylim(-max_val*0.2, max_val * 1.6)
    
    # --- Subplot 2: Control Signal ---
    ax2.step(range(batch_size), control_codes, where='post', color='darkorange', lw=2.5)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['PROCEED', 'DOWNWEIGHT', 'DEFER', 'ABSTAIN'], fontsize=10)
    ax2.axvline(x=shift_index, color='red', linestyle=':', lw=2)
    
    # Annotate Transition
    # Point to the step up
    ax2.annotate('Logic Transition:\nPROCEED \u2192 ABSTAIN', 
                 xy=(shift_index, 3), 
                 xytext=(shift_index - 15, 2.8),
                 arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1.5),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkorange', alpha=0.9))
    
    ax2.set_ylabel('RLCS Control Signal', fontsize=12)
    ax2.set_title('Governance Logic Trace', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Batch Index', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 3.5)
    
    # Global Title
    fig.suptitle('Figure 2: System Resilience to Sudden Shift', fontsize=16, y=1.02)
    
    output_path = 'docs/figures/figure2_resed_on_off.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure 2 saved to {output_path}")

if __name__ == "__main__":
    generate_figure2()