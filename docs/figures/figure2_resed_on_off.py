"""
Figure 3: System Resilience to Sudden Shift (On/Off).

Compares governed vs ungoverned system behavior.
Includes an example trace and a statistical aggregate over 1,000 trials.
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

def run_simulation(n_trials=1000):
    d_in, d_z, d_out = 10, 16, 5
    block = ResEdBlock(d_in, d_z, d_out)
    rng = np.random.default_rng(42)
    
    # Random weights
    W_enc = rng.uniform(-0.5, 0.5, (d_in, d_z))
    block.encoder.set_weights(W_enc, np.zeros(d_z))
    U_dec = rng.uniform(-0.5, 0.5, (d_z, d_out))
    block.decoder.set_weights(U_dec, np.zeros(d_out))
    
    # Stats
    rejection_rates = []
    suppression_norms = [] # Norm of output under shock (should be 0)
    hallucination_norms = [] # Norm of ungoverned output under shock
    
    for _ in range(n_trials):
        # Clean Z
        x = np.zeros((1, d_in))
        z_enc, s_enc = block.encoder.encode(x)
        
        # Inject Shock
        z_fail = inject_distribution_shift(z_enc.copy(), shift_magnitude=10.0)
        
        # Governed
        signals, _ = block.governance.diagnose(z_fail, s_enc)
        y_on = block.decoder.decode(z_fail, signals[0])
        
        # Ungoverned
        y_off = block.decoder.decode(z_fail, RlcsSignal.PROCEED)
        
        rejection_rates.append(1.0 if signals[0] == RlcsSignal.ABSTAIN else 0.0)
        suppression_norms.append(np.linalg.norm(y_on) if y_on is not None else 0.0)
        hallucination_norms.append(np.linalg.norm(y_off))
        
    return rejection_rates, suppression_norms, hallucination_norms

def generate_figure3():
    # 1. Run Aggregate Simulation
    print("Running 1,000-trial simulation...")
    rejection_rates, suppression_norms, hallucination_norms = run_simulation(1000)
    
    # 2. Setup Example Trace (Same as old logic)
    d_in, d_z, d_out = 10, 16, 5
    block = ResEdBlock(d_in, d_z, d_out)
    rng = np.random.default_rng(42)
    block.encoder.set_weights(rng.uniform(-0.5, 0.5, (d_in, d_z)), np.zeros(d_z))
    block.decoder.set_weights(rng.uniform(-0.5, 0.5, (d_z, d_out)), np.zeros(d_out))
    
    batch_size = 50
    shift_index = 25
    x = np.zeros((batch_size, d_in))
    z_enc, s_enc = block.encoder.encode(x)
    z_fail = z_enc.copy()
    z_fail[shift_index:] = inject_distribution_shift(z_fail[shift_index:], shift_magnitude=10.0)
    
    outputs_off = []
    outputs_on = []
    signals, _ = block.governance.diagnose(z_fail, s_enc)
    for i in range(batch_size):
        y_off = block.decoder.decode(z_fail[i:i+1], RlcsSignal.PROCEED)
        outputs_off.append(np.linalg.norm(y_off))
        y_on = block.decoder.decode(z_fail[i:i+1], signals[i])
        outputs_on.append(np.linalg.norm(y_on) if y_on is not None else 0.0)

    # Plotting
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    
    # Left: Trace (spanning 2 rows?) No, let's keep trace on top, hist on bottom.
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(outputs_off, label='Ungoverned (Bypass)', color='gray', linestyle='--', alpha=0.5)
    ax1.plot(outputs_on, label='Governed (resED ON)', color='forestgreen', lw=3)
    ax1.axvline(x=shift_index, color='red', linestyle=':')
    ax1.set_title("Example System Trace under Shock", fontweight='bold')
    ax1.set_ylabel("Output Norm ||Y||")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Bottom Left: Logic Trace
    ax2 = fig.add_subplot(gs[1, 0])
    signal_map = {RlcsSignal.PROCEED: 0, RlcsSignal.DOWNWEIGHT: 1, RlcsSignal.DEFER: 2, RlcsSignal.ABSTAIN: 3}
    codes = [signal_map[s] for s in signals]
    ax2.step(range(batch_size), codes, color='darkorange', lw=2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['PROC', 'DOWN', 'DEFER', 'ABS'])
    ax2.set_title("Governance Logic (Example)", fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Bottom Right: Aggregate Histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(hallucination_norms, bins=30, alpha=0.4, label='Ungoverned', color='gray')
    ax3.hist(suppression_norms, bins=30, alpha=0.8, label='Governed', color='forestgreen')
    ax3.set_title(f"Statistical Aggregate (N=1,000 Trials)", fontweight='bold')
    ax3.set_xlabel("Mean Output Norm under Shock")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    fig.suptitle("Figure 3: System Resilience to Shock (Governed vs. Ungoverned)", fontsize=16, y=1.02)
    
    output_path = 'docs/figures/figure2_resed_on_off.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure 3 saved to {output_path}")

if __name__ == "__main__":
    generate_figure3()
