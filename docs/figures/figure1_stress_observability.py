"""
Figure 1: Stress Signal Observability.

Demonstrates that representation-level failures are detectable by RLCS sensors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.system.resed_block import ResEdBlock
from resed.validation.faults import inject_gradual_drift, inject_distribution_shift

def generate_figure1():
    # Setup
    d_in, d_z, d_out = 10, 16, 5
    block = ResEdBlock(d_in, d_z, d_out)
    
    # 1. Gradual Drift Scenario
    batch_size = 50
    x = np.zeros((batch_size, d_in))
    z_enc, s_enc = block.encoder.encode(x)
    
    # Inject drift
    z_drifted = inject_gradual_drift(z_enc, drift_rate=0.2)
    
    # Get diagnostics
    signals, diag = block.governance.diagnose(z_drifted, s_enc)
    
    t_scores = diag['temporal_consistency']
    d_scores = diag['population_consistency']
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(d_scores, label='Population Distance (ResLik)', color='crimson', lw=2)
    ax1.axhline(y=3.0, color='black', linestyle='--', alpha=0.5, label='TAU_D')
    ax1.set_ylabel('Discrepancy Score')
    ax1.set_title('Sensor Response to Gradual Latent Drift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_scores, label='Temporal Consistency (TCS)', color='steelblue', lw=2)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='TAU_T')
    ax2.set_ylabel('Consistency Score')
    ax2.set_xlabel('Batch Index (Time)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'docs/figures/figure1_stress_observability.png'
    plt.savefig(output_path)
    print(f"Figure 1 saved to {output_path}")

if __name__ == "__main__":
    generate_figure1()
