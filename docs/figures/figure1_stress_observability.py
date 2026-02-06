"""
Figure 1: Stress Signal Observability.

Demonstrates that representation-level failures are detectable by RLCS sensors.
Updated for enhanced readability and non-overlapping text.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.system.resed_block import ResEdBlock
from resed.validation.faults import inject_gradual_drift

def generate_figure1():
    # Setup
    d_in, d_z, d_out = 10, 16, 5
    block = ResEdBlock(d_in, d_z, d_out)
    
    # Initialize weights for realistic latent projection
    rng = np.random.default_rng(42)
    W_enc = rng.uniform(-0.5, 0.5, (d_in, d_z))
    b_enc = np.zeros(d_z)
    block.encoder.set_weights(W_enc, b_enc)
    
    # 1. Gradual Drift Scenario
    batch_size = 50
    drift_rate = 0.2
    x = np.zeros((batch_size, d_in))
    z_enc, s_enc = block.encoder.encode(x)
    
    # Inject drift
    z_drifted = inject_gradual_drift(z_enc, drift_rate=drift_rate)
    
    # Get diagnostics
    signals, diag = block.governance.diagnose(z_drifted, s_enc)
    
    t_scores = diag['temporal_consistency']
    d_scores = diag['population_consistency']
    
    # Plotting
    # Use constrained_layout for better auto-spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    
    # --- Subplot 1: Population Consistency ---
    ax1.plot(d_scores, label='ResLik Score (D)', color='crimson', lw=3)
    ax1.axhline(y=3.0, color='black', linestyle='--', alpha=0.7, label='Threshold (TAU_D = 3.0)')
    
    # Info Box (Top Left, fixed)
    ax1.text(0.02, 0.9, f'Fault: Gradual Drift\nRate = {drift_rate} / step', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Annotate Response (find crossing)
    cross_idx = np.where(d_scores > 3.0)[0]
    if len(cross_idx) > 0:
        idx = cross_idx[0]
        # Move text relative to the point to avoid covering the line
        # Pointing to the specific crossing point
        ax1.annotate('RLCS Violation\n(Signal: ABSTAIN)', 
                     xy=(idx, 3.0), 
                     xytext=(idx - 15, 3.0 + 1.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))
        
        # Highlight Unsafe Region
        ax1.axvspan(idx, batch_size, color='red', alpha=0.1)
        ax1.text(idx + 2, max(d_scores) * 0.9, "UNSAFE REGION", color='darkred', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Population Distance (D)', fontsize=12)
    ax1.set_title('Sensor 1: Population Consistency vs. Drift', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(d_scores) * 1.1) # Add headroom
    
    # --- Subplot 2: Temporal Consistency ---
    ax2.plot(t_scores, label='TCS Score (T)', color='steelblue', lw=3)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold (TAU_T = 0.5)')
    
    # Annotate Response (find drop)
    drop_idx = np.where(t_scores < 0.5)[0]
    if len(drop_idx) > 0:
        idx = drop_idx[0]
        # Place text in the "clean" area (usually above the curve as it drops)
        ax2.annotate('Consistency Drop\n(Signal: DEFER)', 
                     xy=(idx, 0.5), 
                     xytext=(idx + 5, 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))
        
        ax2.axvspan(idx, batch_size, color='orange', alpha=0.1)
        ax2.text(idx + 2, 0.2, "UNSTABLE REGION", color='darkorange', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Temporal Consistency (T)', fontsize=12)
    ax2.set_title('Sensor 2: Temporal Consistency vs. Drift', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Batch Index (Time Step)', fontsize=12)
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1) # Fixed range for probability-like score
    
    # Global Title
    plt.suptitle("Figure 2: RLCS Sensor Observability under Stress", fontsize=16)
    
    output_path = 'docs/figures/figure1_stress_observability.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure 1 saved to {output_path}")

if __name__ == "__main__":
    generate_figure1()
