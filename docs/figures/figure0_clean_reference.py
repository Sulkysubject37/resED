"""
Figure 0: Clean Reference Baseline (Phase 7-B).

Captures normal RLCS behavior on clean, unperturbed ResNet-50 embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.system.resed_block import ResEdBlock
from resed.rlcs.types import RlcsSignal

def generate_figure0():
    # Paths
    embedding_path = "experiments/benchmarks/cifar10_resnet50_embeddings.npy"
    if not os.path.exists(embedding_path):
        print(f"Error: Embeddings not found at {embedding_path}")
        return

    # Load Embeddings (X)
    X_clean = np.load(embedding_path)
    # Use subset for visualization clarity if needed, or full batch
    # Phase 7-A saved 2000. Let's visualize first 200 for clarity in plots.
    X_vis = X_clean[:200]
    
    # System Setup
    d_in = X_clean.shape[1] # 2048
    d_z = 64
    d_out = 10 # Dummy output dim
    block = ResEdBlock(d_in, d_z, d_out)
    
    # Initialize weights (Simulate trained model)
    rng = np.random.default_rng(42)
    block.encoder.set_weights(
        rng.uniform(-0.1, 0.1, (d_in, d_z)), 
        np.zeros(d_z)
    )
    # Set reference stats for RLCS based on this clean batch?
    # Phase 6 contract says "Pipelines where reference statistics are stable and estimable".
    # We should estimate mu/sigma from a "train" split, but here we only have validation.
    # For baseline visualization, we can use the batch stats as reference (self-referenced) 
    # to show "ideal" behavior, or use a subset.
    # Let's use the first 100 as "reference" and the next 100 as "test" (visualized).
    # Wait, X_vis is 200.
    
    # Encode all
    z, s = block.encoder.encode(X_vis)
    
    # Calculate Reference Stats (mu, sigma) from the batch itself to simulate "In-Distribution"
    # In a real scenario, these come from training data.
    ref_mu = np.mean(z, axis=0)
    ref_std = np.std(z, axis=0)
    # Handle zero std if any
    ref_std[ref_std < 1e-6] = 1.0 
    # We pass scalar sigma to RLCS sensors? 
    # resed/rlcs/sensors.py: population_consistency(z, mu, sigma). 
    # If mu is vector, sigma can be scalar or vector? 
    # "sigma: Reference standard deviation (scalar)". 
    # Phase 2 implementation uses scalar sigma in signature but can broadcast if numpy handles it.
    # Let's check `resed/utils/stats.py` or `sensors.py`.
    # `dist / (sigma + epsilon)`. If sigma is vector, it works elementwise?
    # But `dist` is norm (scalar per sample).
    # So `sigma` should be scalar (global std) or we need Mahalanobis?
    # Phase 2 `population_consistency`: "D_i = ||z_i - mu||_2 / (sigma + epsilon)".
    # This implies isotropic sigma (scalar).
    # So we compute scalar std (mean of stds or std of norms?).
    # Usually average std across dims.
    scalar_sigma = np.mean(ref_std)
    
    # Run Governance
    signals, diag = block.governance.diagnose(z, s, mu=ref_mu, sigma=scalar_sigma)
    
    # Metrics
    d_scores = diag['population_consistency']
    t_scores = diag['temporal_consistency']
    
    # Signals
    signal_map = {RlcsSignal.PROCEED: 0, RlcsSignal.DOWNWEIGHT: 1, RlcsSignal.DEFER: 2, RlcsSignal.ABSTAIN: 3}
    signal_codes = [signal_map[s] for s in signals]
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # Panel 1: Population Consistency (D)
    axs[0, 0].plot(d_scores, color='forestgreen', lw=1.5, alpha=0.8)
    axs[0, 0].axhline(3.0, color='gray', linestyle='--', label='Threshold (3.0)')
    axs[0, 0].set_title("Population Consistency (ResLik)", fontweight='bold')
    axs[0, 0].set_ylabel("Score (D)")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Panel 2: Temporal Consistency (T)
    # Note: These are independent images, so temporal consistency is artificial here.
    # But we visualize it as "batch stability".
    axs[0, 1].plot(t_scores, color='steelblue', lw=1.5, alpha=0.8)
    axs[0, 1].axhline(0.5, color='gray', linestyle='--', label='Threshold (0.5)')
    axs[0, 1].set_title("Temporal Consistency (TCS)", fontweight='bold')
    axs[0, 1].set_ylabel("Score (T)")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # Panel 3: Signal Distribution
    axs[1, 0].hist(signal_codes, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='purple', alpha=0.7, rwidth=0.8)
    axs[1, 0].set_xticks([0, 1, 2, 3])
    axs[1, 0].set_xticklabels(['PROCEED', 'DOWNWEIGHT', 'DEFER', 'ABSTAIN'])
    axs[1, 0].set_title("Control Signal Distribution", fontweight='bold')
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].grid(axis='y', alpha=0.3)
    
    # Panel 4: Latent Norm Distribution
    latent_norms = np.linalg.norm(z, axis=1)
    axs[1, 1].hist(latent_norms, bins=30, color='gray', alpha=0.7)
    axs[1, 1].set_title("Latent Norm Distribution", fontweight='bold')
    axs[1, 1].set_xlabel("||z||")
    axs[1, 1].grid(axis='y', alpha=0.3)
    
    fig.suptitle(f"Figure 0: Clean Reference Baseline (N={len(X_vis)})", fontsize=16)
    
    output_path = "docs/figures/figure0_clean_reference.pdf"
    plt.savefig(output_path)
    print(f"Figure 0 saved to {output_path}")

if __name__ == "__main__":
    generate_figure0()
