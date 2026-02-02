"""
Figure 3: RLCS Stress Response (Phase 7-B).

Compares clean reference behavior against various stress conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.system.resed_block import ResEdBlock
from resed.rlcs.types import RlcsSignal
from resed.validation.faults import (
    inject_gradual_drift, 
    inject_distribution_shift, 
    inject_single_point_shock, 
    inject_view_disagreement
)

def generate_figure3():
    # Setup
    embedding_path = "experiments/benchmarks/cifar10_resnet50_embeddings.npy"
    if not os.path.exists(embedding_path):
        print(f"Error: Embeddings not found at {embedding_path}")
        return
        
    X_clean = np.load(embedding_path)[:200]
    d_in = X_clean.shape[1]
    d_z = 64
    d_out = 10
    block = ResEdBlock(d_in, d_z, d_out)
    rng = np.random.default_rng(42)
    block.encoder.set_weights(rng.uniform(-0.1, 0.1, (d_in, d_z)), np.zeros(d_z))
    
    # Reference Stats
    z_ref, _ = block.encoder.encode(X_clean)
    ref_mu = np.mean(z_ref, axis=0)
    ref_std = np.mean(np.std(z_ref, axis=0))
    if ref_std < 1e-6: ref_std = 1.0

    # Scenarios
    scenarios = {
        "Clean": z_ref,
        "Gradual Drift": None,
        "Sudden Shock": None,
        "High Noise": None
    }
    
    # 1. Gradual Drift
    # Apply to X or Z? Faults apply to Z usually in Phase 5.
    # But here we have X. Phase 5 `faults.py` functions take `z`.
    # So we encode first, then perturb Z.
    scenarios["Gradual Drift"] = inject_gradual_drift(z_ref.copy(), drift_rate=0.1)
    
    # 2. Sudden Shock (at idx 100)
    scenarios["Sudden Shock"] = inject_single_point_shock(z_ref.copy(), index=100, magnitude=20.0)
    
    # 3. High Noise (View Disagreement / Gaussian)
    scenarios["High Noise"] = inject_view_disagreement(z_ref.copy(), noise_magnitude=5.0)
    
    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, constrained_layout=True)
    
    signal_map = {RlcsSignal.PROCEED: 0, RlcsSignal.DOWNWEIGHT: 1, RlcsSignal.DEFER: 2, RlcsSignal.ABSTAIN: 3}
    
    for i, (name, z_perturbed) in enumerate(scenarios.items()):
        # Run Governance
        # We pass s_enc as zeros because `resENC` stats are computed on X->Z. 
        # Here we perturbed Z directly. Stats S should ideally reflect Z.
        # But `rlcs_control` uses S only if we pass it? 
        # `rlcs_control` uses S? Let's check `resed/rlcs/control_surface.py`.
        # It accepts S but Phase 2 sensors (Population, Temporal, Agreement) don't seem to use S explicitly?
        # `population_consistency` uses z, mu, sigma.
        # `temporal_consistency` uses z.
        # `agreement_consistency` uses z, z_prime.
        # S (from encoder) was [norm, var, entropy, sparsity]. 
        # Phase 2 implementation didn't seem to bind S to logic?
        # Let's check `resed/rlcs/control_surface.py` logic.
        # Logic: 1. ABSTAIN if D > TAU. 2. DEFER if T < TAU. 3. DOWNWEIGHT if A < TAU.
        # It doesn't use S. So passing dummy S is fine.
        s_dummy = np.zeros((z_perturbed.shape[0], 4))
        
        signals, diag = block.governance.diagnose(z_perturbed, s_dummy, mu=ref_mu, sigma=ref_std)
        
        d_scores = diag['population_consistency']
        codes = [signal_map[s] for s in signals]
        
        # Plot D-Score
        ax = axs[i]
        ax.plot(d_scores, color='crimson' if name != "Clean" else 'green', label='ResLik Score')
        ax.axhline(3.0, color='black', linestyle='--', label='Threshold')
        
        # Overlay Signals as background color
        # We can use pcolor or vspan.
        # Let's step plot the signals on secondary axis? 
        # Or just color regions.
        # Simple approach: Plot D-Score and mark violations.
        
        ax.set_ylabel("ResLik Score")
        ax.set_title(f"Scenario: {name}", fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(d_scores.max(), 4.0) * 1.1)
        
        # Secondary axis for Signals
        ax2 = ax.twinx()
        ax2.step(range(len(codes)), codes, color='gray', alpha=0.5, where='post', lw=1)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['PROC', 'DOWN', 'DEFER', 'ABS'], fontsize=8)
        ax2.set_ylabel("Signal")
        ax2.set_ylim(-0.5, 3.5)

    fig.suptitle("Figure 3: RLCS Stress Response across Scenarios", fontsize=16)
    
    output_path = "docs/figures/figure3_rlcs_stress_response.pdf"
    plt.savefig(output_path)
    print(f"Figure 3 saved to {output_path}")

if __name__ == "__main__":
    generate_figure3()
