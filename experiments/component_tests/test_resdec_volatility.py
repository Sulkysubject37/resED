"""
Component Test 3: resDEC Volatility .

Quantifies output instability relative to latent corruption.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())

from resed.decoders.resdec import ResDEC, PROCEED
from resed.rlcs.sensors import population_consistency

OUTPUT_DIR = "experiments/component_tests"
FIGURE_DIR = "docs/figures"
D_Z = 64
D_OUT = 10
BATCH_SIZE = 100

def setup_decoder():
    dec = ResDEC(D_Z, D_OUT)
    rng = np.random.default_rng(42)
    U = rng.uniform(-0.1, 0.1, (D_Z, D_OUT))
    c = np.zeros(D_OUT)
    dec.set_weights(U, c)
    return dec

def run_test():
    print("Starting resDEC Volatility Test...")
    decoder = setup_decoder()
    
    rng = np.random.default_rng(42)
    Z_clean = rng.normal(0, 1, (BATCH_SIZE, D_Z))
    Y_clean = decoder.decode(Z_clean, PROCEED)
    
    # Reference Stats
    mu = np.mean(Z_clean, axis=0)
    sigma = np.mean(np.std(Z_clean, axis=0))
    
    results = []
    
    noise_levels = np.linspace(0.0, 2.0, 10)
    for noise in noise_levels:
        Z_pert = Z_clean + rng.normal(0, noise, Z_clean.shape)
        Y_pert = decoder.decode(Z_pert, PROCEED)
        
        # Output Divergence
        div = np.mean(np.linalg.norm(Y_pert - Y_clean, axis=1))
        
        # Sensitivity Ratio (dY / dZ)
        # dZ is roughly sqrt(d_z) * noise
        # dY is div
        # Ratio = div / (noise * sqrt(d_z) + eps)
        dz = noise * np.sqrt(D_Z)
        ratio = div / (dz + 1e-6)
        
        # RLCS Correlation
        d_scores = population_consistency(Z_pert, mu, sigma)
        mean_d = np.mean(d_scores)
        
        results.append({
            "Noise": noise,
            "Output_Divergence": div,
            "Sensitivity_Ratio": ratio,
            "RLCS_D": mean_d
        })
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "resdec_volatility_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved logs to {csv_path}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(df["Noise"], df["Output_Divergence"], marker='o', label="Output Divergence")
    ax.set_xlabel("Latent Noise Sigma")
    ax.set_ylabel("Output Divergence (L2)")
    ax.set_title("resDEC Volatility vs Latent Noise")
    ax.grid(alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(df["Noise"], df["RLCS_D"], marker='s', color='orange', label="RLCS D-Score")
    ax2.set_ylabel("RLCS ResLik Score")
    
    plot_path = os.path.join(FIGURE_DIR, "figure_component_resdec_volatility.pdf")
    plt.savefig(plot_path)
    print(f"Saved figure to {plot_path}")

if __name__ == "__main__":
    run_test()
