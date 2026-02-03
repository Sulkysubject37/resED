"""
Component Test 1: resENC Stability (Phase 10-A).

Measures latent representation stability under structured input perturbations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.encoders.resenc import ResENC
from resed.rlcs.sensors import population_consistency
from resed.utils.math import cosine_similarity, l2_norm

# Configuration
OUTPUT_DIR = "experiments/component_tests"
FIGURE_DIR = "docs/figures"
D_IN = 128
D_Z = 64
BATCH_SIZE = 100

def setup_encoder():
    enc = ResENC(D_IN, D_Z)
    rng = np.random.default_rng(42)
    W = rng.uniform(-0.1, 0.1, (D_IN, D_Z))
    b = np.zeros(D_Z)
    enc.set_weights(W, b)
    return enc

def apply_noise(x, sigma):
    rng = np.random.default_rng(42)
    return x + rng.normal(0, sigma, x.shape)

def apply_dropout(x, rate):
    rng = np.random.default_rng(42)
    mask = rng.random(x.shape) > rate
    return x * mask

def apply_spike(x, magnitude=10.0):
    rng = np.random.default_rng(42)
    x_out = x.copy()
    # Spike one random feature per sample
    indices = rng.integers(0, x.shape[1], size=x.shape[0])
    rows = np.arange(x.shape[0])
    x_out[rows, indices] += magnitude
    return x_out

def run_test():
    print("Starting resENC Stability Test...")
    encoder = setup_encoder()
    
    # Reference Data
    rng = np.random.default_rng(42)
    X_clean = rng.normal(0, 1, (BATCH_SIZE, D_IN))
    Z_clean, _ = encoder.encode(X_clean)
    
    # Reference Stats for RLCS
    mu = np.mean(Z_clean, axis=0)
    sigma = np.mean(np.std(Z_clean, axis=0))
    
    results = []
    
    # 1. Gaussian Noise
    sigmas = [0.01, 0.05, 0.1, 0.3]
    for s in sigmas:
        X_pert = apply_noise(X_clean, s)
        Z_pert, _ = encoder.encode(X_pert)
        
        l2 = np.mean([l2_norm(Z_pert[i] - Z_clean[i]) for i in range(BATCH_SIZE)])
        cos = np.mean([cosine_similarity(Z_pert[i], Z_clean[i]) for i in range(BATCH_SIZE)])
        var_ratio = np.var(Z_pert) / np.var(Z_clean)
        
        # RLCS Observability
        d_scores = population_consistency(Z_pert, mu, sigma)
        mean_d = np.mean(d_scores)
        
        results.append({
            "Perturbation": "Noise",
            "Intensity": s,
            "L2_Dist": l2,
            "Cosine_Sim": cos,
            "Var_Inflation": var_ratio,
            "RLCS_D": mean_d
        })
        
    # 2. Dropout
    rates = [0.05, 0.10, 0.20]
    for r in rates:
        X_pert = apply_dropout(X_clean, r)
        Z_pert, _ = encoder.encode(X_pert)
        
        l2 = np.mean([l2_norm(Z_pert[i] - Z_clean[i]) for i in range(BATCH_SIZE)])
        cos = np.mean([cosine_similarity(Z_pert[i], Z_clean[i]) for i in range(BATCH_SIZE)])
        var_ratio = np.var(Z_pert) / np.var(Z_clean)
        
        d_scores = population_consistency(Z_pert, mu, sigma)
        mean_d = np.mean(d_scores)
        
        results.append({
            "Perturbation": "Dropout",
            "Intensity": r,
            "L2_Dist": l2,
            "Cosine_Sim": cos,
            "Var_Inflation": var_ratio,
            "RLCS_D": mean_d
        })
        
    # 3. Spikes
    mags = [5.0, 10.0, 20.0]
    for m in mags:
        X_pert = apply_spike(X_clean, m)
        Z_pert, _ = encoder.encode(X_pert)
        
        l2 = np.mean([l2_norm(Z_pert[i] - Z_clean[i]) for i in range(BATCH_SIZE)])
        cos = np.mean([cosine_similarity(Z_pert[i], Z_clean[i]) for i in range(BATCH_SIZE)])
        var_ratio = np.var(Z_pert) / np.var(Z_clean)
        
        d_scores = population_consistency(Z_pert, mu, sigma)
        mean_d = np.mean(d_scores)
        
        results.append({
            "Perturbation": "Spike",
            "Intensity": m,
            "L2_Dist": l2,
            "Cosine_Sim": cos,
            "Var_Inflation": var_ratio,
            "RLCS_D": mean_d
        })

    # Save Logs
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "resenc_stability_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved logs to {csv_path}")
    
    # Plotting (Optional Figure)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    subset = df[df["Perturbation"] == "Noise"]
    ax.plot(subset["Intensity"], subset["L2_Dist"], marker='o', label="L2 Distortion")
    ax.set_xlabel("Noise Sigma")
    ax.set_ylabel("Latent Distortion (L2)")
    ax.set_title("resENC Stability: Distortion vs Noise")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(subset["Intensity"], subset["RLCS_D"], marker='s', color='orange', label="RLCS D-Score")
    ax2.set_ylabel("RLCS ResLik Score")
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    
    plot_path = os.path.join(FIGURE_DIR, "figure_component_resenc_stability.pdf")
    plt.savefig(plot_path)
    print(f"Saved figure to {plot_path}")

if __name__ == "__main__":
    run_test()
