"""
 RLCS Evaluation with Recalibrated Reference.

Re-evaluates RLCS behavior using explicitly persisted biological reference statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal
from resed.validation.faults import (
    inject_gradual_drift, 
    inject_distribution_shift, 
    inject_single_point_shock, 
    inject_view_disagreement
)

# Configuration
EMBEDDING_PATH = "experiments/benchmarks/bioteque/bioteque_gen_omnipath_embeddings.npz"
REF_STATS_PATH = "experiments/benchmarks/bioteque/bioteque_gen_reference_stats.npz"
OUTPUT_DIR = "docs/figures"


# Actually, I can import from `figure4...`? No, that's a script.
# I will redefine them to ensure identical behavior.

def perturb_gaussian_noise(z, sigma):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, z.shape)
    return z + noise

def perturb_shock(z, shock_prob=0.02, factor=5.0):
    rng = np.random.default_rng(42)
    mask = rng.random(z.shape[0]) < shock_prob
    z_out = z.copy()
    z_out[mask] *= factor
    return z_out

def perturb_drift(z, factor=0.5):
    # Drift: Shift mean
    return z + factor

def perturb_dropout(z, drop_prob=0.2):
    rng = np.random.default_rng(42)
    d_z = z.shape[1]
    mask = rng.random(d_z) > drop_prob
    z_out = z.copy()
    z_out[:, ~mask] = 0.0
    return z_out

def evaluate():
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(REF_STATS_PATH):
        print("Error: Artifacts not found.")
        return

    # 1. Load Data
    data = np.load(EMBEDDING_PATH)
    z_clean = data['embeddings']
    
    # 2. Load Reference
    ref_data = np.load(REF_STATS_PATH)
    mu_bio = ref_data['mu']
    sigma_bio_vec = ref_data['sigma']
    
    
    # Using mean of vector sigma
    sigma_bio = np.mean(sigma_bio_vec)
    if sigma_bio < 1e-6: sigma_bio = 1.0
    
    print(f"Loaded Reference: mu_norm={np.linalg.norm(mu_bio):.2f}, sigma={sigma_bio:.4f}")

    
    conditions = [
        ("Clean", z_clean),
        ("Noise (0.1)", perturb_gaussian_noise(z_clean, 0.1)),
        ("Noise (0.3)", perturb_gaussian_noise(z_clean, 0.3)),
        ("Noise (0.6)", perturb_gaussian_noise(z_clean, 0.6)),
        ("Shock (5x)", perturb_shock(z_clean, factor=5.0)),
        ("Drift (0.5)", perturb_drift(z_clean, factor=0.5)),
        ("Dropout (20%)", perturb_dropout(z_clean, drop_prob=0.2))
    ]
    
    results = {}
    
    for name, z_cond in conditions:
        s_dummy = np.zeros((len(z_cond), 4))
        diag = {}
        signals = rlcs_control(z_cond, s_dummy, diagnostics=diag, mu=mu_bio, sigma=sigma_bio)
        
        d_scores = diag['population_consistency']
        results[name] = {
            "signals": signals,
            "d_mean": np.mean(d_scores),
            "d_std": np.std(d_scores)
        }

    # 4. Figure 1: Sensor Response
    fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    names = [c[0] for c in conditions]
    d_means = [results[n]["d_mean"] for n in names]
    d_stds = [results[n]["d_std"] for n in names]
    
    x_pos = np.arange(len(names))
    ax1.bar(x_pos, d_means, yerr=d_stds, capsize=5, color='steelblue', alpha=0.8, label='ResLik Score')
    ax1.axhline(3.0, color='red', linestyle='--', label='Threshold (3.0)')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel("Population Consistency (D)")
    ax1.set_title("RLCS Sensor Response (Recalibrated Reference)")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    path1 = os.path.join(OUTPUT_DIR, "figure_bioteque_recalibrated_sensor_response.pdf")
    plt.savefig(path1)
    print(f"Saved {path1}")
    
    # 5. Figure 2: Control Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    signal_types = [RlcsSignal.PROCEED, RlcsSignal.DOWNWEIGHT, RlcsSignal.DEFER, RlcsSignal.ABSTAIN]
    colors = ['forestgreen', 'gold', 'orange', 'crimson']
    bottom = np.zeros(len(names))
    
    for sig_type, color in zip(signal_types, colors):
        counts = []
        for n in names:
            sigs = results[n]["signals"]
            count = sigs.count(sig_type)
            counts.append(count)
        ax2.bar(x_pos, counts, bottom=bottom, label=sig_type.value, color=color, alpha=0.8)
        bottom += np.array(counts)
        
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel("Sample Count")
    ax2.set_title("RLCS Control Distribution (Recalibrated)")
    ax2.legend(title="Signal", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    path2 = os.path.join(OUTPUT_DIR, "figure_bioteque_recalibrated_control_distribution.pdf")
    plt.savefig(path2)
    print(f"Saved {path2}")

if __name__ == "__main__":
    evaluate()
