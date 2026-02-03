"""
Final Biological Validation (Phase 9+).

Re-runs the Phase 8-B biological evaluation with the Phase 9 Calibration 
Layer enabled. Demonstrates that calibration restores system utility on 
high-dimensional biological embeddings while maintaining safety.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal
from resed.calibration.calibrator import RlcsCalibrator

# --- Configuration ---
EMBEDDING_PATH = "experiments/benchmarks/bioteque/bioteque_gen_omnipath_embeddings.npz"
OUTPUT_DIR = "docs/figures"

# --- Perturbations (Reproduced from Phase 8-B) ---

def perturb_gaussian_noise(z, sigma):
    rng = np.random.default_rng(42)
    return z + rng.normal(0, sigma, z.shape)

def perturb_shock(z, shock_prob=0.02, factor=5.0):
    rng = np.random.default_rng(42)
    mask = rng.random(z.shape[0]) < shock_prob
    z_out = z.copy()
    z_out[mask] *= factor
    return z_out

def perturb_drift(z, factor=0.5):
    return z + factor

def perturb_dropout(z, drop_prob=0.2):
    rng = np.random.default_rng(42)
    mask = rng.random(z.shape[1]) > drop_prob
    z_out = z.copy()
    z_out[:, ~mask] = 0.0
    return z_out

# --- Evaluation ---

def run_evaluation():
    if not os.path.exists(EMBEDDING_PATH):
        print(f"Error: {EMBEDDING_PATH} not found.")
        return

    # 1. Load Clean Data
    data = np.load(EMBEDDING_PATH)
    z_clean = data['embeddings']
    
    # 2. Setup Reference Stats
    mu = np.mean(z_clean, axis=0)
    sigma = np.mean(np.std(z_clean, axis=0))
    if sigma < 1e-6: sigma = 1.0
    
    # 3. Fit Calibrator using Clean Reference
    # We need a raw diagnostic pass to fit the calibrator
    s_dummy = np.zeros((len(z_clean), 4))
    diag_ref = {}
    rlcs_control(z_clean, s_dummy, diagnostics=diag_ref, mu=mu, sigma=sigma)
    
    calibrator = RlcsCalibrator()
    calibrator.fit(diag_ref)
    print("RLCS Calibrator fitted on biological reference.")

    # 4. Define Scenarios
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
        # Evaluate sample-by-sample to enforce independence (TCS bypass)
        # This reflects the biological 'set' nature of the data
        signals = []
        d_scores_calibrated = []
        
        for i in range(len(z_cond)):
            zi = z_cond[i:i+1]
            si = s_dummy[i:i+1]
            diag = {}
            # Call with calibrator
            sig = rlcs_control(zi, si, diagnostics=diag, calibrator=calibrator, mu=mu, sigma=sigma)[0]
            signals.append(sig)
            
            # Extract calibrated D-score manually for visualization
            # (rlcs_control populates diagnostics with RAW scores)
            raw_d = diag['population_consistency'][0]
            cal_d = calibrator.calibrate('population_consistency', raw_d)
            d_scores_calibrated.append(cal_d)
            
        results[name] = {
            "signals": signals,
            "d_calibrated": np.array(d_scores_calibrated)
        }

    # 5. Figure 1: Calibrated Sensor Response
    fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    names = [c[0] for c in conditions]
    means = [np.mean(results[n]["d_calibrated"]) for n in names]
    stds = [np.std(results[n]["d_calibrated"]) for n in names]
    
    x_pos = np.arange(len(names))
    ax1.bar(x_pos, means, yerr=stds, capsize=5, color='forestgreen', alpha=0.7, label='Calibrated ResLik (Z-Score)')
    ax1.axhline(3.0, color='red', linestyle='--', label='Threshold (3.0)')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel("Calibrated Discrepancy (Z)")
    ax1.set_title("Calibrated Sensor Response (High-Dim Bio Embeddings)")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figure5_bioteque_calibrated_sensor_response.pdf"))
    
    # 6. Figure 2: Calibrated Control Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    signal_types = [RlcsSignal.PROCEED, RlcsSignal.DOWNWEIGHT, RlcsSignal.DEFER, RlcsSignal.ABSTAIN]
    colors = ['forestgreen', 'gold', 'orange', 'crimson']
    bottom = np.zeros(len(names))
    
    for sig_type, color in zip(signal_types, colors):
        counts = [results[n]["signals"].count(sig_type) for n in names]
        ax2.bar(x_pos, counts, bottom=bottom, label=sig_type.value, color=color, alpha=0.8)
        bottom += np.array(counts)
        
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Calibrated Control Distribution (High-Dim Bio Embeddings)")
    ax2.legend(title="Signal", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figure5_bioteque_calibrated_control_distribution.pdf"))
    print("Figures generated in docs/figures/")

if __name__ == "__main__":
    run_evaluation()
