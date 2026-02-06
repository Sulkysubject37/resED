"""
Figure: D-Score Histograms.

Visualizes separation between Reference, Shifted Valid, and OOD Noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import KMeans

sys.path.append(os.getcwd())

from resed.calibration.calibrator import RlcsCalibrator
from resed.rlcs.control_surface import rlcs_control

DATA_PATH = "experiments/benchmarks/cifar10_resnet50_embeddings.npy"
OUTPUT_PATH = "docs/figures/figure_d_score_histograms.pdf"

def generate_figure():
    if not os.path.exists(DATA_PATH):
        return
        
    data = np.load(DATA_PATH)
    
    # 1. Cluster
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(data)
    c0 = data[labels == 0] # Ref
    c1 = data[labels == 1] # Shifted
    
    # 2. Noise OOD
    rng = np.random.default_rng(42)
    noise = c0 + rng.normal(0, 0.5, c0.shape)
    
    # 3. Calibrate on C0
    ref_mu = np.mean(c0, axis=0)
    ref_std = np.mean(np.std(c0, axis=0))
    if ref_std < 1e-6: ref_std = 1.0
    
    s_dummy = np.zeros((len(c0), 4))
    diag_ref = {}
    rlcs_control(c0, s_dummy, diagnostics=diag_ref, mu=ref_mu, sigma=ref_std)
    calibrator = RlcsCalibrator()
    calibrator.fit(diag_ref)
    
    # 4. Get Scores
    def get_scores(z):
        diag = {}
        rlcs_control(z, np.zeros((len(z),4)), diagnostics=diag, calibrator=calibrator, mu=ref_mu, sigma=ref_std)
        # return calibrated D
        return calibrator.calibrate_batch('population_consistency', diag['population_consistency'])

    s_ref = get_scores(c0)
    s_shift = get_scores(c1)
    s_noise = get_scores(noise)
    
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(s_ref, bins=30, alpha=0.6, label='Reference (In-Dist)', density=True, color='green')
    plt.hist(s_shift, bins=30, alpha=0.6, label='Shifted Valid (Cluster 1)', density=True, color='orange')
    plt.hist(s_noise, bins=30, alpha=0.6, label='OOD Noise', density=True, color='red')
    
    plt.axvline(3.0, color='black', linestyle='--', label='Threshold (3.0)')
    plt.xlabel('Calibrated ResLik Score (Z)')
    plt.ylabel('Density')
    plt.title('Figure 7: Distribution of Risk Scores under Shift and Corruption')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
