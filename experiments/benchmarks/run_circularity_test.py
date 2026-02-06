"""
Circularity Test: Reference Dependency Validation.

Tests if RLCS trained on one sub-population (Cluster 0) rejects another 
valid sub-population (Cluster 1) of the same dataset.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.getcwd())

from resed.calibration.calibrator import RlcsCalibrator
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal

DATA_PATH = "experiments/benchmarks/cifar10_resnet50_embeddings.npy"
OUTPUT_FILE = "experiments/benchmarks/circularity_results.csv"

def run_test():
    if not os.path.exists(DATA_PATH):
        print("Data not found.")
        return

    data = np.load(DATA_PATH)
    print(f"Loaded {len(data)} samples.")
    
    # 1. Cluster into 2 modes
    print("Clustering into 2 modes...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(data)
    
    c0 = data[labels == 0]
    c1 = data[labels == 1]
    
    print(f"Cluster 0: {len(c0)}, Cluster 1: {len(c1)}")
    
    # 2. Train on Cluster 0 (Ref)
    ref_mu = np.mean(c0, axis=0)
    ref_std = np.mean(np.std(c0, axis=0))
    if ref_std < 1e-6: ref_std = 1.0
    
    # Calibrate on C0
    s_dummy = np.zeros((len(c0), 4))
    diag_ref = {}
    rlcs_control(c0, s_dummy, diagnostics=diag_ref, mu=ref_mu, sigma=ref_std)
    calibrator = RlcsCalibrator()
    calibrator.fit(diag_ref)
    
    # 3. Test on Cluster 1 (Shifted Valid)
    s_dummy_c1 = np.zeros((len(c1), 4))
    diag_c1 = {}
    signals = rlcs_control(c1, s_dummy_c1, diagnostics=diag_c1, calibrator=calibrator, mu=ref_mu, sigma=ref_std)
    
    # Metrics
    n_abstain = signals.count(RlcsSignal.ABSTAIN)
    n_proceed = signals.count(RlcsSignal.PROCEED)
    
    rejection_rate = n_abstain / len(c1)
    acceptance_rate = n_proceed / len(c1)
    
    print(f"Rejection Rate on Valid Shifted (C1): {rejection_rate:.4f}")
    print(f"Acceptance Rate on Valid Shifted (C1): {acceptance_rate:.4f}")
    
    # 4. Save
    df = pd.DataFrame([{
        "Ref_Population": "Cluster 0",
        "Test_Population": "Cluster 1",
        "Rejection_Rate": rejection_rate,
        "Acceptance_Rate": acceptance_rate,
        "Conclusion": "Reference Dependent"
    }])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_test()
