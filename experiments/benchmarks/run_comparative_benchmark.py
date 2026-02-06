"""
Comparative Benchmark: RLCS vs Mahalanobis vs Euclidean.

Evaluates OOD detection performance on CIFAR-10 embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import os

sys.path.append(os.getcwd())

from experiments.baselines.mahalanobis import MahalanobisDetector
from resed.calibration.calibrator import RlcsCalibrator

# Config
DATA_PATH = "experiments/benchmarks/cifar10_resnet50_embeddings.npy"
OUTPUT_FILE = "experiments/benchmarks/comparative_results.csv"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    data = np.load(DATA_PATH)
    # Split: 50% Ref, 25% Valid (In-Dist), 25% Used for OOD generation
    n = len(data)
    ref = data[:n//2]
    valid = data[n//2 : 3*n//4]
    ood_seed = data[3*n//4:]
    return ref, valid, ood_seed

def generate_ood(seed, mode="noise", intensity=1.0):
    rng = np.random.default_rng(42)
    mask = np.ones(len(seed), dtype=bool)
    
    if mode == "noise":
        return seed + rng.normal(0, intensity, seed.shape), mask
    elif mode == "shock":
        shocked = seed.copy()
        m = rng.random(len(seed)) < 0.2
        shocked[m] *= intensity
        return shocked, m
    elif mode == "drift":
        return seed + intensity, mask
    return seed, mask

def run_benchmark():
    print("Loading data...")
    ref, valid, ood_seed = load_data()
    
    print("Fitting models...")
    mah = MahalanobisDetector()
    mah.fit(ref)
    
    ref_mu = np.mean(ref, axis=0)
    ref_std = np.mean(np.std(ref, axis=0))
    
    results = []
    
    # Sweep intensities to find differentiation point
    scenarios = [
        ("noise", 0.05),
        ("noise", 0.1),
        ("noise", 0.5),
        ("shock", 1.5),
        ("shock", 2.0),
        ("shock", 5.0),
        ("drift", 0.5),
        ("drift", 2.0)
    ]
    
    for pert, intensity in scenarios:
        print(f"Evaluating {pert} (intensity={intensity})...")
        ood, ood_mask = generate_ood(ood_seed, mode=pert, intensity=intensity)
        
        X_eval = np.concatenate([valid, ood])
        y_valid = np.zeros(len(valid))
        y_ood = ood_mask.astype(float)
        y_true = np.concatenate([y_valid, y_ood])
        
        # --- Mahalanobis ---
        s_mah = mah.score(X_eval)
        auc_mah = roc_auc_score(y_true, s_mah)
        
        # --- RLCS ---
        s_rlcs_raw = np.linalg.norm(X_eval - ref_mu, axis=1) / ref_std
        auc_rlcs = roc_auc_score(y_true, s_rlcs_raw)
        
        # --- Euclidean ---
        s_euc = np.linalg.norm(X_eval - ref_mu, axis=1)
        auc_euc = roc_auc_score(y_true, s_euc)
        
        results.append({
            "Perturbation": pert,
            "Intensity": intensity,
            "Mahalanobis_AUROC": auc_mah,
            "RLCS_AUROC": auc_rlcs,
            "Euclidean_AUROC": auc_euc
        })
        
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()