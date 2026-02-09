"""
Reference Recalibration Script .

Computes and persists reference statistics (mu, sigma) from the biological embedding population.
This isolates reference calibration as a variable for RLCS evaluation.
"""

import os
import json
import numpy as np

# Config
INPUT_PATH = "experiments/benchmarks/bioteque/bioteque_gen_omnipath_embeddings.npz"
OUTPUT_PATH = "experiments/benchmarks/bioteque/bioteque_gen_reference_stats.npz"

def recalibrate():
    print("Starting reference recalibration...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input not found at {INPUT_PATH}")
        return

    # 1. Load Embeddings
    data = np.load(INPUT_PATH)
    z = data['embeddings']
    print(f"Loaded {z.shape[0]} embeddings (dim={z.shape[1]}).")
    
    # 2. Compute Statistics
    # Mean vector (d,)
    mu = np.mean(z, axis=0)
    
    # Standard deviation
    # RLCS 
    # D = ||z - mu|| / (sigma + eps)
    # To be consistent with "recalibrated reference", we should compute the
    # sigma that reflects the typical variation.
    # If we use scalar sigma (average std), we assume isotropy.
    # However, 
    # If we want to test "reference recalibration", we should calculate
    # the sigma that creates a "Z-score like" distribution for D?
    # No, 
    # It does NOT say "Change formula".
    #  `(z, mu, sigma)`.

    # I will calculate `std` vector and take mean for scalar compatibility,
    # OR save vector std if supported?
    #  If sigma is vector, it broadcasts.
    # If I save vector sigma, I get Mahalanobis-like scaling (per dimension).
    # This is a stronger calibration.
    # Prompt says "Compute Mean (mu_bio), Standard deviation (sigma_bio)".
    # I will compute vector sigma to allow full calibration.
    sigma = np.std(z, axis=0)
    
    # 3. Persist
    print(f"Saving reference stats to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH, mu=mu, sigma=sigma)
    
    # Log summary
    print(f"Mean norm: {np.linalg.norm(mu):.4f}")
    print(f"Avg Sigma: {np.mean(sigma):.4f}")
    print("Recalibration complete.")

if __name__ == "__main__":
    recalibrate()
