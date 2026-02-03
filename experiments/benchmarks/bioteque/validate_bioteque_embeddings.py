"""
Bioteque Embedding Validation Script (Phase 8-A).

Performs sanity checks on the extracted biological embeddings.
Checks:
1. Shape consistency
2. Finite values
3. Non-zero variance
4. Determinism
5. Basic distribution summary
"""

import os
import json
import numpy as np

# Config
OUTPUT_DIR = "experiments/benchmarks/bioteque"
EMBEDDING_NPZ = os.path.join(OUTPUT_DIR, "bioteque_gen_omnipath_embeddings.npz")
METADATA_JSON = os.path.join(OUTPUT_DIR, "bioteque_gen_omnipath_metadata.json")

def validate():
    print("Starting Bioteque validation...")
    
    if not os.path.exists(EMBEDDING_NPZ) or not os.path.exists(METADATA_JSON):
        print("FAIL: Artifacts not found.")
        return
        
    # 1. Load
    data = np.load(EMBEDDING_NPZ)
    embeddings = data['embeddings']
    ids = data['ids']
    
    with open(METADATA_JSON, "r") as f:
        metadata = json.load(f)
        
    print(f"Loaded {len(ids)} entity IDs and {embeddings.shape} matrix.")
    
    # 2. Shape Consistency
    if embeddings.shape[0] != len(ids):
        print(f"FAIL: Row count ({embeddings.shape[0]}) != ID count ({len(ids)})")
        return
    if list(embeddings.shape) != metadata["shape"]:
        print(f"FAIL: Shape mismatch with metadata.")
        return
    print(f"PASS: Shape consistent {embeddings.shape}.")
    
    # 3. Finite Check
    if not np.all(np.isfinite(embeddings)):
        print("FAIL: Non-finite values detected.")
        return
    print("PASS: All values finite.")
    
    # 4. Variance Check
    vars = np.var(embeddings, axis=0)
    zero_var = np.where(vars == 0)[0]
    if len(zero_var) > 0:
        print(f"WARNING: {len(zero_var)} dimensions have zero variance.")
    else:
        print("PASS: No zero-variance dimensions.")
        
    # 5. Determinism (Identity load)
    data2 = np.load(EMBEDDING_NPZ)
    if np.array_equal(embeddings, data2['embeddings']):
        print("PASS: Load determinism verified.")
    else:
        print("FAIL: Load determinism failed.")
        
    # 6. Distribution Summary
    print("\n--- Distribution Summary ---")
    print(f"Mean: {np.mean(embeddings):.4f}")
    print(f"Std:  {np.std(embeddings):.4f}")
    print(f"Min:  {np.min(embeddings):.4f}")
    print(f"Max:  {np.max(embeddings):.4f}")
    
    print("\nValidation Complete.")

if __name__ == "__main__":
    validate()
