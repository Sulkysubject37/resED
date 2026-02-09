"""
Bioteque Embedding Extraction Script .

Extracts precomputed gene embeddings from the Bioteque resource for the 
GEN-_dph-GEN metapath (Omnipath dataset).

Protocol:
1. Download all_datasets_embeddings.tar from Bioteque.
2. Extract tar/gz layers to reach GEN_emb.h5 and GEN_ids.txt.
3. Load HDF5 embeddings into NumPy.
4. Save artifacts as .npz and metadata .json.

No biological interpretation is performed.
"""

import os
import tarfile
import json
import h5py
import numpy as np
import requests

# Configuration
ENTITY_TYPE = "GEN"
METAPATH = "GEN-_dph-GEN"
DATASET = "omnipath"
DOWNLOAD_URL = "https://bioteque.irbbarcelona.org/downloads/embeddings%3EGEN%3EGEN-_dph-GEN/all_datasets_embeddings.tar"

OUTPUT_DIR = "experiments/benchmarks/bioteque"
EMBEDDING_NPZ = os.path.join(OUTPUT_DIR, "bioteque_gen_omnipath_embeddings.npz")
METADATA_JSON = os.path.join(OUTPUT_DIR, "bioteque_gen_omnipath_metadata.json")

def main():
    print(f"Starting Bioteque extraction for {ENTITY_TYPE} ({METAPATH})...")
    
    # 1. Download
    tar_path = os.path.join(OUTPUT_DIR, "all_datasets_embeddings.tar")
    if not os.path.exists(tar_path):
        print(f"Downloading from {DOWNLOAD_URL}...")
        r = requests.get(DOWNLOAD_URL, stream=True)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # 2. Extract
    print("Extracting layers...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=OUTPUT_DIR)
        
    omnipath_dir = os.path.join(OUTPUT_DIR, DATASET)
    gz_path = os.path.join(omnipath_dir, "embeddings.tar.gz")
    
    with tarfile.open(gz_path, "r:gz") as tar:
        tar.extractall(path=omnipath_dir)
        
    # 3. Load Data
    h5_path = os.path.join(omnipath_dir, "GEN_emb.h5")
    ids_path = os.path.join(omnipath_dir, "GEN_ids.txt")
    
    print(f"Loading embeddings from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Bioteque H5 usually has a dataset named 'emb' or similar
        # Based on common bioteque format:
        if 'm' in f:
            embeddings = f['m'][:]
        elif 'emb' in f:
            embeddings = f['emb'][:]
        else:
            # Fallback to first dataset
            key = list(f.keys())[0]
            embeddings = f[key][:]
            
    with open(ids_path, 'r') as f:
        entity_ids = [line.strip() for line in f]
        
    # 4. Save Final Artifacts
    print(f"Saving artifacts to {EMBEDDING_NPZ}...")
    np.savez(EMBEDDING_NPZ, embeddings=embeddings, ids=entity_ids)
    
    metadata = {
        "entity_type": ENTITY_TYPE,
        "metapath": METAPATH,
        "dataset": DATASET,
        "source": "Bioteque",
        "url": DOWNLOAD_URL,
        "num_entities": len(entity_ids),
        "dimension": embeddings.shape[1],
        "shape": list(embeddings.shape)
    }
    
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Extraction complete.")

if __name__ == "__main__":
    main()
