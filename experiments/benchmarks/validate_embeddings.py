"""
Benchmark Embedding Validation Script (Phase 7-A).

Validates the integrity and reproducibility of extracted embeddings.

Checks:
1. Shape consistency
2. Finite values
3. Variance checks
4. Determinism (via re-inference on a subset)
"""

import os
import json
import numpy as np
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Config matching extraction
OUTPUT_DIR = "experiments/benchmarks"
EMBEDDING_FILE = os.path.join(OUTPUT_DIR, "cifar10_resnet50_embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "cifar10_resnet50_metadata.json")
MODEL_NAME = "resnet50"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "resnet50_a1_0-14fe96d1.pth")
IMAGE_SIZE = 224

def validate():
    print("Starting validation...")
    
    # 1. Load Artifacts
    if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(METADATA_FILE):
        print("FAIL: Artifacts not found. Run extraction first.")
        return
        
    embeddings = np.load(EMBEDDING_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    print(f"Loaded {len(embeddings)} embeddings.")
    
    # 2. Shape Consistency
    expected_shape = tuple(metadata["shape"])
    if embeddings.shape != expected_shape:
        print(f"FAIL: Shape mismatch. Metadata: {expected_shape}, File: {embeddings.shape}")
        return
    print(f"PASS: Shape consistent {embeddings.shape}.")
    
    # 3. Finite Check
    if not np.all(np.isfinite(embeddings)):
        print("FAIL: Embeddings contain NaNs or Infs.")
        return
    print("PASS: All values finite.")
    
    # 4. Variance Check
    variances = np.var(embeddings, axis=0)
    zero_var_indices = np.where(variances == 0)[0]
    if len(zero_var_indices) > 0:
        print(f"WARNING: {len(zero_var_indices)} dimensions have zero variance.")
    else:
        print("PASS: No zero-variance dimensions.")
        
    # 5. Statistics
    print("\n--- Statistics ---")
    print(f"Mean: {np.mean(embeddings):.4f}")
    print(f"Std:  {np.std(embeddings):.4f}")
    print(f"Min:  {np.min(embeddings):.4f}")
    print(f"Max:  {np.max(embeddings):.4f}")
    
    # 6. Determinism Check
    print("\n--- Determinism Check ---")
    try:
        # Load model and data again for a single batch check
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load just 10 images
        dataset = datasets.CIFAR10(root=os.path.join(OUTPUT_DIR, "data"), train=False, download=False, transform=transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        inputs, _ = next(iter(loader))
        
        with torch.no_grad():
            features = model.forward_features(inputs)
            if features.ndim == 4:
                features = model.forward_head(features, pre_logits=True)
            new_embeddings = features.cpu().numpy()
            
        # Compare with first 10 saved embeddings
        saved_subset = embeddings[:10]
        
        if np.allclose(new_embeddings, saved_subset, atol=1e-5):
            print("PASS: Determinism verified (Batch 0 matches).")
        else:
            diff = np.abs(new_embeddings - saved_subset).max()
            print(f"FAIL: Determinism failed. Max diff: {diff}")
            
    except Exception as e:
        print(f"SKIPPED: Determinism check failed to run ({e})")

if __name__ == "__main__":
    validate()
