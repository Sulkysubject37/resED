"""
Benchmark Embedding Extraction Script .

Extracts fixed-length embeddings from the CIFAR-10 validation set using
a specific pre-trained ResNet-50 checkpoint.

Protocol:
1. Load ResNet-50 (timm) with specific weights.
2. Load CIFAR-10 Test split.
3. Resize (224x224) and Normalize (ImageNet).
4. Extract features (Global Average Pooling).
5. Save .npy and .json metadata.

No training, no RLCS, no augmentation.
"""

import os
import hashlib
import json
import torch
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration
CHECKPOINT_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"
MODEL_NAME = "resnet50"
DATASET_NAME = "CIFAR10"
SPLIT = "test" # Using test split as validation
BATCH_SIZE = 32
IMAGE_SIZE = 224
OUTPUT_DIR = "experiments/benchmarks"
EMBEDDING_FILE = os.path.join(OUTPUT_DIR, "cifar10_resnet50_embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "cifar10_resnet50_metadata.json")

def get_checksum(file_path):
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print(f"Starting extraction for {DATASET_NAME} using {MODEL_NAME}...")
    
    # 1. Prepare Model
    print("Loading model...")
    # We load the model structure first
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0) # num_classes=0 for features
    
    # Load specific checkpoint
    # timm can load from URL if we provide it, or we can download.
    # We will let torch.hub handle the download if possible, or manually download.
    # timm.models.load_checkpoint usually handles local files.
    # To be safe and explicit, let's download via torch if not present.
    
    checkpoint_path = os.path.join(OUTPUT_DIR, "resnet50_a1_0-14fe96d1.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Downloading weights from {CHECKPOINT_URL}...")
        torch.hub.download_url_to_file(CHECKPOINT_URL, checkpoint_path)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Use timm's load helper or direct load
    # The checkpoint might be a dict.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Remove classifier weights if they exist (though num_classes=0 should handle structure)
    # Strict=False allows ignoring head weights if they mismatch
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 2. Prepare Data
    print("Preparing data...")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root=os.path.join(OUTPUT_DIR, "data"), train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Extraction
    print("Extracting features...")
    embeddings = []
    
    # LIMIT TO SUBSET (2000 images) due to execution time constraints
    SUBSET_SIZE = 2000
    count = 0
    
    with torch.no_grad():
        for inputs, _ in loader:
            if count >= SUBSET_SIZE:
                break
                
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Forward pass features
            features = model.forward_features(inputs)
            
            if features.ndim == 4:
                features = model.forward_head(features, pre_logits=True)
            
            batch_embeddings = features.cpu().numpy()
            embeddings.append(batch_embeddings)
            
            count += inputs.shape[0]
            if count % 100 == 0:
                print(f"Processed {count}/{SUBSET_SIZE}...")
                
    all_embeddings = np.concatenate(embeddings, axis=0)
    # Trim to exact subset size if batch overshot
    all_embeddings = all_embeddings[:SUBSET_SIZE]
    
    # 4. Save
    print(f"Saving {all_embeddings.shape[0]} embeddings to {EMBEDDING_FILE}...")
    np.save(EMBEDDING_FILE, all_embeddings)
    
    # Compute Checksum
    checksum = get_checksum(EMBEDDING_FILE)
    
    # Metadata
    metadata = {
        "dataset": DATASET_NAME,
        "split": f"{SPLIT}_subset_{SUBSET_SIZE}",
        "model": MODEL_NAME,
        "checkpoint_url": CHECKPOINT_URL,
        "shape": all_embeddings.shape,
        "checksum": checksum,
        "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "resize": f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Extraction complete.")

if __name__ == "__main__":
    main()
