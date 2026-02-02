# Benchmark Data Extraction (Phase 7-A)

This document describes the protocol used to extract standardized latent embeddings for benchmarking the resED system.

## 1. Dataset

*   **Name**: CIFAR-10
*   **Split**: Test (Validation)
*   **Subset**: First 2,000 images (Indices 0-1999)
*   **Resolution**: Resized to 224x224 (Bicubic interpolation)

## 2. Encoder Model

*   **Architecture**: ResNet-50
*   **Source**: `timm` (PyTorch Image Models)
*   **Weights**: `resnet50_a1_0-14fe96d1.pth` (RSB A1 recipe)
*   **Pretrained**: Yes (ImageNet-1k)
*   **Mode**: `eval()` (No gradients, no training)

## 3. Extraction Protocol

1.  **Preprocessing**:
    *   Resize to 224x224.
    *   ToTensor (Scale [0, 1]).
    *   Normalize: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225].
    *   No augmentation (no flip, no crop).

2.  **Inference**:
    *   Forward pass through backbone.
    *   Global Average Pooling (GAP) applied.
    *   Final dimensionality: **2048**.

3.  **Output**:
    *   `experiments/benchmarks/cifar10_resnet50_embeddings.npy`: Float32 array of shape (2000, 2048).
    *   `experiments/benchmarks/cifar10_resnet50_metadata.json`: Provenance metadata.

## 4. Validation Results

*   **Shape**: (2000, 2048)
*   **Integrity**: No NaNs or Infs.
*   **Determinism**: Verified via re-inference of the first batch.
*   **Statistics**:
    *   Mean: ~0.045
    *   Sparsity: High (ReLU activations), minor zero-variance dimensions noted.

## 5. Limitations

*   **Subset**: Only 20% of the full test set is used to ensure rapid reproducibility in CI/CD environments.
*   **Domain**: CIFAR-10 upscaled to 224x224 may introduce scaling artifacts compared to native high-res data.
