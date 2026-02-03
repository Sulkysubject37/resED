# Biological Embedding Extraction (Phase 8-A)

This document describes the protocol for extracting biological benchmark embeddings from the **Bioteque** resource.

## 1. Motivation
Bioteque provides pre-calculated knowledge graph embeddings across various biological entities. Using these pre-trained embeddings allows the **resED** system to be validated on high-dimensional biological data without the overhead of graph construction or model training.

## 2. Selected Embedding
*   **Entity Type**: Gene (GEN)
*   **Metapath**: `GEN-_dph-GEN` (Indirect gene association through drug/phenotype)
*   **Dataset Source**: Omnipath
*   **Extraction URL**: [Bioteque Downloads](https://bioteque.irbbarcelona.org/downloads/embeddings%3EGEN%3EGEN-_dph-GEN/all_datasets_embeddings.tar)

## 3. Extraction Protocol
1.  **Ingress**: Downloaded `all_datasets_embeddings.tar` directly from the Bioteque resource.
2.  **Processing**: Extracted nested `embeddings.tar.gz` to retrieve `GEN_emb.h5`.
3.  **Refinement**: Loaded HDF5 dataset using `h5py`. preserved the mapping between vectors and Ensembl/Uniprot IDs in `GEN_ids.txt`.
4.  **Serialization**: Saved verified matrix as a compressed NumPy artifact (`.npz`).

## 4. Validation Results
*   **Entity Count**: 260 genes
*   **Dimensionality**: 128 dimensions per gene
*   **Integrity**:
    *   No NaNs or non-finite values.
    *   No zero-variance dimensions (information spread preserved).
    *   Verified bit-exact determinism across repeated loads.
*   **Statistics**:
    *   Mean: ~0.0055
    *   Range: [-1.12, 1.15]

## 5. Explicit Constraints
*   **No Biological Interpretation**: No functional enrichment or biological significance is claimed. The data is treated as a high-dimensional feature matrix for system stress testing.
*   **No KG Rebuilding**: We used static precomputed vectors; no graph traversal was performed.

## 6. Limitations
*   **Batch Size**: The selected subset (260 entities) is relatively small, optimized for rapid pipeline verification in the Phase 8-B RLCS evaluation.
