# Universality Validation (Phase 12)

This document details the evaluation of RLCS governance across heterogeneous model architectures to determine if reliability is an architecture-agnostic system property.

## 1. Experimental Design

We evaluated three distinct model families using identical inputs and perturbations:
1.  **resENC (Baseline)**: Deterministic MLP-based encoder.
2.  **VAE (Probabilistic)**: Standard Variational Autoencoder (encoder only, mean output).
3.  **Transformer (Attention)**: Multi-head self-attention with mean pooling and LayerNorm.

**Protocol**:
*   **Calibration**: Each model was calibrated on clean data (N=200).
*   **Stress**: Gaussian Noise, Shock (5% samples $\times$ 10), Drift (+2.0), Dropout.
*   **Metric**: Mean Calibrated ResLik Z-Score and Control Signal Distribution.

## 2. Results

### Monotonicity and Ordering
**Figure**: `docs/figures/figure_universality_sensor_response.pdf`

*   **resENC & VAE**: Show perfect monotonicity. Discrepancy scores rise sharply with Noise (0.1 $\to$ 0.5) and Drift. Shock is detected in the specific samples affected.
*   **Transformer**: Shows monotonicity for Drift ($Z \approx$ 4.0), but **reduced sensitivity** to Noise and Shock.

### Governance Consistency
**Figure**: `docs/figures/figure_universality_governance.pdf`

| Scenario | resENC | VAE | Transformer | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Clean** | 0.5% ABSTAIN | 0.5% ABSTAIN | 0.5% ABSTAIN | **Universal Success** |
| **Noise (0.5)** | 81% ABSTAIN | 71% ABSTAIN | 0.5% ABSTAIN | **Divergence** |
| **Drift (+2.0)** | 100% ABSTAIN | 100% ABSTAIN | 81% ABSTAIN | **Universal Success** |
| **Shock** | 5% ABSTAIN | 5% ABSTAIN | 0.5% ABSTAIN | **Divergence** |

## 3. Interpretation

### The "Normalization Blindness" Phenomenon
The Transformer's reduced sensitivity to Noise and Shock is explained by **Layer Normalization**.
*   RLCS (ResLik) measures Euclidean distance from the mean.
*   Shock/Noise primarily inflate the magnitude of the input vector.
*   `resENC` and `VAE` (Linear/ReLU) propagate this inflation to the latent space ($||z||$ increases).
*   `Transformer` applies `LayerNorm`, which projects the vector back to the unit hypersphere (approximately). The magnitude information is lost.

### Universality Conclusion
RLCS governance is **structurally universal** but **metric-sensitive**.
*   It works across all architectures for **Distributional Shifts** (Drift) that alter the direction of the latent vector.
*   For architectures with internal normalization (Transformers), RLCS cannot detect pure **Magnitude Anomalies** using Euclidean distance alone.

## 5. Inference

The Phase 12 validation reveals a critical boundary condition for representation-level governance:

1.  **Metric-Architecture Alignment**: RLCS relies on the latent space geometry reflecting the input stress.
    *   **Linear/ReLU Models (resENC, VAE)**: Input stress (noise/shock) propagates linearly to latent magnitude ($||z|| \uparrow$). RLCS detects this robustly.
    *   **Normalized Models (Transformer)**: LayerNorm constrains the latent vector to a fixed manifold (hypersphere). Input magnitude stress is normalized away. RLCS detects *directional* shifts (Drift) but misses pure *magnitude* violations.

2.  **Universality is Conditional**: The claim "Reliability is a System Property" holds, but the "System" must include the *Encoder's preservation of error*. If an encoder suppresses the error signal (via normalization) before the governor sees it, the system cannot govern it.

**Recommendation**: For normalized architectures, RLCS should be augmented with a pre-normalization magnitude sensor, or the encoder should expose the normalization statistics as part of the side-channel ($S$).
