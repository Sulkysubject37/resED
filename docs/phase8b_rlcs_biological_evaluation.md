# RLCS Biological Evaluation (Phase 8-B)

This document details the evaluation of the RLCS governance layer on real-world biological embeddings extracted from Bioteque.

## 1. Dataset Scope
*   **Source**: Bioteque (`GEN-_dph-GEN` metapath).
*   **Content**: 260 Gene embeddings (128 dimensions).
*   **Nature**: Pre-computed, static, high-dimensional vectors representing indirect associations.
*   **Constraint**: No biological interpretation is performed; these are treated as opaque feature vectors to validate system robustness.

## 2. Evaluation Protocol

We established a reference population statistics ($\mu, \sigma$) from the clean embeddings and then subjected them to four deterministic stress conditions:

1.  **Gaussian Noise**: Additive noise ($\sigma \in \{0.1, 0.3, 0.6\}$) to test sensitivity to distribution widening.
2.  **Localized Shock**: Scaling a random subset (2%) of vectors by $5\times$ to simulate outliers.
3.  **Smooth Drift**: Shifting the entire population mean to test drift detection.
4.  **Dimensional Dropout**: Zeroing out 20% of features to simulate sparsity/corruption.

## 3. Observed Behavior

### Sensor Response
**Figure**: `docs/figures/figure_bioteque_sensor_response.pdf`

*   **Clean**: ResLik scores are low and stable, well below the $\tau_D=3.0$ threshold.
*   **Noise**: Monotonic increase in ResLik score as noise $\sigma$ increases. High noise ($\sigma=0.6$) reliably pushes the population mean score above the threshold.
*   **Shock**: Drastically increases the variance (standard deviation) of the ResLik scores, reflecting the presence of extreme outliers.

### Control Distribution
**Figure**: `docs/figures/figure_bioteque_control_distribution.pdf`

*   **Clean**: Dominantly `PROCEED`. The system accepts the valid biological data.
*   **Noise (0.1)**: Mostly `PROCEED`, showing tolerance for minor noise.
*   **Noise (0.6)**: Shift to `ABSTAIN` dominance. The system correctly rejects the corrupted data.
*   **Shock**: A small fraction of samples transition to `ABSTAIN` (the shocked ones), while the rest remain `PROCEED`. This demonstrates **granular, per-sample governance**.
*   **Drift/Dropout**: Triggers `ABSTAIN` or `DEFER` depending on the severity relative to the reference manifold.

## 4. Conclusion
The RLCS layer successfully generalizes to biological data. It establishes a stable "safe" baseline for the clean embeddings and exhibits conservative, monotonic failure modes when those embeddings are perturbed. The system does not require retraining to protect against OOD biological inputs.
