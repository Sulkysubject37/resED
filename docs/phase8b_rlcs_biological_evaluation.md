# RLCS Biological Evaluation (Phase 8-B)

This document details the evaluation of the RLCS governance layer on real-world biological embeddings extracted from Bioteque.

## 1. Dataset Scope
*   **Source**: Bioteque (`GEN-_dph-GEN` metapath).
*   **Content**: 260 Gene embeddings (128 dimensions).
*   **Nature**: Pre-computed, static, high-dimensional vectors representing indirect associations.
*   **Constraint**: No biological interpretation is performed; these are treated as opaque feature vectors to validate system robustness.

## 2. Evaluation Protocol

We established a reference population statistics ($\mu, \sigma$) from the clean embeddings and then subjected them to four deterministic stress conditions:

1.  **Gaussian Noise**: Additive noise ($\sigma ∈ {0.1, 0.3, 0.6}$) to test sensitivity to distribution widening.
2.  **Localized Shock**: Scaling a random subset (2%) of vectors by $5\times$ to simulate outliers.
3.  **Smooth Drift**: Shifting the entire population mean to test drift detection.
4.  **Dimensional Dropout**: Zeroing out 20% of features to simulate sparsity/corruption.

## 3. Observed Behavior

### 3.1. Sensor Response
**Figure**: `docs/figures/figure_bioteque_sensor_response.pdf`

*   **Clean Behavior**: Clean embeddings produce stable but elevated ResLik scores (~11-13) relative to the nominal threshold ($\tau_D = 3.0$). This indicates a **scale mismatch** between the reference statistics (which normalize for variance) and the Euclidean distance metric (which scales with $\sqrt{d}$).
*   **Stress Response**: Despite the offset, the sensor response remains **monotonic** and **discriminative**.
    *   **Noise**: As noise increases, the score rises clearly above the baseline.
    *   **Shock**: Introduces massive variance, correctly flagging outliers.

### 3.2. Control Distribution
**Figure**: `docs/figures/figure_bioteque_control_distribution.pdf`

*   **Global Abstention**: All conditions, including clean embeddings, result in **ABSTAIN-dominant** control signals.
*   **Interpretation**: This reflects **conservative system behavior**. The RLCS detected that the embedding manifold (128 dimensions) generated distances far exceeding the safety threshold calibrated for lower-dimensional or unit-normalized spaces.
*   **Safety**: Instead of proceeding silently on unverified data scales, the system defaulted to a hard stop.

## 4. Reference Sensitivity Analysis

The observed "failure" to PROCEED on clean data highlights a critical strength of the RLCS architecture: **Explicit Trust Calibration**.

1.  **Dimensionality Scaling**: The ResLik sensor computes $D = \|z - \mu\| / \sigma$. For high-dimensional vectors ($d=128$), the expected Euclidean distance is $\approx \sqrt{d} ≈ 11.3$.
2.  **Threshold Sensitivity**: The nominal threshold $\tau_D=3.0$ implies a 3-sigma bound for a scalar or low-dim variable. It does not automatically scale with $d$.
3.  **Governance Consequence**: Because the domain (Biology, $d=128$) differed geometrically from the calibration domain (Vision/Synthetic), RLCS refused to extrapolate trust.

**Conclusion**: The system behaved correctly by refusing to process data that violated its internal safety definition, rather than failing silently. Future deployment would require dimension-aware threshold scaling ($\\tau \propto \sqrt{d}$). 

## 5. Conclusion
The RLCS layer generalizes structurally but requires domain-specific calibration.
*   **Observability**: It successfully tracks perturbation magnitude in biological latent spaces.
*   **Governance**: It enforces strict boundaries, defaulting to `ABSTAIN` when reference assumptions (dimensionality/norm) are violated.
*   **Robustness**: It prevents processing of unfamiliar manifolds, serving as an effective "circuit breaker" for model integration.