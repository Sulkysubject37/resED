# Benchmark Validation 

This document details the validation of the resED system using standardized ResNet-50 embeddings extracted from CIFAR-10.

## 1. Clean Reference Baseline

**Figure 0**: `docs/figures/figure0_clean_reference.pdf`

*   **Dataset**: CIFAR-10 Test Subset (N=200).
*   **Encoder**: ResNet-50 (frozen) -> resENC (random projection).
*   **Observation**:
    *   **Population Consistency (ResLik)**: Scores remain consistently below the $\tau_D=3.0$ threshold, indicating the "clean" data is accepted as in-distribution.
    *   **Temporal Consistency**: Since samples are independent images, TCS fluctuates but generally stays stable.
    *   **Control Signals**: Predominantly `PROCEED`, verifying that the system does not reject valid data.

## 2. Stress Response

**Figure 3**: `docs/figures/figure3_rlcs_stress_response.pdf`

We subjected the clean embeddings to three deterministic stress conditions:

1.  **Gradual Drift**:
    *   *Protocol*: Add accumulating shift vector.
    *   *Response*: ResLik score rises monotonically. System transitions `PROCEED` -> `ABSTAIN` as drift exceeds tolerance.

2.  **Sudden Shock**:
    *   *Protocol*: Inject massive noise at index 100.
    *   *Response*: Immediate spike in ResLik score. Instant `ABSTAIN` signal for the shocked sample, returning to `PROCEED` immediately after.

3.  **High Noise**:
    *   *Protocol*: Add Gaussian noise ($\sigma=5.0$).
    *   *Response*: Global elevation of ResLik scores. Constant `ABSTAIN` signal, effectively rejecting the entire corrupted batch.

## 3. Conclusions

*   **Selectivity**: The RLCS layer distinguishes between clean (accepted) and perturbed (rejected) representations without training on the specific failure modes.
*   **Responsiveness**: The system reacts instantaneously to single-point failures (Shock) and trend-based failures (Drift).
*   **Robustness**: The governance logic holds up under high-dimensional embedding stress.

## 4. Scope Limitations

*   **Synthetic Projection**: The `resENC` used here is a random projector. In a deployed system, this would be a trained component, but the *governance mechanics* validated here remain identical.
*   **Independent Samples**: Temporal consistency checks on CIFAR-10 are artificial but serve to validate the logic.
