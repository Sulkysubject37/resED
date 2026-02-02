# Failure Modes and Mitigation

This document details the failure modes detectable by the resED system and the corresponding mitigation strategies employed by the RLCS layer.

## 1. Taxonomy of Failures

We distinguish between **Representation Failures** (detectable in Latent Space) and **Semantic Failures** (detectable only in Output Space). resED focuses exclusively on Representation Failures.

| Failure Type | Description | Detectable by resED? |
| :--- | :--- | :--- |
| **Out-of-Distribution (OOD)** | Input is statistically distinct from training population. | **YES** (ResLik) |
| **Temporal Drift** | Latent state changes too rapidly between steps. | **YES** (TCS) |
| **View Disagreement** | Encoder views (e.g., audio vs video) conflict. | **YES** (Agreement) |
| **Encoder Collapse** | All inputs map to a single point. | **PARTIAL** (Variance check) |
| **Semantic Error** | Output is fluent but factually wrong. | **NO** |
| **Adversarial Attack** | Small perturbation flips semantic meaning. | **NO** (if distance small) |

## 2. Detection Mechanisms

### 2.1. Population Deviation (The "Stranger" Problem)
*   **Symptom**: Latent vector $z$ has large Mahalanobis/Euclidean distance from $\mu$.
*   **Sensor**: **ResLik** (Residual Likelihood).
*   **Threshold**: $\tau_D$ (typically 3-sigma).
*   **Response**: `ABSTAIN`. The system refuses to decode.

### 2.2. Temporal Instability (The "Jitter" Problem)
*   **Symptom**: $z_t$ is far from $z_{t-1}$ despite continuity expectation.
*   **Sensor**: **TCS** (Temporal Consistency Score).
*   **Threshold**: $\tau_T$.
*   **Response**: `DEFER`. The system pauses updates or suppresses output to maintain stability.

### 2.3. Consensus Failure (The "Liar" Problem)
*   **Symptom**: $z_{view1}$ is orthogonal to $z_{view2}$.
*   **Sensor**: **Agreement**.
*   **Threshold**: $\tau_A$.
*   **Response**: `DOWNWEIGHT`. The system decodes but flags the output as low-confidence.

## 3. Residual Risks (Known Unknowns)

The system remains vulnerable to:
1.  **In-Distribution Hallucinations**: If the encoder confidently maps a nonsensical input to the center of the latent distribution, RLCS will `PROCEED`.
2.  **Threshold Sensitivity**: Incorrectly calibrated $\tau$ values may lead to high False Rejection Rates (over-conservatism).
3.  **Reference Drift**: If the world changes but reference statistics ($\mu, \sigma$) are not updated, valid new data will be rejected.
