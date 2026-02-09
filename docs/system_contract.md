# resED System Contract

This document formally defines the **resED** system as a governed representation system. It establishes the scope, input assumptions, output semantics, and the specific failure modes the system is designed to detect.

## 1. System Definition

**resED** is a modular encoder-transformer-decoder architecture governed by a Representation-Level Control Surface (RLCS).

The system is defined by the transformation:
$$ Y = \text{System}(X, \Omega) $$
Where:
*   $X$ is the input data.
*   $\Omega$ is the set of governing parameters (thresholds, reference statistics).
*   $Y$ is the output, which may be a decoded value or a null result (suppression).

Crucially, the system does not guarantee an output for every input. It guarantees that an output is produced **only if** the internal representation satisfies the governance constraints defined by $\Omega$.

## 2. Input Assumptions

The system assumes the following about input $X$:
1.  **Format**: Inputs must conform to the dimensionality expected by the encoder ($d_{in}$).
2.  **Domain**: Inputs are expected to be drawn from a distribution related to the reference population used to calibrate $\Omega$.
3.  **Independence**: Unless processing sequential data (where temporal order matters), samples in a batch are treated as statistically independent.

## 3. Output Semantics

The output $Y$ carries explicit semantic meaning based on the governance decision:

*   **Value $Y \in \mathbb{R}^{d_{out}}$**: The system validated the representation and successfully decoded it. The output is considered "governance-compliant."
*   **Null Output ($\varnothing$)**: The system actively rejected the representation. This is not a software error; it is a valid system state indicating that the input was deemed unsafe or unstable.

## 4. Failure Detection Scope

### detectable Failure Modes
The system explicitly detects and mitigates the following representation-level failures:
1.  **Out-of-Distribution (OOD) Latents**: Representations that deviate significantly from the reference population centroid (high Mahalanobis or Euclidean distance).
2.  **Temporal Instability**: Latent trajectories that exhibit unrealistic jumps or drift between consecutive time steps (when applicable).
3.  **View Disagreement**: Inconsistencies between the primary encoding and an auxiliary view (if provided).

### Non-Detectable Failure Modes
The system does **not** claim to detect:
1.  **Semantic Errors**: An output can be statistically stable but semantically incorrect (e.g., a fluent but factually wrong sentence).
2.  **Adversarial Perturbations**: Small-norm perturbations designed to fool the encoder but remain within statistical bounds.
3.  **Encoder Collapse**: If the encoder maps all inputs to a single valid point, RLCS will see "high stability" and proceed, despite the failure.

## 5. Scope of Applicability

resED is applicable to:
*   Inference-time safety gating.
*   Systems requiring "fail-safe" or "abstain" capabilities.
*   Pipelines where reference statistics ($\mu, \sigma$) are stable and estimable.

resED is **not** applicable to:
*   Online learning or few-shot adaptation (unless $\Omega$ is recalibrated).
*   Scenarios requiring 100% output yield (no-refusal systems).
