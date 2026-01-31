# Statistical Layer

The Statistical Layer acts as the gatekeeper between the Encoder and the Decoder in the resED architecture.

## Role and Philosophy

In standard encoder-decoder systems, the latent space is often treated as a passive conduit. In resED, mimicking the **resLik** population sensing approach, this space is treated as an active control surface.

The placement of statistical sensing between the encoder and decoder serves a critical verification purpose:

1.  **Decoupling Validity from Fidelity**: The encoder may produce a high-fidelity representation that is nonetheless out-of-distribution (OOD) or statistically anomalous. The statistical layer detects these anomalies before they propagate to the decoder.
2.  **Population consistency**: By maintaining running statistics (mean, variance) of the latent representations, the system can assess the "typicality" of any single inference pass relative to the historical population.

## Connection to resLik

This layer implements the **RLCS** (Representation-Level Control Surfaces) paradigm. It utilizes the same fundamental sensors:

*   **ResLik (Residual Likelihood)**: Measures the alignment of the current representation with the expected manifold.
*   **TCS (Temporal Consistency)**: Monitors stability over time (for sequential data).
*   **Agreement**: Checks consensus if multiple encoding views are available.

## Mechanism

The layer operates on purely statistical principles, avoiding learned "discriminator" logic which can be brittle. It calculates standard metrics such as Euclidean distances, cosine similarities, and Z-scores against a reference background. These metrics are then aggregated to drive the Control Surface's decision to `PROCEED`, `DEFER`, or `ABSTAIN`.
