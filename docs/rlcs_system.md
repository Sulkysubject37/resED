# RLCS System

The **Representation-Level Control Surface (RLCS)** is the core reliability governance layer of the resED architecture. It acts as an autonomous "gatekeeper" between the Encoder and Decoder.

## 1. System Role

The RLCS does not modify the latent representation $Z$. Instead, it inspects it and emits a **Control Signal** ($\pi$) that dictates how the Decoder should treat the representation.

### Inputs
1.  **Latent Representation ($Z$)**: The vector output from resENC.
2.  **Statistical Summary ($S$)**: The side-channel statistics computed by resENC.
3.  **Reference Statistics** (Optional): Historical population parameters ($\mu, \sigma$).
4.  **Alternate View ($Z'$)** (Optional): A parallel encoding for consistency checking.

### Outputs
*   **Control Signal ($\pi$)**: A discrete signal per sample: `{PROCEED, DOWNWEIGHT, DEFER, ABSTAIN}`.
*   **Diagnostics**: A dictionary of computed scores ($D_i, T_i, A_i$).

## 2. Control Logic

The control decision uses **Conservative OR** logic. This means if *any* sensor raises an alarm, the system escalates the intervention level. The signals are prioritized by severity:

1.  **ABSTAIN** (Highest Severity): The representation is statistically anomalous (OOD).
2.  **DEFER**: The representation is temporally unstable (drifting).
3.  **DOWNWEIGHT**: The representation lacks consensus with an alternate view.
4.  **PROCEED** (Lowest Severity): All checks pass.

## 3. Architecture Independence

The RLCS layer is:
*   **Deterministic**: Same inputs yield same signals.
*   **Stateless**: It evaluates the current batch (and immediate history for temporal checks) without updating internal weights.
*   **Unsupervised**: It relies on statistical distances, not learned classifiers.
