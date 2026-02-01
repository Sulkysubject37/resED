# resED System Validation

This document outlines the validation and stress testing of the resED system under controlled failure scenarios.

## Validation Overview

The resED system is validated by observing how the RLCS governance layer responds to deterministic latent perturbations. We focus on **observability** (detecting faults) and **governance** (changing system behavior to mitigate faults).

## Figure 1: Stress Signal Observability

**Location**: `docs/figures/figure1_stress_observability.png`

### Description
This figure demonstrates the sensitivity of RLCS sensors (ResLik and TCS) to a **gradual latent drift**.

*   **Population Distance (ResLik)**: Tracks the deviation of the latent representation from the reference population. As drift increases, the score rises monotonically.
*   **Temporal Consistency (TCS)**: Monitors the rate of change between steps. Drastic or cumulative shifts lead to a collapse in consistency.

### Conclusion
Representation-level failures are reliably observable through statistical distance metrics before they reach the decoding stage.

## Figure 2: resED OFF vs resED ON

**Location**: `docs/figures/figure2_resed_on_off.png`

### Description
This figure compares the behavior of an ungoverned system (resED OFF) vs. the RLCS-governed system (resED ON) during a **sudden distribution shift** (at batch index 25).

*   **resED OFF**: Continues to decode anomalous representations, producing high-variance or potentially nonsensical outputs (visualized by output norm).
*   **resED ON**: Detects the shift via the ResLik sensor, escalates the control signal from `PROCEED` to `ABSTAIN`, and successfully suppresses the decoder output (norm goes to 0).

### Conclusion
RLCS governance successfully intervenes in the system execution path, preventing generation from OOD latents without requiring learned error-detection parameters.

## Explicit Non-Claims

1.  **No Generalization Claims**: This validation uses deterministic simulation; performance on real-world noise may vary based on threshold tuning.
2.  **No Accuracy Gains**: The system does not "fix" the encoder; it refuses to generate when the encoder fails.
3.  **No Learning**: All detections are purely statistical and threshold-based.
