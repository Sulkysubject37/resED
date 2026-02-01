# resED System Validation

This document outlines the validation and stress testing of the resED system under controlled failure scenarios.

## Validation Overview

The resED system is validated by observing how the RLCS governance layer responds to deterministic latent perturbations. We focus on **observability** (detecting faults) and **governance** (changing system behavior to mitigate faults).

## Figure 1: Stress Signal Observability

**Location**: `docs/figures/figure1_stress_observability.png`

### Description
This figure demonstrates the sensitivity of RLCS sensors (ResLik and TCS) to a **Gradual Drift** failure mode.
*   **Fault Parameters**: Drift Rate = 0.2 per step.
*   **Panel 1 (ResLik)**: Tracks the Population Distance ($D$). The red dashed line indicates the safety threshold ($	au_D = 3.0$). The plot is annotated to show exactly where the drift causes the system to enter the **ABSTAIN** (Unsafe) region.
*   **Panel 2 (TCS)**: Tracks the Temporal Consistency Score ($T$). The score drops as the drift accelerates relative to the previous state, crossing the **DEFER** threshold ($	au_T = 0.5$).

### Conclusion
Representation-level failures are reliably observable through statistical distance metrics. The sensors provide clear, monotonic signals that cross predefined safety thresholds before the representation degenerates completely.

## Figure 2: resED OFF vs resED ON

**Location**: `docs/figures/figure2_resed_on_off.png`

### Description
This figure compares the behavior of an ungoverned system (resED OFF) vs. the RLCS-governed system (resED ON) during a **Sudden Distribution Shift**.
*   **Fault Parameters**: Magnitude = 10.0, Injected at Index = 25.
*   **Panel 1 (Output)**:
    *   **OFF (Gray Dashed)**: The decoder continues to process the corrupted latent, resulting in a high-norm output (Hallucination/Noise).
    *   **ON (Green Solid)**: Immediately upon fault injection, the output norm drops to 0. The system successfully suppresses the invalid generation.
*   **Panel 2 (Control Signal)**: Traces the decision logic. At Index 25, the signal strictly transitions from `PROCEED` to `ABSTAIN`.

### Conclusion
RLCS governance successfully intervenes in the system execution path. The comparison proves that the system's robustness is a property of the governance layer, not the decoder's inherent tolerance.

## Explicit Non-Claims

1.  **No Generalization Claims**: This validation uses deterministic simulation; performance on real-world noise may vary based on threshold tuning.
2.  **No Accuracy Gains**: The system does not "fix" the encoder; it refuses to generate when the encoder fails.
3.  **No Learning**: All detections are purely statistical and threshold-based.