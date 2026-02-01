# RLCS-Governed System Architecture

This document describes the Phase 4 integration of the Representation-Level Control Surface (RLCS) into the resED pipeline.

## System Overview

resED is no longer just a sequence of components. It is a governed system where the flow of representations is explicitly controlled by statistical reliability checks.

**Pipeline Flow:**
```
Input -> [resENC] -> (z, s) -> [RLCS Governance] -> Signal -> [resTR] -> [resDEC] -> Output
```

## Governance Logic

The `RlcsGovernance` module acts as the system brain. It does not learn; it judges.

1.  **Observe**: It consumes the latent representation $Z$ and its statistics $S$.
2.  **Diagnose**: It runs the RLCS sensor suite (Population, Temporal, Agreement).
3.  **Signal**: It emits a discrete control signal (`PROCEED`, `DOWNWEIGHT`, `DEFER`, `ABSTAIN`).
4.  **Route**: It configures the downstream components based on the signal.

### Control Routing Table

| RLCS Signal | resTR Behavior | resDEC Behavior |
| :--- | :--- |
| **PROCEED** | **Nominal**: Runs with requested $\alpha, \beta$. | **Normal**: Decodes $Z_{ref}$. |
| **DOWNWEIGHT** | **Nominal**: Runs with requested $\alpha, \beta$. | **Scaled**: Output is downweighted by $\alpha_{dec}$. |
| **DEFER** | **Attenuated**: Runs with scaled parameters ($\alpha \times 0.5$). | **Suppressed**: Returns `None`. |
| **ABSTAIN** | **Bypassed**: Identity function ($\alpha=\beta=0$). | **Blocked**: Returns `None`. |

## Key Components

*   **`ResEdBlock`**: The main system entrypoint. It handles batch splitting to ensure mixed signals in a single batch are routed correctly.
*   **`RlcsGovernance`**: The adapter that wraps `rlcs_control` and implements the routing logic.
*   **`ResTR`**: The residual transformer, now acting as a *conditional* refinement layer.

## Guarantees

*   **Determinism**: The same input and configuration will always yield the same execution path.
*   **Safety**: Out-of-distribution inputs (high population distance) trigger `ABSTAIN`, preventing the decoder from hallucinating on invalid latents.
*   **Completeness**: Every sample is assigned a signal; no sample is processed "blindly."
