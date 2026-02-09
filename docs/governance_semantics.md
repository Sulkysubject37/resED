# Governance Semantics

This document formally defines the semantics of the control signals emitted by the Representation-Level Control Surface (RLCS) within the resED system.

## 1. Governance Signals

The RLCS emits a discrete signal $\pi \in \{\text{PROCEED}, \text{DOWNWEIGHT}, \text{DEFER}, \text{ABSTAIN}\}$ for each input sample.

### PROCEED
*   **Definition**: The representation satisfies all statistical checks.
*   **Operational Consequence**: The pipeline executes nominally. `resTR` refines the representation with full strength ($\alpha, \beta$), and `resDEC` decodes without suppression.
*   **Semantic Meaning**: "Valid for generation."

### DOWNWEIGHT
*   **Definition**: The representation is valid but exhibits weak consensus (e.g., disagreement between views).
*   **Operational Consequence**: `resTR` executes normally. `resDEC` scales the output amplitude by a factor $\alpha < 1$.
*   **Semantic Meaning**: "Valid but low confidence."

### DEFER
*   **Definition**: The representation exhibits temporal instability or drift.
*   **Operational Consequence**: `resTR` executes in an attenuated state (parameters scaled by factor $< 1$) to minimize state updates. `resDEC` suppresses the output (returns `None`).
*   **Semantic Meaning**: "Unstable; do not act, but maintain state."

### ABSTAIN
*   **Definition**: The representation is statistically anomalous (out-of-distribution).
*   **Operational Consequence**: `resTR` is bypassed ($\\alpha=\\beta=0$). `resDEC` is blocked (returns `None`).
*   **Semantic Meaning**: "Invalid; reject immediately."

## 2. Signal Resolution Logic

The system resolves conflicting sensor inputs using a **Conservative OR** hierarchy. The most severe signal detected dominates the decision.

**Hierarchy (Highest to Lowest Severity):**
1.  **ABSTAIN** (Population Violation)
2.  **DEFER** (Temporal Violation)
3.  **DOWNWEIGHT** (Agreement Violation)
4.  **PROCEED** (No Violation)

**Example**: If a sample triggers both `DOWNWEIGHT` (agreement issue) and `ABSTAIN` (population issue), the final signal is **ABSTAIN**.

## 3. Granularity

*   **Per-Sample**: Governance decisions are made independently for each sample in a batch.
*   **Mixed Execution**: A single batch may contain samples that `PROCEED` alongside samples that `ABSTAIN`. The `ResEdBlock` handles this by routing execution paths dynamically.

## 4. Limitations of Governance

*   **No Correction**: Governance does not "fix" the representation. It only gates the downstream consequences.
*   **No Learning**: Governance logic is static during inference. It does not update thresholds or weights based on outcomes.
*   **No Feedback**: There is no backpropagation from the decision logic to the encoder.
