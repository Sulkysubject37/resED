# resED System Architecture

This document provides a comprehensive architectural definition of the **resED** (Reliability-First Encoder-Decoder) system.

## 1. High-Level Design

resED is a **governed generative pipeline**. Unlike standard black-box autoencoders, it inserts a deterministic control surface between encoding and decoding.

**Data Flow:**
```
Input (X) 
  |
  v
[resENC] -> (Z, S)
              |
              v
           [RLCS Governance]
              |
              +--> Diagnostics (D, T, A)
              +--> Control Signal (PROCEED, DEFER, ABSTAIN)
              |
              v
[resTR] <----(Gating)
  |
  v
[resDEC] <---(Gating)
  |
  v
Output (Y) or None
```

## 2. Core Components

### 2.1. resENC (Reference Encoder)
*   **Role**: Maps input $X$ to latent $Z$ and computes statistical summary $S$.
*   **Properties**:
    *   **Deterministic**: Fixed weights, no stochasticity.
    *   **Side-Channel**: Emits $S = [\|z\|, \text{Var}(z), \text{Entropy}, \text{Sparsity}]$.
    *   **Bounded**: Output bounded by activation (e.g., tanh).

### 2.2. RLCS (Representation-Level Control Surface)
*   **Role**: The system "brain". Evaluates $Z$ against reference statistics.
*   **Sensors**:
    *   **ResLik**: Population consistency ($D$).
    *   **TCS**: Temporal consistency ($T$).
    *   **Agreement**: Multi-view consensus ($A$).
*   **Logic**: Conservative OR. The most severe violation dictates the signal.

### 2.3. resTR (Residual Transformer)
*   **Role**: Optional refinement of $Z$.
*   **Structure**: $Z_{out} = Z_{in} + \alpha \cdot \text{Attn}(Z_{in}) + \beta \cdot \text{FFN}(\dots)$.
*   **Governance**: $\alpha, \beta$ are modulated by RLCS. If Signal is `ABSTAIN`, $\alpha=\beta=0$ (Identity bypass).

### 2.4. resDEC (Reference Decoder)
*   **Role**: Maps $Z$ to $Y$ under strict control.
*   **Governance**:
    *   **PROCEED**: Normal execution.
    *   **DEFER**: Suppressed output (or partial).
    *   **ABSTAIN**: Hard block (returns `None`).

## 3. System Invariants

1.  **Observability**: No latent representation passes to the decoder without being scored by RLCS.
2.  **Determinism**: The control logic is purely functional; identical inputs yield identical signals.
3.  **Fail-Safe**: The default behavior under high uncertainty is `ABSTAIN` (suppression), not hallucination.
4.  **Shape Preservation**: resTR is guaranteed to preserve latent dimensionality.

## 4. Design Philosophy

resED treats reliability as a **system property**, not a model property. By externalizing the "trust" logic into RLCS, the system avoids the pitfall of models being "confidently wrong" because the checking mechanism is independent of the generating mechanism.
