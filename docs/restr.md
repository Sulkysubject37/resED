# resTR: Residual Transformer

## Overview

**resTR** (Residual Transformer) is an auxiliary refinement module located between the Encoder and Decoder in the resED architecture. Unlike standard Transformers, it is not a backbone feature extractor but a controlled, residual refinement layer.

## Core Philosophy

1.  **Residual-Only**:
    The module strictly adheres to the residual form:
    $$ Z_{out} = Z_{in} + \text{Refinement}(Z_{in}) $$
    This ensures that in the limit of zero control (\(\alpha=\beta=0\)), the module is an identity function, preserving the original encoding exactly.

2.  **Externally Gated**:
    resTR contains no internal learned gates or decision logic. The magnitude of refinement is controlled explicitly by scalar coefficients (\(\alpha, \beta\)) provided by the RLCS layer (or manual configuration).

3.  **Invariant-Guarded**:
    The module enforces strict mathematical invariants at runtime:
    *   **Shape Preservation**: \(Z_{out}\) must match \(Z_{in}\) dimensions.
    *   **Finite Values**: No NaNs or Infs allowed.
    *   **Norm Inflation Bound**: \(\left\|Z_{out}\right\| \le (1 + \epsilon)\left\|Z_{in}\right\|
). This prevents "runaway" refinement where the Transformer destroys the latent geometry.

## Architecture

The module consists of two optional sub-blocks:

### 1. Self-Attention Block
$$ Z_1 = Z_{in} + \alpha \cdot \text{MHSA}(Z_{in}) $$
*   **MHSA**: Minimal Multi-Head Self-Attention (Single layer).
*   **\(\alpha\)**: External control scalar [0, 1].

### 2. Feedforward Block
$$ Z_{out} = Z_1 + \beta \cdot \text{FFN}(Z_1) $$
*   **FFN**: 2-layer ReLU network (No normalization, no internal residuals).
*   **\(\beta\)**: External control scalar [0, 1].

## RLCS Governance

The RLCS layer governs resTR by dynamically modulating \(\alpha\) and \(\beta\). For example:
*   If **Agreement Consistency** is high, RLCS may permit higher \(\alpha\) to refine the representation.
*   If **Population Consistency** is low (outlier detection), RLCS forces \(\alpha=\beta=0\) to prevent hallucination.

## Non-Goals

*   resTR is **not** a generative backbone.
*   resTR does **not** perform sequence-to-sequence translation.
*   resTR does **not** include "learning safety" or internal reliability estimation.
