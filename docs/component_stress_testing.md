# Component-Level Stress Testing 

This document characterizes the intrinsic failure modes of the resED components (`resENC`, `resTR`, `resDEC`) under controlled stress, validating that the RLCS governance layer is observing the correct signals.

## 1. resENC: Encoder Stability

**Test Protocol**: `experiments/component_tests/test_resenc_stability.py`
*   **Perturbations**: Gaussian Noise ($\sigma ∈ [0.01, 0.3]$), Dropout, Spikes.
*   **Metric**: Latent distortion (L2), Cosine Similarity, RLCS ResLik Score.

**Findings**:
*   **Radial Distortion**: Input noise primarily inflates the *variance* (magnitude) of the latent vector (Var Inflation: 1.0 $\to$ 1.35), while angular semantics (Cosine Sim) remain robust (>0.99).
*   **Observability**: RLCS ResLik score scales monotonically with this variance inflation (8.0 $\to$ 9.3), confirming that **RLCS effectively detects the primary failure mode of the encoder (distribution shift)**.

## 2. resTR: Transformer Sensitivity

**Test Protocol**: `experiments/component_tests/test_restr_attention_sensitivity.py`
*   **Perturbations**: Token corruption (1 to 5 tokens).
*   **Metric**: Attention Entropy, Norm Inflation.

**Findings**:
*   **Attention Collapse**: Under heavy corruption (5 tokens), attention entropy drops (2.76 $\to$ 2.02) and concentration spikes (0.17 $\to$ 0.41). The mechanism effectively "fixates" on the loud noise.
*   **Safety Interlock**: The resulting latent vector violated the **Norm Inflation Invariant** ($>5\%$ growth), triggering a `RuntimeError`.
*   **Conclusion**: The `resTR` component is intrinsically volatile under shock, validating the necessity of the **Norm Check** and **RLCS Gating** (which would ABSTAIN/DEFER before execution).

## 3. resDEC: Decoder Volatility

**Test Protocol**: `experiments/component_tests/test_resdec_volatility.py`
*   **Perturbations**: Latent Noise.
*   **Metric**: Output Divergence, Sensitivity Ratio.

**Findings**:
*   **Linear Propagation**: The decoder exhibits a constant sensitivity ratio ($\Delta Y / \Delta Z ≈ 0.18$). It transmits latent errors linearly to the output.
*   **Governance Implication**: Since the decoder does not suppress errors itself, **RLCS suppression (returning None) is the only line of defense** against corrupt latents. The strong correlation between RLCS scores and Decoder Divergence confirms RLCS is monitoring the correct proxy for output risk.

## 4. Summary

| Component | Failure Mode | RLCS/System Detection |
| :--- | :--- | :--- |
| **resENC** | Variance Inflation (Radial drift) | **ResLik Sensor** (High D-score) |
| **resTR** | Attention Collapse / Amplification | **Invariant Check** (Norm) |
| **resDEC** | Linear Error Propagation | **Governed Suppression** (RLCS ABSTAIN) |

The component tests confirm that no component is safe on its own; stability is an emergent property of the governed system.
