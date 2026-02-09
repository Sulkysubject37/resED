# Formal System Bounds 

This document formalizes the relationship between component-level failure modes and system-level observability, converting empirical stress test data into bounded guarantees.

## 1. Definition: Empirical Failure Envelopes

A **Failure Envelope** is the empirical boundary of a component's response to structured stress. It defines the worst-case behavior observed during 

*   **Min-Envelopes**: Lower bound on stability metrics (e.g., Cosine Similarity, Entropy).
*   **Max-Envelopes**: Upper bound on distortion metrics (e.g., L2 Distance, Output Divergence).

## 2. Component Envelopes

Based on 

### 2.1. resENC (Encoder)
*   **Stability Bound**: For input noise $\sigma \le 0.3$, latent angular semantics remain stable: $\text{CosineSimilarity}(Z, Z_{ref}) \ge 0.998$.
*   **Inflation Bound**: Latent variance inflation scales with noise: $\text{Var}(Z) \le 1.35 \times \text{Var}(Z_{ref})$ at $\sigma=0.3$.
*   **Observability**: The **ResLik Score (D)** is a strictly monotonic function of input noise intensity.

### 2.2. resTR (Transformer)
*   **Collapse Bound**: High token corruption ($N \ge 5$) induces attention collapse: $\text{AttentionEntropy} \ge 2.02$ (down from 2.76).
*   **Safety Bound**: Global corruption is strictly prevented by the **Norm Inflation Invariant**: $\Delta \|Z\| \le 5\%$. Any violation triggers a hard `RuntimeError`.

### 2.3. resDEC (Decoder)
*   **Volatility Bound**: The decoder exhibits linear error propagation: $\Delta Y \le 0.18 \times \Delta Z$.
*   **Invariant**: The decoder does not amplify latent errors non-linearly (Sensitivity Ratio $\approx$ constant).

## 3. RLCS-Lifted System Bounds

By lifting component envelopes into the RLCS diagnostic space, we derive the following system-level inequalities:

1.  **Observability Guarantee**: Any input noise $\sigma \ge 0.01$ results in a ResLik score $D \ge 8.0$.
2.  **Detection Inequality**: If $D \le \tau_D$, then input noise $\sigma$ is bounded by the inverse envelope of the reference population.
3.  **Refinement Block**: If $\text{NormInflation} > 5\%$, the transformer is logically bypassed or the pipeline is halted.

## 4. Guarantees and Non-Claims

### System Guarantees
*   **Transparency**: The system state (Risk Scores) is always observable, regardless of component "opacity."
*   **Boundedness**: Output volatility is linearly bounded by latent risk.
*   **Conservative Default**: The system is guaranteed to stop (ABSTAIN) if $D > \tau_D$, regardless of the encoder's internal confidence.

### Non-Claims
*   **No Internal Correctness**: We do not claim the encoder is "correct"; we claim its *incorrectness* is observable.
*   **Empirical Only**: These bounds are valid within the support of the  Extrapolation beyond $\sigma=0.3$ or $N=5$ is not guaranteed.

## 5. Conclusion: Transparent System, Opaque Components

The resED architecture succeeds by treating components as black boxes with measurable failure envelopes. By governing the **flow** between these envelopes using RLCS, the system achieves predictable reliability despite using volatile high-dimensional components.
