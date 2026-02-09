# RLCS Mathematics

This document defines the formal mathematical specification for the RLCS sensors used in 

## 1. Population Consistency ($D_i$)

Measures how far a latent vector $z_i$ deviates from the expected population distribution.

$$ D_i = \frac{\|z_i - \mu\|_2}{\sigma + \epsilon} $$

*   $z_i$: Latent vector for sample $i$.
*   $\mu$: Reference mean (centroid) of the training population.
*   $\sigma$: Reference standard deviation (spread) of the population.
*   $\epsilon$: Stability constant ($10^{-8}$). 

**Threshold**: $\tau_D = 3.0$ (approx. 3-sigma rule).
**Signal**: `ABSTAIN` if $D_i > \tau_D$.

## 2. Temporal Consistency ($T_i$)

Measures the stability of the latent trajectory between consecutive time steps. Defined only for sequential data.

$$ T_i = \exp(-\|z_i - z_{i-1}\|_2) $$

*   $z_{i-1}$: Latent vector of the previous step.
*   **Boundary Condition**: $T_0 = 1.0$.

**Threshold**: $\tau_T = 0.5$.
**Signal**: `DEFER` if $T_i < \tau_T$.

## 3. Agreement Consistency ($A_i$)

Measures the consensus between the primary representation $z_i$ and an alternate view $z'_i$ (e.g., from a different encoder or augmented view).

$$ A_i = \frac{z_i \cdot z'_i}{\|z_i\|_2 \|z'_i\|_2 + \epsilon} $$

*   $z'_i$: Alternate latent vector.
*   Note: This is the Cosine Similarity.

**Threshold**: $\tau_A = 0.8$.
**Signal**: `DOWNWEIGHT` if $A_i < \tau_A$.
