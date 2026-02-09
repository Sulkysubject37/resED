# Interface Contracts

This document stabilizes the module interfaces across the resED pipeline. These contracts represent the immutable boundary conditions for future development.

## 1. Encoder (resENC)

*   **Input**: Tensor $X$ of shape `(batch_size, d_in)`.
*   **Output**: Tuple `(Z, S)`.
    *   $Z$: Latent tensor `(batch_size, d_z)`.
    *   $S$: Statistics tensor `(batch_size, k)` containing `[L2, Var, Entropy, Sparsity]`.
*   **Contract**: The encoder must be deterministic. It must strictly respect the input dimensions. It must provide $S$ for every $Z$.

## 2. Governance (RLCS)

*   **Input**:
    *   Latent $Z$: `(batch_size, d_z)`.
    *   Stats $S$: `(batch_size, k)`.
    *   Context (Optional): `mu`, `sigma`, `z_prime`.
*   **Output**:
    *   Signals: List of `RlcsSignal` (length `batch_size`).
    *   Diagnostics: Dictionary mapping sensor names to score arrays.
*   **Contract**: Every sample in the batch must be assigned exactly one signal. The operation must be stateless and side-effect free.

## 3. Residual Transformer (resTR)

*   **Input**:
    *   Latent $Z$: `(batch_size, d_z)` (or sequence equivalent).
    *   Control scalars: `alpha` (Attention), `beta` (FFN).
*   **Output**: Refined Latent $Z_{ref}$ matching shape of $Z$.
*   **Contract**:
    *   If `alpha=beta=0`, output must strictly equal input ($Z_{ref} \equiv Z$).
    *   Norm inflation must be bounded: $\|Z_{ref}\| \le (1+\epsilon)\|Z\|$.
    *   Must handle variable batch sizes (due to routing splits).

## 4. Decoder (resDEC)

*   **Input**:
    *   Latent $Z$: `(batch_size, d_z)`.
    *   Control Signal: Single `RlcsSignal` applying to the sub-batch.
*   **Output**:
    *   Reconstruction $Y$: `(batch_size, d_out)` OR `None`.
*   **Contract**:
    *   If Signal is `ABSTAIN` or `DEFER`, output **must** be `None`.
    *   If Signal is `DOWNWEIGHT`, output norm must be scaled.
    *   Must be deterministic for a given $Z$ and Signal.
