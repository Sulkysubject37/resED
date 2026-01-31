# resENC: Reference Encoder

## 1. Purpose
The **resENC** (Reference Encoder) is the ingress component of the resED system. Its primary role is to deterministically transform raw input data into a latent representation ($Z$) while simultaneously calculating a statistical side-channel summary ($S$). This summary is consumed by the RLCS layer to validate the representation's quality and safety.

## 2. Mathematical Formulation
The encoder performs a deterministic projection followed by a fixed nonlinearity:

$$ Z = \phi(XW + b) $$

Where:
*   $X \in \mathbb{R}^{n \times d_{in}}$ is the input batch.
*   $W \in \mathbb{R}^{d_{in} \times d_z}$ is the weight matrix.
*   $b \in \mathbb{R}^{d_z}$ is the bias vector.
*   $\phi$ is the element-wise activation function (default: `tanh`).

## 3. Input / Output Contract
*   **Input**: Tensor $X$ of shape `(batch_size, d_in)`.
*   **Output 1 (Latent)**: Tensor $Z$ of shape `(batch_size, d_z)`.
*   **Output 2 (Stats)**: Tensor $S$ of shape `(batch_size, 4)`.

## 4. Statistical Channel Definition
For each sample $z_i$ in the batch, the statistical summary $S_i$ is computed as:

$$
S_i = \begin{bmatrix}
\|z_i\|_2 \\
\text{var}(z_i) \\
\text{entropy\_proxy}(z_i) \\
\text{sparsity}(z_i)
\end{bmatrix}
$$

*   **L2 Norm**: Measure of vector magnitude.
*   **Variance**: Measure of information spread across dimensions.
*   **Entropy Proxy**: Shannon entropy of the softmax-normalized vector.
*   **Sparsity**: $L1(z_i) / \sqrt{d_z}$ (scale-aware proxy).

## 5. Failure Modes
*   **Shape Mismatch**: Raises `ValueError` if input $X$ does not match $d_{in}$.
*   **Dimensionality**: Requires 2D input (batch, features).
*   **NaN Propagation**: Standard floating-point propagation; no internal guards against NaNs in input.
