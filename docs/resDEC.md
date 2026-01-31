# resDEC: Reference Decoder

## 1. Purpose
The **resDEC** (Reference Decoder) is the egress component of the resED system. It maps latent representations back to the output space. Crucially, its execution is strictly gated by an external control signal provided by the RLCS layer. It does *not* make reliability decisions itself; it simply obeys the signal.

## 2. Mathematical Definition
The nominal decoding operation is a deterministic linear map followed by an activation:

$$ \hat{Y} = \psi(ZU + c) $$

Where:
*   $Z \in \mathbb{R}^{n \times d_z}$ is the latent input.
*   $U \in \mathbb{R}^{d_z \times d_{out}}$ is the weight matrix.
*   $c \in \mathbb{R}^{d_{out}}$ is the bias vector.
*   $\psi$ is the output activation function (e.g., identity, sigmoid).

## 3. Control Interface
The decoder accepts a control signal $\pi$ from the set:
`{PROCEED, DOWNWEIGHT, DEFER, ABSTAIN}`.

## 4. Output Semantics
The final output $\hat{Y}_\pi$ is determined by the control signal:

$$
\hat{Y}_\pi =
\begin{cases}
\hat{Y} & \pi = \text{PROCEED} \\
\alpha \hat{Y} & \pi = \text{DOWNWEIGHT} \\
\varnothing & \pi \in \{\text{DEFER}, \text{ABSTAIN}\} \\
\end{cases}
$$

*   **PROCEED**: Standard decoding.
*   **DOWNWEIGHT**: Output is scaled by $\alpha \in (0, 1)$ (default $\alpha=0.5$). Used when confidence is marginal.
*   **DEFER / ABSTAIN**: Returns `None`. The system refuses to generate an output.

## 5. Failure Modes
*   **Invalid Signal**: Raises `ValueError` if an unknown control signal is passed.
*   **Shape Mismatch**: Raises `ValueError` if input $Z$ does not match $d_z$.
*   **Null Output**: Downstream systems must handle `None` returns for DEFER/ABSTAIN cases.
