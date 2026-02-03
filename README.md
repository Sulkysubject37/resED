# resED: Reliability-First Encoder-Decoder

**resED** is a modular encoder-transformer-decoder system derived from the **resLik / RLCS** paradigm. It prioritizes reliability and control over raw generation capability.

## Architecture

The system is composed of four strictly governed layers:

1.  **resENC (Encoder)**: Deterministic projection to latent space with a statistical side-channel.
2.  **RLCS (Governance)**: A statistical control surface that guards the latent representation. It emits signals (`PROCEED`, `DEFER`, `ABSTAIN`) based on Population, Temporal, and Agreement consistency.
3.  **resTR (Transformer)**: A residual-only, externally gated refinement module. It never executes "blindly"; its depth and influence are modulated by RLCS.
4.  **resDEC (Decoder)**: A deterministic decoder that accepts control signals to gate or suppress output.

## Status

*   **Phases 0–6 (Core)**: Implementation, Integration, and Formal Contracts. (Complete)
*   **Phase 7 (Benchmarks)**: Vision (ResNet-50) Benchmarks. (Complete)
*   **Phase 8 (Bio)**: Biological (Bioteque) Benchmarks. (Complete)
*   **Phase 9 (Calibration)**: Reference-conditioned risk calibration. (Complete)
*   **Phase 10 (Bounds)**: Empirical failure envelopes and formal system bounds. (Complete)

## Phase 10: Component Analysis & System Bounds

We have empirically characterized the intrinsic failure modes of every component and formalized them into system-level bounds.
*   **Failure Envelopes**: Characterized resENC stability, resTR sensitivity, and resDEC volatility.
*   **Formal Bounds**: Derived observability guarantees linking latent stress to RLCS diagnostics.
*   **Key Finding**: System reliability is an emergent property of governance, not component robustness.
*   **Docs**: [docs/phase10a_component_stress_testing.md](docs/phase10a_component_stress_testing.md), [docs/phase10b_formal_bounds.md](docs/phase10b_formal_bounds.md).

## Usage

```python
from resed.system.resed_block import ResEdBlock

# Initialize system
block = ResEdBlock(d_in=128, d_z=64, d_out=10)

# Run inference
# The system automatically runs RLCS checks.
outputs, diagnostics = block.forward(input_data)

# Check diagnostics
print(diagnostics['population_consistency'])
```
