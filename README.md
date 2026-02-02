# resED: Reliability-First Encoder-Decoder

**resED** is a modular encoder-transformer-decoder system derived from the **resLik / RLCS** paradigm. It prioritizes reliability and control over raw generation capability.

## Architecture

The system is composed of four strictly governed layers:

1.  **resENC (Encoder)**: Deterministic projection to latent space with a statistical side-channel.
2.  **RLCS (Governance)**: A statistical control surface that guards the latent representation. It emits signals (`PROCEED`, `DEFER`, `ABSTAIN`) based on Population, Temporal, and Agreement consistency.
3.  **resTR (Transformer)**: A residual-only, externally gated refinement module. It never executes "blindly"; its depth and influence are modulated by RLCS.
4.  **resDEC (Decoder)**: A deterministic decoder that accepts control signals to gate or suppress output.

## Status

*   **Phase 0 (Core)**: Utilities and scaffolding. (Complete)
*   **Phase 1 (Components)**: `resENC` and `resDEC` implemented. (Complete)
*   **Phase 2 (RLCS)**: Sensors and Control Surface logic. (Complete)
*   **Phase 3 (resTR)**: Residual Transformer logic. (Complete)
*   **Phase 4 (Integration)**: Full system wiring and governance. (Complete)
*   **Phase 5 (Validation)**: Stress testing and observability figures. (Complete)
*   **Phase 6 (Formalization)**: System contracts and governance semantics. (Complete)
*   **Phase 7-A (Benchmarks)**: Benchmark representation extraction. (Complete)
*   **Phase 7-B (Stress Testing)**: Validation on real-world ResNet-50 embeddings. (Complete)

## Phase 7-B: Real-World Stress Testing

The system has been validated using standardized ResNet-50 embeddings (CIFAR-10) under controlled perturbation.
*   **Clean Baseline**: `docs/figures/figure0_clean_reference.pdf`
*   **Stress Response**: `docs/figures/figure3_rlcs_stress_response.pdf`
*   **Report**: [docs/phase7b_benchmark_validation.md](docs/phase7b_benchmark_validation.md)

## System Formalization (Phase 6)

The system is now formally specified. Please refer to the following documents for semantic definitions:

*   **[System Contract](docs/phase6_system_contract.md)**: Input/Output scope and detectable failures.
*   **[Governance Semantics](docs/phase6_governance_semantics.md)**: Formal meaning of `PROCEED`, `DEFER`, `ABSTAIN`.
*   **[Interface Contracts](docs/phase6_interface_contracts.md)**: Module boundary definitions.
*   **[Claims and Non-Claims](docs/phase6_claims_and_nonclaims.md)**: Explicit boundaries of system capability.

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