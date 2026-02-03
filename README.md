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
*   **Phase 7-A (Benchmarks)**: Benchmark representation extraction (Vision). (Complete)
*   **Phase 7-B (Stress Testing)**: Validation on real-world ResNet-50 embeddings. (Complete)
*   **Phase 8-A (Bio Benchmarks)**: Biological embedding extraction (Bioteque). (Complete)
*   **Phase 8-B (Bio Evaluation)**: RLCS evaluation on biological embeddings. (Complete)
*   **Phase 8-C (Recalibration)**: Reference recalibration evaluation. (Complete)

## Phase 8: Biological Validation

We have validated RLCS on real-world biological embeddings (Bioteque Genes).
*   **Extraction**: [docs/phase8a_biological_embedding_extraction.md](docs/phase8a_biological_embedding_extraction.md)
*   **Evaluation**: [docs/phase8b_rlcs_biological_evaluation.md](docs/phase8b_rlcs_biological_evaluation.md)
*   **Recalibration**: [docs/phase8c_reference_recalibration.md](docs/phase8c_reference_recalibration.md)
*   **Key Result**: The system exhibits structural conservatism on high-dimensional data, defaulting to `ABSTAIN` due to geometric scaling ($\sqrt{d}$) rather than reference misalignment.

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
