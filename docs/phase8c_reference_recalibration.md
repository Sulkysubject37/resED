# Reference Recalibration Evaluation (Phase 8-C)

This document details the evaluation of the RLCS governance layer using explicitly recalibrated reference statistics from the biological embedding population.

## 1. Motivation
In Phase 8-B, the system exhibited conservative behavior (100% `ABSTAIN`) even on clean biological embeddings. This phase investigates whether this was due to using incorrect (synthetic) reference statistics or if it reflects a deeper structural property of the system.

## 2. Methodology
*   **Variable Changed**: Reference statistics ($\mu, \sigma$) were explicitly computed from the Bioteque `GEN-_dph-GEN` population and persisted to `bioteque_gen_reference_stats.npz`.
*   **Variables Constant**: All thresholds ($\tau_D=3.0$), sensor mathematics (Euclidean distance), and perturbation protocols remained identical to Phase 8-B.

## 3. Results

### Comparison: Phase 8-B vs Phase 8-C

| Metric | Phase 8-B (Previous) | Phase 8-C (Recalibrated) | Difference |
| :--- | :--- | :--- | :--- |
| **Reference $\mu$** | Computed in-memory | Loaded from artifact | None (Numerical Identity) |
| **Reference $\sigma$** | Computed in-memory | Loaded from artifact | None (Numerical Identity) |
| **Clean ResLik Score** | ~11-13 | ~11-13 | **Identical** |
| **Clean Signal** | `ABSTAIN` | `ABSTAIN` | **Identical** |

### Interpretation
The results are identical because Phase 8-B effectively performed the same calibration in-memory. The persistence of these statistics in Phase 8-C confirms that the high ResLik scores are **not** an artifact of using default/synthetic references (e.g., $\mu=0, \sigma=1$).

Instead, the behavior is structural:
*   The Euclidean distance in 128-dimensional space scales with $\sqrt{d} \approx 11.3$.
*   The fixed threshold $\tau_D=3.0$ is calibrated for low-dimensional or unit-normalized distances.
*   Even with correct $\mu$ and $\sigma$, the expected distance for a valid sample ($\\approx 11.3$) exceeds the threshold ($3.0$).

## 4. Conclusion
Reference recalibration alone is insufficient to adapt RLCS to high-dimensional biological manifolds. The system correctly identifies that the data geometry violates the assumptions of the fixed scalar threshold.

**Next Steps**: This finding strongly motivates **Phase 9 (Formal Calibration Layer)**, which must introduce dimension-aware thresholding (e.g., $\\tau \\propto \\sqrt{d}$) to normalize trust across domains.
