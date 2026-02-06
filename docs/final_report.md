# Final Claims Resolution Report

**Date**: 2026-02-05
**Target**: IEEE TNNLS (Resubmission)

## 1. Summary of Changes
This revision systematically addresses the reviewer's critique regarding baseline comparisons, architectural scope, and empirical rigor.

### Key Actions
*   **Baseline Implementation**: Implemented and benchmarked **Mahalanobis Distance** and **Euclidean Distance** against RLCS on CIFAR-10 embeddings.
*   **Empirical Verification**: Ran comparative benchmarks for varying intensities of Noise, Shock, and Drift.
    *   *Differentiation*: At low noise intensity ($\sigma=0.05$), Mahalanobis significantly outperforms RLCS (AUROC 0.98 vs 0.79), confirming that covariance estimation is necessary for subtle OOD detection.
    *   *Parity*: At operational noise levels ($\sigma \ge 0.1$) and for Drift, RLCS achieves parity (AUROC $\approx 1.0$).
    *   *Advantage*: RLCS shows slight superiority in detecting low-magnitude Shock ($1.5\times$), suggesting Euclidean distance is a more direct proxy for magnitude inflation failure modes.
*   **Architectural Scope**: Added a "Architectural Boundary Conditions" section explicitly documenting the "Normalization Blindness" of Transformers (LayerNorm) to magnitude-based shocks.
*   **Formal Novelty**: Clarified that ResLik is a *standardized Euclidean* metric (scalar Mahalanobis), positioning RLCS as a "Safety Circuit Breaker" (catastrophic failure detection) rather than a precision OOD detector.

## 2. Resolved Criticisms
| Reviewer Criticism | Resolution | Evidence |
| :--- | :--- | :--- |
| **Missing Baselines** | Added Mahalanobis & Euclidean benchmarks | `experiments/benchmarks/comparative_results.csv` |
| **Unclear Novelty** | Explicitly defined ResLik as standard metric; novelty is Governance | Methodology Section |
| **Transformer Scope** | Defined "Normalization Blindness" boundary | Limitations Section |
| **Data Realism** | Added "Shock" (Adversarial-like) perturbation | `run_comparative_benchmark.py` |

## 3. Remaining Limitations
*   **Adversarial Completeness**: RLCS remains vulnerable to optimized attacks. This is now explicitly out-of-scope.
*   **Reference Staticity**: The system assumes a static reference population.

## 4. Venue Recommendation
1.  **IEEE TNNLS (Resubmission)**: The manuscript now meets the rigor and baseline requirements. The "System Property" thesis aligns with TNNLS's interest in complex learning systems.
2.  **NeurIPS (Systems Track)**: If TNNLS rejects, the architectural contribution fits the Systems track.
3.  **ICML (Reliable ML Workshop)**: Good venue for the "Governance" concept if main tracks require SOTA metric performance.
