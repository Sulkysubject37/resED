# Claims and Non-Claims

This document explicitly delineates the novelty, capabilities, and limitations of the resED system. It serves to prevent overinterpretation of the system's performance or purpose.

## 1. Explicit Claims

resED **does** claim the following:

1.  **Architecture-Agnostic Governance**: The RLCS layer can govern any encoder-decoder pair that exposes a latent vector, provided reference statistics are available.
2.  **Deterministic Reliability**: The system's refusal to generate (ABSTAIN) is a deterministic function of the representation's statistical properties, not a stochastic or learned probability.
3.  **Observability of Failure**: Representation-level failures (drift, shift, collapse) are observable via the provided sensors before decoding occurs.
4.  **Conservative Failure Mode**: The system defaults to suppressing output when data is ambiguous, favoring Type II errors (false negatives/refusals) over Type I errors (hallucinations).

## 2. Explicit Non-Claims

resED **does NOT** claim the following:

1.  **Improved Accuracy**: The system does not improve the predictive accuracy of the underlying encoder/decoder on in-distribution data.
2.  **Adversarial Robustness**: The system is not verified against adversarial attacks designed to minimize statistical distance while maximizing semantic error.
3.  **Uncertainty Estimation**: The scores ($D, T, A$) are reliability heuristics, not calibrated probabilities or Bayesian uncertainty estimates.
4.  **Training Method**: resED is an inference-time architecture. It describes how to *run* models, not how to *train* them.
5.  **Semantic Awareness**: The system does not "understand" the content. A statistically typical representation of gibberish will result in `PROCEED`.

## 3. Rationale for Boundaries

### Why not a training method?
Embedding rejection logic into the loss function often leads to optimization collapse or adversarial satisfaction of the metric. resED separates generation (Encoder) from governance (RLCS) to preserve the independence of the reliability check.

### Why not an uncertainty estimator?
Uncertainty estimation often requires ensemble methods (computationally expensive) or Bayesian layers (complex to train). resED prioritizes low-overhead, deterministic checks compatible with standard deterministic models.
