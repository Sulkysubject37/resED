# RLCS Calibration Layer (Phase 9)

This document describes the **reference-conditioned calibration layer (quantile-estimated, Z-score mapped)** introduced to normalize raw RLCS diagnostics into actionable risk scores.

## 1. Motivation
In Phase 8-B, the system exhibited **conservative collapse** (100% ABSTAIN) on clean biological embeddings. This was diagnosed as a **dimensionality scaling mismatch**: the Euclidean distances in 128-dim space naturally exceeded the scalar threshold TAU_D=3.0 derived from lower-dimensional contexts.

Rather than tuning thresholds (which is brittle), we introduced a **Calibration Layer** that maps raw sensor outputs to a normalized coordinate system based on a reference distribution.

## 2. Methodology

### 2.1. Quantile-to-Normal Mapping
The calibration uses a non-parametric quantile mapping followed by a Z-score transformation:
1.  **Reference**: A clean dataset establishes the empirical cumulative distribution function (CDF) of the sensor scores.
2.  **Mapping**: New scores are mapped to their quantile rank *q* in the range [0, 1].
3.  **Normalization**: The rank is converted to a standard normal Z-score using the inverse normal CDF (probit function).

This transformation ensures that:
*   In-distribution data maps to a standard normal distribution (mean=0, std=1).
*   The threshold TAU_D=3.0 regains its semantic meaning ("3-sigma rarity" relative to the reference).

### 2.2. Architectural Placement
The calibrator sits between the Sensors and the Control Surface:
```
resENC -> Sensors -> [Calibrator] -> Control Surface -> resTR
```
The Control Surface logic remains unchanged; it simply operates on calibrated Z-scores instead of raw distances.

### 2.3. Selective Calibration
*   **Population Consistency (ResLik)**: Calibrated. This sensor is unbounded and scale-sensitive. Calibration is essential for domain transfer.
*   **Temporal/Agreement**: Uncalibrated. These sensors produce bounded metrics ([0, 1]) with absolute semantic thresholds (0.5, 0.8). Calibrating them to Z-scores would break the logic.

## 3. Guarantees

1.  **Monotonicity**: The quantile mapping is strictly monotonic. Higher raw error always equals a higher risk score.
2.  **Determinism**: The calibration depends only on the fixed reference set.
3.  **Domain Transfer**: By normalizing the score distribution, the same control thresholds (TAU=3.0) become applicable across domains with vastly different intrinsic geometries (e.g., Vision vs. Biology).

## 4. Final Biological Validation



We re-evaluated the system on the Bioteque gene embeddings (128-dim) with the calibration layer enabled.



### Calibrated Sensor Response

**Figure**: `docs/figures/figure5_bioteque_calibrated_sensor_response.pdf`

By mapping raw Euclidean distances to Z-scores, the "Clean" data distribution is now centered at $Z \approx 0$. This aligns the biological manifold with the global system threshold $\tau_D=3.0$.



### Calibrated Control Distribution

**Figure**: `docs/figures/figure5_bioteque_calibrated_control_distribution.pdf`

*   **Clean Data**: 99.6% **PROCEED**. The "conservative collapse" observed in Phase 8 is resolved.

*   **Fault Detection**: Despite the shift in baseline, safety is preserved.

    *   **Noise (0.6)**: 100% **ABSTAIN**.

    *   **Shock (5x)**: Granular detection of outliers (ABSTAIN signals for shocked samples).

    *   **Drift/Dropout**: Graded escalation to ABSTAIN as the distribution deviates from the reference.



## 5. Conclusion

The reference-conditioned calibration layer (quantile-estimated, Z-score mapped) successfully decouples the **governance logic** (thresholds) from the **data geometry** (dimensionality). This allows resED to operate as a unified system across diverse embedding domains (Vision, Biology, Synthetic) without per-task threshold tuning.
