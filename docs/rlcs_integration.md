# RLCS Integration Guide

How to integrate the RLCS layer into the resED pipeline.

## 1. Import Structure

```python
from resed.encoders.resenc import ResENC
from resed.decoders.resdec import ResDEC
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal
```

## 2. Pipeline Execution Flow

The integration strictly follows the **Enc -> Res -> Dec** pattern:

1.  **Encode**:
    ```python
    z, s = encoder.encode(x)
    ```

2.  **Control (RLCS)**:
    ```python
    diagnostics = {}
    signals = rlcs_control(
        z, s, 
        diagnostics=diagnostics,
        mu=ref_mu, sigma=ref_sigma, z_prime=z_alt
    )
    ```

3.  **Decode**:
    ```python
    outputs = []
    for i, signal in enumerate(signals):
        # The decoder handles the signal logic (gating/scaling)
        y_hat = decoder.decode(z[i:i+1], signal) 
        outputs.append(y_hat)
    ```

## 3. Configuration

The RLCS behavior is governed by `resed/rlcs/thresholds.py`. These constants should be tuned based on validation data distributions during a calibration phase (not implemented in 

*   `TAU_D`: Controls outlier rejection rate.
*   `TAU_T`: Controls sensitivity to jitter/drift.
*   `TAU_A`: Controls reliance on multi-view consensus.
