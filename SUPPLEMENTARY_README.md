# Supplementary Material Index - resED

This archive contains the data, code, and documentation required to reproduce the claims of the paper "Reliability is a System Property".

## Directory Structure

- `experiments/`: Full experimental suite.
  - `benchmarks/`: CIFAR-10 (Vision) and Bioteque (Biology) evaluation scripts.
  - `component_tests/`: Stress testing for resENC, resTR, and resDEC.
  - `synthetic/`: Minimal demonstrations of the governance logic.
  - `baselines/`: Mahalanobis distance baseline implementation.
  - `universality/`: Cross-architecture validation scripts.
- `docs/`: Technical documentation and phase-specific reports.
  - `architecture.md`: Detailed system topology.
  - `rlcs_math.md`: Formal mathematical definitions of sensors.
  - `failure_modes.md`: Catalog of observed architectural blind spots.
- `tests/`: Unit tests for core Python modules.
- `resed/`: Reference Python implementation of the resED framework.
- `environment.yml` / `pyproject.toml`: Dependency and environment configuration.
- `LICENSE`: MIT License.
- `CITATION.cff`: Citation metadata.

## Reproduction Guide

1. Create the environment: `conda env create -f environment.yml`
2. Run the vision benchmark: `python experiments/benchmarks/run_comparative_benchmark.py`
3. Run the biological benchmark: `python experiments/benchmarks/bioteque/evaluate_calibrated_rlcs_bioteque.py`

For the R reference implementation of RLCS sensors, please refer to the `resLIK` package on CRAN.
