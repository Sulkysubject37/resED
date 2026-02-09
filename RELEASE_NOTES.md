# Release Notes - v1.3.0-tnnls-submission

## Overview
This release marks the formal submission of the manuscript **"Reliability is a System Property"** to the **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**. The repository has been refined to meet archival standards, with an emphasis on transparency, reproducibility, and professional hygiene.

## Key Changes

### 1. Manuscript Submission
- **Submission Target**: IEEE TNNLS Regular Paper.
- **Anonymization**: The main manuscript has been anonymized for double-blind review.
- **Supporting Documents**: Generated official Title Page (DOCX), Conflict of Interest Statement (PDF), and Cover Letter (PDF).
- **Packaging**: Created `main_manuscript.zip` and `supplementary_material.zip` following IEEE guidelines.

### 2. Codebase Professionalization
- **Editorial Cleanup**: Removed AI-generated narrative slop and redundant internal development markers (Phases 0-12) from all Python source files.
- **Comment Pruning**: Editorial refinement of inline comments to focus on mathematical invariants and safety contracts.
- **Consistency**: Standardized naming conventions across `resed/`, `experiments/`, and `tests/`.

### 3. Documentation & Transparency
- **Documentation Overhaul**: Renamed and restructured all technical reports in `docs/` to follow an academic, descriptive format (e.g., `universality_validation.md`, `biological_evaluation.md`).
- **Supplementary Index**: Added a comprehensive `SUPPLEMENTARY_README.md` to guide reviewers through the experimental and validation suite.
- **Reference Implementation**: Linked to the `resLIK` R package on CRAN as the formal reference for RLCS logic.

### 4. Technical Refinements
- **Limitation Framing**: Explicitly documented "Normalization Blindness" in LayerNorm-based architectures.
- **Numerical Integrity**: Verified that all core experimental results (AUROC scores) remain bit-exact matches to the reported figures in the manuscript.

## Repository Status
- **Branch**: `main`
- **Tag**: `v1.3.0-tnnls-submission`
- **License**: MIT License
- **Copyright**: (c) 2026 MD. Arshad

---
*The resED repository is provided for transparency and reproducibility; it is not required to reproduce the claims of the paper.*
