# resED

**resED** is a system-level encoder–decoder architecture designed for robustness,
reliability, and interpretability under distribution shift.

Rather than treating reliability as a learned property, resED introduces a
deterministic **Representation-Level Control Surface (RLCS)** between the encoder
and decoder. This intermediate system monitors latent representations and governs
downstream behavior using explicit signals.

## Core Idea

Encoders and decoders often fail silently under noise, drift, or domain shift.
resED addresses this by inserting a control layer that:

- Observes latent representations
- Measures population consistency, temporal stability, and cross-view agreement
- Produces deterministic control signals:
  - PROCEED
  - DEFER
  - ABSTAIN

The decoder never acts blindly.

## Architecture
Encoder → Latent Representation → RLCS → Decoder
The RLCS layer is model-agnostic and does not require retraining.

## Repository Structure

- `resed/` — Core Python package
- `resed/encoders/` — Encoder implementations
- `resed/decoders/` — Decoder implementations
- `resed/rlcs/` — Sensors and control logic
- `resed/system/` — Enc–Res–Dec wiring
- `experiments/` — Demonstrations and simulations
- `docs/` — Architecture and theory
- `tests/` — Unit and system tests

## Design Principles

- System correctness over optimization
- Deterministic behavior
- No gradient dependence
- Clear separation between sensing and acting

## Status

This repository defines the reference architecture for resED.
Implementations are intentionally minimal and explicit.

