# Contributing to ATOM

## Where the project is now

- Optical attention **scores** match digital scaled-dot-product attention (binary phase).
- **Primary evidence:** hybrid generate on real **Hugging Face safetensors** — SmolLM2-135M-Instruct, all **30 layers**, optical greedy **==** digital on fixed and natural-language prompts (`docs/results_smollm2_safetensors.md`).
- Text eval script: `examples/16_safetensors_text_eval.py`.
- Optional integration test: `ATOM_SAFETENSORS_MODEL=... pytest tests/test_hybrid_identity.py`.
- Noise models, M# capacity defaults, phase encoding, and a unified safetensors/GGUF loader are in-tree.

Open work is mostly systems, media physics, and hardware — not whether the score path runs on a real checkpoint.

Do not commit model weight files.

---

## Materials science

**Problem:** Readout erasure, scatter, dynamic range.

**Open:** Thermal fixing / two-color gating under dense multiplexing; measured M#; scatter models.

**Hook:** `atom/capacity.py`.

---

## Integrated photonics / FPGA

**Problem:** Score kernel is software-only.

**Open:** Fixed-point complex MAC; weight layout; SLM / detector / ADC energy.

**Hook:** `atom/attention.py`, `atom/hybrid_model.py`.

---

## Noise and evaluation

**Present:** Phase quantisation, phase noise, angular jitter, soft crosstalk; text-prompt optical vs digital compare.

**Open:** Bragg-shaped crosstalk; detector noise; longer prompts; larger safetensors models; chat-quality metrics (separate from score identity).

**Hook:** `atom/noise.py`, `examples/16_*`.

---

## ML systems

**Present:** Full-depth safetensors hybrid generate; streaming; integration test hook.

**Open:** KV cache; faster multi-head score path; larger models; CI with a tiny public checkpoint artifact if licensing allows.

**Hook:** `atom/hybrid_model.py`.

---

## Theory

**Present:** Score-level interference equivalence.

**Open:** Optical softmax / values; multi-head packing in one volume; training without full optical backprop.

---

## General

Issues for bugs and over-claims. PRs for code and docs. Math errors are highest priority.
