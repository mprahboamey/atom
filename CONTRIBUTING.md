# Contributing to ATOM

## Where the project is now

- Optical attention **scores** match digital scaled-dot-product attention (binary phase).
- **Primary end-to-end evidence:** hybrid generate on real **Hugging Face safetensors** — SmolLM2-135M-Instruct, all **30 layers**, optical greedy token sequence identical to digital (`docs/results_smollm2_safetensors.md`).
- Loader reads HF `config.json` for head_dim / GQA (e.g. 9 heads, 3 KV, head_dim 64).
- Noise models, M# capacity defaults (Fe:LiNbO₃), and phase encoding tools are in-tree.
- GGUF remains supported as an alternate input format; **published claims should lead with safetensors results.**

What is still open is mostly systems, media physics, and hardware — not the score identity on a real checkpoint.

Do not commit model weight files or large `optical_weights_*` dumps.

---

## Materials science

**Problem:** Readout erasure, scatter, and dynamic range in photorefractive / polymer media.

**Open:** Thermal fixing and two-photon gating under dense angular multiplexing; measured M# and SNR vs hologram count; scatter models.

**Hook:** `atom/capacity.py` (M#, η_min).

---

## Integrated photonics / FPGA

**Problem:** Score kernel exists in software; FPGA / optical I/O does not.

**Open:** Fixed-point complex MAC for optical scores; host↔device weight layout; peripheral energy (SLM, detectors, ADCs).

**Hook:** `atom/attention.py`, `atom/hybrid_model.py`.

---

## Noise and evaluation

**Present:** Phase quantisation, Gaussian phase noise, angular jitter, soft crosstalk.

**Open:** Bragg-shaped crosstalk from measured selectivity; detector noise; thermal drift; prompt-level eval with tokenizer and reported metrics on safetensors models.

**Hook:** `atom/noise.py`, `examples/14_*`.

---

## ML systems

**Present:** Safetensors hybrid generate (full depth on SmolLM2); unified checkpoint API; layer streaming.

**Open:** KV cache for long generate; larger safetensors models under streaming; integration tests that pin optical vs digital greedy match; multi-module partition.

**Hook:** `atom/hybrid_model.py`.

---

## Theory

**Present:** Score-level interference equivalence.

**Open:** Optical softmax / value paths; multi-head packing in one volume; training without full backprop through the optical model.

---

## General

Issues for bugs and incorrect claims. PRs for code and documentation. If the math is wrong, that is the highest-priority report.
