# Contributing to ATOM

## Where the project is now

- Optical attention **scores** match digital scaled-dot-product attention (binary phase), including on all 32 layers of a converted Mistral-7B Q4_K_M checkpoint.
- Hybrid **attention blocks** (GQA) match digital block outputs on those weights when noise is off.
- Full **GGUF hybrid generate** loads embed, norms, MLP, and lm_head from the same file; scores use the optical path (`atom/gguf_model.py`).
- Noise models, M# capacity defaults (Fe:LiNbO₃), and conversion from GGUF / safetensors / PyTorch are in-tree.

What is still open is mostly systems, media physics, and hardware — not the score identity itself.

Do not commit model weights, GGUF files, or `optical_weights_*` directories.

---

## Materials science

**Problem:** Readout erasure, scatter, and dynamic range in photorefractive / polymer media.

**Open:** Thermal fixing and two-photon gating under dense angular multiplexing; measured M# and SNR vs hologram count; scatter models.

**Hook:** `atom/capacity.py` (M#, η_min). Calibrated curves can replace defaults.

---

## Integrated photonics / FPGA

**Problem:** The score kernel is defined in software; a digital accelerator (FPGA) or eventual optical I/O stack is not.

**Open:** Fixed-point complex MAC for optical scores; host↔device weight layout (amplitude + phase); peripheral energy (SLM, detectors, ADCs).

**Hook:** `atom/attention.py`, `atom/hybrid_block.py`, weight layout from `atom/convert.py`.

---

## Noise and evaluation

**Present:** Phase quantisation, Gaussian phase noise, angular jitter, soft crosstalk.

**Open:** Measured Bragg selectivity kernels; detector noise; thermal drift; prompt-level eval with a real tokenizer and reported metrics; fix remaining edge cases in bit-width sweeps when continuous-phase quantisation interacts with equal positions.

**Hook:** `atom/noise.py`, `examples/10_*`, `examples/11_*`, `examples/14_*`.

---

## ML systems

**Present:** Conversion; hybrid block; full GGUF hybrid generate.

**Open:** Faster / lower-memory GGUF load; KV cache for long generate; safetensors parity runs; multi-module partition (rack-scale composition); integration tests that assert optical vs digital greedy match on short prompts.

**Hook:** `atom/gguf_model.py`, `atom/convert.py`.

---

## Theory

**Present:** Score-level interference equivalence.

**Open:** Optical softmax / value paths; multi-head packing in one volume; training without full backprop through the optical model.

---

## General

Issues for bugs and incorrect claims. PRs for code and documentation. If the math is wrong, that is the highest-priority report.
