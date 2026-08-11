# Contributing to ATOM

## Where the project is now

- Binary-phase attention **scores** equal digital scaled-dot-product attention **by construction** (unit-tested).
- Hybrid generate on real **Hugging Face safetensors** (SmolLM2-135M-Instruct, 30 layers) matches digital greedy on logged prompts — that validates **wiring**, not a crystal.
- Stubs: noise/Bragg hooks, M# capacity, readout refresh, multi-crystal rack placement, FPGA **scaffold** (no board data).

**Before proposing marketing claims or new capacity numbers, read:**
[`docs/validation_audit.md`](docs/validation_audit.md) and [`docs/audit_limits.md`](docs/audit_limits.md).

Do not commit model weight files.

---

## Materials science

**Problem:** Readout erasure, scatter, dynamic range (M#).

**Open:** Measured M# and erase curves for a chosen medium; thermal fixing / two-color under dense multiplexing.

**Hook:** `atom/capacity.py`, `atom/refresh.py` (placeholders only).

---

## Integrated photonics / FPGA

**Problem:** Score path is software or CPU-reference; FPGA tree is scaffold only.

**Open:** Vitis build, board joules/latency, multi-board scaling.

**Hook:** `fpga/`.

---

## Noise and evaluation

**Present:** Phase quantisation, phase noise, angular jitter, soft and Bragg-shaped crosstalk; text-prompt optical vs digital compare.

**Open:** Detector noise; calibrated Bragg combs; longer prompts; larger models.

---

## ML systems

**Present:** Full-depth safetensors hybrid; streaming; rack **placement** stub.

**Open:** Distributed hybrid runtime from `ClusterPlan`; small-weight write scaling; bit-stable KV cache.

---

## Theory

**Present:** Score-level interference equivalence (binary phase).

**Open:** Optical softmax/values; training without full optical backprop — research, not claimed done.

---

## General

Issues for bugs and over-claims. PRs for code and docs. Math and claim errors are highest priority.
