![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Transformer **attention scores** from optical interference — with an exact
match to scaled dot-product attention.

Encode query and key as amplitudes with binary phase (0 / π). Interference
then recovers the real inner product term by term:

```text
Re(q_wave · conj(k_wave))  =  q · k
```

so the score matrix is identical to `QKᵀ / √d`. That is the mechanism claim:
a physical two-level phase encoding is not an approximation to attention
scores; under that encoding it **is** attention scores. The identity is
unit-tested; the hybrid stack runs it on real model weights.

**Architecture:** optical (or optical-model) path computes scores; digital
path handles softmax, values, MLP, norms, and the lm head — a deliberate
hybrid split, not a half-finished pure-optical fantasy.

## Results

**SmolLM2-135M-Instruct** (Hugging Face safetensors, all **30 layers**):

- Optical-score greedy decode **matches** digital-score decode on fixed and
  natural-language prompts
- Full-forward logits: **100%** top-1 agreement on the logged short sequence

That run shows the mechanism plugged into a full transformer, not only a
toy matmul.

Details: [evidence](docs/evidence_status.md) · [result log](docs/results_smollm2_safetensors.md)

## Stack

| Piece | Role |
|-------|------|
| `atom/attention.py` | Interference score algebra |
| `atom/hybrid_model.py` | Safetensors / GGUF hybrid generate |
| `atom/capacity.py` | M#-aware capacity model |
| `atom/refresh.py` | Readout-erase / rewrite planning stub |
| `atom/rack.py` | Multi-crystal placement (datacenter-style scale-out) |
| `fpga/` | Digital hybrid path scaffold for measured energy/latency later |

## Install

```bash
pip install -e .
pip install torch safetensors transformers pytest
```

Do not commit model weight files.

```bash
python examples/16_safetensors_text_eval.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --max-new 4

pytest tests/test_certainty.py -q
```

## Scope

This repo is the **software and systems** foundation: exact score mechanism,
hybrid inference, capacity/refresh/rack planning, FPGA hooks.

A physical photorefractive write/read loop is **future work** — not claimed
as done. Geometric storage ceilings and material defaults are models with
stated assumptions; see [validation audit](docs/validation_audit.md) when
citing capacity or hardware numbers.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md)
