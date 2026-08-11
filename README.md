![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Software stack for transformer attention **scores** via optical interference algebra, with hybrid inference on **real Hugging Face safetensors** checkpoints.

Optical step: attention score matrix. Digital: softmax, values, MLP, norms, lm_head.

## Primary measured result

On **SmolLM2-135M-Instruct** (public safetensors, full **30 layers**):

```text
optical greedy sequence == digital greedy sequence
```

Same token ids when attention scores are computed with the optical path vs standard matmul. Details: [`docs/evidence_status.md`](docs/evidence_status.md), [`docs/results_smollm2_safetensors.md`](docs/results_smollm2_safetensors.md).

## Status

| Area | Status |
|------|--------|
| Optical scores = digital (binary phase) | Verified |
| Hybrid generate on real **safetensors** (SmolLM2, 30 layers) | **Measured — sequences identical** |
| Unified loader (safetensors + GGUF) | `atom/hybrid_model.py` |
| Config-driven head_dim / GQA (e.g. head_dim 64) | Implemented |
| Noise + M# capacity (Fe:LiNbO₃ defaults) | Implemented |
| Physical crystal / FPGA | Not built |

## Install

```bash
pip install -e .
pip install torch safetensors transformers
# optional: gguf  (if you also use llama.cpp files)
```

Do not commit model weight files.

## Workflow (safetensors first)

```bash
# HF folder with config.json + model.safetensors
python examples/14_hybrid_generate_mistral.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --stream-layers \
  --max-new 6 \
  --compare-digital
```

`--compare-digital` checks that optical-score and digital-score greedy outputs match.

## Layout

```
atom/
  hybrid_model.py   Safetensors (and GGUF) hybrid transformer
  attention.py      Optical scores
  convert.py        Attention weights -> phase encode
  noise.py capacity.py hybrid.py hybrid_block.py
examples/
docs/evidence_status.md
docs/results_smollm2_safetensors.md
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
