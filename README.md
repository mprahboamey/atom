![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Software stack for transformer attention **scores** via optical interference algebra, with hybrid inference on real checkpoints (GGUF today, Hugging Face safetensors when you have them).

Optical step: score matrix. Digital: softmax, values, MLP, norms, lm_head.

## Status

| Area | Status |
|------|--------|
| Optical scores = digital (binary phase) | Verified |
| Noise + M# capacity (Fe:LiNbO₃ defaults) | Implemented |
| Convert GGUF / safetensors / PyTorch attention weights | Implemented |
| Mistral-7B: all 32 layers score audit | Measured |
| Hybrid block parity (GQA) | Measured |
| Hybrid generate from full GGUF weights | Measured (2 and 8 layers; streaming for 32) |
| Unified GGUF + safetensors model loader | `atom/hybrid_model.py` |
| Physical crystal / FPGA | Not built |

See [`docs/evidence_status.md`](docs/evidence_status.md).

## Install

```bash
pip install -e .
pip install torch gguf safetensors
# optional: transformers  (text prompts)
```

Do not commit multi-GB model files.

## Workflow

```bash
# Attention encode + audits
python examples/08_convert_weights.py --model /path/to/model.gguf --out ./optical_weights_mistral7b
python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b
python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0

# Hybrid generate (same flag works for GGUF or safetensors dir)
python examples/14_hybrid_generate_mistral.py \
  --model /path/to/model.gguf \
  --stream-layers \
  --max-new 4 \
  --compare-digital

# Full depth (32 layers), streaming
python examples/15_full_depth_generate.py \
  --model /path/to/model.gguf \
  --max-new 4 \
  --compare-digital
```

## Layout

```
atom/
  hybrid_model.py   Unified GGUF/safetensors hybrid transformer
  convert.py        Attention weight -> phase encode
  attention.py      Optical scores
  noise.py capacity.py hybrid.py hybrid_block.py
examples/           01–15
docs/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
