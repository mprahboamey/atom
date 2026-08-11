![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Software stack for transformer attention **scores** via optical interference algebra, with hybrid inference on real **Hugging Face safetensors** checkpoints.

Optical step: attention score matrix. Digital: softmax, values, MLP, norms, lm_head.

## Primary measured result

**SmolLM2-135M-Instruct** (public safetensors, full **30 layers**):

- Fixed token prompts: optical greedy **==** digital greedy  
- Natural language prompt (`"The capital of France is"`): same token sequence and same decoded text both paths  
- Full-forward logits: **100%** top-1 agreement  

Details: [`docs/evidence_status.md`](docs/evidence_status.md) · [`docs/results_smollm2_safetensors.md`](docs/results_smollm2_safetensors.md)

## Status

| Area | Status |
|------|--------|
| Optical scores = digital (binary phase) | Verified |
| Hybrid generate on real safetensors (30 layers) | Measured |
| Text-prompt optical vs digital eval | `examples/16_safetensors_text_eval.py` |
| Unified loader (safetensors + GGUF) | `atom/hybrid_model.py` |
| Noise + M# capacity | Implemented |
| Physical crystal / FPGA | Not built |

## Install

```bash
pip install -e .
pip install torch safetensors transformers pytest
```

Do not commit model weight files.

## Workflow

```bash
python examples/16_safetensors_text_eval.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --max-new 4

# optional integration test
ATOM_SAFETENSORS_MODEL=/path/to/SmolLM2-135M-Instruct pytest tests/test_hybrid_identity.py -q
```

## Layout

```
atom/hybrid_model.py   Safetensors (and GGUF) hybrid transformer
atom/attention.py      Optical scores
examples/16_...        Text-prompt eval
tests/test_hybrid_identity.py
docs/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
