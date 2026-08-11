![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Software stack for transformer attention **scores** via optical interference **algebra**, with hybrid inference on real **Hugging Face safetensors** checkpoints.

- **Optical step (modeled):** attention score matrix  
- **Digital step:** softmax, values, MLP, norms, lm_head  

Binary-phase scores equal scaled dot-product attention **by construction**. Matching on a real model checks hybrid wiring, not a physical crystal.

Read [`docs/audit_limits.md`](docs/audit_limits.md) and [`docs/validation_audit.md`](docs/validation_audit.md) before citing results.

## Primary measured result (software)

**SmolLM2-135M-Instruct** (public safetensors, full **30 layers**):

- Fixed and natural-language prompts: optical greedy **==** digital greedy  
- Full-forward logits: **100%** top-1 agreement on the logged short sequence  

Details: [`docs/evidence_status.md`](docs/evidence_status.md) · [`docs/results_smollm2_safetensors.md`](docs/results_smollm2_safetensors.md)

## Status

| Area | Status |
|------|--------|
| Binary-phase scores = digital (algebra) | Exact; unit-tested |
| Hybrid generate on real safetensors (30 layers) | Measured in software |
| Noise / Bragg-shaped crosstalk hooks | Implemented |
| M# capacity + readout-refresh **stubs** | Implemented (placeholder material numbers) |
| Multi-crystal rack **placement** stub | Implemented |
| FPGA hybrid path | **Scaffold only** (CPU parity + HLS stubs; no board data) |
| Physical crystal write/read | **Not built** |

Geometric capacity ceilings (e.g. “90T” under infinite dynamic range) are **not** usable capacity. See `docs/benchmarks.md` and `atom/capacity.py`.

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

python fpga/host/parity_check.py
pytest tests/test_certainty.py tests/test_refresh.py tests/test_rack.py -q
```

## Layout

```text
atom/attention.py      Score algebra
atom/hybrid_model.py   Safetensors / GGUF hybrid
atom/capacity.py       M#-limited capacity
atom/refresh.py        Readout erase budget stub
atom/rack.py           Multi-crystal placement stub
fpga/                  Digital hybrid FPGA scaffold
docs/validation_audit.md
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
