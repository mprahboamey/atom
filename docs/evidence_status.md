# Evidence status

Software measurements only. No physical optical hardware is claimed.

## Primary result (safetensors)

**Checkpoint:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

| Check | Result |
|-------|--------|
| Full 30-layer hybrid generate, fixed ids | Optical greedy **==** digital |
| Natural language prompt ("The capital of France is") | Optical greedy **==** digital |
| Decoded text | Same string both paths |
| Full-forward logits top-1 | **100%** agreement |
| Logits MSE | ~1e-3 (float16 accumulation; does not change greedy tokens here) |

Full log: [`results_smollm2_safetensors.md`](results_smollm2_safetensors.md).

## Supporting

| Item | Status |
|------|--------|
| Binary-phase score identity | Verified |
| Noise + M# capacity modules | Implemented |
| Unified safetensors / GGUF loader | `atom/hybrid_model.py` |
| Config-driven head_dim / GQA | Yes (head_dim 64 on SmolLM2) |
| Layer streaming | Yes |
| Text prompt eval script | `examples/16_safetensors_text_eval.py` |
| Optional integration test | `tests/test_hybrid_identity.py` |

## Not claimed

- Physical device results
- Chat quality / factual accuracy of generations
- Bit-identical logits under float16 (greedy match is the bar used here)

## Reproduce

```bash
python examples/16_safetensors_text_eval.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --max-new 4
```
