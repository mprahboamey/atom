# Evidence status

Software measurements only. No physical optical hardware is claimed.

## Primary result (safetensors)

**Checkpoint:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)  
**Format:** real Hugging Face `model.safetensors` + `config.json` (not GGUF, not placeholders)

| Property | Value |
|----------|--------|
| Layers | 30 (full model) |
| Hidden size | 576 |
| Attention heads | 9 |
| KV heads | 3 |
| Head dim | 64 |
| Vocab | 49152 |
| Loader | `HybridTransformer.from_checkpoint(..., stream_layers=True)` |

**Hybrid generate (greedy):** optical-score path and digital-score path produced the **same token sequence**.

```text
prompt:  [1, 2, 3, 4]
optical: [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
digital: [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
identical: True
```

That is end-to-end inference on a real published weight file: embed, all 30 layers (attention + MLP + norms), lm_head (tied), RoPE, GQA. Only the attention **scores** use the optical interference formulation; the rest is digital matmul on the checkpoint tensors.

## Supporting results

| Item | Result |
|------|--------|
| Binary-phase optical scores = scaled dot-product | Float agreement (unit tests) |
| Noise hooks + M# capacity (Fe:LiNbO₃ defaults) | Implemented |
| Config-driven head_dim / GQA for HF models | Required for SmolLM2 (head_dim 64) |
| Unified checkpoint API | GGUF or safetensors folder, same generate API |
| Layer streaming | Enables full-depth runs under limited RAM |

## Not claimed

- Physical crystal or FPGA measurements
- Leaderboard / chat quality benchmarks
- That quantized GGUF dumps are the primary evidence base (safetensors is)

## Reproduce

```bash
pip install -e . torch safetensors transformers

# download HF folder (config.json + model.safetensors), then:
python examples/14_hybrid_generate_mistral.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --stream-layers \
  --max-new 6 \
  --compare-digital
```

Do not commit model weight files to git.
