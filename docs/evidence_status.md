# Evidence status

Software measurements only.

## Measured

| Item | Result |
|------|--------|
| Binary-phase optical scores = scaled dot-product | Float agreement |
| Mistral-7B Q4_K_M: 128 attention tensors converted | Local run |
| Optical vs digital scores, all 32 layers | MSE ~1e-16, top-1 100% |
| Hybrid attention block (GQA) vs digital | MSE ~1e-18 |
| Hybrid generate 2-layer and 8-layer from GGUF | Optical greedy == digital greedy |
| **SmolLM2-135M Instruct safetensors, full 30 layers** | Optical greedy == digital greedy |
| Unified checkpoint API (GGUF + safetensors) | `atom/hybrid_model.py` |
| Layer streaming for full-depth generate | `stream_layers=True` |
| Config-driven head_dim / GQA for HF models | e.g. head_dim=64, 9 heads, 3 kv |

## Checkpoint plug-in

```text
HybridTransformer.from_checkpoint(path)
  path = *.gguf              -> GGUF loader
  path = HF safetensors dir  -> safetensors loader (same internal layout)
```

Inference API does not change when you move from GGUF to safetensors.

## Reproduce

```bash
# GGUF path
python examples/15_full_depth_generate.py --model /path/to/model.gguf --max-new 4 --compare-digital

# Safetensors path (HF folder with config.json + model.safetensors)
python examples/14_hybrid_generate_mistral.py \
  --model /path/to/hf-folder --stream-layers --max-new 6 --compare-digital
```

Do not commit model weights to git.
