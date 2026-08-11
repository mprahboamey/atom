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
| Unified checkpoint API (GGUF + safetensors) | `atom/hybrid_model.py` |
| Layer streaming for full-depth generate | `stream_layers=True` |

## Checkpoint plug-in

```text
HybridTransformer.from_checkpoint(path)
  path = *.gguf              -> GGUF loader
  path = HF safetensors dir  -> safetensors loader (same internal layout)
```

Inference API does not change when you move from GGUF to safetensors.

## Reproduce

```bash
# audits on converted attention weights
python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b
python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0

# hybrid generate (GGUF today)
python examples/14_hybrid_generate_mistral.py \
  --model /path/to/model.gguf --stream-layers --max-new 4 --compare-digital

# full depth streaming
python examples/15_full_depth_generate.py \
  --model /path/to/model.gguf --max-new 4 --compare-digital

# later: same commands with a safetensors folder as --model
```

Do not commit GGUF, safetensors, or optical_weights_* to git.
