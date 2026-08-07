# Evidence status (software path)

This page lists what has been measured in software. Nothing here is a hardware measurement.

## Proven / measured

| Claim | Evidence |
|-------|----------|
| Binary-phase optical scores = scaled dot-product | Unit tests + `examples/04_validate_model.py` |
| Noise hooks (phase quant, phase noise, jitter, crosstalk) | `atom/noise.py`, tests |
| M#-aware capacity (Fe:LiNbO₃ defaults) | `atom/capacity.py` |
| Hybrid module (optical scores + digital remainder) | `atom/hybrid.py` |
| GGUF / safetensors / PyTorch → optical weight encode | `atom/convert.py` |
| Real Mistral-7B Q4_K_M attention weights converted | Local run: 128 tensors |
| Layer-0 optical vs digital scores on those weights | MSE ~1e-15, top-1 100% (`examples/09`) |
| All-layer score audit | `examples/10_all_layer_audit.py` |
| Phase-bit / noise sweep on real layer | `examples/11_phase_noise_sweep.py` |
| Hybrid **block output** parity (GQA) | `atom/hybrid_block.py`, `examples/12_hybrid_block_parity.py` |
| Structural multi-layer token loop (toy lm head) | `examples/13_minimal_hybrid_generate.py` |

## Not claimed

- Full-quality chat or benchmark scores
- MLP / embed / lm_head loaded from GGUF in the generate demo
- Physical Fe:LiNbO₃ device or FPGA measurements
- Energy / latency of a real optical system

## How to reproduce (local weights only)

```bash
# after convert to ./optical_weights_mistral7b
python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b
python examples/11_phase_noise_sweep.py --weights ./optical_weights_mistral7b --layer 0
python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0
python examples/13_minimal_hybrid_generate.py --weights ./optical_weights_mistral7b
```

Do not commit GGUF files or `optical_weights_*` directories to git.

## Fundable narrative (bounded)

Software verifies that attention scores for a real 7B model’s weight matrices can be computed via the optical interference formulation at float agreement when noise is off, and that a hybrid attention **block** (GQA) matches digital block outputs on those weights. Remaining work is systems integration (full weights in generate, FPGA kernel, materials), not the score identity itself.
