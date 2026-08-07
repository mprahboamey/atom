# Evidence status

Software measurements only. No physical optical hardware results are claimed.

## Measured

| Item | Result |
|------|--------|
| Binary-phase optical scores = scaled dot-product | Float agreement (unit tests, validate script) |
| Continuous-phase path, noise hooks, M# capacity, hybrid module | Implemented and tested |
| GGUF / safetensors / PyTorch → phase-encoded attention weights | `atom/convert.py` |
| Mistral-7B-Instruct Q4_K_M: 128 attention tensors converted | Local conversion run |
| Optical vs digital **scores**, all 32 layers | MSE ~1e-16, top-1 100% every layer (`examples/10`) |
| Hybrid **attention block** output vs digital (GQA) | MSE ~1e-18 on layer 0 (`examples/12`) |
| Full GGUF hybrid forward (embed + all layers + MLP + lm_head) | `atom/gguf_model.py`, `examples/14_hybrid_generate_mistral.py` |

## Generate path

`examples/14_hybrid_generate_mistral.py` loads **all** required tensors from the same GGUF used for attention conversion. Optical path is used only for attention scores; remaining ops are digital on dequantized checkpoint weights. Optional `--compare-digital` checks whether greedy token sequences match when scores are computed digitally instead.

Token **text** decoding requires a tokenizer (e.g. Hugging Face `Mistral-7B-Instruct-v0.2` tokenizer) applied offline to the printed token ids. The generate script itself prints token ids.

## Not claimed

- Physical Fe:LiNbO₃ or FPGA measurements
- Benchmark leaderboard scores
- Bit-identical long generations under every noise setting
- That Q4 dequantization equals an FP16 safetensors run

## Reproduce (local model file only — do not commit weights)

```bash
python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b
python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0
python examples/14_hybrid_generate_mistral.py \
  --gguf /path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --max-new 8 \
  --compare-digital
```

Use `--max-layers 2` first if RAM is tight.
