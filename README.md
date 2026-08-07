![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

Software research stack for transformer **attention scores** computed via the same algebra as optical wave interference (phase-encoded Q/K), with a hybrid path for real checkpoints.

Core result: with binary phase (0 or π), interference scores are algebraically identical to scaled dot-product attention, verified to float precision. Continuous-phase encoding, noise models, M#-aware capacity (Fe:LiNbO₃ defaults), weight conversion (safetensors / PyTorch / GGUF), and hybrid inference on full GGUF weights are implemented in this repo.

The optical step is the **score matrix**. Softmax, values, MLP, norms, and the lm_head stay digital. Hardware (photorefractive crystal, FPGA) is future work — not claimed as built.

---

## Status

| Area | Status |
|------|--------|
| Optical scores = digital scores (binary phase) | Verified |
| Noise model (phase quant, phase noise, jitter, crosstalk) | Implemented |
| M# capacity model (Fe:LiNbO₃ defaults) | Implemented |
| Convert local HF / GGUF attention weights → phase encode | Implemented |
| Mistral-7B Q4_K_M: all 32 layers score audit | Measured (MSE ~1e-16, top-1 100%) |
| Hybrid attention block vs digital (GQA) | Measured |
| Full GGUF hybrid generate (embed + layers + MLP + lm_head) | Implemented (`atom/gguf_model.py`) |
| Physical crystal / FPGA | Not built |
| Leaderboard task scores | Not claimed |

Details: [`docs/evidence_status.md`](docs/evidence_status.md), [`docs/model.md`](docs/model.md), [`docs/benchmarks.md`](docs/benchmarks.md).

---

## Install

```bash
git clone https://github.com/mprahboamey/atom.git
cd atom
pip install -e .
pip install torch gguf   # gguf needed for GGUF checkpoints
```

Do **not** commit multi-GB model files or `optical_weights_*` directories into git.

---

## Typical workflow (local Mistral GGUF)

```bash
# 1) Encode attention weights for audits / block tests
python examples/08_convert_weights.py \
  --model /path/to/model.Q4_K_M.gguf \
  --out ./optical_weights_mistral7b \
  --phase-bits 8

# 2) All-layer score audit
python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b

# 3) Hybrid block parity
python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0

# 4) Full hybrid generate from the same GGUF (needs RAM)
python examples/14_hybrid_generate_mistral.py \
  --gguf /path/to/model.Q4_K_M.gguf \
  --max-new 8 \
  --compare-digital
```

Use `--max-layers 2` on example 14 if memory is limited while bringing the stack up.

---

## Layout

```
atom/
  attention.py      Optical score math
  noise.py          Phase / angular / crosstalk noise
  capacity.py       M#-aware capacity
  convert.py        Checkpoint → phase-encoded attention weights
  hybrid.py         Generic hybrid attention module
  hybrid_block.py   Loaded hybrid attention from optical_weights.pt
  gguf_model.py     Full GGUF hybrid Mistral/Llama forward + generate
  optical_weights_io.py
examples/           01–14 runnable scripts
docs/               model, benchmarks, evidence_status
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas: richer noise tied to measured media, FPGA kernel for the score path, multi-module system design, and eval harnesses on real prompts with a tokenizer.

---

## References

- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Psaltis, D., Brady, D., & Wagner, K. (1988). Adaptive optical networks using photorefractive crystals. *Applied Optics*, 27(9), 1752–1759.
- Psaltis, D., & Mok, F. (1995). Holographic memories. *Scientific American*, 273(5), 70–76.
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004–1008.
- Miller, D. A. B. (2017). Attojoule optoelectronics. *Journal of Lightwave Technology*, 35(3), 346–396.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
