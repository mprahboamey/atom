# 1B-class phase score check without a laptop

This sandbox has ~2GB RAM and no GPU. A full 1B hybrid train cannot run here.

## What we can do here

- Score identity at **1B-class shapes** (heads, seq, head dim)
- Scripts + docs + GitHub

## What you can do from a phone

### Google Colab (free GPU)

1. Open [Google Colab](https://colab.research.google.com) in the phone browser  
2. Runtime → Change runtime type → **GPU**  
3. Cell:

```python
!pip install -q transformers accelerate torch
!wget -q https://raw.githubusercontent.com/mprahboamey/atom/main/scripts/colab_1b_phase_score_check.py
!python colab_1b_phase_score_check.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-layers 4
```

Lighter model if memory fails:

```python
!python colab_1b_phase_score_check.py --model Qwen/Qwen2.5-0.5B-Instruct --max-layers 8
```

### What the script claims

On a **real ~1B HF checkpoint**, for the first N layers, **phase score path vs digital QKᵀ/√d** on that layer’s Q,K (max abs error). That is the 1B-class **software** check without your laptop.

Full hybrid generate parity on all layers of 1B needs more VRAM and a tighter integration with RoPE/GQA; extend the script once the layer audit passes.
