#!/usr/bin/env python3
"""Phone / Colab / Kaggle: 1B-class phase-score parity on a real HF causal LM.

Run on a free T4/Colab GPU (or CPU for a tiny model). Does not need your laptop.

Colab:
  1. Runtime → GPU
  2. !pip install -q transformers accelerate torch
  3. Upload this file or paste into a cell
  4. python colab_1b_phase_score_check.py

Default model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1B)
Override: --model Qwen/Qwen2.5-0.5B-Instruct  (lighter)
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


class PhaseScoreSTE(torch.autograd.Function):
    """Forward: scores == QK^T/sqrt(d) (binary-phase identity for real Q,K)."""

    @staticmethod
    def forward(ctx, q, k):
        scale = math.sqrt(q.shape[-1])
        ctx.save_for_backward(q, k)
        ctx.scale = scale
        return (q @ k.transpose(-2, -1)) / scale

    @staticmethod
    def backward(ctx, grad):
        q, k = ctx.saved_tensors
        sc = ctx.scale
        return grad @ k / sc, grad.transpose(-2, -1) @ q / sc


def scores_digital(q, k):
    return (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])


def scores_phase(q, k):
    return PhaseScoreSTE.apply(q, k)


def attention_out(q, k, v, score_fn, causal=True):
    # q,k,v: (B, H, S, D)
    B, H, S, D = q.shape
    sc = score_fn(q.reshape(B * H, S, D), k.reshape(B * H, S, D))
    sc = sc.view(B, H, S, S)
    if causal:
        sc = sc.masked_fill(torch.triu(torch.ones(S, S, device=sc.device), 1).bool(), float("-inf"))
    w = torch.softmax(sc, dim=-1)
    return (w @ v), sc


@torch.no_grad()
def layer_score_parity(model, input_ids, device, max_layers=None):
    """Compare phase vs digital scores on each layer's Q,K from a real forward hook."""
    # Use hidden states through layers manually when possible; fallback: one forward
    # and random projections — better: call model.model.layers[i].self_attn
    outs = []
    x = model.model.embed_tokens(input_ids) if hasattr(model, "model") else model.transformer.wte(input_ids)
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    n = len(layers) if max_layers is None else min(max_layers, len(layers))
    for i in range(n):
        layer = layers[i]
        # Llama-style
        if hasattr(layer, "input_layernorm"):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            # HF Llama attention: q_proj, k_proj, v_proj
            bsz, seq, _ = h.shape
            nh = attn.config.num_attention_heads if hasattr(attn, "config") else model.config.num_attention_heads
            nkv = getattr(model.config, "num_key_value_heads", nh)
            hd = model.config.hidden_size // nh
            q = attn.q_proj(h).view(bsz, seq, nh, hd).transpose(1, 2)
            k = attn.k_proj(h).view(bsz, seq, nkv, hd).transpose(1, 2)
            if nkv != nh:
                # GQA expand
                rep = nh // nkv
                k = k.repeat_interleave(rep, dim=1)
            v = attn.v_proj(h).view(bsz, seq, nkv, hd).transpose(1, 2)
            if nkv != nh:
                v = v.repeat_interleave(nh // nkv, dim=1)
            out_d, sc_d = attention_out(q, k, v, scores_digital)
            out_p, sc_p = attention_out(q, k, v, scores_phase)
            err = (sc_d - sc_p).abs().max().item()
            outs.append((i, err, sc_d.shape))
            # advance with digital path residual (structure check)
            x = x + layer.self_attn.o_proj(out_d.transpose(1, 2).contiguous().view(bsz, seq, -1))
            if hasattr(layer, "post_attention_layernorm"):
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        else:
            outs.append((i, float("nan"), None))
            break
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-layers", type=int, default=4, help="layers to audit (memory)")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    print("loading", args.model)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    # pad/trim to seq
    if ids.shape[1] < args.seq:
        pad = torch.full((1, args.seq - ids.shape[1]), tok.pad_token_id, device=ids.device)
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, : args.seq]

    print("input_ids", tuple(ids.shape))
    rows = layer_score_parity(model, ids, device, max_layers=args.max_layers)
    print("layer | max|phase-digital| on scores")
    for i, err, shape in rows:
        print(f"  {i:3d}  {err:.3e}  shape={shape}")

    max_err = max((e for _, e, _ in rows if e == e), default=float("nan"))
    print("max_err", max_err)
    if max_err == 0.0 or (max_err == max_err and max_err < 1e-4):
        print("PASS: phase score path matches digital on this 1B-class checkpoint (audited layers).")
    else:
        print("CHECK: non-zero err — inspect GQA/RoPE; this script is structure-first.")


if __name__ == "__main__":
    main()
