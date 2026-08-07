"""Minimal structural generate loop using hybrid optical attention.

Attention weights are real (from convert). Embed / lm_head are toy random
tables — token ids are not meaningful language. Purpose: closed loop that
calls the optical score path on real weights.

  python examples/13_minimal_hybrid_generate.py --weights ./optical_weights_mistral7b
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom.hybrid_block import LoadedHybridAttention
from atom.optical_weights_io import load_payload, layer_indices


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    args = p.parse_args()

    payload = load_payload(args.weights)
    available = layer_indices(payload)
    use_layers = available[: args.layers]
    print(f"stacking layers {use_layers} (attention only; toy lm head)")

    blocks = [
        LoadedHybridAttention.from_optical_dir(
            args.weights, layer=L, num_heads=args.heads, num_kv_heads=args.kv_heads
        )
        for L in use_layers
    ]
    dim = blocks[0].w_q.shape[1]
    torch.manual_seed(args.seed)
    vocab = 32000
    embed = torch.randn(vocab, dim) * 0.02
    lm_head = torch.randn(vocab, dim) * 0.02

    token_ids = [1, 2, 3, 4]
    x = embed[token_ids].unsqueeze(0)

    for step in range(args.steps):
        h = x
        for block in blocks:
            attn_out, _ = block(h, use_optical=True)
            h = h + attn_out
        logits = h[:, -1, :] @ lm_head.T
        next_id = int(logits.argmax(dim=-1).item())
        token_ids.append(next_id)
        x = torch.cat([x, embed[next_id].view(1, 1, dim)], dim=1)
        if x.shape[1] > args.seq:
            x = x[:, -args.seq :, :]
        print(f"step {step}: next_id={next_id}  ctx_len={x.shape[1]}")

    print(f"token id sequence (toy): {token_ids}")
    print("note: ids are not decoded language; MLP/embed/lm_head not from GGUF.")


if __name__ == "__main__":
    main()
