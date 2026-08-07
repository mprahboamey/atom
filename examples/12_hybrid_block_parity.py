"""Hybrid attention block output vs pure digital on one converted layer.

  python examples/12_hybrid_block_parity.py --weights ./optical_weights_mistral7b --layer 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom.hybrid_block import LoadedHybridAttention
from atom.noise import NoiseConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    layer = LoadedHybridAttention.from_optical_dir(
        args.weights,
        layer=args.layer,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        noise=None,
    )
    in_dim = layer.w_q.shape[1]
    x = torch.randn(args.batch, args.seq, in_dim)

    out_opt, _ = layer(x, use_optical=True)
    out_dig, _ = layer(x, use_optical=False)
    mse = (out_opt - out_dig).pow(2).mean().item()
    max_abs = (out_opt - out_dig).abs().max().item()
    print(f"layer {args.layer} hybrid optical vs digital block output")
    print(f"  MSE={mse:.6e}  max|diff|={max_abs:.6e}")
    print(f"  shapes out={tuple(out_opt.shape)}")

    noisy = LoadedHybridAttention.from_optical_dir(
        args.weights,
        layer=args.layer,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        noise=NoiseConfig(phase_sigma=0.05, crosstalk=0.05),
    )
    out_n, _ = noisy(x, use_optical=True)
    mse_n = (out_n - out_dig).pow(2).mean().item()
    print(f"  with noise MSE vs digital={mse_n:.6e}")


if __name__ == "__main__":
    main()
