"""Compare optical vs digital attention scores on one converted layer.

Loads optical_weights.pt from examples/08_convert_weights.py, reconstructs
float q_proj / k_proj for a chosen layer, projects random token activations,
and checks optical_scores against scaled dot-product attention.

Does not run full model inference. Does not upload weights.

Example:
  python examples/09_optical_vs_digital_layer.py \
    --weights ./optical_weights_mistral7b \
    --layer 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from atom.attention import optical_scores, optical_scores_general
from atom.noise import NoiseConfig


def reconstruct_weight(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Recover signed weight from amplitude + 0/pi phase."""
    sign = torch.where(phase.abs() < (math.pi / 2), torch.ones_like(phase), -torch.ones_like(phase))
    return amplitude * sign


def load_layer_qk(weights_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(weights_dir / "optical_weights.pt", map_location="cpu", weights_only=True)
    q_key = f"model__layers__{layer}__self_attn__q_proj__weight"
    k_key = f"model__layers__{layer}__self_attn__k_proj__weight"
    # keys were sanitized with . -> __
    q_amp = payload[f"{q_key}.amplitude"]
    q_phase = payload[f"{q_key}.phase"]
    k_amp = payload[f"{k_key}.amplitude"]
    k_phase = payload[f"{k_key}.phase"]
    w_q = reconstruct_weight(q_amp, q_phase)
    w_k = reconstruct_weight(k_amp, k_phase)
    return w_q, w_k


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True, help="Dir with optical_weights.pt")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase-sigma", type=float, default=0.0)
    p.add_argument("--crosstalk", type=float, default=0.0)
    args = p.parse_args()

    weights_dir = Path(args.weights)
    meta_path = weights_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"source tensors: {meta.get('n_converted')}  phase_bits={meta.get('phase_bits')}")

    w_q, w_k = load_layer_qk(weights_dir, args.layer)
    # Linear is y = x @ W.T for nn.Linear; W shape (out, in)
    in_dim = w_q.shape[1]
    q_out, k_out = w_q.shape[0], w_k.shape[0]
    print(f"layer {args.layer}: W_q {tuple(w_q.shape)}  W_k {tuple(w_k.shape)}")

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.seq, in_dim)
    # Project: (B,S,in) @ (in,out) = (B,S,out)
    q = x @ w_q.T
    k = x @ w_k.T

    # Digital reference uses the shared head dim of q (full q dim)
    # For GQA, k has fewer rows; scores are still q @ k.T over feature dim of k
    # Align feature dim: use min dim for a fair score matrix
    d = min(q.shape[-1], k.shape[-1])
    q_ = q[..., :d]
    k_ = k[..., :d]

    digital = (q_ @ k_.transpose(-2, -1)) / math.sqrt(d)
    optical = optical_scores(q_, k_)

    mse = (digital - optical).pow(2).mean().item()
    max_abs = (digital - optical).abs().max().item()
    print(f"ideal optical vs digital:  MSE={mse:.6e}  max|diff|={max_abs:.6e}")

    ideal_attn = F.softmax(digital, dim=-1)
    opt_attn = F.softmax(optical, dim=-1)
    kl = F.kl_div(opt_attn.clamp_min(1e-12).log(), ideal_attn, reduction="batchmean").item()
    top1 = (ideal_attn.argmax(-1) == opt_attn.argmax(-1)).float().mean().item()
    print(f"softmax KL={kl:.6e}  top-1 agree={top1:.1%}")

    if args.phase_sigma > 0 or args.crosstalk > 0:
        noise = NoiseConfig(phase_sigma=args.phase_sigma, crosstalk=args.crosstalk)
        positions = torch.arange(args.seq, dtype=torch.float32)
        noisy = optical_scores_general(
            q_, k_,
            query_positions=positions.unsqueeze(0).expand(args.batch, -1),
            key_positions=positions.unsqueeze(0).expand(args.batch, -1),
            noise=noise,
        )
        # same positions cancel continuous phase -> near binary; still applies phase_sigma/crosstalk knobs
        mse_n = (digital - noisy).pow(2).mean().item()
        print(f"with noise (sigma={args.phase_sigma}, xtalk={args.crosstalk}): MSE vs digital={mse_n:.6e}")

    print("done")


if __name__ == "__main__":
    main()
