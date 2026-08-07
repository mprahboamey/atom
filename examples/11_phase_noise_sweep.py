"""Phase-bit and noise sweep on a real converted layer.

  python examples/11_phase_noise_sweep.py --weights ./optical_weights_mistral7b --layer 0
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
from atom.optical_weights_io import reconstruct_weight, load_payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/phase_noise_sweep.json")
    args = p.parse_args()

    payload = load_payload(args.weights)
    q_name = f"model__layers__{args.layer}__self_attn__q_proj__weight"
    k_name = f"model__layers__{args.layer}__self_attn__k_proj__weight"
    w_q = reconstruct_weight(payload[f"{q_name}.amplitude"], payload[f"{q_name}.phase"])
    w_k = reconstruct_weight(payload[f"{k_name}.amplitude"], payload[f"{k_name}.phase"])

    torch.manual_seed(args.seed)
    x = torch.randn(1, args.seq, w_q.shape[1])
    q = x @ w_q.T
    k = x @ w_k.T
    d = min(q.shape[-1], k.shape[-1])
    q_, k_ = q[..., :d], k[..., :d]
    digital = (q_ @ k_.transpose(-2, -1)) / math.sqrt(d)

    rows = []
    print(f"{'bits':>4} {'sigma':>6} {'xtalk':>6} | {'MSE':>12} {'top1':>8}")
    for bits in (4, 6, 8, None):
        for sigma in (0.0, 0.02, 0.05):
            for xtalk in (0.0, 0.05):
                if bits is None and sigma == 0 and xtalk == 0:
                    optical = optical_scores(q_, k_)
                else:
                    noise = NoiseConfig(
                        phase_bits=bits,
                        phase_sigma=sigma,
                        crosstalk=xtalk,
                    )
                    pos = torch.arange(args.seq, dtype=torch.float32).unsqueeze(0)
                    optical = optical_scores_general(
                        q_, k_, query_positions=pos, key_positions=pos, noise=noise
                    )
                mse = (digital - optical).pow(2).mean().item()
                top1 = (
                    F.softmax(digital, -1).argmax(-1) == F.softmax(optical, -1).argmax(-1)
                ).float().mean().item()
                rows.append(
                    {
                        "phase_bits": bits,
                        "phase_sigma": sigma,
                        "crosstalk": xtalk,
                        "mse": mse,
                        "top1": top1,
                    }
                )
                bstr = "inf" if bits is None else str(bits)
                print(f"{bstr:>4} {sigma:6.2f} {xtalk:6.2f} | {mse:12.4e} {top1:7.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"layer": args.layer, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
