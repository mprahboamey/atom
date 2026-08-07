"""Audit optical vs digital scores on every converted attention layer.

Writes a JSON report under results/ (local only; do not commit large artifacts).

  python examples/10_all_layer_audit.py --weights ./optical_weights_mistral7b
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

from atom.attention import optical_scores
from examples.utils_optical_weights import reconstruct_weight, load_payload, layer_indices


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/all_layer_audit.json")
    args = p.parse_args()

    payload = load_payload(args.weights)
    layers = layer_indices(payload)
    torch.manual_seed(args.seed)
    rows = []

    for layer in layers:
        q_name = f"model__layers__{layer}__self_attn__q_proj__weight"
        k_name = f"model__layers__{layer}__self_attn__k_proj__weight"
        w_q = reconstruct_weight(payload[f"{q_name}.amplitude"], payload[f"{q_name}.phase"])
        w_k = reconstruct_weight(payload[f"{k_name}.amplitude"], payload[f"{k_name}.phase"])
        in_dim = w_q.shape[1]
        x = torch.randn(1, args.seq, in_dim)
        q = x @ w_q.T
        k = x @ w_k.T
        d = min(q.shape[-1], k.shape[-1])
        q_, k_ = q[..., :d], k[..., :d]
        digital = (q_ @ k_.transpose(-2, -1)) / math.sqrt(d)
        optical = optical_scores(q_, k_)
        mse = (digital - optical).pow(2).mean().item()
        ideal_attn = F.softmax(digital, dim=-1)
        opt_attn = F.softmax(optical, dim=-1)
        top1 = (ideal_attn.argmax(-1) == opt_attn.argmax(-1)).float().mean().item()
        rows.append({"layer": layer, "mse": mse, "top1": top1, "d": d})
        print(f"layer {layer:2d}  MSE={mse:.3e}  top1={top1:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_layers": len(rows),
        "max_mse": max(r["mse"] for r in rows),
        "min_top1": min(r["top1"] for r in rows),
        "layers": rows,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}  max_mse={summary['max_mse']:.3e}  min_top1={summary['min_top1']:.1%}")


if __name__ == "__main__":
    main()
