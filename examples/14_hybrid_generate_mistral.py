"""Hybrid generate from a full Mistral/Llama GGUF (real weights end-to-end).

Loads embeddings, every layer (attention + MLP + norms), and lm_head from
the same GGUF file you already have. Attention scores use the optical path;
everything else is digital matmuls on dequantized checkpoint weights.

This is still a software simulation of the optical score step — not a
physical crystal — but the weights are the model, not placeholders.

Requires: pip install gguf
RAM: plan for ~12–20 GB when loading a 7B Q4 model into float32.

  python examples/14_hybrid_generate_mistral.py \
    --gguf /path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --prompt-tokens 1,2,3,4 \
    --max-new 8

Optional: --max-layers 4 for a shorter stack while debugging memory.
Optional: --compare-digital runs the same generate with digital scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom.gguf_model import HybridMistralFromGGUF


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=str, required=True)
    p.add_argument(
        "--prompt-tokens",
        type=str,
        default="1,2,3,4",
        help="Comma-separated token ids (use a real tokenizer offline if you need text)",
    )
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--compare-digital", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    print(f"Loading GGUF (this can take several minutes and substantial RAM)...")
    print(f"  {args.gguf}")
    model = HybridMistralFromGGUF.from_gguf(
        args.gguf, max_layers=args.max_layers
    )
    cfg = model.cfg
    print(
        f"config: layers={cfg.n_layers} hidden={cfg.hidden_size} "
        f"heads={cfg.n_heads} kv_heads={cfg.n_kv_heads} vocab={cfg.vocab_size}"
    )

    prompt = [int(x) for x in args.prompt_tokens.split(",") if x.strip() != ""]
    print(f"prompt token ids: {prompt}")

    out_opt = model.generate(
        prompt,
        max_new_tokens=args.max_new,
        use_optical=True,
        temperature=args.temperature,
    )
    print(f"optical-score generate: {out_opt}")

    if args.compare_digital:
        out_dig = model.generate(
            prompt,
            max_new_tokens=args.max_new,
            use_optical=False,
            temperature=args.temperature,
        )
        print(f"digital-score generate: {out_dig}")
        match = out_opt == out_dig
        print(f"sequences identical: {match}")

    print("done")


if __name__ == "__main__":
    main()
