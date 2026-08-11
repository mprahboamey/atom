"""Hybrid generate from GGUF or Hugging Face safetensors.

Same command for both checkpoint types:

  # GGUF (what you have now)
  python examples/14_hybrid_generate_mistral.py \
    --model /path/to/model.Q4_K_M.gguf \
    --stream-layers \
    --max-new 8 \
    --compare-digital

  # Later: safetensors folder (HF layout)
  python examples/14_hybrid_generate_mistral.py \
    --model /path/to/mistral-hf-folder \
    --stream-layers \
    --prompt "Hello" \
    --tokenizer mistralai/Mistral-7B-Instruct-v0.2

--stream-layers loads one transformer layer at a time (needed for 32-layer
runs on smaller machines). --max-layers still works for shorter stacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from atom.hybrid_model import HybridTransformer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        "--gguf",
        dest="model",
        type=str,
        required=True,
        help="Path to .gguf file OR HF safetensors directory",
    )
    p.add_argument("--prompt-tokens", type=str, default="1,2,3,4")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (needs --tokenizer)")
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer id or local path (optional)",
    )
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument(
        "--stream-layers",
        action="store_true",
        help="Load one layer at a time (use for full 32-layer runs)",
    )
    p.add_argument("--compare-digital", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    print(f"Loading checkpoint from {args.model}")
    print(f"  stream_layers={args.stream_layers}  max_layers={args.max_layers}")
    model = HybridTransformer.from_checkpoint(
        args.model,
        max_layers=args.max_layers,
        stream_layers=args.stream_layers,
    )
    cfg = model.cfg
    print(
        f"config: source={cfg.source} layers={cfg.n_layers} "
        f"hidden={cfg.hidden_size} heads={cfg.n_heads} kv={cfg.n_kv_heads} vocab={cfg.vocab_size}"
    )

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install transformers for --tokenizer") from e
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.prompt is not None:
        if tok is None:
            raise SystemExit("--prompt requires --tokenizer")
        prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
    else:
        prompt_ids = [int(x) for x in args.prompt_tokens.split(",") if x.strip() != ""]

    print(f"prompt token ids: {prompt_ids}")

    out_opt = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new,
        use_optical=True,
        temperature=args.temperature,
    )
    print(f"optical-score generate: {out_opt}")
    if tok is not None:
        print(f"optical text: {tok.decode(out_opt, skip_special_tokens=False)}")

    if args.compare_digital:
        out_dig = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new,
            use_optical=False,
            temperature=args.temperature,
        )
        print(f"digital-score generate: {out_dig}")
        if tok is not None:
            print(f"digital text: {tok.decode(out_dig, skip_special_tokens=False)}")
        print(f"sequences identical: {out_opt == out_dig}")

    print("done")


if __name__ == "__main__":
    main()
