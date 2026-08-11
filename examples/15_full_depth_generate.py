"""Full-depth (up to 32 layer) hybrid generate with layer streaming.

Designed for machines that OOM when all layers are loaded at once.

  python examples/15_full_depth_generate.py \
    --model /path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --max-new 4 \
    --compare-digital
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from atom.hybrid_model import HybridTransformer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompt-tokens", type=str, default="1,2,3,4")
    p.add_argument("--max-new", type=int, default=4)
    p.add_argument("--compare-digital", action="store_true")
    args = p.parse_args()

    print("Full-depth hybrid generate with stream_layers=True")
    model = HybridTransformer.from_checkpoint(
        args.model,
        max_layers=None,
        stream_layers=True,
    )
    print(
        f"source={model.cfg.source} layers={model.cfg.n_layers} "
        f"hidden={model.cfg.hidden_size}"
    )
    prompt = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]
    out_opt = model.generate(prompt, max_new_tokens=args.max_new, use_optical=True)
    print(f"optical: {out_opt}")
    if args.compare_digital:
        out_dig = model.generate(prompt, max_new_tokens=args.max_new, use_optical=False)
        print(f"digital: {out_dig}")
        print(f"identical: {out_opt == out_dig}")
    print("done")


if __name__ == "__main__":
    main()
