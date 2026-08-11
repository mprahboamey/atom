"""Text-prompt hybrid eval on a Hugging Face safetensors folder.

  python examples/16_safetensors_text_eval.py \
    --model /path/to/SmolLM2-135M-Instruct \
    --prompt "The capital of France is" \
    --max-new 4

Reports optical vs digital greedy sequences and decoded text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from atom.hybrid_model import HybridTransformer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True, help="HF folder with safetensors")
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max-new", type=int, default=4)
    p.add_argument("--out", type=str, default="results/safetensors_text_eval.json")
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("pip install transformers") from e

    tok = AutoTokenizer.from_pretrained(args.model)
    model = HybridTransformer.from_checkpoint(args.model, stream_layers=True)
    cfg = model.cfg
    print(
        f"source={cfg.source} layers={cfg.n_layers} "
        f"hidden={cfg.hidden_size} heads={cfg.n_heads} kv={cfg.n_kv_heads}"
    )

    prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
    print(f"prompt: {args.prompt!r}")
    print(f"prompt ids: {prompt_ids}")

    out_opt = model.generate(prompt_ids, max_new_tokens=args.max_new, use_optical=True)
    out_dig = model.generate(prompt_ids, max_new_tokens=args.max_new, use_optical=False)
    identical = out_opt == out_dig
    print(f"optical ids: {out_opt}")
    print(f"digital ids: {out_dig}")
    print(f"identical: {identical}")
    print(f"optical text: {tok.decode(out_opt)!r}")
    print(f"digital text: {tok.decode(out_dig)!r}")

    x = torch.tensor([prompt_ids[: min(8, len(prompt_ids))]])
    lo = model.forward(x, use_optical=True)
    ld = model.forward(x, use_optical=False)
    mse = (lo.float() - ld.float()).pow(2).mean().item()
    top1 = (lo.argmax(-1) == ld.argmax(-1)).float().mean().item()
    print(f"logits MSE={mse:.6e}  top-1 agree={top1:.1%}")

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "optical": out_opt,
        "digital": out_dig,
        "identical": identical,
        "optical_text": tok.decode(out_opt),
        "digital_text": tok.decode(out_dig),
        "logits_mse": mse,
        "logits_top1_agree": top1,
        "layers": cfg.n_layers,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
