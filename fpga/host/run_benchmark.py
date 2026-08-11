#!/usr/bin/env python3
"""Benchmark the CPU reference score kernel; same metrics schema for FPGA runs.

On FPGA, replace the timed region with a device enqueue and fill
energy_joules from board sensors / power rail logs when available.

  python fpga/host/run_benchmark.py --seq 128 --dim 64 --repeat 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


def score_kernel(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scale = math.sqrt(q.shape[-1])
    return (q @ k.transpose(-2, -1)) / scale


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--out", type=str, default="fpga/results/cpu_benchmark.json")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Label only until a real FPGA backend is wired",
    )
    args = p.parse_args()

    q = torch.randn(args.batch, args.seq, args.dim)
    k = torch.randn(args.batch, args.seq, args.dim)

    # warmup
    for _ in range(5):
        score_kernel(q, k)

    t0 = time.perf_counter()
    for _ in range(args.repeat):
        out = score_kernel(q, k)
    t1 = time.perf_counter()
    _ = float(out.sum())  # keep result live

    total_s = t1 - t0
    per_s = total_s / args.repeat
    # Energy unknown on CPU path without RAPL/etc.
    payload = {
        "device": args.device,
        "backend": "pytorch_cpu_reference",
        "batch": args.batch,
        "seq": args.seq,
        "dim": args.dim,
        "repeat": args.repeat,
        "total_seconds": total_s,
        "seconds_per_call": per_s,
        "ms_per_call": per_s * 1e3,
        "energy_joules_total": None,
        "energy_joules_per_call": None,
        "notes": [
            "energy_joules_* filled on FPGA from power measurement",
            "score kernel is digital stand-in for hybrid scores (= dot product)",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
