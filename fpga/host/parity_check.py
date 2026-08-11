#!/usr/bin/env python3
"""CPU parity: FPGA score stand-in math vs atom.optical_scores.

Run from repo root:
  python fpga/host/parity_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom.attention import optical_scores


def cpu_score_kernel(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Same contract as the HLS score kernel: (B,S,D) x (B,S,D) -> (B,S,S)."""
    scale = math.sqrt(q.shape[-1])
    return (q @ k.transpose(-2, -1)) / scale


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 16, 32)
    k = torch.randn(2, 16, 32)
    a = optical_scores(q, k)
    b = cpu_score_kernel(q, k)
    max_diff = (a - b).abs().max().item()
    ok = max_diff < 1e-5
    print(f"max|optical_scores - cpu_score_kernel| = {max_diff:.3e}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
