"""Shared helpers for examples that read optical_weights.pt."""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch


def reconstruct_weight(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    sign = torch.where(
        phase.abs() < (math.pi / 2), torch.ones_like(phase), -torch.ones_like(phase)
    )
    return amplitude * sign


def load_payload(weights_dir: str | Path) -> dict:
    path = Path(weights_dir) / "optical_weights.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def layer_indices(payload: dict) -> list[int]:
    found = set()
    for k in payload:
        m = re.search(r"model__layers__(\d+)__self_attn__q_proj__weight\.amplitude", k)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)
