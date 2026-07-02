"""Optical-interference attention helpers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def encode_signed_values(values: torch.Tensor) -> torch.Tensor:
    """Encode real values as complex wave amplitudes with sign in phase."""
    amplitude = values.abs()
    phase = torch.where(values >= 0, torch.zeros_like(values), torch.full_like(values, math.pi))
    return amplitude * torch.exp(1j * phase)


def optical_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Return attention scores from atomal interference.

    Input shape is `(..., sequence, features)`. The result shape is
    `(..., query_sequence, key_sequence)`.

    With `normalize=False`, this matches scaled dot-product attention:
    `query @ key.T / sqrt(features)`.

    With `normalize=True`, it returns cosine-like similarity scores.
    """
    if normalize:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        scale = 1.0
    else:
        scale = math.sqrt(query.shape[-1])

    q_wave = encode_signed_values(query)
    k_wave = encode_signed_values(key)
    scores = torch.einsum("...qd,...kd->...qk", q_wave, torch.conj(k_wave)).real
    return scores / scale


class OpticalSelfAttention(nn.Module):
    """A compact self-attention layer whose scores use optical interference math."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = torch.softmax(optical_scores(q, k), dim=-1)
        return self.output(weights @ v), weights
