"""Optical-interference attention helpers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .noise import (
    quantize_phase,
    add_phase_noise,
    add_angular_jitter,
    apply_crosstalk,
    apply_bragg_crosstalk,
    NoiseConfig,
)


def encode_signed_values(values: torch.Tensor) -> torch.Tensor:
    """Encode real values as complex wave amplitudes with sign in phase.

    Phase only takes 0 or pi so Re(q_wave * conj(k_wave)) equals q*k termwise.
    """
    amplitude = values.abs()
    phase = torch.where(values >= 0, torch.zeros_like(values), torch.full_like(values, math.pi))
    return amplitude * torch.exp(1j * phase)


def encode_angular_phase(
    values: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    phase_bits: int | None = None,
    phase_sigma: float = 0.0,
) -> torch.Tensor:
    """Encode values with continuous phase from sign and token position."""
    amplitude = values.abs()
    sign_phase = torch.where(values >= 0, torch.zeros_like(values), torch.full_like(values, math.pi))
    dim = values.shape[-1]
    freq_index = torch.arange(dim, device=values.device, dtype=values.dtype)
    freq = base ** (-freq_index / dim)
    angular_phase = positions.unsqueeze(-1) * freq
    total_phase = sign_phase + angular_phase
    if phase_bits is not None:
        total_phase = quantize_phase(total_phase, phase_bits)
    if phase_sigma > 0:
        total_phase = add_phase_noise(total_phase, phase_sigma)
    return amplitude * torch.exp(1j * total_phase)


def optical_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Binary-phase optical scores; exact match to scaled dot-product."""
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


def optical_scores_general(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    normalize: bool = False,
    phase_bits: int | None = None,
    phase_sigma: float = 0.0,
    angular_jitter: float = 0.0,
    crosstalk: float = 0.0,
    crosstalk_kernel: int = 3,
    noise: NoiseConfig | None = None,
) -> torch.Tensor:
    """Scores with continuous phase and optional noise / Bragg crosstalk.

    Ideal binary path only when positions are None, phase_sigma is 0, and
    phase_bits is None. phase_sigma without positions is applied (not silent).
    """
    bragg_strength = 0.0
    bragg_sinc_width = 1.5
    if noise is not None:
        phase_bits = noise.phase_bits if noise.phase_bits is not None else phase_bits
        phase_sigma = noise.phase_sigma
        angular_jitter = noise.angular_jitter
        crosstalk = noise.crosstalk
        crosstalk_kernel = noise.crosstalk_kernel
        bragg_strength = getattr(noise, "bragg_strength", 0.0)
        bragg_sinc_width = getattr(noise, "bragg_sinc_width", 1.5)

    if normalize:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        scale = 1.0
    else:
        scale = math.sqrt(query.shape[-1])

    use_binary = (
        query_positions is None
        and key_positions is None
        and phase_sigma == 0.0
        and phase_bits is None
    )
    if use_binary:
        q_wave = encode_signed_values(query)
        k_wave = encode_signed_values(key)
    else:
        if query_positions is None:
            query_positions = torch.zeros(
                query.shape[:-1], device=query.device, dtype=query.dtype
            )
        if key_positions is None:
            key_positions = torch.zeros(
                key.shape[:-1], device=key.device, dtype=key.dtype
            )

        if angular_jitter > 0:
            query_positions = add_angular_jitter(query_positions, angular_jitter)
            key_positions = add_angular_jitter(key_positions, angular_jitter)

        q_wave = encode_angular_phase(
            query, query_positions, phase_bits=phase_bits, phase_sigma=phase_sigma
        )
        k_wave = encode_angular_phase(
            key, key_positions, phase_bits=phase_bits, phase_sigma=phase_sigma
        )

    scores = torch.einsum("...qd,...kd->...qk", q_wave, torch.conj(k_wave)).real
    scores = scores / scale

    if bragg_strength > 0:
        scores = apply_bragg_crosstalk(
            scores, sinc_width=bragg_sinc_width, strength=bragg_strength
        )
    elif crosstalk > 0:
        scores = apply_crosstalk(scores, strength=crosstalk, kernel_size=crosstalk_kernel)

    return scores


def optical_scores_multihead(
    query: torch.Tensor,
    key: torch.Tensor,
    noise: NoiseConfig | None = None,
) -> torch.Tensor:
    """Vectorized multi-head optical scores.

    query/key: (batch, heads, seq, dim). Returns (batch, heads, q_seq, k_seq).
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("expected (batch, heads, seq, dim)")
    b, h, sq, d = query.shape
    sk = key.shape[2]
    q = query.reshape(b * h, sq, d)
    k = key.reshape(b * h, sk, d)
    if noise is None or (
        noise.phase_sigma == 0
        and noise.angular_jitter == 0
        and noise.crosstalk == 0
        and getattr(noise, "bragg_strength", 0) == 0
        and noise.phase_bits is None
    ):
        scores = optical_scores(q, k)
    else:
        scores = optical_scores_general(q, k, noise=noise)
    return scores.reshape(b, h, sq, sk)


class OpticalSelfAttention(nn.Module):
    """Self-attention layer whose scores use optical interference math."""

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
