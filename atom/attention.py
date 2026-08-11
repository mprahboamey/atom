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
    NoiseConfig,
)


def encode_signed_values(values: torch.Tensor) -> torch.Tensor:
    """Encode real values as complex wave amplitudes with sign in phase.

    Phase here only ever takes two values, 0 or pi. That is not a
    simplification -- it is the *unique* phase choice for which
    `Re(q_wave * conj(k_wave))` reduces exactly, term for term, to the
    real product `q * k`. Exact term-wise recovery requires
    `cos(theta_q - theta_k) = sign(q * k)`, and cosine only hits +/-1
    at 0 and pi. So this function is the degenerate, binary-phase case
    used to prove exact equivalence to scaled dot-product attention
    (see `optical_scores`) -- it is intentionally not a demonstration
    of genuine continuous-phase interference. For phase that carries
    real, continuous information (and produces interference effects a
    plain dot product cannot reproduce), see `encode_angular_phase`
    and `optical_scores_general`.
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
    """Encode values as waves with a genuine continuous phase.

    Amplitude carries magnitude as before, but phase is now a
    continuous function of both the value's sign *and* its token's
    angular position -- modeling angular multiplexing, where each
    sequence position is addressed at a distinct Bragg angle inside
    the crystal. This is the same construction as rotary position
    embeddings, applied here as an optical angle rather than an
    abstract index.

    `phase_bits`, if given, quantizes the resulting phase to that many
    bits before encoding -- models a real SLM/crystal that can only
    address `2**phase_bits` discrete phase levels, instead of an
    idealized continuous angle. `None` (default) keeps phase exact,
    matching the ideal theoretical case tested elsewhere.

    `phase_sigma` adds independent Gaussian phase jitter (radians) after
    any quantization. Default 0.0 leaves phase unchanged.

    IMPORTANT: this phase only produces genuine interference when
    query and key tokens carry *different* positions -- if both sides
    of an interference term rotate by the same angle, that rotation
    cancels in `Re(q_wave * conj(k_wave))` (cos of a zero difference)
    and you're silently back to the binary case. Callers must pass
    distinct position tensors for query and key (see
    `optical_scores_general`), not shared ones.

    `values` is `(..., seq, features)`. `positions` is `(..., seq)`,
    one angular offset per token, broadcast across feature frequency
    channels. `positions == 0` everywhere recovers `encode_signed_values`
    exactly.
    """
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
    """Return attention scores from atomal interference.

    Input shape is `(..., sequence, features)`. The result shape is
    `(..., query_sequence, key_sequence)`.

    With `normalize=False`, this matches scaled dot-product attention:
    `query @ key.T / sqrt(features)`.

    With `normalize=True`, it returns cosine-like similarity scores.

    This uses `encode_signed_values`, the binary-phase (0/pi) case --
    see that function's docstring for why phase is degenerate here by
    construction. For continuous phase, use `optical_scores_general`.
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
    """Attention scores from interference with continuous phase and optional noise.

    Ideal binary path (exact match to scaled dot-product) only when positions
    are None, phase_sigma is 0, and phase_bits is None. If phase_sigma > 0
    without positions, zero positions are used so noise is still applied
    (previously phase_sigma was silently ignored).
    """
    if noise is not None:
        phase_bits = noise.phase_bits if noise.phase_bits is not None else phase_bits
        phase_sigma = noise.phase_sigma
        angular_jitter = noise.angular_jitter
        crosstalk = noise.crosstalk
        crosstalk_kernel = noise.crosstalk_kernel

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
            query_positions = torch.zeros(query.shape[:-1], device=query.device, dtype=query.dtype)
        if key_positions is None:
            key_positions = torch.zeros(key.shape[:-1], device=key.device, dtype=key.dtype)

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

    if crosstalk > 0:
        scores = apply_crosstalk(scores, strength=crosstalk, kernel_size=crosstalk_kernel)

    return scores


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
