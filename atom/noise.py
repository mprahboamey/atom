"""Noise modelling: physical constraints the pure-math simulator ignores.

Starts with phase quantization -- CONTRIBUTING.md flags this as the most
tractable open noise problem, and the natural next question after proving
the continuous-phase math works in the ideal case (see attention.py):
a real crystal can't write an infinite-precision phase angle. It writes
theta to some finite number of bits. This module answers "how many bits
before that actually matters."

Also covers Gaussian phase noise, angular (Bragg) position jitter, and
soft inter-channel crosstalk on attention scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def quantize_phase(phase: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a phase angle (radians, any range) to `bits`-bit precision.

    Models a real spatial light modulator or crystal write mechanism that
    can only address 2**bits distinct phase levels around the unit circle,
    rather than an idealized continuous angle. Phase is wrapped to
    [0, 2*pi) before quantizing, since phase is cyclic -- an SLM has no
    concept of "phase 400 degrees," it only has 2**bits discrete steps
    around one full turn.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = 2 ** bits
    wrapped = torch.remainder(phase, 2 * math.pi)
    step = 2 * math.pi / levels
    quantized = torch.round(wrapped / step) * step
    return torch.remainder(quantized, 2 * math.pi)


def add_phase_noise(phase: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add independent Gaussian phase jitter in radians.

    Covers write noise, residual SLM error, or uncorrected phase drift.
    sigma is the standard deviation. sigma=0 is a no-op.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return phase
    return phase + torch.randn_like(phase) * sigma


def add_angular_jitter(positions: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian jitter to angular / Bragg positions.

    Models thermal or mechanical drift of the incidence angle.
    sigma uses the same units as `positions`. sigma=0 is a no-op.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return positions
    return positions + torch.randn_like(positions) * sigma


def apply_crosstalk(
    scores: torch.Tensor,
    strength: float,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Soft leakage between neighbouring angular channels.

    `scores` shape is (..., query_seq, key_seq). Mixing happens along the
    key (angular) axis. `strength` in [0, 1] sets how much of the local
    neighbourhood is blended in (0 = none, 1 = full average over the kernel).

    Uses a uniform box kernel for now. A measured Bragg selectivity curve
    can replace it later without changing the call signature.
    """
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0 or kernel_size <= 1:
        return scores
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    pad = kernel_size // 2
    padded = torch.nn.functional.pad(scores, (pad, pad), mode="replicate")
    mixed = torch.zeros_like(scores)
    for i in range(kernel_size):
        mixed = mixed + padded[..., i : i + scores.shape[-1]]
    mixed = mixed / kernel_size

    return (1.0 - strength) * scores + strength * mixed


@dataclass
class NoiseConfig:
    """Optional noise parameters, all defaulting to ideal / off.

    Pass this (or the individual kwargs) into the optical score functions.
    """

    phase_bits: int | None = None
    phase_sigma: float = 0.0
    angular_jitter: float = 0.0
    crosstalk: float = 0.0
    crosstalk_kernel: int = 3
    bragg_strength: float = 0.0
    bragg_sinc_width: float = 1.5


def bragg_selectivity_kernel(
    n_channels: int,
    sinc_width: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """1D Bragg-like selectivity weights for relative channel offset.

    Approximates volume-hologram angular selectivity as a squared-sinc
    profile over discrete channel index lag. Returns shape (n_channels,)
    centered so lag 0 is peak transmission. Not a full volume-grating
    simulation — a calibrated curve can replace this later.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    if sinc_width <= 0:
        raise ValueError("sinc_width must be positive")
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    lags = torch.arange(n_channels, device=device, dtype=dtype) - (n_channels // 2)
    x = lags / sinc_width
    w = torch.sinc(x) ** 2
    w = w / w.sum().clamp_min(1e-12)
    return w


def apply_bragg_crosstalk(
    scores: torch.Tensor,
    sinc_width: float = 1.5,
    strength: float = 1.0,
) -> torch.Tensor:
    """Mix key-axis channels with a Bragg-shaped kernel.

    scores: (..., query, key). strength in [0, 1] blends ideal vs mixed.
    """
    if strength <= 0:
        return scores
    if strength > 1:
        raise ValueError("strength must be in [0, 1]")
    n = scores.shape[-1]
    kernel = bragg_selectivity_kernel(
        n, sinc_width=sinc_width, device=scores.device, dtype=scores.dtype
    )
    flat = scores.reshape(-1, 1, n)
    k = kernel.view(1, 1, n)
    pad = n // 2
    flat_pad = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    mixed = torch.nn.functional.conv1d(flat_pad, k)
    mixed = mixed[..., :n]
    mixed = mixed.reshape(scores.shape)
    return (1.0 - strength) * scores + strength * mixed
