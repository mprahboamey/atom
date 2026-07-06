"""Noise modelling: physical constraints the pure-math simulator ignores.

Starts with phase quantization -- CONTRIBUTING.md flags this as the most
tractable open noise problem, and the natural next question after proving
the continuous-phase math works in the ideal case (see attention.py):
a real crystal can't write an infinite-precision phase angle. It writes
theta to some finite number of bits. This module answers "how many bits
before that actually matters."
"""

from __future__ import annotations

import math

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
    # Rounding near the top of the circle can land exactly on 2*pi, which
    # is the same physical angle as 0 but a distinct float value -- wrap
    # again so that boundary collapses to the single level it actually is.
    return torch.remainder(quantized, 2 * math.pi)
