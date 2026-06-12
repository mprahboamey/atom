"""Diffractive layers and compact optical network blocks."""

from __future__ import annotations

import math

import torch
from torch import nn

from .propagation import WavePropagator


class DiffractiveLayer(nn.Module):
    """A trainable phase-only optical layer."""

    def __init__(self, height: int, width: int, init_phase: torch.Tensor | None = None):
        super().__init__()
        if init_phase is None:
            init_phase = torch.rand(height, width) * (2 * math.pi)
        if init_phase.shape != (height, width):
            raise ValueError("init_phase must match (height, width)")
        self.phase = nn.Parameter(init_phase)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(self.phase, 2 * math.pi)
        return field * torch.exp(1j * phase)

    def phase_mask(self) -> torch.Tensor:
        return torch.remainder(self.phase.detach(), 2 * math.pi)


class DiffractiveNetwork(nn.Module):
    """A small stack of phase masks with propagation between masks."""

    def __init__(
        self,
        shape: tuple[int, int],
        num_layers: int,
        distance: float,
        wavelength: float = 405e-9,
        pixel_size: float = 1e-6,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        h, w = shape
        self.layers = nn.ModuleList(DiffractiveLayer(h, w) for _ in range(num_layers))
        self.propagator = WavePropagator(wavelength=wavelength, pixel_size=pixel_size)
        self.distance = distance

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        x = field
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = self.propagator(x, self.distance)
        return x
