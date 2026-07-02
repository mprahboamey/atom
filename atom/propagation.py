"""Wave-field helpers and Angular Spectrum Method propagation."""

from __future__ import annotations

import math

import torch
from torch import nn


def complex_field(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Build a complex field from amplitude and phase tensors."""
    return amplitude * torch.exp(1j * phase)


def intensity(field: torch.Tensor) -> torch.Tensor:
    """Detector intensity: squared complex magnitude."""
    return field.abs().square()


def gaussian_field(
    size: int,
    pixel_size: float,
    waist: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a centered zero-phase Gaussian beam."""
    coords = (torch.arange(size, device=device) - size // 2) * pixel_size
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    amplitude = torch.exp(-(x.square() + y.square()) / (2 * waist**2))
    phase = torch.zeros_like(amplitude)
    return complex_field(amplitude, phase)


class WavePropagator(nn.Module):
    """Propagate a 2D or batched complex field with the Angular Spectrum Method."""

    def __init__(self, wavelength: float = 405e-9, pixel_size: float = 1e-6):
        super().__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size

    def forward(self, field: torch.Tensor, distance: float) -> torch.Tensor:
        return self.propagate(field, distance)

    def propagate(self, field: torch.Tensor, distance: float) -> torch.Tensor:
        if not torch.is_complex(field):
            raise TypeError("field must be a complex tensor")

        h, w = field.shape[-2:]
        fy = torch.fft.fftfreq(h, d=self.pixel_size, device=field.device)
        fx = torch.fft.fftfreq(w, d=self.pixel_size, device=field.device)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

        k = 2 * math.pi / self.wavelength
        kx = 2 * math.pi * fx_grid
        ky = 2 * math.pi * fy_grid
        kz_squared = k**2 - kx.square() - ky.square()
        propagating = kz_squared >= 0

        kz = torch.sqrt(torch.clamp(kz_squared, min=0.0))
        transfer = torch.zeros_like(kz, dtype=field.dtype)
        transfer[propagating] = torch.exp(1j * kz[propagating] * distance)

        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        return torch.fft.ifft2(spectrum * transfer, dim=(-2, -1))
