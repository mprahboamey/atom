"""Core simulation tools for diffractive optical networks."""

from .propagation import WavePropagator, gaussian_field, intensity
from .diffractive import DiffractiveLayer, DiffractiveNetwork
from .attention import optical_scores, OpticalSelfAttention

__all__ = [
    "WavePropagator",
    "gaussian_field",
    "intensity",
    "DiffractiveLayer",
    "DiffractiveNetwork",
    "optical_scores",
    "OpticalSelfAttention",
]
