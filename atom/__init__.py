"""Core simulation tools for diffractive optical networks."""

from .propagation import WavePropagator, gaussian_field, intensity
from .diffractive import DiffractiveLayer, DiffractiveNetwork
from .attention import (
    optical_scores,
    optical_scores_general,
    encode_signed_values,
    encode_angular_phase,
    OpticalSelfAttention,
)

__all__ = [
    "WavePropagator",
    "gaussian_field",
    "intensity",
    "DiffractiveLayer",
    "DiffractiveNetwork",
    "optical_scores",
    "optical_scores_general",
    "encode_signed_values",
    "encode_angular_phase",
    "OpticalSelfAttention",
]
