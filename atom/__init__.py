"""Core simulation tools for diffractive optical networks."""

from .propagation import WavePropagator, gaussian_field, intensity
from .diffractive import DiffractiveLayer, DiffractiveNetwork
from .noise import (
    quantize_phase,
    add_phase_noise,
    add_angular_jitter,
    apply_crosstalk,
    NoiseConfig,
)
from .attention import (
    optical_scores,
    optical_scores_general,
    encode_signed_values,
    encode_angular_phase,
    OpticalSelfAttention,
)
from .capacity import (
    CapacityParams,
    max_usable_channels,
    geometric_capacity,
    usable_capacity,
    capacity_summary,
)
from .hybrid import (
    HybridOpticalAttention,
    AccountingReport,
    optical_time_of_flight,
    estimate_digital_flops,
)

__all__ = [
    "WavePropagator",
    "gaussian_field",
    "intensity",
    "DiffractiveLayer",
    "DiffractiveNetwork",
    "quantize_phase",
    "add_phase_noise",
    "add_angular_jitter",
    "apply_crosstalk",
    "NoiseConfig",
    "optical_scores",
    "optical_scores_general",
    "encode_signed_values",
    "encode_angular_phase",
    "OpticalSelfAttention",
    "CapacityParams",
    "max_usable_channels",
    "geometric_capacity",
    "usable_capacity",
    "capacity_summary",
    "HybridOpticalAttention",
    "AccountingReport",
    "optical_time_of_flight",
    "estimate_digital_flops",
]
