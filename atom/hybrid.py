"""Hybrid optical-QK + digital remainder attention.

Scores are computed by wave interference (optical_scores_general).
Softmax, value aggregation, residual, and LayerNorm stay digital.
This is the practical composition path: the part that is algebraically
identical to scaled dot-product attention runs optically; everything
else uses ordinary PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import optical_scores_general
from .noise import NoiseConfig


@dataclass
class AccountingReport:
    """First-order energy / latency breakdown.

    Optical numbers are pure time-of-flight from geometry.
    Digital numbers are rough FLOP counts for the remainder.
    Neither is a hardware measurement.
    """

    optical_tof_s: float
    digital_flops: float
    seq_len: int
    dim: int
    notes: str = (
        "optical_tof_s is L*n/c for the stated path length; "
        "digital_flops counts softmax + matmul + residual/norm only; "
        "projections and periphery (SLM, detector, ADC) are not included."
    )


def optical_time_of_flight(path_cm: float = 1.0, refractive_index: float = 1.5) -> float:
    """Propagation delay through the medium (seconds)."""
    c = 2.998e8  # m/s
    return (path_cm * 1e-2) * refractive_index / c


def estimate_digital_flops(seq_len: int, dim: int, with_norm: bool = True) -> float:
    """Rough FLOP count for softmax + V aggregation + residual (+ LayerNorm).

    Softmax ~ 5 * S^2, attn @ V ~ 2 * S^2 * D, residual ~ S * D,
    LayerNorm ~ 5 * S * D. Order-of-magnitude only.
    """
    s, d = float(seq_len), float(dim)
    flops = 5.0 * s * s + 2.0 * s * s * d + s * d
    if with_norm:
        flops += 5.0 * s * d
    return flops


class HybridOpticalAttention(nn.Module):
    """Attention block with optical scores and digital remainder.

    Forward path:
      x -> Q, K, V (digital linears)
        -> optical_scores_general(Q, K, positions, noise=...)
        -> softmax -> (weights @ V) -> output projection
        -> optional residual + LayerNorm
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        path_cm: float = 1.0,
        use_layer_norm: bool = True,
        noise: NoiseConfig | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.path_cm = path_cm
        self.noise = noise

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim) if use_layer_norm else None

    def _reshape_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, S, D) -> (B, H, S, head_dim)
        b, s, _ = t.shape
        return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, H, S, head_dim) -> (B, S, D)
        b, h, s, d = t.shape
        return t.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, dim)
            positions: optional (batch, seq) or (seq,) angular positions
                       for continuous-phase encoding. None -> binary phase.

        Returns:
            output: (batch, seq, dim)
            weights: (batch, num_heads, seq, seq)
        """
        b, s, d = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        # Scores per head: optical path
        # optical_scores_general expects (..., seq, features)
        scores_list = []
        for h in range(self.num_heads):
            qh, kh = q[:, h], k[:, h]
            pos_q = positions
            pos_k = positions
            if positions is not None and positions.ndim == 1:
                pos_q = positions.unsqueeze(0).expand(b, -1)
                pos_k = pos_q
            elif positions is not None and positions.ndim == 2:
                pos_q = positions
                pos_k = positions

            sh = optical_scores_general(
                qh, kh,
                query_positions=pos_q,
                key_positions=pos_k,
                noise=self.noise,
            )
            scores_list.append(sh)
        scores = torch.stack(scores_list, dim=1)  # (B, H, S, S)

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v  # (B, H, S, head_dim)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        if self.norm is not None:
            out = self.norm(x + out)
        else:
            out = x + out

        return out, weights

    def accounting(self, seq_len: int) -> AccountingReport:
        """First-order optical ToF + digital FLOP estimate for one forward."""
        return AccountingReport(
            optical_tof_s=optical_time_of_flight(self.path_cm),
            digital_flops=estimate_digital_flops(
                seq_len, self.dim, with_norm=self.norm is not None
            ),
            seq_len=seq_len,
            dim=self.dim,
        )
