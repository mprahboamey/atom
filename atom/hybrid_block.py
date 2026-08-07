"""Hybrid attention block loaded from converted optical weight tensors.

Reconstructs q/k/v/o from amplitude+phase, runs digital projections, optical
scores, digital softmax/V/out/residual. Supports GQA (e.g. Mistral 7B).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import optical_scores, optical_scores_general
from .noise import NoiseConfig


def reconstruct_weight(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    sign = torch.where(
        phase.abs() < (math.pi / 2), torch.ones_like(phase), -torch.ones_like(phase)
    )
    return amplitude * sign


def _safe_key(hf_name: str) -> str:
    return hf_name.replace("/", "__").replace(".", "__")


def load_reconstructed_weight(payload: dict, hf_name: str) -> torch.Tensor:
    base = _safe_key(hf_name)
    return reconstruct_weight(payload[f"{base}.amplitude"], payload[f"{base}.phase"])


class LoadedHybridAttention(nn.Module):
    """One attention layer: weights from optical_weights.pt, scores optical."""

    def __init__(
        self,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        noise: NoiseConfig | None = None,
    ):
        super().__init__()
        self.w_q = nn.Parameter(w_q, requires_grad=False)
        self.w_k = nn.Parameter(w_k, requires_grad=False)
        self.w_v = nn.Parameter(w_v, requires_grad=False)
        self.w_o = nn.Parameter(w_o, requires_grad=False)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = w_q.shape[0] // num_heads
        self.noise = noise
        if w_q.shape[0] % num_heads != 0:
            raise ValueError("q out features must divide num_heads")

    @classmethod
    def from_optical_dir(
        cls,
        weights_dir: str | Path,
        layer: int,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        noise: NoiseConfig | None = None,
    ) -> "LoadedHybridAttention":
        path = Path(weights_dir) / "optical_weights.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        prefix = f"model.layers.{layer}.self_attn"
        w_q = load_reconstructed_weight(payload, f"{prefix}.q_proj.weight")
        w_k = load_reconstructed_weight(payload, f"{prefix}.k_proj.weight")
        w_v = load_reconstructed_weight(payload, f"{prefix}.v_proj.weight")
        w_o = load_reconstructed_weight(payload, f"{prefix}.o_proj.weight")
        return cls(w_q, w_k, w_v, w_o, num_heads=num_heads, num_kv_heads=num_kv_heads, noise=noise)

    def _shape_q(self, t: torch.Tensor) -> torch.Tensor:
        b, s, _ = t.shape
        return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, t: torch.Tensor) -> torch.Tensor:
        b, s, _ = t.shape
        return t.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        use_optical: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, S, D_in); W: (out, in)
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T
        q = self._shape_q(q)
        k = self._shape_kv(k)
        v = self._shape_kv(v)

        # GQA: repeat kv heads to match q heads
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = math.sqrt(self.head_dim)
        scores_list = []
        for h in range(self.num_heads):
            qh, kh = q[:, h], k[:, h]
            if use_optical:
                if self.noise is None:
                    sh = optical_scores(qh, kh)
                else:
                    sh = optical_scores_general(qh, kh, noise=self.noise)
            else:
                sh = (qh @ kh.transpose(-2, -1)) / scale
            scores_list.append(sh)
        scores = torch.stack(scores_list, dim=1)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        b, h, s, d = out.shape
        out = out.transpose(1, 2).contiguous().view(b, s, h * d)
        out = out @ self.w_o.T
        return out, weights


def digital_block_reference(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 8,
) -> torch.Tensor:
    """Pure digital attention with the same weights (no residual/norm)."""
    layer = LoadedHybridAttention(
        w_q, w_k, w_v, w_o, num_heads=num_heads, num_kv_heads=num_kv_heads, noise=None
    )
    out, _ = layer(x, use_optical=False)
    return out
