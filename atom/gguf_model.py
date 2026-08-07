"""Load a full Llama/Mistral-style GGUF and run hybrid optical-score inference.

Attention QK scores use the optical interference path. Embeddings, norms,
MLP, RoPE, softmax, V, output projection, and lm_head use weights loaded
from the same GGUF file (dequantized). This is a software hybrid, not a
physical optical device.

Requires: pip install gguf
Memory: a 7B Q4 model dequantized to float32 needs on the order of 10+ GB RAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import optical_scores, optical_scores_general
from .noise import NoiseConfig


def _dequant_tensor(tensor) -> torch.Tensor:
    import numpy as np
    import gguf
    from gguf import GGMLQuantizationType

    data = tensor.data
    ttype = tensor.tensor_type
    try:
        if ttype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            arr = np.asarray(data).astype(np.float32, copy=False)
        else:
            arr = np.asarray(gguf.dequantize(data, ttype), dtype=np.float32)
    except Exception:
        arr = np.asarray(gguf.dequantize(data, ttype), dtype=np.float32)
    shape = tuple(int(s) for s in reversed(list(tensor.shape)))
    if arr.size == int(np.prod(shape)):
        arr = arr.reshape(shape)
    return torch.from_numpy(np.ascontiguousarray(arr))


@dataclass
class ModelConfig:
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 14336
    vocab_size: int = 32000
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: (B, H, S, D)
    dim = q.shape[-1]
    half = dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device, dtype=q.dtype) / dim))
    # positions: (S,) or (B, S)
    if positions.ndim == 1:
        t = positions.float()
    else:
        t = positions[0].float()
    angles = torch.outer(t, freq)  # (S, half)
    cos = torch.cos(angles).to(q.dtype)
    sin = torch.sin(angles).to(q.dtype)
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)  # (S, D)
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class HybridMistralFromGGUF(nn.Module):
    """Mistral/Llama-style model: optical attention scores, digital remainder."""

    def __init__(self, weights: dict, cfg: ModelConfig, noise: NoiseConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.noise = noise
        self.weights = weights

    @classmethod
    def from_gguf(
        cls,
        path: str | Path,
        noise: NoiseConfig | None = None,
        max_layers: int | None = None,
    ) -> "HybridMistralFromGGUF":
        try:
            import gguf
        except ImportError as e:
            raise ImportError("pip install gguf") from e

        reader = gguf.GGUFReader(str(path))
        raw = {}
        for t in reader.tensors:
            raw[t.name] = _dequant_tensor(t)

        # Infer config from tensor shapes
        emb = raw["token_embd.weight"]
        vocab, hidden = emb.shape[0], emb.shape[1]
        n_layers = 0
        while f"blk.{n_layers}.attn_q.weight" in raw:
            n_layers += 1
        if max_layers is not None:
            n_layers = min(n_layers, max_layers)

        q_out = raw["blk.0.attn_q.weight"].shape[0]
        k_out = raw["blk.0.attn_k.weight"].shape[0]
        head_dim = 128
        if q_out % 128 == 0:
            head_dim = 128
        n_heads = q_out // head_dim
        n_kv_heads = k_out // head_dim
        inter = raw["blk.0.ffn_gate.weight"].shape[0]

        cfg = ModelConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
            intermediate_size=inter,
            vocab_size=vocab,
        )
        # Keep only needed tensors for selected layers
        keep = {
            "token_embd.weight": raw["token_embd.weight"],
            "output_norm.weight": raw["output_norm.weight"],
            "output.weight": raw["output.weight"],
        }
        for i in range(n_layers):
            for key in (
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ):
                name = f"blk.{i}.{key}"
                if name in raw:
                    keep[name] = raw[name]
        return cls(keep, cfg, noise=noise)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weights["token_embd.weight"])

    def layer_forward(
        self,
        x: torch.Tensor,
        layer: int,
        positions: torch.Tensor,
        use_optical: bool = True,
    ) -> torch.Tensor:
        cfg = self.cfg
        w = self.weights
        h = _rms_norm(x, w[f"blk.{layer}.attn_norm.weight"], cfg.rms_eps)

        q = h @ w[f"blk.{layer}.attn_q.weight"].T
        k = h @ w[f"blk.{layer}.attn_k.weight"].T
        v = h @ w[f"blk.{layer}.attn_v.weight"].T

        b, s, _ = q.shape
        q = q.view(b, s, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, positions, cfg.rope_theta)

        if cfg.n_kv_heads != cfg.n_heads:
            rep = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = math.sqrt(cfg.head_dim)
        scores_h = []
        for hi in range(cfg.n_heads):
            qh, kh = q[:, hi], k[:, hi]
            if use_optical:
                if self.noise is None:
                    sh = optical_scores(qh, kh)
                else:
                    sh = optical_scores_general(qh, kh, noise=self.noise)
            else:
                sh = (qh @ kh.transpose(-2, -1)) / scale
            scores_h.append(sh)
        scores = torch.stack(scores_h, dim=1)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, s, cfg.n_heads * cfg.head_dim)
        out = out @ w[f"blk.{layer}.attn_output.weight"].T
        x = x + out

        h = _rms_norm(x, w[f"blk.{layer}.ffn_norm.weight"], cfg.rms_eps)
        gate = h @ w[f"blk.{layer}.ffn_gate.weight"].T
        up = h @ w[f"blk.{layer}.ffn_up.weight"].T
        h = F.silu(gate) * up
        h = h @ w[f"blk.{layer}.ffn_down.weight"].T
        return x + h

    def forward(
        self,
        input_ids: torch.Tensor,
        use_optical: bool = True,
    ) -> torch.Tensor:
        """Return logits (B, S, vocab)."""
        x = self.embed(input_ids)
        s = input_ids.shape[1]
        positions = torch.arange(s, device=input_ids.device)
        for layer in range(self.cfg.n_layers):
            x = self.layer_forward(x, layer, positions, use_optical=use_optical)
        x = _rms_norm(x, self.weights["output_norm.weight"], self.cfg.rms_eps)
        logits = x @ self.weights["output.weight"].T
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 16,
        use_optical: bool = True,
        temperature: float = 0.0,
    ) -> list[int]:
        ids = list(input_ids)
        for _ in range(max_new_tokens):
            t = torch.tensor([ids], dtype=torch.long)
            logits = self.forward(t, use_optical=use_optical)
            next_logits = logits[0, -1]
            if temperature and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(next_logits.argmax().item())
            ids.append(next_id)
        return ids
