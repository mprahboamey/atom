"""Load a Llama/Mistral-style GGUF and run hybrid optical-score inference.

Attention QK scores use the optical interference path. Embeddings, norms,
MLP, RoPE, softmax, V, output projection, and lm_head use weights from the
same GGUF (dequantized). Software hybrid only — not a physical device.

Requires: pip install gguf

Memory: only tensors for the selected layers are dequantized. Weights are
kept in float16. A 2-layer slice should fit in a few GB; full 32-layer 7B
still needs a large machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .attention import optical_scores, optical_scores_general
from .noise import NoiseConfig


def _dequant_tensor(tensor) -> torch.Tensor:
    import gguf
    from gguf import GGMLQuantizationType

    data = tensor.data
    ttype = tensor.tensor_type
    try:
        if ttype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            arr = np.array(data, dtype=np.float32, copy=True)
        else:
            arr = np.array(gguf.dequantize(data, ttype), dtype=np.float32, copy=True)
    except Exception:
        arr = np.array(gguf.dequantize(data, ttype), dtype=np.float32, copy=True)

    shape = tuple(int(s) for s in reversed(list(tensor.shape)))
    if arr.size == int(np.prod(shape)):
        arr = arr.reshape(shape)
    # float16 cuts RAM roughly in half vs float32 weights
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float16)


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
    # compute in float32 for stability, return in x dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return (weight.float() * x_f).to(dtype=x.dtype)


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
    dim = q.shape[-1]
    freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim
        )
    )
    t = positions.float() if positions.ndim == 1 else positions[0].float()
    angles = torch.outer(t, freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.stack((cos, cos), dim=-1).flatten(-2).to(dtype=q.dtype)
    sin = torch.stack((sin, sin), dim=-1).flatten(-2).to(dtype=q.dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


def _needed_tensor_names(n_layers: int) -> set[str]:
    names = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
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
            names.add(f"blk.{i}.{key}")
    return names


class HybridMistralFromGGUF(torch.nn.Module):
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
        # Count layers from names without dequantizing everything
        n_layers_total = 0
        name_to_tensor = {t.name: t for t in reader.tensors}
        while f"blk.{n_layers_total}.attn_q.weight" in name_to_tensor:
            n_layers_total += 1
        n_layers = n_layers_total
        if max_layers is not None:
            n_layers = min(n_layers_total, max_layers)

        needed = _needed_tensor_names(n_layers)
        # Always need blk.0 shapes for config even if max_layers is 0 (should not happen)
        print(f"dequantizing {len(needed)} tensors for {n_layers} layers (of {n_layers_total})...")

        raw = {}
        for name in sorted(needed):
            if name not in name_to_tensor:
                raise KeyError(f"missing tensor in GGUF: {name}")
            raw[name] = _dequant_tensor(name_to_tensor[name])
            print(f"  loaded {name} {tuple(raw[name].shape)}")

        emb = raw["token_embd.weight"]
        vocab, hidden = emb.shape[0], emb.shape[1]
        q_out = raw["blk.0.attn_q.weight"].shape[0]
        k_out = raw["blk.0.attn_k.weight"].shape[0]
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
        return cls(raw, cfg, noise=noise)

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

        # matmul in float32 for numerical headroom, results back to x dtype
        q = (h.float() @ w[f"blk.{layer}.attn_q.weight"].float().T).to(dtype=x.dtype)
        k = (h.float() @ w[f"blk.{layer}.attn_k.weight"].float().T).to(dtype=x.dtype)
        v = (h.float() @ w[f"blk.{layer}.attn_v.weight"].float().T).to(dtype=x.dtype)

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
            qh, kh = q[:, hi].float(), k[:, hi].float()
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
        out = attn @ v.float()
        out = out.transpose(1, 2).contiguous().view(b, s, cfg.n_heads * cfg.head_dim)
        out = (out @ w[f"blk.{layer}.attn_output.weight"].float().T).to(dtype=x.dtype)
        x = x + out

        h = _rms_norm(x, w[f"blk.{layer}.ffn_norm.weight"], cfg.rms_eps)
        gate = h.float() @ w[f"blk.{layer}.ffn_gate.weight"].float().T
        up = h.float() @ w[f"blk.{layer}.ffn_up.weight"].float().T
        h = F.silu(gate) * up
        h = (h @ w[f"blk.{layer}.ffn_down.weight"].float().T).to(dtype=x.dtype)
        return x + h

    def forward(
        self,
        input_ids: torch.Tensor,
        use_optical: bool = True,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        s = input_ids.shape[1]
        positions = torch.arange(s, device=input_ids.device)
        for layer in range(self.cfg.n_layers):
            x = self.layer_forward(x, layer, positions, use_optical=use_optical)
        x = _rms_norm(x, self.weights["output_norm.weight"], self.cfg.rms_eps)
        logits = x.float() @ self.weights["output.weight"].float().T
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
