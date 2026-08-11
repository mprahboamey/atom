"""Unified hybrid optical-score model: GGUF or Hugging Face safetensors.

Same internal weight layout and forward path for both sources. Plug in a
GGUF file today; later point at a safetensors folder without changing the
inference API.

Attention scores use the optical path. Embed, norms, MLP, RoPE, V, out,
lm_head stay digital on checkpoint weights.

Layer streaming (stream_layers=True) keeps only one transformer layer in
memory at a time so a full 32-layer 7B run is more likely to fit on a
machine that OOMs when all layers are resident.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .attention import optical_scores, optical_scores_general
from .noise import NoiseConfig


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
    source: str = "unknown"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
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
        theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim)
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


def _to_f16(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float16:
        return t.contiguous()
    return t.detach().to(torch.float16).contiguous()


# ---- GGUF ----

def _gguf_dequant(tensor) -> torch.Tensor:
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
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float16)


def _gguf_layer_keys(i: int) -> list[str]:
    return [
        f"blk.{i}.attn_norm.weight",
        f"blk.{i}.ffn_norm.weight",
        f"blk.{i}.attn_q.weight",
        f"blk.{i}.attn_k.weight",
        f"blk.{i}.attn_v.weight",
        f"blk.{i}.attn_output.weight",
        f"blk.{i}.ffn_gate.weight",
        f"blk.{i}.ffn_up.weight",
        f"blk.{i}.ffn_down.weight",
    ]


def _load_gguf_bundle(
    path: Path,
    max_layers: int | None,
    stream_layers: bool,
) -> tuple[dict, ModelConfig, Any]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    name_to_tensor = {t.name: t for t in reader.tensors}
    n_total = 0
    while f"blk.{n_total}.attn_q.weight" in name_to_tensor:
        n_total += 1
    n_layers = n_total if max_layers is None else min(n_total, max_layers)

    # Always load embed + head; layers either all or on demand
    shared_names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    weights: dict = {}
    for name in shared_names:
        weights[name] = _gguf_dequant(name_to_tensor[name])

    if not stream_layers:
        for i in range(n_layers):
            for name in _gguf_layer_keys(i):
                weights[name] = _gguf_dequant(name_to_tensor[name])

    # Config from shapes (need layer 0 q/k/gate — load temporarily if streaming)
    if f"blk.0.attn_q.weight" in weights:
        q0 = weights["blk.0.attn_q.weight"]
        k0 = weights["blk.0.attn_k.weight"]
        g0 = weights["blk.0.ffn_gate.weight"]
    else:
        q0 = _gguf_dequant(name_to_tensor["blk.0.attn_q.weight"])
        k0 = _gguf_dequant(name_to_tensor["blk.0.attn_k.weight"])
        g0 = _gguf_dequant(name_to_tensor["blk.0.ffn_gate.weight"])

    emb = weights["token_embd.weight"]
    head_dim = 128
    cfg = ModelConfig(
        n_layers=n_layers,
        n_heads=q0.shape[0] // head_dim,
        n_kv_heads=k0.shape[0] // head_dim,
        head_dim=head_dim,
        hidden_size=emb.shape[1],
        intermediate_size=g0.shape[0],
        vocab_size=emb.shape[0],
        source="gguf",
    )
    backend = {"kind": "gguf", "name_to_tensor": name_to_tensor if stream_layers else None}
    return weights, cfg, backend


# ---- Hugging Face safetensors ----

_HF_TO_INTERNAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def _hf_layer_map(i: int) -> dict[str, str]:
    return {
        f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
    }


def _load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError("pip install safetensors") from e
    return load_file(str(path))


def _load_hf_bundle(
    path: Path,
    max_layers: int | None,
    stream_layers: bool,
) -> tuple[dict, ModelConfig, Any]:
    path = Path(path)
    if path.is_file() and path.suffix == ".safetensors":
        files = [path]
        root = path.parent
    else:
        root = path
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors under {path}")

    # Merge shards into a key index: key -> (file, tensor_name) without loading all
    # Simplest reliable path: load only needed keys from each file
    all_keys: dict[str, Path] = {}
    for f in files:
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise ImportError("pip install safetensors") from e
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                all_keys[k] = f

    def get_tensor(key: str) -> torch.Tensor:
        f = all_keys[key]
        from safetensors import safe_open

        with safe_open(str(f), framework="pt", device="cpu") as sf:
            return _to_f16(sf.get_tensor(key))

    # Count layers
    n_total = 0
    while f"model.layers.{n_total}.self_attn.q_proj.weight" in all_keys:
        n_total += 1
    n_layers = n_total if max_layers is None else min(n_total, max_layers)

    weights: dict = {}
    for hf, internal in _HF_TO_INTERNAL.items():
        if hf not in all_keys:
            # some models tie lm_head to embed
            if hf == "lm_head.weight" and "model.embed_tokens.weight" in all_keys:
                weights[internal] = get_tensor("model.embed_tokens.weight")
                continue
            raise KeyError(f"Missing {hf} in {path}")
        weights[internal] = get_tensor(hf)

    if not stream_layers:
        for i in range(n_layers):
            for hf, internal in _hf_layer_map(i).items():
                weights[internal] = get_tensor(hf)

    q0 = (
        weights["blk.0.attn_q.weight"]
        if "blk.0.attn_q.weight" in weights
        else get_tensor("model.layers.0.self_attn.q_proj.weight")
    )
    k0 = (
        weights["blk.0.attn_k.weight"]
        if "blk.0.attn_k.weight" in weights
        else get_tensor("model.layers.0.self_attn.k_proj.weight")
    )
    g0 = (
        weights["blk.0.ffn_gate.weight"]
        if "blk.0.ffn_gate.weight" in weights
        else get_tensor("model.layers.0.mlp.gate_proj.weight")
    )
    emb = weights["token_embd.weight"]
    head_dim = 128
    cfg = ModelConfig(
        n_layers=n_layers,
        n_heads=q0.shape[0] // head_dim,
        n_kv_heads=k0.shape[0] // head_dim,
        head_dim=head_dim,
        hidden_size=emb.shape[1],
        intermediate_size=g0.shape[0],
        vocab_size=emb.shape[0],
        source="safetensors",
    )

    # optional config.json rope / eps
    cfg_path = root / "config.json"
    if cfg_path.exists():
        meta = json.loads(cfg_path.read_text())
        cfg.rms_eps = float(meta.get("rms_norm_eps", cfg.rms_eps))
        cfg.rope_theta = float(meta.get("rope_theta", cfg.rope_theta))

    backend = {
        "kind": "safetensors",
        "get_tensor": get_tensor if stream_layers else None,
        "hf_layer_map": _hf_layer_map if stream_layers else None,
    }
    return weights, cfg, backend


class HybridTransformer(torch.nn.Module):
    """Hybrid optical-score transformer from GGUF or safetensors."""

    def __init__(
        self,
        weights: dict,
        cfg: ModelConfig,
        backend: dict | None = None,
        noise: NoiseConfig | None = None,
        stream_layers: bool = False,
    ):
        super().__init__()
        self.weights = weights
        self.cfg = cfg
        self.backend = backend or {}
        self.noise = noise
        self.stream_layers = stream_layers

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        max_layers: int | None = None,
        stream_layers: bool = False,
        noise: NoiseConfig | None = None,
    ) -> "HybridTransformer":
        path = Path(path)
        if path.is_file() and path.suffix.lower() == ".gguf":
            weights, cfg, backend = _load_gguf_bundle(path, max_layers, stream_layers)
        elif path.is_dir() or (path.is_file() and path.suffix == ".safetensors"):
            weights, cfg, backend = _load_hf_bundle(path, max_layers, stream_layers)
        else:
            raise ValueError(
                f"Unsupported checkpoint {path}. Use a .gguf file or a HF safetensors directory."
            )
        return cls(weights, cfg, backend=backend, noise=noise, stream_layers=stream_layers)

    def _ensure_layer(self, layer: int) -> None:
        key = f"blk.{layer}.attn_q.weight"
        if key in self.weights:
            return
        kind = self.backend.get("kind")
        if kind == "gguf":
            name_to_tensor = self.backend["name_to_tensor"]
            for name in _gguf_layer_keys(layer):
                self.weights[name] = _gguf_dequant(name_to_tensor[name])
        elif kind == "safetensors":
            get_tensor = self.backend["get_tensor"]
            for hf, internal in self.backend["hf_layer_map"](layer).items():
                self.weights[internal] = get_tensor(hf)
        else:
            raise RuntimeError("stream_layers set but no backend for on-demand load")

    def _release_layer(self, layer: int) -> None:
        if not self.stream_layers:
            return
        for name in _gguf_layer_keys(layer):
            self.weights.pop(name, None)
        gc.collect()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weights["token_embd.weight"])

    def layer_forward(
        self,
        x: torch.Tensor,
        layer: int,
        positions: torch.Tensor,
        use_optical: bool = True,
    ) -> torch.Tensor:
        self._ensure_layer(layer)
        cfg = self.cfg
        w = self.weights
        h = _rms_norm(x, w[f"blk.{layer}.attn_norm.weight"], cfg.rms_eps)

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
        x = x + h
        self._release_layer(layer)
        return x

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


# Back-compat alias
HybridMistralFromGGUF = HybridTransformer
