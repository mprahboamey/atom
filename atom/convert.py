"""Convert digital attention weights into optical phase encodings.

Loads a local checkpoint and maps attention projection matrices into the
amplitude/phase representation used by the optical score path.

Supported inputs (same convert_checkpoint entry point):
  - Hugging Face / safetensors directory or file
  - PyTorch .pt / .bin state dict
  - GGUF file (llama.cpp), including quantized types such as Q4_K_M

Default phase quantisation is 8 bits. This module does not download models.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .noise import quantize_phase

# HF / GPT-2 / Llama-style names
_Q_PATTERNS = re.compile(
    r"(?:^|[.])(?:q_proj|query|q_lin|to_q)$", re.IGNORECASE
)
_K_PATTERNS = re.compile(
    r"(?:^|[.])(?:k_proj|key|k_lin|to_k)$", re.IGNORECASE
)
_V_PATTERNS = re.compile(
    r"(?:^|[.])(?:v_proj|value|v_lin|to_v)$", re.IGNORECASE
)
_O_PATTERNS = re.compile(
    r"(?:^|[.])(?:o_proj|out_proj|c_proj|to_out\.0)$", re.IGNORECASE
)

# llama.cpp GGUF tensor names -> synthetic HF-style keys
_GGUF_ATTN = re.compile(
    r"^blk\.(\d+)\.attn_(q|k|v|output)\.weight$"
)
_GGUF_KIND = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "output": "o_proj"}


@dataclass
class OpticalWeightTensor:
    """One weight matrix encoded for the optical path."""

    name: str
    amplitude: torch.Tensor
    phase: torch.Tensor
    shape: tuple
    phase_bits: int | None

    def to_complex(self) -> torch.Tensor:
        return self.amplitude * torch.exp(1j * self.phase)


@dataclass
class ConversionResult:
    weights: dict
    meta: dict


def encode_weight_matrix(
    weight: torch.Tensor,
    phase_bits: int | None = 8,
) -> tuple:
    """Map a real weight matrix to amplitude + phase.

    Sign is stored as binary phase (0 / pi). If phase_bits is set, phase is
    quantised to that many levels around the circle (default 8).
    """
    w = weight.detach().float().cpu()
    amplitude = w.abs()
    phase = torch.where(w >= 0, torch.zeros_like(w), torch.full_like(w, math.pi))
    if phase_bits is not None:
        phase = quantize_phase(phase, phase_bits)
    return amplitude, phase


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _gguf_tensor_to_torch(tensor) -> torch.Tensor:
    """Dequantize a GGUF ReaderTensor to float32 torch.Tensor."""
    import numpy as np
    import gguf

    data = tensor.data
    ttype = tensor.tensor_type
    # F32 / F16 often already usable; other types need dequantize
    try:
        from gguf import GGMLQuantizationType

        if ttype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            arr = np.asarray(data).astype(np.float32, copy=False)
        else:
            arr = np.asarray(gguf.dequantize(data, ttype), dtype=np.float32)
    except Exception:
        arr = np.asarray(gguf.dequantize(data, ttype), dtype=np.float32)

    # GGUF stores shapes in reverse order relative to PyTorch convention
    shape = tuple(int(s) for s in reversed(list(tensor.shape)))
    if arr.size != int(np.prod(shape)):
        # Fall back to flat dequant size if reshape is ambiguous
        return torch.from_numpy(np.ascontiguousarray(arr))
    return torch.from_numpy(np.ascontiguousarray(arr.reshape(shape)))


def _load_gguf(path: Path) -> dict:
    """Load attention weights from a GGUF file into a state-dict-like mapping.

    Requires: pip install gguf
    Quantized tensors (e.g. Q4_K_M) are dequantized to float32.
    """
    try:
        import gguf
    except ImportError as e:
        raise ImportError(
            "Reading GGUF requires the gguf package. Install with: pip install gguf"
        ) from e

    reader = gguf.GGUFReader(str(path))
    out = {}
    for tensor in reader.tensors:
        name = tensor.name
        m = _GGUF_ATTN.match(name)
        if not m:
            continue
        layer_idx, kind = m.group(1), m.group(2)
        hf_name = (
            f"model.layers.{layer_idx}.self_attn.{_GGUF_KIND[kind]}.weight"
        )
        out[hf_name] = _gguf_tensor_to_torch(tensor)
    if not out:
        raise ValueError(
            f"No attention tensors found in GGUF file {path}. "
            "Expected names like blk.N.attn_q.weight"
        )
    return out


def _load_state_dict(path: Path) -> dict:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return _load_gguf(path)

    if path.is_file():
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "safetensors is required to load .safetensors files"
                ) from e
            return load_file(str(path))
        obj = _torch_load(str(path))
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if torch.is_tensor(v)}
        raise ValueError(f"Unsupported checkpoint format: {path}")

    st = list(path.glob("*.safetensors"))
    if st:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required to load .safetensors files"
            ) from e
        out = {}
        for f in st:
            out.update(load_file(str(f)))
        return out

    bins = list(path.glob("*.bin")) + list(path.glob("pytorch_model*.bin"))
    if bins:
        out = {}
        for f in bins:
            part = _torch_load(str(f))
            if isinstance(part, dict):
                out.update({k: v for k, v in part.items() if torch.is_tensor(v)})
        return out

    ggufs = list(path.glob("*.gguf"))
    if len(ggufs) == 1:
        return _load_gguf(ggufs[0])
    if len(ggufs) > 1:
        raise ValueError(
            f"Multiple GGUF files under {path}; pass a single file path instead"
        )

    raise FileNotFoundError(f"No weights found under {path}")


def _classify_key(key: str) -> str | None:
    k = key
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    if k.endswith("c_attn") or k.endswith(".attn.c_attn"):
        return "c_attn"
    if _O_PATTERNS.search(k):
        return "o"
    if _Q_PATTERNS.search(k):
        return "q"
    if _K_PATTERNS.search(k):
        return "k"
    if _V_PATTERNS.search(k):
        return "v"
    return None


def convert_state_dict(
    state: dict,
    phase_bits: int | None = 8,
    include_output_proj: bool = True,
) -> ConversionResult:
    """Convert attention-related tensors in a state dict to optical form."""
    optical = {}
    skipped = []

    for key, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 1:
            continue
        kind = _classify_key(key)
        if kind is None:
            skipped.append(key)
            continue
        if kind == "o" and not include_output_proj:
            skipped.append(key)
            continue

        if kind == "c_attn":
            if tensor.ndim != 2:
                skipped.append(key)
                continue
            d = tensor.shape[0] // 3
            if d * 3 != tensor.shape[0]:
                skipped.append(key)
                continue
            for name, chunk in zip(
                ("q", "k", "v"),
                (tensor[:d], tensor[d : 2 * d], tensor[2 * d :]),
            ):
                amp, phase = encode_weight_matrix(chunk, phase_bits=phase_bits)
                full_name = f"{key}.{name}"
                optical[full_name] = OpticalWeightTensor(
                    name=full_name,
                    amplitude=amp,
                    phase=phase,
                    shape=tuple(chunk.shape),
                    phase_bits=phase_bits,
                )
            continue

        amp, phase = encode_weight_matrix(tensor, phase_bits=phase_bits)
        optical[key] = OpticalWeightTensor(
            name=key,
            amplitude=amp,
            phase=phase,
            shape=tuple(tensor.shape),
            phase_bits=phase_bits,
        )

    meta = {
        "phase_bits": phase_bits,
        "n_converted": len(optical),
        "n_skipped": len(skipped),
        "converted_keys": sorted(optical.keys()),
    }
    return ConversionResult(weights=optical, meta=meta)


def convert_checkpoint(
    path,
    phase_bits: int | None = 8,
    include_output_proj: bool = True,
) -> ConversionResult:
    """Load a local checkpoint (safetensors, PyTorch, or GGUF) and convert."""
    state = _load_state_dict(Path(path))
    result = convert_state_dict(
        state, phase_bits=phase_bits, include_output_proj=include_output_proj
    )
    result.meta["source"] = str(path)
    result.meta["format"] = (
        "gguf" if str(path).lower().endswith(".gguf") else "torch_or_safetensors"
    )
    return result


def save_conversion(result: ConversionResult, out_dir) -> None:
    """Write amplitude/phase tensors and metadata to a directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, ow in result.weights.items():
        safe = name.replace("/", "__").replace(".", "__")
        payload[f"{safe}.amplitude"] = ow.amplitude
        payload[f"{safe}.phase"] = ow.phase
    torch.save(payload, out / "optical_weights.pt")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(result.meta, f, indent=2)
