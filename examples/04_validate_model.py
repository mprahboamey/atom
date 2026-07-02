from pathlib import Path
import math
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from atom import DiffractiveLayer, WavePropagator, gaussian_field, intensity, optical_scores


def assert_close(name: str, value: float, limit: float) -> None:
    print(f"{name}: {value:.3e}  limit={limit:.1e}")
    if value > limit:
        raise AssertionError(f"{name} failed: {value} > {limit}")


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().sum() / (a.abs().sum() + 1e-12)).item()


def check_propagation_energy() -> None:
    print("\n1. Propagation energy conservation")
    for size in (16, 32, 64, 128):
        field = gaussian_field(size=size, pixel_size=1e-6, waist=8e-6)
        propagator = WavePropagator(pixel_size=1e-6)
        before = intensity(field).sum()
        for distance in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
            after = intensity(propagator(field, distance)).sum()
            error = abs(after.item() - before.item()) / before.item()
            assert_close(f"size={size} distance={distance:g}", error, 2e-5)


def check_round_trip() -> None:
    print("\n2. Forward/backward propagation round trip")
    torch.manual_seed(2)
    size = 64
    real = torch.randn(size, size)
    imag = torch.randn(size, size)
    field = torch.complex(real, imag)
    propagator = WavePropagator(pixel_size=1e-6)
    out = propagator(propagator(field, 2e-3), -2e-3)
    assert_close("round-trip relative field error", relative_error(out, field), 3e-5)


def check_phase_mask() -> None:
    print("\n3. Phase mask preserves intensity")
    torch.manual_seed(3)
    field = torch.randn(8, 32, 32, dtype=torch.complex64)
    layer = DiffractiveLayer(32, 32)
    error = (intensity(layer(field)) - intensity(field)).abs().max().item()
    assert_close("max intensity difference", error, 2e-5)


def check_attention_scores() -> None:
    print("\n4. Optical attention score equivalence")
    torch.manual_seed(4)
    q = torch.randn(3, 5, 7)
    k = torch.randn(3, 6, 7)

    optical = optical_scores(q, k)
    digital = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    assert_close("scaled dot-product max error", (optical - digital).abs().max().item(), 1e-6)

    optical_cos = optical_scores(q, k, normalize=True)
    digital_cos = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    assert_close("cosine-score max error", (optical_cos - digital_cos).abs().max().item(), 1e-6)

    anti_q = torch.tensor([[[1.0, -2.0, 3.0]]])
    anti_k = -anti_q
    anti_score = optical_scores(anti_q, anti_k, normalize=True).item()
    print(f"anti-correlation score: {anti_score:.3f}")
    if anti_score > -0.999:
        raise AssertionError("negative correlation was not preserved")


def check_gradients() -> None:
    print("\n5. Gradients through phase mask and propagation")
    torch.manual_seed(5)
    size = 32
    layer = DiffractiveLayer(size, size)
    propagator = WavePropagator(pixel_size=2e-6)
    field = torch.ones(size, size, dtype=torch.complex64)
    output = propagator(layer(field), distance=5e-4)
    loss = intensity(output)[size // 2, size // 2]
    loss.backward()

    grad = layer.phase.grad
    if grad is None:
        raise AssertionError("phase gradient is missing")
    if not torch.isfinite(grad).all():
        raise AssertionError("phase gradient contains nan or inf")
    print(f"gradient abs max: {grad.abs().max().item():.3e}")
    print(f"gradient abs mean: {grad.abs().mean().item():.3e}")


def main() -> None:
    check_propagation_energy()
    check_round_trip()
    check_phase_mask()
    check_attention_scores()
    check_gradients()
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
