"""Hybrid optical-QK attention on a small random tensor."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom import HybridOpticalAttention, NoiseConfig


def main() -> None:
    torch.manual_seed(0)
    dim, seq, batch = 32, 8, 2

    model = HybridOpticalAttention(dim=dim, num_heads=4, path_cm=1.0)
    x = torch.randn(batch, seq, dim)
    out, weights = model(x)

    print(f"input  shape : {tuple(x.shape)}")
    print(f"output shape : {tuple(out.shape)}")
    print(f"weights shape: {tuple(weights.shape)}")
    print(f"output finite: {torch.isfinite(out).all().item()}")

    report = model.accounting(seq_len=seq)
    print(f"\noptical ToF  : {report.optical_tof_s * 1e12:.1f} ps")
    print(f"digital FLOPs: {report.digital_flops:.0f}")
    print(f"note: {report.notes}")

    # Noisy path
    noisy = HybridOpticalAttention(
        dim=dim,
        num_heads=4,
        noise=NoiseConfig(phase_sigma=0.02, crosstalk=0.05),
    )
    positions = torch.arange(seq, dtype=torch.float32)
    out_n, _ = noisy(x, positions=positions)
    print(f"\nnoisy output finite: {torch.isfinite(out_n).all().item()}")


if __name__ == "__main__":
    main()
