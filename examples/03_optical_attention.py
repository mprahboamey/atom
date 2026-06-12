from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from optic import optical_scores


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)

    optical = optical_scores(q, k)
    digital = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)

    print("Optical attention scores")
    print(f"max difference from scaled dot product: {(optical - digital).abs().max().item():.8f}")


if __name__ == "__main__":
    main()
