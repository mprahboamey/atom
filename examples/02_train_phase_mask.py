from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom import DiffractiveLayer, WavePropagator, intensity


def main() -> None:
    torch.manual_seed(0)

    size = 48
    pixel_size = 2e-6
    layer = DiffractiveLayer(size, size)
    propagator = WavePropagator(pixel_size=pixel_size)

    field = torch.ones(size, size, dtype=torch.complex64)

    coords = (torch.arange(size) - size // 2) * pixel_size
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    target = torch.exp(-(x.square() + y.square()) / (2 * (6e-6) ** 2))
    target = target / target.max()

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

    print("Training one phase mask")
    for step in range(80):
        optimizer.zero_grad()
        output = propagator(layer(field), distance=1e-3)
        measured = intensity(output)
        measured = measured / (measured.max() + 1e-9)
        loss = torch.mean((measured - target) ** 2)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 79:
            print(f"step {step:02d}: loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
