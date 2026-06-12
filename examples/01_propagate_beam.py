from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from optic import WavePropagator, gaussian_field, intensity


def main() -> None:
    propagator = WavePropagator(wavelength=405e-9, pixel_size=1e-6)
    field = gaussian_field(size=128, pixel_size=1e-6, waist=10e-6)
    output = propagator(field, distance=1e-3)

    before = intensity(field)
    after = intensity(output)

    print("Beam propagation")
    print(f"input total intensity:  {before.sum().item():.6f}")
    print(f"output total intensity: {after.sum().item():.6f}")
    print(f"input peak intensity:   {before.max().item():.6f}")
    print(f"output peak intensity:  {after.max().item():.6f}")


if __name__ == "__main__":
    main()
