import math
import unittest

import torch

from atom import DiffractiveLayer, WavePropagator, gaussian_field, intensity, optical_scores


class CoreTests(unittest.TestCase):
    def test_phase_layer_preserves_intensity(self):
        field = torch.randn(16, 16, dtype=torch.complex64)
        layer = DiffractiveLayer(16, 16)
        self.assertTrue(torch.allclose(intensity(field), intensity(layer(field)), atol=1e-5))

    def test_propagation_approximately_preserves_total_intensity(self):
        field = gaussian_field(size=32, pixel_size=1e-6, waist=8e-6)
        propagator = WavePropagator(pixel_size=1e-6)
        output = propagator(field, distance=5e-4)
        self.assertAlmostEqual(intensity(field).sum().item(), intensity(output).sum().item(), places=4)

    def test_optical_scores_match_scaled_dot_product(self):
        torch.manual_seed(1)
        q = torch.randn(2, 3, 5)
        k = torch.randn(2, 4, 5)
        optical = optical_scores(q, k)
        digital = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        self.assertTrue(torch.allclose(optical, digital, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
