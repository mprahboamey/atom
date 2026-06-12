import math
import unittest

import torch
from torch.nn import functional as F

from optic import DiffractiveLayer, WavePropagator, gaussian_field, intensity, optical_scores


class MathStressTests(unittest.TestCase):
    def test_propagation_conserves_energy_across_sizes_and_distances(self):
        for size in (16, 32, 64):
            field = gaussian_field(size=size, pixel_size=1e-6, waist=8e-6)
            propagator = WavePropagator(pixel_size=1e-6)
            before = intensity(field).sum().item()
            for distance in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
                after = intensity(propagator(field, distance)).sum().item()
                relative_error = abs(after - before) / before
                self.assertLess(relative_error, 2e-5)

    def test_propagation_round_trip_reconstructs_field(self):
        torch.manual_seed(12)
        field = torch.randn(32, 32, dtype=torch.complex64)
        propagator = WavePropagator(pixel_size=1e-6)
        recovered = propagator(propagator(field, 1e-3), -1e-3)
        relative_error = ((recovered - field).abs().sum() / field.abs().sum()).item()
        self.assertLess(relative_error, 3e-5)

    def test_phase_layer_preserves_batched_intensity(self):
        torch.manual_seed(13)
        field = torch.randn(4, 24, 24, dtype=torch.complex64)
        layer = DiffractiveLayer(24, 24)
        self.assertTrue(torch.allclose(intensity(layer(field)), intensity(field), atol=2e-5))

    def test_attention_scores_match_digital_math(self):
        torch.manual_seed(14)
        q = torch.randn(2, 3, 5)
        k = torch.randn(2, 4, 5)

        scaled = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        self.assertTrue(torch.allclose(optical_scores(q, k), scaled, atol=1e-6))

        cosine = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        self.assertTrue(torch.allclose(optical_scores(q, k, normalize=True), cosine, atol=1e-6))

    def test_attention_preserves_negative_correlation(self):
        q = torch.tensor([[[1.0, -2.0, 3.0]]])
        k = -q
        self.assertLess(optical_scores(q, k, normalize=True).item(), -0.999)

    def test_gradients_are_finite(self):
        torch.manual_seed(15)
        size = 24
        field = torch.ones(size, size, dtype=torch.complex64)
        layer = DiffractiveLayer(size, size)
        propagator = WavePropagator(pixel_size=2e-6)

        measured = intensity(propagator(layer(field), 5e-4))
        loss = measured[size // 2, size // 2]
        loss.backward()

        self.assertIsNotNone(layer.phase.grad)
        self.assertTrue(torch.isfinite(layer.phase.grad).all())


if __name__ == "__main__":
    unittest.main()
