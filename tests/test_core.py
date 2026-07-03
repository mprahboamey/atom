import math
import unittest

import torch

from atom import (
    DiffractiveLayer,
    WavePropagator,
    gaussian_field,
    intensity,
    optical_scores,
    optical_scores_general,
)


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


    def test_general_scores_reduce_to_binary_case_when_positions_none(self):
        torch.manual_seed(2)
        q = torch.randn(2, 3, 5)
        k = torch.randn(2, 4, 5)
        general = optical_scores_general(q, k)
        binary = optical_scores(q, k)
        self.assertTrue(torch.allclose(general, binary, atol=1e-6))

    def test_general_scores_reduce_to_binary_case_when_relative_offset_is_zero(self):
        # Relative phase is (query_position - key_position) * freq, so it
        # only cancels back to the binary case for the query/key pair at
        # equal position -- pin that on the diagonal of a self-attention
        # setup, where every token's position matches itself.
        torch.manual_seed(3)
        x = torch.randn(5, 5)
        same_positions = torch.arange(5, dtype=torch.float32)
        general = optical_scores_general(x, x, query_positions=same_positions, key_positions=same_positions)
        binary = optical_scores(x, x)
        self.assertTrue(torch.allclose(general.diagonal(), binary.diagonal(), atol=1e-5))
        # off-diagonal (unequal position) pairs are where relative phase
        # actually does something -- they should generally NOT match.
        self.assertFalse(torch.allclose(general, binary, atol=1e-3))

    def test_continuous_phase_produces_genuine_interference(self):
        # With distinct query/key positions, scores must diverge from
        # the plain dot product -- this is the actual claim under test:
        # relative continuous phase does real work that a real-valued
        # dot product cannot reproduce, unlike the binary-phase case.
        torch.manual_seed(4)
        q = torch.randn(3, 8)
        k = torch.randn(4, 8)
        q_positions = torch.arange(3, dtype=torch.float32)
        k_positions = torch.arange(4, dtype=torch.float32) + 10.0
        general = optical_scores_general(q, k, query_positions=q_positions, key_positions=k_positions)
        plain_dot = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        self.assertFalse(torch.allclose(general, plain_dot, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
