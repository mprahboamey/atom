import math
import unittest

import torch
from torch.nn import functional as F

from atom import HybridOpticalAttention, NoiseConfig, optical_scores
from atom.hybrid import optical_time_of_flight, estimate_digital_flops


class TestHybridOpticalAttention(unittest.TestCase):
    def test_output_shapes(self):
        torch.manual_seed(0)
        m = HybridOpticalAttention(dim=32, num_heads=4, use_layer_norm=True)
        x = torch.randn(2, 8, 32)
        out, weights = m(x)
        self.assertEqual(out.shape, (2, 8, 32))
        self.assertEqual(weights.shape, (2, 4, 8, 8))

    def test_ideal_scores_match_digital(self):
        # With no noise and binary phase (positions=None), scores from the
        # optical path must match scaled dot-product attention.
        torch.manual_seed(1)
        dim = 16
        m = HybridOpticalAttention(dim=dim, num_heads=1, use_layer_norm=False, noise=None)
        # Freeze projections so we can compare scores only
        x = torch.randn(1, 6, dim)
        q = m.q_proj(x)
        k = m.k_proj(x)
        optical = optical_scores(q, k)
        scale = math.sqrt(dim)
        digital = (q @ k.transpose(-2, -1)) / scale
        self.assertTrue(torch.allclose(optical, digital, atol=1e-5))

    def test_noise_path_runs(self):
        torch.manual_seed(2)
        noise = NoiseConfig(phase_sigma=0.05, crosstalk=0.1)
        m = HybridOpticalAttention(dim=16, num_heads=2, noise=noise)
        x = torch.randn(1, 5, 16)
        positions = torch.arange(5, dtype=torch.float32)
        out, weights = m(x, positions=positions)
        self.assertEqual(out.shape, (1, 5, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradients_flow(self):
        torch.manual_seed(3)
        m = HybridOpticalAttention(dim=16, num_heads=2)
        x = torch.randn(1, 5, 16, requires_grad=True)
        out, _ = m(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_accounting(self):
        m = HybridOpticalAttention(dim=32, num_heads=4, path_cm=2.0)
        report = m.accounting(seq_len=16)
        self.assertGreater(report.optical_tof_s, 0.0)
        self.assertGreater(report.digital_flops, 0.0)
        # 2 cm * 1.5 / c ≈ 100 ps
        expected = optical_time_of_flight(2.0)
        self.assertAlmostEqual(report.optical_tof_s, expected)


class TestAccountingHelpers(unittest.TestCase):
    def test_tof_positive(self):
        self.assertGreater(optical_time_of_flight(1.0), 0.0)

    def test_flops_scale(self):
        f1 = estimate_digital_flops(8, 32)
        f2 = estimate_digital_flops(16, 32)
        self.assertGreater(f2, f1)


if __name__ == "__main__":
    unittest.main()
