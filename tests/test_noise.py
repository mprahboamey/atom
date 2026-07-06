import math
import unittest

import torch

from atom import quantize_phase, optical_scores_general


class TestPhaseQuantization(unittest.TestCase):
    def test_quantize_phase_has_correct_level_count(self):
        # Sweep a fine grid across a full turn and confirm the quantizer
        # only ever produces exactly 2**bits distinct output values.
        phase = torch.linspace(0, 2 * math.pi, steps=10_000)
        for bits in (1, 2, 4, 8):
            quantized = quantize_phase(phase, bits)
            distinct = torch.unique(torch.round(quantized, decimals=6))
            self.assertLessEqual(distinct.numel(), 2 ** bits)

    def test_quantize_phase_wraps_cyclically(self):
        # Phase is cyclic -- an angle and that angle plus a full turn must
        # quantize to the same level, since an SLM has no concept of
        # "angle 400 degrees."
        phase = torch.tensor([0.3, 0.3 + 2 * math.pi, 0.3 - 2 * math.pi])
        quantized = quantize_phase(phase, bits=6)
        self.assertTrue(torch.allclose(quantized[0], quantized[1], atol=1e-5))
        self.assertTrue(torch.allclose(quantized[0], quantized[2], atol=1e-5))

    def test_quantize_phase_rejects_nonpositive_bits(self):
        with self.assertRaises(ValueError):
            quantize_phase(torch.zeros(4), bits=0)

    def test_high_bit_quantization_converges_to_continuous_case(self):
        # At high enough precision, quantized scores must approach the
        # ideal continuous-phase case arbitrarily closely -- if this
        # didn't hold, the quantizer itself would be broken, independent
        # of any real hardware question.
        torch.manual_seed(0)
        q = torch.randn(6, 16)
        k = torch.randn(6, 16)
        positions = torch.arange(6, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        high_precision = optical_scores_general(
            q, k, query_positions=positions, key_positions=positions, phase_bits=16
        )
        self.assertTrue(torch.allclose(ideal, high_precision, atol=1e-3))

    def test_low_bit_quantization_visibly_degrades_scores(self):
        # Conversely, 1-bit phase (effectively back to the binary case
        # plus rounding) should NOT match the continuous case -- if it
        # did, quantization wouldn't be doing anything, meaning the test
        # above would be vacuous.
        torch.manual_seed(1)
        q = torch.randn(6, 16)
        k = torch.randn(6, 16)
        positions = torch.arange(6, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        low_precision = optical_scores_general(
            q, k, query_positions=positions, key_positions=positions, phase_bits=1
        )
        self.assertFalse(torch.allclose(ideal, low_precision, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
