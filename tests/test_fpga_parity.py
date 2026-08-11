"""FPGA score stand-in matches optical_scores on CPU."""

import math
import torch
from atom.attention import optical_scores


def test_score_kernel_matches_optical():
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16)
    k = torch.randn(2, 8, 16)
    scale = math.sqrt(16)
    cpu = (q @ k.transpose(-2, -1)) / scale
    opt = optical_scores(q, k)
    assert torch.allclose(cpu, opt, atol=1e-5)
