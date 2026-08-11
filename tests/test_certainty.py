"""Hard certainty tests: algebra, noise non-noop, Bragg, multi-head."""

from __future__ import annotations

import math

import torch

from atom.attention import optical_scores, optical_scores_general, optical_scores_multihead
from atom.noise import NoiseConfig, bragg_selectivity_kernel, apply_bragg_crosstalk


def test_binary_optical_exact_dot_product():
    torch.manual_seed(0)
    q = torch.randn(3, 11, 16)
    k = torch.randn(3, 9, 16)
    opt = optical_scores(q, k)
    dig = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    assert torch.allclose(opt, dig, atol=1e-5, rtol=1e-5)


def test_phase_sigma_not_silent():
    torch.manual_seed(1)
    q = torch.randn(1, 5, 8)
    k = torch.randn(1, 5, 8)
    ideal = optical_scores(q, k)
    noisy = optical_scores_general(q, k, noise=NoiseConfig(phase_sigma=0.4))
    assert (noisy - ideal).abs().max().item() > 0.05


def test_zero_sigma_still_exact():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 6, 8)
    a = optical_scores(q, k)
    b = optical_scores_general(q, k, noise=NoiseConfig(phase_sigma=0.0))
    assert torch.allclose(a, b, atol=1e-6)


def test_bragg_kernel_peaks_at_center():
    k = bragg_selectivity_kernel(21, sinc_width=1.2)
    assert k.argmax().item() == 10
    assert torch.isclose(k.sum(), torch.tensor(1.0), atol=1e-5)


def test_bragg_crosstalk_changes_scores():
    torch.manual_seed(3)
    s = torch.randn(2, 8, 16)
    mixed = apply_bragg_crosstalk(s, sinc_width=1.5, strength=0.5)
    assert mixed.shape == s.shape
    assert (mixed - s).abs().mean().item() > 1e-4


def test_multihead_matches_loop():
    torch.manual_seed(4)
    q = torch.randn(2, 4, 7, 8)
    k = torch.randn(2, 4, 5, 8)
    batched = optical_scores_multihead(q, k)
    refs = []
    for h in range(4):
        refs.append(optical_scores(q[:, h], k[:, h]))
    ref = torch.stack(refs, dim=1)
    assert torch.allclose(batched, ref, atol=1e-5)


def test_bragg_noise_config_path():
    torch.manual_seed(5)
    q = torch.randn(1, 6, 8)
    k = torch.randn(1, 6, 8)
    ideal = optical_scores(q, k)
    n = NoiseConfig(bragg_strength=0.4, bragg_sinc_width=1.2)
    out = optical_scores_general(q, k, noise=n)
    assert (out - ideal).abs().mean().item() > 1e-5
