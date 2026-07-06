"""M3.1 -- Phase quantization sensitivity.

A real crystal / spatial light modulator can't write an infinite-precision
phase angle -- it addresses some finite number of discrete phase levels.
The theory (attention.py) assumes continuous phase. This experiment asks
the honest engineering question CONTRIBUTING.md flags as the most
tractable open noise problem: how many bits of phase precision are
actually needed before attention degrades meaningfully from the ideal
case?

Metrics, in order of how much they should actually be trusted:
  1. Raw score error (MSE) vs ideal continuous phase -- least meaningful
     on its own, since attention runs scores through softmax before they
     do anything.
  2. KL divergence between quantized and ideal softmax attention
     distributions per query -- this is the metric that matters, since
     it's what the transformer actually consumes downstream.
  3. Top-1 agreement -- does the model attend to the *same* token as the
     max-weight target, even if the distribution around it shifted.

Run: python examples/05_phase_quantization_sweep.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from atom.attention import optical_scores_general


def run_sweep(
    seq_len: int = 32,
    dim: int = 64,
    n_trials: int = 20,
    bit_range: range = range(1, 13),
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)

    print(f"{'bits':>4} | {'levels':>7} | {'score MSE':>12} | {'mean KL':>10} | {'top-1 agree':>12}")
    print("-" * 60)

    for bits in bit_range:
        score_errors = []
        kls = []
        top1_agreements = []

        for trial in range(n_trials):
            q = torch.randn(seq_len, dim)
            k = torch.randn(seq_len, dim)
            positions = torch.arange(seq_len, dtype=torch.float32)

            ideal_scores = optical_scores_general(
                q, k, query_positions=positions, key_positions=positions
            )
            quantized_scores = optical_scores_general(
                q, k, query_positions=positions, key_positions=positions,
                phase_bits=bits,
            )

            score_errors.append((ideal_scores - quantized_scores).pow(2).mean().item())

            ideal_attn = F.softmax(ideal_scores, dim=-1)
            quantized_attn = F.softmax(quantized_scores, dim=-1)

            # KL(ideal || quantized) per query row, averaged
            kl = F.kl_div(quantized_attn.log(), ideal_attn, reduction="none").sum(-1)
            kls.append(kl.mean().item())

            ideal_top1 = ideal_attn.argmax(dim=-1)
            quantized_top1 = quantized_attn.argmax(dim=-1)
            top1_agreements.append((ideal_top1 == quantized_top1).float().mean().item())

        mse = sum(score_errors) / n_trials
        mean_kl = sum(kls) / n_trials
        mean_top1 = sum(top1_agreements) / n_trials

        print(f"{bits:>4} | {2**bits:>7} | {mse:>12.6f} | {mean_kl:>10.6f} | {mean_top1:>11.1%}")


if __name__ == "__main__":
    run_sweep()
