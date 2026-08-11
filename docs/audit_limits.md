# Audit: what is actually proven

This note is the result of stress-testing claims, not only re-running happy-path scripts.

## 1. Binary-phase "optical scores" equal digital scores by construction

`encode_signed_values` maps each real number to amplitude = |x| and phase ∈ {0, π}.
Then:

```text
Re(q_wave * conj(k_wave)) = |q| |k| cos(θq − θk) = q * k
```

term by term, because cos is ±1 exactly when phases are only 0 or π.

**Measured max |optical − digital| on random tensors: 0.**

So matching scores on SmolLM2 or any other model does **not** independently discover that interference equals attention. It checks that:

- the einsum / scaling path is wired correctly
- the hybrid module feeds the same Q/K the digital path would use
- float16 / GQA / RoPE / MLP stacking did not break greedy identity

That is **software integration validation**, and it is valuable. It is not evidence that a crystal has been simulated in full physical detail.

## 2. What the safetensors generate match does prove

On SmolLM2-135M-Instruct (real HF weights, 30 layers):

- greedy token sequences match for fixed ids and for a natural-language prompt
- full-forward logits top-1 agreement was 100% on a short sequence

So the hybrid stack (embed → N layers → lm_head) is consistent under optical vs digital **score functions** for those runs.

It does **not** prove:

- energy or latency of a physical optical score unit
- behavior under real Bragg crosstalk, scatter, detector noise
- that continuous-phase angular encoding is more than a RoPE-like software model

## 3. Continuous-phase path

`encode_angular_phase` uses `position * freq` with `freq = base^(-i/dim)` — the same structure as RoPE-style angular features. Relative positions change scores relative to pure binary phase. That is expected math, not a lab measurement of angular multiplexing in LiNbO₃.

## 4. Bug found and fixed: silent noise drop

`optical_scores_general(..., noise=NoiseConfig(phase_sigma=σ))` with **no** positions used to call `encode_signed_values` and **ignore** `phase_sigma` entirely (max diff vs digital = 0 even for σ = 0.5).

After the fix, σ > 0 forces the angular encoder (with zero positions if needed) so phase jitter is applied. Ideal σ = 0 with no positions remains exact.

## 5. Capacity / M#

`capacity.py` correctly treats geometric channel count as an upper bound and M# as the dynamic-range limit. Defaults are labeled conservative Fe:LiNbO₃, not measured for this repo.

## 6. Practical rule for claims

| Claim | OK? |
|-------|-----|
| Binary interference algebra = dot product | Yes |
| Hybrid software path matches digital greedy on real safetensors | Yes |
| Therefore a photorefractive accelerator is demonstrated | **No** |
| Continuous-phase software ≠ digital attention | Yes (by design) |
| Continuous-phase software = measured crystal physics | **No** |
