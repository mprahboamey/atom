# Scale training plan (2T → 20T → 200T)

This document and `atom/scale_plan.py` fix the **shape** of a large hybrid MoE
training job so moving from 2T to 200T is the same formula with larger inputs.

## Idea

| Quantity | Driven by |
|----------|-----------|
| Storage banks / crystals | **Total** parameters (M#-usable capacity per volume) |
| Token budget + FLOP proxy | **Active** parameters (top-k experts + shared layers) |
| Optical role | Score path fraction + measured speedup (default 1× until measured) |

Same planner row for every target. No separate theory at 200T.

## Default ladder

```bash
python examples/16_scale_plan.py
python examples/16_scale_plan.py --total 20e12 --experts 128 --top-k 2
python examples/16_scale_plan.py --optical-speedup 10 --tflops 500
```

Defaults: 64 experts, top-2, token_multiplier=20 × active_params,
`CapacityParams.m_number=2`, digital 100 TFLOP/s × 0.4 utilization.

## What a 20T job looks like under those defaults

- **Total params:** 2×10¹³
- **Active params:** shared + (2/64) of expert pool (see `MoEShape.from_total`)
- **Tokens:** ~20 × active (change `--token-mult` for denser training)
- **Volumes:** ceil(total / usable_capacity) under Fe:LiNbO₃-class M# model
- **Wall time:** FLOP proxy / assumed TFLOP/s — **assumption-limited** until hardware numbers replace `ThroughputAssumptions`

## Hybrid mapping

1. Attention **scores** — phase / optical path (software identity already in repo).
2. Softmax, V, router — digital.
3. **Experts** — digital and/or one bank per expert group on the rack.
4. Training — digital optimizers; optional later optical write path does not change the scale arithmetic.

## Predictability rule

If plan P works at total N with active A, tokens T, volumes V, then
αN uses the same code paths with active ~αA, tokens ~αT, volumes ~αV
when expert fractions and top-k stay fixed. That is the scalability contract.

## Not included

Calendar dates for a real 20T run, measured optical joules, or quality guarantees.
Those require data, optimizers, and hardware fills of the assumption slots.
