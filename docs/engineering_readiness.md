# Engineering readiness

Target: software path is credible and test-gated (~80% of what can be done without hardware).

## Done

| Item | Status |
|------|--------|
| Binary score algebra = digital (exact) | Proven + unit tested |
| Noise not silently dropped | Fixed + unit tested |
| Vectorized multi-head optical scores | `optical_scores_multihead` |
| Bragg-shaped crosstalk option | `NoiseConfig.bragg_strength` |
| KV cache generate (optional) | `use_cache=True` (default off; float16 can drift) |
| Real safetensors full-depth identity | SmolLM2-135M, 30 layers |
| Text-prompt eval script | examples/16 |
| Certainty tests | tests/test_certainty.py |
| Honest limits doc | docs/audit_limits.md |

## Remaining for ~100% software

- float32 / stable cache path that matches full recompute bit-for-bit
- Detector / shot-noise model on scores
- Larger public safetensors model in CI (license + size)
- FPGA fixed-point score kernel reference

## Remaining outside software

- Measured M# and Bragg curves from a real medium
- Thermal fixing / packaging
- End-to-end optical bench

## Claim language

Safe: hybrid software scores match digital greedy on real HF weights under ideal binary phase; noise and Bragg hooks exist and are tested.

Not safe: physical optical LLM demonstrated.
