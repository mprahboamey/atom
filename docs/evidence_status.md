# Evidence status

Software measurements only. No physical optical hardware is claimed.

**Read first:** [`audit_limits.md`](audit_limits.md) — what generate-match does and does not prove.

## Primary result (safetensors)

**Checkpoint:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

| Check | Result |
|-------|--------|
| Full 30-layer hybrid generate, fixed ids | Optical greedy **==** digital |
| Natural language prompt | Optical greedy **==** digital |
| Full-forward logits top-1 | **100%** agreement |

Full log: [`results_smollm2_safetensors.md`](results_smollm2_safetensors.md).

## Interpretation (strict)

Binary-phase optical scores equal digital scores **by construction** (phase ∈ {0, π}). Matching on SmolLM2 validates the **hybrid wiring** (projections, GQA, RoPE, MLP, lm_head), not independent optical physics.

## Supporting

| Item | Status |
|------|--------|
| Noise + M# modules | Implemented |
| phase_sigma no longer silently dropped without positions | Fixed |
| Unified safetensors / GGUF loader | Yes |

## Not claimed

- Physical device results
- That software match implies crystal feasibility alone
- Chat quality of small-model generations
