# Evidence status

Software measurements only. No physical optical hardware is claimed.

**Read first:** [`audit_limits.md`](audit_limits.md) · [`validation_audit.md`](validation_audit.md)

## Primary result (safetensors)

**Checkpoint:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

| Check | Result |
|-------|--------|
| Full 30-layer hybrid generate, fixed ids | Optical greedy **==** digital |
| Natural language prompt | Optical greedy **==** digital |
| Full-forward logits top-1 | **100%** on logged short sequence |

Binary-phase equality is **algebraic**. The model run validates the hybrid stack.

Full log: [`results_smollm2_safetensors.md`](results_smollm2_safetensors.md).

## Supporting (software)

| Item | Status |
|------|--------|
| Noise + Bragg crosstalk hooks | Implemented + tests |
| M# capacity model | Implemented; defaults not lab-measured here |
| Readout refresh stub | Placeholder erase curve |
| Rack placement stub | Logical shards/links only |
| FPGA path | Scaffold; no board joules |

## Not claimed

- Physical device results  
- Operational “90T” storage  
- FPGA energy/latency  
- Continuous-phase path = measured Bragg physics  
