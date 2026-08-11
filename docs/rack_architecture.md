# Multi-crystal / rack architecture (stub)

## Idea

Same as a GPU cluster:

| GPU datacenter | ATOM rack stub |
|----------------|----------------|
| One GPU’s VRAM holds a shard of weights | One crystal holds a shard of holograms |
| N GPUs hold a larger model | N crystals hold a larger model |
| NVLink / Ethernet move activations | Interconnect moves activations + control |
| Host loads checkpoints | Host injects / refreshes weights |

Weights do **not** stream over the fabric every token. Activations do.

## Sharding modes (`atom/rack.py`)

1. **Layer pipeline** — crystal 0 runs layers 0..k-1, crystal 1 runs k..2k-1, …
   - Link: `activation_bus` residual between stages
   - Good when a full model does not fit in one volume

2. **Tensor parallel** — each crystal holds a slice of every layer’s weights
   - Link: `reduce_fabric` (all-reduce style partial results)
   - Good when layers are wide but depth is modest

3. **Data parallel** — full replicas (throughput, not model size)

## Interconnect roles

| Kind | Carries |
|------|--------|
| `activation_bus` | Hidden state between pipeline stages |
| `reduce_fabric` | TP partial sums |
| `control_and_weight_host` | Write, refresh, telemetry (not per-token) |
| `rack_backplane` | Same, across rack boundary |

Optical backplane vs copper/ethernet is a **hardware choice**. The stub only names the logical traffic.

## Capacity

`CrystalCapacity.from_material()` uses `capacity.estimate_capacity` (M#-limited usable params).  
`crystals_needed_for_params(N)` → how many volumes to host an N-parameter model under that ceiling.

## Example

```python
from atom.rack import plan_cluster, ShardMode

plan = plan_cluster(
    n_params=1e11,   # 100B
    n_layers=80,
    mode=ShardMode.LAYER_PIPELINE,
    crystals_per_rack=8,
)
print(plan.summary())
```

## Not implemented

- Real optical or electrical link simulation
- Deadlock-free schedules, quantization on the bus
- Automatic binding to `HybridTransformer` multi-process runtime
- Physical write path (see `docs/readout_refresh.md`)
