# Readout erase and rack refresh (stub)

## Problem

In unfixed photorefractive media, **read light erases stored holograms**
(weights). The crystal bulk is usually fine; the *gratings* fade.

## Software model (`atom/refresh.py`)

First-order fade:

```text
eta(n) ≈ eta0 * exp(-n / n_erase)
```

| Symbol | Meaning |
|--------|--------|
| `eta0` | Diffraction efficiency after a fresh write |
| `n_erase` | Read count scale for 1/e fade at reference intensity |
| `eta_min` | Floor still usable for the optical score path |
| `reads_to_eta_min` | Planning budget before mandatory rewrite |

Defaults are **Fe:LiNbO₃-class placeholders** for *unfixed* operation.
Thermal fixing / two-color readout can change `n_erase` by large factors.

## Rack policy (not a tiny monolithic gadget)

- Digital host holds the weight master.
- Each **bank** (crystal or sub-volume) tracks `read_count`.
- When remaining budget falls below a safety margin, schedule
  **`optical_rewrite_in_place`** — reinject weights into that bank.
- Stripe traffic so one bank can refresh while others serve (same idea as
  RAID rebuild or DRAM refresh, at rack scale).

## Writing is unsolved hardware

This stub does **not** claim a write engine. Open hardware work:

- Exposure schedule and closed-loop eta sensing
- Angular addressing without destroying neighbors
- Avoiding a **mechanical** refresh tax (no robot reseat every N reads)

The intended direction: rewrite is **integrated in the rack** as an
optical path from the host (or a dedicated write laser + SLM), stable
alignment, electronic shutters — not a jittery cartridge swap.

## API sketch

```python
from atom.refresh import default_fe_linbo3_rack

rack = default_fe_linbo3_rack(n_banks=8)
rack.record_read_on_bank(0, n=50)
print(rack.summary())
print(rack.plan_refresh())  # jobs for a future write driver
```

## Relation to capacity

`atom/capacity.py` bounds how many holograms fit (M#). This module bounds
how long they *last under read* before reinjection. Both are required for
a rack ops story.
