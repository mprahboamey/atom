# Readout erase and rack refresh (stub)

## Problem

In **unfixed, single-color** photorefractive recording, read light can erase
stored holograms (weights). The crystal bulk is usually fine; the *gratings*
fade.

## Decay shape

First-order model used in `atom/refresh.py`:

```text
eta(n) ≈ eta0 * exp(-n / n_erase)
```

(or equivalently exponential in time / fluence at fixed intensity).

**Literature:** monoexponential erasure under illumination is the standard
first-order description for photorefractive gratings in Fe:LiNbO₃-class media
(after accounting for absorption, decays often fit simple exponentials).
Some recent storage work prefers a **stretched** exponential for erasure during
multiplexed writing (`β < 1`); `β = 1` recovers ordinary exponential. Using
exponential as the default ops model is **empirically grounded**, not arbitrary.

## Numbers depend on recording regime

| Regime | Read durability (order of magnitude) | Role of refresh |
|--------|--------------------------------------|-----------------|
| Unfixed, single-color continuous readout | **Short** — continuous interrogation can wipe usable η quickly | Aggressive reinjection / rewrite may be required |
| Two-color / photon-gated (e.g. IR write + visible gate) | **Much longer** — e.g. Lee et al., APL 2002, stoichiometric LiNbO₃:Tb,Fe: estimated **~8×10⁷** readouts at 1 Gbit/s before diffracted signal halves | Refresh may be rare or mainly dark-decay / system maintenance |
| Thermal / ionic fixing | Persistence improves; tradeoffs on process complexity | Different failure modes (ionic conductivity, etc.) |

Default `n_erase = 500` in code is a **placeholder for the unfixed single-color
ops story** — “how many full-bank reads until η hits the floor” in a **volatile**
regime. It is **not** a measured constant for a specific crystal in this repo,
and it is **not** appropriate as a default if the product commits to two-color
or fixed holograms.

**Design choice (not yet coded as an enum):** pick a recording regime, then set
`n_erase` (or a fluence-based τ) from that regime’s data. Two-color can shrink
the need for aggressive weight reinjection by orders of magnitude at the cost of
gating optics and material process complexity.

## Software model (`atom/refresh.py`)

| Symbol | Meaning |
|--------|--------|
| `eta0` | Diffraction efficiency after a fresh write |
| `n_erase` | Read-count scale for 1/e fade at reference intensity |
| `eta_min` | Floor still usable for the optical score path |
| `reads_to_eta_min` | Planning budget before mandatory rewrite |

## Rack policy

Digital host holds the weight master. Banks track `read_count`. Below a safety
margin, schedule **optical rewrite in place** (not mechanical reseat). Stripe so
one bank refreshes while others serve.

## Writing is unsolved hardware

Exposure schedule, closed-loop η sensing, neighbor erase during write, thermal
limits — open. This module only plans.

## API sketch

```python
from atom.refresh import default_fe_linbo3_rack

rack = default_fe_linbo3_rack(n_banks=8)
rack.record_read_on_bank(0, n=50)
print(rack.summary())
print(rack.plan_refresh())
```

## Relation to capacity

`atom/capacity.py` bounds how many holograms fit (M#). This module bounds how
long they last under read before reinjection — and that lifetime is **regime-
dependent**.
