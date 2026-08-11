# Benchmark methodology and projections

Every number here is a **simulation or geometric derivation**, or a **placeholder** material parameter. Nothing is measured hardware in this repository.

For claim limits see [`validation_audit.md`](validation_audit.md).

---

## Why volumetric *storage* changes geometric ceilings

Flat silicon stores weights in area (~d²). A holographic volume adds depth and angular multiplexing, so a **geometric** count of addressable degrees of freedom can look much larger than a 2D chip under aggressive assumptions.

That is a statement about **possible addressing geometry**, not a demonstration that a photorefractive crystal has delivered that many independent, readable weights at usable SNR—or that attention runs optically end-to-end.

---

## Simulation parameters (assumptions)

| Parameter | Value | Notes |
|-----------|-------|--------|
| Wavelength | 405 nm | Common write band in demos; not unique |
| Pixel size | 1 µm | Assumed spatial resolution |
| Layer spacing | 10 µm | Assumed depth slicing |
| Angular increment | 0.1° | **Assumed** independent channel spacing — not a measured Bragg comb for this repo |
| Angular range | 90° | Assumed addressable range |
| Refractive index | ~2.2 | Order of magnitude for **LiNbO₃** (not 1.5). Generic organics differ |

Bragg selectivity **narrows with thickness**; treating 900 angles as fully independent is optimistic unless validated for the actual crystal and geometry.

Schematic Bragg condition (definitions of θ and Λ depend on geometry):

```text
2 n Λ sin(θ_B) ≈ m λ
```

---

## Parameter capacity

### Geometric ceiling (infinite dynamic range)

Under the table above, a 1 cm³-style count is on the order of:

```text
Z-layers      ≈ 1 cm / 10 µm     = 1000
Angular lanes ≈ 90° / 0.1°       = 900
Pixels/layer  ≈ (1 cm / 1 µm)²   = 1e8
Product       ≈ 9e13   (~90T “slots”)
```

**Label this only as a geometric ceiling.** It assumes infinite dynamic range, perfect selectivity, and ideal pixels.

### Usable capacity (M#)

Multiplexed equal-efficiency holograms obey:

```text
η ≈ (M# / M)²
```

so the number of usable channels is limited by material/system **M#** and minimum readable η (Mok / Psaltis and follow-on holographic storage literature).

`atom/capacity.py` applies this. Default M# / η_min are **conservative placeholders** (Fe:LiNbO₃ *class*), not measurements from this project. 90° geometry often reports **lower** M# than transmission geometry in published Fe:LN work.

**Never quote 90T as operational capacity.**

---

## Attention score algebra (software)

Binary phase {0, π}:

```text
Re(Σ q_wave conj(k_wave)) / √d  =  (q @ k.T) / √d
```

exactly. Tests lock this. Continuous-phase helpers are a **software** generalization (RoPE-like angular terms), not a lab Bragg dataset.

Correct algebra does **not** by itself imply system latency “at the speed of light” or a working optical accelerator: detection, modulation, hybrid digital stages, and SNR dominate real systems.

---

## Limits

This repository does not claim:

- a fabricated optical processor  
- measured optical task accuracy  
- measured hardware latency or energy (FPGA scaffold has no board data yet)  
- experimental noise, insertion loss, or phase drift on a crystal  

Experimental validation requires a physical device and calibrated material numbers.
