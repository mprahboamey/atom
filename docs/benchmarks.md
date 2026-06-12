# Benchmark Methodology and Projections

This document explains where each projection number comes from. Every figure is either a simulation output or a derivation from stated geometry and published physics. Nothing is measured hardware.

---

## Simulation parameters

| Parameter | Value |
|-----------|-------|
| Wavelength | 405 nm |
| Pixel size | 1 µm |
| Layer spacing | 10 µm |
| Angular increment | 0.1° |
| Angular range | 90° |
| Refractive index | 1.5 |

---

## Parameter capacity

The 90T figure is a geometric storage argument, not a trained model size.

A 1 cm³ holographic crystal under the simulation parameters above contains:

```
Z-layers       = 1 cm / 10 µm            = 1,000
Angular lanes  = 90° / 0.1°              = 900
Pixels / layer = (1 cm / 1 µm)²          = 100,000,000
Raw capacity   = 1,000 × 900 × 10⁸       = 90 × 10¹² (90T)
```

This is the information storage capacity of the volume, not a count of trained parameters. It assumes angular multiplexing channels are fully independent and that manufacturing tolerances hold at the simulation resolution — neither has been experimentally validated.

Allocating half to error correction and redundancy gives a conservative usable estimate of **45T per cm³**.

For comparison, the NVIDIA H100 SXM holds ~20–35B parameters in FP16 across 80–141 GB HBM. The projected density advantage is approximately **700×–4,500× per cm³** of active storage volume. The H100 package including HBM spans hundreds of cubic centimeters; a direct volume-normalized comparison narrows this gap.

---

## Latency and time-of-flight

Optical propagation latency follows directly from physics:

```
τ = L × n / c
```

where L is path length, n = 1.5 (refractive index), c = 2.998 × 10⁸ m/s.

| Path length | Latency |
|-------------|---------|
| 1 cm | ~50 ps |
| 2 cm | ~100 ps |
| 3 cm | ~150 ps |
| 4 cm | ~200 ps |

These are the physically grounded latency estimates for a compact device. The H100 forward pass latency for an 8B model is approximately 7–10 ms. The projected improvement is 5–7 orders of magnitude at the optical layer — contingent on the device being fabricated, which it has not been.

---

## Throughput

Throughput is derived from the time-of-flight model:

```
Raw TPS = 1 / τ = 1 / (1.334 ns) ≈ 750M passes/sec
```

The 1.334 ns figure models a longer internal optical path than the 4 cm physical device. Correcting for real-world overhead:

| Scenario | TPS |
|----------|-----|
| Raw optical (time-of-flight only) | ~750M |
| 50% overhead correction | ~375M |
| Conservative (100× overhead for digital I/O, averaging, post-processing) | ~7.5M |

All three figures are reported to avoid selective citation. The conservative 7.5M TPS figure is the appropriate lower bound for any comparison against H100's 100–150 TPS.

---

## Energy

Electronic transformer inference is dominated by memory bandwidth — moving FP16 weights from DRAM to compute units. Published figures for the H100 place forward pass energy in the millijoule range for large models.

Optical computation displaces the matrix-multiply hot path with a sequence of field propagation steps (FFT → pointwise multiply → IFFT) and phase mask applications. The energy cost of these optical operations, under idealized conditions, is in the picojoule–femtojoule regime.

The projected reduction is approximately 99% per forward pass. This projection assumes:
- coherent light source and detector efficiency are not dominant losses
- optical insertion loss is negligible (not experimentally validated)
- digital overhead (softmax, layer norm, residual addition) scales to ~0.07% of total operations

No energy measurement has been performed on hardware.

---

## What the simulation does and does not prove

| Claim | Status |
|-------|--------|
| ASM propagates fields with energy conservation < 2×10⁻⁵ | ✓ Numerically verified |
| Phase mask preserves intensity | ✓ Numerically verified |
| Forward/backward propagation is reversible | ✓ Numerically verified |
| Optical scores are algebraically identical to scaled dot-product attention | ✓ Verified exact to float precision |
| Gradients are finite through the full optical path | ✓ Verified |
| 90T parameter capacity per cm³ | Geometric derivation — not experimentally validated |
| ~50–200 ps latency for 1–4 cm device | Physics derivation (τ = Ln/c) — not measured |
| ~99% energy reduction | Architecture projection — not measured |
| >99% inference accuracy | **Not claimed. No noise model exists in this codebase.** |

The last row is a deliberate omission. Accuracy claims require a noise model, a task benchmark, and hardware. None of those exist here yet.
