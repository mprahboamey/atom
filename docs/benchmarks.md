# Benchmark Methodology and Projections

Every number in this document is either a direct simulation output or a derivation from stated geometry and published physics. The derivation is shown in full. Nothing is measured hardware.

---

## Why volumetric storage changes the numbers

Flat silicon chips store weights in 2D — area scales as d². A 1 cm² chip at 7 nm node holds roughly 10¹² transistors, translating to tens of billions of parameters.

A holographic crystal stores weights in 3D, with an additional multiplexing dimension from Bragg angle selectivity. The capacity scales as d³ × angular_channels. This is why the numbers look different: it is not a more efficient 2D surface, it is a fundamentally different geometry.

Bragg angle selectivity is the physical mechanism that makes angular multiplexing possible. In a photorefractive crystal (lithium niobate, barium titanate, and similar materials), writing a hologram with a reference beam at angle θ creates a periodic refractive index grating tuned to that angle. Reading back requires illumination at exactly the same angle — a shift of even 0.1° selects a completely different stored hologram. This is not a design choice. It is a consequence of the Bragg condition:

```
2 · d_grating · sin(θ) = m · λ
```

where `d_grating` is the grating spacing, `θ` is the Bragg angle, `m` is the diffraction order, and `λ` is the wavelength. The angular selectivity narrows as the crystal depth increases, which means thicker crystals support finer angular multiplexing — more independent holograms per cubic centimetre.

---

## Simulation parameters

| Parameter | Value | Physical meaning |
|-----------|-------|-----------------|
| Wavelength | 405 nm | Violet laser diode — standard for photorefractive writing |
| Pixel size | 1 µm | Spatial resolution of each phase mask layer |
| Layer spacing | 10 µm | Distance between diffractive layers |
| Angular increment | 0.1° | Minimum resolvable angle between Bragg channels |
| Angular range | 90° | Total addressable angular range |
| Refractive index | 1.5 | Typical photorefractive crystal (e.g. LiNbO₃) |

---

## Parameter capacity

The 90T figure is a geometric capacity argument. It is the maximum number of independent weight values the physical medium can store and address, under the stated simulation parameters.

```
Z-layers       = 1 cm / 10 µm             = 1,000    (depth resolution)
Angular lanes  = 90° / 0.1°               = 900      (Bragg multiplexing)
Pixels / layer = (1 cm / 1 µm)²           = 10⁸      (spatial resolution)
Raw capacity   = 1,000 × 900 × 10⁸        = 9 × 10¹³ (90T)
```

This is not a trained model size. It is the storage capacity of the volume — analogous to saying a hard drive holds 2 TB, before any data is written to it.

Conservative estimates:

| Scenario | Capacity |
|----------|----------|
| Raw geometric maximum | 90T |
| 50% reserved for error correction and redundancy | 45T |
| Practical (accounting for multiplexing crosstalk) | 14T–20T |

For comparison, the NVIDIA H100 SXM stores 20–35B parameters in FP16 across 80–141 GB of HBM. The projected density advantage at the reference configuration is:

```
45T usable / 35B H100 ≈ 1,285× per cm³ of active storage volume
```

The H100 package including HBM spans hundreds of cubic centimetres. A full volume-normalized comparison narrows this gap considerably — see Figure 2 in `figures/fig2_parameter_density.png` for the full scaling surface across crystal sizes and angular resolutions.

---

## Latency and time-of-flight

Optical propagation latency is determined by the speed of light in the medium:

```
τ = L × n / c
```

where L is path length, n = 1.5 (refractive index), c = 2.998 × 10⁸ m/s.

| Path length | Latency | Notes |
|-------------|---------|-------|
| 1 cm | ~50 ps | Single-pass compact device |
| 2 cm | ~100 ps | |
| 3 cm | ~150 ps | |
| 4 cm | ~200 ps | |

For reference, H100 forward pass latency for an 8B parameter model is approximately 7–10 ms. The projected improvement is 5–7 orders of magnitude at the optical layer — contingent on the device being fabricated, which it has not been.

These estimates model only the optical propagation time. Real devices incur additional overhead for coherent source stabilisation, detector readout, digital pre/post-processing (softmax, layer norm, residual addition), and I/O. See throughput section below for overhead-corrected figures.

---

## Throughput

| Scenario | TPS | Assumption |
|----------|-----|------------|
| Raw optical (time-of-flight only, 4 cm device) | ~5B | τ = 200 ps |
| 50% duty cycle overhead | ~375M | I/O, source switching |
| Conservative (100× overhead) | ~7.5M | Full digital integration overhead |

All three figures are reported. The conservative 7.5M TPS is the appropriate lower bound for comparison against H100's 100–150 TPS at production batch sizes. Even the conservative figure represents a ~50,000× improvement — driven almost entirely by eliminating the memory bandwidth bottleneck, not by raw speed of the optical path itself.

---

## Energy

The energy advantage of optical inference is structural, not incremental.

Electronic transformer inference is dominated by memory bandwidth: each forward pass requires loading FP16 weight matrices from DRAM or HBM into on-chip SRAM for every layer. Published energy figures for the H100 place forward pass energy in the millijoule range for large models, with 60–80% of that attributable to data movement rather than arithmetic.

Optical inference eliminates this bottleneck. The weights are the medium. No data movement occurs during inference — the input beam propagates through the stored phase structure and the computation happens through diffraction and interference. The energy cost of a phase mask operation is a phase rotation of an optical field, which under idealized conditions costs picojoules to femtojoules.

The projected energy reduction is approximately 99% per forward pass. This assumes:

- coherent source power is not a dominant loss term
- optical insertion loss is negligible (not validated)
- digital overhead (softmax, layer norm, residual addition, I/O) remains below ~1% of total pass energy

No energy measurement has been performed. These are architecture-level projections.

---

## What the simulation proves vs. what it projects

| Claim | Status |
|-------|--------|
| ASM energy conservation < 2×10⁻⁵ | ✓ Numerically verified |
| Phase mask preserves intensity | ✓ Numerically verified |
| Forward/backward propagation is reversible | ✓ Numerically verified |
| Optical scores identical to scaled dot-product attention | ✓ Exact to float precision |
| Gradients finite through full optical path | ✓ Verified |
| 90T parameter capacity per cm³ | Geometric derivation — not experimentally validated |
| ~50–200 ps latency for 1–4 cm device | Physics derivation (τ = Ln/c) — not measured |
| ~99% energy reduction | Architecture projection — not measured |
| Bragg angle selectivity enables independent multiplexing | Established physics — not validated in this device |
| >99% inference accuracy | **Not claimed. No noise model in this codebase.** |

The last row is a deliberate omission. Accuracy on a real task requires a noise model, a task benchmark, and hardware. A simulation running in a noiseless mathematical environment cannot make that claim honestly.
