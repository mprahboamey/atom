# OPTIC

**What if AI weights didn't live in memory, but in light?**

This repository documents a numerically verified simulation experiment showing that neural network weights, encoded as optical phases masks inside a holographic medium, performs attention computation exactly, not approximately, through the principle of wave interference .

The core result: when AI weights are stored as phase structure in a volumetric crystal and a coherent beam passes through them, the resulting interference pattern produces attention scores algebraically identical to scaled dot-product attention. Zero error. Verified in code you can run in one command.

This repository provides the differentiable simulator, the numerical proofs, and the architecture-level projections for what a device built on these principles would look like at scale.

---

## The idea in plain terms

A holographic crystal doesn't store images on a flat surface, it stores them *through the volume*, using the angle of the reference beam to write and read each pattern independently. This is Bragg angle selectivity: only light arriving at the correct angle reconstructs a specific hologram. Change the angle slightly, and you address a completely different stored pattern.

Applied to AI: encode each weight matrix as a phase mask written into the crystal at a specific Bragg angle. To execute a forward pass, illuminate the crystal with a coherent beam encoding the input. The crystal, without any digital computation performs the matrix operation through diffraction and interference. Read the output intensity at the detector plane.

The volumetric geometry is what makes the capacity numbers interesting. A 1 cm³ crystal at 1 µm pixel resolution, 10 µm layer spacing, and 0.1° angular multiplexing steps holds:

```
1,000 depth layers × 900 angular channels × 100M pixels/layer = 90 trillion parameters
```

That is not a trained model. It is the storage capacity of the medium, the number of independent weight values the physics can hold and address. For comparison, the H100 holds 20–35B parameters in 80–141 GB of HBM silicon. The projected density advantage is 700×–4,500× per cm³.

---

## Verified results

These are numerical outputs from running the code. Not projections.

| Check | Result | Tolerance |
|-------|--------|-----------|
| Propagation energy conservation | < 2.3 × 10⁻⁷ relative error | 2 × 10⁻⁵ |
| Forward/backward round-trip field error | 2.8 × 10⁻⁷ relative error | 3 × 10⁻⁵ |
| Phase mask intensity preservation | < 1.9 × 10⁻⁶ absolute error | 2 × 10⁻⁵ |
| Optical scores vs scaled dot-product attention | 0.00 error (exact to float precision) | 1 × 10⁻⁶ |
| Cosine similarity equivalence | 0.00 error (exact to float precision) | 1 × 10⁻⁶ |
| Gradients through full optical path | finite, non-zero | — |

```bash
python examples/04_validate_model.py
```

---

## Architecture projections

Derived from simulation geometry and published optical physics. Not measured hardware.

| Metric | NVIDIA H100 | OPTIC (projected) | Basis |
|--------|-------------|-------------------|-------|
| Parameter capacity | ~20–35B | ~90T raw / ~45T usable | Volumetric: 1,000 Z-layers × 900 Bragg channels × 10⁸ pixels/layer in 1 cm³ |
| Latency | ~7–10 ms | ~50–200 ps (1–4 cm path) | Time-of-flight: τ = L·n/c, n = 1.5 |
| Energy per forward pass | order of mJ | order of pJ–fJ | Optical ops displace DRAM-bandwidth-bound matmul |
| Throughput | 100–150 TPS | 7.5M–375M TPS (projected) | Time-of-flight with 50–100× overhead correction |

See `docs/benchmarks.md` for the full derivation of every figure.

---

## What the simulator models

**Holographic weight storage.** Neural network weights are encoded as optical phase masks `exp(1j × θ)` where `θ` is the weight value mapped to a phase angle. In a physical device these masks would be written into a photorefractive crystal using Bragg holography, with each angular channel storing an independent weight matrix. In this simulator, `θ` is a trainable `nn.Parameter` that learns via standard gradient descent.

**Propagation.** The Angular Spectrum Method (ASM) propagates the complex optical field between layers. The field is decomposed into plane-wave components in the Fourier domain, each acquires a propagation phase, and the field is reconstructed via inverse FFT. Evanescent components are suppressed. Total intensity is conserved to within 2.3 × 10⁻⁷ relative error.

**Attention via interference.** Query and key vectors are encoded as complex wave amplitudes, magnitude carries the absolute value, phase carries the sign (0 for positive, π for negative). The interference of these waves produces a score:

```
score = Re( Σ q_wave · conj(k_wave) ) / √d
```

This is algebraically identical to scaled dot-product attention. The equivalence holds exactly to floating-point precision across dimensions d = 8 to 2048.

---

## Figures

| Figure | Description |
|--------|-------------|
| `figures/fig1_phase_mask.png` | Trained phase mask pattern and input/output field intensities |
| `figures/fig2_parameter_density.png` | Volumetric capacity surface as a function of crystal size and angular resolution |

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the examples

```bash
python examples/01_propagate_beam.py      # Gaussian beam propagation, intensity before/after
python examples/02_train_phase_mask.py    # train a phase mask to focus light onto a target
python examples/03_optical_attention.py   # optical vs digital attention scores, exact match
python examples/04_validate_model.py      # all numerical checks with explicit tolerances
```

## Run everything

```bash
python scripts/run_all.py
```

---

## Project layout

```
optic/
├── propagation.py    # Angular Spectrum Method, Gaussian beam helpers
├── diffractive.py    # trainable phase masks, stacked diffractive network
└── attention.py      # interference-based attention scores

examples/             # one runnable script per concept
figures/              # publication-quality figures generated from simulation
scripts/run_all.py    # runs everything, writes results
tests/                # unit tests for core behaviour
docs/                 # model derivation, benchmark methodology
results/              # validation output (generated at runtime)
```

Suggested reading order:

1. `optic/propagation.py`
2. `examples/01_propagate_beam.py`
3. `optic/diffractive.py`
4. `examples/02_train_phase_mask.py`
5. `optic/attention.py`
6. `docs/model.md`
7. `docs/benchmarks.md`

---

## Scope

This is a software simulation. Verified results cover: energy conservation, field reversibility, phase-mask behaviour, and attention score equivalence. Architecture projections are derived from simulation geometry and published optical physics.

Claims requiring physical validation, insertion loss, phase stability over temperature, manufacturing yield, measured inference accuracy on fabricated hardware, are not made here. That work requires a lab.

---

## References

- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004–1008.
- Psaltis, D., & Mok, F. (1995). Holographic memories. *Scientific American*, 273(5), 70–76.
- Miller, D. A. B. (2017). Attojoule optoelectronics for low-energy information processing and communications. *Journal of Lightwave Technology*, 35(3), 346–396.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
