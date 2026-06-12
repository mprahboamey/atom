# OPTIC

A PyTorch simulator for diffractive optical neural networks, with verified numerical results and architecture-level projections for optical AI inference.

OPTIC models a coherent optical field moving through a stack of programmable diffractive elements. Trainable phase masks shape the wavefront. The Angular Spectrum Method propagates the field between layers. Wave interference computes attention scores. All of this runs in PyTorch, is end-to-end differentiable, and produces verifiable numerical results.

The repository also documents what an optical inference device built on these principles would look like at scale — parameter capacity, energy, and throughput — with projections derived transparently from the simulation geometry. Every projection is labeled as such.

---

## Verified results

These are not projections. These are numerical results from running the code.

| Check | Result | Tolerance |
|-------|--------|-----------|
| Propagation energy conservation | < 2.3 × 10⁻⁷ relative error | 2 × 10⁻⁵ |
| Forward/backward round-trip field error | 2.8 × 10⁻⁷ relative error | 3 × 10⁻⁵ |
| Phase mask intensity preservation | < 1.9 × 10⁻⁶ absolute error | 2 × 10⁻⁵ |
| Optical scores vs scaled dot-product attention | 0.00 error (exact to float precision) | 1 × 10⁻⁶ |
| Cosine similarity equivalence | 0.00 error (exact to float precision) | 1 × 10⁻⁶ |
| Gradients through full optical path | finite, non-zero | — |

Run them yourself:

```bash
python examples/04_validate_model.py
```

---

## Architecture projections

These figures are derived from the simulation geometry. They are not measured hardware results.

| Metric | NVIDIA H100 | OPTIC (projected) | Basis |
|--------|-------------|-------------------|-------|
| Parameter capacity | ~20–35B | ~90T raw / ~45T usable | Geometric: 1,000 Z-layers × 900 angular channels × 10⁸ pixels/layer in 1 cm³ |
| Latency | ~7–10 ms | ~50–200 ps (1–4 cm path) | Time-of-flight: τ = L·n/c, n=1.5 |
| Energy per forward pass | order of mJ | order of pJ–fJ | Optical operations displace DRAM-bandwidth-bound matmul |
| Throughput | 100–150 TPS | 7.5M–375M TPS (projected) | Derived from time-of-flight with 50–100× overhead correction |

The density advantage of ~700×–4,500× per cm³ follows directly from the geometry above. The parameter count is not a trained model — it is the storage capacity of the holographic volume under the stated manufacturing assumptions.

> **Note on latency figures:** The 1–4 cm time-of-flight estimates (50–200 ps) are the physically grounded numbers for a compact device. Longer optical path configurations would increase latency proportionally via τ = L·n/c.

---

## What it models

**Propagation.** The Angular Spectrum Method propagates a complex optical field through free space. The field is decomposed into plane-wave components in the Fourier domain, each component acquires a propagation phase, and the field is reconstructed via inverse FFT. Evanescent components are suppressed. Total intensity is conserved.

**Diffraction.** A diffractive layer multiplies the field by `exp(1j × θ)` where `θ` is a trainable parameter. The phase mask reshapes the wavefront without absorbing energy. After propagation, constructive and destructive interference redirect optical power. The phase parameters are differentiable — the entire optical path trains end-to-end with gradient descent.

**Attention via interference.** Real-valued query and key vectors are encoded as complex wave amplitudes (sign carried in phase). Their interference produces scores that are algebraically identical to scaled dot-product attention. This is not an approximation — the equivalence holds exactly to floating-point precision across all tested dimensions (d = 8 to 2048).

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
python examples/02_train_phase_mask.py    # train a phase mask to focus light on a target spot
python examples/03_optical_attention.py   # optical vs digital attention scores, exact match
python examples/04_validate_model.py      # all numerical checks with tolerances
```

## Run everything

```bash
python scripts/run_all.py
```

Runs all examples and the unit test suite. Writes `results/latest_run.json` and per-command logs to `results/logs/`.

---

## Project layout

```
optic/
├── propagation.py    # Angular Spectrum Method, Gaussian beam helpers
├── diffractive.py    # trainable phase masks, stacked diffractive network
└── attention.py      # interference-based attention scores

examples/             # one runnable script per concept
scripts/run_all.py    # runs everything, writes results
tests/                # unit tests for core behaviour
docs/                 # model derivation, benchmark methodology, integration notes
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

This is a software simulation. Verified results cover numerical properties of the optical model: energy conservation, reversibility, phase-mask behavior, and attention score equivalence. Architecture projections are derived from simulation geometry and published optical physics, not from fabricated hardware.

Claims that require physical validation — insertion loss, phase stability over temperature, manufacturing yield, measured inference accuracy — are not made here.

---

## References

- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004–1008.
- Miller, D. A. B. (2017). Attojoule optoelectronics for low-energy information processing and communications. *Journal of Lightwave Technology*, 35(3), 346–396.
