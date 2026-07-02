# ATOM
**Angular-Multiplexed Transformer Optical Model**

What if a neural network's weights didn't exist as floating points in memory but in holograms?

This repo is a numerically verified simulation showing that AI weights, encoded as phase structure inside a holographic crystal, can perform transformer attention exactly through wave interference. Not approximately. Algebraically identical to. Verified in code you can run in one command.

The math is proved. The hardware doesn't exist yet. That's the point of open sourcing this.

---

## What this actually is

A standard transformer spends most of its energy moving weights from memory to compute and back. Every forward pass, every token, billions of numbers shuffling across a bus.

This project asks: what if the weights *were* the computer?

A photorefractive crystal stores holograms through its entire volume using Bragg angle selectivity — write a pattern at a specific angle, and only light arriving at that exact angle reads it back. Change the angle by a fraction of a degree and you're reading a completely different stored pattern. This is angular multiplexing: hundreds of independent weight matrices living in the same cubic centimetre of crystal, each addressed by angle.

The ATOM simulator models this in PyTorch. It proves that when query and key vectors are encoded as optical wave amplitudes and interfere inside such a crystal, the output is algebraically identical to scaled dot-product attention. The computation happens through physics — diffraction and interference — not digital arithmetic.

For the full mathematical derivation of how the proof works, see [`docs/model.md`](docs/model.md).

---

## The capacity numbers (and what they actually mean)

A 1 cm³ crystal at reference parameters holds:

```
1,000 depth layers × 900 angular channels × 100M pixels/layer = 90 trillion addressable weight values
```

This is a geometric ceiling — the storage capacity of the medium before any data is written, like saying a hard drive holds 2 TB before formatting. Here's the honest range of what's usable:

| Scenario | Capacity | Notes |
|----------|----------|-------|
| Geometric ceiling | 90T | Pure math, no physical losses |
| 50% error correction allocation | 45T | Half reserved for redundancy |
| Realistic (SNR + crosstalk degradation) | 4.5T – 9T | 5–10% of ceiling under real multiplexing conditions |
| NVIDIA H100 (for reference) | 20–35B | Measured hardware |

The 5–10% realistic figure is where the engineering gets hard. It's also where most of the interesting open problems live. For the full derivation of every number in this table — assumptions, sources, and what's measured vs projected — see [`docs/benchmarks.md`](docs/benchmarks.md).

---

## What's proved vs what's projected

| Claim | Status |
|-------|--------|
| Optical interference = scaled dot-product attention | ✓ Proved and verified to float precision |
| Angular Spectrum Method conserves energy | ✓ < 2.3×10⁻⁷ relative error |
| Phase masks preserve intensity | ✓ Verified |
| Field propagation is reversible | ✓ Verified |
| Gradients flow through the full optical path | ✓ Verified |
| 90T geometric capacity per cm³ | Geometric derivation — not experimentally validated |
| 50–200 ps latency | Physics derivation — not measured |
| ~99% energy reduction | Architecture projection — not measured |
| Inference accuracy on real tasks | **Not claimed. There is no noise model.** |

The last row is intentional. Accuracy claims require a noise model, a task benchmark, and hardware. None of those exist here yet.

---

## Install and run

```bash
git clone https://github.com/mprahboamey/atom.git
cd atom
pip install -e .
```

If you don't have PyTorch yet:
```bash
pip install "atom-optic[torch]"
```

If you want everything in one shot:
```bash
pip install "atom-optic[full]"
```

Run everything and see all validation results:

```bash
python scripts/run_all.py
```

Or step through the concepts in order:

```bash
python examples/01_propagate_beam.py      # Gaussian beam through free space
python examples/02_train_phase_mask.py    # train a phase mask to focus light
python examples/03_optical_attention.py   # optical vs digital attention — exact match
python examples/04_validate_model.py      # all numerical checks with tolerances
```

---

## Project layout

```
atom/                        ← core library
├── propagation.py           ← Angular Spectrum Method, field helpers
├── diffractive.py           ← trainable phase masks, diffractive network
└── attention.py             ← interference-based attention scores

examples/                    ← one concept per script, run in order
scripts/run_all.py           ← runs everything, writes results/
tests/                       ← unit tests
results/                     ← validation output (generated at runtime)
figures/                     ← plots from simulation

docs/
├── model.md                 ← full mathematical derivation of the proof
└── benchmarks.md            ← every projection with full assumptions and sources
```

---

## Where this needs to go next

The math is done. The simulator works. What doesn't exist yet is everything physical — and that's what this repo is for. See [CONTRIBUTING.md](CONTRIBUTING.md) for specific open problems by domain.

Building a real device requires solving noise, materials, readout, and system integration problems that this codebase deliberately does not claim to have solved. If you work in any of those areas, there is a concrete problem in CONTRIBUTING.md with your name on it.

---

## References

- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Psaltis, D., Brady, D., & Wagner, K. (1988). Adaptive optical networks using photorefractive crystals. *Applied Optics*, 27(9), 1752–1759.
- Psaltis, D., & Mok, F. (1995). Holographic memories. *Scientific American*, 273(5), 70–76.
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004–1008.
- Miller, D. A. B. (2017). Attojoule optoelectronics. *Journal of Lightwave Technology*, 35(3), 346–396.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
