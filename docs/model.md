# Model Notes

This document describes the physical and mathematical foundations of the simulator. It assumes basic familiarity with complex numbers and neural networks, but not with optics.

---

## The core idea: weights as light

In a standard neural network, weights are numbers sitting in DRAM. To run a forward pass, the GPU reads those numbers, multiplies them by the input, and writes the result back. The bottleneck is not just computation but moving data between memory and compute.

This project asks a different question: what if the weights were not numbers in memory, but phase structure encoded into a physical medium, specifically, a holographic crystal?

A holographic crystal stores information through its volume using Bragg angle selectivity. When a reference beam writes a hologram at a specific angle, only light arriving at that exact angle later reconstructs it. Change the angle by a fraction of a degree and you address a completely different stored pattern. This is angular multiplexing; the same cubic centimetre of crystal holds hundreds or thousands of independent holograms, each retrievable by angle.

Applied to AI weights: encode each weight matrix as a phase mask written into the crystal at a specific Bragg angle. When a coherent input beam illuminates the crystal, the stored phase structure diffracts and interferes with it. The output intensity at the detector encodes the result of the computation with no digital arithmetic, no memory bandwidth, and no clock cycles.

This simulator models that process in software, using optical physics, with numerical verification at every step.

---

## Field representation

The simulator stores a monochromatic optical field as a complex tensor:

```
u(x, y) = A(x, y) * exp(1j * phi(x, y))
```

`A` is amplitude. `phi` is phase in radians. A single complex value carries both at each grid point.

Detectors measure intensity, not phase:

```
I(x, y) = |u(x, y)|²
```

This is the quantity used whenever the field is read out as an image or matched against a target pattern.

---

## Propagation

Free-space propagation between layers uses the Angular Spectrum Method (ASM).

Any 2D optical field can be decomposed into plane waves travelling in different directions. Each plane wave accumulates a known phase shift as it propagates forward through free space. ASM performs this decomposition in the Fourier domain, applies the phase shift to each component, and reconstructs the output field via inverse FFT.

```
U(fx, fy, 0) = FFT[ u(x, y, 0) ]
U(fx, fy, z) = U(fx, fy, 0) * H(fx, fy, z)
u(x, y, z)   = IFFT[ U(fx, fy, z) ]
```

The transfer function is:

```
H(fx, fy, z) = exp(1j * kz * z)
kz = sqrt(k² - (2π fx)² - (2π fy)²)
k  = 2π / wavelength
```

Components where `kz²` is negative correspond to evanescent waves fields that decay exponentially rather than propagate. These are suppressed.

**Energy conservation.** Total intensity is conserved to within 2.3 × 10⁻⁷ relative error across all tested grid sizes and propagation distances; approximately 100× better than the stated tolerance. This confirms the simulator is not leaking or creating energy, which is the prerequisite for all projection claims downstream.

---

## Phase masks as optical weights

A phase mask is the optical representation of a weight matrix:

```
u_out(x, y) = u_in(x, y) * exp(1j * theta(x, y))
```

`theta` is a trainable `nn.Parameter`. Each pixel of the mask is one weight value, encoded as a phase angle between 0 and 2π. The mask modifies the wavefront reshaping the direction and interference pattern of the field, without absorbing energy. Intensity at the mask plane is unchanged.

This is the key property that makes optical computation energy-efficient: the weights modify the field through phase rotation, not amplitude attenuation. No energy is consumed by the weight application itself.

After propagation, the accumulated phase differences from multiple mask layers produce constructive and destructive interference that concentrates optical power at specific output locations performing the equivalent of a learned linear transformation.

Because `theta` is differentiable, a loss computed from the output intensity backpropagates through the propagation step and into the mask parameters. The full optical path trains end-to-end with standard gradient descent. Example 02 demonstrates this: a single phase mask is trained to focus a plane wave onto a Gaussian target spot, with loss decreasing from 0.025 to 0.004 over 150 steps.

---

## Attention scores via interference

The attention module addresses a specific question: when weights are encoded as phase masks and inputs are encoded as optical fields, does the resulting interference produce the same scores as scaled dot-product attention?

The base encoding is deliberately minimal:

- positive values → complex amplitude with phase 0
- negative values → complex amplitude with phase π

This is a binary phase encoding, and that's not incidental — it's the *only* phase choice for which `Re(q_wave · conj(k_wave))` recovers `q · k` exactly, term for term. Exact recovery requires `cos(θ_q − θ_k) = sign(q · k)`, and cosine only equals ±1 at 0 and π. So the base proof below is intentionally the degenerate case of interference, chosen to make the equivalence to standard attention exact rather than approximate. It is not, by itself, a demonstration that continuous phase is doing anything — see "Continuous phase" below for that.

The interference score between a query wave and a key wave is then:

```
score = Re( Σ q_wave · conj(k_wave) ) / √d
```

Expanding algebraically, this is identical to:

```
score = q @ k.T / √d
```

The validation script confirms both expressions produce the same values to floating-point precision zero error at d = 8, 64, 512, and sub-10⁻⁶ at d = 2048 where floating-point rounding accumulates.

Sign preservation is also verified explicitly: a query vector and its negation produce an anti-correlation score of exactly −1.000, confirming that semantic opposition is correctly encoded and recovered through the optical path.

This result is the numerical foundation for the architecture projections. If the interference mechanism correctly computes attention, then a physical device running this interference on real light fields performs transformer attention — with the weights stored in the holographic medium and the computation happening at the speed of light through the crystal volume.

### Continuous phase

`encode_angular_phase` and `optical_scores_general` (`atom/attention.py`) generalize the base encoding: each token accumulates a continuous phase offset from its angular position — modeling angular multiplexing, where sequence positions are addressed at distinct Bragg angles inside the crystal, the same construction rotary position embeddings use for sequence index. With query and key positions equal, the relative phase cancels and this reduces exactly to the binary case above. With distinct query/key positions it does not reduce to a plain dot product — scores become sensitive to relative angular offset, which is the actual interference effect binary phase can't produce. This is a strict superset of the base proof, not a replacement for it: the exact-equivalence result above still holds as the zero-offset special case.

---

## Limits

This repository validates numerical assumptions in software. It does not claim:

- a fabricated optical processor
- measured optical inference accuracy on real tasks
- measured hardware latency or energy consumption
- experimental characterization of noise, insertion loss, or phase drift

Those claims require a physical device, which does not yet exist. The simulation establishes that the underlying mathematics is correct. Experimental validation is the natural next step.
