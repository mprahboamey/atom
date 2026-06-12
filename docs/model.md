# Model Notes

This document describes the physics and assumptions behind the simulator. It is written to be readable without hiding the math.

---

## Field representation

The simulator stores a monochromatic optical field as a complex tensor:

```
u(x, y) = A(x, y) * exp(1j * phi(x, y))
```

`A` is amplitude. `phi` is phase in radians. A single complex value carries both quantities at each grid point, which keeps the physics compact and the operations straightforward.

Detectors measure intensity, not phase:

```
I(x, y) = abs(u(x, y)) ** 2
```

This is the quantity used in examples whenever the field is read out as an image or matched against a target pattern.

---

## Propagation

Free-space propagation uses the Angular Spectrum Method (ASM).

The idea: any 2D field can be decomposed into a sum of plane waves travelling in different directions. Each plane wave accumulates a known phase shift as it travels forward. ASM computes that shift in the frequency domain, which makes it efficient and exact within the scalar wave approximation.

**Steps:**

1. Take the 2D FFT of the input field.
2. Multiply each spatial-frequency component by a transfer function.
3. Take the inverse FFT to get the output field.

In compact notation:

```
U(fx, fy, 0) = FFT[ u(x, y, 0) ]
U(fx, fy, z) = U(fx, fy, 0) * H(fx, fy, z)
u(x, y, z)   = IFFT[ U(fx, fy, z) ]
```

where the transfer function is:

```
H(fx, fy, z) = exp(1j * kz * z)
kz = sqrt(k² - (2π fx)² - (2π fy)²)
k  = 2π / wavelength
```

Components where `kz²` is negative would correspond to evanescent waves — fields that decay exponentially rather than propagate. The implementation sets their contribution to zero.

**Energy conservation.** A propagating field with no absorbing element should not lose or gain energy. The validation script confirms that total intensity is conserved to within `2e-5` relative error across a range of grid sizes and propagation distances.

---

## Diffractive layers

A diffractive layer is a phase-only mask:

```
u_out(x, y) = u_in(x, y) * exp(1j * theta(x, y))
```

`theta` is a trainable `nn.Parameter`. The mask modifies the wavefront without absorbing energy — intensity at the mask plane is unchanged. After the field propagates, phase differences produce interference patterns that can concentrate or redirect light.

Because `theta` is differentiable, a loss computed from the output intensity can be backpropagated through the propagation step and into the mask parameters. Example 02 demonstrates this: a single phase mask is trained to focus a plane wave onto a Gaussian target spot.

---

## Attention scores via interference

The attention module asks a specific question: can optical interference reproduce the score used by scaled dot-product attention?

The answer is yes, exactly, when real-valued queries and keys are encoded as complex wave amplitudes with sign carried in phase:

- positive values → amplitude with zero phase
- negative values → amplitude with phase π

The interference score between a query wave `q_wave` and a key wave `k_wave` is:

```
score = real( sum(q_wave * conj(k_wave)) ) / sqrt(d)
```

This is algebraically equivalent to:

```
score = q @ k.T / sqrt(d)
```

The validation script confirms the two expressions agree to within `1e-6` absolute error. A normalized mode (dividing each vector by its norm before encoding) produces cosine similarity scores by the same mechanism.

The module also checks sign preservation: a query and its negation should produce a maximally negative correlation score. That check is included because sign information — which encodes semantic opposition — is easy to lose when moving between real and complex representations.

---

## Limits

This repository validates numerical assumptions in software. It does not claim:

- a fabricated optical processor
- measured optical inference accuracy
- measured hardware latency or energy consumption
- a verified large-scale implementation

Those claims would require separate experimental evidence outside the scope of this codebase.
