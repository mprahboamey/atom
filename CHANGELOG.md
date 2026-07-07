# Changelog

All notable changes to ATOM will be documented here.

## [Unreleased]
### Added
- `encode_angular_phase` and `optical_scores_general`: continuous-phase attention that reduces exactly to the binary-phase proof as a special case, and produces genuine relative-position-sensitive interference otherwise
- `atom/noise.py`: `quantize_phase`, modeling finite SLM/crystal phase write precision
- M3.1 experiment (`examples/05_phase_quantization_sweep.py`): measures attention fidelity vs phase bit-depth; ~8 bits sufficient for ~99.5% top-1 agreement with the ideal continuous case on synthetic data, single architecture, no other hardware noise sources included yet
### Changed
- Documentation (`docs/model.md`, `README.md`) now explicitly discloses that the base proof's binary phase (0/π) is the unique phase choice required for exact term-wise equivalence to dot-product attention, not an incidental simplification

## [0.1.1] - 2026-07-02
### Changed
- Added project URLs (homepage, repository, bug tracker) to package metadata

## [0.1.0] - 2026-06-19
### Added
- Initial public release
- Exact proof: optical wave interference = scaled dot-product attention, verified to floating-point precision
- Angular Spectrum Method propagation with energy conservation < 2.3×10⁻⁷ relative error
- Trainable diffractive phase masks
- Interference-based attention scoring (`optical_scores`, `OpticalSelfAttention`)
- Full validation suite across embedding dimensions d=8 to d=512
- Architecture projections: volumetric capacity, latency, energy vs H100
- Examples: beam propagation, phase mask training, optical attention, full validation
