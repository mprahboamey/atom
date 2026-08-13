# Related hardware (verified against sources)

This note records what external results **do** and **do not** settle for ATOM.

---

## 1. Optical interference and attention on real chips

### What is real

**Tian et al., PhotoniX (2025)** — *Photonic transformer chip: interference is all you need*
https://doi.org/10.1186/s43074-025-00182-7

- Fabricated **silicon photonic** transformer chip (PTC).
- Attention via runtime-programmable **optical interference** (Kramers–Kronig attention / KKA).
- Measured mean error **≈ 8.9×10⁻⁴** across **10,000** attention-matrix elements.
- Small prototype; MNIST / ViT-style demo — not a full LLM stack.

### What this does **not** mean for ATOM

| Claim | PTC (Tian 2025) | ATOM software path |
|-------|-----------------|---------------------|
| Interference used for attention-like scores | Yes | Yes (target) |
| Encoding = binary phase {0, π} so scores **equal** `QKᵀ/√d` exactly | **No** | **Yes** |
| Photorefractive Fe:LiNbO₃ volume holograms | **No** | Capacity/refresh story |
| Hybrid LLM on HF weights | **No** | Software hybrid yes |

---

## 2. Fe:LiNbO₃ and M#

Holographic storage literature reports **M#** with η ≈ (M#/M)².
For **90° geometry**, lightly Fe-doped LiNbO₃, published typical values are on the
order of **~2 cm⁻¹**. Transmission geometry often much higher.

ATOM `m_number = 2.0` is a **conservative literature-class placeholder**, not a
measurement in this repo. Watch **per cm** vs total-crystal normalization.

---

## 3. Readout erase / decay shape and regimes

### Exponential form

Under continuous readout illumination, photorefractive grating strength in
Fe:LiNbO₃-class media is widely modeled as **exponential** in time (or fluence),
with τ inversely related to intensity. After absorption corrections, many
experimental erase curves fit monoexponential forms. Some multiplex-write
studies use **stretched** exponentials; the ordinary exponential is the
`β = 1` special case and remains a sound default for ops planning.

`atom/refresh.py`: `eta(n) ≈ eta0 * exp(-n / n_erase)` — **right shape**,
placeholder scale.

### Regime dependence (orders of magnitude)

- **Unfixed single-color:** continuous read can erase usable diffraction efficiency
  on practical timescales; aggressive refresh / reinjection is a real systems problem.
- **Two-color / gated recording:** much higher read durability. Example:
  **Lee et al., Appl. Phys. Lett. 81, 4511 (2002)** — two-color multiplexing in
  stoichiometric LiNbO₃:Tb,Fe; estimated **~80 million** readouts at 1 Gbit/s before
  diffracted intensity falls to half.
- **Fixing (thermal/ionic):** further persistence options with process tradeoffs.

Default `n_erase=500` is scoped as **unfixed single-color planning**, not a
universal material constant. Choosing two-color or fixing is a **product design
decision** that can reduce reinjection pressure dramatically.

---

## 4. Fit to our pipeline

```text
ATOM software:  binary-phase scores ≡ digital QKᵀ/√d
PTC hardware:   interference attention (KKA) on SiPh — related, different encoding
Fe:LN M#:       storage dynamic range — capacity.py
Erase model:    exponential form OK; n_erase regime-dependent — refresh.py
```
