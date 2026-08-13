# Related hardware (verified against sources)

This note records what external results **do** and **do not** settle for ATOM.

---

## 1. Optical interference and attention on real chips

### What is real

**Tian et al., PhotoniX (2025)** — *Photonic transformer chip: interference is all you need*  
https://doi.org/10.1186/s43074-025-00182-7

- Fabricated **silicon photonic** transformer chip (PTC).
- Attention via runtime-programmable **optical interference** (their “Kramers–Kronig attention” / KKA).
- Measured mean error **≈ 8.9×10⁻⁴** across **10,000** attention-matrix elements (10×10×100 samples); they relate this to ~8-bit-class precision.
- Small prototype (e.g. 10×1 interference unit array); MNIST / ViT-style demo, not a full LLM stack.

Broader context: optical **dot-product / MAC** hardware has multiple prior demonstrations; interference-based linear algebra is not science fiction.

### What this does **not** mean for ATOM

| Claim | PTC (Tian 2025) | ATOM software path |
|-------|-----------------|---------------------|
| Interference used for attention-like scores | Yes | Yes (target) |
| Encoding = binary phase {0, π} so scores **equal** `QKᵀ/√d` exactly | **No** — KKA uses amplitude–phase coupling; SoftMax-free variant | **Yes** — that identity is our score contract |
| Photorefractive Fe:LiNbO₃ volume holograms | **No** — integrated silicon photonics | Storage story uses Fe:LN-class media in capacity/refresh stubs |
| Hybrid LLM on HF weights | **No** | Software hybrid yes; optical loop not built |

**Correct use in narrative:**  
“Optical interference can implement attention-related computation on **real fabricated photonics**, with published error ~10⁻³ on attention matrix elements.”  

**Incorrect use:**  
“A 2025 chip already proved ATOM’s binary-phase score path / crystal pipeline.”

---

## 2. Fe:LiNbO₃ and M#

### What is real

Holographic storage literature (including Caltech-era work on LiNbO₃:Fe, 90° vs transmission geometry) reports **M#** as the dynamic-range figure of merit with

```text
η ≈ (M# / M)²
```

for equalized multiplexed holograms.

For **90° geometry**, lightly iron-doped LiNbO₃, published **typical** values are on the order of **~2 cm⁻¹** (e.g. summaries citing ~2.0 cm⁻¹; measured samples in the same literature often ~2–3+ cm⁻¹ depending on doping and thickness). **Transmission geometry** often shows **much larger** M# (roughly an order of magnitude higher in comparative tables).

ATOM’s `CapacityParams.m_number = 2.0` is a **conservative placeholder** aligned with that **90° Fe:LN order of magnitude**, not a measurement performed in this repo on a specific 5 mm sample.

### Units caution

Papers often quote **M# per cm** of thickness. A “M# = 2” headline may mean **2 cm⁻¹**, so a 5 mm (0.5 cm) crystal is not automatically “total M# = 2” without checking the paper’s normalization. Keep defaults labeled **placeholder / literature-class**, and replace with a cited lab curve when you have one.

### What this does **not** mean

- Multiplexing and M# for Fe:LN are **established** for **storage**.
- That does **not** by itself deliver optical LLM inference or ATOM’s hybrid score path in crystal.

---

## 3. Fit to our pipeline

```text
ATOM software:  binary-phase scores ≡ digital QKᵀ/√d  →  hybrid HF generate
PTC hardware:   interference attention (KKA) on SiPh chip     →  related, different encoding
Fe:LN M#:       bounds holographic storage channels          →  capacity.py / refresh planning
```

External work **strengthens** the case that (a) interference-based attention is experimentally real and (b) Fe:LN M# defaults are not pulled from thin air.  
It does **not** close ATOM’s remaining gap: **our** encoding and **our** hybrid loop on **our** weights under **our** medium.
