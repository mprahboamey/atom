# Validation audit (senior pass)

Cross-check of claims in README / docs / code against math and literature.
No hardware was measured in this repository.

For external photonic/transformer chips and Fe:LN M#, see also [`related_work_hardware.md`](related_work_hardware.md).

---

## 1. Binary-phase scores = digital attention

**Claim in places:** optical scores verified equal to scaled dot-product.

**Fact:** For phase ∈ {0, π},
`Re(q_wave conj(k_wave)) = q·k` **identically**.
Random-tensor max error in tests: 0.

**Caveat:** This is not an experimental discovery about light.
**Literature:** Standard complex encoding; any two-level phase product recovers the real product.

**Related hardware (2025):** Tian et al. (PhotoniX) demonstrate interference-based attention on a **silicon photonic** chip with mean attention-matrix error ~8.9×10⁻⁴. That supports “interference attention on real chips,” **not** equivalence to ATOM’s binary-phase = `QKᵀ/√d` contract (they use a different KKA mechanism).

**Implication for docs:** Say exact software identity + hybrid wiring on safetensors. Cite PTC as related photonics, not as our crystal result.

---

## 2. SmolLM2 optical == digital generate

**Fact:** Software integration on real HF weights (30 layers) matched greedy tokens for the logged prompts.

**Caveat:** Proves stack consistency under the binary score function, not optical physics, energy, or latency.

**Secondary caveat:** Logits can show small MSE under float16 with 100% top-1; do not claim bit-identical logits unless measured in float32.

---

## 3. Geometric “90T” capacity

**Fact in benchmarks:**
`1000 depth × 900 angles × 10^8 pixels ≈ 9×10^13` under **assumed** 1 µm pixels, 10 µm layers, 0.1° steps, 90° range, **infinite dynamic range**.

**Caveat A — M#:** Usable holograms obey η ≈ (M#/M)² (Mok / Psaltis). Typical Fe:LiNbO₃ **90°** geometry: M# often **~O(1–few) per cm** in published tables; transmission geometry often higher. Code default `m_number=2.0` is a **conservative literature-class placeholder**, not a lab measurement in this repo.

**Caveat B — Angular channels are not free:** Bragg selectivity width scales with thickness and index; **0.1° × 900 independent channels is a modeling assumption**, not a measured comb for 1 cm³ here.

**Caveat C — Noise / scatter / BER:** Storage density papers show practical density << geometric bound once SNR and crosstalk matter.

**Loop:** Geometric bound → M# limit → geometry- and SNR-limited density → still **storage**, not automatic **optical matmul energy** advantage.

---

## 4. Refractive index in benchmarks

**LiNbO₃:** n_o / n_e roughly **~2.2 / ~2.3** in the visible. Do not label 1.5 as LiNbO₃.

---

## 5. Bragg formula wording

Schematic in-medium form: `2 n Λ sin(θ_B) ≈ m λ`. Treat as schematic, not a design equation for 0.1° spacing.

---

## 6. Destructive readout / refresh

**Fact:** Same-wavelength readout can erase unfixed photorefractive gratings.
**Software:** `atom/refresh.py` uses η(n) ≈ η0 exp(−n/n_erase) with **placeholder** n_erase.
**Caveat:** n_erase is **not** calibrated to a published erase curve for a specific crystal/intensity.
**Literature path:** thermal fixing, two-color / gated recording → non-volatility tradeoffs vs sensitivity and M#.

---

## 7. Continuous-phase “angular multiplexing” in software

**Fact:** `encode_angular_phase` is **RoPE-structured** (`position * base^(-i/dim)`).
**Caveat:** Relative-position effects are expected math, **not** a measured Bragg angular multiplex experiment.

---

## 8. “Speed of light through the crystal” leap

Correct interference math ≠ system latency claim. Detection, modulation, hybrid digital stages, and SNR dominate real systems.

---

## 9. FPGA path

**Fact:** `fpga/` is a **scaffold** (CPU parity + HLS stubs).
**Caveat:** No bitstream, no joules, no board run in this repository.

---

## 10. Rack / multi-crystal

**Fact:** `atom/rack.py` plans shards and logical interconnects.
**Caveat:** No distributed runtime, no measured link model.

---

## Claim cheatsheet

| Statement | Allowed? |
|-----------|----------|
| Binary-phase scores equal digital scores by algebra | Yes |
| Hybrid software matches digital greedy on SmolLM2 for logged runs | Yes |
| Interference-based attention exists on fabricated SiPh chips (e.g. Tian 2025) | Yes, with citation |
| That chip is our binary-phase / Fe:LN pipeline | **No** |
| Geometric 90T is an infinite-dynamic-range ceiling under stated assumptions | Yes, if labeled |
| M#≈2 is a conservative 90° Fe:LN literature-class default | Yes, as placeholder |
| M#=2 measured on our crystal | **No** |
| Photorefractive optical LLM demonstrated | **No** |
| FPGA energy measured | **No** |
