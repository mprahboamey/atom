# Validation audit (senior pass)

Cross-check of claims in README / docs / code against math and literature.
No hardware was measured for this note.

---

## 1. Binary-phase scores = digital attention

**Claim in places:** optical scores verified equal to scaled dot-product.

**Fact:** For phase ∈ {0, π},  
`Re(q_wave conj(k_wave)) = q·k` **identically**.  
Random-tensor max error in tests: 0.

**Caveat:** This is not an experimental discovery about light.  
**Literature:** Standard complex encoding; any two-level phase product recovers the real product.

**Implication for docs:** Say **by construction** + **hybrid wiring verified on safetensors**.  
Do not imply a crystal was tested.

---

## 2. SmolLM2 optical == digital generate

**Fact:** Software integration on real HF weights (30 layers) matched greedy tokens for the logged prompts.

**Caveat:** Proves stack consistency under the binary score function, not optical physics, energy, or latency.

**Secondary caveat:** Logits can show small MSE under float16 with 100% top-1; do not claim bit-identical logits unless measured in float32.

---

## 3. Geometric “90T” capacity

**Fact in benchmarks:**  
`1000 depth × 900 angles × 10^8 pixels ≈ 9×10^13` under **assumed** 1 µm pixels, 10 µm layers, 0.1° steps, 90° range, **infinite dynamic range**.

**Caveat A — M#:** Usable holograms obey η ≈ (M#/M)² (Mok / Psaltis).  
Typical Fe:LiNbO₃: M# often **O(1)–O(10)** depending on thickness, doping, geometry; 90° geometry often **worse** than transmission (literature cites ~1–2 scale for 90° Fe:LN class).  
**Code:** `capacity.usable_capacity` applies this; geometric ceiling must never be sold as deliverable capacity.

**Caveat B — Angular channels are not free:** Bragg selectivity width scales with thickness and index; **0.1° × 900 independent channels is a modeling assumption**, not a measured comb for 1 cm³ in this repo.

**Caveat C — Noise / scatter / BER:** Storage density papers show practical density << geometric bound once SNR and crosstalk matter.

**Loop:** Geometric bound → M# limit → geometry- and SNR-limited density → still **storage**, not automatic **optical matmul energy** advantage.

---

## 4. Refractive index in benchmarks

**Doc had:** n ≈ 1.5 “typical photorefractive.”  
**LiNbO₃:** n_o / n_e roughly **~2.2 / ~2.3** in the visible (order of magnitude).  
1.5 understates LN; fine as a generic placeholder, **wrong as “LiNbO₃.”** Corrected in benchmarks language.

---

## 5. Bragg formula wording

Benchmarks showed `2 · d_grating · sin(θ) = m · λ`.  
In-medium form is `2 n Λ sin(θ_B) = m λ` (definitions of θ and Λ vary by geometry).  
Treat as **schematic**, not a design equation for 0.1° spacing.

---

## 6. Destructive readout / refresh

**Fact:** Same-wavelength readout can erase unfixed photorefractive gratings.  
**Software:** `atom/refresh.py` uses η(n) ≈ η0 exp(−n/n_erase) with **placeholder** n_erase (e.g. 500).  
**Caveat:** n_erase is **not** calibrated to a published erase curve for a specific crystal/intensity.  
**Literature path:** thermal fixing, two-color / gated recording → non-volatility tradeoffs vs sensitivity and M# (Adibi / Buse / Psaltis and related work).

---

## 7. Continuous-phase “angular multiplexing” in software

**Fact:** `encode_angular_phase` is **RoPE-structured** (`position * base^(-i/dim)`).  
**Caveat:** Relative-position effects are expected math, **not** a measured Bragg angular multiplex experiment.

---

## 8. “Speed of light through the crystal” leap

**Old risk in benchmarks:** correct interference math ⇒ physical device does attention at c.  
**Reject as written:** ignores detection, SLM, electronics, SNR, multiplexing schedule, and that **hybrid** softmax/MLP remain digital.  
Audit rule: math correctness ≠ system latency claim.

---

## 9. FPGA path

**Fact:** `fpga/` is a **scaffold** (CPU parity + HLS stubs).  
**Caveat:** No bitstream, no joules, no board run in this repository.

---

## 10. Rack / multi-crystal

**Fact:** `atom/rack.py` plans shards and logical interconnects.  
**Caveat:** No distributed runtime, no optical/electrical link model with measured bandwidth.

---

## Claim cheatsheet (use in README / external talk)

| Statement | Allowed? |
|-----------|----------|
| Binary-phase scores equal digital scores by algebra | Yes |
| Hybrid software matches digital greedy on SmolLM2 for logged runs | Yes |
| Geometric 90T is an infinite-dynamic-range ceiling under stated assumptions | Yes, if labeled |
| Usable capacity is M#-limited and defaults are placeholders | Yes |
| Photorefractive optical LLM demonstrated | **No** |
| FPGA energy measured | **No** |
| Continuous-phase path = lab Bragg multiplex | **No** |
| n_erase = 500 is measured | **No** |

---

## Residual open (software vs physics)

**Software can still add:** small-weight write scaling, rack↔refresh control loop, staged hybrid sim, stronger noise models.  
**Physics must supply:** measured M#, erase curves, Bragg comb, write SNR for *your* medium.
