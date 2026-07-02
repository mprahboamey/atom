# Contributing to ATOM

The math is proved. The simulator works. Everything else is an open problem.

This document breaks down exactly what needs to happen next, by domain. Find your expertise, read your section, fork and build. If you're not sure where you fit, open an issue and say what you do — we'll find a place.

---

## Materials Science

**The core problem:** Photorefractive crystals like lithium niobate store holograms well, but reading erases them. The same light you use to read a hologram also partially overwrites it. Fix this and you've solved one of the oldest problems in holographic storage.

**Open problems:**
- Fixing holograms without destroying angular multiplexing fidelity (thermal fixing, two-photon gating)
- Reducing shrinkage during polymerization in holographic polymer media
- Characterizing M# (dynamic range figure of merit) for candidate materials under dense angular multiplexing
- Modelling SNR degradation as a function of hologram count for lithium niobate and barium titanate

**How to contribute:** Add a `materials/` module with noise and SNR models. Even a calibrated estimate of M# degradation under M=900 multiplexed holograms would close the biggest gap between the geometric capacity ceiling and the realistic usable number.

---

## Integrated Photonics

**The core problem:** Getting light into and out of the crystal fast enough to be useful. Spatial light modulators are slow. Photodetectors add noise. The optical interaction itself is femtojoule-level; everything around it is not.

**Open problems:**
- Input encoding: SLM vs direct waveguide coupling — speed vs precision tradeoff
- Detector array design for reading interference patterns with acceptable SNR
- Coherent source stabilisation: how much thermal drift can the Bragg condition tolerate before a channel becomes unreadable?
- ADC energy budget: analog-to-digital conversion per output channel is a known bottleneck — what's the minimum viable detector count for attention readout?

**How to contribute:** Add to `docs/benchmarks.md` with realistic peripheral energy figures. The current benchmark models only the optical interaction; the periphery is where the real system-level energy lives.

---

## Noise Modelling

**The core problem:** The simulator runs in a noiseless mathematical environment. A real device has phase noise, thermal drift, inter-hologram crosstalk, detector shot noise, and write precision limits. None of these are modelled. Until they are, inference accuracy is completely unknown.

**Open problems:**
- Phase quantisation: what happens to attention scores when `theta` is quantised to N bits?
- Crosstalk model: simulate SNR degradation as angular channel count increases from M=1 to M=900
- Thermal drift: model Bragg angle shift as a function of temperature and crystal coefficient of thermal expansion
- End-to-end noisy inference: run a real task benchmark with a calibrated noise model and report top-1 accuracy

**How to contribute:** Add a `noise/` module. Start with phase quantisation — it's the most tractable and gives immediate signal on how sensitive the attention computation is to imperfect weight encoding. Add tests under `tests/` and document assumptions in `docs/`.

---

## ML Systems

**The core problem:** ATOM proves the attention score computation can happen optically. Everything else — softmax, layer norm, residual addition, value aggregation — still runs digitally. Figuring out how to compose optical attention with a real transformer inference stack is unsolved.

**Open problems:**
- Weight conversion pipeline: given any HuggingFace model, produce the phase mask values for each layer ready for crystal writing
- Hybrid inference stack: optical QK scores + digital softmax/V aggregation — profile latency and energy for a real model
- Quantisation-aware weight encoding: what's the minimum phase precision needed to maintain attention accuracy within 1% of digital?
- Forward-pass-only training: can the optical path be trained without backpropagation? (See gradient estimation literature)

**How to contribute:** The weight conversion stub lives in `atom/attention.py`. A full pipeline that takes a `.safetensors` or `.gguf` checkpoint and outputs phase mask values would be the highest-leverage ML contribution possible right now.

---

## Theory

**The core problem:** The attention–interference equivalence is proved for the linear QK score. Softmax is not optical. Value aggregation is not optical. Multi-head attention is not optical. Extending the theoretical framework to cover more of the transformer forward pass is wide open.

**Open problems:**
- Optical softmax approximations: is there a physical mechanism that produces softmax-like normalisation without digital conversion?
- Multi-head attention in a single crystal volume: can multiple heads be addressed by different angular sub-bands?
- Optical activation functions: what nonlinearities are physically achievable in photorefractive media?
- Training without backprop: forward-pass-only gradient estimation (SPSA, perturbation methods) for updating holographic weights without a full differentiable path

**How to contribute:** Open a discussion issue with a proof sketch. If you can prove or disprove any of the above, that is a paper, not just a PR.

---

## General

- **Issues** for questions, bugs, or pointing out something that doesn't add up
- **Discussions** for open-ended ideas, architecture questions, or "I work in X and I think I can help"
- **PRs** for actual code, models, or documentation

If you find an error in the math, open an issue. If the proof is wrong somewhere, that is the most important contribution possible and it will be credited as such.
