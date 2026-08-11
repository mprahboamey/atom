# FPGA hybrid digital path (scaffold)

**Scope:** real silicon path for the *digital* half of hybrid inference
(score stand-in, softmax, residual plumbing). **No optics.**

Goal: measure latency and energy on cloud or local FPGA so scaling is not
only a PyTorch simulation.

## Layout

```text
fpga/
  README.md                 this file
  host/run_benchmark.py     CPU reference + metrics log format
  host/parity_check.py      compare CPU kernel vs atom.attention scores
  hls/score_kernel.cpp      HLS-oriented QK^T style score kernel
  hls/softmax_kernel.cpp    naive softmax stub for HLS experiments
  scripts/measure_template.sh   what to capture on the board
  docs/cloud_fpga_steps.md  AWS-style deploy checklist
```

## What runs *here* without an FPGA

```bash
# from repo root
python fpga/host/parity_check.py
python fpga/host/run_benchmark.py --seq 64 --dim 64 --repeat 50
```

Writes `fpga/results/cpu_benchmark.json` (create dir as needed).

## What needs your laptop / cloud

1. Xilinx Vitis / Vivado (or vendor cloud build)
2. AWS F2/F1 (or other) FPGA instance + FPGA Developer AMI
3. Synthesize `hls/*.cpp` (or rewrite as RTL), package xclbin / agfi
4. Run host against the board, fill the same JSON fields with real `joules` / `ms`

See `docs/cloud_fpga_steps.md`.

## Relation to ATOM software

| Component | Today | FPGA target |
|-----------|--------|-------------|
| Binary-phase scores (= dot product) | `atom.attention.optical_scores` | `score_kernel` |
| Softmax + rest of hybrid | PyTorch | later kernels / host |
| Crystal / Bragg | not this tree | never on FPGA |
