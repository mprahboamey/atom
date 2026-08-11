# Cloud FPGA steps (you complete on laptop / cloud)

## Goal

Produce one real measurement file with the same schema as
`fpga/results/cpu_benchmark.json`, but `backend: "fpga"` and non-null energy
if the platform exposes power.

## Suggested path (AWS-shaped)

1. **Account** with FPGA instance access (e.g. F2 / legacy F1) and quota.
2. **FPGA Developer AMI** (or vendor image) with Vitis.
3. **HLS** `fpga/hls/score_kernel.cpp` → `.xo` → link → `xclbin`.
4. **Host** C/C++ or Python+XOCL that:
   - copies Q,K to device
   - enqueues `score_kernel`
   - copies scores back
   - times wall clock for N repeats
5. **Power**: instance metrics, `xbutil`, or external PDU — convert average
   watts × seconds → `energy_joules_total`.
6. **Parity**: same Q,K on CPU (`parity_check.py` / host CPU path) vs FPGA
   outputs; max abs error gate (e.g. 1e-3 for fp32).
7. **Scale**: second instance or multi-device when ready; plot ms and J vs N.

## What this repo already guarantees

- Score math matches `atom.attention.optical_scores` on CPU
- Metric schema for apples-to-apples CPU vs FPGA logs

## What this repo cannot do from CI alone

- Synthesize a bitstream without Vitis
- Boot your cloud FPGA
- Read physical power rails
