#!/usr/bin/env bash
# Template: run on the FPGA instance after xclbin is loaded.
# Replace DEVICE_ID / XCLBIN / host binary with your build outputs.
set -euo pipefail

OUT_JSON="${1:-fpga/results/fpga_benchmark.json}"
mkdir -p "$(dirname "$OUT_JSON")"

echo "1. Load xclbin / agfi for this instance"
echo "2. Run host that enqueues score_kernel N times"
echo "3. Read board power (vendor tool or external meter)"
echo "4. Write JSON with the same keys as fpga/host/run_benchmark.py"

cat <<EOF
Example JSON fields:
  device, backend, batch, seq, dim, repeat,
  total_seconds, seconds_per_call, ms_per_call,
  energy_joules_total, energy_joules_per_call, notes
EOF

echo "Template only — no FPGA detected in this environment."
