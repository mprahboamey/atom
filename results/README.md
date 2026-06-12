# Results

This folder is where validation output is written.

Run:

```bash
python scripts/run_all.py
```

The command creates:

- `results/latest_run.json`: machine-readable run summary
- `results/logs/`: stdout and stderr from each runnable script

`expected.json` records the checks that should pass. The generated files are
local run artifacts and may vary slightly by Python, PyTorch, CPU, or GPU.
