# Entry Points

Three categories of files live in this repository.

---

## Library code

These modules are imported by examples and tests. They are not meant to be run directly.

| Path | Purpose |
|---|---|
| `optic/propagation.py` | Complex field helpers and Angular Spectrum Method propagation |
| `optic/diffractive.py` | Trainable phase masks and stacked diffractive network |
| `optic/attention.py` | Interference-based attention score computation |

---

## Examples

Run from the repository root. Each script exercises one concept independently.

| Command | What it does | Output |
|---|---|---|
| `python examples/01_propagate_beam.py` | Propagates a Gaussian beam 1 mm through free space | Total and peak intensity before and after propagation |
| `python examples/02_train_phase_mask.py` | Trains a phase mask to focus light onto a Gaussian target | Loss at steps 0, 20, 40, 60, 79 |
| `python examples/03_optical_attention.py` | Computes optical and digital attention scores and compares them | Max absolute difference between the two |
| `python examples/04_validate_model.py` | Runs all numerical checks with explicit tolerances | Pass/fail value and limit for each check |

---

## Full validation

Run this before sharing or submitting the repository:

```bash
python scripts/run_all.py
```

Runs every example and the unit test suite. Writes:

| Path | Contents |
|---|---|
| `results/latest_run.json` | Summary of the most recent run |
| `results/logs/*.txt` | Captured stdout from each command |
| `results/expected.json` | Expected checks and their source locations |
