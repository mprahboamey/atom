"""Run all public examples and validation checks.

The script writes captured output to `results/` so a reviewer can see what was
run and what each command printed.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"

COMMANDS = [
    {
        "name": "example_01_propagate_beam",
        "kind": "example",
        "command": [sys.executable, "examples/01_propagate_beam.py"],
    },
    {
        "name": "example_02_train_phase_mask",
        "kind": "example",
        "command": [sys.executable, "examples/02_train_phase_mask.py"],
    },
    {
        "name": "example_03_optical_attention",
        "kind": "example",
        "command": [sys.executable, "examples/03_optical_attention.py"],
    },
    {
        "name": "example_04_validate_model",
        "kind": "validation",
        "command": [sys.executable, "examples/04_validate_model.py"],
    },
    {
        "name": "example_05_phase_quantization_sweep",
        "kind": "example",
        "command": [sys.executable, "examples/05_phase_quantization_sweep.py"],
    },
    {
        "name": "unit_tests",
        "kind": "test",
        "command": [sys.executable, "-m", "unittest", "discover", "tests"],
    },
]


def torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def run_command(item: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(
        item["command"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.perf_counter() - start

    log_text = ""
    if proc.stdout:
        log_text += proc.stdout
    if proc.stderr:
        if log_text:
            log_text += "\n"
        log_text += "[stderr]\n" + proc.stderr

    log_path = LOGS / f"{item['name']}.txt"
    log_path.write_text(log_text, encoding="utf-8")

    return {
        "name": item["name"],
        "kind": item["kind"],
        "command": " ".join(item["command"]),
        "returncode": proc.returncode,
        "duration_seconds": round(duration, 4),
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    runs = [run_command(item) for item in COMMANDS]
    passed = all(run["returncode"] == 0 for run in runs)

    report = {
        "passed": passed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version(),
        "runs": runs,
    }
    (RESULTS / "latest_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for run in runs:
        status = "PASS" if run["returncode"] == 0 else "FAIL"
        print(f"{status} {run['name']} ({run['duration_seconds']}s)")

    print(f"\nReport: {RESULTS.relative_to(ROOT) / 'latest_run.json'}")
    print(f"Logs:   {LOGS.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
