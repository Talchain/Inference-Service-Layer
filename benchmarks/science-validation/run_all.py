"""One-command runner for the science-validation harness.

Runs the four experiments in priority order, each as a separate process, and
prints a pass/fail summary. Results land in benchmarks/science-validation/
results/ with full provenance (command, git SHA, seeds).

Usage:
    poetry run python benchmarks/science-validation/run_all.py --quick   # smoke (~2 min)
    poetry run python benchmarks/science-validation/run_all.py          # full  (~20-30 min)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent

EXPERIMENTS = [
    "exp1_higher_k.py",
    "exp2_evpi_floor.py",
    "exp3_determinism.py",
    "exp4_calibration.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced grids for smoke testing")
    args = parser.parse_args()

    flags = ["--quick"] if args.quick else []
    failures = []
    t0 = time.time()
    for script in EXPERIMENTS:
        print(f"\n=== {script} {' '.join(flags)} ===", flush=True)
        proc = subprocess.run([sys.executable, str(HARNESS_DIR / script), *flags], cwd=REPO_ROOT)
        if proc.returncode != 0:
            failures.append(script)

    elapsed = time.time() - t0
    print(f"\n=== run_all complete in {elapsed:.0f}s ===")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
