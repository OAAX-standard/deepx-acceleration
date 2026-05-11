#!/usr/bin/env python3
"""
Stage 1 orchestrator: converts ONNX models to DXNN using the DeepX toolchain.

Runs pytest test suites that download models and convert them.
Output goes to tests/compiled_models/.

Usage:
    python3 tests/stage1.py [--yolo]

Options:
    --yolo    Also convert YOLO models (requires ultralytics)
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent


def run_pytest(extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-s",
        str(TESTS_DIR / "test_conversion.py"),
    ] + extra_args
    print(f"\n{'='*60}")
    print("Running: " + " ".join(cmd))
    print("=" * 60 + "\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", action="store_true", help="Also convert YOLO models (requires ultralytics)")
    args = parser.parse_args()

    # Always run classification models
    rc = run_pytest(["-k", "Classification"])
    if rc != 0:
        return rc

    if args.yolo:
        rc = run_pytest(["-k", "Yolo"])

    return rc


if __name__ == "__main__":
    sys.exit(main())
