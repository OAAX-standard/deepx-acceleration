#!/usr/bin/env python3
"""
Stage 1 orchestrator: converts ONNX models to DXNN using the DeepX toolchain.

Runs the conversion test suite, which downloads models from the DeepX Model Zoo
and converts them to DXNN format. Output goes to tests/compiled_models/.

Usage:
    uv run python tests/stage1.py              # convert all models
    uv run python tests/stage1.py -k mobilenet # convert only models matching 'mobilenet'
    uv run python tests/stage1.py -v           # verbose output

Any extra arguments are forwarded directly to pytest.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR / "test_conversion.py"),
    ] + sys.argv[1:]
    print(f"\n{'='*60}")
    print("Running: " + " ".join(cmd))
    print("=" * 60 + "\n")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
