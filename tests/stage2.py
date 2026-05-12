#!/usr/bin/env python3
"""
Stage 2: run compiled DXNN models on DeepX hardware and report inference metrics.

Discovers .dxnn files under tests/compiled_models/, builds the inference_runner
C++ binary if needed, and reports latency/throughput per model.

Usage:
    python3 tests/stage2.py [options]

Environment:
    DEEPX_RUNTIME_LIB_DIR   Override path to directory containing libRuntimeLibrary.so
    DEEPX_RUNNER_DIR        Override path to directory containing the inference_runner binary
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
COMPILED_DIR = TESTS_DIR / "compiled_models"
RUNTIME_BUILD_DIR = Path(os.environ.get("DEEPX_RUNNER_DIR", str(TESTS_DIR / "runtime" / "build")))
RUNNER_BIN = RUNTIME_BUILD_DIR / "inference_runner"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", metavar="NAME", help="Model names to run (default: all found in compiled_models/)"
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of timed inference runs per model (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs (default: 10)")
    parser.add_argument("--csv", metavar="FILE", help="Write results to a CSV file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _discover_models(filter_names=None) -> list[dict]:
    """Return list of {name, dxnn_path, input_size} for available compiled models."""
    # Import here so the module works even without models.py dependencies
    try:
        from tests.models import TEST_MODELS, input_data_size
    except ImportError:
        from models import TEST_MODELS, input_data_size  # type: ignore

    models = []
    for dxnn in sorted(COMPILED_DIR.glob("*/*.dxnn")):
        name = dxnn.parent.name
        if filter_names and name not in filter_names:
            continue
        if name not in TEST_MODELS:
            print(f"Warning: {name}.dxnn found but not in TEST_MODELS — skipping")
            continue
        models.append(
            {
                "name": name,
                "dxnn_path": dxnn,
                "input_size": input_data_size(name),
                "input_shape": TEST_MODELS[name]["input_shape"],
            }
        )
    return models


def _build_runner() -> bool:
    """Build inference_runner if not already built. Returns True on success."""
    if RUNNER_BIN.exists():
        return True

    build_script = TESTS_DIR / "runtime" / "build-tests.sh"
    if not build_script.exists():
        print(f"Error: build script not found: {build_script}", file=sys.stderr)
        return False

    env = os.environ.copy()
    if "DEEPX_RUNTIME_LIB_DIR" in os.environ:
        extra = ["--runtime-lib-dir", os.environ["DEEPX_RUNTIME_LIB_DIR"]]
    else:
        extra = []

    print("Building inference_runner ...")
    result = subprocess.run(
        ["bash", str(build_script)] + extra,
        cwd=REPO_ROOT,
        env=env,
    )
    return result.returncode == 0


def _run_model(model: dict, runs: int, warmup: int) -> dict | None:
    """Run inference_runner for one model. Returns result dict or None on failure."""
    shape_str = ",".join(str(d) for d in model["input_shape"])
    cmd = [
        str(RUNNER_BIN),
        "--model",
        str(model["dxnn_path"]),
        "--input-size",
        str(model["input_size"]),
        "--input-shape",
        shape_str,
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {model['name']}: {result.stderr.strip()}", file=sys.stderr)
        return None

    line = result.stdout.strip()
    parts = line.split(",")
    if len(parts) != 8:
        print(f"Unexpected output from inference_runner: {line}", file=sys.stderr)
        return None

    return {
        "model": parts[0],
        "runs": int(parts[1]),
        "avg_ms": float(parts[2]),
        "p50_ms": float(parts[3]),
        "p95_ms": float(parts[4]),
        "min_ms": float(parts[5]),
        "max_ms": float(parts[6]),
        "throughput_fps": float(parts[7]),
    }


def _print_table(results: list[dict]) -> None:
    if not results:
        return
    header = (
        f"{'Model':<20} {'Runs':>5} {'Avg ms':>8} {'P50 ms':>8} {'P95 ms':>8} {'Min ms':>8} {'Max ms':>8} {'FPS':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<20} {r['runs']:>5} {r['avg_ms']:>8.2f} {r['p50_ms']:>8.2f} "
            f"{r['p95_ms']:>8.2f} {r['min_ms']:>8.2f} {r['max_ms']:>8.2f} {r['throughput_fps']:>8.1f}"
        )
    print("=" * len(header))


def main() -> int:
    args = _parse_args()

    models = _discover_models(args.models)
    if not models:
        print("No compiled DXNN models found under tests/compiled_models/ — skipping inference tests.")
        return 0

    print(f"Found {len(models)} model(s): {[m['name'] for m in models]}")

    if not _build_runner():
        print("Error: failed to build inference_runner.", file=sys.stderr)
        print("Ensure the runtime library is built:", file=sys.stderr)
        print("  cd runtime-library && ./build-runtimes.sh --ubuntu_version 22.04", file=sys.stderr)
        return 1

    results = []
    for model in models:
        print(
            f"\nRunning {model['name']} (input_size={model['input_size']} bytes, "
            f"runs={args.runs}, warmup={args.warmup}) ..."
        )
        r = _run_model(model, args.runs, args.warmup)
        if r:
            results.append(r)
            print(f"  avg={r['avg_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  " f"throughput={r['throughput_fps']:.1f} fps")
        else:
            print("  FAILED")

    _print_table(results)

    if args.csv and results:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model", "runs", "avg_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_fps"]
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nResults appended to {csv_path}")

    failed = len(models) - len(results)
    if failed:
        print(f"\n{failed} model(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
