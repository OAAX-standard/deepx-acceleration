#!/bin/bash
# Environment-variable-driven wrapper for stage2.py.
#
# Configuration (all optional):
#   MODELS              Space-separated list of model names to run (default: all)
#   RUNS                Number of timed inference runs per model (default: 100)
#   WARMUP              Number of warmup runs (default: 10)
#   PIPELINE_DEPTH      In-flight inference requests (default: 4)
#   CSV                 Path to output CSV file (default: no CSV)
#   LOG_LEVEL           Logging verbosity: DEBUG|INFO|WARNING|ERROR (default: INFO)
#   DEEPX_RUNTIME_LIB_DIR  Path to directory with libRuntimeLibrary.so
#
# Example:
#   MODELS="yolo8n yolo11n" RUNS=200 PIPELINE_DEPTH=8 CSV=results.csv ./tests/run_stage2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-10}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-4}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$REPO_ROOT"

# Activate virtualenv if present
if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

ARGS=(
    --runs "$RUNS"
    --warmup "$WARMUP"
    --pipeline-depth "$PIPELINE_DEPTH"
    --log-level "$LOG_LEVEL"
)

if [[ -n "$MODELS" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(--models $MODELS)
fi

if [[ -n "$CSV" ]]; then
    ARGS+=(--csv "$CSV")
fi

python3 tests/stage2.py "${ARGS[@]}"
