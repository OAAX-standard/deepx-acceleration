#!/bin/bash
# Build the inference_runner binary.
#
# Usage:
#   ./build-tests.sh [--runtime-lib-dir <path>]
#
# If --runtime-lib-dir is not given, the script searches the standard artifact
# directories produced by runtime-library/build-runtimes.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ARTIFACTS_DIR="$REPO_ROOT/runtime-library/artifacts"

RUNTIME_LIB_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-lib-dir) RUNTIME_LIB_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Auto-discover library directory if not provided
if [[ -z "$RUNTIME_LIB_DIR" ]]; then
    for candidate in \
        "$ARTIFACTS_DIR/x86_64-ubuntu22.04" \
        "$ARTIFACTS_DIR/x86_64-ubuntu20.04" \
        "$ARTIFACTS_DIR/x86_64-ubuntu24.04" \
        "$ARTIFACTS_DIR/aarch64-ubuntu22.04" \
        "$ARTIFACTS_DIR/aarch64-ubuntu20.04" \
        "$ARTIFACTS_DIR/aarch64-ubuntu24.04"
    do
        if [[ -f "$candidate/libRuntimeLibrary.so" ]]; then
            RUNTIME_LIB_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$RUNTIME_LIB_DIR" ]]; then
    echo "Error: libRuntimeLibrary.so not found under $ARTIFACTS_DIR" >&2
    echo "Build the runtime library first:" >&2
    echo "  cd runtime-library && ./build-runtimes.sh --ubuntu_version 22.04" >&2
    exit 1
fi

echo "Using runtime library from: $RUNTIME_LIB_DIR"

mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRUNTIME_LIB_DIR="$RUNTIME_LIB_DIR" \
    -DRUNTIME_INCLUDE_DIR="$REPO_ROOT/runtime-library/include"

cmake --build "$BUILD_DIR" --config Release

# Symlink the shared library next to the binary so $ORIGIN RPATH works
if [[ ! -f "$BUILD_DIR/libRuntimeLibrary.so" ]]; then
    ln -sf "$RUNTIME_LIB_DIR/libRuntimeLibrary.so" "$BUILD_DIR/libRuntimeLibrary.so"
fi

echo ""
echo "Build complete: $BUILD_DIR/inference_runner"
