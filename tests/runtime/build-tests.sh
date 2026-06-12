#!/bin/bash
# Build the inference_runner binary.
#
# Usage:
#   ./build-tests.sh [--runtime-lib-dir <path>] [--arch x86_64|aarch64] [--build-dir <path>]
#
# If --runtime-lib-dir is not given, the script searches the standard artifact
# directories produced by runtime-library/build-runtimes.sh.
#
# Use --arch aarch64 to cross-compile for aarch64 (requires aarch64-linux-gnu-g++).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACTS_DIR="$REPO_ROOT/runtime-library/artifacts"

RUNTIME_LIB_DIR=""
TARGET_ARCH="${TARGET_ARCH:-$(uname -m)}"
BUILD_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-lib-dir) RUNTIME_LIB_DIR="$2"; shift 2 ;;
        --arch)            TARGET_ARCH="$2"; shift 2 ;;
        --build-dir)       BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$BUILD_DIR" ]] && BUILD_DIR="$SCRIPT_DIR/build"

# Auto-discover library directory if not provided, preferring the target arch
if [[ -z "$RUNTIME_LIB_DIR" ]]; then
    for candidate in \
        "$ARTIFACTS_DIR/${TARGET_ARCH}-ubuntu22.04" \
        "$ARTIFACTS_DIR/${TARGET_ARCH}-ubuntu20.04" \
        "$ARTIFACTS_DIR/${TARGET_ARCH}-ubuntu24.04"
    do
        if [[ -f "$candidate/libRuntimeLibrary.so" ]]; then
            RUNTIME_LIB_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$RUNTIME_LIB_DIR" ]]; then
    echo "Error: libRuntimeLibrary.so not found under $ARTIFACTS_DIR for arch '$TARGET_ARCH'" >&2
    echo "Build the runtime library first:" >&2
    echo "  cd runtime-library && ./build-runtimes.sh --ubuntu_version 22.04" >&2
    exit 1
fi

echo "Using runtime library from: $RUNTIME_LIB_DIR"

# Select compiler based on target arch
CMAKE_EXTRA_ARGS=()
HOST_ARCH="$(uname -m)"
if [[ "$TARGET_ARCH" != "$HOST_ARCH" && "$TARGET_ARCH" == "aarch64" ]]; then
    CROSS_CXX="aarch64-linux-gnu-g++"
    if ! command -v "$CROSS_CXX" &>/dev/null; then
        echo "Error: $CROSS_CXX not found. Install with: apt-get install g++-aarch64-linux-gnu" >&2
        exit 1
    fi
    CMAKE_EXTRA_ARGS+=(
        -DCMAKE_CXX_COMPILER="$CROSS_CXX"
        -DCMAKE_SYSTEM_NAME=Linux
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=NEVER
        -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=NEVER
    )
fi

mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRUNTIME_LIB_DIR="$RUNTIME_LIB_DIR" \
    -DRUNTIME_INCLUDE_DIR="$REPO_ROOT/runtime-library/include" \
    "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "$BUILD_DIR" --config Release

# Symlink the shared library next to the binary so $ORIGIN RPATH works
if [[ ! -f "$BUILD_DIR/libRuntimeLibrary.so" ]]; then
    ln -sf "$RUNTIME_LIB_DIR/libRuntimeLibrary.so" "$BUILD_DIR/libRuntimeLibrary.so"
fi

echo ""
echo "Build complete: $BUILD_DIR/inference_runner"
