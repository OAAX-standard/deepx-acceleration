#!/bin/bash
# Create a self-contained stage2 archive for execution on a remote machine.
#
# Builds runtime libraries and inference_runner for all supported Linux
# architectures (x86_64, aarch64), then bundles everything with the compiled
# DXNN models into a single archive.
#
# Usage:
#   ./tests/package_stage2.sh [--output <archive.tar.gz>] [--ubuntu-version <ver>]
#
# Options:
#   --output <path>         Output archive path (default: deepx-stage2-<version>.tar.gz)
#   --ubuntu-version <ver>  Ubuntu version to build against (default: 20.04)
#                           Ubuntu 20.04 gives the widest glibc compatibility.
#
# The resulting archive can be unpacked and run on a target machine:
#   tar -xzf deepx-stage2-<version>.tar.gz
#   cd deepx-stage2-<version>
#   ./run_stage2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILED_DIR="$SCRIPT_DIR/compiled_models"
VERSION_FILE="$REPO_ROOT/VERSION"
ARTIFACTS_DIR="$REPO_ROOT/runtime-library/artifacts"

OUTPUT=""
UBUNTU_VERSION="20.04"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)         OUTPUT="$2"; shift 2 ;;
        --ubuntu-version) UBUNTU_VERSION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

VERSION="$(cat "$VERSION_FILE" 2>/dev/null || echo "dev")"
PKG_NAME="deepx-stage2-${VERSION}"
[[ -z "$OUTPUT" ]] && OUTPUT="$REPO_ROOT/${PKG_NAME}.tar.gz"

ARCHS=("x86_64" "aarch64")

# --- Sanity checks ---

DXNN_COUNT=$(find "$COMPILED_DIR" -name "*.dxnn" 2>/dev/null | wc -l)
if [[ "$DXNN_COUNT" -eq 0 ]]; then
    echo "Error: no .dxnn files found under $COMPILED_DIR" >&2
    echo "Run stage 1 first: uv run python tests/stage1.py" >&2
    exit 1
fi

# --- Build runtime libraries for all architectures ---

echo "========================================="
echo "Building runtime libraries (ubuntu${UBUNTU_VERSION})"
echo "========================================="

cd "$REPO_ROOT/runtime-library"

for arch in "${ARCHS[@]}"; do
    echo ""
    echo "--- Building libRuntimeLibrary.so for $arch ---"
    if [[ "$arch" == "aarch64" ]]; then
        ./build-runtimes.sh --ubuntu_version "$UBUNTU_VERSION" --cross-aarch64
    else
        ./build-runtimes.sh --ubuntu_version "$UBUNTU_VERSION"
    fi
done

cd "$REPO_ROOT"

# --- Build inference_runner for all architectures ---

echo ""
echo "========================================="
echo "Building inference_runner for all architectures"
echo "========================================="

for arch in "${ARCHS[@]}"; do
    echo ""
    echo "--- Building inference_runner for $arch ---"
    build_dir="$SCRIPT_DIR/runtime/build-${arch}"
    bash "$SCRIPT_DIR/runtime/build-tests.sh" \
        --arch "$arch" \
        --build-dir "$build_dir" \
        --runtime-lib-dir "$ARTIFACTS_DIR/${arch}-ubuntu${UBUNTU_VERSION}"
done

# --- Assemble package ---

echo ""
echo "========================================="
echo "Assembling package: $PKG_NAME"
echo "========================================="

TMP_PKG="$(mktemp -d)"
trap 'rm -rf "$TMP_PKG"' EXIT
PKG_DIR="$TMP_PKG/$PKG_NAME"
mkdir -p "$PKG_DIR"

# Per-arch binaries and libraries
for arch in "${ARCHS[@]}"; do
    bin_dir="$PKG_DIR/bin/$arch"
    mkdir -p "$bin_dir"

    build_dir="$SCRIPT_DIR/runtime/build-${arch}"
    cp "$build_dir/inference_runner" "$bin_dir/inference_runner"
    chmod +x "$bin_dir/inference_runner"

    lib_src="$(realpath "$ARTIFACTS_DIR/${arch}-ubuntu${UBUNTU_VERSION}/libRuntimeLibrary.so")"
    cp "$lib_src" "$bin_dir/libRuntimeLibrary.so"

    echo "  bin/$arch/ : inference_runner + libRuntimeLibrary.so"
done

# Compiled DXNN models
find "$COMPILED_DIR" -name "*.dxnn" | while read -r dxnn; do
    rel="${dxnn#"$COMPILED_DIR"/}"
    dest="$PKG_DIR/compiled_models/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$dxnn" "$dest"
done
echo "  compiled_models/ : $DXNN_COUNT .dxnn file(s)"

# Python helpers
cp "$SCRIPT_DIR/stage2.py" "$PKG_DIR/stage2.py"
cp "$SCRIPT_DIR/models.py"  "$PKG_DIR/models.py"
echo "  stage2.py, models.py"

# Patch stage2.py: fix import path and COMPILED_DIR to work standalone
sed -i 's/from tests\.models/from models/' "$PKG_DIR/stage2.py"

# Launcher script (auto-detects arch at runtime)
cat > "$PKG_DIR/run_stage2.sh" << 'EOF'
#!/bin/bash
# Run DeepX stage2 inference tests on this machine.
#
# Configuration (all optional env vars):
#   MODELS      Space-separated model names (default: all)
#   RUNS        Number of timed inference runs (default: 100)
#   WARMUP      Number of warmup runs (default: 10)
#   CSV         Path to output CSV file
#   LOG_LEVEL   DEBUG|INFO|WARNING|ERROR (default: INFO)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ARCH="$(uname -m)"
BIN_DIR="$SCRIPT_DIR/bin/$ARCH"

if [[ ! -d "$BIN_DIR" ]]; then
    echo "Error: no binaries bundled for architecture '$ARCH'" >&2
    echo "Available: $(ls "$SCRIPT_DIR/bin/" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

export DEEPX_RUNNER_DIR="$BIN_DIR"
export DEEPX_RUNTIME_LIB_DIR="$BIN_DIR"

ARGS=(
    --runs   "${RUNS:-100}"
    --warmup "${WARMUP:-10}"
    --log-level "${LOG_LEVEL:-INFO}"
)
[[ -n "$MODELS" ]] && ARGS+=(--models $MODELS)
[[ -n "$CSV"    ]] && ARGS+=(--csv "$CSV")

python3 "$SCRIPT_DIR/stage2.py" "${ARGS[@]}"
EOF
chmod +x "$PKG_DIR/run_stage2.sh"

# README
cat > "$PKG_DIR/README.txt" << EOF
DeepX Stage 2 Test Package (v${VERSION})
========================================

Prerequisites:
  - DeepX NPU hardware (DX-M1 or DX-H1)
  - Python 3.8+
  - DX-RT ${UBUNTU_VERSION} system libraries installed

Supported architectures: ${ARCHS[*]}

Usage:
  ./run_stage2.sh

With options:
  MODELS="yolov8n" RUNS=200 CSV=results.csv ./run_stage2.sh

Results are printed to stdout and optionally written to a CSV file.
EOF

# --- Archive ---

tar -czf "$OUTPUT" -C "$TMP_PKG" "$PKG_NAME"

echo ""
echo "========================================="
echo "Package created: $OUTPUT"
echo "Contents:"
tar -tzf "$OUTPUT" | sed 's/^/  /'
echo ""
echo "To use on a target machine:"
echo "  tar -xzf $(basename "$OUTPUT")"
echo "  cd $PKG_NAME"
echo "  ./run_stage2.sh"
