#!/bin/bash
# Create a self-contained stage2 archive for execution on a remote machine.
#
# The archive includes:
#   - inference_runner binary
#   - libRuntimeLibrary.so (and any symlinks, resolved)
#   - All compiled .dxnn models
#   - stage2.py, run_stage2.sh, models.py
#
# Usage:
#   ./tests/package_stage2.sh [--output <archive.tar.gz>]
#
# The resulting archive can be unpacked and run on a target machine:
#   tar -xzf deepx-stage2-<version>.tar.gz
#   cd deepx-stage2-<version>
#   ./run_stage2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_BIN="$SCRIPT_DIR/runtime/build/inference_runner"
COMPILED_DIR="$SCRIPT_DIR/compiled_models"
VERSION_FILE="$REPO_ROOT/VERSION"

OUTPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

VERSION="$(cat "$VERSION_FILE" 2>/dev/null || echo "dev")"
PKG_NAME="deepx-stage2-${VERSION}"
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$REPO_ROOT/${PKG_NAME}.tar.gz"
fi

# --- Sanity checks ---

if [[ ! -f "$RUNNER_BIN" ]]; then
    echo "Error: inference_runner not found at $RUNNER_BIN" >&2
    echo "Build it first:" >&2
    echo "  bash tests/runtime/build-tests.sh" >&2
    exit 1
fi

DXNN_COUNT=$(find "$COMPILED_DIR" -name "*.dxnn" 2>/dev/null | wc -l)
if [[ "$DXNN_COUNT" -eq 0 ]]; then
    echo "Error: no .dxnn files found under $COMPILED_DIR" >&2
    echo "Run stage 1 first:" >&2
    echo "  python3 tests/stage1.py" >&2
    exit 1
fi

# Find libRuntimeLibrary.so (follow symlinks to get the real file)
RUNNER_DIR="$(dirname "$RUNNER_BIN")"
LIB_PATH=""
for candidate in \
    "$RUNNER_DIR/libRuntimeLibrary.so" \
    "$(ldconfig -p 2>/dev/null | awk '/libRuntimeLibrary/{print $NF}' | head -1)"
do
    if [[ -f "$candidate" ]]; then
        LIB_PATH="$(realpath "$candidate")"
        break
    fi
done

if [[ -z "$LIB_PATH" ]]; then
    echo "Error: libRuntimeLibrary.so not found." >&2
    echo "Ensure it is linked in $RUNNER_DIR or discoverable via ldconfig." >&2
    exit 1
fi

echo "Packaging:"
echo "  inference_runner : $RUNNER_BIN"
echo "  runtime library  : $LIB_PATH"
echo "  compiled models  : $DXNN_COUNT .dxnn file(s)"
echo "  output archive   : $OUTPUT"
echo ""

# --- Build package directory in a temp location ---

TMP_PKG="$(mktemp -d)"
trap 'rm -rf "$TMP_PKG"' EXIT
PKG_DIR="$TMP_PKG/$PKG_NAME"
mkdir -p "$PKG_DIR"

# Binaries
cp "$RUNNER_BIN"   "$PKG_DIR/inference_runner"
cp "$LIB_PATH"     "$PKG_DIR/libRuntimeLibrary.so"
chmod +x "$PKG_DIR/inference_runner"

# Compiled models (preserve directory structure)
find "$COMPILED_DIR" -name "*.dxnn" | while read -r dxnn; do
    rel="${dxnn#"$COMPILED_DIR"/}"
    dest="$PKG_DIR/compiled_models/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$dxnn" "$dest"
done

# Python helpers
cp "$SCRIPT_DIR/stage2.py"     "$PKG_DIR/stage2.py"
cp "$SCRIPT_DIR/run_stage2.sh" "$PKG_DIR/run_stage2.sh"
cp "$SCRIPT_DIR/models.py"     "$PKG_DIR/models.py"
chmod +x "$PKG_DIR/run_stage2.sh"

# Patch stage2.py import path so it works without the package installed
sed -i 's/from tests\.models/from models/' "$PKG_DIR/stage2.py" 2>/dev/null || true

# Simple README
cat > "$PKG_DIR/README.txt" << EOF
DeepX Stage 2 Test Package (v${VERSION})
========================================

Prerequisites:
  - DeepX NPU hardware (DX-M1 or DX-H1)
  - Python 3.8+

Usage:
  ./run_stage2.sh

With options:
  MODELS="yolov8n" RUNS=200 CSV=results.csv ./run_stage2.sh

Results are printed to stdout and optionally written to a CSV file.
EOF

# --- Archive ---
tar -czf "$OUTPUT" -C "$TMP_PKG" "$PKG_NAME"
echo "Archive created: $OUTPUT"
echo ""
echo "To use on a target machine:"
echo "  tar -xzf $(basename "$OUTPUT")"
echo "  cd $PKG_NAME"
echo "  ./run_stage2.sh"
